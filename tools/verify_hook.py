"""Commit verification gate and stamp tracker for Claude Code / local runs.

Implements an ``off`` / ``observe`` / ``enforce`` gate inspired by the Claude
Code team's verification-gate pattern:

  * ``tools/check.py`` captures a working-tree fingerprint before running its
    stages and verifies the tree did not change while checks ran, then writes a
    gitignored stamp (``.claude/.verify-stamp.json``). The fingerprint hashes
    file contents across ``git diff --no-renames HEAD``, ``git diff --cached
    --no-renames``, and untracked non-ignored paths, so staging files (including
    renames) via ``git add`` keeps the stamp valid while any post-check edit,
    partial staging mismatch, or new commit invalidates it.
  * Wired as a Claude Code ``PreToolUse`` hook on ``Bash`` in
    ``.claude/settings.json`` (mode controlled by ``DIKW_VERIFY_GATE_MODE``,
    defaulting to ``enforce``; set ``observe`` or ``off`` to downgrade).
  * In ``enforce`` mode, an unverified ``git commit`` is blocked (exit 2)
    unless:
      1. the target worktree's stamp matches the tree and staged content (and no
         preceding command in the same Bash call mutates files before commit), OR
      2. the commit command carries an explicit ``[skip-verify: <reason>]``
         explanation, OR
      3. the same session has already been blocked twice (escape hatch so an
         agent cannot deadlock).

Stdlib-only and Python 3.9+ compatible so the hook executes in milliseconds
even when launched with a system ``python3`` outside the project venv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

GateMode = Literal["off", "observe", "enforce"]

STAMP_REL = Path(".claude") / ".verify-stamp.json"
STATE_REL = Path(".claude") / ".verify-gate-state.json"
OBSERVE_LOG_REL = Path(".claude") / "verify-gate-observe.jsonl"
_INTERNAL_STATE_PATHS = frozenset(
    {STAMP_REL.as_posix(), STATE_REL.as_posix(), OBSERVE_LOG_REL.as_posix()}
)

ENV_MODE_VAR = "DIKW_VERIFY_GATE_MODE"
DEFAULT_MODE: GateMode = "enforce"
MAX_SESSION_BLOCKS = 2

_SEPARATOR_CHARS = frozenset(";&|\n")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SKIP_REASON_RE = re.compile(r"\[skip-verify:\s*([^\]\r\n]+?)\s*\]", re.IGNORECASE)
# Global git flags that consume the next token as their argument.
_GIT_FLAGS_WITH_ARG = frozenset(
    {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix", "--config-env"}
)
# Short/long `git commit` options that ALWAYS consume a following argument
# (attached or next token). Note: `-S`/`--gpg-sign` and `-u`/`--untracked-files`
# take OPTIONAL args that Git requires to be attached, so bare `-S -a` must NOT
# consume `-a`, while `-Skey` consumes the rest of the token as the key id.
# `-o` is `--only` (boolean, no argument).
_COMMIT_SHORT_VALUE_FLAGS = frozenset("mFCct")
_COMMIT_SHORT_OPTIONAL_VALUE_FLAGS = frozenset("Su")
_COMMIT_SHORT_KNOWN_BOOL_FLAGS = frozenset("qvsenzohai" + "p")
_COMMIT_LONG_VALUE_FLAGS = frozenset(
    {
        "--message",
        "--file",
        "--reuse-message",
        "--reedit-message",
        "--fixup",
        "--squash",
        "--template",
        "--author",
        "--date",
        "--cleanup",
        "--trailer",
        "--pathspec-from-file",
    }
)
_COMMIT_LONG_OPTIONAL_VALUE_FLAGS = frozenset({"--gpg-sign", "--untracked-files"})
_COMMIT_LONG_KNOWN_BOOL_FLAGS = frozenset(
    {
        "--quiet",
        "--no-quiet",
        "--verbose",
        "--no-verbose",
        "--reset-author",
        "--no-reset-author",
        "--signoff",
        "--no-signoff",
        "--edit",
        "--no-edit",
        "--status",
        "--no-status",
        "--only",
        "--no-only",
        "--no-verify",
        "--verify",
        "--branch",
        "--no-branch",
        "--ahead-behind",
        "--no-ahead-behind",
        "--amend",
        "--no-amend",
        "--no-post-rewrite",
        "--post-rewrite",
        "--allow-empty",
        "--allow-empty-message",
        "--no-gpg-sign",
        "--no-untracked-files",
        "--pathspec-file-nul",
        "--no-pathspec-file-nul",
    }
)
# Allowlist of `git add` flags known not to narrow, skip, dry-run, force-stage
# ignored files, or alter index modes. Any other option (including `--force` or
# abbreviations like `--dry`) fails closed so partial-staging parity checks
# remain active.
_ADD_SAFE_LONG_FLAGS = frozenset(
    {"--all", "--update", "--verbose", "--no-dry-run", "--no-ignore-removal"}
)
_ADD_SAFE_SHORT_FLAGS = frozenset("Auv")
# Repo-local Git env vars that could redirect subprocesses away from `cwd` when
# invoked from a user Git hook.
_GIT_LOCAL_ENV_VARS = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_PREFIX",
    }
)
# Commands safe to precede `git commit` in the same Bash call without mutating
# file contents after verification.
_BENIGN_PRE_COMMIT_GIT = frozenset({"add", "stage", "status", "diff", "log", "branch", "rev-parse"})
_BENIGN_PRE_COMMIT_CMDS = frozenset({"cd", "echo", "printf", "pwd", "true"})


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    exit_code: int
    message: str


@dataclass(frozen=True)
class CommitInvocation:
    """Parsed target worktree and flags for a ``git commit`` shell segment."""

    target_dir: Path
    commit_all: bool
    preceded_by_mutator: bool
    preceded_by_full_stage: bool = False
    preceded_by_selective_stage: bool = False


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()  # noqa: UP017


def resolve_mode(raw: str | None = None) -> GateMode:
    val = (raw if raw is not None else os.environ.get(ENV_MODE_VAR, DEFAULT_MODE)).strip().lower()
    if val in ("off", "observe", "enforce"):
        return val  # type: ignore[return-value]
    return DEFAULT_MODE


def extract_skip_reason(command: str) -> str | None:
    """Return the non-empty explanation inside ``[skip-verify: <reason>]``, or None."""
    match = _SKIP_REASON_RE.search(command)
    if match is None:
        return None
    reason = match.group(1).strip()
    return reason or None


def _strip_shell_comments(command: str) -> str:
    """Strip unquoted ``# ...`` shell comments while preserving newlines."""
    out: list[str] = []
    in_single = False
    in_double = False
    escaped = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if escaped:
            out.append(ch)
            escaped = False
            i += 1
            continue
        if in_single:
            out.append(ch)
            if ch == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            out.append(ch)
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            i += 1
            continue
        if ch == "'":
            in_single = True
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_double = True
            out.append(ch)
            i += 1
            continue
        if ch == "#" and (i == 0 or command[i - 1] in " \t\r\n;&|"):
            while i < n and command[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _is_separator(tok: str) -> bool:
    return bool(tok) and all(ch in _SEPARATOR_CHARS for ch in tok)


def _split_shell_segments_with_ops(command: str) -> list[tuple[str | None, list[str]]] | None:
    """Split ``command`` into ``(preceding_separator, tokens)`` pairs."""
    cleaned = _strip_shell_comments(command)
    try:
        lexer = shlex.shlex(cleaned, posix=True, punctuation_chars=";&|<>\n")
        lexer.commenters = ""
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None

    segments: list[tuple[str | None, list[str]]] = []
    current: list[str] = []
    pending_op: str | None = None
    for tok in tokens:
        if _is_separator(tok):
            # shlex joins adjacent punctuation such as `||\n` or `&&\n` into one
            # token; strip continuation newlines so operator comparisons match.
            norm_op = tok.replace("\n", "") or "\n"
            if current:
                segments.append((pending_op, current))
                current = []
                pending_op = norm_op
            elif pending_op is None or norm_op != "\n":
                # Preserve `&&` / `||` across a following line-continuation newline.
                pending_op = norm_op
        else:
            current.append(tok)
    if current:
        segments.append((pending_op, current))
    return segments


def _split_shell_segments(command: str) -> list[list[str]] | None:
    paired = _split_shell_segments_with_ops(command)
    if paired is None:
        return None
    return [seg for _, seg in paired]


def _segment_has_output_redirection(tokens: list[str]) -> bool:
    """True if any unquoted shell token performs output redirection (e.g. ``>``, ``>>``)."""
    return any(tok.startswith(">") or tok in ("&>", "&>>") for tok in tokens)


def _segment_has_input_redirection(tokens: list[str]) -> bool:
    """True if any unquoted shell token performs input redirection (e.g. ``<``, ``<<``)."""
    return any(tok.startswith("<") for tok in tokens)


def _git_add_may_force_stage_ignored(args: list[str]) -> bool:
    """True if ``git add`` passes ``-f`` / ``--force`` (or an abbreviation of it)."""
    for tok in args:
        if tok == "--":
            break
        if tok.startswith("--"):
            if tok != "--" and "--force".startswith(tok) and len(tok) >= 3:
                return True
            continue
        if tok.startswith("-") and len(tok) > 1 and "f" in tok[1:]:
            return True
    return False


def _git_add_stages_all_tracked(args: list[str], target_dir: Path | None = None) -> bool:
    """True when ``git add`` is guaranteed to stage the full working tree.

    Uses a strict allowlist of safe options so unknown flags or abbreviations
    (e.g. ``--dry``, ``--chmod``, ``-p``, ``--pathspec-from-file``) fail closed.
    Matches ``git add -A/--all/-u/--update`` with no narrowing pathspec, and
    ``git add .`` (or ``./``) when executed at the Git repository root.
    """
    has_all_flag = False
    positional: list[str] = []
    after_dd = False
    for tok in args:
        if after_dd:
            positional.append(tok)
            continue
        if tok == "--":
            after_dd = True
            continue
        if tok.startswith("--"):
            if tok not in _ADD_SAFE_LONG_FLAGS:
                return False
            if tok in ("--all", "--update"):
                has_all_flag = True
            continue
        if tok.startswith("-") and len(tok) > 1:
            flags = tok[1:]
            if any(ch not in _ADD_SAFE_SHORT_FLAGS for ch in flags):
                return False
            if any(ch in "Au" for ch in flags):
                has_all_flag = True
            continue
        positional.append(tok)

    if has_all_flag and not positional:
        return True
    if positional and all(p in (".", "./") for p in positional) and target_dir is not None:
        repo_root = _try_resolve_repo_root(target_dir)
        return repo_root is not None and target_dir.resolve() == repo_root
    return False


def _segment_command_name(tokens: list[str]) -> str | None:
    idx = 0
    while idx < len(tokens) and (
        _ENV_ASSIGN_RE.match(tokens[idx]) or tokens[idx] in {"env", "command"}
    ):
        idx += 1
    if idx >= len(tokens):
        return None
    return Path(tokens[idx]).name


def _parse_git_segment(
    tokens: list[str], base_dir: Path
) -> tuple[str | None, Path, list[str], bool]:
    """Return ``(subcommand, effective_dir, subcommand_args, uses_custom_git_env)``."""
    idx = 0
    uses_custom_git_env = False
    while idx < len(tokens) and (
        _ENV_ASSIGN_RE.match(tokens[idx]) or tokens[idx] in {"env", "command"}
    ):
        if _ENV_ASSIGN_RE.match(tokens[idx]):
            var_name = tokens[idx].split("=", 1)[0]
            if var_name in _GIT_LOCAL_ENV_VARS:
                uses_custom_git_env = True
        idx += 1
    if idx >= len(tokens) or Path(tokens[idx]).name != "git":
        return None, base_dir, [], uses_custom_git_env

    idx += 1
    effective_dir = base_dir
    work_tree: Path | None = None
    while idx < len(tokens):
        tok = tokens[idx]
        if tok == "-C" and idx + 1 < len(tokens):
            arg = Path(tokens[idx + 1])
            effective_dir = arg if arg.is_absolute() else (effective_dir / arg)
            idx += 2
            continue
        if tok.startswith("-C") and len(tok) > 2:
            arg = Path(tok[2:])
            effective_dir = arg if arg.is_absolute() else (effective_dir / arg)
            idx += 1
            continue
        if tok == "--work-tree" and idx + 1 < len(tokens):
            arg = Path(tokens[idx + 1])
            work_tree = arg if arg.is_absolute() else (effective_dir / arg)
            uses_custom_git_env = True
            idx += 2
            continue
        if tok.startswith("--work-tree="):
            arg = Path(tok.split("=", 1)[1])
            work_tree = arg if arg.is_absolute() else (effective_dir / arg)
            uses_custom_git_env = True
            idx += 1
            continue
        if tok == "--git-dir" and idx + 1 < len(tokens):
            uses_custom_git_env = True
            idx += 2
            continue
        if tok.startswith("--git-dir="):
            uses_custom_git_env = True
            idx += 1
            continue
        if tok in _GIT_FLAGS_WITH_ARG:
            idx += 2
            continue
        if tok.startswith("-"):
            idx += 1
            continue
        return tok, (work_tree or effective_dir), tokens[idx + 1 :], uses_custom_git_env
    return None, (work_tree or effective_dir), [], uses_custom_git_env


@dataclass(frozen=True)
class _ParsedCommitArgs:
    read_only: bool
    commits_worktree: bool


def _parse_commit_args(
    args: list[str], *, has_shell_redirection: bool = False
) -> _ParsedCommitArgs:
    """Inspect ``git commit`` arguments for read-only previews and worktree commits.

    ``read_only`` is True for ``-h``/``--help``, or when ``--dry-run``,
    ``-z``/``--null``, or an explicit status preview format (``--short``,
    ``--porcelain[=...]``, ``--long``) remains active after honoring their
    respective ``--no-*`` negations and value-taking flags (so
    ``-m "--dry-run"`` and ``-Skeyh`` are never treated as previews).

    ``commits_worktree`` is True only when all options are recognized and Git
    commits working-tree bytes rather than the pre-existing index: either
    ``-a``/``--all`` or explicit positional pathspecs (with no shell input
    redirection that could mimic an operand) without ``-i``/``--include``,
    ``-p``/``--patch``, or ``--interactive``. Unrecognized/abbreviated options
    (e.g. ``--incl``) and ``--pathspec-from-file`` fail closed so index parity
    checks stay enabled.
    """
    dry_run = False
    status_format: str | None = None
    use_null = False
    is_help = False
    has_all = False
    has_include = False
    has_patch = False
    has_interactive = False
    has_pathspec = False
    has_unknown_flag = False

    idx = 0
    n = len(args)
    while idx < n:
        tok = args[idx]
        if tok == "--":
            if idx + 1 < n:
                has_pathspec = True
            break
        if tok == "--help":
            is_help = True
            idx += 1
            continue
        if tok == "--dry-run":
            dry_run = True
            idx += 1
            continue
        if tok == "--no-dry-run":
            dry_run = False
            idx += 1
            continue
        if tok == "--short":
            status_format = "short"
            idx += 1
            continue
        if tok == "--porcelain" or tok.startswith("--porcelain="):
            status_format = "porcelain"
            idx += 1
            continue
        if tok == "--long":
            status_format = "long"
            idx += 1
            continue
        if tok in ("--no-short", "--no-porcelain", "--no-long"):
            status_format = None
            idx += 1
            continue
        if tok == "--null":
            use_null = True
            idx += 1
            continue
        if tok == "--no-null":
            use_null = False
            idx += 1
            continue
        if tok == "--all":
            has_all = True
            idx += 1
            continue
        if tok == "--no-all":
            has_all = False
            idx += 1
            continue
        if tok == "--include":
            has_include = True
            idx += 1
            continue
        if tok == "--no-include":
            has_include = False
            idx += 1
            continue
        if tok == "--patch":
            has_patch = True
            idx += 1
            continue
        if tok == "--no-patch":
            has_patch = False
            idx += 1
            continue
        if tok == "--interactive":
            has_interactive = True
            idx += 1
            continue
        if tok == "--no-interactive":
            has_interactive = False
            idx += 1
            continue
        if tok.startswith("--"):
            if tok in _COMMIT_LONG_VALUE_FLAGS:
                idx += 2
                continue
            flag_name, has_eq, _ = tok.partition("=")
            if has_eq and (
                flag_name in _COMMIT_LONG_VALUE_FLAGS
                or flag_name in _COMMIT_LONG_OPTIONAL_VALUE_FLAGS
            ):
                idx += 1
                continue
            if not has_eq and (
                tok in _COMMIT_LONG_KNOWN_BOOL_FLAGS or tok in _COMMIT_LONG_OPTIONAL_VALUE_FLAGS
            ):
                idx += 1
                continue
            # Unrecognized or abbreviated long flag (e.g. `--incl`, `--no-shor`):
            # fail closed so neither preview nor worktree exemptions trigger.
            has_unknown_flag = True
            idx += 1
            continue
        if tok.startswith("-") and len(tok) > 1:
            flags = tok[1:]
            consumed_next = False
            for pos, ch in enumerate(flags):
                if ch == "h":
                    is_help = True
                elif ch == "z":
                    use_null = True
                elif ch == "a":
                    has_all = True
                elif ch == "i":
                    has_include = True
                elif ch == "p":
                    has_patch = True
                elif ch in _COMMIT_SHORT_VALUE_FLAGS:
                    # Remaining chars in this token (if any) are the option's value
                    # (e.g. `-m"add tests"`). Only consume the next token when this
                    # flag is the final character of the cluster.
                    if pos == len(flags) - 1:
                        consumed_next = True
                    break
                elif ch in _COMMIT_SHORT_OPTIONAL_VALUE_FLAGS:
                    # `-S[<key-id>]` / `-u[<mode>]`: any remaining chars in this
                    # token are the attached argument, not more flags; bare `-S`
                    # never consumes the next token.
                    break
                elif ch not in _COMMIT_SHORT_KNOWN_BOOL_FLAGS:
                    has_unknown_flag = True
            idx += 2 if consumed_next else 1
            continue
        has_pathspec = True
        idx += 1

    is_preview = (dry_run or status_format is not None or use_null) and not has_unknown_flag
    confirmed_pathspec = has_pathspec and not has_shell_redirection
    commits_worktree = (
        (not has_unknown_flag)
        and (not has_patch)
        and (not has_interactive)
        and (has_all or (confirmed_pathspec and not has_include))
    )
    return _ParsedCommitArgs(read_only=(is_help or is_preview), commits_worktree=commits_worktree)


def _commit_has_all_flag(args: list[str]) -> bool:
    """Return True iff ``git commit`` commits working-tree bytes directly."""
    return _parse_commit_args(args).commits_worktree


def find_commit_invocations(command: str, base_dir: Path) -> list[CommitInvocation]:
    """Return all mutating ``git commit`` invocations in ``command`` with target dirs."""
    if "git" not in command or "commit" not in command:
        return []
    paired_segments = _split_shell_segments_with_ops(command)
    if paired_segments is None:
        # Unbalanced quotes — fall back conservatively to unquoted segment check.
        if re.search(r"(?:^|[;&|\n]\s*)git(?:\s+\S+)*\s+commit\b", command):
            return [
                CommitInvocation(
                    target_dir=base_dir, commit_all=False, preceded_by_mutator=False
                )
            ]
        return []

    invocations: list[CommitInvocation] = []
    cwd = base_dir
    saw_mutator = False
    full_staged_dirs: set[Path] = set()
    selective_staged_dirs: set[Path] = set()
    for preceding_op, seg in paired_segments:
        # A full-stage exemption only carries forward while the next command
        # depends on the preceding command succeeding (`&&`). `;`, `||`, `|`,
        # `&`, or newline can execute the commit even if staging failed or was
        # skipped by an earlier conditional.
        if preceding_op != "&&":
            full_staged_dirs.clear()
        if _segment_has_output_redirection(seg):
            saw_mutator = True
            continue
        has_input_redir = _segment_has_input_redirection(seg)
        cmd_name = _segment_command_name(seg)
        if cmd_name == "cd" and len(seg) >= 2:
            target = Path(seg[1])
            cwd = target if target.is_absolute() else (cwd / target)
            continue
        subcmd, target_dir, sub_args, uses_custom_git_env = _parse_git_segment(seg, cwd)
        resolved_target = target_dir.resolve()
        if subcmd == "commit":
            parsed = _parse_commit_args(sub_args, has_shell_redirection=has_input_redir)
            if parsed.read_only:
                continue
            invocations.append(
                CommitInvocation(
                    target_dir=resolved_target,
                    commit_all=parsed.commits_worktree,
                    preceded_by_mutator=saw_mutator,
                    preceded_by_full_stage=(
                        not uses_custom_git_env and resolved_target in full_staged_dirs
                    ),
                    preceded_by_selective_stage=resolved_target in selective_staged_dirs,
                )
            )
            continue
        if subcmd in ("add", "stage"):
            if _git_add_may_force_stage_ignored(sub_args):
                # Force-staging can add ignored files absent from the stamped
                # working-tree fingerprint; require fresh verification first.
                full_staged_dirs.discard(resolved_target)
                saw_mutator = True
                continue
            if (
                not uses_custom_git_env
                and not has_input_redir
                and preceding_op not in ("||", "|")
                and _git_add_stages_all_tracked(sub_args, resolved_target)
            ):
                full_staged_dirs.add(resolved_target)
            else:
                # Any non-full add (including abbreviated `--chmod`) may leave
                # index modes or entries out of sync with the stamped tree.
                full_staged_dirs.discard(resolved_target)
                selective_staged_dirs.add(resolved_target)
            continue
        if subcmd in _BENIGN_PRE_COMMIT_GIT or cmd_name in _BENIGN_PRE_COMMIT_CMDS:
            continue
        saw_mutator = True
    return invocations


def is_git_commit_command(command: str) -> bool:
    """Return True iff any top-level shell segment invokes ``git commit``."""
    return bool(find_commit_invocations(command, Path(".")))


def _clean_git_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _GIT_LOCAL_ENV_VARS}


def _git(repo_root: Path, args: list[str]) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        env=_clean_git_env(),
        check=True,
        capture_output=True,
    ).stdout


def _try_resolve_repo_root(target_dir: Path) -> Path | None:
    if not target_dir.is_dir():
        return None
    try:
        top = _git(target_dir, ["rev-parse", "--show-toplevel"]).decode("utf-8").strip()
        return Path(top).resolve() if top else None
    except subprocess.CalledProcessError:
        return None


def _resolve_repo_root(target_dir: Path, fallback: Path) -> Path:
    resolved = _try_resolve_repo_root(target_dir)
    return resolved if resolved is not None else fallback


def _git_head_sha(repo_root: Path) -> str:
    try:
        return _git(repo_root, ["rev-parse", "HEAD"]).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return "UNBORN"


def _git_name_set(repo_root: Path, args: list[str], *, strict: bool = False) -> set[str]:
    try:
        raw = _git(repo_root, args)
    except subprocess.CalledProcessError:
        if strict:
            raise
        return set()
    out: set[str] = set()
    for entry in raw.split(b"\x00"):
        if entry:
            rel = entry.decode("utf-8", errors="surrogateescape")
            if rel not in _INTERNAL_STATE_PATHS:
                out.add(rel)
    return out


def compute_tree_fingerprint(repo_root: Path) -> tuple[str, str]:
    """Return ``(head_sha, fingerprint)`` invariant under ``git add`` (incl. renames).

    Hashes the current working-tree bytes and Git-relevant executable mode bit
    of every path differing from HEAD (staged or unstaged, with ``--no-renames``)
    plus untracked non-ignored files. Any post-check edit, mode change, or new
    file changes the fingerprint; staging already-checked files or renames does
    not.
    """
    head_sha = _git_head_sha(repo_root)
    paths: set[str] = set()
    for cmd in (
        ["diff", "--no-renames", "--name-only", "-z", "HEAD"],
        ["diff", "--cached", "--no-renames", "--name-only", "-z"],
        ["ls-files", "--others", "--exclude-standard", "-z"],
    ):
        paths.update(_git_name_set(repo_root, cmd))

    digest = hashlib.sha256()
    digest.update(head_sha.encode("utf-8"))
    digest.update(b"\n")
    for rel in sorted(paths):
        digest.update(rel.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\x00")
        target = repo_root / rel
        if target.is_symlink():
            digest.update(b"SYMLINK:")
            digest.update(os.readlink(target).encode("utf-8", errors="surrogateescape"))
        elif target.is_file():
            mode_tag = b"EXEC:" if (target.stat().st_mode & 0o111) else b"FILE:"
            digest.update(mode_tag)
            digest.update(hashlib.sha256(target.read_bytes()).hexdigest().encode("ascii"))
        else:
            digest.update(b"DELETED")
        digest.update(b"\n")
    return head_sha, digest.hexdigest()


def _partially_staged_paths(repo_root: Path) -> list[str] | None:
    """Paths whose staged (index) content differs from the verified working tree.

    Returns ``None`` if Git fails to answer either diff query (fail-closed).
    """
    try:
        staged = _git_name_set(
            repo_root, ["diff", "--cached", "--no-renames", "--name-only", "-z"], strict=True
        )
        if not staged:
            return []
        unstaged = _git_name_set(
            repo_root, ["diff", "--no-renames", "--name-only", "-z"], strict=True
        )
    except subprocess.CalledProcessError:
        return None
    return sorted(staged & unstaged)


def write_stamp(
    repo_root: Path,
    *,
    stages: list[str],
    expected_head_sha: str | None = None,
    expected_fingerprint: str | None = None,
) -> dict[str, Any] | None:
    """Record a verification stamp for the current working tree.

    When ``expected_head_sha`` / ``expected_fingerprint`` are supplied (captured
    *before* checks ran), returns ``None`` without writing if the tree changed
    mid-run so edits made after a check read a file can never be stamped green.
    """
    head_sha, fingerprint = compute_tree_fingerprint(repo_root)
    if expected_head_sha is not None and head_sha != expected_head_sha:
        return None
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        return None

    payload: dict[str, Any] = {
        "head_sha": head_sha,
        "tree_fingerprint": fingerprint,
        "stages": list(stages),
        "timestamp": _utc_now_iso(),
    }
    stamp_path = repo_root / STAMP_REL
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def verify_stamp(
    repo_root: Path, *, check_staged_parity: bool = True
) -> tuple[bool, str]:
    """Return ``(ok, reason)`` checking whether the saved stamp matches the tree."""
    stamp_path = repo_root / STAMP_REL
    if not stamp_path.is_file():
        return False, "no verification stamp found (run `uv run python tools/check.py`)"
    try:
        data = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable verification stamp: {exc}"
    if not isinstance(data, dict):
        return False, "malformed verification stamp"

    head_sha, fingerprint = compute_tree_fingerprint(repo_root)
    if data.get("head_sha") != head_sha:
        return False, "verification stamp is stale (HEAD changed since last check)"
    if data.get("tree_fingerprint") != fingerprint:
        return False, "verification stamp is stale (files changed since last check)"
    if check_staged_parity:
        partial = _partially_staged_paths(repo_root)
        if partial is None:
            return False, "could not verify staged index parity (git diff failed)"
        if partial:
            preview = ", ".join(partial[:3])
            suffix = f" (+{len(partial) - 3} more)" if len(partial) > 3 else ""
            return (
                False,
                f"staged index differs from verified working tree for: {preview}{suffix} "
                "(stage the tested files in a separate command before committing)",
            )
    return True, "verified"


def _load_state(repo_root: Path) -> dict[str, int]:
    path = repo_root / STATE_REL
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): int(v) for k, v in raw.items() if isinstance(v, int)}


def _save_state(repo_root: Path, state: dict[str, int]) -> None:
    """Best-effort state persistence; I/O errors must never cancel a gate decision."""
    path = repo_root / STATE_REL
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def _append_observe_log(repo_root: Path, entry: dict[str, Any]) -> None:
    """Best-effort audit logging; I/O errors must never override a gate decision."""
    log_path = repo_root / OBSERVE_LOG_REL
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def evaluate_commit_gate(
    repo_root: Path,
    *,
    command: str,
    mode: GateMode | None = None,
    session_id: str = "default",
    command_cwd: Path | None = None,
) -> GateDecision:
    """Evaluate whether a ``git commit`` command should be allowed or blocked.

    ``repo_root`` stores hook state / observe logs; ``command_cwd`` (the Bash
    tool's working directory, defaulting to ``repo_root``) resolves relative
    targets and bare ``git commit`` invocations.
    """
    effective_mode = resolve_mode(mode)
    if effective_mode == "off":
        return GateDecision(allowed=True, exit_code=0, message="")

    base_cwd = (command_cwd or repo_root).resolve()
    invocations = find_commit_invocations(command, base_cwd)
    if not invocations:
        return GateDecision(allowed=True, exit_code=0, message="")

    skip_reason = extract_skip_reason(command)
    if skip_reason is not None:
        _append_observe_log(
            repo_root,
            {
                "timestamp": _utc_now_iso(),
                "mode": effective_mode,
                "session_id": session_id,
                "would_block": False,
                "skip_reason": skip_reason,
            },
        )
        return GateDecision(
            allowed=True,
            exit_code=0,
            message=f"[verify-gate:{effective_mode}] bypass accepted: {skip_reason}",
        )

    failures: list[tuple[Path, str]] = []
    for inv in invocations:
        target_root = _resolve_repo_root(inv.target_dir, repo_root)
        if inv.preceded_by_mutator:
            failures.append(
                (
                    target_root,
                    "compound command runs another step before `git commit` that may "
                    "modify files; run `tools/check.py` and `git commit` in separate calls",
                )
            )
            continue
        check_staged = not (inv.commit_all or inv.preceded_by_full_stage)
        ok, reason = verify_stamp(target_root, check_staged_parity=check_staged)
        if not ok:
            if inv.preceded_by_selective_stage and "staged index differs" in reason:
                reason = (
                    f"{reason}; run `git add` in a separate command before `git commit` "
                    "so staged parity can be verified after staging"
                )
            failures.append((target_root, reason))

    if not failures:
        state = _load_state(repo_root)
        if state.get(session_id, 0) != 0:
            state[session_id] = 0
            _save_state(repo_root, state)
        return GateDecision(allowed=True, exit_code=0, message="")

    target_root, reason = failures[0]
    entry = {
        "timestamp": _utc_now_iso(),
        "mode": effective_mode,
        "session_id": session_id,
        "target_repo": str(target_root),
        "would_block": True,
        "reason": reason,
        "command_preview": command[:160],
    }

    if effective_mode == "observe":
        _append_observe_log(repo_root, entry)
        return GateDecision(
            allowed=True,
            exit_code=0,
            message=(
                f"[verify-gate:observe] would block commit: {reason}. "
                "Run `uv run python tools/check.py` before committing "
                "(or add `[skip-verify: <reason>]` to the commit message)."
            ),
        )

    # enforce mode with 2-block session escape hatch
    state = _load_state(repo_root)
    prior_blocks = state.get(session_id, 0)
    if prior_blocks >= MAX_SESSION_BLOCKS:
        state[session_id] = 0
        _save_state(repo_root, state)
        entry["escape_hatch"] = True
        _append_observe_log(repo_root, entry)
        return GateDecision(
            allowed=True,
            exit_code=0,
            message=(
                f"[verify-gate:enforce] escape hatch activated after {prior_blocks} prior "
                f"blocks in session {session_id!r}; allowing commit through ({reason})."
            ),
        )

    state[session_id] = prior_blocks + 1
    _save_state(repo_root, state)
    _append_observe_log(repo_root, entry)
    return GateDecision(
        allowed=False,
        exit_code=2,
        message=(
            f"[verify-gate:enforce] commit blocked ({prior_blocks + 1}/{MAX_SESSION_BLOCKS}): "
            f"{reason}. Run `uv run python tools/check.py` first, or include "
            "`[skip-verify: <reason>]` in the commit message."
        ),
    )


def _handle_claude_pretool(repo_root: Path, mode: GateMode | None) -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    if not isinstance(payload, dict):
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return 0
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return 0
    session_id = str(payload.get("session_id") or "default")
    raw_cwd = payload.get("cwd")
    command_cwd = Path(raw_cwd) if isinstance(raw_cwd, str) and raw_cwd.strip() else None
    decision = evaluate_commit_gate(
        repo_root,
        command=command,
        mode=mode,
        session_id=session_id,
        command_cwd=command_cwd,
    )
    if decision.message:
        print(decision.message, file=sys.stderr)
    return decision.exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (defaults to parent of tools/)",
    )
    parser.add_argument(
        "--mode",
        choices=("off", "observe", "enforce"),
        default=None,
        help=f"override {ENV_MODE_VAR} (default: {DEFAULT_MODE})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("claude-pretool", help="Claude Code PreToolUse Bash hook handler")
    sub.add_parser("fingerprint", help="print current HEAD and tree fingerprint as JSON")
    stamp_p = sub.add_parser("stamp", help="record a fresh verification stamp")
    stamp_p.add_argument("--stage", action="append", default=[])
    stamp_p.add_argument("--expect-head", default=None)
    stamp_p.add_argument("--expect-fingerprint", default=None)
    sub.add_parser("check", help="read-only check of current verification stamp status")

    args = parser.parse_args(argv)
    repo_root: Path = args.repo_root.resolve()

    if args.cmd == "claude-pretool":
        return _handle_claude_pretool(repo_root, args.mode)
    if args.cmd == "fingerprint":
        head_sha, fp = compute_tree_fingerprint(repo_root)
        print(json.dumps({"head_sha": head_sha, "tree_fingerprint": fp}))
        return 0
    if args.cmd == "stamp":
        stages = list(args.stage) if args.stage else ["check"]
        stamped = write_stamp(
            repo_root,
            stages=stages,
            expected_head_sha=args.expect_head,
            expected_fingerprint=args.expect_fingerprint,
        )
        if stamped is None:
            print(
                "verify-hook: working tree changed while checks ran; stamp NOT recorded.",
                file=sys.stderr,
            )
            return 1
        print("verify-hook: stamp recorded.")
        return 0
    if args.cmd == "check":
        ok, reason = verify_stamp(repo_root)
        if ok:
            print("verify-hook: stamp is valid.")
            return 0
        print(f"verify-hook: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
