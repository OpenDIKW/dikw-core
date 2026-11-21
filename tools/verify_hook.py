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
    defaulting to ``observe`` so new rollout surfaces false positives before
    blocking).
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
DEFAULT_MODE: GateMode = "observe"
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
# consume `-a`.
_COMMIT_SHORT_VALUE_FLAGS = frozenset("mFCcto")
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


def _split_shell_segments(command: str) -> list[list[str]] | None:
    cleaned = _strip_shell_comments(command)
    try:
        lexer = shlex.shlex(cleaned, posix=True, punctuation_chars=";&|>\n")
        lexer.commenters = ""
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None

    segments: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if _is_separator(tok):
            if current:
                segments.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        segments.append(current)
    return segments


def _segment_has_output_redirection(tokens: list[str]) -> bool:
    """True if any unquoted shell token performs output redirection (e.g. ``>``, ``>>``)."""
    return any(tok.startswith(">") or tok in ("&>", "&>>") for tok in tokens)


def _git_add_stages_all_tracked(args: list[str]) -> bool:
    """True for ``git add -A/--all/-u/--update`` with no narrowing pathspec."""
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
        if tok in ("-A", "--all", "-u", "--update"):
            has_all_flag = True
            continue
        if tok.startswith("-"):
            continue
        positional.append(tok)
    return has_all_flag and not positional


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
) -> tuple[str | None, Path, list[str]]:
    """Return ``(subcommand, effective_dir, subcommand_args)`` for a git segment."""
    idx = 0
    while idx < len(tokens) and (
        _ENV_ASSIGN_RE.match(tokens[idx]) or tokens[idx] in {"env", "command"}
    ):
        idx += 1
    if idx >= len(tokens) or Path(tokens[idx]).name != "git":
        return None, base_dir, []

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
            idx += 2
            continue
        if tok.startswith("--work-tree="):
            arg = Path(tok.split("=", 1)[1])
            work_tree = arg if arg.is_absolute() else (effective_dir / arg)
            idx += 1
            continue
        if tok in _GIT_FLAGS_WITH_ARG:
            idx += 2
            continue
        if tok.startswith("-"):
            idx += 1
            continue
        return tok, (work_tree or effective_dir), tokens[idx + 1 :]
    return None, (work_tree or effective_dir), []


def _commit_has_all_flag(args: list[str]) -> bool:
    """Return True iff ``git commit`` args include ``-a`` or ``--all``."""
    idx = 0
    n = len(args)
    while idx < n:
        tok = args[idx]
        if tok == "--":
            break
        if tok == "--all":
            return True
        if tok.startswith("--"):
            if tok in _COMMIT_LONG_VALUE_FLAGS:
                idx += 2
            else:
                idx += 1
            continue
        if tok.startswith("-") and len(tok) > 1:
            flags = tok[1:]
            consumed_next = False
            for pos, ch in enumerate(flags):
                if ch == "a":
                    return True
                if ch in _COMMIT_SHORT_VALUE_FLAGS:
                    # Remaining chars in this token (if any) are the option's value
                    # (e.g. `-m"add tests"`). Only consume the next token when this
                    # flag is the final character of the cluster.
                    if pos == len(flags) - 1:
                        consumed_next = True
                    break
            idx += 2 if consumed_next else 1
            continue
        idx += 1
    return False


def find_commit_invocations(command: str, base_dir: Path) -> list[CommitInvocation]:
    """Return all ``git commit`` invocations in ``command`` with their target dirs."""
    if "git" not in command or "commit" not in command:
        return []
    segments = _split_shell_segments(command)
    if segments is None:
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
    for seg in segments:
        if _segment_has_output_redirection(seg):
            saw_mutator = True
            continue
        cmd_name = _segment_command_name(seg)
        if cmd_name == "cd" and len(seg) >= 2:
            target = Path(seg[1])
            cwd = target if target.is_absolute() else (cwd / target)
            continue
        subcmd, target_dir, sub_args = _parse_git_segment(seg, cwd)
        resolved_target = target_dir.resolve()
        if subcmd == "commit":
            invocations.append(
                CommitInvocation(
                    target_dir=resolved_target,
                    commit_all=_commit_has_all_flag(sub_args),
                    preceded_by_mutator=saw_mutator,
                    preceded_by_full_stage=resolved_target in full_staged_dirs,
                    preceded_by_selective_stage=resolved_target in selective_staged_dirs,
                )
            )
            continue
        if subcmd in ("add", "stage"):
            if _git_add_stages_all_tracked(sub_args):
                full_staged_dirs.add(resolved_target)
            else:
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


def _resolve_repo_root(target_dir: Path, fallback: Path) -> Path:
    if not target_dir.is_dir():
        return fallback
    try:
        top = _git(target_dir, ["rev-parse", "--show-toplevel"]).decode("utf-8").strip()
        return Path(top).resolve() if top else fallback
    except subprocess.CalledProcessError:
        return fallback


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
