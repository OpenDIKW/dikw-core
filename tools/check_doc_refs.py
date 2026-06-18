"""Doc-reference drift checker — CLI verbs + ``DIKW_`` env vars cited in docs must resolve in source.

Catches the recurring "rename a CLI verb / env var, miss a doc" drift
(``feedback_grep_cli_typos_across_docs``) that ruff/mypy/pytest are blind to.
Two checks over the documentation set:

  * **CLI verbs** — every ``dikw <command chain>`` referenced in a *code context*
    (a fenced block or an inline ``code`` span) must be a valid command path in
    the Typer app, validated against ``tests/cli_command_tree.golden.txt`` (the
    same golden the CLI-surface test pins). The golden's ``[group]`` / ``[command]``
    annotation disambiguates an unknown subcommand (drift — a group expects a known
    subcommand) from a positional arg after a leaf command (fine). CLI refs are
    read ONLY from code spans because *prose* noun phrases ("the dikw base", "a
    dikw server") would otherwise read as ``dikw <subcommand>`` and false-positive.
    A residual hole remains — a noun phrase that is itself code-formatted
    (`` `the dikw base layout` ``) still false-positives — but it is rare (docs
    wrap *commands* in code, not "dikw <noun>" phrases) and the allowlist is the
    escape hatch when it happens.

  * **Env vars** — every ``DIKW_<NAME>`` token anywhere in the doc set must be
    read somewhere in the codebase: ``src/`` (runtime vars), ``tests/`` + ``.github/``
    (test / CI vars like ``DIKW_TEST_POSTGRES_DSN``), or ``.env.example`` (declared).
    The token is distinctive enough that scanning prose too is low-FP.

Routes (``/v1/...``) are a deliberate follow-up: FastAPI path-template
normalization (param-name + ``:path`` converter differences) carries enough
false-positive risk that it is not worth shipping in this first advisory pass.

Doc set: ``CLAUDE.md`` + ``docs/**/*.md`` + ``README.md`` + ``.claude/skills/**/*.md``.
``CHANGELOG.md`` is excluded — it is a historical record that intentionally cites
removed verbs/vars.

The pure :func:`check_doc_refs` is unit-tested in ``tests/test_check_doc_refs.py``,
which also asserts the live repo is drift-free — so this is a **hard gate** (that
test runs in CI / ``tools/check.py``), not merely advisory. The roadmap pencilled
it in as advisory-first out of false-positive caution; the tool is tuned to zero
findings with conservative, unit-tested FP rules + the two allowlists as the
escape hatch, so it ships as a real gate (a soft check that goes red without
blocking is itself close to a false-green). :func:`main` adds the repo-root
plumbing for a standalone / pre-commit run.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

GOLDEN_REL = "tests/cli_command_tree.golden.txt"

# The only command-kind annotations the golden tree uses. A bracketed token
# outside this set is a typo (or an unhandled new Typer shape) — fail fast
# rather than silently store a bogus kind that ``validate_cli_ref`` would then
# treat as a non-group leaf.
_GOLDEN_KINDS = frozenset({"group", "command"})

# ``dikw`` followed by >=1 lowercase command tokens. ``\bdikw\b`` won't fire
# inside ``dikw-core`` / ``dikw_core`` / ``dikw.yml`` (no following space), and a
# bare ``dikw`` with no command word never matches (the group is required). The
# ``(?<![=/])`` lookbehind drops ``dikw`` as a *value* — a DSN ``user=dikw
# password=…`` or a path segment ``.../dikw foo`` — which is not a command.
_CLI_RE = re.compile(r"(?<![=/])\bdikw((?:\s+[a-z][a-z0-9-]*)+)")
# Distinctive enough to scan prose without false positives. ``DIKW_*`` (a glob)
# does not match — the class requires >=1 of [A-Z0-9_] after the underscore. The
# ``(?<![A-Z0-9_])`` lookbehind anchors the left edge so a compound identifier
# (``MY_DIKW_THING``) doesn't yield a phantom ``DIKW_THING`` from its middle.
_ENV_RE = re.compile(r"(?<![A-Z0-9_])DIKW_[A-Z0-9_]+")
# Inline code spans: `one` or ``two-backtick`` runs. Captures the inner text.
_INLINE_CODE_RE = re.compile(r"`+([^`]+)`+")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

# Env vars that are legitimately doc-only / external and need no source read.
# Empty today; an entry here is a conscious exception, not a silent skip.
_ENV_ALLOWLIST: frozenset[str] = frozenset()

# CLI references that intentionally name a command that doesn't resolve today —
# a *documented future* verb. Each is a conscious exception, not a silent skip;
# a ref matches if it equals an entry or extends it with positional args.
_CLI_ALLOWLIST: frozenset[str] = frozenset(
    {
        # CLAUDE.md + docs/architecture.md + ADR-0005 + CHANGELOG: the never-built
        # single-page reindex verb that PR3's stale_index/untracked_file drift
        # lint superseded. The docs still name it (as the superseded command), so
        # it must resolve here even though it has no implementation.
        "dikw client reindex",
        # docs/providers.md: planned verb to migrate the multimodal vec table
        # after switching the embedding model (v1 ships a stub). Distinct concept
        # from ``dikw client reindex`` above, hence its own entry.
        "dikw embed reindex",
    }
)


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    kind: str  # "cli" | "env"
    ref: str
    reason: str


def load_command_paths(golden_text: str) -> dict[tuple[str, ...], str]:
    """Parse ``cli_command_tree.golden.txt`` into ``path tuple -> "group"|"command"``.

    Each line is ``dikw a b c [group|command]``; the trailing token is the kind,
    the rest is the command path (including the ``dikw`` root).
    """
    paths: dict[tuple[str, ...], str] = {}
    for raw in golden_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        kind_tok = tokens[-1]
        if not (kind_tok.startswith("[") and kind_tok.endswith("]")):
            continue
        kind = kind_tok[1:-1]
        if kind not in _GOLDEN_KINDS:
            raise ValueError(
                f"unknown command-kind annotation {kind_tok!r} in golden tree "
                f"(line: {line!r}); expected one of {sorted(_GOLDEN_KINDS)}"
            )
        path = tuple(tokens[:-1])
        if path:
            paths[path] = kind
    return paths


def validate_cli_ref(
    chain: list[str], command_paths: dict[tuple[str, ...], str]
) -> str | None:
    """Walk a ``dikw`` command chain against the known paths.

    Returns a drift reason, or ``None`` if the chain resolves. The ``dikw`` root
    is an implicit group. A token that doesn't extend the current path is an
    *unknown subcommand* (drift) when the current node is a group, but a
    *positional argument* (fine — stop here) when it is a leaf command.
    """
    # An empty chain is unreachable here — ``_CLI_RE`` requires >=1 command token
    # — so it falls through to ``return None`` (a bare ``dikw`` is never scanned).
    path: tuple[str, ...] = ("dikw",)
    for tok in chain:
        candidate = (*path, tok)
        if candidate in command_paths:
            path = candidate
            continue
        is_group = path == ("dikw",) or command_paths.get(path) == "group"
        if is_group:
            return f"unknown subcommand: {' '.join(path)} {tok}"
        # current node is a leaf command → the rest are positional args
        return None
    return None


def _iter_doc_files(repo_root: Path) -> list[Path]:
    """The documentation set, in stable order.

    Excludes ``CHANGELOG.md`` and ``docs/adr/**`` — both are historical records
    that intentionally cite removed verbs/vars to document what changed, so
    gating them on current source would false-positive by design.

    Includes ``.claude/skills/**`` — skill docs cite CLI verbs and env vars
    the same way ``docs/**`` does, and drift there silently breaks the agent
    workflows that follow them.
    """
    files: list[Path] = []
    claude = repo_root / "CLAUDE.md"
    if claude.is_file():
        files.append(claude)
    readme = repo_root / "README.md"
    if readme.is_file():
        files.append(readme)
    docs_dir = repo_root / "docs"
    if docs_dir.is_dir():
        adr_dir = docs_dir / "adr"
        files.extend(
            p for p in sorted(docs_dir.rglob("*.md")) if adr_dir not in p.parents
        )
    skills_dir = repo_root / ".claude" / "skills"
    if skills_dir.is_dir():
        files.extend(sorted(skills_dir.rglob("*.md")))
    return files


def _known_env_vars(repo_root: Path) -> set[str]:
    """Every ``DIKW_*`` token read anywhere authoritative: src / tests / CI / examples / .env.example.

    ``examples/`` is included so a container-orchestration-only var (read by a
    docker-compose file, never by Python) documented in ``deployment-docker.md``
    resolves rather than false-positiving.
    """
    known: set[str] = set()
    roots = [
        repo_root / "src",
        repo_root / "tests",
        repo_root / ".github",
        repo_root / "examples",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".yml", ".yaml", ".toml"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            known.update(_ENV_RE.findall(text))
    env_example = repo_root / ".env.example"
    if env_example.is_file():
        known.update(_ENV_RE.findall(env_example.read_text(encoding="utf-8")))
    return known


def _scan_doc(
    text: str, command_paths: dict[tuple[str, ...], str], known_env: set[str], rel: str
) -> list[Finding]:
    findings: list[Finding] = []
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        # Env vars: scan the whole line (distinctive token, prose-safe) — including
        # fenced ``#`` shell comments, where an env var named in a comment should
        # still resolve. Only the CLI scan below skips those comment lines.
        for m in _ENV_RE.finditer(line):
            var = m.group(0)
            if var in known_env or var in _ENV_ALLOWLIST:
                continue
            findings.append(
                Finding(rel, lineno, "env", var, f"env var {var} is not read in src/tests/.github/.env.example")
            )
        # CLI verbs: code context only. Scan each code fragment SEPARATELY (the
        # whole fenced line, or each inline `code` span on its own) — joining
        # spans would fabricate a fake command from two unrelated ones
        # (`` `dikw client` `` + `` `subgroup` `` → "dikw client subgroup"). A
        # ``#``-led line inside a fence is a shell comment (prose), not a command.
        if in_fence and line.lstrip().startswith("#"):
            continue
        fragments = [line] if in_fence else _INLINE_CODE_RE.findall(line)
        for code in fragments:
            for m in _CLI_RE.finditer(code):
                chain = m.group(1).split()
                ref = f"dikw {' '.join(chain)}"
                if any(ref == a or ref.startswith(a + " ") for a in _CLI_ALLOWLIST):
                    continue
                reason = validate_cli_ref(chain, command_paths)
                if reason is not None:
                    findings.append(Finding(rel, lineno, "cli", ref, reason))
    return findings


def check_doc_refs(repo_root: Path) -> list[Finding]:
    """Scan the doc set; return every CLI/env reference that doesn't resolve in source."""
    golden = (repo_root / GOLDEN_REL).read_text(encoding="utf-8")
    command_paths = load_command_paths(golden)
    known_env = _known_env_vars(repo_root)
    findings: list[Finding] = []
    for path in _iter_doc_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        findings.extend(
            _scan_doc(path.read_text(encoding="utf-8"), command_paths, known_env, rel)
        )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repo root (defaults to the parent of tools/)",
    )
    args = parser.parse_args(argv)
    findings = check_doc_refs(args.repo_root)
    if not findings:
        print("doc-refs: OK — every CLI verb and DIKW_ env var in the docs resolves in source.")
        return 0
    print(f"doc-refs: {len(findings)} drift finding(s):\n")
    for f in findings:
        print(f"  {f.file}:{f.line}  [{f.kind}]  {f.ref!r}\n      {f.reason}")
    print("\nFix the doc (or, for a deliberate exception, the allowlist in tools/check_doc_refs.py).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
