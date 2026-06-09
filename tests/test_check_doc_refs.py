"""Tests for ``tools/check_doc_refs.py`` — the doc-reference drift checker.

Two layers: pure-function unit tests (command-path parsing, CLI-chain
validation), behaviour tests over a synthetic temp repo (the FP-avoidance rules
that make the tool trustworthy), and the load-bearing **repo-clean gate** —
``check_doc_refs(REPO_ROOT)`` must return no findings, so a future doc that
renames a CLI verb or env var without updating the prose fails CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load tools/check_doc_refs.py directly — tools/ is not an importable package.
# Register in sys.modules before exec so the module-level @dataclass can resolve
# its own __module__ (dataclasses looks the class's module up in sys.modules).
_spec = importlib.util.spec_from_file_location(
    "check_doc_refs", REPO_ROOT / "tools" / "check_doc_refs.py"
)
assert _spec is not None and _spec.loader is not None
cdr = importlib.util.module_from_spec(_spec)
sys.modules["check_doc_refs"] = cdr
_spec.loader.exec_module(cdr)


_GOLDEN = """\
dikw client [group]
dikw client synth [command]
dikw client lint [group]
dikw client lint propose [command]
dikw serve [command]
"""


def _paths() -> dict[tuple[str, ...], str]:
    return cdr.load_command_paths(_GOLDEN)


# ---- pure: load_command_paths -------------------------------------------


def test_load_command_paths_parses_kind_and_path() -> None:
    paths = _paths()
    assert paths[("dikw", "client")] == "group"
    assert paths[("dikw", "client", "synth")] == "command"
    assert paths[("dikw", "client", "lint", "propose")] == "command"
    assert paths[("dikw", "serve")] == "command"


def test_load_command_paths_rejects_unknown_kind() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown command-kind"):
        cdr.load_command_paths("dikw client splice [commnd]\n")


# ---- pure: validate_cli_ref ---------------------------------------------


def test_valid_command_resolves() -> None:
    assert cdr.validate_cli_ref(["client", "synth"], _paths()) is None


def test_group_alone_resolves() -> None:
    assert cdr.validate_cli_ref(["client", "lint"], _paths()) is None


def test_positional_arg_after_leaf_command_is_fine() -> None:
    # synth is a leaf command → the trailing token is an argument, not drift.
    assert cdr.validate_cli_ref(["client", "synth", "somearg"], _paths()) is None


def test_unknown_subcommand_after_group_is_drift() -> None:
    reason = cdr.validate_cli_ref(["client", "bogus"], _paths())
    assert reason is not None and "bogus" in reason


def test_unknown_top_level_is_drift() -> None:
    reason = cdr.validate_cli_ref(["nope"], _paths())
    assert reason is not None and "nope" in reason


def test_nested_command_resolves() -> None:
    assert cdr.validate_cli_ref(["client", "lint", "propose"], _paths()) is None


# ---- behaviour: check_doc_refs over a synthetic repo --------------------


def _make_repo(tmp_path: Path, doc_body: str) -> Path:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "cli_command_tree.golden.txt").write_text(
        _GOLDEN, encoding="utf-8"
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "x.py").write_text(
        'KEY = os.environ["DIKW_REAL_VAR"]\n', encoding="utf-8"
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(doc_body, encoding="utf-8")
    return tmp_path


def test_bad_cli_ref_in_inline_code_is_caught(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "Run `dikw client bogus` to do the thing.\n")
    findings = cdr.check_doc_refs(repo)
    assert [f for f in findings if f.kind == "cli" and "bogus" in f.ref]


def test_bad_env_var_is_caught(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "Set `DIKW_BOGUS_VAR` before running.\n")
    findings = cdr.check_doc_refs(repo)
    assert [f for f in findings if f.kind == "env" and f.ref == "DIKW_BOGUS_VAR"]


def test_good_cli_and_env_refs_pass(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path,
        "Run `dikw client synth` with `DIKW_REAL_VAR` set.\n",
    )
    assert cdr.check_doc_refs(repo) == []


def test_prose_noun_phrase_is_not_a_cli_ref(tmp_path: Path) -> None:
    # "the dikw base" reads as `dikw base` but it's prose (no code formatting) —
    # CLI refs are only extracted from code spans, so this must not fire.
    repo = _make_repo(tmp_path, "Open the dikw base in your editor.\n")
    assert cdr.check_doc_refs(repo) == []


def test_fence_shell_comment_is_not_a_cli_ref(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path,
        "```bash\n# the first call to dikw will lazy-load the token\n"
        "dikw client synth\n```\n",
    )
    assert cdr.check_doc_refs(repo) == []


def test_dikw_as_value_after_equals_is_not_a_cli_ref(tmp_path: Path) -> None:
    # A DSN `user=dikw password=...` is not a command invocation.
    repo = _make_repo(
        tmp_path, '```yaml\ndsn: "user=dikw password=secret dbname=dikw"\n```\n'
    )
    assert cdr.check_doc_refs(repo) == []


def test_compound_identifier_is_not_an_env_ref(tmp_path: Path) -> None:
    # A token that merely CONTAINS DIKW_ (`MY_DIKW_THING`) must not yield a
    # phantom `DIKW_THING` — the env regex is left-anchored.
    repo = _make_repo(tmp_path, "Tune `MY_DIKW_THING` for throughput.\n")
    assert cdr.check_doc_refs(repo) == []


def test_examples_dir_is_an_env_authority(tmp_path: Path) -> None:
    # A container-orchestration-only var defined in examples/ (never read by
    # Python) must resolve, not false-positive.
    repo = _make_repo(tmp_path, "Set `DIKW_COMPOSE_ONLY` in your compose file.\n")
    examples = repo / "examples"
    examples.mkdir()
    (examples / "docker-compose.yml").write_text(
        "services:\n  app:\n    environment:\n      DIKW_COMPOSE_ONLY: '1'\n",
        encoding="utf-8",
    )
    assert cdr.check_doc_refs(repo) == []


def test_allowlisted_future_verb_passes(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path, "A future `dikw client reindex <path>` will close the gap.\n"
    )
    assert cdr.check_doc_refs(repo) == []


def test_adr_dir_is_excluded(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "clean doc\n")
    adr = repo / "docs" / "adr"
    adr.mkdir()
    # A removed verb cited in an ADR (historical record) must not fire.
    (adr / "0001-x.md").write_text("We removed `dikw client bogus`.\n", encoding="utf-8")
    assert cdr.check_doc_refs(repo) == []


# ---- the repo-clean gate (regression guard) -----------------------------


def test_repo_docs_have_no_reference_drift() -> None:
    findings = cdr.check_doc_refs(REPO_ROOT)
    assert findings == [], "doc-reference drift:\n" + "\n".join(
        f"  {f.file}:{f.line} [{f.kind}] {f.ref!r} — {f.reason}" for f in findings
    )
