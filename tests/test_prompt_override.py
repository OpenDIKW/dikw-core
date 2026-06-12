"""Per-base prompt override resolution + validation (ADR-0003, workstream 2).

Covers ``prompts.resolve`` (containment + placeholder / output-marker contract)
and the ``dikw client check`` leg ``api_health._check_prompt_overrides`` that
surfaces the same failures before a synth/lint run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import prompts
from dikw_core.api_health import _check_prompt_overrides
from dikw_core.api_types import CheckReport, ProbeResult
from dikw_core.config import DikwConfig, LintConfig, SynthConfig
from dikw_core.prompts import PromptOverrideError

# A synthesize override that satisfies the contract: every placeholder the
# engine fills, the four ``<page category="..." slug="...">`` output markers,
# and the ``## Knowledge-base context`` H2 container the renderer's H3
# sub-sections nest under.
SYNTH_OK = (
    "Categories:\n{categories}\n"
    "## Knowledge-base context\n{existing_pages_section}\n"
    "Source path: {source_path}\n"
    "Source body:\n{source_body}\n"
    "Outline: {group_outline}\n"
    "Group {group_index} of {group_total}, emit at most {max_pages} pages.\n"
    'Emit blocks like <page category="entity" slug="x">...</page>.\n'
)

# An orphan-merge override (the other fixer-overridable prompt).
ORPHAN_OK = (
    "Target {target_path} {target_category} {target_slug} {target_title}\n"
    "{target_body}\n"
    "Orphan {orphan_path}\n{orphan_body}\n"
    "Why: {score_reason}\n"
    'Output <page category="{target_category}" slug="{target_slug}">...</page>\n'
)


def _write(base: Path, rel: str, text: str) -> str:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return rel


# ---- prompts.resolve ------------------------------------------------------


def test_resolve_no_override_returns_packaged_default(tmp_path: Path) -> None:
    assert prompts.resolve("synthesize") == prompts.load("synthesize")
    # base_root is irrelevant when nothing is overridden.
    assert prompts.resolve("synthesize", base_root=tmp_path) == prompts.load("synthesize")


def test_resolve_valid_override_returns_override_text(tmp_path: Path) -> None:
    rel = _write(tmp_path, "prompts/my_synth.md", SYNTH_OK)
    out = prompts.resolve("synthesize", override_path=rel, base_root=tmp_path)
    assert out == SYNTH_OK
    assert out != prompts.load("synthesize")


def test_resolve_valid_orphan_merge_override(tmp_path: Path) -> None:
    rel = _write(tmp_path, "prompts/orphan.md", ORPHAN_OK)
    out = prompts.resolve("lint_fix_orphan_merge", override_path=rel, base_root=tmp_path)
    assert out == ORPHAN_OK


def test_resolve_override_escaping_base_rejected(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    # File lives in the parent, outside the base — a ``..`` traversal.
    (tmp_path / "evil.md").write_text(SYNTH_OK, encoding="utf-8")
    with pytest.raises(PromptOverrideError, match="outside the base"):
        prompts.resolve("synthesize", override_path="../evil.md", base_root=base)


def test_resolve_override_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(PromptOverrideError, match="not found"):
        prompts.resolve("synthesize", override_path="prompts/nope.md", base_root=tmp_path)


def test_resolve_override_without_base_root_rejected(tmp_path: Path) -> None:
    with pytest.raises(PromptOverrideError, match="requires a base_root"):
        prompts.resolve("synthesize", override_path="prompts/x.md")


def test_resolve_missing_required_placeholder_rejected(tmp_path: Path) -> None:
    bad = SYNTH_OK.replace("Source body:\n{source_body}\n", "Source body: (removed)\n")
    rel = _write(tmp_path, "prompts/bad.md", bad)
    with pytest.raises(PromptOverrideError, match=r"missing required placeholder.*source_body"):
        prompts.resolve("synthesize", override_path=rel, base_root=tmp_path)


def test_resolve_stray_unknown_placeholder_rejected(tmp_path: Path) -> None:
    bad = SYNTH_OK + "Stray: {bogus}\n"
    rel = _write(tmp_path, "prompts/stray.md", bad)
    with pytest.raises(PromptOverrideError, match=r"unknown placeholder.*bogus"):
        prompts.resolve("synthesize", override_path=rel, base_root=tmp_path)


def test_resolve_missing_output_marker_rejected(tmp_path: Path) -> None:
    # Drop the closing </page> marker while keeping every placeholder.
    bad = SYNTH_OK.replace("</page>", "")
    rel = _write(tmp_path, "prompts/nomarker.md", bad)
    with pytest.raises(PromptOverrideError, match=r"output-format marker.*</page>"):
        prompts.resolve("synthesize", override_path=rel, base_root=tmp_path)


def test_resolve_override_missing_context_heading_rejected(tmp_path: Path) -> None:
    # ``_render_existing_section`` / ``_render_priority_targets`` emit H3
    # sub-sections that assume a ``## Knowledge-base context`` H2 container in
    # the template. An override written against the pre-0.5.x layout (its own
    # ``## Existing pages`` heading, no container) would silently nest the
    # dynamic sections under the wrong parent — so the contract requires the
    # container and ``dikw client check`` fails loudly instead.
    bad = SYNTH_OK.replace("## Knowledge-base context\n", "## Existing pages\n")
    rel = _write(tmp_path, "prompts/old-layout.md", bad)
    with pytest.raises(PromptOverrideError, match=r"Knowledge-base context"):
        prompts.resolve("synthesize", override_path=rel, base_root=tmp_path)


def test_resolve_override_demoted_context_heading_rejected(tmp_path: Path) -> None:
    # ``### Knowledge-base context`` contains the H2 string as a SUBSTRING —
    # a bare ``in`` check would wave the demoted heading through and the H3
    # sub-sections would nest under an H3 parent. The marker is line-anchored.
    bad = SYNTH_OK.replace(
        "## Knowledge-base context\n", "### Knowledge-base context\n"
    )
    rel = _write(tmp_path, "prompts/demoted.md", bad)
    with pytest.raises(PromptOverrideError, match=r"Knowledge-base context"):
        prompts.resolve("synthesize", override_path=rel, base_root=tmp_path)


def test_resolve_non_overridable_prompt_rejected(tmp_path: Path) -> None:
    # eval_judge_synth has no override contract — deliberately not user-overridable.
    rel = _write(tmp_path, "prompts/judge.md", "anything")
    with pytest.raises(PromptOverrideError, match="not overridable"):
        prompts.resolve("eval_judge_synth", override_path=rel, base_root=tmp_path)


def test_resolve_malformed_template_raises_prompt_override_error(tmp_path: Path) -> None:
    # A syntactically malformed format token (a lone unescaped ``{``) makes
    # ``string.Formatter().parse`` raise a bare ``ValueError``. ``resolve``
    # must surface it as a ``PromptOverrideError`` (its documented failure
    # type) so synth/lint and ``dikw client check`` get a typed, catchable
    # error instead of a raw crash.
    rel = _write(tmp_path, "prompts/malformed.md", SYNTH_OK + "lone { brace\n")
    with pytest.raises(PromptOverrideError, match="not a valid template"):
        prompts.resolve("synthesize", override_path=rel, base_root=tmp_path)


def test_check_malformed_template_returns_failure_not_raises(tmp_path: Path) -> None:
    # The ``dikw client check`` leg must report a malformed override as a
    # failed ProbeResult, never let a bare ValueError escape and crash the
    # whole check command.
    rel = _write(tmp_path, "prompts/malformed.md", SYNTH_OK + "lone { brace\n")
    cfg = DikwConfig(synth=SynthConfig(prompt_path=rel))
    results = _check_prompt_overrides(cfg, tmp_path)
    assert len(results) == 1
    assert results[0].ok is False
    assert "not a valid template" in results[0].detail


# ---- dikw check leg: _check_prompt_overrides ------------------------------


def test_check_no_overrides_configured_returns_empty(tmp_path: Path) -> None:
    assert _check_prompt_overrides(DikwConfig(), tmp_path) == []


def test_check_valid_synth_override_ok(tmp_path: Path) -> None:
    rel = _write(tmp_path, "prompts/my_synth.md", SYNTH_OK)
    cfg = DikwConfig(synth=SynthConfig(prompt_path=rel))
    results = _check_prompt_overrides(cfg, tmp_path)
    assert len(results) == 1
    assert results[0].ok is True
    assert results[0].target == rel


def test_check_invalid_synth_override_fails(tmp_path: Path) -> None:
    bad = SYNTH_OK.replace("Source body:\n{source_body}\n", "(removed)\n")
    rel = _write(tmp_path, "prompts/bad.md", bad)
    cfg = DikwConfig(synth=SynthConfig(prompt_path=rel))
    results = _check_prompt_overrides(cfg, tmp_path)
    assert len(results) == 1
    assert results[0].ok is False
    assert "source_body" in results[0].detail


def test_check_valid_fixer_override_ok(tmp_path: Path) -> None:
    rel = _write(tmp_path, "prompts/orphan.md", ORPHAN_OK)
    cfg = DikwConfig(lint=LintConfig(fixer_prompts={"orphan_merge": rel}))
    results = _check_prompt_overrides(cfg, tmp_path)
    assert len(results) == 1
    assert results[0].ok is True


def test_check_report_ok_false_when_prompt_leg_fails() -> None:
    # A green provider leg must not mask a broken prompt override.
    report = CheckReport(
        llm=ProbeResult(ok=True, target="(provider default)", detail="ok"),
        embed=ProbeResult(ok=True, target="x", detail="ok"),
        prompts=[ProbeResult(ok=False, target="prompts/bad.md", detail="invalid")],
    )
    assert report.ok is False


def test_check_report_ok_true_when_prompts_valid() -> None:
    report = CheckReport(
        llm=ProbeResult(ok=True, target="(provider default)", detail="ok"),
        prompts=[ProbeResult(ok=True, target="prompts/ok.md", detail="valid")],
    )
    assert report.ok is True
