"""Tests for the per-rule lint fixers.

PR1 ships only ``broken_wikilink`` with a heuristic-only path: fuzzy match
the broken target against existing K-layer page titles and propose a
``[[link]]`` rewrite when the match is confident enough; otherwise the
fixer returns ``None`` so the orchestrator skips the issue (LLM stub
fallback lands in PR2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dikw_core.domains.knowledge.lint import LintIssue
from dikw_core.domains.knowledge.lint_fix import (
    FixerContext,
    FixOperation,
    FixProposal,
    WikiPageMeta,
)
from dikw_core.domains.knowledge.lint_fixers.broken_wikilink import (
    BrokenWikilinkFixer,
)
from dikw_core.domains.knowledge.wiki import build_page

from .fakes import FakeLLM


def _make_page(title: str, body: str) -> Any:
    """Build a real ``WikiPage`` so tests can write the same path layout
    on disk that production synth would produce."""
    return build_page(title=title, body=body, type_="concept")


def _meta_from(page: Any) -> WikiPageMeta:
    return WikiPageMeta(path=page.path, title=page.title)


def _ctx(*, pages: list[Any], wiki_root: Path) -> FixerContext:
    return FixerContext(
        storage=None,
        llm=FakeLLM(),  # type: ignore[arg-type]
        embedding=None,
        wiki_root=wiki_root,
        all_pages=[_meta_from(p) for p in pages],
    )


@pytest.mark.asyncio
async def test_fuzzy_match_above_threshold_proposes_update_page(
    tmp_path: Path,
) -> None:
    """A broken ``[[foo bar]]`` should rewrite to the existing ``Foo Bar``
    page when normalized titles match within the 0.85 ratio band."""
    wiki_root = tmp_path
    target_page = _make_page("Foo Bar", "# Foo Bar\nbody\n")
    src_page = _make_page(
        "Source Page",
        "# Source Page\n\nSee [[foo  bar]] for context.\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source Page\n---\n\n"
        "# Source Page\n\nSee [[foo  bar]] for context.\n",
        encoding="utf-8",
    )

    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[foo  bar]] has no matching wiki page",
        line=3,
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[target_page, src_page], wiki_root=wiki_root),
        reporter=_NullReporter(),
    )

    assert proposal is not None
    assert isinstance(proposal, FixProposal)
    assert proposal.source == "heuristic"
    assert proposal.issue_path == src_page.path
    assert len(proposal.operations) == 1
    op = proposal.operations[0]
    assert isinstance(op, FixOperation)
    assert op.kind == "update_page"
    assert op.path == src_page.path
    assert op.expected_hash is not None  # we will hash the on-disk file
    assert op.new_body is not None
    # Rewrite should land an exact-title link.
    assert "[[Foo Bar]]" in op.new_body
    assert "[[foo  bar]]" not in op.new_body


@pytest.mark.asyncio
async def test_fuzzy_match_miss_returns_none_in_pr1(tmp_path: Path) -> None:
    """When no existing title is close enough, PR1 returns ``None``
    (LLM stub fallback ships in PR2)."""
    wiki_root = tmp_path
    other = _make_page("Completely Different", "# Completely Different\n")
    src_page = _make_page(
        "Source Page",
        "# Source Page\n\nSee [[xyz123abc]] here.\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source Page\n---\n\n"
        "# Source Page\n\nSee [[xyz123abc]] here.\n",
        encoding="utf-8",
    )

    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[xyz123abc]] has no matching wiki page",
        line=3,
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[other, src_page], wiki_root=wiki_root),
        reporter=_NullReporter(),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_match_excludes_self_page(tmp_path: Path) -> None:
    """A page must not propose a link to itself even if the broken
    target normalizes to the page's own title."""
    wiki_root = tmp_path
    src_page = _make_page(
        "Source Page",
        "# Source Page\n\nSee [[source page]].\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source Page\n---\n\n"
        "# Source Page\n\nSee [[source page]].\n",
        encoding="utf-8",
    )

    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[source page]] has no matching wiki page",
        line=3,
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[src_page], wiki_root=wiki_root),
        reporter=_NullReporter(),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_fuzzy_match_works_on_non_ascii_titles(tmp_path: Path) -> None:
    """The earlier ASCII-only ``[a-z0-9]`` normalize regex stripped
    every CJK glyph and made the fixer a no-op for multilingual bases.
    Verify a mixed-case CJK target now resolves."""
    wiki_root = tmp_path
    target_page = _make_page("卡尔曼滤波", "# 卡尔曼滤波\nbody\n")
    src_page = _make_page(
        "Source CJK",
        "# Source CJK\n\nReference [[卡尔曼 滤波]] here.\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source CJK\n---\n\n"
        "# Source CJK\n\nReference [[卡尔曼 滤波]] here.\n",
        encoding="utf-8",
    )

    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[卡尔曼 滤波]] has no matching wiki page",
        line=3,
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[target_page, src_page], wiki_root=wiki_root),
        reporter=_NullReporter(),
    )
    assert proposal is not None, "non-ASCII titles must not be silently skipped"
    assert "[[卡尔曼滤波]]" in proposal.operations[0].new_body  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_extracts_target_from_detail_when_issue_lacks_explicit_target(
    tmp_path: Path,
) -> None:
    """The lint scanner formats detail as ``[[<target>]] has no matching ...``;
    the fixer must parse that to recover the broken target text."""
    wiki_root = tmp_path
    target_page = _make_page("Karpathy Rule", "# Karpathy Rule\nbody\n")
    src_page = _make_page(
        "Source",
        "# Source\n\nApply [[karpathy rule]] here.\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        "---\ntitle: Source\n---\n\n"
        "# Source\n\nApply [[karpathy rule]] here.\n",
        encoding="utf-8",
    )

    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail="[[karpathy rule]] has no matching wiki page",
        line=3,
    )
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[target_page, src_page], wiki_root=wiki_root),
        reporter=_NullReporter(),
    )
    assert proposal is not None
    assert "[[Karpathy Rule]]" in proposal.operations[0].new_body  # type: ignore[arg-type]


class _NullReporter:
    """Minimal ``ProgressReporter`` for fixer-unit tests."""

    def __init__(self) -> None:
        from dikw_core.progress import CancelToken

        self._token = CancelToken()

    async def progress(
        self, *, phase: str, current: int = 0, total: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        return None

    async def log(self, level: str, message: str) -> None:
        return None

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        return None

    def cancel_token(self) -> Any:
        return self._token
