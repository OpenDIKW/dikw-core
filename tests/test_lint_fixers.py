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


def _ctx(
    *,
    pages: list[Any],
    wiki_root: Path,
    llm: Any | None = None,
    enable_llm: bool = False,
    cfg: Any = None,
) -> FixerContext:
    """Build a FixerContext for unit tests.

    LLM-enabled paths require a non-None ``cfg`` in production (see
    fixer guards); default to ``_default_cfg()`` so tests don't have
    to thread it through every call. Heuristic-only tests keep
    ``cfg=None`` and ``enable_llm=False`` — that branch never reads cfg.
    """
    if enable_llm and cfg is None:
        cfg = _default_cfg()
    return FixerContext(
        storage=None,
        llm=llm if llm is not None else FakeLLM(),  # type: ignore[arg-type]
        embedding=None,
        wiki_root=wiki_root,
        all_pages=[_meta_from(p) for p in pages],
        enable_llm=enable_llm,
        cfg=cfg,
    )


def _default_cfg() -> Any:
    """Build a minimal ``DikwConfig`` for fixer tests.

    Used by both ``broken_wikilink`` LLM-stub tests and the
    ``non_atomic_page`` tests so the fixer's ``ctx.cfg.provider.*`` /
    ``ctx.cfg.schema_.*`` / ``ctx.cfg.synth.*`` reads land on real
    defaults rather than ``None``.
    """
    from dikw_core.config import DikwConfig

    return DikwConfig()


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


# --- PR2: broken_wikilink LLM stub fallback ----------------------------------


def _make_broken_link_setup(
    tmp_path: Path, broken_target: str = "Whole New Topic"
) -> tuple[Path, Any, Any]:
    """Build a wiki-root + a source page that references a missing target.

    No existing K-page resembles ``broken_target``, so the heuristic
    must miss; LLM-stub tests then assert that the fallback fires when
    enabled and is silent when not.
    """
    wiki_root = tmp_path
    src_page = _make_page(
        "Source",
        f"# Source\n\nSee [[{broken_target}]] for context.\n",
    )
    src_abs = wiki_root / src_page.path
    src_abs.parent.mkdir(parents=True, exist_ok=True)
    src_abs.write_text(
        f"---\ntitle: Source\n---\n\n# Source\n\nSee [[{broken_target}]] for context.\n",
        encoding="utf-8",
    )
    issue = LintIssue(
        kind="broken_wikilink",
        path=src_page.path,
        detail=f"[[{broken_target}]] has no matching wiki page",
        line=3,
    )
    return wiki_root, src_page, issue


_STUB_LLM_RESPONSE = (
    "<page path=\"wiki/concepts/whole-new-topic.md\" type=\"concept\">\n"
    "---\n"
    "tags: [stub]\n"
    "---\n"
    "\n"
    "# Whole New Topic\n"
    "\n"
    "TODO: stub page for the broken `[[Whole New Topic]]` reference. "
    "Replace this body with real content.\n"
    "</page>\n"
)


@pytest.mark.asyncio
async def test_broken_wikilink_llm_stub_when_enabled(tmp_path: Path) -> None:
    """A heuristic miss with ``enable_llm=True`` should produce a
    ``create_page`` proposal whose body comes from the LLM."""
    wiki_root, src_page, issue = _make_broken_link_setup(tmp_path)
    fake = FakeLLM(response_text=_STUB_LLM_RESPONSE)
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[src_page], wiki_root=wiki_root, llm=fake, enable_llm=True),
        reporter=_NullReporter(),
    )

    assert proposal is not None, "LLM stub fallback must fire when enabled"
    assert isinstance(proposal, FixProposal)
    assert proposal.source == "llm"
    assert proposal.issue_path == src_page.path
    assert len(proposal.operations) == 1
    op = proposal.operations[0]
    assert isinstance(op, FixOperation)
    assert op.kind == "create_page"
    assert op.expected_hash is None  # create_page never asserts a prior hash
    assert op.path.startswith("wiki/")
    assert op.path.endswith(".md")
    assert op.new_body is not None
    assert "Whole New Topic" in op.new_body
    # FakeLLM captured the prompt — verify the broken target was injected
    # so the model has the context it needs to emit a relevant stub.
    assert fake.last_user is not None
    assert "Whole New Topic" in fake.last_user


@pytest.mark.asyncio
async def test_broken_wikilink_llm_disabled_returns_none(tmp_path: Path) -> None:
    """When ``enable_llm`` is False, a heuristic miss must still return
    None — the LLM path is opt-in and must not fire by default."""
    wiki_root, src_page, issue = _make_broken_link_setup(tmp_path)
    fake = FakeLLM(response_text=_STUB_LLM_RESPONSE)
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[src_page], wiki_root=wiki_root, llm=fake, enable_llm=False),
        reporter=_NullReporter(),
    )
    assert proposal is None
    # The LLM must NOT have been called when the flag is off.
    assert fake.last_user is None


@pytest.mark.asyncio
async def test_broken_wikilink_llm_failure_returns_none(tmp_path: Path) -> None:
    """An LLM exception in the stub path must not fail the whole propose
    task — return None so the orchestrator records a skip."""
    wiki_root, src_page, issue = _make_broken_link_setup(tmp_path)

    class _RaisingLLM:
        async def complete(self, **kwargs: Any) -> Any:
            raise RuntimeError("simulated LLM outage")

    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(
            pages=[src_page],
            wiki_root=wiki_root,
            llm=_RaisingLLM(),
            enable_llm=True,
        ),
        reporter=_NullReporter(),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_broken_wikilink_llm_unparseable_returns_none(tmp_path: Path) -> None:
    """LLM emits no usable ``<page>`` block (e.g. apologies / refusal):
    fixer must skip rather than synthesise an empty page."""
    wiki_root, src_page, issue = _make_broken_link_setup(tmp_path)
    fake = FakeLLM(response_text="Sorry, I cannot draft that stub.")
    fixer = BrokenWikilinkFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(pages=[src_page], wiki_root=wiki_root, llm=fake, enable_llm=True),
        reporter=_NullReporter(),
    )
    assert proposal is None


# --- PR2: non_atomic_page fixer ---------------------------------------------


_NON_ATOMIC_LLM_RESPONSE = (
    "<page path=\"wiki/concepts/topic-a.md\" type=\"concept\">\n"
    "---\n"
    "tags: [child]\n"
    "---\n"
    "\n"
    "# Topic A\n"
    "\n"
    "First atomic child page.\n"
    "</page>\n"
    "\n"
    "<page path=\"wiki/concepts/topic-b.md\" type=\"concept\">\n"
    "---\n"
    "tags: [child]\n"
    "---\n"
    "\n"
    "# Topic B\n"
    "\n"
    "Second atomic child page.\n"
    "</page>\n"
)


def _make_fat_page_on_disk(tmp_path: Path) -> tuple[Path, Any, Any]:
    """Write a non-atomic K-page to disk + the matching ``LintIssue``."""
    wiki_root = tmp_path
    page_path = "wiki/concepts/grab-bag.md"
    abs_path = wiki_root / page_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "---\n"
        "id: K-grab\n"
        "type: concept\n"
        "title: Grab Bag\n"
        "---\n"
        "\n"
        "# Grab Bag\n"
        "\n"
        "## Topic A\n\nFirst topic discussion.\n\n"
        "## Topic B\n\nSecond topic discussion.\n"
    )
    abs_path.write_text(body, encoding="utf-8")
    page = _make_page("Grab Bag", body)  # only used for ctx.all_pages
    issue = LintIssue(
        kind="non_atomic_page",
        path=page_path,
        detail="page looks like multiple atomic notes glued together: 2 H2 sections",
        line=None,
    )
    return wiki_root, page, issue


@pytest.mark.asyncio
async def test_non_atomic_page_splits_into_n_create_plus_one_delete(
    tmp_path: Path,
) -> None:
    """Happy path: LLM emits 2 child pages → fixer proposes 2 create_page
    + 1 delete_page (original)."""
    from dikw_core.domains.knowledge.lint_fixers.non_atomic_page import (
        NonAtomicPageFixer,
    )

    wiki_root, page, issue = _make_fat_page_on_disk(tmp_path)
    fake = FakeLLM(response_text=_NON_ATOMIC_LLM_RESPONSE)
    fixer = NonAtomicPageFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(
            pages=[page],
            wiki_root=wiki_root,
            llm=fake,
            enable_llm=True,
            cfg=_default_cfg(),
        ),
        reporter=_NullReporter(),
    )

    assert proposal is not None
    assert proposal.source == "llm"
    assert proposal.issue_kind == "non_atomic_page"
    creates = [op for op in proposal.operations if op.kind == "create_page"]
    deletes = [op for op in proposal.operations if op.kind == "delete_page"]
    assert len(creates) == 2, "LLM emitted 2 child pages"
    assert len(deletes) == 1, "original page must be deleted"
    assert deletes[0].path == issue.path
    assert deletes[0].expected_hash is not None  # concurrent-edit guard
    # External wikilinks are intentionally NOT rewritten — the design
    # leaves [[Grab Bag]] in other pages to be fixed by a follow-up
    # broken_wikilink propose run (decision A in the plan).
    paths = {op.path for op in creates}
    assert "wiki/concepts/grab-bag.md" not in paths
    assert all(p.startswith("wiki/") and p.endswith(".md") for p in paths)


@pytest.mark.asyncio
async def test_non_atomic_page_skips_when_synth_returns_one(tmp_path: Path) -> None:
    """If the LLM only finds one atomic page in the body, splitting is
    pointless — fixer returns None."""
    from dikw_core.domains.knowledge.lint_fixers.non_atomic_page import (
        NonAtomicPageFixer,
    )

    wiki_root, page, issue = _make_fat_page_on_disk(tmp_path)
    one_page_response = (
        "<page path=\"wiki/concepts/single-atomic.md\" type=\"concept\">\n"
        "---\ntags: [single]\n---\n\n# Single Atomic\n\nOnly one.\n"
        "</page>\n"
    )
    fake = FakeLLM(response_text=one_page_response)
    fixer = NonAtomicPageFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(
            pages=[page],
            wiki_root=wiki_root,
            llm=fake,
            enable_llm=True,
            cfg=_default_cfg(),
        ),
        reporter=_NullReporter(),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_non_atomic_page_skips_on_synth_error(tmp_path: Path) -> None:
    """Hard parse failure (no <page> blocks) → fixer returns None
    rather than failing the whole propose task."""
    from dikw_core.domains.knowledge.lint_fixers.non_atomic_page import (
        NonAtomicPageFixer,
    )

    wiki_root, page, issue = _make_fat_page_on_disk(tmp_path)
    fake = FakeLLM(response_text="LLM refused to split.")
    fixer = NonAtomicPageFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(
            pages=[page],
            wiki_root=wiki_root,
            llm=fake,
            enable_llm=True,
            cfg=_default_cfg(),
        ),
        reporter=_NullReporter(),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_non_atomic_page_skips_when_llm_disabled(tmp_path: Path) -> None:
    """Without ``enable_llm``, the fixer cannot run (no heuristic-only
    path makes sense for this rule) — must skip silently."""
    from dikw_core.domains.knowledge.lint_fixers.non_atomic_page import (
        NonAtomicPageFixer,
    )

    wiki_root, page, issue = _make_fat_page_on_disk(tmp_path)
    fake = FakeLLM(response_text=_NON_ATOMIC_LLM_RESPONSE)
    fixer = NonAtomicPageFixer()
    proposal = await fixer.propose(
        issue,
        _ctx(
            pages=[page],
            wiki_root=wiki_root,
            llm=fake,
            enable_llm=False,
            cfg=_default_cfg(),
        ),
        reporter=_NullReporter(),
    )
    assert proposal is None
    assert fake.last_user is None  # no LLM call
