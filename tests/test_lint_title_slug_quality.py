"""``title_slug_quality`` lint — deterministic K-page title/slug hygiene.

Three zero-false-positive sub-cases, none of which fire on correct synth
output (synth always emits a ``# Title`` H1, writes a matching frontmatter
``title:``, and — for non-ASCII titles — an ASCII/pinyin slug):

* **missing / empty / punctuation-only** ``# Title`` heading in the body.
* **frontmatter ``title:`` != body ``# H1``** — the genuine title drift
  (``write_page`` always writes both equal; a hand-edit to one diverges them).
* **degenerate slug** — the filename stem is the ``untitled`` fallback,
  which only happens when ``slugify`` collapsed a non-ASCII title that the
  LLM gave no ASCII/pinyin slug for.

What is deliberately NOT here: any ``slugify(title) == stem`` comparison.
Slugs are LLM-chosen and intentionally differ from ``slugify(title)``
(``The DIKW Pyramid`` -> ``dikw-pyramid``; ``神经网络`` -> ``shen-jing-wang-luo``),
and wikilinks resolve by title not slug, so that comparison would red-flag
the engine's own correct output. "Is this title too generic?" is a
probabilistic judgement left to the LLM-judge leg, never this lexical lint.

The pure helper ``check_title_slug_quality`` is unit-tested in isolation;
``run_lint`` integration confirms wiring, KNOWLEDGE-layer scoping, and
``lint: {skip: [...]}`` suppression.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.domains.data.path_norm import doc_id_for
from dikw_core.domains.knowledge.lint import check_title_slug_quality, run_lint
from dikw_core.domains.knowledge.page import build_page, write_page
from dikw_core.schemas import DocumentRecord, Layer

from .fakes import FakeEmbeddings, ingest_wisdom_files, init_test_base

# ---- pure helper: the three sub-cases + the no-false-positive regressions ----


def test_clean_title_and_slug_has_no_violations() -> None:
    assert (
        check_title_slug_quality(
            body="# DIKW Pyramid\n\nFour layers.",
            frontmatter_title="DIKW Pyramid",
            stem="dikw-pyramid",
        )
        == ()
    )


def test_missing_h1_heading_flagged() -> None:
    out = check_title_slug_quality(
        body="No heading here, just a paragraph.\n",
        frontmatter_title="Something",
        stem="something",
    )
    assert len(out) == 1
    assert "no usable" in out[0].lower()


def test_blank_h1_heading_flagged_as_missing() -> None:
    # ``# `` with no text never matches the capturing H1 regex, so it reads
    # as an absent heading rather than a separate "empty" case.
    out = check_title_slug_quality(
        body="# \n\nbody text\n",
        frontmatter_title=None,
        stem="ok-slug",
    )
    assert len(out) == 1
    assert "no usable" in out[0].lower()


def test_punctuation_only_h1_flagged() -> None:
    out = check_title_slug_quality(
        body="# ...\n\nbody\n",
        frontmatter_title=None,
        stem="ok-slug",
    )
    assert len(out) == 1
    assert "punctuation" in out[0].lower()


def test_cjk_h1_is_not_punctuation_only() -> None:
    # CJK characters are Unicode word characters — a Chinese title must NOT
    # be mistaken for a punctuation-only heading.
    assert (
        check_title_slug_quality(
            body="# 神经网络\n\n正文。",
            frontmatter_title="神经网络",
            stem="shen-jing-wang-luo",
        )
        == ()
    )


def test_frontmatter_title_mismatch_flagged() -> None:
    out = check_title_slug_quality(
        body="# Different Heading\n\nbody\n",
        frontmatter_title="Canonical Name",
        stem="canonical-name",
    )
    assert len(out) == 1
    assert "Canonical Name" in out[0]
    assert "Different Heading" in out[0]


def test_frontmatter_title_match_is_clean() -> None:
    assert (
        check_title_slug_quality(
            body="# Same\n\nbody\n",
            frontmatter_title="Same",
            stem="same",
        )
        == ()
    )


def test_absent_frontmatter_title_skips_mismatch_leg() -> None:
    # A page with a good H1 but no frontmatter title (hand-written) must not
    # trip the mismatch leg — there is nothing to disagree with.
    assert (
        check_title_slug_quality(
            body="# Standalone\n\nbody\n",
            frontmatter_title=None,
            stem="standalone",
        )
        == ()
    )


def test_untitled_stem_flagged() -> None:
    out = check_title_slug_quality(
        body="# 神经网络\n\n正文。",
        frontmatter_title="神经网络",
        stem="untitled",
    )
    assert len(out) == 1
    assert "untitled" in out[0].lower()


def test_untitled_counter_stem_is_not_flagged() -> None:
    # There is no ``-NNN`` collision suffix in the engine (same-slug pages are
    # merged, never counter-suffixed), so a hand-created ``untitled-1`` stem is
    # NOT the degenerate fallback and must not be flagged — only bare
    # ``untitled`` is.
    assert (
        check_title_slug_quality(
            body="# 机器学习\n\n正文。",
            frontmatter_title="机器学习",
            stem="untitled-1",
        )
        == ()
    )


def test_atx_closing_hashes_agree_with_frontmatter_title() -> None:
    # CommonMark lets a heading carry a closing hash sequence (``# Title #``).
    # synthesize.py's ``_ATX_TITLE`` strips it when deriving the frontmatter
    # ``title:``, so the body H1 extractor here must strip it too — otherwise
    # the title-drift leg false-fires on the engine's own correct output.
    assert (
        check_title_slug_quality(
            body="# Reward shaping #\n\nbody\n",
            frontmatter_title="Reward shaping",
            stem="reward-shaping",
        )
        == ()
    )
    assert (
        check_title_slug_quality(
            body="# Reward shaping ###\n\nbody\n",
            frontmatter_title="Reward shaping",
            stem="reward-shaping",
        )
        == ()
    )


def test_internal_hash_in_title_is_preserved() -> None:
    # A ``#`` that is NOT a trailing closing sequence stays part of the title,
    # matching ``_ATX_TITLE`` — so a body/frontmatter pair that both keep it
    # does not false-fire.
    assert (
        check_title_slug_quality(
            body="# Section #2 overview\n\nbody\n",
            frontmatter_title="Section #2 overview",
            stem="section-2-overview",
        )
        == ()
    )


def test_stopword_dropping_ascii_slug_is_clean() -> None:
    # ``slugify('The DIKW Pyramid')`` == 'the-dikw-pyramid', but the LLM chose
    # 'dikw-pyramid'. This intentional divergence must NOT be flagged.
    assert (
        check_title_slug_quality(
            body="# The DIKW Pyramid\n\nbody\n",
            frontmatter_title="The DIKW Pyramid",
            stem="dikw-pyramid",
        )
        == ()
    )


def test_fenced_code_heading_is_not_the_title() -> None:
    # A ``# comment`` inside a fenced code block must not be read as the page
    # heading — the real H1 below it is.
    body = "```bash\n# install deps\nuv sync\n```\n\n# Real Title\n\nbody\n"
    assert (
        check_title_slug_quality(
            body=body, frontmatter_title="Real Title", stem="real-title"
        )
        == ()
    )


def test_fenced_code_heading_only_reads_as_missing() -> None:
    body = "```bash\n# install deps\nuv sync\n```\n\nplain paragraph, no heading\n"
    out = check_title_slug_quality(
        body=body, frontmatter_title=None, stem="ok-slug"
    )
    assert len(out) == 1
    assert "no usable" in out[0].lower()


def test_multiple_violations_all_reported() -> None:
    out = check_title_slug_quality(
        body="no heading at all\n",
        frontmatter_title="Ghost",
        stem="untitled",
    )
    # missing H1 + degenerate slug (mismatch leg can't fire without an H1).
    assert len(out) == 2


# ---- run_lint integration: wiring, scoping, suppression ----------------------


async def _seed_page(
    *,
    base_root: Path,
    title: str,
    body: str,
    path: str | None = None,
    extras: dict | None = None,
) -> str:
    page = build_page(
        title=title,
        body=body,
        category="concept",
        tags=[],
        sources=[],
        path=path,
        extras=extras or {},
    )
    write_page(base_root, page)
    _cfg, _root, storage = await api._with_storage(base_root)
    try:
        await storage.upsert_document(
            DocumentRecord(
                doc_id=doc_id_for(Layer.KNOWLEDGE, page.path),
                path=page.path,
                title=page.title,
                hash=f"hash-{page.path}",
                mtime=0.0,
                layer=Layer.KNOWLEDGE,
                active=True,
            )
        )
    finally:
        await storage.close()
    return page.path


async def _run_lint(base_root: Path):
    _cfg, root, storage = await api._with_storage(base_root)
    try:
        return await run_lint(storage, root=root)
    finally:
        await storage.close()


@pytest.fixture()
def empty_wiki(tmp_path: Path) -> Path:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    return wiki


@pytest.mark.asyncio
async def test_clean_page_emits_no_title_slug_issue(empty_wiki: Path) -> None:
    await _seed_page(
        base_root=empty_wiki,
        title="Tidy Page",
        body="# Tidy Page\n\nA single subject, well titled.\n",
    )
    report = await _run_lint(empty_wiki)
    assert report.by_kind().get("title_slug_quality", 0) == 0


@pytest.mark.asyncio
async def test_frontmatter_title_drift_surfaces(empty_wiki: Path) -> None:
    # build_page writes ``title: Canonical Name`` to frontmatter, body keeps
    # the divergent H1 — a realistic hand-edit drift.
    path = await _seed_page(
        base_root=empty_wiki,
        title="Canonical Name",
        body="# Different Heading\n\nbody\n",
    )
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "title_slug_quality"]
    assert len(issues) == 1
    assert issues[0].path == path
    assert "Canonical Name" in issues[0].detail


@pytest.mark.asyncio
async def test_untitled_slug_surfaces(empty_wiki: Path) -> None:
    path = await _seed_page(
        base_root=empty_wiki,
        title="神经网络",
        body="# 神经网络\n\n正文。\n",
        path="knowledge/concept/untitled.md",
    )
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "title_slug_quality"]
    assert len(issues) == 1
    assert issues[0].path == path
    assert "untitled" in issues[0].detail.lower()


@pytest.mark.asyncio
async def test_skip_frontmatter_suppresses_title_slug_quality(
    empty_wiki: Path,
) -> None:
    path = await _seed_page(
        base_root=empty_wiki,
        title="神经网络",
        body="# 神经网络\n\n正文。\n",
        path="knowledge/concept/untitled.md",
        extras={"lint": {"skip": ["title_slug_quality"], "reason": "known"}},
    )
    report = await _run_lint(empty_wiki)
    assert report.by_kind().get("title_slug_quality", 0) == 0
    assert path in report.acknowledged_leaves


@pytest.mark.asyncio
async def test_wisdom_page_is_not_title_slug_checked(tmp_path: Path) -> None:
    # title_slug_quality is a synth-output (K-layer) quality gate; hand-written
    # wisdom pages may legitimately carry a frontmatter title distinct from the
    # body H1, so the rule must not fire on Layer.WISDOM.
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    wpath = wiki / "wisdom" / "me" / "note.md"
    wpath.parent.mkdir(parents=True, exist_ok=True)
    wpath.write_text(
        "---\ntitle: Frontmatter Title\n---\n# Body Heading\n\nbody\n",
        encoding="utf-8",
    )
    await ingest_wisdom_files(
        wiki, ["wisdom/me/note.md"], embedder=FakeEmbeddings()
    )
    report = await _run_lint(wiki)
    assert report.by_kind().get("title_slug_quality", 0) == 0
