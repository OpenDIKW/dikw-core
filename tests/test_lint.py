"""Tests for ``run_lint`` — focused on the ``non_atomic_page`` heuristic.

The other three issue kinds (``broken_wikilink`` / ``orphan_page`` /
``duplicate_title``) are exercised end-to-end by
``test_synthesize_pipeline.py``; this file pins down the atomicity
backstop in isolation so threshold tweaks are easy to validate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.domains.knowledge.lint import run_lint
from dikw_core.domains.knowledge.wiki import build_page, write_page
from dikw_core.schemas import DocumentRecord, Layer

from .fakes import init_test_wiki


async def _seed_page(
    *,
    wiki_root: Path,
    title: str,
    body: str,
    type_: str = "concept",
    tags: list[str] | None = None,
) -> str:
    """Write ``page`` to disk + register it in storage so lint can see it."""
    page = build_page(
        title=title,
        body=body,
        type_=type_,
        tags=list(tags or []),
        sources=[],
        path=None,
        extras={},
    )
    write_page(wiki_root, page)

    _cfg, _root, storage = await api._with_storage(wiki_root)
    try:
        await storage.upsert_document(
            DocumentRecord(
                doc_id=f"K:{page.path}",
                path=page.path,
                title=page.title,
                hash=f"hash-{page.path}",
                mtime=0.0,
                layer=Layer.WIKI,
                active=True,
            )
        )
    finally:
        await storage.close()
    return page.path


async def _run_lint(wiki_root: Path):
    _cfg, root, storage = await api._with_storage(wiki_root)
    try:
        return await run_lint(storage, root=root)
    finally:
        await storage.close()


@pytest.fixture()
def empty_wiki(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    return wiki


@pytest.mark.asyncio
async def test_atomic_page_emits_no_non_atomic_issue(empty_wiki: Path) -> None:
    body = (
        "# Tight Page\n\n"
        "Two short paragraphs about a single subject.\n\n"
        "Linked to [[Some Other Concept]] for context.\n"
    )
    await _seed_page(wiki_root=empty_wiki, title="Tight Page", body=body)
    report = await _run_lint(empty_wiki)
    kinds = report.by_kind()
    assert kinds.get("non_atomic_page", 0) == 0


@pytest.mark.asyncio
async def test_long_body_triggers_non_atomic(empty_wiki: Path) -> None:
    body = "# Long Page\n\n" + ("paragraph filler " * 200) + "\n"
    await _seed_page(wiki_root=empty_wiki, title="Long Page", body=body)
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert len(issues) == 1
    assert "body" in issues[0].detail and "chars" in issues[0].detail


@pytest.mark.asyncio
async def test_many_h2_sections_trigger_non_atomic(empty_wiki: Path) -> None:
    body = (
        "# Multi-section Page\n\n"
        "intro\n\n"
        "## Section One\n\nbody\n\n"
        "## Section Two\n\nbody\n\n"
        "## Section Three\n\nbody\n\n"
        "## Section Four\n\nbody\n"
    )
    await _seed_page(
        wiki_root=empty_wiki, title="Multi-section Page", body=body
    )
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert len(issues) == 1
    assert "H2 sections" in issues[0].detail


@pytest.mark.asyncio
async def test_many_wikilinks_trigger_non_atomic(empty_wiki: Path) -> None:
    body = (
        "# Hub Page\n\n"
        "References [[A]], [[B]], [[C]], [[D]], [[E]], [[F]], "
        "[[G]], [[H]], [[I]] all in one breath.\n"
    )
    await _seed_page(wiki_root=empty_wiki, title="Hub Page", body=body)
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert len(issues) == 1
    assert "wikilinks" in issues[0].detail


@pytest.mark.asyncio
async def test_tags_across_domains_trigger_non_atomic(empty_wiki: Path) -> None:
    """A page tagged across two top-level namespaces (ml/* and biz/*) is
    almost certainly N atomic notes glued together. The Stage A prompt
    can't easily catch this — lint as a backstop must."""
    body = (
        "# Cross-Domain Page\n\n"
        "Discusses both ML research and startup operations in one breath.\n"
    )
    await _seed_page(
        wiki_root=empty_wiki,
        title="Cross-Domain Page",
        body=body,
        tags=["ml/research", "biz/startup"],
    )
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert len(issues) == 1
    assert "tags span" in issues[0].detail


@pytest.mark.asyncio
async def test_tags_single_domain_does_not_trigger(empty_wiki: Path) -> None:
    """Multiple tags within ONE top-level namespace is normal taxonomy
    practice (ml/research + ml/eval), not a cross-domain violation."""
    body = "# Single-Domain Page\n\nA focused page on ML research.\n"
    await _seed_page(
        wiki_root=empty_wiki,
        title="Single-Domain Page",
        body=body,
        tags=["ml/research", "ml/eval"],
    )
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert issues == []


@pytest.mark.asyncio
async def test_no_tags_does_not_trigger_tag_violation(empty_wiki: Path) -> None:
    """Empty tag list must never trigger the cross-domain heuristic."""
    body = "# Untagged Page\n\nNo frontmatter tags.\n"
    await _seed_page(wiki_root=empty_wiki, title="Untagged Page", body=body, tags=[])
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert issues == []


@pytest.mark.asyncio
async def test_tags_no_namespace_treats_each_as_domain(empty_wiki: Path) -> None:
    """Flat tags without ``/`` separators are treated as separate domains
    each — being conservative pushes the user toward namespaced tags."""
    body = "# Flat-Tagged Page\n\nMixes unrelated flat tags.\n"
    await _seed_page(
        wiki_root=empty_wiki,
        title="Flat-Tagged Page",
        body=body,
        tags=["foo", "bar"],
    )
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert len(issues) == 1
    assert "tags span" in issues[0].detail


@pytest.mark.asyncio
async def test_multiple_violations_collapse_to_one_issue(empty_wiki: Path) -> None:
    """A page that trips multiple heuristics should yield ONE issue with
    all violations in detail — avoid drowning the report in duplicates."""
    body = (
        "# Mega Page\n\n"
        + ("paragraph filler " * 200)
        + "\n\n## A\n\nbody\n\n## B\n\nbody\n\n## C\n\nbody\n\n## D\n\nbody\n\n"
        + "Links: [[a]] [[b]] [[c]] [[d]] [[e]] [[f]] [[g]] [[h]] [[i]].\n"
    )
    await _seed_page(wiki_root=empty_wiki, title="Mega Page", body=body)
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert len(issues) == 1
    assert "body" in issues[0].detail
    assert "H2 sections" in issues[0].detail
    assert "wikilinks" in issues[0].detail
