"""Per-page ``lint:`` frontmatter suppression.

A page can declare ``lint: {skip: [<kind>, ...], reason: ...}`` to tell
``run_lint`` to skip specified rules for that page. The skipped issue
still surfaces in ``LintReport.acknowledged_leaves`` (path-keyed) so
operators can audit which pages they've intentionally exempted.

The intended use is the ``mark_as_leaf`` orphan-page fixer (Step 3): a
user accepts a page as a valid terminal note, and the fixer writes the
frontmatter once so the next lint pass doesn't keep reporting it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.domains.data.path_norm import doc_id_for
from dikw_core.domains.knowledge.lint import run_lint
from dikw_core.domains.knowledge.page import build_page, write_page
from dikw_core.schemas import DocumentRecord, Layer

from .fakes import init_test_base


async def _seed_page(
    *,
    base_root: Path,
    title: str,
    body: str,
    extras: dict | None = None,
) -> str:
    page = build_page(
        title=title,
        body=body,
        category="concept",
        tags=[],
        sources=[],
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
async def test_orphan_page_suppressed_by_frontmatter(empty_wiki: Path) -> None:
    """A page with ``lint: {skip: [orphan_page]}`` must not appear in
    the orphan_page issue list. The acknowledged_leaves bucket records
    the path so it stays visible in audit / `dikw client lint --format json`."""
    path = await _seed_page(
        base_root=empty_wiki,
        title="Intentional Leaf",
        body="# Intentional Leaf\n\nValid terminal note.\n",
        extras={"lint": {"skip": ["orphan_page"], "reason": "valid leaf note"}},
    )
    report = await _run_lint(empty_wiki)
    orphan_issues = [i for i in report.issues if i.kind == "orphan_page"]
    assert all(i.path != path for i in orphan_issues), (
        "page with lint.skip[orphan_page] still got reported as orphan"
    )
    assert path in report.acknowledged_leaves


@pytest.mark.asyncio
async def test_other_rules_still_apply_when_only_orphan_skipped(
    empty_wiki: Path,
) -> None:
    """Skipping ``orphan_page`` must not silence other rules. A page
    with both no inbound links AND a broken wikilink should still
    report the broken_wikilink issue, just not the orphan one."""
    path = await _seed_page(
        base_root=empty_wiki,
        title="Leaf With Broken Link",
        body="# Leaf With Broken Link\n\nSee [[Missing Target]].\n",
        extras={"lint": {"skip": ["orphan_page"]}},
    )
    report = await _run_lint(empty_wiki)
    kinds = {(i.kind, i.path) for i in report.issues}
    assert ("orphan_page", path) not in kinds
    assert ("broken_wikilink", path) in kinds


@pytest.mark.asyncio
async def test_malformed_lint_frontmatter_is_ignored(empty_wiki: Path) -> None:
    """Non-dict / non-list / non-string entries must be skipped silently
    (no rule suppression, no crash). The frontmatter is user-editable,
    so robustness matters more than strict validation."""
    path = await _seed_page(
        base_root=empty_wiki,
        title="Bad Lint Block",
        body="# Bad Lint Block\n\nValid body.\n",
        # Various bad shapes: list at top instead of dict; skip as
        # comma-string instead of list; an int kind. None should
        # accidentally suppress a real rule.
        extras={"lint": ["this is wrong"]},
    )
    report = await _run_lint(empty_wiki)
    orphan_issues = [i for i in report.issues if i.kind == "orphan_page"]
    assert any(i.path == path for i in orphan_issues), (
        "malformed lint block should not have suppressed the orphan rule"
    )
    assert path not in report.acknowledged_leaves


@pytest.mark.asyncio
async def test_acknowledged_leaves_empty_when_no_suppression(
    empty_wiki: Path,
) -> None:
    """A wiki with no ``lint:`` frontmatter anywhere reports an empty
    acknowledged_leaves list — the bucket exists by default but stays
    empty unless a page opts in."""
    await _seed_page(
        base_root=empty_wiki,
        title="Plain Page",
        body="# Plain Page\n\nbody.\n",
    )
    report = await _run_lint(empty_wiki)
    assert report.acknowledged_leaves == []
