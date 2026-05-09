"""``_persist_wiki_page`` link-set reconciliation contract.

Editing a wiki page to drop a ``[[wikilink]]`` must remove the
corresponding edge from storage. Without this, the ``links`` table
accumulates ghost edges as users edit pages — polluting the
graph-leg retrieval channel (``neighbor_chunks_via_links``) and
quietly breaking ``orphan_page`` / ``broken_wikilink`` lint
detection.

``test_links.py`` covers the parser; this file pins the
persistence-layer round-trip so the engine-side reconciliation
invariant can't regress silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.domains.knowledge.wiki import build_page, write_page
from dikw_core.schemas import Layer, LinkType

from .fakes import init_test_wiki


async def _persist(storage, root: Path, page) -> None:
    """Thin wrapper — embedder/version pinned to None so the test
    doesn't drag the whole embed pipeline in. ``_persist_wiki_page``
    skips embedding when either is None."""
    await api._persist_wiki_page(
        storage=storage,
        root=root,
        page=page,
        embedder=None,
        embedding_model="fake",
        text_version_id=None,
    )


@pytest.mark.asyncio
async def test_persist_wiki_page_drops_removed_wikilinks(tmp_path: Path) -> None:
    """Rewriting a page to swap ``[[Target A]]`` for ``[[Target B]]``
    must leave only the B edge in storage. Pre-fix this assertion
    fails because the A edge was never deleted."""
    wiki_root = tmp_path / "wiki"
    init_test_wiki(wiki_root)
    _cfg, root, storage = await api._with_storage(wiki_root)
    try:
        target_a = build_page(title="Target A", body="A body.\n", type_="concept")
        target_b = build_page(title="Target B", body="B body.\n", type_="concept")
        for p in (target_a, target_b):
            write_page(root, p)
            await _persist(storage, root, p)

        src_v1 = build_page(
            title="Src",
            body="See [[Target A]] for details.\n",
            type_="concept",
            path="wiki/src.md",
        )
        write_page(root, src_v1)
        await _persist(storage, root, src_v1)

        src_doc_id = api._doc_id_for(Layer.WIKI, "wiki/src.md")
        wikilinks_v1 = [
            link for link in await storage.links_from(src_doc_id)
            if link.link_type == LinkType.WIKILINK
        ]
        assert {link.dst_path for link in wikilinks_v1} == {target_a.path}

        # Rewrite the same page so the [[Target A]] reference is gone.
        src_v2 = build_page(
            title="Src",
            body="See [[Target B]] instead.\n",
            type_="concept",
            path="wiki/src.md",
        )
        write_page(root, src_v2)
        await _persist(storage, root, src_v2)

        wikilinks_v2 = [
            link for link in await storage.links_from(src_doc_id)
            if link.link_type == LinkType.WIKILINK
        ]
        # Without reconciliation, target_a still shows up here as a
        # ghost edge — that's the bug this test pins.
        assert {link.dst_path for link in wikilinks_v2} == {target_b.path}
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_persist_wiki_page_drops_all_wikilinks_when_body_loses_them(
    tmp_path: Path,
) -> None:
    """Rewriting a page to remove every ``[[wikilink]]`` must leave
    zero outgoing wikilink edges. Catches the edge case where a
    ``DELETE … WHERE src_doc_id = ?`` works for "swap A for B" but
    a different implementation (e.g. "delete edges that no longer
    appear in the new body") might miss the empty-target set."""
    wiki_root = tmp_path / "wiki"
    init_test_wiki(wiki_root)
    _cfg, root, storage = await api._with_storage(wiki_root)
    try:
        target = build_page(title="Target", body="t body.\n", type_="concept")
        write_page(root, target)
        await _persist(storage, root, target)

        src_v1 = build_page(
            title="Src",
            body="See [[Target]] for context.\n",
            type_="concept",
            path="wiki/src.md",
        )
        write_page(root, src_v1)
        await _persist(storage, root, src_v1)

        src_doc_id = api._doc_id_for(Layer.WIKI, "wiki/src.md")
        wikilinks_v1 = [
            link for link in await storage.links_from(src_doc_id)
            if link.link_type == LinkType.WIKILINK
        ]
        assert len(wikilinks_v1) == 1

        src_v2 = build_page(
            title="Src",
            body="Plain prose with no links anymore.\n",
            type_="concept",
            path="wiki/src.md",
        )
        write_page(root, src_v2)
        await _persist(storage, root, src_v2)

        wikilinks_v2 = [
            link for link in await storage.links_from(src_doc_id)
            if link.link_type == LinkType.WIKILINK
        ]
        assert wikilinks_v2 == []
    finally:
        await storage.close()
