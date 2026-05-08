"""HybridSearcher's optional 4th leg: K-layer wikilink graph expansion.

Verifies that when ``graph_enabled=True`` the searcher walks one hop via
``Storage.neighbor_chunks_via_links`` from the BM25/vector top-K, folds
the neighbors into the fused ranking with ``graph_weight``, and leaves
the historical 3-leg behaviour byte-identical when the flag is off.
"""

from __future__ import annotations

import time

import pytest

from dikw_core.domains.info.search import HybridSearcher
from dikw_core.schemas import (
    ChunkRecord,
    DocumentRecord,
    Layer,
    LinkRecord,
    LinkType,
)
from dikw_core.storage.sqlite import SQLiteStorage


def _doc(path: str) -> DocumentRecord:
    return DocumentRecord(
        doc_id=f"K::{path}",
        path=path,
        title=path.rsplit("/", 1)[-1].rstrip(".md"),
        hash=f"hash-{path}",
        mtime=time.time(),
        layer=Layer.WIKI,
        active=True,
    )


@pytest.fixture()
async def linked_wiki(tmp_path):
    """Three K-layer pages: A links to B and C via wikilinks. Bodies
    are arranged so a search for ``"alpha"`` matches A only — making the
    extra B/C hits a clear graph-leg signal."""
    storage = SQLiteStorage(tmp_path / "test.sqlite")
    await storage.connect()
    await storage.migrate()

    page_a = _doc("wiki/concepts/alpha.md")
    page_b = _doc("wiki/concepts/bravo.md")
    page_c = _doc("wiki/concepts/charlie.md")
    for d in (page_a, page_b, page_c):
        await storage.upsert_document(d)

    a_chunks = await storage.replace_chunks(
        page_a.doc_id,
        [
            ChunkRecord(
                doc_id=page_a.doc_id,
                seq=0,
                start=0,
                end=30,
                text="alpha alpha alpha background",
            )
        ],
    )
    b_chunks = await storage.replace_chunks(
        page_b.doc_id,
        [
            ChunkRecord(
                doc_id=page_b.doc_id,
                seq=0,
                start=0,
                end=30,
                text="bravo bravo unrelated body",
            )
        ],
    )
    c_chunks = await storage.replace_chunks(
        page_c.doc_id,
        [
            ChunkRecord(
                doc_id=page_c.doc_id,
                seq=0,
                start=0,
                end=30,
                text="charlie charlie disjoint",
            )
        ],
    )

    for dst in ("wiki/concepts/bravo.md", "wiki/concepts/charlie.md"):
        await storage.upsert_link(
            LinkRecord(
                src_doc_id=page_a.doc_id,
                dst_path=dst,
                link_type=LinkType.WIKILINK,
                anchor=None,
                line=1,
            )
        )

    yield {
        "storage": storage,
        "a_chunk": a_chunks[0],
        "b_chunk": b_chunks[0],
        "c_chunk": c_chunks[0],
    }
    await storage.close()


@pytest.mark.asyncio
async def test_graph_disabled_returns_only_text_match(linked_wiki) -> None:
    storage = linked_wiki["storage"]
    searcher = HybridSearcher(storage, embedder=None, graph_enabled=False)
    hits = await searcher.search("alpha", limit=10)
    chunk_ids = [h.chunk_id for h in hits]
    assert linked_wiki["a_chunk"] in chunk_ids
    assert linked_wiki["b_chunk"] not in chunk_ids
    assert linked_wiki["c_chunk"] not in chunk_ids


@pytest.mark.asyncio
async def test_graph_enabled_pulls_in_wikilink_neighbors(linked_wiki) -> None:
    storage = linked_wiki["storage"]
    searcher = HybridSearcher(storage, embedder=None, graph_enabled=True)
    hits = await searcher.search("alpha", limit=10)
    chunk_ids = [h.chunk_id for h in hits]
    assert linked_wiki["a_chunk"] in chunk_ids, (
        "the BM25-matching seed must still rank — graph leg augments, not replaces"
    )
    assert linked_wiki["b_chunk"] in chunk_ids, (
        "wikilink target should surface via the graph leg"
    )
    assert linked_wiki["c_chunk"] in chunk_ids


@pytest.mark.asyncio
async def test_graph_seed_top_k_caps_seed_count(linked_wiki) -> None:
    """If only the top-1 seed is used, the graph leg still pulls in
    A's neighbors because A is the BM25 top hit. Verifies the cap path
    runs without error and still yields neighbors."""
    storage = linked_wiki["storage"]
    searcher = HybridSearcher(
        storage, embedder=None, graph_enabled=True, graph_seed_top_k=1
    )
    hits = await searcher.search("alpha", limit=10)
    chunk_ids = [h.chunk_id for h in hits]
    assert linked_wiki["b_chunk"] in chunk_ids


@pytest.mark.asyncio
async def test_graph_enabled_with_no_text_match_is_safe(linked_wiki) -> None:
    """A query that matches nothing in BM25 produces zero seeds → the
    graph leg silently contributes nothing rather than blowing up."""
    storage = linked_wiki["storage"]
    searcher = HybridSearcher(storage, embedder=None, graph_enabled=True)
    hits = await searcher.search("zzznosuchword", limit=10)
    assert hits == []
