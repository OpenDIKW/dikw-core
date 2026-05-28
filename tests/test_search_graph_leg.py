"""HybridSearcher's optional 4th leg: K-layer knowledgelink graph expansion.

Verifies that when ``graph_enabled=True`` the searcher walks one hop via
``Storage.neighbor_chunks_via_links`` from the BM25/vector top-K, folds
the neighbors into the fused ranking with ``graph_weight``, and leaves
the historical 3-leg behaviour byte-identical when the flag is off.

Parameterised over SQLite + Postgres backends — the graph leg is built
on storage primitives, so PG must produce the same hits as SQLite.
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


def _doc(path: str) -> DocumentRecord:
    return DocumentRecord(
        doc_id=f"K::{path}",
        path=path,
        title=path.rsplit("/", 1)[-1].rstrip(".md"),
        hash=f"hash-{path}",
        mtime=time.time(),
        layer=Layer.KNOWLEDGE,
        active=True,
    )


@pytest.fixture()
async def linked_wiki(parametrized_storage):
    """Three K-layer pages: A links to B and C via wikilinks. Bodies
    are arranged so a search for ``"alpha"`` matches A only — making the
    extra B/C hits a clear graph-leg signal."""
    storage = parametrized_storage
    page_a = _doc("knowledge/concepts/alpha.md")
    page_b = _doc("knowledge/concepts/bravo.md")
    page_c = _doc("knowledge/concepts/charlie.md")
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

    for dst in ("knowledge/concepts/bravo.md", "knowledge/concepts/charlie.md"):
        await storage.upsert_link(
            LinkRecord(
                src_doc_id=page_a.doc_id,
                dst_path=dst,
                link_type=LinkType.WIKILINK,
                anchor=None,
                line=1,
            )
        )

    return {
        "storage": storage,
        "a_chunk": a_chunks[0],
        "b_chunk": b_chunks[0],
        "c_chunk": c_chunks[0],
    }


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


@pytest.mark.asyncio
async def test_text_match_seed_ranks_above_graph_only_neighbor(linked_wiki) -> None:
    """The graph leg must augment without overpowering text matches.
    If graph_weight pushes graph-only neighbors above the BM25-matching
    seed, the recall improvement comes at the cost of precision."""
    storage = linked_wiki["storage"]
    searcher = HybridSearcher(storage, embedder=None, graph_enabled=True)
    hits = await searcher.search("alpha", limit=10)
    chunk_ids = [h.chunk_id for h in hits]
    a_idx = chunk_ids.index(linked_wiki["a_chunk"])
    b_idx = chunk_ids.index(linked_wiki["b_chunk"])
    assert a_idx < b_idx, (
        f"BM25-matching seed (A) must rank above graph-only neighbor (B); "
        f"got A@{a_idx} vs B@{b_idx} — graph_weight may be too aggressive"
    )


@pytest.mark.asyncio
async def test_prod_weights_keep_bm25_above_graph_only(linked_wiki) -> None:
    """Same precision invariant under production ``RetrievalConfig``
    defaults (bm25_weight=0.3, graph_weight matched). Without this
    pin, raising ``graph_weight`` above ``bm25_weight`` lets a
    rank-1 graph-only neighbor outscore an exact BM25 match
    (0.5/61 vs 0.3/61) when the vector leg is absent or misses."""
    from dikw_core.config import RetrievalConfig

    cfg = RetrievalConfig()
    storage = linked_wiki["storage"]
    searcher = HybridSearcher(
        storage,
        embedder=None,
        graph_enabled=True,
        bm25_weight=cfg.bm25_weight,
        vector_weight=cfg.vector_weight,
        graph_weight=cfg.graph_weight,
    )
    hits = await searcher.search("alpha", limit=10)
    chunk_ids = [h.chunk_id for h in hits]
    a_idx = chunk_ids.index(linked_wiki["a_chunk"])
    b_idx = chunk_ids.index(linked_wiki["b_chunk"])
    assert a_idx < b_idx, (
        f"BM25-matching seed (A) must rank above graph-only neighbor (B) "
        f"under prod weights; got A@{a_idx} vs B@{b_idx} — "
        f"graph_weight {cfg.graph_weight} must be ≤ bm25_weight {cfg.bm25_weight}"
    )


@pytest.mark.asyncio
async def test_graph_weight_zero_treats_leg_as_off(linked_wiki) -> None:
    """``graph_weight=0`` must opt out cleanly — appending a zero-weight
    leg lets graph-only chunks land at score 0 (visible whenever limit
    exceeds positive-score hits) and can break CombMNZ's leg-count
    accounting. Treating zero as "leg off" matches user intent and
    keeps fusion math safe."""
    storage = linked_wiki["storage"]
    searcher = HybridSearcher(
        storage, embedder=None, graph_enabled=True, graph_weight=0.0
    )
    hits = await searcher.search("alpha", limit=10)
    chunk_ids = [h.chunk_id for h in hits]
    assert linked_wiki["a_chunk"] in chunk_ids
    assert linked_wiki["b_chunk"] not in chunk_ids
    assert linked_wiki["c_chunk"] not in chunk_ids


@pytest.mark.asyncio
async def test_layer_filter_propagates_to_graph_leg(parametrized_storage) -> None:
    """The graph leg must respect the same layer filter as FTS/vector.
    Otherwise a wikilink from a K-layer page to a D-layer source would
    leak across the layer boundary on a ``layer=Layer.KNOWLEDGE`` search."""
    storage = parametrized_storage
    page_a = _doc("knowledge/concepts/alpha.md")
    source_b = DocumentRecord(
        doc_id="D::sources/bravo.md",
        path="sources/bravo.md",
        title="bravo",
        hash="hash-source-bravo",
        mtime=time.time(),
        layer=Layer.SOURCE,
        active=True,
    )
    for d in (page_a, source_b):
        await storage.upsert_document(d)
    a_chunks = await storage.replace_chunks(
        page_a.doc_id,
        [ChunkRecord(doc_id=page_a.doc_id, seq=0, start=0, end=30, text="alpha alpha alpha")],
    )
    b_chunks = await storage.replace_chunks(
        source_b.doc_id,
        [ChunkRecord(doc_id=source_b.doc_id, seq=0, start=0, end=30, text="bravo bravo unrelated")],
    )
    await storage.upsert_link(
        LinkRecord(
            src_doc_id=page_a.doc_id,
            dst_path="sources/bravo.md",
            link_type=LinkType.WIKILINK,
            anchor=None,
            line=1,
        )
    )
    searcher = HybridSearcher(storage, embedder=None, graph_enabled=True)
    hits = await searcher.search("alpha", limit=10, layer=Layer.KNOWLEDGE)
    chunk_ids = [h.chunk_id for h in hits]
    assert a_chunks[0] in chunk_ids
    assert b_chunks[0] not in chunk_ids, (
        "graph leg leaked a Layer.SOURCE neighbor under layer=Layer.KNOWLEDGE"
    )


@pytest.mark.asyncio
async def test_graph_leg_skipped_in_bm25_mode(linked_wiki) -> None:
    """Single-leg modes (``bm25`` / ``vector``) are diagnostic
    ablations used by ``dikw client eval --retrieval all`` to compare against
    published baselines. The graph leg must NOT activate in those
    modes — even with ``graph_enabled=True`` — or the bm25 / vector
    numbers stop being pure-leg measurements. ``vector`` mode shares
    the same gate but requires a real embedder; covering ``bm25``
    here gives the gate sufficient coverage."""
    storage = linked_wiki["storage"]
    searcher = HybridSearcher(storage, embedder=None, graph_enabled=True)
    hits = await searcher.search("alpha", limit=10, mode="bm25")
    chunk_ids = [h.chunk_id for h in hits]
    assert linked_wiki["a_chunk"] in chunk_ids
    assert linked_wiki["b_chunk"] not in chunk_ids
    assert linked_wiki["c_chunk"] not in chunk_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("fusion", ["rrf", "combsum", "combmnz"])
async def test_graph_leg_works_with_all_fusion_modes(linked_wiki, fusion) -> None:
    """Graph leg must integrate cleanly with all three fusion algorithms,
    not just RRF. CombSUM/CombMNZ require score lists, RRF requires rank
    lists — the graph leg adapter inside HybridSearcher must produce
    whichever shape the active fusion expects."""
    storage = linked_wiki["storage"]
    searcher = HybridSearcher(
        storage, embedder=None, graph_enabled=True, fusion=fusion
    )
    hits = await searcher.search("alpha", limit=10)
    chunk_ids = [h.chunk_id for h in hits]
    assert linked_wiki["a_chunk"] in chunk_ids
    assert linked_wiki["b_chunk"] in chunk_ids
    assert linked_wiki["c_chunk"] in chunk_ids
