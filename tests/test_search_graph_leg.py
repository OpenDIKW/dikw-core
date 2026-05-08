"""HybridSearcher's optional 4th leg: K-layer wikilink graph expansion.

Verifies that when ``graph_enabled=True`` the searcher walks one hop via
``Storage.neighbor_chunks_via_links`` from the BM25/vector top-K, folds
the neighbors into the fused ranking with ``graph_weight``, and leaves
the historical 3-leg behaviour byte-identical when the flag is off.

Parameterised over SQLite + Postgres backends — the graph leg is built
on storage primitives, so PG must produce the same hits as SQLite.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from dikw_core.domains.info.search import HybridSearcher
from dikw_core.schemas import (
    ChunkRecord,
    DocumentRecord,
    Layer,
    LinkRecord,
    LinkType,
)
from dikw_core.storage.base import Storage
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


@pytest.fixture(
    params=[
        pytest.param("sqlite", id="sqlite"),
        pytest.param(
            "postgres",
            id="postgres",
            marks=pytest.mark.skipif(
                not os.environ.get("DIKW_TEST_POSTGRES_DSN"),
                reason="Postgres adapter tests require DIKW_TEST_POSTGRES_DSN",
            ),
        ),
    ]
)
async def storage(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[Storage]:
    backend = request.param
    if backend == "sqlite":
        s: Storage = SQLiteStorage(tmp_path / "test.sqlite", cjk_tokenizer="jieba")
        schema: str | None = None
    elif backend == "postgres":
        from dikw_core.storage.postgres import PostgresStorage

        dsn = os.environ["DIKW_TEST_POSTGRES_DSN"]
        schema = f"dikw_test_{abs(hash(str(tmp_path))) % 10_000_000:07d}"
        s = PostgresStorage(dsn, schema=schema, pool_size=2, cjk_tokenizer="jieba")
    else:
        raise RuntimeError(f"unreachable: adapter {backend}")

    await s.connect()
    await s.migrate()
    try:
        yield s
    finally:
        if backend == "postgres":
            from psycopg import AsyncConnection

            conn = await AsyncConnection.connect(os.environ["DIKW_TEST_POSTGRES_DSN"])
            try:
                async with conn.cursor() as cur:
                    await cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
                await conn.commit()
            finally:
                await conn.close()
        await s.close()


@pytest.fixture()
async def linked_wiki(storage):
    """Three K-layer pages: A links to B and C via wikilinks. Bodies
    are arranged so a search for ``"alpha"`` matches A only — making the
    extra B/C hits a clear graph-leg signal."""
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
