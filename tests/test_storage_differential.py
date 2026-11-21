"""Stateful differential tests for the ``Storage`` Protocol.

Two complementary checks driven by Hypothesis-generated operation sequences
(``upsert_document``, ``replace_chunks``, ``replace_links_from``,
``replace_provenance_from``, ``deactivate_document``, ``delete_document``):

  1. **SQLite vs in-memory reference oracle** — runs in every fast suite pass
     with zero external services. Catches state-machine regressions across
     replace-never-union edges, ``active=False`` visibility, hard-delete
     cascades (preserving inbound links from other docs), and FTS sync.
  2. **SQLite vs Postgres differential parity** — runs whenever
     ``DIKW_TEST_POSTGRES_DSN`` is set (wired into CI's ``postgres-contract``
     job). Executes the exact same random operation sequence against both
     adapters and asserts identical observable state and FTS hit sets.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dikw_core.domains.data.path_norm import doc_id_for, normalize_path
from dikw_core.schemas import (
    ChunkRecord,
    DocumentRecord,
    Layer,
    LinkRecord,
    LinkType,
)
from dikw_core.storage.base import Storage
from dikw_core.storage.sqlite import SQLiteStorage

# Vocabulary of clean ASCII words that tokenize identically in SQLite FTS5
# (unicode61/jieba) and Postgres `simple` tsvector — avoids stemming/stopword
# divergence while still exercising multi-token FTS indexing & deletion.
_VOCAB = (
    "aurora",
    "beacon",
    "cobalt",
    "dynamo",
    "eclipse",
    "falcon",
    "glacier",
    "horizon",
    "isotope",
    "jupiter",
)

_DOC_SLOTS: tuple[tuple[Layer, str], ...] = (
    (Layer.SOURCE, "sources/alpha.md"),
    (Layer.SOURCE, "sources/beta.md"),
    (Layer.KNOWLEDGE, "knowledge/concept/gamma.md"),
    (Layer.KNOWLEDGE, "knowledge/entity/delta.md"),
    (Layer.WISDOM, "wisdom/me/epsilon.md"),
)

_TARGET_PATHS: tuple[str, ...] = (
    "knowledge/concept/gamma.md",
    "knowledge/entity/delta.md",
    "knowledge/concept/missing-target.md",
    "wisdom/me/epsilon.md",
)

_SOURCE_REFS: tuple[str, ...] = (
    "sources/alpha.md",
    "sources/Alpha.md",  # case-variant testing normalized dedup
    "sources/beta.md",
)


@dataclass(frozen=True)
class _Op:
    kind: Literal[
        "upsert_doc",
        "replace_chunks",
        "replace_links",
        "replace_provenance",
        "deactivate",
        "delete",
    ]
    slot: int
    version: int = 0
    chunk_texts: tuple[str, ...] = ()
    link_targets: tuple[str, ...] = ()
    prov_sources: tuple[str, ...] = ()


def _op_strategy() -> st.SearchStrategy[_Op]:
    slot = st.integers(min_value=0, max_value=len(_DOC_SLOTS) - 1)
    chunk_text = st.lists(st.sampled_from(_VOCAB), min_size=1, max_size=4).map(" ".join)
    return st.one_of(
        st.builds(
            _Op,
            kind=st.just("upsert_doc"),
            slot=slot,
            version=st.integers(min_value=1, max_value=100),
        ),
        st.builds(
            _Op,
            kind=st.just("replace_chunks"),
            slot=slot,
            chunk_texts=st.lists(chunk_text, min_size=0, max_size=3).map(tuple),
        ),
        st.builds(
            _Op,
            kind=st.just("replace_links"),
            slot=slot,
            link_targets=st.lists(st.sampled_from(_TARGET_PATHS), min_size=0, max_size=3).map(
                tuple
            ),
        ),
        st.builds(
            _Op,
            kind=st.just("replace_provenance"),
            slot=slot,
            prov_sources=st.lists(st.sampled_from(_SOURCE_REFS), min_size=0, max_size=3).map(
                tuple
            ),
        ),
        st.builds(_Op, kind=st.just("deactivate"), slot=slot),
        st.builds(_Op, kind=st.just("delete"), slot=slot),
    )


def _doc_for_slot(slot: int, version: int) -> DocumentRecord:
    layer, path = _DOC_SLOTS[slot]
    return DocumentRecord(
        doc_id=doc_id_for(layer, path),
        path=path,
        title=f"Title-{slot}-v{version}",
        hash=f"h-{slot}-{version}",
        mtime=float(1000 + version),
        layer=layer,
        active=True,
    )


# ---------------------------------------------------------------------------
# Pure-Python reference oracle for Storage semantics
# ---------------------------------------------------------------------------


class _RefStorageModel:
    """Minimal in-memory oracle mirroring the Storage Protocol invariants."""

    def __init__(self) -> None:
        self.docs: dict[str, DocumentRecord] = {}
        self.chunks: dict[str, list[tuple[int, int, int, str]]] = {}
        self.links: dict[str, list[tuple[str, str, LinkType, str | None, int]]] = {}
        self.prov: dict[str, dict[str, str]] = {}  # doc_id -> {source_path_key: raw_path}

    def apply(self, op: _Op) -> None:
        layer, path = _DOC_SLOTS[op.slot]
        doc_id = doc_id_for(layer, path)
        if op.kind == "upsert_doc":
            self.docs[doc_id] = _doc_for_slot(op.slot, op.version)
        elif op.kind == "replace_chunks":
            if doc_id not in self.docs:
                return
            rows: list[tuple[int, int, int, str]] = []
            offset = 0
            for seq, text in enumerate(op.chunk_texts):
                end = offset + len(text)
                rows.append((seq, offset, end, text))
                offset = end + 2
            self.chunks[doc_id] = rows
        elif op.kind == "replace_links":
            if doc_id not in self.docs:
                return
            self.links[doc_id] = [
                (doc_id, target, LinkType.WIKILINK, None, idx + 1)
                for idx, target in enumerate(op.link_targets)
            ]
        elif op.kind == "replace_provenance":
            if doc_id not in self.docs:
                return
            dedup: dict[str, str] = {}
            for src in op.prov_sources:
                key = normalize_path(src)
                dedup.setdefault(key, src)
            self.prov[doc_id] = dedup
        elif op.kind == "deactivate":
            doc = self.docs.get(doc_id)
            if doc is not None:
                self.docs[doc_id] = doc.model_copy(update={"active": False})
        elif op.kind == "delete":
            self.docs.pop(doc_id, None)
            self.chunks.pop(doc_id, None)
            self.links.pop(doc_id, None)
            self.prov.pop(doc_id, None)

    def snapshot(self) -> dict[str, object]:
        active_docs = sorted(
            (d.doc_id, d.path_key, d.title, d.hash, d.layer.value, d.active)
            for d in self.docs.values()
            if d.active
        )
        all_docs = sorted(
            (d.doc_id, d.path_key, d.title, d.hash, d.layer.value, d.active)
            for d in self.docs.values()
        )
        chunks_by_doc = {
            doc_id: self.chunks.get(doc_id, [])
            for doc_id in sorted(self.docs)
        }
        links_from_map = {
            doc_id: sorted(self.links.get(doc_id, []))
            for doc_id in sorted(self.docs)
        }
        links_to_map = {
            target: sorted(
                row
                for rows in self.links.values()
                for row in rows
                if row[1] == target
            )
            for target in _TARGET_PATHS
        }
        prov_from_map = {
            doc_id: sorted((doc_id, k) for k in self.prov.get(doc_id, {}))
            for doc_id in sorted(self.docs)
        }
        prov_to_map = {
            normalize_path(src): sorted(
                (doc_id, normalize_path(src))
                for doc_id, mapping in self.prov.items()
                if normalize_path(src) in mapping
            )
            for src in ("sources/alpha.md", "sources/beta.md")
        }
        fts_hits: dict[str, list[tuple[str, int]]] = {}
        for word in _VOCAB[:4]:
            matched: list[tuple[str, int]] = []
            for doc_id, doc in self.docs.items():
                if not doc.active:
                    continue
                for seq, _s, _e, text in self.chunks.get(doc_id, []):
                    if word in text.split():
                        matched.append((doc_id, seq))
            fts_hits[word] = sorted(matched)
        doc_counts: dict[str, int] = {}
        for d in self.docs.values():
            if d.active:
                doc_counts[d.layer.value] = doc_counts.get(d.layer.value, 0) + 1
        # Both SQLite and Postgres count every stored chunk (deactivation hides
        # a document from search without deleting its chunks).
        chunk_count = sum(len(rows) for rows in self.chunks.values())
        link_count = sum(len(rows) for rows in self.links.values())
        return {
            "active_docs": active_docs,
            "all_docs": all_docs,
            "chunks_by_doc": chunks_by_doc,
            "links_from": links_from_map,
            "links_to": links_to_map,
            "prov_from": prov_from_map,
            "prov_to": prov_to_map,
            "fts_hits": fts_hits,
            "counts": (doc_counts, chunk_count, link_count),
        }


async def _apply_storage_op(storage: Storage, op: _Op) -> None:
    layer, path = _DOC_SLOTS[op.slot]
    doc_id = doc_id_for(layer, path)
    if op.kind == "upsert_doc":
        await storage.upsert_document(_doc_for_slot(op.slot, op.version))
    elif op.kind == "replace_chunks":
        if await storage.get_document(doc_id) is None:
            return
        records: list[ChunkRecord] = []
        offset = 0
        for seq, text in enumerate(op.chunk_texts):
            end = offset + len(text)
            records.append(
                ChunkRecord(doc_id=doc_id, seq=seq, start=offset, end=end, text=text)
            )
            offset = end + 2
        await storage.replace_chunks(doc_id, records)
    elif op.kind == "replace_links":
        if await storage.get_document(doc_id) is None:
            return
        links = [
            LinkRecord(
                src_doc_id=doc_id,
                dst_path=target,
                link_type=LinkType.WIKILINK,
                anchor=None,
                line=idx + 1,
            )
            for idx, target in enumerate(op.link_targets)
        ]
        await storage.replace_links_from(doc_id, links)
    elif op.kind == "replace_provenance":
        if await storage.get_document(doc_id) is None:
            return
        await storage.replace_provenance_from(doc_id, op.prov_sources)
    elif op.kind == "deactivate":
        await storage.deactivate_document(doc_id)
    elif op.kind == "delete":
        await storage.delete_document(doc_id)


async def _snapshot_storage(
    storage: Storage, *, fts_granularity: Literal["chunk", "doc"] = "chunk"
) -> dict[str, object]:
    active_list = list(await storage.list_documents(active=True))
    all_list = list(await storage.list_documents(active=None))
    active_docs = sorted(
        (d.doc_id, d.path_key, d.title, d.hash, d.layer.value, d.active) for d in active_list
    )
    all_docs = sorted(
        (d.doc_id, d.path_key, d.title, d.hash, d.layer.value, d.active) for d in all_list
    )
    doc_ids = sorted(d.doc_id for d in all_list)

    chunks_by_doc: dict[str, list[tuple[int, int, int, str]]] = {}
    chunk_seq_by_id: dict[int, int] = {}
    for doc_id in doc_ids:
        chunks = await storage.list_chunks(doc_id)
        chunks_by_doc[doc_id] = [(c.seq, c.start, c.end, c.text) for c in chunks]
        for c in chunks:
            if c.chunk_id is not None:
                chunk_seq_by_id[c.chunk_id] = c.seq

    links_from_map = {
        doc_id: sorted(
            (lk.src_doc_id, lk.dst_path, lk.link_type, lk.anchor, lk.line)
            for lk in await storage.links_from(doc_id)
        )
        for doc_id in doc_ids
    }
    links_to_map = {
        target: sorted(
            (lk.src_doc_id, lk.dst_path, lk.link_type, lk.anchor, lk.line)
            for lk in await storage.links_to(target)
        )
        for target in _TARGET_PATHS
    }
    prov_from_map = {
        doc_id: sorted((e.src_doc_id, e.source_path_key) for e in await storage.provenance_from(doc_id))
        for doc_id in doc_ids
    }
    prov_to_map = {
        normalize_path(src): sorted(
            (e.src_doc_id, e.source_path_key)
            for e in await storage.provenance_to(normalize_path(src))
        )
        for src in ("sources/alpha.md", "sources/beta.md")
    }
    fts_hits: dict[str, list[object]] = {}
    for word in _VOCAB[:4]:
        hits = await storage.fts_search(word, limit=50)
        if fts_granularity == "doc":
            # Postgres fts_search deduplicates to the highest-ranked chunk per
            # document, whereas SQLite returns every matching chunk — compare
            # matching document IDs across adapters.
            fts_hits[word] = sorted({h.doc_id for h in hits})
        else:
            fts_hits[word] = sorted(
                (h.doc_id, chunk_seq_by_id[h.chunk_id])
                for h in hits
                if h.chunk_id is not None and h.chunk_id in chunk_seq_by_id
            )
    counts = await storage.counts()
    return {
        "active_docs": active_docs,
        "all_docs": all_docs,
        "chunks_by_doc": chunks_by_doc,
        "links_from": links_from_map,
        "links_to": links_to_map,
        "prov_from": prov_from_map,
        "prov_to": prov_to_map,
        "fts_hits": fts_hits,
        "counts": (counts.documents_by_layer, counts.chunks, counts.links),
    }


@settings(max_examples=25, deadline=None)
@given(ops=st.lists(_op_strategy(), min_size=1, max_size=18))
def test_sqlite_matches_reference_model_under_random_ops(ops: list[_Op]) -> None:
    async def _run() -> None:
        ref = _RefStorageModel()
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStorage(Path(tmp) / "diff.sqlite", cjk_tokenizer="jieba")
            await store.connect()
            await store.migrate()
            try:
                for op in ops:
                    ref.apply(op)
                    await _apply_storage_op(store, op)
                expected = ref.snapshot()
                actual = await _snapshot_storage(store)
                assert actual == expected
            finally:
                await store.close()

    asyncio.run(_run())


@pytest.mark.skipif(
    not os.environ.get("DIKW_TEST_POSTGRES_DSN"),
    reason="SQLite vs Postgres differential test requires DIKW_TEST_POSTGRES_DSN",
)
@settings(max_examples=15, deadline=None)
@given(ops=st.lists(_op_strategy(), min_size=1, max_size=18))
def test_sqlite_vs_postgres_differential_parity(ops: list[_Op]) -> None:
    dsn = os.environ["DIKW_TEST_POSTGRES_DSN"]

    async def _run() -> None:
        from psycopg import AsyncConnection

        from dikw_core.storage.postgres import PostgresStorage

        schema = f"dikw_diff_{uuid.uuid4().hex[:12]}"
        with tempfile.TemporaryDirectory() as tmp:
            sqlite_store = SQLiteStorage(Path(tmp) / "diff.sqlite", cjk_tokenizer="jieba")
            pg_store = PostgresStorage(dsn, schema=schema, pool_size=2, cjk_tokenizer="jieba")
            await sqlite_store.connect()
            await sqlite_store.migrate()
            await pg_store.connect()
            await pg_store.migrate()
            try:
                for op in ops:
                    await _apply_storage_op(sqlite_store, op)
                    await _apply_storage_op(pg_store, op)
                sqlite_snap = await _snapshot_storage(sqlite_store, fts_granularity="doc")
                pg_snap = await _snapshot_storage(pg_store, fts_granularity="doc")
                assert pg_snap == sqlite_snap
            finally:
                await sqlite_store.close()
                await pg_store.close()
                conn = await AsyncConnection.connect(dsn)
                try:
                    async with conn.cursor() as cur:
                        await cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
                    await conn.commit()
                finally:
                    await conn.close()

    asyncio.run(_run())
