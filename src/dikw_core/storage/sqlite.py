"""SQLite + sqlite-vec + FTS5 storage adapter (MVP).

Uses stdlib ``sqlite3`` synchronously under the hood and exposes an ``async``
surface on the Storage Protocol via ``asyncio.to_thread``. This is intentional:
SQLite's pattern is one-writer-at-a-time; wrapping it in a worker thread lets
the rest of the engine stay ``async`` without pulling in aiosqlite.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import time
from collections.abc import Iterable, Sequence
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import sqlite_vec

from ..domains.data.hashing import hash_bytes
from ..domains.info.tokenize import CjkTokenizer, initialize_jieba, preprocess_for_fts
from ..schemas import (
    AssetEmbeddingRow,
    AssetKind,
    AssetRecord,
    AssetVecHit,
    CachedEmbeddingRow,
    ChunkAssetRef,
    ChunkNeighborRecord,
    ChunkRecord,
    DocumentRecord,
    EmbeddingRow,
    EmbeddingVersion,
    FTSHit,
    Layer,
    LinkRecord,
    LinkType,
    StorageCounts,
    VecHit,
    WikiLogEntry,
    WisdomEmbeddingRow,
    WisdomEvidence,
    WisdomItem,
    WisdomKind,
    WisdomStatus,
    WisdomVecHit,
    dump_media_meta,
    load_media_meta,
)
from ._schema import SCHEMA_VERSION, SCHEMA_VERSION_KEY, mismatch_message
from ._vec_codec import deserialize_vec as _deserialize_vec
from ._vec_codec import serialize_vec as _serialize_vec
from .base import NotSupported, StorageError

MIGRATIONS_PACKAGE = "dikw_core.storage.migrations.sqlite"

_REBUILD_HINT = (
    "dikw-core is pre-alpha — the on-disk format is allowed to change, "
    "and there is no in-place upgrade path. Delete the SQLite database "
    "file (the path SQLiteStorage was constructed with, typically "
    "``.dikw/dikw.sqlite``) and re-run ``dikw ingest`` to rebuild."
)

# vec0 defaults to L2; we want cosine for parity with the legacy
# ``vec_distance_cosine`` ranking so existing BASELINES.md thresholds
# stay valid. Baked into ``CREATE VIRTUAL TABLE`` and verified at
# migrate-time on legacy DBs.
_VEC_DISTANCE_METRIC = "cosine"

# Over-fetch factor when a ``layer`` filter post-filters the KNN heap.
# A skewed corpus (e.g. 1% WIKI in a SOURCE-heavy vault) can have
# fewer than ``limit`` of the requested layer in the top-``k``, so we
# pull a larger candidate set and trim. Tunable; 10x picked to absorb
# 90/10 skew at ``limit=20`` without an exponential-backoff retry loop.
#
# Known limitation: sqlite-vec MATCH can't constrain on the external
# ``documents.layer`` column, so the layer filter happens after KNN.
# A query whose first matching-layer chunk ranks beyond
# ``limit * _LAYER_FILTER_OVER_FETCH`` globally will under-fill (or
# return ``[]``). Acceptable today because dikw layers (SOURCE/WIKI/
# WISDOM) are not typically 99.x% skewed. Follow-up: retry with
# exponential backoff or fall back to brute-force when the candidate
# set under-fills.
_LAYER_FILTER_OVER_FETCH = 10


class SQLiteStorage:
    """SQLite-backed Storage implementation.

    Each ``embed_versions`` row produces its own per-version vector table
    (``vec_chunks_v<id>`` for text, ``vec_assets_v<id>`` for multimodal),
    created lazily because sqlite-vec needs the dim baked into the
    CREATE statement. Switching a model = new version row = new table;
    prior vectors survive in place.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        cjk_tokenizer: CjkTokenizer = "none",
    ) -> None:
        self._path = Path(path)
        self._conn: sqlite3.Connection | None = None
        # Must match `_sanitize_fts` on the query side — locked at first
        # ingest via `RetrievalConfig.cjk_tokenizer`.
        self._cjk_tokenizer: CjkTokenizer = cjk_tokenizer

    # ---- lifecycle -------------------------------------------------------

    async def connect(self) -> None:
        def _open() -> sqlite3.Connection:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            return conn

        self._conn = await asyncio.to_thread(_open)
        # Warm up jieba's dictionary now so the ~0.3 s load lands in
        # `connect()` instead of the first `replace_chunks` call.
        if self._cjk_tokenizer == "jieba":
            await asyncio.to_thread(initialize_jieba)

    async def close(self) -> None:
        conn = self._conn
        if conn is None:
            return
        self._conn = None
        await asyncio.to_thread(conn.close)

    async def migrate(self) -> None:
        """Apply ``schema.sql`` to a fresh DB, or refuse a stale fingerprint.

        See ``storage/_schema.py`` for the rebuild-on-incompatibility
        policy and the ``SCHEMA_VERSION`` fingerprint contract.
        """

        def _run() -> None:
            conn = self._require_conn()
            with conn:
                # meta_kv must exist before the version-read query.
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS meta_kv ("
                    "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                stored = _read_schema_version_sqlite(conn)
                if stored is None:
                    # ``stored is None`` should mean "fresh DB" but a
                    # pre-fingerprint install (older than PR #39) has
                    # the same shape: tables exist, no version row.
                    # ``schema.sql`` is all ``CREATE … IF NOT EXISTS``,
                    # so silently applying it would leave the legacy
                    # tables in place and stamp the DB as current —
                    # later reads against new columns would fail with
                    # opaque SQL errors. Loud-fail instead, consistent
                    # with the rebuild-on-incompatibility policy.
                    legacy = conn.execute(
                        "SELECT 1 FROM sqlite_master "
                        "WHERE type = 'table' AND name = 'documents'"
                    ).fetchone()
                    if legacy is not None:
                        raise StorageError(
                            "DB has tables but no schema_version row — "
                            "likely a pre-fingerprint install. "
                            f"{_REBUILD_HINT}"
                        )
                    sql = (
                        resources.files(MIGRATIONS_PACKAGE)
                        .joinpath("schema.sql")
                        .read_text(encoding="utf-8")
                    )
                    conn.executescript(sql)
                    _write_schema_version_sqlite(conn, SCHEMA_VERSION)
                elif stored != SCHEMA_VERSION:
                    raise StorageError(mismatch_message(stored, _REBUILD_HINT))
            # Vec tables are runtime-created and not in schema.sql, so
            # the cosine invariant is checked outside the version-branch
            # logic. Cheap fixed-cost lookup.
            self._verify_vec_tables_use_cosine(conn)

        await asyncio.to_thread(_run)

    # ---- D layer ---------------------------------------------------------

    async def upsert_document(self, doc: DocumentRecord) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    """
                    INSERT INTO documents(
                        doc_id, path, path_key, title, hash, mtime, layer, active
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                        path = excluded.path,
                        path_key = excluded.path_key,
                        title = excluded.title,
                        hash = excluded.hash,
                        mtime = excluded.mtime,
                        layer = excluded.layer,
                        active = excluded.active
                    """,
                    (
                        doc.doc_id,
                        doc.path,
                        doc.path_key,
                        doc.title,
                        doc.hash,
                        doc.mtime,
                        doc.layer.value,
                        int(doc.active),
                    ),
                )

        await asyncio.to_thread(_run)

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        def _run() -> DocumentRecord | None:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            return _row_to_document(row) if row is not None else None

        return await asyncio.to_thread(_run)

    async def get_documents(
        self, doc_ids: Iterable[str]
    ) -> list[DocumentRecord]:
        ids = list(doc_ids)
        if not ids:
            return []

        def _run() -> list[DocumentRecord]:
            conn = self._require_conn()
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT * FROM documents WHERE doc_id IN ({placeholders})",
                ids,
            ).fetchall()
            return [_row_to_document(r) for r in rows]

        return await asyncio.to_thread(_run)

    async def list_documents(
        self,
        *,
        layer: Layer | None = None,
        active: bool | None = True,
        since_ts: float | None = None,
    ) -> Iterable[DocumentRecord]:
        def _run() -> list[DocumentRecord]:
            conn = self._require_conn()
            q = "SELECT * FROM documents WHERE 1=1"
            params: list[Any] = []
            if layer is not None:
                q += " AND layer = ?"
                params.append(layer.value)
            if active is not None:
                q += " AND active = ?"
                params.append(int(active))
            if since_ts is not None:
                q += " AND mtime >= ?"
                params.append(since_ts)
            rows = conn.execute(q, params).fetchall()
            return [_row_to_document(r) for r in rows]

        return await asyncio.to_thread(_run)

    async def deactivate_document(self, doc_id: str) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    "UPDATE documents SET active = 0 WHERE doc_id = ?", (doc_id,)
                )

        await asyncio.to_thread(_run)

    # ---- I layer ---------------------------------------------------------

    async def replace_chunks(
        self, doc_id: str, chunks: Sequence[ChunkRecord]
    ) -> list[int]:
        def _run() -> list[int]:
            conn = self._require_conn()
            ids: list[int] = []
            with conn:
                # Drop FTS rows for this doc BEFORE deleting the chunks
                # they reference — the rowid lookup goes through the
                # ``chunks`` table, so once chunks are gone we can't
                # find the FTS rows to delete anymore.
                conn.execute(
                    "DELETE FROM documents_fts WHERE rowid IN ("
                    "SELECT chunk_id FROM chunks WHERE doc_id = ?)",
                    (doc_id,),
                )
                conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                for c in chunks:
                    cur = conn.execute(
                        "INSERT INTO chunks(doc_id, seq, start_off, end_off, text) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (c.doc_id, c.seq, c.start, c.end, c.text),
                    )
                    ids.append(int(cur.lastrowid or 0))
                # rowid = chunk_id lets `fts_search` return chunk_id so
                # `chunk_asset_refs` can attach to FTS-only hits. Body
                # goes through the same preprocessor the query side
                # uses in `_sanitize_fts`.
                for cid, c in zip(ids, chunks, strict=True):
                    body = preprocess_for_fts(
                        c.text, tokenizer=self._cjk_tokenizer
                    )
                    conn.execute(
                        "INSERT INTO documents_fts(rowid, body) "
                        "VALUES (?, ?)",
                        (cid, body),
                    )
            return ids

        return await asyncio.to_thread(_run)

    async def upsert_embeddings(self, rows: Sequence[EmbeddingRow]) -> None:
        if not rows:
            return

        def _run() -> None:
            conn = self._require_conn()
            # Group by version so each per-version vec table is touched
            # once; a single ingest run usually has one version_id but
            # mixed batches stay correct.
            by_version: dict[int, list[EmbeddingRow]] = {}
            for r in rows:
                by_version.setdefault(r.version_id, []).append(r)
            with conn:
                for version_id, batch in by_version.items():
                    version = _fetch_version(conn, version_id)
                    if version is None:
                        raise StorageError(
                            f"unknown embed version_id={version_id}; "
                            "call upsert_embed_version first"
                        )
                    for r in batch:
                        if len(r.embedding) != version.dim:
                            raise StorageError(
                                f"embedding dim {len(r.embedding)} != "
                                f"version {version_id} dim {version.dim}"
                            )
                    self._ensure_vec_table(conn, "chunks", version_id, version.dim)
                    vec_table = f"vec_chunks_v{version_id}"
                    for r in batch:
                        conn.execute(
                            "INSERT OR REPLACE INTO chunk_embed_meta"
                            "(chunk_id, version_id) VALUES (?, ?)",
                            (r.chunk_id, version_id),
                        )
                        conn.execute(
                            f"INSERT OR REPLACE INTO {vec_table}"
                            "(rowid, embedding) VALUES (?, ?)",
                            (r.chunk_id, _serialize_vec(r.embedding)),
                        )

        await asyncio.to_thread(_run)

    async def get_cached_embeddings(
        self, content_hashes: Sequence[str], *, version_id: int
    ) -> dict[str, list[float]]:
        hashes = list(content_hashes)
        if not hashes:
            return {}

        def _run() -> dict[str, list[float]]:
            conn = self._require_conn()
            placeholders = ",".join("?" * len(hashes))
            rows = conn.execute(
                f"SELECT content_hash, dim, embedding FROM embed_cache "
                f"WHERE version_id = ? AND content_hash IN ({placeholders})",
                [version_id, *hashes],
            ).fetchall()
            return {
                str(r["content_hash"]): _deserialize_vec(r["embedding"], int(r["dim"]))
                for r in rows
            }

        return await asyncio.to_thread(_run)

    async def cache_embeddings(self, rows: Sequence[CachedEmbeddingRow]) -> None:
        if not rows:
            return

        def _run() -> None:
            conn = self._require_conn()
            now = time.time()
            with conn:
                for r in rows:
                    # INSERT OR IGNORE: idempotent on (content_hash, version_id);
                    # vectors for the same content under the same version
                    # identity must be deterministic, so we never overwrite.
                    conn.execute(
                        "INSERT OR IGNORE INTO embed_cache"
                        "(content_hash, version_id, dim, embedding, created_ts) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            r.content_hash,
                            r.version_id,
                            r.dim,
                            _serialize_vec(list(r.embedding)),
                            now,
                        ),
                    )

        await asyncio.to_thread(_run)

    async def list_chunks_missing_embedding(
        self, *, version_id: int
    ) -> list[ChunkRecord]:
        def _run() -> list[ChunkRecord]:
            conn = self._require_conn()
            rows = conn.execute(
                "SELECT chunk_id, doc_id, seq, start_off, end_off, text "
                "FROM chunks "
                "WHERE chunk_id NOT IN "
                "(SELECT chunk_id FROM chunk_embed_meta WHERE version_id = ?) "
                "ORDER BY chunk_id",
                (version_id,),
            ).fetchall()
            return [
                ChunkRecord(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    seq=r["seq"],
                    start=r["start_off"],
                    end=r["end_off"],
                    text=r["text"],
                )
                for r in rows
            ]

        return await asyncio.to_thread(_run)

    async def get_chunk(self, chunk_id: int) -> ChunkRecord | None:
        def _run() -> ChunkRecord | None:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT chunk_id, doc_id, seq, start_off, end_off, text "
                "FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            return _row_to_chunk(row) if row is not None else None

        return await asyncio.to_thread(_run)

    async def list_chunks(self, doc_id: str) -> list[ChunkRecord]:
        def _run() -> list[ChunkRecord]:
            conn = self._require_conn()
            rows = conn.execute(
                "SELECT chunk_id, doc_id, seq, start_off, end_off, text "
                "FROM chunks WHERE doc_id = ? ORDER BY seq",
                (doc_id,),
            ).fetchall()
            return [_row_to_chunk(r) for r in rows]

        return await asyncio.to_thread(_run)

    async def get_chunks(self, chunk_ids: Iterable[int]) -> list[ChunkRecord]:
        ids = list(chunk_ids)
        if not ids:
            return []

        def _run() -> list[ChunkRecord]:
            conn = self._require_conn()
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(
                "SELECT chunk_id, doc_id, seq, start_off, end_off, text "
                f"FROM chunks WHERE chunk_id IN ({placeholders})",
                ids,
            ).fetchall()
            return [_row_to_chunk(r) for r in rows]

        return await asyncio.to_thread(_run)

    async def fts_search(
        self, q: str, *, limit: int = 20, layer: Layer | None = None
    ) -> list[FTSHit]:
        def _run() -> list[FTSHit]:
            conn = self._require_conn()
            # ``documents_fts`` is body-only; doc_id and layer come back
            # via JOIN through ``chunks`` (rowid = chunk_id) onto
            # ``documents``. Snippet column index 0 == the only indexed
            # column (``body``).
            sql = (
                "SELECT documents_fts.rowid AS chunk_id, d.doc_id AS doc_id, "
                "snippet(documents_fts, 0, '<mark>', '</mark>', '…', 10) AS snip, "
                "bm25(documents_fts) AS score "
                "FROM documents_fts "
                "JOIN chunks c ON c.chunk_id = documents_fts.rowid "
                "JOIN documents d ON d.doc_id = c.doc_id "
                "WHERE documents_fts MATCH ? AND d.active = 1"
            )
            params: list[Any] = [q]
            if layer is not None:
                sql += " AND d.layer = ?"
                params.append(layer.value)
            sql += " ORDER BY score LIMIT ?"
            params.append(limit)
            hits: list[FTSHit] = []
            for row in conn.execute(sql, params).fetchall():
                # bm25 returns lower-is-better; invert so higher = better
                hits.append(
                    FTSHit(
                        doc_id=row["doc_id"],
                        chunk_id=int(row["chunk_id"]),
                        score=-float(row["score"]),
                        snippet=row["snip"],
                    )
                )
            return hits

        return await asyncio.to_thread(_run)

    async def vec_search(
        self,
        embedding: list[float],
        *,
        version_id: int | None = None,
        limit: int = 20,
        layer: Layer | None = None,
    ) -> list[VecHit]:
        def _run() -> list[VecHit]:
            conn = self._require_conn()
            resolved = version_id
            if resolved is None:
                row = conn.execute(
                    "SELECT version_id, dim FROM embed_versions "
                    "WHERE modality = 'text' AND is_active = 1 "
                    "ORDER BY version_id DESC LIMIT 1"
                ).fetchone()
                if row is None:
                    raise NotSupported("no text embeddings indexed yet")
                resolved = int(row["version_id"])
                resolved_dim = int(row["dim"])
            else:
                version = _fetch_version(conn, resolved)
                if version is None:
                    raise NotSupported(
                        f"no text embeddings for version_id={resolved}"
                    )
                resolved_dim = version.dim
            if len(embedding) != resolved_dim:
                raise StorageError(
                    f"query embedding dim {len(embedding)} != "
                    f"version {resolved} dim {resolved_dim}"
                )
            vec_table = f"vec_chunks_v{resolved}"
            # If the per-version vec table doesn't exist yet (version
            # registered but no chunk embeddings written), surface as
            # NotSupported so the caller can degrade gracefully.
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = ?",
                (vec_table,),
            ).fetchone()
            if exists is None:
                raise NotSupported(
                    f"no chunk vectors for version_id={resolved}"
                )
            fetch_k = limit * _LAYER_FILTER_OVER_FETCH if layer is not None else limit
            ranked = _knn(conn, vec_table, embedding, fetch_k)
            if not ranked:
                return []
            chunk_ids = [cid for cid, _ in ranked]
            placeholders = ",".join("?" * len(chunk_ids))
            # ``d.active = 1`` mirrors fts_search above (line 506) and the
            # postgres adapter — without it, chunks of a doc deactivated
            # by ``api.ingest``'s storage_error retry path would still
            # leak into retrieve / query results until the doc is
            # re-ingested or the chunks fall out of cache. The fts side
            # already filters; this brings the dense leg in line.
            join_sql = (
                f"SELECT c.chunk_id, c.doc_id FROM chunks c "
                f"JOIN documents d ON d.doc_id = c.doc_id "
                f"WHERE c.chunk_id IN ({placeholders}) AND d.active = 1"
            )
            join_params: list[Any] = list(chunk_ids)
            if layer is not None:
                join_sql += " AND d.layer = ?"
                join_params.append(layer.value)
            doc_id_by_chunk: dict[int, str] = {
                int(r["chunk_id"]): r["doc_id"]
                for r in conn.execute(join_sql, join_params).fetchall()
            }
            hits: list[VecHit] = []
            for chunk_id, dist in ranked:
                doc_id = doc_id_by_chunk.get(chunk_id)
                if doc_id is None:
                    continue  # filtered by layer or chunk row gone
                hits.append(VecHit(doc_id=doc_id, chunk_id=chunk_id, distance=dist))
                if len(hits) >= limit:
                    break
            return hits

        return await asyncio.to_thread(_run)

    # ---- K layer ---------------------------------------------------------

    async def upsert_link(self, link: LinkRecord) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO links(src_doc_id, dst_path, link_type, anchor, line) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (link.src_doc_id, link.dst_path, link.link_type.value, link.anchor, link.line),
                )

        await asyncio.to_thread(_run)

    async def links_from(self, src_doc_id: str) -> list[LinkRecord]:
        def _run() -> list[LinkRecord]:
            conn = self._require_conn()
            rows = conn.execute(
                "SELECT * FROM links WHERE src_doc_id = ?", (src_doc_id,)
            ).fetchall()
            return [_row_to_link(r) for r in rows]

        return await asyncio.to_thread(_run)

    async def links_to(self, dst_path: str) -> list[LinkRecord]:
        def _run() -> list[LinkRecord]:
            conn = self._require_conn()
            rows = conn.execute(
                "SELECT * FROM links WHERE dst_path = ?", (dst_path,)
            ).fetchall()
            return [_row_to_link(r) for r in rows]

        return await asyncio.to_thread(_run)

    async def neighbor_chunks_via_links(
        self,
        seed_chunk_ids: Sequence[int],
        *,
        layer: Layer | None = None,
        limit: int = 200,
    ) -> list[ChunkNeighborRecord]:
        if not seed_chunk_ids:
            return []

        def _run() -> list[ChunkNeighborRecord]:
            conn = self._require_conn()
            seed_list = list(seed_chunk_ids)
            seeds_in = ",".join("?" * len(seed_list))
            params: list[object] = [*seed_list, *seed_list]
            sql = (
                "SELECT c2.chunk_id, c2.doc_id, COUNT(*) AS edge_count "
                "FROM chunks c1 "
                "JOIN documents d1 ON c1.doc_id = d1.doc_id "
                "JOIN links l ON l.src_doc_id = d1.doc_id "
                "    AND l.link_type = 'wikilink' "
                "JOIN documents d2 ON l.dst_path = d2.path AND d2.active = 1 "
                "JOIN chunks c2 ON c2.doc_id = d2.doc_id "
                f"WHERE c1.chunk_id IN ({seeds_in}) "
                f"  AND c2.chunk_id NOT IN ({seeds_in}) "
            )
            if layer is not None:
                sql += " AND d2.layer = ? "
                params.append(layer.value)
            sql += (
                "GROUP BY c2.chunk_id, c2.doc_id "
                "ORDER BY edge_count DESC, c2.chunk_id "
                "LIMIT ?"
            )
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [
                ChunkNeighborRecord(
                    chunk_id=row[0], doc_id=row[1], edge_count=row[2], hop=1
                )
                for row in rows
            ]

        return await asyncio.to_thread(_run)

    async def append_wiki_log(self, entry: WikiLogEntry) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    "INSERT INTO wiki_log(ts, action, src, dst, note) VALUES (?, ?, ?, ?, ?)",
                    (entry.ts, entry.action, entry.src, entry.dst, entry.note),
                )

        await asyncio.to_thread(_run)

    async def list_wiki_log(
        self, *, since_ts: float | None = None, limit: int | None = None
    ) -> list[WikiLogEntry]:
        def _run() -> list[WikiLogEntry]:
            conn = self._require_conn()
            sql = "SELECT id, ts, action, src, dst, note FROM wiki_log"
            params: list[Any] = []
            if since_ts is not None:
                sql += " WHERE ts >= ?"
                params.append(since_ts)
            # (ts, id) tie-break: float-second ts can collide for events
            # in the same ingest batch; AUTOINCREMENT id preserves
            # insertion order within a second.
            sql += " ORDER BY ts ASC, id ASC"
            if limit is not None:
                sql += " LIMIT ?"
                params.append(int(limit))
            rows = conn.execute(sql, params).fetchall()
            return [
                WikiLogEntry(
                    id=int(r["id"]),
                    ts=float(r["ts"]),
                    action=r["action"],
                    src=r["src"],
                    dst=r["dst"],
                    note=r["note"],
                )
                for r in rows
            ]

        return await asyncio.to_thread(_run)

    # ---- W layer ---------------------------------------------------------

    async def put_wisdom(
        self, item: WisdomItem, evidence: Sequence[WisdomEvidence]
    ) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    """
                    INSERT INTO wisdom_items(
                        item_id, kind, status, path, title, body,
                        confidence, created_ts, approved_ts
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(item_id) DO UPDATE SET
                        kind = excluded.kind,
                        status = excluded.status,
                        path = excluded.path,
                        title = excluded.title,
                        body = excluded.body,
                        confidence = excluded.confidence,
                        approved_ts = excluded.approved_ts
                    """,
                    (
                        item.item_id,
                        item.kind.value,
                        item.status.value,
                        item.path,
                        item.title,
                        item.body,
                        item.confidence,
                        item.created_ts,
                        item.approved_ts,
                    ),
                )
                conn.execute("DELETE FROM wisdom_evidence WHERE item_id = ?", (item.item_id,))
                for e in evidence:
                    conn.execute(
                        "INSERT INTO wisdom_evidence(item_id, doc_id, excerpt, line) "
                        "VALUES (?, ?, ?, ?)",
                        (item.item_id, e.doc_id, e.excerpt, e.line),
                    )

        await asyncio.to_thread(_run)

    async def list_wisdom(
        self,
        *,
        status: WisdomStatus | None = None,
        kind: WisdomKind | None = None,
    ) -> list[WisdomItem]:
        def _run() -> list[WisdomItem]:
            conn = self._require_conn()
            sql = "SELECT * FROM wisdom_items WHERE 1=1"
            params: list[Any] = []
            if status is not None:
                sql += " AND status = ?"
                params.append(status.value)
            if kind is not None:
                sql += " AND kind = ?"
                params.append(kind.value)
            sql += " ORDER BY created_ts DESC"
            rows = conn.execute(sql, params).fetchall()
            return [_row_to_wisdom(r) for r in rows]

        return await asyncio.to_thread(_run)

    async def set_wisdom_status(
        self,
        item_id: str,
        status: WisdomStatus,
        *,
        approved_ts: float | None = None,
    ) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                if approved_ts is None:
                    conn.execute(
                        "UPDATE wisdom_items SET status = ? WHERE item_id = ?",
                        (status.value, item_id),
                    )
                else:
                    conn.execute(
                        "UPDATE wisdom_items SET status = ?, approved_ts = ? "
                        "WHERE item_id = ?",
                        (status.value, approved_ts, item_id),
                    )

        await asyncio.to_thread(_run)

    async def get_wisdom(self, item_id: str) -> WisdomItem | None:
        def _run() -> WisdomItem | None:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT * FROM wisdom_items WHERE item_id = ?", (item_id,)
            ).fetchone()
            return _row_to_wisdom(row) if row is not None else None

        return await asyncio.to_thread(_run)

    async def get_wisdom_evidence(self, item_id: str) -> list[WisdomEvidence]:
        def _run() -> list[WisdomEvidence]:
            conn = self._require_conn()
            rows = conn.execute(
                "SELECT id, doc_id, excerpt, line FROM wisdom_evidence "
                "WHERE item_id = ? ORDER BY id ASC",
                (item_id,),
            ).fetchall()
            return [
                WisdomEvidence(
                    id=int(r["id"]),
                    doc_id=r["doc_id"],
                    excerpt=r["excerpt"],
                    line=int(r["line"]) if r["line"] is not None else None,
                )
                for r in rows
            ]

        return await asyncio.to_thread(_run)

    # ---- D layer: multimedia assets --------------------------------------

    async def upsert_asset(self, asset: AssetRecord) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    """
                    INSERT INTO assets(
                        asset_id, kind, mime, stored_path, original_paths,
                        bytes, media_meta, created_ts
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(asset_id) DO UPDATE SET
                        kind = excluded.kind,
                        mime = excluded.mime,
                        stored_path = excluded.stored_path,
                        original_paths = excluded.original_paths,
                        bytes = excluded.bytes,
                        media_meta = excluded.media_meta
                    """,
                    (
                        asset.asset_id,
                        asset.kind.value,
                        asset.mime,
                        asset.stored_path,
                        json.dumps(asset.original_paths),
                        asset.bytes,
                        dump_media_meta(asset.media_meta),
                        asset.created_ts,
                    ),
                )

        await asyncio.to_thread(_run)

    async def get_asset(self, asset_id: str) -> AssetRecord | None:
        def _run() -> AssetRecord | None:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            return _row_to_asset(row) if row is not None else None

        return await asyncio.to_thread(_run)

    async def get_assets(self, asset_ids: Iterable[str]) -> list[AssetRecord]:
        ids = list(asset_ids)
        if not ids:
            return []

        def _run() -> list[AssetRecord]:
            conn = self._require_conn()
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT * FROM assets WHERE asset_id IN ({placeholders})",
                ids,
            ).fetchall()
            return [_row_to_asset(r) for r in rows]

        return await asyncio.to_thread(_run)

    # ---- I layer: chunk ↔ asset bridge -----------------------------------

    async def replace_chunk_asset_refs(
        self, chunk_id: int, refs: Sequence[ChunkAssetRef]
    ) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    "DELETE FROM chunk_asset_refs WHERE chunk_id = ?", (chunk_id,)
                )
                for r in refs:
                    if r.chunk_id != chunk_id:
                        raise StorageError(
                            f"ChunkAssetRef.chunk_id={r.chunk_id} doesn't match "
                            f"target chunk_id={chunk_id}"
                        )
                    conn.execute(
                        """
                        INSERT INTO chunk_asset_refs(
                            chunk_id, asset_id, ord, alt,
                            start_in_chunk, end_in_chunk
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            r.chunk_id,
                            r.asset_id,
                            r.ord,
                            r.alt,
                            r.start_in_chunk,
                            r.end_in_chunk,
                        ),
                    )

        await asyncio.to_thread(_run)

    async def chunk_asset_refs_for_chunks(
        self, chunk_ids: Sequence[int]
    ) -> dict[int, list[ChunkAssetRef]]:
        def _run() -> dict[int, list[ChunkAssetRef]]:
            conn = self._require_conn()
            out: dict[int, list[ChunkAssetRef]] = {cid: [] for cid in chunk_ids}
            if not chunk_ids:
                return out
            placeholders = ",".join("?" for _ in chunk_ids)
            rows = conn.execute(
                f"""
                SELECT chunk_id, asset_id, ord, alt, start_in_chunk, end_in_chunk
                FROM chunk_asset_refs
                WHERE chunk_id IN ({placeholders})
                ORDER BY chunk_id, ord
                """,
                tuple(chunk_ids),
            ).fetchall()
            for r in rows:
                out[int(r["chunk_id"])].append(
                    ChunkAssetRef(
                        chunk_id=int(r["chunk_id"]),
                        asset_id=r["asset_id"],
                        ord=int(r["ord"]),
                        alt=r["alt"],
                        start_in_chunk=int(r["start_in_chunk"]),
                        end_in_chunk=int(r["end_in_chunk"]),
                    )
                )
            return out

        return await asyncio.to_thread(_run)

    async def chunks_referencing_assets(
        self, asset_ids: Sequence[str]
    ) -> dict[str, list[int]]:
        # SQLite's default ``SQLITE_MAX_VARIABLE_NUMBER`` is 999 on
        # legacy builds and 32766 on modern; chunk the IN-list so a
        # large backfill (e.g., text-only → multimodal migration with
        # thousands of unembedded images) doesn't trip
        # ``OperationalError: too many SQL variables``.
        _SQL_VARS_PER_CALL = 500

        def _run() -> dict[str, list[int]]:
            conn = self._require_conn()
            out: dict[str, list[int]] = {aid: [] for aid in asset_ids}
            if not asset_ids:
                return out
            ids_seq = list(asset_ids)
            for start in range(0, len(ids_seq), _SQL_VARS_PER_CALL):
                shard = ids_seq[start : start + _SQL_VARS_PER_CALL]
                placeholders = ",".join("?" for _ in shard)
                rows = conn.execute(
                    f"""
                    SELECT asset_id, chunk_id
                    FROM chunk_asset_refs
                    WHERE asset_id IN ({placeholders})
                    ORDER BY asset_id, chunk_id
                    """,
                    tuple(shard),
                ).fetchall()
                for r in rows:
                    out[r["asset_id"]].append(int(r["chunk_id"]))
            return out

        return await asyncio.to_thread(_run)

    # ---- I layer: asset embeddings (multimodal) --------------------------

    async def list_assets_missing_embedding(
        self, *, version_id: int
    ) -> list[AssetRecord]:
        def _run() -> list[AssetRecord]:
            conn = self._require_conn()
            rows = conn.execute(
                "SELECT * FROM assets "
                "WHERE asset_id NOT IN "
                "(SELECT asset_id FROM asset_embed_meta WHERE version_id = ?) "
                "ORDER BY asset_id",
                (version_id,),
            ).fetchall()
            return [_row_to_asset(r) for r in rows]

        return await asyncio.to_thread(_run)

    async def list_wisdom_missing_embedding(
        self, *, version_id: int
    ) -> list[WisdomItem]:
        def _run() -> list[WisdomItem]:
            conn = self._require_conn()
            rows = conn.execute(
                "SELECT * FROM wisdom_items "
                "WHERE item_id NOT IN "
                "(SELECT item_id FROM wisdom_embed_meta WHERE version_id = ?) "
                "ORDER BY item_id",
                (version_id,),
            ).fetchall()
            return [_row_to_wisdom(r) for r in rows]

        return await asyncio.to_thread(_run)

    async def upsert_asset_embeddings(
        self, rows: Sequence[AssetEmbeddingRow]
    ) -> None:
        if not rows:
            return

        def _run() -> None:
            conn = self._require_conn()
            by_version: dict[int, list[AssetEmbeddingRow]] = {}
            for r in rows:
                by_version.setdefault(r.version_id, []).append(r)
            for version_id, batch in by_version.items():
                version = _fetch_version(conn, version_id)
                if version is None:
                    raise StorageError(
                        f"unknown embed version_id={version_id}; "
                        "call upsert_embed_version first"
                    )
                for r in batch:
                    if len(r.embedding) != version.dim:
                        raise StorageError(
                            f"asset embedding dim {len(r.embedding)} != "
                            f"version {version_id} dim {version.dim}"
                        )
                self._ensure_vec_table(conn, "assets", version_id, version.dim)
                rowid_table = f"asset_vec_rowid_v{version_id}"
                conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {rowid_table}("
                    "rowid INTEGER PRIMARY KEY, "
                    "asset_id TEXT NOT NULL UNIQUE)"
                )
                vec_table = f"vec_assets_v{version_id}"
                with conn:
                    for r in batch:
                        # sqlite-vec needs an integer rowid; derive a stable
                        # 60-bit int from the sha256 asset_id. Birthday
                        # collisions are ~2^-30 even at 10^9 assets, but
                        # detect them explicitly so a hit corrupts no data.
                        rowid = _asset_id_to_rowid(r.asset_id)
                        existing = conn.execute(
                            f"SELECT asset_id FROM {rowid_table} WHERE rowid = ?",
                            (rowid,),
                        ).fetchone()
                        if existing is not None and existing["asset_id"] != r.asset_id:
                            raise StorageError(
                                f"asset rowid collision in version {version_id}: "
                                f"rowid {rowid} already used by "
                                f"{existing['asset_id']!r}, refused for "
                                f"{r.asset_id!r}"
                            )
                        conn.execute(
                            "INSERT OR REPLACE INTO asset_embed_meta"
                            "(asset_id, version_id) VALUES (?, ?)",
                            (r.asset_id, r.version_id),
                        )
                        conn.execute(
                            f"INSERT OR REPLACE INTO {vec_table}(rowid, embedding) "
                            "VALUES (?, ?)",
                            (rowid, _serialize_vec(r.embedding)),
                        )
                        conn.execute(
                            f"INSERT OR REPLACE INTO {rowid_table}"
                            "(rowid, asset_id) VALUES (?, ?)",
                            (rowid, r.asset_id),
                        )

        await asyncio.to_thread(_run)

    async def vec_search_assets(
        self,
        embedding: list[float],
        *,
        version_id: int,
        limit: int = 20,
        layer: Layer | None = None,
    ) -> list[AssetVecHit]:
        # `layer` is reserved for future use (e.g. when assets get layer
        # attribution); v1 assets are always D-layer adjacent.
        del layer

        def _run() -> list[AssetVecHit]:
            conn = self._require_conn()
            version = _fetch_version(conn, version_id)
            if version is None:
                raise NotSupported(
                    f"no embed version_id={version_id} registered"
                )
            if len(embedding) != version.dim:
                raise StorageError(
                    f"query embedding dim {len(embedding)} != "
                    f"version {version_id} dim {version.dim}"
                )
            table = f"vec_assets_v{version_id}"
            row_table = f"asset_vec_rowid_v{version_id}"
            # Empty index → no hits. Skip the SQL to avoid sqlite-vec errors
            # on a non-existent virtual table.
            tbl_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if tbl_exists is None:
                return []
            ranked = _knn(conn, table, embedding, limit)
            if not ranked:
                return []
            rowids = [rid for rid, _ in ranked]
            placeholders = ",".join("?" * len(rowids))
            asset_id_by_rowid: dict[int, str] = {
                int(r["rowid"]): r["asset_id"]
                for r in conn.execute(
                    f"SELECT rowid, asset_id FROM {row_table} "
                    f"WHERE rowid IN ({placeholders})",
                    rowids,
                ).fetchall()
            }
            return [
                AssetVecHit(asset_id=asset_id_by_rowid[rid], distance=dist)
                for rid, dist in ranked
                if rid in asset_id_by_rowid
            ]

        return await asyncio.to_thread(_run)

    # ---- W layer: wisdom embeddings --------------------------------------

    async def upsert_wisdom_embeddings(
        self, rows: Sequence[WisdomEmbeddingRow]
    ) -> None:
        if not rows:
            return

        def _run() -> None:
            conn = self._require_conn()
            by_version: dict[int, list[WisdomEmbeddingRow]] = {}
            for r in rows:
                by_version.setdefault(r.version_id, []).append(r)
            for version_id, batch in by_version.items():
                version = _fetch_version(conn, version_id)
                if version is None:
                    raise StorageError(
                        f"unknown embed version_id={version_id}; "
                        "call upsert_embed_version first"
                    )
                if version.modality != "text":
                    raise StorageError(
                        f"wisdom embeddings require modality='text'; "
                        f"version {version_id} has modality={version.modality!r} "
                        "— wisdom rides on the active text version so chunks "
                        "and wisdom share one cosine space"
                    )
                for r in batch:
                    if len(r.embedding) != version.dim:
                        raise StorageError(
                            f"wisdom embedding dim {len(r.embedding)} != "
                            f"version {version_id} dim {version.dim}"
                        )
                self._ensure_vec_table(conn, "wisdom", version_id, version.dim)
                rowid_table = f"wisdom_vec_rowid_v{version_id}"
                conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {rowid_table}("
                    "rowid INTEGER PRIMARY KEY, "
                    "item_id TEXT NOT NULL UNIQUE)"
                )
                vec_table = f"vec_wisdom_v{version_id}"
                with conn:
                    for r in batch:
                        # sqlite-vec needs an integer rowid; derive a stable
                        # 60-bit int from the (variable-length, not-pure-hex)
                        # item_id by hashing it. Same shape as
                        # ``_asset_id_to_rowid`` but the input domain is
                        # arbitrary text (e.g. ``W-3a8f1c``), not a sha256
                        # hex string. Birthday collisions are detected
                        # explicitly so a hit corrupts no data.
                        rowid = _wisdom_id_to_rowid(r.item_id)
                        existing = conn.execute(
                            f"SELECT item_id FROM {rowid_table} WHERE rowid = ?",
                            (rowid,),
                        ).fetchone()
                        if existing is not None and existing["item_id"] != r.item_id:
                            raise StorageError(
                                f"wisdom rowid collision in version {version_id}: "
                                f"rowid {rowid} already used by "
                                f"{existing['item_id']!r}, refused for "
                                f"{r.item_id!r}"
                            )
                        conn.execute(
                            "INSERT OR REPLACE INTO wisdom_embed_meta"
                            "(item_id, version_id) VALUES (?, ?)",
                            (r.item_id, r.version_id),
                        )
                        # sqlite-vec's vec0 virtual table doesn't honor
                        # INSERT OR REPLACE on the rowid PK — re-upserting
                        # the same item_id would fail with a UNIQUE
                        # violation. Delete-then-insert mimics upsert
                        # semantics cleanly.
                        conn.execute(
                            f"DELETE FROM {vec_table} WHERE rowid = ?",
                            (rowid,),
                        )
                        conn.execute(
                            f"INSERT INTO {vec_table}(rowid, embedding) "
                            "VALUES (?, ?)",
                            (rowid, _serialize_vec(r.embedding)),
                        )
                        conn.execute(
                            f"INSERT OR REPLACE INTO {rowid_table}"
                            "(rowid, item_id) VALUES (?, ?)",
                            (rowid, r.item_id),
                        )

        await asyncio.to_thread(_run)

    async def vec_search_wisdom(
        self,
        embedding: list[float],
        *,
        version_id: int,
        limit: int = 20,
    ) -> list[WisdomVecHit]:
        def _run() -> list[WisdomVecHit]:
            conn = self._require_conn()
            version = _fetch_version(conn, version_id)
            if version is None:
                raise NotSupported(
                    f"no embed version_id={version_id} registered"
                )
            if version.modality != "text":
                raise StorageError(
                    f"vec_search_wisdom requires modality='text'; "
                    f"version {version_id} has modality={version.modality!r}"
                )
            if len(embedding) != version.dim:
                raise StorageError(
                    f"query embedding dim {len(embedding)} != "
                    f"version {version_id} dim {version.dim}"
                )
            table = f"vec_wisdom_v{version_id}"
            row_table = f"wisdom_vec_rowid_v{version_id}"
            # Empty index → no hits. Skip the SQL to avoid sqlite-vec errors
            # on a non-existent virtual table.
            tbl_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if tbl_exists is None:
                return []
            ranked = _knn(conn, table, embedding, limit)
            if not ranked:
                return []
            rowids = [rid for rid, _ in ranked]
            placeholders = ",".join("?" * len(rowids))
            item_id_by_rowid: dict[int, str] = {
                int(r["rowid"]): r["item_id"]
                for r in conn.execute(
                    f"SELECT rowid, item_id FROM {row_table} "
                    f"WHERE rowid IN ({placeholders})",
                    rowids,
                ).fetchall()
            }
            return [
                WisdomVecHit(item_id=item_id_by_rowid[rid], distance=dist)
                for rid, dist in ranked
                if rid in item_id_by_rowid
            ]

        return await asyncio.to_thread(_run)

    # ---- Embedding version registry --------------------------------------

    async def upsert_embed_version(self, v: EmbeddingVersion) -> int:
        def _run() -> int:
            conn = self._require_conn()
            with conn:
                # Match key includes modality so a CLIP-style model can
                # register both text and multimodal versions side by side.
                row = conn.execute(
                    """
                    SELECT version_id FROM embed_versions
                    WHERE provider = ? AND model = ? AND revision = ?
                      AND dim = ? AND normalize = ? AND distance = ?
                      AND modality = ?
                    """,
                    (
                        v.provider,
                        v.model,
                        v.revision,
                        v.dim,
                        int(v.normalize),
                        v.distance,
                        v.modality,
                    ),
                ).fetchone()
                if row is not None:
                    found_id = int(row["version_id"])
                    # Reactivate the matched row + demote any other actives
                    # of the same modality. Otherwise A→B→A flips leave B
                    # active forever, and writes routed to A's vec table
                    # become invisible to query()'s active-version resolve.
                    conn.execute(
                        "UPDATE embed_versions SET is_active = (version_id = ?) "
                        "WHERE modality = ?",
                        (found_id, v.modality),
                    )
                    return found_id
                # New version: insert and demote prior active versions of
                # the same modality.
                conn.execute(
                    "UPDATE embed_versions SET is_active = 0 WHERE modality = ?",
                    (v.modality,),
                )
                created_ts = (
                    v.created_ts if v.created_ts is not None else time.time()
                )
                cur = conn.execute(
                    """
                    INSERT INTO embed_versions(
                        provider, model, revision, dim, normalize, distance,
                        modality, created_ts, is_active
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        v.provider,
                        v.model,
                        v.revision,
                        v.dim,
                        int(v.normalize),
                        v.distance,
                        v.modality,
                        created_ts,
                    ),
                )
                return int(cur.lastrowid or 0)

        return await asyncio.to_thread(_run)

    async def get_active_embed_version(
        self, *, modality: Literal["text", "multimodal"]
    ) -> EmbeddingVersion | None:
        def _run() -> EmbeddingVersion | None:
            conn = self._require_conn()
            row = conn.execute(
                """
                SELECT * FROM embed_versions
                WHERE modality = ? AND is_active = 1
                ORDER BY version_id DESC LIMIT 1
                """,
                (modality,),
            ).fetchone()
            return _row_to_embed_version(row) if row is not None else None

        return await asyncio.to_thread(_run)

    async def list_embed_versions(
        self, *, modality: Literal["text", "multimodal"] | None = None
    ) -> list[EmbeddingVersion]:
        def _run() -> list[EmbeddingVersion]:
            conn = self._require_conn()
            if modality is None:
                rows = conn.execute(
                    "SELECT * FROM embed_versions ORDER BY version_id"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM embed_versions WHERE modality = ? "
                    "ORDER BY version_id",
                    (modality,),
                ).fetchall()
            return [_row_to_embed_version(r) for r in rows]

        return await asyncio.to_thread(_run)

    # ---- diagnostics -----------------------------------------------------

    async def counts(self) -> StorageCounts:
        def _run() -> StorageCounts:
            conn = self._require_conn()
            by_layer: dict[str, int] = {}
            for row in conn.execute(
                "SELECT layer, COUNT(*) AS n FROM documents WHERE active = 1 GROUP BY layer"
            ).fetchall():
                by_layer[row["layer"]] = int(row["n"])
            chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            embeddings = int(conn.execute("SELECT COUNT(*) FROM chunk_embed_meta").fetchone()[0])
            links = int(conn.execute("SELECT COUNT(*) FROM links").fetchone()[0])
            by_status: dict[str, int] = {}
            for row in conn.execute(
                "SELECT status, COUNT(*) AS n FROM wisdom_items GROUP BY status"
            ).fetchall():
                by_status[row["status"]] = int(row["n"])
            last_log = conn.execute("SELECT MAX(ts) AS ts FROM wiki_log").fetchone()
            assets_count = int(conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0])
            asset_emb_count = int(
                conn.execute("SELECT COUNT(*) FROM asset_embed_meta").fetchone()[0]
            )
            return StorageCounts(
                documents_by_layer=by_layer,
                chunks=chunks,
                embeddings=embeddings,
                links=links,
                wisdom_by_status=by_status,
                last_wiki_log_ts=float(last_log["ts"]) if last_log and last_log["ts"] else None,
                assets=assets_count,
                asset_embeddings=asset_emb_count,
            )

        return await asyncio.to_thread(_run)

    # ---- internals -------------------------------------------------------

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StorageError("SQLiteStorage is not connected; call `connect()` first")
        return self._conn

    def _verify_vec_tables_use_cosine(self, conn: sqlite3.Connection) -> None:
        """Refuse to open a DB whose vec0 tables predate distance_metric=cosine.

        ``CREATE VIRTUAL TABLE IF NOT EXISTS`` makes the cosine clause a
        no-op against an existing table, so an upgraded user would
        silently get vec0's L2 default ranking — wrong order, no
        exception. Inspect the stored CREATE statement and bail out
        loudly with rebuild instructions before the engine serves
        miscalibrated results.
        """
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND sql LIKE 'CREATE VIRTUAL TABLE%vec0%'"
        ).fetchall()
        for row in rows:
            sql = row["sql"] or ""
            if f"distance_metric={_VEC_DISTANCE_METRIC}" not in sql:
                raise StorageError(
                    f"vector table {row['name']!r} was created without "
                    f"distance_metric={_VEC_DISTANCE_METRIC} and would rank "
                    f"by sqlite-vec's L2 default. {_REBUILD_HINT}"
                )

    def _ensure_vec_table(
        self,
        conn: sqlite3.Connection,
        kind: Literal["chunks", "assets", "wisdom"],
        version_id: int,
        dim: int,
    ) -> None:
        """Create ``vec_{kind}_v<version_id>`` lazily. sqlite-vec needs
        the embedding dim baked into the table at CREATE time, so each
        version gets its own dim-locked virtual table — switching model
        creates a new version + new table, leaving prior data intact.

        ``kind`` widens to include ``"wisdom"`` so the W layer can ride
        on the active text version's vector space (chunks and wisdom
        coexist in the same cosine space; the per-kind vec table only
        differs by which identity column the rowid maps to).
        """
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_{kind}_v{version_id} "
            f"USING vec0(embedding float[{dim}] "
            f"distance_metric={_VEC_DISTANCE_METRIC})"
        )


# ---- module-level helpers ------------------------------------------------


def _read_schema_version_sqlite(conn: sqlite3.Connection) -> int | None:
    """Return ``meta_kv['schema_version']`` as an int, or ``None`` if absent
    or unparseable. ``None`` keys the fresh-DB branch in ``migrate()``.
    """
    row = conn.execute(
        "SELECT value FROM meta_kv WHERE key = ?", (SCHEMA_VERSION_KEY,)
    ).fetchone()
    if row is None:
        return None
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return None


def _write_schema_version_sqlite(conn: sqlite3.Connection, n: int) -> None:
    conn.execute(
        "INSERT INTO meta_kv(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (SCHEMA_VERSION_KEY, str(n)),
    )


def _knn(
    conn: sqlite3.Connection, table: str, embedding: list[float], k: int
) -> list[tuple[int, float]]:
    """Run sqlite-vec MATCH+k= on ``table`` and return (rowid, distance) pairs.

    Drops zero-vector rows whose distance comes back NULL/NaN — sqlite-vec
    returns NULL for cosine on the zero vector (mirrors postgres' guard
    after commit 6ecd539); ``float(None)`` would otherwise crash the
    caller. Result is in KNN order (ascending distance).

    Known limitation: vec0's MATCH path doesn't accept
    ``WHERE distance IS NOT NULL`` constraints, so degenerate rows
    consume KNN slots and the caller may under-fill ``limit`` when
    the index contains zero-vector rows. Tolerable today because zero
    vectors are exceptional (provider degeneracies, not typical
    workload). Follow-up: filter at upsert_embeddings time instead.
    """
    serialized = _serialize_vec(embedding)
    rows = conn.execute(
        f"SELECT rowid, distance AS dist "
        f"FROM {table} WHERE embedding MATCH ? AND k = ?",
        (serialized, k),
    ).fetchall()
    ranked: list[tuple[int, float]] = []
    for r in rows:
        d = r["dist"]
        if d is None:
            continue
        df = float(d)
        if math.isnan(df):
            continue
        ranked.append((int(r["rowid"]), df))
    return ranked


def _row_to_chunk(row: sqlite3.Row) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        seq=row["seq"],
        start=row["start_off"],
        end=row["end_off"],
        text=row["text"],
    )


def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row["doc_id"],
        path=row["path"],
        path_key=row["path_key"],
        title=row["title"],
        hash=row["hash"],
        mtime=float(row["mtime"] or 0.0),
        layer=Layer(row["layer"]),
        active=bool(row["active"]),
    )


def _row_to_link(row: sqlite3.Row) -> LinkRecord:
    return LinkRecord(
        src_doc_id=row["src_doc_id"],
        dst_path=row["dst_path"],
        link_type=LinkType(row["link_type"]),
        anchor=row["anchor"],
        line=int(row["line"]),
    )


def _row_to_wisdom(row: sqlite3.Row) -> WisdomItem:
    return WisdomItem(
        item_id=row["item_id"],
        kind=WisdomKind(row["kind"]),
        status=WisdomStatus(row["status"]),
        path=row["path"],
        title=row["title"],
        body=row["body"],
        confidence=float(row["confidence"]),
        created_ts=float(row["created_ts"]),
        approved_ts=float(row["approved_ts"]) if row["approved_ts"] is not None else None,
    )


def _row_to_asset(row: sqlite3.Row) -> AssetRecord:
    return AssetRecord(
        asset_id=row["asset_id"],
        kind=AssetKind(row["kind"]),
        mime=row["mime"],
        stored_path=row["stored_path"],
        original_paths=json.loads(row["original_paths"]),
        bytes=int(row["bytes"]),
        media_meta=load_media_meta(row["media_meta"]),
        created_ts=float(row["created_ts"]),
    )


def _row_to_embed_version(row: sqlite3.Row) -> EmbeddingVersion:
    return EmbeddingVersion(
        version_id=int(row["version_id"]),
        provider=row["provider"],
        model=row["model"],
        revision=row["revision"],
        dim=int(row["dim"]),
        normalize=bool(row["normalize"]),
        distance=row["distance"],
        modality=row["modality"],
        created_ts=float(row["created_ts"]),
        is_active=bool(row["is_active"]),
    )


def _fetch_version(
    conn: sqlite3.Connection, version_id: int
) -> EmbeddingVersion | None:
    row = conn.execute(
        "SELECT * FROM embed_versions WHERE version_id = ?", (version_id,)
    ).fetchone()
    return _row_to_embed_version(row) if row is not None else None


def _asset_id_to_rowid(asset_id: str) -> int:
    """Stable, signed-INT64-safe rowid derived from the sha256 asset_id.

    sqlite-vec uses INTEGER rowids; we need a deterministic mapping from
    the hex asset_id to a Python int that fits in sqlite3's signed 64-bit
    rowid space. Take the first 15 hex chars (60 bits) so the result is
    always non-negative and safely below 2**63.
    """
    return int(asset_id[:15], 16)


def _wisdom_id_to_rowid(item_id: str) -> int:
    """Stable, signed-INT64-safe rowid derived from a wisdom item_id.

    Mirror of ``_asset_id_to_rowid`` for the W-layer. Wisdom item_ids are
    short text (e.g. ``W-3a8f1c``) — not raw sha256 hex — so we hash
    first to get a uniform high-entropy domain, then take the first 15
    hex chars (60 bits, ~2^-30 birthday collision odds at 10^9 items).
    Collisions are detected explicitly at upsert time so a hit corrupts
    no data.
    """
    return int(hash_bytes(item_id.encode("utf-8"))[:15], 16)
