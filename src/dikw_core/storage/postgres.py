"""Postgres + pgvector storage adapter (Phase 5, optional extra).

Uses ``psycopg[binary,pool]`` v3 asynchronously and the ``pgvector`` Python
bindings to mirror the SQLite adapter's contract against a multi-writer
database. Install via ``uv pip install dikw-core[postgres]``.

Schema lives in ``storage/migrations/postgres/``; the per-version
``vec_chunks_v<id>`` / ``vec_assets_v<id>`` tables are created lazily at
first embedding insert so the vector dimension is parameterised into
the ``vector(<dim>)`` column type. Mirrors the SQLite adapter exactly.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Iterable, Sequence
from importlib import resources
from typing import TYPE_CHECKING, Any, Literal

from ..domains.data.path_norm import normalize_path as _normalize_path
from ..domains.info.tokenize import (
    WORD_OR_CJK_CHARS,
    CjkTokenizer,
    initialize_jieba,
    preprocess_for_fts,
)
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
    ProvenanceEdge,
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
from ._vec_codec import deserialize_vec, serialize_vec
from .base import NotSupported, StorageError

if TYPE_CHECKING:  # imports happen in connect() so base install works without pg deps
    from psycopg import AsyncConnection
    from psycopg_pool import AsyncConnectionPool


MIGRATIONS_PACKAGE = "dikw_core.storage.migrations.postgres"

_REBUILD_HINT = (
    "dikw-core is pre-alpha — the on-disk format is allowed to change, "
    "and there is no in-place upgrade path. Drop the configured Postgres "
    "schema (e.g. ``DROP SCHEMA <schema> CASCADE``) and re-run "
    "``dikw client ingest`` to rebuild."
)


class PostgresStorage:
    """Storage Protocol impl on top of Postgres + pgvector."""

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "dikw",
        pool_size: int = 10,
        cjk_tokenizer: CjkTokenizer = "none",
    ) -> None:
        self._dsn = dsn
        self._schema = schema
        self._pool_size = pool_size
        self._pool: AsyncConnectionPool | None = None
        # Must match `_sanitize_fts` on the query side — locked at first
        # ingest via `RetrievalConfig.cjk_tokenizer`.
        self._cjk_tokenizer: CjkTokenizer = cjk_tokenizer

    # ---- lifecycle -------------------------------------------------------

    async def connect(self) -> None:
        try:
            import psycopg
            from psycopg_pool import AsyncConnectionPool
        except ImportError as e:  # pragma: no cover - exercised only without extras
            raise StorageError(
                "Postgres adapter requires the `postgres` extra — "
                "install via `uv pip install dikw-core[postgres]`"
            ) from e

        # The vector extension must exist before the pool hands out
        # connections because the SQL below uses ``::vector`` casts;
        # those need the extension type to be present.
        boot = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
        try:
            async with boot.cursor() as cur:
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")
        finally:
            await boot.close()

        # pgvector types flow through explicit ``::vector`` casts in the SQL
        # so we don't need a type-registration configure hook here — the
        # extension only needs to exist (which we guaranteed above).

        self._pool = AsyncConnectionPool(
            conninfo=self._dsn,
            min_size=1,
            max_size=self._pool_size,
            kwargs={"autocommit": False},
            open=False,
        )
        await self._pool.open()

        # Warm up jieba's dictionary now so the ~0.3 s load lands in
        # `connect()` instead of the first `replace_chunks` call.
        if self._cjk_tokenizer == "jieba":
            await asyncio.to_thread(initialize_jieba)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def migrate(self) -> None:
        """Apply ``schema.sql`` to a fresh DB, or refuse a stale fingerprint.

        See ``storage/_schema.py`` for the rebuild-on-incompatibility
        policy and the ``SCHEMA_VERSION`` fingerprint contract.
        Extensions + the search-path schema were already created in
        ``connect()`` so the pool can register pgvector before this
        method runs.
        """
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                # meta_kv must exist before the version-read query.
                await cur.execute(
                    "CREATE TABLE IF NOT EXISTS meta_kv ("
                    "  key TEXT PRIMARY KEY, value TEXT NOT NULL"
                    ")"
                )
                stored = await _read_schema_version_pg(cur)
                if stored is None:
                    # ``stored is None`` should mean "fresh DB" but a
                    # pre-fingerprint install has the same shape:
                    # tables exist, no version row. ``schema.sql`` is
                    # all ``CREATE … IF NOT EXISTS``, so silently
                    # applying it would leave legacy tables in place
                    # and stamp the DB as current. Loud-fail instead,
                    # consistent with the rebuild-on-incompatibility
                    # policy.
                    await cur.execute("SELECT to_regclass('documents')")
                    legacy_row = await cur.fetchone()
                    if legacy_row is not None and legacy_row[0] is not None:
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
                    await cur.execute(sql)
                    await _write_schema_version_pg(cur, SCHEMA_VERSION)
                elif stored != SCHEMA_VERSION:
                    raise StorageError(mismatch_message(stored, _REBUILD_HINT))
            await conn.commit()

    # ---- D layer ---------------------------------------------------------

    async def upsert_document(self, doc: DocumentRecord) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO documents(
                        doc_id, path, path_key, title, hash, mtime, layer, active
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        path = EXCLUDED.path,
                        path_key = EXCLUDED.path_key,
                        title = EXCLUDED.title,
                        hash = EXCLUDED.hash,
                        mtime = EXCLUDED.mtime,
                        layer = EXCLUDED.layer,
                        active = EXCLUDED.active
                    """,
                    (
                        doc.doc_id,
                        doc.path,
                        doc.path_key,
                        doc.title,
                        doc.hash,
                        doc.mtime,
                        doc.layer.value,
                        doc.active,
                    ),
                )
            await conn.commit()

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT doc_id, path, path_key, title, hash, mtime, layer, active "
                "FROM documents WHERE doc_id = %s",
                (doc_id,),
            )
            row = await cur.fetchone()
        return _row_to_document(row) if row else None

    async def get_documents(
        self, doc_ids: Iterable[str]
    ) -> list[DocumentRecord]:
        ids = list(doc_ids)
        if not ids:
            return []
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT doc_id, path, path_key, title, hash, mtime, layer, active "
                "FROM documents WHERE doc_id = ANY(%s)",
                (ids,),
            )
            rows = await cur.fetchall()
        return [_row_to_document(r) for r in rows]

    async def list_documents(
        self,
        *,
        layer: Layer | None = None,
        active: bool | None = True,
        since_ts: float | None = None,
    ) -> Iterable[DocumentRecord]:
        sql = (
            "SELECT doc_id, path, path_key, title, hash, mtime, layer, active "
            "FROM documents WHERE TRUE"
        )
        params: list[Any] = []
        if layer is not None:
            sql += " AND layer = %s"
            params.append(layer.value)
        if active is not None:
            sql += " AND active = %s"
            params.append(bool(active))
        if since_ts is not None:
            sql += " AND mtime >= %s"
            params.append(since_ts)
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [_row_to_document(r) for r in rows]

    async def deactivate_document(self, doc_id: str) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE documents SET active = FALSE WHERE doc_id = %s",
                    (doc_id,),
                )
            await conn.commit()

    async def delete_document(self, doc_id: str) -> None:
        # FK cascade chain on the Postgres schema:
        #   documents → chunks (CASCADE)
        #   chunks → chunk_embed_meta, chunk_asset_refs (CASCADE)
        #   chunks → vec_chunks_v<v> (CASCADE, per ``_ensure_vec_table``)
        # ``links`` and ``wisdom_evidence`` both reference documents
        # WITHOUT cascade, so wipe them explicitly first. Inbound
        # (``dst_path``) links stay so the next lint surfaces
        # broken_wikilink from the referrers; the wisdom_item rows
        # themselves stay — they may have evidence from other docs.
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM links WHERE src_doc_id = %s", (doc_id,)
                )
                # ``provenance`` is FK-less by design (see ADR-0001);
                # mirrors the explicit ``links`` cleanup above. Inbound
                # rows (other K-pages whose frontmatter points at this
                # doc) survive — api.read_provenance surfaces them with
                # ``resolved=False``.
                await cur.execute(
                    "DELETE FROM provenance WHERE src_doc_id = %s", (doc_id,)
                )
                await cur.execute(
                    "DELETE FROM wisdom_evidence WHERE doc_id = %s", (doc_id,),
                )
                await cur.execute(
                    "DELETE FROM documents WHERE doc_id = %s", (doc_id,)
                )
            await conn.commit()

    # ---- I layer ---------------------------------------------------------

    async def replace_chunks(
        self, doc_id: str, chunks: Sequence[ChunkRecord]
    ) -> list[int]:
        async with self._acquire() as conn:
            ids: list[int] = []
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
                for chunk in chunks:
                    # `chunks.text` stores the original text; `chunks.fts`
                    # is built from the same `preprocess_for_fts` output
                    # the SQLite path uses, so jieba segmentation lands
                    # in the index-side tsvector and matches the
                    # `_sanitize_fts` query side byte-for-byte.
                    fts_text = preprocess_for_fts(
                        chunk.text, tokenizer=self._cjk_tokenizer
                    )
                    await cur.execute(
                        "INSERT INTO chunks(doc_id, seq, start_off, end_off, text, fts) "
                        "VALUES (%s, %s, %s, %s, %s, to_tsvector('simple', %s)) "
                        "RETURNING chunk_id",
                        (
                            doc_id,
                            chunk.seq,
                            chunk.start,
                            chunk.end,
                            chunk.text,
                            fts_text,
                        ),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise StorageError("failed to insert chunk")
                    ids.append(int(row[0]))
            await conn.commit()
            return ids

    async def upsert_embeddings(self, rows: Sequence[EmbeddingRow]) -> None:
        if not rows:
            return
        # Group by version so each per-version vec table is touched once.
        by_version: dict[int, list[EmbeddingRow]] = {}
        for r in rows:
            by_version.setdefault(r.version_id, []).append(r)
        async with self._acquire() as conn:
            for version_id, batch in by_version.items():
                version = await _fetch_version_pg(conn, version_id)
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
                await self._ensure_vec_table(conn, "chunks", version_id, version.dim)
                vec_table = f"vec_chunks_v{version_id}"
                async with conn.cursor() as cur:
                    for r in batch:
                        await cur.execute(
                            "INSERT INTO chunk_embed_meta(chunk_id, version_id) "
                            "VALUES (%s, %s) "
                            "ON CONFLICT (chunk_id, version_id) DO NOTHING",
                            (r.chunk_id, version_id),
                        )
                        await cur.execute(
                            f"INSERT INTO {vec_table}(chunk_id, embedding) "
                            "VALUES (%s, %s::vector) "
                            "ON CONFLICT (chunk_id) DO UPDATE "
                            "SET embedding = EXCLUDED.embedding",
                            (r.chunk_id, list(r.embedding)),
                        )
            await conn.commit()

    async def get_cached_embeddings(
        self, content_hashes: Sequence[str], *, version_id: int
    ) -> dict[str, list[float]]:
        hashes = list(content_hashes)
        if not hashes:
            return {}
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT content_hash, dim, embedding FROM embed_cache "
                "WHERE version_id = %s AND content_hash = ANY(%s)",
                (version_id, hashes),
            )
            rows = await cur.fetchall()
        # Cross-backend byte-exact contract: see storage/_vec_codec.py.
        return {str(r[0]): deserialize_vec(bytes(r[2]), int(r[1])) for r in rows}

    async def get_chunk_embeddings(
        self,
        chunk_ids: Sequence[int],
        *,
        version_id: int | None = None,
    ) -> dict[int, list[float]]:
        ids = list(chunk_ids)
        if not ids:
            return {}
        async with self._acquire() as conn:
            resolved = version_id
            if resolved is None:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT version_id FROM embed_versions "
                        "WHERE modality = 'text' AND is_active = TRUE "
                        "ORDER BY version_id DESC LIMIT 1"
                    )
                    row = await cur.fetchone()
                if row is None:
                    return {}
                resolved = int(row[0])
            else:
                version = await _fetch_version_pg(conn, resolved)
                if version is None:
                    return {}
            vec_table = f"vec_chunks_v{resolved}"
            # Per-version vec table is created lazily on first
            # ``upsert_embeddings``. Returning ``{}`` lets callers
            # (synth's retrieval-gated branch) degrade rather than
            # crash on a version with zero indexed chunks.
            async with conn.cursor() as cur:
                await cur.execute("SELECT to_regclass(%s)", (vec_table,))
                exists_row = await cur.fetchone()
            if exists_row is None or exists_row[0] is None:
                return {}

            # ``embedding::real[]`` casts pgvector's ``vector`` to a
            # PostgreSQL real array; psycopg deserialises that as a
            # Python ``list[float]`` without needing pgvector type
            # registration on the connection.
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT chunk_id, embedding::real[] AS vec "
                    f"FROM {vec_table} WHERE chunk_id = ANY(%s)",
                    (ids,),
                )
                rows = await cur.fetchall()
        return {int(r[0]): [float(x) for x in r[1]] for r in rows}

    async def cache_embeddings(self, rows: Sequence[CachedEmbeddingRow]) -> None:
        if not rows:
            return
        now = time.time()
        params = [
            (r.content_hash, r.version_id, r.dim, serialize_vec(list(r.embedding)), now)
            for r in rows
        ]
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                # ON CONFLICT DO NOTHING — idempotent contract; vectors
                # for the same (content_hash, version_id) are deterministic
                # so we never overwrite.
                await cur.executemany(
                    "INSERT INTO embed_cache"
                    "(content_hash, version_id, dim, embedding, created_ts) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (content_hash, version_id) DO NOTHING",
                    params,
                )
            await conn.commit()

    async def list_chunks_missing_embedding(
        self, *, version_id: int
    ) -> list[ChunkRecord]:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT chunk_id, doc_id, seq, start_off, end_off, text "
                "FROM chunks "
                "WHERE chunk_id NOT IN "
                "(SELECT chunk_id FROM chunk_embed_meta WHERE version_id = %s) "
                "ORDER BY chunk_id",
                (version_id,),
            )
            rows = await cur.fetchall()
        return [_row_to_chunk(r) for r in rows]

    async def get_chunk(self, chunk_id: int) -> ChunkRecord | None:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT chunk_id, doc_id, seq, start_off, end_off, text "
                "FROM chunks WHERE chunk_id = %s",
                (chunk_id,),
            )
            row = await cur.fetchone()
        return _row_to_chunk(row) if row is not None else None

    async def get_chunks(self, chunk_ids: Iterable[int]) -> list[ChunkRecord]:
        ids = list(chunk_ids)
        if not ids:
            return []
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT chunk_id, doc_id, seq, start_off, end_off, text "
                "FROM chunks WHERE chunk_id = ANY(%s)",
                (ids,),
            )
            rows = await cur.fetchall()
        return [_row_to_chunk(r) for r in rows]

    async def list_chunks(self, doc_id: str) -> list[ChunkRecord]:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT chunk_id, doc_id, seq, start_off, end_off, text "
                "FROM chunks WHERE doc_id = %s ORDER BY seq",
                (doc_id,),
            )
            rows = await cur.fetchall()
        return [_row_to_chunk(r) for r in rows]

    async def fts_search(
        self, q: str, *, limit: int = 20, layer: Layer | None = None
    ) -> list[FTSHit]:
        # ``info/search.py:_sanitize_fts`` produces SQLite-flavored FTS5
        # input like ``'"foo" OR "bar"'``. PG's ``plainto_tsquery`` would
        # re-tokenize that string and treat ``OR`` as a literal search
        # word — broken for any multi-word query. Translate the
        # SQLite form into PG ``to_tsquery`` syntax (``'foo | bar'``)
        # and short-circuit empty queries (``to_tsquery('')`` raises).
        parsed = _fts_to_tsquery_string(q)
        if not parsed:
            return []
        sql = (
            "SELECT c.doc_id, c.chunk_id, "
            # normalization=1: divides rank by 1 + log(document length).
            # Default 0 ignores doclength entirely, which gives long docs an
            # unfair advantage on CJK where common tokens dominate IDF-less
            # ts_rank. Closer to BM25's length normalization.
            "ts_rank(c.fts, to_tsquery('simple', %s), 1) AS score, "
            "ts_headline('simple', c.text, to_tsquery('simple', %s), "
            "  'StartSel=<mark>,StopSel=</mark>,ShortWord=2,MaxWords=25,MinWords=5') AS snip "
            "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE d.active = TRUE AND c.fts @@ to_tsquery('simple', %s)"
        )
        params: list[Any] = [parsed, parsed, parsed]
        if layer is not None:
            sql += " AND d.layer = %s"
            params.append(layer.value)
        sql += " ORDER BY score DESC LIMIT %s"
        params.append(limit)

        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

        best_by_doc: dict[str, FTSHit] = {}
        for row in rows:
            doc_id = row[0]
            if doc_id in best_by_doc:
                continue
            best_by_doc[doc_id] = FTSHit(
                doc_id=doc_id,
                chunk_id=int(row[1]),
                score=float(row[2]),
                snippet=row[3],
            )
        return list(best_by_doc.values())[:limit]

    async def vec_search(
        self,
        embedding: list[float],
        *,
        version_id: int | None = None,
        limit: int = 20,
        layer: Layer | None = None,
    ) -> list[VecHit]:
        async with self._acquire() as conn:
            resolved = version_id
            resolved_dim: int
            if resolved is None:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT version_id, dim FROM embed_versions "
                        "WHERE modality = 'text' AND is_active = TRUE "
                        "ORDER BY version_id DESC LIMIT 1"
                    )
                    row = await cur.fetchone()
                if row is None:
                    raise NotSupported("no text embeddings indexed yet")
                resolved = int(row[0])
                resolved_dim = int(row[1])
            else:
                version = await _fetch_version_pg(conn, resolved)
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
            async with conn.cursor() as cur:
                await cur.execute("SELECT to_regclass(%s)", (vec_table,))
                exists_row = await cur.fetchone()
            if exists_row is None or exists_row[0] is None:
                raise NotSupported(
                    f"no chunk vectors for version_id={resolved}"
                )

            # See storage/_vec_codec.py for the NaN/zero-vec guard rationale —
            # pgvector returns NaN on zero vectors which slips past IS NOT NULL.
            vec = list(embedding)
            sql = (
                f"SELECT cv.chunk_id, c.doc_id, (cv.embedding <=> %s::vector) AS dist "
                f"FROM {vec_table} cv JOIN chunks c ON c.chunk_id = cv.chunk_id "
                f"JOIN documents d ON d.doc_id = c.doc_id "
                f"WHERE d.active = TRUE AND (cv.embedding <=> %s::vector) IS NOT NULL"
            )
            params: list[Any] = [vec, vec]
            if layer is not None:
                sql += " AND d.layer = %s"
                params.append(layer.value)
            sql += " ORDER BY dist ASC LIMIT %s"
            params.append(limit)

            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
        return [
            VecHit(chunk_id=int(r[0]), doc_id=r[1], distance=float(r[2]))
            for r in rows
            if r[2] is not None and not math.isnan(r[2])
        ]

    # ---- K layer ---------------------------------------------------------

    async def upsert_link(self, link: LinkRecord) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO links(src_doc_id, dst_path, link_type, anchor, line)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (src_doc_id, dst_path, line) DO UPDATE SET
                        link_type = EXCLUDED.link_type,
                        anchor = EXCLUDED.anchor
                    """,
                    (link.src_doc_id, link.dst_path, link.link_type.value, link.anchor, link.line),
                )
            await conn.commit()

    async def links_from(self, src_doc_id: str) -> list[LinkRecord]:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT src_doc_id, dst_path, link_type, anchor, line "
                "FROM links WHERE src_doc_id = %s",
                (src_doc_id,),
            )
            rows = await cur.fetchall()
        return [_row_to_link(r) for r in rows]

    async def links_to(self, dst_path: str) -> list[LinkRecord]:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT src_doc_id, dst_path, link_type, anchor, line "
                "FROM links WHERE dst_path = %s",
                (dst_path,),
            )
            rows = await cur.fetchall()
        return [_row_to_link(r) for r in rows]

    async def replace_links_from(
        self, src_doc_id: str, links: Sequence[LinkRecord]
    ) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM links WHERE src_doc_id = %s", (src_doc_id,)
                )
                if links:
                    await cur.executemany(
                        """
                        INSERT INTO links(src_doc_id, dst_path, link_type, anchor, line)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (src_doc_id, dst_path, line) DO UPDATE SET
                            link_type = EXCLUDED.link_type,
                            anchor = EXCLUDED.anchor
                        """,
                        [
                            (
                                link.src_doc_id,
                                link.dst_path,
                                link.link_type.value,
                                link.anchor,
                                link.line,
                            )
                            for link in links
                        ],
                    )
            await conn.commit()

    # ---- K layer: provenance --------------------------------------------

    async def replace_provenance_from(
        self, src_doc_id: str, source_paths: Iterable[str]
    ) -> None:
        # Dedupe in Python so first-occurrence-wins on raw spelling is
        # adapter-invariant; PG's ON CONFLICT would keep EXCLUDED (last
        # one) and drift from the SQLite path.
        seen: dict[str, str] = {}
        for raw in source_paths:
            key = _normalize_path(raw)
            if key not in seen:
                seen[key] = raw

        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM provenance WHERE src_doc_id = %s",
                    (src_doc_id,),
                )
                if seen:
                    await cur.executemany(
                        """
                        INSERT INTO provenance(src_doc_id, source_path, source_path_key)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (src_doc_id, source_path_key) DO UPDATE SET
                            source_path = EXCLUDED.source_path
                        """,
                        [
                            (src_doc_id, raw, key)
                            for key, raw in seen.items()
                        ],
                    )
            await conn.commit()

    async def provenance_from(self, src_doc_id: str) -> list[ProvenanceEdge]:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT src_doc_id, source_path, source_path_key "
                "FROM provenance WHERE src_doc_id = %s "
                "ORDER BY source_path_key",
                (src_doc_id,),
            )
            rows = await cur.fetchall()
        return [
            ProvenanceEdge(
                src_doc_id=r[0], source_path=r[1], source_path_key=r[2]
            )
            for r in rows
        ]

    async def provenance_to(
        self, source_path_key: str
    ) -> list[ProvenanceEdge]:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT src_doc_id, source_path, source_path_key "
                "FROM provenance WHERE source_path_key = %s "
                "ORDER BY src_doc_id",
                (source_path_key,),
            )
            rows = await cur.fetchall()
        return [
            ProvenanceEdge(
                src_doc_id=r[0], source_path=r[1], source_path_key=r[2]
            )
            for r in rows
        ]

    async def neighbor_chunks_via_links(
        self,
        seed_chunk_ids: Sequence[int],
        *,
        layer: Layer | None = None,
        limit: int = 200,
    ) -> list[ChunkNeighborRecord]:
        if not seed_chunk_ids:
            return []
        seed_list = list(seed_chunk_ids)
        params: list[object] = [seed_list, seed_list]
        sql = (
            "SELECT c2.chunk_id, c2.doc_id, COUNT(*) AS edge_count "
            "FROM chunks c1 "
            "JOIN documents d1 ON c1.doc_id = d1.doc_id "
            "JOIN links l ON l.src_doc_id = d1.doc_id "
            "    AND l.link_type = 'wikilink' "
            "JOIN documents d2 ON l.dst_path = d2.path AND d2.active = TRUE "
            "JOIN chunks c2 ON c2.doc_id = d2.doc_id "
            "WHERE c1.chunk_id = ANY(%s::bigint[]) "
            "  AND c2.chunk_id <> ALL(%s::bigint[]) "
        )
        if layer is not None:
            sql += "AND d2.layer = %s "
            params.append(layer.value)
        sql += (
            "GROUP BY c2.chunk_id, c2.doc_id "
            "ORDER BY edge_count DESC, c2.chunk_id "
            "LIMIT %s"
        )
        params.append(limit)
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [
            ChunkNeighborRecord(
                chunk_id=row[0], doc_id=row[1], edge_count=row[2]
            )
            for row in rows
        ]

    async def append_wiki_log(self, entry: WikiLogEntry) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO wiki_log(ts, action, src, dst, note) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (entry.ts, entry.action, entry.src, entry.dst, entry.note),
                )
            await conn.commit()

    async def list_wiki_log(
        self, *, since_ts: float | None = None, limit: int | None = None
    ) -> list[WikiLogEntry]:
        sql = "SELECT id, ts, action, src, dst, note FROM wiki_log"
        params: list[Any] = []
        if since_ts is not None:
            sql += " WHERE ts >= %s"
            params.append(since_ts)
        # (ts, id) tie-break: float-second ts can collide for events
        # in the same ingest batch; BIGSERIAL id preserves insertion
        # order within a second.
        sql += " ORDER BY ts ASC, id ASC"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(int(limit))
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [
            WikiLogEntry(
                id=int(r[0]),
                ts=float(r[1]),
                action=r[2],
                src=r[3],
                dst=r[4],
                note=r[5],
            )
            for r in rows
        ]

    # ---- W layer ---------------------------------------------------------

    async def put_wisdom(
        self, item: WisdomItem, evidence: Sequence[WisdomEvidence]
    ) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO wisdom_items(
                        item_id, kind, status, path, title, body,
                        confidence, created_ts, approved_ts
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (item_id) DO UPDATE SET
                        kind = EXCLUDED.kind,
                        status = EXCLUDED.status,
                        path = EXCLUDED.path,
                        title = EXCLUDED.title,
                        body = EXCLUDED.body,
                        confidence = EXCLUDED.confidence,
                        approved_ts = EXCLUDED.approved_ts
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
                await cur.execute(
                    "DELETE FROM wisdom_evidence WHERE item_id = %s", (item.item_id,)
                )
                for ev in evidence:
                    await cur.execute(
                        "INSERT INTO wisdom_evidence(item_id, doc_id, excerpt, line) "
                        "VALUES (%s, %s, %s, %s)",
                        (item.item_id, ev.doc_id, ev.excerpt, ev.line),
                    )
            await conn.commit()

    async def list_wisdom(
        self,
        *,
        status: WisdomStatus | None = None,
        kind: WisdomKind | None = None,
    ) -> list[WisdomItem]:
        sql = (
            "SELECT item_id, kind, status, path, title, body, confidence, "
            "       created_ts, approved_ts FROM wisdom_items WHERE TRUE"
        )
        params: list[Any] = []
        if status is not None:
            sql += " AND status = %s"
            params.append(status.value)
        if kind is not None:
            sql += " AND kind = %s"
            params.append(kind.value)
        sql += " ORDER BY created_ts DESC"
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [_row_to_wisdom(r) for r in rows]

    async def set_wisdom_status(
        self,
        item_id: str,
        status: WisdomStatus,
        *,
        approved_ts: float | None = None,
    ) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                if approved_ts is None:
                    await cur.execute(
                        "UPDATE wisdom_items SET status = %s WHERE item_id = %s",
                        (status.value, item_id),
                    )
                else:
                    await cur.execute(
                        "UPDATE wisdom_items SET status = %s, approved_ts = %s "
                        "WHERE item_id = %s",
                        (status.value, approved_ts, item_id),
                    )
            await conn.commit()

    async def get_wisdom(self, item_id: str) -> WisdomItem | None:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT item_id, kind, status, path, title, body, confidence, "
                "       created_ts, approved_ts FROM wisdom_items WHERE item_id = %s",
                (item_id,),
            )
            row = await cur.fetchone()
        return _row_to_wisdom(row) if row else None

    async def get_wisdom_evidence(self, item_id: str) -> list[WisdomEvidence]:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT id, doc_id, excerpt, line FROM wisdom_evidence "
                "WHERE item_id = %s ORDER BY id ASC",
                (item_id,),
            )
            rows = await cur.fetchall()
        return [
            WisdomEvidence(
                id=int(r[0]),
                doc_id=r[1],
                excerpt=r[2],
                line=int(r[3]) if r[3] is not None else None,
            )
            for r in rows
        ]

    # ---- diagnostics -----------------------------------------------------

    async def counts(self) -> StorageCounts:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT layer, COUNT(*) FROM documents "
                "WHERE active = TRUE GROUP BY layer"
            )
            by_layer = {row[0]: int(row[1]) for row in await cur.fetchall()}
            await cur.execute("SELECT COUNT(*) FROM chunks")
            chunks = int((await cur.fetchone())[0])
            await cur.execute("SELECT COUNT(*) FROM chunk_embed_meta")
            embeddings = int((await cur.fetchone())[0])
            await cur.execute("SELECT COUNT(*) FROM links")
            links = int((await cur.fetchone())[0])
            await cur.execute(
                "SELECT status, COUNT(*) FROM wisdom_items GROUP BY status"
            )
            by_status = {row[0]: int(row[1]) for row in await cur.fetchall()}
            await cur.execute("SELECT MAX(ts) FROM wiki_log")
            last = (await cur.fetchone())[0]
            await cur.execute("SELECT COUNT(*) FROM assets")
            assets_count = int((await cur.fetchone())[0])
            await cur.execute("SELECT COUNT(*) FROM asset_embed_meta")
            asset_emb_count = int((await cur.fetchone())[0])

        return StorageCounts(
            documents_by_layer=by_layer,
            chunks=chunks,
            embeddings=embeddings,
            links=links,
            wisdom_by_status=by_status,
            last_wiki_log_ts=float(last) if last is not None else None,
            assets=assets_count,
            asset_embeddings=asset_emb_count,
        )

    # ---- multimedia assets ------------------------------------------------

    async def upsert_asset(self, asset: AssetRecord) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO assets (
                        asset_id, kind, mime, stored_path,
                        original_paths, bytes, media_meta, created_ts
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (asset_id) DO UPDATE SET
                        kind = EXCLUDED.kind,
                        mime = EXCLUDED.mime,
                        stored_path = EXCLUDED.stored_path,
                        original_paths = EXCLUDED.original_paths,
                        bytes = EXCLUDED.bytes,
                        media_meta = EXCLUDED.media_meta
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
            await conn.commit()

    async def list_assets_missing_embedding(
        self, *, version_id: int
    ) -> list[AssetRecord]:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT asset_id, kind, mime, stored_path, "
                "original_paths, bytes, media_meta, created_ts "
                "FROM assets "
                "WHERE asset_id NOT IN "
                "(SELECT asset_id FROM asset_embed_meta WHERE version_id = %s) "
                "ORDER BY asset_id",
                (version_id,),
            )
            rows = await cur.fetchall()
        return [_row_to_asset(r) for r in rows]

    async def list_wisdom_missing_embedding(
        self, *, version_id: int
    ) -> list[WisdomItem]:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT item_id, kind, status, path, title, body, confidence, "
                "       created_ts, approved_ts FROM wisdom_items "
                "WHERE item_id NOT IN "
                "(SELECT item_id FROM wisdom_embed_meta WHERE version_id = %s) "
                "ORDER BY item_id",
                (version_id,),
            )
            rows = await cur.fetchall()
        return [_row_to_wisdom(r) for r in rows]

    async def get_asset(self, asset_id: str) -> AssetRecord | None:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT asset_id, kind, mime, stored_path, "
                "original_paths, bytes, media_meta, created_ts "
                "FROM assets WHERE asset_id = %s",
                (asset_id,),
            )
            row = await cur.fetchone()
        return _row_to_asset(row) if row else None

    async def get_assets(self, asset_ids: Iterable[str]) -> list[AssetRecord]:
        ids = list(asset_ids)
        if not ids:
            return []
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT asset_id, kind, mime, stored_path, "
                "original_paths, bytes, media_meta, created_ts "
                "FROM assets WHERE asset_id = ANY(%s)",
                (ids,),
            )
            rows = await cur.fetchall()
        return [_row_to_asset(r) for r in rows if r is not None]

    async def replace_chunk_asset_refs(
        self, chunk_id: int, refs: Sequence[ChunkAssetRef]
    ) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM chunk_asset_refs WHERE chunk_id = %s",
                    (chunk_id,),
                )
                for ref in refs:
                    await cur.execute(
                        "INSERT INTO chunk_asset_refs"
                        "(chunk_id, asset_id, ord, alt, start_in_chunk, end_in_chunk) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (
                            ref.chunk_id,
                            ref.asset_id,
                            ref.ord,
                            ref.alt,
                            ref.start_in_chunk,
                            ref.end_in_chunk,
                        ),
                    )
            await conn.commit()

    async def chunk_asset_refs_for_chunks(
        self, chunk_ids: Sequence[int]
    ) -> dict[int, list[ChunkAssetRef]]:
        ids = list(chunk_ids)
        if not ids:
            return {}
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT chunk_id, asset_id, ord, alt, start_in_chunk, end_in_chunk "
                "FROM chunk_asset_refs WHERE chunk_id = ANY(%s) "
                "ORDER BY chunk_id, ord",
                (ids,),
            )
            rows = await cur.fetchall()
        out: dict[int, list[ChunkAssetRef]] = {cid: [] for cid in ids}
        for r in rows:
            out[int(r[0])].append(
                ChunkAssetRef(
                    chunk_id=int(r[0]),
                    asset_id=r[1],
                    ord=int(r[2]),
                    alt=r[3],
                    start_in_chunk=int(r[4]),
                    end_in_chunk=int(r[5]),
                )
            )
        return out

    async def chunks_referencing_assets(
        self, asset_ids: Sequence[str]
    ) -> dict[str, list[int]]:
        ids = list(asset_ids)
        if not ids:
            return {}
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT asset_id, chunk_id FROM chunk_asset_refs "
                "WHERE asset_id = ANY(%s) ORDER BY asset_id, chunk_id",
                (ids,),
            )
            rows = await cur.fetchall()
        out: dict[str, list[int]] = {aid: [] for aid in ids}
        for r in rows:
            out[r[0]].append(int(r[1]))
        return out

    async def upsert_asset_embeddings(
        self, rows: Sequence[AssetEmbeddingRow]
    ) -> None:
        if not rows:
            return
        by_version: dict[int, list[AssetEmbeddingRow]] = {}
        for r in rows:
            by_version.setdefault(r.version_id, []).append(r)
        async with self._acquire() as conn:
            for version_id, batch in by_version.items():
                version = await _fetch_version_pg(conn, version_id)
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
                await self._ensure_vec_table(conn, "assets", version_id, version.dim)
                vec_table = f"vec_assets_v{version_id}"
                async with conn.cursor() as cur:
                    for r in batch:
                        await cur.execute(
                            "INSERT INTO asset_embed_meta(asset_id, version_id) "
                            "VALUES (%s, %s) "
                            "ON CONFLICT (asset_id, version_id) DO NOTHING",
                            (r.asset_id, version_id),
                        )
                        await cur.execute(
                            f"INSERT INTO {vec_table}(asset_id, embedding) "
                            "VALUES (%s, %s::vector) "
                            "ON CONFLICT (asset_id) DO UPDATE "
                            "SET embedding = EXCLUDED.embedding",
                            (r.asset_id, list(r.embedding)),
                        )
            await conn.commit()

    async def vec_search_assets(
        self,
        embedding: list[float],
        *,
        version_id: int,
        limit: int = 20,
        layer: Layer | None = None,
    ) -> list[AssetVecHit]:
        async with self._acquire() as conn:
            version = await _fetch_version_pg(conn, version_id)
            if version is None:
                raise NotSupported(
                    f"no asset embeddings for version_id={version_id}"
                )
            if len(embedding) != version.dim:
                raise StorageError(
                    f"query embedding dim {len(embedding)} != "
                    f"version {version_id} dim {version.dim}"
                )
            vec_table = f"vec_assets_v{version_id}"
            async with conn.cursor() as cur:
                await cur.execute("SELECT to_regclass(%s)", (vec_table,))
                exists_row = await cur.fetchone()
            if exists_row is None or exists_row[0] is None:
                return []
            vec = list(embedding)
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT av.asset_id, (av.embedding <=> %s::vector) AS dist "
                    f"FROM {vec_table} av "
                    f"WHERE (av.embedding <=> %s::vector) IS NOT NULL "
                    f"ORDER BY dist ASC LIMIT %s",
                    (vec, vec, limit),
                )
                rows = await cur.fetchall()
        return [
            AssetVecHit(asset_id=r[0], distance=float(r[1]))
            for r in rows
            if r[1] is not None and not math.isnan(r[1])
        ]

    # ---- W layer: wisdom embeddings --------------------------------------

    async def upsert_wisdom_embeddings(
        self, rows: Sequence[WisdomEmbeddingRow]
    ) -> None:
        if not rows:
            return
        by_version: dict[int, list[WisdomEmbeddingRow]] = {}
        for r in rows:
            by_version.setdefault(r.version_id, []).append(r)
        async with self._acquire() as conn:
            for version_id, batch in by_version.items():
                version = await _fetch_version_pg(conn, version_id)
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
                await self._ensure_vec_table(conn, "wisdom", version_id, version.dim)
                vec_table = f"vec_wisdom_v{version_id}"
                async with conn.cursor() as cur:
                    for r in batch:
                        await cur.execute(
                            "INSERT INTO wisdom_embed_meta(item_id, version_id) "
                            "VALUES (%s, %s) "
                            "ON CONFLICT (item_id, version_id) DO NOTHING",
                            (r.item_id, version_id),
                        )
                        await cur.execute(
                            f"INSERT INTO {vec_table}(item_id, embedding) "
                            "VALUES (%s, %s::vector) "
                            "ON CONFLICT (item_id) DO UPDATE "
                            "SET embedding = EXCLUDED.embedding",
                            (r.item_id, list(r.embedding)),
                        )
            await conn.commit()

    async def vec_search_wisdom(
        self,
        embedding: list[float],
        *,
        version_id: int,
        limit: int = 20,
    ) -> list[WisdomVecHit]:
        async with self._acquire() as conn:
            version = await _fetch_version_pg(conn, version_id)
            if version is None:
                raise NotSupported(
                    f"no wisdom embeddings for version_id={version_id}"
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
            vec_table = f"vec_wisdom_v{version_id}"
            async with conn.cursor() as cur:
                await cur.execute("SELECT to_regclass(%s)", (vec_table,))
                exists_row = await cur.fetchone()
            if exists_row is None or exists_row[0] is None:
                return []
            vec = list(embedding)
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT wv.item_id, (wv.embedding <=> %s::vector) AS dist "
                    f"FROM {vec_table} wv "
                    f"WHERE (wv.embedding <=> %s::vector) IS NOT NULL "
                    f"ORDER BY dist ASC LIMIT %s",
                    (vec, vec, limit),
                )
                rows = await cur.fetchall()
        return [
            WisdomVecHit(item_id=r[0], distance=float(r[1]))
            for r in rows
            if r[1] is not None and not math.isnan(r[1])
        ]

    async def upsert_embed_version(self, v: EmbeddingVersion) -> int:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                # Match key includes modality so a CLIP-style model can
                # register both text and multimodal versions side by side.
                await cur.execute(
                    """
                    SELECT version_id FROM embed_versions
                    WHERE provider = %s AND model = %s AND revision = %s
                      AND dim = %s AND normalize = %s AND distance = %s
                      AND modality = %s
                    """,
                    (
                        v.provider,
                        v.model,
                        v.revision,
                        v.dim,
                        v.normalize,
                        v.distance,
                        v.modality,
                    ),
                )
                row = await cur.fetchone()
                if row is not None:
                    found_id = int(row[0])
                    # Reactivate the matched row + demote any other actives
                    # of the same modality (see SQLite mirror).
                    await cur.execute(
                        "UPDATE embed_versions "
                        "SET is_active = (version_id = %s) "
                        "WHERE modality = %s",
                        (found_id, v.modality),
                    )
                    await conn.commit()
                    return found_id
                # New version: insert and demote prior actives of the
                # same modality.
                await cur.execute(
                    "UPDATE embed_versions SET is_active = FALSE WHERE modality = %s",
                    (v.modality,),
                )
                created_ts = (
                    v.created_ts if v.created_ts is not None else time.time()
                )
                await cur.execute(
                    """
                    INSERT INTO embed_versions(
                        provider, model, revision, dim, normalize, distance,
                        modality, created_ts, is_active
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, TRUE)
                    RETURNING version_id
                    """,
                    (
                        v.provider,
                        v.model,
                        v.revision,
                        v.dim,
                        v.normalize,
                        v.distance,
                        v.modality,
                        created_ts,
                    ),
                )
                inserted = await cur.fetchone()
            await conn.commit()
        if inserted is None:
            raise StorageError("upsert_embed_version: insert returned no row")
        return int(inserted[0])

    async def get_active_embed_version(
        self, *, modality: Literal["text", "multimodal"]
    ) -> EmbeddingVersion | None:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT version_id, provider, model, revision, dim, normalize, "
                "distance, modality, created_ts, is_active "
                "FROM embed_versions "
                "WHERE modality = %s AND is_active = TRUE "
                "ORDER BY version_id DESC LIMIT 1",
                (modality,),
            )
            row = await cur.fetchone()
        return _row_to_embed_version_pg(row) if row else None

    async def list_embed_versions(
        self, *, modality: Literal["text", "multimodal"] | None = None
    ) -> list[EmbeddingVersion]:
        sql = (
            "SELECT version_id, provider, model, revision, dim, normalize, "
            "distance, modality, created_ts, is_active "
            "FROM embed_versions"
        )
        params: list[Any] = []
        if modality is not None:
            sql += " WHERE modality = %s"
            params.append(modality)
        sql += " ORDER BY version_id"
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [_row_to_embed_version_pg(r) for r in rows]

    # ---- internals -------------------------------------------------------

    def _acquire(self) -> Any:
        if self._pool is None:
            raise StorageError("PostgresStorage is not connected; call `connect()` first")
        return _ConnectionContext(self._pool, self._schema)

    async def _ensure_vec_table(
        self,
        conn: AsyncConnection,
        kind: Literal["chunks", "assets", "wisdom"],
        version_id: int,
        dim: int,
    ) -> None:
        """Create ``vec_{kind}_v<version_id>`` lazily. pgvector needs the
        embedding dim parameterised into the column type, so each version
        gets its own dim-locked table — switching model creates a new
        version + new table, leaving prior data intact.

        ``kind`` widens to include ``"wisdom"`` so the W layer can ride
        on the active text version's vector space (see SQLite mirror's
        rationale).
        """
        if kind == "chunks":
            pk_col, fk_table = "chunk_id BIGINT", "chunks(chunk_id)"
        elif kind == "assets":
            pk_col, fk_table = "asset_id TEXT", "assets(asset_id)"
        else:  # wisdom
            pk_col, fk_table = "item_id TEXT", "wisdom_items(item_id)"
        async with conn.cursor() as cur:
            await cur.execute(
                f"CREATE TABLE IF NOT EXISTS vec_{kind}_v{version_id} ("
                f"  {pk_col} PRIMARY KEY "
                f"  REFERENCES {fk_table} ON DELETE CASCADE,"
                f"  embedding vector({dim}) NOT NULL"
                f")"
            )
        await conn.commit()


async def _read_schema_version_pg(cur: Any) -> int | None:
    """Return ``meta_kv['schema_version']`` as an int, or ``None`` if absent
    or unparseable. ``None`` keys the fresh-DB branch in ``migrate()``.
    """
    await cur.execute(
        "SELECT value FROM meta_kv WHERE key = %s", (SCHEMA_VERSION_KEY,)
    )
    row = await cur.fetchone()
    if row is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


async def _write_schema_version_pg(cur: Any, n: int) -> None:
    await cur.execute(
        "INSERT INTO meta_kv(key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        (SCHEMA_VERSION_KEY, str(n)),
    )


class _ConnectionContext:
    """``async with``-friendly wrapper that sets the schema search path per conn."""

    def __init__(self, pool: AsyncConnectionPool, schema: str) -> None:
        self._pool = pool
        self._schema = schema
        self._cm: Any = None

    async def __aenter__(self) -> AsyncConnection:
        self._cm = self._pool.connection()
        conn: AsyncConnection = await self._cm.__aenter__()
        async with conn.cursor() as cur:
            await cur.execute(f"SET search_path TO {self._schema}, public")
        return conn

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._cm.__aexit__(*exc_info)


# ---- row → DTO helpers ---------------------------------------------------


def _row_to_chunk(row: Any) -> ChunkRecord:
    # Column order: chunk_id, doc_id, seq, start_off, end_off, text
    return ChunkRecord(
        chunk_id=int(row[0]),
        doc_id=row[1],
        seq=int(row[2]),
        start=int(row[3]),
        end=int(row[4]),
        text=row[5],
    )


def _row_to_document(row: Any) -> DocumentRecord:
    # Column order matches every SELECT in postgres.py:
    # doc_id, path, path_key, title, hash, mtime, layer, active
    return DocumentRecord(
        doc_id=row[0],
        path=row[1],
        path_key=row[2],
        title=row[3],
        hash=row[4],
        mtime=float(row[5] or 0.0),
        layer=Layer(row[6]),
        active=bool(row[7]),
    )


def _row_to_asset(row: Any) -> AssetRecord:
    return AssetRecord(
        asset_id=row[0],
        kind=AssetKind(row[1]),
        mime=row[2],
        stored_path=row[3],
        original_paths=json.loads(row[4]),
        bytes=int(row[5]),
        media_meta=load_media_meta(row[6]),
        created_ts=float(row[7]),
    )


def _row_to_link(row: Any) -> LinkRecord:
    return LinkRecord(
        src_doc_id=row[0],
        dst_path=row[1],
        link_type=LinkType(row[2]),
        anchor=row[3],
        line=int(row[4]),
    )


def _row_to_embed_version_pg(row: Any) -> EmbeddingVersion:
    return EmbeddingVersion(
        version_id=int(row[0]),
        provider=row[1],
        model=row[2],
        revision=row[3],
        dim=int(row[4]),
        normalize=bool(row[5]),
        distance=row[6],
        modality=row[7],
        created_ts=float(row[8]),
        is_active=bool(row[9]),
    )


async def _fetch_version_pg(
    conn: AsyncConnection, version_id: int
) -> EmbeddingVersion | None:
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT version_id, provider, model, revision, dim, normalize, "
            "distance, modality, created_ts, is_active "
            "FROM embed_versions WHERE version_id = %s",
            (version_id,),
        )
        row = await cur.fetchone()
    return _row_to_embed_version_pg(row) if row else None


def _row_to_wisdom(row: Any) -> WisdomItem:
    return WisdomItem(
        item_id=row[0],
        kind=WisdomKind(row[1]),
        status=WisdomStatus(row[2]),
        path=row[3],
        title=row[4],
        body=row[5],
        confidence=float(row[6]),
        created_ts=float(row[7]),
        approved_ts=float(row[8]) if row[8] is not None else None,
    )


_TS_TOKEN_RE = re.compile(r'"([^"]+)"')
# Word-or-CJK runs in the raw input become independent tokens. Anything
# else (whitespace, punctuation, tsquery operators ``& | ! ( )``) is a
# split boundary — never carried into a token, so user input can't be
# parsed as tsquery syntax.
_TS_RAW_TOKEN_RE = re.compile(rf"[{WORD_OR_CJK_CHARS}]+")


def _fts_to_tsquery_string(q: str) -> str:
    """Translate any ``Storage.fts_search`` query into PG to_tsquery form.

    Two input shapes flow into this helper:

    1. **Pre-sanitized form** from ``info/search.py:_sanitize_fts``:
       ``'"foo" OR "bar"'`` — the SQLite-flavored quoted bag-of-words.
       Strip the quotes, join with ``' | '``.
    2. **Raw form** from any direct ``Storage.fts_search`` caller
       (e.g. the ``test_chunks_and_fts_search`` contract test passes
       ``"brown"`` straight through). Pull every word/CJK run out as
       a token via ``re.findall`` so punctuation acts as a token
       boundary the same way ``plainto_tsquery`` would split it —
       ``"retrieval.rrf_k"`` becomes two tokens (``retrieval``,
       ``rrf_k``), not one collapsed lexeme.

    Empty input → empty output; the caller must short-circuit because
    ``to_tsquery('')`` raises in PG.
    """
    # Sanitized form (``'"foo" OR "bar"'``): lift tokens from inside
    # the quotes. Raw form (``foo bar`` / ``foo&bar``): pull every
    # word-or-CJK run out via the same regex the inverted character
    # class would split on. Empty/whitespace inputs return an empty
    # list either way; the caller short-circuits.
    quoted = _TS_TOKEN_RE.findall(q)
    tokens = [t for t in quoted if t] if quoted else _TS_RAW_TOKEN_RE.findall(q)
    return " | ".join(tokens)
