"""Storage contract tests.

Parameterised over SQLite and Postgres so engine code can't grow
backend-specific assumptions. Postgres requires
``DIKW_TEST_POSTGRES_DSN`` and the contract job in CI provides one;
locally the Postgres parametrisation skips when the env var is unset.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from dikw_core.schemas import (
    AssetEmbeddingRow,
    AssetKind,
    AssetRecord,
    CachedEmbeddingRow,
    ChunkAssetRef,
    ChunkRecord,
    DocumentRecord,
    EmbeddingRow,
    EmbeddingVersion,
    ImageMediaMeta,
    Layer,
    LinkRecord,
    LinkType,
    WikiLogEntry,
    WisdomEmbeddingRow,
    WisdomEvidence,
    WisdomItem,
    WisdomKind,
    WisdomStatus,
)
from dikw_core.storage._schema import SCHEMA_VERSION, SCHEMA_VERSION_KEY
from dikw_core.storage.base import NotSupported, Storage
from dikw_core.storage.sqlite import SQLiteStorage

from .fakes import register_text_version


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
async def storage(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Storage]:
    backend = request.param
    if backend == "sqlite":
        s: Storage = SQLiteStorage(
            tmp_path / "index.sqlite", cjk_tokenizer="jieba"
        )
    elif backend == "postgres":
        from dikw_core.storage.postgres import PostgresStorage

        dsn = os.environ["DIKW_TEST_POSTGRES_DSN"]
        # Use a schema derived from the test tmpdir so parallel runs don't collide.
        schema = f"dikw_test_{abs(hash(str(tmp_path))) % 10_000_000:07d}"
        s = PostgresStorage(dsn, schema=schema, pool_size=2, cjk_tokenizer="jieba")
    else:
        raise RuntimeError(f"unreachable: adapter {backend}")

    await s.connect()
    await s.migrate()
    try:
        yield s
    finally:
        # Drop the Postgres schema to keep the test DB clean between runs.
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


def _make_doc(path: str, layer: Layer = Layer.SOURCE) -> DocumentRecord:
    body_hash = f"hash-{path}"
    return DocumentRecord(
        doc_id=f"doc::{path}",
        path=path,
        title=path.rsplit("/", 1)[-1],
        hash=body_hash,
        mtime=time.time(),
        layer=layer,
        active=True,
    )


async def _column_info(storage: Storage, table: str) -> dict[str, dict[str, int]]:
    """Return ``{col_name: {'notnull': 0|1}}`` for a SQL adapter's table.

    Hides the SQLite (``PRAGMA table_info``) vs Postgres
    (``information_schema.columns``) split so schema-shape contract
    tests don't repeat the introspection branch every time.
    """
    cls_name = type(storage).__name__
    if cls_name == "SQLiteStorage":
        conn = storage._conn  # type: ignore[attr-defined]
        rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        return {r["name"]: {"notnull": int(r["notnull"])} for r in rows}
    if cls_name == "PostgresStorage":
        async with storage._acquire() as conn, conn.cursor() as cur:  # type: ignore[attr-defined]
            await cur.execute(
                "SELECT column_name, is_nullable FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = %s",
                (table,),
            )
            rows = await cur.fetchall()
        return {r[0]: {"notnull": 0 if r[1] == "YES" else 1} for r in rows}
    raise AssertionError(f"_column_info unsupported for adapter {cls_name}")


def _integrity_error_types() -> tuple[type[BaseException], ...]:
    """Adapter-specific integrity-error classes for ``pytest.raises``.

    SQLite raises ``sqlite3.IntegrityError``; Postgres raises
    ``psycopg.errors.IntegrityError``. The Postgres extra is optional,
    so import it lazily and fall back to ``sqlite3`` only.
    """
    import sqlite3

    types: list[type[BaseException]] = [sqlite3.IntegrityError]
    try:
        import psycopg

        types.append(psycopg.errors.IntegrityError)
    except ImportError:
        pass
    return tuple(types)


async def test_migrate_is_idempotent(storage: Storage) -> None:
    await storage.migrate()
    await storage.migrate()
    counts = await storage.counts()
    assert counts.chunks == 0
    assert counts.documents_by_layer == {}


async def test_meta_kv_value_is_not_null(storage: Storage) -> None:
    """``meta_kv.value`` must reject NULLs on both SQL adapters so the
    schema_version writer can never silently drop the version.
    """
    await storage.migrate()

    cols = await _column_info(storage, "meta_kv")
    value_col = cols.get("value")
    assert value_col is not None, "meta_kv.value column missing"
    assert value_col["notnull"] == 1, (
        f"meta_kv.value must be NOT NULL; got {value_col!r}"
    )


async def test_migrate_records_schema_version(storage: Storage) -> None:
    """After ``migrate()`` the SQL adapters must record ``SCHEMA_VERSION``
    in ``meta_kv['schema_version']`` so a subsequent connect that sees a
    matching fingerprint short-circuits to a no-op.
    """
    await storage.migrate()
    assert await _read_schema_version(storage) == SCHEMA_VERSION

    # Re-running migrate must not regress the version.
    await storage.migrate()
    assert await _read_schema_version(storage) == SCHEMA_VERSION


def test_sqlite_schema_declares_meta_kv() -> None:
    """``meta_kv`` must be declared in the SQLite ``schema.sql`` file
    for parity with the Postgres schema. The Python-side inline
    ``CREATE TABLE`` in ``SQLiteStorage.migrate()`` still runs first
    (it has to, so the schema-version reader can hit the row before
    ``schema.sql`` applies on a fresh DB), but a schema-diff between the
    two adapters' ``migrations/`` trees should not surface a phantom
    "Postgres has meta_kv, SQLite doesn't" because of where the table
    happens to be declared.
    """
    from importlib import resources

    sql = (
        resources.files("dikw_core.storage.migrations.sqlite")
        .joinpath("schema.sql")
        .read_text(encoding="utf-8")
    )
    assert "CREATE TABLE IF NOT EXISTS meta_kv" in sql, (
        "schema.sql must declare meta_kv for parity with the PG schema; "
        "the inline create in sqlite.py:migrate() stays but the .sql "
        "file is the documentation source of truth."
    )


async def test_schema_version_mismatch_raises(storage: Storage) -> None:
    """A DB whose ``meta_kv['schema_version']`` doesn't match the code's
    ``SCHEMA_VERSION`` must be refused at ``migrate()`` time with
    rebuild instructions. Pre-alpha policy is rebuild-on-incompatibility.
    """
    from dikw_core.storage.base import StorageError

    await storage.migrate()
    await _write_schema_version(storage, 99999)

    with pytest.raises(StorageError) as excinfo:
        await storage.migrate()
    msg = str(excinfo.value)
    assert "rebuild" in msg.lower(), msg
    # Mentions both numbers so log readers can tell what they're looking at.
    assert "v99999" in msg, msg


async def test_legacy_unfingerprinted_db_raises(storage: Storage) -> None:
    """A pre-fingerprint DB (tables present, no ``schema_version`` row)
    must be refused at ``migrate()`` time. The fresh-DB branch in
    ``migrate()`` keys off ``stored is None``, which is the same shape
    a legacy install presents — silently applying ``schema.sql``
    against legacy tables would stamp the DB as current and let later
    reads fail on missing columns. Loud-fail instead, consistent with
    the rebuild-on-incompatibility policy.
    """
    from dikw_core.storage.base import StorageError

    await storage.migrate()  # creates documents/etc + writes fingerprint
    await _delete_schema_version(storage)  # simulate pre-fingerprint shape

    with pytest.raises(StorageError) as excinfo:
        await storage.migrate()
    msg = str(excinfo.value)
    assert "rebuild" in msg.lower(), msg
    assert "schema_version" in msg, msg


async def _read_schema_version(storage: Storage) -> int:
    """Adapter-aware ``meta_kv['schema_version']`` reader for tests."""
    cls_name = type(storage).__name__
    if cls_name == "SQLiteStorage":
        conn = storage._conn  # type: ignore[attr-defined]
        row = conn.execute(
            "SELECT value FROM meta_kv WHERE key = ?", (SCHEMA_VERSION_KEY,)
        ).fetchone()
        return 0 if row is None else int(row["value"])
    if cls_name == "PostgresStorage":
        async with storage._acquire() as conn, conn.cursor() as cur:  # type: ignore[attr-defined]
            await cur.execute(
                "SELECT value FROM meta_kv WHERE key = %s", (SCHEMA_VERSION_KEY,)
            )
            row = await cur.fetchone()
        return 0 if row is None else int(row[0])
    raise AssertionError(f"unknown adapter {cls_name}")


async def _write_schema_version(storage: Storage, value: int) -> None:
    """Adapter-aware ``meta_kv['schema_version']`` writer for tests.

    Mirrors ``_read_schema_version`` so legacy/mismatch scenarios can be
    set up without re-doing the cls_name + private-member dance inline.
    """
    cls_name = type(storage).__name__
    if cls_name == "SQLiteStorage":
        conn = storage._conn  # type: ignore[attr-defined]
        with conn:
            conn.execute(
                "UPDATE meta_kv SET value = ? WHERE key = ?",
                (str(value), SCHEMA_VERSION_KEY),
            )
        return
    if cls_name == "PostgresStorage":
        async with storage._acquire() as conn, conn.cursor() as cur:  # type: ignore[attr-defined]
            await cur.execute(
                "UPDATE meta_kv SET value = %s WHERE key = %s",
                (str(value), SCHEMA_VERSION_KEY),
            )
            await conn.commit()
        return
    raise AssertionError(f"unknown adapter {cls_name}")


async def _delete_schema_version(storage: Storage) -> None:
    """Drop the ``meta_kv['schema_version']`` row to simulate a
    pre-fingerprint install for the legacy-DB regression test.
    """
    cls_name = type(storage).__name__
    if cls_name == "SQLiteStorage":
        conn = storage._conn  # type: ignore[attr-defined]
        with conn:
            conn.execute(
                "DELETE FROM meta_kv WHERE key = ?", (SCHEMA_VERSION_KEY,)
            )
        return
    if cls_name == "PostgresStorage":
        async with storage._acquire() as conn, conn.cursor() as cur:  # type: ignore[attr-defined]
            await cur.execute(
                "DELETE FROM meta_kv WHERE key = %s", (SCHEMA_VERSION_KEY,)
            )
            await conn.commit()
        return
    raise AssertionError(f"unknown adapter {cls_name}")


async def test_document_roundtrip(storage: Storage) -> None:
    doc = _make_doc("sources/a.md")
    await storage.upsert_document(doc)

    fetched = await storage.get_document(doc.doc_id)
    assert fetched is not None
    assert fetched.path == "sources/a.md"
    assert fetched.layer == Layer.SOURCE

    docs = list(await storage.list_documents(layer=Layer.SOURCE))
    assert len(docs) == 1

    await storage.deactivate_document(doc.doc_id)
    active_docs = list(await storage.list_documents(layer=Layer.SOURCE, active=True))
    assert active_docs == []


async def test_storage_has_no_put_content(storage: Storage) -> None:
    """The legacy ``put_content`` Protocol method is gone — D-layer schema
    no longer carries a write-only ``content`` table. Sentinel that catches
    accidental re-introduction of the API."""
    assert not hasattr(storage, "put_content")


async def test_upsert_document_without_put_content(storage: Storage) -> None:
    """``upsert_document`` must succeed without any prior content-table write.

    Pre-refactor this failed on SQLite/Postgres because ``documents.hash``
    was an FK to ``content(hash)``. After the refactor the FK is gone and
    ``documents.hash`` is just an indexed text column, so the call stands
    on its own.
    """
    doc = _make_doc("sources/no-content.md")
    await storage.upsert_document(doc)
    fetched = await storage.get_document(doc.doc_id)
    assert fetched is not None
    assert fetched.hash == doc.hash


async def test_documents_hash_indexed(storage: Storage) -> None:
    """``documents.hash`` must be indexed so content-addressed lookups
    (``WHERE hash = ?``) don't fall back to a sequential scan once the
    legacy ``content`` table is gone.
    """
    # Schema-introspection assertion — by design the Storage Protocol exposes no
    # raw connection, so this contract test reaches into adapter internals to
    # verify the migration actually created the index. Adding a Protocol method
    # just to dedupe one assertion would leak SQL up to the engine.
    if isinstance(storage, SQLiteStorage):
        conn = storage._conn
        assert conn is not None
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND tbl_name='documents'"
        ).fetchall()
        names = {r[0] for r in rows}
    else:  # PostgresStorage
        async with storage._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'documents' "
                "AND schemaname = current_schema()"
            )
            rows = await cur.fetchall()
        names = {r[0] for r in rows}

    assert "documents_hash_idx" in names, (
        f"expected documents_hash_idx among {sorted(names)}"
    )


async def test_get_documents_batch(storage: Storage) -> None:
    """Batch fetch is the N+1 fix used by chunk-level retrieval — every
    adapter must satisfy it (missing ids are dropped silently, not raised).
    """
    a = _make_doc("sources/batch_a.md")
    b = _make_doc("sources/batch_b.md")
    await storage.upsert_document(a)
    await storage.upsert_document(b)

    fetched = await storage.get_documents([a.doc_id, b.doc_id, "missing:nope"])
    by_id = {d.doc_id: d for d in fetched}
    assert set(by_id.keys()) == {a.doc_id, b.doc_id}
    assert by_id[a.doc_id].path == "sources/batch_a.md"
    assert by_id[b.doc_id].path == "sources/batch_b.md"

    # Empty input → empty output, no DB hit needed.
    assert await storage.get_documents([]) == []


async def test_get_chunks_batch(storage: Storage) -> None:
    """Batch chunk fetch — same contract as ``get_documents`` for the
    chunk side. Used by the search path to avoid N+1 over retrieved
    chunk_ids."""
    doc = _make_doc("sources/chunks_batch.md")
    await storage.upsert_document(doc)
    ids = await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=4, text="aaaa"),
            ChunkRecord(doc_id=doc.doc_id, seq=1, start=4, end=8, text="bbbb"),
            ChunkRecord(doc_id=doc.doc_id, seq=2, start=8, end=12, text="cccc"),
        ],
    )
    fetched = await storage.get_chunks([ids[0], ids[2], 999_999])
    by_id = {c.chunk_id: c for c in fetched}
    assert set(by_id.keys()) == {ids[0], ids[2]}
    assert by_id[ids[0]].text == "aaaa"
    assert by_id[ids[2]].text == "cccc"
    assert await storage.get_chunks([]) == []


async def test_list_chunks_by_doc_returns_seq_ordered(storage: Storage) -> None:
    """``list_chunks(doc_id)`` returns every chunk of one doc in ``seq``
    order, regardless of insertion order. Eval anchor-resolution depends
    on this — it scans the list to find the chunk covering a char position.
    """
    doc = _make_doc("sources/list_chunks.md")
    await storage.upsert_document(doc)
    other = _make_doc("sources/other.md")
    await storage.upsert_document(other)
    await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=5, text="alpha"),
            ChunkRecord(doc_id=doc.doc_id, seq=1, start=5, end=10, text="beta-"),
            ChunkRecord(doc_id=doc.doc_id, seq=2, start=10, end=15, text="gamma"),
        ],
    )
    await storage.replace_chunks(
        other.doc_id,
        [ChunkRecord(doc_id=other.doc_id, seq=0, start=0, end=4, text="zzzz")],
    )

    chunks = await storage.list_chunks(doc.doc_id)
    assert [c.seq for c in chunks] == [0, 1, 2]
    assert [c.text for c in chunks] == ["alpha", "beta-", "gamma"]
    # All chunks belong to the requested doc — no leakage from siblings.
    assert all(c.doc_id == doc.doc_id for c in chunks)
    # Empty list for an unknown doc.
    assert await storage.list_chunks("does-not-exist") == []


async def test_chunks_and_fts_search(storage: Storage) -> None:
    doc = _make_doc("sources/chunked.md")
    await storage.upsert_document(doc)

    await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=18, text="The quick brown fox"),
            ChunkRecord(doc_id=doc.doc_id, seq=1, start=18, end=36, text="jumps over the fence"),
        ],
    )
    hits = await storage.fts_search("brown", limit=5)
    assert any(h.doc_id == doc.doc_id for h in hits)
    # snippet highlighting is available
    assert any(h.snippet and "brown" in h.snippet.lower() for h in hits)


async def test_sqlite_documents_fts_body_only(storage: Storage) -> None:
    """SQLite ``documents_fts`` must scope to ``body`` only — no
    ``path UNINDEXED`` / ``title`` / ``layer UNINDEXED`` columns —
    and use ``tokenize = "unicode61"`` (no ``remove_diacritics``).

    The PG side has always indexed only ``chunks.text`` via the
    generated ``chunks.fts`` tsvector; aligning SQLite to the same
    column scope removes silent recall divergence on title-heavy
    queries. Diacritics are preserved on both sides post-alignment.
    """
    if type(storage).__name__ != "SQLiteStorage":
        pytest.skip("documents_fts is SQLite-only")
    await storage.migrate()
    conn = storage._conn  # type: ignore[attr-defined]
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'documents_fts'"
    ).fetchone()
    assert row is not None, "documents_fts virtual table missing"
    sql = row["sql"]
    assert "path UNINDEXED" not in sql, (
        f"documents_fts must not index path; got:\n{sql}"
    )
    assert "layer UNINDEXED" not in sql, (
        f"documents_fts must not index layer; got:\n{sql}"
    )
    # ``remove_diacritics 0`` must be explicit: unicode61 defaults to
    # ``1`` which still strips diacritics. We want the byte-level
    # behavior of PG's ``to_tsvector('simple', ...)``.
    assert "remove_diacritics 0" in sql, (
        f"documents_fts must explicitly pin remove_diacritics=0; got:\n{sql}"
    )


async def test_fts_preserves_diacritics(storage: Storage) -> None:
    """Cross-adapter contract: FTS preserves diacritics. ``café`` is a
    different token from ``cafe``. Searching ``"café"`` returns the
    diacritic-bearing chunk; searching ``"cafe"`` does not.

    Pre-PR SQLite stripped diacritics (``unicode61 remove_diacritics
    2``); aligning to PG behaviour (preserve as-is) keeps cross-backend
    recall consistent without pulling in the ``unaccent`` extension.
    """
    doc = _make_doc("sources/diacritics.md")
    await storage.upsert_document(doc)
    await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(
                doc_id=doc.doc_id,
                seq=0,
                start=0,
                end=18,
                text="café au lait recipe",
            ),
        ],
    )

    hits_with_diacritic = await storage.fts_search('"café"', limit=5)
    assert any(h.doc_id == doc.doc_id for h in hits_with_diacritic), (
        "querying with the exact diacritic-bearing token must hit"
    )

    hits_without_diacritic = await storage.fts_search('"cafe"', limit=5)
    assert not any(h.doc_id == doc.doc_id for h in hits_without_diacritic), (
        "querying without the diacritic must NOT hit a diacritic-only "
        "indexed token (i.e. no implicit unaccent / remove_diacritics)"
    )


async def test_fts_search_cjk_query_round_trip(storage: Storage) -> None:
    """CJK ingest → CJK query must hit on both backends under jieba.

    Regression guard for the asymmetry where PG's ``chunks.fts`` was a
    ``GENERATED ALWAYS AS to_tsvector('simple', text)`` column: the
    index side bypassed Python's jieba while the query side
    (``_sanitize_fts``) still pre-segmented, so almost no Chinese
    query could land a BM25 hit. Storage fixture pins
    ``cjk_tokenizer='jieba'`` on both adapters; this test proves the
    round-trip closes on both.
    """
    from dikw_core.domains.info.search import _sanitize_fts

    doc = _make_doc("sources/cjk.md")
    await storage.upsert_document(doc)
    await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(
                doc_id=doc.doc_id,
                seq=0,
                start=0,
                end=12,
                text="搜索引擎是信息检索的核心",
            )
        ],
    )

    sanitized = _sanitize_fts("搜索引擎", cjk_tokenizer="jieba")
    hits = await storage.fts_search(sanitized, limit=10)
    assert any(h.doc_id == doc.doc_id for h in hits), (
        f"CJK round-trip failed on {type(storage).__name__}; "
        f"sanitized query={sanitized!r}, hits={[h.doc_id for h in hits]}"
    )


def test_pg_fts_to_tsquery_string_translates_or_form() -> None:
    """``_fts_to_tsquery_string`` translates the SQLite-flavored output
    of ``info/search.py:_sanitize_fts`` (``'"foo" OR "bar"'``) into the
    PG ``to_tsquery`` form (``'foo | bar'``).

    Pure function — exercised without a Postgres fixture so the
    translation contract is locked even when CI runs without PG.
    """
    from dikw_core.storage.postgres import _fts_to_tsquery_string

    assert _fts_to_tsquery_string('"foo" OR "bar"') == "foo | bar"
    assert _fts_to_tsquery_string('"single"') == "single"
    assert _fts_to_tsquery_string("") == ""
    # Empty token list (after stripping) → empty (caller short-circuits)
    assert _fts_to_tsquery_string('""') == ""
    # CJK passes through (the helper imports WORD_OR_CJK_CHARS so
    # jieba-segmented Chinese tokens survive translation)
    assert _fts_to_tsquery_string('"机器" OR "学习"') == "机器 | 学习"
    # Raw (un-sanitized) input must also work — direct callers like
    # ``test_chunks_and_fts_search`` pass plain words. Single token:
    assert _fts_to_tsquery_string("brown") == "brown"
    # Multi-token raw input: whitespace acts as token boundary
    assert _fts_to_tsquery_string("alpha bravo") == "alpha | bravo"
    # Punctuation is also a token boundary on the raw path — same as
    # ``plainto_tsquery`` would split it. ``foo&bar`` and
    # ``retrieval.rrf_k`` must become independent tokens, not a
    # collapsed lexeme.
    assert _fts_to_tsquery_string("foo&bar") == "foo | bar"
    assert _fts_to_tsquery_string("retrieval.rrf_k") == "retrieval | rrf_k"


async def test_pg_fts_search_multi_word_or(storage: Storage) -> None:
    """PG ``fts_search`` must honor the ``OR``-joined sanitized query
    that ``info/search.py:_sanitize_fts`` produces. Pre-PR PG used
    ``plainto_tsquery('simple', q)`` which re-tokenized the literal
    string ``'"alpha" OR "bravo"'`` and treated ``OR`` as a search
    word — yielding 0 hits for any multi-word query.
    """
    if type(storage).__name__ != "PostgresStorage":
        pytest.skip("PG-specific to_tsquery translation")
    from dikw_core.domains.info.search import _sanitize_fts

    doc_a = _make_doc("sources/a.md")
    doc_b = _make_doc("sources/b.md")
    doc_c = _make_doc("sources/c.md")
    await storage.upsert_document(doc_a)
    await storage.upsert_document(doc_b)
    await storage.upsert_document(doc_c)
    await storage.replace_chunks(
        doc_a.doc_id,
        [ChunkRecord(doc_id=doc_a.doc_id, seq=0, start=0, end=5, text="alpha")],
    )
    await storage.replace_chunks(
        doc_b.doc_id,
        [ChunkRecord(doc_id=doc_b.doc_id, seq=0, start=0, end=5, text="bravo")],
    )
    await storage.replace_chunks(
        doc_c.doc_id,
        [ChunkRecord(doc_id=doc_c.doc_id, seq=0, start=0, end=7, text="charlie")],
    )

    sanitized = _sanitize_fts("alpha bravo")
    assert " OR " in sanitized, (
        f"_sanitize_fts contract changed; helper expects OR form: {sanitized!r}"
    )

    hits = await storage.fts_search(sanitized, limit=10)
    hit_doc_ids = {h.doc_id for h in hits}
    assert doc_a.doc_id in hit_doc_ids, "alpha-only doc must hit on OR query"
    assert doc_b.doc_id in hit_doc_ids, "bravo-only doc must hit on OR query"
    assert doc_c.doc_id not in hit_doc_ids, (
        "charlie has neither token; must not hit"
    )


async def test_pg_fts_search_empty_query_returns_empty(storage: Storage) -> None:
    """Sanitizer can produce an empty string when every token is
    a reserved word or punctuation. PG ``fts_search`` must short-circuit
    rather than feeding ``''`` to ``to_tsquery`` (which would raise).
    """
    if type(storage).__name__ != "PostgresStorage":
        pytest.skip("PG-specific to_tsquery short-circuit")
    hits = await storage.fts_search("", limit=5)
    assert hits == []


async def test_embeddings_and_vec_search(storage: Storage) -> None:
    doc = _make_doc("sources/vec.md")
    await storage.upsert_document(doc)
    ids = await storage.replace_chunks(
        doc.doc_id,
        [ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=5, text="alpha")],
    )
    assert len(ids) == 1
    cid = ids[0]

    try:
        version_id = await register_text_version(storage, dim=4, model="test-embed")
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")
    await storage.upsert_embeddings(
        [EmbeddingRow(chunk_id=cid, version_id=version_id, embedding=[1.0, 0.0, 0.0, 0.0])]
    )
    hits = await storage.vec_search([1.0, 0.0, 0.0, 0.0], limit=3)
    assert hits and hits[0].chunk_id == cid
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)


async def test_vec_search_skips_zero_vector_embeddings(storage: Storage) -> None:
    """Zero-vector indexed embeddings have undefined cosine distance.

    Surfaces in practice when a hashed bag-of-words embedder hits text
    outside its alphabet (e.g., FakeEmbeddings on CJK), or when an
    upstream provider returns degenerate output. Adapters must skip
    such rows rather than crash on a NULL/NaN distance.
    """
    a = _make_doc("sources/normal.md")
    b = _make_doc("sources/zero.md")
    for d in (a, b):
        await storage.upsert_document(d)
    a_ids = await storage.replace_chunks(
        a.doc_id, [ChunkRecord(doc_id=a.doc_id, seq=0, start=0, end=5, text="alpha")]
    )
    b_ids = await storage.replace_chunks(
        b.doc_id, [ChunkRecord(doc_id=b.doc_id, seq=0, start=0, end=5, text="zero")]
    )
    try:
        version_id = await register_text_version(storage, dim=4, model="test-embed")
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")
    await storage.upsert_embeddings(
        [
            EmbeddingRow(
                chunk_id=a_ids[0], version_id=version_id, embedding=[1.0, 0.0, 0.0, 0.0]
            ),
            EmbeddingRow(
                chunk_id=b_ids[0], version_id=version_id, embedding=[0.0, 0.0, 0.0, 0.0]
            ),
        ]
    )

    hits = await storage.vec_search([1.0, 0.0, 0.0, 0.0], limit=10)
    chunk_ids = {h.chunk_id for h in hits}
    assert a_ids[0] in chunk_ids, "non-zero indexed embedding must surface"
    assert b_ids[0] not in chunk_ids, (
        "zero-vector indexed embedding must be skipped (cosine distance is undefined)"
    )
    # No hit should carry NaN/None — the float() coercion would crash.
    for h in hits:
        assert h.distance == h.distance, f"NaN distance: {h}"  # NaN != NaN


# ---- embed_cache (chunk-level content-hash cache) ------------------------
#
# The cache is keyed by (sha256(chunk.text), model) and decouples
# embedding reuse from chunks.chunk_id, so re-ingest under
# replace_chunks's delete-and-reinsert semantics doesn't lose API spend
# on byte-identical chunks.


async def test_get_cached_embeddings_empty_input_no_roundtrip(
    storage: Storage,
) -> None:
    """Empty input list must short-circuit before touching the DB."""
    try:
        result = await storage.get_cached_embeddings([], version_id=1)
    except NotSupported:
        pytest.skip("backend doesn't implement embed cache")
    assert result == {}


async def test_cache_then_get_roundtrip(storage: Storage) -> None:
    """cache_embeddings → get_cached_embeddings returns the same vectors."""
    try:
        version_id = await register_text_version(storage, dim=4, model="test-embed")
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")
    rows = [
        CachedEmbeddingRow(
            content_hash="a" * 64, version_id=version_id, dim=4,
            embedding=[1.0, 0.0, 0.0, 0.0],
        ),
        CachedEmbeddingRow(
            content_hash="b" * 64, version_id=version_id, dim=4,
            embedding=[0.0, 1.0, 0.0, 0.0],
        ),
        CachedEmbeddingRow(
            content_hash="c" * 64, version_id=version_id, dim=4,
            embedding=[0.0, 0.0, 1.0, 0.0],
        ),
    ]
    try:
        await storage.cache_embeddings(rows)
        got = await storage.get_cached_embeddings(
            ["a" * 64, "b" * 64, "c" * 64, "z" * 64], version_id=version_id
        )
    except NotSupported:
        pytest.skip("backend doesn't implement embed cache")
    assert set(got.keys()) == {"a" * 64, "b" * 64, "c" * 64}
    assert got["a" * 64] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert got["b" * 64] == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert got["c" * 64] == pytest.approx([0.0, 0.0, 1.0, 0.0])


async def test_cache_embeddings_idempotent(storage: Storage) -> None:
    """Inserting the same (content_hash, version_id) twice is a no-op.

    Vectors for the same content + version are deterministic by definition,
    so the second insert must NOT raise and MUST NOT clobber the first
    (the cache is a content-addressed lookup table, not a version log).
    """
    try:
        version_id = await register_text_version(storage, dim=4, model="test-embed")
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")
    row = CachedEmbeddingRow(
        content_hash="d" * 64, version_id=version_id, dim=4,
        embedding=[0.5, 0.5, 0.5, 0.5],
    )
    try:
        await storage.cache_embeddings([row])
        await storage.cache_embeddings([row])  # second call: no-op, no exception
        got = await storage.get_cached_embeddings(["d" * 64], version_id=version_id)
    except NotSupported:
        pytest.skip("backend doesn't implement embed cache")
    assert got["d" * 64] == pytest.approx([0.5, 0.5, 0.5, 0.5])


async def test_cache_partitioned_by_version(storage: Storage) -> None:
    """Same content_hash under two different versions = two independent rows.

    Lookup must filter by version_id so a content-hash can carry vectors
    for multiple embedding versions in parallel without collision.
    """
    h = "e" * 64
    try:
        v1 = await register_text_version(storage, dim=4, model="m1")
        v2 = await register_text_version(storage, dim=4, model="m2")
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")
    try:
        await storage.cache_embeddings(
            [
                CachedEmbeddingRow(
                    content_hash=h, version_id=v1, dim=4,
                    embedding=[1.0, 0.0, 0.0, 0.0],
                ),
                CachedEmbeddingRow(
                    content_hash=h, version_id=v2, dim=4,
                    embedding=[0.0, 1.0, 0.0, 0.0],
                ),
            ]
        )
        got_v1 = await storage.get_cached_embeddings([h], version_id=v1)
        got_v2 = await storage.get_cached_embeddings([h], version_id=v2)
    except NotSupported:
        pytest.skip("backend doesn't implement embed cache")
    assert got_v1[h] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert got_v2[h] == pytest.approx([0.0, 1.0, 0.0, 0.0])


async def test_list_chunks_missing_embedding(storage: Storage) -> None:
    """Resume-scan: returns chunks without a ``chunk_embed_meta`` row for
    ``version_id``.

    A chunk that's been embedded under a DIFFERENT version still counts
    as "missing for version X" — version is part of the dedup key, the
    same chunk text could legitimately be re-embedded under a new
    version (model swap).
    """
    doc = _make_doc("sources/scan.md")
    await storage.upsert_document(doc)
    chunk_ids = await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=10, text="aaa"),
            ChunkRecord(doc_id=doc.doc_id, seq=1, start=10, end=20, text="bbb"),
            ChunkRecord(doc_id=doc.doc_id, seq=2, start=20, end=30, text="ccc"),
        ],
    )
    try:
        v1 = await register_text_version(storage, dim=4, model="m1")
        v2 = await register_text_version(storage, dim=4, model="m2")
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")
    # Embed chunk 0 under version v1, chunk 1 under v2; chunk 2 unembedded.
    try:
        await storage.upsert_embeddings(
            [
                EmbeddingRow(
                    chunk_id=chunk_ids[0], version_id=v1, embedding=[1.0, 0.0, 0.0, 0.0]
                ),
                EmbeddingRow(
                    chunk_id=chunk_ids[1], version_id=v2, embedding=[0.0, 1.0, 0.0, 0.0]
                ),
            ]
        )
    except NotSupported:
        pytest.skip("backend doesn't implement embeddings")
    missing = await storage.list_chunks_missing_embedding(version_id=v1)
    missing_ids = {c.chunk_id for c in missing}
    # Under v1: chunk 0 has it; chunks 1 + 2 don't.
    assert missing_ids == {chunk_ids[1], chunk_ids[2]}
    # Returned ChunkRecords carry the original text so the caller can
    # re-embed without a separate fetch.
    by_id = {c.chunk_id: c for c in missing}
    assert by_id[chunk_ids[1]].text == "bbb"
    assert by_id[chunk_ids[2]].text == "ccc"


async def test_vec_search_returns_in_distance_order(storage: Storage) -> None:
    """KNN MATCH must return rows ordered by ascending cosine distance.

    Builds a deterministic 5-chunk corpus where each chunk's embedding is
    a known angular distance from the query vector, then asserts the
    returned ranking matches that distance order.
    """
    doc = _make_doc("sources/order.md")
    await storage.upsert_document(doc)
    # 5 chunks, each with a 4-dim vector at increasing angle from
    # [1, 0, 0, 0]. Closest first.
    embeddings = [
        [1.00, 0.00, 0.00, 0.00],  # cosine dist = 0.0
        [0.99, 0.10, 0.00, 0.00],  # ~0.005
        [0.80, 0.60, 0.00, 0.00],  # ~0.20
        [0.50, 0.866, 0.00, 0.00],  # ~0.50
        [0.00, 1.00, 0.00, 0.00],  # 1.0
    ]
    chunks = [
        ChunkRecord(doc_id=doc.doc_id, seq=i, start=i * 5, end=i * 5 + 5, text=f"c{i}")
        for i in range(len(embeddings))
    ]
    chunk_ids = await storage.replace_chunks(doc.doc_id, chunks)
    try:
        version_id = await register_text_version(storage, dim=4, model="test-embed")
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")
    await storage.upsert_embeddings(
        [
            EmbeddingRow(chunk_id=cid, version_id=version_id, embedding=emb)
            for cid, emb in zip(chunk_ids, embeddings, strict=True)
        ]
    )
    hits = await storage.vec_search([1.0, 0.0, 0.0, 0.0], limit=5)
    assert len(hits) == 5
    # Returned ranking must follow the embedding insertion order (which
    # is also the distance order by construction).
    assert [h.chunk_id for h in hits] == chunk_ids
    # Distances must be monotone non-decreasing.
    distances = [h.distance for h in hits]
    assert distances == sorted(distances), f"distances not in order: {distances}"


async def test_vec_search_layer_filter(storage: Storage) -> None:
    """Layer filter applied after KNN must return only the requested layer.

    Inserts a SOURCE-layer doc and a WIKI-layer doc whose embeddings
    bracket the query; without the filter both surface, with
    ``layer=WIKI`` only the wiki chunk surfaces.
    """
    src_doc = _make_doc("sources/src.md", layer=Layer.SOURCE)
    wiki_doc = _make_doc("wiki/page.md", layer=Layer.WIKI)
    for d in (src_doc, wiki_doc):
        await storage.upsert_document(d)
    src_ids = await storage.replace_chunks(
        src_doc.doc_id,
        [ChunkRecord(doc_id=src_doc.doc_id, seq=0, start=0, end=5, text="src")],
    )
    wiki_ids = await storage.replace_chunks(
        wiki_doc.doc_id,
        [ChunkRecord(doc_id=wiki_doc.doc_id, seq=0, start=0, end=5, text="wiki")],
    )
    # Source chunk is the closer match; wiki chunk is the farther one.
    try:
        version_id = await register_text_version(storage, dim=4, model="test-embed")
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")
    await storage.upsert_embeddings(
        [
            EmbeddingRow(
                chunk_id=src_ids[0], version_id=version_id, embedding=[1.0, 0.0, 0.0, 0.0]
            ),
            EmbeddingRow(
                chunk_id=wiki_ids[0], version_id=version_id, embedding=[0.0, 1.0, 0.0, 0.0]
            ),
        ]
    )
    # Unfiltered: both come back, source first.
    all_hits = await storage.vec_search([1.0, 0.0, 0.0, 0.0], limit=10)
    assert {h.chunk_id for h in all_hits} == {src_ids[0], wiki_ids[0]}
    # Layer-filtered to WIKI: only the wiki chunk surfaces, even though
    # the over-fetch would have pulled the source chunk into the KNN
    # candidate set.
    wiki_hits = await storage.vec_search(
        [1.0, 0.0, 0.0, 0.0], limit=10, layer=Layer.WIKI
    )
    assert [h.chunk_id for h in wiki_hits] == [wiki_ids[0]]
    # Symmetric check.
    src_hits = await storage.vec_search(
        [1.0, 0.0, 0.0, 0.0], limit=10, layer=Layer.SOURCE
    )
    assert [h.chunk_id for h in src_hits] == [src_ids[0]]


async def test_vec_search_excludes_deactivated_doc(storage: Storage) -> None:
    """``deactivate_document`` must hide the doc's chunks from vec_search.

    The fts_search path (and the Postgres adapter's vec path) already
    filter ``documents.active``; the SQLite vec path was not — so a
    doc parked inactive by an ``api.ingest`` storage_error retry could
    leak its stale chunks into ``/v1/retrieve`` results until re-ingest.
    This test locks the contract on both adapters.
    """
    keep_doc = _make_doc("sources/keep.md", layer=Layer.SOURCE)
    drop_doc = _make_doc("sources/drop.md", layer=Layer.SOURCE)
    for d in (keep_doc, drop_doc):
        await storage.upsert_document(d)
    keep_ids = await storage.replace_chunks(
        keep_doc.doc_id,
        [ChunkRecord(doc_id=keep_doc.doc_id, seq=0, start=0, end=4, text="keep")],
    )
    drop_ids = await storage.replace_chunks(
        drop_doc.doc_id,
        [ChunkRecord(doc_id=drop_doc.doc_id, seq=0, start=0, end=4, text="drop")],
    )
    try:
        version_id = await register_text_version(storage, dim=4, model="test-embed")
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")
    await storage.upsert_embeddings(
        [
            EmbeddingRow(
                chunk_id=keep_ids[0], version_id=version_id, embedding=[1.0, 0.0, 0.0, 0.0]
            ),
            EmbeddingRow(
                chunk_id=drop_ids[0], version_id=version_id, embedding=[1.0, 0.0, 0.0, 0.0]
            ),
        ]
    )
    # Both surface while both are active.
    pre = await storage.vec_search([1.0, 0.0, 0.0, 0.0], limit=10)
    assert {h.chunk_id for h in pre} == {keep_ids[0], drop_ids[0]}

    # Deactivate one — its chunk must vanish from vec_search results.
    await storage.deactivate_document(drop_doc.doc_id)
    post = await storage.vec_search([1.0, 0.0, 0.0, 0.0], limit=10)
    assert [h.chunk_id for h in post] == [keep_ids[0]]


async def test_get_chunk_embeddings_round_trip(storage: Storage) -> None:
    """Stored chunk embeddings must be recoverable raw by chunk_id.

    Synth's existing-pages retrieval-gated mode (PR2) needs to query
    vec_search with the SAME vectors it embedded a source's chunks
    under during ingest. Re-embedding would re-pay the API cost on
    every synth call; instead we add a row-level fetch primitive that
    both adapters can satisfy from their existing chunk-embedding
    tables (SQLite vec0 ``rowid = chunk_id``; PG ``chunk_id PK``).
    """
    doc = _make_doc("sources/embed-fetch.md")
    await storage.upsert_document(doc)
    chunk_ids = await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=5, text="alpha"),
            ChunkRecord(doc_id=doc.doc_id, seq=1, start=5, end=10, text="bravo"),
        ],
    )
    assert len(chunk_ids) == 2

    try:
        version_id = await register_text_version(storage, dim=4, model="test-embed")
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")

    vec_a = [1.0, 0.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0, 0.0]
    await storage.upsert_embeddings(
        [
            EmbeddingRow(chunk_id=chunk_ids[0], version_id=version_id, embedding=vec_a),
            EmbeddingRow(chunk_id=chunk_ids[1], version_id=version_id, embedding=vec_b),
        ]
    )

    embs = await storage.get_chunk_embeddings(chunk_ids, version_id=version_id)
    assert set(embs.keys()) == set(chunk_ids)
    assert embs[chunk_ids[0]] == pytest.approx(vec_a, abs=1e-6)
    assert embs[chunk_ids[1]] == pytest.approx(vec_b, abs=1e-6)


async def test_get_chunk_embeddings_skips_unembedded(storage: Storage) -> None:
    """Chunks that exist but were never embedded are simply absent from
    the result dict — callers iterate over what they got back rather
    than guarding every lookup.
    """
    doc = _make_doc("sources/partial-embed.md")
    await storage.upsert_document(doc)
    chunk_ids = await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=5, text="alpha"),
            ChunkRecord(doc_id=doc.doc_id, seq=1, start=5, end=10, text="bravo"),
        ],
    )

    try:
        version_id = await register_text_version(storage, dim=4, model="test-embed")
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")
    # Embed only the first chunk.
    await storage.upsert_embeddings(
        [
            EmbeddingRow(
                chunk_id=chunk_ids[0],
                version_id=version_id,
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
        ]
    )

    embs = await storage.get_chunk_embeddings(chunk_ids, version_id=version_id)
    assert chunk_ids[0] in embs
    assert chunk_ids[1] not in embs


async def test_get_chunk_embeddings_empty_input_no_roundtrip(
    storage: Storage,
) -> None:
    """Empty input must short-circuit before touching the DB — mirrors
    ``get_cached_embeddings`` so synth's retrieval-gated branch stays
    cheap when a group has zero embeddable chunks.
    """
    embs = await storage.get_chunk_embeddings([], version_id=1)
    assert embs == {}


async def test_link_graph(storage: Storage) -> None:
    src_doc = _make_doc("wiki/a.md", layer=Layer.WIKI)
    dst_doc = _make_doc("wiki/b.md", layer=Layer.WIKI)
    for d in (src_doc, dst_doc):
        await storage.upsert_document(d)

    link = LinkRecord(
        src_doc_id=src_doc.doc_id,
        dst_path="wiki/b.md",
        link_type=LinkType.WIKILINK,
        anchor=None,
        line=3,
    )
    await storage.upsert_link(link)

    out = await storage.links_from(src_doc.doc_id)
    inb = await storage.links_to("wiki/b.md")
    assert len(out) == 1 and len(inb) == 1
    assert out[0].dst_path == "wiki/b.md"


async def test_replace_links_from_atomically_swaps_outgoing_set(
    storage: Storage,
) -> None:
    """``replace_links_from(src, links)`` atomically deletes all prior
    edges from ``src`` and inserts the new set in one transaction.
    Pass ``[]`` to wipe entirely; pass a fresh source's first set to
    no-op the leading delete. Used by ``_persist_wiki_page`` so
    removing a ``[[wikilink]]`` from the body actually drops the
    edge — without this the link table accumulates ghost edges as
    wiki pages are edited, polluting graph-leg retrieval and
    orphan/broken-link lint."""
    src_a = _make_doc("wiki/a.md", layer=Layer.WIKI)
    src_other = _make_doc("wiki/other.md", layer=Layer.WIKI)
    for d in (
        src_a,
        src_other,
        _make_doc("wiki/b.md", layer=Layer.WIKI),
        _make_doc("wiki/c.md", layer=Layer.WIKI),
        _make_doc("wiki/d.md", layer=Layer.WIKI),
    ):
        await storage.upsert_document(d)

    def _link(src: str, dst: str, line: int) -> LinkRecord:
        return LinkRecord(
            src_doc_id=src,
            dst_path=dst,
            link_type=LinkType.WIKILINK,
            anchor=None,
            line=line,
        )

    # Fresh src — leading DELETE no-ops, the inserts land.
    await storage.replace_links_from(
        src_a.doc_id,
        [_link(src_a.doc_id, "wiki/b.md", 1), _link(src_a.doc_id, "wiki/c.md", 2)],
    )
    await storage.upsert_link(_link(src_other.doc_id, "wiki/b.md", 1))
    initial_a = await storage.links_from(src_a.doc_id)
    assert {link.dst_path for link in initial_a} == {"wiki/b.md", "wiki/c.md"}

    # Swap src_a's set entirely: drop b/c, install d.
    await storage.replace_links_from(
        src_a.doc_id, [_link(src_a.doc_id, "wiki/d.md", 1)]
    )
    a_out = await storage.links_from(src_a.doc_id)
    assert {link.dst_path for link in a_out} == {"wiki/d.md"}
    # links_to scoped by dst_path: src_a no longer points at b;
    # src_other's edge to b survives the unrelated replace call.
    inbound_b = await storage.links_to("wiki/b.md")
    assert {link.src_doc_id for link in inbound_b} == {src_other.doc_id}

    # Empty list wipes the source's outgoing edges — the explicit
    # contract for "page body lost every wikilink".
    await storage.replace_links_from(src_a.doc_id, [])
    assert await storage.links_from(src_a.doc_id) == []
    # And src_other is still untouched.
    assert len(await storage.links_from(src_other.doc_id)) == 1

    # Idempotent: replacing an already-empty source with an empty
    # set is a no-op, not an error — _persist_wiki_page invokes
    # this unconditionally on every persist including the first.
    await storage.replace_links_from(src_a.doc_id, [])
    assert await storage.links_from(src_a.doc_id) == []


async def test_neighbor_chunks_via_links_one_hop(storage: Storage) -> None:
    """Wikilink-graph leg of HybridSearcher: from a seed chunk in
    ``page_a``, get to the chunks of ``page_b`` and ``page_c`` via
    the wikilinks ``page_a -> page_b`` and ``page_a -> page_c``."""
    page_a = _make_doc("wiki/a.md", layer=Layer.WIKI)
    page_b = _make_doc("wiki/b.md", layer=Layer.WIKI)
    page_c = _make_doc("wiki/c.md", layer=Layer.WIKI)
    unlinked = _make_doc("wiki/d.md", layer=Layer.WIKI)
    for d in (page_a, page_b, page_c, unlinked):
        await storage.upsert_document(d)

    a_chunks = await storage.replace_chunks(
        page_a.doc_id,
        [
            ChunkRecord(doc_id=page_a.doc_id, seq=0, start=0, end=10, text="a-0"),
        ],
    )
    b_chunks = await storage.replace_chunks(
        page_b.doc_id,
        [
            ChunkRecord(doc_id=page_b.doc_id, seq=0, start=0, end=10, text="b-0"),
            ChunkRecord(doc_id=page_b.doc_id, seq=1, start=10, end=20, text="b-1"),
        ],
    )
    c_chunks = await storage.replace_chunks(
        page_c.doc_id,
        [
            ChunkRecord(doc_id=page_c.doc_id, seq=0, start=0, end=10, text="c-0"),
        ],
    )
    await storage.replace_chunks(
        unlinked.doc_id,
        [
            ChunkRecord(doc_id=unlinked.doc_id, seq=0, start=0, end=10, text="d-0"),
        ],
    )

    for dst in ("wiki/b.md", "wiki/c.md"):
        await storage.upsert_link(
            LinkRecord(
                src_doc_id=page_a.doc_id,
                dst_path=dst,
                link_type=LinkType.WIKILINK,
                anchor=None,
                line=1,
            )
        )
    # Markdown-style link should NOT pull anything in (we only walk wikilinks).
    await storage.upsert_link(
        LinkRecord(
            src_doc_id=page_a.doc_id,
            dst_path="wiki/d.md",
            link_type=LinkType.MARKDOWN,
            anchor=None,
            line=2,
        )
    )

    seed_id = a_chunks[0]
    assert seed_id is not None
    neighbors = await storage.neighbor_chunks_via_links([seed_id])
    neighbor_ids = {n.chunk_id for n in neighbors}

    assert neighbor_ids == set(b_chunks) | set(c_chunks)
    assert seed_id not in neighbor_ids
    # ``neighbor_ids`` is exact above — markdown-link-only ``wiki/d.md``
    # is implicitly excluded — but lock the doc_id projection too.
    for n in neighbors:
        assert n.edge_count >= 1
        assert n.doc_id in {page_b.doc_id, page_c.doc_id}


async def test_neighbor_chunks_via_links_edge_count_ranking(
    storage: Storage,
) -> None:
    """edge_count counts links from the seed set to each neighbor —
    a neighbor reached from many seeds outranks one reached from few.
    """
    pages = {
        name: _make_doc(f"wiki/{name}.md", layer=Layer.WIKI)
        for name in ("p1", "p2", "p3", "popular", "lonely")
    }
    for d in pages.values():
        await storage.upsert_document(d)

    seed_chunks: list[int] = []
    for key in ("p1", "p2", "p3"):
        cids = await storage.replace_chunks(
            pages[key].doc_id,
            [ChunkRecord(
                doc_id=pages[key].doc_id, seq=0, start=0, end=10, text=key
            )],
        )
        seed_chunks.extend(cids)
    pop_chunks = await storage.replace_chunks(
        pages["popular"].doc_id,
        [ChunkRecord(
            doc_id=pages["popular"].doc_id, seq=0, start=0, end=10, text="pop"
        )],
    )
    lonely_chunks = await storage.replace_chunks(
        pages["lonely"].doc_id,
        [ChunkRecord(
            doc_id=pages["lonely"].doc_id, seq=0, start=0, end=10, text="lo"
        )],
    )

    # All three seeds link to ``popular``; only p1 links to ``lonely``.
    for src in ("p1", "p2", "p3"):
        await storage.upsert_link(
            LinkRecord(
                src_doc_id=pages[src].doc_id,
                dst_path="wiki/popular.md",
                link_type=LinkType.WIKILINK,
                anchor=None,
                line=1,
            )
        )
    await storage.upsert_link(
        LinkRecord(
            src_doc_id=pages["p1"].doc_id,
            dst_path="wiki/lonely.md",
            link_type=LinkType.WIKILINK,
            anchor=None,
            line=2,
        )
    )

    neighbors = await storage.neighbor_chunks_via_links(seed_chunks)
    by_chunk = {n.chunk_id: n for n in neighbors}
    assert by_chunk[pop_chunks[0]].edge_count == 3
    assert by_chunk[lonely_chunks[0]].edge_count == 1
    # Returned in edge_count desc order.
    assert neighbors[0].chunk_id == pop_chunks[0]
    assert neighbors[-1].chunk_id == lonely_chunks[0]


async def test_neighbor_chunks_via_links_layer_filter(storage: Storage) -> None:
    """``layer`` filters the *neighbor* docs — useful when the caller
    wants the graph leg constrained to WIKI pages.
    """
    src_page = _make_doc("wiki/src.md", layer=Layer.WIKI)
    wiki_target = _make_doc("wiki/target.md", layer=Layer.WIKI)
    source_target = _make_doc("sources/target.md", layer=Layer.SOURCE)
    for d in (src_page, wiki_target, source_target):
        await storage.upsert_document(d)

    src_chunk = (await storage.replace_chunks(
        src_page.doc_id,
        [ChunkRecord(doc_id=src_page.doc_id, seq=0, start=0, end=10, text="x")],
    ))[0]
    wiki_chunk = (await storage.replace_chunks(
        wiki_target.doc_id,
        [ChunkRecord(
            doc_id=wiki_target.doc_id, seq=0, start=0, end=10, text="w"
        )],
    ))[0]
    source_chunk = (await storage.replace_chunks(
        source_target.doc_id,
        [ChunkRecord(
            doc_id=source_target.doc_id, seq=0, start=0, end=10, text="s"
        )],
    ))[0]

    for dst in ("wiki/target.md", "sources/target.md"):
        await storage.upsert_link(
            LinkRecord(
                src_doc_id=src_page.doc_id,
                dst_path=dst,
                link_type=LinkType.WIKILINK,
                anchor=None,
                line=1,
            )
        )

    wiki_only = await storage.neighbor_chunks_via_links(
        [src_chunk], layer=Layer.WIKI
    )
    assert {n.chunk_id for n in wiki_only} == {wiki_chunk}
    assert source_chunk not in {n.chunk_id for n in wiki_only}


async def test_neighbor_chunks_via_links_empty_seeds(storage: Storage) -> None:
    assert await storage.neighbor_chunks_via_links([]) == []


async def test_wiki_log_append(storage: Storage) -> None:
    ts = time.time()
    await storage.append_wiki_log(
        WikiLogEntry(ts=ts, action="ingest", src="sources/a.md", dst=None, note="hello")
    )
    counts = await storage.counts()
    assert counts.last_wiki_log_ts is not None
    assert counts.last_wiki_log_ts == pytest.approx(ts, abs=1.0)


async def test_wiki_log_same_ts_tiebreak(storage: Storage) -> None:
    """Events appended within the same float-second must come back in
    insertion order; ts collisions are real (one ingest run can append
    multiple entries inside a single second).
    """
    ts = time.time()
    notes = [f"event-{i}" for i in range(5)]
    for note in notes:
        await storage.append_wiki_log(
            WikiLogEntry(ts=ts, action="ingest", src="sources/a.md", note=note)
        )

    rows = await storage.list_wiki_log(since_ts=ts)
    same_ts = [r for r in rows if r.ts == ts]
    assert [r.note for r in same_ts] == notes
    # storage layer must assign a non-None monotonic id per row
    assigned_ids = [r.id for r in same_ts]
    assert all(i is not None for i in assigned_ids)
    assert assigned_ids == sorted(assigned_ids)
    assert len(set(assigned_ids)) == len(assigned_ids)


async def test_wisdom_lifecycle(storage: Storage) -> None:
    doc = _make_doc("wiki/concept.md", layer=Layer.WIKI)
    await storage.upsert_document(doc)

    ts = time.time()
    item = WisdomItem(
        item_id="W-000001",
        kind=WisdomKind.PRINCIPLE,
        status=WisdomStatus.CANDIDATE,
        path="wisdom/_candidates/prefer-determinism.md",
        title="Prefer deterministic scoping",
        body="Use deterministic structure before invoking an LLM.",
        confidence=0.8,
        created_ts=ts,
        approved_ts=None,
    )
    evidence = [
        WisdomEvidence(doc_id=doc.doc_id, excerpt="Karpathy argues...", line=12),
        WisdomEvidence(doc_id=doc.doc_id, excerpt="...", line=18),
    ]
    await storage.put_wisdom(item, evidence)

    candidates = await storage.list_wisdom(status=WisdomStatus.CANDIDATE)
    assert len(candidates) == 1 and candidates[0].item_id == "W-000001"

    await storage.set_wisdom_status("W-000001", WisdomStatus.APPROVED)
    approved = await storage.list_wisdom(status=WisdomStatus.APPROVED)
    assert len(approved) == 1


async def test_wisdom_evidence_id_preserves_insertion_order(
    storage: Storage,
) -> None:
    """``get_wisdom_evidence`` must return rows in insertion order across
    all backends. SQLite gained an explicit AUTOINCREMENT id (mirroring
    PG's BIGSERIAL) so both adapters' ``ORDER BY id ASC`` is stable.
    """
    doc = _make_doc("wiki/concept.md", layer=Layer.WIKI)
    await storage.upsert_document(doc)
    ts = time.time()
    item = WisdomItem(
        item_id="W-EV-ORDER",
        kind=WisdomKind.PRINCIPLE,
        title="Test ordering",
        body="...",
        confidence=0.5,
        created_ts=ts,
    )
    evidence = [
        WisdomEvidence(doc_id=doc.doc_id, excerpt=f"piece-{i}", line=i)
        for i in range(5)
    ]
    await storage.put_wisdom(item, evidence)

    rows = await storage.get_wisdom_evidence("W-EV-ORDER")
    assert [r.excerpt for r in rows] == [f"piece-{i}" for i in range(5)]
    assert all(r.id is not None for r in rows)
    ids = [r.id for r in rows]
    assert ids == sorted(ids)


# ---- Multimedia assets + chunk_asset_refs (Phase F) ----------------------


def _make_asset(
    asset_id: str,
    *,
    original_path: str = "img.png",
    width: int | None = 100,
    height: int | None = 80,
) -> AssetRecord:
    return AssetRecord(
        asset_id=asset_id,
        kind=AssetKind.IMAGE,
        mime="image/png",
        stored_path=f"assets/{asset_id[:2]}/{asset_id[:8]}-img.png",
        original_paths=[original_path],
        bytes=42,
        media_meta=ImageMediaMeta(width=width, height=height),
        created_ts=time.time(),
    )


async def test_asset_upsert_and_get_roundtrip(storage: Storage) -> None:
    a = _make_asset("ab3f12ef" + "0" * 56)
    try:
        await storage.upsert_asset(a)
    except NotSupported:
        pytest.skip("backend doesn't implement assets yet")

    fetched = await storage.get_asset(a.asset_id)
    assert fetched is not None
    assert fetched.asset_id == a.asset_id
    assert fetched.mime == "image/png"
    assert fetched.original_paths == ["img.png"]
    assert isinstance(fetched.media_meta, ImageMediaMeta)
    assert fetched.media_meta.width == 100
    assert fetched.media_meta.height == 80


async def test_get_assets_batch_drops_missing_ids(storage: Storage) -> None:
    """``get_assets`` is the batch equivalent of ``get_asset``. Single
    SQL round-trip — the search path needs every retrieved chunk's
    asset bundle, and N+1 ``get_asset`` over a shared sqlite connection
    has tripped sqlite3.InterfaceError on real workloads.
    """
    a = _make_asset("aaaa1234" + "0" * 56, original_path="a.png")
    b = _make_asset("bbbb5678" + "0" * 56, original_path="b.png")
    try:
        await storage.upsert_asset(a)
    except NotSupported:
        pytest.skip("backend doesn't implement assets yet")
    await storage.upsert_asset(b)

    fetched = await storage.get_assets(
        [a.asset_id, b.asset_id, "ffff0000" + "0" * 56]
    )
    by_id = {x.asset_id: x for x in fetched}
    assert set(by_id.keys()) == {a.asset_id, b.asset_id}
    assert by_id[a.asset_id].original_paths == ["a.png"]
    assert by_id[b.asset_id].original_paths == ["b.png"]
    # Empty input is a no-op — equivalent to ``get_chunks([])``.
    assert await storage.get_assets([]) == []


async def test_asset_media_meta_none_round_trips(storage: Storage) -> None:
    """SVG / WebP / probe-miss path stores media_meta=None — the round-trip
    must preserve None rather than coerce to an empty ImageMediaMeta()."""
    a = _make_asset("dead0001" + "0" * 56).model_copy(update={"media_meta": None})
    try:
        await storage.upsert_asset(a)
    except NotSupported:
        pytest.skip("backend doesn't implement assets yet")

    fetched = await storage.get_asset(a.asset_id)
    assert fetched is not None
    assert fetched.media_meta is None


async def test_asset_media_meta_partial_dimensions(storage: Storage) -> None:
    """Some image probes recover only one dimension — the JSON round-trip
    must keep the missing side as None (not 0, not absent-as-error)."""
    a = _make_asset("dead0002" + "0" * 56, width=120, height=None)
    try:
        await storage.upsert_asset(a)
    except NotSupported:
        pytest.skip("backend doesn't implement assets yet")

    fetched = await storage.get_asset(a.asset_id)
    assert fetched is not None
    assert isinstance(fetched.media_meta, ImageMediaMeta)
    assert fetched.media_meta.width == 120
    assert fetched.media_meta.height is None


async def test_asset_upsert_replaces_with_new_metadata(storage: Storage) -> None:
    a = _make_asset("cafe1234" + "0" * 56, original_path="a.png")
    try:
        await storage.upsert_asset(a)
    except NotSupported:
        pytest.skip("backend doesn't implement assets yet")

    a2 = a.model_copy(update={"original_paths": ["a.png", "b.png"]})
    await storage.upsert_asset(a2)
    fetched = await storage.get_asset(a.asset_id)
    assert fetched is not None
    assert fetched.original_paths == ["a.png", "b.png"]


async def test_chunk_asset_refs_roundtrip(storage: Storage) -> None:
    doc = _make_doc("sources/with-image.md")
    await storage.upsert_document(doc)
    ids = await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(
                doc_id=doc.doc_id, seq=0, start=0, end=20, text="Has ![](x.png) inline"
            ),
        ],
    )
    cid = ids[0]
    a1 = _make_asset("aa" + "0" * 62, original_path="x.png")
    a2 = _make_asset("bb" + "0" * 62, original_path="y.png")
    try:
        await storage.upsert_asset(a1)
        await storage.upsert_asset(a2)
        await storage.replace_chunk_asset_refs(
            cid,
            [
                ChunkAssetRef(
                    chunk_id=cid,
                    asset_id=a1.asset_id,
                    ord=0,
                    alt="x",
                    start_in_chunk=4,
                    end_in_chunk=14,
                ),
                ChunkAssetRef(
                    chunk_id=cid,
                    asset_id=a2.asset_id,
                    ord=1,
                    alt="y",
                    start_in_chunk=14,
                    end_in_chunk=20,
                ),
            ],
        )
    except NotSupported:
        pytest.skip("backend doesn't implement chunk_asset_refs yet")

    refs = await storage.chunk_asset_refs_for_chunks([cid])
    assert len(refs[cid]) == 2
    assert [r.asset_id for r in refs[cid]] == [a1.asset_id, a2.asset_id]
    assert [r.ord for r in refs[cid]] == [0, 1]

    # Reverse lookup
    by_asset = await storage.chunks_referencing_assets([a1.asset_id, a2.asset_id])
    assert by_asset[a1.asset_id] == [cid]
    assert by_asset[a2.asset_id] == [cid]


async def test_chunk_asset_refs_replaced_on_reupsert(storage: Storage) -> None:
    """Calling replace_chunk_asset_refs again must wipe the previous set."""
    doc = _make_doc("sources/replace.md")
    await storage.upsert_document(doc)
    ids = await storage.replace_chunks(
        doc.doc_id,
        [ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=10, text="hello")],
    )
    cid = ids[0]
    a = _make_asset("dd" + "0" * 62)
    try:
        await storage.upsert_asset(a)
        await storage.replace_chunk_asset_refs(
            cid,
            [
                ChunkAssetRef(
                    chunk_id=cid,
                    asset_id=a.asset_id,
                    ord=0,
                    alt="",
                    start_in_chunk=0,
                    end_in_chunk=5,
                )
            ],
        )
        # Replace with empty list — refs should disappear.
        await storage.replace_chunk_asset_refs(cid, [])
    except NotSupported:
        pytest.skip("backend doesn't implement chunk_asset_refs yet")

    assert await storage.chunk_asset_refs_for_chunks([cid]) == {cid: []}


async def test_chunk_asset_refs_constraint_zero_length_span(
    storage: Storage,
) -> None:
    """The schema must reject ``start_in_chunk == end_in_chunk`` (degenerate
    span). Today no chunker path produces these, but the CHECK is the
    boundary safety net so a future refactor can't slip past silently."""
    doc = _make_doc("sources/zero-span.md")
    await storage.upsert_document(doc)
    ids = await storage.replace_chunks(
        doc.doc_id,
        [ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=10, text="hi")],
    )
    cid = ids[0]
    a = _make_asset("ff" + "0" * 62)
    try:
        await storage.upsert_asset(a)
    except NotSupported:
        pytest.skip("backend doesn't implement chunk_asset_refs yet")

    with pytest.raises(_integrity_error_types()):
        await storage.replace_chunk_asset_refs(
            cid,
            [
                ChunkAssetRef(
                    chunk_id=cid,
                    asset_id=a.asset_id,
                    ord=0,
                    alt="",
                    start_in_chunk=5,
                    end_in_chunk=5,
                )
            ],
        )


async def test_chunk_asset_refs_constraint_duplicate_span(
    storage: Storage,
) -> None:
    """Two refs at the same ``[start, end)`` byte range within a single
    chunk must be rejected. The chunker can't legitimately produce
    duplicates; a violation indicates a regression worth catching loudly."""
    doc = _make_doc("sources/dup-span.md")
    await storage.upsert_document(doc)
    ids = await storage.replace_chunks(
        doc.doc_id,
        [ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=20, text="text")],
    )
    cid = ids[0]
    a = _make_asset("11" + "0" * 62, original_path="x.png")
    b = _make_asset("22" + "0" * 62, original_path="y.png")
    try:
        await storage.upsert_asset(a)
        await storage.upsert_asset(b)
    except NotSupported:
        pytest.skip("backend doesn't implement chunk_asset_refs yet")

    with pytest.raises(_integrity_error_types()):
        await storage.replace_chunk_asset_refs(
            cid,
            [
                ChunkAssetRef(
                    chunk_id=cid,
                    asset_id=a.asset_id,
                    ord=0,
                    alt="",
                    start_in_chunk=4,
                    end_in_chunk=14,
                ),
                ChunkAssetRef(
                    chunk_id=cid,
                    asset_id=b.asset_id,
                    ord=1,
                    alt="",
                    start_in_chunk=4,
                    end_in_chunk=14,
                ),
            ],
        )


async def test_chunk_asset_refs_cascade_on_chunk_delete(storage: Storage) -> None:
    """When the parent chunk is replaced (DELETE-then-INSERT pattern), the
    chunk_asset_refs rows must cascade so they don't dangle."""
    doc = _make_doc("sources/cascade.md")
    await storage.upsert_document(doc)
    ids = await storage.replace_chunks(
        doc.doc_id,
        [ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=10, text="hello")],
    )
    cid = ids[0]
    a = _make_asset("ee" + "0" * 62)
    try:
        await storage.upsert_asset(a)
        await storage.replace_chunk_asset_refs(
            cid,
            [
                ChunkAssetRef(
                    chunk_id=cid,
                    asset_id=a.asset_id,
                    ord=0,
                    alt="",
                    start_in_chunk=0,
                    end_in_chunk=5,
                )
            ],
        )
    except NotSupported:
        pytest.skip("backend doesn't implement chunk_asset_refs yet")

    # Replace chunks for this doc — old chunk row is gone, refs should follow.
    new_ids = await storage.replace_chunks(
        doc.doc_id,
        [ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=10, text="rewritten")],
    )
    new_cid = new_ids[0]
    refs = await storage.chunk_asset_refs_for_chunks([cid, new_cid])
    assert refs == {cid: [], new_cid: []}
    # Asset row still exists (only chunk_asset_refs cascade, not assets).
    assert await storage.get_asset(a.asset_id) is not None


# ---- Embedding versioning + asset embeddings (Phase F) -------------------


async def test_embed_version_upsert_idempotent(storage: Storage) -> None:
    v = EmbeddingVersion(
        provider="gitee_multimodal",
        model="jina-clip-v2",
        revision="",
        dim=768,
        normalize=True,
        distance="cosine",
        modality="multimodal",
    )
    try:
        vid_a = await storage.upsert_embed_version(v)
        vid_b = await storage.upsert_embed_version(v)
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")
    assert vid_a == vid_b
    versions = await storage.list_embed_versions()
    assert sum(1 for x in versions if x.version_id == vid_a) == 1


async def test_embed_version_new_demotes_prior_active(storage: Storage) -> None:
    """A new version with the same modality must mark prior versions of
    that modality as is_active=0; other modalities are untouched."""
    text_v1 = EmbeddingVersion(
        provider="legacy",
        model="text-embedding-3-small",
        dim=1536,
        normalize=True,
        distance="cosine",
        modality="text",
    )
    mm_v1 = EmbeddingVersion(
        provider="gitee_multimodal",
        model="jina-clip-v2",
        dim=768,
        normalize=True,
        distance="cosine",
        modality="multimodal",
    )
    mm_v2 = mm_v1.model_copy(update={"dim": 1024})
    try:
        await storage.upsert_embed_version(text_v1)
        await storage.upsert_embed_version(mm_v1)
        await storage.upsert_embed_version(mm_v2)
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")

    active_text = await storage.get_active_embed_version(modality="text")
    active_mm = await storage.get_active_embed_version(modality="multimodal")
    assert active_text is not None
    assert active_text.dim == 1536  # text untouched by mm bump
    assert active_mm is not None
    assert active_mm.dim == 1024  # latest multimodal wins


async def test_asset_embeddings_upsert_and_search(storage: Storage) -> None:
    """End-to-end: register version → upsert asset embedding → vec_search
    finds it."""
    v = EmbeddingVersion(
        provider="fake_mm",
        model="fake-mm-v1",
        dim=4,
        normalize=True,
        distance="cosine",
        modality="multimodal",
    )
    a = _make_asset("ff" + "0" * 62)
    try:
        version_id = await storage.upsert_embed_version(v)
        await storage.upsert_asset(a)
        await storage.upsert_asset_embeddings(
            [
                AssetEmbeddingRow(
                    asset_id=a.asset_id,
                    version_id=version_id,
                    embedding=[1.0, 0.0, 0.0, 0.0],
                )
            ]
        )
        hits = await storage.vec_search_assets(
            [1.0, 0.0, 0.0, 0.0], version_id=version_id, limit=5
        )
    except NotSupported:
        pytest.skip("backend doesn't implement asset embeddings yet")

    assert len(hits) == 1
    assert hits[0].asset_id == a.asset_id
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)


async def test_asset_embeddings_dim_mismatch_raises(storage: Storage) -> None:
    v = EmbeddingVersion(
        provider="fake_mm",
        model="fake-mm-v1-dim4",
        dim=4,
        normalize=True,
        distance="cosine",
        modality="multimodal",
    )
    a = _make_asset("12" + "0" * 62)
    try:
        version_id = await storage.upsert_embed_version(v)
        await storage.upsert_asset(a)
    except NotSupported:
        pytest.skip("backend doesn't implement asset embeddings yet")

    from dikw_core.storage.base import StorageError

    with pytest.raises(StorageError):
        await storage.upsert_asset_embeddings(
            [
                AssetEmbeddingRow(
                    asset_id=a.asset_id,
                    version_id=version_id,
                    embedding=[1.0, 0.0],  # wrong dim
                )
            ]
        )


async def test_chunks_referencing_assets_chunks_large_input(
    storage: Storage,
) -> None:
    """``chunks_referencing_assets`` must not trip
    ``OperationalError: too many SQL variables`` on a backfill-sized
    asset list. Default SQLite ``SQLITE_MAX_VARIABLE_NUMBER`` is 999
    on legacy builds; modern is 32766. The implementation chunks the
    IN-list internally so callers can hand it the full backlog.
    """
    doc = _make_doc("sources/many.md")
    await storage.upsert_document(doc)
    chunk_ids = await storage.replace_chunks(
        doc.doc_id,
        [ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=4, text="body")],
    )

    # Three asset IDs ACTUALLY referenced by chunk 0; the remaining
    # 1500 are random hashes the function should return empty lists for.
    real_ids = [f"a{i:04d}" + "0" * 60 for i in range(3)]
    decoy_ids = [f"d{i:04d}" + "0" * 60 for i in range(1500)]
    all_ids = real_ids + decoy_ids

    try:
        for aid in real_ids:
            await storage.upsert_asset(_make_asset(aid))
        await storage.replace_chunk_asset_refs(
            chunk_ids[0],
            [
                ChunkAssetRef(
                    chunk_id=chunk_ids[0],
                    asset_id=aid,
                    ord=i,
                    alt="",
                    start_in_chunk=i,
                    end_in_chunk=i + 1,
                )
                for i, aid in enumerate(real_ids)
            ],
        )
    except NotSupported:
        pytest.skip("backend doesn't implement chunk_asset_refs yet")

    # The ask: 1503 IDs in one call. Must not raise.
    result = await storage.chunks_referencing_assets(all_ids)
    assert len(result) == len(all_ids)
    for aid in real_ids:
        assert result[aid] == [chunk_ids[0]]
    for aid in decoy_ids:
        assert result[aid] == []


async def test_list_assets_missing_embedding(storage: Storage) -> None:
    """Backfill scan: returns assets without an ``asset_embed_meta`` row
    for ``version_id``.

    Assets stored under a text-only ingest (``materialize_asset``
    decoupled from ``mm_cfg``) get rows in ``assets`` but no embedding.
    The next mm-enabled ingest needs to find them; without this scan
    they'd stay un-embedded forever (was_new=False suppresses the
    new-asset queue entry).
    """
    v = EmbeddingVersion(
        provider="fake_mm",
        model="fake-mm-v1-missing",
        dim=4,
        normalize=True,
        distance="cosine",
        modality="multimodal",
    )
    v2 = EmbeddingVersion(
        provider="fake_mm",
        model="fake-mm-v2-missing",
        dim=4,
        normalize=True,
        distance="cosine",
        modality="multimodal",
    )
    a1 = _make_asset("a1" + "0" * 62, original_path="a1.png")
    a2 = _make_asset("a2" + "0" * 62, original_path="a2.png")
    a3 = _make_asset("a3" + "0" * 62, original_path="a3.png")
    try:
        v1_id = await storage.upsert_embed_version(v)
        v2_id = await storage.upsert_embed_version(v2)
        for asset in (a1, a2, a3):
            await storage.upsert_asset(asset)
        # Embed only a1 + a2 under v1; a3 unembedded under v1.
        await storage.upsert_asset_embeddings(
            [
                AssetEmbeddingRow(
                    asset_id=a1.asset_id,
                    version_id=v1_id,
                    embedding=[1.0, 0.0, 0.0, 0.0],
                ),
                AssetEmbeddingRow(
                    asset_id=a2.asset_id,
                    version_id=v1_id,
                    embedding=[0.0, 1.0, 0.0, 0.0],
                ),
            ]
        )
    except NotSupported:
        pytest.skip("backend doesn't implement asset embeddings yet")

    missing_v1 = await storage.list_assets_missing_embedding(version_id=v1_id)
    assert {a.asset_id for a in missing_v1} == {a3.asset_id}
    # Returned AssetRecords carry the original metadata so the caller can
    # hand them to ``embed_assets`` directly without a refetch.
    assert missing_v1[0].original_paths == ["a3.png"]

    # Different version_id (v2) — none have embeddings yet → all 3 missing.
    missing_v2 = await storage.list_assets_missing_embedding(version_id=v2_id)
    assert {a.asset_id for a in missing_v2} == {
        a1.asset_id,
        a2.asset_id,
        a3.asset_id,
    }


# ---- Wisdom embeddings (P0) ---------------------------------------------


async def _seed_wisdom_item(
    storage: Storage, item_id: str, *, body: str = "...", title: str = "T"
) -> str:
    """Create a wiki-layer doc + a candidate wisdom item with two evidence
    entries and return the item_id. Used by every wisdom-embedding test
    so they don't repeat the boilerplate.
    """
    doc = _make_doc(f"wiki/{item_id.lower()}.md", layer=Layer.WIKI)
    await storage.upsert_document(doc)
    item = WisdomItem(
        item_id=item_id,
        kind=WisdomKind.PRINCIPLE,
        title=title,
        body=body,
        confidence=0.7,
        created_ts=time.time(),
    )
    evidence = [
        WisdomEvidence(doc_id=doc.doc_id, excerpt="e1", line=1),
        WisdomEvidence(doc_id=doc.doc_id, excerpt="e2", line=2),
    ]
    await storage.put_wisdom(item, evidence)
    return item_id


async def test_wisdom_embeddings_upsert_and_search(storage: Storage) -> None:
    """End-to-end mirror of ``test_asset_embeddings_upsert_and_search``:
    register text version → put_wisdom → upsert wisdom embedding →
    ``vec_search_wisdom`` finds it. Modality is reused: wisdom rides on
    the active text ``embed_versions`` row (chunks and wisdom share one
    cosine space)."""
    try:
        version_id = await register_text_version(
            storage, dim=4, model="fake-text-v1-wisdom"
        )
        item_id = await _seed_wisdom_item(storage, "W-EMB-001")
        await storage.upsert_wisdom_embeddings(
            [
                WisdomEmbeddingRow(
                    item_id=item_id,
                    version_id=version_id,
                    embedding=[1.0, 0.0, 0.0, 0.0],
                )
            ]
        )
        hits = await storage.vec_search_wisdom(
            [1.0, 0.0, 0.0, 0.0], version_id=version_id, limit=5
        )
    except NotSupported:
        pytest.skip("backend doesn't implement wisdom embeddings")

    assert len(hits) == 1
    assert hits[0].item_id == item_id
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)


async def test_wisdom_embeddings_dim_mismatch_raises(storage: Storage) -> None:
    """Embedding dim must match the dim recorded on ``embed_versions`` —
    same invariant as chunk and asset embeddings."""
    try:
        version_id = await register_text_version(
            storage, dim=4, model="fake-text-v1-wisdom-dim"
        )
        item_id = await _seed_wisdom_item(storage, "W-DIM-001")
    except NotSupported:
        pytest.skip("backend doesn't implement wisdom embeddings")

    from dikw_core.storage.base import StorageError

    with pytest.raises(StorageError):
        await storage.upsert_wisdom_embeddings(
            [
                WisdomEmbeddingRow(
                    item_id=item_id,
                    version_id=version_id,
                    embedding=[1.0, 0.0],  # wrong dim
                )
            ]
        )


async def test_wisdom_embeddings_idempotent_on_reupsert(
    storage: Storage,
) -> None:
    """Writing the same ``(item_id, version_id)`` twice must replace
    in-place (no duplicate vec rows). Mirror of asset/chunk semantics."""
    try:
        version_id = await register_text_version(
            storage, dim=4, model="fake-text-v1-wisdom-idem"
        )
        item_id = await _seed_wisdom_item(storage, "W-IDEM-001")
        await storage.upsert_wisdom_embeddings(
            [
                WisdomEmbeddingRow(
                    item_id=item_id,
                    version_id=version_id,
                    embedding=[1.0, 0.0, 0.0, 0.0],
                )
            ]
        )
        # Re-upsert with a different vector — should overwrite, not duplicate.
        await storage.upsert_wisdom_embeddings(
            [
                WisdomEmbeddingRow(
                    item_id=item_id,
                    version_id=version_id,
                    embedding=[0.0, 1.0, 0.0, 0.0],
                )
            ]
        )
        hits_a = await storage.vec_search_wisdom(
            [1.0, 0.0, 0.0, 0.0], version_id=version_id, limit=5
        )
        hits_b = await storage.vec_search_wisdom(
            [0.0, 1.0, 0.0, 0.0], version_id=version_id, limit=5
        )
    except NotSupported:
        pytest.skip("backend doesn't implement wisdom embeddings")

    # Exactly one wisdom row across the two probes; the second probe
    # (matching the most recently written vector) is the close hit.
    assert len(hits_b) == 1
    assert hits_b[0].item_id == item_id
    assert hits_b[0].distance == pytest.approx(0.0, abs=1e-6)
    # The first probe should now be far (orthogonal), not zero — proves
    # the original vector was overwritten rather than duplicated.
    assert len(hits_a) == 1
    assert hits_a[0].distance > 0.5


async def test_list_wisdom_missing_embedding(storage: Storage) -> None:
    """Symmetric to ``test_list_assets_missing_embedding`` — wisdom items
    without a ``wisdom_embed_meta`` row for ``version_id``.

    No engine call site uses this primitive yet (wisdom embedding
    pipeline is not wired up); the test just locks the storage
    contract so future work has the recovery path ready.
    """
    try:
        v1 = await register_text_version(
            storage, dim=4, model="fake-wisdom-missing-v1"
        )
        v2 = await register_text_version(
            storage, dim=4, model="fake-wisdom-missing-v2"
        )
        i1 = await _seed_wisdom_item(storage, "W-MISS-001")
        i2 = await _seed_wisdom_item(storage, "W-MISS-002")
        i3 = await _seed_wisdom_item(storage, "W-MISS-003")
        await storage.upsert_wisdom_embeddings(
            [
                WisdomEmbeddingRow(
                    item_id=i1, version_id=v1, embedding=[1.0, 0.0, 0.0, 0.0]
                ),
                WisdomEmbeddingRow(
                    item_id=i2, version_id=v1, embedding=[0.0, 1.0, 0.0, 0.0]
                ),
            ]
        )
    except NotSupported:
        pytest.skip("backend doesn't implement wisdom embeddings")

    missing_v1 = await storage.list_wisdom_missing_embedding(version_id=v1)
    assert {it.item_id for it in missing_v1} == {i3}
    # Different version → none embedded under it yet.
    missing_v2 = await storage.list_wisdom_missing_embedding(version_id=v2)
    assert {it.item_id for it in missing_v2} == {i1, i2, i3}


async def test_vec_search_wisdom_returns_in_distance_order(
    storage: Storage,
) -> None:
    """KNN ranking parity with ``test_vec_search_returns_in_distance_order``:
    multiple wisdom embeddings → ``vec_search_wisdom`` returns them sorted
    by ascending cosine distance against the query vector.

    Vectors are constructed so the angle from the query (1,0,0,0) grows
    monotonically across items; the returned ranking must mirror that.
    """
    # 4 wisdom items at angles 0°, 30°, 60°, 90° from the query.
    import math as _math

    angles_deg = [0.0, 30.0, 60.0, 90.0]
    embeddings: list[list[float]] = [
        [_math.cos(_math.radians(a)), _math.sin(_math.radians(a)), 0.0, 0.0]
        for a in angles_deg
    ]
    item_ids = [f"W-ORD-{i}" for i in range(len(angles_deg))]
    try:
        version_id = await register_text_version(
            storage, dim=4, model="fake-text-v1-wisdom-order"
        )
        for iid in item_ids:
            await _seed_wisdom_item(storage, iid)
        await storage.upsert_wisdom_embeddings(
            [
                WisdomEmbeddingRow(item_id=iid, version_id=version_id, embedding=emb)
                for iid, emb in zip(item_ids, embeddings, strict=True)
            ]
        )
        hits = await storage.vec_search_wisdom(
            [1.0, 0.0, 0.0, 0.0], version_id=version_id, limit=10
        )
    except NotSupported:
        pytest.skip("backend doesn't implement wisdom embeddings")

    # All items returned; ranking matches insertion (=angle) order; distances
    # strictly increasing.
    from itertools import pairwise

    assert [h.item_id for h in hits] == item_ids
    distances = [h.distance for h in hits]
    assert distances == sorted(distances)
    assert all(d_b > d_a for d_a, d_b in pairwise(distances))


async def test_upsert_wisdom_embeddings_rejects_non_text_modality(
    storage: Storage,
) -> None:
    """Wisdom rides on the text modality; upserting against a multimodal
    version must raise ``StorageError`` instead of silently mixing
    cosine spaces. Without this guard, dims-happen-to-match would let
    multimodal vectors leak into ``vec_wisdom_v<id>`` and
    ``vec_search_wisdom`` would return meaningless rankings.
    """
    mm_v = EmbeddingVersion(
        provider="fake_mm",
        model="fake-mm-v1-wisdom-modality",
        dim=4,
        normalize=True,
        distance="cosine",
        modality="multimodal",
    )
    try:
        version_id = await storage.upsert_embed_version(mm_v)
        item_id = await _seed_wisdom_item(storage, "W-MM-001")
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning")

    from dikw_core.storage.base import StorageError

    with pytest.raises(StorageError, match="modality"):
        await storage.upsert_wisdom_embeddings(
            [
                WisdomEmbeddingRow(
                    item_id=item_id,
                    version_id=version_id,
                    embedding=[1.0, 0.0, 0.0, 0.0],
                )
            ]
        )


async def test_vec_search_wisdom_rejects_non_text_modality(
    storage: Storage,
) -> None:
    """Same invariant as upsert: searching wisdom against a multimodal
    version must fail loudly, not return silently wrong results."""
    mm_v = EmbeddingVersion(
        provider="fake_mm",
        model="fake-mm-v1-wisdom-modality-search",
        dim=4,
        normalize=True,
        distance="cosine",
        modality="multimodal",
    )
    try:
        version_id = await storage.upsert_embed_version(mm_v)
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning")

    from dikw_core.storage.base import StorageError

    with pytest.raises(StorageError, match="modality"):
        await storage.vec_search_wisdom(
            [1.0, 0.0, 0.0, 0.0], version_id=version_id, limit=5
        )


async def test_vec_search_wisdom_empty_index_returns_empty(
    storage: Storage,
) -> None:
    """Querying before any wisdom embedding has been written must return
    an empty list, not raise — same shape as ``vec_search`` /
    ``vec_search_assets`` over an empty index."""
    try:
        version_id = await register_text_version(
            storage, dim=4, model="fake-text-v1-wisdom-empty"
        )
        hits = await storage.vec_search_wisdom(
            [1.0, 0.0, 0.0, 0.0], version_id=version_id, limit=5
        )
    except NotSupported:
        pytest.skip("backend doesn't implement wisdom embeddings")

    assert hits == []


# ---- T6 + text-versioning regression tests ------------------------------


async def test_embed_version_modality_in_unique_key(storage: Storage) -> None:
    """T6: a CLIP-style provider that exposes the same model name as both
    a text and a multimodal encoder must produce TWO distinct ``version_id``s.

    Pre-fix the UNIQUE was ``(provider, model, revision, dim, normalize, distance)``
    — modality was missing — so the second registration silently collapsed
    onto the first row, overwriting modality semantically.
    """
    base = {
        "provider": "clip-style",
        "model": "bge-m3",
        "revision": "",
        "dim": 1024,
        "normalize": True,
        "distance": "cosine",
    }
    text = EmbeddingVersion(**base, modality="text")
    mm = EmbeddingVersion(**base, modality="multimodal")
    try:
        vid_text = await storage.upsert_embed_version(text)
        vid_mm = await storage.upsert_embed_version(mm)
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")
    assert vid_text != vid_mm, (
        f"T6 regression: same model under different modalities collapsed "
        f"to the same version_id={vid_text}"
    )


async def test_text_versioning_isolation(storage: Storage) -> None:
    """Two text versions with different dims coexist — switching a text
    embedding model creates a new ``vec_chunks_v<id>`` next to the old
    one rather than crashing the index.
    """
    doc = _make_doc("sources/iso.md")
    await storage.upsert_document(doc)
    cids = await storage.replace_chunks(
        doc.doc_id,
        [ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=5, text="alpha")],
    )
    try:
        v1 = await register_text_version(storage, dim=4, model="m1")
        v2 = await register_text_version(storage, dim=8, model="m2")
        await storage.upsert_embeddings(
            [EmbeddingRow(chunk_id=cids[0], version_id=v1, embedding=[1.0] * 4)]
        )
        await storage.upsert_embeddings(
            [EmbeddingRow(chunk_id=cids[0], version_id=v2, embedding=[1.0] * 8)]
        )
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")
    # Each version's vec_search uses its own table.
    hits_v1 = await storage.vec_search([1.0] * 4, version_id=v1, limit=5)
    hits_v2 = await storage.vec_search([1.0] * 8, version_id=v2, limit=5)
    assert hits_v1 and hits_v1[0].chunk_id == cids[0]
    assert hits_v2 and hits_v2[0].chunk_id == cids[0]


async def test_vec_search_resolves_active_text_version_when_omitted(
    storage: Storage,
) -> None:
    """``vec_search(version_id=None)`` resolves to the active text version.

    Bumping to a new active version must redirect subsequent unqualified
    queries to the new table — no caller threading required.
    """
    doc = _make_doc("sources/active.md")
    await storage.upsert_document(doc)
    cids = await storage.replace_chunks(
        doc.doc_id,
        [ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=5, text="alpha")],
    )
    try:
        v1 = await register_text_version(storage, dim=4, model="m1")
        await storage.upsert_embeddings(
            [EmbeddingRow(chunk_id=cids[0], version_id=v1, embedding=[1.0] * 4)]
        )
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")
    hits_a = await storage.vec_search([1.0] * 4, limit=5)
    assert hits_a and hits_a[0].chunk_id == cids[0]
    # Bump to a new active version with a different dim. The unqualified
    # vec_search must now route to v2 — i.e. fail dim check on the v1
    # query vector.
    v2 = await register_text_version(storage, dim=8, model="m2")
    await storage.upsert_embeddings(
        [EmbeddingRow(chunk_id=cids[0], version_id=v2, embedding=[1.0] * 8)]
    )
    from dikw_core.storage.base import StorageError as _SE

    with pytest.raises(_SE):
        await storage.vec_search([1.0] * 4, limit=5)
    # Querying with the new dim against active version_id resolves cleanly.
    hits_b = await storage.vec_search([1.0] * 8, limit=5)
    assert hits_b and hits_b[0].chunk_id == cids[0]
    # Sanity: get_active_embed_version reflects the latest registration.
    active = await storage.get_active_embed_version(modality="text")
    assert active is not None and active.version_id == v2
