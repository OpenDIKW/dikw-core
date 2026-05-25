-- dikw-core SQLite schema (alpha, single source of truth).
--
-- Applied verbatim to a fresh DB by ``SQLiteStorage.migrate()``; bumping
-- ``storage/_schema.py:SCHEMA_VERSION`` invalidates any DB carrying an
-- older fingerprint. There is no in-place upgrade path — alpha
-- policy is rebuild on incompatibility.
--
-- ``sqlite-vec`` must be loaded into the connection before this file
-- runs. Per-version ``vec_chunks_v<version_id>`` and
-- ``vec_assets_v<version_id>`` tables are NOT declared here: sqlite-vec
-- needs the embedding dim parameterized into ``CREATE VIRTUAL TABLE``,
-- so each new ``embed_versions`` row triggers ``_ensure_vec_table()``
-- in ``storage/sqlite.py`` to materialize its dim-locked vec table at
-- runtime. Switching a model = new ``embed_versions`` row = new vec
-- table; prior vectors survive in-place under their own version_id.
--
-- W layer (wisdom) is being refactored in 0.3.0 to first-class
-- documents under ``wisdom/<author>/<slug>.md``. PR1 removes the
-- legacy ``wisdom_items`` / ``wisdom_evidence`` / ``wisdom_embed_meta``
-- tables; PR2 wires wisdom files into ``documents`` via
-- ``layer = 'wisdom'``.
--
-- ``meta_kv`` is redeclared here (it is also created inline ahead of
-- this file by ``migrate()`` so the schema-version reader has a row to
-- query against) so a schema diff sees the full shape in one place.
CREATE TABLE IF NOT EXISTS meta_kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ---- D layer -------------------------------------------------------------

-- ``path`` carries the user's spelling (display path); ``path_key`` is
-- the engine's NFC + casefold lookup key. Splitting the two lets the
-- same logical file under different macOS NFD / NTFS-case spellings
-- resolve to a single row while ``dikw client status`` still shows whichever
-- spelling is on disk. Uniqueness lives on ``path_key``; ``path`` stays
-- NOT NULL but plain so a rename-with-case-change updates the display
-- value in place. See data/path_norm.py.
CREATE TABLE IF NOT EXISTS documents (
    doc_id   TEXT PRIMARY KEY,
    path     TEXT NOT NULL,
    path_key TEXT NOT NULL UNIQUE,
    title    TEXT,
    hash     TEXT NOT NULL,
    mtime    REAL,
    layer    TEXT NOT NULL CHECK (layer IN ('source','wiki','wisdom')),
    active   INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS documents_layer_active
    ON documents(layer, active);

-- Reverse lookup by content hash (idempotent ingest short-circuit).
CREATE INDEX IF NOT EXISTS documents_hash_idx ON documents(hash);

-- ---- I layer -------------------------------------------------------------

-- FTS5 over chunk body; ``rowid`` aligns with ``chunks.chunk_id``.
-- Scope is intentionally narrow:
--   * Only ``body`` is indexed — ``path`` / ``title`` / ``layer`` come
--     back via a JOIN onto ``chunks`` + ``documents`` at search time,
--     mirroring the PG side which indexes only ``chunks.text``.
--   * ``unicode61 remove_diacritics 0`` so that ``café`` and ``cafe``
--     are different tokens — same byte-level behavior as PG's
--     ``to_tsvector('simple', ...)``. The ``0`` is intentional and
--     explicit because the unicode61 default is ``1``, which still
--     strips most diacritics.
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    body,
    tokenize = "unicode61 remove_diacritics 0"
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id    TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    seq       INTEGER NOT NULL,
    start_off INTEGER NOT NULL,
    end_off   INTEGER NOT NULL,
    text      TEXT NOT NULL,
    UNIQUE (doc_id, seq)
);

CREATE INDEX IF NOT EXISTS chunks_doc_seq ON chunks(doc_id, seq);

-- ---- K layer -------------------------------------------------------------

CREATE TABLE IF NOT EXISTS links (
    src_doc_id TEXT NOT NULL,
    dst_path   TEXT NOT NULL,
    link_type  TEXT NOT NULL CHECK (link_type IN ('wikilink','markdown','url')),
    anchor     TEXT,
    line       INTEGER NOT NULL,
    PRIMARY KEY (src_doc_id, dst_path, line)
);

CREATE INDEX IF NOT EXISTS links_dst ON links(dst_path);

-- K-page → D-source attribution ("provenance"), reconciled from the
-- K-page's ``sources:`` frontmatter on every persist. Deliberately a
-- separate table from ``links`` (see docs/adr/0001-provenance-as-separate-edge.md):
-- frontmatter entries carry no body line/anchor, must not pollute the
-- wikilink graph that feeds graph-leg retrieval and orphan/broken-link
-- lint, and reuse-with-link_type='derived_from' would force every
-- downstream consumer to filter forever. PK is ``(src_doc_id,
-- source_path_key)`` — not ``source_path`` — so case/NFC drift on
-- Windows/macOS collapses to one row. No FK on either side: mirrors
-- ``links`` so a user can stage frontmatter ahead of ingesting the
-- referenced source, and cleanup happens explicitly in
-- ``delete_document``. The index on ``source_path_key`` powers
-- ``provenance_to`` (reverse "which K-pages claim this source?").
CREATE TABLE IF NOT EXISTS provenance (
    src_doc_id      TEXT NOT NULL,
    source_path     TEXT NOT NULL,
    source_path_key TEXT NOT NULL,
    PRIMARY KEY (src_doc_id, source_path_key)
);

CREATE INDEX IF NOT EXISTS provenance_source_key
    ON provenance(source_path_key);

-- ``id`` is an explicit AUTOINCREMENT column so monotonic ordering is
-- preserved when multiple events share the same ``ts`` (``time.time()``
-- is float-second resolution; an ingest batch can append multiple rows
-- in the same second). Mirrors the Postgres ``BIGSERIAL`` column.
CREATE TABLE IF NOT EXISTS wiki_log (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     REAL NOT NULL,
    action TEXT NOT NULL,
    src    TEXT,
    dst    TEXT,
    note   TEXT
);

CREATE INDEX IF NOT EXISTS wiki_log_ts ON wiki_log(ts);

-- ---- Multimedia assets ---------------------------------------------------

CREATE TABLE IF NOT EXISTS assets (
    asset_id       TEXT PRIMARY KEY,         -- sha256 hex of the bytes; content-addressed identity
    kind           TEXT NOT NULL CHECK (kind IN ('image')),
    mime           TEXT NOT NULL,
    stored_path    TEXT NOT NULL,            -- relative to project_root
    original_paths TEXT NOT NULL,            -- JSON list of source-side names
    bytes          INTEGER NOT NULL,
    media_meta     TEXT,                     -- per-kind JSON; image: {"kind":"image","width":...,"height":...}
    created_ts     REAL NOT NULL
);

-- CHECK guarantees a non-degenerate span; the markdown image regex
-- already enforces this on the application side, so the constraint is
-- the schema-level safety net for future chunker variants. UNIQUE
-- forbids two refs landing in the same byte range within one chunk —
-- duplicates would indicate a chunker bug rather than legitimate intent.
CREATE TABLE IF NOT EXISTS chunk_asset_refs (
    chunk_id        INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    asset_id        TEXT    NOT NULL REFERENCES assets(asset_id),
    ord             INTEGER NOT NULL,        -- 0-based ordinal within chunk
    alt             TEXT    NOT NULL DEFAULT '',
    start_in_chunk  INTEGER NOT NULL,
    end_in_chunk    INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, ord),
    CHECK (start_in_chunk < end_in_chunk),
    UNIQUE (chunk_id, start_in_chunk, end_in_chunk)
);

CREATE INDEX IF NOT EXISTS chunk_asset_refs_asset
    ON chunk_asset_refs(asset_id);

-- ---- Embedding versioning ------------------------------------------------

-- Composite identity: any field different = different version = different
-- vector table. ``revision`` lets users force a new version when a
-- provider silently refreshes weights behind a stable model name.
-- ``modality`` is part of the UNIQUE so a single CLIP-style model can
-- register both a text and a multimodal version without collision.
CREATE TABLE IF NOT EXISTS embed_versions (
    version_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    provider    TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    revision    TEXT    NOT NULL DEFAULT '',
    dim         INTEGER NOT NULL,
    normalize   INTEGER NOT NULL,            -- 0/1
    distance    TEXT    NOT NULL CHECK (distance IN ('cosine','l2','dot')),
    modality    TEXT    NOT NULL CHECK (modality IN ('text','multimodal')),
    created_ts  REAL    NOT NULL,
    is_active   INTEGER NOT NULL DEFAULT 1,
    UNIQUE (provider, model, revision, dim, normalize, distance, modality)
);

CREATE INDEX IF NOT EXISTS embed_versions_active
    ON embed_versions(modality, is_active);

-- Per-chunk embedding metadata. The vector itself lives in the
-- vec_chunks_v<version_id> virtual table created at runtime; see the
-- file header for the rationale.
CREATE TABLE IF NOT EXISTS chunk_embed_meta (
    chunk_id   INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    version_id INTEGER NOT NULL REFERENCES embed_versions(version_id),
    PRIMARY KEY (chunk_id, version_id)
);

CREATE INDEX IF NOT EXISTS chunk_embed_meta_version
    ON chunk_embed_meta(version_id);

-- Per-asset embedding metadata. Mirror of chunk_embed_meta. The vector
-- lives in vec_assets_v<version_id> (also runtime-created).
CREATE TABLE IF NOT EXISTS asset_embed_meta (
    asset_id   TEXT    NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    version_id INTEGER NOT NULL REFERENCES embed_versions(version_id),
    PRIMARY KEY (asset_id, version_id)
);

CREATE INDEX IF NOT EXISTS asset_embed_meta_version
    ON asset_embed_meta(version_id);

-- ---- Embedding cache ----------------------------------------------------

-- Content-addressed embedding cache, decoupled from chunks.chunk_id so
-- re-ingest under replace_chunks's delete-and-reinsert semantics doesn't
-- lose API spend on byte-identical chunk text. Lookup by
-- (sha256(chunk.text), version_id). Keying on version_id (not raw model
-- name) keeps the cache strictly version-isolated: a normalize/distance
-- flip without a model name change is a different version_id and
-- therefore a different cache key, so stale vectors can't leak across
-- version boundaries. ``dim`` duplicates len(embedding) but is stored
-- explicitly so a cleanup query can filter cheaply without unpacking
-- blobs. Vector blob format: float32 little-endian, same packing as
-- storage/sqlite.py::_serialize_vec.
CREATE TABLE IF NOT EXISTS embed_cache (
    content_hash TEXT    NOT NULL,
    version_id   INTEGER NOT NULL REFERENCES embed_versions(version_id),
    dim          INTEGER NOT NULL,
    embedding    BLOB    NOT NULL,
    created_ts   REAL    NOT NULL,
    PRIMARY KEY (content_hash, version_id)
);
