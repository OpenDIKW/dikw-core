-- dikw-core Postgres schema (pre-alpha, single source of truth).
--
-- Applied verbatim to a fresh DB by ``PostgresStorage.migrate()``;
-- bumping ``storage/_schema.py:SCHEMA_VERSION`` invalidates any DB
-- carrying an older fingerprint. There is no in-place upgrade path —
-- pre-alpha policy is rebuild on incompatibility.
--
-- The ``vector`` extension is enabled at connect-time in Python (not
-- here) so the SQL below can use ``::vector`` casts on the first
-- connection. Per-version ``vec_chunks_v<id>`` / ``vec_assets_v<id>``
-- tables are created lazily in Python so the engine can parameterise
-- the embedding dimension at first insert. See storage/postgres.py.
--
-- FTS surface: ``chunks.fts`` is a plain ``tsvector NOT NULL`` indexed
-- by GIN. The Python adapter populates it on INSERT via
-- ``to_tsvector('simple', preprocess_for_fts(text, tokenizer=cjk_tokenizer))``
-- so the index side runs the *same* CJK tokenizer (jieba, when
-- configured) as the query side's ``info/search.py:_sanitize_fts`` —
-- guaranteeing byte-level symmetry with the SQLite backend's FTS5
-- path. (A ``GENERATED ALWAYS AS to_tsvector('simple', text)`` column
-- can't reach Python's jieba and would leave the index unsegmented for
-- CJK input, dropping BM25 recall on Chinese queries.) ``fts_search``
-- consumes the OR-form sanitizer output via ``to_tsquery`` with a
-- per-call format adapter ``_fts_to_tsquery_string``;
-- ``plainto_tsquery`` would re-tokenize the sanitizer output and
-- silently break multi-word queries.

CREATE TABLE IF NOT EXISTS meta_kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ---- D layer -------------------------------------------------------------

-- See migrations/sqlite/schema.sql for the path / path_key split rationale.
CREATE TABLE IF NOT EXISTS documents (
    doc_id   TEXT PRIMARY KEY,
    path     TEXT NOT NULL,
    path_key TEXT NOT NULL UNIQUE,
    title    TEXT,
    hash     TEXT NOT NULL,
    mtime    DOUBLE PRECISION,
    layer    TEXT NOT NULL CHECK (layer IN ('source','wiki','wisdom')),
    active   BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS documents_layer_active ON documents(layer, active);

-- Reverse lookup by content hash.
CREATE INDEX IF NOT EXISTS documents_hash_idx ON documents(hash);

-- ---- I layer -------------------------------------------------------------

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    start_off   INTEGER NOT NULL,
    end_off     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    fts         tsvector NOT NULL,
    UNIQUE (doc_id, seq)
);

CREATE INDEX IF NOT EXISTS chunks_doc_seq ON chunks(doc_id, seq);
CREATE INDEX IF NOT EXISTS chunks_fts ON chunks USING GIN (fts);

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

-- See migrations/sqlite/schema.sql for the design rationale. Mirror of
-- the SQLite shape; no FK on either side, explicit cleanup in
-- ``delete_document``.
CREATE TABLE IF NOT EXISTS provenance (
    src_doc_id      TEXT NOT NULL,
    source_path     TEXT NOT NULL,
    source_path_key TEXT NOT NULL,
    PRIMARY KEY (src_doc_id, source_path_key)
);

CREATE INDEX IF NOT EXISTS provenance_source_key
    ON provenance(source_path_key);

CREATE TABLE IF NOT EXISTS wiki_log (
    id     BIGSERIAL PRIMARY KEY,
    ts     DOUBLE PRECISION NOT NULL,
    action TEXT NOT NULL,
    src    TEXT,
    dst    TEXT,
    note   TEXT
);

CREATE INDEX IF NOT EXISTS wiki_log_ts ON wiki_log(ts);

-- ---- W layer -------------------------------------------------------------

CREATE TABLE IF NOT EXISTS wisdom_items (
    item_id     TEXT PRIMARY KEY,
    kind        TEXT NOT NULL CHECK (kind IN ('principle','lesson','pattern')),
    status      TEXT NOT NULL DEFAULT 'candidate'
                CHECK (status IN ('candidate','approved','archived')),
    path        TEXT,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    confidence  DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    created_ts  DOUBLE PRECISION NOT NULL,
    approved_ts DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS wisdom_status ON wisdom_items(status);
CREATE INDEX IF NOT EXISTS wisdom_kind   ON wisdom_items(kind);

CREATE TABLE IF NOT EXISTS wisdom_evidence (
    id      BIGSERIAL PRIMARY KEY,
    item_id TEXT NOT NULL REFERENCES wisdom_items(item_id) ON DELETE CASCADE,
    doc_id  TEXT NOT NULL REFERENCES documents(doc_id),
    excerpt TEXT NOT NULL,
    line    INTEGER
);

CREATE INDEX IF NOT EXISTS wisdom_evidence_item ON wisdom_evidence(item_id);

-- ---- Multimedia assets ---------------------------------------------------

-- ``bytes`` widens to BIGINT because PG INTEGER is 32-bit; SQLite
-- INTEGER is variable-width (effectively 64-bit). Conservative side of
-- the cross-backend size contract.
CREATE TABLE IF NOT EXISTS assets (
    asset_id       TEXT PRIMARY KEY,        -- sha256 hex of the bytes; content-addressed identity
    kind           TEXT NOT NULL CHECK (kind IN ('image')),
    mime           TEXT NOT NULL,
    stored_path    TEXT NOT NULL,
    original_paths TEXT NOT NULL,           -- JSON list (parity with SQLite)
    bytes          BIGINT NOT NULL,
    media_meta     TEXT,                    -- per-kind JSON (image: {"kind","width","height"})
    created_ts     DOUBLE PRECISION NOT NULL
);

-- See migrations/sqlite/schema.sql for the CHECK / UNIQUE rationale.
CREATE TABLE IF NOT EXISTS chunk_asset_refs (
    chunk_id        BIGINT  NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
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

CREATE TABLE IF NOT EXISTS embed_versions (
    version_id  BIGSERIAL PRIMARY KEY,
    provider    TEXT             NOT NULL,
    model       TEXT             NOT NULL,
    revision    TEXT             NOT NULL DEFAULT '',
    dim         INTEGER          NOT NULL,
    normalize   BOOLEAN          NOT NULL,
    distance    TEXT             NOT NULL CHECK (distance IN ('cosine','l2','dot')),
    modality    TEXT             NOT NULL CHECK (modality IN ('text','multimodal')),
    created_ts  DOUBLE PRECISION NOT NULL,
    is_active   BOOLEAN          NOT NULL DEFAULT TRUE,
    UNIQUE (provider, model, revision, dim, normalize, distance, modality)
);

CREATE INDEX IF NOT EXISTS embed_versions_active
    ON embed_versions(modality, is_active);

-- Per-chunk embedding metadata. Vector lives in vec_chunks_v<version_id>
-- (created at runtime).
CREATE TABLE IF NOT EXISTS chunk_embed_meta (
    chunk_id   BIGINT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    version_id BIGINT NOT NULL REFERENCES embed_versions(version_id),
    PRIMARY KEY (chunk_id, version_id)
);

CREATE INDEX IF NOT EXISTS chunk_embed_meta_version
    ON chunk_embed_meta(version_id);

-- Per-asset embedding metadata. Mirror of chunk_embed_meta.
CREATE TABLE IF NOT EXISTS asset_embed_meta (
    asset_id   TEXT   NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    version_id BIGINT NOT NULL REFERENCES embed_versions(version_id),
    PRIMARY KEY (asset_id, version_id)
);

CREATE INDEX IF NOT EXISTS asset_embed_meta_version
    ON asset_embed_meta(version_id);

-- Per-wisdom-item embedding metadata. Mirror of chunk_embed_meta and
-- asset_embed_meta. The per-version vector lives in
-- vec_wisdom_v<version_id> (runtime-created in storage/postgres.py
-- because pgvector needs the dim parameterised into the vector(<dim>)
-- column type). Wisdom reuses the text modality on embed_versions.
CREATE TABLE IF NOT EXISTS wisdom_embed_meta (
    item_id    TEXT   NOT NULL REFERENCES wisdom_items(item_id) ON DELETE CASCADE,
    version_id BIGINT NOT NULL REFERENCES embed_versions(version_id),
    PRIMARY KEY (item_id, version_id)
);

CREATE INDEX IF NOT EXISTS wisdom_embed_meta_version
    ON wisdom_embed_meta(version_id);

-- ---- Embedding cache ----------------------------------------------------

-- Content-addressed embedding cache. See migrations/sqlite/schema.sql
-- for the full rationale; this is the PG mirror with BYTEA in place of
-- BLOB. The byte-level packing is identical (struct-packed float32
-- little-endian) so the cross-adapter cache contract is byte-exact.
CREATE TABLE IF NOT EXISTS embed_cache (
    content_hash TEXT             NOT NULL,
    version_id   BIGINT           NOT NULL REFERENCES embed_versions(version_id),
    dim          INTEGER          NOT NULL,
    embedding    BYTEA            NOT NULL,
    created_ts   DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (content_hash, version_id)
);
