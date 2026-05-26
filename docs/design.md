# dikw-core — AI-Native Knowledge Engine (Plan)

> **⚠ 0.3.0 W-layer refactor in flight (PR2 of 4) — some sections
> below are stale.** This document still describes the prior W layer
> as an LLM-distilled candidate/review pipeline in places (`distill` /
> `_candidates/` / `wisdom_items` table / `review approve|reject` /
> `GET /v1/wisdom/applicable` / `WisdomItem` / `WisdomKind`). All of
> those were removed in 0.3.0 PR1 and are being replaced with a
> hand-written first-class document layer under
> `wisdom/<author>/<slug>.md`. **The "Wisdom Layer Design" section
> and the Storage-Abstraction section have been rewritten for PR2 and
> are current.** Treat other distill/wisdom_items mentions as
> historical until PR4 rewrites the spec end-to-end. See CHANGELOG
> for the full migration log.

## Context

`dikw-core` is a greenfield, open-source project. The goal is an **AI-native knowledge engine** inspired by Karpathy's "LLM Wiki" pattern ([gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)), but extended end-to-end across the **DIKW pyramid** — Data → Information → Knowledge → Wisdom.

Why this project exists:
- Karpathy's LLM-Wiki pattern captures a real gap in today's RAG stacks: **knowledge should be a compounding artifact, not a query-time search result.** His pattern stops at Knowledge (markdown wiki with index.md + log.md).
- Existing reference tools (`mineru-doc-explorer`, `qmd`) implement the pattern in TypeScript/Node, local-first with GGUF models and SQLite+sqlite-vec. They cover D→I→K well but do not treat Wisdom (principles, lessons, transferable judgment) as a first-class layer.
- The user wants a **Python-native** implementation that (a) makes all four DIKW layers first-class, (b) is pluggable across LLM providers via API, (c) targets personal and enterprise knowledge bases, (d) is packaged with `uv` and hosted on GitHub.

Design decisions already locked in (via clarifying Q&A):
- **Scope:** full D→I→K→W four layers, with Wisdom as the differentiator.
- **Providers:** API-first, pluggable. First-party: Anthropic + OpenAI-compatible (covers OpenAI, Azure, Ollama, DeepSeek, Gemini-compat, etc.). Local (llama-cpp-python) deferred.
- **MVP source format:** Markdown only. Other formats via a backend-registry extension point later.

## Vision & Principles

1. **DIKW as first-class layers** — each layer has its own storage, schemas, and operations. The pipeline between layers is explicit (not an implicit by-product of retrieval).
2. **Wiki-as-artifact** — Knowledge & Wisdom layers are plain markdown on disk, versioned with git by the user, editable by humans and LLMs. The engine is a tool; the wiki is the product.
3. **Scoping deterministic, reasoning probabilistic** (Karpathy) — navigation uses deterministic structure (index.md, link graph, FTS); LLM calls are reserved for synthesis, distillation, and answering.
4. **Server-as-the-engine, CLI-as-the-client** — the engine is a long-lived `dikw serve` (FastAPI + NDJSON streaming) process that owns storage and provider connections; humans drive it through `dikw client *`, agents through HTTP. There is no in-process import path for end-user operations.
5. **Local-first data, pluggable compute** — the wiki lives on the user's filesystem; the default index is a local SQLite DB; only LLM calls leave the machine (and are provider-abstracted).
6. **Pluggable storage** — the engine talks to an abstract **Storage** interface, not to SQL directly. Two backends ship: **SQLite+sqlite-vec** (default, single-user local) and **Postgres+pgvector** (enterprise, multi-user). Swapping backends is a config change.
7. **Obsidian-compatible on-disk format** — the K & W layers are written as a plain markdown tree that Obsidian (or any MD editor) opens as a vault: `[[wikilinks]]`, YAML front-matter with tags, folder-based organization, daily-note conventions. The engine is a collaborator, not a walled garden; the user owns the files.
8. **YAGNI + extension points** — ship a tight MVP, but put named seams (provider adapter, storage adapter, source-backend registry, prompt registry) where known growth vectors are.

## The Four Layers (concrete definitions)

| Layer | What it is | Storage | Who writes it |
|---|---|---|---|
| **D — Data** | Raw, immutable sources (markdown files the user curates) | filesystem + indexed `documents` table in SQLite (path, content hash, layer, active) | human |
| **I — Information** | Parsed, chunked, embedded, indexed — enables fast lookup | SQLite FTS5 + sqlite-vec (`.dikw/index.sqlite`) | engine (deterministic) |
| **K — Knowledge** | LLM-authored wiki pages: summaries, entities, concepts, cross-refs, `index.md`, `log.md`; each page's `sources:` frontmatter is reconciled into a dedicated **provenance** edge (page → D-source attribution, separate from body `[[wikilinks]]` — see [ADR-0001](adr/0001-provenance-as-separate-edge.md)) | markdown files in `wiki/` | LLM, human-editable |
| **W — Wisdom** | Distilled principles, heuristics, lessons, patterns — transferable beyond a single source | markdown files in `wisdom/` with explicit provenance & review status | LLM proposes, human confirms |

The W layer is the novel bit and is spelled out in "Wisdom Layer Design" below.

## Target Architecture

```
                 ┌──────────────────────────────────────────┐
 User & Agents → │  Remote CLI (dikw client …)              │
                 │  Typer + httpx + rich + NDJSON           │
                 └────────────────┬─────────────────────────┘
                                  │ HTTP + NDJSON streaming
                 ┌────────────────▼─────────────────────────┐
                 │  Server (dikw serve — FastAPI + Uvicorn) │
                 │  sync RPC + async task subsystem +       │
                 │  ProgressBus + per-task NDJSON event tap │
                 └────────────────┬─────────────────────────┘
                                  │
                 ┌────────────────▼─────────────────────────┐
                 │  Core API (dikw_core.api)                │
                 │  ingest · synthesize · distill · retrieve│
                 │  · lint · status                         │
                 └────────────────┬─────────────────────────┘
          ┌───────────────────────┼────────────────────────┐
          ▼                       ▼                        ▼
 ┌────────────────┐     ┌───────────────────┐   ┌────────────────────┐
 │  Data (D)      │     │ Information (I)   │   │ Knowledge (K) /    │
 │  sources.py    │     │ chunk · embed ·   │   │ Wisdom (W)         │
 │  backends/md   │──▶  │ index · search    │◀─▶│ wiki/ · wisdom/    │
 │  (content-hash)│     │ (FTS5 + vec + RRF)│   │ links · log        │
 └────────┬───────┘     └─────────┬─────────┘   └──────────┬─────────┘
          │                       │                        │
          └───────────────────────▼────────────────────────┘
                ┌─────────────────▼───────────────────────────┐
                │  Storage adapter  (dikw_core.storage)       │
                │  base · sqlite (default) · postgres         │
                └─────────────────┬───────────────────────────┘
                                  │
        ┌─────────────────────────┴───────────────────────┐
        ▼                                                 ▼
 SQLite+sqlite-vec+FTS5                  Postgres+pgvector+tsvector
 (single-user, local)                    (multi-user, enterprise)
                                  │
                 ┌────────────────▼─────────────────────────┐
                 │ Providers (LLM + Embedding)              │
                 │ base · anthropic_compat · openai_compat  │
                 └──────────────────────────────────────────┘
```

Module boundaries are chosen so each subpackage fits in a single reading pass and has a named interface. Engine code depends only on the **Storage** Protocol — never on raw SQL or backend-specific tables — which keeps the SQLite/Postgres seam sharp.

## Tech Stack

- **Language**: Python 3.12+
- **Packaging**: `uv` → `pyproject.toml` (PEP 621), `uv.lock` committed; single source layout under `src/dikw_core/`
- **Storage (default)**: stdlib `sqlite3` + `sqlite-vec` (pip) for vectors; FTS5 built into SQLite. Behind a `Storage` Protocol.
- **Storage (enterprise)**: Postgres 15+ with `pgvector` ≥0.6 and `tsvector` + GIN for full-text, via `psycopg[binary,pool]`. Optional extra: `uv pip install dikw-core[postgres]`.
- **Schemas**: Pydantic v2 for config, records, tool I/O
- **Markdown**: `markdown-it-py` + `python-frontmatter`; wiki-link parsing via a small in-repo module (not a heavy dep)
- **LLM SDKs**: `anthropic`, `openai` (the `openai` SDK covers all OpenAI-compatible endpoints), behind a thin provider interface
- **Embeddings**: default through an OpenAI-compatible `embeddings` endpoint (works for OpenAI, Ollama, TEI, etc.); Anthropic path uses OpenAI-compat for embeddings since Anthropic has no embeddings API
- **HTTP server**: `fastapi` + `uvicorn[standard]` + `python-multipart` for source-import multipart payloads
- **CLI & output**: `typer` + `httpx` + `rich`
- **Quality**: `pytest`, `pytest-asyncio`, `ruff`, `mypy --strict` where practical
- **CI**: GitHub Actions — lint + type-check + tests on 3.12/3.13

Known patterns to reuse from references (concrete sources):
- **Hybrid search pipeline (BM25 + vector + RRF + rerank)** — `mineru-doc-explorer/src/hybrid-search.ts`, `mineru-doc-explorer/src/search.ts`. Port the RRF fusion + position-aware blending logic.
- **SQLite schema design + content-addressed storage** — `mineru-doc-explorer/src/db-schema.ts`, `mineru-doc-explorer/src/store.ts` (documents table with indexed content hash, links table, wiki_log).
- **Smart markdown chunking (~900 tokens, 15% overlap, heading-aware)** — `mineru-doc-explorer/src/store.ts` chunking section; `qmd/src/store.ts` lines ~257–310.
- **Wikilink parsing + forward/backward graph** — `mineru-doc-explorer/src/links.ts`, `mineru-doc-explorer/src/wiki/{log,lint,index-gen}.ts`. Port to a small `knowledge/links.py`.
- **HTTP route grouping** — server endpoints map 1:1 to `dikw_core.api` methods, grouped under `/v1/{sync,tasks,import,retrieve}` so the wire surface mirrors the engine seam. Long ops (ingest / synth / distill / eval) return a `task_id` whose progress is consumed via the paged JSON cursor at `GET /v1/tasks/{id}/events` (long-poll with `wait>0`); retrieve streams inline NDJSON (no task_id, short-lived); sync ops return JSON directly. **LLM synthesis is not a dikw-core verb** — agents call `retrieve` and run their own LLM on the returned chunks.
- **YAML config + schema validation** — `mineru-doc-explorer/src/config-schema.ts` (Zod) → Pydantic v2 equivalent in `dikw_core/config.py`.
- **Strong-signal short-circuit** (skip expensive LLM expansion when FTS already gives a confident top hit) — `qmd/src/store.ts:4057–4076`.

## Package Layout

```
dikw-core/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
├── .python-version           # 3.12
├── .github/workflows/ci.yml
├── .gitignore
├── src/dikw_core/
│   ├── __init__.py
│   ├── api.py                # engine facade — server routes + eval runner depend on it
│   ├── config.py             # Pydantic models + YAML loader
│   ├── schemas.py            # cross-layer record types
│   │
│   ├── storage/              # Storage adapters — SQLite (default) + Postgres
│   │   ├── __init__.py       # factory: resolves backend from config
│   │   ├── base.py           # Storage Protocol + typed DTOs
│   │   ├── sqlite.py         # SQLite + sqlite-vec + FTS5 implementation (default)
│   │   ├── postgres.py       # tsvector + pgvector (optional [postgres] extra)
│   │   └── migrations/       # per-backend schema (single schema.sql each)
│   │       ├── sqlite/       #   schema.sql
│   │       └── postgres/     #   schema.sql
│   │
│   ├── domains/              # DIKW domain model — the four layers
│   │   ├── data/             # D layer
│   │   │   ├── sources.py    # source registry, hashing, mtime tracking
│   │   │   └── backends/
│   │   │       ├── __init__.py  # backend registry (extension point)
│   │   │       └── markdown.py  # MD parser + front-matter + deep-read
│   │   │
│   │   ├── info/             # I layer
│   │   │   ├── chunk.py      # heading-aware markdown chunking
│   │   │   ├── embed.py      # batched embedding via provider
│   │   │   ├── index.py      # FTS5 + sqlite-vec writes
│   │   │   └── search.py     # BM25 + vector + RRF + optional rerank
│   │   │
│   │   ├── knowledge/        # K layer
│   │   │   ├── wiki.py       # page read/write, front-matter conventions
│   │   │   ├── synthesize.py # ingest → wiki pages (LLM-driven)
│   │   │   ├── links.py      # wikilink/markdown/URL link graph
│   │   │   ├── indexgen.py   # regenerate index.md from wiki/
│   │   │   └── log.py        # append-only wiki_log + log.md renderer
│   │   │
│   │   └── wisdom/           # W layer (see dedicated section)
│   │       ├── distill.py    # propose principles/lessons/patterns
│   │       ├── review.py     # human-confirmation workflow
│   │       └── apply.py      # rank wisdom items applicable to a question (exposed via /v1/wisdom/applicable)
│   │
│   ├── providers/            # LLM + embedding abstraction
│   │   ├── base.py           # LLMProvider, EmbeddingProvider protocols
│   │   ├── anthropic.py      # claude sonnet/haiku via anthropic SDK
│   │   └── openai_compat.py  # openai SDK pointed at any compat endpoint
│   │
│   ├── prompts/              # versioned prompt templates (Jinja2-lite strings)
│   │   ├── synthesize.md
│   │   ├── distill.md
│   │   └── lint.md
│   │
│   ├── server/               # FastAPI app, auth, sync + task routes, NDJSON streamer
│   ├── client/               # remote Typer CLI + httpx transport + NDJSON progress + sources importer
│   └── cli.py                # top-level typer app: version, init, serve + dikw client subgroup
│
├── tests/
│   ├── fixtures/             # small MD corpora
│   ├── test_chunk.py
│   ├── test_search.py        # FTS + vector + RRF behavior on golden set
│   ├── test_wiki.py
│   ├── test_distill.py
│   ├── test_providers.py     # uses recorded responses
│   ├── test_storage_contract.py  # same contract test runs against every backend
│   ├── server/               # HTTP-level tests against an in-memory ASGI app
│   └── client/               # transport, config, importer, progress renderer tests
└── examples/
    └── personal-wiki/        # runnable demo wiki
```

## On-Disk User Wiki Layout (convention, not code)

```
my-wiki/
├── dikw.yml                  # config: sources, provider, schema
├── sources/                  # user-curated raw markdown (D layer)
├── wiki/                     # K layer (LLM-authored, human-editable)
│   ├── index.md              # auto-generated catalog
│   ├── log.md                # append-only chronology
│   ├── entities/
│   ├── concepts/
│   └── notes/
├── wisdom/                   # W layer
│   ├── principles.md
│   ├── lessons.md
│   ├── patterns.md
│   └── _candidates/          # LLM proposals awaiting human review
└── .dikw/                    # engine-managed, gitignored by default
    ├── index.sqlite          # I layer when storage.backend=sqlite
    └── cache/                # model/artifact caches (backend-agnostic)
```

**Obsidian vault compatibility** — `my-wiki/` is itself a valid Obsidian vault. The engine follows these conventions so Obsidian (or any plain MD editor) can open it and edit alongside the engine without conflict:
- `[[Wikilinks]]` — the canonical link form in `wiki/` and `wisdom/`. `[[Page#Heading]]` and `[[Page|alias]]` supported.
- **YAML front-matter** — every engine-authored page has `---`-delimited front-matter with at least `id`, `kind` (for wisdom) or `type` (for wiki), `created`, `updated`, and `tags: [...]`. Obsidian reads `tags` natively.
- **Folder = category** — `wiki/entities/`, `wiki/concepts/`, `wiki/notes/`, `wisdom/_candidates/`. Matches Obsidian's default folder-sort behavior.
- **Daily-note style log** — `wiki/log.md` keeps Karpathy's chronological format; optionally daily files under `wiki/log/YYYY/MM/YYYY-MM-DD.md` for vaults that already use Obsidian's daily-notes plugin (opt-in via `schema.log_style: daily`).
- **Engine state stays out of the vault** — the `.dikw/` sidecar directory is gitignored and Obsidian-ignored (`.obsidian/app.json` `userIgnoreFilters` receives a `.dikw/` entry on `dikw init`).
- **No bespoke syntax in MD bodies** — only standard Markdown + wikilinks + front-matter, so a human editing in Obsidian never sees engine-only constructs that would get stripped on round-trip.

`dikw.yml` example:
```yaml
provider:
  llm: anthropic_compat    # or: openai_compat (both are protocol names)
  llm_model: claude-sonnet-4-6
  embedding: openai_compat
  embedding_model: text-embedding-3-small
  embedding_base_url: https://api.openai.com/v1
storage:
  backend: sqlite          # sqlite | postgres
  # --- sqlite-specific (default) ---
  path: .dikw/index.sqlite
  # --- postgres-specific ---
  # dsn: postgresql://user:pass@host:5432/dikw
  # schema: dikw            # isolates multi-tenant deployments
  # pool_size: 10
schema:
  description: "Personal research wiki on AI safety"
  page_types: [entity, concept, note]
  wisdom_kinds: [principle, lesson, pattern]
sources:
  - path: ./sources
    pattern: "**/*.md"
    ignore: ["drafts/**"]
```

## Data Model

The logical model is backend-agnostic; the SQL below is the **SQLite reference schema** used by the MVP adapter. The Postgres adapter maps the same logical entities to equivalent structures — `tsvector` + GIN for FTS, `pgvector` for embeddings, regular tables for the rest — behind the same `Storage` Protocol.

### SQLite reference schema (MVP)

```sql
-- D
-- ``path`` carries the user's spelling (display path); ``path_key`` is
-- the engine's NFC + casefold lookup key. Splitting the two lets the
-- same logical file under different macOS NFD / NTFS-case spellings
-- resolve to a single row while ``dikw client status`` still shows whichever
-- spelling is on disk. ``data/path_norm.normalize_path`` is the single
-- source of truth for the transformation.
CREATE TABLE documents (
    doc_id   TEXT PRIMARY KEY,
    path     TEXT NOT NULL,
    path_key TEXT NOT NULL UNIQUE,
    title    TEXT,
    hash     TEXT NOT NULL,             -- sha256 of body; indexed for reverse lookup
    mtime    REAL,
    layer    TEXT CHECK (layer IN ('source','wiki','wisdom')) NOT NULL,
    active   INTEGER DEFAULT 1
);
CREATE INDEX documents_hash_idx ON documents(hash);

-- I
-- Body-only FTS5 over chunk text; rowid aligns with chunks.chunk_id.
-- ``remove_diacritics 0`` is intentional and matches PG's
-- ``to_tsvector('simple', text)`` byte-level behavior.
CREATE VIRTUAL TABLE documents_fts USING fts5(
    body, tokenize = "unicode61 remove_diacritics 0"
);
CREATE TABLE chunks (
    chunk_id  INTEGER PRIMARY KEY,
    doc_id    TEXT REFERENCES documents(doc_id),
    seq       INTEGER,
    start_off INTEGER, end_off INTEGER,
    text      TEXT
);
-- Per-version vec table (one per embed_versions row, lazy-created at first
-- ingest). The dim is locked at CREATE time per sqlite-vec, so a model swap
-- registers a fresh `embed_versions` row + a fresh `vec_chunks_v<n>` table
-- — old vectors stay queryable, no rebuild needed.
CREATE VIRTUAL TABLE vec_chunks_v<n> USING vec0(embedding float[<dim>]);
CREATE TABLE chunk_embed_meta (
    chunk_id   INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    version_id INTEGER NOT NULL REFERENCES embed_versions(version_id),
    PRIMARY KEY (chunk_id, version_id)
);

-- K (link graph spans K and W)
CREATE TABLE links (
    src_doc_id TEXT, dst_path TEXT,
    link_type  TEXT CHECK (link_type IN ('wikilink','markdown','url')),
    anchor     TEXT, line INTEGER,
    PRIMARY KEY (src_doc_id, dst_path, line)
);
CREATE TABLE wiki_log (
    ts INTEGER, action TEXT, src TEXT, dst TEXT, note TEXT
);

-- W
CREATE TABLE wisdom_items (
    item_id       TEXT PRIMARY KEY,
    kind          TEXT CHECK (kind IN ('principle','lesson','pattern')),
    status        TEXT CHECK (status IN ('candidate','approved','archived')) DEFAULT 'candidate',
    path          TEXT,          -- wisdom/<file>.md anchor
    title         TEXT,
    body          TEXT,
    confidence    REAL,
    created_ts    INTEGER, approved_ts INTEGER
);
CREATE TABLE wisdom_evidence (
    item_id  TEXT REFERENCES wisdom_items(item_id),
    doc_id   TEXT REFERENCES documents(doc_id),
    excerpt  TEXT, line INTEGER
);
```

### Multimedia assets (v1)

Image binaries referenced from markdown sources are content-addressed by the
sha256 of their bytes (stored verbatim in `asset_id`, mirroring the role of
`documents.hash` for sources). The chunk → asset bridge keeps text positions
recoverable across file moves; per-asset embeddings live in their own
runtime-created vec table, parallel to `vec_chunks_v<n>`.

```sql
-- D (multimedia, v1: images only)
CREATE TABLE assets (
    asset_id       TEXT PRIMARY KEY,        -- sha256 hex of the bytes
    kind           TEXT CHECK (kind IN ('image')) NOT NULL,
    mime           TEXT NOT NULL,
    stored_path    TEXT NOT NULL,           -- relative to project_root
    original_paths TEXT NOT NULL,           -- JSON list of source-side names
    bytes          INTEGER NOT NULL,
    media_meta     TEXT,                    -- per-kind JSON; image: {width,height}
    created_ts     REAL NOT NULL
);

-- I (chunk → asset bridge)
CREATE TABLE chunk_asset_refs (
    chunk_id       INTEGER REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    asset_id       TEXT    REFERENCES assets(asset_id),
    ord            INTEGER NOT NULL,        -- 0-based ordinal within chunk
    alt            TEXT NOT NULL DEFAULT '',
    start_in_chunk INTEGER NOT NULL,
    end_in_chunk   INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, ord)
);

-- I (asset embedding metadata; vector lives in vec_assets_v<n>, lazy-created
--    by the same dim-locked CREATE pattern as vec_chunks_v<n>)
CREATE TABLE asset_embed_meta (
    asset_id   TEXT    NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    version_id INTEGER NOT NULL REFERENCES embed_versions(version_id),
    PRIMARY KEY (asset_id, version_id)
);
```

`media_meta` is a per-kind discriminated union — for v1 images it carries
`width` / `height`; future modalities (audio, video) slot in their own fields
without an `ALTER TABLE`. `embed_versions` is the cross-modal embedding
registry shared with `chunk_embed_meta` (text and multimodal versions
coexist; each row produces its own `vec_*_v<n>` table).

## Core Operations

Each operation is implemented in `dikw_core.api` and surfaced over HTTP by the server (`dikw_core.server`); the remote CLI (`dikw_core.client`) and any agent / web UI consume the same wire contract.

| Op | Input | Output | Notes |
|---|---|---|---|
| `ingest(paths)` | file paths | updated `documents`/`chunks`/`documents_fts`/`vec_chunks_v<id>` | D→I; deterministic; idempotent by content hash |
| `synthesize(scope)` | source doc_ids (or "new since log") | new/updated wiki pages + wiki_log entries | I→K; LLM call with prompts/synthesize.md |
| `retrieve(q)` | user question | ranked chunks + page refs (no LLM call) | hybrid search via `info/search.py` (BM25 + vec RRF); **LLM synthesis is the agent's responsibility, not dikw-core's** |
| `lint()` | — | report of broken links, orphan pages, duplicated entities | K+W hygiene; prompts/lint.md |
| `status()` | — | counts per layer, last-ingest, last-synthesize, pending review | for CLI and HTTP `/v1/status` |

## Wisdom Layer Design

W layer is a first-class document layer alongside K. A wisdom page is a
plain markdown file under `wisdom/<author>/<slug>.md` that a human writes
by hand in Obsidian — there is no LLM proposal, no candidate queue, no
review state machine. Authorship is encoded by directory
(`wisdom/elon-musk/...` attributes the page to `elon-musk`); a wisdom
file directly under `wisdom/<slug>.md` (no author subdirectory) is
allowed and indexed with `author = None`.

Wisdom pages share the K-page contract:

- **YAML frontmatter:** optional `sources: [...]` list (same semantics as
  wiki pages — populates the `provenance` table), optional
  `status: draft | published | favorite | archived` enum (omitted ≡
  published), free-form additional keys. Wiki and Source layer documents
  may carry `status:` in YAML but the engine forces
  `DocumentRecord.status = NULL` for them — status is wisdom-only.
- **Body:** markdown with `[[wikilinks]]` that resolve across both wiki
  and wisdom layers via title. Title collisions across layers fall
  through the existing refuse-to-resolve mechanism (lint surfaces the
  ambiguity); the user disambiguates with the path form
  `[[wisdom/elon-musk/be-relentless|Be Relentless]]`.
- **Persistence:** `dikw ingest` scans `<root>/wisdom/**/*.md` after the
  `sources:` scan, hash-idempotent, runs the same `persist_page`
  pipeline as wiki (`documents` row + `chunks` + per-version embedding +
  `links` + `provenance`). `wisdom/_candidates/` and the legacy
  `principles.md` / `lessons.md` / `patterns.md` aggregate files are
  hard-coded skips so an upgrading base does not accidentally index
  drained queue artifacts.

In 0.3.0 PR2 (this commit) wisdom pages flow through ingest and into
`documents`. PR3 will wire `dikw client retrieve` to surface them
alongside wiki hits (`Hit.layer == "wisdom"`) and extend
`broken_wikilink` / `missing_provenance` / `orphan_page` lint coverage
to the wisdom layer; PR4 is the docs/CHANGELOG/ADR final pass.

There is no `≥N evidence` gate, no `kind` taxonomy, and no review state
machine. `status` is a flat enum the human sets; the engine validates it
via the `invalid_wisdom_status` lint kind (added in PR2, non-blocking
warning) but does not yet consume it for retrieval filtering or boost.

## Storage Abstraction

Goal: one engine, many backends. The rest of `dikw_core` never touches SQL; it calls into a Protocol.

`storage/base.py` (sketch):
```python
class Storage(Protocol):
    # lifecycle
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def migrate(self) -> None: ...             # idempotent schema bring-up

    # D layer
    async def upsert_document(self, doc: DocumentRecord) -> None: ...
    async def get_document(self, doc_id: str) -> DocumentRecord | None: ...
    async def list_documents(self, *, layer: Layer, active: bool | None = True,
                             since_ts: float | None = None) -> Iterable[DocumentRecord]: ...
    async def deactivate_document(self, doc_id: str) -> None: ...

    # I layer
    async def replace_chunks(self, doc_id: str, chunks: Sequence[ChunkRecord]) -> None: ...
    async def upsert_embeddings(self, rows: Sequence[EmbeddingRow]) -> None: ...
    async def fts_search(self, q: str, *, limit: int, layer: Layer | None = None
                        ) -> list[FTSHit]: ...
    async def vec_search(self, embedding: list[float], *, limit: int,
                         layer: Layer | None = None) -> list[VecHit]: ...
    async def get_chunk(self, chunk_id: int) -> ChunkRecord | None: ...

    # K + W layer (both flow through documents / chunks; wisdom-only fields)
    async def upsert_link(self, link: LinkRecord) -> None: ...
    async def links_from(self, src_doc_id: str) -> list[LinkRecord]: ...
    async def links_to(self, dst_path: str) -> list[LinkRecord]: ...
    async def append_wiki_log(self, entry: WikiLogEntry) -> None: ...
    async def replace_provenance_from(self, src_doc_id: str,
                                      source_paths: Sequence[str]) -> None: ...

    # diagnostics
    async def counts(self) -> StorageCounts: ...
```

The W layer shares the K-layer Storage surface — both flow through the
`documents` row + `chunks` + `links` + `provenance` quartet, separated
only by `documents.layer` (`WIKI` vs `WISDOM`) and the wisdom-only
`documents.status` column (CHECK-constrained to the four enum values;
NULL for non-wisdom rows, enforced application-side at write time).

Design constraints:
- **No leaky query objects.** All inputs and outputs are plain Pydantic DTOs. No `cursor`, no `Session`, no backend-specific types crossing the boundary.
- **Hybrid search stays outside storage.** `info/search.py` owns RRF fusion / reranking; storage exposes only the two primitives (`fts_search`, `vec_search`) since the two backends express those very differently. This is the right abstraction seam — high enough to hide dialect, low enough to avoid re-implementing the fusion in each adapter.
- **Migrations are backend-owned.** `storage/migrations/sqlite/` ships SQL files; `storage/migrations/postgres/` will ship equivalents. A shared `Migrator` drives `await storage.migrate()`.
- **Contract tests.** `tests/test_storage_contract.py` defines a single pytest suite parameterized over `[sqlite, postgres]`; the Postgres variant skips unless `DIKW_TEST_POSTGRES_DSN` is set. This keeps engine code from growing backend-specific assumptions.
- **Transactional boundary.** One unit of work per engine operation (e.g., a single `ingest(path)` is one transaction on the adapter). Adapters are responsible for honoring that — SQLite via `BEGIN IMMEDIATE`, Postgres via `psycopg` transactions.

The Postgres adapter is installed as an **optional extra** so SQLite users never pay for the `psycopg` dependency footprint:
```toml
[project.optional-dependencies]
postgres = ["psycopg[binary,pool] >=3.2", "pgvector >=0.3"]
```

## Provider Abstraction

`providers/base.py`:
```python
class LLMProvider(Protocol):
    async def complete(self, *, system: str, user: str, model: str,
                       max_tokens: int, temperature: float,
                       tools: list[ToolSpec] | None = None) -> LLMResponse: ...

class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]: ...
```

`providers/anthropic.py` wraps the official `anthropic` SDK for LLM; raises for embedding (unsupported). `providers/openai_compat.py` wraps the official `openai` SDK and takes `base_url` + `api_key` from env/config, covering OpenAI proper, Azure OpenAI, Ollama, vLLM, TEI-style embedding endpoints, and any Claude Code-style OpenAI-compat. `providers/__init__.py` resolves instances from `dikw.yml`; swapping providers is a config-only change.

Prompt caching: when the provider is Anthropic, use the `cache_control` param on the system prompt and large wiki blocks in `synthesize` — the wiki schema is near-static per session and is the prime caching target. (Query-time prompt caching is the agent's concern, not dikw-core's, since dikw-core does not call the LLM at retrieve time.)

## Interfaces

**Local CLI** (run in this process; no server required):
- `dikw version` — print package version
- `dikw init [path]` — scaffold `dikw.yml`, `sources/`, `wiki/`, `wisdom/`, `.dikw/`
- `dikw serve --base <path>` — start the FastAPI + NDJSON server bound to one base

**Remote CLI** (`dikw client *`, also reachable via top-level aliases):
- `dikw client status` — counts per layer
- `dikw client check [--llm-only|--embed-only]` — provider connectivity probe
- `dikw client import <path>` — pre-flight + import markdown packages (md + referenced assets) into the server's `sources/`
- `dikw client ingest [--no-embed]` — chunk + embed the server's `sources/` tree, stream progress
- `dikw client synth [--all]` — K synthesis
- `dikw client distill` — propose W-layer candidates
- `dikw client review {list,approve,reject}` — drive the W review state machine
- `dikw client retrieve "<q>"` — streamed retrieval (ranked chunks + page refs, no LLM call); agent supplies its own LLM synthesis on the result
- `dikw client wisdom applicable "<q>"` — _(PR-5)_ list approved wisdom items applicable to a question; until then use `GET /v1/wisdom?status=approved` over HTTP
- `dikw client lint` — hygiene report
- `dikw client eval [--dataset]` — run retrieval-quality evaluation
- `dikw client tasks {list,show,follow,cancel}` — inspect the server's async task queue

**HTTP surface** (the server is the canonical wire contract):
- Sync RPC under `/v1/` — `status`, `check`, `lint`, `init`, wiki page list/read, doc search, chunk fetch, wisdom list/approve/reject.
- Async tasks under `/v1/{ingest,synth,distill,eval}` — submit returns `task_id`; `GET /v1/tasks` paginates the queue via cursor JSON (`TaskListPage`, summary rows); `GET /v1/tasks/{id}/events?from_seq=N&wait=K` long-polls a paged JSON event cursor (`EventsPage`); `/result` and `/cancel` complete the lifecycle.
- Streaming retrieve — `POST /v1/retrieve` returns NDJSON: `retrieve_started → retrieval_done → final{succeeded|failed|cancelled}`. The final event payload carries ranked chunks (with full text) plus page refs. **No LLM tokens stream from the server** — synthesis is the agent's job. Applicable-wisdom surfacing lands in PR-5 via a separate `GET /v1/wisdom/applicable?q=...` endpoint, not on the retrieve payload.
- Sources import — `POST /v1/import` accepts a manifest + tar.gz (multipart upload at the transport layer), validates sha256, stages atomically, then commits per-package into `<base>/sources/` before ingest reads from disk.

## Phasing

- **Phase 0 — Scaffold (small):** repo layout, `uv` init, CI, ruff/mypy, typer CLI with `init`/`status`, config loader, **`Storage` Protocol + DTOs in `storage/base.py`**, SQLite bootstrap in `storage/sqlite.py`, `storage/__init__.py` factory, contract-test skeleton, minimal `providers/base.py` + Anthropic stub, a golden-path test that runs end-to-end on an empty wiki.
- **Phase 1 — D + I (foundation):** markdown backend, content-hash store, heading-aware chunker, embedding batch pipeline via OpenAI-compat, FTS5 index and sqlite-vec index implemented on the SQLite adapter, RRF hybrid `search` (fusion lives in `info/search.py`, calling `storage.fts_search` + `storage.vec_search`), `ingest` + `retrieve` CLI + HTTP routes. Acceptance: ingest a 50-file corpus, `retrieve` returns ranked chunks in <2s warm.
- **Phase 2 — K (wiki):** `synthesize` prompt + worker, wiki page writer, link graph, `index.md` regenerator, `log.md` append, `lint`, wiki HTTP routes. Acceptance: running `synth` on the Phase-1 corpus produces a non-empty `wiki/` with valid cross-links; `lint` reports 0 errors.
- **Phase 3 — W (wisdom, the differentiator):** `distill` prompt + worker, `wisdom_items` table, `_candidates/` flow, interactive `review`, `wisdom.apply` exposed via `GET /v1/wisdom/applicable?q=...` (agent-driven application, not server-internal injection), tests covering candidate→approved transitions and the "at least N=2 evidence" invariant.
- **Phase 4 — Polish:** OpenAI-compat provider completeness (Ollama and Azure verified), prompt-caching on Anthropic paths, packaging for PyPI (`pip install dikw-core`), docs site, GitHub Actions release automation.
- **Phase 5 — Alternate storage adapters:**
  - **Postgres (enterprise):** `storage/postgres.py` using `psycopg[binary,pool]` + `pgvector`, `migrations/postgres/schema.sql` with `tsvector`+GIN for FTS and `vector(N)` for embeddings. Contract test suite runs green against a `postgres:16`+`pgvector` container in CI. Packaged as `dikw-core[postgres]` optional extra.
  - Acceptance: the Phase 1–3 verification script runs end-to-end against the Postgres adapter with only `storage.backend` flipped in `dikw.yml`.

Each phase is a landable slice: CI green, tests added, docs updated.

## Critical Files to Create (first wave)

- `pyproject.toml` — declares package, pins runtime deps, `[project.optional-dependencies] postgres = [...]`, configures ruff/mypy/pytest
- `src/dikw_core/config.py` — Pydantic config + YAML loader (includes `storage:` block)
- `src/dikw_core/storage/base.py` — `Storage` Protocol + DTOs
- `src/dikw_core/storage/sqlite.py` — SQLite + sqlite-vec + FTS5 implementation
- `src/dikw_core/storage/migrations/sqlite/schema.sql` — reference schema
- `src/dikw_core/storage/__init__.py` — factory resolving backend from config
- `src/dikw_core/domains/data/backends/markdown.py` — MD parser + front-matter
- `src/dikw_core/domains/info/chunk.py` — heading-aware chunker (port logic from qmd `store.ts:257–310`)
- `src/dikw_core/domains/info/search.py` — RRF fusion on top of `storage.fts_search` + `storage.vec_search` (port from `mineru-doc-explorer/src/hybrid-search.ts`)
- `src/dikw_core/providers/{base,anthropic,openai_compat}.py`
- `src/dikw_core/cli.py`, `src/dikw_core/server/app.py`, `src/dikw_core/client/cli_app.py`
- `tests/test_storage_contract.py` — parameterized over backends
- `.github/workflows/ci.yml`

## Verification (how we'll know it works end-to-end)

1. `uv sync` resolves cleanly; `uv run pytest` green; `uv run ruff check` + `uv run mypy src` clean.
2. `uv run dikw init examples/personal-wiki && cd examples/personal-wiki` scaffolds the expected tree.
3. Populate `sources/` with ~20 markdown notes (fixtures); `uv run dikw client ingest`; confirm FTS and vec rows via a diagnostic `dikw client status`.
4. `uv run dikw client retrieve "what is DIKW?" --format json` returns at least one chunk hit with `path`, `text`, and `score`; LLM synthesis on top of these chunks is the agent's responsibility.
5. `uv run dikw client synth`; check `wiki/index.md` and `wiki/log.md` updated, at least one `entities/`/`concepts/` page created, all wikilinks resolve in `lint`.
6. `uv run dikw client distill --recent` creates ≥1 candidate in `wisdom/_candidates/`; `uv run dikw client review` accepts one; the corresponding `wisdom_items.status` flips to `approved` and a rendered page exists in `wisdom/principles.md` (or kind-appropriate file).
7. `uv run dikw client wisdom applicable "what principles apply when choosing retrieval over a wiki?" --format json` lists the approved wisdom item; the agent injects it into its own LLM prompt to produce a wisdom-grounded answer.
8. `uv run dikw serve --base .` launches; a `GET /v1/wisdom/applicable?q=...` round-trip from any HTTP client returns the same wisdom items as step 7, and a `POST /v1/retrieve` round-trip returns chunks consumable by any HTTP agent.
9. Swap provider in `dikw.yml` from Anthropic to OpenAI-compatible (pointed at Ollama locally or OpenAI) and repeat step 4 — works unchanged.
10. (After Phase 5, Postgres) `docker compose up postgres` (with `pgvector` image), set `storage.backend: postgres` in `dikw.yml`, rerun steps 3–8 against the Postgres adapter — every assertion holds, no engine code changes. The storage contract test suite runs green under `DIKW_TEST_POSTGRES_DSN=...` in CI.

## Open execution-time decisions (not blockers for plan approval)

- Exact embedding model default (text-embedding-3-small vs bge-small) — pick in Phase 1 after a tiny retrieval eval on the fixtures.
- Whether `knowledge/links.py` parses MDX-style links (probably no — Karpathy's pattern is vanilla MD).
- Whether to ship a `dikw.yml` schema JSON for editor autocomplete — nice-to-have in Phase 4.
- License choice (MIT vs Apache-2.0) — ask user before publishing.

## Multimedia Assets — v1 (images only)

dikw-core's v1 brings images into the retrieval surface. The design honours
the four invariants from the rest of this doc — Obsidian-vault-native
on-disk format, deterministic scoping, the Storage Protocol as the only
seam, and reuse of named extension points.

**On disk.** Image binaries referenced from a markdown source (either the
standard `![alt](path)` form or Obsidian's `![[file|alias]]`) are copied
into an engine-managed directory under the project root:

```
<project_root>/assets/<sha256[:2]>/<sha256[:8]>-<sanitized-name>.<ext>
```

The 2-char hash prefix shards the directory (256-way) so the asset
folder stays Finder-, rsync-, and Dropbox-friendly even at six-figure
asset counts. The 8-hex prefix in the filename guarantees uniqueness
even when two different binaries share a sanitized stem; the trailing
sanitized name preserves human-readable semantics (Obsidian shows
`ab3f12ef-architecture-diagram.png`, not an opaque hash). Sanitization
NFC-normalizes Unicode and whitelists letters/numbers (any script,
including CJK / JP / KR / Cyrillic / Greek) plus `-` and `_`; everything
else collapses to `-`. Length is capped at 150 UTF-8 bytes on a
character boundary.

**Reference preservation across path rewrite.** The user's source
markdown is never modified — `![alt](./diagrams/foo.png)` survives
verbatim in `chunks.text` (Layer 1). Structural mapping lives in the
new `chunk_asset_refs` bridge table (Layer 2) — `(chunk_id, asset_id,
ord, alt, start_in_chunk, end_in_chunk)` — so the relationship
between chunk text positions and the materialized asset is recoverable
regardless of file moves. A pure `info/render.py::render_chunk` helper
produces a self-contained Markdown rendering with `stored_path`
substituted into the original positions on demand (Layer 3).

**Span-aware chunking.** The paragraph-aligned chunker takes an
`atomic_spans` parameter and hard-fails (rather than silently splitting)
if any image reference would cross a chunk boundary. For valid
single-line markdown image syntax this is a no-op; the explicit
post-condition catches pathological inputs and future chunker variants.

**Native multimodal embedding.** v1 introduces a
`MultimodalEmbeddingProvider` Protocol (text + image inputs → shared
vector space). The first concrete impl wraps Gitee AI's hosted
multimodal embedding endpoint over httpx; other providers (Voyage,
Cohere v4, native Jina) drop in as siblings with no engine changes.
Chunk text and asset binaries both flow through the same provider so
their vectors share one space — a text query naturally matches
against image semantics, no caption-to-text intermediary required.

**Embedding versioning.** Every embedding generation is identified by
a composite tuple `(provider, model, revision, dim, normalize, distance)`
in the new `embed_versions` table. The `revision` field is the
user-facing escape hatch when a vendor silently refreshes weights
behind a stable model name. Each version gets its own per-version
sqlite-vec virtual table (`vec_assets_v<id>`) so dim changes are
naturally isolated and prior data survives a model swap.

**Three-leg retrieval.** `info/search.HybridSearcher` now fuses three
ranked lists via RRF: BM25 over chunk text, vector search over chunk
vectors, and vector search over asset vectors with reverse-lookup
into parent chunks. Asset-vector hits promote chunks that *reference*
a matching image even when the chunk text contains no matching tokens.
The asset channel is opt-in — installations without multimodal config
get the original 2-leg behavior unchanged.

**Per-kind metadata via `media_meta`.** Modality-specific fields
(image dimensions today; audio sample rate / channels and video
duration / fps tomorrow) live in a single `media_meta TEXT` column
holding JSON validated through a pydantic discriminated union
(`Annotated[ImageMediaMeta | …, Field(discriminator="kind")]`). This
keeps the `assets` table schema modality-agnostic — adding audio /
video doesn't require an `ALTER TABLE`, just a new `MediaMeta` sibling
on the DTO and a new `AssetKind` enum member. The same JSON wire
format is shared between SQLite (`TEXT`) and Postgres (`TEXT`, with a
non-breaking upgrade path to `JSONB` if/when query-side JSON ops earn
their keep).

**Scope (v1) vs. roadmap.** v1 ships images only. Audio / video
transcription with timestamp anchors, true mixed text+image chunk
encoding, image-bytes-in-prompt LLM synthesis, and content-generation
features such as automatic captions are all designed to slot into the
existing seams without re-shaping `assets` itself. `AssetKind` is an
enum with reserved values, `MediaMeta` extends through pydantic
discriminator dispatch, `chunk_asset_refs` PK is `(chunk_id, ord)`
not tied to text-only assumptions, and `LLMProvider` can grow an
`images` parameter behind a capability flag. Generated content (e.g.
LLM captions or transcripts) is **not** intended to live as columns
on `assets`; it lands in a dedicated provider-driven side table or as
K-layer wiki annotations so a re-run with a different model is a pure
write, never a schema change.

