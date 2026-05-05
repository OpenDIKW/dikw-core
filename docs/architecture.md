# Architecture

`dikw-core` is structured around two ideas, both fighting for air in most
RAG stacks today:

1. Knowledge should be a **compounding artifact**, not a query-time search
   result. The wiki you can read in Obsidian is the product; the engine is
   the scribe. (This is Karpathy's LLM-wiki framing.)
2. **All four DIKW layers deserve first-class treatment.** Data, Information,
   Knowledge, and Wisdom each have their own storage, schemas, and
   operations. The pipeline between them is explicit.

Everything else is plumbing.

## The four layers

| Layer                | What lives here                                  | Writer                         |
| -------------------- | ------------------------------------------------ | ------------------------------ |
| **D** — Data         | raw source files (markdown)                      | human                          |
| **I** — Information  | parsed, chunked, FTS-indexed, embedded           | engine (deterministic)         |
| **K** — Knowledge    | LLM-authored wiki pages, link graph, `index.md`  | LLM, human-editable            |
| **W** — Wisdom       | principles / lessons / patterns with evidence    | LLM proposes, human approves   |

The W layer is the differentiator. Every item must cite **≥ 2 pieces of
evidence** drawn from the K or D layers, and every state change passes
through the review flow (`dikw review approve|reject`). Approved items
surface automatically at query time as "operating principles".

## Module map

```text
src/dikw_core/
├── api.py                 thin facade (init_wiki, ingest, synth, distill,
│                          review, query, lint, status)
├── config.py              Pydantic config + YAML loader
├── schemas.py             cross-layer DTOs
├── domains/                 DIKW domain model (the four layers)
│   ├── data/
│   │   ├── sources.py       source-file scanner (glob + ignore)
│   │   ├── assets.py        image/asset materialization (sha-streamed)
│   │   ├── hashing.py       streaming + in-memory SHA-256 helpers
│   │   └── backends/        registry-dispatched parsers
│   │       ├── base.py      SourceBackend Protocol + registry
│   │       └── markdown.py  .md / .markdown
│   ├── info/
│   │   ├── chunk.py         heading-aware paragraph chunker
│   │   ├── tokenize.py      CJK-aware preprocessing + token counting
│   │   ├── embed.py         batched embedding worker
│   │   └── search.py        RRF-fused FTS + vector hybrid
│   ├── knowledge/
│   │   ├── wiki.py          WikiPage I/O (Obsidian-compatible front-matter)
│   │   ├── synthesize.py    LLM -> <page> blocks -> WikiPage
│   │   ├── links.py         [[wikilinks]] + md + URL parser
│   │   ├── indexgen.py      regenerate wiki/index.md
│   │   ├── log.py           render wiki/log.md from wiki_log rows
│   │   └── lint.py          broken wikilinks, orphans, duplicate titles
│   └── wisdom/
│       ├── distill.py       LLM -> <wisdom> blocks; enforces N>=2 evidence
│       ├── io.py            candidate files + aggregate regenerators
│       ├── review.py        approve/reject state machine
│       └── apply.py         stem-aware token overlap at query time
├── providers/
│   ├── base.py            LLMProvider + EmbeddingProvider Protocols
│   ├── anthropic.py       anthropic SDK, system-prompt cache_control
│   └── openai_compat.py   openai SDK; any base_url
├── storage/
│   ├── base.py              Storage Protocol (engine depends only on this)
│   ├── sqlite.py            SQLite + sqlite-vec + FTS5 (default)
│   ├── postgres.py          Postgres + pgvector + tsvector (optional extra)
│   └── migrations/
│       ├── sqlite/          schema SQL
│       └── postgres/        schema SQL (vector extension)
├── prompts/               versioned LLM prompts loaded via importlib.resources
├── server/                FastAPI app + auth + sync/task/upload/query routes + task subsystem
├── client/                Remote Typer CLI + httpx transport + NDJSON progress
└── cli.py                 top-level Typer app: version, init, serve + dikw client subgroup
```

## Seams on purpose

Three extension points are sharper than they look, because the rest of the
engine depends only on their Protocol / abstract interface:

1. **`SourceBackend`** — adding a format (PDF, Quarto, `.ipynb`) means
   writing one class and calling `register()`. No other change.
2. **`Storage`** — every I/O crosses typed Pydantic DTOs. SQLite
   (default) and Postgres (`[postgres]` extra) both slot in via
   `storage/__init__.py`'s factory without touching engine code; new
   adapters land the same way.
3. **`LLMProvider` / `EmbeddingProvider`** — Anthropic and any
   OpenAI-compatible endpoint are wired today; llama-cpp-python for local
   inference is a drop-in.

## Storage schema

**Policy: pre-alpha "rebuild on incompatibility".** Each adapter ships a
single `schema.sql` under `storage/migrations/{sqlite,postgres}/` that
represents the desired shape. `migrate()` applies the file verbatim to
a fresh DB and writes the code's `SCHEMA_VERSION` constant
(`storage/_schema.py`) into `meta_kv['schema_version']`. On a subsequent
connect:

* fingerprint matches → no-op,
* fingerprint missing → fresh-DB branch (apply schema.sql, record version),
* fingerprint differs → loud `StorageError` telling the user to
  delete the storage directory and re-ingest.

There is **no in-place upgrade path** — bumping `SCHEMA_VERSION` in code
invalidates every existing DB at the next connect. This is fit for
pre-alpha (`CLAUDE.md` warns "APIs and on-disk formats will change");
when we declare alpha we'll introduce a real migration framework.
Schema history lives in `git log` on `migrations/`, not in a
deprecated-tables inventory.

The runtime-created vector tables (`vec_chunks_v<id>` / `vec_assets_v<id>`)
are intentionally NOT in `schema.sql` — sqlite-vec / pgvector both need
the embedding dim parameterised into the CREATE statement, so each
`embed_versions` row materialises its own dim-locked vec table on first
upsert. `SQLiteStorage._verify_vec_tables_use_cosine` is a defensive
runtime invariant check (not a legacy migration) that refuses to open
a DB whose vec0 tables predate the cosine-distance fix.

### Cross-adapter shape: where the two SQL backends intentionally differ

The SQLite and Postgres tables match column-for-column with the
expected dialect aliases (`INTEGER`/`BIGINT`, `REAL`/`DOUBLE PRECISION`,
`BLOB`/`BYTEA`, `INTEGER 0/1`/`BOOLEAN`). One table is shaped
differently **on purpose** because the two engines implement FTS via
different mechanisms:

| Where text indexing lives | SQLite | Postgres |
|---|---|---|
| Search index | separate FTS5 virtual table `documents_fts` (body-only; `rowid` aligns with `chunks.chunk_id`) | plain `chunks.fts tsvector NOT NULL` column populated by the Python adapter via `to_tsvector('simple', preprocess_for_fts(text, tokenizer=cjk_tokenizer))`, indexed by `GIN` |

Both adapters expose the same `fts_search` method on the `Storage`
Protocol returning the same `FTSHit` DTOs — the engine never sees the
divergence. Schema-parity diff tools should treat `chunks.fts` (PG-only)
and `documents_fts` (SQLite-only) as the dual implementations of the
same logical capability. Their **column scope** is identical: both
index only chunk body text.

**Tokenization** is also aligned: SQLite uses `unicode61
remove_diacritics 0` (the `0` is explicit because the unicode61
default is `1`, which still strips diacritics) so `café` and `cafe`
are different tokens — same byte-level behavior as PG's
`to_tsvector('simple', text)`. CJK input flows through
`info.tokenize.preprocess_for_fts` (jieba when
`cjk_tokenizer="jieba"`) on both adapters: SQLite inserts the
segmented body into `documents_fts`; PG feeds the same segmented
string through `to_tsvector('simple', …)` into a plain `chunks.fts
tsvector NOT NULL` column populated by the Python adapter on INSERT.
A `GENERATED ALWAYS AS to_tsvector('simple', text)` column would
bypass Python's jieba and silently regress CJK BM25 — see
`storage/migrations/postgres/schema.sql` for the rationale.

The PG `fts_search` consumes the `info/search.py:_sanitize_fts`
output via `to_tsquery` (with a small `_fts_to_tsquery_string`
adapter that translates SQLite's `'"foo" OR "bar"'` form into PG's
`'foo | bar'`). Earlier versions used `plainto_tsquery`, which
re-tokenized the sanitizer output and treated `OR` as a literal
search word — broken for any multi-word query.

Both adapters now apply the `documents.active = TRUE` filter inside
`fts_search` so soft-deleted documents never surface in BM25 hits.
Pre-PR the SQLite adapter skipped this filter (PG always applied
it); the post-PR JOIN on `documents` makes the alignment cheap and
removes a silent recall divergence on inactive docs.

`SQLiteStorage` also reports `notnull=0` on every `INTEGER`/`TEXT`
PRIMARY KEY column via `PRAGMA table_info` (a documented SQLite quirk
— PK columns implicitly enforce NOT NULL). Postgres `information_schema`
reports `notnull=1` for the same columns, which is the actual contract
both adapters honor at write time.

## What stays out of the adapters

Hybrid search fusion (RRF), chunking, link-graph parsing, wisdom scoring,
and prompt templating all live **outside** the storage and provider
adapters. The Storage Protocol exposes only the primitives (`fts_search`,
`vec_search`, …); fusion happens in `info/search.py`. This is the right
abstraction height: high enough to hide SQL dialects, low enough that
each adapter doesn't re-implement RRF.

## Karpathy's rule, applied

> "Scoping should be deterministic, reasoning should be probabilistic."

We take that seriously. Every navigation step (source listing, chunk
lookup, link traversal, wisdom retrieval-by-title) is deterministic SQL +
file I/O. LLM calls only enter at synthesis, distillation, and the
natural-language answering step of `query`.
