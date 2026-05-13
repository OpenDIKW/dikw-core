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
are exposed to agents (via `GET /v1/wisdom/applicable?q=...`, PR-5)
so the agent can inject them into its own LLM prompt; dikw-core itself
no longer performs answer synthesis.

## Module map

```text
src/dikw_core/
├── api.py                 thin facade (init_wiki, ingest, synth, distill,
│                          review, retrieve, lint, status)
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
│   │   ├── page_index.py    persist_wiki_page — K-layer indexing entrypoint reused by synth + lint apply
│   │   ├── synthesize.py    LLM -> <page> blocks -> WikiPage
│   │   ├── links.py         [[wikilinks]] + md + URL parser; fuzzy resolve + collision refusal
│   │   ├── indexgen.py      regenerate wiki/index.md
│   │   ├── log.py           render wiki/log.md from wiki_log rows
│   │   ├── lint.py          broken wikilinks, orphans, duplicate titles; lint.skip frontmatter suppression
│   │   ├── lint_fix.py      Fixer Protocol + apply orchestrator (multi-op atomicity, trash redirect)
│   │   └── lint_fixers/     broken_wikilink, non_atomic_page, orphan_page (4-strategy router)
│   └── wisdom/
│       ├── distill.py       LLM -> <wisdom> blocks; enforces N>=2 evidence
│       ├── io.py            candidate files + aggregate regenerators
│       ├── review.py        approve/reject state machine
│       └── apply.py         stem-aware token overlap; surfaced to agents via /v1/wisdom/applicable (PR-5)
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
├── server/                FastAPI app + auth + sync/task/import/retrieve routes + task subsystem
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
file I/O. LLM calls only enter at synthesis and distillation — the two
engine-internal authoring legs that write the K and W layers. Answer
synthesis happens **outside** dikw-core, in the agent layer, with the
agent's own LLM and conversation context.

### Wikilink resolve, as a concrete example

`resolve_links` (in `domains/knowledge/links.py`) walks three lookup
stages, all deterministic:

1. **Exact title match** — `[[Tesla]]` against the K-layer title
   index.
2. **Fuzzy normalize** — NFKC + casefold + ASCII/CJK punctuation
   strip + ASCII trailing-plural stem (`-s`/`-es`/`-ies`). This
   catches the typing variations users hit in practice
   (`[[Neural Networks]]` to `Neural Network`, `[[Elon Musk.]]` to
   `Elon Musk`, full-width 中文 punctuation trailing the title) without
   ever calling an LLM.
3. **Collision refusal** — when normalize maps a wikilink to a key
   whose index entry holds two or more distinct paths (e.g., `Tesla`
   the company and `tesla` the SI unit both normalize to `tesla`),
   we **refuse to guess** and return the link as `UnresolvedLink`.
   `dikw lint` then surfaces the ambiguity to the user. Wrong-merge
   is irreversible; missed-resolve is a fixable lint warning — so
   we tolerate the latter to avoid the former.

Stronger fuzzy techniques (jaro-winkler, embedding similarity,
abbreviation dictionaries) are deliberately out of scope: their
false-merge risk is materially higher and the fixable broken-link
trade-off is the wrong way for K-layer pages users will edit by hand.
LLM-aware "is this candidate semantically a duplicate of an existing
page?" judgement happens upstream at synth time (see the next section)
— the resolve step itself stays deterministic.

### Synth-time existing-pages awareness, as a concrete example

`resolve_links` only sees variants of titles that actually appear in a
page body. Some duplicates never get a chance to be resolved because
the LLM, generating new pages without seeing the existing wiki, simply
**writes a fresh `<page>` block under a different title** — a true
semantic duplicate that no string-distance trick can absorb.
`_synth_pages_from_source` (in `api.py`) closes that loop by feeding
two prompt sections to every group:

1. **`## Already created in this batch`** — a per-source accumulator
   listing the `Title (type)` of every page emitted by groups
   `0..N-1` of the SAME source. Stage A 1:N fan-out runs groups
   serially; without this, group 2 reinvents what group 1 wrote.
2. **`## Existing wiki pages`** — a snapshot of the base K-layer.
   Below `synth.existing_pages_max_bytes` (default 16384 B ≈ 500
   pages × ~25 B/line) we render the full list. Above that the
   prompt would balloon as the wiki grows, so we switch to a
   `vec_search`-gated top-K driven by the group's own chunk
   embeddings — the per-chunk embeddings already exist from ingest,
   so the only new Storage primitive is `get_chunk_embeddings`
   (a pure SELECT over the existing per-version vec table). Top-K
   defaults to `synth.existing_pages_top_k = 50`.

The S2 prompt strategy: strong instruction + zero-block escape hatch.
On a detected duplicate the LLM is told to emit **zero `<page>` blocks
for that candidate** and reference the existing page via `[[Title]]`
in its other pages instead. The "zero blocks" path is the only clean
way the LLM can comply without partial-output ambiguity.

Why per-chunk vec_search → union → top-K (rather than re-embed the
group text once)? The locked design keeps the original "per-chunk
vec_search → union dedup → score sort" semantics so retrieval
faithfully reflects each chunk's local topic. Re-embedding would have
collapsed a multi-topic group into one query vector and missed pages
relevant to chunks the LLM hadn't focused on yet. The
`get_chunk_embeddings` SELECT is cheap enough that this faithfulness
costs nothing measurable.

Both new fields (`existing_pages_max_bytes`, `existing_pages_top_k`)
live on `SynthConfig` so a base-level `dikw.yml` can tune them per
deployment — a wiki targeting tiny local models can drop the byte
threshold; a wiki targeting Claude Opus's full context window can
raise it.
