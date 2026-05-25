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
| **W** — Wisdom       | hand-written markdown under `wisdom/<author>/`   | human (Obsidian)               |

The W layer is being refactored across 0.3.0 (this is PR2 of 4). PR1
removed the prior LLM-distilled candidate/review pipeline. PR2 (this
commit) wires `dikw ingest` to scan `<root>/wisdom/**/*.md` through the
same `persist_page` pipeline as wiki pages, indexes them as
`Layer.WISDOM` documents with chunks/embeddings/links/provenance, and
adds the wisdom-only `documents.status` column (CHECK-constrained to
`draft | published | favorite | archived`) validated by the new
`invalid_wisdom_status` lint kind. PR3 surfaces wisdom hits on
`dikw client retrieve` with `Hit.layer == "wisdom"` and extends
`broken_wikilink` / `missing_provenance` / `orphan_page` lint coverage
to the wisdom layer. dikw-core does not perform answer synthesis —
`retrieve` returns ranked chunks + page refs and the agent layer runs
its own LLM.

## Module map

```text
src/dikw_core/
├── api.py                 thin facade — init_wiki, ingest, retrieve,
│                          synthesize, lint (+ lint_propose / lint_apply),
│                          list_pages, read_page, list_links, read_provenance,
│                          list_graph, read_asset, status, health,
│                          check_providers
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
│   │   ├── page_index.py    persist_page(layer=...) — K + W layer indexing entrypoint reused by synth, ingest, lint apply
│   │   ├── synthesize.py    LLM -> <page> blocks -> WikiPage
│   │   ├── links.py         [[wikilinks]] + md + URL parser; fuzzy resolve + collision refusal
│   │   ├── indexgen.py      regenerate wiki/index.md
│   │   ├── log.py           render wiki/log.md from wiki_log rows
│   │   ├── lint.py          broken wikilinks, orphans, duplicate titles, missing_provenance; lint.skip frontmatter suppression
│   │   ├── lint_fix.py      Fixer Protocol + apply orchestrator (multi-op atomicity, trash redirect, reconcile_provenance op)
│   │   └── lint_fixers/     broken_wikilink, non_atomic_page, orphan_page (4-strategy router), missing_provenance (deterministic)
│   └── wisdom/
│       └── page.py          author_from_path — wisdom/<author>/<slug>.md attribution
├── providers/
│   ├── base.py              LLMProvider + EmbeddingProvider + MultimodalEmbeddingProvider Protocols
│   ├── anthropic_compat.py  anthropic SDK, system-prompt cache_control; retargets via llm_base_url
│   ├── openai_compat.py     openai SDK; any base_url (OpenAI, Azure, Ollama, DeepSeek, GLM, Gemini-compat, …)
│   ├── openai_codex.py      ChatGPT-only GPT-5 family via dikw-managed OAuth at <base>/.dikw/auth.json
│   ├── codex_auth.py        device-code + import + refresh for the codex OAuth store
│   ├── gitee_multimodal.py  Gitee AI multimodal-embedding HTTP client (text + image inputs)
│   └── _http.py             shared httpx pool helpers
├── storage/
│   ├── base.py              Storage Protocol (engine depends only on this)
│   ├── sqlite.py            SQLite + sqlite-vec + FTS5 (default)
│   ├── postgres.py          Postgres + pgvector + tsvector (optional extra)
│   └── migrations/
│       ├── sqlite/          schema SQL
│       └── postgres/        schema SQL (vector extension)
├── prompts/               versioned LLM prompts loaded via importlib.resources
├── server/                FastAPI app + auth + sync/task/import/retrieve/pages/assets/graph routes + task subsystem
├── client/                Remote Typer CLI + httpx transport + NDJSON progress + sources importer + converter dispatch
├── auth_cli.py            local `dikw auth {login,import,status,list,logout}` for the per-base OAuth store
├── logging.py             init_logging() — DIKW_LOG_LEVEL + httpx/httpcore/urllib3 clamp
├── md_inspect.py          standalone markdown preflight (frontmatter + image-ref extraction)
└── cli.py                 top-level Typer app: version, init, serve, auth subgroup, dikw client subgroup
                           (HTTP-bound commands live exclusively under `dikw client <verb>` — there
                           are no top-level short aliases)
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

**Policy: alpha "rebuild on incompatibility".** Each adapter ships a
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
alpha (`CLAUDE.md` warns "APIs, on-disk formats, database schema, and
CLI will change"); a real migration framework lands when we declare
beta. Schema history lives in `git log` on `migrations/`, not in a
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
lookup, link traversal, provenance lookup) is deterministic SQL + file
I/O. LLM calls only enter at synth — the engine-internal authoring leg
that writes the K layer. The W layer is hand-authored markdown the user
writes in Obsidian; ingest/persistence indexes it (wisdom-as-documents
lands in 0.3.0 PR2 — see CHANGELOG). Answer synthesis happens **outside**
dikw-core, in the agent layer, with the agent's own LLM and conversation
context.

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
   `dikw client lint` then surfaces the ambiguity to the user. Wrong-merge
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

## Asset exposure to remote clients

Markdown source pages routinely embed images (`![alt](./figs/x.png)`
or Obsidian `![[assets/foo.jpg]]`). Ingest already materialises each
image into `<base>/assets/<h2>/<h8>-<stem>.<ext>` (content-addressed by
SHA-256) and writes both an `AssetRecord` and the chunk → asset bridge
rows (`chunk_asset_refs`). The server makes those bytes reachable from
a remote process via two pieces, designed to be the **minimum surface
that keeps the wiki tree itself unchanged**:

1. **`PageReadResult.assets[]` on `GET /v1/base/pages/{path}`** —
   page-level asset list, deduped by `asset_id`, ordered by first
   appearance (chunk seq → ref ord). Each entry carries
   `original_paths` (what the user typed in markdown), `mime`, `bytes`,
   `media_meta`, and `url` (always `/v1/assets/{asset_id}` — fixed
   shape so the client zero-parses). The page `body` itself is
   returned **verbatim** — no in-place URL rewriting — because the
   `wiki/` tree is the product (Obsidian-compatible, user-owned),
   and rewriting would diverge from the on-disk file.

2. **`GET /v1/assets/{asset_id}`** — streams the raw bytes with the
   stored `mime` as `Content-Type`. Because content is addressed by
   SHA-256, the response is naturally immutable and the route emits
   `ETag: "<asset_id>"` + `Cache-Control: public, max-age=31536000,
   immutable` so well-behaved clients (browsers, CDN fronts, future
   client cache) can revalidate with a single `If-None-Match`.

Failure surface is uniform 404: unknown id, malformed id (anything
that isn't 64 lower-case hex), file vanished, or a row whose
`stored_path` would resolve outside `<base>/<assets.dir>` all produce
the same `{"error": {"code": "asset_not_found", ...}}` envelope. The
last case is a defence-in-depth guard against DB corruption /
migration drift — without it a tampered row could point the route at
an arbitrary file on disk. Distinguishing the four causes on the wire
would let an attacker probe which ids exist, so we deliberately don't.

`Storage` Protocol is unchanged — the engine reuses `get_asset`,
`get_assets`, `chunk_asset_refs_for_chunks`, and `list_chunks` that
the ingest + retrieve channels already depend on. No new adapter
behaviour, no new contract-suite cases.

## Base graph exposure to remote clients

`dikw-web`'s Knowledge Graph view used to loop
`GET /v1/base/pages/{path}` for every page and re-do wikilink
resolution in the browser. That works for a 50-page demo wiki but
breaks on large bases (N HTTP requests, single hang stalls the whole
view) and silently drifts from the engine's K-layer link semantics.
Issue #89 moves graph construction into the engine:

1. **`api.list_graph` (engine)** — one read-only pass over all docs:
   read body via `parse_any` (front-matter stripped, in `asyncio.to_thread`
   so disk I/O doesn't stall the loop), parse via
   `domains/knowledge/links.parse_links`, resolve in-line with the same
   three-stage `exact title → fuzzy normalize → collision-refuse` rules
   `resolve_links` uses, aggregate edges with `weight` and unresolved
   counts. No new storage primitives — `Storage.list_documents` is the
   only DB call. Resolution universe equals the response's node set, so
   in default `active=True` mode a wikilink to a deactivated page
   surfaces as unresolved (matches the user-visible "this page is
   hidden, treat the link as broken" expectation).

2. **`GET /v1/base/graph` (`server/routes_graph.py`)** — thin handler
   over `api.list_graph`; query is just `active` (mirrors
   `/v1/base/pages`). `response_model=GraphResult` keeps the wire shape
   pinned in pydantic so client codegen doesn't drift.

3. **`dikw client graph get` (`client/cli_app.py`)** — agent-first JSON
   to stdout (no `--format table` — graph data isn't tabular); pipes
   straight into `jq`.

`base_revision` is `sha256(sorted (path, mtime, active) triples)` —
content-addressed and cheap (microseconds), so a client can
short-circuit unchanged graphs by comparing the revision string before
re-rendering. Not a cryptographic commitment to body content; clients
that need that should hash the response themselves. v1 deliberately
omits ghost nodes for unresolved targets, the `layer` query knob, and
`anchor_count` / `suggestions` per-node fields — kept for follow-up
once `dikw-web` has exercised the v1 surface.
