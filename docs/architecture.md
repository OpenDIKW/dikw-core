# Architecture

`dikw-core` is structured around two ideas, both fighting for air in most
RAG stacks today:

1. Knowledge should be a **compounding artifact**, not a query-time search
   result. The open Markdown knowledge tree is the product; the engine owns,
   authors, and reconciles it. (This is Karpathy's LLM-wiki framing.)
2. **All four DIKW layers deserve first-class treatment.** Data, Information,
   Knowledge, and Wisdom each have their own storage, schemas, and
   operations. The pipeline between them is explicit.

Everything else is plumbing.

## The four layers

| Layer                | What lives here                                  | Writer                         |
| -------------------- | ------------------------------------------------ | ------------------------------ |
| **D** — Data         | raw source files (markdown)                      | human                          |
| **I** — Information  | parsed, chunked, FTS-indexed, embedded           | engine (deterministic)         |
| **K** — Knowledge    | LLM-authored knowledge pages (filed under a configurable `category` tree), link graph  | LLM, human-editable            |
| **W** — Wisdom       | hand-written markdown under `wisdom/<author>/`   | human                          |

The W layer is hand-written first-class documents under
`wisdom/<author>/<slug>.md`. Since 0.4.0, wisdom is indexed
**exclusively through the dedicated write surface**
(`api.write_wisdom_page` / `POST /v1/base/wisdom` /
`dikw client wisdom write`) — `dikw client ingest` no longer scans
`<base>/wisdom/`. The write surface runs the same
`persist_wisdom` pipeline as the previous ingest loop did (chunks
/ embeddings / `[[wikilinks]]` / `provenance` edges), so the on-disk
schema and storage row shape are unchanged.
`Layer.WISDOM` documents carry a wisdom-only `documents.status`
column (CHECK-constrained to `draft | published | favorite |
archived`) validated by the `invalid_wisdom_status` lint kind.
`dikw client retrieve` returns wisdom chunks tagged
`Hit.layer == "wisdom"`; `read_page` / `list_links` /
`read_provenance` accept wisdom paths and resolve cross-layer
edges; `broken_wikilink` / `orphan_page` / `missing_provenance`
lint scans both K + W layers, crediting cross-layer wikilinks in
the orphan inbound counter so a knowledge page cited only from
wisdom is not falsely flagged. Hand-edits to wisdom files on
disk are not auto-reindexed *live*, but the `stale_index`
drift lint (ADR-0005) detects them — `dikw client lint apply`
re-projects the edited bytes (or re-run `dikw client wisdom
write`). See
`docs/adr/0002-wisdom-as-first-class-documents.md` for the
rationale. dikw-core does not perform answer synthesis —
`retrieve` returns ranked chunks + page refs and the agent layer
runs its own LLM.

## Module map

```text
src/dikw_core/
├── api.py                 thin re-export facade — surfaces every verb so
│                          the public `api.X` surface (+ `__all__`) stays
│                          byte-stable; defines nothing. Each verb lives in
│                          a focused `api_*` cluster module below.
├── api_core.py            base scaffold (init_base / load_base / status),
│                          storage open+migrate (`_with_storage`), embed-version helpers
├── api_types.py           cross-cutting DTOs + exceptions (IngestReport,
│                          SynthReport, HealthReport, PageNotFound, …)
├── api_health.py          health, check_providers + provider probes
├── api_ingest.py          ingest (D-layer write entry)
├── api_pages.py           list_pages, read_page, read_asset
├── api_graph.py           list_links, read_provenance, list_graph
├── api_retrieve.py        retrieve (RRF-fused hybrid search; no LLM)
├── api_synth.py           synthesize (K-layer authoring leg — only LLM entry)
├── api_lint.py            lint, lint_propose, lint_apply
├── api_wisdom.py          write_wisdom_page (W-layer write entry)
├── api_delete.py          delete_page (immediate D/K/W soft-delete to trash/)
├── api_path_safety.py     `_assert_within` base-escape guard
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
│   ├── data/
│   │   └── persist.py       persist_source — D-layer write entry (doc + chunk + FTS + chunk_asset_refs)
│   ├── knowledge/
│   │   ├── page.py          KnowledgePage I/O (YAML front-matter + wikilinks)
│   │   ├── page_index.py    persist_knowledge — K-layer write entry (synth + lint apply); also defines the private _persist_layered_page shared with W
│   │   ├── synthesize.py    LLM -> <page> blocks -> KnowledgePage
│   │   ├── links.py         [[wikilinks]] + md + URL parser; fuzzy resolve + collision refusal
│   │   ├── lint.py          broken wikilinks, orphans, duplicate titles, non-atomic pages, missing_provenance, invalid_wisdom_status, uncategorized, title_slug_quality, missing_file (D/K/W orphaned-row drift), stale_index + untracked_file (K/W disk↔index drift), dangling_provenance (K/W cited-source-gone drift, read-only/no fixer); lint.skip frontmatter suppression
│   │   ├── lint_fix.py      Fixer Protocol + apply orchestrator (multi-op atomicity, trash redirect, reconcile_provenance + purge_document + reindex_page ops)
│   │   └── lint_fixers/     broken_wikilink, non_atomic_page, orphan_page (4-strategy router), missing_provenance (deterministic), missing_file (deterministic — purge orphaned row, D/K/W), reindex (deterministic — re-project on-disk bytes for stale_index + untracked_file, K/W)
│   ├── wisdom/
│   │   ├── page.py          author_from_path — wisdom/<author>/<slug>.md attribution
│   │   └── persist.py       persist_wisdom — W-layer write entry (sole engine caller: api.write_wisdom_page)
│   └── trash.py            move_to_trash — cross-layer soft-delete primitive (lint delete_page fixer + the delete verb)
├── providers/
│   ├── base.py              LLMProvider + EmbeddingProvider + MultimodalEmbeddingProvider Protocols
│   ├── anthropic_compat.py  anthropic SDK, system-prompt cache_control; retargets via llm_base_url
│   ├── openai_compat.py     openai SDK; any base_url (OpenAI, Azure, Ollama, DeepSeek, GLM, Gemini-compat, …)
│   ├── openai_codex.py      ChatGPT-only codex model family (gpt-5.5, …) via dikw-managed OAuth at <base>/.dikw/auth.json
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
├── prompts/               versioned LLM prompts (importlib.resources); `resolve()` honours per-base
│                          overrides (`synth.prompt_path` / `lint.fixer_prompts`) validated against `_contract.py`
├── server/                FastAPI app + auth + sync/task/import/retrieve/pages/assets/graph routes + task subsystem
├── client/                Remote Typer CLI + httpx transport + NDJSON progress + sources importer + converter dispatch
├── auth_cli.py            local `dikw auth {login,import,status,list,logout}` for the per-base OAuth store
├── logging.py             init_logging() — DIKW_LOG_LEVEL + DIKW_LOG_FORMAT (text/json) + httpx/httpcore/urllib3 clamp
├── telemetry.py           OTel seam (optional [otel] extra) — accessors + dikw.*/gen_ai.* keys + span/metric helpers + entry-only SDK bootstrap; imports only opentelemetry+stdlib (never server). See docs/observability.md
├── md_inspect.py          standalone markdown preflight (frontmatter + image-ref extraction)
└── cli.py                 top-level Typer app: version, init, serve, auth subgroup, client subgroup
                           (HTTP-bound commands live exclusively under `dikw client <verb>` — there
                           are no top-level short aliases)
```

## Chunk → FTS → embed pipeline per layer

Each DIKW layer has exactly one programmatic write entry that owns
its full `upsert_document` + `chunk_markdown` + `replace_chunks`
(FTS side-effect, same transaction) + optional inline embed +
`replace_links_from` + `replace_provenance_from` pipeline. Public
callers fan into these entries; no other engine code mutates K / W
documents and their derived rows.

| Layer | Write entry | Trigger surface | Embed timing |
|---|---|---|---|
| **D** (source) | `persist_source` (`domains/data/persist.py`) | `api.ingest` (one call per scanned source file) | **Deferred** — `api.ingest` accumulates chunks across files and runs one bulk embed at end-of-scan for throughput |
| **K** (knowledge) | `persist_knowledge` (`domains/knowledge/page_index.py`) | `api.synthesize` (synth) / `api.lint_apply` | **Inline** — synth always wires an embedder; lint apply wires one when the configured `provider.embedding_api_key_env` var is set, otherwise defers |
| **W** (wisdom) | `persist_wisdom` (`domains/wisdom/persist.py`) | `api.write_wisdom_page` (CLI `dikw client wisdom write` / HTTP `POST /v1/base/wisdom`) | **Inline** unless the caller passes `no_embed=True` |

**Cross-layer resume scan.** At the end of every `dikw client ingest`,
the engine runs `storage.list_chunks_missing_embedding(version_id)`
and embeds any chunks whose vector is absent — across D / K / W
layers, irrespective of which path created them. This is the
**eventual-consistency contract** of the embedding leg: every chunk
eventually has a vector once an embedder is reachable. It backstops
three legitimate "no vector yet" paths: ingest crash recovery
(D-layer chunks written before the bulk embed ran), lint apply
without the configured `provider.embedding_api_key_env` var set (K-layer
chunks deferred), and `write_wisdom_page --no-embed` (W-layer chunks deferred).

**Per-batch retry-skip.** Inside every persist path the embed leg
runs through `consume_embedding_stream`, which catches
`TransientProviderError` per batch (5xx, 408/429, timeout, connect
drop, parse failure), retries up to `cfg.provider.embedding_error_retries`
(default 2) with `embedding_error_retry_backoff_seconds` (default
2.0s) linear backoff before skipping the batch and continuing. A bare
(permanent) `ProviderError` — 401/403/404, missing key, invalid model
id — is **not** caught here: it propagates so misconfig fails fast
instead of being silently retried-then-skipped. Skipped chunks remain
in storage without vectors and the next ingest's resume scan picks
them up. The persist function's return shape surfaces
`chunks_embedded` and `chunks_pending_embedding` so the caller can
warn the user (CLI `lint apply` prints both).

**Persist failure → deactivate (`documents.active` as the commit
marker).** A persist pipeline is five separately-committed Storage
calls with no enclosing transaction, so a hard exception mid-pipeline
(a permanent `ProviderError` from inline embed, or `replace_chunks` /
`replace_links_from` / `replace_provenance_from` raising) would leave
the `active=True` document row + chunks committed while later steps
never ran — a half-written page that still surfaces in retrieval. The
invariant that closes this: **a document is `active=True` only if its
full pipeline completed; on any hard exception the caller
`deactivate_document(doc_id)`s it** (`active=False`), which hides it
from every retrieval leg (`fts_search` / `vec_search` /
`neighbor_chunks_via_links`) and from `read_page` / `list_pages`. All
four write entries enforce this at the call site: D in `api.ingest`'s
`storage_error` arm, W in `write_wisdom_page`, and K in **both** synth's
per-page loop and `lint_apply`'s Phase 1 loop (K was the last to gain
it). The same call sites also catch `asyncio.CancelledError` (a
`BaseException` a bare `except Exception` would miss) — they deactivate
the in-flight doc and re-raise, so a mid-persist cancellation can't strand
a half-written `active=True` row either. A transient embed skip is *not* a failure — it leaves the page
`active=True` with `chunks_pending_embedding > 0` for the resume scan.
Recovery is layer-specific: D re-activates on the next ingest (it
re-scans `sources/` and the early-skip arm falls through for
`active=False` rows); K and W now have a scan-based reindex too — a
deactivated page's file is still on disk with no *active* row, so the
next default `lint` flags it `untracked_file` (or `stale_index` if a
stale active row lingers) and `lint apply` re-projects the on-disk bytes,
re-activating it without re-running synth (ADR-0005). For a synth page,
synth *additionally* writes a `synth_source_failed` knowledge_log marker
that invalidates any prior `synth_source_done` for that source (done/failed
markers apply in `ts ASC, id ASC` log order, last-writer-wins — so even a
`synth --all` re-synth that fails doesn't strand the page), letting the
next default `synth` re-process and rebuild it from the D-source. Synth
surfaces failed pages as `SynthReport.persist_errors`; lint apply as
`ApplyReport.persist_errors`.

**FTS is always synchronous.** `replace_chunks` writes the FTS index
inside the same storage transaction as the `chunks` rows — there is
no "FTS deferred" code path. SQLite uses the `documents_fts` virtual
table (rowid aligns with `chunks.chunk_id`); Postgres populates
`chunks.fts tsvector` via the Python adapter at INSERT. CJK
preprocessing (jieba) happens before the SQL call on both adapters.

**Reconciling hand-edits.** Hand-edits to K or W files on disk, and
hand-written new pages dropped under `knowledge/`/`wisdom/`, are reconciled
by the `stale_index` / `untracked_file` drift lint kinds (ADR-0005): default
`lint` detects them and `dikw client lint propose --rule stale_index` (or
`untracked_file`) → `lint apply` re-projects the on-disk bytes into storage
(re-chunk / re-link / re-provenance / inline-or-deferred embed) **without
rewriting the file or re-running synth** — so the hand-edit is preserved, not
regenerated from the D-source. The engine still doesn't *watch* the
filesystem (and `ingest` still only scans `<base>/sources/`), so
reconciliation happens on the next `lint` run, not live. This supersedes the
never-built `dikw client reindex <path>`.

## Seams on purpose

Four extension points are sharper than they look, because the rest of the
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
4. **`telemetry` accessors** — engine code emits spans/metrics through
   `get_tracer` / `get_meter` + the `gen_ai_span` / `op_span` / `record_*`
   helpers, never `import opentelemetry` directly, so the whole OTel stack is
   an optional `[otel]` extra (hand-rolled no-ops when absent) and **only the
   process entry** (server lifespan / client CLI root) wires the SDK — exactly
   how `init_logging` is wired from entry points, not engine code. A background
   task's root span is **linked** back to the submitting request rather than
   parented, the OTel idiom for fire-and-forget work that outlives its trigger.
   See [`observability.md`](observability.md).

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

Hybrid search fusion (RRF), chunking, link-graph parsing, and prompt
templating all live **outside** the storage and provider adapters. The Storage Protocol exposes only the primitives (`fts_search`,
`vec_search`, …); fusion happens in `info/search.py`. This is the right
abstraction height: high enough to hide SQL dialects, low enough that
each adapter doesn't re-implement RRF.

## Karpathy's rule, applied

> "Scoping should be deterministic, reasoning should be probabilistic."

We take that seriously. Every navigation step (source listing, chunk
lookup, link traversal, provenance lookup) is deterministic SQL + file
I/O. LLM calls only enter at synth — the engine-internal authoring leg
that writes the K layer. The W layer is hand-authored markdown the user
writes through `dikw client wisdom write` (or its HTTP / Python
equivalents); the write surface drives the same `persist_wisdom`
pipeline that the K layer uses via `persist_knowledge`, so chunks /
embeddings / `[[wikilinks]]` / `provenance` edges land symmetrically.
Answer synthesis happens **outside** dikw-core, in the agent layer,
with the agent's own LLM and conversation context.

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
the LLM, generating new pages without seeing the existing knowledge base, simply
**writes a fresh `<page>` block under a different title** — a true
semantic duplicate that no string-distance trick can absorb.
`_synth_pages_from_source` (in `api_synth.py`) closes that loop by feeding
two prompt sections to every group (rendered as H3 sub-sections nested
under the template's `## Knowledge-base context` heading):

1. **`### Already created in this batch`** — a per-source accumulator
   listing the `Title [slug] (category)` of every page emitted by groups
   `0..N-1` of the SAME source. Stage A 1:N fan-out runs groups
   serially; without this, group 2 reinvents what group 1 wrote.
2. **`### Existing knowledge pages`** — a snapshot of the base K-layer.
   Below `synth.existing_pages_max_bytes` (default 16384 B ≈ 500
   pages × ~25 B/line) we render the full list. Above that the
   prompt would balloon as the knowledge base grows, so we switch to a
   `vec_search`-gated top-K driven by the group's own chunk
   embeddings — the per-chunk embeddings already exist from ingest,
   so the only new Storage primitive is `get_chunk_embeddings`
   (a pure SELECT over the existing per-version vec table). Top-K
   defaults to `synth.existing_pages_top_k = 50`.

Both render each page as `- Title [slug] (category)`: the kebab-case
file stem disambiguates two same-titled pages, while the prompt still
tells the model to write `[[Title]]` (never the slug) when linking.

A third section, **`### Priority targets (create if relevant)`**, is
prepended for every group after the first. It lists the top-5 wikilink
targets earlier groups of the SAME source referenced but that resolve
to no page yet — checked against the snapshot **and** the batch with
the same exact → fuzzy → collision rules as `resolve_links`, re-resolved
each group so a target a prior group has since authored drops off, and
ranked by how many distinct pages want it. This nudges a later group
whose content covers one to author it at the right title instead of
stranding a broken `[[wikilink]]`. It is pure deterministic scoping over
data the loop already has (parsed wikilinks + the title index) — no
extra LLM call, no new Storage primitive.

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
deployment — a base targeting tiny local models can drop the byte
threshold; a base targeting Claude Opus's full context window can
raise it.

## Asset exposure to remote clients

Markdown source pages routinely embed images (`![alt](./figs/x.png)`
or the wiki-style `![[assets/foo.jpg]]` embed). Ingest already materialises each
image into `<base>/assets/<h2>/<h8>-<stem>.<ext>` (content-addressed by
SHA-256) and writes both an `AssetRecord` and the chunk → asset bridge
rows (`chunk_asset_refs`). The server makes those bytes reachable from
a remote process via two pieces, designed to be the **minimum surface
that keeps the knowledge tree itself unchanged**:

1. **`PageReadResult.assets[]` on `GET /v1/base/pages/{path}`** —
   page-level asset list, deduped by `asset_id`, ordered by first
   appearance (chunk seq → ref ord). Each entry carries
   `original_paths` (what the user typed in markdown), `mime`, `bytes`,
   `media_meta`, and `url` (always `/v1/assets/{asset_id}` — fixed
   shape so the client zero-parses). The page `body` itself is
   returned **verbatim** — no in-place URL rewriting — because the
   `knowledge/` tree is the product (open Markdown, user-owned),
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
resolution in the browser. That works for a 50-page demo base but
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

`base_revision` is a `sha256` over the path-sorted docs, mixing each
doc's `path`, `title`, `layer`, `mtime`, current on-disk body hash (or
a missing-body sentinel), and `active` — content-addressed and cheap
(microseconds), so a client can
short-circuit unchanged graphs by comparing the revision string before
re-rendering. Not a cryptographic commitment to body content; clients
that need that should hash the response themselves. v1 deliberately
omits ghost nodes for unresolved targets, the `layer` query knob, and
`anchor_count` / `suggestions` per-node fields — kept for follow-up
once `dikw-web` has exercised the v1 surface.
