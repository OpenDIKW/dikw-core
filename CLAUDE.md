# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`dikw-core` is a Python 3.12+ AI-native knowledge engine spanning the full
DIKW pyramid (**D**ata → **I**nformation → **K**nowledge → **W**isdom).
Status: **pre-alpha** — APIs and on-disk formats will change.

Architecture is **client/server**: a `dikw serve` process (FastAPI + NDJSON)
hosts the engine; every HTTP-bound command lives under `dikw client …`,
spelled out — there are no top-level short aliases. The only top-level
commands that run in-process are `dikw version`, `dikw init`, `dikw serve`,
and the `dikw auth {login,import,status,list,logout}` subgroup (local OAuth
token management for the `openai_codex` provider).

Canonical docs (read these before designing changes):
- `docs/design.md` — approved design doc, source of truth for intent
- `docs/architecture.md` — module map, layer contracts, seams
- `docs/getting-started.md` — end-user walkthrough
- `docs/providers.md` — per-vendor config cookbook + production gotchas
  (batch size, dim locking, retry, prompt caching) when swapping LLM or
  embedding providers
- `docs/eval-plan.md` — methodology (retrieval-only Phase A, triggers for LLM-as-judge)
- `evals/README.md` — dataset three-file contract, how to add new datasets

## Dev workflow

Package manager is **`uv`** (not pip/poetry). Python **3.12+**.

```bash
uv sync --all-extras          # install (includes [postgres] + dev group)
uv run ruff check .           # lint
uv run mypy src               # strict type-check
uv run pytest -v              # tests (asyncio_mode=auto)
uv run pytest tests/test_storage_contract.py   # storage-contract tests (also run in CI against real Postgres)
uv run dikw <cmd>             # exercise the CLI against a scratch wiki
```

CI (`.github/workflows/ci.yml`) gates PRs on ruff + mypy + pytest across
Python 3.12 and 3.13, and runs the storage contract suite against a
`pgvector/pgvector:pg16` Postgres service. Release tags (`vX.Y.Z`) publish
to PyPI via trusted publishing (`.github/workflows/release.yml`).

Tooling config lives in `pyproject.toml`:
- ruff: line-length 100, rules `E,F,W,I,UP,B,SIM,C4,RUF` (E501 ignored)
- mypy: `strict = true`, `packages = ["dikw_core"]`, `mypy_path = "src"`
- pytest: `asyncio_mode = "auto"`, `testpaths = ["tests"]`

## Architecture at a glance

```
src/dikw_core/
├── api.py                 engine facade — ingest, retrieve, synthesize, lint (+ propose/apply),
│                          distill, review, list_pages, read_page, list_links, read_provenance,
│                          list_graph, read_asset, status, health, check_providers
├── cli.py                 top-level Typer app: version, init, serve, auth subgroup, dikw client subgroup
│                          (client commands are also spliced as top-level aliases — `dikw status`,
│                          `dikw retrieve`, `dikw serve-and-run`, …)
├── auth_cli.py            `dikw auth {login,import,status,list,logout}` — local OAuth token store at <base>/.dikw/auth.json
├── logging.py             init_logging() — DIKW_LOG_LEVEL clamp; clamps httpx/httpcore/urllib3 to WARNING
├── md_inspect.py          standalone markdown preflight — frontmatter + image-ref extraction (no engine deps)
├── progress.py            ProgressReporter Protocol + CancelToken (engine-side progress contract)
├── config.py              pydantic config + YAML loader (dikw.yml)
├── schemas.py             cross-layer DTOs (cross the Storage Protocol boundary — no SQL types)
├── domains/               DIKW domain model — the four layers grouped together
│   ├── data/              D layer — sources + assets + SourceBackend registry (markdown only)
│   ├── info/              I layer — chunk, tokenize, embed, render, RRF-fused hybrid search
│   ├── knowledge/         K layer — wiki pages, [[wikilinks]], frontmatter `sources:` ↔ provenance edge,
│   │                                  index.md, log.md, lint (incl. missing_provenance), lint_fix + lint_fixers/
│   └── wisdom/            W layer — distill, review state machine, apply-at-retrieve (PR-5)
├── providers/             LLMProvider + EmbeddingProvider + MultimodalEmbeddingProvider Protocols
│                          (anthropic_compat, openai_compat, openai_codex, gitee_multimodal)
├── storage/               Storage Protocol + adapters (sqlite, postgres) + migrations/{sqlite,postgres}
├── eval/                  retrieval + synth-quality eval — metrics, judge, dataset loader, runner, fake embedder
├── prompts/               versioned LLM prompts (importlib.resources)
├── server/                FastAPI app, auth, sync + task + import + retrieve + pages + assets + graph routes,
│                          NDJSON streaming, task subsystem
└── client/                Remote Typer CLI + httpx transport + NDJSON progress + sources importer + converter dispatch
```

### Layering invariants

- `server/*` may import `dikw_core.api`, `schemas`, `storage`, `providers`. The reverse is forbidden — engine code must not depend on FastAPI / uvicorn / server task plumbing.
- `client/*` only depends on `schemas` (for response type alignment) and stdlib + httpx + typer + rich. It must not import any `dikw_core.{api,storage,providers,server}` symbol — the client is meant to be packagable as a standalone wheel later.

### Named seams — extend here, not elsewhere

1. **`SourceBackend`** (`domains/data/backends/base.py`) — new formats: one subclass + `register()`. Reference impl: `domains/data/backends/markdown.py`.
2. **`Storage` Protocol** (`storage/base.py`) — two backends ship (sqlite, postgres); engine code depends only on the Protocol. Hybrid-search fusion (RRF), chunking, link-graph parsing, and wisdom scoring live **outside** adapters — adapters expose primitives only.
3. **`LLMProvider` / `EmbeddingProvider`** (`providers/base.py`) — Anthropic uses `cache_control` on the system prompt; openai_compat works against any base URL.

### Core invariants

- **Karpathy's rule:** *scoping is deterministic, reasoning is probabilistic*. Navigation (source listing, chunk lookup, link traversal, wisdom lookup-by-title) is deterministic SQL/file I/O. LLMs enter only at synth and distill — the two engine-internal authoring legs that write the K and W layers. **Answer synthesis is not a `dikw-core` verb**; `retrieve` returns ranked chunks + page refs and the agent layer runs its own LLM on the result.
- **W layer gate:** every wisdom item must cite **≥ 2 pieces of evidence** from K or D; state transitions go through `dikw review approve|reject`.
- **On-disk format is the product.** `wiki/` and `wisdom/` are plain markdown with YAML front-matter and `[[wikilinks]]` — an Obsidian vault the user owns. The engine writes, the user reads/edits with any editor.
- **Idempotent ingest.** Files whose content hash is unchanged are skipped.
- **Link reconciliation.** Re-persisting a wiki page **replaces** — not unions — its outgoing link set. `persist_wiki_page` (in `domains/knowledge/page_index.py`, single source of truth shared by synth and `lint apply`) calls `storage.replace_links_from(doc_id, resolved)` (atomic delete + insert in one transaction, mirrors `replace_chunks`), so removing a `[[wikilink]]` from the body actually drops it from storage. Without this the `links` table accumulates ghost edges as users edit pages, polluting the graph-leg retrieval channel and silently miscounting `orphan_page` / `broken_wikilink` lint. The `lint apply` path passes `embedder=None` to keep apply provider-free — embeddings reconcile on the next `dikw ingest` via `doc.hash` drift, so a heuristic-only fix can land without an embedder configured.
- **Provenance reconciliation.** A K-page's `sources:` frontmatter list is a second page-level edge — page → D-source attribution, distinct from `[[wikilink]]` (body). `persist_wiki_page` calls `storage.replace_provenance_from(doc_id, source_paths)` next to `replace_links_from` so the dedicated `provenance` table self-heals when a user edits frontmatter. Stored separately from `links` (see `docs/adr/0001-provenance-as-separate-edge.md`) because provenance has no body line/anchor, must NOT pollute the wikilink graph that feeds graph-leg retrieval, and reuse with `link_type='derived_from'` would force every downstream consumer to filter forever. Reverse lookup ("which K-pages claim this source?") via `GET /v1/base/pages/{source_path}/provenance?direction=in` (and `dikw client pages provenance`); forward dangling entries surface with `resolved=False`. Legacy bases bootstrap via the new `missing_provenance` LintKind + deterministic `MissingProvenanceFixer` — one pass of `dikw client lint fix apply --kind missing_provenance` backfills the table.
- **Wikilink fuzzy resolve.** `resolve_links` falls through three stages: exact title match, then a deterministic fuzzy normalize (NFKC + casefold + ASCII/CJK punctuation strip + ASCII trailing-plural stem) so `[[Neural Networks]]` resolves to `Neural Network`, `[[Elon Musk.]]` to `Elon Musk`, etc. When normalize maps a wikilink to a key whose index entry holds **two or more** distinct paths, we **refuse to resolve** — the link stays broken so `dikw lint` surfaces the ambiguity. Karpathy's rule: wrong-merge is irreversible damage, missed-resolve is a fixable lint warning. The unresolved count surfaces per-run via `SynthReport.unresolved_wikilinks` so users see broken-link drift without waiting for a separate `dikw lint` pass.
- **Synth-time existing-pages awareness.** Each synth LLM call within `_synth_pages_from_source` receives two prompt sections: `## Already created in this batch:` (per-source accumulator of pages emitted by earlier groups in the SAME source — Stage A 1:N fan-out runs groups serially, so group N must see what groups 0..N-1 wrote) and `## Existing wiki pages:` (full base K-layer snapshot below `synth.existing_pages_max_bytes`, default 16384 B; vec_search-gated top-K driven by the group's own chunk embeddings above that, capped at `synth.existing_pages_top_k`, default 50). When the LLM detects that a candidate page would semantically duplicate an entry in either section, it emits **zero `<page>` blocks** for that candidate and references the existing one via `[[Title]]` in its other pages instead. Karpathy's rule again: deterministic scoping (which pages exist, in this batch and in the base) feeds probabilistic reasoning (whether a candidate is a true duplicate). Without this awareness the LLM regenerates pages it cannot see, polluting the wiki with semantic duplicates that PR1's fuzzy resolver cannot absorb.
- **Orphan-page governance.** `OrphanPageFixer` (`domains/knowledge/lint_fixers/orphan_page.py`) routes each orphan to one of four strategies, deterministic-first: (1) `delete_page` for empty/TODO/unattributed stubs under 40 B; (2) `merge_into_existing_page` LLM-only when a candidate scores ≥ `MERGE_THRESHOLD = 6.0` (e.g. two shared `sources` entries) AND `--enable-llm`; (3) `link_from_existing_page` for scores ≥ `LINK_THRESHOLD = 3.0`, appending `[[orphan-title]]` under a stable `## 相关` heading on the parent; (4) `mark_as_leaf` writes `lint: {skip: [orphan_page], reason: …}` into the orphan's frontmatter, suppressed by `run_lint` on the next pass and surfaced via `LintReport.acknowledged_leaves`. Scoring weights live in `orphan_page.py`: shared `sources` ×3.0, shared full tag ×1.0, namespaced tag-domain ×0.5, title-jaccard ×2.0, embedding cosine ×3.0. The embedding leg goes through `storage.list_chunks` → `get_chunk_embeddings` → `vec_search(layer=Layer.WIKI)` with `asyncio.gather`; absence of an embedder degrades silently to pure heuristic. Ambiguous-title orphans (≥ 2 K-pages share the title) skip merge and link entirely and fall to leaf, because the backlink would resolve to the wrong page.
- **Soft-delete via `<base>/trash/`.** `_apply_one_op` (`lint_fix.py`) executes `delete_page` ops by purging storage rows first (`storage.delete_document(doc_id)` — documents + chunks + embeddings + outgoing links + wisdom_evidence references) then `shutil.move`-ing the file to `<base>/trash/wiki/<original-rel-path>` with a `trashed: {at, reason, proposal_id}` frontmatter block injected for audit. Same-second collisions get a `-NNN` counter suffix (000…999). Recovery: drag the file back into `wiki/` and rerun `dikw ingest`; the `trashed:` block is harmless residue users can hand-strip. `ingest` only scans `wiki/`, so `trash/` is naturally excluded. `Storage.delete_document(doc_id)` is now a Protocol method (`storage/base.py`) — sqlite cleans virtual tables (`documents_fts`, `vec_chunks_v*`) explicitly because they don't honor FK cascades; postgres relies on FK cascade + explicit `links` / `wisdom_evidence` deletes.

## Conventions

- Types: code is fully typed; mypy runs strict. Don't widen types to silence errors — fix the root cause. Missing-import overrides (`sqlite_vec`, `frontmatter`, `markdown_it`, `pgvector`, `jieba`) live in `pyproject.toml`; extend deliberately.
- DTOs: anything crossing the Storage Protocol is a pydantic model in `schemas.py`. No SQL types, ORM handles, or cursors leak out of adapters.
- Tests: `tests/fakes.py` provides in-memory fakes; prefer them over mocks. Storage adapters are validated via `tests/test_storage_contract.py` — add new adapter behavior to the contract, not to ad-hoc tests.
- Prompts: versioned markdown files under `src/dikw_core/prompts/`, loaded via `importlib.resources`. Don't inline prompts in code.
- Logging: `DIKW_LOG_LEVEL` (DEBUG/INFO/WARNING/ERROR/CRITICAL, default INFO) controls the root logger level for both CLI and `dikw serve`. It's an env var (not a `dikw.yml` field) because CLI parsing happens before any base is loaded. `init_logging()` is idempotent — safe to wire from multiple entry points; non-`dikw_core` loggers (httpx, httpcore, urllib3) are clamped to WARNING so per-request noise doesn't drown synth/embed progress.
- Secrets: `OPENAI_API_KEY` (openai_compat LLM), `ANTHROPIC_API_KEY` (anthropic LLM), and `DIKW_EMBEDDING_API_KEY` (every embedding call) are read from env. The embedding leg never falls back to `OPENAI_API_KEY` — set `DIKW_EMBEDDING_API_KEY` explicitly so LLM and embedding keys can differ (e.g., MiniMax LLM + Gitee AI embeddings). **`.env` is for secrets only**; non-secret config (URLs, models, dims, batch, display labels) lives in `dikw.yml`. Never hardcode or commit; `.env`/`.env.*` are gitignored (except `.env.example`). The `openai_codex` LLM is the exception: it doesn't read an env API key — it manages ChatGPT OAuth tokens in dikw's own per-wiki store at `<wiki>/.dikw/auth.json` (separate from codex CLI's `~/.codex/auth.json`, to avoid refresh_token rotation conflicts). Bootstrap with `dikw auth login openai-codex` (device-code flow) or `dikw auth import openai-codex` (one-shot copy from `~/.codex/auth.json`); dikw refreshes the access_token automatically before each call.

## Things not to do

- Don't call SQL or touch adapter internals from engine code — go through the `Storage` Protocol.
- Don't implement search fusion inside a storage adapter — it belongs in `info/search.py`.
- Don't add a new source format without registering a `SourceBackend`.
- Don't change on-disk wiki/wisdom layout without updating `docs/design.md` first — users open these trees in Obsidian.
- Don't ship K-layer (`domains/knowledge/`) or Retrieval (`domains/info/`, `RetrievalConfig`) changes without an entry in `evals/BASELINES.md` showing real-data outcome. K-layer changes get an `elon-musk.md` baseline **plus** the seven `synth/*` metrics from `dikw eval mvp --eval synth`; retrieval gets an ablation across packaged datasets. See `docs/eval-plan.md` "Acceptance gates for K-layer and Retrieval changes".
