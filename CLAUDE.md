# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`dikw-core` is a Python 3.12+ AI-native knowledge engine spanning the full
DIKW pyramid (**D**ata → **I**nformation → **K**nowledge → **W**isdom).
Status: **pre-alpha** — APIs and on-disk formats will change.

Architecture is **client/server**: a `dikw serve` process (FastAPI + NDJSON)
hosts the engine; `dikw client …` (the default user surface, plus top-level
aliases like `dikw status`) talks to it over HTTP. The local-only commands
are `dikw version`, `dikw init`, and `dikw serve` itself.

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
├── api.py                 engine facade (ingest, query, synth, distill, review, lint, status)
├── cli.py                 top-level Typer app: version, init, serve + dikw client subgroup
├── progress.py            ProgressReporter Protocol + CancelToken (engine-side progress contract)
├── config.py              pydantic config + YAML loader (dikw.yml)
├── schemas.py             cross-layer DTOs (cross the Storage Protocol boundary — no SQL types)
├── domains/               DIKW domain model — the four layers grouped together
│   ├── data/              D layer — sources + SourceBackend registry (markdown only)
│   ├── info/              I layer — chunk, embed, RRF-fused hybrid search
│   ├── knowledge/         K layer — wiki pages, [[wikilinks]], index.md, log.md, lint
│   └── wisdom/            W layer — distill, review state machine, apply-at-query
├── providers/             LLMProvider + EmbeddingProvider Protocols (anthropic, openai_compat)
├── storage/               Storage Protocol + adapters (sqlite, postgres)
├── eval/                  retrieval-quality eval — metrics, dataset loader, runner, packaged datasets
├── prompts/               versioned LLM prompts (importlib.resources)
├── server/                FastAPI app, auth, sync + task routes, NDJSON streaming, task subsystem
└── client/                Remote Typer CLI + httpx transport + NDJSON progress renderer + sources upload
```

### Layering invariants

- `server/*` may import `dikw_core.api`, `schemas`, `storage`, `providers`. The reverse is forbidden — engine code must not depend on FastAPI / uvicorn / server task plumbing.
- `client/*` only depends on `schemas` (for response type alignment) and stdlib + httpx + typer + rich. It must not import any `dikw_core.{api,storage,providers,server}` symbol — the client is meant to be packagable as a standalone wheel later.

### Named seams — extend here, not elsewhere

1. **`SourceBackend`** (`domains/data/backends/base.py`) — new formats: one subclass + `register()`. Reference impl: `domains/data/backends/markdown.py`.
2. **`Storage` Protocol** (`storage/base.py`) — two backends ship (sqlite, postgres); engine code depends only on the Protocol. Hybrid-search fusion (RRF), chunking, link-graph parsing, and wisdom scoring live **outside** adapters — adapters expose primitives only.
3. **`LLMProvider` / `EmbeddingProvider`** (`providers/base.py`) — Anthropic uses `cache_control` on the system prompt; openai_compat works against any base URL.

### Core invariants

- **Karpathy's rule:** *scoping is deterministic, reasoning is probabilistic*. Navigation (source listing, chunk lookup, link traversal, wisdom lookup-by-title) is deterministic SQL/file I/O. LLMs enter only at synth, distill, and the final answer step of query.
- **W layer gate:** every wisdom item must cite **≥ 2 pieces of evidence** from K or D; state transitions go through `dikw review approve|reject`.
- **On-disk format is the product.** `wiki/` and `wisdom/` are plain markdown with YAML front-matter and `[[wikilinks]]` — an Obsidian vault the user owns. The engine writes, the user reads/edits with any editor.
- **Idempotent ingest.** Files whose content hash is unchanged are skipped.
- **Link reconciliation.** Re-persisting a wiki page **replaces** — not unions — its outgoing link set. `_persist_wiki_page` calls `storage.replace_links_from(doc_id, resolved)` (atomic delete + insert in one transaction, mirrors `replace_chunks`), so removing a `[[wikilink]]` from the body actually drops it from storage. Without this the `links` table accumulates ghost edges as users edit pages, polluting the graph-leg retrieval channel and silently miscounting `orphan_page` / `broken_wikilink` lint.

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
- Don't ship K-layer (`domains/knowledge/`) or Retrieval (`domains/info/`, `RetrievalConfig`) changes without an entry in `evals/BASELINES.md` showing real-data outcome. K-layer changes get an `elon-musk.md` baseline; retrieval gets an ablation across packaged datasets. See `docs/eval-plan.md` "Acceptance gates for K-layer and Retrieval changes".
