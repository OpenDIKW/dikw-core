# CLAUDE.md

Guidance for Claude Code in the `dikw-core` repository.

## What this is

`dikw-core` is a Python 3.12+ AI-native knowledge engine for the DIKW pyramid
(**D**ata → **I**nformation → **K**nowledge → **W**isdom). Status: **alpha**.
APIs, on-disk formats, database schema, and CLI will change.

- Architecture is **client/server**. A `dikw serve` process (FastAPI + NDJSON) hosts the engine.
- Every HTTP-bound command is under `dikw client …`, spelled out. There are no top-level short aliases.
- Only these top-level commands run in-process: `dikw version`, `dikw init`, `dikw serve`, and the
  `dikw auth {login,import,status,list,logout}` subgroup (local OAuth tokens for the `openai_codex` provider).

Canonical docs. Read the relevant one before you design a change:

- `docs/design.md` — approved design, source of truth for intent.
- `docs/architecture.md` — module map, layer contracts, seams.
- `docs/getting-started.md` — end-user walkthrough.
- `docs/providers.md` — per-vendor config and production gotchas (batch size, dim locking, retry, prompt caching). Read it before you swap an LLM or embedding provider.
- `docs/eval-plan.md` — eval method (retrieval-only Phase A, triggers for LLM-as-judge) and the acceptance gates for K-layer and Retrieval changes.
- `evals/README.md` — dataset three-file contract, and how to add a dataset.

## Commands

Package manager is **`uv`** (not pip or poetry). Python **3.12+**.

```bash
uv sync --all-extras          # install (includes [postgres] + dev group)
uv run python tools/check.py  # local CI mirror: ruff + mypy + fast pytest in CI order — run before every commit
uv run pre-commit install     # (once) git pre-commit hook: ruff + mypy on every commit
uv run ruff check .           # lint
uv run mypy src               # strict type-check
uv run pytest -v              # tests (asyncio_mode=auto)
uv run pytest tests/test_storage_contract.py   # storage contract (CI also runs it against real Postgres)
uv run dikw <cmd>             # exercise the CLI against a scratch base
uv run python tools/e2e_verify.py --mode local   # real-env e2e: every `dikw client` verb vs a live server
```

- `tools/check.py` is the one in-loop gate. It runs without `--cov`, because `--cov` flakes ASGI/CliRunner tests on Windows.
- `tools/e2e_verify.py` runs real-provider legs only when `.env` has the key vars that the active profile names. Otherwise those legs SKIP loudly.
- CI gates PRs on ruff + mypy + pytest (Python 3.12 and 3.13) and on the storage contract against `pgvector/pgvector:0.8.2-pg18`.
- Release tags (`vX.Y.Z`) publish to PyPI through trusted publishing.
- Details of the gate, the e2e harness, and tool config: `.claude/rules/verification-tools.md`.

## Where things live

- `src/dikw_core/api.py` is a thin re-export facade. It defines nothing. Put a verb body in its `api_*.py` module, never in `api.py`.
- Each `api_*.py` module imports `api_core`, `api_types`, and its domains. It never imports the facade.
- `domains/data`, `domains/info`, `domains/knowledge`, `domains/wisdom` hold the D, I, K, and W layers.
- `storage/` holds the `Storage` Protocol and the sqlite and postgres adapters.
- `providers/` holds the LLM, embedding, multimodal, and rerank Protocols and adapters.
- `server/` holds the FastAPI app and the task subsystem. `client/` holds the remote Typer CLI.
- `prompts/` holds versioned LLM prompts.
- Full module map: `.claude/rules/module-map.md` and `docs/architecture.md`.

### Layering invariants

- `server/*` may import `dikw_core.api`, `schemas`, `storage`, `providers`. The reverse is forbidden — engine code must not depend on FastAPI / uvicorn / server task plumbing.
- `client/*` only depends on `schemas` (for response type alignment) and stdlib + httpx + typer + rich. It must not import any `dikw_core.{api,storage,providers,server,eval}` symbol — the client is meant to be packagable as a standalone wheel later.

### Named seams — extend here, not elsewhere

1. **`SourceBackend`** (`domains/data/backends/base.py`) — new formats: one subclass + `register()`. Reference impl: `domains/data/backends/markdown.py`.
2. **`Storage` Protocol** (`storage/base.py`) — two backends ship (sqlite, postgres); engine code depends only on the Protocol. Hybrid-search fusion (RRF), chunking, and link-graph parsing live **outside** adapters — adapters expose primitives only.
3. **`LLMProvider` / `EmbeddingProvider`** (`providers/base.py`) — Anthropic uses `cache_control` on the system prompt; openai_compat works against any base URL.

## Core invariants

Each line below is a summary. The full text is in `.claude/rules/`. A rule loads when you read or edit a file in its area.
Before you design a change in an area, read its rule file.

- **Karpathy's rule:** scoping is deterministic, reasoning is probabilistic. LLMs enter only at synth. `retrieve` never calls a generative LLM. A cross-encoder reranker is scoping; an LLM reranker is excluded (ADR-0006). → `retrieval.md`
- **Rerank** is on once `provider.rerank` is set. It reorders the fused pool and never adds a chunk to it. → `retrieval.md`
- **On-disk format is the product.** K and W pages are plain Markdown + YAML front matter + `[[wikilinks]]`. The synth LLM contributes only the `tags` front-matter key. → `knowledge-layer.md`
- **Categories are a closed set** from `schema.categories`. A page that fits no category goes to `schema.fallback`. → `knowledge-layer.md`
- **Re-persisting a K page replaces** its outgoing links and provenance edges. It does not merge them. → `knowledge-layer.md`
- **Provenance is a separate edge.** A K page's `sources:` front matter is stored in the `provenance` table, never in the wikilink graph (ADR-0001). → `knowledge-layer.md`
- **Wikilink resolve refuses an ambiguous fuzzy match.** A wrong merge is irreversible; a broken link is a fixable lint warning. → `knowledge-layer.md`
- **Synth sees the existing pages.** Each synth call gets the pages already written in its batch and in the base, so it links to them instead of writing duplicates. → `knowledge-layer.md`
- **Orphan pages are routed deterministically first:** delete a tiny stub, merge (LLM, opt-in), link from a parent, or mark as a leaf. → `knowledge-layer.md`
- **Ingest is idempotent.** An unchanged content hash is skipped; only a broken stored `mtime` makes a row re-persist once. → `persist-pipeline.md`
- **One write entry per layer:** `persist_source` (D), `persist_knowledge` (K), `persist_wisdom` (W). Only `ingest` may register a new embed version. → `persist-pipeline.md`
- **`documents.active` is the commit marker.** A document is active only after its full pipeline completes. On a hard exception, deactivate it. → `persist-pipeline.md`
- **Delete purges storage rows first**, then moves the file to `<base>/trash/`. → `delete-trash.md`
- **Write serialization:** every SQLite adapter method runs through `self._locked`. Every base-mutating task takes `ServerRuntime.ingest_lock`. → `storage-concurrency.md`
- **Telemetry** goes only through `telemetry.py`. Never import `opentelemetry` directly in engine code. → `observability.md`
- **Key env-var names** come from `ProviderConfig`. There is no hardcoded name and no fallback. → `providers-secrets.md`

## Working rules

### Clarify before coding

- State your assumptions.
- If a decision blocks you, ask one question with the AskUserQuestion tool. Put your recommended answer first.
- If a simpler approach exists, say so before you write code.
- Before a K-layer or W-layer change, read `docs/design.md`.

### Keep the change small

- Write the minimum code that solves the request. No speculative features, no single-use abstractions, no error handling for impossible cases.
- Apply Karpathy's rule to your own design: deterministic scoping does not need an LLM; probabilistic reasoning does not need a state machine.
- Change only what the request needs. Do not reformat, rename, or refactor code outside it. Match the surrounding style.
- Report unrelated dead code or smells. Do not change them without approval.
- Remove the imports, helpers, and tests that your change made unused. Leave dead code that was already there.
- If the diff grows well past what the request implies, rewrite it before review.

### Test first

- Turn each task into `step → check` pairs. Loop until the check passes.
- Bug fix: reproduce the bug in a failing test first.
- K-layer and Retrieval changes: write the failing test before the implementation. This is mandatory.
- Refactor: the same tests pass before and after.

### Language

- Write plans and reports in Chinese.
- Write code, commits, PR titles, and identifiers in English.

## Autonomy

- When a step does not need my input, continue. Put status notes in the same message as your next action.
- Treat explicit steps in a `/goal` request as approval for those steps. The delivery loop is approval to commit, push, open the PR, squash-merge, and delete the merged feature branch.
- Stop and ask only when one of these is true:
  - You cannot continue without my decision.
  - A block signal from the `dikw-core-delivery-workflow` skill fires.
  - The next action is destructive and not approved above: delete data or files you did not create, or change anything outside this repository.
- Do not end a turn in these ways while work is still owed:
  1. A summary that announces the next step but does not take it.
  2. An offer to continue "unless you prefer otherwise".
  3. A list of decisions that do not block the remaining work.
  4. A pause only because the turn was long or a milestone is done.
- **WARNING:** Never force-push. Describe the situation and let me do it.

## Finish line

A non-trivial change is done when all of these are true:

- The PR is squash-merged. Local `main` is fast-forwarded. The feature branch is deleted.
- Every required check is green. Every actionable review comment is fixed or answered.
- The PR body has a `## Delivery receipt` section.

If you cannot reach this, stop on a block signal and report it.

## Report

End every run with these three headings:

- **需要你决定** — decisions or approvals you wait for. Write "无" if there are none.
- **改动** — what changed, with PR links.
- **发现** — what you found. Mark each claim you could not confirm, and say where you looked.

For an architecture or flow explanation, use a Mermaid diagram or an HTML page when it is clearer than prose.

## Delivery loop

- For any non-trivial change, run the `dikw-core-delivery-workflow` skill. It is the only definition of the steps, the review tiers, and the block signals.
- For a trivial edit (typo, comment, one-line fix), skip the loop. Still run `tools/check.py`.
- `uv run python tools/loop_metrics.py` reports how well the loop works: first-pass-green rate, escape rate, codex rounds.

## Conventions

- **Types:** code is fully typed; mypy runs strict. Do not widen types to silence errors. Fix the root cause. Missing-import overrides (`sqlite_vec`, `frontmatter`, `markdown_it`, `pgvector`, `jieba`) are in `pyproject.toml`. Extend them deliberately.
- **DTOs:** anything that crosses the Storage Protocol is a pydantic model in `schemas.py`. No SQL types, ORM handles, or cursors leave the adapters.
- **Tests:** use the in-memory fakes in `tests/fakes.py`, not mocks. Add new adapter behavior to `tests/test_storage_contract.py`, not to ad-hoc tests.
- **Prompts:** versioned Markdown files under `src/dikw_core/prompts/`. Never inline a prompt in code. Details: `knowledge-layer.md`.
- **Logging:** `DIKW_LOG_LEVEL` and `DIKW_LOG_FORMAT` (`text` or `json`) are env vars, not `dikw.yml` fields. Details: `observability.md`.
- **Secrets:** `.env` holds secrets only. Non-secret config goes in `dikw.yml`. Never hardcode or commit a key. Details: `providers-secrets.md`.
- **Area rules:** when your change alters behavior that a file in `.claude/rules/` describes, update that rule in the same PR.

## Things not to do

- Don't call SQL or touch adapter internals from engine code — go through the `Storage` Protocol.
- Don't implement search fusion inside a storage adapter — it belongs in `info/search.py`.
- Don't add a new source format without registering a `SourceBackend`.
- Don't change on-disk knowledge/wisdom layout without updating `docs/design.md` first — users open these trees in any Markdown editor.
- Don't ship K-layer (`domains/knowledge/`) or Retrieval (`domains/info/`, `RetrievalConfig`) changes without an entry in `evals/BASELINES.md` showing real-data outcome. K-layer changes get an `elon-musk.md` baseline **plus** the seven `synth/*` metrics from `dikw client eval --dataset mvp --eval synth`; retrieval gets an ablation across packaged datasets. See `docs/eval-plan.md` "Acceptance gates for K-layer and Retrieval changes".
