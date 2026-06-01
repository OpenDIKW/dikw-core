# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`dikw-core` is a Python 3.12+ AI-native knowledge engine spanning the full
DIKW pyramid (**D**ata → **I**nformation → **K**nowledge → **W**isdom).
Status: **alpha** — APIs, on-disk formats, database schema, and CLI will change.

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
uv run dikw <cmd>             # exercise the CLI against a scratch base
```

CI (`.github/workflows/ci.yml`) gates PRs on ruff + mypy + pytest across
Python 3.12 and 3.13, and runs the storage contract suite against a
`pgvector/pgvector:0.8.2-pg18` Postgres service. Release tags (`vX.Y.Z`) publish
to PyPI via trusted publishing (`.github/workflows/release.yml`).

Tooling config lives in `pyproject.toml`:
- ruff: line-length 100, rules `E,F,W,I,UP,B,SIM,C4,RUF` (E501 ignored)
- mypy: `strict = true`, `packages = ["dikw_core"]`, `mypy_path = "src"`
- pytest: `asyncio_mode = "auto"`, `testpaths = ["tests"]`

## Architecture at a glance

```
src/dikw_core/
├── api.py                 thin re-export facade — surfaces every verb (ingest, retrieve,
│                          synthesize, lint (+ propose/apply), list_pages, read_page,
│                          list_links, read_provenance, list_graph, read_asset, status,
│                          health, check_providers, write_wisdom_page) so the public
│                          `api.X` surface + `__all__` stay byte-stable; defines nothing
├── api_*.py               verb clusters the facade re-exports: api_core (scaffold +
│                          `_with_storage` + embed-version helpers), api_types (DTOs +
│                          exceptions), api_health, api_ingest, api_pages, api_graph,
│                          api_retrieve, api_synth, api_lint, api_wisdom, api_path_safety.
│                          Each imports api_core/api_types + its domains, never the facade
│                          (acyclic). Move a verb's body here, not into api.py
├── cli.py                 top-level Typer app: version, init, serve, auth subgroup, dikw client subgroup
│                          (HTTP-bound commands live exclusively under `dikw client <verb>` —
│                          there are no top-level short aliases)
├── auth_cli.py            `dikw auth {login,import,status,list,logout}` — local OAuth token store at <base>/.dikw/auth.json
├── logging.py             init_logging() — DIKW_LOG_LEVEL clamp; clamps httpx/httpcore/urllib3 to WARNING
├── md_inspect.py          standalone markdown preflight — frontmatter + image-ref extraction (no engine deps)
├── progress.py            ProgressReporter Protocol + CancelToken (engine-side progress contract)
├── config.py              pydantic config + YAML loader (dikw.yml)
├── schemas.py             cross-layer DTOs (cross the Storage Protocol boundary — no SQL types)
├── domains/               DIKW domain model — the four layers grouped together
│   ├── data/              D layer — sources + assets + SourceBackend registry (markdown only)
│   ├── info/              I layer — chunk, tokenize, embed, render, RRF-fused hybrid search
│   ├── knowledge/         K layer — knowledge pages filed under a configurable `category` tree
│   │                                  (`knowledge/<category>/<slug>.md`), [[wikilinks]], frontmatter `sources:` ↔
│   │                                  provenance edge, lint (incl. missing_provenance, uncategorized), lint_fix + lint_fixers/
│   └── wisdom/            W layer — hand-written documents; `page.py::author_from_path`
│                                    (`wisdom/<author>/<slug>.md` → author); indexed exclusively by
│                                    `api.write_wisdom_page` (`dikw client wisdom write` / `POST /v1/base/wisdom`).
│                                    `dikw client ingest` does NOT scan `<base>/wisdom/` (0.4.0 BREAKING).
├── providers/             LLMProvider + EmbeddingProvider + MultimodalEmbeddingProvider Protocols
│                          (anthropic_compat, openai_compat, openai_codex, gitee_multimodal)
├── storage/               Storage Protocol + adapters (sqlite, postgres) + migrations/{sqlite,postgres}
├── eval/                  retrieval + synth-quality eval — metrics, judge, dataset loader, runner, fake embedder
├── prompts/               versioned LLM prompts (importlib.resources); `resolve()` + `_contract.py` validate
│                          per-base overrides (`synth.prompt_path` / `lint.fixer_prompts`)
├── server/                FastAPI app, auth, sync + task + import + retrieve + pages + assets + graph routes,
│                          NDJSON streaming, task subsystem
└── client/                Remote Typer CLI + httpx transport + NDJSON progress + sources importer + converter dispatch
```

### Layering invariants

- `server/*` may import `dikw_core.api`, `schemas`, `storage`, `providers`. The reverse is forbidden — engine code must not depend on FastAPI / uvicorn / server task plumbing.
- `client/*` only depends on `schemas` (for response type alignment) and stdlib + httpx + typer + rich. It must not import any `dikw_core.{api,storage,providers,server}` symbol — the client is meant to be packagable as a standalone wheel later.

### Named seams — extend here, not elsewhere

1. **`SourceBackend`** (`domains/data/backends/base.py`) — new formats: one subclass + `register()`. Reference impl: `domains/data/backends/markdown.py`.
2. **`Storage` Protocol** (`storage/base.py`) — two backends ship (sqlite, postgres); engine code depends only on the Protocol. Hybrid-search fusion (RRF), chunking, and link-graph parsing live **outside** adapters — adapters expose primitives only.
3. **`LLMProvider` / `EmbeddingProvider`** (`providers/base.py`) — Anthropic uses `cache_control` on the system prompt; openai_compat works against any base URL.

### Core invariants

- **Karpathy's rule:** *scoping is deterministic, reasoning is probabilistic*. Navigation (source listing, chunk lookup, link traversal) is deterministic SQL/file I/O. LLMs enter only at synth — the engine-internal authoring leg that writes the K layer. **Answer synthesis is not a `dikw-core` verb**; `retrieve` returns ranked chunks + page refs and the agent layer runs its own LLM on the result. W layer is hand-written in 0.3.0 (no engine LLM call writes wisdom).
- **On-disk format is the product.** `knowledge/` and `wisdom/` are plain markdown with YAML front-matter and `[[wikilinks]]` — an Obsidian vault the user owns. The engine writes K (`synth`), the user writes W; user reads/edits both with any editor.
- **Configurable knowledge taxonomy (0.5.0 BREAKING, ADR-0003/0004).** K-page classification is a configurable, hierarchical **`category`** tree declared in `dikw.yml` under `schema.categories` (each a `path` + optional `desc`), not the fixed single-axis `type` (entity/concept/note) it replaced — `entity`/`concept`/`note` is now merely the *default* taxonomy. The set is **closed**: synth's LLM (`<page category="…" slug="…">`, no `path=`) may only file under a declared `path`; anything unplaceable lands in `schema.fallback` (default `未分类`) and is flagged by the `uncategorized` lint. Karpathy's rule — a wrong category is a cheap re-file, an invented folder is irreversible drift, so the parser refuses to honour an out-of-set category and there is no path-prefix *recovery* (the brittle 0.4.6 normalizer was deleted). `category_from_path` recovers a page's category at any depth; `default_page_path(category, title)` → `knowledge/<category>/<slug>.md`. No `type:` frontmatter, no pluralized folders, no `SchemaConfig.page_types`/`log_style`; a base carrying any of them trips `BaseUpgradeRequired` (rebuild policy, no in-place migration). K-layer authoring prompts (`synthesize`, `lint_fix_orphan_merge`, `lint_fix_broken_wikilink_grounded`) are per-base overridable via `synth.prompt_path` / `lint.fixer_prompts` — `prompts.resolve` enforces base-containment + the `_contract.py` placeholder/output-marker contract at load and at `dikw client check`. `synth` no longer generates `knowledge/index.md` / `log.md` — the category folder tree is the catalogue and the `knowledge_log` table (+ `list_knowledge_log`) is the history; `indexgen.py` / `log.py` are deleted.
- **Idempotent ingest.** Files whose content hash is unchanged are skipped — except a source row whose stored `mtime` is broken (`<= 0`, a legacy byte-stable import that landed `st_mtime == 0`) re-persists once so it self-heals; new source ingests fall back to wall-clock when the file carries no usable mtime, preferring an already-stored positive value only for an unchanged re-persist (same body hash) so it doesn't flap the graph change-hash while a real content change still advances it (#145).
- **Link reconciliation.** Re-persisting a knowledge page **replaces** — not unions — its outgoing link set. `persist_knowledge` (in `domains/knowledge/page_index.py`, single source of truth shared by synth and `lint apply`) calls `storage.replace_links_from(doc_id, resolved)` (atomic delete + insert in one transaction, mirrors `replace_chunks`), so removing a `[[wikilink]]` from the body actually drops it from storage. Without this the `links` table accumulates ghost edges as users edit pages, polluting the graph-leg retrieval channel and silently miscounting `orphan_page` / `broken_wikilink` lint.
- **chunk → FTS → embed pipeline per layer (0.4.0).** Each DIKW layer has exactly one programmatic write entry that owns its full `upsert_document` + `chunk_markdown` + `replace_chunks` (FTS side-effect) + optional inline embed + `replace_links_from` + `replace_provenance_from` pipeline: `persist_source` (`domains/data/persist.py`) for D, `persist_knowledge` for K (synth + lint apply), `persist_wisdom` (`domains/wisdom/persist.py`) for W. `api.ingest` is **D-only** — it scans `<base>/sources/` and runs every file through `persist_source` with embed deferred to one bulk pass at end-of-scan (throughput optimisation). `synth` / `lint apply` write K pages with **inline embed when an embedder is wired AND the active text version's `(provider, model)` identity matches `cfg.provider`** (lint apply checks `DIKW_EMBEDDING_API_KEY` and auto-builds). `write_wisdom_page` writes W pages with inline embed under the same conditions unless `no_embed=True`. Lint apply / wisdom write **never register-and-activate** a new text version on their own — that would flip `embed_versions.is_active` and strand every other vector in the now-inactive table; only `ingest` (which re-embeds the whole corpus) is allowed to register. When the active identity drifts from cfg (user edited `dikw.yml` between full ingests), inline embed defers to the next ingest's resume scan instead of silently mixing vector spaces under the old version. Embed-batch failures classified by `TransientProviderError` vs bare `ProviderError`: the per-batch retry-skip in `consume_embedding_stream` (`cfg.provider.embedding_error_retries` / `embedding_error_retry_backoff_seconds`) retries only transient (5xx, 408/429, timeout, connect drop, parse failure); permanent (401, 403, 404, missing key, invalid model id) propagates so misconfig fails fast instead of being silently retried-then-skipped. Failing chunks remain in storage without vectors and the per-call return surfaces `chunks_pending_embedding`. `api.ingest` then runs **one cross-layer resume scan** via `storage.list_chunks_missing_embedding(version_id)` that reconciles D/K/W chunks deferred by any of the above paths, so the eventual-consistency contract is "every chunk eventually has a vector once an embedder is reachable." Hand-edits to K or W files on disk, and out-of-order writes (referring page `A` written before its `[[B]]` target page), are NOT auto-reindexed — a future `dikw client reindex <path>` will close both gaps; until then re-write the referring page via the normal write entry (`dikw client synth` / `dikw client wisdom write`) to re-resolve.
- **Provenance reconciliation.** A K-page's `sources:` frontmatter list is a second page-level edge — page → D-source attribution, distinct from `[[wikilink]]` (body). `persist_knowledge` calls `storage.replace_provenance_from(doc_id, source_paths)` next to `replace_links_from` so the dedicated `provenance` table self-heals when a user edits frontmatter. Stored separately from `links` (see `docs/adr/0001-provenance-as-separate-edge.md`) because provenance has no body line/anchor, must NOT pollute the wikilink graph that feeds graph-leg retrieval, and reuse with `link_type='derived_from'` would force every downstream consumer to filter forever. Reverse lookup ("which K-pages claim this source?") via `GET /v1/base/pages/{source_path}/provenance?direction=in` (and `dikw client pages provenance`) — gated on `Layer.SOURCE` so a malformed K-page that lists a `knowledge/...` path in `sources:` cannot surface as a `derived_pages` entry on the target. Forward dangling entries surface with `resolved=False`. Legacy bases bootstrap via the new `missing_provenance` LintKind + deterministic `MissingProvenanceFixer` — run `dikw client lint propose --rule missing_provenance` then `dikw client lint apply <task_id>` to backfill the table. The lint detector compares `{normalized_key: raw_path}` dicts (not key sets), so raw-spelling drift (`Sources/Foo.md` → `sources/foo.md`) surfaces alongside the four key-level sub-cases (zero / partial / stale / cleared). Every list-of-strings frontmatter read (`sources:`, `tags:`) goes through `frontmatter_str_list` in `page.py` so a hand-written YAML scalar collapses to `[]` instead of being iterated per character.
- **Wikilink fuzzy resolve.** `resolve_links` falls through three stages: exact title match, then a deterministic fuzzy normalize (NFKC + casefold + ASCII/CJK punctuation strip + ASCII trailing-plural stem) so `[[Neural Networks]]` resolves to `Neural Network`, `[[Elon Musk.]]` to `Elon Musk`, etc. When normalize maps a wikilink to a key whose index entry holds **two or more** distinct paths, we **refuse to resolve** — the link stays broken so `dikw client lint` surfaces the ambiguity. Karpathy's rule: wrong-merge is irreversible damage, missed-resolve is a fixable lint warning. The unresolved count surfaces per-run via `SynthReport.unresolved_wikilinks` so users see broken-link drift without waiting for a separate `dikw client lint` pass.
- **Synth-time existing-pages awareness.** Each synth LLM call within `_synth_pages_from_source` receives two prompt sections: `## Already created in this batch:` (per-source accumulator of pages emitted by earlier groups in the SAME source — Stage A 1:N fan-out runs groups serially, so group N must see what groups 0..N-1 wrote) and `## Existing knowledge pages:` (full base K-layer snapshot below `synth.existing_pages_max_bytes`, default 16384 B; vec_search-gated top-K driven by the group's own chunk embeddings above that, capped at `synth.existing_pages_top_k`, default 50). When the LLM detects that a candidate page would semantically duplicate an entry in either section, it emits **zero `<page>` blocks** for that candidate and references the existing one via `[[Title]]` in its other pages instead. Karpathy's rule again: deterministic scoping (which pages exist, in this batch and in the base) feeds probabilistic reasoning (whether a candidate is a true duplicate). Without this awareness the LLM regenerates pages it cannot see, polluting the knowledge base with semantic duplicates that PR1's fuzzy resolver cannot absorb.
- **Orphan-page governance.** `OrphanPageFixer` (`domains/knowledge/lint_fixers/orphan_page.py`) routes each orphan to one of four strategies, deterministic-first: (1) `delete_page` for empty/TODO/unattributed stubs under 40 B; (2) `merge_into_existing_page` LLM-only when a candidate scores ≥ `MERGE_THRESHOLD = 6.0` (e.g. two shared `sources` entries) AND `--enable-llm`; (3) `link_from_existing_page` for scores ≥ `LINK_THRESHOLD = 3.0`, appending `[[orphan-title]]` under a stable `## 相关` heading on the parent; (4) `mark_as_leaf` writes `lint: {skip: [orphan_page], reason: …}` into the orphan's frontmatter, suppressed by `run_lint` on the next pass and surfaced via `LintReport.acknowledged_leaves`. Scoring weights live in `orphan_page.py`: shared `sources` ×3.0, shared full tag ×1.0, namespaced tag-domain ×0.5, title-jaccard ×2.0, embedding cosine ×3.0. The embedding leg goes through `storage.list_chunks` → `get_chunk_embeddings` → `vec_search(layer=Layer.KNOWLEDGE)` with `asyncio.gather`; absence of an embedder degrades silently to pure heuristic. Ambiguous-title orphans (≥ 2 K-pages share the title) skip merge and link entirely and fall to leaf, because the backlink would resolve to the wrong page.
- **Soft-delete via `<base>/trash/`.** `_apply_one_op` (`lint_fix.py`) executes `delete_page` ops by purging storage rows first (`storage.delete_document(doc_id)` — documents + chunks + embeddings + outgoing links) then `shutil.move`-ing the file to `<base>/trash/knowledge/<original-rel-path>` with a `trashed: {at, reason, proposal_id}` frontmatter block injected for audit. Same-second collisions get a `-NNN` counter suffix (000…999). Recovery: drag the file back into `knowledge/` — the `trashed:` block is harmless residue users can hand-strip — then re-run `dikw client synth` against the source that originally produced it (the K-layer has no scan-based reindex entry; this is the same parallel limitation as a K obsidian hand-edit). `ingest` only scans `<base>/sources/`, so the `trash/` tree is naturally outside its purview. `Storage.delete_document(doc_id)` is now a Protocol method (`storage/base.py`) — sqlite cleans virtual tables (`documents_fts`, `vec_chunks_v*`) explicitly because they don't honor FK cascades; postgres relies on FK cascade + explicit `links` deletes.

## Working principles

Behavioral defaults — bias toward caution over speed; for trivial edits use judgment. These sit alongside `Conventions` and `Things not to do` below.

- **Think before coding.** State assumptions; if uncertain ask one focused question instead of guessing. Surface tradeoffs and name alternatives — don't silently pick one. If a simpler approach exists, say so up-front rather than waiting for `/code-review` to surface it (cf. `feedback_codex_review_loop`). `docs/design.md` is the source of truth for K- and W-layer intent — read it before designing changes there.
- **Simplicity first.** Minimum code that solves the stated problem — no speculative features, no abstractions for single-use code, no flexibility that wasn't requested, no error handling for impossible scenarios. Karpathy's rule (above) is the project-level form: deterministic scoping does not deserve an LLM, probabilistic reasoning does not deserve a state machine. If the diff grew well past what the request implied, rewrite it before review.
- **Surgical changes.** Touch only what the request requires. Don't reformat, rename, or refactor adjacent code outside the blast radius; match surrounding style even if you'd write it differently. If you notice unrelated dead code or smells, mention them — don't delete or rewrite without approval. Remove orphans (imports, helpers, tests) that *your* change made unused; don't sweep pre-existing dead code. Every changed line should trace to the request or a direct consequence.
- **Goal-driven execution.** Transform tasks into `step → verify` pairs before starting and loop until verify passes. "Add validation" → failing test for the bad input first, then make it pass. "Fix the bug" → reproduce in a test first (K-layer / retrieval changes mandate this — see `feedback_tdd_discipline`). "Refactor X" → same tests pass before and after. Explicit steps in a `/goal` invocation count as pre-approval (see `feedback_goal_explicit_approves_steps`); only stop on real blockers (CHANGES_REQUESTED, red CI, mergeStateStatus ≠ CLEAN).

Working when diffs stay small, rewrites due to overcomplication are rare, and clarifying questions arrive before implementation rather than after.

## Delivery loop

End-to-end protocol for any non-trivial change. Run it autonomously — only pause at the block signals listed at the bottom.

1. **Clarify the request.** Don't start coding from a vague ask. Restate the goal, list assumptions, surface alternatives. For multi-decision work, escalate to the `grill-with-docs` skill or the `superpowers:brainstorming` skill until a written plan exists.
2. **Plan in the user's language, default TDD.** Write the plan in the language the user uses (Chinese in this repo — see `feedback_language_chinese`); keep code, commits, and technical identifiers English. Each step lands as `failing test → implementation → passing test`. K-layer / retrieval changes mandate this (see `feedback_tdd_discipline`).
3. **Codex review loop, up to 3 rounds.** When the implementation is feature-complete, run `/codex:review --background` and address each finding. Repeat up to 3 rounds (see `feedback_codex_review_loop`). When a finding implicates one CLI string / symbol / doc string, grep the whole repo before declaring it fixed (see `feedback_grep_cli_typos_across_docs` and `feedback_defensive_guard_grep_read_sites`).
4. **Code review pass.** Run `/code-review` once the codex loop quiets down; resolve every finding before continuing. Doc-only PRs included — see `feedback_code_review_not_optional` for why this step is never optional.
5. **Doc sync.** Audit all markdown (CLAUDE.md, CONTEXT.md, `docs/**`, CHANGELOG.md, plans, ADRs, GUIDE_FOR_AGENTS.md) against the diff. Update anything stale — especially CLI spellings, frontmatter keys, env vars, HTTP routes.
6. **Commit + push + PR.** Local commit is fine without approval; `git push` and `gh pr create` proceed once steps 1–5 are done — the loop itself is the standing approval (see `feedback_pr_workflow`). K-layer / retrieval PRs need an `evals/BASELINES.md` entry or the `no-baseline-needed` label before CI can pass — handle this in step 6, not at merge time.
7. **Watch CI + PR comments to green.** Monitor checks and reviewer comments (CodeRabbit / human). Fix every actionable finding; reject nitpicks with a one-line reason. Don't stop until every check is green and `mergeStateStatus` is CLEAN.
8. **Squash merge + sync local.** Squash merge the PR, fast-forward local main, delete the feature branch.

### Block signals — stop and ask

- A required check is failing and the cause isn't obvious.
- A reviewer marks `CHANGES_REQUESTED` or raises a design-level concern.
- A finding requires widening a Storage / Provider Protocol or changing on-disk knowledge/wisdom layout (see `Things not to do`).
- A merge conflict appears that needs a domain decision (not just a mechanical resolve).
- A force-push would be needed — **forbidden**; describe the situation and let the user handle it manually (see `feedback_pr_workflow`).

Nitpicks, style preferences, and non-actionable suggestions are **not** block signals — note them and move on (see `feedback_goal_explicit_approves_steps`).

## Conventions

- Types: code is fully typed; mypy runs strict. Don't widen types to silence errors — fix the root cause. Missing-import overrides (`sqlite_vec`, `frontmatter`, `markdown_it`, `pgvector`, `jieba`) live in `pyproject.toml`; extend deliberately.
- DTOs: anything crossing the Storage Protocol is a pydantic model in `schemas.py`. No SQL types, ORM handles, or cursors leak out of adapters.
- Tests: `tests/fakes.py` provides in-memory fakes; prefer them over mocks. Storage adapters are validated via `tests/test_storage_contract.py` — add new adapter behavior to the contract, not to ad-hoc tests.
- Prompts: versioned markdown files under `src/dikw_core/prompts/`, loaded via `importlib.resources`. Don't inline prompts in code. A base may override the K-layer authoring prompts (`synthesize`, `lint_fix_orphan_merge`, `lint_fix_broken_wikilink_grounded`) with its own markdown under `<base>/prompts/` via `synth.prompt_path` / `lint.fixer_prompts`; load via `prompts.resolve(name, override_path=…, base_root=…)` (never raw `load()` on the call path), and register any new overridable prompt's placeholder/output-marker contract in `prompts/_contract.py`.
- Logging: `DIKW_LOG_LEVEL` (DEBUG/INFO/WARNING/ERROR/CRITICAL, default INFO) controls the root logger level for both CLI and `dikw serve`. It's an env var (not a `dikw.yml` field) because CLI parsing happens before any base is loaded. `init_logging()` is idempotent — safe to wire from multiple entry points; non-`dikw_core` loggers (httpx, httpcore, urllib3) are clamped to WARNING so per-request noise doesn't drown synth/embed progress.
- Secrets: `OPENAI_API_KEY` (openai_compat LLM), `ANTHROPIC_API_KEY` (anthropic LLM), and `DIKW_EMBEDDING_API_KEY` (every embedding call) are read from env. The embedding leg never falls back to `OPENAI_API_KEY` — set `DIKW_EMBEDDING_API_KEY` explicitly so LLM and embedding keys can differ (e.g., MiniMax LLM + Gitee AI embeddings). **`.env` is for secrets only**; non-secret config (URLs, models, dims, batch, display labels) lives in `dikw.yml`. Never hardcode or commit; `.env`/`.env.*` are gitignored (except `.env.example`). The `openai_codex` LLM is the exception: it doesn't read an env API key — it manages ChatGPT OAuth tokens in dikw's own per-base store at `<base>/.dikw/auth.json` (separate from codex CLI's `~/.codex/auth.json`, to avoid refresh_token rotation conflicts). Bootstrap with `dikw auth login openai-codex` (device-code flow) or `dikw auth import openai-codex` (one-shot copy from `~/.codex/auth.json`); dikw refreshes the access_token automatically before each call.

## Things not to do

- Don't call SQL or touch adapter internals from engine code — go through the `Storage` Protocol.
- Don't implement search fusion inside a storage adapter — it belongs in `info/search.py`.
- Don't add a new source format without registering a `SourceBackend`.
- Don't change on-disk knowledge/wisdom layout without updating `docs/design.md` first — users open these trees in Obsidian.
- Don't ship K-layer (`domains/knowledge/`) or Retrieval (`domains/info/`, `RetrievalConfig`) changes without an entry in `evals/BASELINES.md` showing real-data outcome. K-layer changes get an `elon-musk.md` baseline **plus** the seven `synth/*` metrics from `dikw client eval --dataset mvp --eval synth`; retrieval gets an ablation across packaged datasets. See `docs/eval-plan.md` "Acceptance gates for K-layer and Retrieval changes".
