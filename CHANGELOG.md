# Changelog

All notable changes to `dikw-core` are tracked here. The project is
**alpha** and follows [SemVer](https://semver.org) loosely — until
1.0, breaking changes can land in any minor version. The status notes
on each entry call out exactly what shape changes break.

## Unreleased

### Added

- **Official container image published to GHCR on every release.** The release
  workflow's new `publish-image` job builds `examples/docker/Dockerfile` (the
  same image the Trivy PR-scan builds) and pushes a public, multi-arch
  (`linux/amd64` + `linux/arm64`) image to
  `ghcr.io/opendikw/dikw-core:X.Y.Z` after the PyPI publish (it waits for the
  wheel to be installable, then `pip install`s it). There is intentionally **no
  floating `:latest`** — downstream pins an exact `X.Y.Z` so its debug
  environment stays reproducible. `examples/docker/docker-compose.yml` now pulls
  this image by default (pinned via the required `DIKW_VERSION` env var, with the
  `build:` block retained as a local-source fallback), giving downstream systems
  a stable, ready-to-run dikw-core to develop and debug their HTTP / `dikw
  client` integration against. See `examples/docker/README.md` and
  `docs/deployment-docker.md`.
- **Client/server version handshake.** `dikw client` now probes `GET /v1/info`
  once per invocation and compares the server's `engine_version` to the client's
  own installed `dikw-core` version. A confirmed mismatch **hard-fails** the
  command (exit 1, a `version skew:` line naming both versions) so a downstream
  system catches silent wire drift immediately — dikw-core is alpha, so a skewed
  client/server pair can misbehave in subtle ways. Ambiguous cases (server
  unreachable, `/v1/info` non-200, `engine_version` missing, or the client run
  from an uninstalled source checkout) skip the check and let the real request
  surface its own error, so the handshake never raises a *false* skew. Set
  `DIKW_ALLOW_VERSION_SKEW=1` to downgrade the hard-fail to a one-line stderr
  warning for deliberate mixed-version debugging. The probe is layering-clean
  (reads its own version via `importlib.metadata`, never imports the engine).
- **GitHub Release is now cut automatically on every tag.** `release.yml` gains a
  `github-release` job (`needs: publish`) that creates the GitHub Release for the
  pushed tag once the PyPI publish succeeds: the body is the tag's `CHANGELOG.md`
  section (extracted from its `## X.Y.Z` heading to the next `## `; falls back to
  GitHub's auto-generated notes with a `::warning::` when no section matches), and
  the built wheel + sdist ride along as downloadable assets. Previously the
  workflow only published to PyPI + GHCR and opened the Dockerfile-bump PR —
  Releases were created by hand, and several tags (v0.6.0 and earlier) shipped
  without one. A backfilled `v0.6.0` GitHub Release was created out-of-band.

### Fixed

- **Concurrency: serialize the SQLite adapter and the synth/lint-apply write
  path.** Three race conditions could surface under concurrent task execution
  on a single base. (1) Each `SQLiteStorage` instance shares one
  `sqlite3.Connection` across the `asyncio.to_thread` workers a verb fans out
  (retrieval already runs its fts/vec/asset legs via `asyncio.create_task`),
  and that connection's Python-level state is not thread-safe — overlapping
  workers could trip `sqlite3.InterfaceError` / phantom rows (the same hazard
  already worked around three times in `storage/base.py`, `eval/runner.py`, and
  `server/tasks/store_sqlite.py`). Every adapter method body now runs under a
  per-instance `threading.RLock` (acquired inside the worker thread), closing
  the window for *every* call site rather than one ad-hoc gather at a time.
  (2) `synth` and `lint apply` previously took **no** lock, so they could
  interleave with `ingest`/`delete`/`wisdom write`/each other — racing the same
  deterministic `doc_id` rows + on-disk page with no enclosing transaction
  (silent anchor loss, embed-version drift, or a healthy page mistakenly
  deactivated). Both now acquire the server's existing base write lock
  (`ServerRuntime.ingest_lock`), which already covered ingest/import/wisdom/
  delete, so the whole D/K/W write surface is single-writer per base within a
  process. (3) The SQLite adapter now sets an explicit `busy_timeout=30000`
  (and opens with `connect(timeout=30)` so the budget also covers the
  connection-time pragmas) — parity with the task store — so a cross-connection
  WAL writer blocks-then-succeeds instead of immediately raising `database is
  locked`. **No on-disk format, schema, Storage Protocol, or CLI change** —
  these are internal serialization fixes. `ingest_lock` stays per-process, so
  multi-replica Postgres deployments are unchanged (cross-replica doc-level
  locking is tracked separately). See `docs/server.md` § Storage concurrency.

### Docs

- **Install-from-PyPI path for downstream consumers.** `docs/getting-started.md`
  §1 now splits installation into Option A (`uv pip install 'dikw-core[...]'` —
  the published wheel, for systems that *use* dikw-core) and Option B (`git clone`
  + `uv sync` — for contributors), and adds an **optional-extras matrix**
  documenting all three user-facing extras (`postgres`, `cjk`, `otel`): what each
  pulls in, when to install it, and how the feature degrades without it. The
  README install section gains a matching "Install from PyPI" block. Previously
  both entry points led only with the from-source checkout flow, leaving the
  pip-install consumer path (and the `cjk` extra entirely) undocumented.

## 0.6.0 — config-driven provider API-key env vars (BREAKING); DeepSeek V4 Pro + Gitee bge-m3; horizontal model comparison

### Changed

- **BREAKING — provider API-key env var is now config-driven, and `DIKW_EMBEDDING_API_KEY`
  is removed.** `ProviderConfig` gains two **required** fields, `llm_api_key_env` and
  `embedding_api_key_env`, naming the environment variable that holds each leg's key.
  The engine no longer hardcodes any key var name: `anthropic_compat`/`openai_compat`
  read exactly the var named in `dikw.yml`, with no fallback. The dikw-invented
  `DIKW_EMBEDDING_API_KEY` magic name is gone — embedding keys now use **vendor-canonical**
  names (`OPENAI_API_KEY`, `GITEE_API_KEY`, …) chosen via `embedding_api_key_env`. The
  LLM/embedding "two separate keys" separation is now achieved by *naming distinct vars*
  (point both legs at one var to share a key, or at different vars to split vendors)
  rather than by a special name + no-fallback rule. **Migration:** add the two fields to
  every `dikw.yml` `provider:` block (a fresh `dikw init` scaffold writes them), and in
  `.env` rename `DIKW_EMBEDDING_API_KEY` → the vendor var your config names; a same-vendor
  Anthropic+MiniMax `.env` that reused `ANTHROPIC_API_KEY` for a MiniMax key should move
  the MiniMax key to `MINIMAX_API_KEY` and set `llm_api_key_env: MINIMAX_API_KEY`. Wipe
  the local `evals/.cache/snapshots/` after upgrading (its snapshot `dikw.yml`s predate
  the fields). `/v1/health`'s `api_key_present` and the `dikw client check` probe now key
  off the configured var; the `tools/e2e_verify.py` real-leg gate derives its required
  keys from the active profile's `provider.{llm,embedding}_api_key_env`.

### Added

- **DeepSeek V4 Pro (LLM) + Gitee AI bge-m3 (embeddings) support — config-only.** DeepSeek
  runs via the existing `anthropic_compat` protocol against its Anthropic-compatible
  endpoint (`llm_base_url: https://api.deepseek.com/anthropic`, `llm_model: deepseek-v4-pro`,
  key in `DEEPSEEK_API_KEY`); DeepSeek **ignores** the `cache_control` field the provider
  sends (no error — only the Anthropic prompt-cache discount is absent, same cost note as
  `openai_compat`). bge-m3 runs via `openai_compat` embeddings against Gitee
  (`embedding_base_url: https://ai.gitee.com/v1`, `embedding_model: bge-m3`,
  `embedding_dim: 1024`, `embedding_batch_size: 16`, key in `GITEE_API_KEY`). No engine
  code; a committed reference config ships at `tests/fixtures/live-deepseek-gitee-bgem3.dikw.yml`.
  See `docs/providers.md`.
- **Horizontal model-comparison harness (`evals/tools/compare_models.py`).** A dev tool
  (not shipped in the wheel) that runs the same eval dataset against N model arms and emits
  an arm-by-metric comparison matrix + per-arm JSON. `compare` compares **embedding** models
  via retrieval eval (deterministic, 1 run/arm: hit@k / mrr / nDCG@10 / recall@100);
  `compare-synth` compares **LLM** models via synth eval (N runs/arm + a Welch t-test of each
  arm vs the baseline arm: grounding / atomicity / duplicate / wikilink / language, plus judge
  dims with `--judge`). Each arm carries a full `provider:` block, so two same-protocol
  vendors (DeepSeek + MiniMax) resolve distinct keys via their `*_api_key_env`. Reuses the
  tested statistics from `ab_experiment.py` and the direction rule from `client/baseline.py`.
  See `evals/README.md` and `docs/providers.md`.
- **Real-environment end-to-end verification harness (`tools/e2e_verify.py`).** A dev
  tool (not shipped in the wheel) that drives **every** `dikw client` verb against a
  live server in one of two throwaway environments, then destroys it: `--mode local`
  (temp-dir base + long-lived `dikw serve` on SQLite) and `--mode docker` (server +
  `pgvector` Postgres via a generated compose project, image built **from the local
  working tree** — not the released PyPI `examples/docker/Dockerfile`). CLI coverage is
  asserted against the live Typer tree, so adding a verb without a sequence step fails
  the run. Provider posture is tiered + skip-loud: structural legs (`ingest --no-embed`,
  `pages`/`graph`/`lint`/`delete`/`tasks`) run with no keys; real legs
  (`check`/embed/`synth`/vector-`retrieve`/`eval`) run when the keys named by the
  active profile's `provider.{llm,embedding}_api_key_env` are present (from `.env`)
  and SKIP loudly otherwise. Both modes
  use a free host port (never a fixed `8765`) so concurrent runs don't collide; docker
  teardown is guaranteed (`down -v --rmi local` removes containers, volumes **and the
  built image**; `--prune` sweeps crashed-run leftovers by label/name). `--observe` wires the
  `docs/observability` OTel stack and surfaces a Jaeger trace link on failure. Registered
  as a `cli/server/client` leg in the `dikw-core-verify` skill; wrapped by
  `tests/test_e2e_verify_{local,docker}.py` (`-m slow`). Default provider profile is the
  committed MiniMax + Qwen3-Embedding-0.6B template; swap vendor/model via
  `--provider-profile <dikw.yml>`.
- **`dangling_provenance` drift lint kind — flag a K/W page citing a deleted source
  (read-only).** A new deterministic `lint` kind that flags a `knowledge/` (K) or
  `wisdom/` (W) page whose `sources:` **provenance** edge points at a source file that
  no longer exists on disk. It is **read-only — surfaced, never auto-repaired**: there
  is no fixer (like `duplicate_title`, `lint propose` reports it for human triage and
  lands every issue in `skipped`), because the `sources:` frontmatter is the user's to
  edit (ADR-0001's non-cascade design — delete never rewrites another page's content).
  Disk is the source of truth (ADR-0005), so detection stats the *file*, not the
  `documents` projection: a source present on disk but not yet `ingest`-ed (no active D
  row) is **not** dangling — there the fix is `ingest`, not editing frontmatter. A
  provenance path that escapes the base is dangling and its external target is never
  stat-ed. Runs in the default `lint` scan, sharing the per-page provenance read with
  `missing_provenance` (zero extra storage round-trips); suppressible per page via
  `lint: {skip: [dangling_provenance]}`. Final slice of ADR-0005
  (filesystem-as-source-of-truth) — the arc (the `delete` verb + `missing_file` /
  `untracked_file` / `stale_index` / `dangling_provenance` drift kinds) is now complete,
  and `docs/design.md` gains a "Disk is the source of truth" invariant section.
- **`stale_index` + `untracked_file` drift lint kinds — re-project hand-edited /
  hand-written K/W pages (and unlock hand-authored knowledge pages as first-class).**
  Two new deterministic `lint` kinds, both fixed by one `ReindexPageFixer`:
  `stale_index` flags an *active* `knowledge/` (K) or `wisdom/` (W) row whose on-disk
  body hash no longer matches the indexed `hash` (a hand-edit outside dikw);
  `untracked_file` flags a `.md` / `.markdown` file under `knowledge/` or `wisdom/`
  with no active row (hand-written, or restored outside dikw). Both propose a single
  `reindex_page` op that re-projects the *current* on-disk bytes through
  `persist_knowledge` / `persist_wisdom` — re-chunk, re-link, re-provenance,
  inline-or-deferred re-embed — **without rewriting the file** (disk is the source of
  truth, ADR-0005) and **without re-running `synth`** (so a hand-edit is preserved, not
  regenerated from the D-source). Run in the default `lint` scan; fix with
  `dikw client lint propose --rule stale_index` (or `untracked_file`) →
  `dikw client lint apply <task_id>`. `untracked_file` closes the "hand-write a K page,
  the engine never indexes it" gap and makes hand-authored pages first-class;
  `stale_index` closes the "edit a K/W file on disk, the storage projection silently
  drifts" gap. Detection is near-free: `stale_index` reuses the per-page read the
  other lexical checks already do (no separate mtime-prefiltered hashing pass), and
  `untracked_file` is a cheap disk walk (stat + membership, no read) rooted at
  `knowledge/` + `wisdom/` so the sibling `trash/` / `.dikw/` / `assets/` trees are
  naturally excluded and `.gitkeep` / non-markdown files never trip. Both are K/W-only
  (D-layer adds/edits stay `ingest`'s job); a page failing its re-projection is
  deactivated and surfaced via `ApplyReport.persist_errors`, successes under
  `ApplyReport.reindexed_documents`. Third slice of ADR-0005 (`dangling_provenance`
  is the fourth, above). This supersedes the never-built `dikw client reindex <path>` — the
  reindex story is now `dikw client lint propose --rule stale_index` (or
  `--rule untracked_file`) followed by `dikw client lint apply <task_id>`.
- **`missing_file` drift lint kind — purge orphaned document rows (D/K/W).** A new
  deterministic `lint` kind (with `MissingFileFixer`) that detects an *active*
  `documents` row whose backing file is gone from disk — a `sources/` (D),
  `knowledge/` (K), or `wisdom/` (W) file deleted outside dikw — and proposes a
  single `purge_document` op that drops the orphaned row + its outgoing edges via
  `Storage.delete_document`. Runs in the default `lint` scan; fix it with
  `dikw client lint propose --rule missing_file` → `dikw client lint apply <task_id>`.
  Closes the original gap where deleting a source file left its row stuck at
  `active=True` forever (`run_lint` never scanned D rows). Inbound `[[wikilink]]`s
  from live pages are left to surface as `broken_wikilink` (delete_document clears
  only outgoing edges; the kind never rewrites a user's page); a truly dangling edge
  (both ends purged) clears itself. The op carries the resolved `layer`, re-checks
  at apply time that the file is still absent and the row still exists (propose→apply
  race / restored-file safety), and reports purged paths under
  `ApplyReport.purged_documents`. Second slice of ADR-0005
  (filesystem-as-source-of-truth); `untracked_file` / `stale_index` /
  `dangling_provenance` land in follow-ups.
- **`dikw client delete <path>` — first-class document deletion (D/K/W).** A new
  immediate verb (`api.delete_page` / `POST /v1/base/delete`) that deletes any
  registered document — a `sources/` file, a `knowledge/` page, or a `wisdom/`
  page — by path: it purges the storage row + its outgoing links/provenance
  (`Storage.delete_document`) and soft-deletes the on-disk file to
  `<base>/trash/<layer>/<rel>` with an audit `trashed:` block (recover with a plain
  `mv` back into place). It is symmetric with `wisdom write`: explicitly-targeted,
  immediate (no propose/apply — `trash/` is the safety net), `--wait` by default,
  `--reason` for an audit note. Closes the gap where deletion existed only as a side
  effect of the `lint` `orphan_page`/`non_atomic_page` fixers (K-layer stubs only) —
  arbitrary K pages and all D/W documents were previously undeletable.
  Inbound `[[wikilink]]`s from live pages are left dangling and surface as
  `broken_wikilink` on the next `dikw client lint` — delete never rewrites another
  page. First slice of ADR-0005 (filesystem-as-source-of-truth); the drift `lint`
  kinds (`missing_file` / `untracked_file` / `stale_index` / `dangling_provenance`)
  land in follow-ups. Internally, the soft-delete primitive `move_to_trash` was
  promoted out of `domains/knowledge/lint_fix.py` into the shared, layer-agnostic
  `domains/trash.py` so D/W deletes reuse it.

### Fixed

- **OTel validation stack now runs on arm64 (Apple Silicon).** The
  `docs/observability/docker-compose.yml` collector was pinned to
  `otel/opentelemetry-collector-contrib:0.116.0`, whose **arm64** binary is
  dynamically linked (`interpreter /lib/ld-linux-aarch64.so.1`) while the image
  is `FROM scratch` — so on Apple Silicon the container exited immediately with
  `exec /otelcol-contrib: no such file or directory` and the stack came up with
  jaeger/prometheus/grafana healthy but **zero traces**. Bumped to `0.117.0`,
  the nearest release that restored the static arm64 build (verified: boots
  clean against the existing `otel-collector-config.yaml`); amd64 was
  unaffected. This also fixes `tools/e2e_verify.py --observe` on arm64, which
  drives this same compose file.

- **Synth front-matter is whitelisted to `tags`; `write_page` guards reserved
  keys.** Enforces in code the forbidden-key policy 0.5.3 added to the synth prompt
  (the *"Synth forbids `sources`/`lint` in emitted front-matter"* entry below):
  that change only reworded the prompt — the parser still routed every non-`tags`
  key into `extras` and `write_page` merged it over the engine's authoritative
  fields, so a disobedient LLM (or a hand-edited file flowing through lint-apply's
  `update_page`) could still override `sources`/`category`/`id`, inject a `lint:`
  block that suppressed lint on a fresh page, or — via a `handler`/`content` key
  colliding with `frontmatter.Post(**meta)` — silently collapse the whole file to a
  literal string. Now: the synth parser (`_parse_one_page_block`) drops every
  non-`tags` front-matter key the LLM emits (`title` comes from the body `# H1`,
  `category`/`slug` from the `<page>` attributes, the rest engine-managed), covering
  every LLM-sourced page (synth fan-out + the lint grounded/split/merge fixers that
  share the parser) at one point; and the shared `write_page` sink filters caller
  `extras` against `_RESERVED_FRONTMATTER_KEYS` and assigns metadata via
  `post.metadata.update`, mirroring the W-layer `write_wisdom_file` guard. User
  `extras` (e.g. an Obsidian `aliases:` list) still pass through, and the `lint:`
  block written by `orphan_page.mark_as_leaf` is deliberately not reserved.
  Behaviour-preserving for conformant synth output (which emits only `tags`).

### Security

- **Raise the `python-multipart` floor to `>=0.0.31` (security floor) — clears the
  open Dependabot form-parsing advisories.** The declared floor was `>=0.0.26`, which
  let the published wheel resolve a `python-multipart` vulnerable to the
  `multipart/form-data` resource-exhaustion / DoS chain (GHSA-5rvq-cxj2-64vf and the
  `<0.0.31` follow-ups GHSA-v9pg-7xvm-68hf / GHSA-6jv3-5f52-599m / GHSA-vffw-93wf-4j4q).
  The lock was already bumped to `0.0.31` by Dependabot (#209), but the manifest floor
  still permitted a downstream install below the fix; raising it hardens the
  published-wheel contract and, by re-touching `uv.lock`, lets GitHub's dependency
  graph re-ingest the already-patched resolution (`python-multipart 0.0.31`,
  `starlette 1.3.1`) so the eight stale alerts auto-resolve. Starlette's matching
  `request.form()` limit-bypass / DoS fixes (≥1.3.1, GHSA-82w8-qh3p-5jfq and the
  `<1.1.0` advisories) already ship transitively via `fastapi` (locked) — it is not a
  direct dependency, so no direct pin is added. Metadata-only: no resolved-version or
  code change (`uv.lock` diff is the recorded root specifier alone).

## 0.5.3 — OpenTelemetry observability arc (traces + metrics + logs) + synth prompt restructure

### Added

- **Observability docs + validation stack (PR5 of the OTel arc — arc complete).**
  New [`docs/observability.md`](docs/observability.md) operator cookbook: the full
  span/metric/log inventory, how to enable telemetry at server (`dikw.yml`
  `telemetry:` section) and client (`OTEL_*` env) ends, the `DIKW_LOG_FORMAT=json`
  log↔trace correlation, a dashboard PromQL cookbook (GenAI token spend, LLM
  latency, Anthropic cache hit-ratio, pipeline volume), and the zero-cost-when-off
  guarantee. New `docs/observability/` validation stack — a `docker-compose.yml`
  wiring **OTel Collector (`:4318`) → Jaeger (traces) + Prometheus (metrics) →
  Grafana (dashboards)** with pre-provisioned datasources, so an operator can run
  one `docker compose up -d`, point dikw at `http://localhost:4318`, and see a full
  client → server → task → engine → provider trace end to end. `telemetry.py` is
  now in the module maps (CLAUDE.md + `docs/architecture.md`) with a fourth "seam
  on purpose"; `docs/providers.md` documents GenAI token metering. Docs-only — no
  code change, no behavior change. This closes the five-PR OpenTelemetry arc
  (traces + metrics + logs across server and client).
- **Log ↔ trace correlation + JSON logging (PR4 of the OTel arc).** New
  `DIKW_LOG_FORMAT` env var: the default `text` keeps the human-readable terminal
  formatter byte-for-byte; `json` opts into one JSON object per log record
  (`ts`/`level`/`logger`/`message`, plus any `extra={…}` fields and an
  `exception` traceback) — the machine-readable form a log aggregator parses.
  When telemetry is active, records emitted inside a span also carry
  `trace_id`/`span_id`/`service`, so a log line pivots straight to its trace.
  `configure_telemetry` (server) and `configure_client_telemetry_from_env`
  (client) wire the OTel `LoggingInstrumentor` via a log hook with
  `set_logging_format=False` + `enable_log_auto_instrumentation=False`, so
  `init_logging` keeps full handler/format ownership and no OTLP log-export
  handler is bolted onto the root logger (log export stays deferred). Degrades
  gracefully without the `[otel]` extra or outside a span (no trace fields, no
  crash); the `text` default is unchanged. Like `DIKW_LOG_LEVEL`, it's an env var
  (CLI parses before any base loads), not a `dikw.yml` field.
- **OpenTelemetry dikw-domain metrics (PR3b of the OTel arc).** The engine now
  emits domain counters + duration histograms mapped from the per-call report
  DTOs and existing instrumentation, so a Prometheus/Grafana dashboard sees
  ingest/synth/embed/retrieve/task volume without parsing logs. All flow through
  the same `_meter_provider` gate as the PR3a GenAI metrics (no-op when the
  `[otel]` extra is absent or telemetry is off) via five `record_*` helpers in
  `telemetry.py` that take plain scalars (so the seam stays decoupled from the
  DTOs):
  - `dikw.ingest.files` (tag `dikw.result` = `added`/`updated`/`unchanged`),
    `dikw.ingest.chunks`, `dikw.ingest.errors` (tag `dikw.error.kind`) —
    emitted once at `ingest`'s return.
  - `dikw.synth.pages` (tag `dikw.result` = `created`/`updated`),
    `dikw.synth.unresolved_wikilinks`, `dikw.synth.persist_errors` — at
    `synthesize`'s return.
  - `dikw.embed.chunks` / `dikw.embed.skipped` / `dikw.embed.retries` — from the
    shared chunk-embed consume seam (`consume_embedding_stream`), so they cover
    every layer's inline embed (D/K/W), not just ingest.
  - `dikw.retrieve.leg.duration` (`s`, tag `dikw.retrieval.leg`) — each fusion
    leg's wall-clock, recorded from the existing `dikw.retrieve.leg` span seam.
  - `dikw.task.duration` (`s`, tags `dikw.op` + `dikw.status` =
    `ok`/`error`/`cancelled`) — one point per background task; unlike the GenAI
    op-duration metric, a cancelled task keeps its duration on its own status
    series.
  Purely additive instrumentation — retrieval results and synth output are
  byte-identical with telemetry on or off (the leg-duration wrapper only times
  the existing await).
- **OpenTelemetry GenAI metrics (PR3a of the OTel arc).** The server now exports
  GenAI metrics over OTLP/HTTP (`/v1/metrics`) alongside traces — surfacing the
  LLM/embedding token usage that was previously discarded. `configure_telemetry`
  registers a `MeterProvider` + `PeriodicExportingMetricReader` + OTLP/HTTP metric
  exporter (with semconv-advised histogram bucket views), and every provider call
  (chat + embeddings) emits two histograms straight from the existing
  `gen_ai_span` / `trace_llm_stream` seam, so **no provider-side code changed**:
  - `gen_ai.client.token.usage` (`{token}`) — one point per token class, tagged
    `gen_ai.token.type` (`input` / `output`, plus `cache_read` / `cache_creation`
    for Anthropic prompt caching), `gen_ai.operation.name`, `gen_ai.system`,
    `gen_ai.request.model`.
  - `gen_ai.client.operation.duration` (`s`) — request latency, tagged
    `error.type` on a failed call; a cancelled / abandoned stream records no point
    (its cut-short time would skew the latency series).
  FastAPI HTTP-server metrics flow for free from the same meter provider. All a
  no-op when the `[otel]` extra is absent or telemetry is off; the remote `dikw
  client` stays trace-only (it makes no LLM calls). dikw-domain counters land in a
  follow-up (PR3b). Touches only `telemetry.py` — behavior-preserving for the
  existing span paths.
- **OpenTelemetry tracing — client→server `traceparent` propagation (PR2c of the
  OTel arc).** The `dikw client` CLI now joins the same trace as the server it
  calls. Because the remote client has no `dikw.yml`, its telemetry is **env-only**:
  set the standard `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`)
  before a `dikw client …` command and — when the `[otel]` extra is installed and
  `OTEL_SDK_DISABLED` is unset — the CLI wires a `TracerProvider` (`service.name`
  from `OTEL_SERVICE_NAME`, default `dikw-client`) and the global httpx
  instrumentation, so the outbound request injects a W3C `traceparent` header that
  `dikw serve`'s FastAPI instrumentation adopts as the parent span — one trace now
  spans client → HTTP server → background task → engine op → provider call. The
  bootstrap is gated to the `client` subgroup (local `version` / `init` / `serve` /
  `auth` commands pay zero cost, and `serve` keeps wiring its own server-side
  telemetry from `dikw.yml`) and a plain `dikw client` invocation with no `OTEL_*`
  env stays a no-op. A short-lived CLI flushes the exporter via an `atexit` hook.
- **OpenTelemetry tracing — engine op-level spans (PR2b of the OTel arc).**
  The engine verbs now open their own op span so a trace shows the full tree —
  and so direct / eval callers (which have no server task span) still get a
  root. `ingest` / `synthesize` / `lint propose` / `lint apply` open a
  `dikw.{ingest,synth,lint.propose,lint.apply}` span (carrying `dikw.layer` +
  `dikw.op`); `retrieve` opens `dikw.retrieve` (`dikw.retrieve.limit` +
  `dikw.retrieve.hit_count`). Inside hybrid search, **each fusion leg** (BM25,
  vector, asset, graph) now emits a `dikw.retrieve.leg` span with
  `dikw.retrieval.leg` + `dikw.retrieve.leg.hit_count` — the legs are pure
  in-process work with no provider call, so they were dark to the PR2a
  `gen_ai.*` spans; the span opens inside each leg's own task for accurate
  concurrent-leg timing. New `op_span` / `traced_op` telemetry seams centralise
  the cancel (`dikw.cancelled`, not error) / `GeneratorExit` / error / OK
  outcome handling. All no-ops when the `[otel]` extra is absent, and purely
  span-wrapping — retrieval results and synth output are byte-identical with
  telemetry on or off. Per-source / per-group / per-batch sub-spans are
  deferred (they are largely covered by the PR2a `gen_ai.*` provider spans);
  the client→server `traceparent` leg follows in PR2c.
- **OpenTelemetry tracing — task linking + provider spans (PR2 of the OTel arc).**
  Building on the PR1 no-op-safe seam, `configure_telemetry()` now registers the
  global **httpx instrumentation**, so every server-side provider HTTP call is
  auto-traced and carries the W3C `traceparent` header. Each background task
  opens a `dikw.task.<op>` **root span linked** back to the submitting request
  span (the OTel idiom for request-triggered fire-and-forget work — the detached
  task outlives the request span), held current across the run so downstream
  spans nest under it, stamped with `dikw.op` / `dikw.task_id` / `dikw.base_id`;
  a user-requested cancel is recorded as `dikw.cancelled` rather than an error
  status, and a consumer that breaks out of a streamed call early
  (`GeneratorExit`) is likewise a graceful terminal, not an error. Every LLM
  provider (`openai_compat`, `anthropic_compat`,
  `openai_codex`) emits a `gen_ai.chat` span carrying `gen_ai.request.model` +
  `gen_ai.usage.{input,output}_tokens` (surfacing the token usage the synth call
  site previously discarded, plus Anthropic `cache_read`/`cache_creation`
  tokens), and both embedders emit a `gen_ai.embeddings` span. All no-ops when
  the `[otel]` extra is absent. Engine op-level spans (ingest/synth/retrieve/
  lint sub-spans) and the client→server `traceparent` leg follow in the next PRs.
- **OpenTelemetry observability foundations (PR1 of a 5-PR arc).** New optional
  `[otel]` extra (`uv sync --extra otel`) and a `dikw_core.telemetry` seam:
  `get_tracer()` / `get_meter()` accessors that instrument against the OTel API
  and degrade to **zero-overhead no-ops** when the extra is absent, the
  `dikw.*` semantic-convention attribute keys, and an idempotent
  `configure_telemetry()` SDK bootstrap (TracerProvider + OTLP/HTTP exporter,
  `ParentBased(TraceIdRatio)` sampling). A new `telemetry:` section in
  `dikw.yml` (`enabled` / `endpoint` / `service_name` / `sample_ratio`, all
  **off by default**) drives server-side export; the standard `OTEL_SDK_DISABLED`
  kill-switch and `OTEL_EXPORTER_OTLP_ENDPOINT` fallback are honoured. `dikw serve`
  bootstraps it from the lifespan (after cfg load) and auto-instruments FastAPI so
  every `/v1/*` request becomes an HTTP server span. **No engine spans/metrics
  yet** — this PR only lands the no-op-safe wiring; tracing of the ingest/synth/
  retrieve/provider seams, GenAI token metrics, and log↔trace correlation follow
  in later PRs.

### Changed

- **Synth prompt two-tier restructure (SP spine + slim UP).** `DEFAULT_SYNTH_SYSTEM`
  — the cached system prompt shared by the synth fan-out leg and the
  non-atomic-page lint splitter — is rewritten from a ~100-word paragraph into a
  structured **standing-policy spine**: six named invariants (atomicity,
  faithfulness, reuse over regeneration, closed taxonomy, honest linking, source
  language). It keeps the deliberate **link-density-as-ceiling** posture
  ("precisely-linked", never "dense"), and restores two rules a prior draft had
  dropped — *never translate a source-language concept* and *never add precision
  the source does not state* ("recent growth" ↛ "grew 40% in 2023"). The
  `prompts/synthesize.md` user prompt is correspondingly **slimmed**: the prose
  that merely restated those principles (the Atomicity, Category, and
  Faithfulness sections) is gone, leaving the operational detail each invariant
  defers to — the quantitative length norms, the 2–4-links/500-chars ceiling, the
  tag vocabulary, the slug/pinyin mechanics — plus the two worked examples, the
  exact output format, and the per-call inputs. **Single source of truth:** each
  rule now has one home, so the cached SP and the per-call UP cannot drift — bar
  two rules deliberately restated in both tiers (the source-language rule, a
  second-line defence if the UP is ever truncated; and "category omission is a
  last resort", reminded at the point of emission), each pinned in both by a
  guard test. The `synthesize` placeholder
  set, output markers, and `## Knowledge-base context` container are unchanged, so
  `synth.prompt_path` overrides keep validating and `_contract.py` is untouched.

- **Synth forbids `sources`/`lint` in emitted front-matter.** The output-format
  forbidden-key list now names `sources` and `lint` alongside
  `title`/`id`/`category`/`created`/`updated`. The parser routes every non-`tags`
  front-matter key into `extras`, and `page.py` applies the engine's authoritative
  `sources` then `meta.update(extras)` — so an LLM that emitted `sources:`
  overwrote the real provenance, and `lint:` injected a leaf-acknowledgement block
  that suppressed lint on a fresh page. Both are now explicitly engine-owned.

## 0.5.2 — synth-quality measurement + prompt tuning + post-synth self-check

### Changed

- **Synth prompt quality pass (SP + layout, PR2).** Two structural changes on
  top of PR1's content revisions: (1) `DEFAULT_SYNTH_SYSTEM` — the cached
  system prompt shared by the synth fan-out leg and the non-atomic-page lint
  splitter (the orphan-merge / broken-wikilink fixers carry their own) —
  no longer pushes "dense [[wikilinks]]" / "favour many tightly-linked atomic
  pages", which fought the user prompt's link-density-as-ceiling and
  "complete, then concise" framing; it now asks for precise linking and
  complete single-subject pages. (2) The `synthesize` template moves **every**
  `{placeholder}` into a dynamic tail zone (`## Category list` → `## Task` →
  `## Knowledge-base context` → source block) after the static instruction
  sections, so the instruction prefix is byte-stable across calls and
  OpenAI-compatible prefix caching (and codex) cover it; the system prompt
  stays separately cached via `cache_control` on Anthropic-compatible
  providers. Placeholder set, output markers, and the `## Knowledge-base
  context` container requirement are all unchanged — existing
  `synth.prompt_path` overrides keep validating. Also sweeps the PR1-deferred
  wording fixes: the Output-format bullet no longer re-legitimizes category
  omission ("last-resort case described under Category"), the duplicate rule
  scopes itself to the two existing-page lists and explicitly exempts
  `Priority targets` (pages that do **not** exist yet), and stale
  "existing-pages section above" references are gone. Category guidance and
  the worked examples deliberately stay in the overridable template —
  per-base taxonomy customization goes through `synth.prompt_path` (see
  `docs/providers.md`), never through mechanical example rewriting.

- **Synth prompt quality pass (UP, PR1).** Six targeted revisions to
  `prompts/synthesize.md` + the prompt-assembly rendering, each tied to a
  measured weakness in the `evals/BASELINES.md` MiniMax-M3 runs:
  the two worked examples now model the `category=` attribute (examples that
  omitted it taught the model to omit it → `fallback_ratio_max` 0.308–0.47) and
  omission is framed as a last resort; every `[[wikilink]]` target must be an
  existing page, a page emitted in the same response, or a deliberately
  page-worthy forward link — passing mentions are excluded, and the 2–4 links
  /500 chars guidance is now a ceiling, not a quota (→ broken-link drift,
  `wikilink_resolved_ratio` 0.31–0.71); faithfulness now requires every
  specific (number, date, name, causal claim) to be traceable to the section
  text (→ `fact_entailment_ratio` `partial` verdicts); atomicity gains an
  "atomic ≠ thin" counterweight and rule 2 becomes "be complete, then concise"
  (→ judge `completeness` 3.4/5 vs ≥4.8 on the other three dims); fan-out
  gains a truncation defence (most-important page first, never open a block
  you cannot finish); and the dynamic prompt sections now nest as H3
  (`### Already created in this batch` / `### Existing knowledge pages` /
  `### Priority targets (create if relevant)`) under a neutral
  `## Knowledge-base context` heading, so the priority-create *directive* no
  longer sits under a heading claiming those pages exist. Placeholder set and
  output markers are unchanged, but the `synthesize` override contract now
  also requires the `## Knowledge-base context` container — a
  `synth.prompt_path` override written against the old layout fails
  `dikw client check` loudly instead of silently nesting the H3 sections
  under the wrong parent heading; add the H2 line above
  `{existing_pages_section}` to migrate. Note: the `non_atomic_page` lint
  fixer's LLM split resolves the same template, so the revised guidance
  reaches that (opt-in) path too — unmeasured by this PR's A/B, which covers
  the synth fan-out only.

### Added

- **`synth/semantic_atomicity_ratio` — LLM judge for one-concept-per-page
  atomicity.** `atomicity_score` is a *form* heuristic (body chars, H1/H2
  counts, distinct wikilink targets, tag domains) — blind to a short paragraph
  stuffed with three unrelated concepts, and conversely to a thorough
  single-concept page that trips the length counters. The new judge reads each
  page's title + body alone and answers `yes`/`partial`/`no` (one concept /
  dominant concept plus a developed tangent / multiple concepts bolted
  together → `1.0`/`0.5`/`0.0`); `[[wikilink]]` references and passing
  mentions never count against atomicity. Opt-in per dataset via
  `judge.semantic_atomicity_enabled: true` **and** `--judge` (`$0` unless both
  are on); informational (never gated), surfaced as
  `SynthEvalReport.semantic_atomicity_summary` with a bootstrap 95% CI and
  mirrored into `informational` for the A/B harness. The packaged `mvp`
  dataset enables it. This completes the Phase 0b judge-metric set
  (entailment / category / wikilink / atomicity).

- **`synth/wikilink_correctness_ratio` — LLM judge for resolved-link referent
  correctness.** `wikilink_resolved_ratio` counts how many `[[wikilinks]]`
  resolved; it is blind to whether each resolved link points at the *right*
  page — the fuzzy resolver makes a wrong-referent link (`[[Mercury]]` in a
  planetary context resolving to the chemical-element page) look *more*
  resolved, never less. The new judge reads each resolved page→page link in its
  body context next to the target page the engine resolved it to (the `links`
  table is the truth, fuzzy results included) and answers `yes`/`partial`/`no`
  (right referent / related-but-imprecise / wrong thing → `1.0`/`0.5`/`0.0`).
  Opt-in per dataset via `judge.wikilink_correctness_enabled: true` **and**
  `--judge` (`$0` unless both are on); informational (never gated), surfaced as
  `SynthEvalReport.wikilink_summary` with a bootstrap 95% CI and mirrored into
  `informational` for the A/B harness. The packaged `mvp` dataset enables it.

- **`synth/fact_entailment_ratio` is now a conditional (judge-only) gate.** The
  LLM entailment metric — previously informational-only — can now be declared as
  a `synth/<metric>` threshold in a dataset and is enforced by `run_synth_eval`
  **only when the judge actually ran**. A non-judge run (the hermetic CI synth
  half, or a plain `--eval synth` without `--judge`) drops the threshold instead
  of recording a spurious `observed=None` miss, so the gate bites on real-LLM
  `--judge` acceptance runs only. The ratio still mirrors into `informational`
  (it is never promoted into the deterministic `metrics` set). The packaged
  `mvp` dataset gates it at `0.55`, calibrated against the 2026-06-05 real-LLM
  run (observed `0.775`, 95% CI `[0.65, 0.90]`, n=20 on MiniMax-M3 — floor set
  below the CI lower bound to absorb judge noise). See `evals/BASELINES.md`.
- **`dikw client synth --verify --judge` — report-only grounding leg.** Adds the
  one probabilistic check the deterministic `--verify` legs can't make: it
  samples this run's K-page claims, grounds each against the source chunks the
  page cites (reusing the eval grounding pipeline), and asks the synth LLM
  whether the evidence entails the claim — surfacing an entailment ratio + 95%
  CI. It is **report-only**: the ratio is *never* folded into the pass/fail
  verdict (an LLM judge is noisy; the pass/fail call over the ratio belongs to
  the orchestrating agent/skill, not a hard CLI gate). Needs both an embedder and
  an LLM; when either is missing it **loud-skips** (`grounding_requested` true,
  `grounding_checked` false) rather than silently reporting nothing, and a
  failure inside the leg degrades to that same skip instead of failing the synth.
  `--judge` implies `--verify`. The sample size is `synth.verify_judge_sample`
  (default `25`, matching `eval.judge.recommended_judge_sample()`).
- **`dikw client eval --against` / `--write-baseline` — machine-readable
  regression gate.** `--write-baseline <path>` dumps a run's metrics to a
  committed JSON; `--against <path>` re-runs the eval and exits 1 when any metric
  moved the wrong way past the baseline's `tolerance` (default `0.02`). The
  comparison is **direction-aware** — a `_max` metric (e.g.
  `synth/fallback_ratio_max`) regresses when it *rises* — mirroring the engine's
  naming convention. Both imply `--wait` and require a single `--dataset` + one
  `--eval` mode (so the result carries one metrics set); they are mutually
  exclusive. This is a single-run regression *gate*, not an A/B significance
  test — keep the tolerance tight for deterministic retrieval evals and generous
  for LLM-driven synth evals. The statistical A/B path (Welch t-test) stays in
  `evals/tools/ab_experiment.py`. Client-only (pure `dikw_core.client.baseline`);
  baselines live under `evals/baselines/` (see its README).
- **`title_slug_quality` lint — deterministic K-page title/slug hygiene.** A new
  read-only lint kind, also wired as a `dikw client synth --verify` gated leg,
  that flags three zero-false-positive defects on a knowledge page: a body with
  no usable `# Title` heading (absent / blank / punctuation-only — a CJK title
  is *not* punctuation-only), a frontmatter `title:` that disagrees with the body
  `# H1` (the genuine title drift — `write_page` always serialises the two equal,
  so a divergence is a hand-edit to one side), and a degenerate `untitled`
  filename slug (only reachable when `slugify` collapsed a non-ASCII title the LLM
  gave no ASCII/pinyin slug for). It is deliberately **not** a
  `slugify(title) == stem` comparison — slugs are LLM-chosen and intentionally
  diverge from `slugify(title)` (stop-word dropping, pinyin for CJK), and
  wikilinks resolve by title, so that comparison would red-flag the engine's own
  correct output; whether a well-formed title is *too generic* is a probabilistic
  judgement left to a future LLM-judge leg, never this lexical lint. Scoped to
  `Layer.KNOWLEDGE` (hand-written wisdom is exempt) and suppressible per-page via
  `lint: {skip: [title_slug_quality]}`.
- **`dikw client synth --verify` — post-synth self-check.** After synth writes
  K pages, `--verify` runs a deterministic, no-extra-LLM check scoped to just
  this run's created/updated pages and emits one PASS/FAIL verdict (exit
  non-zero on fail; implies `--wait`). Three gated legs: **persist**
  (`SynthReport.persist_errors == 0`), **lint** (a full-base `run_lint` whose
  results are filtered to this run's pages — the scan needs the whole base to
  resolve wikilinks — gated on `broken_wikilink` / `duplicate_title` /
  `non_atomic_page` / `uncategorized` / `missing_provenance` /
  `title_slug_quality`), and
  **duplicate** (semantic `duplicate_ratio_max` over
  this run's page bodies ≤ `synth.verify_max_duplicate_ratio`, default `0.05`,
  at cosine tau `synth.verify_duplicate_cosine_tau`, default `0.85`).
  `orphan_page` is surfaced but **not** gated (a fresh page is legitimately
  orphan until cited). With no embedder the duplicate leg is **skipped loudly**
  (a warning, never a silent pass). The result carries a `SynthVerifyReport`
  under `SynthReport.verify` / the `/v1/synth` task result's `verify` key.
  Purely additive — `--verify` only READS synth output, it never changes the
  generated pages.
- **synth-quality measurement foundation (Phase 0a).** Tooling that lets a
  later prompt/pipeline tuning PR *prove* it helped rather than eyeball a
  single noisy run. No synth/retrieval generation behavior changes — this is
  measurement only.
  - **Deterministic synth-eval diagnostics** in `dikw client eval --eval synth`
    (informational, never gated): `synth/source_chunk_coverage`
    (under-generation — source chunks that no page claim grounds), `synth/fallback_ratio_max`
    (taxonomy miscalibration — share of pages filed under the fallback category),
    `synth/slug_merge_ratio_max` (over-generation — fraction of fan-out pages the
    slug dedup collapsed), and a per-category `category_distribution`. The `_max`
    suffix marks the two lower-is-better metrics for the direction convention.
  - **`SynthReport.slug_merge_count`** — a raw run-total of pages collapsed by
    `dedup_pages_by_slug`, surfaced in the `dikw client synth` summary table as
    a visible over-generation signal.
  - **LLM-judge bootstrap 95% CIs** — `JudgeSummary` now carries a deterministic
    `ci_<dimension>` per score so a small-sample mean isn't mistaken for a real
    move.
  - **A/B experiment harness** (`evals/tools/ab_experiment.py`, developer tooling)
    — runs the same synth eval N times per arm and compares them with a Welch
    two-sample t-test, Cohen's d, and a direction-aware ship gate (`p < p_max`
    **and** `improvement > effect_min`). Pure-Python stats (no scipy). `collect`
    runs the arms (live LLM); `compare` is offline.
  - **`--target-tokens` override** on `ab_experiment collect` (threaded through
    `run_synth_eval` / `_materialise_base`, default = production 3600) — fans a
    small packaged corpus into multiple groups so grouping-sensitive synth
    changes (the Phase 2 priority-create / existing-pages features) can be A/B'd
    on `mvp` instead of needing a bespoke large-source dataset. No behavior
    change when unset.
- **`synth/fact_entailment_ratio` — LLM grounding judge (Phase 0b).** The
  embedding `fact_grounding_ratio` reduces to a cosine, so "GPT-4 is 4x faster
  than GPT-3" (a fabricated ratio) and "GPT-4 is faster than GPT-3" (supported)
  land in the same band — it cannot tell a true claim from a sharpened one. The
  new judge pairs each page claim with its nearest source chunk (reusing the
  grounding argmax — no re-embedding) and asks an LLM whether the evidence
  *entails* the claim (`yes`/`partial`/`no` → `1.0`/`0.5`/`0.0`), catching
  invented numbers, dates, ratios, superlatives, causal links, and contradicted
  claims. Surfaced as an informational metric (never gated) with a deterministic
  bootstrap 95% CI on `SynthEvalReport.entailment_summary`. **Opt-in and `$0` by
  default**: runs only when `--judge` is set **and** the dataset declares
  `judge.entailment_grounding_enabled: true`. New overridable-by-nobody prompt
  `prompts/eval_judge_entailment.md`.
- **`synth/category_correctness_ratio` — LLM taxonomy judge (Phase 0b).** The
  embedding metrics see *where* synth filed each page (`category_distribution`,
  `fallback_ratio_max`) but never whether the filing is *right* — a named-tool
  page mis-filed under `concept` looks identical to a correctly filed one. The
  new judge re-derives the best category independently from the page body and
  the closed declared set (fallback included), then scores synth's actual
  choice: exact `1.0`, a judge-acknowledged co-equal `0.5`, wrong `0.0`. The
  closed set is enforced at parse time — a verdict naming an undeclared category
  is a parse failure, never a silent re-file (mirrors synth's own refusal).
  Informational (never gated) with a bootstrap 95% CI on
  `SynthEvalReport.category_summary`. **Opt-in and `$0` by default**: runs only
  when `--judge` is set **and** the dataset declares
  `judge.category_correctness_enabled: true`. New prompt
  `prompts/eval_judge_category.md`.
- **`--judge-sample auto` — calibrated judge sample size (Phase 0b).** Both
  real-LLM judge calibrations *cleared* the ±0.2 CI half-width target — but
  category only barely (entailment n=20 → ±0.13, category n=8 → ±0.19), and only
  because its scores were low-variance. At n=8 the worst-case (50/50) half-width
  is ±0.35, so a metric nearer an even split would have blown past ±0.2: the
  target was being met by luck, not design. That raised "how many items
  *guarantee* a trustworthy ratio regardless of variance?". A [0,1] judge
  ratio's bootstrap 95% CI half-width is at most `1.96 * 0.5 / sqrt(n)`, so
  `n ≥ 25` guarantees ≤ ±0.2 for *any* score distribution — a worst-case bound
  that holds on every dataset, so no per-corpus sweep can push it higher. `dikw
  client eval --judge-sample auto` resolves (server-side) to
  `eval.judge.recommended_judge_sample()` = that `n`, clamped to `[5, 50]`;
  datasets with fewer items are judged in full.

### Changed

- **LLM prompts now name and introduce `dikw-core`.** Every engine prompt that
  referenced the product (`synthesize.md`, the two `lint_fix_*` prompts, the
  three `eval_judge_*` prompts, plus the synth / lint-fix / merge system strings)
  now writes the name as code-formatted `` `dikw-core` `` and, in the six prompt
  templates, adds one shared appositive — *"an AI-native knowledge engine that
  refines raw sources up the Data → Information → Knowledge → Wisdom (DIKW)
  pyramid"* — so the model has product context instead of a bare unexplained
  name. Descriptive grounding only: no instruction, output-format, placeholder,
  or behavior change, and no new LLM calls.
- **`provider.llm_max_tokens_synth` default 2048 → 3072 (Phase 1).** The old
  default left only ~512 tokens per page at `max_pages_per_group=4`, so a dense
  fan-out group could truncate mid-page (losing the last page or its closing
  body). 3072 gives ~768/page. Override per-base in `dikw.yml`; reasoning models
  still need `>= 8192` (their hidden chain-of-thought draws on the same budget —
  see `docs/providers.md`). A new test pins the default to leave `>= 512` tokens
  per page so a future `max_pages_per_group` bump can't silently re-introduce the
  clip. `dikw client synth` now also logs the per-page budget at DEBUG.
- **Enriched synthesis prompt (Phase 1).** `prompts/synthesize.md` and the
  default system prompt now frame the K layer as a Zettelkasten of small,
  densely-linked atomic notes and add the guidance the heuristics implied but the
  prompt never stated: page-size (a `< ~200`-char body is **rarely** worth its
  own page — fold it into a neighbour; typical pages run 300–1500 chars), inline
  wikilink density (`~2–4` links per 500 chars, never a trailing "see also"
  list), a reusable tag vocabulary (`entity`/`concept`/`process`/…, one namespace
  domain per page), and an explicit duplicate-vs-facet rule. The existing-pages
  section moves **ahead** of category/links so the model reads "reuse beats
  regenerate" before drafting, and a new `## Example` section carries two worked
  pages (one English, one Chinese) that round-trip through the real parser and
  clear the atomicity heuristic — a regression test pins this so a malformed
  example can't ship. Prompt-only: no engine code, no placeholder/marker contract
  change, fully overridable per base via `synth.prompt_path`.
- **Synth prompt now disambiguates existing pages by slug + nudges
  priority-create (Phase 2).** Two deterministic-scoping additions to the
  per-group fan-out prompt; neither adds an LLM call.
  - **Existing-pages slug.** Every existing-pages / batch-accumulator bullet
    renders as `- Title [slug] (category)` (was `- Title (category)`). The slug
    is the deterministic kebab-case file stem, surfaced so the model can tell
    two same-titled pages apart; the prompt still instructs it to link by
    **title**, never slug.
  - **Priority-create feedback.** Wikilink targets an earlier group of the same
    source referenced but that resolve to no page yet (existing snapshot **or**
    in-batch, via the same exact → fuzzy → collision rules as `resolve_links`)
    are surfaced to later groups under a `## Priority targets (create if
    relevant)` section, so a group whose content covers one creates it at the
    right title instead of leaving the graph broken. Re-resolved each group (a
    target a prior group satisfied is dropped), ranked by how many distinct
    pages want it, and recorded in `knowledge_log`. Empty / anchor-only /
    punctuation-only targets (`[[#sec]]`, `[[ ]]`, `[[...]]`) are excluded — no
    usable key, uncreatable.

### Fixed

- **Synth/fixer now detect a clean-but-truncated LLM response via
  `finish_reason`.** PR2's "never open a `<page>` block you cannot finish"
  prompt prose lets a budget-starved model end *cleanly* — no unclosed tag —
  so `parse_synthesis_response` saw no truncation, the source was marked
  `synth_source_done`, and the dropped tail pages were stranded (the K layer
  has no scan-based reindex). The destructive `non_atomic_page` split fixer had
  the same exposure: a clean stop below `_MAX_CHILDREN_CEILING` would
  `delete_page` the original after writing only the children that fit.
  `parse_synthesis_response` now also consults the provider `finish_reason`,
  treating the cross-provider truncation set `{"length", "max_tokens"}` as a
  retry signal — note `anthropic_compat` (MiniMax-M3's provider) passes
  Anthropic's raw `stop_reason` (`"max_tokens"`, not `"length"`) through, so a
  literal `== "length"` check would have missed the synth workhorse. Zero
  parsed blocks under truncation becomes a hard `SynthesisError`; surviving
  blocks raise `SynthesisPartialError(retry=True)` so the synth fan-out
  withholds the done-marker and the fixer refuses the split. No behavior change
  on a clean `finish_reason`.
- **`openai_compat` embeddings are re-ordered by the response `index` before
  use.** The OpenAI embeddings response carries an explicit per-item `index`
  precisely because list order is not contractual — any compatible gateway
  (Ollama, vLLM, TEI, …) may return items out of order, and the consumer pairs
  vectors to chunks positionally before persisting them into the content-hash
  embedding cache. An unsorted response silently mis-assigned vectors *and*
  poisoned the cache. `gitee_multimodal` already sorted; `openai_compat` now
  applies the same defence (items missing `index` keep their list order).
- **`synth --verify` no longer fails the whole task when the duplicate leg's
  embed pass fails.** The semantic-duplicate leg performs a verify-only second
  embed pass over the run's page bodies; a transient 503 (or a permanent
  provider misconfig surfacing only there) propagated uncaught and flipped a
  synth whose pages were *all* persisted successfully into a FAILED task,
  discarding the SynthReport. The leg now degrades to the same loud skip the
  grounding leg already used (`duplicate_checked=False`, a warning, never a
  silent pass); `CancelledError` still propagates.
- **`eval --against` no longer exits 1 after printing SHIP.** Dataset-declared
  thresholds drove the exit code even when the user chose the baseline gate,
  so a run could print `SHIP — no regressions` and then exit 1 because an
  unrelated dataset threshold failed — CI reading the exit code saw a phantom
  regression. Under `--against` the exit code now reflects only the baseline
  verdict (the mirror of the existing exit-2 skip); a failed dataset gate is
  downgraded to a loud warning, not silenced, because judge-only gates (an
  `observed=None` dead-judge miss) are not pinnable in a baseline.
  `--write-baseline` keeps the exit-1: nothing gated that run, and exiting 0
  would let a thresholds-failing run silently pin its regressed metrics as
  the future reference (the file is still written; the exit code says look
  before committing it).
- **A majority-errored entailment judge no longer green-lights the gate or the
  verify report.** The `fact_entailment_ratio` denominator counts only
  successfully-judged claims, so a half-dead judge (19 timeouts + 1 `yes`, or
  a 50/50 tie) published a sliver ratio over a meaningless denominator —
  `ratio=1.0` passing the `0.55` floor. The reliability rule now lives on
  `EntailmentSummary.trustworthy` — with any judge errors, distinct
  successful judge *calls* (`n_calls_ok`, not `n_judged`: cached duplicate
  claims and no-evidence zeros pad `n_judged` without an LLM call) must
  strictly outnumber them — and is applied by BOTH consumers: the eval gate
  fold withholds the ratio and keeps the declared threshold as an
  `observed=None` loud miss (same fail-loud shape as the existing
  `n_judged == 0` guard), and `synth --verify --judge`'s grounding leg
  reports `None` (with the error counts visible) instead of "claims fully
  grounded". `trustworthy` is a `computed_field`, so the verdict also
  survives into the eval JSON payload — the client renderer (which cannot
  import the rule) reads it from `entailment_summary` and prints
  `withheld (judge errors outnumber successful verdicts)` instead of the
  sliver ratio.
- **All judge prompt fills are single-pass.** The entailment and four-field
  synth judges filled their templates with chained `str.replace`, so a
  page-authored value literally containing a later placeholder token (a claim
  citing `{evidence}`) had real content spliced into it. Every judge formatter
  now goes through one shared single-pass `_fill_template` (the wikilink and
  atomicity judges already did this individually).
- **`lint apply` no longer rewrites scalar `tags:`/`sources:` frontmatter as
  per-character garbage.** `_build_page_from_op` iterated the raw frontmatter
  lists bare, so a hand-written scalar (`sources: foo.md`) passed through a
  fixer became `['f','o','o',…]` *on disk*. Because apply REWRITES the page
  (and `sources:` feeds `replace_provenance_from`, where a cleared value is
  unrecoverable — the missing_provenance fixer backfills FROM frontmatter),
  the new read HEALS instead of collapsing: a scalar string becomes a
  one-item list, numeric list entries (`tags: [2024]`) are stringified, and
  only truly shapeless values (dict / null / bool) drop.
- **`dikw client check` fails the embed probes on empty embeddings.** A text
  probe returning no vectors (or a zero-dim one) without raising was reported
  `ok=True` with `dim=0` buried in the detail, and the multimodal probe
  green-lit two zero-dim vectors (they pass both the count check and the
  dim-equality check, `0 == 0`) — a green check before an ingest whose every
  embed call produced nothing. Both probes now fail with an explicit message,
  matching `_probe_llm`'s empty-completion check.
- **`tools/check_doc_refs.py` now scans `.claude/skills/**/*.md`.** Skill docs
  cite CLI verbs and `DIKW_*` env vars the same way `docs/**` does; they were
  outside the gate, so a typo'd verb in a skill drifted silently.
- **LLM completions no longer time out on long reasoning-model syntheses
  (`anthropic_compat` + `openai_compat`).** Both providers' `complete()` issued a
  **non-streaming** request whose read timeout (`llm_timeout_seconds`, default
  120 s) bounded the *whole response*, so a reasoning model (e.g. MiniMax-M3)
  spending minutes on a 16k-token synthesis raised `APITimeoutError` mid-receipt
  — and the SDK's `max_retries` only re-waited the full timeout, so every retry
  failed identically. `complete()` is now a collapse of the existing
  `complete_stream()`, so the read timeout applies **per SSE event** (token /
  thinking / keepalive) instead of per whole response; a steadily-streaming
  generation never trips it regardless of length (measured on MiniMax-M3: max
  inter-event gap ~2 s). The stream path also now **classifies** SDK failures —
  timeout / connection drop / 5xx / 408 / 429 → `TransientProviderError` (the
  synth group loop retries it); 401 / 403 / 404 / other 4xx → permanent
  `ProviderError` that fails fast instead of being retried-then-skipped (closing
  the asymmetry where only the embedding leg classified). `asyncio.CancelledError`
  propagates untouched. Behavior change for `openai_compat`: `complete()` now
  streams, so on gateways that don't emit a usage-only final chunk (older vLLM /
  TEI) it reports empty `usage` — observability only, not correctness. See
  `docs/providers.md` §3 for the two-tier retry model and the keepalive-stall
  limitation. (`openai_codex` already streamed; its exception classification is a
  follow-up.)
- **LLM judge no longer truncates to empty on reasoning models.** Both eval
  judges (`judge_synthesis`, `judge_entailment`) capped `max_tokens` at 512 / 256
  — too small for a reasoning LLM, which spends a hidden chain-of-thought against
  that budget before emitting the JSON. Measured against MiniMax-M3, a dense
  entailment judgment needs ~1350 output tokens, so the old caps truncated such
  responses to empty text and the judge logged them as parse errors (~75% of
  entailment calls, ~58% of page-judge calls on a real mvp run). Both defaults
  are now a reasoning-safe `4096`; non-reasoning models stop at `end_turn` far
  below it, so the higher ceiling bounds rather than pads (no extra spend). This
  is what makes the Phase 0b entailment metric trustworthy on the MiniMax
  baseline provider.
- **`dikw client check` no longer false-fails a reasoning-model LLM.** The LLM
  connectivity probe handed the model a fixed `max_tokens=32`; a reasoning model
  (e.g. MiniMax-M3) spends its hidden chain-of-thought against that budget before
  any visible token, so the probe got back an empty completion and reported a
  perfectly healthy provider as **down** (the same empty-completion shape as the
  issue #160 fault it was meant to catch). The probe now hands the model the
  configured `provider.llm_max_tokens_synth` budget instead, so a green check
  predicts the synth path's success — and a model whose budget genuinely is too
  small to clear its own reasoning still reports down (truthful, not a false
  red). Non-reasoning models stop at `end_turn` after `OK`, so the larger ceiling
  bounds rather than pads.

## 0.5.1 — codex empty-final recovery + K-layer persist fault-tolerance

### Fixed

- **`openai_codex` no longer drops a completion when the backend's final
  response carries `output=[]` (issue #160).** The ChatGPT codex backend
  sometimes ships a terminal `response.completed` whose `output` is an empty
  **list** — distinct from the `output=None` reducer bug, so the SDK hands back
  a well-formed `Response` and the provider trusted it, returning `text=""`
  even though valid `response.output_text.delta` events had already streamed
  the full answer (observed live: every synth group returning
  `response_chars: 0` while the model had actually produced `<page>` blocks, so
  `dikw client synth` reported `succeeded / created=0 / errors=0` and created
  no `knowledge/` files). The provider now falls back to the streamed delta
  text when the final response carries **no output items at all** but deltas
  did arrive. Narrow by design: a final with an explicit empty *message* item
  (`output=[message("")]` — a real cleared turn) still surfaces as `text=""`.
- **`dikw client check --llm-only` now fails on an empty completion.**
  `_probe_llm` previously reported `ok` whenever `complete()` did not raise,
  never inspecting the returned text — so a provider that silently yields no
  text (the #160 failure mode) passed the pre-flight check. The probe now
  verifies the completion is non-empty and reports the `finish_reason` /
  `output_tokens` when it isn't; the probe budget was raised (`max_tokens`
  4 → 32) so a reasoning model isn't starved into a false-red.
- **K-layer persist is now fault-tolerant via `documents.active`, matching D
  and W.** A hard storage failure mid-`persist_knowledge` (a permanent
  `ProviderError` from inline embed, or `replace_chunks` /
  `replace_links_from` / `replace_provenance_from` raising) previously left a
  half-written knowledge page with `active=True` that still surfaced in
  retrieval, and aborted the whole synth / lint-apply run. Now both K write
  paths — synth's per-page loop and `lint apply`'s Phase 1 loop — catch the
  exception, `deactivate_document` the in-flight page (so it is hidden from
  `fts_search` / `vec_search` / the wikilink graph leg / `read_page` /
  `list_pages`), record it, and continue with the remaining pages. This is the
  same deactivate-on-failure contract D (`api.ingest`) and W
  (`write_wisdom_page`) already enforced. A transient embed retry-skip is
  **not** treated as a failure — the page stays `active=True` with
  `chunks_pending_embedding > 0` for the next ingest's resume scan.
- **A failed page invalidates its source's `synth_source_done` marker.** Synth
  writes a new `synth_source_failed` knowledge_log marker when a page in a
  source failed to persist; it invalidates any prior `synth_source_done` for
  that source (markers apply in log order, last-writer-wins), so the next
  default `dikw client synth` re-processes the source and rebuilds the page
  parked inactive — even after a `synth --all` re-synth. K has no scan-based
  reindex, so this is the recovery path for a transient failure.
- **D (`api.ingest`) now deactivates the in-flight doc on cancellation too.**
  `asyncio.CancelledError` inherits from `BaseException`, so the per-file
  `except Exception` arm missed it: a cancel arriving mid-`persist_source`
  (after `upsert_document` committed `active=True`) left a half-indexed doc
  active forever, since the next ingest's unchanged-hash early-skip kept it
  stranded. D now catches cancellation, deactivates, and re-raises — closing
  the last gap so the deactivate-on-failure invariant holds uniformly across
  D / K / W for both hard exceptions and cancellation.
- **lint apply no longer reports a persist-failed page as a live change.** A
  page deactivated by a Phase-1 persist failure is now excluded from
  `ApplyReport.knowledge_paths_changed` (it is surfaced via `persist_errors`
  instead), matching synth — whose `created` / `updated` counters already
  excluded failed pages.

### Added

- `SynthReport.persist_errors` (tuple of `PagePersistError{path, message}`) and
  `ApplyReport.persist_errors` (list of `{path, message}`) surface pages
  deactivated by a mid-pipeline persist failure. The CLI renders them as a
  `path | message` table under the synth / lint-apply report.
- Synth's per-source progress event carries a `persist_failed` boolean in its
  `detail`, so a stream-only consumer can tell a source whose pages failed
  persist (still emitted as `outcome="no_pages"`, vocabulary kept stable) apart
  from one that legitimately produced zero pages.

## 0.5.0 — configurable knowledge taxonomy + overridable prompts; drop index.md/log.md

This release generalizes the fixed K-layer classification into a
user-configurable taxonomy, opens the K-layer authoring prompts to per-base
overrides, and removes the generated `knowledge/index.md` / `knowledge/log.md`.
All three are **breaking** for any base built on ≤0.4.7 — there is no in-place
migration (rebuild-on-incompatibility policy); a base carrying the old shape
trips `BaseUpgradeRequired` with the exact rebuild command. See
[ADR-0003](docs/adr/0003-configurable-knowledge-taxonomy.md) and
[ADR-0004](docs/adr/0004-drop-generated-index-and-log.md).

### Breaking

- **K-page classification is now a configurable, hierarchical `category` tree.**
  The fixed single-axis `type` (`entity`/`concept`/`note`) filed pages under
  pluralized folders (`knowledge/concepts/foo.md`). It is replaced by
  `schema.categories` in `dikw.yml` — a **closed set** of declared `path`
  (+ optional `desc`) entries of arbitrary depth, e.g. `产品/移动端`,
  `技术/架构`. A page filed under `技术/架构` lands at
  `knowledge/技术/架构/<slug>.md` with `category: 技术/架构` in its frontmatter.
  `entity`/`concept`/`note` is now merely the *default* taxonomy a fresh base
  ships with (folders are **singular**: `knowledge/entity/`, not `entities/`).
  Synth's LLM emits `<page category="…" slug="…">` (no `path=`/`type=`) and may
  only file under a declared `path`; anything it can't place lands in
  `schema.fallback` (default `未分类/`) and is flagged by the new
  `uncategorized` lint kind. Karpathy's rule: a wrong category is a cheap
  re-file, an invented folder is irreversible drift — the parser refuses an
  out-of-set category, and the brittle 0.4.6 "missing `knowledge/` prefix"
  path-recovery normalizer is **removed** (the engine now owns path
  construction entirely, so there is no model-supplied path to recover).
  Removed: `type:` frontmatter key, `SchemaConfig.page_types`,
  `SchemaConfig.log_style`, `type_to_folder`/`type_from_path` (→
  `category_from_path`).
- **K-layer authoring prompts are overridable per base.** `synthesize` (via
  `synth.prompt_path`; also used by the `non_atomic_page` fixer),
  `lint_fix_orphan_merge` and `lint_fix_broken_wikilink_grounded` (via
  `lint.fixer_prompts.{orphan_merge,broken_wikilink}`) can point at your own
  markdown under `<base>/prompts/`. `prompts.resolve` enforces that the override
  stays inside the base and carries the engine's required `{placeholders}` +
  `<page …>` output markers (declared in `prompts/_contract.py`), failing fast
  at load **and** surfacing via `dikw client check` (new `prompts` leg on the
  check report). New config: `synth.prompt_path`, the `lint:` block with
  `lint.fixer_prompts`.
- **`knowledge/index.md` and `knowledge/log.md` are no longer generated.** The
  category folder tree is the catalogue (browse it in Obsidian + `retrieve`),
  and the `knowledge_log` storage table (readable via `list_knowledge_log`)
  remains the authoritative history — only the markdown render views are gone.
  `dikw init` no longer scaffolds them (nor the pluralized type folders); it
  now creates `knowledge/<category>/.gitkeep` for the default taxonomy plus a
  `prompts/.gitkeep`. The `indexgen.py` and `log.py` modules are deleted.

### Migration

- Rebuild on upgrade: write your `schema.categories` into `dikw.yml`, move the
  old `knowledge/` (and `.dikw/`) tree aside, and re-run `dikw client synth` to
  re-author pages under the new taxonomy. `BaseUpgradeRequired` fires on a base
  that still carries `schema.page_types` (or the legacy `wiki/` layout) and
  prints the command.
- If you have a local `evals/.cache` snapshot built before this release, clear
  it (or run evals with `cache_mode="rebuild"`): a cached snapshot's `dikw.yml`
  carries the old `page_types` shape and would otherwise trip
  `BaseUpgradeRequired` on reuse. CI builds snapshots cold, so this only affects
  local dev caches.

## 0.4.7 — ingest source-path containment

### Security

- **Source discovery refuses a `sources[].path` that escapes the base.**
  `iter_source_files` (the D-layer ingest scan entry) now validates every
  configured source root up front and raises if one resolves outside the base
  — a `../` prefix or an absolute path elsewhere. `sources` is a managed tree
  under the base, so an escaping entry is a config error, not a license to read
  + index arbitrary files into the `Layer.SOURCE` index (whose doc-ids would
  also degrade to absolute paths). Fails the whole ingest with a clear message
  before any file is read (no partial index). Completes the K/W write-sink
  containment shipped in 0.4.6 — the last guarded-vs-unguarded asymmetry in the
  path-handling surface.

## 0.4.6 — ingest mtime fallback + synth path normalization + write-sink containment

### Fixed

- **Synth no longer drops a knowledge page when the model omits the
  `knowledge/` path prefix.** A `<page>` whose `path` uses a valid type
  folder but forgets the layer prefix (`entities/foo.md` instead of
  `knowledge/entities/foo.md` — common with smaller / open-weight models
  that follow the type-folder convention but drop the parent) used to raise
  `SynthesisError`; via per-group failure (#134) that discarded *every*
  page in the group and left the source un-marked, so the task reported
  success while the knowledge base was silently partial. The parser now
  normalizes a recognized type folder (built-in `entities`/`concepts`/`notes`
  **or** a custom `SchemaConfig.page_types` folder) by prepending
  `knowledge/`, preserving the model's own slug — the pinyin/ASCII spelling
  the prompt asks for and `slugify` would otherwise collapse to `untitled`
  for CJK titles. A stale pre-0.4.0 `wiki/` prefix or a truly unrecognized
  head still raises. The same parse-time guard now also rejects a `..`
  traversal segment (`entities/../…`, an already-prefixed `knowledge/../…`,
  or one hidden behind Windows `\` separators) and a bare type folder with no
  filename, so a prompt-injected source document can't steer synth into
  writing a page outside the base. (#146)
- **Source docs imported via a byte-stable tarball no longer store
  `mtime=0`.** dikw-web zeroes the tar `mtime` field so identical bytes
  dedup to one `package_sha256`; the extracted file landed with
  `st_mtime == 0`, every client rendered it as `1970-01-01`, and the
  constant `0` gave the graph change-hash (`api_graph.py`) no entropy
  across re-imports.
  `ingest` now falls back to ingest wall-clock (`time.time()`) when a
  source file carries no usable mtime (`<= 0`); an unchanged re-persist
  (e.g. an image-bearing doc, same body hash) keeps its already-stored
  timestamp so it doesn't flap the change-hash, while a genuine content
  change still advances it. Legacy rows already stored
  with `mtime=0` self-heal on the next `dikw client ingest`: a broken
  stored mtime now also forces one re-persist (alongside hash drift /
  `active=False` / asset refs). D-layer only — knowledge / wisdom pages
  are engine-written with a real mtime and were never affected. (#145)

### Security

- **K/W page writes now refuse a path that escapes the base.** The
  knowledge write sink (`write_page`) and the shared persist leg behind
  `persist_knowledge` / `persist_wisdom` (`_persist_layered_page`) now
  resolve `root / path` and reject it — before any `mkdir` / `write_text`
  or `parse_any` read — when it resolves outside the base (a `..`
  traversal segment or an absolute path). Previously only the read leg
  (`api_path_safety._assert_within`) and the wisdom file writer were
  guarded; these two K-side sinks were not, so a caller that bypassed the
  synth parser (#146/#149 closed that one vector) could steer a write — or
  a read-into-index — outside the base. The guard uses `Path.relative_to`
  on the *resolved* path, so it is platform-correct without a lexical
  backslash special-case. Follow-up to #149.

## 0.4.5 — api facade decomposition + dead-code cleanup

### Removed

- **Deprecated persist aliases dropped.** `persist_page`,
  `persist_knowledge_page`, and `wiki_doc_id` — the
  `tuple[int, str]`-returning compatibility shims kept through 0.4.0 —
  are gone. Call `persist_knowledge` / `persist_wisdom` /
  `persist_source` and `doc_id_for(Layer.KNOWLEDGE, …)` directly.
- **Orphaned eval `--dump-raw` plumbing removed.** `run_eval`'s
  `raw_dump_path` parameter and its `_dump_raw_ranked` JSONL writer had
  no CLI surface after the client/server split; both are deleted.
  `evals/tools/sweep_rrf.py` stays as a manual offline RRF-sweep tool —
  prepare its input JSONL by hand.

### Internal

- **`api.py` decomposed into per-verb cluster modules.** The 3.9k-line
  engine facade is now a ~170-line thin re-export surface; each verb lives
  in a focused `api_*` module (`api_core`, `api_types`, `api_health`,
  `api_ingest`, `api_pages`, `api_graph`, `api_retrieve`, `api_synth`,
  `api_lint`, `api_wisdom`, `api_path_safety`). The public `api.X` surface
  and `__all__` are byte-identical, so server routes and the eval runner
  are unaffected. Contributors add a verb's body to its cluster module and
  re-export it from `api.py`. Note for test authors: a function resolves
  its bare global names against the module it is *defined* in, so
  `monkeypatch.setattr` targets move with the verb (e.g. `ingest`'s
  `_with_storage` is now patched on `api_ingest`, not `api`).
- **K/W persist leg enforces `title_to_path`/`fuzzy_index` pairing.** When
  `_persist_layered_page` rebuilds `title_to_path` from storage (caller passed
  `None`), it now rebuilds `fuzzy_index` from the same title set via
  `build_title_indexes` and discards any caller-supplied `fuzzy_index`,
  closing a latent footgun where a stale fuzzy index resolved wikilinks
  against a different key space than the fresh exact index. Behavior-preserving
  for every production caller (synth, `lint apply`, `write_wisdom_page`), which
  already pass the two indexes as a matched pair.

## 0.4.0 — chunk→FTS→embed pipeline governance + ingest scope narrowing

### ⚠️ Breaking

- **`dikw client ingest` no longer scans `<base>/wisdom/`.** Wisdom
  pages are indexed exclusively when written via `dikw client wisdom
  write` (CLI) or `POST /v1/base/wisdom` (HTTP). Hand-edits to wisdom
  files in Obsidian are no longer auto-reindexed — the same
  limitation already applied to knowledge pages (a future `dikw
  client reindex <path>` will close this gap symmetrically). The old
  scan loop and its legacy aggregate-file skip-list have been
  removed; obsolete tests have been dropped, and the `ingest_wisdom_files`
  helper in `tests/fakes.py` lets test authors seed wisdom rows via
  the per-file `persist_wisdom` path.

### Added

- **Per-batch embed retry-skip.** `consume_embedding_stream` now
  catches `ProviderError` per batch and retries up to
  `cfg.provider.embedding_error_retries` (default 2) with
  `embedding_error_retry_backoff_seconds` (default 2.0s) linear
  backoff before skipping the batch and moving on. Skipped chunks
  remain in storage without vectors and the next ingest's
  missing-embedding resume scan reconciles them — synth /
  `lint apply` / `wisdom write` / ingest's bulk pass are all
  durable through transient embedding-provider failures now.
- **`lint apply` inline-embeds when an embedder is configured.**
  Setting `DIKW_EMBEDDING_API_KEY` makes `dikw client lint apply`
  re-embed every rebuilt page in the same pass so the fix is
  retrievable on return. Without the key, behaviour is unchanged:
  every chunk falls into `chunks_pending_embedding` and the next
  ingest's resume scan picks them up. `ApplyReport` gains
  `chunks_embedded` and `chunks_pending_embedding`; the CLI summary
  prints both.

### Changed

- **Refactor: `persist_page` split into three layer-specific
  functions.** `persist_source` (D, `domains/data/persist.py`),
  `persist_knowledge` (K, `domains/knowledge/page_index.py`), and
  `persist_wisdom` (W, `domains/wisdom/persist.py`) each own their
  layer's full upsert + chunk + FTS + (optional inline embed) +
  links + provenance pipeline. The legacy `persist_page` and
  `persist_knowledge_page` symbols remain as deprecated aliases
  returning the old `tuple[int, str]` shape; they will be removed
  in a follow-up.
- `api.ingest` is now D-layer-only plus the cross-layer
  missing-embedding resume scan that reconciles deferred chunks
  from D / K / W.

### Fixed

- **Synth's per-group retry-skip now catches only
  `TransientProviderError`** — symmetric with the 0.4.0 embed-batch
  retry change. Without this, a permanent LLM `ProviderError` (typo
  in `cfg.provider.llm_model`, missing key, invalid model id) was
  silently retried-then-skipped, producing "synth succeeded with
  0 pages" runs instead of failing fast. The `openai_codex`
  reducer-bug path (issue #134 / #135) now raises
  `TransientProviderError` so synth's narrowed retry still catches
  it.
- **`embed_assets` gained per-batch retry-skip** — symmetric with
  `embed_chunks` / `embed_chunks_multimodal`. Without it, a single
  5xx / timeout mid-pass aborted the whole asset embedding run; the
  retry-then-skip path now persists prior batches and the resume
  scan reconciles missing asset vectors on the next ingest. Both
  retries pass `cfg.provider.embedding_error_retries` and backoff.
- **`lint apply` now builds and threads `fuzzy_index` through
  `persist_knowledge` and Phase 2 referrer reconciliation** —
  without it, fuzzy-resolvable wikilinks like `[[Neural Networks]] →
  Neural Network` silently broke inside lint apply and the next
  lint propose flagged them as `broken_wikilink`, causing churn.
- **`WisdomWriteReport` gained `chunks_pending_embedding`** — fully
  symmetric with `ApplyReport`. Non-zero values surface when
  `no_embed=True`, when the inline-embed path defers (no active
  text version yet or `cfg.provider` drift), or when transient
  embed failures exhaust the retry budget. CLI render prints
  "N pending — next dikw client ingest reconciles them" mirroring
  the lint apply message.
- **`lint apply` / `wisdom write` no longer flip the active text
  embed version, defer inline embed on cfg drift, and preflight the
  embedder before mutating files.** Both paths now reuse the active
  version returned by `storage.get_active_embed_version("text")`
  instead of registering-and-activating a new identity from
  `cfg.provider`. Activating here would have stranded every other
  vector in the now-inactive table and gutted dense retrieval until
  the next full ingest. **Full-identity drift detection**: when the
  active version's `(provider, model, revision, dim, normalize,
  distance)` differs from `cfg.provider` (the user edited
  `dikw.yml` between full ingests), inline embed is deferred —
  otherwise the cfg-built embedder would produce different-dim or
  different-space vectors that get stored under the old version's
  table (silent corruption, or a hard StorageError mid-persist
  after files were already mutated). **Preflight**: each call also
  performs a one-token embed call before mutating any files, so a
  permanent provider error (bad API key, 401, invalid model id)
  surfaces while state is still clean instead of after Phase 0 has
  rewritten / deleted files. When no active version exists yet
  (fresh base), inline embed is deferred to the next ingest's
  resume scan, which goes through the full register-and-activate
  path. Mirrors `synthesize`'s long-standing reuse pattern.
- **Embedding provider errors classified as transient vs. permanent.**
  `OpenAICompatEmbeddings.embed` and `GiteeMultimodalEmbedding.embed`
  now classify exceptions into `TransientProviderError` (retryable:
  timeouts, rate limits, 5xx, 408/429, connect drops, parse failures)
  vs. plain `ProviderError` (permanent: 401, 403, 404, invalid model
  id, missing API key). The per-batch retry-skip in
  `consume_embedding_stream` retries only `TransientProviderError`;
  permanent errors propagate so misconfig fails fast instead of being
  silently retried-then-skipped (a single missing/wrong API key would
  otherwise have produced "success, 0 vectors" runs).
- **Synth forwards `cfg.provider.embedding_error_retries` to
  `persist_knowledge`.** The K-layer inline embed inside synthesize
  was silently using `retries=0` regardless of the configured
  retry budget; ingest, lint apply, and wisdom write already
  forwarded it.

### Known limitations

- **W→W forward-ref wikilinks**: a wisdom page `A` written before
  its `[[B]]` target wisdom page exists keeps the link unresolved
  in storage until `A` is re-written via `dikw client wisdom write`.
  Symmetric to the long-standing K→K limitation — synth doesn't
  re-resolve K-page links when a later page introduces the target
  either. A future `dikw client reindex <path>` will close both
  gaps; until then, re-writing the referring page is the
  user-facing workaround.

## 0.4.0 — BREAKING term rename: K layer "wiki" → "knowledge"

⚠️ **Breaking change for every existing base.** The K-layer
on-disk directory, the `Layer` enum value, the `wiki_log` SQL
table, and every `wiki_*` API symbol have been renamed to use
`knowledge` consistently. Bases initialised by ≤0.3.6 are **not
readable** by 0.4.0; no in-place migration is provided.

**Manual upgrade for an existing base:**

```bash
cd <base>
mv wiki knowledge       # rename the K-layer directory
rm -rf .dikw            # drop the SQLite + auth + task ledger
# (dikw.yml stays — your existing config is reused)
dikw serve --base . &   # start the server
dikw client ingest      # reindex sources + knowledge pages
```

Opening a 0.3.x base with 0.4.0 raises `BaseUpgradeRequired`
with the exact command above; the server refuses to start until
the rename is done.

### Why

`wiki` was historically overloaded — it named the K-layer DIKW
role, the on-disk directory, and the `[[wikilink]]` body
syntax. CONTEXT.md called this out as a long-standing
ambiguity. In 0.4.0 the term is reserved exclusively for
**wikilink** syntax; everything else uses `knowledge`. This
removes a recurring source of confusion in LLM prompts, public
APIs, and new-contributor onboarding.

### Renames

- **On disk:** `<base>/wiki/` → `<base>/knowledge/`,
  `<base>/trash/wiki/` → `<base>/trash/knowledge/`,
  `wiki/index.md` → `knowledge/index.md`,
  `wiki/log.md` → `knowledge/log.md`.
- **Layer enum / DB:** `Layer.WIKI` → `Layer.KNOWLEDGE`
  (string value `'wiki'` → `'knowledge'`); SQL table `wiki_log`
  → `knowledge_log`; storage CHECK constraint values updated.
- **Engine API:** `WikiPage` → `KnowledgePage`,
  `WikiPageMeta` → `KnowledgePageMeta`,
  `WikiLogEntry` → `KnowledgeLogEntry`;
  `persist_wiki_page` → `persist_knowledge_page`;
  `init_wiki` / `load_wiki` / `resolve_wiki_root` →
  `init_base` / `load_base` / `resolve_base_root`;
  module `domains/knowledge/wiki.py` → `domains/knowledge/page.py`.
- **Field rename:** `SynthReport.wiki_pages` →
  `knowledge_pages`; progress `wiki_pages_changed` →
  `knowledge_pages_changed`; `StorageCounts.last_wiki_log_ts`
  → `last_knowledge_log_ts`.
- **HTTP / CLI:** response field `"wiki_root"` →
  `"base_root"`; `dikw auth ... --wiki, -w` →
  `--base, -b` (consistent with `dikw serve --base`).
- **Storage protocol:** `append_wiki_log` /
  `list_wiki_log` → `append_knowledge_log` /
  `list_knowledge_log`.
- **Runtime:** `<base>/.dikw/wiki_id` → `<base>/.dikw/base_id`;
  env var `DIKW_WIKI_INSTANCE_ID` → `DIKW_BASE_INSTANCE_ID`.

### Preserved (still spelled "wiki")

The `[[wikilink]]` syntax keeps its name everywhere:
- `LinkType.WIKILINK`, regex `WIKILINK_RE`, lint kind
  `broken_wikilink`, prompt file
  `lint_fix_broken_wikilink_grounded.md`,
  `SynthReport.unresolved_wikilinks`,
  storage method `replace_links_from`, and every
  docstring/comment that talks about `[[Target]]` resolution.

### Added — synth retries `ProviderError` per group, then skips

Fixes [#134]. A single `ProviderError` raised by `llm.complete` for
one source group used to abort the whole `synth` task — prior groups'
work was lost and later sources never ran. The canonical trigger was
the `openai_codex` empty-streaming-response edge case (`response.
output=None` + zero text deltas), but any `ProviderError` (auth flap,
quota, refusal) hit the same dead-end.

`_synth_pages_from_source` now wraps `llm.complete` in a bounded
per-group retry loop: up to `cfg.synth.provider_error_retries`
retries (default `2`, so `3` attempts total) with linear backoff
(`provider_error_retry_backoff_seconds`, default `2.0` → `2s`/`4s`).
After the retries are exhausted the group is recorded as a parse-
style error and **skipped**; subsequent groups in the same source
and later sources continue to process. The skip is counted in
`outcome.parse_errors`, so `synth_source_done` is **not** written
and re-running `synth` retries the flaky group.

Per-group NDJSON progress events surface the new states:
`status="retrying"` (one per failed attempt that still has retries
left, carries `attempt` / `max_attempts` / `error_kind` /
`error_msg`) and `status="skipped"` (terminal, carries
`reason="provider_error"` / `attempts` / `error_kind` /
`error_msg`). Set `provider_error_retries: 0` for one-attempt-no-retry
(the group is still skipped, not re-raised — by design, since the
whole point of the fix is that a single bad group must not abort the
task).

[#134]: https://github.com/OpenDIKW/dikw-core/issues/134

### Added — `read_page` surfaces parsed frontmatter

`api.read_page` and `GET /v1/base/pages/{path}` now return the parsed
YAML front-matter as a `frontmatter: dict[str, Any]` on
`PageReadResult`. Callers that previously had to re-read the file
themselves to look up `tags`, `sources`, `status`, or custom keys can
now read them off the same response. The field is `{}` when the file
has no front-matter or when the YAML failed to parse externally
(parse-failure also still emits `anchors=[]`, so the empty-anchors
signal disambiguates "no metadata" from "broken file"). Backwards
compatible: existing clients ignore the new field.

## 0.3.6 — 2026-05-28

- **Fixed:** `openai_codex` survives the ChatGPT codex backend's `response.output = None` reducer bug — falls back to locally-collected delta text and raises `ProviderError` on a zero-delta hit so synth never silently drops a page (#129).

## 0.3.5 — 2026-05-27

- **Added: wisdom write surface.** `api.write_wisdom_page` / `POST /v1/base/wisdom` / `dikw client wisdom write` create or update a single `wisdom/[<author>/]<slug>.md` and index it inline (chunks + FTS + embeddings + wikilinks + provenance) without a full `ingest`. Upsert on `(author, slug)`; `--no-embed` defers embedding; empty body is rejected (422). Reads still go through `dikw client pages get wisdom/...` (#126).

## 0.3.0 — 2026-05-26

- **BREAKING: Wisdom layer reshaped** from an LLM-distilled candidate/review pipeline into a hand-written first-class document layer under `wisdom/<author>/<slug>.md`, sharing the K-layer `documents` / `chunks` / `links` / `provenance` shape (4-PR arc #120–#123). Removed the `distill` and `review {list,approve,reject}` verbs, `POST /v1/distill`, `GET /v1/wisdom`, the `wisdom_items` / `wisdom_evidence` / `wisdom_embed_meta` / `vec_wisdom_v<n>` tables, the `WisdomKind` taxonomy, and the `provider.llm_max_tokens_distill` / `schema.wisdom_kinds` config keys. Added the wisdom-only `documents.status` column (`draft | published | favorite | archived`) and the `invalid_wisdom_status` lint kind. Wisdom now participates in retrieve (`Hit.layer == "wisdom"`), the page read APIs, and the `broken_wikilink` / `orphan_page` / `missing_provenance` lint scan. `SCHEMA_VERSION` bumped 3 → 5 (rebuild required). Rationale: `docs/adr/0002-wisdom-as-first-class-documents.md`.
- **Changed:** project status pre-alpha → alpha; `examples/docker/Dockerfile` `DIKW_VERSION` now auto-syncs via a post-publish `sync-dockerfile` PR plus a `dockerfile-version-guard` CI check (#118, #119).

## 0.2.7 — 2026-05-24

- **Docs/tests:** swept stale `--enable-llm` copy that still described the pre-#83 "TODO-stub" fixer behaviour (it became an evidence-backed grounded repair in #83) and added the missing `orphan_page` mentions across CLI / OpenAPI / engine docstrings. No runtime change.

## 0.2.6 — 2026-05-23

- **Added: provenance edge** — each K-page's `sources:` frontmatter is now a queryable page → D-source attribution edge, distinct from body `[[wikilinks]]`: new `provenance` table, `replace_provenance_from` / `provenance_from` / `provenance_to` Storage methods, `GET /v1/base/pages/{path}/provenance`, and `dikw client pages provenance`. Rationale: `docs/adr/0001-provenance-as-separate-edge.md`.
- **Added:** `missing_provenance` lint kind + deterministic `MissingProvenanceFixer` to backfill the table from frontmatter (no LLM).
- **Changed (storage):** `SCHEMA_VERSION` → 3 (rebuild required).

## 0.2.5 — 2026-05-21

- **BREAKING (CLI):** agent-first default-JSON audit completed — `lint`, `lint proposals`, `tasks list`, `import` (and the then-extant `review` commands) default to JSON; humans opt into `--format table` / `--pretty`.

## 0.2.0 — 2026-05-19

- **BREAKING (HTTP):** `GET /v1/tasks` returns a `TaskListPage` envelope (`{tasks, next_cursor, has_more}`) with **summary** rows (no `result` / `error`) instead of a bare `list[TaskRow]`; full detail via `GET /v1/tasks/{id}` or `.../result`. Added keyset cursor pagination (`?cursor=`, `?limit=`, composing with `?status=` / `?op=`); `dikw client tasks list` gained `--all` / `--cursor`.
- **Fixed:** `dikw serve --help` no longer claims NDJSON for task events (they are cursor JSON + long-poll; only `POST /v1/retrieve` streams NDJSON).

## 0.1.0 — 2026-05-18

First tagged release; consolidates the client/server architecture and the agent-first kernel surface built during pre-release development.

- **BREAKING (architecture):** the in-process CLI was replaced by a long-lived `dikw serve` (FastAPI) server + a `dikw client` HTTP client.
- **BREAKING (CLI):** top-level short names for HTTP commands removed — every HTTP-bound verb lives exclusively under `dikw client <verb>` (no `dikw status` / `dikw retrieve` / `dikw serve-and-run` aliases). The four local-only surfaces (`dikw version`, `dikw init`, `dikw serve`, `dikw auth …`) are unchanged.
- **BREAKING:** in-engine answer synthesis removed — `POST /v1/query`, `dikw client query`, the `QueryResult` / `Citation` DTOs, and `provider.llm_max_tokens_query` are gone. `retrieve` returns ranked chunks + page refs; the agent layer runs its own LLM.
- **BREAKING (CLI + HTTP):** `dikw client init` and `POST /v1/init` removed — the server refuses to start without a `dikw.yml`, so local scaffolding is `dikw init <path>` only.
- **BREAKING:** source-import verb renamed `upload` → `import` and decoupled from `ingest` (commit bytes into `sources/` vs chunk + embed); the `--wiki` flag renamed `--base`.
- **Added:** `GET /v1/base/graph` (+ `dikw client graph get`) — the full base graph (nodes + edges + unresolved wikilinks) in one read-only request (#89). `GET /v1/base/pages/{path}/links` exposes the K-layer link graph at a page boundary.
- **Fixed:** `broken_wikilink --enable-llm` became an evidence-backed grounded repair (D-layer hybrid search gates the LLM; `TODO` / `stub page` / `placeholder` / sub-200-char outputs rejected) instead of fabricating TODO stubs (#83).
- **Changed:** `synth` preserves the source's dominant language in K-layer pages; `dikw client status` / `check` default to JSON.
