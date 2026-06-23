# Running `dikw serve`

This document covers operating the FastAPI + NDJSON server (`dikw serve`)
and the wire contract between server and client. For the in-process
engine surface, see [`docs/architecture.md`](./architecture.md); for the
on-disk format the engine produces, see [`docs/design.md`](./design.md).

## TL;DR

```bash
# Bind to loopback, no auth — the typical single-user laptop workflow.
uv run dikw serve --base ./my-base

# In another terminal:
uv run dikw client status
uv run dikw client retrieve "what does Karpathy say about scoping?"
```

For one-shot commands without keeping a server running, use
`serve-and-run`:

```bash
uv run dikw client serve-and-run --base ./my-base -- ingest --no-embed
```

## Wire shape

The server speaks JSON over HTTP under `/v1/`. Two route families:

| family | examples | shape |
|---|---|---|
| **Sync** (millisecond-level) | `GET /v1/status`, `POST /v1/check`, `POST /v1/lint`, `GET /v1/base/pages`, `GET /v1/base/pages/{path}`, `GET /v1/base/pages/{path}/links`, `GET /v1/base/pages/{path}/provenance`, `GET /v1/base/graph`, `POST /v1/doc/search` | request / response JSON |
| **Async tasks** (seconds–minutes) | `POST /v1/{ingest,synth,eval}`, `POST /v1/lint/{propose,apply}` → `task_id`; `GET /v1/tasks?cursor=<opaque>&limit=M&op=…&status=…` (cursor JSON, summary rows); `GET /v1/tasks/{id}/events?from_seq=N&limit=M&wait=K` (cursor JSON, long-poll); `GET /v1/tasks/{id}` / `GET /v1/tasks/{id}/result`; `POST /v1/tasks/{id}/cancel` | submit JSON → paged JSON cursor → final JSON |
| **Streaming retrieve** | `POST /v1/retrieve` | NDJSON: `retrieve_started → retrieval_done → final`. **No LLM tokens stream from the server** — agents compose chunks with their own LLM. |
| **Import** | `POST /v1/import` | multipart: tar.gz payload + packages-aware manifest JSON; commits straight into `<base>/sources/` |

Every error follows one envelope:

```json
{ "error": { "code": "not_found", "message": "...", "detail": {...} } }
```

`code` is the stable identifier — clients branch on it, never on the
free-form `message`.

### `GET /v1/base/graph` — full base graph

One request returns every node + every edge + every unresolved wikilink
in the base, so a web Knowledge Graph view doesn't have to loop
`GET /v1/base/pages/{path}` and re-parse `[[wikilinks]]` in the
browser. Query: `active` (`true` default = only active docs; `false` =
deactivated subset; mirrors `GET /v1/base/pages`). Response shape:

```json
{
  "base_revision": "<sha256 over (path, title, layer, mtime, body_sha256, active)>",
  "generated_at": "2026-05-14T10:00:00Z",
  "nodes":      [ { "id", "path", "title", "layer", "active", "mtime", "inbound", "outbound" } ],
  "edges":      [ { "id", "source", "target", "type", "target_text", "anchor", "weight" } ],
  "unresolved": [ { "source", "target_text", "anchor", "count" } ],
  "stats":      { "node_count", "edge_count", "unresolved_count" }
}
```

Determinism: identical base state hashes the same `base_revision`, and
nodes / edges / unresolved are sorted (path; then `(source, target,
target_text, anchor)`; then `(source, target_text, anchor)`) so two
back-to-back calls return byte-equivalent payloads modulo
`generated_at`. Endpoint is read-only — never triggers ingest, synth,
or lint apply. Repeated `(source, target, target_text, anchor)` edges
collapse to one edge with `weight > 1`. URLs and out-of-base markdown
links are intentionally dropped (neither edge nor unresolved). Issue
#89 v1 deliberately omits ghost nodes for unresolved targets and the
`layer` query knob — clients filter the node set themselves.

## Bind and authentication

`dikw serve` binds to `127.0.0.1:8765` by default. There is **no
authentication on loopback** — the implicit threat model is "the user
running the CLI also owns the base on disk." If you need to expose the
server to other hosts:

```bash
export DIKW_SERVER_TOKEN=$(openssl rand -hex 32)
uv run dikw serve --base ./my-base --host 0.0.0.0 --token $DIKW_SERVER_TOKEN
```

Hard rule, enforced at startup: **`--host 0.0.0.0` (or any non-loopback
address) refuses to start without a token.** The runtime would rather
fail loudly than silently expose an unauthenticated base to the
network.

Clients pick the token up via:

1. `--token` CLI flag (highest precedence)
2. `DIKW_SERVER_TOKEN` env var
3. `~/.config/dikw/client.toml` (or `%APPDATA%\dikw\client.toml` on
   Windows) under `[default]`
4. Built-in default (no token; only valid against a no-auth server)

## Operational concerns

### Process lifecycle

`dikw serve` is a long-lived process. Run it under a supervisor in
production:

* **systemd** — `ExecStart=/usr/bin/dikw serve --base /var/lib/dikw/...`,
  `Restart=on-failure`. Set `Environment=DIKW_SERVER_TOKEN=...` (or
  `EnvironmentFile=` to a 600-perm file) so the token doesn't end up in
  the unit listing.
* **Docker** — see [`../examples/docker/`](../examples/docker/) for a
  ready-to-run `docker-compose.yml` (dikw-core +
  `pgvector/pgvector:0.8.2-pg18`) and
  [`deployment-docker.md`](deployment-docker.md) for the bootstrap
  walkthrough. The server expects to own its bound base — don't share
  the same `.dikw/` directory across multiple containers.
* **Foreground / dev** — `uv run dikw serve` is fine for laptop work;
  `serve-and-run` is the right tool for one-shot commands.

The server lifespan boots storage on the first request, runs migrations
idempotently, and keeps adapters open until shutdown. SIGTERM triggers
a graceful drain (FastAPI lifespan teardown closes adapters before the
socket).

### Server-restart semantics for in-flight tasks

When the server restarts mid-task (e.g., systemd restart, OOM kill,
graceful shutdown), **any task previously in `pending` or `running`
status is marked `failed{reason=server_restart}`** by the lifespan
startup hook — automatically for the per-base SQLite task store, and
for a shared Postgres task store only when `DIKW_TASK_REAP_ON_START=1`
is set (otherwise it is skipped so a healthy peer replica's in-flight
tasks aren't reaped). The TaskManager doesn't attempt to resume —
engine ops are idempotent
(content-hash skip on ingest, deterministic page paths on synth) so the
correct recovery is to re-submit the task, not to half-resume one whose
in-memory state is gone.

### Storage concurrency

Two layers of write serialization keep concurrent tasks from racing a base:

* **Per-adapter (SQLite).** Each `SQLiteStorage` instance shares one
  `sqlite3.Connection` across the `asyncio.to_thread` workers a single verb
  fans out (e.g. retrieval's fts/vec/asset legs run via `asyncio.create_task`).
  That connection's Python-level state isn't thread-safe, so every method body
  runs under a per-instance `threading.RLock` — overlapping workers on one
  instance are serialized instead of tripping `sqlite3.InterfaceError` /
  phantom rows.
* **Cross-connection (SQLite WAL).** Distinct verbs open distinct connections
  to one WAL file, so two writers can still collide at the file-lock level. The
  adapter sets `busy_timeout=30000` (and opens with `connect(timeout=30)` so
  the same budget covers the connection-time pragmas), so the loser **blocks up
  to 30 s and then succeeds** rather than immediately raising `database is
  locked`. A base also mutated by a hand-edited script out-of-band can still
  exceed that window.
* **Base-level write lock (process).** The server holds one `asyncio.Lock`
  (`ServerRuntime.ingest_lock`) across every base-mutating task — ingest,
  import, wisdom write, delete, **synth, and lint apply** — so two writers on
  one base never interleave their (un-transaction-wrapped) row + on-disk writes
  within a process. Read paths (retrieve, list, read) don't take it.
* **Postgres** — multiple `dikw serve` instances against one base are
  supported by the storage layer (each task is one transaction), but there's no
  orchestration logic, and `ingest_lock` is **per-process** — it does not
  serialize writers across replicas. If you need multi-server topologies, put a
  load balancer in front and accept that ingest/synth tasks racing on the same
  source will produce one winner per `(path, content_hash)` pair via
  storage-level upsert.
* **Filesystem backend** — single-writer only by design. Don't run two
  servers against the same base.

### Observability

* `GET /v1/healthz` — liveness, no dependencies. Returns `{"status":"ok"}`
  immediately. Suitable for k8s readiness/liveness probes and the
  `serve-and-run` ready-poll.
* `GET /v1/readyz` — confirms the storage adapter is connected and
  migrated. Returns `{"status":"ready", ...}` with HTTP 200 once the
  lifespan startup hook has run (requests aren't routed before then, so
  there's no cold-start 503). It deliberately doesn't probe providers —
  use `/v1/check` for that.
* `GET /v1/info` — engine version, storage backend, configured
  providers (without secrets), auth posture. `dikw client` probes this
  once per invocation to run a **version handshake**: if the server's
  `engine_version` differs from the client's own installed `dikw-core`
  version it hard-fails the command (`version skew:` + exit 1), since a
  skewed pair can silently misbehave while dikw-core is alpha. Ambiguous
  cases (unreachable, non-200, field missing) skip the check rather than
  raise a false skew. Set `DIKW_ALLOW_VERSION_SKEW=1` to downgrade the
  refusal to a stderr warning for deliberate mixed-version debugging.
* `GET /v1/tasks?limit=...` — list of past tasks for debugging long
  ingests / failed synth runs. Persists across server restart (backed
  by the same storage adapter as the base itself).

#### OpenTelemetry export

> For the full operator cookbook — the complete span/metric inventory, a
> dashboard query cookbook, and a ready-to-run Collector → Jaeger / Prometheus /
> Grafana stack — see [`observability.md`](observability.md). This section is the
> in-context summary.

Optional, **off by default**, and a zero-overhead no-op unless the `[otel]`
extra is installed (`uv pip install 'dikw-core[otel]'`). Turn it on with a
`telemetry:` section in `dikw.yml`:

```yaml
telemetry:
  enabled: true
  endpoint: http://collector:4318   # OTLP/HTTP; /v1/traces + /v1/metrics appended for you
  service_name: dikw-core
  sample_ratio: 1.0                 # ParentBased(TraceIdRatio) head sampling
```

The standard `OTEL_SDK_DISABLED` kill-switch is honoured, and an unset
`endpoint` falls back to `OTEL_EXPORTER_OTLP_ENDPOINT`. The server bootstraps
the SDK from its lifespan (after config load) and traces are exported to any
OTLP backend (Jaeger/Tempo, Grafana, Datadog, …). A single trace now spans:
the `/v1/*` HTTP server span → a `dikw.task.<op>` span for the background task
it submits (a **root span linked** back to the request, since the detached task
outlives it) → a `dikw.<op>` engine span (`dikw.ingest` / `dikw.synth` /
`dikw.retrieve` / `dikw.lint.{propose,apply}`) carrying `dikw.layer` — for a
retrieve, this also has a `dikw.retrieve.leg` child span per fusion leg (BM25,
vector, asset, graph) with its hit count → a `gen_ai.chat` span per LLM call
carrying the model + token usage (incl. Anthropic prompt-cache tokens) and a
`gen_ai.embeddings` span per embedding call → the outbound provider HTTP call,
auto-traced via httpx with W3C `traceparent` propagation. (The `dikw.<op>`
engine spans are also the root for direct / eval callers, which have no task
span.)

Alongside traces, the server also exports **GenAI metrics** over OTLP/HTTP
(`/v1/metrics`, on a periodic timer). Every LLM/embedding call — completed or
failed — emits two histograms from the same span seam, so no provider-specific
metric code exists:

* `gen_ai.client.token.usage` (`{token}`) — one point per token class, tagged
  `gen_ai.token.type` (`input` / `output`, plus `cache_read` / `cache_creation`
  for Anthropic prompt caching), `gen_ai.operation.name` (`chat` / `embeddings`),
  `gen_ai.system`, and `gen_ai.request.model`.
* `gen_ai.client.operation.duration` (`s`) — request latency, tagged
  `error.type` on a failed call. A cancelled / abandoned stream records no point
  (its cut-short time would skew the latency series).

FastAPI HTTP-server metrics flow for free from the same meter provider.

The engine also exports **dikw-domain metrics** mapped from the per-call report
DTOs + existing instrumentation, so a dashboard sees pipeline volume without
parsing logs (same meter provider, all no-ops when telemetry is off):

* `dikw.ingest.files` (tag `dikw.result`), `dikw.ingest.chunks`,
  `dikw.ingest.errors` (tag `dikw.error.kind`) — counters from the final
  `IngestReport`.
* `dikw.synth.pages` (tag `dikw.result`), `dikw.synth.unresolved_wikilinks`,
  `dikw.synth.persist_errors` — counters from the final `SynthReport`.
* `dikw.embed.chunks` / `dikw.embed.skipped` / `dikw.embed.retries` — chunk-embed
  volume + durability, from the shared consume seam (covers D/K/W inline embed).
* `dikw.retrieve.leg.duration` (`s`, tag `dikw.retrieval.leg`) — per-leg latency.
* `dikw.task.duration` (`s`, tags `dikw.op` + `dikw.status`) — background-task
  wall-clock per terminal outcome.

**Log ↔ trace correlation.** Set `DIKW_LOG_FORMAT=json` (default `text` keeps the
human-readable terminal line byte-for-byte) and every log line becomes one JSON
object (`ts` / `level` / `logger` / `message`, plus any `extra={…}` fields and an
`exception` traceback). When telemetry is active, records emitted inside a span
also carry `trace_id` / `span_id` / `service`, so a log aggregator can pivot from
a log line straight to its trace. The OTel `LoggingInstrumentor` injects those ids
via a log hook (`init_logging` keeps handler + format ownership; no OTLP log
*export* handler is attached — that stays deferred). The same `DIKW_LOG_FORMAT`
applies to the `dikw client` CLI, whose records correlate to the client side of
the trace below.

Finer per-source / per-group / per-batch sub-spans land in subsequent releases.

The remote `dikw client` CLI has no base config (no `dikw.yml`), so its
telemetry is driven purely by the standard `OTEL_*` env vars. Set
`OTEL_EXPORTER_OTLP_ENDPOINT` (or the per-signal
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) before a `dikw client …` command and —
with the `[otel]` extra installed and `OTEL_SDK_DISABLED` unset — the CLI
auto-instruments its outbound httpx calls, injecting a W3C `traceparent` that
the server adopts as the parent. The result is **one trace** from the client
process through the HTTP server span, the background task, the engine op, and
the provider call:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4318
export OTEL_SERVICE_NAME=dikw-client   # optional; defaults to dikw-client
dikw client ingest                     # this trace now stitches to dikw serve
```

The bootstrap fires only for the `client` subgroup — local `version` / `init` /
`serve` / `auth` commands make no HTTP calls and pay zero cost, and `serve`
keeps wiring its own server-side telemetry from the `telemetry:` section above.
A plain `dikw client` invocation with no `OTEL_*` env is a no-op. Because the
CLI is short-lived, spans are flushed when it exits (an `atexit` hook drains the
`BatchSpanProcessor`), so even a sub-second command exports its span.

### Client config

Per-machine defaults live at `~/.config/dikw/client.toml`
(or `%APPDATA%\dikw\client.toml` on Windows):

```toml
[default]
server_url = "http://my-server.example:8765"
token = "..."   # optional; prefer env or --token to keep secrets out of files
```

The hierarchy is **explicit > env > file > built-in default**. Each
layer is independent: setting only the URL via env works fine if the
token comes from the file, and so on.

### Networking gotchas

* **Reverse proxies and NDJSON streams** — `POST /v1/retrieve` is the
  only NDJSON-streaming endpoint left after the task-first cursor
  flip. Disable response buffering on any proxy that fronts the
  server. nginx: `proxy_buffering off;`. Without this, clients see
  events arrive in batches at the buffer flush boundary. (Everything
  else is request/response JSON or multipart: `POST /v1/import` is a
  multipart upload returning an ``ImportResponse`` JSON body.)
* **Task events use cursor JSON, not streaming** — `GET /v1/tasks/{id}/events`
  is a long-poll JSON endpoint (server holds the response up to `wait`
  seconds, capped at 60s), not an open NDJSON stream. Each response is
  a single `EventsPage` with `events`, `next_from_seq`, `has_more`,
  `last_seq`, `task_status`. UIs page with `wait=0`; agents follow
  with `wait>0` and re-issue with the returned `next_from_seq`. No
  heartbeat needed because the response cycle itself bounds connection
  lifetime to ≤ `wait`.
* **Tasks list uses cursor JSON too** — `GET /v1/tasks` returns a
  `TaskListPage` envelope (`{tasks, next_cursor, has_more}`) since
  0.2.0; rows are **summary** projections of `TaskRow` (no `result`,
  no `error`) so a 50 KB synth result never crosses the wire just
  because someone browsed the list. Pull full detail through
  `GET /v1/tasks/{id}` (whole row) or `GET /v1/tasks/{id}/result`
  (terminal payload). The `next_cursor` is an opaque base64url token
  encoding `(created_at, task_id)` under the
  `(created_at DESC, task_id ASC)` keyset — clients must treat it as
  opaque and replay it verbatim on the next request. A tampered or
  forged cursor surfaces as `400 invalid_cursor`.
* **Import payload size** — `POST /v1/import` accepts up to 1 GiB by
  default. Override via `DIKW_SERVER_MAX_IMPORT_BYTES=<int>`.

### Sources import (`POST /v1/import`)

Multipart form-data with two parts:

* `payload` — tar.gz; every member's path must start with `sources/`
  (assets ride along under `sources/<rel>` so the engine's
  sibling-of-md asset resolution still works).
* `manifest` — JSON of shape:

  ```json
  {
    "files":    [{"path": "sources/...", "size": N, "sha256": "<lc-hex>"}],
    "packages": [
      {"id": 0, "md_path": "sources/note.md",
       "asset_paths": ["sources/diagram.png"],
       "package_sha256": "<sha256(sorted([md_sha, *asset_shas]).join(\"\\n\"))>"}
    ],
    "total_bytes": N
  }
  ```

The server stages the tarball under
`<base>/.dikw/staging/<import_id>/`, validates schema (orphan
file, duplicate `md_path`, missing/extra files), recomputes every
file sha256 + each `package_sha256`, then commits well-formed
packages straight into `<base>/sources/` via `os.replace` and
rmtrees the staging tree.

Response:

```json
{
  "import_id": "...", "files_count": N, "bytes": M, "applied_at": "...",
  "committed": [0, 2, ...],
  "rejected": [{"id": 1, "code": "...", "detail": {...}}]
}
```

Error codes:

| code | layer | meaning |
|---|---|---|
| `import_too_large` | request | exceeds `DIKW_SERVER_MAX_IMPORT_BYTES` |
| `tar_path_traversal` / `tar_link_forbidden` / `tar_unexpected_path` / `tar_invalid` | tar safety | refused before extraction |
| `manifest_malformed` / `manifest_invalid` | schema | manifest JSON / pydantic validation |
| `manifest_packages_missing` | schema | legacy files-only manifest no longer accepted |
| `manifest_missing_files` / `manifest_extra_files` | schema | tar / manifest disagree |
| `manifest_orphan_file` | schema | file in manifest not referenced by any package |
| `manifest_duplicate_md_path` | schema | one `md_path` in multiple packages |
| `manifest_package_unknown_file` | schema | a package references a file not declared in `files` |
| `manifest_sha256_mismatch` | per-package | file-level sha mismatch; affected packages go to `rejected` |
| `manifest_package_sha256_mismatch` | per-package | package-level sha mismatch |
| `package_commit_failed` | per-package | `os.replace` failed (disk full, permissions) |

Schema-level failures return 4xx; per-package failures return 200 with
the failing packages listed in `rejected` so the client can retry just
those.
