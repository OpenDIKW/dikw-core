# Running `dikw serve`

This document covers operating the FastAPI + NDJSON server (`dikw serve`)
and the wire contract between server and client. For the in-process
engine surface, see [`docs/architecture.md`](./architecture.md); for the
on-disk format the engine produces, see [`docs/design.md`](./design.md).

## TL;DR

```bash
# Bind to loopback, no auth — the typical single-user laptop workflow.
uv run dikw serve --wiki ./my-wiki

# In another terminal:
uv run dikw client status
uv run dikw client query "what does Karpathy say about scoping?"
```

For one-shot commands without keeping a server running, use
`serve-and-run`:

```bash
uv run dikw serve-and-run --wiki ./my-wiki -- ingest --no-embed
```

## Wire shape

The server speaks JSON over HTTP under `/v1/`. Two route families:

| family | examples | shape |
|---|---|---|
| **Sync** (millisecond-level) | `GET /v1/status`, `POST /v1/check`, `POST /v1/lint`, `GET /v1/base/pages`, `GET /v1/base/pages/{path}`, `POST /v1/doc/search`, `GET /v1/wisdom`, `POST /v1/wisdom/{id}/approve` | request / response JSON |
| **Async tasks** (seconds–minutes) | `POST /v1/{ingest,synth,distill,eval}` → `task_id`; `GET /v1/tasks/{id}/events` (NDJSON); `GET /v1/tasks/{id}/result`; `POST /v1/tasks/{id}/cancel` | submit JSON → stream NDJSON → final JSON |
| **Streaming query** | `POST /v1/query` | NDJSON: `query_started → retrieval_done → llm_token* → final` |
| **Upload** | `POST /v1/upload/sources` | multipart: tar.gz payload + manifest JSON |

Every error follows one envelope:

```json
{ "error": { "code": "not_found", "message": "...", "detail": {...} } }
```

`code` is the stable identifier — clients branch on it, never on the
free-form `message`.

## Bind and authentication

`dikw serve` binds to `127.0.0.1:8765` by default. There is **no
authentication on loopback** — the implicit threat model is "the user
running the CLI also owns the wiki on disk." If you need to expose the
server to other hosts:

```bash
export DIKW_SERVER_TOKEN=$(openssl rand -hex 32)
uv run dikw serve --wiki ./my-wiki --host 0.0.0.0 --token $DIKW_SERVER_TOKEN
```

Hard rule, enforced at startup: **`--host 0.0.0.0` (or any non-loopback
address) refuses to start without a token.** The runtime would rather
fail loudly than silently expose an unauthenticated wiki to the
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

* **systemd** — `ExecStart=/usr/bin/dikw serve --wiki /var/lib/dikw/...`,
  `Restart=on-failure`. Set `Environment=DIKW_SERVER_TOKEN=...` (or
  `EnvironmentFile=` to a 600-perm file) so the token doesn't end up in
  the unit listing.
* **Docker** — base image with `uv pip install dikw-core[postgres]`,
  mount the wiki tree at `/wiki`, expose 8765. The server expects to
  own its bound wiki — don't share the same `.dikw/` directory across
  multiple containers.
* **Foreground / dev** — `uv run dikw serve` is fine for laptop work;
  `serve-and-run` is the right tool for one-shot commands.

The server lifespan boots storage on the first request, runs migrations
idempotently, and keeps adapters open until shutdown. SIGTERM triggers
a graceful drain (FastAPI lifespan teardown closes adapters before the
socket).

### Server-restart semantics for in-flight tasks

When the server restarts mid-task (e.g., systemd restart, OOM kill,
graceful shutdown), **any task previously in `running` status is marked
`failed{reason=server_restart}`** by the lifespan startup hook. The
TaskManager doesn't attempt to resume — engine ops are idempotent
(content-hash skip on ingest, deterministic page paths on synth) so the
correct recovery is to re-submit the task, not to half-resume one whose
in-memory state is gone.

### Storage concurrency

* **SQLite** — single writer at any time. `dikw serve` is the only
  intended writer; running `dikw serve` against a wiki that's also
  being mutated by a hand-edited script will trigger `database is
  locked` errors.
* **Postgres** — multiple `dikw serve` instances against one wiki are
  supported by the storage layer (each task is one transaction), but
  there's no orchestration logic. If you need multi-server topologies,
  put a load balancer in front and accept that ingest/synth/distill
  tasks racing on the same source will produce one winner per `(path,
  content_hash)` pair via storage-level upsert.
* **Filesystem backend** — single-writer only by design. Don't run two
  servers against the same vault.

### Observability

* `GET /v1/healthz` — liveness, no dependencies. Returns `{"status":"ok"}`
  immediately. Suitable for k8s readiness/liveness probes and the
  `serve-and-run` ready-poll.
* `GET /v1/readyz` — confirms the storage adapter is connected and
  migrated. Returns 503 during cold start until the lifespan startup
  hook completes.
* `GET /v1/info` — engine version, storage backend, configured
  providers (without secrets), auth posture. Useful for client-side
  schema-drift checks.
* `GET /v1/tasks?limit=...` — list of past tasks for debugging long
  ingests / failed synth runs. Persists across server restart (backed
  by the same storage adapter as the wiki itself).

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

* **Reverse proxies and NDJSON** — disable response buffering on any
  proxy that fronts the server. nginx: `proxy_buffering off;`.
  Traefik: `--providers.file ... HTTP middleware Buffering` removed.
  Without this, the client sees streaming events arrive in batches at
  the buffer flush boundary, which makes the LLM token streaming feel
  broken.
* **Heartbeat** — task event streams emit a `{"type":"heartbeat"}`
  event every 15s while idle, just to defeat reverse-proxy idle
  timeouts. Clients drop heartbeats at the transport layer; server
  operators don't need to do anything special.
* **Upload size** — `POST /v1/upload/sources` accepts up to 1 GiB by
  default. Override via `DIKW_SERVER_MAX_UPLOAD_BYTES=<int>`.

## Disabling lifecycle endpoints

For production wikis whose tree should never be re-scaffolded by a
client:

```bash
export DIKW_SERVER_DISABLE_INIT=1
uv run dikw serve --wiki /var/lib/dikw/prod
```

`POST /v1/init` then returns `409 Conflict {"code":"init_disabled"}`
regardless of whether the wiki tree exists. The CLI's `dikw client init`
falls through cleanly; the local-only `dikw init` (which writes files
directly) is unaffected.
