# Observability

dikw-core ships **OpenTelemetry** instrumentation across all three signals —
**traces**, **metrics**, and **logs** — so its runtime data integrates with any
OTLP-compatible backend (Jaeger / Tempo, Prometheus / Grafana, Datadog, Honeycomb,
…) instead of being a black box. This is the operator-facing observability channel;
the user-facing channel is the `ProgressReporter` event stream over NDJSON (see
[`server.md`](server.md)). Don't confuse the two.

The whole stack is an **optional `[otel]` extra and off by default** — a plain
`pip install dikw-core` pulls in zero OpenTelemetry packages, and with the extra
installed but telemetry disabled (or `OTEL_SDK_DISABLED` set) every span / metric /
log-hook is a hand-rolled no-op that costs ~nothing. You turn it on deliberately.

This doc is the operator cookbook: what's instrumented, how to enable it, a
ready-to-run validation stack, and a dashboard query cookbook. The shorter
in-context summary lives in [`server.md` → OpenTelemetry export](server.md#opentelemetry-export);
the engine-side seam is `src/dikw_core/telemetry.py` (see
[`architecture.md`](architecture.md)).

## What you get

With telemetry on at both ends, a single `dikw client <verb>` produces **one
distributed trace** that stitches the whole request path:

```
dikw-client (CLI process)
└─ HTTP POST /v1/base/...            ← client httpx span, injects W3C traceparent
   └─ /v1/base/... (FastAPI server span)   ← server adopts the traceparent
      └─ dikw.task.<op>             ← background-task ROOT span, LINKED to the request
         └─ dikw.<op>               ← engine op span (dikw.ingest / dikw.synth / …)
            ├─ dikw.retrieve.leg     ← one child per fusion leg (retrieve only)
            └─ chat <model>          ← gen_ai span per LLM call (token usage on it)
               └─ POST https://…     ← outbound provider HTTP call (httpx auto-span)
```

Token usage flows into metrics, and every log line emitted inside a span carries
the `trace_id` / `span_id` so a log aggregator pivots straight to the trace.

## Enabling it

### 1. Install the extra

```bash
uv pip install 'dikw-core[otel]'
# or, from a checkout:  uv sync --all-extras
```

The extra bundles `opentelemetry-{api,sdk}`, the OTLP/HTTP exporter, and the
FastAPI / httpx / logging instrumentations. Without it, the steps below are inert.

### 2. Server — the `telemetry:` section in `dikw.yml`

`dikw serve` reads its telemetry config from the base's `dikw.yml` and bootstraps
the SDK from the server lifespan (after config load):

```yaml
telemetry:
  enabled: true
  endpoint: http://localhost:4318   # OTLP/HTTP base; /v1/traces + /v1/metrics appended for you
  service_name: dikw-core           # shows up as service.name on every span/metric
  sample_ratio: 1.0                 # ParentBased(TraceIdRatio) head sampling — traces only
```

- `endpoint` is the OTLP/**HTTP** base URL of your collector (transport is fixed to
  OTLP/HTTP — no `grpcio` dependency). Leave it `null` to fall back to the standard
  `OTEL_EXPORTER_OTLP_ENDPOINT` env var (the SDK then appends the per-signal paths
  itself).
- `sample_ratio` governs **trace** sampling only; metrics are never sampled.
- `OTEL_SDK_DISABLED=true` is honoured as a kill-switch regardless of `enabled`.

### 3. Client — `OTEL_*` env vars

The remote `dikw client` CLI has no base config, so its telemetry is driven purely
by the standard `OTEL_*` env vars. It activates only when an OTLP endpoint is set
(and the extra is installed and `OTEL_SDK_DISABLED` is unset):

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_SERVICE_NAME=dikw-client   # optional; defaults to dikw-client
dikw client ingest                     # this trace now stitches to dikw serve
```

The bootstrap fires only for the `client` subgroup — local `version` / `init` /
`serve` / `auth` commands make no HTTP calls and pay zero cost. A plain
`dikw client` invocation with no `OTEL_*` env is a no-op. Because the CLI is
short-lived, an `atexit` hook drains the `BatchSpanProcessor` on exit so even a
sub-second command exports its span.

### 4. Logs — `DIKW_LOG_FORMAT=json`

Set `DIKW_LOG_FORMAT=json` (default `text` keeps the human-readable terminal line
byte-for-byte) and every log line becomes one JSON object — the form a log
aggregator parses:

```bash
DIKW_LOG_FORMAT=json dikw serve --base ./base
```

```json
{"ts": "2026-06-16 12:01:02,123", "level": "INFO", "logger": "dikw_core.api_ingest", "message": "ingest complete", "trace_id": "7b1f…", "span_id": "a3c…", "service": "dikw-core"}
```

`trace_id` / `span_id` / `service` appear only on records emitted **inside a span**
while telemetry is active (degrades away otherwise — no crash). `DIKW_LOG_FORMAT`
is an env var (not a `dikw.yml` field), like `DIKW_LOG_LEVEL`, because CLI parsing
happens before any base loads. It applies to both `dikw serve` and `dikw client`.
No OTLP **log export** handler is attached — log export over OTLP stays deferred;
the supported path today is shipping the JSON stdout to your log aggregator.

## What's instrumented

### Traces

| Span | Where | Key attributes |
|---|---|---|
| `dikw.task.<op>` | background-task root (linked to the request) | `dikw.op`, `dikw.task_id`, `dikw.base_id` |
| `dikw.ingest` | `api.ingest` | `dikw.layer=data` |
| `dikw.synth` | `api.synthesize` | `dikw.layer=knowledge` |
| `dikw.retrieve` | `api.retrieve` | `dikw.layer=info`, `dikw.retrieve.limit`, `dikw.retrieve.hit_count` |
| `dikw.retrieve.leg` | `domains/info/search.py` (per fusion leg) | `dikw.retrieval.leg` (`bm25`/`vector`/`asset`/`graph`), `dikw.retrieve.leg.hit_count` |
| `dikw.lint.propose` / `dikw.lint.apply` | `api_lint.py` | `dikw.layer=knowledge`, `dikw.op` |
| `chat <model>` | every LLM provider call (`gen_ai_span` / `trace_llm_stream`) | `gen_ai.operation.name=chat`, `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.{input,output}_tokens` (+ Anthropic `cache_read`/`cache_creation`) |
| `embeddings <model>` | every embedding provider call | `gen_ai.operation.name=embeddings`, `gen_ai.system`, `gen_ai.request.model` |
| `POST https://…` | outbound provider HTTP | httpx auto-instrumentation; W3C `traceparent` propagation |

Notes on span semantics:

- **`dikw.task.<op>` is a root span LINKED back to the request**, not a child. The
  detached `asyncio` task outlives the HTTP request span (the route returns `202`-style
  before the work finishes), so a parent-child edge would dangle — a `trace.Link` is
  the OTel idiom for request-triggered fire-and-forget work. The `dikw.<op>` engine
  spans nest under it; for a direct / eval caller with no task, the engine op span is
  the trace root.
- **A cancel is not an error.** An `asyncio.CancelledError` (cooperative cancel) or a
  `GeneratorExit` (a consumer abandoning a stream early) flags the span
  `dikw.cancelled=true` and leaves the status `UNSET`, so cancels never pollute
  error-rate dashboards. Only a real exception sets `StatusCode.ERROR`.

### Metrics — GenAI (OTel semantic convention)

Every LLM/embedding call — completed or failed — emits two histograms from the same
span seam, so there is no provider-specific metric code:

| Metric | Unit | Tags |
|---|---|---|
| `gen_ai.client.token.usage` | `{token}` | `gen_ai.token.type` (`input` / `output`, plus `cache_read` / `cache_creation` for Anthropic prompt caching), `gen_ai.operation.name`, `gen_ai.system`, `gen_ai.request.model` |
| `gen_ai.client.operation.duration` | `s` | same base tags, plus `error.type` on a failed call |

A cancelled / abandoned stream records **no** duration point (its cut-short time
would skew the latency series). The two histograms use explicit bucket boundaries
tuned for token counts (1 → 64 M) and LLM latencies (10 ms → 80 s) — the SDK
defaults (top out at 10 k) fit neither.

FastAPI HTTP-server metrics (`http.server.request.duration`, …) flow for free from
the same meter provider.

### Metrics — dikw domain

Mapped from the per-call report DTOs + existing instrumentation, so a dashboard sees
pipeline volume without parsing logs. Counters skip zero-valued series to keep
cardinality honest.

| Metric | Type | Tags | Source |
|---|---|---|---|
| `dikw.ingest.files` | Counter | `dikw.result` (`added`/`updated`/`unchanged`) | final `IngestReport` |
| `dikw.ingest.chunks` | Counter | — | final `IngestReport` |
| `dikw.ingest.errors` | Counter | `dikw.error.kind` | final `IngestReport` |
| `dikw.synth.pages` | Counter | `dikw.result` (`created`/`updated`) | final `SynthReport` |
| `dikw.synth.unresolved_wikilinks` | Counter | — | final `SynthReport` |
| `dikw.synth.persist_errors` | Counter | — | final `SynthReport` |
| `dikw.embed.chunks` / `dikw.embed.skipped` / `dikw.embed.retries` | Counter | — | shared embed consume seam (D/K/W) |
| `dikw.retrieve.leg.duration` | Histogram (`s`) | `dikw.retrieval.leg` | per fusion leg |
| `dikw.task.duration` | Histogram (`s`) | `dikw.op`, `dikw.status` (`ok`/`error`/`cancelled`) | task runner |

Unlike the GenAI duration metric, a **cancelled task** does record its duration —
on its own `dikw.status=cancelled` series, so it never skews the `ok` latency.

### Logs

When `DIKW_LOG_FORMAT=json` and telemetry is active, each in-span record carries
`trace_id` / `span_id` / `service` (a `LoggingInstrumentor` log hook stamps them;
`init_logging` keeps handler + format ownership). Point your log shipper at the
JSON on stdout and correlate on `trace_id`.

## Run the validation stack

A self-contained OTLP backend — **OTel Collector → Jaeger (traces) + Prometheus
(metrics) + Grafana (dashboards)** — lives at
[`observability/docker-compose.yml`](observability/docker-compose.yml). It is a
*local validation / demo* stack, not a production deployment (see
[`deployment-docker.md`](deployment-docker.md) for that).

```bash
cd docs/observability
docker compose up -d
# Collector now listens on http://localhost:4318 (OTLP/HTTP)
```

Point dikw at the collector and generate some traffic:

```bash
# Server-side (in the base's dikw.yml):
#   telemetry: { enabled: true, endpoint: http://localhost:4318 }
dikw serve --base ./base &

# Client-side:
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
dikw client ingest
dikw client synth --all
dikw client retrieve "some query"
```

Then open:

- **Jaeger** — <http://localhost:16686> — pick service `dikw-core` (or `dikw-client`)
  and find a trace; you should see the full client → server → task → engine → provider
  tree above.
- **Prometheus** — <http://localhost:9090> — query `gen_ai_client_token_usage_count`
  or `dikw_ingest_files_total` to confirm metrics arrive.
- **Grafana** — <http://localhost:3000> (anonymous admin, no login) — the Prometheus
  and Jaeger datasources are pre-provisioned; build panels from the cookbook below.

Tear down with `docker compose down -v`.

## Dashboard cookbook

The Collector's Prometheus exporter sanitizes OTLP metric names: dots become
underscores, counters/histograms gain the usual `_total` / `_bucket` / `_sum` /
`_count` suffixes, and — by default — a metric's **unit** is appended too, so an
`s`-unit histogram like `gen_ai.client.operation.duration` becomes
`gen_ai_client_operation_duration_seconds_*` (annotation units in braces like
`{token}` / `{file}` are dropped, so those keep no unit infix). Exact spelling
depends on the collector/Prometheus version, so confirm names under
<http://localhost:9090> first. Representative PromQL:

**GenAI token spend, by model** (the high-value cost panel):

```promql
sum by (gen_ai_request_model, gen_ai_token_type) (
  rate(gen_ai_client_token_usage_sum[5m])
)
```

**LLM call latency, p95, by model:**

```promql
histogram_quantile(0.95,
  sum by (le, gen_ai_request_model) (rate(gen_ai_client_operation_duration_seconds_bucket[5m]))
)
```

**Anthropic prompt-cache hit ratio** (cache reads vs. all input tokens — caching
reports `cache_read` / `cache_creation` separately so you can split the cost tiers):

```promql
sum(rate(gen_ai_client_token_usage_sum{gen_ai_token_type="cache_read"}[5m]))
/
sum(rate(gen_ai_client_token_usage_sum{gen_ai_token_type=~"input|cache_read|cache_creation"}[5m]))
```

**Ingest / synth pipeline volume:**

```promql
sum by (dikw_result) (rate(dikw_ingest_files_total[5m]))
sum by (dikw_result) (rate(dikw_synth_pages_total[5m]))
```

**Retrieval leg latency, p95, by leg:**

```promql
histogram_quantile(0.95,
  sum by (le, dikw_retrieval_leg) (rate(dikw_retrieve_leg_duration_seconds_bucket[5m]))
)
```

For traces, use Grafana's Jaeger datasource (or Jaeger's own UI) to drill into a
slow `dikw.synth` and see which `chat <model>` call dominated, then pivot to the
correlated logs via `trace_id`.

## Zero-cost guarantee

The instrumentation is engineered to be free when unused:

- **No `[otel]` extra installed** → `get_tracer()` / `get_meter()` return hand-rolled
  no-op objects; every `op_span` / `gen_ai_span` / `record_*` call is a method that
  does nothing. Engine code emits spans/metrics unconditionally and never branches on
  whether telemetry is present.
- **Extra installed but telemetry off** (config `enabled: false`, no client `OTEL_*`,
  or `OTEL_SDK_DISABLED`) → the SDK is never bootstrapped, so the API objects stay in
  their no-op-until-provider state and metric instruments are never created (the
  `record_*` helpers gate on dikw's own meter provider).
- **Telemetry must never break the engine.** Every metric-recording helper is wrapped
  so a telemetry failure logs at `debug` and never propagates into the ingest / synth /
  task path it runs in.

## How it's wired (layering)

The engine instruments against the OTel **API** and lets the operator wire the
**SDK** at the process entry — exactly mirroring how `init_logging` is wired from
the CLI / app factory:

- `src/dikw_core/telemetry.py` sits at the engine root and imports only
  `opentelemetry` (optional) + stdlib — **never** `server` / FastAPI. Engine modules
  call its accessors (`get_tracer` / `get_meter`) + the `gen_ai_span` / `op_span` /
  `record_*` helpers + the `dikw.*` / `gen_ai.*` attribute-key constants.
- **Only the entry point calls `configure_telemetry`** — the server lifespan
  (`server/runtime.py`) after cfg load, or `configure_client_telemetry_from_env` from
  the `dikw client` CLI root. The FastAPI auto-instrumentation lives in
  `server/app.py` (server code may import FastAPI), gated on the resolved telemetry
  setting (`telemetry_should_activate` — `enabled` AND `[otel]` installed AND not
  `OTEL_SDK_DISABLED`), so the web-framework import never leaks into the engine and
  a disabled server adds no middleware.
- The `client/*` package depends only on `schemas` + stdlib + httpx + typer + rich and
  imports **no** `telemetry` symbol — its spans + `traceparent` injection come for free
  from the global `HTTPXClientInstrumentor` patch, keeping the client a clean
  standalone-wheel candidate.

See [`architecture.md`](architecture.md) for the full module map and seam contracts.
