---
paths:
  - "src/dikw_core/logging.py"
  - "src/dikw_core/telemetry.py"
  - "src/dikw_core/server/app.py"
  - "src/dikw_core/client/cli_app.py"
  - "docs/observability.md"
  - "docs/observability/**"
  - "tests/test_{logging,telemetry,engine_op_spans,synth_observability}*.py"
---

# Logging and telemetry

Read these rules before you add logging, spans, metrics, or telemetry bootstrap code.

- Logging: `DIKW_LOG_LEVEL` (DEBUG/INFO/WARNING/ERROR/CRITICAL, default INFO) controls the root logger level for both CLI and `dikw serve`. `DIKW_LOG_FORMAT` selects the line shape — default `text` keeps the human-readable terminal formatter byte-for-byte, `json` opts into one JSON object per record (`_JsonFormatter`, stdlib-only) that surfaces `trace_id`/`span_id`/`service` for log↔trace correlation (injected by the OTel `LoggingInstrumentor` log hook wired in `telemetry.configure_telemetry`, `enable_log_auto_instrumentation=False` so `init_logging` keeps handler/format ownership; degrades gracefully without the `[otel]` extra or outside a span) and passes through `extra={…}` fields. Both are env vars (not `dikw.yml` fields) because CLI parsing happens before any base is loaded. `init_logging()` is idempotent — safe to wire from multiple entry points; non-`dikw_core` loggers (httpx, httpcore, urllib3) are clamped to WARNING so per-request noise doesn't drown synth/embed progress.
- Telemetry: OpenTelemetry is an **optional `[otel]` extra, off by default**. Engine code (`api_*`, `domains`, `providers`, server task subsystem) emits spans/metrics **only** through `telemetry.py`'s accessors (`get_tracer`/`get_meter`) + helpers (`gen_ai_span`/`op_span`/`traced_op`/`task_span`, `record_*`) + the `dikw.*`/`gen_ai.*` attribute-key constants — never `import opentelemetry` directly, so the extra stays optional (no-op shim when absent) and the no-op + active paths share one gate. `telemetry.py` imports only `opentelemetry`+stdlib, **never** `server`/FastAPI (engine-root layering). **Only the entry point bootstraps the SDK** — the server lifespan calls `configure_telemetry(**cfg.telemetry)` after cfg load (`dikw.yml` `telemetry:` section: `enabled`/`endpoint`/`service_name`/`sample_ratio`), the `dikw client` CLI root calls `configure_client_telemetry_from_env` (driven by standard `OTEL_*` env, no base cfg) — mirroring how `init_logging` is wired from entry points, never from engine code. `OTEL_SDK_DISABLED` is a kill-switch; metric `record_*` helpers gate on dikw's own meter provider and are wrapped so a telemetry failure logs at debug and never propagates into ingest/synth/task. Full operator cookbook + validation compose stack: `docs/observability.md`.
