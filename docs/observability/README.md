# dikw-core observability validation stack

A self-contained OTLP backend for **local validation / demo** — not a production
deployment. See [`../observability.md`](../observability.md) for the full operator
cookbook (what's instrumented, enabling it, dashboard queries) and
[`../deployment-docker.md`](../deployment-docker.md) for running `dikw serve` itself
in a container.

```
OTel Collector (:4318 OTLP/HTTP)  →  Jaeger (traces, UI :16686)
                                  →  Prometheus (metrics, UI :9090)  →  Grafana (UI :3000)
```

## Quick start

```bash
cd docs/observability
docker compose up -d
```

The collector now listens on `http://localhost:4318` (OTLP/HTTP). Point dikw at it:

- **Server** — in the base's `dikw.yml`:
  ```yaml
  telemetry:
    enabled: true
    endpoint: http://localhost:4318
  ```
  then `dikw serve --base ./base`.
- **Client** — `export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318` before a
  `dikw client …` command.

Generate some traffic (`dikw client ingest` / `dikw client synth --all` /
`dikw client retrieve "…"`), then open:

| UI | URL | Use |
|---|---|---|
| Jaeger | <http://localhost:16686> | find a trace for service `dikw-core` / `dikw-client` |
| Prometheus | <http://localhost:9090> | confirm metrics arrive (`gen_ai_client_token_usage_count`) |
| Grafana | <http://localhost:3000> | datasources pre-provisioned; build panels from the cookbook |

Grafana opens with anonymous admin access (no login) — **demo posture only**.

## Tear down

```bash
docker compose down -v
```

## Files

| File | Role |
|---|---|
| `docker-compose.yml` | the four-service stack |
| `otel-collector-config.yaml` | OTLP receiver → Jaeger (traces) + Prometheus exporter (metrics) |
| `prometheus.yml` | scrapes the collector's `:8889` Prometheus endpoint |
| `grafana/provisioning/datasources/datasources.yml` | Prometheus + Jaeger datasources |
