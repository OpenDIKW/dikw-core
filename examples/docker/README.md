# dikw-core Docker example

A ready-to-run compose stack that boots `dikw serve` against a pinned
`pgvector/pgvector:0.8.2-pg18` Postgres backend. The image installs
`dikw-core[postgres]` from PyPI — no source build required.

Full walkthrough (image internals, bootstrap, SQLite-only variant,
upgrade flow, multi-container caveats):
[`../../docs/deployment-docker.md`](../../docs/deployment-docker.md).

## Quick start

```bash
cp .env.example .env       # then edit secrets
mkdir base
docker compose run --rm dikw-core init /base
# edit base/dikw.yml — storage.backend: postgres + the libpq keyword form:
#   storage:
#     backend: postgres
#     dsn: "host=postgres port=5432 user=dikw password=<POSTGRES_PASSWORD> dbname=dikw"
docker compose up -d
set -a; . ./.env; set +a   # only needed for the next curl line
curl -H "Authorization: Bearer $DIKW_SERVER_TOKEN" http://localhost:8765/v1/healthz
```

## Files

| File | Purpose |
| --- | --- |
| `Dockerfile` | `python:3.12-slim` + `pip install dikw-core[postgres]` |
| `docker-compose.yml` | dikw-core + Postgres services with health checks |
| `.env.example` | Required and optional environment variables |
| `pg-init/01-extensions.sql` | Creates `vector` and `pg_trgm` extensions on first start |

## Pinning a specific dikw-core version

```bash
docker compose build --build-arg DIKW_VERSION=0.2.7
```

The Dockerfile default `DIKW_VERSION` advances via an auto-opened
chore PR after each PyPI publish — see the `sync-dockerfile` job in
`.github/workflows/release.yml`. Merging that PR moves the default on
`main` forward; until the PR lands, `main` keeps the prior default.
The `dockerfile-version-guard` job in
`.github/workflows/reusable-ci.yml` fails CI on any default that's
neither equal to `pyproject.toml`'s version nor already published on
PyPI.
