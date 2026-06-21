# dikw-core Docker debug environment

A ready-to-run compose stack that boots `dikw serve` against a pinned
`pgvector/pgvector:0.8.2-pg18` Postgres backend — a stable local
environment for **downstream systems** to develop and debug their
HTTP / `dikw client` integration against a real dikw-core.

By default it **pulls the official multi-arch image published to GHCR**
on each release (`ghcr.io/opendikw/dikw-core:<version>`, `linux/amd64`
+ `linux/arm64`) — no source build required. The version is pinned via
`DIKW_VERSION` so the environment always rides a known release.

Full walkthrough (image internals, bootstrap, SQLite-only variant,
upgrade flow, multi-container caveats):
[`../../docs/deployment-docker.md`](../../docs/deployment-docker.md).

## Quick start (pull the official image)

```bash
cp .env.example .env       # set DIKW_VERSION + secrets (real provider keys)
mkdir base
docker compose pull        # fetch ghcr.io/opendikw/dikw-core:$DIKW_VERSION
docker compose run --rm dikw-core init /base
# edit base/dikw.yml — storage.backend: postgres + the libpq keyword form:
#   storage:
#     backend: postgres
#     dsn: "host=postgres port=5432 user=dikw password=<POSTGRES_PASSWORD> dbname=dikw"
#   provider: ...   # your LLM/embedding base URLs, models, and *_api_key_env
docker compose up -d
set -a; . ./.env; set +a   # only needed for the next curl line
curl -H "Authorization: Bearer $DIKW_SERVER_TOKEN" http://localhost:8765/v1/healthz
```

The base starts empty — bring your own content: drop markdown under
`base/sources/`, then `docker compose run --rm dikw-core client ingest`
(and `… client synth`) to populate the D / K layers, and hit the server
over HTTP or `dikw client …` from your downstream system.

> The GHCR package is public — `docker compose pull` needs no login. If
> you ever pin a yanked/unpublished tag the pull fails fast with a clear
> manifest-not-found error.

## Files

| File | Purpose |
| --- | --- |
| `docker-compose.yml` | dikw-core (GHCR image, `build:` fallback) + Postgres, with health checks |
| `Dockerfile` | local-source fallback: `python:3.12-slim` + `pip install dikw-core[postgres]` |
| `.env.example` | `DIKW_VERSION` pin + required/optional environment variables |
| `pg-init/01-extensions.sql` | Creates `vector` and `pg_trgm` extensions on first start |

## Pinning / changing the version

Edit `DIKW_VERSION` in `.env` (see the
[releases](https://github.com/OpenDIKW/dikw-core/releases)) and
`docker compose pull && docker compose up -d`. Pinning an exact
`X.Y.Z` keeps the debug environment reproducible — there is no
floating `:latest` tag, by design (dikw-core is alpha; breaking
changes can land in any minor).

`dikw client` runs a version handshake on every call: if your client's
installed `dikw-core` differs from the server's `engine_version` it
hard-fails with a `version skew:` message. Keep the client and
`DIKW_VERSION` on the same release, or set `DIKW_ALLOW_VERSION_SKEW=1`
for deliberate mixed-version debugging.

## Building from local source instead

To debug an unreleased change, rebuild from the working tree's
Dockerfile rather than pulling:

```bash
docker compose up -d --build --build-arg DIKW_VERSION=0.2.7
```

The Dockerfile default `DIKW_VERSION` advances via an auto-opened
chore PR after each PyPI publish — see the `sync-dockerfile` job in
`.github/workflows/release.yml`. The published GHCR image is built from
this same Dockerfile (see the `publish-image` job), so the image you
pull is the one that's Trivy-scanned on every PR.
