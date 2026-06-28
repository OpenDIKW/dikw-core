# Docker deployment

This doc covers running `dikw serve` from a container, including the
`examples/docker/` compose stack. For systemd / foreground patterns and
shared operational concerns (restart semantics, storage concurrency,
auth model), see [`server.md`](server.md).

The example stack lives at [`../examples/docker/`](../examples/docker/).
This document is the long-form companion — the example's `README.md`
keeps the quick-start under a screenful.

## What the image contains

`examples/docker/Dockerfile` builds a single-layer image:

- **Base**: `python:3.12-slim` (not `alpine`). The `[postgres]` extra
  pulls `psycopg[binary,pool]` and `pgvector`, both of which ship glibc
  wheels but not musl wheels — `alpine` would force a source build with
  PostgreSQL headers and a C toolchain. The slim image is ~140 MB before
  the install; the final image lands around 280 MB.
- **Install**: `pip install dikw-core[postgres]==${DIKW_VERSION}` from
  PyPI. No source tree is copied — the image is a pure consumer of the
  published wheel, which means the build is reproducible per version and
  the build context stays empty.
- **User**: non-root `dikw` (UID 1000). Mount points need
  `chown 1000:1000` on the host (or use Docker Desktop's user namespacing).
- **Entry**: `ENTRYPOINT ["dikw"]` plus `CMD ["serve", "--base", "/base",
  "--host", "0.0.0.0", "--port", "8765"]`. The split lets you reuse the
  locally built image (`docker compose build` tags it as
  `docker-dikw-core` by default) for one-shot commands:

  ```bash
  docker run --rm -v ./base:/base docker-dikw-core init /base
  docker run --rm docker-dikw-core version
  ```

  Each release publishes this same image to GHCR (the `publish-image`
  job in `.github/workflows/release.yml`) as
  `ghcr.io/opendikw/dikw-core:X.Y.Z` — public, multi-arch
  (`linux/amd64` + `linux/arm64`), no leading `v`. The compose stack
  pulls it by default (pinned via `DIKW_VERSION`); `docker run` works
  against it too:

  ```bash
  docker run --rm -v ./base:/base ghcr.io/opendikw/dikw-core:0.6.4 init /base
  docker run --rm ghcr.io/opendikw/dikw-core:0.6.4 version
  ```

  There is intentionally **no floating `:latest`** — downstream pins an
  exact release so its debug environment stays reproducible.

  > **Maintainer, one-time:** a GHCR package pushed by `GITHUB_TOKEN` is
  > created **private**. After the first release that runs `publish-image`,
  > set the package to **Public** (org → Packages → `dikw-core` → Package
  > settings → Change visibility) so downstream `docker compose pull`
  > needs no login.

## The compose stack

`examples/docker/docker-compose.yml` runs two services:

```
┌──────────────────┐         ┌──────────────────────────────────────┐
│  dikw-core       │ ──────▶ │  pgvector/pgvector:0.8.2-pg18        │
│  (port 8765)     │  5432   │  (volume: postgres-data)             │
│  volume: ./base  │         │  init: pg-init/01-extensions.sql     │
└──────────────────┘         └──────────────────────────────────────┘
```

- **Postgres image pin**: `pgvector/pgvector:0.8.2-pg18` ties the
  pgvector extension to v0.8.2 and Postgres to the 18.x line. The CI
  pipeline uses the same tag, so passing CI implies the same backend
  the compose stack uses. Bare `pg18` tracks upstream `latest` and is
  not safe for environments where you care about reproducibility.
- **Extensions**: `pg-init/01-extensions.sql` runs once on first
  Postgres start (`/docker-entrypoint-initdb.d` convention) and issues
  `CREATE EXTENSION IF NOT EXISTS vector;` plus the same for `pg_trgm`.
  The image bundles the extension files; `CREATE EXTENSION` still has
  to fire per-database.
- **Health checks**: Postgres uses `pg_isready`. dikw-core hits
  `/v1/healthz` from inside the container — the probe has to pass
  `DIKW_SERVER_TOKEN` because the whole `/v1` router is auth-gated.
  For orchestrators that don't propagate secrets into the probe
  (vanilla k8s liveness, some PaaS) you'll need a sidecar or wrapper
  that injects the header; a future unauthenticated `GET /healthz`
  outside the `/v1` prefix would remove the need for that.
  `dikw-core` declares `depends_on: postgres: condition: service_healthy`,
  so the engine doesn't start until Postgres is accepting connections.

## First-run bootstrap

The compose stack does **not** auto-generate `dikw.yml`. The reason:
`dikw init` is a deliberate prerequisite, and hiding it behind an
entrypoint shim would conceal the real first-run surface — meaning you'd
hit a confusing failure later when the file's contents matter (provider
config, dim locking, prompt overrides).

> **Windows + Git Bash gotcha**: `docker compose run --rm dikw-core init /base`
> from Git Bash silently rewrites `/base` to `C:/Program Files/Git/base`
> via MSYS path translation, so the container creates a nested
> `/base/C:/Program Files/Git/base` tree instead of populating `/base`.
> Use PowerShell or cmd, or prefix the command with `MSYS_NO_PATHCONV=1`.

Concrete steps:

```bash
cd examples/docker
cp .env.example .env
# Edit .env: POSTGRES_PASSWORD, DIKW_SERVER_TOKEN are required;
# plus the key vars your base's dikw.yml names via
# provider.llm_api_key_env / provider.embedding_api_key_env
# (e.g. ANTHROPIC_API_KEY for the LLM + GITEE_API_KEY for embeddings).

mkdir base
docker compose run --rm dikw-core init /base
```

The `run --rm` call mounts `./base` into a one-shot container, executes
`dikw init /base`, and exits. You now have:

```
base/
├── dikw.yml
├── knowledge/
├── wisdom/
├── sources/
└── .dikw/
```

Now wire Postgres into the base storage layer by editing
`base/dikw.yml`:

```yaml
storage:
  backend: postgres
  dsn: "host=postgres port=5432 user=dikw password=<POSTGRES_PASSWORD> dbname=dikw"
```

(Substitute the password you put in `.env`. The hostname `postgres`
matches the compose service name — Docker networking resolves it.)
Both the base DSN above and `DIKW_SERVER_TASKS_DSN` in `docker-compose.yml`
use libpq's keyword conninfo form (`host=… password=… dbname=…`) rather
than `postgresql://` URLs, so generated strong passwords containing
`/ # ? % @ :` work without URL-encoding.

Finally:

```bash
docker compose up -d
# .env is only read by Compose; export the token into your shell for curl:
set -a; . ./.env; set +a
curl -H "Authorization: Bearer $DIKW_SERVER_TOKEN" http://localhost:8765/v1/healthz
```

## Two DSNs, two paths

The compose stack reveals an asymmetry that's intentional:

| Storage layer | Configured via | Why |
| --- | --- | --- |
| Base documents / chunks / embeddings | `storage.dsn` in `dikw.yml` | Base storage is per-base and tied to the on-disk format declared in `dikw.yml`. Cross-base coordination is impossible. |
| Server task store | `DIKW_SERVER_TASKS_DSN` env var | Task store is server-scoped (not per-base), and in production it's the operator's choice whether to share an instance with the base DB. Env var keeps it out of the on-disk format. |

In the example compose, both point at the same Postgres instance (the
task store creates its own table; no cross-talk). In production you may
want them on different clusters.

## Variant: SQLite only (no Postgres service)

If you don't need Postgres, trim the stack:

1. Leave `base/dikw.yml` at its default (`storage.backend: sqlite`).
2. Remove the `postgres` service and `depends_on` from
   `docker-compose.yml`.
3. Drop the `DIKW_SERVER_TASKS_DSN` env from the `dikw-core` service —
   the task store falls back to `base/.dikw/server-tasks.db`.
4. Drop the `volumes:` block at the bottom.

Base data and task tape both land under `./base/.dikw/` then. This is
the right setup for single-machine personal use.

## Upgrading

Pulling the published image is the default path — bump `DIKW_VERSION`
in `.env` to a newer [release](https://github.com/OpenDIKW/dikw-core/releases)
and re-pull:

```bash
docker compose pull        # fetch ghcr.io/opendikw/dikw-core:$DIKW_VERSION
docker compose up -d
```

To upgrade by rebuilding from local source instead (e.g. an unreleased
change), use the `build:` fallback:

```bash
docker compose up -d --build --build-arg DIKW_VERSION=0.2.7
```

Omit `--build-arg` to use the default baked into the Dockerfile. That
default tracks the latest PyPI release via an auto-opened chore PR
after each publish (see the `sync-dockerfile` job in
`.github/workflows/release.yml`) — there's a short window after a new
release where the default still points at the prior version until that
PR lands.

## Single-writer constraint

Repeating the warning from [`server.md`](server.md): the server expects
to own its bound base. **Don't** run two `dikw-core` containers against
the same `./base` mount — `.dikw/base_id` and SQLite write locks both
assume single-writer.

The compose stack sets `DIKW_TASK_REAP_ON_START=1` for the same reason:
the Postgres task store skips orphan reaping by default (so a peer's
in-flight tasks aren't clobbered in a multi-replica deployment), and a
single-server stack has to opt back in or it leaves `running` tasks
stuck after every restart. If you ever fan out to multiple `dikw-core`
replicas sharing one task DSN, **unset** this env var.
