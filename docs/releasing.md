# Releasing

Maintainer-facing notes for cutting a `dikw-core` release. End users do not
need any of this — see [`getting-started.md`](./getting-started.md) instead.

## TL;DR

The version is declared in exactly one place — `pyproject.toml` (`[project].version`).
Every runtime read (`dikw version`, `/v1/info`, health, the client/server version
handshake) resolves it at runtime via `importlib.metadata.version("dikw-core")`, so a
release is: bump that one field, write the CHANGELOG section, tag, push.

```bash
# 1. Bump pyproject.toml [project].version to X.Y.Z (and let uv.lock pick it up).
# 2. Rename the CHANGELOG "## Unreleased" section to "## X.Y.Z — <subtitle>"
#    and add a fresh empty "## Unreleased" above it.
# 3. Open a normal PR, get it green + merged.
# 4. Tag the merged commit and push the tag:
git tag vX.Y.Z
git push origin vX.Y.Z
```

Pushing the tag is the irreversible, outward-facing step — it publishes to PyPI
(a version number can never be reused) and to GHCR.

## What the tag triggers

A tagged push (`v*`) runs [`.github/workflows/release.yml`](../.github/workflows/release.yml):

1. **`pre-release-gate`** — the full test matrix, same as a PR gate.
2. **`build`** — `uv build` produces the sdist + wheel.
3. **`publish`** — uploads to [PyPI via **trusted publishing**](https://docs.pypi.org/trusted-publishers/)
   (OIDC, no API token in repo secrets), in the `pypi` GitHub environment.
4. **`publish-image`** — waits for the PyPI CDN to serve the new version, then builds
   a multi-arch (`linux/amd64` + `linux/arm64`) image and pushes
   `ghcr.io/opendikw/dikw-core:X.Y.Z`. There is **no `:latest` tag** — deployments
   pin an explicit version.
5. **`github-release`** — creates the GitHub Release for the tag. The body is this
   version's `CHANGELOG.md` section (falling back to auto-generated notes if the
   section is missing); the built wheel + sdist are attached as assets.
6. **`sync-dockerfile`** — opens a `chore(docker): bump DIKW_VERSION to vX.Y.Z` PR
   against `main`, keeping [`examples/docker/Dockerfile`](../examples/docker/Dockerfile)
   in lockstep with the latest published wheel. Merge it to clear the post-release
   queue. The `dockerfile-version-guard` job in `reusable-ci.yml` enforces the
   invariant on every PR (the Dockerfile's `ARG DIKW_VERSION` must equal `pyproject`
   or already be published on PyPI), so it tolerates the Dockerfile lagging by one
   release until that chore PR merges.

## One-time setup

### PyPI trusted publisher

On the `dikw-core` project's *Publishing* page on PyPI, add a GitHub trusted
publisher:

- owner: `OpenDIKW`
- repository: `dikw-core`
- workflow: `release.yml`
- environment: `pypi`

After that, `git tag vX.Y.Z && git push origin vX.Y.Z` is sufficient — no token
lives in repo secrets.

### `RELEASE_PR_PAT` secret (for the Dockerfile-bump PR)

The `sync-dockerfile` chore PR's required CI only runs if the PR is opened with a
PAT: a PR opened by the default `GITHUB_TOKEN` cannot trigger other workflows, so
its checks stay `action_required` and it is born blocked. Add a repo secret
**`RELEASE_PR_PAT`** holding a fine-grained PAT scoped to this repo with
**Contents: read/write** + **Pull requests: read/write** (no Workflows scope —
this PR only edits the Dockerfile). With it set, the chore PR runs CI normally and
is directly mergeable. Without it, the PR still opens but you must push an empty
commit (or close/re-open) to run its gate.

### GHCR package visibility (first publish only)

The **first** time `publish-image` runs, GHCR creates the
`ghcr.io/opendikw/dikw-core` package **private** by default, so the anonymous
`docker compose pull` that the compose stack and docs assume fails with a 401.
Making it public is a one-time, web-UI-only fix (no REST API for container-package
visibility):

1. **Org policy** — at `https://github.com/organizations/OpenDIKW/settings/packages`,
   under "Package creation", ensure **Public** is allowed.
2. **Package visibility** — at
   `https://github.com/orgs/OpenDIKW/packages/container/dikw-core/settings`,
   Danger zone → Change visibility → **Public**.

Verify anonymously:

```bash
curl -s "https://ghcr.io/token?scope=repository:opendikw/dikw-core:pull"
# public → {"token":"..."}; private → {"errors":[{"code":"UNAUTHORIZED",...}]}
```

Every subsequent release's tag is then anonymously pullable — this is per-package,
not per-release.

## Gotchas

- **Trivy CDN race on the bump PR.** The `Scan dikw-core image` (Trivy) check on the
  Dockerfile-bump PR can fail fast with `pip install dikw-core[...]==X.Y.Z` →
  `No matching distribution found`, because Trivy builds the image locally before
  PyPI's simple index has propagated the just-published version. It is a transient
  CDN race, not a real failure: wait until `https://pypi.org/simple/dikw-core/`
  lists the version, then re-run the failed Trivy job. Trivy is non-blocking, but
  re-running confirms the image actually builds.
