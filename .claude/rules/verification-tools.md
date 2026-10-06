---
paths:
  - "tools/**"
  - "tests/test_e2e_verify_*.py"
  - ".github/workflows/**"
  - ".pre-commit-config.yaml"
---

# Local gate and real-environment e2e

Read this rule before you change `tools/check.py`, `tools/e2e_verify.py`, or CI.

`tools/check.py` is the single in-loop gate to run before every commit; it runs
the same ruff/mypy/pytest CI runs, in CI order, without `--cov` (which flakes
ASGI/CliRunner tests locally on Windows).

`tools/e2e_verify.py` is the real-environment e2e harness (delivery-loop step 3,
`cli/server/client` bucket): it spins a throwaway server, drives **every** `dikw
client` verb (coverage asserted against the live Typer tree, so a new verb without
a step fails the run), then tears the environment down. Structural legs run with no
keys; the real-provider legs (`check`/embed/`synth`/vector-`retrieve`/`eval`) run
when `.env` carries the key vars named by the active profile's
`provider.{llm,embedding}_api_key_env` (the default profile uses
`MINIMAX_API_KEY` + `GITEE_API_KEY`), else they SKIP
loudly. `--mode docker` builds the image from the local tree (not the released
`examples/docker/Dockerfile`). Wrapped by `tests/test_e2e_verify_{local,docker}.py`
(`-m slow`).

CI (`.github/workflows/ci.yml`) gates PRs on ruff + mypy + pytest across
Python 3.12 and 3.13, and runs the storage contract suite against a
`pgvector/pgvector:0.8.2-pg18` Postgres service. Release tags (`vX.Y.Z`) publish
to PyPI via trusted publishing (`.github/workflows/release.yml`).

Tooling config lives in `pyproject.toml`:
- ruff: line-length 100, rules `E,F,W,I,UP,B,SIM,C4,RUF` (E501 ignored)
- mypy: `strict = true`, `packages = ["dikw_core"]`, `mypy_path = "src"`
- pytest: `asyncio_mode = "auto"`, `testpaths = ["tests"]`
