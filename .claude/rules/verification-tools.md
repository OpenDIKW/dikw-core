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
the same harness-doctor, ruff, mypy and pytest checks as CI, in CI order, without `--cov` (which flakes
ASGI/CliRunner tests locally on Windows), and writes a gitignored verification
stamp (`.claude/.verify-stamp.json`) when all stages pass and the working tree
stayed unchanged during the run.

The three clients call `tools/verify_hook.py` before commit commands. Mode is controlled by `DIKW_VERIFY_GATE_MODE`
(`off` / `observe` / `enforce` [default]): `enforce` blocks unverified commits
(exit 2) unless the target worktree's stamp matches the tree and staged
content, the command carries `[skip-verify: <reason>]`, or the same session has
already been blocked twice (escape hatch); `observe` logs unverified commits to
`.claude/verify-gate-observe.jsonl` without blocking.

Codex 0.160.0 reports the shell command without its execution directory. Use `git -C "<absolute worktree path>" commit` in Codex so the hook can verify the intended worktree. Forward slashes and quotes work in both PowerShell and Bash.

`.github/workflows/eval-gate.yml` runs on `pull_request_target` with read-only
`contents` permission, checking out the base commit and fetching the PR head
strictly as git diff objects, so both the workflow and `tools/check_baselines.py`
stay base-branch-controlled and a PR cannot weaken the gate that judges it.

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
Python 3.12 and 3.13, runs the storage contract and SQLite-vs-Postgres
differential suites (`tests/test_storage_differential.py`) against a
`pgvector/pgvector:0.8.2-pg18` Postgres service, and runs hermetic
`@pytest.mark.perf` budgets (`perf-budgets` job). A scheduled/manual workflow
(`.github/workflows/e2e-nightly.yml`) exercises `tools/e2e_verify.py --mode docker`.
Release tags (`vX.Y.Z`) publish to PyPI via trusted publishing
(`.github/workflows/release.yml`).

Tooling config lives in `pyproject.toml`:
- ruff: line-length 100, rules `E,F,W,I,UP,B,SIM,C4,RUF` (E501 ignored)
- mypy: `strict = true`, `packages = ["dikw_core"]`, `mypy_path = "src"`
- pytest: `asyncio_mode = "auto"`, `testpaths = ["tests"]`
- hypothesis: property-based invariants live in `tests/test_properties.py` and
  stateful storage differential tests in `tests/test_storage_differential.py`
