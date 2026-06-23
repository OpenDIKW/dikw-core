# Contributing to dikw-core

Thanks for your interest in contributing! `dikw-core` is in **alpha** —
APIs, on-disk formats, the database schema, and the CLI are still moving,
so the most useful contributions right now are bug reports, focused fixes,
and small well-scoped improvements. Please open an issue to discuss larger
changes before investing in a big PR.

## Development setup

The package manager is [`uv`](https://docs.astral.sh/uv/) (not pip/poetry),
and the project targets **Python 3.12+**.

```bash
git clone https://github.com/OpenDIKW/dikw-core
cd dikw-core
uv sync --all-extras          # installs every extra + the dev group
uv run pre-commit install     # (once) wire the ruff + mypy git pre-commit hook
```

## The local CI gate

Run this before every commit — it mirrors what CI runs, in CI order:

```bash
uv run python tools/check.py  # ruff + mypy + fast pytest
```

You can also run the legs individually:

```bash
uv run ruff check .           # lint (line-length 100; rules E,F,W,I,UP,B,SIM,C4,RUF)
uv run mypy src               # strict type-check
uv run pytest -v              # tests (asyncio_mode=auto)
```

Tooling config lives in `pyproject.toml`. The code is **fully typed** and mypy
runs in `strict` mode — fix the root cause rather than widening types to silence
an error.

### Storage and server changes

- **Storage adapter changes** (`src/dikw_core/storage/**`) are validated by the
  shared contract suite against both backends. Run it locally against a real
  Postgres when you touch an adapter:
  ```bash
  uv run pytest tests/test_storage_contract.py
  ```
  CI runs the same suite against a `pgvector/pgvector` service.
- **Server / client changes** have a real-environment end-to-end harness:
  ```bash
  uv run python tools/e2e_verify.py --mode local
  ```
- **Knowledge-layer (`domains/knowledge/`) and Retrieval (`domains/info/`)
  changes** require an entry in [`evals/BASELINES.md`](./evals/BASELINES.md)
  showing a real-data outcome (or the `no-baseline-needed` label when the change
  is mechanical). See [`docs/eval-plan.md`](./docs/eval-plan.md).

### Documentation changes

Docs are written in **English**. `tools/check_doc_refs.py` (also a pytest gate)
asserts that every `dikw <verb>` and `DIKW_*` env var mentioned in the docs
resolves against the actual CLI tree and source — so keep CLI spellings, routes,
frontmatter keys, and env-var names accurate.

## Pull request workflow

1. **Branch** off `main` — never commit to `main` directly.
2. **Write tests first** where it fits (the project defaults to TDD; K-layer and
   retrieval changes mandate a failing test before the fix).
3. **Keep changes surgical** — touch only what the change requires; match the
   surrounding style; don't reformat or refactor adjacent code.
4. **Green the local gate** (`uv run python tools/check.py`) before pushing.
5. **Open a PR** and fill in the template. Keep the description focused on the
   *why* and the user-visible effect.
6. **Update the docs** that your change affects (README, `docs/**`, CHANGELOG,
   ADRs).
7. **Get CI green** and address review comments. Don't force-push shared branches.

### Commit messages

Use clear, conventional-style subjects (`feat:`, `fix:`, `docs:`, `ci:`,
`refactor:`, …). When a change is co-authored with an AI assistant, end the
commit message with the appropriate `Co-Authored-By:` trailer.

## Architecture invariants

Before changing core behavior, read [`docs/design.md`](./docs/design.md) (intent),
[`docs/architecture.md`](./docs/architecture.md) (module map + seams), and the
`CLAUDE.md` invariants. A few that come up often:

- Engine code talks only to the `Storage` Protocol — no SQL or adapter internals
  from engine modules.
- New source formats register a `SourceBackend`; search fusion (RRF) lives in
  `info/search.py`, not in an adapter.
- Don't change the on-disk `knowledge/` / `wisdom/` layout without updating
  `docs/design.md` first — the open Markdown tree is the product.

## Code of Conduct

This project follows the [Contributor Covenant](./CODE_OF_CONDUCT.md). By
participating, you agree to uphold it.

## Security

Please report vulnerabilities privately — see [`SECURITY.md`](./SECURITY.md).
Do not open a public issue for a security problem.
