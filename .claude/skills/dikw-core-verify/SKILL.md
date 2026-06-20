---
name: dikw-core-verify
description: The delivery-loop step-3 in-loop verify router — classify what the diff touches, run the shared deterministic floor (tools/check.py) plus only the change-specific legs that path needs (storage→Postgres contract, info→retrieval ablation, knowledge→dikw-core-verify-synth, providers→contract+check, cli/server/client→e2e+import-direction+CLI-grep, docs→ref-resolve), emit a per-leg pass/fail table, and self-heal red legs by fixing and re-running. Use at delivery-loop step 3 after implementing a non-trivial change, before the codex/fresh-review passes.
---

<what-this-is>

The **build-time feedback loop** of dikw-core's verification story
(`[[project_verification_capability_roadmap]]`): the "edit → self-verify → fix →
re-verify" inner loop made runnable as delivery-loop **step 3**. It does not
invent checks — it **routes by what the diff touches** to the deterministic
signals that already exist, runs only the relevant ones, and reports a per-leg
verdict. The spine `[[dikw-core-delivery-workflow]]` calls this at step 3
("if present").

**Karpathy's rule, applied to verification itself.** Routing is *deterministic
scoping* — `git diff` path → which legs run — so it's plain dispatch, not a
judgment call. Every leg here is a signal the agent can run **itself** (no human
in the loop): `tools/check.py`, the Postgres contract suite, a retrieval
ablation, the provider contract, server-e2e, a doc-ref grep. The one
probabilistic-adjacent branch (K-layer synth quality) is delegated to
`[[dikw-core-verify-synth]]`; the pre-merge clean-eyes judgment is a different
skill (`[[dikw-core-fresh-review]]`, step 5). This skill is the fast,
re-runnable, deterministic half.

**Self-heal, don't hand back.** A red leg here is something the agent can fix and
re-run — do that (`[[feedback_autonomy_default]]`). Never return a runnable check
to the user; only stop on a real block signal (a Protocol-widening or on-disk
layout finding, or a Postgres-only failure you can't reproduce locally).

</what-this-is>

<checklist>

Create one TodoWrite item per leg that fires. **Run the shared floor first**, then
the change-specific legs; any red → fix → re-run before moving on.

## 0. Classify the diff
```
git diff --name-only main...HEAD     # (or vs the working tree if pre-commit)
```
Bucket the paths: `storage/**` · `domains/info/**` + `RetrievalConfig` ·
`domains/knowledge/**` + `api_synth.py` + authoring prompts · `providers/**` ·
`cli.py` / `server/**` / `client/**` · `docs/**` + `*.md` · `config.py` / other.
A diff can hit several buckets — run every leg that matches.

## 1. Shared floor (every change, no exceptions)
```
uv run python tools/check.py   # ruff + mypy + fast pytest in CI order (no --cov — it flakes ASGI/CliRunner locally)
```
The cheap stages (ruff + mypy) also run as a git pre-commit hook once
`uv run pre-commit install` is done. Red → fix → re-run.

## 2. Change-specific legs (route by bucket)

| diff touches | also run |
|---|---|
| `storage/**` | local Postgres contract (`[[feedback_run_pg_locally]]`): spin `pgvector/pgvector:0.8.2-pg18` (the exact CI pin), then `uv run pytest tests/test_storage_contract.py tests/server/test_task_store_contract.py` against `DIKW_TEST_POSTGRES_DSN`. A PG-only failure you can't reproduce locally is a **block signal**. |
| `domains/info/**`, `RetrievalConfig` | `uv run pytest tests/test_search.py tests/test_retrieval_quality.py`; real-data ablation `dikw client eval --retrieval all` on ≥1 packaged dataset — assert nDCG@10 / hit@k non-regression vs the `evals/BASELINES.md` row (`[[feedback_real_data_validation]]`). |
| `domains/knowledge/**`, `api_synth.py`, the LLM authoring prompts | run **`[[dikw-core-verify-synth]]`** (the K-layer leg: `synth --verify [--judge]` + scoped lint + `eval --eval synth` on the real elon-musk corpus). |
| `providers/**` | provider contract harness + retry/error tests (`[[feedback_provider_backend_invariants]]`: SDK fake green ≠ backend green — confirm a sentinel fixture exists); `dikw client check` against a real/stub endpoint, assert exit 0 + sane dims. |
| `cli.py`, `server/**`, `client/**` | `uv run pytest tests/server tests/client`; `uv run pytest -v -m slow` (server-e2e); the **real-environment e2e harness** `uv run python tools/e2e_verify.py --mode local` (+ `--mode docker` when a Docker daemon is up, else SKIPPED-loud) — spins a live server and drives **every** `dikw client` verb (coverage is asserted against the live Typer tree, so a new verb without a step fails the run); tier-2 legs (`check`/embed/`synth`/vector-`retrieve`/`eval`) need `.env` keys and SKIP-loud without them; assert `client/*` imports no `dikw_core.{api,storage,providers,server,eval}` symbol (`tests/test_layering_contract.py`). |
| `docs/**`, `*.md` only | `uv run python tools/check_doc_refs.py` (asserts every `dikw <verb>` + `DIKW_*` env var in the docs resolves in source — also a pytest gate); eyeball any `/v1/...` route or frontmatter key shown (not yet auto-checked); `/code-review` stays mandatory (`[[feedback_code_review_not_optional]]`). |

## 3. CLI-string grep gate (any rename/removal)
For each CLI verb / route / env var / public symbol the diff **renamed or
removed**, ripgrep the whole repo (`CLAUDE.md` + `docs/**` + `CHANGELOG.md` +
`.claude/skills/**`) for the old spelling — a surviving reference is a finding
(`[[feedback_grep_cli_typos_across_docs]]`, `[[feedback_defensive_guard_grep_read_sites]]`).

## 4. Emit the per-leg pass/fail table
Print a compact table — **leg · status · detail** — with the floor and each fired
leg as PASS / FAIL (or SKIPPED-loudly when a prerequisite like an embedder or
Docker is genuinely absent — a skip is not a pass). A clean table is the green
light into step 4 (codex) and step 5 (`[[dikw-core-fresh-review]]`).

</checklist>

<notes>

- **Deterministic in, deterministic out.** Routing is path-based dispatch, not a
  judgment — that's why it lives in a skill the agent runs unattended. The
  judgment calls (synth quality, design fit) are delegated to
  `[[dikw-core-verify-synth]]` and `[[dikw-core-fresh-review]]`.
- **The shared floor is non-negotiable** — even a one-line change runs
  `tools/check.py`. It mirrors CI order so a green floor here predicts a green
  `lint-type-test` job.
- **A loud skip is not a pass.** No Docker → the Postgres leg is SKIPPED, surfaced
  as such (CI runs it against a real service anyway, `[[feedback_run_pg_locally]]`
  is the local mirror); no embedder → retrieval/synth vector legs degrade and say
  so. Never paper a skip over as green.
- **K-layer / Retrieval legs carry their own discipline:** TDD-first + an
  `evals/BASELINES.md` real-data entry (`[[feedback_tdd_discipline]]`,
  `[[feedback_real_data_validation]]`) — `dikw-core-verify-synth` enforces the
  BASELINES shape at step 7, not here.

</notes>
