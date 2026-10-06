---
name: dikw-core-verify
description: The delivery-loop step-3 in-loop verify router — classify what the diff touches, run the shared deterministic floor (tools/check.py) plus only the change-specific legs that path needs (storage→Postgres contract, info→retrieval ablation, knowledge→dikw-core-verify-synth, providers→contract+check, cli/server/client→e2e+import-direction+CLI-grep, docs→ref-resolve), emit a per-leg pass/fail table, and self-heal red legs by fixing and re-running. Use at delivery-loop step 3 after implementing a non-trivial change, before the review step.
---

<what-this-is>

This is step 3 of `dikw-core-delivery-workflow`: the edit → verify → fix → verify loop.
It adds no new checks. It routes the diff to the deterministic checks that exist, runs only those, and reports one verdict per leg.

- **Routing is deterministic.** The `git diff` paths decide which legs run. This is dispatch, not judgment (Karpathy's rule).
- **Every leg runs without a human:** `tools/check.py`, the Postgres contract, a retrieval ablation, the provider contract, server e2e, a doc-ref check.
- **Judgment lives elsewhere.** K-layer synth quality goes to `dikw-core-verify-synth`. The design review is `dikw-core-fresh-review` (step 4, tier L).
- **Fix red legs yourself.** Fix the cause and run the leg again. Do not hand a runnable check back to the user. Stop only on a real block signal: a Protocol or on-disk layout change, or a Postgres-only failure that you cannot reproduce locally.

</what-this-is>

<checklist>

Track each leg that fires. Run the shared floor first, then the change-specific legs. If a leg is red, fix it and run it again before you continue.

## 0. Classify the diff

```
git diff --name-only main...HEAD     # (or vs the working tree if pre-commit)
```

Put each path in a bucket:

- `storage/**`
- `domains/info/**` + `RetrievalConfig`
- `domains/knowledge/**` + `api_synth.py` + the authoring prompts
- `providers/**`
- `cli.py` / `server/**` / `client/**`
- `docs/**` + `*.md` + `.claude/**`
- `config.py` / other

A diff can hit several buckets. Run every leg that matches.

## 1. Shared floor (every change, no exceptions)

```
uv run python tools/check.py   # ruff + mypy + fast pytest in CI order (no --cov — it flakes ASGI/CliRunner locally)
```

The cheap stages (ruff + mypy) also run as a git pre-commit hook after `uv run pre-commit install`. If the floor is red, fix it and run it again.

## 2. Change-specific legs (route by bucket)

| diff touches | also run |
|---|---|
| `storage/**` | Local Postgres contract. Start `pgvector/pgvector:0.8.2-pg18` (the exact CI pin), then run `uv run pytest tests/test_storage_contract.py tests/server/test_task_store_contract.py` with `DIKW_TEST_POSTGRES_DSN` set. Run it locally; do not wait for CI to find it. A Postgres-only failure that you cannot reproduce locally is a **block signal**. |
| `domains/info/**`, `RetrievalConfig` | `uv run pytest tests/test_search.py tests/test_retrieval_quality.py`. Then a real-data ablation: `dikw client eval --retrieval all` on at least one packaged dataset. Assert no nDCG@10 or hit@k regression against the `evals/BASELINES.md` row. Fake data does not count. |
| `domains/knowledge/**`, `api_synth.py`, the LLM authoring prompts | Run **`dikw-core-verify-synth`** (`synth --verify [--judge]` + scoped lint + `eval --eval synth` on the real elon-musk corpus). |
| `providers/**` | The provider contract harness and the retry/error tests. A green SDK fake does not prove the real backend works: confirm that a sentinel fixture exists for the backend behavior you changed. Then `dikw client check` against a real or stub endpoint; assert exit 0 and sane dims. |
| `cli.py`, `server/**`, `client/**` | `uv run pytest tests/server tests/client`; `uv run pytest -v -m slow` (server e2e). Then the real-environment harness `uv run python tools/e2e_verify.py --mode local`, plus `--mode docker` when a Docker daemon runs (else SKIPPED, said loudly). It starts a live server and drives **every** `dikw client` verb; a new verb without a step fails the run. Tier-2 legs (`check`/embed/`synth`/vector-`retrieve`/`eval`) need `.env` keys and SKIP loudly without them. Assert that `client/*` imports no `dikw_core.{api,storage,providers,server,eval}` symbol (`tests/test_layering_contract.py`). |
| `docs/**`, `*.md`, `.claude/**` only | `uv run python tools/check_doc_refs.py`. It asserts that every `dikw <verb>` and `DIKW_*` env var in `CLAUDE.md`, `docs/**`, `README.md`, and `.claude/{skills,rules}/**` resolves in source. It is also a pytest gate. Read any `/v1/...` route or front-matter key in the diff by eye; nothing checks them yet. |

## 3. CLI-string grep gate (any rename or removal)

For each CLI verb, route, env var, or public symbol that the diff **renamed or removed**:

- Grep the whole repo for the old spelling: `CLAUDE.md`, `docs/**`, `CHANGELOG.md`, `.claude/**`.
- Also grep the read sites of any guard you added or removed.
- A reference that survives is a finding.

## 4. Print the per-leg table

Print a compact table: **leg · status · detail**. Include the floor and each leg that fired.

- Status is PASS, FAIL, or SKIPPED.
- Use SKIPPED only when a prerequisite (an embedder, Docker) is really absent. Say so loudly. A skip is not a pass.
- A clean table is the entry to step 4 (review).

</checklist>

<notes>

- **The shared floor is mandatory.** A one-line change also runs `tools/check.py`. It mirrors CI order, so a green floor predicts a green `lint-type-test` job.
- **A loud skip is not a pass.** No Docker: the Postgres leg is SKIPPED, and you say so. CI still runs it on a real service. No embedder: the retrieval and synth vector legs degrade, and you say so. Never report a skip as green.
- **K-layer and Retrieval legs** also need test-first work and an `evals/BASELINES.md` entry with real-data numbers. `dikw-core-verify-synth` checks the BASELINES shape at step 7, not here.

</notes>
