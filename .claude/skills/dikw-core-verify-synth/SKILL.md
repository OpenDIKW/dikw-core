---
name: dikw-core-verify-synth
description: The K-layer leg of step-3 in-loop verify — runs dikw-core's shipped synth self-check verbs (synth --verify [--judge], lint, eval --eval synth [--against]) against the real elon-musk corpus and folds them into one per-leg pass/fail table, applying the probabilistic grounding judgment the CLI deliberately leaves report-only. Use when a change touches the K layer (domains/knowledge/, api_synth.py, the LLM authoring prompts) or whenever iterating synth output quality.
---

<what-this-is>

The **K-layer self-check** for dikw-core — the "open the vault and click around"
pass made runnable. It is the `domains/knowledge/**` branch of
`dikw-core-delivery-workflow` step 3 (in-loop verify), and it is also useful
standalone while iterating synth quality (`[[project_synth_quality_optimization]]`).

It does not invent new checks — it **orchestrates the verbs that already shipped**
(roadmap Phase 1) into one verdict, against the **real** elon-musk corpus the
acceptance gates mandate (`docs/eval-plan.md` "Acceptance gates for K-layer and
Retrieval changes"; `[[feedback_real_data_validation]]`):

| leg | verb | gate kind |
|---|---|---|
| persist / lint / duplicate | `dikw client synth --verify` | **deterministic — hard gate** |
| grounding (entailment) | `dikw client synth --verify --judge` | **probabilistic — interpreted here, NOT a CLI gate** |
| standalone lint | `dikw client lint` on the produced vault | deterministic |
| synth-quality metrics | `dikw client eval --dataset mvp --eval synth [--against]` | numeric / regression |

**The Karpathy split this skill exists to honour.** `synth --verify --judge` is
**report-only** by design — the engine surfaces an entailment ratio but never
folds it into `passed`, because a noisy LLM judge must not false-red the
flagship verdict. *This skill is the upstream layer that makes the probabilistic
call*: it reads the ratio + CI (and which pages scored `no`/`partial`) and
decides whether to investigate. Deterministic scoping (persist/lint/duplicate)
hard-gates in the engine; the probabilistic read lives here.

**Run autonomously** (`[[feedback_autonomy_default]]`). Self-heal red legs; only
stop on a real block signal (a deterministic leg that stays red after a fix, or a
required Protocol / on-disk-layout change).

</what-this-is>

<checklist>

Create one TodoWrite item per leg. Run the floor first; a red **deterministic**
leg → fix → re-run. The grounding leg is interpreted, not pass/fail-gated.

## 0. Scope — does this skill apply?
Confirm `git diff --name-only main...HEAD` touches the K layer:
`src/dikw_core/domains/knowledge/**`, `src/dikw_core/api_synth.py`, or an LLM
authoring prompt (`src/dikw_core/prompts/{synthesize,lint_fix_orphan_merge,lint_fix_broken_wikilink_grounded}.md`).
If it does not, this skill is a no-op — return to the caller. K-layer changes
also **mandate test-first** (`[[feedback_tdd_discipline]]`): the failing test
should already exist from step 2 before you get here.

## 1. Fast K-layer test subset (the floor)
```
uv run python tools/check.py           # ruff + mypy + fast pytest, CI order
uv run pytest -k "lint or synth or atomicity or wisdom or grounding or verify"
```
Any red → fix → re-run before spending an LLM call below.

## 2. Live self-check on the real corpus (synth --verify --judge)
Use the **elon-musk** corpus (the mandated K-layer baseline) via the
`openai_codex` provider so the **LLM cost is zero** (ChatGPT-subscription OAuth;
see `[[reference_openai_codex_setup]]`). Do **not** use MiniMax here — its
moderation hard-blocks the elon biography (`[[reference_minimax_elon_moderation]]`).

```
# A scratch base whose dikw.yml points at openai_codex (gpt-5.x). Seed the
# elon-musk.md source (dikw-data/datasets/markdown-books/elon-musk.md), ingest,
# then synth THIS run's pages with the full self-check + grounding leg:
uv run dikw client synth --all --verify --judge --plain
```
Read the `SynthVerifyReport` (the command exits non-zero iff a **deterministic**
leg failed):
- **persist** (`persist_ok`) — must be PASS; a deactivated page is never clean output.
- **lint** (`lint_ok`) — must be PASS; no `broken_wikilink` / `duplicate_title` /
  `non_atomic_page` / `uncategorized` / `missing_provenance` / `title_slug_quality`
  on this run's pages.
- **duplicate** (`duplicate_ok`) — must be PASS (or loud-skip if no embedder is
  wired; a skip is not a pass — set the base's configured
  `provider.embedding_api_key_env` var).
- **grounding** (`grounding_entailment_ratio` + `grounding_ci`) — **interpret, do
  not gate**:
  - `grounding_checked: false` → the leg loud-skipped (no embedder/LLM, or it
    errored). Surface it; it is NOT a green grounding result.
  - A ratio in line with this corpus's history → fine.
  - A ratio that **drops materially** versus the BASELINES.md history, or a CI
    whose lower bound is alarming, is the signal to **inspect the low-scoring
    pages** for hallucination — re-run with a larger `synth.verify_judge_sample`
    to tighten the CI before concluding, since the default n=25 carries ±0.2.
    Don't fail the build on judge jitter; do investigate a real, repeatable drop.

## 3. Standalone lint on the produced vault
```
uv run dikw client lint --format table
```
No new `broken_wikilink` / `orphan_page` / `duplicate_title` / `uncategorized` /
`title_slug_quality` over the baseline. (This is the whole-base view; step 2's
lint leg is scoped to just this run's pages.)

## 4. Synth-quality metrics (the seven K-layer metrics + regression)
```
uv run dikw client eval --dataset mvp --eval synth --pretty
```
Captures the seven K-layer metrics — the **gated five** `fact_grounding_ratio`,
`atomicity_score`, `wikilink_resolved_ratio`, `language_fidelity`,
`duplicate_ratio_max` (plus `expected_coverage` when the dataset declares
expectations), and the **informational** `page_density`. If a committed baseline exists, gate the
run against it:
```
uv run dikw client eval --dataset mvp --eval synth \
  --against evals/baselines/mvp-synth.json
```
`--against` exits 1 on a direction-aware regression past tolerance (a `_max`
metric regresses when it *rises*). To refresh after a deliberate, justified
change: `--write-baseline evals/baselines/mvp-synth.json`.

## 5. Emit the per-leg verdict
Print a compact table — leg · status · detail — with the deterministic legs as
PASS/FAIL and the grounding leg as an interpreted ratio (or loud-skip). A single
deterministic FAIL means the K-layer change is not ready.

## 6. BASELINES.md entry (when wired into a PR)
A K-layer PR needs an `evals/BASELINES.md` entry citing the elon-musk outcome
(or a non-destructiveness claim for a read-only/additive change), or `eval-gate`
blocks (`docs/eval-plan.md`; mirror the entries for #179/#180/#182). Read-only
additions (a new lint detector, a report-only leg) use the non-destructiveness
shape; a change to the authoring path needs real numbers.

</checklist>

<notes>

- **Zero-cost LLM** for step 2/4 comes from `openai_codex` — keep the scratch
  base's `dikw.yml` pointed at it. Embeddings still need a real key — set the
  var the base's `provider.embedding_api_key_env` names (e.g. `GITEE_API_KEY`
  for Gitee AI); each leg reads exactly its configured var, no fallback.
- The grounding leg's report-only contract is deliberate
  (`SynthVerifyReport` docstring; roadmap defers *gating* entailment to a
  calibrated Phase 2.2 threshold). Do not "fix" it by gating the ratio in the CLI
  — the judgment belongs here.
- A failure *inside* the grounding leg never fails the synth (it degrades to a
  loud skip) — so a `grounding_checked: false` on a wired base means the leg
  errored; check the WARN log before trusting the deterministic verdict alone.

</notes>
