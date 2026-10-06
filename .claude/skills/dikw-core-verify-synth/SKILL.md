---
name: dikw-core-verify-synth
description: The K-layer leg of step-3 in-loop verify — runs dikw-core's shipped synth self-check verbs (synth --verify [--judge], lint, eval --eval synth [--against]) against the real elon-musk corpus and folds them into one per-leg pass/fail table, applying the probabilistic grounding judgment the CLI deliberately leaves report-only. Use when a change touches the K layer (domains/knowledge/, api_synth.py, the LLM authoring prompts) or whenever iterating synth output quality.
---

<what-this-is>

This is the K-layer self-check: the "open the vault and click around" pass, made runnable.
It is the `domains/knowledge/**` leg of step 3 in `dikw-core-delivery-workflow`. It is also useful on its own when you tune synth quality.

It adds no new checks. It runs the verbs that already ship and folds them into one verdict.
It uses the **real** elon-musk corpus, as the acceptance gates require (`docs/eval-plan.md`, "Acceptance gates for K-layer and Retrieval changes"). Fake data does not count.

| leg | verb | gate kind |
|---|---|---|
| persist / lint / duplicate | `dikw client synth --verify` | **deterministic — hard gate** |
| grounding (entailment) | `dikw client synth --verify --judge` | **probabilistic — you interpret it; NOT a CLI gate** |
| standalone lint | `dikw client lint` on the produced vault | deterministic |
| synth-quality metrics | `dikw client eval --dataset mvp --eval synth [--against]` | numeric / regression |

**The Karpathy split.** `synth --verify --judge` is **report-only** by design.
The engine shows an entailment ratio but never folds it into `passed`, because a noisy LLM judge must not turn the main verdict red.
This skill makes the probabilistic call: it reads the ratio, the CI, and which pages scored `no` or `partial`, and decides whether to investigate.

**Run autonomously.** Fix red legs yourself. Stop only on a real block signal: a deterministic leg that stays red after a fix, or a required Protocol or on-disk layout change.

</what-this-is>

<checklist>

Track each leg. Run the floor first. A red **deterministic** leg: fix it and run it again. You interpret the grounding leg; it does not gate.

## 0. Scope — does this skill apply?

Confirm that `git diff --name-only main...HEAD` touches the K layer:
`src/dikw_core/domains/knowledge/**`, `src/dikw_core/api_synth.py`, or an LLM authoring prompt (`src/dikw_core/prompts/{synthesize,lint_fix_orphan_merge,lint_fix_broken_wikilink_grounded}.md`).

- If it does not, this skill does nothing. Return to the caller.
- K-layer changes are test-first. The failing test must already exist from step 2.

## 1. Fast K-layer test subset (the floor)

```
uv run python tools/check.py           # ruff + mypy + fast pytest, CI order
uv run pytest -k "lint or synth or atomicity or wisdom or grounding or verify"
```

If a test is red, fix it and run it again before you spend an LLM call below.

## 2. Live self-check on the real corpus (synth --verify --judge)

- Use the **elon-musk** corpus (the required K-layer baseline) through the `openai_codex` provider. Its LLM cost is zero: it uses ChatGPT-subscription OAuth (`dikw auth login openai-codex`, or `dikw auth import openai-codex` from an existing Codex login).
- Do **not** use MiniMax here. Its moderation blocks the elon biography.

```
# A scratch base whose dikw.yml points at openai_codex (gpt-5.x). Seed the
# elon-musk.md source (dikw-data/datasets/markdown-books/elon-musk.md), ingest,
# then synth THIS run's pages with the full self-check + grounding leg:
uv run dikw client synth --all --verify --judge --plain
```

Read the `SynthVerifyReport`. The command exits non-zero only when a **deterministic** leg failed.

- **persist** (`persist_ok`) — must PASS. A deactivated page is never clean output.
- **lint** (`lint_ok`) — must PASS. No `broken_wikilink`, `duplicate_title`, `non_atomic_page`, `uncategorized`, `missing_provenance`, or `title_slug_quality` on this run's pages.
- **duplicate** (`duplicate_ok`) — must PASS. If no embedder is wired it skips loudly; a skip is not a pass. Set the var that the base's `provider.embedding_api_key_env` names.
- **grounding** (`grounding_entailment_ratio` + `grounding_ci`) — **interpret it; do not gate on it**:
  - `grounding_checked: false` means the leg skipped (no embedder or LLM, or an error). Report it. It is NOT a green result.
  - A ratio in line with this corpus's history is fine.
  - A ratio that **drops materially** against the BASELINES.md history, or a CI with an alarming lower bound, means: **inspect the low-scoring pages** for hallucination.
  - Before you conclude, run again with a larger `synth.verify_judge_sample` to tighten the CI. The default n=25 carries ±0.2.
  - Do not fail the build on judge jitter. Do investigate a real drop that repeats.

## 3. Standalone lint on the produced vault

```
uv run dikw client lint --format table
```

No new `broken_wikilink`, `orphan_page`, `duplicate_title`, `uncategorized`, or `title_slug_quality` over the baseline.
This is the whole-base view. Step 2's lint leg covers only this run's pages.

## 4. Synth-quality metrics (the seven K-layer metrics + regression)

```
uv run dikw client eval --dataset mvp --eval synth --pretty
```

This captures the seven K-layer metrics:

- The **gated five:** `fact_grounding_ratio`, `atomicity_score`, `wikilink_resolved_ratio`, `language_fidelity`, `duplicate_ratio_max`.
- `expected_coverage`, when the dataset declares expectations.
- The **informational** `page_density`.

If a committed baseline exists, gate the run against it:

```
uv run dikw client eval --dataset mvp --eval synth \
  --against evals/baselines/mvp-synth.json
```

- `--against` exits 1 on a direction-aware regression past tolerance. A `_max` metric regresses when it *rises*.
- After a deliberate, justified change, refresh the baseline: `--write-baseline evals/baselines/mvp-synth.json`.

## 5. Print the per-leg verdict

Print a compact table: leg · status · detail.

- Deterministic legs: PASS or FAIL.
- Grounding: the interpreted ratio, or SKIPPED (said loudly).
- One deterministic FAIL means the K-layer change is not ready.

## 6. BASELINES.md entry (when the change goes into a PR)

- A K-layer PR needs an `evals/BASELINES.md` entry that cites the elon-musk outcome. Without it, `eval-gate` blocks (`docs/eval-plan.md`; see the entries for #179/#180/#182).
- A read-only or additive change (a new lint detector, a report-only leg) can use a non-destructiveness claim instead.
- A change to the authoring path needs real numbers.

</checklist>

<notes>

- **Zero-cost LLM** for steps 2 and 4 comes from `openai_codex`. Keep the scratch base's `dikw.yml` pointed at it. Embeddings still need a real key: set the var that `provider.embedding_api_key_env` names (for example `GITEE_API_KEY` for Gitee AI). Each leg reads exactly its configured var; there is no fallback.
- **The grounding leg is report-only on purpose** (`SynthVerifyReport` docstring). Gating on entailment waits for a calibrated threshold. Do not "fix" it by gating the ratio in the CLI. The judgment belongs in this skill.
- **A failure inside the grounding leg never fails the synth.** It becomes a loud skip. So `grounding_checked: false` on a base that has a judge wired means the leg errored. Read the WARN log before you trust the deterministic verdict alone.

</notes>
