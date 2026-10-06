---
name: dikw-core-fresh-review
description: The pre-merge design review for tier-L changes — spawn a clean subagent that did NOT write the code, hand it ONLY the diff + dikw-core's design-invariant rubric + the relevant docs/design.md and .claude/rules sections, and have it judge correctness, scope discipline, and invariant adherence (the checks ruff/mypy/pytest and /code-review's bug-hunt cannot make), plus sample-read synth-produced pages for K-layer changes. Emits a pass/blocking verdict with a triaged true-positive/false-positive table. Use at delivery-loop step 4 for tier-L diffs, after the codex rounds are quiet, before the PR merges.
---

<what-this-is>

A reviewer that did **not** write the code reads the diff against the project's invariants and design intent.
This skill makes that review runnable. `dikw-core-delivery-workflow` calls it at step 4 for tier-L changes.

It is **not** a bug hunt and **not** a test run:

| lens | tool | what it catches |
|---|---|---|
| deterministic floor | `tools/check.py` (step 3) | ruff / mypy / pytest |
| bug hunt | `/code-review` (step 4) | concrete defects, edge cases |
| **invariant / design / scope** | **this skill** (step 4, tier L) | fixes at the wrong depth, seam violations, scope drift, Karpathy-rule breaks |

Run `/code-review` **and** this skill. They are different lenses, not substitutes.
This skill scores the diff against `docs/verification-rubric.md` (a checkable form of the invariants) and the intent in `docs/design.md`:
is this the right change, at the right depth, through the named seams, with scoping deterministic and reasoning probabilistic?

**Clean context is the point.** Spawn the reviewer with the Agent tool and a **self-contained** prompt (diff + rubric + design excerpts).
Never give it the build conversation. A reviewer that knows the author's reasons approves too easily.

**Run autonomously.** Fix every confirmed blocking finding. Reject false positives and nitpicks with a one-line reason.
Stop only on a block signal: a design-level concern or `CHANGES_REQUESTED`, or a finding that needs a wider Protocol or an on-disk layout change.

</what-this-is>

<checklist>

## 0. Scope — when this runs

- Step 4 of the delivery loop, for tier-L diffs, after the codex rounds are quiet.
- If `git diff main...HEAD` is empty, diff the working tree. The review often runs before the commit.

## 1. Gather the review packet

The reviewer sees only this packet:

- The unified diff: `git diff main...HEAD` (or the PR, or the working tree).
- `docs/verification-rubric.md` — the yes / no / N-A invariant checklist.
- The `docs/design.md` sections for what the diff touches (K-layer authoring, retrieval fusion, persist pipeline, …).
- The `.claude/rules/*.md` files for the touched areas — they hold the full text of the core invariants.

## 2. Spawn the clean reviewer(s)

Use the Agent tool. Scale the panel to the change:

- **Default (1 reviewer):** score every rubric line yes / no / N-A, and read the diff for correctness and scope drift.
- **Large or cross-cutting diff:** add a 2nd reviewer. One owns invariant adherence. The other owns scope, simplicity, and design fit.
- **K-layer diff** (`domains/knowledge/**`, `api_synth.py`, the authoring prompts): add the synth-page sampler (step 3).

Give each reviewer these instructions:

- You did NOT write this code. Judge the artifact against the rubric and the design. Do not guess the author's intent.
- Report only problems that would block the merge. A rubric `no` is blocking.
- For each finding give: `file`, `line`, `rubric_line_or_concern`, why it is wrong, and how to show that it fails.
- If you suspect a rubric `no` but cannot prove it, report it and mark it `UNCONFIRMED`. Say where you looked.

## 3. K-layer only — read the produced vault

When the diff touches synth or the K layer, a fresh agent opens some synth-produced pages.
Produce them with the elon-musk subset through `openai_codex` (no LLM cost; see `dikw-core-verify-synth` step 2), or read pages from a recent run.
The agent judges each page on the bar that metrics cannot fully see:

- **Grounded** — each claim traces to a cited source.
- **Atomic** — one subject per page.
- **Well-titled** — specific, not generic.
- **Not a duplicate** of another page.
- **Coherent** — reads like a human note, not LLM filler.

## 4. Triage → the TP/FP table

A reviewer can invent a finding. Before you act, **check each finding against the source**: read the cited file and line. Then fill this table:

| finding | file:line | rubric line / concern | verdict | action |
|---|---|---|---|---|
| … | … | … | **TP** / **FP** | fix / reject (one-line reason) |

- A rubric `no` that the source confirms is **blocking**.
- An `UNCONFIRMED` finding that you cannot confirm in the source is FP. Reject it with a one-line reason.
- The verdict is **pass** (no confirmed blocking finding) or **blocking** (one or more).

## 5. Resolve and re-verify

- Fix every confirmed blocking finding and every actionable TP. If you changed code, run `uv run python tools/check.py` again.
- If a fix touches one CLI string, symbol, route, or env var, grep the whole repo for it.
- **STOP** (block signal) if a confirmed finding needs a wider Storage or Provider Protocol, or an on-disk knowledge/wisdom layout change (update `docs/design.md` first), or if a reviewer raised a design-level `CHANGES_REQUESTED`.

</checklist>

<notes>

- **Keep the context clean.** Never paste the implementation conversation into the reviewer prompt. If you start to explain "why" to the reviewer, you have already biased it.
- **This skill adds to the other lenses; it does not replace them.** `/code-review` runs beside it. Step 3 already gated the deterministic floor. This skill owns the invariant, design, and scope judgment that ruff, mypy, and pytest cannot make.
- **The review can be wrong.** Step 4's TP/FP triage handles that. Honest triage is better than both approving everything and chasing invented findings.
- **Karpathy framing.** Step 3 hard-gates the deterministic floor. This skill is the probabilistic pre-merge judgment (right change? right depth? right seam?). The same split applies to `synth --verify --judge`, whose grounding pass/fail is left to `dikw-core-verify-synth`.

</notes>
