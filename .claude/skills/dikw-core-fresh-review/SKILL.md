---
name: dikw-core-fresh-review
description: The layer-2 pre-merge review — spawn a clean subagent that did NOT write the code, hand it ONLY the diff + dikw-core's design-invariant rubric + the relevant docs/design.md, and have it judge correctness, scope discipline, and invariant adherence (the taste-level checks ruff/mypy/pytest and /code-review's bug-hunt are blind to), plus sample-read synth-produced pages for K-layer changes. Emits a pass/blocking verdict with a triaged true-positive/false-positive table. Use at delivery-loop step 5, after the codex loop quiets, before opening or merging a PR.
---

<what-this-is>

The **second feedback loop** from the article this verification arc is built on
(`[[project_verification_capability_roadmap]]`): the first loop is build-time
self-verification (deterministic signals the agent already runs — step 3); the
second is a reviewer who **did not write the code**, reading the diff against the
project's *invariants* and *design intent* before merge. This skill makes that
second loop runnable as delivery-loop **step 5** (it's the `dikw-core-fresh-review`
the spine `[[dikw-core-delivery-workflow]]` calls "if present").

It is **not** a bug-hunt and **not** a re-run of the test suite:

| lens | tool | what it catches |
|---|---|---|
| deterministic floor | `tools/check.py` (step 3) | ruff / mypy / pytest |
| recall bug-hunt | `/code-review` (step 5) | concrete defects, edge cases |
| **invariant / design / scope** | **this skill** (step 5) | wrong-depth fixes, seam violations, scope drift, Karpathy-rule breaks |

Run `/code-review` **and** this skill at step 5 — they are complementary lenses,
not substitutes. This one scores the diff against `docs/verification-rubric.md`
(the checkable restatement of CLAUDE.md's invariants) and `docs/design.md` intent:
*is this the right change, implemented at the right depth, honoring the named
seams and "scoping deterministic / reasoning probabilistic"?*

**The defining property is clean context.** Spawn the reviewer with the Agent tool
and a **self-contained** prompt (the diff + rubric + design excerpts) — never the
build conversation. A reviewer that inherits the author's rationale rubber-stamps;
one that sees only the artifact + the contract is what "did not write the code"
means.

**Run autonomously** (`[[feedback_autonomy_default]]`): resolve every confirmed
blocking finding, reject false positives / nitpicks with a one-line reason
(`[[feedback_goal_explicit_approves_steps]]`). Only stop on a real block signal
(a design-level concern / `CHANGES_REQUESTED`, or a finding that needs widening a
Protocol or changing on-disk layout).

</what-this-is>

<checklist>

Create one TodoWrite item per step. Even a doc-only PR earns this pass
(`[[feedback_code_review_not_optional]]`).

## 0. Scope — when this runs
Step 5, after the codex loop (step 4) has quieted. Any non-trivial PR, doc-only
included. If `git diff main...HEAD` is empty, also diff the working tree — review
often runs pre-commit.

## 1. Gather the review packet
Assemble the **self-contained** inputs the fresh reviewer will get (it sees
nothing else):
- the unified diff: `git diff main...HEAD` (or the PR / working tree);
- `docs/verification-rubric.md` (the yes/no/N-A invariant checklist);
- the **relevant** `docs/design.md` sections for what the diff touches (K-layer
  authoring, retrieval fusion, persist pipeline, …) — design intent is the bar;
- CLAUDE.md's *Core invariants* + *Layering invariants* (paste the headers; the
  rubric already distils them).

## 2. Spawn the clean fresh-reviewer(s)
Use the Agent tool. Scale the panel to the change:
- **default (1 reviewer):** score every rubric line yes/no/N-A and read the diff
  for correctness + scope drift.
- **large / cross-cutting diff:** add a 2nd lens so one agent owns
  invariant-adherence and the other owns scope/simplicity + design-fit.
- **K-layer diff** (`domains/knowledge/**`, `api_synth.py`, the authoring
  prompts): add the synth-page sampler (step 3 below).

Tell each agent: it did NOT write this code; judge the artifact against the
rubric + design, not against any assumed intent; a rubric **`no` is a blocking
finding**; return findings as `{file, line, severity, rubric_line_or_concern,
why}`; be precise (the findings will be triaged against source, so a vague claim
wastes a round). Recall over precision on **invariants** — surface a suspected
`no` even if unsure; triage (step 4) filters it.

## 3. K-layer only — read the produced vault
When the diff touches synth/K-layer, a fresh agent opens a handful of the
synth-produced pages (run the elon-musk subset via `openai_codex` for zero LLM
cost — see `[[dikw-core-verify-synth]]` step 2 for the exact synth invocation —
or read pages from a recent run) and judges each on the Zettelkasten bar
the metrics can't fully see: **grounded** (claims trace to a cited source),
**atomic** (one subject), **well-titled** (specific, not generic), **non-
duplicate**, **coherent** (reads like a human note, not LLM filler). This is the
"open the vault and click around" pass made a review leg
(`[[feedback_real_data_validation]]`).

## 4. Triage → the TP/FP table
A fresh agent can hallucinate a finding. Before acting, **verify each against
source** (read the cited file/line). Build a table:

| finding | file:line | rubric line / concern | verdict | action |
|---|---|---|---|---|
| … | … | … | **TP** / **FP** | fix / reject (one-line reason) |

A rubric `no` confirmed against source is **blocking**. An FP gets a one-line
rejection (don't thrash on a hallucinated finding). The verdict is **pass** (no
confirmed blocking) or **blocking** (≥1).

## 5. Resolve and re-verify
- Fix every confirmed blocking + actionable TP. If you changed code, re-run the
  step-3 floor (`tools/check.py`) before declaring done.
- When a fix touches one CLI string / symbol / route / env var, grep the whole
  repo for it (`[[feedback_grep_cli_typos_across_docs]]`).
- **STOP** (block signal) if a confirmed finding needs widening a Storage /
  Provider Protocol or changing on-disk knowledge/wisdom layout (update
  `docs/design.md` first), or a reviewer raised a design-level `CHANGES_REQUESTED`.

</checklist>

<notes>

- **Clean context is non-negotiable.** Never paste the implementation conversation
  into the reviewer prompt — the self-contained packet (diff + rubric + design) is
  exactly what makes the review a *fresh* set of eyes. If you find yourself
  explaining "why" to the reviewer, you've already biased it.
- **Complements, never replaces.** `/code-review` (recall bug-hunt) runs alongside
  this at step 5, and step 3's deterministic self-check (`dikw-core-verify` once it
  lands; routed inline by the spine until then) already gated the floor. This leg
  owns the invariant / design / scope judgment the other lenses structurally cannot
  make (ruff/mypy/pytest are blind to every rubric line — that's why a *reading*
  reviewer exists).
- **The review is itself fallible** — that's what step 4's TP/FP triage is for. A
  confirmed `no` blocks; an unconfirmable claim is an FP, rejected with a reason.
  Honest triage beats both rubber-stamping and thrashing on phantom findings.
- **Karpathy framing.** Step 3 hard-gates the deterministic floor; this skill is
  the *probabilistic* pre-merge call (right change? right depth? right seam?) — the
  agent-layer judgment the engine deliberately leaves to a reading reviewer, mirror
  of how `synth --verify --judge` leaves the grounding pass/fail to the
  `[[dikw-core-verify-synth]]` skill.

</notes>
