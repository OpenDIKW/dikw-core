---
name: dikw-core-delivery-workflow
description: The unified dikw-core delivery loop — drives a non-trivial change end-to-end through clarify → plan/TDD → in-loop verify → codex review → fresh review → doc-sync → PR → watch-CI-green → squash, with verification woven into each step and STOP points on the 6 block signals. Use when implementing any non-trivial change in dikw-core (feature, bugfix, refactor) so the sequence is run, not remembered.
---

<what-this-is>

This skill is the **spine** of dikw-core's verification story. It encodes CLAUDE.md's
8-step **Delivery loop** (steps 1–8) as one runnable checklist — prefixed by a **step-0
resume preamble** that reads the delivery artifact (mirroring `dikw-core-verify`'s step-0
"classify"), with the right verification leg embedded at each step. It does not invent a
parallel process — it *is* the CLAUDE.md Delivery loop, made executable.

Two companion skills are called from inside this one as they land (walking-skeleton →
grows legs over phases):

- `dikw-core-verify` — step 3 in-loop change-type router (classify the diff → run only the legs that path needs); step 3's table below is the inline fallback if it's unavailable.
- `dikw-core-verify-synth` — the K-layer leg of step 3 (synth output self-check), dispatched by `dikw-core-verify`.
- `dikw-core-fresh-review` — step 5 fresh-agent pre-merge review (clean subagent vs diff + rubric + design); run it alongside `/code-review`.

Every run keeps one **delivery artifact** — `.claude/delivery/<branch>.md` — that is both
**resumable STATE** (step 0 reads it, so an interrupted task continues instead of
cold-starting) and an **auditable receipt** (each step records its *machine output*, which
step 7 renders into the PR body). It turns the soft verify/review steps from "ran on the
honor system" into an **auditable trail** a reviewer can check at PR time. Honest limit: the
receipt *records* those steps, it doesn't *re-run* them the way CI re-runs the deterministic
floor — so it raises the cost of a silently-skipped step and surfaces it, but is not itself a
hard gate. Template + lifecycle in `<delivery-artifact>` below.

**Run autonomously.** Per `feedback_autonomy_default` + `feedback_pr_workflow`, the full
loop is standing approval to commit/push/PR/squash. Only stop on the 6 block signals below.

</what-this-is>

<checklist>

Create one TodoWrite item per step. Do not skip a step because it "feels unnecessary"
(`feedback_code_review_not_optional`: even doc-only PRs surface real findings).

## 0. Resume — read the delivery artifact first
Before anything, read `.claude/delivery/<branch>.md` (the current branch, `/`→`-`). If it
exists, it records which steps are done + the single **Next action** — resume there, don't
cold-start. If absent (new task), create it from the `<delivery-artifact>` template
(back-fill **Goal** once step 1 has clarified it). After **every** step, keep it current —
both the **header fields** (Goal, Next action, PR number) and that step's **status** + its
**machine evidence**, pasted into the Evidence section per the template. On *why* the
evidence must be real machine output rather than a free-text "I ran it", see the rule in
`<delivery-artifact>`.

## 1. Clarify the request
- Restate the goal, list assumptions, surface alternatives.
- Multi-decision work → escalate to the `grill-with-docs` skill (or `superpowers:brainstorming`) until a written plan exists.
- **STOP** if the request is genuinely ambiguous on a decision you cannot resolve from code/docs.

## 2. Plan in Chinese, default TDD
- Plan prose in Chinese; code/commits/identifiers in English (`feedback_language_chinese`).
- Each step lands as **failing test → implementation → passing test**.
- **K-layer (`domains/knowledge/`) and Retrieval (`domains/info/`) changes MANDATE test-first** (`feedback_tdd_discipline`): write the failing test before touching implementation.

## 3. In-loop verify (the core feedback loop)
Run `dikw-core-verify` if present. Until it lands, route inline by what the `git diff vs main` touches.
**Always run the shared floor first**, then the change-specific legs. Any red → fix → re-run (self-heal, do not hand back to the user for a check you can run).

**Shared floor (every change):**
```
uv run python tools/check.py   # ruff + mypy + fast pytest in CI order (no --cov locally — it flakes ASGI/CliRunner)
```
Or run the three directly:
`uv run ruff check .` · `uv run mypy src` · `uv run pytest -m "not slow and not perf"`.
The cheap stages (ruff + mypy) also run as a git pre-commit hook once `uv run pre-commit install` is done.

**Change-specific legs (route by diff path):**

| diff touches | also run |
|---|---|
| `storage/**` | local Postgres contract (see `feedback_run_pg_locally`): spin `pgvector/pgvector:pg18`, then `uv run pytest tests/test_storage_contract.py tests/server/test_task_store_contract.py` against `DIKW_TEST_POSTGRES_DSN` |
| `domains/info/**`, `RetrievalConfig` | `uv run pytest tests/test_search.py tests/test_retrieval_quality.py`; real-data ablation `dikw client eval --retrieval all` on ≥1 packaged dataset, assert nDCG@10/hit@k non-regression vs the BASELINES.md row |
| `domains/knowledge/**` (synth/lint) | run `dikw-core-verify-synth` if present; else: `uv run pytest -k "lint or synth or atomicity or wisdom"`, then synth the elon-musk subset via `openai_codex` (zero LLM cost) + `dikw client lint` on the produced vault (no new broken_wikilink/orphan/duplicate/uncategorized/title_slug_quality over baseline) + `dikw client eval --dataset mvp --eval synth` |
| `providers/**` | provider contract harness + retry/error tests (`feedback_provider_backend_invariants`: SDK fake green ≠ backend green — confirm a sentinel fixture exists); `dikw client check` against a real/stub endpoint, assert exit 0 + sane dims |
| `cli.py`, `server/**`, `client/**` | `uv run pytest tests/server tests/client`; `uv run pytest -v -m slow` (server-e2e); confirm `client/*` imports no `dikw_core.{api,storage,providers,server,eval}` symbol (`tests/test_layering_contract.py`); grep every renamed CLI verb/route/env-var across `CLAUDE.md` + `docs/**` + `CHANGELOG.md` (`feedback_grep_cli_typos_across_docs`) |
| `docs/**`, `*.md` only | `uv run python tools/check_doc_refs.py` (asserts every `dikw <verb>` + `DIKW_*` env var in the docs resolves in source — also a pytest gate); eyeball any `/v1/...` route or frontmatter key shown; `/code-review` is still mandatory (`feedback_code_review_not_optional`) |

## 4. Codex review loop (≤ 3 rounds)
- Run `/codex:review --background`; address each finding; reflect; repeat up to 3 rounds (`feedback_codex_review_loop`).
- When a finding implicates one CLI string / symbol / doc string, **grep the whole repo** before declaring it fixed (`feedback_grep_cli_typos_across_docs`, `feedback_defensive_guard_grep_read_sites`).

## 5. Fresh-agent review (pre-merge, layer 2)
- Run `/code-review` (never optional, doc-only included).
- Run `dikw-core-fresh-review` if present; else spawn a clean subagent with **only** the diff + CLAUDE.md core invariants + `docs/design.md` + `docs/verification-rubric.md`, and have it rate correctness, scope-drift, and invariant adherence, emitting pass / blocking findings.
- Resolve every actionable finding; reject nitpicks with a one-line reason (`feedback_goal_explicit_approves_steps`).

## 6. Doc sync
- Audit all markdown (CLAUDE.md, CONTEXT.md, `docs/**`, CHANGELOG.md, plans, ADRs) against the diff — CLI spellings, frontmatter keys, env vars, HTTP routes.

## 7. Commit + push + PR
- Local commit + `git push` + `gh pr create` proceed without re-asking (the loop is the approval).
- **K-layer / Retrieval / storage PRs need an `evals/BASELINES.md` entry** (real-data outcome) or the `no-baseline-needed` label, or `eval-gate` blocks. Handle here, not at merge.
- **Render the delivery receipt** into the PR body under a `## Delivery receipt` section — the artifact's Evidence — then set the artifact's `PR:` field to the new number. This is the durable, reviewable form of the otherwise self-graded soft steps. **Redact first:** strip absolute home paths, endpoints / base-URLs, and any key or token value from the pasted output before it enters the world-readable PR body (`.env` stays the only home for secrets).

## 8. Watch CI green → squash → sync local
- Monitor `gh pr checks` + reviewer comments (CodeRabbit/human). Fix every actionable finding.
- Only 4 CI checks hard-block (`project_pr_merge_gating_ruleset`); codex/coderabbit/eval-gate are soft — handle them as self-discipline.
- Don't stop until every check is green and `mergeStateStatus` is CLEAN.
- Squash merge, fast-forward local main, delete the feature branch (`feedback_delivery_loop_tail_no_reask`).

</checklist>

<delivery-artifact>

Every run maintains one markdown file, **`.claude/delivery/<branch>.md`** (replace `/` with
`-` in the branch name). It is **gitignored** — the existing `.claude/*` rule already covers
it, so no `.gitignore` change is needed — and lives in the working checkout (lifetime = the
branch's worktree); its durable, shareable form is the `## Delivery receipt` step 7 renders
into the PR body. Three roles across the lifecycle:

- **STATE** (in progress) — step 0 reads it to resume an interrupted task mid-loop.
- **Receipt** (at PR) — proves the soft verify/review steps actually ran, against
  "loops fail quietly"; rendered into the PR body at step 7.
- **Metrics source** (after merge) — a future `tools/loop_metrics.py` reads the rendered
  `## Delivery receipt` PR bodies across merged PRs (codex rounds, fresh-review TP/FP) for
  the `cost per accepted change` readout — the durable PR-body receipts, not these
  gitignored local files.

Template:

```markdown
# Delivery — <branch>

- **PR:** —            (set to #N once opened)
- **Goal:** <one line restated from step 1>
- **Next action:** <the single next step to take on resume>

## Steps
| # | step            | status | evidence                                   |
|---|-----------------|--------|--------------------------------------------|
| 0 | resume          | …      | fresh task / resumed-from-prior-session    |
| 1 | clarify         | …      | assumptions / plan link                    |
| 2 | plan / TDD      | …      | failing-test refs                          |
| 3 | verify          | …      | per-leg table ↓                            |
| 4 | codex (≤3)      | …      | rounds; resolved/rejected                  |
| 5 | fresh-review    | …      | verdict; TP/FP ↓                           |
| 6 | doc-sync        | …      | BASELINES link / no-baseline-needed / N-A  |
| 7 | commit+push+PR  | …      | PR #                                       |
| 8 | CI green→squash | …      | run links; mergeStateStatus                |

## Evidence
### step 3 — verify
<paste the dikw-core-verify per-leg PASS/FAIL/SKIPPED table verbatim>
### step 4 — codex
<rounds run; links to findings resolved / rejected>
### step 5 — fresh-review
<pass/blocking verdict + the TP/FP table verbatim>

## Open findings
- <in-flight findings the next session must still resolve>
```

Paste **real machine output** into Evidence — never a free-text "I ran it"; a self-asserted
receipt is just another soft gate (`feedback_real_data_validation`, in spirit). *The agent
forgets, the file does not.* Redact secrets/paths/endpoints before any of it is rendered
into a public PR body (step 7).

</delivery-artifact>

<block-signals>

STOP and ask the user only when one of these appears (everything else: keep going):

1. A required check is failing and the cause isn't obvious.
2. A reviewer marks `CHANGES_REQUESTED` or raises a design-level concern.
3. A finding requires widening a Storage / Provider Protocol or changing on-disk knowledge/wisdom layout (update `docs/design.md` first).
4. A merge conflict needs a domain decision (not a mechanical resolve).
5. A **force-push** would be needed — **forbidden**; describe the situation and let the user handle it manually (`feedback_pr_workflow`).
6. The request itself is ambiguous on a decision you cannot resolve from code/docs (step 1).

Nitpicks, style preferences, and non-actionable suggestions are **not** block signals — note them and move on.

</block-signals>
