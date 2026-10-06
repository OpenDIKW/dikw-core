---
name: dikw-core-delivery-workflow
description: The dikw-core delivery loop — drive a non-trivial change from request to squash-merged PR through resume → clarify → plan/TDD → in-loop verify → review tiered by risk (codex, /code-review, fresh-review) → doc-sync → PR with delivery receipt → CI green → squash. Stops only on its block signals. Use for any non-trivial dikw-core change (feature, bugfix, refactor), so the sequence is run, not remembered.
---

<what-this-is>

This skill is the only definition of the dikw-core delivery loop. CLAUDE.md points here.

**Finish line.** The loop is done when all of these are true:

- The PR is squash-merged, local `main` is fast-forwarded, and the feature branch is deleted.
- Every required check is green, and every actionable review comment is fixed or answered.
- The PR body has the `## Delivery receipt` section.

If you cannot reach the finish line, stop on a block signal and report it.

**Run autonomously.** The loop is standing approval to commit, push, open the PR, squash-merge, and delete the merged feature branch.
Do not ask again at the tail of the loop. Stop only on a block signal (see `<block-signals>`).
Put status notes in the same message as your next tool call.

Companion skills:

- `dikw-core-verify` — step 3. Routes the diff to the checks it needs.
- `dikw-core-verify-synth` — the K-layer leg of step 3. `dikw-core-verify` calls it.
- `dikw-core-fresh-review` — step 4 for tier-L changes. A clean subagent reviews the diff against the invariants.

</what-this-is>

<checklist>

Track the steps in the delivery artifact's Steps table. It survives context compaction; an in-context to-do list does not.
Do not skip a step because it "feels unnecessary".

## 0. Resume — read the delivery artifact first

- Read `.claude/delivery/<branch>.md`. The branch name is the path; a `/` makes a subdirectory (`feat/foo` → `.claude/delivery/feat/foo.md`).
- If the file exists, resume at its **Next action**. Do not start again from step 1.
- If it does not exist, create it from the `<delivery-artifact>` template. Fill in **Goal** after step 1.
- After every step, update the header fields (Goal, Next action, PR) and that step's status and machine evidence.

## 1. Clarify the request

- Restate the goal. List your assumptions. Name the alternatives.
- For work with several design decisions, use the `grill-with-docs` skill until a written plan exists.
- **STOP** and ask only if a decision blocks you and the code and docs cannot answer it. Then ask one question with the AskUserQuestion tool, with your recommended answer first.

## 2. Plan in Chinese, test first

- Write the plan in Chinese. Write code, commits, and identifiers in English.
- Each step lands as **failing test → implementation → passing test**.
- K-layer (`domains/knowledge/`) and Retrieval (`domains/info/`) changes: write the failing test before you touch the implementation. This is mandatory.
- Pick the review tier now (see step 4) and record it in the artifact.

## 3. In-loop verify

- Run the `dikw-core-verify` skill. It runs the shared floor (`uv run python tools/check.py`) and the legs that the diff needs.
- A red leg is yours to fix. Fix it and run it again. Do not hand back a check that you can run.

## 4. Review — tiered by risk

Find the highest tier that any file in the diff hits. When in doubt, go one tier up.

| tier | the diff touches | review |
|---|---|---|
| **S** | only docs outside `src/` (`*.md`, `docs/**`, `.claude/**`); or a version / release / `DIKW_VERSION` bump | `/code-review` once |
| **M** | anything else: code, tests, tools, dependencies, CI | codex (≤ 3 rounds) + `/code-review` |
| **L** | `src/dikw_core/domains/knowledge/**`, `src/dikw_core/api_synth.py`, `src/dikw_core/prompts/**`, `src/dikw_core/domains/info/**`, `RetrievalConfig` in `src/dikw_core/config.py`, `src/dikw_core/storage/**`, `src/dikw_core/providers/base.py`, the persist pipeline (`src/dikw_core/domains/data/persist.py`, `src/dikw_core/domains/wisdom/persist.py`, `src/dikw_core/domains/knowledge/page_index.py`), `src/dikw_core/server/auth.py`; or any other `src/` file that matches the `paths:` of `.claude/rules/{knowledge-layer,persist-pipeline,retrieval,storage-concurrency,delete-trash}.md` | tier M + `dikw-core-fresh-review` |

Run the reviews in this order:

1. **Codex (tiers M and L).** Run `codex review --base main` in the background. The `/codex:review` slash command is user-only, so call the CLI. Fix the findings, then run it again. Stop after 3 rounds or when a round has no new actionable finding.
2. **`/code-review` (all tiers).** It is never optional. Doc-only PRs also get real findings from it.
3. **`dikw-core-fresh-review` (tier L).** Run it after the codex rounds are quiet.

Triage every finding the same way:

- Ask each reviewer to report merge-blocking problems first, each with the file and line, why it is wrong, and how to show that it fails.
- Before you fix a finding, read the cited code. A reviewer can be wrong.
- Fix every actionable finding, blocking or not.
- Reject a nitpick or a false positive with a one-line reason.
- If a finding names one CLI string, symbol, route, or env var, grep the whole repo for it before you call it fixed. Include `CLAUDE.md`, `docs/**`, `CHANGELOG.md`, `.claude/**`.

## 5. Re-verify after review fixes

- If a review fix changed code, run `uv run python tools/check.py` again.
- Record the review results (rounds, verdicts, rejected findings) in the artifact.

## 6. Doc sync

Check every Markdown file against the diff: `CLAUDE.md`, `.claude/rules/**`, `CONTEXT.md`, `docs/**`, `CHANGELOG.md`, plans, ADRs, `GUIDE_FOR_AGENTS.md`.
Look for stale CLI spellings, front-matter keys, env vars, and HTTP routes. Fix each one in this PR.

## 7. Commit, push, open the PR

- Commit, push, and run `gh pr create`. Do not ask first.
- **K-layer, Retrieval, and storage PRs** need an `evals/BASELINES.md` entry with the real-data outcome, or the `no-baseline-needed` label. Without it `eval-gate` blocks. Handle it here, not at merge time.
- Set the artifact's `PR:` field as soon as the PR exists.
- Render the artifact's Evidence into the PR body under `## Delivery receipt`.
- The PR body is public. Scan the receipt for secrets, absolute paths, and endpoints before you post it.

## 8. CI green → squash → sync

- Watch `gh pr checks` and the review comments (CodeRabbit, humans). Fix every actionable finding with the step-4 triage rules.
- Required checks on `main`: `ci / lint-type-test (3.12)`, `ci / lint-type-test (3.13)`, `ci / Postgres contract tests`, `ci / Server e2e (serve-and-run)`, `codecov/patch`. Codex, CodeRabbit, and `eval-gate` are not required. Still treat their findings with the triage rules.
- Continue until every check is green and `mergeStateStatus` is `CLEAN`.
- Squash-merge. Fast-forward local `main`. Delete the feature branch.
- End with the Report section from CLAUDE.md.

</checklist>

<delivery-artifact>

Every run keeps one Markdown file: **`.claude/delivery/<branch>.md`**.

- The existing `.claude/*` rule in `.gitignore` already ignores it.
- It lives in the working checkout, so its lifetime is the branch's worktree.
- Its durable, public form is the `## Delivery receipt` that step 7 puts in the PR body.

It has three roles:

- **State** (in progress) — step 0 reads it to resume a task that a session left unfinished.
- **Receipt** (at the PR) — records that the review and verify steps ran, with their machine output. It does not re-run them the way CI re-runs the deterministic floor, so it is not a hard gate. It makes a silently skipped step visible.
- **Metrics source** (after merge) — `tools/loop_metrics.py` reads the receipts in merged PR bodies, plus the GitHub API.

Template:

```markdown
# Delivery — <branch>

- **PR:** —            (set to #N once opened)
- **Goal:** <one line restated from step 1>
- **Tier:** <S | M | L — and the path that set it>
- **Next action:** <the single next step to take on resume>

## Steps
| # | step            | status | evidence                                   |
|---|-----------------|--------|--------------------------------------------|
| 0 | resume          | …      | fresh task / resumed-from-prior-session    |
| 1 | clarify         | …      | assumptions / plan link                    |
| 2 | plan / TDD      | …      | failing-test refs                          |
| 3 | verify          | …      | per-leg table ↓                            |
| 4 | review          | …      | codex (N rounds); /code-review; fresh-review **pass** / **blocking** / N-A |
| 5 | re-verify       | …      | check.py after review fixes                |
| 6 | doc-sync        | …      | BASELINES link / no-baseline-needed / N-A  |
| 7 | commit+push+PR  | …      | PR #                                       |
| 8 | CI green→squash | …      | run links; mergeStateStatus                |

## Evidence
### step 3 — verify
<paste the dikw-core-verify per-leg PASS/FAIL/SKIPPED table verbatim>
### step 4 — review
- codex (N rounds): <findings fixed / rejected>
- /code-review: <result>
- fresh-review **pass** | **blocking** (tier L only): <TP/FP table>

## Open findings
- <findings the next session must still resolve>
```

Rules for the Evidence section:

- Keep `codex (N rounds)` and `fresh-review **<verdict>**` each on one line. `tools/loop_metrics.py` parses exactly these forms.

- Paste **real machine output**. A free-text "I ran it" is not evidence.
- **Redact when you write**, not later. Remove secrets, absolute paths, and endpoints before the text goes into the file. `.env` is the only place for secrets.

</delivery-artifact>

<block-signals>

STOP and ask the user only when one of these occurs. For everything else, continue.

1. A required check fails and the cause is not clear.
2. A reviewer sets `CHANGES_REQUESTED` or raises a design-level concern.
3. A finding needs a wider Storage or Provider Protocol, or a change to the on-disk knowledge/wisdom layout. (Update `docs/design.md` first.)
4. A merge conflict needs a domain decision, not a mechanical resolve.
5. **WARNING:** a force-push would be necessary. Force-push is forbidden. Describe the situation and let the user do it.
6. The request is ambiguous on a decision that the code and docs cannot answer (step 1).
7. The next action is destructive and this loop does not already approve it: deleting data or files you did not create, or changing anything outside this repository.

Nitpicks, style preferences, and non-actionable suggestions are **not** block signals. Note them and continue.

</block-signals>
