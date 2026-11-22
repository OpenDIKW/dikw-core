# dikw-core contributor instructions

## Start each new session

The SessionStart hook runs the repository's compatibility check. If this session has no `harness-doctor: OK` receipt, run the following command from the current worktree root before working.

```sh
uv run --offline --no-project python scripts/harness-doctor check
```

If the check reports drift, inspect the named paths. Preserve intentional edits in `harness/`, run the same command with `sync`, then repeat `check`. Reconcile unregistered files by hand. The check is read-only. A missing runtime or an untrusted hook needs a new check before claiming success.

Before designing or editing a file, run `uv run --offline --no-project python scripts/harness-doctor rules <path> [<path> ...]` from the root and read **every** returned file. Use this lookup for planned files and when the client omits automatic path rules. Edit the canonical bodies, scope patterns, agent definitions and hooks under `harness/`. Regenerate and check the adapters in the same change.

## Working agreement

- Write plans and reports in Chinese. Use English for code, identifiers, commits and PR titles. Use `CONTEXT.md` for domain language.
- Read `docs/design.md` before changing the K or W layers. Use deterministic scoping for retrieval and probabilistic reasoning for synth. Existing Markdown, wikilinks and provenance edges are user-facing contracts.
- Preserve the client/server import boundaries and the Storage Protocol. New adapter behavior needs the shared storage contract and the real Postgres leg.
- Reproduce bugs with a failing test. K-layer and Retrieval changes require test-first work and a real-data entry in `evals/BASELINES.md`. For K, include the elon-musk baseline and seven synth metrics. For retrieval, compare packaged datasets. See `docs/eval-plan.md`.
- Run `uv run python tools/check.py` before committing. It records this worktree's verification stamp. In Codex use `git -C "<absolute worktree path>" commit` so the hook receives the target. Follow `dikw-core-verify` for additional legs and report missing real-provider prerequisites.
- For a non-trivial change follow `dikw-core-delivery-workflow`, including the risk-tier reviews, resumable `.claude/delivery/<branch>.md`, and the PR's **Delivery receipt**. A high-risk design review starts with a fresh reviewer and the project's design-invariant rubric.
- State material assumptions. Keep the change scoped, verify evidence from this run, and continue until the requested completion condition. Ask when a decision blocks progress.
- The delivery workflow covers commit, push, PR, reviewed squash merge and branch cleanup. Never force-push. Keep real credentials in the local environment.
- Close reports with decisions needed, changes and findings.
