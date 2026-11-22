# Agent harness

Edit `manifest.json` for skill metadata, rule scopes, role names and the session hook. Rule and role files contain Markdown bodies. Each role's sibling JSON file declares its native capabilities. Each skill directory contains its body in `SKILL.md` and any relative resources. `policies/default.rules` holds Codex command-prefix decisions. The shared routing examples are in `tests/routing.json`.

From the current worktree root, use `uv run --offline --no-project python scripts/harness-doctor` with one of these arguments:

- `check` (default) compares the checked-out adapters with the canonical sources.
- `sync` regenerates managed files. Review an unexpected file and incorporate or retire it explicitly.
- `rules <path> ...` prints the Codex area documents for each match. Paths are relative to the caller's directory, or absolute within the worktree. Scope patterns use repository-relative forward slashes, `*`, `**`, `?`, and brace alternatives such as `src/{a,b}.ts`.

The Git shell alias in each startup hook locates this worktree's root even from a nested directory. The doctor delivers a short success or diagnostic message using the client's SessionStart protocol. Keep local settings and session history outside the managed adapter trees.

The doctor manages the declared hooks in `.claude/settings.json` and preserves other project settings. If it finds an unregistered handler, review it, then move personal hooks to `.claude/settings.local.json` or declare shared hooks in the manifest and its adapters. A parent directory's Claude instructions can shadow the repository's `AGENTS.md`. The Claude startup receipt asks the agent to read it when absent.

Skill attachments retain their Git executable bit. On Windows, use the exact `git add --chmod` command from the doctor to stage that bit for new or changed executable attachments and their adapters, then repeat `check`.

The commit hooks call the same verification-stamp implementation in `tools/verify_hook.py` using each client's native input and decision format.
On Antigravity CLI 1.5.12 a permitted operation leaves hook stdout empty so the client applies its own permissions. A rejected commit emits a JSON deny decision. Returning an empty JSON object for normal operations caused this CLI to reject a permitted command in the loader test.

## Client upgrade check

Start a fresh session from a clean checkout with each client. Inspect the startup receipt, the root instructions, the available project skills and the `reviewer` role. Query the scope for `src/dikw_core/config.py` and open its matching documents. Repeat after changing a generated rule in a scratch checkout; the new session should report its path. Restore with `sync` and repeat `check`.

Use `codex execpolicy check --rules .codex/rules/default.rules -- git push --force origin HEAD` to evaluate the command prefix. Compare with `git push origin HEAD` through the same checker. Record the client versions and actual results when changing the supported baseline.

Codex matches literal command prefixes. The shared prohibition in `AGENTS.md` also covers arbitrary remotes, ref names, and forced refspecs.

### Migration results (2026-11-22)

The `rules` lookup for `src/dikw_core/config.py` returned knowledge-layer, module-map, observability, providers-secrets and retrieval. Clean new sessions on Claude Code 2.1.325, Codex 0.160.0 (trusted project hooks) and Antigravity CLI 1.5.12 received the root instructions and a separate `harness-doctor: OK` receipt. All three discovered the five project skills and a reviewer. The loader tests used temporary profiles and local model endpoints.

In the network-isolated Antigravity CLI, changing a generated rule in a scratch checkout produced a `harness-doctor: FAIL` receipt naming the file. Reading the representative file with the headless Gemini provider did not inject the glob rule bodies. The CLI then ran the `rules` lookup from `AGENTS.md` and read every returned document. Claude Code loaded the same rules automatically after a file read.
