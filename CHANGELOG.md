# Changelog

All notable changes to `dikw-core` are tracked here. The project is
**pre-alpha** and follows [SemVer](https://semver.org) loosely — until
1.0, breaking changes can land in any minor version. The status notes
on each entry call out exactly what shape changes break.

## Unreleased

### `lint propose` / `lint apply` — repair closure for broken_wikilink

* **Added**: `dikw client lint propose [--rule <kind>] [--limit N]` runs lint
  + dispatches per-rule fixers, collecting structured `FixProposal`s.
  Result lives in the existing `tasks.result` JSON column — no new
  storage layer, no Storage Protocol changes.
* **Added**: `dikw client lint apply <proposal-task-id> [--pick a,b] [--skip c]`
  reads a successful propose task's result, validates each
  `expected_hash` against the on-disk file (concurrent-edit guard),
  mutates `wiki/` via `wiki.write_page` / unlink, and reconciles
  outgoing wikilinks via `storage.replace_links_from`.
* **Added**: `dikw client lint proposals` lists succeeded propose tasks
  with proposal counts and an "applied?" derived field.
* **Added**: `BrokenWikilinkFixer` — heuristic-only path. Normalizes the
  broken target with Unicode-aware `\w` (CJK / Cyrillic / Greek work,
  not just ASCII), fuzzy-matches existing K-layer titles via
  `difflib.SequenceMatcher`, proposes an in-place `[[link]]` rewrite
  when the ratio crosses 0.85. Targets that the engine's own
  `resolve_links` fuzzy stage would already handle never reach this
  fixer — it covers the typo / edit-distance cases beyond the
  engine's deterministic normalize.
* **Apply safety**: paths are sandboxed under `<base>/wiki/` (rejects
  absolute paths, `..` traversal, and base-relative targets like
  `sources/foo.md`); `update_page` / `delete_page` ops require a
  non-empty `expected_hash`; multiple ops on the same path within one
  apply pass are detected and the second one skipped with an explicit
  "superseded" reason rather than a misleading hash mismatch.
* **Apply contract**: only `lint.propose` task results are accepted as
  proposal sources — passing an unrelated SUCCEEDED task id surfaces
  as `proposal_wrong_op` instead of silently no-op'ing on an empty
  proposal report.
* **Followups**: PR2 plans an LLM stub-page fallback for
  `broken_wikilink` misses + a `non_atomic_page` fixer that reuses
  the synth 1:N fan-out; PR3 adds `orphan_page` + `duplicate_title`
  fixers.

### Agent ergonomics + `--wiki` → `--base` rename

* **BREAKING**: `dikw serve --wiki <path>` is now `dikw serve --base <path>`
  (and `dikw client serve-and-run --wiki` → `--base`). The old flag is
  removed; pre-alpha rebuild policy applies. The on-disk `wiki/`
  subdirectory keeps its name — only the CLI flag pointing at the
  bound directory changed, since "base" is the consistent term for the
  whole tree (containing `sources/`, `wiki/`, `wisdom/`, `.dikw/`,
  `dikw.yml`).
* **Added**: `--format json|table` on `status`, `lint`, `tasks list`,
  `review list` — same contract as `health`, `retrieve`, `pages list`.
  JSON output is unbuffered, suitable for piping into `jq` or feeding
  back into an agent loop.
* **Added**: `--help` epilogs with example invocations on `serve`,
  `init`, `health`, `check`, `retrieve`, `query`, `ingest`, `pages
  list`, `pages get`.
* **Added**: top-level `AGENTS.md` and `INSTALL_FOR_AGENTS.md` for AI
  agents that *use* dikw-core as a knowledge backend (vs. CLAUDE.md
  which targets coding assistants contributing to the engine).

### **BREAKING**: client/server architecture replaces in-process CLI

Phases 0–6 of the `dikw-core-client-server-eventual-clarke` plan
collapse the in-process invocation model. The engine now runs as a
long-lived `dikw serve` process; the CLI is a thin httpx + NDJSON
client that talks to it over `/v1/`.

* **Removed**: `dikw mcp` subcommand and the entire `mcp_server.py`
  module. The MCP runtime dependency is gone from `pyproject.toml` and
  every `mcp.*` reference in code and docs is scrubbed (eval-dataset
  fixture text under `evals/datasets/` is left alone — that's corpus
  content, not engine docs).
* **Removed**: in-process implementations of `dikw status`,
  `dikw check`, `dikw ingest`, `dikw query`, `dikw synth`,
  `dikw lint`, `dikw distill`, `dikw review *`, `dikw eval`. These
  commands are now thin HTTP clients; running any of them requires a
  reachable `dikw serve` instance (or `dikw serve-and-run` for one-shot
  use).
* **Added**: `dikw serve --base <path>` — FastAPI + Uvicorn server.
  Defaults to `127.0.0.1:8765` with no auth on loopback; non-loopback
  hosts require `DIKW_SERVER_TOKEN`. Routes documented in
  [`docs/server.md`](./docs/server.md).
* **Added**: `dikw client *` subcommand group — full remote surface
  (status / check / init / ingest / query / synth / lint / distill /
  eval / review / tasks). Top-level aliases (`dikw status` etc.) keep
  the previous muscle memory working; they now route through the same
  HTTP client.
* **Added**: `dikw client serve-and-run -- <cmd> [args]` — spawns a
  local server, waits for `/v1/healthz`, runs the inner command
  against it, and tears it down. Use this for one-off ingest/query
  flows when you don't want to manage a long-lived server. Picks a
  free port automatically; `--keep-alive` leaves the server up.
* **Added**: NDJSON streaming for long ops (`ingest`, `synth`,
  `distill`, `eval`) and for `query`. Each event is a JSON line; the
  transport drops `heartbeat` events at the client layer. Task
  endpoints support `?from_seq=N` resume so a disconnected client can
  rejoin without missing events.
* **Added**: `POST /v1/upload/sources` — multipart tar.gz + manifest
  upload that the client packs from a local directory. Server
  validates sha256 per file before staging, and the ingest task
  references the staged tree by `upload_id`.
* **Added**: `dikw_core.progress.ProgressReporter` Protocol — the
  engine emits structured progress events (`progress`, `log`,
  `partial`) via this hook; the server bridges them onto the NDJSON
  task event stream, in-process callers (tests, the eval runner) can
  pass `NoopReporter()` and ignore them.
* **Changed**: dependencies. Added `fastapi`, `uvicorn[standard]`,
  `python-multipart`. Dropped `mcp`. The `dikw-core[postgres]` extra is
  unchanged.

### Migration

If you scripted against the old in-process CLI:

* **`dikw status`** etc. — still works, but the server must be
  running. Replace `dikw status` with either `dikw serve-and-run --
  status` or run `dikw serve` in a separate terminal first.
* **`dikw mcp --stdio`** — gone. There is no shim. Adapt to the HTTP
  surface (the wire shape is documented in
  [`docs/server.md`](./docs/server.md)) or pin to the last release
  containing MCP.
* **Configuration** — `dikw.yml` is unchanged. Client-side settings
  (server URL, token) live in `~/.config/dikw/client.toml` or
  `DIKW_SERVER_URL` / `DIKW_SERVER_TOKEN` env vars.
