# Changelog

All notable changes to `dikw-core` are tracked here. The project is
**pre-alpha** and follows [SemVer](https://semver.org) loosely — until
1.0, breaking changes can land in any minor version. The status notes
on each entry call out exactly what shape changes break.

## Unreleased

### feat(server): `GET /v1/base/graph` exposes the full base graph (#89)

* **Wire (additive)**: new `GET /v1/base/graph` returns the entire base
  graph in one read-only request. Replaces `dikw-web`'s old workaround
  of looping `GET /v1/base/pages/{path}` and re-parsing wikilinks in
  the browser. Query: `active` (`true` default = active subset, `false`
  = deactivated subset; matches `GET /v1/base/pages` semantics).
  Response: `{base_revision, generated_at, nodes[{id, path, title,
  layer, active, mtime, inbound, outbound}], edges[{id, source, target,
  type, target_text, anchor, weight}], unresolved[{source, target_text,
  anchor, count}], stats[{node_count, edge_count, unresolved_count}]}`.
* **Determinism contract**: identical base state hashes the same
  `base_revision` (sha256 over sorted per-doc
  `(path, title, layer, mtime, body_sha256, active)` tuples —
  observes current on-disk bodies AND title/metadata changes that
  re-ingest persists without touching bytes; defence-in-depth drops
  any docs whose stored path resolves outside the base before
  hashing) so a client can cheaply skip re-render when nothing
  changed.
  `nodes` / `edges` / `unresolved` are sorted (by path; then
  `(source, target, target_text, anchor)`; then `(source, target_text,
  anchor)`) so two back-to-back calls yield byte-equivalent payloads
  modulo `generated_at`.
* **Aggregation rules**: repeated byte-identical
  `(source, target, target_text, anchor)` edges collapse to one entry
  with `weight > 1`; `inbound` / `outbound` count *distinct* connected
  pages, not raw link occurrences. Unresolved entries aggregate
  byte-identical `(source, target_text, anchor)` pairs the same way.
* **Read-only**: never triggers ingest, synth, or lint apply. Existing
  `/v1/base/pages` and `/v1/base/pages/{path}` contracts unchanged.
* **Engine reuse**: `api.list_graph` reuses
  `domains/knowledge/links.parse_links` + `build_fuzzy_index` +
  `normalize_for_match` — wikilink resolution stays in one place
  (exact title → fuzzy normalize → collision-refuse). URLs are dropped
  from both `edges` and `unresolved` (out-of-graph by design); markdown
  links count as edges only when their href matches a base node.
* **Issue #89 v1 omissions** (deliberate, deferred): no ghost nodes
  for unresolved targets; no `layer=wiki|source|all` query (clients
  filter the node set themselves); no `anchor_count` per node; no
  `suggestions` on unresolved entries.
* **New (CLI)**: `dikw client graph get [--no-active]` mirrors the
  endpoint, agent-first JSON to stdout. Pipe into `jq` for slicing.

### fix(lint): broken_wikilink `--enable-llm` is now evidence-backed (#83)

* **Semantics change**: `dikw client lint propose --rule broken_wikilink
  --enable-llm` no longer creates TODO-stub placeholder pages. The
  LLM is invoked only when the D/I-layer has enough source evidence to
  ground a real K-page; insufficient-evidence cases stay visible in
  the next `dikw lint` run as unresolved `broken_wikilink`.
* **Three rejection paths**, each surfaced as a structured skip reason
  in `FixProposalReport.skipped[].reason` (agent-visible in the
  propose-task result JSON, not just on the live stream):
  * `evidence_insufficient: N chunks, M chars` — D-layer hybrid search
    returned fewer than 1 chunk or under 200 chars total.
  * `rejected_todo_marker` — LLM body still contained `TODO` / `stub
    page` / `placeholder` (defence-in-depth against prompt drift).
  * `rejected_body_too_short` — body cleared the marker check but was
    shorter than 200 chars (rejects "Topic A is a topic." filler).
* **New skip signal**: `FixerSkip(reason)` in
  `domains/knowledge/lint_fix.py` lets any fixer record a structured
  product-semantic skip reason on the propose report. Other fixers are
  unaffected; the orchestrator continues to record `"fixer returned
  None"` for the unstructured-skip path.
* **Prompt rename**: `prompts/lint_fix_broken_wikilink_stub.md` →
  `prompts/lint_fix_broken_wikilink_grounded.md`. The new prompt
  injects retrieved evidence chunks and offers an explicit `REFUSE:
  insufficient evidence` exit when grounding fails.
* **Internal API**: `BrokenWikilinkFixer` now reads `ctx.storage` and
  `ctx.embedding` to retrieve evidence via `HybridSearcher`. No
  changes to `FixerContext` shape, server routes, or client CLI flags
  — `--enable-llm` still toggles the LLM path, the LLM path just
  behaves correctly now.

### Agent-first CLI evolution + remove `query`

* **BREAKING (HTTP)**: `POST /v1/query` is **removed**. dikw-core no longer
  performs in-engine LLM answer synthesis. Agents call `POST /v1/retrieve`
  to get ranked chunks + page refs, then compose the answer with their own
  LLM. Rationale: query rewrite, query expansion, and conversation context
  all live in the agent layer; dikw-core is stateless and structurally
  cannot do query well from inside the engine. (See
  `~/.claude/plans/agent-dikw-resilient-swing.md`.)
* **BREAKING (CLI)**: `dikw client query "..."` is **removed**. Use
  `dikw client retrieve "..."` and run an LLM on the result, or write a
  short shell helper. The `dikw client retrieve` JSON output is stable
  and agent-friendly by default.
* **BREAKING (config)**: `provider.llm_max_tokens_query` field removed
  from `dikw.yml`. `llm_max_tokens_synth` and `llm_max_tokens_distill`
  remain — those are the only legs where dikw-core still calls the LLM
  internally.
* **BREAKING (wire)**: `QueryResult` / `Citation` DTOs removed.
  `AppliedWisdomRef` retained — PR-5 will surface it on a new
  `/v1/wisdom/applicable?q=...` endpoint so agents can preview which
  wisdom items would shape an answer.
* **Internal removal**: `src/dikw_core/server/routes_query.py`,
  `prompts/query.md`, `api.query()`, `_format_applicable_wisdom`,
  `_build_excerpts`, and `QueryStreamRenderer` are all gone. The
  "codex SSE large-input hang" known issue (in-engine streaming LLM
  path) goes with them.
* **Docs**: `docs/design.md`, `docs/architecture.md`, `docs/server.md`,
  `docs/getting-started.md`, `AGENTS.md`, `INSTALL_FOR_AGENTS.md` all
  rewritten to reflect "dikw-core is a knowledge kernel; agents compose
  answers" as the new product invariant.
* **Wire (additive)**: `retrieval_done.hits[].text` now carries the
  **full chunk body** instead of being stripped. Agents consuming the
  intermediate partial event can now prompt directly off it without
  waiting for `final` (or paying a second round-trip for chunk bodies).
  Cost: payload roughly doubles at `limit=100` since chunks duplicate
  on `final.result.chunks`; clients that only need the final result can
  stop reading the stream after `final`. Clients ignoring unknown fields
  are unaffected.
* **BREAKING (CLI)**: `dikw client status` default output flips from
  rich-rendered table to JSON. Human operators add `--format table`
  to recover the previous behavior. Rationale: agent-first principle
  — JSON is the zero-friction format for the dominant caller.
* **BREAKING (CLI)**: `dikw client check` gains a `--format json|table`
  flag (previously rendered table only). Default is `json`. Add
  `--format table` for the previous human-friendly probe summary. Exit
  code (0 / 1) still mirrors per-leg `ok` regardless of format.
* **Fix (CLI)**: `dikw client info` and `dikw client tasks show` now
  emit clean JSON via `console.print_json` instead of
  `console.print(json.dumps(...))`. The old path let rich's soft-wrap
  inject newlines mid-string at long paths, URLs, or error messages,
  breaking `jq` / `json.loads` on agent stdout.
* **Wire (additive)**: `GET /v1/base/pages/{path}/links` exposes the
  K-layer link graph at a page boundary. Query params: `direction=in|out|both`
  (default `both`), `limit=N` (`ge=0`; caps each list independently — a
  hub page with many edges on both sides sees both halves trimmed, not a
  total split, and `limit=0` symmetrically returns empty lists on both
  sides). Response shape: `{path, outgoing[{dst_path, link_type, anchor,
  line}], incoming[{src_doc_id, src_path, link_type, anchor, line}]}`.
  **Graph-hop contract**: every returned edge resolves to an active
  document — bare URLs, markdown links to non-indexed files, and edges
  pointing to deactivated docs are filtered on both sides so the caller
  can always feed `dst_path` / `src_path` back into
  `GET /v1/base/pages/{path}` without 404. Path safety is index-driven,
  same as `GET /v1/base/pages/{path}` — unindexed lookup paths return
  404 with `error.code = page_not_found`.
* **New (CLI)**: `dikw client pages links <path> [--direction in|out|both]
  [--limit N] [--format json|table]` mirrors the new HTTP endpoint.
  Default `--format json` (agent contract); `--format table` renders two
  stacked tables (outgoing / incoming) for humans. Used together with
  `dikw client pages get`, an agent can walk neighbours from a retrieve
  hit without re-parsing wiki bodies for `[[wikilinks]]`.

### `upload` → `import` — rename the source-import verb top-to-bottom

* **BREAKING (CLI)**: `dikw client upload <path>` is now
  `dikw client import <path>`. Top-level alias `dikw upload` is gone;
  use `dikw import`. `dikw auth import` is **unchanged** — it sits in
  the `auth` subgroup and targets the OAuth token store, not the
  base's `sources/`.
* **BREAKING (HTTP)**: `POST /v1/upload/sources` is now
  `POST /v1/import`. Manifest and response shapes are unchanged
  except that the response field `upload_id` is now `import_id`.
* **BREAKING (env)**: `DIKW_SERVER_MAX_UPLOAD_BYTES` is now
  `DIKW_SERVER_MAX_IMPORT_BYTES`. The old name is not read as a
  fallback (pre-alpha; nobody outside this repo depends on it).
* **BREAKING (error code)**: `upload_too_large` is now
  `import_too_large`. Other error codes (`tar_*`, `manifest_*`,
  `package_*`) are unchanged.
* **BREAKING (staging path)**: per-request staging directory moves
  from `<base>/.dikw/upload-staging/<id>/` to
  `<base>/.dikw/staging/<id>/`. The orphan-cleanup pass in
  `runtime.py` additionally rmtrees the legacy
  `.dikw/upload-staging/` once on next startup so users upgrading
  don't leak abandoned transient bytes.
* **BREAKING (backup suffix)**: the per-file backup created during
  the atomic in-place replace inside `_commit_one_file` changes from
  `.bak.upload` to `.bak.import`. The new server doesn't sweep stale
  `.bak.upload` files left over from a pre-rename crash mid-commit
  — they're rare enough that we leave them for the user to delete
  manually.
* **Why**: `upload` is HTTP-wire terminology; the user-facing verb
  for "bring external files into the base" is `import`. The DIKW
  pipeline now reads as `import → ingest → synth → distill`, four
  verbs each pinned to one transition between layers. The HTTP path
  describes what the caller asks the server to do; multipart upload
  remains the **transport mechanism** but is no longer surfaced as
  the **business verb**. See `CONTEXT.md` for the term boundaries.
* **Code moves**: `client/upload.py` → `client/importer.py`;
  `server/routes_upload.py` → `server/routes_import.py`. Class
  rename: `UploadError` → `SourceImportError` (avoids shadowing
  Python's builtin `ImportError`); `UploadResponse` →
  `ImportResponse`; `UploadBundle` → `ImportBundle`. Function rename:
  `build_upload` → `build_import`; `render_upload_report` →
  `render_import_report`.

### `synth` — preserve dominant source language in K-layer pages

* **Changed**: synth prompt (`prompts/synthesize.md`) gains an `## Output
  language` section that instructs the LLM to detect the dominant language
  of the SOURCE DOCUMENT and emit page titles, body H1, body paragraphs,
  tags, and **new** wikilink titles in that same language. Chinese sources
  no longer get translated into English K-pages by default; English sources
  remain unchanged.
* **Changed**: `DEFAULT_SYNTH_SYSTEM` (`domains/knowledge/synthesize.py`)
  now reinforces the same rule as a second-line defence — keeps the
  directive in scope when the user prompt is later truncated under
  context-window pressure. The split `non_atomic_page` lint fixer reuses
  this constant, so its in-place page splits inherit the language rule
  for free.
* **Invariant kept**: `path` and `slug` remain lowercase ASCII kebab-case
  regardless of title language — Obsidian / cross-OS portability of the
  on-disk wiki tree depends on it. For non-ASCII titles the LLM is
  instructed to use a short pinyin or English-equivalent slug; the title
  itself stays in the source language.
* **Tests**: new `test_synth_prompt_preserves_source_language` in
  `tests/test_synthesize_pipeline.py` asserts both the user-prompt template
  and the `DEFAULT_SYNTH_SYSTEM` system prompt carry the rule end-to-end.
* **Out of scope**: `distill` and `query` prompts are NOT yet language-aware
  — Chinese K-pages still risk producing English wisdom and English answers.
  Tracked separately.

### `lint apply` — storage-sync closure + CJK / cross-link correctness

* **Fixed**: `dikw lint apply` for `create_page` ops now registers the
  new K page in storage (document row + chunks + outgoing links)
  instead of just writing the file to disk. Before this fix, a
  freshly-created stub showed up on disk but was invisible to the
  next `run_lint` (which builds its title map from
  `storage.list_documents`) — so users saw the same `broken_wikilink`
  reported again and assumed apply did nothing. There is no separate
  ingest path that closes the gap; lint apply has to do it itself.
* **Fixed**: `lint apply` now reconciles outgoing wikilinks on the
  *referrer* page (the source page that contained the broken
  `[[Title]]`), not just on the page the proposal mutated. Without
  this, a `broken_wikilink → create_page` LLM stub fix would land a
  new K page that `run_lint` immediately reported as `orphan_page`
  because `storage.links_from(source)` was stale.
* **Fixed**: intra-batch cross-links resolve in a single apply pass.
  `non_atomic_page` splits that emit Topic A + Topic B (where A's
  body links to `[[Topic B]]`) used to silently drop A→B because
  `paths_changed` iterated alphabetically — A persisted before B's
  title entered the resolver index. Phase 0 now pre-populates
  `title_to_path` from `op.new_frontmatter` before any persist call
  runs.
* **Fixed**: `BrokenWikilinkFixer` now lets short CJK targets
  (`[[秦朝]]`, `[[疫苗]]`, `[[抗体]]`) reach the LLM stub fallback
  when `--enable-llm` is set. The 4-char heuristic gate (a guard
  against 3-char ASCII substring noise) was applied at the top of
  `propose()` instead of inside the heuristic branch, so 2-3 char
  Chinese entity titles were silently dropped before the LLM path
  could fire — exactly the case Chinese wiki users hit most.
* **Fixed**: `lint apply` now threads the configured
  `retrieval.cjk_tokenizer` (default `jieba`) through to the
  K-layer indexer. Before, lint-apply chunks were always split with
  the no-op `none` tokenizer, diverging from the `doc.hash`
  lint-apply itself wrote and breaking the next embedding backfill
  on Chinese content.
* **Refactored**: K-layer page indexing (document upsert + chunks +
  embeddings + outgoing-link reconciliation) now lives in a single
  `domains/knowledge/page_index.persist_wiki_page` shared by synth
  and lint apply. The function takes `(path, title=None)` and reads
  title fallbacks from disk, so callers don't double-parse the file.
  `wiki.path_slug_title` centralises the path-stem-to-title
  convention previously duplicated in three places.
* **Apply contract**: `_op_title` is the single source of truth for
  "what title should this op produce" — phase 0's resolver index and
  `_build_page_from_op`'s `WikiPage` construction now compute the
  same value (raw frontmatter title stripped of leading/trailing
  whitespace, falling back to `path_slug_title` when missing or
  non-string), so a fixer that omits `title` in `new_frontmatter`
  still gets sibling links resolved correctly.

### Upload decoupled from ingest — new `dikw client upload` command

* **BREAKING**: `dikw client ingest --from <dir>` is removed. Upload
  is now a separate command (see below); `dikw client ingest` only
  scans the server's existing `<base>/sources/` tree.
* **BREAKING**: `POST /v1/ingest` no longer accepts an `upload_id`
  field (`extra="forbid"` rejects it). `commit_staging` and the old
  upload→ingest chain in `server/ingest_op.py` are deleted.
* **BREAKING**: `POST /v1/upload/sources` manifest schema upgrades
  to `{"files": [...], "packages": [...], "total_bytes": N}`. The
  legacy files-only shape returns `manifest_packages_missing`.
  Response gains `committed: list[int]` + `rejected: list[{id, code,
  detail}]`; the legacy `staging_path` field is removed.
* **Added**: `dikw client upload <path>` (top-level alias `dikw upload
  <path>`) — accepts a single `.md` file or a directory whose
  `**/*.md` becomes one package each. Pre-flight inspection
  (frontmatter parse, asset-existence, non-empty body, orphan-asset)
  runs locally; failures exit 2 before the network round trip.
* **Added**: `src/dikw_core/md_inspect.py` — shared module exposing
  `extract_image_refs(body)` and `inspect_markdown(path, *,
  project_root)`. The D-layer `domains/data/backends/markdown.py`
  re-exports `extract_image_refs` so existing callers stay intact.
* **Added**: per-package commit semantics — server validates each
  package's `package_sha256 = sha256(sorted([md_sha, *asset_shas])
  .join("\n"))`, commits the well-formed packages straight into
  `<base>/sources/` via `os.replace`, and reports failed packages
  via `rejected` (still 200, so partial successes don't force a
  retry of the whole batch).
* **Added**: server-startup orphan-staging cleanup —
  `<base>/.dikw/upload-staging/*` is wiped on `build_runtime` to
  cover crash-recovery (a `finally` rmtree in the upload route
  handles the normal path).
* **Added**: server error codes `manifest_packages_missing`,
  `manifest_orphan_file`, `manifest_duplicate_md_path`,
  `manifest_package_unknown_file`, `manifest_package_sha256_mismatch`,
  `package_commit_failed`.
* **Tightened**: tar `_ALLOWED_TOP_DIRS` reduced to `("sources",)`
  — assets ride along under `sources/<rel>` to preserve sibling-of-md
  asset resolution; `assets/` as a top-level archive directory is no
  longer accepted.

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

### `lint propose` / `lint apply` — PR2: LLM stub fallback + non_atomic_page splitter

* **Added**: `dikw client lint propose --enable-llm` opts the configured
  LLM into the per-rule fixers. Default off — heuristic-only propose
  stays free of token spend; users opt in explicitly because each
  issue may incur a real LLM call.
* **Added**: `BrokenWikilinkFixer` LLM stub fallback. When the
  fuzzy-match heuristic misses, the fixer asks the LLM to draft a
  stub page (matching title + TODO marker, no invented facts) so the
  wikilink resolves on the next lint pass. Refuses to overwrite
  existing K-pages and strips Obsidian alias / anchor syntax from
  the broken target so `[[X|alias]]` and `[[X#section]]` resolve to
  the bare `X` title the resolver expects.
* **Added**: `NonAtomicPageFixer` — splits a page flagged as
  non-atomic into N atomic children + delete the original.
  LLM-only (no heuristic; `--enable-llm` required). External
  wikilinks pointing at the original are intentionally NOT rewritten
  — the next lint pass surfaces them as `broken_wikilink` issues
  that the stub fallback / fuzzy match handles.
* **Added**: `synthesize_pages_from_text` shared helper in
  `domains/knowledge/synthesize.py` and `safe_synthesize_pages`
  wrapper in `domains/knowledge/lint_fix.py` — single seam for
  "text → N pages" used by both the LLM-stub and split fixers.
  Handles `SynthesisPartialError` with a `strict=True` mode for
  destructive callers (refuse any partial parse) vs `strict=False`
  for additive callers (return parsed pages).
* **Apply atomicity**: `run_lint_apply` now preflights every op of
  every proposal against current disk state (collisions, missing
  files, hash drift, sandbox refusal). If ANY op would fail, the
  whole proposal skips at op #0 — no half-applied state on disk.
  A multi-op proposal where create_page #1 succeeds and create_page
  #2 collides used to leave child #1 orphaned + the original still
  present; preflight closes that gap.
* **Safety guards**:
  - `non_atomic_page` skips bodies > 32 KB to avoid the openai_codex
    SSE keepalive timeout on very large prompts.
  - `non_atomic_page` uses its own 16-child ceiling (decoupled from
    `cfg.synth.max_pages_per_group`) and refuses any split where the
    LLM emitted exactly the ceiling — the model voluntarily stops at
    the cap with no truncation signal, so we can't tell whether
    topic 17 just didn't exist or got dropped silently.
  - `safe_synthesize_pages` returns `None` on `retry=True` partials
    (`max_tokens` truncation — recoverable next run with a bigger
    budget) regardless of caller mode.
  - Both LLM fixers refuse `create_page` paths that collide with
    existing K-pages; the split fixer aborts the entire proposal on
    any child collision rather than silently dropping the colliding
    child's content.
* **Followups**: PR3 still owes `orphan_page` + `duplicate_title`
  fixers per the original lint-fix closure plan.

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
