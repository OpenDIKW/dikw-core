# Changelog

All notable changes to `dikw-core` are tracked here. The project is
**alpha** and follows [SemVer](https://semver.org) loosely — until
1.0, breaking changes can land in any minor version. The status notes
on each entry call out exactly what shape changes break.

## Unreleased

### Changed: Dockerfile DIKW_VERSION now auto-syncs after PyPI publish

`examples/docker/Dockerfile`'s `ARG DIKW_VERSION` used to lag
`pyproject.toml` whenever a maintainer forgot to hand-bump it after a
release — 0.2.5 / 0.2.6 / 0.2.7 all shipped with the Dockerfile still
pointing at 0.2.0, a 7-patch drift. The constraint (the Dockerfile
must point at a version already on PyPI, because Trivy builds the image
on every PR and `pip install`s from PyPI) cannot be solved by simply
syncing the two files inside one commit — there's a real window where
`pyproject.toml` is bumped but PyPI hasn't received the wheel yet.

Two complementary fixes close the loop:

1. **`sync-dockerfile` job in `.github/workflows/release.yml`** — after
   the `publish` job successfully uploads to PyPI, this new job uses
   `peter-evans/create-pull-request` (pinned by SHA, v8.1.1) to open a
   `chore(docker): bump DIKW_VERSION to vX.Y.Z` PR against `main`. The
   PR carries the `chore` / `automated` / `no-baseline-needed` labels.
   The job refuses to edit if the `ARG DIKW_VERSION=…` line shape
   changed (renamed arg, multi-line assignment, etc.) — better to fail
   loudly than produce a malformed Dockerfile.

2. **`dockerfile-version-guard` job in
   `.github/workflows/reusable-ci.yml`** — runs on every PR (and as
   part of the release pre-gate). Enforces the invariant
   `DIKW_VERSION == pyproject.version` OR
   `DIKW_VERSION ∈ published PyPI releases`. The first branch is the
   post-sync steady state; the second is the legitimate publish-window
   transitional state. Anything else (hand-edited drift, typo, stale
   bump) fails the guard with an actionable `::error` annotation. The
   job pins Python 3.12 via `actions/setup-python` (pinned by SHA,
   v6.2.0) rather than relying on whatever `python3` ubuntu-latest
   currently ships — `tomllib` is Python 3.11+ stdlib, so the explicit
   pin keeps the guard valid across future runner-image rolls.

Both halves share the same `^X.Y.Z$` version contract: sync-dockerfile
refuses to write a PEP 440 pre-release (e.g. `0.2.7rc1`, `0.2.7.dev1`)
into the Dockerfile so the guard cannot later jam on a shape-mismatch.
If pre-release support is ever needed, relax the guard regex first.

Caveat: PRs opened by the default `GITHUB_TOKEN` do not trigger other
workflows on the new PR. Maintainers who want Trivy to scan the
sync PR before merging should close and re-open it manually; otherwise
the guard re-validates on the next non-sync PR. This is a documented
GitHub-side limitation rather than a workaround we can paper over
without provisioning a PAT or GitHub App.

The current `examples/docker/Dockerfile` is also bumped from 0.2.0 to
0.2.7 in this entry to absorb the historical drift before the guard
goes live — without this catch-up, the very first PR after merge would
fail the new guard.

### Changed: project status bumped from pre-alpha to alpha

The repo-wide self-description graduates from **pre-alpha** to **alpha**:
the PyPI classifier (`pyproject.toml`) moves to `Development Status :: 3 -
Alpha`, and every doc / docstring / SQL header / runtime hint string that
previously said "pre-alpha" now says "alpha" — README, CLAUDE.md,
`docs/architecture.md`, `docs/eval-plan.md`, `CHANGELOG.md` header,
`.github/pull_request_template.md`, `storage/_schema.py`,
`storage/migrations/{sqlite,postgres}/schema.sql`, and the
`_REBUILD_HINT` strings in `storage/sqlite.py` + `storage/postgres.py`
(which surface in `StorageError` when a user opens a DB whose schema
fingerprint doesn't match the code's `SCHEMA_VERSION`). The
"rebuild-on-incompatibility" storage policy and the "APIs / on-disk
formats / DB schema / CLI may change" disclaimer are both unchanged in
substance — alpha still means breaking changes can land in any minor
version. The architecture-doc forward reference updates accordingly:
the real migration framework now lands when we declare **beta**, not
alpha. Two unrelated Delivery-loop doc fixes ride along: CLAUDE.md
Working principles + step 4 now reference `/code-review` (the
production skill) instead of the retired `/simplify` placeholder, and
step 4 explicitly notes that doc-only PRs are not exempt from
`/code-review` (per `feedback_code_review_not_optional`).

## 0.2.7 — 2026-05-24

### Fixed: stale `--enable-llm` copy from PR2 era + sweep gaps surfaced by /code-review

The `broken_wikilink` lint fixer's `--enable-llm` behavior was upgraded
in #83 (squash-merged to main as #86, commit `c1ae5b2`) from "LLM
fabricates a TODO-laden placeholder page" to a real **evidence-backed
grounded repair** (D-layer hybrid search gates an LLM call; outputs
containing `TODO` / `stub page` / `placeholder` markers or shorter than
200 chars are rejected). The runtime landed correctly, but many
user-facing and internal description sites still described the obsolete
behavior — most visibly the CLI `--enable-llm` help text, the
`POST /v1/lint/propose` OpenAPI schema (via the Pydantic docstring),
and the public engine `lint_propose` docstring. Users reading any of
these would conclude that `--enable-llm` produces stub pages and avoid
opting in, missing the new functionality's value.

This release sweeps the stale copy across every site surfaced by the
initial pass plus a follow-up `/code-review` pass: CLI help, OpenAPI
schema, public engine docstring, the `lint_fixers` package docstring,
the `FixerContext`, `synthesize_pages_from_text`, and
`_build_page_from_op` docstrings, the `non_atomic_page` cross-reference
to `broken_wikilink`, the `_strip_alias_anchor` docstring, an
`orphan_page.py` token-budget comment, an internal comment in
`broken_wikilink.py`, plus several historical references and fixture
strings in `tests/test_lint_*.py` and `tests/test_synthesize_*.py`. The
`broken_wikilink.py` file-level docstring's "this replaces the PR2
TODO-stub fallback" contrast paragraph is intentionally left in
place — it explains the historical change to readers and is the
canonical source for the new phrasing.

This release also tightens three accuracy gaps the `/code-review`
surfaced:

* **`orphan_page` was missing from every public description.** The
  `merge_into_existing_page` strategy is also gated on
  `ctx.enable_llm`, but no `--enable-llm`-related copy (CLI / OpenAPI /
  engine / `FixerContext` / `synthesize_pages_from_text`) mentioned
  it. All five sites now enumerate orphan_page alongside
  broken_wikilink + non_atomic_page.
* **Marker-list shorthand was inaccurate.** New copy said
  "`TODO`/`stub`/`placeholder` are rejected" but the canonical
  `_FORBIDDEN_BODY_TOKENS` tuple is `('TODO', 'stub page',
  'placeholder')` — bare `stub` does NOT trigger rejection. Copy now
  matches the tuple verbatim.
* **A hallucinated class name.** `tests/test_synthesize_pipeline.py`
  fixture body string called the emitting test-stub class `FakeLLM`,
  but the actual class on line 217 is `GroupAwareLLM` (`FakeLLM` is
  an unrelated class in `tests/fakes.py`). Renamed to match the
  declared class.

Test fixture `rationale` fields in `tests/test_lint_apply.py` are now
the neutral string `"test fixture"` (was `"LLM-generated stub"` and
`"evidence-backed grounded repair"`) — fixtures with sub-200-char
bodies containing `TODO`/`stub` markers should not pretend to be real
grounded outputs.

**Pure documentation + test-string change. No runtime, schema,
Protocol, or on-disk-format behavior changed.** Users who already had
`--enable-llm` in production saw the new behavior land in 0.2.x at #83.

## 0.2.6 — 2026-05-23

### Added: provenance edge (K-page → D-source attribution)

The page→source attribution recorded in each K-page's `sources:`
frontmatter is now a queryable edge in its own right — distinct from
body-derived `[[wikilink]]` references. This closes the
"which K-pages were synth-authored from this source?" gap that
previously required scanning every wiki page's frontmatter by hand.
See [`docs/adr/0001-provenance-as-separate-edge.md`](docs/adr/0001-provenance-as-separate-edge.md)
for why provenance lives in a dedicated table.

* New `provenance(src_doc_id, source_path, source_path_key)` storage
  table on both SQLite + Postgres, with a reverse-lookup index on
  `source_path_key`. PK on the normalized key collapses case / NFC
  drift to one row. No FK on either side — mirrors the `links` table.
* New `Storage` Protocol methods: `replace_provenance_from`,
  `provenance_from`, `provenance_to`. `delete_document` now also
  purges provenance rows where the deleted doc is `src_doc_id`.
* `persist_wiki_page` reconciles provenance from frontmatter on every
  persist, mirroring the existing wikilink reconcile. Edits to
  `sources:` self-heal on the next synth / lint-apply pass.
* New HTTP endpoint
  `GET /v1/base/pages/{path}/provenance?direction=in|out|both&limit=N`
  returns a `PageProvenanceResult` with `derived_from` (forward) +
  `derived_pages` (reverse). Forward entries that don't resolve to an
  active `Layer.SOURCE` document are surfaced with `resolved=false`
  rather than silently dropped — agents can detect provenance drift.
* New CLI `dikw client pages provenance <path> [--direction in|out|both]
  [--limit N] [--format json|table]` — JSON default per the
  agent-first contract. Table mode renders ✓/✗ in the resolved column.
* New API: `dikw_core.api.read_provenance(root, path, *, direction,
  limit)` — pure helper used by the HTTP route, exposed for in-process
  agent embeddings.

### Added: `missing_provenance` lint kind + deterministic fixer

Legacy bases that existed before 0.2.6 carry K-pages whose
frontmatter declares `sources:` but whose `provenance` table is empty.
The same drift can happen when a user hand-edits `sources:` outside
synth / lint-apply.

* New `LintKind = "missing_provenance"`. Fires when a page's
  frontmatter `sources:` exists but doesn't match the `provenance`
  table. Suppressible per-page via
  `lint: {skip: [missing_provenance]}` frontmatter (same shape as the
  other kinds).
* New `MissingProvenanceFixer` is pure-deterministic (no LLM call,
  no provider dependencies). It emits a single
  `reconcile_provenance` `FixOperation` carrying the frontmatter
  snapshot + `expected_hash` for concurrent-edit safety.
* New `FixOperation.kind = "reconcile_provenance"` — the narrowest
  possible write: one `storage.replace_provenance_from` call, no file
  mutation, no chunk / embedding / link side effects. Phase-1
  re-persist is skipped; `wiki_paths_changed` stays empty for this
  op kind.
* On a legacy base whose storage *still has the K-layer rows* (i.e.,
  you haven't rebuilt yet — see "Changed (storage)" below), run:

  ```bash
  dikw client lint propose --rule missing_provenance
  dikw client lint apply <task_id>   # the id printed by propose
  ```

  to backfill the new table from existing wiki frontmatter without
  touching any files. Heuristic-only — no LLM call, no embedder
  required.

### Changed (storage): schema fingerprint bumped to v3

`SCHEMA_VERSION = 3` (was 2). The new `provenance` table is additive,
but pre-alpha policy is "rebuild on incompatibility" — any DB at v2
will refuse to migrate in place. The eval-cache fingerprint includes
`SCHEMA_VERSION`, so snapshots from v2 will not be reused under v3.

* **Migration (SQLite)**: `rm -rf <base>/.dikw/index*` then re-run
  `dikw client ingest`.
* **Migration (Postgres)**: drop and recreate the configured schema,
  then re-run `dikw client ingest`.
* `dikw client ingest` only rescans `sources/` — it does not re-index
  `wiki/` (this is the pre-existing gap noted below). After a rebuild,
  the K-layer rows are therefore gone and the `missing_provenance`
  backfill above has no documents to operate on yet. The only
  automated path to repopulate `wiki/` today is `dikw client synth …`
  from the original sources, which regenerates wiki pages from
  scratch — **back up any hand-edits under `wiki/` first**, or hold
  off the rebuild until the "wiki rescan in ingest" follow-up lands.
  Once `wiki/` rows are back in storage, run the propose + apply pair
  shown above to populate `provenance`.

### Notes

* Provenance is a **navigation edge only** — it does not feed RRF
  retrieval, does not affect `orphan_page` / `broken_wikilink` counts,
  and required no eval-baseline update.
* The pre-existing gap that `dikw client ingest` does not rescan
  `wiki/` (and so does not detect hand-edits to wiki bodies or
  frontmatter) is **not** addressed here. It applies equally to the
  `links` table today. Tracked separately as a "wiki rescan in ingest"
  follow-up.

## 0.2.5 — 2026-05-21

### BREAKING (CLI): agent-first default-JSON audit completed

The maintenance-side `dikw client` commands deferred by the 0.1.0
agent-first audit now default to JSON, finishing the matrix: every
`dikw client` command's default output is agent-parseable, and humans
opt into rendered output via `--format table` (or `--pretty`).

* **`dikw client lint`, `lint proposals`, `review list`, `tasks list`** —
  default output flips from a rich table to JSON. Humans add
  `--format table`. Exit codes are unchanged (`lint` still exits 1 when
  issues are found, regardless of format).
* **`dikw client review approve` / `review reject`** — now emit the raw
  `{item_id, new_status}` JSON on stdout by default so agents can pipe to
  `jq` without stripping ANSI. Pass `--pretty` for the colored human
  line (mirrors `dikw client tasks cancel`).
* **`dikw client import`** — gains `--format json|table`, default JSON.
  The committed / rejected summary is now parseable; `--format table`
  renders the previous human summary.
* `--format` help text is unified across every command to
  `Output format: 'json' (default) or 'table'.`.
* **Migration**: any script/agent that scraped the human table or colored
  line from these commands must either add `--format table` / `--pretty`
  to keep the old rendering, or (recommended) switch to consuming the
  JSON now emitted by default.

## 0.2.0 — 2026-05-19

### BREAKING (HTTP): `GET /v1/tasks` response shape changed

* `GET /v1/tasks` no longer returns a bare `list[TaskRow]`. It now
  returns a `TaskListPage` envelope:

  ```json
  {
    "tasks": [TaskRowSummary, ...],
    "next_cursor": "<opaque base64url cursor or null>",
    "has_more": true | false
  }
  ```

* `TaskRowSummary` is a **summary** projection — it omits `result`
  and `error`. A succeeded synth / eval task can stamp tens of KB
  into `result`, and the list view exists to *find* tasks, not to
  read their bodies. For full detail use `GET /v1/tasks/{id}` (whole
  row, including `result` / `error`) or `GET /v1/tasks/{id}/result`
  (terminal payload).
* The CLI mirror `dikw client tasks list --format json` now emits
  the envelope verbatim on a single page; pass `--all` to drain the
  cursor and emit a flat JSON array.
* **Migration**: any client that consumed the old bare-array shape
  must unwrap `.tasks` and treat `result` / `error` as absent. Any
  client that looped through results to read `r.result` (e.g. the
  pre-0.2.0 `dikw client lint proposals` cross-reference) must fan
  out to `GET /v1/tasks/{id}/result` per row — the in-tree client
  already does this.

### Added: cursor pagination on `GET /v1/tasks`

* New `?cursor=<opaque>` query parameter; the server hands back the
  next page's cursor in `next_cursor`. Pagination uses keyset over
  `(created_at DESC, task_id ASC)` so same-millisecond submissions
  page deterministically without skipping or repeating rows.
* `?limit=` (default 100, max 1000) is the page size, not the total
  cap. Pair with the cursor to walk arbitrarily large queues.
* `?status=` and `?op=` filters compose with `?cursor=` — cursor
  positions advance inside the filtered set, not the unfiltered one.
* Malformed cursors fail loudly: `400 invalid_cursor` (stable error
  code, suitable for agent branches).
* CLI: `dikw client tasks list` gains `--all` (drain the cursor and
  emit a flat array; default is single-page envelope passthrough)
  and `--cursor <opaque>` (resume from a prior response).

### Fixed: `dikw serve --help` no longer claims NDJSON for tasks

* The `dikw serve` docstring previously described the server as
  "FastAPI + NDJSON". Task events have been cursor JSON + long-poll
  since the task-first flip; only `POST /v1/retrieve` is still
  NDJSON. The help text now reads "FastAPI, JSON long-poll task
  events".
* `docs/design.md` references that called the task event endpoint a
  "NDJSON streamer" have been corrected to "paged JSON cursor".

## 0.1.0 — 2026-05-18

### BREAKING (CLI): top-level short names for HTTP commands removed

* **All HTTP-bound commands now live exclusively under `dikw client *`.**
  The splice in `cli.py` that previously exposed `dikw status`,
  `dikw retrieve`, `dikw ingest`, `dikw synth`, `dikw distill`,
  `dikw eval`, `dikw lint`, `dikw review`, `dikw pages`, `dikw assets`,
  `dikw graph`, `dikw tasks`, `dikw info`, `dikw health`, `dikw check`,
  `dikw import`, and `dikw serve-and-run` is **gone**. Each of those
  commands now exits non-zero with Typer's `No such command "<name>"`.
* **Migration**: replace `dikw <verb>` with `dikw client <verb>`
  everywhere, including `dikw serve-and-run` → `dikw client serve-and-run`.
* **Why**: the splice gave the appearance that local-only and HTTP-bound
  commands shared a flat namespace, when in reality they have different
  failure modes (a missing `dikw serve` only affects HTTP commands).
  Spelling out the `client` prefix keeps the local/HTTP boundary
  unambiguous for agents and humans, and matches the agent-friendly
  default-JSON contract already shipped under `dikw client *`.
* **Unchanged**: the four local-only top-level surfaces — `dikw version`,
  `dikw init <path>`, `dikw serve`, and the `dikw auth {login, import,
  status, list, logout}` subgroup — keep working.

### BREAKING (CLI + HTTP): `dikw client init` and `POST /v1/init` removed

* **CLI**: `dikw client init` is gone. The server's runtime already
  refuses to start without a `dikw.yml` in place, so the command was a
  permanent no-op (returning 409 `wiki_already_initialised` on every
  call). For *local* scaffolding, the top-level `dikw init <path>`
  command (which writes files directly without contacting any server)
  is unchanged.
* **HTTP**: `POST /v1/init` is removed; calling it now returns 404. The
  `InitRequest` / `InitResponse` DTOs are deleted from `routes_sync.py`.
* **Env**: `DIKW_SERVER_DISABLE_INIT` is no longer read by anything. The
  posture it provided ("production base where clients can never trigger
  rescaffold") is now structurally guaranteed because the endpoint
  itself doesn't exist.

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
