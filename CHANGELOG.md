# Changelog

All notable changes to `dikw-core` are tracked here. The project is
**alpha** and follows [SemVer](https://semver.org) loosely — until
1.0, breaking changes can land in any minor version. The status notes
on each entry call out exactly what shape changes break.

## Unreleased

### Fixed

- **K-layer persist is now fault-tolerant via `documents.active`, matching D
  and W.** A hard storage failure mid-`persist_knowledge` (a permanent
  `ProviderError` from inline embed, or `replace_chunks` /
  `replace_links_from` / `replace_provenance_from` raising) previously left a
  half-written knowledge page with `active=True` that still surfaced in
  retrieval, and aborted the whole synth / lint-apply run. Now both K write
  paths — synth's per-page loop and `lint apply`'s Phase 1 loop — catch the
  exception, `deactivate_document` the in-flight page (so it is hidden from
  `fts_search` / `vec_search` / the wikilink graph leg / `read_page` /
  `list_pages`), record it, and continue with the remaining pages. This is the
  same deactivate-on-failure contract D (`api.ingest`) and W
  (`write_wisdom_page`) already enforced. A transient embed retry-skip is
  **not** treated as a failure — the page stays `active=True` with
  `chunks_pending_embedding > 0` for the next ingest's resume scan.
- **A failed page invalidates its source's `synth_source_done` marker.** Synth
  writes a new `synth_source_failed` knowledge_log marker when a page in a
  source failed to persist; it invalidates any prior `synth_source_done` for
  that source (markers apply in log order, last-writer-wins), so the next
  default `dikw client synth` re-processes the source and rebuilds the page
  parked inactive — even after a `synth --all` re-synth. K has no scan-based
  reindex, so this is the recovery path for a transient failure.

### Added

- `SynthReport.persist_errors` (tuple of `PagePersistError{path, message}`) and
  `ApplyReport.persist_errors` (list of `{path, message}`) surface pages
  deactivated by a mid-pipeline persist failure. The CLI renders them as a
  `path | message` table under the synth / lint-apply report.

## 0.5.0 — configurable knowledge taxonomy + overridable prompts; drop index.md/log.md

This release generalizes the fixed K-layer classification into a
user-configurable taxonomy, opens the K-layer authoring prompts to per-base
overrides, and removes the generated `knowledge/index.md` / `knowledge/log.md`.
All three are **breaking** for any base built on ≤0.4.7 — there is no in-place
migration (rebuild-on-incompatibility policy); a base carrying the old shape
trips `BaseUpgradeRequired` with the exact rebuild command. See
[ADR-0003](docs/adr/0003-configurable-knowledge-taxonomy.md) and
[ADR-0004](docs/adr/0004-drop-generated-index-and-log.md).

### Breaking

- **K-page classification is now a configurable, hierarchical `category` tree.**
  The fixed single-axis `type` (`entity`/`concept`/`note`) filed pages under
  pluralized folders (`knowledge/concepts/foo.md`). It is replaced by
  `schema.categories` in `dikw.yml` — a **closed set** of declared `path`
  (+ optional `desc`) entries of arbitrary depth, e.g. `产品/移动端`,
  `技术/架构`. A page filed under `技术/架构` lands at
  `knowledge/技术/架构/<slug>.md` with `category: 技术/架构` in its frontmatter.
  `entity`/`concept`/`note` is now merely the *default* taxonomy a fresh base
  ships with (folders are **singular**: `knowledge/entity/`, not `entities/`).
  Synth's LLM emits `<page category="…" slug="…">` (no `path=`/`type=`) and may
  only file under a declared `path`; anything it can't place lands in
  `schema.fallback` (default `未分类/`) and is flagged by the new
  `uncategorized` lint kind. Karpathy's rule: a wrong category is a cheap
  re-file, an invented folder is irreversible drift — the parser refuses an
  out-of-set category, and the brittle 0.4.6 "missing `knowledge/` prefix"
  path-recovery normalizer is **removed** (the engine now owns path
  construction entirely, so there is no model-supplied path to recover).
  Removed: `type:` frontmatter key, `SchemaConfig.page_types`,
  `SchemaConfig.log_style`, `type_to_folder`/`type_from_path` (→
  `category_from_path`).
- **K-layer authoring prompts are overridable per base.** `synthesize` (via
  `synth.prompt_path`; also used by the `non_atomic_page` fixer),
  `lint_fix_orphan_merge` and `lint_fix_broken_wikilink_grounded` (via
  `lint.fixer_prompts.{orphan_merge,broken_wikilink}`) can point at your own
  markdown under `<base>/prompts/`. `prompts.resolve` enforces that the override
  stays inside the base and carries the engine's required `{placeholders}` +
  `<page …>` output markers (declared in `prompts/_contract.py`), failing fast
  at load **and** surfacing via `dikw client check` (new `prompts` leg on the
  check report). New config: `synth.prompt_path`, the `lint:` block with
  `lint.fixer_prompts`.
- **`knowledge/index.md` and `knowledge/log.md` are no longer generated.** The
  category folder tree is the catalogue (browse it in Obsidian + `retrieve`),
  and the `knowledge_log` storage table (readable via `list_knowledge_log`)
  remains the authoritative history — only the markdown render views are gone.
  `dikw init` no longer scaffolds them (nor the pluralized type folders); it
  now creates `knowledge/<category>/.gitkeep` for the default taxonomy plus a
  `prompts/.gitkeep`. The `indexgen.py` and `log.py` modules are deleted.

### Migration

- Rebuild on upgrade: write your `schema.categories` into `dikw.yml`, move the
  old `knowledge/` (and `.dikw/`) tree aside, and re-run `dikw client synth` to
  re-author pages under the new taxonomy. `BaseUpgradeRequired` fires on a base
  that still carries `schema.page_types` (or the legacy `wiki/` layout) and
  prints the command.
- If you have a local `evals/.cache` snapshot built before this release, clear
  it (or run evals with `cache_mode="rebuild"`): a cached snapshot's `dikw.yml`
  carries the old `page_types` shape and would otherwise trip
  `BaseUpgradeRequired` on reuse. CI builds snapshots cold, so this only affects
  local dev caches.

## 0.4.7 — ingest source-path containment

### Security

- **Source discovery refuses a `sources[].path` that escapes the base.**
  `iter_source_files` (the D-layer ingest scan entry) now validates every
  configured source root up front and raises if one resolves outside the base
  — a `../` prefix or an absolute path elsewhere. `sources` is a managed tree
  under the base, so an escaping entry is a config error, not a license to read
  + index arbitrary files into the `Layer.SOURCE` index (whose doc-ids would
  also degrade to absolute paths). Fails the whole ingest with a clear message
  before any file is read (no partial index). Completes the K/W write-sink
  containment shipped in 0.4.6 — the last guarded-vs-unguarded asymmetry in the
  path-handling surface.

## 0.4.6 — ingest mtime fallback + synth path normalization + write-sink containment

### Fixed

- **Synth no longer drops a knowledge page when the model omits the
  `knowledge/` path prefix.** A `<page>` whose `path` uses a valid type
  folder but forgets the layer prefix (`entities/foo.md` instead of
  `knowledge/entities/foo.md` — common with smaller / open-weight models
  that follow the type-folder convention but drop the parent) used to raise
  `SynthesisError`; via per-group failure (#134) that discarded *every*
  page in the group and left the source un-marked, so the task reported
  success while the knowledge base was silently partial. The parser now
  normalizes a recognized type folder (built-in `entities`/`concepts`/`notes`
  **or** a custom `SchemaConfig.page_types` folder) by prepending
  `knowledge/`, preserving the model's own slug — the pinyin/ASCII spelling
  the prompt asks for and `slugify` would otherwise collapse to `untitled`
  for CJK titles. A stale pre-0.4.0 `wiki/` prefix or a truly unrecognized
  head still raises. The same parse-time guard now also rejects a `..`
  traversal segment (`entities/../…`, an already-prefixed `knowledge/../…`,
  or one hidden behind Windows `\` separators) and a bare type folder with no
  filename, so a prompt-injected source document can't steer synth into
  writing a page outside the base. (#146)
- **Source docs imported via a byte-stable tarball no longer store
  `mtime=0`.** dikw-web zeroes the tar `mtime` field so identical bytes
  dedup to one `package_sha256`; the extracted file landed with
  `st_mtime == 0`, every client rendered it as `1970-01-01`, and the
  constant `0` gave the graph change-hash (`api_graph.py`) no entropy
  across re-imports.
  `ingest` now falls back to ingest wall-clock (`time.time()`) when a
  source file carries no usable mtime (`<= 0`); an unchanged re-persist
  (e.g. an image-bearing doc, same body hash) keeps its already-stored
  timestamp so it doesn't flap the change-hash, while a genuine content
  change still advances it. Legacy rows already stored
  with `mtime=0` self-heal on the next `dikw client ingest`: a broken
  stored mtime now also forces one re-persist (alongside hash drift /
  `active=False` / asset refs). D-layer only — knowledge / wisdom pages
  are engine-written with a real mtime and were never affected. (#145)

### Security

- **K/W page writes now refuse a path that escapes the base.** The
  knowledge write sink (`write_page`) and the shared persist leg behind
  `persist_knowledge` / `persist_wisdom` (`_persist_layered_page`) now
  resolve `root / path` and reject it — before any `mkdir` / `write_text`
  or `parse_any` read — when it resolves outside the base (a `..`
  traversal segment or an absolute path). Previously only the read leg
  (`api_path_safety._assert_within`) and the wisdom file writer were
  guarded; these two K-side sinks were not, so a caller that bypassed the
  synth parser (#146/#149 closed that one vector) could steer a write — or
  a read-into-index — outside the base. The guard uses `Path.relative_to`
  on the *resolved* path, so it is platform-correct without a lexical
  backslash special-case. Follow-up to #149.

## 0.4.5 — api facade decomposition + dead-code cleanup

### Removed

- **Deprecated persist aliases dropped.** `persist_page`,
  `persist_knowledge_page`, and `wiki_doc_id` — the
  `tuple[int, str]`-returning compatibility shims kept through 0.4.0 —
  are gone. Call `persist_knowledge` / `persist_wisdom` /
  `persist_source` and `doc_id_for(Layer.KNOWLEDGE, …)` directly.
- **Orphaned eval `--dump-raw` plumbing removed.** `run_eval`'s
  `raw_dump_path` parameter and its `_dump_raw_ranked` JSONL writer had
  no CLI surface after the client/server split; both are deleted.
  `evals/tools/sweep_rrf.py` stays as a manual offline RRF-sweep tool —
  prepare its input JSONL by hand.

### Internal

- **`api.py` decomposed into per-verb cluster modules.** The 3.9k-line
  engine facade is now a ~170-line thin re-export surface; each verb lives
  in a focused `api_*` module (`api_core`, `api_types`, `api_health`,
  `api_ingest`, `api_pages`, `api_graph`, `api_retrieve`, `api_synth`,
  `api_lint`, `api_wisdom`, `api_path_safety`). The public `api.X` surface
  and `__all__` are byte-identical, so server routes and the eval runner
  are unaffected. Contributors add a verb's body to its cluster module and
  re-export it from `api.py`. Note for test authors: a function resolves
  its bare global names against the module it is *defined* in, so
  `monkeypatch.setattr` targets move with the verb (e.g. `ingest`'s
  `_with_storage` is now patched on `api_ingest`, not `api`).
- **K/W persist leg enforces `title_to_path`/`fuzzy_index` pairing.** When
  `_persist_layered_page` rebuilds `title_to_path` from storage (caller passed
  `None`), it now rebuilds `fuzzy_index` from the same title set via
  `build_title_indexes` and discards any caller-supplied `fuzzy_index`,
  closing a latent footgun where a stale fuzzy index resolved wikilinks
  against a different key space than the fresh exact index. Behavior-preserving
  for every production caller (synth, `lint apply`, `write_wisdom_page`), which
  already pass the two indexes as a matched pair.

## 0.4.0 — chunk→FTS→embed pipeline governance + ingest scope narrowing

### ⚠️ Breaking

- **`dikw client ingest` no longer scans `<base>/wisdom/`.** Wisdom
  pages are indexed exclusively when written via `dikw client wisdom
  write` (CLI) or `POST /v1/base/wisdom` (HTTP). Hand-edits to wisdom
  files in Obsidian are no longer auto-reindexed — the same
  limitation already applied to knowledge pages (a future `dikw
  client reindex <path>` will close this gap symmetrically). The old
  scan loop and its legacy aggregate-file skip-list have been
  removed; obsolete tests have been dropped, and the `ingest_wisdom_files`
  helper in `tests/fakes.py` lets test authors seed wisdom rows via
  the per-file `persist_wisdom` path.

### Added

- **Per-batch embed retry-skip.** `consume_embedding_stream` now
  catches `ProviderError` per batch and retries up to
  `cfg.provider.embedding_error_retries` (default 2) with
  `embedding_error_retry_backoff_seconds` (default 2.0s) linear
  backoff before skipping the batch and moving on. Skipped chunks
  remain in storage without vectors and the next ingest's
  missing-embedding resume scan reconciles them — synth /
  `lint apply` / `wisdom write` / ingest's bulk pass are all
  durable through transient embedding-provider failures now.
- **`lint apply` inline-embeds when an embedder is configured.**
  Setting `DIKW_EMBEDDING_API_KEY` makes `dikw client lint apply`
  re-embed every rebuilt page in the same pass so the fix is
  retrievable on return. Without the key, behaviour is unchanged:
  every chunk falls into `chunks_pending_embedding` and the next
  ingest's resume scan picks them up. `ApplyReport` gains
  `chunks_embedded` and `chunks_pending_embedding`; the CLI summary
  prints both.

### Changed

- **Refactor: `persist_page` split into three layer-specific
  functions.** `persist_source` (D, `domains/data/persist.py`),
  `persist_knowledge` (K, `domains/knowledge/page_index.py`), and
  `persist_wisdom` (W, `domains/wisdom/persist.py`) each own their
  layer's full upsert + chunk + FTS + (optional inline embed) +
  links + provenance pipeline. The legacy `persist_page` and
  `persist_knowledge_page` symbols remain as deprecated aliases
  returning the old `tuple[int, str]` shape; they will be removed
  in a follow-up.
- `api.ingest` is now D-layer-only plus the cross-layer
  missing-embedding resume scan that reconciles deferred chunks
  from D / K / W.

### Fixed

- **Synth's per-group retry-skip now catches only
  `TransientProviderError`** — symmetric with the 0.4.0 embed-batch
  retry change. Without this, a permanent LLM `ProviderError` (typo
  in `cfg.provider.llm_model`, missing key, invalid model id) was
  silently retried-then-skipped, producing "synth succeeded with
  0 pages" runs instead of failing fast. The `openai_codex`
  reducer-bug path (issue #134 / #135) now raises
  `TransientProviderError` so synth's narrowed retry still catches
  it.
- **`embed_assets` gained per-batch retry-skip** — symmetric with
  `embed_chunks` / `embed_chunks_multimodal`. Without it, a single
  5xx / timeout mid-pass aborted the whole asset embedding run; the
  retry-then-skip path now persists prior batches and the resume
  scan reconciles missing asset vectors on the next ingest. Both
  retries pass `cfg.provider.embedding_error_retries` and backoff.
- **`lint apply` now builds and threads `fuzzy_index` through
  `persist_knowledge` and Phase 2 referrer reconciliation** —
  without it, fuzzy-resolvable wikilinks like `[[Neural Networks]] →
  Neural Network` silently broke inside lint apply and the next
  lint propose flagged them as `broken_wikilink`, causing churn.
- **`WisdomWriteReport` gained `chunks_pending_embedding`** — fully
  symmetric with `ApplyReport`. Non-zero values surface when
  `no_embed=True`, when the inline-embed path defers (no active
  text version yet or `cfg.provider` drift), or when transient
  embed failures exhaust the retry budget. CLI render prints
  "N pending — next dikw client ingest reconciles them" mirroring
  the lint apply message.
- **`lint apply` / `wisdom write` no longer flip the active text
  embed version, defer inline embed on cfg drift, and preflight the
  embedder before mutating files.** Both paths now reuse the active
  version returned by `storage.get_active_embed_version("text")`
  instead of registering-and-activating a new identity from
  `cfg.provider`. Activating here would have stranded every other
  vector in the now-inactive table and gutted dense retrieval until
  the next full ingest. **Full-identity drift detection**: when the
  active version's `(provider, model, revision, dim, normalize,
  distance)` differs from `cfg.provider` (the user edited
  `dikw.yml` between full ingests), inline embed is deferred —
  otherwise the cfg-built embedder would produce different-dim or
  different-space vectors that get stored under the old version's
  table (silent corruption, or a hard StorageError mid-persist
  after files were already mutated). **Preflight**: each call also
  performs a one-token embed call before mutating any files, so a
  permanent provider error (bad API key, 401, invalid model id)
  surfaces while state is still clean instead of after Phase 0 has
  rewritten / deleted files. When no active version exists yet
  (fresh base), inline embed is deferred to the next ingest's
  resume scan, which goes through the full register-and-activate
  path. Mirrors `synthesize`'s long-standing reuse pattern.
- **Embedding provider errors classified as transient vs. permanent.**
  `OpenAICompatEmbeddings.embed` and `GiteeMultimodalEmbedding.embed`
  now classify exceptions into `TransientProviderError` (retryable:
  timeouts, rate limits, 5xx, 408/429, connect drops, parse failures)
  vs. plain `ProviderError` (permanent: 401, 403, 404, invalid model
  id, missing API key). The per-batch retry-skip in
  `consume_embedding_stream` retries only `TransientProviderError`;
  permanent errors propagate so misconfig fails fast instead of being
  silently retried-then-skipped (a single missing/wrong API key would
  otherwise have produced "success, 0 vectors" runs).
- **Synth forwards `cfg.provider.embedding_error_retries` to
  `persist_knowledge`.** The K-layer inline embed inside synthesize
  was silently using `retries=0` regardless of the configured
  retry budget; ingest, lint apply, and wisdom write already
  forwarded it.

### Known limitations

- **W→W forward-ref wikilinks**: a wisdom page `A` written before
  its `[[B]]` target wisdom page exists keeps the link unresolved
  in storage until `A` is re-written via `dikw client wisdom write`.
  Symmetric to the long-standing K→K limitation — synth doesn't
  re-resolve K-page links when a later page introduces the target
  either. A future `dikw client reindex <path>` will close both
  gaps; until then, re-writing the referring page is the
  user-facing workaround.

## 0.4.0 — BREAKING term rename: K layer "wiki" → "knowledge"

⚠️ **Breaking change for every existing base.** The K-layer
on-disk directory, the `Layer` enum value, the `wiki_log` SQL
table, and every `wiki_*` API symbol have been renamed to use
`knowledge` consistently. Bases initialised by ≤0.3.6 are **not
readable** by 0.4.0; no in-place migration is provided.

**Manual upgrade for an existing base:**

```bash
cd <base>
mv wiki knowledge       # rename the K-layer directory
rm -rf .dikw            # drop the SQLite + auth + task ledger
# (dikw.yml stays — your existing config is reused)
dikw serve --base . &   # start the server
dikw client ingest      # reindex sources + knowledge pages
```

Opening a 0.3.x base with 0.4.0 raises `BaseUpgradeRequired`
with the exact command above; the server refuses to start until
the rename is done.

### Why

`wiki` was historically overloaded — it named the K-layer DIKW
role, the on-disk directory, and the `[[wikilink]]` body
syntax. CONTEXT.md called this out as a long-standing
ambiguity. In 0.4.0 the term is reserved exclusively for
**wikilink** syntax; everything else uses `knowledge`. This
removes a recurring source of confusion in LLM prompts, public
APIs, and new-contributor onboarding.

### Renames

- **On disk:** `<base>/wiki/` → `<base>/knowledge/`,
  `<base>/trash/wiki/` → `<base>/trash/knowledge/`,
  `wiki/index.md` → `knowledge/index.md`,
  `wiki/log.md` → `knowledge/log.md`.
- **Layer enum / DB:** `Layer.WIKI` → `Layer.KNOWLEDGE`
  (string value `'wiki'` → `'knowledge'`); SQL table `wiki_log`
  → `knowledge_log`; storage CHECK constraint values updated.
- **Engine API:** `WikiPage` → `KnowledgePage`,
  `WikiPageMeta` → `KnowledgePageMeta`,
  `WikiLogEntry` → `KnowledgeLogEntry`;
  `persist_wiki_page` → `persist_knowledge_page`;
  `init_wiki` / `load_wiki` / `resolve_wiki_root` →
  `init_base` / `load_base` / `resolve_base_root`;
  module `domains/knowledge/wiki.py` → `domains/knowledge/page.py`.
- **Field rename:** `SynthReport.wiki_pages` →
  `knowledge_pages`; progress `wiki_pages_changed` →
  `knowledge_pages_changed`; `StorageCounts.last_wiki_log_ts`
  → `last_knowledge_log_ts`.
- **HTTP / CLI:** response field `"wiki_root"` →
  `"base_root"`; `dikw auth ... --wiki, -w` →
  `--base, -b` (consistent with `dikw serve --base`).
- **Storage protocol:** `append_wiki_log` /
  `list_wiki_log` → `append_knowledge_log` /
  `list_knowledge_log`.
- **Runtime:** `<base>/.dikw/wiki_id` → `<base>/.dikw/base_id`;
  env var `DIKW_WIKI_INSTANCE_ID` → `DIKW_BASE_INSTANCE_ID`.

### Preserved (still spelled "wiki")

The `[[wikilink]]` syntax keeps its name everywhere:
- `LinkType.WIKILINK`, regex `WIKILINK_RE`, lint kind
  `broken_wikilink`, prompt file
  `lint_fix_broken_wikilink_grounded.md`,
  `SynthReport.unresolved_wikilinks`,
  storage method `replace_links_from`, and every
  docstring/comment that talks about `[[Target]]` resolution.

### Added — synth retries `ProviderError` per group, then skips

Fixes [#134]. A single `ProviderError` raised by `llm.complete` for
one source group used to abort the whole `synth` task — prior groups'
work was lost and later sources never ran. The canonical trigger was
the `openai_codex` empty-streaming-response edge case (`response.
output=None` + zero text deltas), but any `ProviderError` (auth flap,
quota, refusal) hit the same dead-end.

`_synth_pages_from_source` now wraps `llm.complete` in a bounded
per-group retry loop: up to `cfg.synth.provider_error_retries`
retries (default `2`, so `3` attempts total) with linear backoff
(`provider_error_retry_backoff_seconds`, default `2.0` → `2s`/`4s`).
After the retries are exhausted the group is recorded as a parse-
style error and **skipped**; subsequent groups in the same source
and later sources continue to process. The skip is counted in
`outcome.parse_errors`, so `synth_source_done` is **not** written
and re-running `synth` retries the flaky group.

Per-group NDJSON progress events surface the new states:
`status="retrying"` (one per failed attempt that still has retries
left, carries `attempt` / `max_attempts` / `error_kind` /
`error_msg`) and `status="skipped"` (terminal, carries
`reason="provider_error"` / `attempts` / `error_kind` /
`error_msg`). Set `provider_error_retries: 0` for one-attempt-no-retry
(the group is still skipped, not re-raised — by design, since the
whole point of the fix is that a single bad group must not abort the
task).

[#134]: https://github.com/OpenDIKW/dikw-core/issues/134

### Added — `read_page` surfaces parsed frontmatter

`api.read_page` and `GET /v1/base/pages/{path}` now return the parsed
YAML front-matter as a `frontmatter: dict[str, Any]` on
`PageReadResult`. Callers that previously had to re-read the file
themselves to look up `tags`, `sources`, `status`, or custom keys can
now read them off the same response. The field is `{}` when the file
has no front-matter or when the YAML failed to parse externally
(parse-failure also still emits `anchors=[]`, so the empty-anchors
signal disambiguates "no metadata" from "broken file"). Backwards
compatible: existing clients ignore the new field.

## 0.3.6 — 2026-05-28

- **Fixed:** `openai_codex` survives the ChatGPT codex backend's `response.output = None` reducer bug — falls back to locally-collected delta text and raises `ProviderError` on a zero-delta hit so synth never silently drops a page (#129).

## 0.3.5 — 2026-05-27

- **Added: wisdom write surface.** `api.write_wisdom_page` / `POST /v1/base/wisdom` / `dikw client wisdom write` create or update a single `wisdom/[<author>/]<slug>.md` and index it inline (chunks + FTS + embeddings + wikilinks + provenance) without a full `ingest`. Upsert on `(author, slug)`; `--no-embed` defers embedding; empty body is rejected (422). Reads still go through `dikw client pages get wisdom/...` (#126).

## 0.3.0 — 2026-05-26

- **BREAKING: Wisdom layer reshaped** from an LLM-distilled candidate/review pipeline into a hand-written first-class document layer under `wisdom/<author>/<slug>.md`, sharing the K-layer `documents` / `chunks` / `links` / `provenance` shape (4-PR arc #120–#123). Removed the `distill` and `review {list,approve,reject}` verbs, `POST /v1/distill`, `GET /v1/wisdom`, the `wisdom_items` / `wisdom_evidence` / `wisdom_embed_meta` / `vec_wisdom_v<n>` tables, the `WisdomKind` taxonomy, and the `provider.llm_max_tokens_distill` / `schema.wisdom_kinds` config keys. Added the wisdom-only `documents.status` column (`draft | published | favorite | archived`) and the `invalid_wisdom_status` lint kind. Wisdom now participates in retrieve (`Hit.layer == "wisdom"`), the page read APIs, and the `broken_wikilink` / `orphan_page` / `missing_provenance` lint scan. `SCHEMA_VERSION` bumped 3 → 5 (rebuild required). Rationale: `docs/adr/0002-wisdom-as-first-class-documents.md`.
- **Changed:** project status pre-alpha → alpha; `examples/docker/Dockerfile` `DIKW_VERSION` now auto-syncs via a post-publish `sync-dockerfile` PR plus a `dockerfile-version-guard` CI check (#118, #119).

## 0.2.7 — 2026-05-24

- **Docs/tests:** swept stale `--enable-llm` copy that still described the pre-#83 "TODO-stub" fixer behaviour (it became an evidence-backed grounded repair in #83) and added the missing `orphan_page` mentions across CLI / OpenAPI / engine docstrings. No runtime change.

## 0.2.6 — 2026-05-23

- **Added: provenance edge** — each K-page's `sources:` frontmatter is now a queryable page → D-source attribution edge, distinct from body `[[wikilinks]]`: new `provenance` table, `replace_provenance_from` / `provenance_from` / `provenance_to` Storage methods, `GET /v1/base/pages/{path}/provenance`, and `dikw client pages provenance`. Rationale: `docs/adr/0001-provenance-as-separate-edge.md`.
- **Added:** `missing_provenance` lint kind + deterministic `MissingProvenanceFixer` to backfill the table from frontmatter (no LLM).
- **Changed (storage):** `SCHEMA_VERSION` → 3 (rebuild required).

## 0.2.5 — 2026-05-21

- **BREAKING (CLI):** agent-first default-JSON audit completed — `lint`, `lint proposals`, `tasks list`, `import` (and the then-extant `review` commands) default to JSON; humans opt into `--format table` / `--pretty`.

## 0.2.0 — 2026-05-19

- **BREAKING (HTTP):** `GET /v1/tasks` returns a `TaskListPage` envelope (`{tasks, next_cursor, has_more}`) with **summary** rows (no `result` / `error`) instead of a bare `list[TaskRow]`; full detail via `GET /v1/tasks/{id}` or `.../result`. Added keyset cursor pagination (`?cursor=`, `?limit=`, composing with `?status=` / `?op=`); `dikw client tasks list` gained `--all` / `--cursor`.
- **Fixed:** `dikw serve --help` no longer claims NDJSON for task events (they are cursor JSON + long-poll; only `POST /v1/retrieve` streams NDJSON).

## 0.1.0 — 2026-05-18

First tagged release; consolidates the client/server architecture and the agent-first kernel surface built during pre-release development.

- **BREAKING (architecture):** the in-process CLI was replaced by a long-lived `dikw serve` (FastAPI) server + a `dikw client` HTTP client.
- **BREAKING (CLI):** top-level short names for HTTP commands removed — every HTTP-bound verb lives exclusively under `dikw client <verb>` (no `dikw status` / `dikw retrieve` / `dikw serve-and-run` aliases). The four local-only surfaces (`dikw version`, `dikw init`, `dikw serve`, `dikw auth …`) are unchanged.
- **BREAKING:** in-engine answer synthesis removed — `POST /v1/query`, `dikw client query`, the `QueryResult` / `Citation` DTOs, and `provider.llm_max_tokens_query` are gone. `retrieve` returns ranked chunks + page refs; the agent layer runs its own LLM.
- **BREAKING (CLI + HTTP):** `dikw client init` and `POST /v1/init` removed — the server refuses to start without a `dikw.yml`, so local scaffolding is `dikw init <path>` only.
- **BREAKING:** source-import verb renamed `upload` → `import` and decoupled from `ingest` (commit bytes into `sources/` vs chunk + embed); the `--wiki` flag renamed `--base`.
- **Added:** `GET /v1/base/graph` (+ `dikw client graph get`) — the full base graph (nodes + edges + unresolved wikilinks) in one read-only request (#89). `GET /v1/base/pages/{path}/links` exposes the K-layer link graph at a page boundary.
- **Fixed:** `broken_wikilink --enable-llm` became an evidence-backed grounded repair (D-layer hybrid search gates the LLM; `TODO` / `stub page` / `placeholder` / sub-200-char outputs rejected) instead of fabricating TODO stubs (#83).
- **Changed:** `synth` preserves the source's dominant language in K-layer pages; `dikw client status` / `check` default to JSON.
