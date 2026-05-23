# Provenance edge — implementation plan

**Status:** **shipped in 0.2.6** (2026-05-23). Kept in-tree as the design-rationale record; for ongoing usage see [ADR-0001](../adr/0001-provenance-as-separate-edge.md), CHANGELOG `0.2.6`, and the in-code module docstrings.
**Related:** [ADR-0001](../adr/0001-provenance-as-separate-edge.md) · [CONTEXT.md](../../CONTEXT.md) (terms: **provenance**, **wikilink**)
**Target version:** `0.2.6` (additive HTTP + Storage Protocol; new LintKind is additive behavior, non-breaking to existing kinds)
**Eval baseline required:** **No** (per decision #6 below — navigation-only, does not feed retrieval or existing lint counts)

**Post-ship corrections** (codex review rounds 1–3 on the feat branch, not in this plan):

- `missing_provenance` detection compares `{normalized_key: raw_path}` dicts, not key sets — catches raw-spelling drift (e.g. `Sources/Foo.md` → `sources/foo.md`) in addition to the four key-level sub-cases (zero / partial / stale / cleared).
- `reconcile_provenance` op does **not** enter `touched_paths` in the apply loop — it doesn't change the file, so sibling `update_page` / `delete_page` ops on the same page are not falsely flagged as "superseded".
- Reverse provenance leg in `api.read_provenance` is gated on `match.layer == Layer.SOURCE` — keeps the documented "WIKI paths have empty reverse provenance" contract honest even when a malformed K-page lists another `wiki/...` path in `sources:`.
- `expected_hash` is re-checked inside `_apply_one_op` for `reconcile_provenance` (not just in preflight) — closes the TOCTOU race between preflight and apply.
- The shared `frontmatter_str_list` helper (in `wiki.py`) guards every list-of-strings frontmatter read against malformed scalars / dicts / nulls — used by `persist_wiki_page`, `run_lint`, and `MissingProvenanceFixer`.

---

## 1. Goal

Make the K-page → D-source attribution that currently lives only in `<base>/wiki/**/*.md` frontmatter (`sources:` list) into a queryable edge, so an agent can ask "which K-pages were synth-authored from this source?" over HTTP.

Forward direction (`page → its sources`) is exposed as a bonus, closing a pre-existing gap where `GET /v1/base/pages/{path}` strips frontmatter entirely.

## 2. Non-goals

- **Retrieval / RRF impact** — provenance is a navigation edge only. It does not feed `info/search.py`, does not change rank ordering, does not warrant an eval baseline (decision #6).
- **Chunk-level provenance** — the `sources:` frontmatter is page-level; we do not synthesize per-chunk attribution.
- **User-edit drift on wiki pages** — the pre-existing gap that `dikw client ingest` does not rescan `wiki/` (and so does not detect user hand-edits to wiki body or frontmatter) is **not** addressed here. The same drift exists today for the `links` table. Out of scope; tracked as a follow-up.
- **Wikilink-to-source reverse query** — already works today via `GET /v1/base/pages/{source_path}/links?direction=in` (`list_links` probes `Layer.SOURCE`). No change needed.

## 3. Design decisions (summary)

| # | Decision | Rationale / pointer |
|---|----------|---------------------|
| 1 | Two distinct edges: **provenance** (frontmatter `sources:`) and **wikilink** (body `[[…]]`). Stored separately. | [ADR-0001](../adr/0001-provenance-as-separate-edge.md); [CONTEXT.md](../../CONTEXT.md) "Edges between pages" |
| 2 | Dedicated `provenance` table, no FK, explicit cleanup. | ADR-0001 |
| 3 | Schema `provenance(src_doc_id, source_path, source_path_key)`. Raw kept for faithful display, key for robust reverse match. Mirrors `DocumentRecord.path` / `path_key`. | §5 |
| 4 | HTTP: `GET /v1/base/pages/{path}/provenance?direction=in\|out\|both` — mirrors `/links`. | §10 |
| 5 | Forward direction returns all frontmatter entries with `resolved`/dangling flag (does NOT silently drop unresolved). | §9 |
| 6 | Navigation-only scope. Does not feed retrieval, does not enter existing lint counts. No eval baseline required. | §2 |
| 7 | Reconcile inside `persist_wiki_page` next to `replace_links_from`. Frontmatter is source of truth, self-heals on every persist. | §8 |
| 8 | `delete_document` deletes provenance rows for the doc. Mirrors links cleanup. | §6 |
| 9 | CLI: `dikw client pages provenance <path> [--direction in|out|both]`, JSON default. Mirrors `pages links`. | §11 |
| 10 | Backfill for pre-existing wiki pages: new `LintKind = "missing_provenance"` + deterministic `MissingProvenanceFixer`. Walks the standard `lint propose → lint apply` flow (`dikw client lint propose --rule missing_provenance` + `dikw client lint apply <task_id>`). | §12 |

---

## 4. Implementation order (TDD)

Follow the [`feedback_tdd_discipline`](../../CONTEXT.md) rule: K-layer / Storage Protocol changes get a failing test first, then implementation. Order:

1. **§5 data model + §6 Storage Protocol** — add Protocol methods (stubbed), extend `tests/test_storage_contract.py` (failing). Local pgvector run per `feedback_run_pg_locally`.
2. **§7 sqlite + postgres adapter impls + migrations** — implement Storage methods + migrations until contract suite passes on both.
3. **§8 persist_wiki_page reconcile + delete_document cleanup** — extend `tests/test_storage_contract.py` for delete cascade; add `tests/test_persist_wiki_page.py` (or extend existing) for reconcile.
4. **§9 engine api.read_provenance** — `tests/test_api_provenance.py` against fakes, then impl.
5. **§10 HTTP route + schemas** — `tests/test_routes_pages.py` add provenance cases; impl route.
6. **§11 CLI** — `tests/test_cli_pages.py` add provenance case; impl.
7. **§12 Lint integration** — `tests/test_lint.py` + `tests/test_lint_fix.py` add missing_provenance cases; impl Fixer + LintKind.
8. **§13 docs + CHANGELOG + version bump**.

Each step is a separate commit and (preferably) a separate PR to keep review manageable. Steps 1–4 are pure engine/storage; 5–7 are surface-layer; 8 is docs.

---

## 5. Data model

### 5.1 Storage table (sqlite + postgres, identical shape)

```sql
CREATE TABLE IF NOT EXISTS provenance (
    src_doc_id      TEXT NOT NULL,    -- K-page doc_id ("wiki:<normalized_path>")
    source_path     TEXT NOT NULL,    -- frontmatter sources: entry, raw spelling
    source_path_key TEXT NOT NULL,    -- normalize_path(source_path), reverse-lookup key
    PRIMARY KEY (src_doc_id, source_path_key)
);

CREATE INDEX IF NOT EXISTS provenance_source_key ON provenance(source_path_key);
```

**No FK** on `src_doc_id` or `source_path_key`. Mirrors `links`:

- `source_path` may legitimately point to a D-layer source that is not yet indexed (user added the frontmatter manually before running ingest).
- `src_doc_id` is always a real K-page (engine never inserts otherwise), but the deliberate absence of FK + ON DELETE CASCADE keeps the cleanup contract identical to `links`: cleanup happens explicitly in `delete_document` (§6).

**PK is `(src_doc_id, source_path_key)`** — not `(src_doc_id, source_path)` — so two frontmatter entries that normalize to the same key (e.g. case drift on Windows) collapse to one row deterministically. We do not preserve duplicates.

**Index on `source_path_key`** powers the reverse lookup. Forward lookup uses the PK directly.

### 5.2 DTOs (schemas.py)

Add three new pydantic models, alongside the existing `OutgoingLink` / `IncomingLink` / `PageLinksResult` cluster:

```python
class ProvenanceSource(BaseModel):
    """One forward provenance edge: K-page → a D-source it claims (frontmatter `sources:`).

    Mirrors OutgoingLink in spirit but carries resolution status instead of link_type/anchor/line.
    `resolved=False` means source_path does not currently index to an active Layer.SOURCE document
    (the page references a source that was deleted, renamed, or never ingested) — surfaced
    faithfully so agents can detect provenance drift.
    """
    source_path: str               # as written in frontmatter (raw)
    doc_id: str | None = None      # resolved Layer.SOURCE doc_id, None if dangling
    title: str | None = None       # resolved source title, None if dangling
    resolved: bool


class DerivedPage(BaseModel):
    """One reverse provenance edge: D-source → a K-page derived from it.

    Mirrors IncomingLink. Always carries doc_id/path/title because src_doc_id always resolves
    to a real K-page (delete_document removes rows on K-page deletion, so dangling on the
    reverse side cannot occur).
    """
    doc_id: str
    path: str
    title: str | None = None


class PageProvenanceResult(BaseModel):
    """Final payload for `GET /v1/base/pages/{path}/provenance`.

    Splits the page-source attribution graph at a page boundary:
    `derived_from` is meaningful when the path is a K-page (Layer.WIKI), `derived_pages` is
    meaningful when the path is a D-source (Layer.SOURCE). For a path that resolves to
    Layer.WIKI, `derived_pages` is always empty; vice versa for Layer.SOURCE. We do not
    filter — agents can ask `direction=both` against any path; the empty list IS the answer.
    `direction=in|out|both` on the request filters which lists are populated.
    """
    path: str
    derived_from: list[ProvenanceSource] = Field(default_factory=list)   # out
    derived_pages: list[DerivedPage] = Field(default_factory=list)        # in
```

Add `ProvenanceDirection = Literal["in", "out", "both"]` next to the existing `LinkDirection`.

---

## 6. Storage Protocol changes

In `src/dikw_core/storage/base.py`, add three async methods on the `Storage` Protocol (positioned next to `replace_links_from` / `links_to` / `links_from`):

```python
async def replace_provenance_from(
    self, src_doc_id: str, source_paths: Iterable[str]
) -> None:
    """Atomic delete-then-insert: replace all provenance edges originating from src_doc_id.

    Each source_path is stored alongside its `normalize_path(source_path)` key.
    Duplicates that collapse to the same key are deduped (last one wins on raw spelling).
    No-op leading delete is fine for fresh pages.

    Mirrors `replace_links_from`. Must run in a single transaction.
    """
    ...

async def provenance_from(self, src_doc_id: str) -> list[ProvenanceEdge]:
    """All forward provenance edges from src_doc_id, in deterministic order
    (source_path_key ASC). Returns raw source_path strings (caller resolves)."""
    ...

async def provenance_to(self, source_path_key: str) -> list[ProvenanceEdge]:
    """All reverse provenance edges pointing at source_path_key. Returns src_doc_id values
    (caller resolves to documents). Caller is responsible for normalizing the input via
    `normalize_path` before calling — engine call sites already have `match.path_key`."""
    ...
```

Add a backend-neutral row type (in `schemas.py` or a sibling, mirroring `LinkRecord`):

```python
class ProvenanceEdge(BaseModel):
    src_doc_id: str
    source_path: str          # raw
    source_path_key: str      # normalized
```

`delete_document(doc_id)` already exists. Extend its contract docstring to say it also deletes provenance rows where `src_doc_id = doc_id`. (Implementation in §7.)

---

## 7. Storage adapter implementations

### 7.1 SQLite (`storage/sqlite.py`)

- `replace_provenance_from`: `BEGIN; DELETE FROM provenance WHERE src_doc_id=?; INSERT OR REPLACE INTO provenance ...; COMMIT;` using a single `executemany` for inserts. Dedupe by `source_path_key` in Python before insert (SQLite's `INSERT OR REPLACE` on the PK handles same-key collisions but we want deterministic raw-string choice — pick the first occurrence in input order).
- `provenance_from`: `SELECT source_path, source_path_key FROM provenance WHERE src_doc_id=? ORDER BY source_path_key`.
- `provenance_to`: `SELECT src_doc_id, source_path, source_path_key FROM provenance WHERE source_path_key=? ORDER BY src_doc_id`. (Index `provenance_source_key` carries this.)
- `delete_document`: add `DELETE FROM provenance WHERE src_doc_id=?` alongside the existing `links` / `chunks` / `documents_fts` / `vec_chunks_v*` deletes.

### 7.2 Postgres (`storage/postgres.py`)

- `replace_provenance_from`: same pattern, `DELETE` then `INSERT ... ON CONFLICT (src_doc_id, source_path_key) DO UPDATE SET source_path = EXCLUDED.source_path` inside one transaction. Use `executemany` via the connection.
- `provenance_from` / `provenance_to`: parameterized SELECTs.
- `delete_document`: add `DELETE FROM provenance WHERE src_doc_id=$1` alongside the existing `links` / `wisdom_evidence` deletes.

### 7.3 Migrations

Both `storage/migrations/sqlite/schema.sql` and `storage/migrations/postgres/schema.sql` get the CREATE TABLE + CREATE INDEX from §5.1 appended at the bottom under a `-- ---- Provenance (K → D edge) ----` comment. Both schemas use `IF NOT EXISTS`, consistent with the rest of the file — no version-bumped migration file needed (the migration runner is idempotent schema replay).

If the repo uses incremental migration files (verify), add `migrations/sqlite/0XX_add_provenance.sql` and `migrations/postgres/0XX_add_provenance.sql`. Otherwise the schema.sql append is sufficient.

---

## 8. `persist_wiki_page` reconcile

In `src/dikw_core/domains/knowledge/page_index.py`, inside `persist_wiki_page`, immediately after the existing `replace_links_from` block:

```python
# Reconcile provenance edges atomically from frontmatter — frontmatter is
# the source of truth (the wiki tree is a user-editable Obsidian vault),
# so re-running this on every persist self-heals when the user edits
# `sources:` directly. Mirrors the wikilink reconcile above; deliberately
# kept off the wikilink graph (separate `provenance` table, see ADR-0001)
# to keep graph-leg retrieval clean.
raw_sources = parsed.frontmatter.get("sources") or []
source_paths = [str(s) for s in raw_sources if isinstance(s, str)]
await storage.replace_provenance_from(doc_id, source_paths)
```

Notes:
- `parsed.frontmatter` is already loaded (`base.py:28`) — zero extra I/O.
- `embedder=None` (lint-apply path) is irrelevant here — provenance reconcile is provider-free, same as wikilink.
- Returns nothing structured; the function signature stays `tuple[int, str]` (unresolved wikilink count, resolved title). Provenance unresolved count is not reported here — it's a query-time concept (§9).

---

## 9. Engine API (`api.py`)

Add `api.read_provenance` mirroring `api.list_links`:

```python
async def read_provenance(
    root: str | Path | None,
    path: str,
    *,
    direction: ProvenanceDirection = "both",
    limit: int | None = None,
) -> PageProvenanceResult:
    """Return the page's provenance neighbourhood: forward sources + reverse derived pages.

    Path safety mirrors `read_page` / `list_links`: probe Layer.SOURCE then Layer.WIKI,
    require active document, else PageNotFound. Inactive/missing src docs on the reverse
    side are filtered. Forward side returns ALL frontmatter entries with resolved/dangling
    flag — provenance drift is intentionally surfaced (decision #5).
    """
```

Implementation outline:

1. Reject malformed `path` (empty, NUL) → `PageNotFound`.
2. Resolve path: probe `Layer.SOURCE` then `Layer.WIKI` via `_doc_id_for` → `storage.get_document`. Inactive → `PageNotFound`. Same pattern as `list_links` (`api.py:1898-1905`).
3. **Forward leg** (`direction in {"out", "both"}`): call `storage.provenance_from(match.doc_id)` → list of `ProvenanceEdge`. For each, probe `_doc_id_for(Layer.SOURCE, e.source_path_key)` (batched via `storage.get_documents([...])`). Build `ProvenanceSource(source_path=e.source_path, doc_id=resolved?.doc_id, title=resolved?.title, resolved=bool(resolved))`. Apply `limit` after building.
4. **Reverse leg** (`direction in {"in", "both"}`): call `storage.provenance_to(match.path_key)`. Batch-resolve `src_doc_id` → documents; filter to active; build `DerivedPage(doc_id, path, title)`. Apply `limit` after filtering.
5. Always release storage in `finally`.
6. Return `PageProvenanceResult(path=match.path, derived_from=..., derived_pages=...)`.

`limit` semantics: cap each list independently (mirror `list_links` — a hub source with many derived pages doesn't starve the forward side).

---

## 10. HTTP route

In `src/dikw_core/server/routes_pages.py`, declare BEFORE the catch-all `{path:path}` get_page handler (same ordering rule as `/links` per the existing comment at `routes_pages.py:49-51`):

```python
@router.get(
    "/base/pages/{path:path}/provenance", response_model=PageProvenanceResult
)
async def get_page_provenance(
    request: Request,
    path: str,
    direction: Annotated[ProvenanceDirection, Query()] = "both",
    limit: Annotated[int | None, Query(ge=0)] = None,
) -> PageProvenanceResult:
    rt: ServerRuntime = get_runtime(request.app)
    try:
        return await api.read_provenance(
            rt.root, path, direction=direction, limit=limit
        )
    except api.PageNotFound as e:
        raise NotFoundError(
            f"page not found: {path!r}", code="page_not_found"
        ) from e
```

No auth changes — `make_router` already wraps everything in `Depends(auth_dep)`.

---

## 11. CLI

In `src/dikw_core/client/cli_app.py`, add a `pages provenance` subcommand alongside `pages_links_cmd` (≈line 1523):

```python
@pages_app.command(
    "provenance",
    help="Show provenance edges (page sources + pages derived from a source).",
)
def pages_provenance_cmd(
    path: Annotated[str, typer.Argument(...)],
    direction: Annotated[str, typer.Option("--direction", "-d")] = "both",
    limit: Annotated[int | None, Query] = None,
    output_format: Annotated[OutputFormat, typer.Option("--format")] = OutputFormat.JSON,
    # ... standard --base / --server / --token resolution
) -> None:
    ...
```

Per [`feedback_cli_agent_first_default`](../../CONTEXT.md): default `--format` is JSON. A `table` format renders two columns (`derived_from` / `derived_pages`) with `resolved` flag rendered as ✓/✗ — keep it consistent with `pages_links_cmd`'s table mode.

Transport: reuse the existing httpx client (`request_json` or equivalent helper) — provenance is a vanilla GET, no NDJSON streaming, no progress events.

---

## 12. Lint integration (backfill path)

Per decision #10, pre-existing wiki pages whose frontmatter has `sources:` but no provenance rows are surfaced as a lint issue and backfilled via the standard fixer pipeline.

### 12.1 New LintKind

In `src/dikw_core/domains/knowledge/lint.py`:

```python
LintKind = Literal[
    "broken_wikilink",
    "orphan_page",
    "duplicate_title",
    "non_atomic_page",
    "missing_provenance",   # NEW
]
```

### 12.2 Detection in `run_lint`

Inside the existing per-page frontmatter loop (`lint.py:217+`), after `page_meta[doc.path] = PageMeta(sources=sources_tuple, ...)`:

```python
# Surface pages whose frontmatter declares sources: but whose provenance
# rows haven't been written yet (typical on bases that existed before
# the provenance feature shipped, or after a user hand-edits sources:
# without re-synth). The fix is deterministic — see MissingProvenanceFixer.
if sources_tuple:
    existing_prov = await storage.provenance_from(doc.doc_id)
    existing_keys = {e.source_path_key for e in existing_prov}
    expected_keys = {normalize_path(s) for s in sources_tuple}
    if existing_keys != expected_keys:
        report.issues.append(
            LintIssue(
                kind="missing_provenance",
                path=doc.path,
                detail=f"frontmatter declares {len(sources_tuple)} source(s); provenance table has {len(existing_prov)} row(s)",
            )
        )
```

The `existing_keys != expected_keys` comparison catches three sub-cases with one check: zero rows (never reconciled), partial rows (interrupted reconcile), and stale rows (user edited frontmatter after a prior reconcile). All resolve to the same fix.

### 12.3 `MissingProvenanceFixer`

New file `src/dikw_core/domains/knowledge/lint_fixers/missing_provenance.py`:

```python
class MissingProvenanceFixer:
    kind: LintKind = "missing_provenance"

    async def propose(
        self,
        issue: LintIssue,
        ctx: FixerContext,
        reporter: Any,
    ) -> FixProposal | None:
        # Read frontmatter sources from the live file (do not trust the lint
        # report's snapshot — file may have changed since scan).
        abs_path = (ctx.wiki_root / issue.path).resolve()
        if not abs_path.is_file():
            return None
        post = frontmatter.load(str(abs_path))
        raw = post.metadata.get("sources") or []
        sources = [str(s) for s in raw if isinstance(s, str)]

        # Resolve doc_id for the page (lint already has it but FixerContext
        # exposes pages by path, not doc_id) — derive from path via wiki_doc_id.
        doc_id = wiki_doc_id(issue.path)

        op = FixOperation(
            kind="reconcile_provenance",   # NEW op kind, see §12.4
            path=issue.path,
            extras={"doc_id": doc_id, "source_paths": sources},
        )
        return FixProposal(
            proposal_id=str(uuid.uuid4()),
            issue_kind=issue.kind,
            issue_path=issue.path,
            issue_detail=issue.detail,
            issue_line=None,
            operations=[op],
            rationale=f"reconcile {len(sources)} provenance edge(s) from frontmatter",
            source="deterministic",
        )
```

Register in the fixer registry alongside the existing four.

### 12.4 `reconcile_provenance` FixOperation

Add a new `FixOperation.kind` value `"reconcile_provenance"` to the existing literal in `lint_fix.py`. Extend `_apply_one_op` to handle it:

```python
elif op.kind == "reconcile_provenance":
    doc_id = op.extras["doc_id"]
    source_paths = op.extras["source_paths"]
    await storage.replace_provenance_from(doc_id, source_paths)
    return ApplyOutcome(...)  # success, no file changes
```

This operation does NOT modify any wiki file, does NOT call `persist_wiki_page`, does NOT touch chunks/embeddings/links. It is the narrowest possible write — one Storage Protocol call.

### 12.5 Self-disabling

Once applied, the page's `existing_keys == expected_keys` check in §12.2 passes → no more issue emitted. Per-page opt-out via `lint: {skip: [missing_provenance]}` in frontmatter works for free — `_read_lint_skip` already validates against `get_args(LintKind)` (`lint.py:125`).

---

## 13. Test plan

### 13.1 Storage contract (`tests/test_storage_contract.py`)

Add a `Provenance` test class run against both sqlite and postgres adapters (mirrors the existing structure). Cases:

1. `test_replace_provenance_from_empty_then_populate`
2. `test_replace_provenance_from_deletes_then_inserts` (atomic — verify mid-state never visible)
3. `test_replace_provenance_from_dedup_by_normalized_key` (`Sources/Foo.md` + `sources/foo.md` collapse to one row deterministically)
4. `test_provenance_from_returns_deterministic_order`
5. `test_provenance_to_returns_all_referencing_docs`
6. `test_delete_document_cascades_provenance` (both src and reverse sides)
7. `test_provenance_no_fk_allows_unindexed_source_path`

Run locally against `pgvector/pgvector:pg16` per [`feedback_run_pg_locally`](../../CONTEXT.md). CI runs both backends already.

### 13.2 `persist_wiki_page` (`tests/test_page_index.py` or extend existing)

1. `test_persist_writes_provenance_from_frontmatter`
2. `test_persist_removes_stale_provenance_when_frontmatter_changes` (atomic replace semantics)
3. `test_persist_with_no_sources_frontmatter_leaves_provenance_empty`

### 13.3 Engine API (`tests/test_api_provenance.py`, new)

Use `tests/fakes.py` storage fake. Cases:

1. `test_read_provenance_forward_marks_dangling_when_source_missing`
2. `test_read_provenance_reverse_filters_inactive_pages`
3. `test_read_provenance_both_direction_populates_both_lists`
4. `test_read_provenance_limit_caps_each_side_independently`
5. `test_read_provenance_path_not_found_raises_page_not_found`
6. `test_read_provenance_resolves_path_via_source_then_wiki_layer`

### 13.4 HTTP route (`tests/test_routes_pages.py`, extend)

1. `test_get_provenance_returns_page_provenance_result`
2. `test_get_provenance_direction_query_filter_works`
3. `test_get_provenance_unknown_path_returns_404_with_code`
4. `test_get_provenance_limit_clamps_at_zero` (negative → 422)
5. Verify route ordering — `/provenance` does not get swallowed by `{path:path}` catch-all.

Per [`feedback_pytest_windows_buffering`](../../CONTEXT.md): use targeted paths + `PYTHONUNBUFFERED=1` when running locally; do NOT use `--cov` on ASGI tests (`CliRunner` flakes with coverage).

### 13.5 CLI (`tests/test_cli_pages.py`, extend)

1. `test_pages_provenance_json_default_emits_provenance_result`
2. `test_pages_provenance_table_format_renders_resolved_flag`
3. `test_pages_provenance_direction_flag`

### 13.6 Lint + Fixer

`tests/test_lint.py`:
1. `test_run_lint_emits_missing_provenance_when_frontmatter_has_sources_but_table_empty`
2. `test_run_lint_no_missing_provenance_when_keys_match`
3. `test_run_lint_skip_frontmatter_suppresses_missing_provenance`

`tests/test_lint_fix.py`:
1. `test_missing_provenance_fixer_proposes_reconcile_op`
2. `test_apply_reconcile_provenance_writes_rows`
3. `test_apply_reconcile_provenance_does_not_modify_wiki_file` (verify file hash unchanged)
4. `test_apply_then_relint_clears_issue`

---

## 14. Migrations / version / docs

### 14.1 Version

Bump `src/dikw_core/__init__.py` `__version__` and `pyproject.toml` `version` to `0.2.6`.

Additive only:
- New HTTP endpoint (existing clients unaffected)
- New Storage Protocol methods (existing impls in tree updated together; out-of-tree storage impls would need to add them, but Protocol consumers in `dikw-core` don't break)
- New LintKind (existing kind consumers don't break; legacy bases will see new lint output on first run — call out in CHANGELOG)
- New CLI subcommand

No breaking changes, no `!` in commit subject.

### 14.2 CHANGELOG

Under `## [0.2.6]`:

```
### Added
- **provenance edge** — K-page → D-source attribution (from frontmatter `sources:`)
  is now a queryable edge in its own `provenance` storage table, exposed via
  `GET /v1/base/pages/{path}/provenance?direction=in|out|both` and the new
  `dikw client pages provenance` CLI. See [ADR-0001](docs/adr/0001-provenance-as-separate-edge.md).
- **`missing_provenance` lint** — surfaces wiki pages whose frontmatter declares
  `sources:` but whose provenance table rows are missing or stale, and a
  deterministic fixer (`source="deterministic"`, no LLM) that backfills via
  the standard `lint propose → lint apply` flow.

### Changed
- `persist_wiki_page` now reconciles provenance edges from frontmatter on every
  call, mirroring the existing wikilink reconcile. No-op when `sources:` is empty.
- `delete_document` now also deletes provenance rows where `src_doc_id` matches.

### Notes
- **Legacy bases:** first `dikw client lint` run after upgrade will report one
  `missing_provenance` issue per pre-existing wiki page that has a `sources:`
  frontmatter. Backfill via the standard two-step propose+apply flow:
  `dikw client lint propose --rule missing_provenance` (returns a `task_id`),
  then `dikw client lint apply <task_id>`. Issues stay resolved unless you
  edit frontmatter by hand.
- Provenance is a **navigation edge only**: it does not feed RRF retrieval,
  does not affect `orphan_page` / `broken_wikilink` counts, and required no
  eval baseline update.
```

### 14.3 Doc updates

- `docs/design.md` — add a short subsection under "The Four Layers" → K layer describing the provenance edge and pointing at ADR-0001.
- `docs/architecture.md` — under "Storage Protocol" add `replace_provenance_from` / `provenance_from` / `provenance_to` to the method list; under "Named seams" no change (no new seam introduced); under K-layer module listing mention `lint_fixers/missing_provenance.py`.
- `docs/getting-started.md` — one example invocation of `dikw client pages provenance` under the "navigating the wiki" section if one exists.
- `CLAUDE.md` — the "Architecture at a glance" tree gets `provenance` added to the api.py facade line; the K-layer module description in the tree picks up `missing_provenance` fixer. Already-captured terms in [CONTEXT.md](../../CONTEXT.md) do not need re-stating.

---

## 15. Out of scope / follow-ups

These are real gaps but explicitly NOT in this PR:

- **User-edit drift on wiki pages.** `dikw client ingest` does not rescan `wiki/`. If a user hand-edits a wiki page's `sources:` (or body — affects `links`) outside of synth / lint-apply, the engine does not detect the change. Existing limitation, applies equally to `links` today. Tracked separately as "wiki rescan in ingest" or "reconcile verb".
- **Provenance lint check `dangling_source`.** Surface forward-direction provenance entries that don't resolve to an active source as a hygiene issue. Cheap follow-up once provenance is in place.
- **Wisdom-layer provenance.** W items already cite K/D evidence via `wisdom_evidence` table — a separate (and pre-existing) attribution mechanism. No unification planned here.
