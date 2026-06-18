# Filesystem is the source of truth; consistency & deletion are lint + a `delete` verb

**Status**: Accepted and fully shipped. PR1 (the `delete <path>` verb), PR2 (the
`missing_file` drift `lint` kind), PR3 (the `stale_index` / `untracked_file` reindex
kinds, both fixed by one `ReindexPageFixer` emitting a `reindex_page` op), and PR4 (the
read-only `dangling_provenance` drift kind + this design's disk-authoritative invariant
written into `docs/design.md`) are all merged.

## Context

Two long-standing gaps in how the engine relates the on-disk markdown trees
(`sources/`, `knowledge/`, `wisdom/`) to the `documents` projection in storage:

1. **No filesystem↔DB consistency check or repair.** `ingest` is disk-first: it
   enumerates files on disk and upserts them, but never enumerates DB rows to compare
   against disk. A source file deleted on disk leaves its `documents` row `active=True`
   forever (orphan, undetected). Hand-edits to K/W files in Obsidian are not
   re-indexed (`ingest` scans only `sources/`; W was deliberately removed from the
   ingest scan in 0.4.0). `status`/`health`/`check` surface no drift.
2. **No first-class deletion + incomplete post-delete governance.** Deletion exists
   only as a side-effect of `lint` fixers (`orphan_page` stub / `non_atomic_page`
   split) — there is no user-facing way to delete an arbitrary page, and D/W cannot
   be deleted at all. `delete_document` clears the doc's *outgoing* links/provenance
   but intentionally leaves *inbound* edges to surface as `broken_wikilink`; with no
   reconciliation pass, repeated delete→edit cycles accumulate ghost inbound edges.

CLAUDE.md already declares **"On-disk format is the product"** and `design.md`
principles #2/#7 ("the knowledge base is the product"; "Obsidian-compatible on-disk
format … the user owns the files") establish the same — and CONTEXT.md already treats
K/W frontmatter/body as the source of truth — but the engine had no mechanism to
reconcile the DB *to* that authoritative disk. A future `dikw client reindex <path>`
was promised but never built.

## Decision

Adopt one invariant and build two complementary surfaces on top of it.

**Invariant — the filesystem is the sole source of truth; the DB is a rebuildable
projection.** Reconciliation is always disk→DB. Excluded from the "rebuildable"
promise: engine-owned state (`.dikw/`, the `knowledge_log` table, the task ledger)
and `synth`'s LLM generation (its output, once on disk, *is* disk content). A
corollary collapses the two gaps into one operation: "reindex a hand-edit" and
"delete" are both a `disk vs DB` diff-and-apply.

1. **fs↔DB reconciliation is modeled as new default `lint` kinds, not a separate
   `reconcile` verb** — reusing lint's propose/apply flow, proposal store,
   `expected_hash` concurrency gate, and `trash/` soft-delete:
   - `missing_file` (D/K/W): a `documents` row whose backing file is gone →
     `delete_document` (purge row + outgoing edges).
   - `untracked_file` (K/W): a `.md` on disk with no row → persist it
     (`persist_knowledge`/`persist_wisdom`), which **unlocks hand-written K pages**.
   - `stale_index` (K/W): `documents.hash ≠ disk hash` → re-project the current bytes
     (re-persist; never re-run `synth`, so hand-edits are preserved).
   - `dangling_provenance` (K/W): a provenance edge whose target source file is gone
     → **read-only surfacing, no fixer** (the user owns the frontmatter).

   Detection stays cheap by construction rather than via a dedicated hashing pass
   (the originally-planned mtime pre-filter proved unnecessary): `stale_index` reuses
   the per-page read the existing lexical checks already perform, comparing the
   in-hand body hash to the row — so a hand-edit is caught with zero extra I/O.
   `untracked_file` is a stat-only membership walk of the `knowledge/`+`wisdom/` trees
   (no file reads), rooted at each layer dir so the sibling `trash/`/`.dikw/`/`assets/`
   trees and `.gitkeep`/non-markdown files are naturally outside scope. `missing_file`
   iterates the active D/K/W rows and checks each backing file's existence (no tree
   walk). `untracked_file`/`stale_index` are K/W-only (D adds/edits remain `ingest`'s
   job, no overlap); `missing_file` spans all three.

2. **Deletion is a new immediate `delete <path>` verb** (`dikw client delete`,
   `POST /v1/base/delete`), spanning D/K/W: `delete_document` + a trash move to
   `<base>/trash/<layer>/<rel>` (audit frontmatter). The soft-delete primitive
   `move_to_trash` was promoted out of `lint_fix.py` into the shared, layer-agnostic
   `domains/trash.py` so D/W deletes reuse it — it was already layer-agnostic (the
   destination mirrors whatever layer prefix the input path carries), so no per-layer
   parameter is needed; the promotion just makes it public and lifts it out of the
   knowledge domain. It is **symmetric with `write_wisdom_page`** and follows the
   resulting write-form principle: **explicitly-targeted single-document writes are
   immediate** (`ingest`/`synth`/`wisdom write`/`delete`); **scan-discovered batch
   hygiene is propose/apply** (`lint`). The verb resolves which layer a path lives in
   by a storage probe (like `read_page`, but matching regardless of `active`) and
   purges rows **before** the trash move, reusing `delete_document` + `move_to_trash`
   directly rather than routing through lint's `_apply_one_op` (which is K-only
   sandboxed and carries the propose/apply `expected_hash` gate an immediate delete
   doesn't need). A row whose backing file is already gone purges cleanly with no
   trash move (`trashed_to=None`).

3. **Inbound edges are not cascade-cleaned on delete.** Inbound links from *live*
   pages stay as `broken_wikilink` (the correct signal); `missing_file` purging an
   orphan row removes its outgoing edges, so genuinely two-sided ghost edges clear
   themselves. The engine **never silently rewrites a user's body or frontmatter**.

4. **No new Storage primitives**; orphaned-asset GC is deferred to backlog.

## Consequences

- **+** Closes both gaps: orphan rows, hand-edit drift, and arbitrary/D/W deletion
  all become first-class. Supersedes the never-built `dikw client reindex`.
- **+** Hand-written K pages become a supported first-class input (any `.md` under
  `knowledge/` is authoritative content, not just `synth` output).
- **+** Ghost-edge accumulation self-heals once `missing_file` exists; no new
  edge-cleaning machinery, no new Storage methods.
- **−** `lint` grows from "is the indexed knowledge well-formed?" into also "does the
  index match the authoritative disk?", and default `lint` now walks the
  `knowledge/`+`wisdom/` trees. The added cost is small — `stale_index` rides the
  per-page read the existing checks already do, and `untracked_file` is a stat-only
  membership walk — so the sync `POST /v1/lint` stays usable (no mtime pre-filter
  needed in the end).
- **−** The drift kinds have asymmetric layer coverage (`untracked_file`/`stale_index`
  K/W-only) — the price of keeping `ingest` as the D add/edit path.

## Alternatives considered

1. **A dedicated `reconcile`/`doctor` verb** parallel to `lint`. Cleaner separation
   of "well-formed?" vs "matches disk?", but more surface for the same propose/apply
   machinery. Rejected in favour of reusing `lint` (minimum new surface).
2. **Report-only doctor** (detect, never repair). Rejected — leaves the user to fix
   drift by hand; the propose/apply fixers are the point.
3. **Auto-apply reconciliation** (`--dry-run` to opt out). Rejected — destructive by
   default; conflicts with the project's caution bias.
4. **DB-authoritative, disk as export.** Rejected — conflicts with "on-disk format is
   the product" and user-owned Obsidian vaults.
5. **Cascade-clean inbound edges / rewrite referrer bodies & frontmatter on delete.**
   Rejected — edits content the user owns; `broken_wikilink` + `dangling_provenance`
   surfacing is the honest signal (consistent with ADR-0001's non-cascade design).
6. **Make `ingest` delete-aware (subsume reconciliation).** Rejected — a routine,
   often-partial `ingest` run would silently prune rows; the destructive leg belongs
   in an opt-in, reviewed flow.

## References

- CLAUDE.md — "On-disk format is the product" (the phrase this ADR builds on).
- `docs/design.md` — principles #2/#7 (knowledge base is the product; Obsidian-compatible
  on-disk format, user owns the files); gained a "Disk is the source of truth" invariant
  section in PR4.
- `docs/architecture.md` — the promised `dikw client reindex <path>` is removed in
  PR3 (the `stale_index`/`untracked_file` reindex kinds + `lint apply` supersede it).
- ADR-0001 — provenance/links non-cascade on delete (the inbound-edge design this ADR
  builds on).
- CONTEXT.md — `drift`, the four drift `LintKind`s, the `delete` verb, and the
  `missing_file` ≠ `orphan_page` disambiguation.
