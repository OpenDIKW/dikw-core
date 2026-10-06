---
paths:
  - "src/dikw_core/domains/trash.py"
  - "src/dikw_core/api_delete.py"
  - "src/dikw_core/server/delete_op.py"
  - "src/dikw_core/domains/knowledge/lint_fix.py"
  - "src/dikw_core/storage/{base,sqlite,postgres}.py"
  - "tests/test_{trash,delete_page}.py"
---

# Soft delete and the trash tree

Read this rule before you change delete, trash, or `Storage.delete_document`.

- **Soft-delete via `<base>/trash/`.** The trash move is one shared, layer-agnostic primitive — `move_to_trash` (`domains/trash.py`): it purges nothing itself but `shutil.move`s a file to `<base>/trash/<rel_path>` (preserving the file's own layer prefix — `trash/sources/…`, `trash/knowledge/…`, or `trash/wisdom/…`) with a `trashed: {at, reason[, proposal_id]}` frontmatter block injected for audit (`proposal_id` omitted for a direct delete that has no proposal). Two-stage write (`.tmp` → atomic replace) + unlink-rollback so a file is never left in both places; same-second collisions get a `.<UTC-timestamp>[.NNN]` suffix (dot-prefixed counter, 000…999). Two callers, both purge storage rows **first** (`storage.delete_document(doc_id)` — documents + chunks + embeddings + outgoing links + provenance) then move the file (purge-first so a failed move leaves the file at its original path, recoverable, rather than stranding an orphaned row pointing at a missing file): (1) the lint `delete_page` fixer via `_apply_one_op` (`lint_fix.py`, K-layer only, propose/apply); (2) the immediate `delete <path>` verb (`api.delete_page` / `dikw client delete` / `POST /v1/base/delete`, spanning D/K/W — resolves the layer by storage probe, `PageNotFound` if unregistered, `trashed_to=None` when the file was already gone or vanished mid-delete, `inbound_broken` counts now-dangling referrers). Recovery: drag the file back into its tree — the `trashed:` block is harmless residue — then re-index it: a D-layer source self-heals on the next `dikw client ingest`; a restored **K/W** page is re-indexed by the `untracked_file` drift lint (see ADR-0005 — the restored file has no active row again, so `dikw client lint propose --rule untracked_file` → `lint apply` re-projects its bytes as-is, no synth), or re-author via `dikw client synth --all` / `wisdom write` for a regenerated version. `ingest` only scans `<base>/sources/`, so the `trash/` tree is naturally outside its purview. `Storage.delete_document(doc_id)` is a Protocol method (`storage/base.py`) — sqlite cleans virtual tables (`documents_fts`, `vec_chunks_v*`) explicitly because they don't honor FK cascades; postgres relies on FK cascade + explicit `links`/`provenance` deletes. Inbound edges from live pages are intentionally NOT cascade-cleaned — they surface as `broken_wikilink` on the next lint; the verb never rewrites another page.
