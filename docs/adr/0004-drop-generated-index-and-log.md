# Drop the generated knowledge/index.md and knowledge/log.md

**Status**: Accepted, 0.5.0.

## Context

Karpathy's "LLM Wiki" pattern — the project's founding inspiration — materialises
two files at the knowledge-base root: `index.md` (an auto-generated catalogue of
every page) and `log.md` (an append-only chronology). dikw-core inherited both:
`indexgen.regenerate_index` rewrote `knowledge/index.md` after every synth/lint,
and `log.render_log` rendered `knowledge/log.md` from the `knowledge_log` table.

Two observations:

1. **The engine never reads either file.** Synth's existing-pages awareness,
   `retrieve`, wikilink resolution, and lint are all storage- and title-index
   driven. `index.md`/`log.md` are write-only vault artifacts; nothing in the
   pipeline consumes them. The authoritative activity history is the
   `knowledge_log` *table*, of which `log.md` is only a rendered view.
2. **The configurable hierarchical taxonomy (ADR-0003) erodes their value.**
   `indexgen` grouped by a single immediate-parent folder name — wrong under an
   arbitrary-depth tree — and the folder tree itself, now mirroring the operator's
   taxonomy, *is* the catalogue Obsidian renders natively. A flat generated index
   duplicates the file tree; a generated log duplicates a DB table.

## Decision

Stop generating both files.

- Delete `domains/knowledge/indexgen.py` and its synth-time regeneration call.
- In `domains/knowledge/log.py`, **keep** the `knowledge_log` table append (the
  authoritative history) and **remove** the markdown rendering (`render_log`,
  `LOG_PATH`, the `log.md` write).
- Drop `index.md`/`log.md` from the `dikw init` scaffold and from the lint
  orphan-exclusion set. Remove `schema.log_style` (it only configured `log.md`).
- Navigation is the Obsidian file tree + `retrieve`; history is the `knowledge_log`
  table (programmatically/SQL queryable; a `dikw client log` read command is a
  possible future addition, deferred to keep this change scoped).

## Consequences

- **+** No write-only vault artifacts to keep in sync; one less thing for the
  hierarchical-taxonomy rework (`indexgen`) to relearn — it is deleted outright.
- **+** The folder tree and the `knowledge_log` table are each a single source of
  truth for their concern (catalogue vs history); no rendered duplicate to drift.
- **−** Diverges from Karpathy's index.md/log.md convention (hence design.md
  principle #3 was amended). A vault opened without dikw loses the at-a-glance
  catalogue/chronology files; users who want them can generate their own from the
  tree / DB.
- **−** History is no longer human-browsable inside the vault until/unless a
  `dikw client log` read command lands.

## Alternatives considered

1. **Keep both, rework `indexgen` for arbitrary depth.** Real effort to make
   index grouping nest correctly, for a file no engine path reads. Rejected.
2. **Keep `index.md`, drop only `log.md`.** index.md is the weaker-justified of the
   two under a folder tree that already catalogues; keeping it would mean retaining
   and depth-fixing `indexgen` for marginal value. Rejected — drop both.
3. **Drop the `knowledge_log` table too.** Loses authoritative history. Rejected —
   the table stays; only its markdown rendering is removed.

## References

- `docs/design.md` principle #3 + "On-Disk Base Layout" — amended in this change.
- ADR-0003 (configurable knowledge taxonomy) — companion K-layout change shipped together.
- CHANGELOG `[0.5.0]`.
