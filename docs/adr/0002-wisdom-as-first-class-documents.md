# Wisdom layer is hand-written first-class documents

**Status**: Accepted, 0.3.0 (PR1 #120, PR2 #121, PR3 #122, PR4 closes the arc).
Amended in 0.4.0 — the W-layer trigger moved off the `dikw client ingest`
scan and onto the dedicated `api.write_wisdom_page` entry (CLI
`dikw client wisdom write` / HTTP `POST /v1/base/wisdom`). On-disk
shape, schema, retrieve / lint / read APIs, and the W↔K link
symmetry are unchanged — only the entry point that drives the
`persist_wisdom` (formerly `persist_page(layer=Layer.WISDOM)`)
pipeline shifted. See CHANGELOG `[0.4.0]` and the 0.4.0 ⚠️ Breaking
entry above it.

## Context

The 0.2.x design treated W (Wisdom) as an LLM-distilled candidate / human-review pipeline: an `api.distill` verb invoked the `distill.md` prompt to propose candidate wisdom items (`principle | lesson | pattern`) from K-layer content, wrote them as `wisdom/_candidates/<kind>-<slug>.md` plus rows in dedicated `wisdom_items` / `wisdom_evidence` / `wisdom_embed_meta` / `vec_wisdom_v<n>` tables, and required `dikw client review approve|reject` to flip the candidate status before a rendered page landed in one of three aggregate files (`wisdom/{principles,lessons,patterns}.md`). The "≥ 2 evidence rows" gate was a hard invariant; agents consumed approved wisdom via a dedicated `GET /v1/wisdom/applicable?q=...` endpoint.

In practice that shape was the wrong product. Three independent signals:

1. **Wrong author.** The user already writes wisdom by hand — first-principles notes, lessons learned, durable opinions — and wants them to participate in retrieve / lint / link graph alongside knowledge pages. The candidate / review queue was overhead for the system designer, not value for the writer. Approved candidates were rare because the LLM proposed too aggressively and the user rejected most.
2. **Off the main pipeline.** Wisdom lived in dedicated tables (`wisdom_items` + `wisdom_evidence` + `wisdom_embed_meta` + `vec_wisdom_v<n>`) parallel to but disjoint from the `documents` / `chunks` / `links` / `provenance` shape K used. That meant separate retrieve / lint / link-graph code paths, separate embedding spend, and a `Hit.layer == "wisdom"` value retrieve never returned because there was no chunk-level wisdom index.
3. **Off-disk format.** `wisdom/_candidates/` and the three aggregate files (`principles.md` / `lessons.md` / `patterns.md`) violated the "the knowledge base is the product" invariant — they were synthetic engine artifacts a human author would not naturally write or organize, and they couldn't be hand-edited without the engine clobbering the changes on the next `distill` pass.

## Decision

Reset W into a layer **structurally symmetric to K**:

- Wisdom is a plain markdown file under `wisdom/<author>/<slug>.md`. The directory name is the author — no frontmatter `author` field. A file directly under `wisdom/<slug>.md` (no author subdirectory) is also indexed, with `author = None`.
- `dikw ingest` scans `<root>/wisdom/**/*.md` after the `sources:` scan, hash-idempotent, and runs each file through the same `_persist_layered_page` pipeline as knowledge pages: `documents` row at `Layer.WISDOM`, chunks, per-version embeddings, outgoing `[[wikilinks]]`, and `provenance` edges from the page's `sources:` frontmatter. Wisdom chunks land in the same `vec_chunks_v<id>` and FTS tables as knowledge / source chunks; there is no separate wisdom vec table.
- `dikw client retrieve` returns wisdom chunks tagged `Hit.layer == "wisdom"` alongside knowledge / source hits. Callers group, weight, or cite by layer in their own assembly step. `read_page` / `list_links` / `read_provenance` HTTP APIs accept wisdom paths and resolve cross-layer edges (wisdom→knowledge, knowledge→wisdom, wisdom→source) symmetrically.
- `broken_wikilink` / `orphan_page` / `missing_provenance` / `duplicate_title` lint scans the unified WIKI + WISDOM page set. The orphan inbound counter credits cross-layer wikilinks so a knowledge page cited only from wisdom is not falsely flagged.
- One wisdom-specific column: `documents.status` (CHECK-constrained to `draft | published | favorite | archived`, NULL otherwise). Knowledge / source rows are clamped to `status = NULL` at both the application layer (`api._to_document`, `_persist_layered_page`) and the storage adapter (`upsert_document` in both sqlite and postgres). Lint kind `invalid_wisdom_status` warns when frontmatter declares an out-of-enum value; ingest stays non-blocking.
- **No LLM authoring path.** No `distill` verb, no candidate queue, no `≥ N evidence` gate, no `kind` taxonomy, no review state machine. The engine writes K via `synth`; the user writes W by hand. Karpathy's rule applied: deterministic scoping (which wisdom pages exist, what they link to, what they cite as sources) feeds probabilistic reasoning (the agent's LLM, which now also sees wisdom hits in retrieve).

## Consequences

- **+** One persist pipeline (`_persist_layered_page`), one retrieve pipeline (`HybridSearcher.search`), one lint pass (`run_lint`), one set of read APIs — wisdom and knowledge share semantics end to end.
- **+** Wisdom is searchable. A user who writes `wisdom/elon-musk/first-principles.md` gets it back from `retrieve "first principles"` tagged `layer=wisdom`. Previously it was invisible to the engine until approved through the review pipeline.
- **+** Wisdom is `[[wikilinkable]]` to and from knowledge pages. Cross-layer same-title collisions stay broken so `lint` surfaces the ambiguity (Karpathy's wrong-merge rule).
- **+** On-disk format honors the open-format promise (the knowledge base is the product). Authors organize by author directory rather than by engine-imposed kind taxonomy.
- **+** Schema slimmed: 4 dedicated wisdom tables removed (`wisdom_items` / `wisdom_evidence` / `wisdom_embed_meta` / `vec_wisdom_v<n>`), 8 Storage Protocol methods removed, 5 CLI subcommands removed (`distill`, `review {list,approve,reject}`), 4 HTTP routes removed (`POST /v1/distill`, `GET /v1/wisdom`, `POST /v1/wisdom/{id}/approve`, `POST /v1/wisdom/{id}/reject`), 6 schema DTOs removed (`WisdomKind`, `WisdomStatus`-old, `WisdomItem`, `WisdomEvidence`, `WisdomEmbeddingRow`, `WisdomVecHit`). `documents.status` adds one column.
- **−** The "≥ 2 evidence" hard invariant is gone. Provenance is still tracked via the `sources:` frontmatter → `provenance` table edge, but the engine no longer enforces a minimum source count. Users (or agent layers) who need that invariant can implement it as a custom lint rule.
- **−** No engine-side entry point for an agent to *propose* wisdom. An agent that wants to write wisdom does so by writing `wisdom/agent/<slug>.md` directly into the tree — same shape as a human author. This is intentional: wisdom proposals from an LLM are a downstream-tool concern, not core-engine concern.
- **−** Schema break. Bases upgrading from 0.2.x must drop and rebuild (`SCHEMA_VERSION` bumped twice across PR1 + PR2).

## Alternatives considered

1. **Keep distill / review and ADD an author directory shape.** Two write paths for the same layer — too much surface area, and the candidate / review path was the one that wasn't producing value. Rejected.
2. **Keep the schema (`wisdom_items` etc.) but stop calling `distill`.** Dead metadata that the engine would have to maintain forever (or migrate-out later). Worse: the dedicated wisdom vec table would diverge from the main `vec_chunks_v<id>` and retrieve still couldn't return wisdom hits without dedicated code. Rejected.
3. **Multi-tenant knowledge base (one knowledge base per author).** Too aggressive for alpha — would require auth, namespacing, cross-base search rules, and would push the "the knowledge base is the product" invariant past comfort. The directory-name-as-author shape gives 80% of the value with 5% of the complexity. Revisit post-1.0.
4. **Frontmatter `author:` field instead of directory.** Less filesystem-native (a directory is visible in any file browser; a frontmatter field is not) and requires a per-file convention that's easy to forget. Directory encoding is mechanical and visible. Rejected.

## References

- PR1 #120 (destructive cleanup), PR2 #121 (wire wisdom into documents), PR3 #122 (read APIs + lint + retrieve coverage), PR4 (this docs/ADR pass).
- `docs/design.md` "Wisdom Layer Design" section — current contract.
- CHANGELOG `[0.3.0]` — the release entry.
- ADR-0001 (provenance edges as separate table) — companion architectural decision; provenance is the shared edge wisdom uses for D-layer attribution.
