# dikw-core design-invariant rubric

A checklist a **fresh-agent reviewer** scores a diff against before merge — a `/code-review`
pass, a clean subagent that did not write the code, or a human. Each line is
**yes / no / N/A**. A `no` on any line is a **blocking finding** — surface it, don't wave it through.

These restate CLAUDE.md's *Core invariants* and *Layering invariants* in checkable form.
ruff/mypy/pytest are blind to all of them — that's exactly why a reading reviewer exists.

## Karpathy's rule — scoping deterministic, reasoning probabilistic
- [ ] No LLM call was added to a **scoping/navigation** path (source listing, chunk lookup, link traversal, lint detection, category placement parsing). Those stay deterministic SQL/file I/O.
- [ ] LLMs enter only at **synth** (K-layer authoring) or **eval judges**. No new "answer synthesis" verb was added to the engine (`retrieve` returns ranked chunks + refs; the agent layer runs its own LLM).
- [ ] A wrong-but-cheap-to-redo decision (mis-filed category, missed wikilink resolve) is left as a fixable lint, **not** "recovered" by inventing folders / force-merging titles (irreversible drift).

## Persist pipeline — `documents.active` is the commit marker
- [ ] Every layer write goes through its single entry: `persist_source` (D) / `persist_knowledge` (K) / `persist_wisdom` (W) — not ad-hoc `upsert_document` + chunk calls.
- [ ] On any **hard** exception mid-pipeline (permanent `ProviderError`, `replace_chunks`/`replace_links_from`/`replace_provenance_from` raising) the caller `deactivate_document(doc_id)`s it. `asyncio.CancelledError` is caught separately (it's a `BaseException`) → deactivate then re-raise.
- [ ] A **transient** embed skip is NOT treated as failure (stays `active=True` with `chunks_pending_embedding > 0`); only hard exceptions deactivate.
- [ ] Deactivated pages are excluded from success tallies (`created`/`updated`, `knowledge_paths_changed`) and surfaced as `persist_errors`.

## Edge reconciliation — replace, never union
- [ ] Re-persisting a page **replaces** its outgoing link set (`replace_links_from`) and provenance (`replace_provenance_from`) — removing a `[[wikilink]]` or a `sources:` entry actually drops the edge. No accumulating ghost edges.
- [ ] Provenance stays a **separate** edge from wikilinks (own `provenance` table; not `link_type='derived_from'` polluting the graph-leg channel).

## Wikilink resolve
- [ ] Resolution is exact → deterministic fuzzy normalize → **refuse on collision** (a fuzzy key mapping to ≥2 distinct paths stays broken for lint to surface). No silent wrong-merge.

## Layering & named seams
- [ ] `server/*` may import `dikw_core.{api,schemas,storage,providers}`; the reverse is forbidden — engine code imports no FastAPI/uvicorn/server task plumbing.
- [ ] `client/*` imports only `schemas` + stdlib + httpx + typer + rich — **no** `dikw_core.{api,storage,providers,server}` symbol (standalone-wheel invariant).
- [ ] New source formats add a `SourceBackend` subclass + `register()` — not a special case elsewhere.
- [ ] Search fusion (RRF), chunking, link-graph parsing live **outside** storage adapters (`info/search.py`), which expose primitives only.
- [ ] Anything crossing the Storage Protocol is a `schemas.py` pydantic model — no SQL types / cursors / ORM handles leak out of adapters.

## On-disk format is the product
- [ ] No change to `knowledge/` or `wisdom/` on-disk layout (frontmatter keys, folder shape, wikilink syntax) without `docs/design.md` updated first — users open these in any Markdown editor.
- [ ] Category taxonomy stays a **closed set**: synth files only under a declared `path`; unplaceable → `schema.fallback` + `uncategorized` lint. No invented folders, no path-prefix recovery.
- [ ] Prompts stay versioned markdown under `prompts/`, loaded via `prompts.resolve(...)` (never raw `load()` on the call path); new overridable prompts register their contract in `prompts/_contract.py`.

## Scope & simplicity
- [ ] The diff is the minimum that solves the stated request — no speculative features, no single-use abstractions, no error handling for impossible scenarios.
- [ ] Only the request's blast radius is touched — no drive-by reformatting/renaming; orphaned imports/helpers/tests that *this* change made unused are removed, pre-existing dead code is left (mentioned, not swept).

## Verification discipline (meta)
- [ ] K-layer / Retrieval changes carry an `evals/BASELINES.md` real-data entry (or `no-baseline-needed` label) and were developed test-first.
- [ ] Any new gate added by the diff was itself proven to go **red** on a known-bad input ("test the test").
