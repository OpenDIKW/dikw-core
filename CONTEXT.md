# dikw-core

The DIKW pyramid is the four-layer mental model the engine is built around. Every command verb corresponds to **one** transition between two layers — overlap between verbs means the model is bleeding through, fix the verb.

For the four DIKW layers themselves (D / I / K / W — what each layer contains, where it lives, who writes it), see [`docs/design.md` § The Four Layers](docs/design.md). This document adds the language and verb boundaries on top of that storage-level definition; the two are meant to be read together.

## Language

### Naming the layers

**K layer** vs **knowledge tree**: say "K layer" when referring to the DIKW role, "knowledge tree" when referring to the on-disk files under `<base>/knowledge/`. The bare term "wiki" is reserved for wikilink syntax (`[[Target]]`) — it never refers to the K layer or the on-disk directory.

### Containers

**base**:
The root directory of one knowledge engine instance. Owned by the user, contains `dikw.yml`, `sources/`, `knowledge/`, `wisdom/`, `.dikw/`. One `dikw serve` process binds to exactly one base.
_Avoid_: knowledge base, workspace, home, root, vault

**source**:
A single markdown file the user authored or curated, sitting under `<base>/sources/`. The input side of the pipeline.
_Avoid_: input file, raw doc, document (which is the indexed-row type, not the file)

**document**:
A `documents` table row — the indexed handle for a source (or K-layer page). Has `doc_id`, `path`, `layer`, `hash`. Crosses the Storage Protocol.
_Avoid_: source (which is the file on disk before it has been indexed)

### Classifying pages

**category**:
A node in the configurable, hierarchical classification tree declared in `dikw.yml` under `schema.categories` (each entry a `path` like `产品/移动端` plus an optional `desc`). It is both the page's frontmatter `category:` value and its on-disk folder: a page filed under `技术/架构` lives at `<base>/knowledge/技术/架构/<slug>.md`. The taxonomy is a **closed set** — synth's LLM may only file a page under a declared `path`; anything it can't place lands in `schema.fallback` (default `未分类/`) and is flagged by the `uncategorized` lint (ADR-0003). `category_from_path` recovers a page's category from its path at any depth.
_Avoid_: type, kind, tag (tags are an orthogonal multi-value frontmatter list, not the filing axis), folder.

**type** *(deprecated 0.5.0)*:
The old single-axis classification (`entity` / `concept` / `note`) that filed pages under fixed pluralized folders (`knowledge/concepts/…`). Generalized into the configurable **category** tree — `entity`/`concept`/`note` is now merely the *default* taxonomy a fresh base ships with. No `type:` frontmatter key, no plural folders, no `SchemaConfig.page_types` remain; a base carrying any of them trips `BaseUpgradeRequired`.

### Edges between pages

Two distinct, deliberately separated relationships connect pages. Conflating them pollutes graph-leg retrieval.

**wikilink** (a.k.a. **link**):
A `[[Target]]` reference in a page **body**, parsed into the `links` table (`src_doc_id → dst_path`, with `link_type`, `anchor`, `line`). Forms the K↔K (and K↔W / W↔W) graph that feeds graph-leg retrieval and orphan/broken-link lint. Reconciled from the body on every `persist_knowledge` / `persist_wisdom` via `replace_links_from`.
_Avoid_: reference, citation (overloaded), provenance (the other edge).

**provenance**:
The page→source attribution recorded in a K/W-page's `sources:` **frontmatter** — "this page was authored from these D-layer sources". A separate edge from **wikilink**: it lives in frontmatter not body, has no body line/anchor, and must NOT enter the wikilink graph. The **frontmatter is the source of truth** (the knowledge tree is user-editable open Markdown), so the edge reconciles from frontmatter on every `persist_knowledge` / `persist_wisdom` — exactly mirroring how **wikilink** reconciles from the body. For pages that pre-existed when provenance shipped, the `missing_provenance` LintKind + its deterministic Fixer backfill them via the standard `lint propose → lint apply` flow (`dikw client lint propose --rule missing_provenance` then `dikw client lint apply <task_id>`).
_Avoid_: reference, link, citation; "sources" alone (that's the frontmatter key, not the relationship).

### Pipeline verbs

**import**:
Take files **outside** the base and commit them into `<base>/sources/`. Markdown inputs (`.md`) pass through after frontmatter + asset validation. Non-markdown single-file inputs (`.pdf`, `.epub`, …) are first converted to md+assets by an installed **client-side converter plugin** (see [`docs/converters.md`](docs/converters.md)); without a plugin for the file's extension the input is rejected. Conversion happens in the client process — the server never loads converter dependencies. Validates frontmatter + assets, packs as multipart, server stages then atomically replaces into place. Does **not** chunk, embed, or touch the D/I layer.
_CLI_: `dikw client import <path>`; `--converter=<name>` overrides the default engine for non-md inputs.
_HTTP_: `POST /v1/import`
_Avoid_: upload (transport-layer term — only correct when describing the HTTP wire), add, push

**ingest**:
Scan `<base>/sources/`, parse markdown, chunk, embed, write into D + I layers. **Only** consumes files already inside the base — never accepts external input.
_CLI_: `dikw client ingest`
_HTTP_: `POST /v1/ingest`
_Avoid_: index (verb), process

**synth**:
LLM-author K-layer knowledge pages from D-layer sources. Files each page under its **category** path (`<base>/knowledge/<category>/<slug>.md`) and appends a row to the `knowledge_log` table. Does **not** generate `index.md` / `log.md` — navigation is the file tree + `retrieve` (0.5.0, see [`docs/adr/0004-drop-generated-index-and-log.md`](docs/adr/0004-drop-generated-index-and-log.md)).
_CLI_: `dikw client synth`
_HTTP_: `POST /v1/synth`
_Avoid_: summarize, build wiki (the term "wiki" is reserved for wikilink syntax)

**retrieve**:
End-of-pipeline read path. Hybrid search (BM25 + vector + RRF) over the I layer returns ranked chunks + page refs. **No LLM call** — the agent owns synthesis (rewrite, expansion, conversation context, the final answer prompt). `dikw-core` no longer ships an in-engine `query` verb.
_CLI_: `dikw client retrieve "..."`
_HTTP_: `POST /v1/retrieve` (streams NDJSON: `retrieve_started → retrieval_done → final`)
_Avoid_: query, ask, search

### Consistency & deletion

The filesystem (`sources/`, `knowledge/`, `wisdom/`) is the **source of truth**; the
`documents` projection in storage is rebuildable from it. Engine-owned state
(`<base>/.dikw/`, the `knowledge_log` table, the task ledger) and `synth`'s LLM output
are *not* part of that promise (once `synth` writes a page to disk, the file is disk
content like any other). The terms below name the divergences between disk and the
projection, and the actions that repair them. They are **cross-cutting maintenance**
(like `lint`), not pipeline transitions — the "one verb, one transition" rule above
governs the pipeline verbs, not these. This is agreed design (ADR-0005), landing
incrementally: `delete` (PR1), the `missing_file` drift kind (PR2), the
`untracked_file` / `stale_index` reindex kinds (PR3), and the read-only
`dangling_provenance` drift kind (PR4) have all shipped.

**drift**:
A divergence between the authoritative on-disk trees and the `documents` projection. Surfaced as `lint` kinds, repaired through `lint apply` — or, for a single named file, `delete`.
_Avoid_: desync, staleness, inconsistency.

**missing_file**:
A **drift** lint kind (D/K/W): a `documents` row whose backing file is gone from disk. Fixer purges the row and its outgoing edges.
_Avoid_: orphan / orphan_document (collides with **orphan_page** — see Flagged ambiguities), dangling document.

**untracked_file**:
A **drift** lint kind (K/W): a markdown file on disk with no `documents` row. Fixer indexes it — the path by which a hand-written knowledge page becomes first-class.
_Avoid_: new file, unindexed.

**stale_index**:
A **drift** lint kind (K/W): a `documents` row whose stored hash no longer matches the file's bytes (a hand-edit). Fixer re-projects the current bytes; it never re-runs `synth`, so the edit is preserved.
_Avoid_: outdated, dirty.

**dangling_provenance**:
A **drift** lint kind (K/W): a **provenance** edge whose target source file is gone. Read-only — surfaced, never auto-repaired (the frontmatter is the user's to edit).
_Avoid_: broken provenance (reserve "broken" for the **wikilink** graph).

**delete**:
Remove one named document — move the live file to `<base>/trash/<layer>/<rel>` (recoverable, with an audit block) and drop its storage row + outgoing edges. Immediate (not propose/apply), symmetric with the `wisdom write` verb. Inbound edges from live pages are left as `broken_wikilink`, never silently rewritten.
_CLI_: `dikw client delete <path>`
_HTTP_: `POST /v1/base/delete`
_Avoid_: remove, trash (the trash/ move is one step of delete, not the verb), purge (DB-only term).

## Relationships

- **import** writes to `<base>/sources/`; **ingest** reads from it. Without import the user puts files there by hand; without ingest the files don't reach D/I.
- A **source** becomes one or more **documents** after **ingest** (markdown front-matter splits, asset attachments, etc.).
- A **document** in the D layer becomes zero or more K-layer **documents** after **synth** (one source can fan out into multiple knowledge pages).
- The user owns `<base>/sources/`, `<base>/knowledge/`, `<base>/wisdom/` — three plain markdown trees. The engine owns `<base>/.dikw/` — opaque state (index, auth tokens, task ledger, staging). Wisdom pages are hand-written and indexed into the documents table, so they participate in retrieve/lint.
- A K-layer **document** carries **provenance** edges back to the **source**(s) it was synth-authored from (`provenance` table, distinct from `links`). The reverse — "which K-pages derive from this source" — is the query this edge exists to answer.
- The filesystem is authoritative; a **document** is a rebuildable projection of a file. **drift** names where the projection has fallen out of step with disk; `lint apply` (scan-discovered) and `delete` (one named file) are how it is brought back. The engine never edits a user's body/frontmatter to repair drift — it only surfaces it.

## Example dialogue

> **Dev:** "User dropped a folder of notes on me — do I `import` or `ingest`?"
> **Maintainer:** "If the folder is outside the base, you `import` it first — that commits the bytes into `<base>/sources/`. Then `ingest` to chunk + embed them. They're two halves of one user mental action ('get my files into the engine'), but they're distinct pipeline stages because (a) import is a network/multipart operation that can fail mid-transfer, (b) ingest is CPU/embedding-bound and may want to retry without re-uploading."
>
> **Dev:** "And if the files were already in `<base>/sources/` because the user `cp`'d them there directly?"
> **Maintainer:** "Skip `import`, just run `ingest`. Import is for getting files **into** the base; if they're already there it's a no-op the user shouldn't have to invoke."

## Flagged ambiguities

- **upload** was used as the user-facing verb for the import action. Resolved: `upload` is reserved for HTTP-wire descriptions only (multipart upload, payload upload). The user-facing verb is `import`. The two are honest at different layers — the CLI speaks domain, the HTTP path speaks transport.
- **wiki** was historically overloaded: K-layer role, on-disk directory, and wikilink syntax. Resolved (0.4.0): say "K layer" for the role, "knowledge tree" for the on-disk files, and reserve **wiki** exclusively for **wikilink** (`[[Target]]`) syntax. The term "wiki" no longer refers to the K layer or its files.
- **document** vs **source**: in the D layer they're nearly synonymous (one source → one document, usually), but **source** is the file on disk and **document** is the indexed row. Keep them distinct because in K + W layers the documents have no corresponding source file — they were LLM-authored.
- **reference / 引用** was used for "source X is used by page Y". Resolved into two distinct edges: **wikilink** (body `[[…]]`, the K↔K graph) and **provenance** (frontmatter `sources:`, page→source attribution). They are stored in separate tables and never mixed — a page can have one without the other.
- **orphan** was ambiguous between "a page with no inbound wikilinks" and "a `documents` row whose file is gone". Resolved: **orphan_page** keeps the no-inbound-links meaning; the file-gone **drift** kind is **missing_file** (never "orphan_document"). "Orphan" alone always refers to the link sense.

## Plugin contract

**converter plugin**:
A pypi package that turns one non-markdown file (`paper.pdf`, `book.epub`, …) into the md+assets a **source** is made of. Plugins are discovered via the `dikw.client.converters` entry-points group, run in-process inside `dikw client`, and live in the sibling [`dikw-plugins`](https://github.com/opendikw/dikw-plugins) repo — never in dikw-core. The contract (`Converter` Protocol, `convert(input_path, output_dir)` signature, output layout) is defined in [`src/dikw_core/client/converters.py`](src/dikw_core/client/converters.py).
_Avoid_: backend (`SourceBackend` is the engine-side D-layer parser — different concern, different layer), loader, importer (verb collision), adapter.

**converter engine name**:
The short label a plugin advertises as `Converter.name` (e.g. `marker`, `mineru`). Used to disambiguate when multiple plugins claim the same extension — via `--converter=<name>` on the CLI or `[default.converters]` in `client.toml`. Lives parallel to the package name (`dikw-converter-pdf` ships the `marker` engine).
_Avoid_: backend name, driver, profile.
