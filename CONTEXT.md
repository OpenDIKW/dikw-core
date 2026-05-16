# dikw-core

The DIKW pyramid is the four-layer mental model the engine is built around. Every command verb corresponds to **one** transition between two layers — overlap between verbs means the model is bleeding through, fix the verb.

For the four DIKW layers themselves (D / I / K / W — what each layer contains, where it lives, who writes it), see [`docs/design.md` § The Four Layers](docs/design.md). This document adds the language and verb boundaries on top of that storage-level definition; the two are meant to be read together.

## Language

### Naming the layers

**K layer** vs **wiki tree**: "wiki" is overloaded — say "K layer" when referring to the role, "wiki tree" when referring to the on-disk files under `<base>/wiki/`. Bare "wiki" is ambiguous.

### Containers

**base**:
The root directory of one knowledge engine instance. Owned by the user, contains `dikw.yml`, `sources/`, `wiki/`, `wisdom/`, `.dikw/`. One `dikw serve` process binds to exactly one base.
_Avoid_: knowledge base, workspace, home, root, vault

**source**:
A single markdown file the user authored or curated, sitting under `<base>/sources/`. The input side of the pipeline.
_Avoid_: input file, raw doc, document (which is the indexed-row type, not the file)

**document**:
A `documents` table row — the indexed handle for a source (or K-layer page). Has `doc_id`, `path`, `layer`, `hash`. Crosses the Storage Protocol.
_Avoid_: source (which is the file on disk before it has been indexed)

### Pipeline verbs

**import**:
Take files **outside** the base and commit them into `<base>/sources/`. Markdown inputs (`.md`) pass through after frontmatter + asset validation. Non-markdown single-file inputs (`.pdf`, `.epub`, …) are first converted to md+assets by an installed **client-side converter plugin** (see [`docs/converters.md`](docs/converters.md)); without a plugin for the file's extension the input is rejected. Conversion happens in the client process — the server never loads converter dependencies. Validates frontmatter + assets, packs as multipart, server stages then atomically replaces into place. Does **not** chunk, embed, or touch the D/I layer.
_CLI_: `dikw client import <path>` (top-level: `dikw import`); `--converter=<name>` overrides the default engine for non-md inputs.
_HTTP_: `POST /v1/import`
_Avoid_: upload (transport-layer term — only correct when describing the HTTP wire), add, push

**ingest**:
Scan `<base>/sources/`, parse markdown, chunk, embed, write into D + I layers. **Only** consumes files already inside the base — never accepts external input.
_CLI_: `dikw client ingest`
_HTTP_: `POST /v1/ingest`
_Avoid_: index (verb), process

**synth**:
LLM-author K-layer wiki pages from D-layer sources. Writes `<base>/wiki/*.md`, updates `index.md` + `log.md`.
_CLI_: `dikw client synth`
_HTTP_: `POST /v1/synth`
_Avoid_: summarize, build wiki

**distill**:
LLM-propose W-layer candidates from K-layer pages. Each candidate must cite ≥ 2 evidence pieces from K or D. Output requires `dikw client review approve` before becoming live wisdom.
_CLI_: `dikw client distill`
_HTTP_: `POST /v1/distill`
_Avoid_: extract, derive

**retrieve**:
End-of-pipeline read path. Hybrid search (BM25 + vector + RRF) over the I layer returns ranked chunks + page refs. **No LLM call** — the agent owns synthesis (rewrite, expansion, conversation context, the final answer prompt). `dikw-core` no longer ships an in-engine `query` verb.
_CLI_: `dikw client retrieve "..."`
_HTTP_: `POST /v1/retrieve` (streams NDJSON: `retrieve_started → retrieval_done → final`)
_Avoid_: query, ask, search

## Relationships

- **import** writes to `<base>/sources/`; **ingest** reads from it. Without import the user puts files there by hand; without ingest the files don't reach D/I.
- A **source** becomes one or more **documents** after **ingest** (markdown front-matter splits, asset attachments, etc.).
- A **document** in the D layer becomes zero or more K-layer **documents** after **synth** (one source can fan out into multiple wiki pages).
- **distill** consumes K-layer documents, never D-layer sources directly.
- The user owns `<base>/sources/`, `<base>/wiki/`, `<base>/wisdom/` — three plain markdown trees. The engine owns `<base>/.dikw/` — opaque state (index, auth tokens, task ledger, staging).

## Example dialogue

> **Dev:** "User dropped a folder of notes on me — do I `import` or `ingest`?"
> **Maintainer:** "If the folder is outside the base, you `import` it first — that commits the bytes into `<base>/sources/`. Then `ingest` to chunk + embed them. They're two halves of one user mental action ('get my files into the engine'), but they're distinct pipeline stages because (a) import is a network/multipart operation that can fail mid-transfer, (b) ingest is CPU/embedding-bound and may want to retry without re-uploading."
>
> **Dev:** "And if the files were already in `<base>/sources/` because the user `cp`'d them there directly?"
> **Maintainer:** "Skip `import`, just run `ingest`. Import is for getting files **into** the base; if they're already there it's a no-op the user shouldn't have to invoke."

## Flagged ambiguities

- **upload** was used as the user-facing verb for the import action. Resolved: `upload` is reserved for HTTP-wire descriptions only (multipart upload, payload upload). The user-facing verb is `import`. The two are honest at different layers — the CLI speaks domain, the HTTP path speaks transport.
- **wiki** is used both for the K-layer role and for the on-disk directory `<base>/wiki/`. Resolved: say "K layer" for the role, "wiki tree" for the files. Bare "wiki" is ambiguous.
- **document** vs **source**: in the D layer they're nearly synonymous (one source → one document, usually), but **source** is the file on disk and **document** is the indexed row. Keep them distinct because in K + W layers the documents have no corresponding source file — they were LLM-authored.

## Plugin contract

**converter plugin**:
A pypi package that turns one non-markdown file (`paper.pdf`, `book.epub`, …) into the md+assets a **source** is made of. Plugins are discovered via the `dikw.client.converters` entry-points group, run in-process inside `dikw client`, and live in the sibling [`dikw-plugins`](https://github.com/opendikw/dikw-plugins) repo — never in dikw-core. The contract (`Converter` Protocol, `convert(input_path, output_dir)` signature, output layout) is defined in [`src/dikw_core/client/converters.py`](src/dikw_core/client/converters.py).
_Avoid_: backend (`SourceBackend` is the engine-side D-layer parser — different concern, different layer), loader, importer (verb collision), adapter.

**converter engine name**:
The short label a plugin advertises as `Converter.name` (e.g. `marker`, `mineru`). Used to disambiguate when multiple plugins claim the same extension — via `--converter=<name>` on the CLI or `[default.converters]` in `client.toml`. Lives parallel to the package name (`dikw-converter-pdf` ships the `marker` engine).
_Avoid_: backend name, driver, profile.
