# GUIDE_FOR_AGENTS.md

This guide is for agents or operators who need to install, configure, and run
`dikw-core` as a knowledge backend.

For day-to-day `dikw client ...` operation, use `dikw-skills`. This file is
only the bootstrap guide: install, create a base, configure providers, start a
server, and verify that the server is reachable.

## What dikw-core Provides

`dikw-core` is a Python service plus CLI for a DIKW knowledge base:

- `sources/` stores source markdown and imported source packages.
- `knowledge/` stores generated K-layer knowledge pages.
- `wisdom/` stores hand-written W-layer pages (principles, lessons, patterns).
- `.dikw/` stores runtime state such as SQLite data and task history.
- `dikw.yml` is the base configuration file.

The bound directory is called a **dikw base**. A running `dikw serve` process
exposes the base over HTTP, and `dikw client ...` is the supported CLI client.

`dikw-core` does not own final answer synthesis. Retrieval returns grounded
chunks, page refs, pages, graph data, and assets; the calling agent composes the
final answer with its own LLM and prompt.

## Install

Python 3.12+ is required. Install the PyPI package; use the `cjk` extra by
default for Chinese/CJK bases:

```bash
pip install "dikw-core[cjk]"
```

Alternative isolated installs:

```bash
pipx install "dikw-core[cjk]"
uv tool install "dikw-core[cjk]"
```

If you manage a project with `uv`, add it to that project's environment:

```bash
uv add "dikw-core[cjk]"
```

Verify:

```bash
dikw version
```

When using `uv run` inside a project, use:

```bash
uv run dikw version
```

## Optional Converter Plugins

Converters are discovered in-process, so install them into the same environment
as `dikw-core`, not as separate isolated tools:

```bash
pip install dikw-converter-mineru
pip install dikw-converter-epub
```

For an isolated `uv tool` install:

```bash
uv tool install "dikw-core[cjk]" --with dikw-converter-mineru
uv tool install "dikw-core[cjk]" --with dikw-converter-epub
```

Install converters only when importing matching non-Markdown material.

## Create a Base

Create a directory that will hold `dikw.yml`, `sources/`, `knowledge/`, `wisdom/`,
and `.dikw/`:

```bash
dikw init ./my-base --description "agent knowledge base"
```

With `uv run`:

```bash
uv run dikw init ./my-base --description "agent knowledge base"
```

Open `my-base/dikw.yml` and configure providers, storage, and retrieval
settings. Provider examples live in `docs/providers.md`.

## Configure Secrets

Secrets belong in environment variables or an untracked `.env`, not in
`dikw.yml`.

Common variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export DIKW_EMBEDDING_API_KEY=sk-...
```

The embedding leg intentionally uses `DIKW_EMBEDDING_API_KEY` so LLM and
embedding credentials can come from different vendors.

## Start the Server

Loopback startup:

```bash
dikw serve --base ./my-base
```

With `uv run`:

```bash
uv run dikw serve --base ./my-base
```

Default bind is `127.0.0.1:8765`. Non-loopback binds require a bearer token:

```bash
export DIKW_SERVER_TOKEN=$(openssl rand -hex 32)
dikw serve --base ./my-base --host 0.0.0.0
```

Leave the server running while clients use the base.

## Verify the Server

In another terminal:

```bash
dikw client health
```

Expected shape:

```json
{
  "status": "ok",
  "version": "0.x.y",
  "base_root": "/abs/path/to/my-base",
  "storage_engine": "sqlite",
  "layer_counts": {
    "sources": 0,
    "knowledge_pages": 0,
    "wisdom_items": 0,
    "chunks": 0
  },
  "providers": {
    "llm": { "provider": "...", "api_key_present": true },
    "embedding": { "provider": "...", "api_key_present": true }
  }
}
```

Provider connectivity:

```bash
dikw client check
dikw client check --llm-only
dikw client check --embed-only
```

## Add Initial Source Material

Put Markdown under `my-base/sources/`, or import local files through the
client:

```bash
dikw client import ./local-sources
dikw client import ./paper.pdf --converter mineru
dikw client import ./book.epub --converter epub
```

Then refresh the index:

```bash
dikw client ingest
```

Long-running mutating commands are async by default and return a task handle.
Use `dikw-skills` for the normal task-following workflow.

## Day-to-Day Agent Operation

Install and use `dikw-skills` for operational workflows:

- observe server/base/provider state
- retrieve chunks, pages, graph links, provenance edges (page ↔ source
  attribution), and assets
- import local source material
- ingest, synthesize (LLM files each page under the configured `category`
  taxonomy), lint, and eval (lint includes `missing_provenance` for backfilling
  the provenance table on legacy bases, `uncategorized` for pages synth filed
  under the fallback bucket, and `missing_file` to purge an orphaned D/K/W row
  whose backing file was deleted outside dikw)
- author W-layer wisdom pages with `dikw client wisdom write` (hand-written
  and indexed on write; `ingest` does not scan `wisdom/`)
- delete any registered document (D/K/W) with `dikw client delete <path>` —
  purges its storage row + outgoing edges and soft-deletes the file to
  `<base>/trash/<layer>/<rel>` (e.g. `trash/knowledge/...`). To recover, move
  the file back: a D source re-indexes on the next `ingest`, but K/W need a
  re-run of `synth --all` / `wisdom write`. Inbound links from live pages
  surface as `broken_wikilink` on the next lint (counted in the report's
  `inbound_broken`)

- follow async tasks with cursor events, status, wait, and cancel

`dikw-skills` is the maintained agent-facing CLI SOP. Keep detailed command
workflow there instead of duplicating it in this repository.

## Operational Notes

- Most `dikw client` data-returning commands default to JSON.
- `retrieve` returns evidence for the calling agent; it does not produce the
  final user-facing answer.
- Async task events are JSON cursor pages exposed through
  `dikw client tasks events`.
- `serve-and-run` is useful for smoke tests, but temporary server logs may mix
  with inner command stdout. Prefer a running server for strict JSON parsing.
- If a page exists on disk but cannot be read through `dikw client pages get`,
  run `dikw client ingest` first; page reads are index-driven.

## Pointers

- `docs/providers.md` - provider configuration cookbook
- `docs/server.md` - HTTP wire contract and security posture
- `docs/deployment-docker.md` - container deployment
- `docs/getting-started.md` - human walkthrough
- `docs/architecture.md` - module map and layer contracts
