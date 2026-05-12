# INSTALL_FOR_AGENTS.md

> Concrete bootstrap for an AI agent (or the human in front of one) to go
> from "no dikw-core anywhere" to "I can call `/v1/retrieve` and get
> grounded chunks back". For the conceptual overview see
> [`AGENTS.md`](./AGENTS.md); for the human walkthrough with screenshots
> and Obsidian notes see [`docs/getting-started.md`](./docs/getting-started.md).

## Prerequisites

- Python **3.12+**
- [`uv`](https://docs.astral.sh/uv/) (the project manager dikw-core uses)
- One of:
  - LLM API key (Anthropic, OpenAI-compatible, or `codex` OAuth) — only
    needed if you'll call `/v1/synth` or `/v1/distill` (engine-internal
    LLM legs that author K-layer wiki pages + W-layer wisdom candidates).
    Answer synthesis is **not** a dikw-core verb; agents run their own
    LLM against retrieve output.
  - Embedding API key on any OpenAI-compatible vendor — only needed if
    you'll embed for vector search; `/v1/ingest --no-embed` skips this

You can do retrieval-only (FTS hits, no LLM, no vectors) with no API
keys at all.

## 1. Install

From source (recommended for now — the package is pre-alpha):

```bash
git clone https://github.com/helebest/dikw-core
cd dikw-core
uv sync
```

Or, when a release is up on PyPI:

```bash
uv tool install dikw-core
```

Verify:

```bash
uv run dikw version
```

## 2. Initialise a base

A **dikw base** is the directory the engine binds to: `dikw.yml` at its
root, `sources/` for raw notes, plus `wiki/`, `wisdom/`, `.dikw/` that
the engine populates. Pick any path:

```bash
uv run dikw init ./my-base --description "agent knowledge base"
```

That creates `my-base/dikw.yml` plus the empty subdirectories. Open
`my-base/dikw.yml` and confirm the defaults match what you want — the
config is the source of truth for **provider, storage, retrieval**
settings.

## 3. Configure the provider (optional, for LLM/embedding work)

Edit `my-base/dikw.yml`. Minimum for an OpenAI-compatible setup:

```yaml
provider:
  llm: openai_compat
  llm_model: gpt-4.1-mini
  llm_base_url: https://api.openai.com/v1
  embedding: openai_compat
  embedding_model: text-embedding-3-small
  embedding_base_url: https://api.openai.com/v1
  embedding_dim: 1536
  embedding_normalize: true
  embedding_distance: cosine
```

Per-vendor cookbook (MiniMax, GLM, Gemini, DeepSeek, Gitee AI, Ollama,
…) lives in [`docs/providers.md`](./docs/providers.md). dikw-core
**never auto-detects** vendor URLs — fill them in by hand.

## 4. Set secrets

Secrets go in environment variables, never in `dikw.yml`. Either export
them or copy [`.env.example`](./.env.example) → `my-base/.env` (gitignored)
and source it:

```bash
# LLM key (provider depends on dikw.yml's llm: setting)
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...

# Embedding key — separate from the LLM key on purpose
export DIKW_EMBEDDING_API_KEY=sk-...
```

The embedding leg reads `DIKW_EMBEDDING_API_KEY` exclusively. There is
**no fallback** to `OPENAI_API_KEY` — that's deliberate so MiniMax-LLM +
Gitee-embedding (and similar splits) don't cross-wire credentials.

## 5. Verify provider connectivity

```bash
uv run dikw client check
```

Pings each leg with one tiny request and emits a JSON `CheckReport`
(default) with endpoint / latency / dim per leg. Pipe into `jq` to
branch. Exit code 0 = both legs OK, 1 = any failure, 2 = flag misuse.
Pass `--llm-only` or `--embed-only` to probe one leg in isolation. Add
`--format table` for a human-readable summary.

> Note: `check` requires a running server. If you haven't started one
> yet, point `serve-and-run` at the base you just initialised:
>
> ```bash
> uv run dikw client serve-and-run --base ./my-base -- check
> ```
>
> That spins one up, runs the inner command, and tears it down. Without
> `--base ./my-base` it falls back to the cwd, which won't contain a
> `dikw.yml` until step 6 has been run.

## 6. Start the server

```bash
uv run dikw serve --base ./my-base
```

Default bind is `127.0.0.1:8765`, no auth (loopback only). For
non-loopback you must set a token:

```bash
export DIKW_SERVER_TOKEN=$(openssl rand -hex 32)
uv run dikw serve --base ./my-base --host 0.0.0.0
```

Leave this running. Your agent talks to it from any HTTP client.

## 7. Probe the server

The first call your agent should make. Confirms the server is up, shows
which base it's bound to, and exposes the resolved provider config:

```bash
uv run dikw client health --format json
```

```json
{
  "status": "ok",
  "version": "0.x.y",
  "base_root": "/abs/path/to/my-base",
  "storage_engine": "sqlite",
  "layer_counts": {
    "sources": 0,
    "wiki_pages": 0,
    "wisdom_items": 0,
    "chunks": 0
  },
  "providers": { "llm": { ... }, "embedding": { ... } }
}
```

## 8. Ingest source material

Drop markdown into `my-base/sources/` (or import a local tree via `dikw client import`). Then:

```bash
# Ingest whatever's already on the server's disk:
uv run dikw client ingest --no-embed         # FTS only, no API calls
uv run dikw client ingest                    # full pipeline (needs embedding key)

# Or push a local tree to the server's sources/ first, then ingest:
uv run dikw client import ./local-sources    # pre-flights md + imports packages
uv run dikw client ingest
```

Per-file errors are non-fatal by default — the run continues and a list
of failures lands on `IngestReport.errors`. Pass `--strict` to make any
file error fail the whole run.

## 9. Retrieve

```bash
uv run dikw client retrieve "your question" --format json
```

Returns a list of chunks (text + path + layer + score) plus page-level
refs. **No LLM call** — answer synthesis is the agent's job. Feed the
JSON into your own LLM with whatever query rewrite, expansion, or
conversation-context handling you want; dikw-core is stateless and does
not own that step.

## 10. Read a full page after a hit

```bash
uv run dikw client pages get sources/notes/alpha.md
```

Returns the parsed body plus chunk anchors aligned to the same
coordinate space, so you can re-locate every chunk hit inside the page.
Use `dikw client pages list` to enumerate registered paths.

## Common failure modes

- **404 from `/v1/base/pages/{path}`** — the path exists on disk but
  hasn't been ingested. Run `dikw client ingest` first, or call
  `GET /v1/base/pages` to see what's actually queryable.
- **Server refuses to start with `--host 0.0.0.0`** — that requires
  `DIKW_SERVER_TOKEN`. The runtime fails fast rather than expose an
  unauthenticated base to the network.
- **`embedding_dim` mismatch on first ingest** — the dim is locked at
  first insert; if your config says 1536 but the endpoint returns 1024,
  the run fails. Fix `dikw.yml` and re-run; switching dims after data is
  in is a rebuild (pre-alpha).
- **`check` reports 401/403** — wrong key for the configured leg. Check
  `OPENAI_API_KEY` vs `ANTHROPIC_API_KEY` vs `DIKW_EMBEDDING_API_KEY`
  and re-export.
- **No data after ingest** — `dikw client status` shows zero rows. Most
  often the source pattern in `dikw.yml` (default `**/*.md`) didn't
  match anything; verify your files actually live under `sources/`.

## Pointers

- [`AGENTS.md`](./AGENTS.md) — what to call, when, why
- [`docs/server.md`](./docs/server.md) — full HTTP wire spec
- [`docs/architecture.md`](./docs/architecture.md) — module map
- [`docs/providers.md`](./docs/providers.md) — provider cookbook
- [`docs/getting-started.md`](./docs/getting-started.md) — human walkthrough
