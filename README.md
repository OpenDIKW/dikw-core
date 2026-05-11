# dikw-core

AI-native knowledge engine that turns your documents into **Data → Information → Knowledge → Wisdom**.

Inspired by [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), extended end-to-end across the full DIKW pyramid. Where Karpathy's pattern stops at a compounding markdown wiki (the K layer), `dikw-core` adds a first-class **Wisdom layer** — distilled principles, lessons, and patterns that apply beyond any single source.

> Status: pre-alpha. Under active construction; APIs and on-disk formats will change.

## What you get

- A local-first knowledge base — the **dikw base** — where the on-disk layout is a **plain markdown tree** your editor (Obsidian, VS Code, …) can open directly.
- Four explicit DIKW layers with their own operations:
  - **D**ata — raw sources you curate.
  - **I**nformation — parsed, chunked, embedded, indexed (FTS5 + vectors).
  - **K**nowledge — LLM-authored wiki pages with `[[wikilinks]]`, `index.md`, and an append-only `log.md`.
  - **W**isdom — evidence-backed principles / lessons / patterns, human-approved, surfaced at query time.
- Pluggable LLM providers (API-first): Anthropic + OpenAI-compatible (covers OpenAI, Azure, Ollama, DeepSeek, Gemini-compat).
- Pluggable storage: SQLite+sqlite-vec (default), Postgres+pgvector (enterprise) — swap by config.
- **Client / server architecture.** A long-lived `dikw serve` (FastAPI + NDJSON) hosts the engine; the `dikw client …` Typer CLI talks to it over HTTP, streams progress events for long ops, and supports cancel / resume.

## Install & quick start

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/helebest/dikw-core
cd dikw-core
uv sync

uv run dikw init my-base --description "my research base"
cd my-base
# Drop some markdown into sources/, then run any single command via
# `serve-and-run` — it spawns a local server, runs the inner command,
# and tears it down.
uv run dikw serve-and-run -- ingest --no-embed
uv run dikw serve-and-run -- query "What does Karpathy mean by deterministic scoping?"
```

For interactive sessions or long iterations, run `dikw serve` once and
keep using `dikw client *` against it:

```bash
uv run dikw serve --base .   # in one terminal
# in another:
uv run dikw client status
uv run dikw client synth               # K layer (needs ANTHROPIC_API_KEY or OpenAI-compat)
uv run dikw client distill             # W-layer candidates
uv run dikw client review list
uv run dikw client review approve W-abcdef123456
uv run dikw client query "What does Karpathy mean by deterministic scoping?"
```

> Top-level aliases (`dikw status`, `dikw query`, …) are kept as shortcuts and
> route through the same client.

Server deployment, security posture, and the wire contract live in
[`docs/server.md`](./docs/server.md).

End-to-end walkthrough: [`docs/getting-started.md`](./docs/getting-started.md).
Architecture brief: [`docs/architecture.md`](./docs/architecture.md).
Approved design doc: [`docs/design.md`](./docs/design.md).

## Commands

Local-only commands run in this process:

| command                     | does                                                                          |
| --------------------------- | ----------------------------------------------------------------------------- |
| `dikw version`              | print the package version                                                     |
| `dikw init <path>`          | scaffold a dikw base (sources / wiki / wisdom / `.dikw/` + `dikw.yml`)        |
| `dikw serve --base <path>`  | start the FastAPI + NDJSON server bound to one base                           |
| `dikw serve-and-run -- <cmd>` | spawn a local server, run an inner client command against it, tear it down |

Everything else lives under `dikw client *` and talks to a running server
(top-level aliases keep the old muscle memory):

| command                     | does                                                                          |
| --------------------------- | ----------------------------------------------------------------------------- |
| `dikw client status`        | counts across DIKW layers                                                     |
| `dikw client check`         | ping the configured LLM + embedding endpoints to verify `dikw.yml` + keys     |
| `dikw client import <path>` | pre-flight + import local md packages (md + referenced assets) into the server's `sources/` |
| `dikw client ingest [--no-embed]` | parse + chunk + FTS-index + embed the server's `sources/` tree           |
| `dikw client query "<q>"`   | hybrid search + LLM answer (streams tokens via NDJSON) with citations         |
| `dikw client synth [--all]` | LLM turns source docs into K-layer wiki pages; maintains `index.md`+`log.md`  |
| `dikw client lint`          | report broken wikilinks, orphan pages, duplicate titles                       |
| `dikw client distill`       | LLM proposes W-layer candidates (each needs ≥ 2 pieces of evidence)           |
| `dikw client review {list,approve,reject}` | drive the candidate -> approved / archived state machine       |
| `dikw client eval [--dataset]` | run retrieval-quality evaluation against packaged or custom datasets       |
| `dikw client tasks {list,show,follow,cancel}` | inspect running / past async tasks on the server               |

## Providers

Configured via `dikw.yml`:

```yaml
provider:
  llm: anthropic_compat         # or: openai_compat
  llm_model: claude-sonnet-4-6
  llm_base_url: null            # set for any Anthropic-protocol-compatible endpoint
  embedding: openai_compat
  embedding_model: text-embedding-3-small
  embedding_base_url: https://api.openai.com/v1
  embedding_dim: 1536           # required: must match what the endpoint returns
  embedding_revision: ""        # bump to force re-embed when vendor refreshes weights silently
  embedding_normalize: true
  embedding_distance: cosine
```

`llm` names a wire **protocol** (which SDK to speak), not a vendor — the
actual vendor is whatever `llm_base_url` points at.

- `anthropic_compat` → uses the `anthropic` async SDK with `cache_control`
  on the system prompt, so repeated synth/query calls hit the prompt cache.
  Set `llm_base_url` to retarget the SDK at any Anthropic-protocol-compatible
  endpoint (e.g., MiniMax's `https://api.minimaxi.com/anthropic`); leave null
  for api.anthropic.com.
- `openai_compat` → uses the `openai` async SDK against any base URL that
  speaks the OpenAI HTTP surface (Azure, Ollama, vLLM, DeepSeek, MiniMax, …).

Full vendor cookbook (MiniMax, GLM, Gemini, DeepSeek, Gitee AI, Ollama, …)
and the production gotchas around batch size, embedding dimensions, and
retry/caching live in [`docs/providers.md`](./docs/providers.md).

### Using MiniMax LLM + Gitee AI embeddings

MiniMax has no embeddings endpoint — pair its Anthropic-compatible LLM surface
with an OpenAI-compatible embedding vendor. The example below uses
[Gitee AI](https://ai.gitee.com/v1) (`Qwen3-Embedding-0.6B`, 1024 native — the
recommended default; swap in `Qwen3-Embedding-8B` with `embedding_dim: 1024`
matryoshka or `4096` native for higher-cost / marginal-quality runs).
Fill the URLs in by hand — dikw-core never auto-detects vendor endpoints:

```yaml
provider:
  llm: anthropic_compat
  llm_model: <MiniMax Anthropic-compatible model name>
  llm_base_url: https://api.minimaxi.com/anthropic
  embedding: openai_compat
  embedding_model: Qwen3-Embedding-0.6B
  embedding_base_url: https://ai.gitee.com/v1
  embedding_dim: 1024               # 0.6B native; locked at first ingest
  embedding_revision: ""            # bump to force re-embed when Qwen weights drift silently
  embedding_normalize: true
  embedding_distance: cosine
  embedding_batch_size: 16          # required: Gitee rejects batches >25
  embedding_provider_label: gitee-ai  # optional; shows up in `dikw check`
```

A working reference copy lives at
[`tests/fixtures/live-minimax-gitee.dikw.yml`](./tests/fixtures/live-minimax-gitee.dikw.yml)
— drop it into a fresh base and fill in your two keys.

Two keys for two vendors — the embedding leg reads `DIKW_EMBEDDING_API_KEY`
exclusively (no `OPENAI_API_KEY` fallback), so misconfigurations fail loudly
rather than cross-wiring credentials:

```bash
export ANTHROPIC_API_KEY=<your-MiniMax-key>
export DIKW_EMBEDDING_API_KEY=<your-Gitee-key>
```

Verify connectivity **before** running ingest/query. The two legs can be
probed separately, which is useful when you set up one vendor first:

```bash
uv run dikw client check --llm-only     # just LLM — useful before Gitee is wired up
uv run dikw client check --embed-only   # just embedding
uv run dikw client check                # both
```

`dikw client check` pings each provider with one tiny request and prints a
status table with endpoint, latency, and dim/tokens. Exit code is 0 on
success, 1 on failure, 2 on flag misuse — scriptable in CI or a shell
one-liner.

## Source formats

Markdown ships out of the box. A new format is one `SourceBackend`
subclass + a `register()` call away — see
[`domains/data/backends/markdown.py`](./src/dikw_core/domains/data/backends/markdown.py)
for the reference impl.

## Storage

Two backends ship, selected in `dikw.yml`:

```yaml
storage:
  backend: sqlite          # sqlite | postgres

  # --- sqlite (default): single-user local ---
  path: .dikw/index.sqlite

  # --- postgres (enterprise): multi-user, pgvector + tsvector ---
  # backend: postgres
  # dsn: postgresql://user:pw@host:5432/dikw
  # schema: dikw
  # pool_size: 10
```

- **SQLite + `sqlite-vec` + FTS5** — the default. No extras required.
- **Postgres + `pgvector`** — install via `uv pip install dikw-core[postgres]`.
  Requires the `pg_trgm` and `vector` extensions (standard on the
  `pgvector/pgvector:pg16` Docker image). The adapter uses `tsvector`+GIN
  for FTS and `vector(N)` for embeddings; the vector dimension is set at
  first insert.

Engine code talks only to the `Storage` Protocol
([`storage/base.py`](./src/dikw_core/storage/base.py)); each adapter
implements the same contract and is swappable by changing `dikw.yml`.

## Releasing

Tagged pushes (`vX.Y.Z`) trigger
[`.github/workflows/release.yml`](./.github/workflows/release.yml), which
builds `sdist` + wheel, re-runs the full test gate, and publishes to PyPI
via **trusted publishing** (no token in repo secrets). One-time setup on
PyPI's side:

1. Create the `dikw-core` project on PyPI.
2. On the project's *Publishing* page, add a GitHub trusted publisher with:
   - owner: `helebest`
   - repository: `dikw-core`
   - workflow: `release.yml`
   - environment: `pypi`

After that, `git tag vX.Y.Z && git push --tags` is enough.

## License

MIT — see [LICENSE](./LICENSE).
