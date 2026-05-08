# Getting started

This walkthrough takes a blank directory to a queryable knowledge base with a
curated Wisdom layer in about five minutes. It only needs Python 3.12+ and
`uv`; LLM keys are optional until you hit `dikw synth` or `dikw query`.

## 1. Install and scaffold

```bash
git clone https://github.com/helebest/dikw-core
cd dikw-core
uv sync

# Pick any directory — `my-base/` below — it will also be a valid Obsidian vault.
uv run dikw init ../my-base --description "my research base"
cd ../my-base
```

The init command creates this tree:

```text
my-base/
├── dikw.yml              # the config the engine reads on every command
├── sources/              # your raw documents go here (Data layer)
├── wiki/                 # LLM-authored knowledge pages, regenerated on synth
│   ├── index.md
│   ├── log.md
│   └── {entities,concepts,notes}/
├── wisdom/               # principles / lessons / patterns (LLM proposes, you approve)
│   ├── principles.md
│   ├── lessons.md
│   ├── patterns.md
│   └── _candidates/
└── .dikw/                # engine state (gitignored)
    └── index.sqlite
```

The whole tree is the **dikw base**; the `wiki/` subdirectory is just
the K-layer slice. Open the folder in Obsidian and you'll see the wiki +
wisdom pages render natively thanks to the `[[wikilink]]` syntax and
YAML front-matter the engine emits.

## 2. Start the server

`dikw-core` runs as a long-lived process; the CLI is a thin client that
talks HTTP + NDJSON to it. Start the server bound to your base in a
spare terminal (or under a process supervisor):

```bash
uv run dikw serve --base .
# bound to http://127.0.0.1:8765 — no auth on loopback
```

Leave it running. Every `dikw <op>` shown below is a top-level alias for
`dikw client <op>` and routes through this server.

## 3. Add source material and ingest

Drop markdown files anywhere under `sources/`, then:

```bash
# Offline mode — indexes FTS only, no API calls.
uv run dikw ingest --no-embed

# Or with embeddings (requires DIKW_EMBEDDING_API_KEY on any OpenAI-compatible
# endpoint — OpenAI, Gitee AI, Ollama, vLLM, …).
export DIKW_EMBEDDING_API_KEY=sk-...
uv run dikw ingest
```

`dikw status` shows document, chunk, and embedding counts per DIKW layer.
Subsequent ingests are idempotent: files whose content hash hasn't changed
are skipped.

## 4. Ask questions (Information layer → LLM)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run dikw query "What does Karpathy mean by deterministic scoping?"
```

Output includes the answer and a table of citations. Every claim cites a
source excerpt by `[#N]`; approved wisdom items cite by `[W1]`.

## 5. Synthesise a Knowledge layer

```bash
uv run dikw synth
```

The LLM reads each source doc and produces a `wiki/<folder>/<slug>.md`
page, cross-linked via `[[wikilinks]]`. `wiki/index.md` and `wiki/log.md`
regenerate automatically. Re-running is a no-op until you add new sources
(or pass `--all` to resynthesise everything).

Run `dikw lint` to check for broken wikilinks, orphans, and duplicate titles.

### Watching synth progress on large sources

A long source (a book-sized markdown) is split into multiple LLM calls
under the hood. The client streams two layers of progress events:

- `phase="synth"` — outer counter, advances once per source (`2/43`).
- `phase="synth_llm"` — inner counter, fires `status="calling"` before
  each LLM round-trip and `status="returned"` after, so you can tell a
  slow LLM call apart from a deadlock. A parser failure surfaces as
  `status="error"` with `error_kind` / `error_msg` fields.

If a single source freezes for minutes without inner events you're
either looking at a provider stall (codex SSE keepalive bug, gateway
buffering) or a real network hang — not a synth-loop issue.

## 6. Distil Wisdom (the W layer)

```bash
uv run dikw distill
uv run dikw review list
uv run dikw review approve W-abcdef123456
```

`distill` prompts the LLM for principles/lessons/patterns supported by **at
least two pieces of evidence** across the wiki. Approval deletes the
candidate file, promotes the item to approved status, and regenerates
`wisdom/<kind>s.md`. Rejected items are archived.

Approved wisdom automatically surfaces in future `dikw query` answers under
an "operating principles" block — cited as `[W1]`, `[W2]`, ….

## 7. Check retrieval quality on your corpus

```bash
# Default: run all packaged datasets (ships with the MVP dogfood corpus).
uv run dikw eval

# Run against your own corpus: create a 3-file directory and point at it.
uv run dikw eval --dataset ./my-corpus/
```

Each query is marked a "hit" at top-k if any `expect_any` doc stem is in
the top-k result. Metrics: `hit@3`, `hit@10`, `MRR`. Exit code 0/1/2.

The full convention (what `dataset.yaml` looks like, how to author
queries, how to convert public benchmarks) lives in [`evals/README.md`](../evals/README.md).

## 8. Bind the server to a non-loopback interface

`dikw serve --host 0.0.0.0` is rejected unless `DIKW_SERVER_TOKEN` is set
— the runtime refuses to expose an unauthenticated base to the network.
Run with the token:

```bash
export DIKW_SERVER_TOKEN=$(openssl rand -hex 32)
uv run dikw serve --base . --host 0.0.0.0
```

Clients pick the same token up via `DIKW_SERVER_TOKEN` (or `--token` /
`~/.config/dikw/client.toml`) and pass it as a bearer header.

## Pluggable providers

Edit `dikw.yml` to swap LLM or embedding providers without changing code:

```yaml
provider:
  llm: openai_compat           # anthropic_compat | openai_compat (protocol names)
  llm_model: gpt-4.1-mini
  llm_base_url: http://localhost:11434/v1   # Ollama, vLLM, Azure, …
  embedding: openai_compat
  embedding_model: text-embedding-3-small
  embedding_base_url: https://api.openai.com/v1
  embedding_dim: 1536          # required: must match what the endpoint returns
  embedding_revision: ""       # bump to force re-embed when vendor refreshes weights silently
  embedding_normalize: true
  embedding_distance: cosine
```

`llm` is a **protocol** name (which SDK to speak), not a vendor name.
`llm_base_url` works for both `anthropic_compat` and `openai_compat`. With
`llm: anthropic_compat` it retargets the official `anthropic` SDK at any
Anthropic-protocol-compatible endpoint (e.g., MiniMax's
`https://api.minimaxi.com/anthropic`), keeping the `cache_control` benefit
on the system prompt.

For a per-vendor config cookbook (MiniMax, GLM, Gemini, DeepSeek,
Gitee AI, Ollama, …), a pre-flight checklist, and the production
gotchas around batch size, embedding dimensions, retries, and prompt
caching, see [`providers.md`](./providers.md).

### Example: MiniMax LLM + Gitee AI embeddings

MiniMax has no embeddings endpoint — pair it with an OpenAI-compatible
embedding vendor. The example below uses Gitee AI's `Qwen3-Embedding-0.6B`
(1024 native, the recommended default; swap in `Qwen3-Embedding-8B` with
`embedding_dim: 1024` matryoshka or `4096` native for higher-cost runs).
dikw-core never auto-detects vendor URLs — fill these in by hand:

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
[`tests/fixtures/live-minimax-gitee.dikw.yml`](../tests/fixtures/live-minimax-gitee.dikw.yml)
— drop it into a fresh base and fill in your two keys.

The two legs use **distinct keys**. The embedding leg reads
`DIKW_EMBEDDING_API_KEY` exclusively — no fallback to `OPENAI_API_KEY` — so
misconfig fails loudly instead of cross-wiring credentials:

```bash
export ANTHROPIC_API_KEY=<your-MiniMax-key>
export DIKW_EMBEDDING_API_KEY=<your-Gitee-key>
```

Or copy [`.env.example`](../.env.example) → `.env` (gitignored) and fill
it in. `.env` holds **secrets only** — all non-secret config (endpoint
URLs, model names, dimensions, batch size, display label) lives in
`dikw.yml`. `pytest-dotenv` auto-loads `.env` for the test suite; for
`dikw` CLI calls either `source` it (`set -a; source .env; set +a`) or
use `uv run --env-file .env dikw …`.

### Verify your provider config

After editing `dikw.yml` and exporting the env vars, run:

```bash
uv run dikw check --llm-only     # just LLM — run this first if you set up vendors one at a time
uv run dikw check --embed-only   # just embedding
uv run dikw check                # both legs
```

Each variant pings the relevant provider with one tiny request and prints
a status table with endpoint, latency, and dim/tokens. Exit code is 0 on
success, 1 on any probe failure, 2 when `--llm-only` and `--embed-only`
are passed together. Do this *before* running `dikw ingest` on a real
corpus so a misconfigured endpoint doesn't burn a full embedding run.

## Pluggable storage

Two backends ship; switch by editing `storage.backend` in `dikw.yml`:

```yaml
storage:
  backend: sqlite                     # default
  path: .dikw/index.sqlite
```

Enterprise / multi-user via Postgres (requires `pip install dikw-core[postgres]`
and a database with the `vector` extension):

```yaml
storage:
  backend: postgres
  dsn: postgresql://user:pw@host:5432/dikw
  schema: dikw
  pool_size: 10
```

Both backends implement the same `Storage` Protocol, so every `dikw`
command behaves identically regardless of which one is active.
