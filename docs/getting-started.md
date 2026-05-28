# Getting started

This walkthrough takes a blank directory to a queryable knowledge base in
about five minutes. It only needs Python 3.12+ and `uv`; LLM keys are
optional until you hit `dikw client synth` (the engine-internal K-layer
authoring leg). Plain `dikw client retrieve` runs without any LLM key.

## 1. Install and scaffold

```bash
git clone https://github.com/OpenDIKW/dikw-core
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
├── knowledge/            # LLM-authored knowledge pages, regenerated on synth
│   ├── index.md
│   ├── log.md
│   └── {entities,concepts,notes}/
├── wisdom/               # hand-written principles / lessons / patterns (you author these in
│   └── .gitkeep          # Obsidian; PR2 of the 0.3.0 W refactor wires them into retrieve)
└── .dikw/                # engine state (gitignored)
    └── index.sqlite
```

The whole tree is the **dikw base**; the `knowledge/` subdirectory is just
the K-layer slice. Open the folder in Obsidian and you'll see the knowledge +
wisdom pages render natively thanks to the `[[wikilink]]` syntax and
YAML front-matter the engine emits.

## Upgrading from 0.3.x to 0.4.0

0.4.0 renamed the K-layer directory from `wiki/` to `knowledge/` (and
the corresponding `Layer` enum, SQL table, and engine API symbols).
There is no in-place migration — opening an 0.3.x base with 0.4.0
raises `BaseUpgradeRequired`. For each existing base run:

```bash
cd <base>
mv wiki knowledge       # rename the K-layer directory
rm -rf .dikw            # drop the SQLite + auth + task ledger
# (dikw.yml stays — your existing config is reused)
dikw serve --base . &   # start the server
dikw client ingest      # reindex sources + knowledge pages
```

If you tracked `<base>/.dikw/auth.json` (OAuth tokens), re-run
`dikw auth login <provider>` (or `dikw auth import <provider>`) after
the rebuild — `rm -rf .dikw` wipes the credential store along with
the SQLite index.

## 2. Start the server

`dikw-core` runs as a long-lived process; the CLI is a thin client that
talks HTTP + NDJSON to it. Start the server bound to your base in a
spare terminal (or under a process supervisor):

```bash
uv run dikw serve --base .
# bound to http://127.0.0.1:8765 — no auth on loopback
```

Leave it running. Every `dikw client <op>` shown below routes through
this server — top-level short aliases like `dikw status` were removed in
0.1.0, so always spell out the `client` prefix.

## 3. Add source material and ingest

Two steps:

* **Import** — pre-flight + ship markdown packages (each md plus the
  assets it embeds) from a local directory into the server's
  `<base>/sources/` tree.
* **Ingest** — chunk + FTS-index + (optionally) embed whatever lives
  under `<base>/sources/`.

```bash
# Import your local notes (file or directory) into the base. Each *.md
# becomes one package together with the images it references; the
# pre-flight rejects bad frontmatter, missing assets, and orphan
# files BEFORE the network round trip. ``import`` commits the bytes
# into ``<base>/sources/``; it does NOT chunk or embed.
uv run dikw client import ./my-notes --format table

# ``ingest`` is the next step: scans ``<base>/sources/``, chunks the
# markdown, and writes the D/I layer. Offline mode indexes FTS only,
# no API calls. Async-by-default: prints a ``{task_id, ...}`` handle
# and exits 0 so an agent can move on; add ``--wait`` to block + render
# the IngestReport + map the final status to the exit code.
uv run dikw client ingest --no-embed --wait

# Or with embeddings (requires DIKW_EMBEDDING_API_KEY on any OpenAI-compatible
# endpoint — OpenAI, Gitee AI, Ollama, vLLM, …).
export DIKW_EMBEDDING_API_KEY=sk-...
uv run dikw client ingest --wait

# Fire-and-forget (default): submit + capture the task_id; follow up
# later with ``dikw client tasks wait <id>`` or page events via
# ``dikw client tasks events <id> --from-seq 0 --limit 100``.
uv run dikw client ingest
```

`import` and `ingest` are two halves of one user intent: import handles
**outside the base** → `sources/`; ingest handles `sources/` →
**chunks + embeddings**. If the server runs on the same machine as your
notes, you can also drop / edit markdown directly under
`<base>/sources/` and skip `dikw client import` — `dikw client ingest` always scans
whatever's on disk.

`dikw client status --format table` shows document, chunk, and embedding counts
per DIKW layer in a human-readable table. The default `dikw client status`
output is JSON so an automation script or agent can pipe it into `jq`
without extra flags. Subsequent ingests are idempotent: files whose
content hash hasn't changed are skipped.

## 4. Retrieve grounded chunks (Information layer)

```bash
uv run dikw client retrieve "What does Karpathy mean by deterministic scoping?" --format table
```

Returns the top-K chunks (with full text, path, layer, and score) plus
page-level refs. `--format table` renders the hits as a human-readable
table. For piping into `jq` or an agent loop, use
`dikw client retrieve "..." --plain` so the rich "retrieving…" status
line stays off stdout; that combination emits just the final JSON
payload.

**dikw-core does not produce the final answer itself.** Answer synthesis
— composing chunks into prose with a particular style, applying query
rewrite or conversation context — belongs in the agent layer (Claude
Code, ChatGPT, your own script). Pipe the retrieve JSON into your LLM of
choice and let it draft the answer with whatever prompt fits your task.

### Working with images

Markdown sources that embed images — either `![alt](path)` or Obsidian
`![[assets/foo.jpg]]` — flow through ingest into a content-addressed
asset store under `<base>/assets/`. Once ingested, two endpoints make
those bytes reachable from a remote client:

```bash
# 1) Discover which assets a source page references.
curl -sS "$DIKW_SERVER_URL/v1/base/pages/sources/some/doc.md" | jq '.assets'
# → [
#     {"asset_id": "a649…", "mime": "image/jpeg",
#      "url": "/v1/assets/a649…",
#      "original_paths": ["assets/images/a649…jpg"], ...}
#   ]

# 2) Fetch the actual bytes by id.
curl -sS "$DIKW_SERVER_URL/v1/assets/a649…" -o /tmp/diagram.jpg

# Or with the CLI client (writes to --output, prints metadata JSON to stdout):
uv run dikw client assets get a649… --output /tmp/diagram.jpg
```

The page `body` is returned **verbatim** — `![[…]]` references are
not rewritten — so the same response round-trips an editor-owned
Obsidian vault without divergence. Clients render images by mapping
each `assets[].original_paths` entry back to `assets[].url`.

Asset responses are immutable (content-addressed by SHA-256), so the
route emits a strong `ETag` + `Cache-Control: immutable, max-age=1y`;
a browser or HTTP cache fronting the server can revalidate with a
single `If-None-Match`.

### Inspecting the link graph

Every wikilink and cross-page markdown link the engine resolved is
exposed as a single read-only graph payload — you don't have to walk
the knowledge tree page-by-page:

```bash
# Whole-base graph: nodes + edges + unresolved wikilinks.
uv run dikw client graph get | jq '.stats'
# → { "node_count": 42, "edge_count": 137, "unresolved_count": 3 }

# What's broken? Each entry carries source path + the literal
# [[target text]] + the count.
uv run dikw client graph get | jq '.unresolved'
```

Use this in `dikw-web`'s Knowledge Graph view, in agent context-expansion
flows, or whenever you need to reason about K-layer connectivity
without re-implementing wikilink resolution per client.

### Inspecting page provenance

Each K-page's `sources:` frontmatter is reconciled into a dedicated
**provenance** edge — distinct from body `[[wikilinks]]`. Two questions
it answers cheaply:

```bash
# Forward: which D-layer sources was this K-page synth-authored from?
uv run dikw client pages provenance knowledge/concepts/topic.md \
  --direction out

# Reverse: which K-pages claim this source in their `sources:`?
uv run dikw client pages provenance sources/foo.md --direction in
```

Forward entries carry `resolved=true|false` — `false` means the
frontmatter references a source the engine has never seen (typo,
deleted source, renamed file). Reverse entries are only populated for
`Layer.SOURCE` paths; asking a `knowledge/...` path for `--direction in`
returns an empty list by design. JSON is the default output; pass
`--format table` for the human ✓/✗ rendering.

## 5. Synthesise a Knowledge layer

```bash
# Async-default: submit + print the task handle, exit 0 right away.
uv run dikw client synth

# Block until the synth task finishes, render the report, and exit
# with succeeded=0 / failed=1 / cancelled=130 / timeout=124.
uv run dikw client synth --wait
```

The LLM reads each source doc and produces a `knowledge/<folder>/<slug>.md`
page, cross-linked via `[[wikilinks]]`. `knowledge/index.md` and `knowledge/log.md`
regenerate automatically. Re-running is a no-op until you add new sources
(or pass `--all` to resynthesise everything).

Run `dikw client lint --format table` to check the K-layer for broken wikilinks, orphans, duplicate titles, non-atomic pages, and missing provenance edges (the default output is agent-facing JSON; add `--format table` for the human view). For `missing_provenance` issues on a legacy base, backfill in one shot via `dikw client lint propose --rule missing_provenance` then `dikw client lint apply <task_id>` — heuristic-only, no LLM required.

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

For server-side detail, raise the log level on the server process:

```bash
DIKW_LOG_LEVEL=DEBUG dikw serve --base $DIKW_BASE
```

DEBUG adds a per-group log line on each side of the LLM call (model,
section count, response chars). A parser failure surfaces at WARNING
even at the default INFO level — operators tailing the server don't
need DEBUG to spot one.

## 6. Author Wisdom (the W layer)

`wisdom/` is yours to fill by hand — there is no LLM-authoring path
and no candidate/review queue. Files live under
`wisdom/<author>/<slug>.md` (e.g. `wisdom/elon-musk/first-principles.md`);
the directory name is the author. A file directly under
`wisdom/<slug>.md` (no author subdirectory) is also valid, with
`author = None`.

Since 0.4.0 the engine indexes wisdom **only when the page is written
through `dikw client wisdom write`** (CLI) or `POST /v1/wisdom/write`
(HTTP) — `dikw client ingest` does NOT scan `<base>/wisdom/`. The
write API takes structured input (slug + title + body + optional
metadata) and runs the same `persist_wisdom` pipeline a manual edit
would have triggered in 0.3.x. Hand-edits to a wisdom file on disk
are NOT auto-reindexed — re-run `dikw client wisdom write` with the
edited body to refresh the row. (The same limitation already applies
to K-layer pages.)

Frontmatter shape that `wisdom write` emits — every field optional:

```yaml
---
title: First Principles                      # falls through to H1 or filename
status: published                            # draft | published | favorite | archived
sources:                                     # populates the provenance table
  - sources/notes/musk-bio.md
tags: [reasoning, mental-model]
---
```

Once a page is written:

```bash
dikw client pages list --layer wisdom        # confirm the page landed
dikw client retrieve "first principles" \
  --format table                             # hit arrives tagged layer=wisdom
dikw client lint                             # broken_wikilink / orphan_page /
                                             # missing_provenance / invalid_wisdom_status
                                             # cover both K and W
```

Wisdom pages can `[[wikilink]]` to knowledge pages (and vice versa);
cross-layer same-title collisions stay broken so `lint` surfaces the
ambiguity — disambiguate with `[[wisdom/elon-musk/be-relentless|Be
Relentless]]`. The wisdom-only `status` enum is validated by the
`invalid_wisdom_status` lint kind (non-blocking warning); the engine
does not yet consume `status` for retrieval filtering or boost.

### Write wisdom programmatically

For agent-driven workflows you can write a wisdom page through the
write API instead of dropping a markdown file by hand. The CLI form:

```bash
dikw client wisdom write \
  --slug first-principles \
  --author elon-musk \
  --title "First Principles" \
  --body "Reason from physics, not analogy." \
  --status published \
  --tag mental-model \
  --source sources/notes/musk-bio.md
```

`--slug` and `--author` must be ASCII kebab-case (`au-thor`, no
spaces / uppercase / underscores) — the path becomes part of the
Obsidian-visible vault layout. Pass `--body-file body.md` to read the
markdown body from a file. `--wait` is the default; `--no-wait` prints
a task handle JSON for async tracking.

Writing the same `(slug, author)` again overwrites the file and
refreshes the row (upsert semantics, same as `lint apply`). Read it
back with `dikw client pages get wisdom/elon-musk/first-principles.md`.
The forward-reference corner — wisdom A linking wisdom B that hasn't
been written yet — surfaces as a non-zero
`unresolved_wikilinks` on the write report; the next `dikw client
wisdom write` for the target page builds out the title index and
reconciles the edge on its own write. If the embedding provider was
transiently down, `chunks_pending_embedding` will be non-zero on the
write report; the next `dikw client ingest`'s cross-layer
missing-embedding resume scan finishes the vector backfill.

> **Replace-on-omit warning.** Every write fully replaces the page's
> frontmatter and provenance edges. Re-running `dikw client wisdom
> write --slug x --title X --body v2` without re-passing `--tag` /
> `--source` / `--status` strips those fields — the typed parameters
> own the on-disk fields, and there is no silent union with the
> existing file. To preserve them on an edit, first `dikw client
> pages get wisdom/<author>/<slug>.md`, then re-pass the values you
> want kept. Empty body is rejected at the schema boundary (422) so
> an accidental `--body ""` cannot wipe an existing page's content.

Upgrading from 0.2.x: delete `wisdom/_candidates/` and the
`wisdom/{principles,lessons,patterns}.md` aggregates (or leave them —
ingest hard-skips both), then re-run `dikw client ingest` against the
new SCHEMA_VERSION. See CHANGELOG `[0.3.0]` for the full migration log.

## 7. Check retrieval quality on your corpus

```bash
# Default: run all packaged datasets (ships with the MVP dogfood corpus).
uv run dikw client eval

# Run against your own corpus: create a 3-file directory and point at it.
uv run dikw client eval --dataset ./my-corpus/
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
  embedding_provider_label: gitee-ai  # optional; shows up in `dikw client check`
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
uv run dikw client check --format table --llm-only    # just LLM (human-readable)
uv run dikw client check --format table --embed-only  # just embedding
uv run dikw client check --format table               # both legs
```

Each variant pings the relevant provider with one tiny request and
reports endpoint / latency / dim/tokens. Drop `--format table` to get
the raw JSON probe report (default, agent-friendly). Exit code is 0 on
success, 1 on any probe failure, 2 when `--llm-only` and `--embed-only`
are passed together. Do this *before* running `dikw client ingest` on a real
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
