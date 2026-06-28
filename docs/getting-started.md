# Getting started

This walkthrough takes a blank directory to a queryable knowledge base in
about five minutes. It only needs Python 3.12+ (plus [`uv`](https://docs.astral.sh/uv/),
or any pip-capable environment); LLM keys are optional until you hit
`dikw client synth` (the engine-internal K-layer authoring leg). Plain
`dikw client retrieve` runs without any LLM key.

## 1. Install and scaffold

There are two ways to get the `dikw` CLI, depending on whether you want to
**use** dikw-core or **work on** it.

### Option A — install the published package (most downstream systems)

dikw-core ships to [PyPI](https://pypi.org/project/dikw-core/) on every
release. Install the wheel into any environment and the `dikw` command lands
on your PATH:

```bash
uv pip install 'dikw-core[postgres]'   # or: pip install 'dikw-core[postgres]'
dikw init my-base --description "my research base"
cd my-base
```

Pin an exact version (`uv pip install 'dikw-core[postgres]==0.6.4'`) so your
client stays on the same release as the server it talks to — `dikw client`
runs a [version handshake](server.md) and hard-fails on a mismatch.

### Option B — develop from a checkout (contributors)

```bash
git clone https://github.com/OpenDIKW/dikw-core
cd dikw-core
uv sync --all-extras    # installs every extra + the dev group

# Pick any directory — `my-base/` below — the server will manage it as a dikw base.
uv run dikw init ../my-base --description "my research base"
cd ../my-base
```

> The rest of this walkthrough spells commands as `uv run dikw …`, the
> checkout (Option B) form. If you installed the wheel (Option A), drop the
> `uv run` prefix and call `dikw …` directly.

### Optional extras

The base install is deliberately dependency-light — SQLite + `sqlite-vec` +
FTS5, no extras required. Three opt-in extras add capabilities; install the
ones your deployment needs:

| extra | pulls in | install when you… |
| --- | --- | --- |
| `postgres` | `psycopg[binary,pool]`, `pgvector` | run the multi-user Postgres backend (`storage.backend: postgres`) instead of the default SQLite. Without it: SQLite only. |
| `cjk` | `jieba` | ingest Chinese (or other CJK Han) text. The default config already selects `retrieval.cjk_tokenizer: jieba`, so **without this extra, ingesting CJK text fails with an `ImportError`** that points you here — install `[cjk]`, or set `retrieval.cjk_tokenizer: "none"` in `dikw.yml` to fall back to single-char tokenization (BM25 recall on CJK collapses). ASCII-only corpora never hit this path. |
| `otel` | OpenTelemetry SDK + OTLP/HTTP exporter + FastAPI/httpx/logging instrumentation | export traces / metrics / logs to an OTLP backend — see [`observability.md`](observability.md). Without it: no-op telemetry. |

Combine them in one install: `uv pip install 'dikw-core[postgres,cjk,otel]'`
(checkout flow: `uv sync --all-extras`).

The init command creates this tree:

```text
my-base/
├── dikw.yml              # the config the engine reads on every command
├── sources/              # your raw documents go here (Data layer)
├── knowledge/            # LLM-authored knowledge pages, filed under the category tree
│   └── {entity,concept,note}/   # the default taxonomy — redeclare via schema.categories
├── prompts/              # optional per-base prompt overrides (synth.prompt_path / lint.fixer_prompts)
│   └── .gitkeep
├── wisdom/               # hand-written principles / lessons / patterns (you author these
│   └── .gitkeep          # by hand or via `dikw client wisdom write`)
└── .dikw/                # engine state (gitignored)
    └── .gitkeep          # index.sqlite is created here on first ingest/serve
```

The whole tree is the **dikw base**; the `knowledge/` subdirectory is just
the K-layer slice. The server manages this tree, but it stays open Markdown —
open the folder in any Markdown editor and the knowledge + wisdom pages render
as a plain `[[wikilink]]` + YAML-front-matter tree.

## 2. Start the server

`dikw-core` runs as a long-lived process; the CLI is a thin client that
talks HTTP + NDJSON to it. Start the server bound to your base in a
spare terminal (or under a process supervisor):

```bash
uv run dikw serve --base .
# bound to http://127.0.0.1:8765 — no auth on loopback
```

Leave it running. Every `dikw client <op>` shown below routes through
this server — top-level short aliases (a bare `status` instead of
`dikw client status`) were removed in 0.1.0, so always spell out the
`client` prefix.

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

# Or with embeddings (requires the env var named by
# ``provider.embedding_api_key_env`` in dikw.yml — e.g. OPENAI_API_KEY for
# OpenAI, GITEE_API_KEY for Gitee AI — on any OpenAI-compatible endpoint:
# OpenAI, Gitee AI, Ollama, vLLM, …).
export OPENAI_API_KEY=sk-...
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

Markdown sources that embed images — either `![alt](path)` or the wiki-style
`![[assets/foo.jpg]]` embed — flow through ingest into a content-addressed
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
not rewritten — so the same response round-trips a hand-edited
Markdown tree without divergence. Clients render images by mapping
each `assets[].original_paths` entry back to `assets[].url`.

Asset responses are immutable (content-addressed by SHA-256), so the
route emits a strong `ETag` + `Cache-Control: public, max-age=31536000, immutable`;
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
uv run dikw client pages provenance knowledge/concept/topic.md \
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

# Self-check this run's pages and exit non-zero if they aren't clean
# (implies --wait). Runs a scoped lint + persist check + semantic-
# duplicate gate over only the pages this run created/updated.
uv run dikw client synth --verify

# Add the report-only grounding leg: have the LLM score whether this
# run's claims are supported by the sources they cite (implies --verify).
uv run dikw client synth --verify --judge
```

`--verify` is the "open the pages and click around" pass made automatic: after
synth writes the K pages, it runs a deterministic, no-extra-LLM check over just
this run's output and prints one PASS/FAIL verdict. Three legs gate the
verdict — **persist** (no page was deactivated mid-write), **lint** (no
`broken_wikilink` / `duplicate_title` / `non_atomic_page` / `uncategorized` /
`missing_provenance` / `title_slug_quality` on the new pages), and **duplicate** (the semantic
near-duplicate ratio over this run's page bodies stays under
`synth.verify_max_duplicate_ratio`, default `0.05`, using cosine tau
`synth.verify_duplicate_cosine_tau`, default `0.85`). Orphan pages are surfaced
but **not** gated — a freshly synthesised page is legitimately orphan until
something cites it. The duplicate leg needs embeddings; when none are available
(no embedder configured, `--no-embed`, or no active embed version yet) — or when
its embed pass itself fails mid-leg — it is **skipped loudly** (a warning, not a
silent pass and never a failed task; every page is already persisted by then) so
a green verdict never reads as "no duplicates" when the check never ran.

Adding `--judge` (which implies `--verify`) runs one more, **report-only** leg:
it samples this run's claims (`synth.verify_judge_sample`, default `25`), pairs
each with the source chunk that best matches it, and asks the synth LLM whether
the evidence entails the claim — printing an entailment ratio + 95% CI. This leg
is the one probabilistic check the others can't make ("are these claims actually
backed by their sources?"), so it is **never** folded into the PASS/FAIL verdict:
an LLM judge is noisy, and the call over the ratio belongs to the agent or skill
driving synth, not a hard CLI gate. It needs both an embedder and an LLM; when
either is missing (or the leg errors) it is **skipped loudly** — never a silent
zero.

Cost notes: the lint leg runs a **full-base** scan (it has to, so wikilinks
resolve against every page) and then filters to this run's pages, so its cost
scales with total knowledge-base size, not with how many pages this run produced; the
duplicate leg performs a second embed pass over the just-written page bodies; and
`--judge` adds a grounding-embed pass plus up to `verify_judge_sample` LLM judge
calls. All only run when you pass the corresponding flag.

The LLM reads each source doc and produces a `knowledge/<category>/<slug>.md`
page, cross-linked via `[[wikilinks]]`. The `<category>` is chosen from the closed
taxonomy you declared in `schema.categories` (default `entity`/`concept`/`note`); a
page the model can't confidently place lands in `schema.fallback` (default
`未分类/`). No `index.md` / `log.md` is generated — the category folder tree is the
catalogue and the `knowledge_log` table is the history (see ADR-0004). Re-running
is a no-op until you add new sources (or pass `--all` to resynthesise everything).

Run `dikw client lint --format table` to check the K-layer for broken wikilinks, orphans, duplicate titles, non-atomic pages, missing provenance edges, pages stranded in the `uncategorized` fallback bucket, and `title_slug_quality` issues (a page with no usable `# Title`, a frontmatter `title:` that disagrees with the body heading, or a degenerate `untitled` filename slug) (the default output is agent-facing JSON; add `--format table` for the human view). For `missing_provenance` issues on a legacy base, backfill in one shot via `dikw client lint propose --rule missing_provenance` then `dikw client lint apply <task_id>` — heuristic-only, no LLM required.

### Customizing the knowledge taxonomy

The default `entity`/`concept`/`note` split is just the taxonomy a fresh base
ships with. Declare your own closed-set, hierarchical tree under
`schema.categories` in `dikw.yml` — paths may use any Unicode (Chinese folder
names land on disk verbatim):

```yaml
schema:
  description: Acme internal knowledge base
  categories:
    - path: 产品/移动端
      desc: 移动端 App 产品相关
    - path: 技术/架构
      desc: 架构与系统设计
    - path: 技术/数据
  fallback: 未分类          # where synth files a page it can't confidently place
```

Synth's LLM may only file a page under a declared `path`; a page filed under
`技术/架构` lands at `knowledge/技术/架构/<slug>.md` with `category: 技术/架构`
in its frontmatter. Anything unplaceable goes to `knowledge/未分类/` and is
flagged by the `uncategorized` lint so a human can re-file it.

You can also override the K-layer authoring prompts per base — point
`synth.prompt_path` (and optionally `lint.fixer_prompts.{orphan_merge,broken_wikilink}`)
at your own markdown under `<base>/prompts/`. Overrides must stay inside the base
and carry the engine's required `{placeholders}`, the `<page …>` output markers,
and (for `synthesize`) the `## Knowledge-base context` heading its dynamic
sections nest under; `dikw client check` validates all of it up front. The
override is also the designed channel for taxonomy customization — a base with
its own category tree can rewrite the template's worked examples so their
`category="…"` values come from its declared paths:

```yaml
synth:
  prompt_path: ./prompts/my_synth.md
lint:
  fixer_prompts:
    orphan_merge: ./prompts/orphan.md
```

Changing the taxonomy or prompts of a base that already has synthesized pages
means re-running `dikw client synth --all` to rebuild under the new rules. The
drift-reindex lint (`stale_index` / `untracked_file`, see §6) re-projects a
page's *current* bytes as-is — it does not re-classify or regenerate — so it is
not a substitute for re-synth when the taxonomy itself changes.

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
DIKW_LOG_LEVEL=DEBUG uv run dikw serve --base $BASE
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
through `dikw client wisdom write`** (CLI) or `POST /v1/base/wisdom`
(HTTP) — `dikw client ingest` does NOT scan `<base>/wisdom/`. The
write API takes structured input (slug + title + body + optional
metadata) and runs the same `persist_wisdom` pipeline a manual edit
would have triggered in 0.3.x. Hand-edits to a wisdom file on disk
are not auto-reindexed *live*, but the `stale_index` drift lint
(ADR-0005) detects them — `dikw client lint apply` re-projects the
edited bytes, or re-run `dikw client wisdom write` with the edited
body. (The same reconciliation applies to hand-edited / hand-written
K-layer pages.)

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
on-disk layout. Pass `--body-file body.md` to read the
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

### Deleting a document

Remove any registered document — a source, a knowledge page, or a wisdom
page — by path:

```bash
dikw client delete knowledge/concept/outdated-note.md
dikw client delete wisdom/elon-musk/draft.md --reason "superseded"
```

`delete` purges the document's storage row and its **outgoing** links +
provenance, then soft-deletes the file to `<base>/trash/<layer>/<rel>` with
a `trashed:` audit block. It is immediate (no propose/apply — `trash/` is
the safety net) and `--wait` by default; pass `--reason` to stamp an audit
note. The report's `inbound_broken` count tells you how many live pages now
have a dangling `[[wikilink]]` to the page you just deleted.

To recover, move the file back into place
(`mv <base>/trash/knowledge/... <base>/knowledge/...`). A **D-layer source**
re-indexes on the next `dikw client ingest`. A recovered **K or W page** is
re-indexed by the `untracked_file` drift lint (the restored file has no active
row again):

```bash
dikw client lint propose --rule untracked_file   # returns a task id
dikw client lint apply <task_id>                 # re-projects the file into storage
```

This re-projects the file's bytes as-is (preserving them); it does not
re-run synth. (You can still rebuild a knowledge page from its source with
`dikw client synth --all`, or re-author a wisdom page with
`dikw client wisdom write`, if you want a regenerated version instead.)

Inbound `[[wikilink]]`s from *other* live pages are left dangling on
purpose — they surface as `broken_wikilink` on the next `dikw client lint`
(and in the delete report's `inbound_broken` count), because silently
rewriting another page's body to drop the link would hide the breakage.
This verb is the way to delete an arbitrary page; the `lint` fixers only
auto-delete empty stubs and merged duplicates.

If you delete a file **outside** dikw (e.g. `rm` it, or remove it in
an external editor) the document row is left behind, stuck `active`. The default
`lint` scan flags it as **`missing_file`** (D/K/W); clean it up with:

```bash
dikw client lint propose --rule missing_file   # returns a task id
dikw client lint apply <task_id>               # purges the orphaned row(s)
```

This is the passive complement to `delete` (which trashes a *live* file):
`missing_file` purges the stale row whose file is *already* gone. Same edge
policy — inbound links from live pages become `broken_wikilink`, never
silently rewritten.

When the gone file was a **source** that a knowledge or wisdom page cites in
its `sources:` front-matter, that page's provenance edge is now **dangling**.
The default `lint` scan flags it as **`dangling_provenance`** (K/W) — a
**read-only** warning naming the missing source. There is no fixer: the
`sources:` front-matter is yours to edit (the engine never rewrites your
content), so fix it by hand (drop or re-point the entry) and the flag clears
on the next `lint`. A source present on disk but not yet `ingest`-ed is *not*
flagged — there the fix is `ingest`, not a front-matter edit.

## 7. Check retrieval quality on your corpus

```bash
# Default: run all packaged datasets (ships with the MVP dogfood corpus).
uv run dikw client eval

# Run against your own corpus: create a 3-file directory and point at it.
uv run dikw client eval --dataset ./my-corpus/
```

Each query is marked a "hit" at top-k if any `expect_any` doc stem is in
the top-k result. Metrics: `hit@3`, `hit@10`, `MRR`. Exit code 0/1/2.

### Gate a run against a committed baseline

Pin a known-good run's metrics, commit the JSON, then fail CI on any
regression beyond a tolerance:

```bash
# Capture the current run's metrics as a baseline (single --dataset + one --eval mode).
uv run dikw client eval --dataset mvp --eval synth --write-baseline evals/baselines/mvp-synth.json

# Later, gate a fresh run against it — exit 1 on regression (implies --wait).
uv run dikw client eval --dataset mvp --eval synth --against evals/baselines/mvp-synth.json
```

The comparison is **direction-aware** (a `_max` metric like
`synth/fallback_ratio_max` regresses when it *rises*) and uses the baseline's
`tolerance` field (default `0.02`). It is a single-run regression *gate*, not an
A/B significance test — keep the tolerance tight for deterministic retrieval
evals and generous for LLM-driven synth evals (so model jitter doesn't trip it).
The statistical A/B path (Welch t-test over sample distributions) lives in
[`evals/tools/ab_experiment.py`](../evals/tools/ab_experiment.py). See
[`evals/baselines/README.md`](../evals/baselines/README.md) for the file format.

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
  llm_api_key_env: OPENAI_API_KEY    # required: names the env var holding the LLM key
  embedding: openai_compat
  embedding_model: text-embedding-3-small
  embedding_base_url: https://api.openai.com/v1
  embedding_api_key_env: OPENAI_API_KEY  # required: names the env var holding the embedding key
  embedding_dim: 1536          # required: must match what the endpoint returns
  embedding_revision: ""       # bump to force re-embed when vendor refreshes weights silently
  embedding_normalize: true
  embedding_distance: cosine
```

Each leg names its own env var via `llm_api_key_env` / `embedding_api_key_env`,
so keys stay vendor-canonical (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`DEEPSEEK_API_KEY`, `MINIMAX_API_KEY`, `GITEE_API_KEY`, …) and multiple
same-protocol vendors (DeepSeek + MiniMax both speak the Anthropic protocol)
coexist in one `.env` — each base picks which var it reads in `dikw.yml`.

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
  llm_api_key_env: MINIMAX_API_KEY  # required: MiniMax gets its own env var
  embedding: openai_compat
  embedding_model: Qwen3-Embedding-0.6B
  embedding_base_url: https://ai.gitee.com/v1
  embedding_api_key_env: GITEE_API_KEY  # required: Gitee gets its own env var
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

The two legs use **distinct keys**. Each leg reads exactly the env var named
in `dikw.yml` (`llm_api_key_env: MINIMAX_API_KEY`, `embedding_api_key_env:
GITEE_API_KEY`), so the keys never cross-wire and a misconfigured name fails
loudly:

```bash
export MINIMAX_API_KEY=<your-MiniMax-key>
export GITEE_API_KEY=<your-Gitee-key>
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
