# Providers

How to pick, configure, and switch LLM / embedding vendors in `dikw-core`.
If you're adding a new vendor, start here before touching code — usually
you don't need to touch any.

## The design seam

`dikw-core` ships with three protocol-level providers, all pluggable via
`dikw.yml` alone. The `llm` field names the **protocol** (which SDK
to speak), not the vendor — vendor is whatever `llm_base_url` points at:

- **`anthropic_compat`** — uses the official `anthropic` async SDK.
  `llm_base_url` retargets it at any Anthropic-protocol-compatible
  endpoint (e.g., MiniMax's `https://api.minimaxi.com/anthropic`).
  Applies `cache_control: ephemeral` on the system prompt, so repeated
  synth calls within the 5-minute TTL hit the prompt cache.
  Leave `llm_base_url` null to talk to api.anthropic.com directly.
- **`openai_compat`** — uses the `openai` async SDK against any
  `llm_base_url` that speaks the OpenAI HTTP surface. Covers OpenAI,
  Azure, Ollama, vLLM, TEI, DeepSeek, GLM, Gemini (OpenAI-compat mode),
  Gitee AI, and most others.
- **`openai_codex`** — uses the `openai` async SDK against the **OpenAI
  Responses API** (not Chat Completions) hosted at the ChatGPT backend.
  Targets the codex model family (`gpt-5.5` / `gpt-5.4-mini` /
  `gpt-5.3-codex` / …) which lives only on `chatgpt.com/backend-api/codex`
  and isn't reachable through the public `api.openai.com`. Authenticates
  with a ChatGPT OAuth `access_token` loaded from `<base>/.dikw/auth.json`
  (dikw's self-managed token store — separate from codex CLI's
  `~/.codex/auth.json`). Dikw refreshes it before each call when it's near
  expiry and writes the rotated tokens back. **No `OPENAI_API_KEY`
  involved.** `llm_base_url` is required (no SDK default exists); a
  `ProviderConfig` validator enforces this at config load. First-time
  bootstrap is one of: `dikw auth login openai-codex` (device-code flow,
  no codex CLI required), `dikw auth import openai-codex` (one-shot copy
  from `~/.codex/auth.json`), or automatic lazy migration on first use
  if `~/.codex/auth.json` already exists.

`anthropic_compat` and `openai_compat` cover most vendors. **`openai_codex`
is the dedicated path for ChatGPT-only models** — the wire shape, auth
mechanism, and required Cloudflare headers all diverge from
`openai_compat`, which is why it's a sibling protocol rather than an
`openai_compat` base_url variant.

## Vendor cookbook

Tested / known-compatible combinations. URLs may evolve — always
cross-check the vendor's own docs.

| Vendor | `llm` | `llm_base_url` | `embedding` | `embedding_base_url` | LLM key env | Embed key env |
|---|---|---|---|---|---|---|
| **OpenAI** (default) | `openai_compat` | `https://api.openai.com/v1` | `openai_compat` | same | `OPENAI_API_KEY` | `DIKW_EMBEDDING_API_KEY` |
| **OpenAI Codex** (GPT-5 series) | `openai_codex` | `https://chatgpt.com/backend-api/codex` *(required)* | *(no embed — pair elsewhere)* | — | *OAuth via `<base>/.dikw/auth.json` — bootstrap with `dikw auth login openai-codex`* | — |
| **Anthropic** | `anthropic_compat` | leave `null` | *(no embed — pair elsewhere)* | — | `ANTHROPIC_API_KEY` | — |
| **MiniMax** | `anthropic_compat` | `https://api.minimaxi.com/anthropic` | *(no embed — pair elsewhere)* | — | `ANTHROPIC_API_KEY` | — |
| **GLM / 智谱** | `openai_compat` | `https://open.bigmodel.cn/api/paas/v4` | `openai_compat` | same | `OPENAI_API_KEY` | `DIKW_EMBEDDING_API_KEY` |
| **Gemini** | `openai_compat` | `https://generativelanguage.googleapis.com/v1beta/openai/` | `openai_compat` | same | `OPENAI_API_KEY` | `DIKW_EMBEDDING_API_KEY` |
| **DeepSeek** | `openai_compat` | `https://api.deepseek.com/v1` | *(no embed — pair elsewhere)* | — | `OPENAI_API_KEY` | — |
| **Gitee AI** | *(often paired as embed only)* | — | `openai_compat` | `https://ai.gitee.com/v1` | — | `DIKW_EMBEDDING_API_KEY` |
| **Ollama / vLLM / TEI** (local) | `openai_compat` | `http://localhost:<port>/v1` | `openai_compat` | same or localhost | `OPENAI_API_KEY` (any non-empty) | `DIKW_EMBEDDING_API_KEY` (any non-empty) |

**Reference configs** (committed in this repo):
- [`tests/fixtures/live-minimax-gitee.dikw.yml`](../tests/fixtures/live-minimax-gitee.dikw.yml)
  — MiniMax LLM + Gitee AI embeddings. Drop-in for a fresh base.

Add more fixtures over time as you verify combinations; PRs welcome.

## Switching procedure

1. **Edit `dikw.yml`** — replace the `provider:` block with the target
   vendor's values from the cookbook above. Don't forget
   `embedding_dim` and `embedding_batch_size` if the new embedder
   differs from the old (see gotchas).
2. **Update `.env`** with the new key values. Variable *names* don't
   change — only their *values*:
   - `ANTHROPIC_API_KEY` → LLM key (Anthropic or MiniMax).
   - `OPENAI_API_KEY` → LLM key (OpenAI, Azure, Ollama, GLM, Gemini, …).
   - `DIKW_EMBEDDING_API_KEY` → embedding key (same or different vendor).
3. **If the embedding model dim changed**, delete `.dikw/index.sqlite`
   (see gotcha #1). Reingestion is required.
4. **Verify** — `dikw client check` talks to a running server, so use
   `serve-and-run` for one-shot probes (it spawns a temporary `dikw
   serve` against `<base>`, runs the inner check, tears it down):
   ```bash
   uv run --env-file .env dikw client serve-and-run --base <base> -- check --llm-only
   uv run --env-file .env dikw client serve-and-run --base <base> -- check --embed-only
   ```
   Each variant pings one endpoint with one tiny request; failures
   print the error inline. Exit 0/1 is scriptable.
5. **`uv run dikw client ingest`** to re-populate the I layer if you wiped the
   SQLite file.

## Production gotchas

In order of likely bite. Not blocking for experimentation, but worth
knowing before committing to a vendor on a real corpus.

### 1. Embedding dimensions are locked at first write

SQLite `vec0` locks vector dimension on the first `upsert_embeddings`
call. Changing `embedding_dim` or `embedding_model` to a
different dim afterwards produces a dimension-mismatch error on the
next insert.

**Only safe migration path today:** delete `.dikw/index.sqlite`, then
`dikw client ingest` fresh. There is no incremental re-embed.

### 2. Batch size varies per vendor

Default `embedding_batch_size: 64` is safe for OpenAI (~2048 cap) but
violates other vendors:

| Vendor | Observed cap | Recommended config |
|---|---|---|
| OpenAI | ~2048 | default (64) |
| Gemini | 100 | default (64) |
| GLM | unverified | start with 32 |
| **Gitee AI** | **25** | `embedding_batch_size: 16` |

Exceeding the cap typically returns HTTP 400. Gitee AI emits
`"No schema matches, </input>"`.

### 3. Streaming completions, per-event timeout, and classified retry

**LLM completions stream internally.** Both `anthropic_compat` and
`openai_compat` implement `complete()` as a collapse of `complete_stream()`
(iterate the SSE event stream, return the terminal assembled text/usage). The
reason is timeouts on **reasoning models**: a non-streaming call bounds the
*whole response* by the read timeout (`llm_timeout_seconds`, default 120 s), so
a model like MiniMax-M3 that spends minutes generating a 16k-token synthesis
times out mid-receipt. Streaming makes the read timeout apply **per SSE event**
instead — each token / thinking / keepalive frame resets the deadline, so a
steadily-streaming generation never trips it regardless of total length.
(Measured on MiniMax-M3: it emits a `thinking_delta` every ~1–2 s throughout
its hidden chain-of-thought, max inter-event gap ~2 s — far under any sane
`llm_timeout_seconds`. The hidden reasoning is *not* a silent gap.)

**Exception classification.** A streamed completion that fails is sorted into
two buckets, mirroring the embedding leg:

| SDK failure | class | synth behavior |
|---|---|---|
| timeout, connection drop, 5xx, 408, 429 | `TransientProviderError` | retried by the synth group loop |
| 401 / 403 / 404 / other 4xx, bad model, missing key | `ProviderError` (permanent) | fails fast — never retried-then-skipped |

`asyncio.CancelledError` propagates untouched (never reclassified), so a cancel
mid-synthesis still honors the per-group cancel contract.

**Two-tier retry.** The SDK's own `max_retries` (`llm_max_retries`, default 5)
covers connection-phase failures on the initial request with exponential
backoff; the engine-level synth loop (`synth.provider_error_retries`, default 2
→ 3 attempts; linear `provider_error_retry_backoff_seconds`) re-issues the
*whole* `complete()` call on a `TransientProviderError` that surfaced mid-stream
(the SDK cannot resume a partially-drained stream). For a genuinely slow-to-
first-byte gateway these can compound — budget worst-case wall-clock at roughly
`llm_max_retries × synth.provider_error_retries` full timeouts when tuning
`llm_timeout_seconds`. **Known limitation:** the per-event timeout replaces the
old whole-response cap, so a pathological gateway that trickles a keepalive
faster than `llm_timeout_seconds` but never completes would stall (a fully
silent gateway still trips the per-event deadline). Real endpoints (MiniMax,
OpenAI, vLLM) don't exhibit this.

`dikw client check` prints provider failures as red cells; `dikw client ingest`
is idempotent via content hash, so re-running resumes without double-embedding
unchanged docs.

### 4. Prompt caching only on the `anthropic_compat` leg

The `AnthropicCompatLLM` provider passes `cache_control: {"type": "ephemeral"}`
on the system prompt, cutting repeat-call input-token cost by ~90%
within a 5-minute TTL. Synth benefits. (Retrieve never calls the LLM —
answer synthesis is the agent's job — so prompt caching is irrelevant on
the read path.)

**`openai_compat` does not expose prompt caching** — GLM / Gemini /
DeepSeek pay full price on every call even with a stable system
prompt. If you plan heavy synth work on an `openai_compat` vendor, the
cost model is different from the Anthropic leg.

### 5. `max_tokens` is per-op, configurable via `dikw.yml`

Default (in [`config.py`](../src/dikw_core/config.py)):
`provider.llm_max_tokens_synth = 3072` — ~768 tokens per page at the
default `max_pages_per_group=4`, so a full fan-out group rarely truncates
mid-page (the old 2048 default left only ~512/page and clipped dense
groups). Some cost-optimized models (a few GLM-Flash variants, smaller
Gemini Nano endpoints) cap responses below 2048 and return 400 — shrink
the field for those. **Reasoning models need much more**: their hidden
chain-of-thought also draws on this budget, so MiniMax-M3 and similar
silently emit zero pages at 3072 — set `>= 8192` (16384 recommended).
A budget cutoff no longer strands content silently: synth reads the
provider `finish_reason` (`"length"` on `openai_compat`/`openai_codex`,
`"max_tokens"` on `anthropic_compat`) and, when it signals truncation,
withholds the `synth_source_done` marker so a re-run with a bigger budget
recovers the dropped pages (the destructive `non_atomic_page` split fixer
refuses outright). Raising `llm_max_tokens_synth` is still the fix — this
just makes an under-budget run loud and recoverable instead of lossy.
Override per-base by adding the field to your `dikw.yml` `provider:`
block — no code change needed. There is no `llm_max_tokens_query` knob;
`retrieve` doesn't call an LLM, so the read-path budget lives on the
agent side. (The 0.2.x `llm_max_tokens_distill` knob was removed in
the 0.3.0 W refactor; pydantic ignores it if still present in your yml.)

`dikw client check`'s LLM connectivity probe also hands the model this
same `llm_max_tokens_synth` budget (not a tiny fixed cap), so a green
check predicts the synth path: a reasoning model whose budget is too
small to clear its own chain-of-thought returns an empty completion and
the probe reports it as **down** rather than green — fix the budget here,
then re-check.

### 6. Two separate keys, on purpose

The embedding leg reads `DIKW_EMBEDDING_API_KEY` **exclusively** — no
silent fallback to `OPENAI_API_KEY`. When LLM and embedding point at
the same OpenAI-compat vendor, you still set two env vars to the same
value. This looks redundant but prevents cross-wiring when the two
legs diverge (the common case — MiniMax LLM + Gitee embeddings, or
Anthropic LLM + OpenAI embeddings).

### 7. CJK corpora need `cjk_tokenizer: jieba`

SQLite FTS5's default `unicode61` tokenizer splits CJK one character
at a time, which collapses BM25 on Chinese / Japanese into single-char
IDF. Measured on CMTEB T2Retrieval: **nDCG@10 = 0.031**, 91.7% of
queries zero-recall — vs ≈ 0.5–0.65 on the published Anserini+jieba
baselines. The default is now `jieba`, so a fresh base picks up the
right tokenizer with no config; pin it explicitly only if you need
the legacy whitespace path:

```yaml
retrieval:
  cjk_tokenizer: jieba       # default since 2026-04 (was "none")
```

…and install the optional extra:

```bash
uv sync --extra cjk          # pulls in jieba ≥ 0.42
```

The preprocessor runs `jieba.cut_for_search` over **CJK runs only** —
ASCII identifiers (``retrieval.rrf_k``, code snippets, …) are passed
verbatim, so mixed English/Chinese dev docs don't get their English
halves shredded.

**Locked at first ingest** — same shape as `embedding_dim`
(gotcha #1). The `documents_fts` rows store whatever segmentation was
in effect when they were written; flipping the config afterwards
produces a mismatch between indexed and queried tokens, silently
dropping CJK hits. To change: wipe `.dikw/index.sqlite` and `dikw client ingest` fresh.

### 8. `openai_codex` self-manages its OAuth tokens (separate from codex CLI)

The codex protocol differs from the other two on every axis worth
flagging — keep these in mind before flipping `llm: openai_codex`:

- **Self-managed token store at `<base>/.dikw/auth.json`.** Dikw keeps its
  own copy of the access_token + refresh_token, separate from codex CLI's
  `~/.codex/auth.json`. Each base owns its own credentials. The store
  follows a multi-provider schema (`{"version":1,"providers":{...}}`) so
  future OAuth providers (e.g. anthropic) can sit alongside.
- **Why a separate store: refresh_token rotation.** ChatGPT's OAuth issuer
  mints a fresh refresh_token on every successful refresh and immediately
  invalidates the old one. If two clients write the same auth file (codex
  CLI + dikw, or hermes + dikw), whichever client refreshes second is
  silently logged out — its refresh_token has just been revoked by the
  first client. Independent stores give each client its own
  refresh_token to rotate.
- **Bootstrap paths** (use whichever fits):
  - `dikw auth login openai-codex` — full device-code OAuth flow inside
    dikw. Doesn't depend on codex CLI being installed.
  - `dikw auth import openai-codex` — one-shot copy from
    `~/.codex/auth.json` (override source via `$CODEX_HOME`). Useful if
    you've already run `codex` and want dikw to inherit that session.
  - **Lazy migration** — the first time dikw needs a token after upgrade,
    if `<base>/.dikw/auth.json` is missing but `~/.codex/auth.json`
    exists with a non-expired access_token, dikw imports it transparently
    and logs a one-line message to stderr telling you it happened. From
    that point on dikw never writes to `~/.codex/auth.json` again, so
    codex CLI and dikw can run side-by-side without colliding.
- **No `OPENAI_API_KEY` involved.** Refresh runs against
  `https://auth.openai.com/oauth/token` using a hard-coded codex CLI
  client_id (the public app identifier).
- **`llm_base_url` is required.** No SDK default exists for the ChatGPT
  backend. The `ProviderConfig` validator rejects `llm: openai_codex` +
  `llm_base_url: null` at config load with a message telling you what to
  paste. Override only if you front the protocol with a custom gateway.
- **gpt-5.5 / gpt-5.4-mini / gpt-5.3-codex are ChatGPT-only.** They are
  not exposed at `api.openai.com`; pointing `llm_base_url` at the public
  OpenAI API will return `model_not_found`.
- **No prompt caching.** Repeated synth calls within the same session
  pay full input-token cost — same caveat as `openai_compat`, unlike
  `anthropic_compat`'s `cache_control: ephemeral`.
- **SDK reducer-bug compatibility shim.** The ChatGPT codex backend
  ships `response.output = None` in its terminal `response.completed`
  payload, while the public OpenAI Responses API ships a list. The
  `openai` Python SDK's high-level `responses.stream(...)` reducer dies
  on this with `TypeError: 'NoneType' object is not iterable`. The dikw
  provider catches this exact signature (and the matching
  `AttributeError("attribute 'output'")` form) and falls back to the
  locally-collected delta text — `finish_reason="error"` distinguishes
  the recovered partial from a clean completion. If zero deltas arrived
  before the reducer fired (auth / quota / refusal failures), the
  provider raises `ProviderError` instead of returning an empty
  response, so synth doesn't silently drop a source page.
- **Empty-final-output recovery (issue #160).** A *different* codex
  backend quirk: the terminal `response.completed` sometimes ships
  `output = []` (an empty **list**, not `None` — so the reducer above
  does *not* fire) even though valid `response.output_text.delta`
  events already streamed the full answer. The SDK hands back a
  well-formed final `Response`, so the provider used to trust it and
  return `text=""`, discarding a complete completion (observed live:
  every synth group returning `response_chars: 0` while the model had
  actually produced `<page>` blocks). The provider now falls back to
  the streamed delta text when the final carries **no output items at
  all** but deltas did arrive. This is deliberately narrow: a final
  holding an explicit empty *message* item (`output=[message("")]` — a
  non-empty list, a real cleared turn) still surfaces as `text=""`.
- **`synth` retries `ProviderError` per group, then skips.** Since
  0.4.0, a single `ProviderError` (the empty-response case above, or
  any other provider failure) on one source group no longer aborts the
  whole `synth` task. The engine retries the call up to
  `synth.provider_error_retries` times (default `2`) with linear
  backoff (`synth.provider_error_retry_backoff_seconds`, default
  `2.0` → `2s` / `4s`), then skips the group and continues with the
  next. The skip is counted as a parse-style error so
  `synth_source_done` is **not** written and a follow-up `dikw
  client synth` will retry the flaky group from scratch. NDJSON
  consumers see new per-group `status="retrying"` and
  `status="skipped"` events. Setting `provider_error_retries: 0`
  disables retries (one attempt only), but the group is still
  *skipped* on failure rather than re-raised — by design, since the
  fix exists precisely so that one bad group cannot abort the task.
- **Reasoning fragments are dropped today.** dikw's `LLMStreamEvent`
  Protocol carries a `reasoning` event type and the codex provider emits
  it for `response.reasoning_summary_text.delta` events, but the
  synth NDJSON renderer only forwards `token` / `done`. Switch to
  reasoning models freely — the chain-of-thought just isn't surfaced to
  the user yet (a follow-up PR will add a `--show-reasoning` toggle).
- **`$CODEX_HOME` is consulted only by `dikw auth import`** as the source
  path. Dikw does not write to that location. The dikw store path
  follows the base — multi-base setups carry independent credentials
  (copy `<old-base>/.dikw/auth.json` to `<new-base>/.dikw/auth.json` to
  share, or run `dikw auth login` per base).
- **Recovery from a stolen / revoked refresh_token.** When you see
  `relogin_required` (e.g., another client rotated the token, or you
  manually revoked the session), re-run `dikw auth login openai-codex`.

## Public-benchmark calibration with Gitee AI

Reproducible workflow for running BEIR / CMTEB benchmarks against
dikw's hybrid retriever, using Gitee AI for embeddings (its free /
low-cost tier makes the 5K–60K passage runs financially trivial). The
benchmark datasets and the converter scripts live under `evals/`; the
runner is the same `dikw client eval` you use on the dogfood mvp set.

Setup once:

```bash
uv run dikw init scratch-bench-base
cd scratch-bench-base
# Edit dikw.yml's provider block:
#   embedding: openai_compat
#   embedding_base_url: https://ai.gitee.com/v1
#   embedding_model: Qwen3-Embedding-0.6B
#   embedding_dim: 1024               # 0.6B native dim
#   embedding_revision: ""
#   embedding_normalize: true
#   embedding_distance: cosine
#   embedding_batch_size: 16          # gotcha #2 — Gitee caps at 25
# Then in .env: DIKW_EMBEDDING_API_KEY=<your gitee-ai key>
uv run --env-file .env dikw client serve-and-run --base . -- check --embed-only
```

Materialise SciFact (BEIR English, 5K passages, ~1 minute on Gitee AI):

```bash
curl -L https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip \
    -o /tmp/scifact.zip
unzip /tmp/scifact.zip -d /tmp/scifact-src/
uv run python evals/tools/convert_beir.py \
    --source /tmp/scifact-src/scifact/ \
    --out evals/datasets/scifact/ \
    --baseline-bm25-ndcg10 0.665    # BEIR paper, Thakur et al. 2021
```

Run the full ablation (the served base's configured embedder is used
automatically — no flag needed):

```bash
uv run --env-file scratch-bench-base/.env \
    dikw client serve-and-run --base ./scratch-bench-base -- \
    eval --dataset scifact --retrieval all
```

Output is a 3-row × 5-metric table (bm25 / vector / hybrid × hit@3 /
hit@10 / mrr / nDCG@10 / recall@100). Compare the `bm25` row's
nDCG@10 to the published 0.665 baseline (treat ±0.10 as in-band —
FTS5 is not Anserini), and check whether `hybrid` actually beats both
single legs.

### Tuning RRF weights for your corpus

The shipped defaults are calibrated on BEIR/SciFact (vector-heavy:
`bm25_weight=0.3, vector_weight=1.5, rrf_k=60`). If your corpus has a
different BM25 / dense balance — keyword-heavy code bases want more
BM25 influence, paraphrase-heavy prose wants more vector — tune by
editing the `retrieval:` block in the served base's `dikw.yml` and
re-running `dikw client eval --dataset <name> --retrieval all` to
compare:

```yaml
retrieval:
  rrf_k: 60           # default; smaller = steeper rank decay
  bm25_weight: 0.5    # raise for keyword-heavy corpora
  vector_weight: 1.0  # raise for paraphrase / semantic match
```

No code change needed — `api.retrieve` and the server's
`POST /v1/doc/search` endpoint pick up the block on next call.
`evals/tools/sweep_rrf.py` re-fuses cached rankings at arbitrary
weights offline, but the raw dump it consumes is no longer emitted by
`dikw client eval` — the `--dump-raw` flag was dropped in the
client/server migration, so the offline sweep isn't reachable through
the CLI today.

### Score-normalised fusion alternatives

`retrieval.fusion` selects the algorithm that combines BM25 + vector +
asset legs. Three options ship:

- `rrf` (default) — Reciprocal Rank Fusion. Rank-only, robust against
  heterogeneous score scales (BM25 unbounded vs cosine `[0, 2]`). The
  safe choice and what every existing baseline in `evals/BASELINES.md`
  was measured under.
- `combsum` — per-leg min-max normalises raw scores to `[0, 1]` then
  weighted-sums across legs. Preserves **magnitude**: a leg's clear
  leader keeps its margin where RRF would collapse it to `1/(k+1)`.
  Reach for it when one leg dominates (vector-strong corpora) or both
  legs are close at the head and rank-based fusion has nothing to
  discriminate (CMTEB-0.6B observation: hybrid `-0.003` vs vector
  nDCG@10 under RRF).
- `combmnz` — `CombSUM × (number of legs that retrieved each key)`.
  Boosts cross-leg consensus on top of CombSUM's magnitude
  preservation. Single-leg corpora collapse to plain CombSUM.

```yaml
retrieval:
  fusion: combsum     # default: rrf
  bm25_weight: 0.3    # weights still apply; per-leg cap on contribution
  vector_weight: 1.5
```

The same `bm25_weight` / `vector_weight` knobs cap the per-leg
contribution under all three modes (CombSUM/CombMNZ multiply the
normalised `[0, 1]` score by the weight; RRF multiplies the rank
reciprocal). `rrf_k` is RRF-only and is ignored under
`combsum` / `combmnz`. Today the offline sweep tool
(`evals/tools/sweep_rrf.py`) only re-fuses RRF — switching fusion mode
requires editing `dikw.yml` and replaying through
`evals/tools/run_phase15_from_snapshot.py` (no API spend, the
embeddings stay cached).

For a Chinese benchmark, repeat with `convert_cmteb.py` against a
HuggingFace download — same workflow, see
[`evals/README.md`](../evals/README.md#public-benchmarks) for the
full command. **Before running any CJK eval**, flip
`retrieval.cjk_tokenizer: jieba` in the scratch base's `dikw.yml`
(gotcha #7) — otherwise the BM25 row in the ablation table will
report 0.03 nDCG@10 regardless of fusion tuning, because FTS5's
default tokenizer doesn't segment Chinese.

## OpenAI Codex (ChatGPT-backend GPT-5 series)

The codex protocol picks up `gpt-5.5`, `gpt-5.4-mini`, `gpt-5.3-codex`,
and the rest of the ChatGPT-only model family. Authentication is OAuth
via dikw's self-managed token store at `<base>/.dikw/auth.json` — see
gotcha #8 for the why and how it differs from codex CLI's
`~/.codex/auth.json`.

**Authenticate** (pick one):

```bash
# Option A — full device-code OAuth flow inside dikw, no codex CLI needed.
uv run dikw auth login openai-codex --base .

# Option B — copy tokens from an already-authenticated codex CLI session.
codex                                           # one-time codex CLI login
uv run dikw auth import openai-codex --base .   # one-shot copy

# Option C — do nothing; if you already have a non-expired
# ~/.codex/auth.json, the first call to dikw will lazy-import on its own
# and print a one-line stderr message.
```

After auth, point `dikw.yml` at the codex protocol:

```yaml
provider:
  llm: openai_codex
  llm_model: gpt-5.5                   # or gpt-5.4-mini / gpt-5.3-codex
  llm_base_url: https://chatgpt.com/backend-api/codex   # required
  embedding: openai_compat             # codex doesn't ship embeddings
  embedding_model: text-embedding-3-small
  embedding_base_url: https://api.openai.com/v1
  embedding_dim: 1536
  embedding_revision: ""
  embedding_normalize: true
  embedding_distance: cosine
```

`.env`:

```
DIKW_EMBEDDING_API_KEY=<your embedding-vendor key>
# CODEX_HOME=/custom/path        # optional — only consulted by `dikw auth import`
```

Verify before running ingest:

```bash
uv run --env-file .env dikw auth status openai-codex --base .
# provider     | status   | expires in | last refresh         | account
# openai-codex | active   | 28m 12s    | 2026-05-06 03:14 UTC | acc-...

uv run --env-file .env dikw client serve-and-run --base . -- check --llm-only
# Expected:
# LLM | https://chatgpt.com/backend-api/codex | OK | <ms>ms
```

If `dikw client check` reports `relogin_required`, the OAuth refresh_token has
been revoked or consumed elsewhere. Recover with
`uv run dikw auth login openai-codex --base .` (the device-code flow
mints a fresh pair).

## Pre-flight checklist for a new vendor

Before running `dikw client ingest` against a real corpus with a new vendor
config:

- [ ] `embedding_dim` matches what the model actually returns.
      Run `dikw client check --embed-only` and read `dim=…` from the output.
- [ ] `embedding_batch_size` is ≤ the vendor's observed cap.
- [ ] `dikw client check --llm-only` and `dikw client check --embed-only` each exit 0.
- [ ] If you're migrating from another vendor, `.dikw/index.sqlite` is
      deleted (see gotcha #1).
- [ ] Costs understood: if the LLM leg is `openai_compat`, you pay full
      input-token price on every synth call — no prompt caching.
- [ ] If the LLM leg is `openai_codex`, you've authenticated via
      `dikw auth login openai-codex --base .` (or `dikw auth import`)
      and `dikw auth status` reports `active` (gotcha #8).

## Overriding the K-layer authoring prompts

Swapping the *model* changes how well pages are written; overriding the
*prompt* changes what you ask it to write. A base can replace the packaged
K-layer authoring prompts with its own markdown — useful when an enterprise
wants house style, domain vocabulary, or a different page structure without
forking the engine.

Three prompts are overridable, each via an explicit config key:

| Prompt (packaged name)              | Config key                          | Used by                                   |
| ----------------------------------- | ----------------------------------- | ----------------------------------------- |
| `synthesize`                        | `synth.prompt_path`                 | `dikw client synth` **and** the `non_atomic_page` lint fixer (it reuses the synth template) |
| `lint_fix_orphan_merge`             | `lint.fixer_prompts.orphan_merge`   | `orphan_page` fixer's merge strategy      |
| `lint_fix_broken_wikilink_grounded` | `lint.fixer_prompts.broken_wikilink`| `broken_wikilink` fixer's grounded rewrite |

```yaml
synth:
  prompt_path: ./prompts/my_synth.md     # relative to the base root
lint:
  fixer_prompts:
    orphan_merge: ./prompts/orphan.md
    broken_wikilink: ./prompts/bw.md
```

Rules the engine enforces (`PromptOverrideError` on any violation, surfaced at
load **and** by `dikw client check`):

- **Inside the base.** The path resolves against the base root and must stay
  within it — a `../` escape or absolute path elsewhere is rejected. Keep
  overrides in the user-owned `<base>/prompts/` tree (committed with the base);
  do **not** put them in the gitignored `.dikw/`.
- **Exact placeholder set.** The override must reference *exactly* the
  `{placeholders}` the engine fills via `str.format` — no missing ones (the LLM
  would be starved of content) and no stray ones (`KeyError` at format time):

  | Prompt | Required `{placeholders}` |
  | ------ | ------------------------- |
  | `synthesize` | `categories`, `existing_pages_section`, `source_path`, `source_body`, `group_outline`, `group_index`, `group_total`, `max_pages` |
  | `lint_fix_orphan_merge` | `target_path`, `target_category`, `target_slug`, `target_title`, `target_body`, `orphan_path`, `orphan_body`, `score_reason` |
  | `lint_fix_broken_wikilink_grounded` | `broken_target`, `source_path`, `source_context`, `evidence_block`, `categories` |

- **Output-format markers.** All three must instruct the `<page category="…"
  slug="…">…</page>` block format — the literal substrings `<page`, `category=`,
  `slug=`, and `</page>` must all appear, or the synth parser can't read the
  response.
- **Context container (`synthesize` only).** The override must carry a
  `## Knowledge-base context` heading at a line start — the engine fills
  `{existing_pages_section}` with H3 sub-sections (existing pages / batch /
  priority targets) that nest under that H2. An override missing or demoting
  the heading is rejected rather than silently mis-nesting the sections.

The placeholder/marker contract is only a *structural* check — it guarantees the
engine can fill and parse your template, not that the wording produces good
pages. Validate quality by running `dikw client synth` on a representative source
and inspecting the output. A long-lived `dikw serve` re-reads the override file
on each call, so prompt edits take effect without a restart.

Two practices worth copying from the packaged template:

- **Taxonomy-specific examples.** The packaged worked examples use the default
  `entity`/`concept`/`note` taxonomy. A base with a custom `schema.categories`
  tree gets better imitation by overriding the template and rewriting the
  examples so their `category="…"` values are its own declared paths — the
  override is the designed channel for category customization (the engine never
  rewrites example categories mechanically; it cannot judge which declared path
  semantically fits an example body).
- **Prompt-cache-friendly layout.** The packaged template keeps every
  `{placeholder}` in a tail zone after the static instruction sections, so the
  instruction prefix is byte-stable across calls — OpenAI-compatible providers'
  automatic prefix caching (and codex) then cover it; the system prompt is
  separately cached via `cache_control` on Anthropic-compatible providers. An
  override that interleaves placeholders with instructions gives that up.

## See also

- [`README.md`](../README.md#providers) — quick config snippets.
- [`docs/getting-started.md`](./getting-started.md#pluggable-providers) —
  end-to-end walkthrough.
- [`docs/architecture.md`](./architecture.md) — where the provider
  seam sits in the module map.
- [`docs/adr/0003-configurable-knowledge-taxonomy.md`](./adr/0003-configurable-knowledge-taxonomy.md) —
  why the taxonomy and prompts are configurable.


## Multimodal embedding (v1: Gitee)

When `dikw.yml` declares an `assets.multimodal` block, the engine
routes both chunk text and image bytes through one
`MultimodalEmbeddingProvider` so they land in the same vector space.
Without the block, the legacy 2-leg text-embedding path is used
exactly as before — multimodal is opt-in.

### Wire format

Gitee's multimodal embeddings endpoint accepts **one shape across every
multimodal model it serves** (Qwen3-VL-Embedding-8B, jina-clip-v2, …):
a list of per-modality input dicts. The model name in
`assets.multimodal.model` discriminates which model runs server-side;
the wire payload does not change.

```http
POST /v1/embeddings
{
  "model": "Qwen3-VL-Embedding-8B",
  "input": [
    {"text": "a blue cat"},
    {"image": "data:image/png;base64,..."}
  ]
}
```

This is **distinct** from Gitee's text-only embeddings shape
(`input: "..."` or `input: ["...", "..."]`), which the legacy text
embedder uses on the `embedding_*` keys at the top of `dikw.yml`.

Each `MultimodalInput` becomes one wire input. Combined text+image on a
single `MultimodalInput` is rejected loudly (Gitee embeds per-modality,
no joint-encode mode), as is multiple images per input. The pipeline
never produces those shapes today (chunks are text-only, assets are
image-only), so the rejection just keeps a future config mistake from
silently dropping a modality.

### Cookbook: Qwen pair on Gitee (recommended)

Both legs on Gitee, same vendor and key. Empirical dims (probed
2026-04-25):

| Model | Native dim | `dimensions` knob | Default if unset |
|---|---|---|---|
| `Qwen3-Embedding-0.6B` (text-only, **recommended**) | **1024 fixed** | n/a | 1024 |
| `Qwen3-Embedding-8B` (text-only, larger / higher-cost) | 4096 | matryoshka 4096 / 1024 / 512 / 256 | **1024** |
| `Qwen3-VL-Embedding-8B` (multimodal) | **1024 fixed** | not accepted | 1024 |

`0.6B` is the recommended default: same 1024 dim as the multimodal leg,
~13x fewer params than `8B`, and on CMTEB-T2 it lands within the noise
floor of `8B` for retrieval quality (see `evals/BASELINES.md`,
2026-04-28 entry). Reach for `8B` only when you have budget headroom and
your eval gates are tight enough that single-percent nDCG matters.

```yaml
provider:
  llm: anthropic_compat                # or whichever LLM you've configured
  embedding: openai_compat             # text leg — single-string input shape
  embedding_base_url: https://ai.gitee.com/v1
  embedding_model: Qwen3-Embedding-0.6B
  embedding_dim: 1024                  # 0.6B native; matches Qwen3-VL so
                                       # both hybrid legs live in the same
                                       # dim space. WARNING: dim locks at
                                       # first ingest (gotcha #1) — don't
                                       # change later. Switch to
                                       # Qwen3-Embedding-8B (with dim 1024
                                       # via matryoshka, or 4096 native)
                                       # for higher-cost runs.
  embedding_revision: ""               # bump to force re-embed when Qwen
                                       # weights drift silently behind the
                                       # stable model name
  embedding_normalize: true
  embedding_distance: cosine
  embedding_batch_size: 16             # Gitee caps at 25 (gotcha #2)
  embedding_provider_label: gitee-ai

assets:
  dir: assets                          # relative to project root
  multimodal:
    provider: gitee_multimodal         # the only literal that ships today
    model: Qwen3-VL-Embedding-8B       # multimodal leg — per-modality dicts
    dim: 1024                          # fixed by the model; Gitee returns
                                       # 4096-byte float32 blobs = 1024 dim.
                                       # Setting any other value here makes
                                       # vec_assets_v<n> dim-mismatch on
                                       # first asset embed.
    revision: ""                       # bump (e.g., "2026-04") if Gitee
                                       # silently refreshes the weights
    normalize: true
    distance: cosine
    batch: 16                          # per-request input cap
    base_url: null                     # null = use the gitee_multimodal default
```

The text leg and the multimodal leg write **separate** vec tables
(`vec_chunks_v<n>` / `vec_assets_v<n>`, one per `embed_versions` row);
their dims do not have to match. Both legs read
`DIKW_EMBEDDING_API_KEY` — never `OPENAI_API_KEY`. If LLM and
embedding target different vendors, set them as distinct env vars.

**Note on chunk routing**: text and multimodal are strictly separate
channels. Chunk text always flows through the **text** embedder
(`provider.embedding_model`) into `vec_chunks_v<text_version_id>`,
even when `assets.multimodal` is configured. The multimodal embedder
embeds **assets only** (image bytes) into
`vec_assets_v<mm_version_id>`. Cross-modal retrieval works because
`info/search.HybridSearcher` runs both legs and asset-vec hits
promote the chunks that reference matching images via
`chunk_asset_refs` — the two vector spaces don't need to coincide.

### Verifying the config end-to-end

`dikw client check --embed-only` automatically routes through the multimodal
embedder when `assets.multimodal` is present, sending one text + one
image input in **a single batched request** (no RTT stacking). Both
modalities probe the same endpoint Gitee will see at ingest time, so a
green check means real ingest will work:

```bash
$ uv run --env-file .env dikw client serve-and-run --base . -- check --embed-only
Embedding | (provider default) | OK | 4234ms, dim=1024, modalities=text+image, provider=gitee-ai
```

(Latency dominated by the ~50ms TLS handshake forced per request to
work around Gitee's idle-keepalive drops — see fdd2cae for the
rationale. Plus a few seconds of Gitee server time on the multimodal
endpoint.)

If `assets.multimodal` is absent, the check falls back to the text-only
probe (one `"ping"` string) — same as before.

### What the multimodal pipeline buys you

- Image binaries referenced from your markdown (`![alt](path)` or
  `![[file]]`) are materialized into `assets/<h2>/<h8>-<name>.<ext>`
  inside your project root, visible in Obsidian.
- Each binary gets a vector via the multimodal model and lives in a
  per-version `vec_assets_v<id>` table (so dim/model changes don't
  collide with prior data).
- Hybrid search adds a third RRF leg over asset vectors that promotes
  chunks via the `chunk_asset_refs` reverse lookup — text queries
  retrieve chunks based on the images they reference even when the
  surrounding prose doesn't match the query.

### Switching the multimodal model

Change `model` (and `dim` if it differs) in `dikw.yml` and re-run
`dikw client ingest`. The engine sees a new identity tuple, mints a new
version row, creates a fresh `vec_assets_v<new_id>` table, and writes
to it. The previous version's data stays in `vec_assets_v<old_id>`
until you run `dikw embed reindex` (v1 ships a stub; v1.5 implements
the real migration).

### Trying another multimodal provider

Drop a new file under `src/dikw_core/providers/` that implements the
`MultimodalEmbeddingProvider` Protocol (one method: `async embed(inputs:
list[MultimodalInput], *, model: str) -> list[list[float]]`), then
add a branch to `build_multimodal_embedder` in
`providers/__init__.py`. The Protocol signature accepts arbitrary
combinations of text and image inputs already — Voyage v3, Cohere v4,
Jina-direct, and self-hosted Nomic Embed Vision all slot in without
engine changes. (Their wire shapes diverge from Gitee's per-modality
dicts, which is why each vendor gets its own provider rather than
sharing one serializer.)
