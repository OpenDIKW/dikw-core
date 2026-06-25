# A reranker is deterministic scoping, consistent with "LLMs only at synth"

**Status**: Accepted and shipped. Adds an optional cross-encoder rerank stage to the
retrieval pipeline (`RerankProvider` Protocol + `build_reranker` factory +
`OpenAICompatReranker`), wired into `HybridSearcher.search` between RRF fusion and the
top-K truncation. Off until a reranker is configured; on once configured.

## Context

The retrieve path is deterministic, no-LLM hybrid search: FTS + vector (+ optional
graph) legs fused by Reciprocal Rank Fusion, a source-diversity penalty, then truncation
to the query `limit` (`domains/info/search.py`). RRF fuses **ranks**, not **relevance**:
it knows only each leg's ordinal position, not how well a chunk answers the query. Two
failure modes survive fusion into the top-K:

1. **Semantically-close-but-wrong** chunks — a chunk embeds near the query (shared
   topic/vocabulary) but does not answer it; the vector leg ranks it high and RRF keeps
   it high.
2. **Lexical-overlap-but-wrong** chunks — keyword match without relevance, from BM25.

The project's own SciFact baseline shows the gap: `recall@100 ≈ 0.97` while top-K
precision is far lower — the right passages are *in the candidate pool, just not ranked
first*. A cross-encoder reranker, which jointly attends to `(query, chunk)`, is the
standard tool for recovering precision@k from that recall.

The blocker is a core invariant. CLAUDE.md and `docs/design.md` (principle #3,
"scoping deterministic, reasoning probabilistic" — Karpathy's rule) and
`docs/architecture.md` state that **navigation is deterministic SQL + file I/O and LLM
calls enter only at synth**; `retrieve` returns ranked chunks with *no LLM call*. Adding
a model to the read path looks, at first glance, like a violation.

## Decision

**Add an optional cross-encoder rerank stage, framed as deterministic scoping — and
exclude LLM-as-reranker.**

A reranker is a **scoring function**: it maps `(query, chunk_text) → scalar relevance`
and reorders the candidate set. It is the same epistemic category as the embedding model
already on the read path — a learned model that scores query↔chunk similarity. A
cross-encoder is simply a more accurate similarity scorer. It is deterministic (same
inputs → same scores), it generates no text, it decides nothing about *what* to retrieve,
and it cannot invent content: the chunks it ranks are exactly the chunks the
deterministic FTS+vector legs already surfaced, re-ordered, never added to. It is part of
**scoping** (ranking the deterministically-retrieved candidate set), not **reasoning**
(synthesising an answer).

So the invariant is sharpened, not broken: *the read path forbids generative LLM
reasoning (answer synthesis), not learned scoring functions.* Under that reading:

- **Dedicated cross-encoder reranker (API or local) — consistent.** Shipped.
- **LLM-as-reranker (prompt a generative `LLMProvider` to score/reorder) — excluded.**
  It puts probabilistic generative reasoning (which can hallucinate, refuse, or reorder
  by latent preference) directly on the read path — exactly what "LLMs only at synth"
  guards against. Not shipped, and not to be added without an explicit, documented
  redefinition of the invariant.

### Shape

- **`RerankProvider` Protocol** (`providers/base.py`): one async method,
  `rerank(query, documents, *, model) -> list[float]`, returning one relevance score per
  document **aligned to input order** (the adapter remaps the endpoint's relevance-sorted
  `index`, mirroring `EmbeddingProvider`'s defensive index handling).
- **`OpenAICompatReranker`** (`providers/rerank.py`): the Jina/Cohere-compatible
  `/rerank` wire shape (`{model, query, documents, top_n}` → `{"results": [{index,
  relevance_score}]}`) that Gitee AI, SiliconFlow, Jina, and Cohere converge on — one
  adapter, vendor picked by `base_url`/`model`/key. Reuses the shared no-keepalive httpx
  defence and the `ProviderError`/`TransientProviderError` classification.
- **`build_reranker(ProviderConfig)`** factory, mirroring `build_embedder`; returns
  `None` when unconfigured.
- **Insertion point**: `HybridSearcher.search`, after RRF fusion + the source-diversity
  penalty, before the `[:limit]` truncation. The top `rerank_candidate_k` (clamped to at
  least `limit`) candidates have their chunk text fetched, are scored, re-sorted by
  rerank score, then truncated to `limit`. The window's chunk records are reused for
  materialization (no second `get_chunks`). This is the seam `design.md` already reserves
  for "RRF fusion / reranking" — **no storage adapter changes**, and `retrieve`'s
  response shape is unchanged.
- **Config**: provider wiring (`rerank`/`rerank_model`/`rerank_base_url`/
  `rerank_api_key_env`/`rerank_timeout_seconds`) on `ProviderConfig`; behavioural knobs
  (`rerank_enabled`, `rerank_candidate_k`) on `RetrievalConfig`. A validator requires the
  three wiring fields once `rerank` is set (fail fast at config load, like the
  `openai_codex` base-url validator).

### "On once configured" (not the `graph_enabled` default-off pattern)

`graph_enabled` defaults **off** and must be explicitly turned on. Rerank instead is
**on once a provider is configured**: `rerank_enabled` defaults `True`, so setting
`provider.rerank` is the opt-in and a base that never configures a reranker runs no
rerank leg regardless. `rerank_enabled=False` is a kill switch that keeps a configured
reranker dark — used by the eval harness to compare rerank-off vs rerank-on at one config.

### Read-path resilience

Rerank is on the interactive read path, so failures must not 500 a query: a
**transient** provider failure (5xx/408/429/timeout/connection drop) degrades to the
fused order (logged); a **permanent** `ProviderError` (401/403/404, bad model/key)
propagates so a misconfig fails loud instead of silently degrading on every query — the
same fail-fast contract the embedding path uses.

## Consequences

- Retrieval can recover top-K precision on large/noisy corpora without changing recall
  (rerank reorders the pool, never expands it — so `recall@100` is unchanged; a change
  there signals a bug).
- Rerank cost is **per-query** (unlike embeddings, paid once at ingest), so it is opt-in
  per base; small corpora where RRF's top-K is already correct gain little and need not
  configure it.
- A fourth provider type and a new read-path failure mode are added, both behind the
  existing provider seam and the deterministic-scoping framing above.
- Eval gate: a retrieval change requires an `evals/BASELINES.md` entry — see that file
  for the SciFact rerank-off-vs-on ablation (Qwen3-Embedding-0.6B + bge-reranker-v2-m3
  via Gitee AI).
