"""Hybrid search: BM25 (FTS5) + vector(s) fused via Reciprocal Rank Fusion.

v1 has two operating modes:

* **Legacy 2-leg** (text embedder, no asset index) — BM25 over chunk
  text + vector search over chunk vectors, fused at chunk-level via RRF.
  Behavior identical to the original implementation.

* **Multimodal 3-leg** (multimodal embedder + asset version) — adds a
  third channel that runs ``vec_search_assets`` against the per-version
  asset vector table; matched assets promote their parent chunks (via
  the ``chunk_asset_refs`` reverse lookup) into the same RRF pool.
  Each returned Hit carries the assets that the chunk references so
  downstream consumers (CLI display, server response, LLM synthesis)
  can render or cite them.

Fusion algorithm is configurable via ``RetrievalConfig.fusion``: ``rrf``
(default, rank-only), ``combsum`` (per-leg min-max → weighted sum), or
``combmnz`` (CombSUM * number of legs that retrieved each key). RRF is
the safe default because BM25 negative-log scores and cosine distances
don't normalize cleanly against each other, but rank ordering does;
CombSUM/CombMNZ trade that safety for magnitude preservation when one
leg dominates or both legs are close at the head — see
``evals/BASELINES.md`` for the empirical motivation.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal, Protocol

from ...providers import EmbeddingProvider, MultimodalEmbeddingProvider
from ...schemas import (
    AssetRecord,
    AssetVecHit,
    ChunkNeighborRecord,
    ChunkRecord,
    FTSHit,
    Hit,
    Layer,
    MultimodalInput,
    VecHit,
)
from ...storage.base import NotSupported, Storage
from .tokenize import WORD_OR_CJK_CHARS, CjkTokenizer, preprocess_for_fts


class RetrievalConfigLike(Protocol):
    """Structural shape of ``config.RetrievalConfig`` for ``from_config``."""

    rrf_k: int
    bm25_weight: float
    vector_weight: float
    fusion: FusionMode
    cjk_tokenizer: CjkTokenizer
    same_doc_penalty_alpha: float
    graph_enabled: bool
    graph_seed_top_k: int
    graph_weight: float


@dataclass(frozen=True)
class MultimodalSearch:
    """Wires the asset-vector retrieval channel into ``HybridSearcher``.

    All three fields are required to activate the channel — the embedder
    embeds the query into the multimodal vector space, the model is the
    name passed to the embedder, and the version_id selects which
    ``vec_assets_v<id>`` table to search.
    """

    embedder: MultimodalEmbeddingProvider
    model: str
    asset_version_id: int


RRF_K = 60

# FTS5 reserved query operators. Stripped from user queries because
# `_sanitize_fts` builds an OR-of-tokens expression itself; an unwary
# user word like "AND" would otherwise become syntax mid-query.
_FTS_RESERVED = frozenset({"AND", "OR", "NOT", "NEAR"})

# Which retrieval legs to fuse. ``hybrid`` is the historical default and
# what `dikw query` uses; ``bm25`` and ``vector`` exist so eval can
# ablate the contribution of each leg against public benchmarks.
RetrievalMode = Literal["bm25", "vector", "hybrid"]


def apply_source_diversity_penalty(
    fused: dict[int, float],
    doc_id_by_chunk: dict[int, str],
    *,
    alpha: float,
) -> dict[int, float]:
    """Diminishing-returns demotion of repeat same-doc chunks.

    Walks ``fused`` in score-desc order. The 1st chunk seen from each
    ``doc_id`` is unpenalized (factor ``1.0``); the N-th chunk (N ≥ 2)
    from the same doc is scaled by ``1 / (1 + alpha * (N - 1))``.
    With ``alpha=0.3``: 1st = 1.0, 2nd ≈ 0.77, 3rd = 0.625, 4th ≈ 0.526.
    With ``alpha=0`` the function is the identity (no-op).

    Returns a new dict with the same key set; the caller re-sorts and
    slices top-K. Pure: no I/O, no side effects, no globals.
    """
    if alpha == 0.0:
        return dict(fused)
    per_doc_seen: dict[str, int] = {}
    out: dict[int, float] = {}
    for chunk_id, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
        doc_id = doc_id_by_chunk.get(chunk_id)
        if doc_id is None:
            out[chunk_id] = score
            continue
        n_seen = per_doc_seen.get(doc_id, 0)
        out[chunk_id] = score / (1.0 + alpha * n_seen)
        per_doc_seen[doc_id] = n_seen + 1
    return out


def reciprocal_rank_fusion[K: Hashable](
    rank_lists: list[list[K]],
    *,
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> dict[K, float]:
    """Reciprocal Rank Fusion. Returns key → fused score (higher = better).

    ``K`` is generic over any hashable identity — historically ``doc_id:
    str``, now also ``chunk_id: int`` once chunk-level fusion lands.

    ``weights`` lets the caller bias fusion toward a stronger leg — e.g.,
    when BM25 is measurably behind the dense leg on a given corpus,
    equal-weight RRF drags the combined ranking toward the weaker signal
    (observed on BEIR/SciFact: hybrid nDCG@10 0.736 < vector 0.773 at
    default k=60, equal weights). Setting ``weights=[0.5, 1.0]`` halves
    BM25's per-rank contribution while keeping every doc it found in the
    pool — better rank quality, no recall loss.

    ``None`` (the default) is equivalent to ``[1.0] * len(rank_lists)``
    — the behaviour before weighting landed, preserved bit-for-bit.
    """
    if weights is None:
        weights = [1.0] * len(rank_lists)
    if len(weights) != len(rank_lists):
        raise ValueError(
            f"weights length {len(weights)} must match rank_lists length "
            f"{len(rank_lists)}"
        )
    scores: dict[K, float] = {}
    for lst, w in zip(rank_lists, weights, strict=True):
        for rank, key in enumerate(lst):
            scores[key] = scores.get(key, 0.0) + w / (k + rank + 1)
    return scores


FusionMode = Literal["rrf", "combsum", "combmnz"]


def _normalise_per_leg[K: Hashable](
    scored: list[tuple[K, float]],
) -> dict[K, float]:
    """Per-leg min-max → ``[0, 1]``. ``max == min`` collapses to all 1.0
    (degenerate single-score leg) instead of dividing by zero.

    Duplicate keys within one leg keep their highest score. Empty input
    returns an empty dict.
    """
    if not scored:
        return {}
    best: dict[K, float] = {}
    for key, score in scored:
        prev = best.get(key)
        if prev is None or score > prev:
            best[key] = score
    lo = min(best.values())
    hi = max(best.values())
    span = hi - lo
    if span == 0.0:
        return dict.fromkeys(best, 1.0)
    return {k: (s - lo) / span for k, s in best.items()}


def comb_sum_fusion[K: Hashable](
    scored_lists: list[list[tuple[K, float]]],
    *,
    weights: list[float] | None = None,
) -> dict[K, float]:
    """CombSUM: per-leg min-max normalise → weighted sum across legs.

    Each leg supplies ``(key, score_higher_is_better)`` pairs. Score
    scales differ across legs (BM25 unbounded vs cosine ``[0, 2]``), so
    each leg is normalised independently before summing — every leg
    contributes at most ``weights[i]`` to a key's fused score.

    Compared to RRF (rank-only), CombSUM preserves magnitude: a clear
    leader by raw score keeps its margin, where RRF would collapse it
    to ``1/(k+1)``. Useful when one leg is much stronger than the
    others or when both legs are close at the head and rank-based
    fusion has nothing to discriminate (CMTEB-0.6B observation).
    """
    if weights is None:
        weights = [1.0] * len(scored_lists)
    if len(weights) != len(scored_lists):
        raise ValueError(
            f"weights length {len(weights)} must match scored_lists length "
            f"{len(scored_lists)}"
        )
    fused: dict[K, float] = {}
    for leg, w in zip(scored_lists, weights, strict=True):
        for key, score in _normalise_per_leg(leg).items():
            fused[key] = fused.get(key, 0.0) + w * score
    return fused


def comb_mnz_fusion[K: Hashable](
    scored_lists: list[list[tuple[K, float]]],
    *,
    weights: list[float] | None = None,
) -> dict[K, float]:
    """CombMNZ: ``CombSUM(key) * (number of legs that retrieved key)``.

    Boosts consensus across legs on top of CombSUM's magnitude
    preservation. A key found by all three legs scores ``3 * CombSUM``;
    a key found by only one scores ``1 * CombSUM`` (i.e., plain
    CombSUM). Single-leg scenarios collapse to CombSUM exactly.

    Zero-weight legs are excluded from the consensus multiplier — a
    leg the caller explicitly disabled via ``weights[i] == 0`` has no
    contribution to ``CombSUM`` and must not bump the leg-count either,
    or ablation runs would behave inconsistently between modes.
    """
    fused = comb_sum_fusion(scored_lists, weights=weights)
    effective_weights = (
        weights if weights is not None else [1.0] * len(scored_lists)
    )
    leg_count: dict[K, int] = {}
    for leg, w in zip(scored_lists, effective_weights, strict=True):
        if w == 0.0:
            continue
        for key in _normalise_per_leg(leg):
            leg_count[key] = leg_count.get(key, 0) + 1
    return {k: s * leg_count[k] for k, s in fused.items()}


class HybridSearcher:
    """Composes FTS + vector search(es) on top of a ``Storage`` backend.

    Pass a ``MultimodalSearch`` to activate the asset-vector retrieval
    leg; otherwise the searcher runs the FTS + (optional) text-vector
    legs only.
    """

    def __init__(
        self,
        storage: Storage,
        embedder: EmbeddingProvider | None,
        *,
        embedding_model: str | None = None,
        text_version_id: int | None = None,
        multimodal: MultimodalSearch | None = None,
        rrf_k: int = RRF_K,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        fusion: FusionMode = "rrf",
        cjk_tokenizer: CjkTokenizer = "none",
        same_doc_penalty_alpha: float = 0.3,
        graph_enabled: bool = False,
        graph_seed_top_k: int = 20,
        graph_weight: float = 0.5,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._embedding_model = embedding_model
        # When set, ``vec_search`` is targeted at this specific text
        # version_id. ``None`` means "let the storage adapter resolve
        # the active text version" — fine for the common single-model
        # case, but eval / migration paths pass an explicit id.
        self._text_version_id = text_version_id
        self._mm = multimodal
        self._rrf_k = rrf_k
        self._bm25_weight = bm25_weight
        self._vector_weight = vector_weight
        self._fusion: FusionMode = fusion
        # Must match the storage adapter's ingest-time tokenizer; a
        # mismatch silently drops CJK hits.
        self._cjk_tokenizer: CjkTokenizer = cjk_tokenizer
        self._same_doc_penalty_alpha = same_doc_penalty_alpha
        self._graph_enabled = graph_enabled
        self._graph_seed_top_k = graph_seed_top_k
        self._graph_weight = graph_weight

    @classmethod
    def from_config(
        cls,
        storage: Storage,
        embedder: EmbeddingProvider | None,
        cfg: RetrievalConfigLike,
        *,
        embedding_model: str | None = None,
        text_version_id: int | None = None,
        multimodal: MultimodalSearch | None = None,
    ) -> HybridSearcher:
        """Unpack a ``RetrievalConfig`` into the keyword kwargs.

        Centralises the knob mapping so adding a new knob is a
        one-file change. ``RetrievalConfigLike`` is any object with
        the listed attributes — pydantic ``RetrievalConfig`` qualifies.
        """
        return cls(
            storage,
            embedder,
            embedding_model=embedding_model,
            text_version_id=text_version_id,
            multimodal=multimodal,
            rrf_k=cfg.rrf_k,
            bm25_weight=cfg.bm25_weight,
            vector_weight=cfg.vector_weight,
            fusion=cfg.fusion,
            cjk_tokenizer=cfg.cjk_tokenizer,
            same_doc_penalty_alpha=cfg.same_doc_penalty_alpha,
            graph_enabled=cfg.graph_enabled,
            graph_seed_top_k=cfg.graph_seed_top_k,
            graph_weight=cfg.graph_weight,
        )

    async def search(
        self,
        q: str,
        *,
        limit: int = 10,
        per_leg_limit: int = 40,
        layer: Layer | None = None,
        mode: RetrievalMode = "hybrid",
    ) -> list[Hit]:
        if not q.strip():
            return []

        run_fts = mode in ("bm25", "hybrid")
        run_vec = mode in ("vector", "hybrid")
        text_vec_active = (
            self._embedder is not None and self._embedding_model is not None
        )
        if mode == "vector" and not (self._mm is not None or text_vec_active):
            raise ValueError(
                "mode='vector' requires either a MultimodalSearch or an "
                "(embedder, embedding_model) pair"
            )

        # Embed once per modality so the text leg searches
        # ``vec_chunks_v<id>`` and the multimodal leg searches
        # ``vec_assets_v<id>``; never share a vector across spaces.
        q_vec_text: list[float] | None = None
        q_vec_mm: list[float] | None = None
        if run_vec:
            if text_vec_active:
                q_vec_text = await self._embed_query_text(q)
            if self._mm is not None:
                q_vec_mm = await self._embed_query_multimodal(q)

        fts_task: asyncio.Task[list[FTSHit]] | None = None
        vec_task: asyncio.Task[list[VecHit]] | None = None
        asset_task: asyncio.Task[list[AssetVecHit]] | None = None
        if run_fts:
            fts_task = asyncio.create_task(
                self._storage.fts_search(
                    _sanitize_fts(q, cjk_tokenizer=self._cjk_tokenizer),
                    limit=per_leg_limit,
                    layer=layer,
                )
            )
        if q_vec_text is not None:
            vec_task = asyncio.create_task(
                self._storage.vec_search(
                    q_vec_text,
                    version_id=self._text_version_id,
                    limit=per_leg_limit,
                    layer=layer,
                )
            )
        if q_vec_mm is not None and self._mm is not None:
            asset_task = asyncio.create_task(
                self._storage.vec_search_assets(
                    q_vec_mm,
                    version_id=self._mm.asset_version_id,
                    limit=per_leg_limit,
                    layer=layer,
                )
            )

        fts_hits: list[FTSHit] = await fts_task if fts_task is not None else []
        vec_hits: list[VecHit] = []
        if vec_task is not None:
            try:
                vec_hits = await vec_task
            except NotSupported:
                vec_hits = []
        asset_hits: list[AssetVecHit] = []
        if asset_task is not None:
            try:
                asset_hits = await asset_task
            except NotSupported:
                asset_hits = []

        # Asset hits promote the chunks that reference them. The first
        # asset that surfaces a chunk wins its rank slot. Score fusion
        # additionally tracks the best (smallest) distance across all
        # assets that reach the chunk into ``asset_dist_by_chunk``; that
        # bookkeeping is gated on ``self._fusion != "rrf"`` so RRF runs
        # are byte-identical to pre-PR behaviour.
        asset_chunk_ranked: list[int] = []
        asset_chunk_doc_ids: dict[int, str] = {}
        asset_dist_by_chunk: dict[int, float] = {}
        track_asset_dist = self._fusion != "rrf"
        if asset_hits:
            chunks_by_asset = await self._storage.chunks_referencing_assets(
                [h.asset_id for h in asset_hits]
            )
            promoted_chunk_ids = list(
                {cid for cids in chunks_by_asset.values() for cid in cids}
            )
            chunk_by_id = {
                c.chunk_id: c
                for c in await self._storage.get_chunks(promoted_chunk_ids)
                if c.chunk_id is not None
            }
            for h in asset_hits:
                for cid in chunks_by_asset.get(h.asset_id, []):
                    chunk = chunk_by_id.get(cid)
                    if chunk is None:
                        continue
                    if track_asset_dist:
                        prev = asset_dist_by_chunk.get(cid)
                        if prev is None or h.distance < prev:
                            asset_dist_by_chunk[cid] = h.distance
                    if cid in asset_chunk_doc_ids:
                        continue
                    asset_chunk_ranked.append(cid)
                    asset_chunk_doc_ids[cid] = chunk.doc_id

        # Build chunk-level rank lists. FTSHit.chunk_id is `int | None` in
        # the schema (legacy compat); every shipped adapter populates it,
        # but defensively skip any None to keep fusion keys homogeneous.
        fts_ranked = [h.chunk_id for h in fts_hits if h.chunk_id is not None]
        vec_ranked = [h.chunk_id for h in vec_hits]

        graph_neighbors = await self._collect_graph_neighbors(
            vec_ranked, fts_ranked, per_leg_limit
        )

        # Asset channel rides the vector weight — same family of signal
        # (semantic similarity in the multimodal space), distinct only in
        # what's embedded (chunk text vs asset bytes). Graph leg lands at
        # the end so it never reorders the historical 3-leg defaults.
        fusion_weights = [
            self._bm25_weight,
            self._vector_weight,
            self._vector_weight,
        ]
        if graph_neighbors:
            fusion_weights = [*fusion_weights, self._graph_weight]
        if self._fusion == "rrf":
            ranked_lists: list[list[int]] = [
                fts_ranked, vec_ranked, asset_chunk_ranked
            ]
            if graph_neighbors:
                ranked_lists.append([n.chunk_id for n in graph_neighbors])
            fused = reciprocal_rank_fusion(
                ranked_lists,
                k=self._rrf_k,
                weights=fusion_weights,
            )
        elif self._fusion in ("combsum", "combmnz"):
            # Score-bearing tuples for CombSUM / CombMNZ. ``FTSHit.score``
            # is already higher-is-better (negated BM25); cosine distances
            # flip via ``-distance`` so per-leg min-max sees a consistent
            # direction. Ties within a leg keep the first occurrence (see
            # ``_normalise_per_leg``).
            fts_scored: list[tuple[int, float]] = [
                (h.chunk_id, h.score) for h in fts_hits if h.chunk_id is not None
            ]
            vec_scored: list[tuple[int, float]] = [
                (h.chunk_id, -h.distance) for h in vec_hits
            ]
            asset_scored: list[tuple[int, float]] = [
                (cid, -asset_dist_by_chunk[cid]) for cid in asset_chunk_ranked
            ]
            scored_lists: list[list[tuple[int, float]]] = [
                fts_scored, vec_scored, asset_scored
            ]
            if graph_neighbors:
                # Graph leg uses ``edge_count`` as its raw score — popular
                # neighbors land near top after per-leg min-max normalises.
                scored_lists.append(
                    [(n.chunk_id, float(n.edge_count)) for n in graph_neighbors]
                )
            fuser = (
                comb_sum_fusion if self._fusion == "combsum" else comb_mnz_fusion
            )
            fused = fuser(
                scored_lists,
                weights=fusion_weights,
            )
        else:
            # Defends direct ``HybridSearcher`` callers that bypass
            # ``RetrievalConfig``'s pydantic ``Literal`` validation —
            # without this raise, an unknown mode falls through to
            # CombMNZ silently and the only signal is degraded ranking
            # quality (very hard to debug).
            raise ValueError(
                f"unknown fusion mode: {self._fusion!r}; "
                f"expected one of 'rrf', 'combsum', 'combmnz'"
            )

        # Per-chunk doc_id lookup, sourced from every leg that knows it.
        # Vec/asset/graph legs always carry doc_id; FTS hits do too. The
        # first writer wins because every leg agrees on chunk_id -> doc_id.
        doc_id_by_chunk: dict[int, str] = {}
        for vh in vec_hits:
            doc_id_by_chunk.setdefault(vh.chunk_id, vh.doc_id)
        for fh in fts_hits:
            if fh.chunk_id is not None:
                doc_id_by_chunk.setdefault(fh.chunk_id, fh.doc_id)
        for cid, did in asset_chunk_doc_ids.items():
            doc_id_by_chunk.setdefault(cid, did)
        for n in graph_neighbors:
            doc_id_by_chunk.setdefault(n.chunk_id, n.doc_id)

        # Stage 3 source-diversity demotion. alpha=0 is a no-op; alpha>0
        # demotes later same-doc chunks via diminishing returns.
        adjusted = apply_source_diversity_penalty(
            fused, doc_id_by_chunk, alpha=self._same_doc_penalty_alpha
        )
        top = sorted(adjusted.items(), key=lambda kv: kv[1], reverse=True)[:limit]

        # Per-chunk snippet lookup from FTS hits (BM25's snippet() preview).
        snippets_by_chunk: dict[int, str] = {
            h.chunk_id: h.snippet or ""
            for h in fts_hits
            if h.chunk_id is not None and h.snippet
        }

        retrieved_chunk_ids = [cid for cid, _ in top]

        refs_by_chunk = await self._storage.chunk_asset_refs_for_chunks(
            retrieved_chunk_ids
        )
        all_asset_ids = list(
            {r.asset_id for refs in refs_by_chunk.values() for r in refs}
        )
        fetched_assets = await self._storage.get_assets(all_asset_ids)
        assets_by_id: dict[str, AssetRecord] = {
            a.asset_id: a for a in fetched_assets
        }

        # Batch-fetch the chunks (for snippet fallback + seq) and unique
        # parent docs (for path/title) — chunk-level fusion repeats
        # doc_ids across hits, so per-hit fetches would N+1 the storage.
        chunk_by_id_all: dict[int, ChunkRecord] = {
            c.chunk_id: c
            for c in await self._storage.get_chunks(retrieved_chunk_ids)
            if c.chunk_id is not None
        }
        unique_doc_ids = list({
            chunk_by_id_all[cid].doc_id
            for cid in retrieved_chunk_ids
            if cid in chunk_by_id_all
        })
        doc_by_id = {
            d.doc_id: d
            for d in await self._storage.get_documents(unique_doc_ids)
        }

        hits: list[Hit] = []
        for chunk_id, score in top:
            chunk = chunk_by_id_all.get(chunk_id)
            if chunk is None:
                # Race: chunk dropped between fusion and materialization.
                # Skip rather than emit a half-formed Hit (TODOS T4 covers
                # the principled "loud failure" path).
                continue
            doc = doc_by_id.get(chunk.doc_id)
            snippet = snippets_by_chunk.get(chunk_id)
            if not snippet:
                snippet = self._render_chunk_snippet(chunk)
            asset_records: list[AssetRecord] = []
            for r in refs_by_chunk.get(chunk_id, []):
                a = assets_by_id.get(r.asset_id)
                if a is not None:
                    asset_records.append(a)
            hits.append(
                Hit(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk_id,
                    seq=chunk.seq,
                    score=score,
                    snippet=snippet,
                    path=doc.path if doc else None,
                    title=doc.title if doc else None,
                    asset_refs=asset_records,
                    layer=doc.layer if doc else None,
                    start=chunk.start,
                    end=chunk.end,
                    text=chunk.text,
                )
            )
        return hits

    async def _collect_graph_neighbors(
        self,
        vec_ranked: list[int],
        fts_ranked: list[int],
        per_leg_limit: int,
    ) -> list[ChunkNeighborRecord]:
        """Optional 4th leg: K-layer wikilink graph.

        Seeds come from the unfused vec + fts top-K (deterministic across
        fusion-mode changes); storage walks one hop via the wikilink
        graph and returns neighbors ordered by edge_count desc. Returns
        an empty list when disabled or when the storage backend lacks
        the neighbor primitive (older adapters).
        """
        if not self._graph_enabled:
            return []
        seeds: list[int] = []
        seen: set[int] = set()
        for cid in (
            *vec_ranked[: self._graph_seed_top_k],
            *fts_ranked[: self._graph_seed_top_k],
        ):
            if cid in seen:
                continue
            seeds.append(cid)
            seen.add(cid)
            if len(seeds) >= self._graph_seed_top_k:
                break
        if not seeds:
            return []
        try:
            return await self._storage.neighbor_chunks_via_links(
                seeds, limit=per_leg_limit
            )
        except NotSupported:
            return []

    async def _embed_query_text(self, q: str) -> list[float] | None:
        assert self._embedder is not None
        assert self._embedding_model is not None
        vectors = await self._embedder.embed([q], model=self._embedding_model)
        return vectors[0] if vectors else None

    async def _embed_query_multimodal(self, q: str) -> list[float] | None:
        assert self._mm is not None
        vectors = await self._mm.embedder.embed(
            [MultimodalInput(text=q)], model=self._mm.model
        )
        return vectors[0] if vectors else None

    @staticmethod
    def _render_chunk_snippet(chunk: ChunkRecord) -> str:
        """Compact one-line preview of a chunk's body for Hit.snippet."""
        snippet = chunk.text.strip().replace("\n", " ")
        return snippet[:240] + ("…" if len(snippet) > 240 else "")


def _sanitize_fts(q: str, *, cjk_tokenizer: CjkTokenizer = "none") -> str:
    """Tokenize a natural-language query into a bag-of-words FTS5 expression.

    The Phase 1 implementation wrapped the whole query in quotes — a
    phrase query — which never matched on multi-word natural-language
    inputs and made BM25-only retrieval (and thus the FTS leg of hybrid
    RRF) return 0 hits in eval.

    Strategy:

    0. If ``cjk_tokenizer != "none"``, pre-segment CJK runs (via the
       same ``preprocess_for_fts`` the ingest path uses) so that the
       whitespace-split below picks up word-level Chinese tokens.
       Symmetry with the indexed form is the whole point; see the
       ``SQLiteStorage.__init__`` note.
    1. Replace anything that isn't a word character, whitespace, or a
       basic CJK ideograph with whitespace. Word characters (``\\w``)
       cover ASCII letters, digits, and underscore, so identifiers like
       ``expect_any`` survive intact; CJK pass-through keeps Chinese
       (e.g. CMTEB) queries from being stripped to nothing.
    2. Split on whitespace and drop FTS5 reserved tokens
       (``AND``/``OR``/``NOT``/``NEAR``) so a user word doesn't accidentally
       turn into an operator.
    3. Quote each token (``"<token>"``) — FTS5 phrase quotes around a
       single term are a no-op semantically but prevent column-qualifier
       interpretation for tokens that happen to contain a colon.
    4. Join with ``OR`` for bag-of-words BM25 retrieval — the same
       semantics published BEIR / CMTEB BM25 baselines use.
    """
    if cjk_tokenizer != "none":
        q = preprocess_for_fts(q, tokenizer=cjk_tokenizer)
    cleaned = re.sub(rf"[^{WORD_OR_CJK_CHARS}\s]", " ", q)
    tokens = [
        t for t in cleaned.split() if t and t.upper() not in _FTS_RESERVED
    ]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)
