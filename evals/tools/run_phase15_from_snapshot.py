"""Replay an eval against an already-ingested wiki, with batched query embeds.

Phase 1.5 of the chunk-level retrieval CEO plan needs CMTEB-T2 + SciFact
real-embedder numbers. The packaged ``dikw eval`` runner re-ingests every
time and embeds queries one-at-a-time inside ``HybridSearcher.search``,
which on Gitee AI's serverless inference takes ~80-150 s per single-text
embed call (vs ~3 s for a 16-text batch). Running 300 queries x 2 modes
that need vectors (vector + hybrid) under that latency takes >12 hours
end-to-end and is what bounced the prior dogfood attempt.

This tool:
  1. Loads a wiki that has already been ingested (typically the snapshot
     of a prior in-flight eval's tempdir, copied out before the eval was
     killed).
  2. Pre-embeds every unique query text in one batched pass (using the
     wiki's own ``embedding_batch_size``).
  3. Wraps the embedder so ``HybridSearcher`` reads from the cache instead
     of re-calling the API per query.
  4. Runs the dataset queries against ``--mode all`` (bm25 / vector /
     hybrid) and prints metrics + threshold pass/fail in the same shape
     as the packaged runner.

It is intentionally a one-off recovery tool, not a runner refactor — the
permanent fix (batching query embeds inside the runner itself) is a
separate change that needs broader review. Keeping the recovery isolated
here avoids touching ``eval/runner.py`` while Phase 1.5 is in flight.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Iterable
from pathlib import Path

from dikw_core import api
from dikw_core.domains.info.search import HybridSearcher, RetrievalMode
from dikw_core.eval.dataset import DatasetSpec, load_dataset
from dikw_core.eval.runner import (
    NegativeRow,
    PerQueryRow,
    _compute_view_metrics,
    _resolve_modes,
)
from dikw_core.providers import build_embedder
from dikw_core.providers.base import EmbeddingProvider
from dikw_core.storage import build_storage
from dikw_core.storage.base import NotSupported, Storage

SEARCH_LIMIT = 100  # mirror runner.py


async def resolve_active_text_version(
    storage: Storage, *, default_model: str
) -> tuple[int | None, str, int | None]:
    """Return ``(version_id, model, dim)`` for the active text version.

    Mirrors ``api.query()``'s pinning: when storage has an active text
    ``embed_versions`` row, queries must use that row's model + dim so
    query vectors land in the same space as indexed chunks even after
    ``dikw.yml`` drifts. Falls back to ``(None, default_model, None)``
    when no active row exists or the backend can't read versions, in
    which case downstream ``build_embedder(dim_override=None)`` will
    use ``cfg.provider.embedding_dim``.
    """
    try:
        active_text = await storage.get_active_embed_version(modality="text")
    except NotSupported:
        active_text = None
    if active_text is not None and active_text.version_id is not None:
        return active_text.version_id, active_text.model, active_text.dim
    return None, default_model, None


class _CachedEmbedder:
    """Embedder wrapper that serves cached vectors and falls back to inner.

    Implements the ``EmbeddingProvider`` Protocol (``embed(texts, *,
    model)``). The cache is keyed by (model, text) so a future tool that
    swaps models mid-flight stays correct, even though Phase 1.5 only
    uses one model end-to-end.
    """

    def __init__(self, inner: EmbeddingProvider, cache: dict[tuple[str, str], list[float]]):
        self._inner = inner
        self._cache = cache

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        miss_idx: list[int] = []
        miss_text: list[str] = []
        for i, t in enumerate(texts):
            cached = self._cache.get((model, t))
            if cached is not None:
                results[i] = cached
            else:
                miss_idx.append(i)
                miss_text.append(t)
        if miss_text:
            new_vecs = await self._inner.embed(miss_text, model=model)
            for i, t, v in zip(miss_idx, miss_text, new_vecs, strict=True):
                results[i] = v
                self._cache[(model, t)] = v
        # All slots are filled by construction.
        return [v for v in results if v is not None]


def _batches(seq: list[str], n: int) -> Iterable[list[str]]:
    for start in range(0, len(seq), n):
        yield seq[start : start + n]


async def _prebatch_query_embeds(
    embedder: EmbeddingProvider,
    *,
    model: str,
    queries: list[str],
    batch_size: int,
) -> dict[tuple[str, str], list[float]]:
    """Embed every unique query once, in batches sized to the provider."""
    unique = sorted(set(queries))
    cache: dict[tuple[str, str], list[float]] = {}
    total_batches = (len(unique) + batch_size - 1) // batch_size
    for i, batch in enumerate(_batches(unique, batch_size), start=1):
        print(f"[query embed] batch {i}/{total_batches} ({len(batch)} texts)", flush=True)
        vectors = await embedder.embed(batch, model=model)
        for text, vec in zip(batch, vectors, strict=True):
            cache[(model, text)] = vec
    return cache


async def replay(base: Path, dataset_name_or_path: str, mode: str) -> int:
    spec: DatasetSpec = load_dataset(dataset_name_or_path)
    cfg, _root = api.load_base(base)
    modes: list[RetrievalMode] = _resolve_modes(mode)  # type: ignore[arg-type]

    storage = build_storage(
        cfg.storage, root=base, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()

    needs_embeddings = any(m in ("vector", "hybrid") for m in modes)

    # Mirror api.query(): pin queries to the snapshot's active text
    # embed_versions row so query vectors land in the same space as
    # indexed chunks even when dikw.yml has drifted since ingest.
    text_version_id, text_query_model, text_query_dim = await resolve_active_text_version(
        storage, default_model=cfg.provider.embedding_model
    )

    embedder: EmbeddingProvider | None = None
    if needs_embeddings:
        inner_embedder = build_embedder(cfg.provider, dim_override=text_query_dim)
        cache = await _prebatch_query_embeds(
            inner_embedder,
            model=text_query_model,
            queries=[q.q for q in spec.queries],
            batch_size=cfg.provider.embedding_batch_size,
        )
        embedder = _CachedEmbedder(inner_embedder, cache)

    searcher = HybridSearcher.from_config(
        storage,
        embedder,
        cfg.retrieval,
        embedding_model=text_query_model if needs_embeddings else None,
        text_version_id=text_version_id,
    )

    per_mode: dict[str, tuple[list[PerQueryRow], list[NegativeRow]]] = {}
    try:
        for m in modes:
            print(f"[search] mode={m} ({len(spec.queries)} queries)", flush=True)
            positives: list[PerQueryRow] = []
            negatives: list[NegativeRow] = []
            for q in spec.queries:
                hits = await searcher.search(q.q, limit=SEARCH_LIMIT, mode=m)
                ranked_stems: list[str] = []
                seen: set[str] = set()
                for h in hits:
                    stem = Path(h.path).stem if h.path else h.doc_id
                    if stem in seen:
                        continue
                    seen.add(stem)
                    ranked_stems.append(stem)
                if q.expect_none:
                    negatives.append(NegativeRow(q=q.q, ranked=tuple(ranked_stems)))
                else:
                    positives.append(
                        PerQueryRow(
                            q=q.q,
                            q_id=q.id,
                            expect_doc_any=tuple(q.doc_positives),
                            expect_chunk_any=(),
                            expect_asset_any=(),
                            ranked_docs=tuple(ranked_stems),
                            ranked_chunks=(),
                            ranked_assets=(),
                        )
                    )
            per_mode[m] = (positives, negatives)
    finally:
        await storage.close()

    canonical: RetrievalMode = "hybrid" if "hybrid" in modes else modes[0]
    canonical_pos = per_mode[canonical][0]
    metrics: dict[str, float] = {}
    if len(modes) == 1:
        metrics.update(_compute_view_metrics(canonical_pos, view="doc"))
    else:
        for m in modes:
            for k, v in _compute_view_metrics(per_mode[m][0], view="doc").items():
                metrics[f"{m}/{k}"] = v
        metrics.update(_compute_view_metrics(canonical_pos, view="doc"))

    print()
    print("=" * 70)
    print(f"dataset = {spec.name}")
    print(f"canonical mode = {canonical}")
    print(f"queries = {len(spec.queries)} ({len(canonical_pos)} positive)")
    print()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print()

    failed: list[tuple[str, float, float]] = []
    for k, threshold in spec.thresholds.items():
        observed = metrics.get(k)
        if observed is None:
            print(f"  [SKIP] {k}: not computed")
            continue
        ok = observed >= threshold
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {k}: observed={observed:.4f}  threshold={threshold:.4f}")
        if not ok:
            failed.append((k, observed, threshold))

    if failed:
        print()
        print(f"FAILED {len(failed)} threshold(s) — Phase 1.5 gate not cleared.")
        return 1
    print()
    print("All thresholds passed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="Path to ingested base")
    parser.add_argument("--dataset", required=True, help="Dataset name or path")
    parser.add_argument("--mode", default="all", help="bm25 | vector | hybrid | all")
    args = parser.parse_args()
    rc = asyncio.run(replay(args.base, args.dataset, args.mode))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
