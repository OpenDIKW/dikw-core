"""Retrieval cluster of the engine facade: ``retrieve`` + its internals.

``retrieve`` runs the RRF-fused hybrid (+ optional multimodal) search and
returns ranked chunks + page-level refs. No LLM — answer synthesis is the
agent layer's job (Karpathy's rule: retrieval is deterministic scoping).

rank3 cluster: imports ``api_core`` (``_with_storage``), providers, the
search primitives, and the leaf schemas — never the ``api`` facade. ``api``
re-exports ``retrieve`` (public, in ``__all__``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .api_core import _with_storage
from .config import DikwConfig
from .domains.info.search import HybridSearcher, MultimodalSearch
from .progress import NoopReporter, ProgressReporter
from .providers import (
    EmbeddingProvider,
    MultimodalEmbeddingProvider,
    build_embedder,
    build_multimodal_embedder,
)
from .schemas import Hit, PageRef, RetrieveResult
from .storage import Storage
from .storage.base import NotSupported

logger = logging.getLogger(__name__)


async def _retrieve_inner(
    storage: Storage,
    cfg: DikwConfig,
    q: str,
    *,
    limit: int,
    embedder: EmbeddingProvider | None = None,
    multimodal_embedder: MultimodalEmbeddingProvider | None = None,
    reporter: ProgressReporter | None = None,
) -> tuple[list[Hit], MultimodalEmbeddingProvider | None]:
    """Run hybrid search and return (hits, owned_mm_embedder).

    Shared helper between ``query`` (LLM-driven RAG) and ``retrieve``
    (retrieval-only). The caller is responsible for closing
    ``owned_mm_embedder`` (returned non-None only when this helper had
    to build the multimodal embedder itself; a caller-supplied embedder
    is never owned here).

    Emits the ``retrieval_done`` partial via ``reporter`` so the
    ``/v1/retrieve`` wire surface reports a stable shape; pass ``None``
    for ``reporter`` to silence the partial.
    """
    _reporter: ProgressReporter = reporter or NoopReporter()
    owned_mm: MultimodalEmbeddingProvider | None = None

    try:
        # Pin the text leg to the active text version's stored model AND
        # dim so a mid-flight cfg edit (new embedding_model /
        # embedding_dim in dikw.yml, no re-ingest) doesn't corrupt query
        # rankings — same anti-drift guard the multimodal path applies
        # below. We resolve the active version BEFORE building the
        # embedder so the override can flow into ``default_dimensions``.
        text_version_id: int | None = None
        text_query_model = cfg.provider.embedding_model
        text_query_dim: int | None = None
        try:
            active_text = await storage.get_active_embed_version(modality="text")
        except NotSupported as e:
            logger.warning(
                "storage backend doesn't support text versioning (%s); "
                "querying with the cfg embedding_model unchecked",
                e,
            )
            active_text = None
        if active_text is not None and active_text.version_id is not None:
            text_version_id = active_text.version_id
            text_query_model = active_text.model
            text_query_dim = active_text.dim

        _embedder = embedder
        if _embedder is None:
            _embedder = build_embedder(cfg.provider, dim_override=text_query_dim)

        mm_search: MultimodalSearch | None = None
        mm_cfg = cfg.assets.multimodal
        if mm_cfg is not None:
            try:
                active = await storage.get_active_embed_version(
                    modality="multimodal"
                )
            except NotSupported as e:
                logger.warning(
                    "storage backend doesn't support multimodal versioning "
                    "(%s); querying with text-only retrieval",
                    e,
                )
                active = None
            if active is not None and active.version_id is not None:
                mm_embedder = multimodal_embedder
                if mm_embedder is None:
                    mm_embedder = build_multimodal_embedder(
                        mm_cfg.provider,
                        base_url=mm_cfg.base_url,
                        batch=mm_cfg.batch,
                    )
                    # Assign immediately so an exception between here and
                    # the `return` below still goes through this scope's
                    # cleanup — caller's finally only sees ``owned_mm``
                    # after a successful return.
                    owned_mm = mm_embedder
                # Use the model recorded on the active version, not the
                # current cfg model — if the user just edited dikw.yml to
                # point at a new model but hasn't re-ingested yet, the
                # asset vectors in vec_assets_v<active> were produced by
                # the OLD model; querying with the new model would either
                # mismatch dim or rank against an incompatible space.
                mm_search = MultimodalSearch(
                    embedder=mm_embedder,
                    model=active.model,
                    asset_version_id=active.version_id,
                )

        searcher = HybridSearcher.from_config(
            storage,
            _embedder,
            cfg.retrieval,
            embedding_model=text_query_model,
            text_version_id=text_version_id,
            multimodal=mm_search,
        )
        hits = await searcher.search(q, limit=limit)
        # Include full ``text`` so a streaming agent can prompt off the
        # partial without waiting for ``final``. Cost: chunk bodies
        # duplicate on ``final.result.chunks`` — clients that don't
        # need the partial can stop reading after ``final``.
        await _reporter.partial(
            "retrieval_done",
            {"hits": [h.model_dump(mode="json") for h in hits]},
        )
        return hits, owned_mm
    except BaseException:
        # Catch ``BaseException`` (not just ``Exception``) so the cleanup
        # runs on ``asyncio.CancelledError`` too — a cancelled retrieve
        # mid-flight must not leak the multimodal embedder we just built.
        #
        # Inner ``except Exception`` (not ``BaseException``) is
        # intentional: if ``aclose`` itself raises ``CancelledError`` /
        # ``SystemExit`` / ``KeyboardInterrupt`` we let it propagate and
        # replace the original exception. asyncio convention treats
        # cancellation as a higher-priority signal that callers must see
        # — masking it under ``raise`` of the original would break
        # cooperative shutdown. Regular cleanup failures (network,
        # provider crash) are logged and the original exception wins.
        if owned_mm is not None and hasattr(owned_mm, "aclose"):
            try:
                await owned_mm.aclose()
            except Exception:
                logger.exception(
                    "multimodal embedder aclose failed during _retrieve_inner cleanup"
                )
        raise


def _build_page_refs(hits: list[Hit]) -> list[PageRef]:
    """Aggregate fusion-ranked chunks into page-level refs.

    ``score`` is the max chunk score for each path so an agent can
    rank pages without re-aggregating. ``hit_chunk_ids`` is captured in
    fusion-rank order (insertion order of hits) so the caller can
    cross-reference back to ``chunks[]`` deterministically. Hits with
    ``path=None`` are dropped — they cannot be cited as a page.
    """
    accum: dict[str, dict[str, Any]] = {}
    for h in hits:
        if h.path is None:
            continue
        bucket = accum.get(h.path)
        if bucket is None:
            accum[h.path] = {
                "path": h.path,
                "layer": h.layer,
                "title": h.title,
                "score": h.score,
                "hit_chunk_ids": [h.chunk_id],
            }
        else:
            bucket["hit_chunk_ids"].append(h.chunk_id)
            if h.score > bucket["score"]:
                bucket["score"] = h.score
    refs = [PageRef(**bucket) for bucket in accum.values()]
    refs.sort(key=lambda r: r.score, reverse=True)
    return refs


async def retrieve(
    q: str,
    path: str | Path | None = None,
    *,
    limit: int = 5,
    embedder: EmbeddingProvider | None = None,
    multimodal_embedder: MultimodalEmbeddingProvider | None = None,
    reporter: ProgressReporter | None = None,
) -> RetrieveResult:
    """Hybrid-search the knowledge base and return chunks + page-level refs only.

    Companion to ``query`` for retrieval-only consumers (typically AI
    agents that intend to assemble their own answer using their own
    LLM): runs the fusion + multimodal pipeline via ``_retrieve_inner``
    and stops there. dikw-core no longer ships an in-engine query verb
    (PR-1 removed it); ``retrieve`` is the sole knowledge-access entry
    point from the engine side.

    ``reporter`` (optional) emits a ``retrieval_done`` partial; the
    route layer wraps this in a ``final{result}`` event for the
    ``POST /v1/retrieve`` NDJSON wire.
    """
    cfg, _root, storage = await _with_storage(path)
    owned_mm: MultimodalEmbeddingProvider | None = None
    try:
        hits, owned_mm = await _retrieve_inner(
            storage,
            cfg,
            q,
            limit=limit,
            embedder=embedder,
            multimodal_embedder=multimodal_embedder,
            reporter=reporter,
        )
        return RetrieveResult(chunks=hits, page_refs=_build_page_refs(hits))
    finally:
        if owned_mm is not None and hasattr(owned_mm, "aclose"):
            await owned_mm.aclose()
        await storage.close()
