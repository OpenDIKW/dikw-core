"""Embedding worker.

Two pathways:

* **Text** — ``embed_chunks(provider: EmbeddingProvider, ...)`` — the
  unimodal path. Returns ``EmbeddingRow(chunk_id, version_id, embedding)``
  for ``Storage.upsert_embeddings`` which routes into the per-version
  ``vec_chunks_v<id>`` table.

* **Multimodal** — ``embed_chunks_multimodal`` and ``embed_assets`` route
  through ``MultimodalEmbeddingProvider`` so chunks (text-only inputs)
  and asset binaries (image-only inputs) land in the *same* shared
  vector space, identified by their own multimodal ``version_id``.

Every embedding row carries a ``version_id`` from ``embed_versions``
so storage can isolate per-version vectors and the cache by identity.

Both pathways batch to keep HTTP round-trips low without overwhelming
providers' per-request input caps.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ...progress import ProgressReporter
from ...providers import EmbeddingProvider, MultimodalEmbeddingProvider
from ...providers.base import ProviderError
from ...schemas import (
    AssetEmbeddingRow,
    AssetRecord,
    CachedEmbeddingRow,
    EmbeddingRow,
    ImageContent,
    MultimodalInput,
)
from ...storage.base import NotSupported, Storage
from ..data.backends.markdown import content_hash

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbedBatchResult:
    """Outcome of one embed batch.

    On success ``rows`` carries the embeddings and ``error`` is None.
    On skip (after exhausted retries) ``rows`` is empty,
    ``skipped_chunk_ids`` lists the chunks that were not embedded, and
    ``error`` is the final failure message.

    Skipped chunks remain in storage without vectors and get picked up
    by the next ingest's ``list_chunks_missing_embedding`` resume scan.
    """

    rows: list[EmbeddingRow]
    skipped_chunk_ids: list[int] = field(default_factory=list)
    error: str | None = None
    attempts: int = 1


@dataclass(frozen=True)
class EmbedConsumeResult:
    """Aggregate outcome of ``consume_embedding_stream``."""

    embedded: int = 0
    batches_skipped: int = 0
    chunks_skipped: int = 0
    errors: list[str] = field(default_factory=list)


async def _safe_cache_lookup(
    storage: Storage, hashes: Sequence[str], version_id: int
) -> dict[str, list[float]]:
    """Cache lookup that degrades to empty on NotSupported / transient faults."""
    try:
        return await storage.get_cached_embeddings(hashes, version_id=version_id)
    except NotSupported:
        return {}
    except Exception as exc:
        logger.warning("embed_cache lookup failed: %s; falling back", exc)
        return {}


async def _safe_cache_write(
    storage: Storage, rows: Sequence[CachedEmbeddingRow]
) -> None:
    """Cache write-back that degrades to no-op on NotSupported / transient faults."""
    try:
        await storage.cache_embeddings(rows)
    except NotSupported:
        return
    except Exception as exc:
        logger.warning("embed_cache write failed: %s; ignoring", exc)


@dataclass(frozen=True)
class ChunkToEmbed:
    chunk_id: int
    text: str


async def embed_chunks(
    provider: EmbeddingProvider,
    chunks: Sequence[ChunkToEmbed],
    *,
    model: str,
    version_id: int,
    storage: Storage | None = None,
    batch_size: int = 64,
    retries: int = 0,
    backoff_seconds: float = 0.0,
    reporter: ProgressReporter | None = None,
) -> AsyncIterator[EmbedBatchResult]:
    """Embed ``chunks`` in fixed-size batches with optional content-hash cache.

    Streams one ``EmbedBatchResult`` per batch so the caller can persist
    each batch immediately. Per-batch failure cost is bounded: prior
    batches are already on disk when the next API call raises.

    Per-batch retry-skip: on ``ProviderError`` the batch is retried up
    to ``retries`` more times with linear backoff (``backoff_seconds``
    * attempt). After exhausted retries the batch yields an
    ``EmbedBatchResult`` with ``error`` set and empty ``rows`` — the
    chunks remain in storage without vectors and get reconciled by the
    next ingest's missing-embedding resume scan. ``retries=0`` (the
    default) means a single attempt then skip on failure, so a caller
    that wants the legacy "first failure raises" behaviour must not
    use this path (or wrap with try/except on the consumer side).

    When ``storage`` is supplied, each batch first looks up
    ``sha256(chunk.text)`` in ``storage.embed_cache`` and routes hits
    around the provider. Misses are sent to the provider; their vectors
    are written to the cache after success. ``storage=None`` skips the
    cache entirely. Adapters that raise ``NotSupported`` silently
    degrade to no-cache.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    async def _call(misses: list[ChunkToEmbed]) -> list[list[float]]:
        return await provider.embed([c.text for c in misses], model=model)

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(chunks), batch_size)):
        batch = chunks[start : start + batch_size]
        # INFO-level so a long-running ingest run shows progress without
        # needing a debug flag — debugging "did the embedder hang?"
        # without per-batch markers means restarting from scratch.
        logger.info(
            "embed batch %d/%d (chunks %d-%d)",
            batch_idx + 1,
            total_batches,
            start,
            start + len(batch) - 1,
        )
        yield await _run_batch_with_retry(
            batch,
            model=model,
            version_id=version_id,
            storage=storage,
            embed_misses=_call,
            error_template="embedding provider returned {got} vectors for {want} texts",
            retries=retries,
            backoff_seconds=backoff_seconds,
            reporter=reporter,
            batch_idx=batch_idx,
            total_batches=total_batches,
        )


async def embed_chunks_multimodal(
    provider: MultimodalEmbeddingProvider,
    chunks: Sequence[ChunkToEmbed],
    *,
    model: str,
    version_id: int,
    storage: Storage | None = None,
    batch_size: int = 16,
    retries: int = 0,
    backoff_seconds: float = 0.0,
    reporter: ProgressReporter | None = None,
) -> AsyncIterator[EmbedBatchResult]:
    """Embed text chunks via the multimodal provider (text-only inputs).

    Same per-batch retry-skip contract as ``embed_chunks`` — see that
    function's docstring for full semantics. Uses the same
    ``EmbeddingRow`` output so chunk vectors can land in the same
    ``vec_chunks_v<version_id>`` table.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not chunks:
        # Still notify the provider so its last_inputs reflects an empty
        # call (a legacy contract some tests rely on). Empty generator
        # otherwise.
        await provider.embed([], model=model)
        return

    async def _call(misses: list[ChunkToEmbed]) -> list[list[float]]:
        return await provider.embed(
            [MultimodalInput(text=c.text) for c in misses], model=model
        )

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(chunks), batch_size)):
        batch = chunks[start : start + batch_size]
        yield await _run_batch_with_retry(
            batch,
            model=model,
            version_id=version_id,
            storage=storage,
            embed_misses=_call,
            error_template="multimodal provider returned {got} vectors for {want} chunk inputs",
            retries=retries,
            backoff_seconds=backoff_seconds,
            reporter=reporter,
            batch_idx=batch_idx,
            total_batches=total_batches,
        )


async def _run_batch_with_retry(
    batch: Sequence[ChunkToEmbed],
    *,
    model: str,
    version_id: int,
    storage: Storage | None,
    embed_misses: Callable[[list[ChunkToEmbed]], Awaitable[list[list[float]]]],
    error_template: str,
    retries: int,
    backoff_seconds: float,
    reporter: ProgressReporter | None,
    batch_idx: int,
    total_batches: int,
) -> EmbedBatchResult:
    """Call ``_embed_one_batch`` with bounded ``ProviderError`` retry-skip.

    On final failure returns a skip ``EmbedBatchResult`` whose
    ``rows`` is empty and ``error`` describes the failure. The caller
    (``consume_embedding_stream``) leaves the chunks in storage without
    vectors so the next ingest's missing-embedding resume scan picks
    them up. ``retries=0`` means a single attempt then skip.
    """
    max_attempts = retries + 1
    last_error: ProviderError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            rows = await _embed_one_batch(
                batch,
                model=model,
                version_id=version_id,
                storage=storage,
                embed_misses=embed_misses,
                error_template=error_template,
            )
            return EmbedBatchResult(rows=rows, attempts=attempt)
        except ProviderError as pe:
            last_error = pe
            if attempt < max_attempts:
                logger.warning(
                    "embed batch %d/%d failed (attempt %d/%d): %s; retrying",
                    batch_idx + 1,
                    total_batches,
                    attempt,
                    max_attempts,
                    pe,
                )
                if reporter is not None:
                    await reporter.progress(
                        phase="embed_batch",
                        current=batch_idx,
                        total=total_batches,
                        detail={
                            "status": "retrying",
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "batch_idx": batch_idx,
                            "error_kind": type(pe).__name__,
                            "error_msg": str(pe)[:200],
                        },
                    )
                if backoff_seconds > 0:
                    await asyncio.sleep(backoff_seconds * attempt)
                if reporter is not None:
                    reporter.cancel_token().raise_if_cancelled()
            else:
                logger.warning(
                    "embed batch %d/%d skipped after %d attempt(s): %s",
                    batch_idx + 1,
                    total_batches,
                    attempt,
                    pe,
                )
                if reporter is not None:
                    await reporter.progress(
                        phase="embed_batch",
                        current=batch_idx + 1,
                        total=total_batches,
                        detail={
                            "status": "skipped",
                            "attempts": attempt,
                            "batch_idx": batch_idx,
                            "error_kind": type(pe).__name__,
                            "error_msg": str(pe)[:200],
                        },
                    )
    # Unreachable when last_error is set (loop above always returns or
    # falls through with last_error populated), but mypy needs a clear
    # post-loop value.
    error_msg = str(last_error) if last_error is not None else "unknown"
    return EmbedBatchResult(
        rows=[],
        skipped_chunk_ids=[c.chunk_id for c in batch],
        error=error_msg,
        attempts=max_attempts,
    )


async def _embed_one_batch(
    batch: Sequence[ChunkToEmbed],
    *,
    model: str,
    version_id: int,
    storage: Storage | None,
    embed_misses: Callable[[list[ChunkToEmbed]], Awaitable[list[list[float]]]],
    error_template: str,
) -> list[EmbeddingRow]:
    """Embed one batch, consulting the cache when ``storage`` is given.

    Returns rows in input order regardless of how hits/misses interleave.
    Within-batch duplicate hashes are sent to the provider once — vectors
    for the same content under the same version_id are deterministic
    by cache contract, so deduping saves a round-trip without changing
    semantics.
    """
    hashes = [content_hash(c.text) for c in batch]
    cached: dict[str, list[float]] = {}
    if storage is not None:
        cached = await _safe_cache_lookup(storage, hashes, version_id)

    miss_chunks: list[ChunkToEmbed] = []
    miss_hashes: list[str] = []
    queued: set[str] = set()
    for chunk, h in zip(batch, hashes, strict=True):
        if h in cached or h in queued:
            continue
        miss_chunks.append(chunk)
        miss_hashes.append(h)
        queued.add(h)

    new_vectors: list[list[float]] = []
    if miss_chunks:
        new_vectors = await embed_misses(miss_chunks)
        if len(new_vectors) != len(miss_chunks):
            raise RuntimeError(
                error_template.format(got=len(new_vectors), want=len(miss_chunks))
            )

    cached.update(zip(miss_hashes, new_vectors, strict=True))
    new_cache_rows = [
        CachedEmbeddingRow(
            content_hash=h, version_id=version_id, dim=len(v), embedding=v
        )
        for h, v in zip(miss_hashes, new_vectors, strict=True)
    ]

    rows = [
        EmbeddingRow(chunk_id=c.chunk_id, version_id=version_id, embedding=cached[h])
        for c, h in zip(batch, hashes, strict=True)
    ]

    if storage is not None and new_cache_rows:
        await _safe_cache_write(storage, new_cache_rows)

    return rows


def is_unembeddable_asset_mime(mime: str) -> bool:
    """Return ``True`` for MIME types that ``embed_assets`` will silently
    skip (no vector + no ``asset_embed_meta`` row produced).

    Currently SVG only — v1 doesn't rasterize. Callers who scan for
    "missing" embeddings (e.g., the resume-scan in ``api.ingest``) use
    this to filter the candidate list, otherwise SVGs reappear in
    every ingest's "needs embedding" pass forever.
    """
    return mime == "image/svg+xml"


async def embed_assets(
    provider: MultimodalEmbeddingProvider,
    assets: Sequence[AssetRecord],
    *,
    project_root: Path,
    model: str,
    version_id: int,
    batch_size: int = 16,
) -> AsyncIterator[list[AssetEmbeddingRow]]:
    """Embed asset binaries via the multimodal provider, streaming
    one ``list[AssetEmbeddingRow]`` per provider call.

    Mirrors ``embed_chunks``'s per-batch yield contract so the caller
    can persist each batch immediately. Without this, a partial-failure
    on a large backfill (e.g. text-only -> multimodal migration with
    thousands of unembedded images) wastes the API spend on every
    successful prior batch — the next retry re-embeds them from scratch.

    Reads each asset's bytes from ``project_root / asset.stored_path``
    just before its batch goes to the provider; skipped assets (SVG,
    unreadable) don't contribute to the request payload but the slice
    still yields (empty list) so the caller's progress bookkeeping
    stays aligned with slice count, not embedded count.

    Output rows are tagged with ``version_id`` so storage can route them
    into the correct ``vec_assets_v<id>`` table.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not assets:
        return

    for start in range(0, len(assets), batch_size):
        slice_ = assets[start : start + batch_size]
        batch_pairs: list[tuple[AssetRecord, ImageContent]] = []
        for asset in slice_:
            if is_unembeddable_asset_mime(asset.mime):
                logger.info(
                    "skipping unembeddable asset (%s): %s",
                    asset.mime,
                    asset.stored_path,
                )
                continue
            path = project_root / asset.stored_path
            try:
                data = path.read_bytes()
            except OSError as e:
                logger.warning(
                    "skipping asset embedding for %s (read failed: %s)",
                    asset.stored_path,
                    e,
                )
                continue
            batch_pairs.append((asset, ImageContent(bytes=data, mime=asset.mime)))

        if not batch_pairs:
            yield []
            continue

        inputs = [MultimodalInput(images=[img]) for _, img in batch_pairs]
        vectors = await provider.embed(inputs, model=model)
        if len(vectors) != len(batch_pairs):
            raise RuntimeError(
                f"multimodal provider returned {len(vectors)} vectors for "
                f"{len(batch_pairs)} asset inputs"
            )
        yield [
            AssetEmbeddingRow(
                asset_id=asset.asset_id, version_id=version_id, embedding=v
            )
            for (asset, _), v in zip(batch_pairs, vectors, strict=True)
        ]


async def consume_embedding_stream(
    stream: AsyncIterator[EmbedBatchResult],
    storage: Storage,
    *,
    on_batch: Callable[[], None] | None = None,
    reporter: ProgressReporter | None = None,
    phase: str = "embed",
    total: int = 0,
) -> EmbedConsumeResult:
    """Drain an embed_chunks-style stream, upserting each batch as it arrives.

    Per-batch upsert is the durability guarantee: a mid-flight provider
    failure leaves prior batches on disk instead of throwing away the
    entire run's API spend. Skipped batches (where ``embed_chunks``
    exhausted retries and yielded ``EmbedBatchResult(rows=[],
    error=…)``) leave their chunks in storage without vectors — the
    next ingest's missing-embedding resume scan picks them up.

    ``on_batch`` keeps the in-process rich progress bar's CLI callback
    surface; ``reporter`` is the new structured channel a server task
    listens on. Both fire per batch so a single ingest call can drive a
    local TTY *and* a remote NDJSON subscriber simultaneously.

    Returns ``EmbedConsumeResult`` so callers can surface skip counts
    in their report (``ApplyReport.chunks_pending_embedding`` etc.).
    """
    embedded = 0
    batches_done = 0
    batches_skipped = 0
    chunks_skipped = 0
    errors: list[str] = []
    async for result in stream:
        if result.rows:
            await storage.upsert_embeddings(result.rows)
            embedded += len(result.rows)
        if result.error is not None:
            batches_skipped += 1
            chunks_skipped += len(result.skipped_chunk_ids)
            errors.append(result.error)
        batches_done += 1
        if on_batch is not None:
            on_batch()
        if reporter is not None:
            await reporter.progress(
                phase=phase, current=batches_done, total=total
            )
            reporter.cancel_token().raise_if_cancelled()
    return EmbedConsumeResult(
        embedded=embedded,
        batches_skipped=batches_skipped,
        chunks_skipped=chunks_skipped,
        errors=errors,
    )
