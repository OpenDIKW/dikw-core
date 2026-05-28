"""Per-batch retry-skip resilience for ``embed_chunks`` + ``consume_embedding_stream``.

Pins the 0.4.0 contract: a single bad embed batch (``ProviderError``
from a transient provider failure that survives SDK-level retries)
MUST NOT abort the whole persist or ingest pipeline. ``embed_chunks``
retries the batch up to ``retries`` times with linear backoff, then
skips it and yields an ``EmbedBatchResult`` with ``error`` set. The
skipped chunks remain in storage without vectors and are reconciled
by the next ingest's ``list_chunks_missing_embedding`` resume scan.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from dikw_core.domains.info.embed import (
    ChunkToEmbed,
    EmbedBatchResult,
    EmbedConsumeResult,
    consume_embedding_stream,
    embed_chunks,
)
from dikw_core.schemas import ChunkRecord, DocumentRecord, Layer
from dikw_core.storage.sqlite import SQLiteStorage

from .fakes import FlakyEmbedder, register_text_version


def _chunks(n: int) -> list[ChunkToEmbed]:
    return [ChunkToEmbed(chunk_id=i + 1, text=f"chunk-{i}") for i in range(n)]


async def _new_storage(tmp_path: Path) -> SQLiteStorage:
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    return storage


async def test_embed_chunks_retries_then_succeeds(tmp_path: Path) -> None:
    """Batch 0 raises on first attempt, retry succeeds on second.

    With retries=2: max_attempts=3. Call 0 raises, call 1 succeeds →
    one yielded result with 2 rows and ``attempts=2``.
    """
    storage = await _new_storage(tmp_path)
    try:
        version_id = await register_text_version(storage)
        # 2 chunks at batch_size=2 = one batch. Call index 0 raises,
        # the retry (call index 1) succeeds.
        embedder = FlakyEmbedder(raise_on_calls={0})

        results: list[EmbedBatchResult] = []
        async for r in embed_chunks(
            embedder,
            _chunks(2),
            model="fake",
            version_id=version_id,
            batch_size=2,
            retries=2,
            backoff_seconds=0.0,
        ):
            results.append(r)

        assert len(results) == 1
        assert len(results[0].rows) == 2
        assert results[0].error is None
        assert results[0].attempts == 2
        assert embedder.embed_calls == 2
    finally:
        await storage.close()


async def test_embed_chunks_skips_batch_after_exhausted_retries(
    tmp_path: Path,
) -> None:
    """All attempts fail → batch is skipped, ``error`` set, ``rows`` empty.

    Subsequent batches still process (skip is local to the failing batch).
    """
    storage = await _new_storage(tmp_path)
    try:
        version_id = await register_text_version(storage)
        # 4 chunks at batch_size=2 → 2 batches. Batch 0's calls (indices
        # 0, 1, 2) all raise (retries=2 → 3 attempts). Batch 1 succeeds
        # on call index 3.
        embedder = FlakyEmbedder(raise_on_calls={0, 1, 2})

        results: list[EmbedBatchResult] = []
        async for r in embed_chunks(
            embedder,
            _chunks(4),
            model="fake",
            version_id=version_id,
            batch_size=2,
            retries=2,
            backoff_seconds=0.0,
        ):
            results.append(r)

        assert len(results) == 2
        # Batch 0 skipped — 3 attempts, no rows, chunk_ids recorded.
        assert results[0].rows == []
        assert results[0].error is not None
        assert "ProviderError" in results[0].error or "FlakyEmbedder" in results[0].error
        assert results[0].attempts == 3
        assert results[0].skipped_chunk_ids == [1, 2]
        # Batch 1 succeeded — single attempt.
        assert len(results[1].rows) == 2
        assert results[1].error is None
        assert results[1].attempts == 1
        # Total provider calls: 3 (batch 0 attempts) + 1 (batch 1) = 4.
        assert embedder.embed_calls == 4
    finally:
        await storage.close()


async def test_embed_chunks_retries_zero_means_no_retry(tmp_path: Path) -> None:
    """``retries=0`` → one attempt then skip on ProviderError.

    Validates the "no retry" off-switch for callers that want fast-fail-
    then-defer behaviour (one attempt, skip on failure — different from
    one-attempt-then-raise, which would require not using this path).
    """
    storage = await _new_storage(tmp_path)
    try:
        version_id = await register_text_version(storage)
        embedder = FlakyEmbedder(raise_on_calls={0})

        results: list[EmbedBatchResult] = []
        async for r in embed_chunks(
            embedder,
            _chunks(2),
            model="fake",
            version_id=version_id,
            batch_size=2,
            retries=0,
            backoff_seconds=0.0,
        ):
            results.append(r)

        assert len(results) == 1
        assert results[0].rows == []
        assert results[0].error is not None
        assert results[0].attempts == 1
        assert embedder.embed_calls == 1
    finally:
        await storage.close()


async def test_consume_stream_returns_embedded_and_skipped_counts(
    tmp_path: Path,
) -> None:
    """``EmbedConsumeResult`` carries both success + skip counts.

    Three batches (4 chunks at batch_size=2 with a third batch added via
    a 5th chunk): batch 0 skipped, batches 1 & 2 succeed → embedded=3,
    chunks_skipped=2, batches_skipped=1.
    """
    storage = await _new_storage(tmp_path)
    try:
        version_id = await register_text_version(storage)
        # Seed 5 chunks via real upsert so the FK from vec_chunks_v<vid>
        # to chunks(chunk_id) is satisfied when batches 1/2 land.
        doc_id = "source:sources/x.md"
        await storage.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                path="sources/x.md",
                title="X",
                hash="h1",
                mtime=time.time(),
                layer=Layer.SOURCE,
                active=True,
            )
        )
        records = [
            ChunkRecord(doc_id=doc_id, seq=i, start=i, end=i + 5, text=f"chunk-{i}")
            for i in range(5)
        ]
        chunk_ids = await storage.replace_chunks(doc_id, records)

        # 5 chunks at batch_size=2 → 3 batches of sizes [2, 2, 1].
        # Batch 0 always fails (calls 0,1,2 at retries=2). Batches 1, 2
        # succeed on calls 3 and 4 respectively.
        embedder = FlakyEmbedder(raise_on_calls={0, 1, 2})
        to_embed = [
            ChunkToEmbed(chunk_id=cid, text=r.text)
            for cid, r in zip(chunk_ids, records, strict=True)
        ]

        result = await consume_embedding_stream(
            embed_chunks(
                embedder,
                to_embed,
                model="fake",
                version_id=version_id,
                batch_size=2,
                retries=2,
                backoff_seconds=0.0,
            ),
            storage,
        )

        assert isinstance(result, EmbedConsumeResult)
        assert result.embedded == 3
        assert result.batches_skipped == 1
        assert result.chunks_skipped == 2
        assert len(result.errors) == 1
    finally:
        await storage.close()


async def test_consume_stream_skipped_chunks_remain_in_storage_without_vectors(
    tmp_path: Path,
) -> None:
    """After a skip, the chunks exist in ``chunks`` but not in
    ``vec_chunks_v<vid>`` — ``list_chunks_missing_embedding`` surfaces
    them for the next ingest's resume scan.
    """
    storage = await _new_storage(tmp_path)
    try:
        version_id = await register_text_version(storage)
        # Seed two source chunks so list_chunks_missing_embedding has
        # something to find. We bypass the full ingest path and write
        # chunks directly via storage to keep the test focused on the
        # embed-side skip behaviour.
        doc_id = "source:sources/x.md"
        await storage.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                path="sources/x.md",
                title="X",
                hash="h1",
                mtime=time.time(),
                layer=Layer.SOURCE,
                active=True,
            )
        )
        records = [
            ChunkRecord(doc_id=doc_id, seq=0, start=0, end=5, text="alpha"),
            ChunkRecord(doc_id=doc_id, seq=1, start=5, end=10, text="bravo"),
        ]
        chunk_ids = await storage.replace_chunks(doc_id, records)

        # Embed with a flaky provider that ALWAYS fails → all chunks
        # skipped, none embedded.
        embedder = FlakyEmbedder(raise_on_calls={0, 1, 2, 3, 4, 5})
        to_embed = [
            ChunkToEmbed(chunk_id=cid, text=r.text)
            for cid, r in zip(chunk_ids, records, strict=True)
        ]
        result = await consume_embedding_stream(
            embed_chunks(
                embedder,
                to_embed,
                model="fake",
                version_id=version_id,
                batch_size=2,
                retries=1,  # 2 attempts then skip
                backoff_seconds=0.0,
            ),
            storage,
        )
        assert result.embedded == 0
        assert result.chunks_skipped == 2

        # The chunks remain in storage (chunks table) but vec table has
        # no rows for them — list_chunks_missing_embedding picks them up.
        missing = await storage.list_chunks_missing_embedding(
            version_id=version_id
        )
        missing_ids = {c.chunk_id for c in missing}
        assert set(chunk_ids).issubset(missing_ids), (
            f"expected {chunk_ids} to be in missing list, got {missing_ids}"
        )
    finally:
        await storage.close()


async def test_embed_chunks_retry_calls_asyncio_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero ``backoff_seconds`` invokes ``asyncio.sleep`` between
    attempts. Stub the sleep so the test stays fast while still
    exercising the wait branch.
    """
    storage = await _new_storage(tmp_path)
    try:
        version_id = await register_text_version(storage)
        sleep_calls: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        # Patch the asyncio.sleep that embed.py imports.
        import dikw_core.domains.info.embed as _embed_mod

        monkeypatch.setattr(_embed_mod.asyncio, "sleep", fake_sleep)

        # Batch fails on first attempt, succeeds on retry. backoff=1.5
        # gives linear backoff: sleep(1.5 * 1) = 1.5s on the one retry.
        embedder = FlakyEmbedder(raise_on_calls={0})
        results: list[EmbedBatchResult] = []
        async for r in embed_chunks(
            embedder,
            _chunks(2),
            model="fake",
            version_id=version_id,
            batch_size=2,
            retries=2,
            backoff_seconds=1.5,
        ):
            results.append(r)

        assert len(results) == 1
        assert results[0].error is None
        assert sleep_calls == [1.5]
    finally:
        await storage.close()
