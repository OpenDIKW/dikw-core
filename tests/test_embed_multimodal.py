"""Tests for the multimodal-aware embed pipeline.

Covers ``embed_chunks_multimodal`` (text chunk → vector via mm provider)
and ``embed_assets`` (image bytes from disk → vector via mm provider).
Both run through the same ``MultimodalEmbeddingProvider`` so chunks
and assets share one vector space.
"""

from __future__ import annotations

import time
from pathlib import Path

from dikw_core.domains.info.embed import ChunkToEmbed, embed_assets, embed_chunks_multimodal
from dikw_core.schemas import (
    AssetEmbeddingRow,
    AssetKind,
    AssetRecord,
    ImageContent,
    MultimodalInput,
)
from tests.fakes import FakeMultimodalEmbedding


def _asset(asset_id: str, *, stored_path: str, mime: str = "image/png") -> AssetRecord:
    return AssetRecord(
        asset_id=asset_id,
        kind=AssetKind.IMAGE,
        mime=mime,
        stored_path=stored_path,
        original_paths=["x.png"],
        bytes=1,
        created_ts=time.time(),
    )


# ---- embed_chunks_multimodal --------------------------------------------


async def _collect(gen: object) -> list:
    """Flatten an async generator of batches into a single list of rows.

    Used by both the chunk-multimodal and asset embed pipelines.
    ``embed_chunks_multimodal`` now yields ``EmbedBatchResult`` (with a
    ``rows`` attribute), while ``embed_assets`` still yields plain
    ``list[AssetEmbeddingRow]`` (asset embeddings have no retry-skip
    contract yet — pending the 0.5.x multimodal pipeline pass).
    """
    out: list = []
    async for batch in gen:  # type: ignore[attr-defined]
        rows = getattr(batch, "rows", None)
        if rows is not None:
            out.extend(rows)
        else:
            out.extend(batch)
    return out


async def test_embed_chunks_multimodal_returns_one_row_per_chunk() -> None:
    fake = FakeMultimodalEmbedding(dim=4)
    chunks = [
        ChunkToEmbed(chunk_id=10, text="alpha"),
        ChunkToEmbed(chunk_id=20, text="beta"),
    ]
    rows = await _collect(
        embed_chunks_multimodal(fake, chunks, model="fake-mm-v1", version_id=1)
    )
    assert len(rows) == 2
    assert rows[0].chunk_id == 10
    assert rows[1].chunk_id == 20
    assert all(len(r.embedding) == 4 for r in rows)
    # The mm provider was given pure text inputs (no images).
    assert all(inp.text is not None and not inp.images for inp in fake.last_inputs)
    assert fake.last_model == "fake-mm-v1"


async def test_embed_chunks_multimodal_empty_returns_empty() -> None:
    fake = FakeMultimodalEmbedding(dim=4)
    rows = await _collect(embed_chunks_multimodal(fake, [], model="any", version_id=1))
    assert rows == []
    assert fake.last_inputs == []


async def test_embed_chunks_multimodal_preserves_order() -> None:
    fake = FakeMultimodalEmbedding(dim=4)
    chunks = [
        ChunkToEmbed(chunk_id=1, text="x"),
        ChunkToEmbed(chunk_id=2, text="y"),
        ChunkToEmbed(chunk_id=3, text="z"),
    ]
    rows = await _collect(embed_chunks_multimodal(fake, chunks, model="m", version_id=1))
    assert [r.chunk_id for r in rows] == [1, 2, 3]


async def test_embed_chunks_multimodal_batches() -> None:
    """Batch boundary handling: 5 chunks at batch_size=2 should still
    produce 5 ordered rows."""
    fake = FakeMultimodalEmbedding(dim=4)
    chunks = [ChunkToEmbed(chunk_id=i, text=str(i)) for i in range(5)]
    rows = await _collect(
        embed_chunks_multimodal(fake, chunks, model="m", version_id=1, batch_size=2)
    )
    assert [r.chunk_id for r in rows] == [0, 1, 2, 3, 4]


# ---- embed_assets --------------------------------------------------------


async def test_embed_assets_reads_bytes_from_stored_path(tmp_path: Path) -> None:
    """embed_assets resolves stored_path against project_root, reads bytes,
    feeds them to the mm provider, and returns one AssetEmbeddingRow per
    asset tagged with the given version_id."""
    project_root = tmp_path / "proj"
    project_root.mkdir()
    asset_dir = project_root / "assets" / "ab"
    asset_dir.mkdir(parents=True)
    (asset_dir / "ab123456-x.png").write_bytes(b"\x89PNG\r\n\x1a\nfakedata")

    a = _asset("ab" + "0" * 62, stored_path="assets/ab/ab123456-x.png")
    fake = FakeMultimodalEmbedding(dim=8)
    rows = await _collect(
        embed_assets(
            fake,
            [a],
            project_root=project_root,
            model="fake-mm-v1",
            version_id=42,
        )
    )
    assert len(rows) == 1
    r = rows[0]
    assert r.asset_id == a.asset_id
    assert r.version_id == 42
    assert len(r.embedding) == 8
    # The mm provider received an image-only input.
    assert len(fake.last_inputs) == 1
    inp = fake.last_inputs[0]
    assert inp.text is None
    assert len(inp.images) == 1
    assert inp.images[0].mime == "image/png"
    assert inp.images[0].bytes == b"\x89PNG\r\n\x1a\nfakedata"


async def test_embed_assets_empty_returns_empty(tmp_path: Path) -> None:
    fake = FakeMultimodalEmbedding(dim=4)
    rows = await _collect(
        embed_assets(fake, [], project_root=tmp_path, model="m", version_id=1)
    )
    assert rows == []


async def test_embed_assets_streams_reads_per_batch(tmp_path: Path) -> None:
    """File bytes are read per-batch, not all upfront. We can't probe
    peak RSS in a unit test, but we can prove the per-batch read by
    deleting the file for batch 2 AFTER batch 1 completes — if the
    function had slurped everything upfront, the test would still
    succeed (bytes already in memory); with per-batch streaming, the
    second batch's read fails and that asset is logged + skipped.
    """
    project_root = tmp_path / "proj"
    project_root.mkdir()
    asset_dir = project_root / "assets" / "00"
    asset_dir.mkdir(parents=True)

    # Two assets across two batches (batch_size=1).
    (asset_dir / "first.png").write_bytes(b"\x89PNG\r\n\x1a\nfirst")
    (asset_dir / "second.png").write_bytes(b"\x89PNG\r\n\x1a\nsecond")
    a1 = _asset("01" + "0" * 62, stored_path="assets/00/first.png")
    a2 = _asset("02" + "0" * 62, stored_path="assets/00/second.png")

    # Wrap the provider so we can delete the second file BEFORE batch 2
    # gets its read. With pre-read (slurp-all), bytes are already in
    # memory and the test would pass with 2 rows. With per-batch read,
    # batch 2's read raises OSError and ``embed_assets`` skips it.
    inner = FakeMultimodalEmbedding(dim=4)

    class _DeleteBetweenBatches:
        async def embed(self, inputs, *, model):
            # Delete the second file the moment batch 1 hits the provider.
            (asset_dir / "second.png").unlink(missing_ok=True)
            return await inner.embed(inputs, model=model)

    rows = await _collect(
        embed_assets(
            _DeleteBetweenBatches(),
            [a1, a2],
            project_root=project_root,
            model="fake-mm-v1",
            version_id=1,
            batch_size=1,
        )
    )
    assert len(rows) == 1, "second batch's read should have failed and been skipped"
    assert rows[0].asset_id == a1.asset_id


async def test_embed_assets_preserves_order(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir()
    bin_dir = project_root / "assets" / "00"
    bin_dir.mkdir(parents=True)
    paths = []
    for i in range(3):
        p = f"assets/00/0000000{i}-x.png"
        (project_root / p).write_bytes(f"data-{i}".encode())
        paths.append(p)
    assets = [_asset("a" + str(i) + "0" * 62, stored_path=p) for i, p in enumerate(paths)]
    fake = FakeMultimodalEmbedding(dim=4)
    rows = await _collect(
        embed_assets(fake, assets, project_root=project_root, model="m", version_id=7)
    )
    assert [r.asset_id for r in rows] == [a.asset_id for a in assets]


async def test_embed_assets_skips_unreadable(tmp_path: Path) -> None:
    """An asset whose stored_path is missing on disk should be silently
    skipped (logged at the call site, but never raise) — the rest of the
    batch still embeds."""
    project_root = tmp_path / "proj"
    project_root.mkdir()
    good_dir = project_root / "assets" / "11"
    good_dir.mkdir(parents=True)
    (good_dir / "11abcdef-good.png").write_bytes(b"good")

    a_good = _asset("11" + "0" * 62, stored_path="assets/11/11abcdef-good.png")
    a_missing = _asset(
        "22" + "0" * 62, stored_path="assets/22/22abcdef-missing.png"
    )

    fake = FakeMultimodalEmbedding(dim=4)
    rows = await _collect(
        embed_assets(
            fake,
            [a_good, a_missing],
            project_root=project_root,
            model="m",
            version_id=1,
        )
    )
    assert len(rows) == 1
    assert rows[0].asset_id == a_good.asset_id


async def test_embed_assets_yields_per_batch_for_durability(
    tmp_path: Path,
) -> None:
    """``embed_assets`` must yield one ``list[AssetEmbeddingRow]`` per
    provider batch so the caller can ``upsert_asset_embeddings`` each
    one immediately. Without this contract a mid-flight failure
    discards every prior successful batch's API spend on retry.
    """
    project_root = tmp_path / "proj"
    project_root.mkdir()
    asset_dir = project_root / "assets" / "00"
    asset_dir.mkdir(parents=True)
    paths = []
    for i in range(5):
        p = f"assets/00/{i:08x}-x.png"
        (project_root / p).write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([i]))
        paths.append(p)
    assets = [_asset(f"a{i:02d}" + "0" * 60, stored_path=p) for i, p in enumerate(paths)]

    fake = FakeMultimodalEmbedding(dim=4)
    yielded: list[list[AssetEmbeddingRow]] = []
    async for batch in embed_assets(
        fake,
        assets,
        project_root=project_root,
        model="m",
        version_id=1,
        batch_size=2,
    ):
        yielded.append(batch)

    # 5 assets / batch_size=2 → 3 yields (sizes 2, 2, 1).
    assert [len(b) for b in yielded] == [2, 2, 1]
    # Provider was called once per non-empty batch — proves the yield
    # alternates with provider calls (vs accumulating then yielding once).
    assert fake.embed_calls == 3


# ---- shared single-input helper ------------------------------------------


def test_multimodal_input_text_only() -> None:
    """Sanity: MultimodalInput accepts text-only construction (chunks)."""
    inp = MultimodalInput(text="hello")
    assert inp.text == "hello"
    assert inp.images == []


def test_multimodal_input_image_only() -> None:
    inp = MultimodalInput(
        images=[ImageContent(bytes=b"raw", mime="image/png")]
    )
    assert inp.text is None
    assert len(inp.images) == 1
