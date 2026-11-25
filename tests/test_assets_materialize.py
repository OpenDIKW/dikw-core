"""Tests for asset materialization.

Materialize_asset is the bridge between MarkdownBackend's symbolic
references and the engine-managed binary on disk. Contract:

  * Local files: read → sha256 → copy to assets/<h2>/<h8>-<name>.<ext>
    → probe dims → upsert AssetRecord. Idempotent by hash.
  * Remote URLs: skip (return None), no side effect.
  * Missing/unreadable files: skip (return None), no side effect.
  * Same hash, different referencing names: one row, original_paths
    accumulates each distinct reference.
"""

from __future__ import annotations

import os
import stat
import struct
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from dikw_core.domains.data.assets import (
    _MIME_SNIFF_BYTES,
    _read_bounded_header,
    materialize_asset,
)
from dikw_core.schemas import AssetKind, AssetRecord, AssetRef

# ---- Helpers -------------------------------------------------------------


def _make_fake_storage() -> tuple[
    dict[str, AssetRecord],
    Callable[[str], Awaitable[AssetRecord | None]],
    Callable[[AssetRecord], Awaitable[None]],
]:
    """Tiny in-memory get/upsert pair for testing the materialize pipeline
    without requiring the full Storage Protocol or a SQLite adapter."""
    store: dict[str, AssetRecord] = {}

    async def get(asset_id: str) -> AssetRecord | None:
        return store.get(asset_id)

    async def upsert(asset: AssetRecord) -> None:
        store[asset.asset_id] = asset

    return store, get, upsert


def _png_with_dims(width: int, height: int) -> bytes:
    """Synthetic minimum-viable PNG: 8-byte signature + IHDR chunk header.
    Enough for the probe to extract width/height; not a renderable image."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_len = struct.pack(">I", 13)
    ihdr_type = b"IHDR"
    ihdr_body = struct.pack(">II", width, height) + bytes([8, 6, 0, 0, 0])
    ihdr_crc = b"\x00\x00\x00\x00"  # probe doesn't validate CRC
    return sig + ihdr_len + ihdr_type + ihdr_body + ihdr_crc


def _jpeg_with_dims(width: int, height: int) -> bytes:
    """Synthetic minimum-viable JPEG: SOI + JFIF APP0 + SOF0 with declared dims."""
    soi = b"\xff\xd8"
    # JFIF APP0: marker FFE0, length 16, "JFIF\0", v1.01, units 0, density 1x1, no thumb
    app0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    # SOF0: marker FFC0, length 17, precision 8, height, width, 3 components
    sof0 = (
        b"\xff\xc0\x00\x11\x08"
        + struct.pack(">HH", height, width)
        + b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    )
    return soi + app0 + sof0


def _gif_with_dims(width: int, height: int) -> bytes:
    """Synthetic minimum-viable GIF89a header: 6 sig + 2 width LE + 2 height LE."""
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00\x00\x00"


def _svg_bytes() -> bytes:
    return b'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" />'


# ---- Tests ---------------------------------------------------------------


async def test_materialize_local_png_copies_and_records(tmp_path: Path) -> None:
    """Happy path: a local PNG referenced from a markdown file is copied
    into the engine vault and its metadata recorded."""
    project_root = tmp_path / "project"
    notes_dir = project_root / "notes"
    notes_dir.mkdir(parents=True)
    md = notes_dir / "doc.md"
    md.write_text("![arch](./diagrams/arch.png)", encoding="utf-8")
    (notes_dir / "diagrams").mkdir()
    png_bytes = _png_with_dims(640, 480)
    (notes_dir / "diagrams" / "arch.png").write_bytes(png_bytes)

    ref = AssetRef(
        original_path="./diagrams/arch.png",
        alt="arch",
        start=0,
        end=29,
        syntax="markdown",
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    rec = None if rec is None else rec[0]
    assert rec is not None
    assert rec.kind == AssetKind.IMAGE
    assert rec.mime == "image/png"
    assert rec.bytes == len(png_bytes)
    assert rec.media_meta is not None
    assert rec.media_meta.width == 640
    assert rec.media_meta.height == 480
    assert rec.original_paths == ["./diagrams/arch.png"]
    # File materialized at the engine path.
    on_disk = project_root / rec.stored_path
    assert on_disk.is_file()
    assert on_disk.read_bytes() == png_bytes
    # Asset row landed in fake storage.
    assert rec.asset_id in store


async def test_materialize_skips_remote_url(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")

    ref = AssetRef(
        original_path="https://example.com/img.png",
        alt="r",
        start=0,
        end=20,
        syntax="markdown",
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is None
    assert store == {}


async def test_materialize_skips_missing_file(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")

    ref = AssetRef(
        original_path="./not-here.png",
        alt="m",
        start=0,
        end=20,
        syntax="markdown",
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is None
    assert store == {}


async def test_materialize_dedup_by_hash_appends_original_paths(
    tmp_path: Path,
) -> None:
    """Same file content referenced from two paths → single AssetRecord,
    original_paths grows; binary copied only once."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    notes = project_root / "notes"
    notes.mkdir()

    png_bytes = _png_with_dims(10, 20)
    a_path = notes / "a.png"
    b_path = notes / "b.png"
    a_path.write_bytes(png_bytes)
    b_path.write_bytes(png_bytes)

    md = notes / "doc.md"
    md.write_text("placeholder", encoding="utf-8")

    store, get, upsert = _make_fake_storage()

    rec1 = await materialize_asset(
        AssetRef(
            original_path="a.png", alt="", start=0, end=10, syntax="markdown"
        ),
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    rec1 = None if rec1 is None else rec1[0]
    assert rec1 is not None
    assert rec1.original_paths == ["a.png"]

    result2 = await materialize_asset(
        AssetRef(
            original_path="b.png", alt="", start=0, end=10, syntax="markdown"
        ),
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert result2 is not None
    rec2, was_new_2 = result2
    # Second materialize hit the existing row — was_new flag tells the
    # caller (api.ingest) it doesn't need to re-embed this asset.
    assert was_new_2 is False
    assert rec2.asset_id == rec1.asset_id  # same content → same id
    assert rec2.original_paths == ["a.png", "b.png"]
    # Only one row.
    assert len(store) == 1


async def test_materialize_cache_hit_does_not_slurp_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: a cache hit must materialize without loading the full
    file into memory. Verified by poisoning ``Path.read_bytes`` after the
    first call so any slurp on the cache-hit path fails loudly.
    """
    project_root = tmp_path / "project"
    project_root.mkdir()
    notes = project_root / "notes"
    notes.mkdir()

    png_bytes = _png_with_dims(4, 4)
    img = notes / "x.png"
    img.write_bytes(png_bytes)
    md = notes / "doc.md"
    md.write_text("placeholder", encoding="utf-8")

    _store, get, upsert = _make_fake_storage()
    ref = AssetRef(original_path="x.png", alt="", start=0, end=10, syntax="markdown")

    # First call: legitimate cache miss — slurp is permitted.
    rec1 = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec1 is not None

    # Now poison read_bytes. If the cache-hit branch ever calls it the
    # test fails with the explicit error below.
    def _explode(_self: Path) -> bytes:
        raise AssertionError("read_bytes() must not be called on cache hit")

    monkeypatch.setattr(Path, "read_bytes", _explode)

    rec2 = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec2 is not None


async def test_materialize_revalidates_cache_hit_against_current_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the file mutates between the initial streaming hash and the
    cache lookup, the materializer must NOT silently attach the new path
    to a stale record. It revalidates by streaming-hashing again before
    trusting the cache hit; on disagreement, it falls through to the
    slurp + canonical path so the persisted hash matches the bytes.
    """
    import hashlib

    project_root = tmp_path / "project"
    project_root.mkdir()
    notes = project_root / "notes"
    notes.mkdir()

    png_bytes = _png_with_dims(5, 5)
    img = notes / "x.png"
    img.write_bytes(png_bytes)
    md = notes / "doc.md"
    md.write_text("placeholder", encoding="utf-8")

    canonical = hashlib.sha256(png_bytes).hexdigest()
    stale_sha = "1" * 64
    # Pre-seed the store with a record at stale_sha so the first
    # `get_asset(stale_sha)` finds it; revalidation hash returns the
    # real canonical sha, exposing the disagreement.
    store, get, upsert = _make_fake_storage()
    store[stale_sha] = AssetRecord(
        asset_id=stale_sha,
        kind=AssetKind.IMAGE,
        mime="image/png",
        stored_path="assets/00/00000000-old.png",
        original_paths=["a.png"],
        bytes=10,
        media_meta=None,
        created_ts=0.0,
    )

    hashes = iter([stale_sha, canonical])
    monkeypatch.setattr(
        "dikw_core.domains.data.assets.hash_file",
        lambda _p: next(hashes),
    )

    rec = await materialize_asset(
        AssetRef(original_path="x.png", alt="", start=0, end=10, syntax="markdown"),
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is not None
    record, was_new = rec
    # The path must be attached to the canonical record, not the stale one.
    assert record.asset_id == canonical
    assert "x.png" not in store[stale_sha].original_paths
    assert was_new is True


async def test_materialize_uses_canonical_hash_when_streaming_hash_disagrees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the streaming hash disagrees with the slurped bytes' hash
    (race: file mutated between the two reads), the persisted record
    must use the canonical hash of the slurped bytes — otherwise we'd
    save bytes under a stale SHA and dedup would silently break for
    the *real* content forever after.
    """
    import hashlib

    project_root = tmp_path / "project"
    project_root.mkdir()
    notes = project_root / "notes"
    notes.mkdir()

    png_bytes = _png_with_dims(3, 3)
    img = notes / "x.png"
    img.write_bytes(png_bytes)
    md = notes / "doc.md"
    md.write_text("placeholder", encoding="utf-8")

    stale_sha = "0" * 64
    monkeypatch.setattr("dikw_core.domains.data.assets.hash_file", lambda _p: stale_sha)

    _store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        AssetRef(original_path="x.png", alt="", start=0, end=10, syntax="markdown"),
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is not None
    record, was_new = rec
    canonical = hashlib.sha256(png_bytes).hexdigest()
    assert record.asset_id == canonical
    assert record.asset_id != stale_sha
    assert was_new is True


async def test_materialize_dedup_idempotent_on_same_path(
    tmp_path: Path,
) -> None:
    """Re-materializing the exact same reference must not duplicate the
    path string in original_paths."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    img = project_root / "img.png"
    img.write_bytes(_png_with_dims(1, 1))

    ref = AssetRef(
        original_path="img.png", alt="", start=0, end=10, syntax="markdown"
    )
    _store, get, upsert = _make_fake_storage()

    r1 = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    r2 = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert r1 is not None and r2 is not None
    _rec1, was_new_1 = r1
    rec2, was_new_2 = r2
    assert was_new_1 is True
    assert was_new_2 is False
    assert rec2.original_paths == ["img.png"]


async def test_materialize_jpeg_dimensions(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    jpg_bytes = _jpeg_with_dims(800, 600)
    (project_root / "photo.jpg").write_bytes(jpg_bytes)

    ref = AssetRef(
        original_path="photo.jpg", alt="", start=0, end=10, syntax="markdown"
    )
    _store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    rec = None if rec is None else rec[0]
    assert rec is not None
    assert rec.mime == "image/jpeg"
    assert rec.media_meta is not None
    assert rec.media_meta.width == 800
    assert rec.media_meta.height == 600


async def test_materialize_gif_dimensions(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    gif_bytes = _gif_with_dims(50, 40)
    (project_root / "loop.gif").write_bytes(gif_bytes)

    ref = AssetRef(
        original_path="loop.gif", alt="", start=0, end=10, syntax="markdown"
    )
    _store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    rec = None if rec is None else rec[0]
    assert rec is not None
    assert rec.mime == "image/gif"
    assert rec.media_meta is not None
    assert rec.media_meta.width == 50
    assert rec.media_meta.height == 40


async def test_materialize_svg_records_no_dims(tmp_path: Path) -> None:
    """SVG is not vectorized in v1 (it's text-based); the materialize step
    still copies it and records mime, but width/height stay None."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    (project_root / "icon.svg").write_bytes(_svg_bytes())

    ref = AssetRef(
        original_path="icon.svg", alt="", start=0, end=10, syntax="markdown"
    )
    _store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    rec = None if rec is None else rec[0]
    assert rec is not None
    assert rec.mime == "image/svg+xml"
    assert rec.media_meta is None


async def test_materialize_unsupported_format_skipped(tmp_path: Path) -> None:
    """Random non-image bytes with a non-image extension → skip."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    (project_root / "data.xyz").write_bytes(b"random binary content here")

    ref = AssetRef(
        original_path="data.xyz", alt="", start=0, end=10, syntax="markdown"
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is None
    assert store == {}


async def test_materialize_uses_engine_vault_path_layout(tmp_path: Path) -> None:
    """The stored_path column must match assets/<h2>/<h8>-<sanitized>.<ext>."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    png = _png_with_dims(1, 1)
    (project_root / "diagram with spaces.png").write_bytes(png)

    ref = AssetRef(
        original_path="diagram with spaces.png",
        alt="",
        start=0,
        end=10,
        syntax="markdown",
    )
    _store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    rec = None if rec is None else rec[0]
    assert rec is not None
    parts = rec.stored_path.split("/")
    assert parts[0] == "assets"
    assert len(parts[1]) == 2  # h2
    assert parts[2].startswith(rec.asset_id[:8] + "-")
    assert parts[2].endswith(".png")
    # Sanitized: spaces collapsed to '-'.
    assert "diagram-with-spaces" in parts[2]


async def test_materialize_rejects_absolute_path_outside_root(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    (project_root / "sources").mkdir(parents=True)
    md = project_root / "sources" / "doc.md"
    md.write_text("x", encoding="utf-8")

    outside = tmp_path / "outside.png"
    outside.write_bytes(_png_with_dims(10, 10))

    ref = AssetRef(
        original_path=str(outside),
        alt="abs",
        start=0,
        end=20,
        syntax="markdown",
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is None
    assert store == {}
    assert not (project_root / "assets").exists()


async def test_materialize_rejects_parent_traversal_outside_root(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    (project_root / "sources").mkdir(parents=True)
    md = project_root / "sources" / "doc.md"
    md.write_text("x", encoding="utf-8")

    outside = tmp_path / "outside.png"
    outside.write_bytes(_png_with_dims(10, 10))

    ref = AssetRef(
        original_path="../../outside.png",
        alt="esc",
        start=0,
        end=20,
        syntax="markdown",
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is None
    assert store == {}
    assert not (project_root / "assets").exists()


async def test_materialize_rejects_dikw_internal_file(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "sources").mkdir(parents=True)
    (project_root / ".dikw").mkdir(parents=True)
    # Even if an internal file started with PNG magic bytes, .dikw/ must be off-limits.
    (project_root / ".dikw" / "auth.json").write_bytes(_png_with_dims(4, 4))
    md = project_root / "sources" / "doc.md"
    md.write_text("x", encoding="utf-8")

    ref = AssetRef(
        original_path="../.dikw/auth.json",
        alt="meta",
        start=0,
        end=20,
        syntax="markdown",
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is None
    assert store == {}


@pytest.mark.parametrize("mode", [stat.S_IFCHR | 0o666, stat.S_IFBLK | 0o660])
async def test_materialize_rejects_device_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: int,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    target = project_root / "dev.png"
    target.write_bytes(_png_with_dims(4, 4))
    target_resolved = target.resolve()

    real_stat = Path.stat

    def _fake_stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        st = real_stat(self, follow_symlinks=follow_symlinks)
        if os.path.realpath(self) == str(target_resolved):
            return os.stat_result((mode, *st[1:]))
        return st

    def _forbid_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if os.path.realpath(self) == str(target_resolved):
            raise AssertionError("device node must never be opened")
        return real_open(self, *args, **kwargs)

    real_open = Path.open
    monkeypatch.setattr(Path, "stat", _fake_stat)
    monkeypatch.setattr(Path, "open", _forbid_open)

    ref = AssetRef(
        original_path="dev.png", alt="", start=0, end=10, syntax="markdown"
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is None
    assert store == {}


async def test_materialize_rejects_fifo(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):  # pragma: no cover - Windows
        pytest.skip("mkfifo unavailable on this platform")
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    os.mkfifo(project_root / "pipe.png")

    ref = AssetRef(
        original_path="pipe.png", alt="", start=0, end=10, syntax="markdown"
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is None
    assert store == {}


async def test_materialize_mime_uses_bounded_header_without_slurp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-image files must be rejected from a bounded header read before
    streaming-hashing or calling ``read_bytes()``."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    target = project_root / "large-non-image.png"
    target.write_bytes(b"NOT_AN_IMAGE_" * 5000)
    target_resolved = target.resolve()

    read_sizes: list[int] = []
    real_open = Path.open

    class _RecordingReader:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def fileno(self) -> int:
            return int(self._inner.fileno())

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return bytes(self._inner.read(size))

        def __enter__(self) -> _RecordingReader:
            return self

        def __exit__(self, *args: object) -> None:
            self._inner.close()

    def _spy_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        fh = real_open(self, *args, **kwargs)
        if os.path.realpath(self) == str(target_resolved):
            return _RecordingReader(fh)
        return fh

    def _explode_read_bytes(_self: Path) -> bytes:
        raise AssertionError("read_bytes() must not be called for non-image")

    def _explode_hash_file(_p: Path) -> str:
        raise AssertionError("hash_file() must not be called for non-image")

    monkeypatch.setattr(Path, "open", _spy_open)
    monkeypatch.setattr(Path, "read_bytes", _explode_read_bytes)
    monkeypatch.setattr("dikw_core.domains.data.assets.hash_file", _explode_hash_file)

    ref = AssetRef(
        original_path="large-non-image.png",
        alt="",
        start=0,
        end=20,
        syntax="markdown",
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is None
    assert store == {}
    assert read_sizes == [_MIME_SNIFF_BYTES]


@pytest.mark.parametrize(
    "filename",
    ["fake.png", "fake.jpg", "fake.jpeg", "fake.gif", "fake.webp", "fake.svg"],
)
async def test_materialize_drops_extension_only_mime_fallback(
    tmp_path: Path,
    filename: str,
) -> None:
    """A non-image payload must NOT be materialized just because its filename
    ends in an image extension."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    (project_root / filename).write_bytes(b'{"fixture": "plain-text-not-image"}\n')

    ref = AssetRef(
        original_path=filename, alt="", start=0, end=20, syntax="markdown"
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is None
    assert store == {}
    assert not (project_root / "assets").exists()


def test_read_bounded_header_rejects_non_regular_and_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_read_bounded_header`` must return ``None`` for non-regular paths
    (at both ``stat`` and ``fstat``) and on ``OSError``."""
    assert _read_bounded_header(tmp_path) is None
    assert _read_bounded_header(tmp_path / "does-not-exist.png") is None

    target = tmp_path / "swapped.png"
    target.write_bytes(_png_with_dims(4, 4))
    real_fstat = os.fstat

    def _fake_fstat(fd: int) -> os.stat_result:
        st = real_fstat(fd)
        return os.stat_result((stat.S_IFIFO | 0o600, *st[1:]))

    monkeypatch.setattr(os, "fstat", _fake_fstat)
    assert _read_bounded_header(target) is None


async def test_materialize_skips_when_header_read_fails_after_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the file becomes unreadable after ``_resolve_local`` succeeds,
    ``materialize_asset`` logs and skips without hashing or slurping."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    target = project_root / "unreadable.png"
    target.write_bytes(_png_with_dims(4, 4))
    target_resolved = target.resolve()

    real_open = Path.open

    def _deny_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if os.path.realpath(self) == str(target_resolved):
            raise PermissionError("simulated unreadable file")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _deny_open)

    ref = AssetRef(
        original_path="unreadable.png",
        alt="",
        start=0,
        end=20,
        syntax="markdown",
    )
    store, get, upsert = _make_fake_storage()
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=get,
        upsert_asset=upsert,
    )
    assert rec is None
    assert store == {}


async def test_materialize_skips_when_file_replaced_with_non_regular_before_slurp(
    tmp_path: Path,
) -> None:
    """If the asset is swapped for a non-regular entry between hashing and
    the final slurp, ``materialize_asset`` rejects it before ``read_bytes``."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    md = project_root / "doc.md"
    md.write_text("x", encoding="utf-8")
    target = project_root / "swapped.png"
    target.write_bytes(_png_with_dims(4, 4))

    store, _get, upsert = _make_fake_storage()

    async def _get_and_swap(_asset_id: str) -> AssetRecord | None:
        if target.is_file():
            target.unlink()
            target.mkdir()
        return None

    ref = AssetRef(
        original_path="swapped.png",
        alt="",
        start=0,
        end=20,
        syntax="markdown",
    )
    rec = await materialize_asset(
        ref,
        source_md_path=md,
        project_root=project_root,
        get_asset=_get_and_swap,
        upsert_asset=upsert,
    )
    assert rec is None
    assert store == {}
    assert not (project_root / "assets").exists()
