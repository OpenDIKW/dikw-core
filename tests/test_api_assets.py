"""Engine-side unit tests for ``api.read_asset``.

Contract guarded here: unknown id, missing file, and ``stored_path``
escape all surface as :class:`api.AssetNotFound` (uniform 404 on the
HTTP boundary). HTTP-layer tests live in ``tests/server/test_routes_assets.py``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dikw_core import api

from .fakes import init_test_base, png_with_dims, seed_asset


def _new_asset_id(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rel_for(asset_id: str) -> str:
    return f"assets/{asset_id[:2]}/{asset_id[:8]}-x.png"


@pytest.mark.asyncio
async def test_read_asset_returns_path_and_record(tmp_path: Path) -> None:
    init_test_base(tmp_path)
    payload = png_with_dims(1, 1)
    asset_id = _new_asset_id(payload)
    rel = _rel_for(asset_id)
    await seed_asset(
        tmp_path, asset_id=asset_id, stored_path=rel, payload=payload
    )

    path, got = await api.read_asset(tmp_path, asset_id)

    assert path == (tmp_path / rel).resolve()
    assert path.read_bytes() == payload
    assert got.asset_id == asset_id
    assert got.mime == "image/png"
    assert got.bytes == len(payload)


@pytest.mark.asyncio
async def test_read_asset_unknown_id_raises_not_found(tmp_path: Path) -> None:
    init_test_base(tmp_path)
    with pytest.raises(api.AssetNotFound):
        await api.read_asset(tmp_path, "deadbeef" * 8)


@pytest.mark.asyncio
async def test_read_asset_missing_file_raises_not_found(tmp_path: Path) -> None:
    """Row registered, file gone — refuse rather than 500 with the path."""
    init_test_base(tmp_path)
    payload = png_with_dims(1, 1)
    asset_id = _new_asset_id(payload)
    rel = _rel_for(asset_id)
    await seed_asset(
        tmp_path,
        asset_id=asset_id,
        stored_path=rel,
        payload=payload,
        drop_file=False,
    )

    with pytest.raises(api.AssetNotFound):
        await api.read_asset(tmp_path, asset_id)


@pytest.mark.asyncio
async def test_read_asset_rejects_stored_path_escape(tmp_path: Path) -> None:
    """Defence-in-depth: a row's ``stored_path`` pointing outside the
    configured assets dir must refuse to serve even when a real file
    sits at that path."""
    init_test_base(tmp_path)
    payload = b"top secret bytes"
    outside_rel = "secrets/outside.bin"
    (tmp_path / outside_rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / outside_rel).write_bytes(payload)
    asset_id = "a" * 64
    await seed_asset(
        tmp_path,
        asset_id=asset_id,
        stored_path=outside_rel,
        payload=payload,
        drop_file=False,
    )

    with pytest.raises(api.AssetNotFound):
        await api.read_asset(tmp_path, asset_id)
