"""HTTP-level tests for ``GET /v1/assets/{asset_id}``.

Tests lock the wire contract: bytes round-trip with the right mime,
content-addressed cache headers, and uniform 404 on every failure path
(unknown id, malformed id, file vanished, ``stored_path`` escape).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from ..fakes import png_with_dims, seed_asset


@pytest.mark.asyncio
async def test_get_asset_returns_bytes_and_mime(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    payload = png_with_dims(1, 1)
    asset_id = hashlib.sha256(payload).hexdigest()
    rel = f"assets/{asset_id[:2]}/{asset_id[:8]}-x.png"
    await seed_asset(
        base_root, asset_id=asset_id, stored_path=rel, payload=payload
    )

    resp = await server_client.get(f"/v1/assets/{asset_id}")
    assert resp.status_code == 200, resp.text
    assert resp.content == payload
    assert resp.headers["content-type"].startswith("image/png")
    assert resp.headers.get("etag") == f'"{asset_id}"'
    cache = resp.headers.get("cache-control", "")
    assert "immutable" in cache
    assert "max-age=" in cache


@pytest.mark.asyncio
async def test_get_asset_unknown_id_404(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    _ = base_root
    resp = await server_client.get(f"/v1/assets/{'0' * 64}")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "asset_not_found"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_id",
    [
        "short",
        "ZZ" * 32,  # right length, non-hex
        "../etc/passwd",
        "a" * 63,
        "a" * 65,
    ],
)
async def test_get_asset_malformed_id_404(
    server_client: httpx.AsyncClient, base_root: Path, bad_id: str
) -> None:
    """Malformed ids must NOT leak existence info — uniform 404."""
    _ = base_root
    resp = await server_client.get(f"/v1/assets/{bad_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_asset_stored_path_escape_404(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    """A row whose ``stored_path`` points outside the assets dir must
    not serve bytes, even when the file exists on disk."""
    payload = b"top secret bytes"
    outside_rel = "secrets/leak.bin"
    (base_root / outside_rel).parent.mkdir(parents=True, exist_ok=True)
    (base_root / outside_rel).write_bytes(payload)
    asset_id = "f" * 64
    await seed_asset(
        base_root,
        asset_id=asset_id,
        stored_path=outside_rel,
        payload=payload,
        drop_file=False,
    )

    resp = await server_client.get(f"/v1/assets/{asset_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_asset_missing_file_404(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    """Row registered, file deleted — still 404, no 500 with disk path."""
    payload = png_with_dims(1, 1)
    asset_id = hashlib.sha256(payload).hexdigest()
    rel = f"assets/{asset_id[:2]}/{asset_id[:8]}-x.png"
    await seed_asset(
        base_root,
        asset_id=asset_id,
        stored_path=rel,
        payload=payload,
        drop_file=False,
    )
    resp = await server_client.get(f"/v1/assets/{asset_id}")
    assert resp.status_code == 404
