"""HTTP-level tests for ``/v1/base/pages*``.

Companion endpoint to ``/v1/retrieve``: once an agent has a chunk hit,
it calls ``GET /v1/base/pages/{path}`` to read the full page + chunk
anchors. Tests lock the wire shape, the path-safety boundary, and the
``layer`` filter on the list route.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from dikw_core import api as api_module

from ..fakes import FakeEmbeddings, png_with_dims


@pytest.mark.asyncio
async def test_list_pages_default_active(
    server_client: httpx.AsyncClient, ingested_wiki: Path
) -> None:
    resp = await server_client.get("/v1/base/pages")
    assert resp.status_code == 200, resp.text
    rows = resp.json()
    assert isinstance(rows, list) and rows, "ingested wiki should have docs"
    for row in rows:
        assert {"doc_id", "path", "layer", "active"} <= set(row.keys())
        assert row["active"] is True


@pytest.mark.asyncio
async def test_list_pages_layer_filter(
    server_client: httpx.AsyncClient, ingested_wiki: Path
) -> None:
    resp = await server_client.get(
        "/v1/base/pages", params={"layer": "source"}
    )
    assert resp.status_code == 200, resp.text
    rows = resp.json()
    assert rows, "fixture corpus is sources/-only — should have hits"
    assert all(r["layer"] == "source" for r in rows)


@pytest.mark.asyncio
async def test_list_pages_rejects_bad_layer(
    server_client: httpx.AsyncClient, ingested_wiki: Path
) -> None:
    resp = await server_client.get(
        "/v1/base/pages", params={"layer": "bogus"}
    )
    # FastAPI surfaces enum mismatch as 422 (validation error) by default.
    assert resp.status_code in (400, 422)


@pytest.mark.asyncio
async def test_get_page_returns_body_and_anchors(
    server_client: httpx.AsyncClient, ingested_wiki: Path
) -> None:
    # Pull a known doc path off the list endpoint so we don't hardcode
    # against fixture renames.
    listed = (await server_client.get("/v1/base/pages")).json()
    target = next(r for r in listed if r["layer"] == "source")
    path = target["path"]

    resp = await server_client.get(f"/v1/base/pages/{path}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["doc_id"] == target["doc_id"]
    assert body["path"] == path
    assert body["layer"] == "source"
    assert isinstance(body["body"], str) and body["body"]
    assert isinstance(body["anchors"], list) and body["anchors"]
    anchor = body["anchors"][0]
    assert {"chunk_id", "seq", "start", "end"} <= set(anchor.keys())
    assert 0 <= anchor["start"] <= anchor["end"] <= len(body["body"])


@pytest.mark.asyncio
async def test_get_page_unknown_path_404(
    server_client: httpx.AsyncClient, ingested_wiki: Path
) -> None:
    resp = await server_client.get(
        "/v1/base/pages/sources/does-not-exist.md"
    )
    assert resp.status_code == 404
    body = resp.json()
    # Error envelope is ``{"error": {"code": "...", ...}}`` — see
    # server/errors.py.
    assert body["error"]["code"] == "page_not_found"


@pytest.mark.asyncio
async def test_get_page_path_escape_404(
    server_client: httpx.AsyncClient, ingested_wiki: Path
) -> None:
    """``..`` traversal is not registered, so it gets the same uniform
    404 as any other unknown path — no special path-escape branch."""
    resp = await server_client.get("/v1/base/pages/../../../etc/passwd")
    # httpx normalises ``..`` segments client-side before sending, so
    # this lands on ``/etc/passwd`` from the server's POV — a 404 from
    # FastAPI's not-found handler (no route matches the path) rather
    # than a 404 from our handler. Either is fine; what we're locking
    # in is "no 200 leaks anything outside the index".
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_page_response_includes_assets(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    """End-to-end: a markdown page with an image ref surfaces the asset
    under ``response.assets[]`` with a directly-usable ``url`` — remote
    callers can fetch image bytes without any server-side rewriting of
    the page body."""
    src_dir = base_root / "sources" / "demo"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "diagram.png").write_bytes(png_with_dims(320, 240))
    rel = "sources/demo/page-with-image.md"
    (base_root / rel).write_text(
        "# With image\n\n"
        "Look at this diagram: ![diagram](./diagram.png)\n\n"
        "Body fodder so the chunker has material to work with.\n",
        encoding="utf-8",
    )
    await api_module.ingest(base_root, embedder=FakeEmbeddings())

    resp = await server_client.get(f"/v1/base/pages/{rel}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assets = body["assets"]
    assert len(assets) == 1, f"expected one image asset, got {assets!r}"
    a = assets[0]
    # The route the client must hit to actually fetch the bytes.
    assert a["url"] == f"/v1/assets/{a['asset_id']}"
    assert a["mime"] == "image/png"
    assert a["bytes"] > 0
    # And that URL must actually work on the same server.
    bytes_resp = await server_client.get(a["url"])
    assert bytes_resp.status_code == 200
    assert bytes_resp.content == (src_dir / "diagram.png").read_bytes()
