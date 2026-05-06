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
