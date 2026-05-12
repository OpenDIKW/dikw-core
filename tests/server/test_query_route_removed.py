"""Guard tests against accidental ``/v1/query`` reintroduction.

PR-1 removed the in-engine query verb; these two tests ensure the
endpoint stays gone — checking both runtime (404, not 405) and the
OpenAPI schema (catches reintroduction via any HTTP method).
"""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.asyncio
async def test_query_route_returns_404(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post("/v1/query", json={"q": "ping", "limit": 1})
    assert resp.status_code == 404, (
        f"/v1/query should be removed entirely, got {resp.status_code}"
    )


@pytest.mark.asyncio
async def test_query_route_absent_from_openapi(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    paths = list(schema.get("paths", {}).keys())
    query_paths = [p for p in paths if "/v1/query" in p]
    assert query_paths == [], (
        f"/v1/query family must be absent from OpenAPI, found: {query_paths}"
    )
