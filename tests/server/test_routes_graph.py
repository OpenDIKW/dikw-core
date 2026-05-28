"""HTTP-level tests for ``GET /v1/base/graph``.

The base graph endpoint (issue #89) lets ``dikw-web`` and agent clients
fetch the full link graph in one request instead of looping
``GET /v1/base/pages/{path}`` and re-parsing wikilinks in the browser.

These tests cover the wire shape, ``active`` query semantics, and the
"this is read-only, no graph-specific knobs leak through unknown query
params" stance.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from dikw_core.schemas import Layer

from ..fakes import seed_doc as _seed


async def _seed_minigraph(root: Path) -> None:
    await _seed(
        root, layer=Layer.KNOWLEDGE, path="knowledge/A.md", title="A",
        body="See [[B]] and [[Missing]].\n",
    )
    await _seed(
        root, layer=Layer.KNOWLEDGE, path="knowledge/B.md", title="B", body="# B\n",
    )


@pytest.mark.asyncio
async def test_get_graph_default_returns_200(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    await _seed_minigraph(base_root)
    resp = await server_client.get("/v1/base/graph")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert {n["path"] for n in body["nodes"]} == {"knowledge/A.md", "knowledge/B.md"}
    edges = body["edges"]
    assert len(edges) == 1
    assert edges[0]["source"] == "knowledge/A.md"
    assert edges[0]["target"] == "knowledge/B.md"
    assert edges[0]["type"] == "wikilink"
    assert body["stats"]["unresolved_count"] == 1


@pytest.mark.asyncio
async def test_response_shape_matches_GraphResult(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    """Lock the wire-level shape so client codegen can rely on it."""
    await _seed_minigraph(base_root)
    resp = await server_client.get("/v1/base/graph")
    assert resp.status_code == 200
    body = resp.json()
    # Top-level keys.
    assert {"base_revision", "generated_at", "nodes", "edges", "unresolved", "stats"} <= body.keys()
    # Node shape.
    n = body["nodes"][0]
    assert {"id", "path", "title", "layer", "active", "mtime", "inbound", "outbound"} <= n.keys()
    # Edge shape.
    e = body["edges"][0]
    assert {"id", "source", "target", "type", "target_text", "anchor", "weight"} <= e.keys()
    # Unresolved shape.
    u = body["unresolved"][0]
    assert {"source", "target_text", "anchor", "count"} <= u.keys()
    # Stats shape.
    assert {"node_count", "edge_count", "unresolved_count"} <= body["stats"].keys()


@pytest.mark.asyncio
async def test_active_false_returns_only_deactivated(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    """Mirrors ``GET /v1/base/pages?active=false`` semantics: returns
    the deactivated subset only. Through-the-wire ``active=None`` is
    not reachable by design (FastAPI ``bool | None`` rejects empty
    strings as 422); engine-side ``active=None`` is covered in
    ``tests/test_api_graph.py``."""
    await _seed(
        base_root, layer=Layer.KNOWLEDGE, path="knowledge/A.md", title="A",
        body="# A\n", active=True,
    )
    await _seed(
        base_root, layer=Layer.KNOWLEDGE, path="knowledge/B.md", title="B",
        body="# B\n", active=False,
    )

    resp = await server_client.get("/v1/base/graph", params={"active": "false"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert {n["path"] for n in body["nodes"]} == {"knowledge/B.md"}


@pytest.mark.asyncio
async def test_active_invalid_returns_422(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    resp = await server_client.get(
        "/v1/base/graph", params={"active": "maybe"}
    )
    # FastAPI bool coercion rejects non-bool-ish strings → 422.
    assert resp.status_code in (400, 422)


@pytest.mark.asyncio
async def test_no_layer_query_param(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    """Issue #89 v1 deliberately drops the ``layer`` knob from the
    proposal; the route returns the full base regardless. A ``?layer=...``
    in the URL must NOT silently filter (or 422 — graceful pass-through
    is the chosen behaviour because forward-compat: a future v2 may
    re-introduce ``layer`` as additive)."""
    await _seed(
        base_root, layer=Layer.KNOWLEDGE, path="knowledge/W.md", title="W", body="# W\n",
    )
    await _seed(
        base_root, layer=Layer.SOURCE, path="sources/S.md", title="S", body="# S\n",
    )

    plain = await server_client.get("/v1/base/graph")
    with_layer = await server_client.get(
        "/v1/base/graph", params={"layer": "wiki"}
    )
    assert plain.status_code == 200 and with_layer.status_code == 200
    plain_paths = {n["path"] for n in plain.json()["nodes"]}
    layered_paths = {n["path"] for n in with_layer.json()["nodes"]}
    # ``layer=wiki`` did NOT filter — both layers still in the node set.
    assert plain_paths == {"knowledge/W.md", "sources/S.md"}
    assert layered_paths == plain_paths


@pytest.mark.asyncio
async def test_base_revision_stable_across_calls(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    """``base_revision`` is content-addressed, so identical state must
    yield identical revision strings — the property a client relies on
    when caching graph responses keyed by revision."""
    await _seed_minigraph(base_root)
    r1 = (await server_client.get("/v1/base/graph")).json()
    r2 = (await server_client.get("/v1/base/graph")).json()
    assert r1["base_revision"] == r2["base_revision"]
