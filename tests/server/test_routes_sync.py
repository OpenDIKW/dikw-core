"""HTTP-level tests for the synchronous route surface.

Goes through the FastAPI app via ``httpx.ASGITransport`` so the auth
dependency, error envelope, and Pydantic response shaping are all
exercised without binding a real socket.
"""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.asyncio
async def test_info_endpoint(server_client: httpx.AsyncClient) -> None:
    resp = await server_client.get("/v1/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["wiki_root"]
    assert body["storage_backend"] in {"sqlite", "postgres"}
    assert body["auth_required"] is False  # localhost-no-token default


@pytest.mark.asyncio
async def test_healthz_and_readyz(server_client: httpx.AsyncClient) -> None:
    h = await server_client.get("/v1/healthz")
    assert h.status_code == 200 and h.json()["status"] == "ok"
    r = await server_client.get("/v1/readyz")
    assert r.status_code == 200 and r.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_status_returns_storage_counts(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.get("/v1/status")
    assert resp.status_code == 200
    body = resp.json()
    # Fresh wiki: docs/chunks/etc all zero.
    assert body["chunks"] == 0
    assert body["embeddings"] == 0
    assert body["links"] == 0


@pytest.mark.asyncio
async def test_check_rejects_both_only_flags(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post(
        "/v1/check", json={"llm_only": True, "embed_only": True}
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "bad_request"


@pytest.mark.asyncio
async def test_lint_runs_against_empty_wiki(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post("/v1/lint")
    assert resp.status_code == 200
    body = resp.json()
    # LintReport's empty shape — exact keys depend on engine but the
    # endpoint should at least return a JSON object.
    assert isinstance(body, dict)


@pytest.mark.asyncio
async def test_unknown_chunk_returns_404(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.get("/v1/doc/chunks/99999")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_wisdom_listing_starts_empty(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.get("/v1/wisdom")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_wisdom_listing_rejects_unknown_kind_with_4xx(
    server_client: httpx.AsyncClient,
) -> None:
    """A typo'd ``?kind=`` query param must be a 400, not a 500 — the
    raw ``WisdomKind(kind)`` cast would otherwise raise ``ValueError``
    and bubble through as a 500."""
    resp = await server_client.get("/v1/wisdom?kind=bogus")
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "invalid_wisdom_kind"


@pytest.mark.asyncio
async def test_approve_unknown_wisdom_is_404_or_409(
    server_client: httpx.AsyncClient,
) -> None:
    # ReviewError → 409 Conflict; if storage raises a different error
    # (e.g. NotFound) the engine surfaces it as 404. Either is acceptable
    # — we just want to not see a 500.
    resp = await server_client.post("/v1/wisdom/no-such-id/approve")
    assert resp.status_code in (404, 409)


@pytest.mark.asyncio
async def test_doc_search_against_empty_wiki(
    server_client: httpx.AsyncClient,
) -> None:
    # ``mode=bm25`` keeps the request path off the dense embedder, which
    # the test wiki's fake provider config can't reach (no real API key).
    resp = await server_client.post(
        "/v1/doc/search",
        json={"q": "anything", "limit": 3, "mode": "bm25"},
    )
    assert resp.status_code == 200
    assert resp.json() == []


# ---- task routes (echo end-to-end) -------------------------------------


@pytest.mark.asyncio
async def test_echo_submit_returns_handle(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post("/v1/echo", json={"count": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["op"] == "echo"
    assert body["task_id"]
    for key in ("self", "events", "result", "cancel"):
        assert key in body["links"]


@pytest.mark.asyncio
async def test_echo_lifecycle_to_terminal_result(
    server_client: httpx.AsyncClient,
) -> None:
    submit = await server_client.post("/v1/echo", json={"count": 3})
    task_id = submit.json()["task_id"]

    # Poll for terminal — echo task is fast (sub-second on a free CPU)
    # but ASGI is single-threaded so we drive the loop with sleeps.
    import asyncio

    for _ in range(50):
        result = await server_client.get(f"/v1/tasks/{task_id}/result")
        if result.status_code == 200:
            break
        await asyncio.sleep(0.05)
    assert result.status_code == 200
    body = result.json()
    assert body["status"] == "succeeded"
    assert body["result"] == {"echoed": 3}


@pytest.mark.asyncio
async def test_task_result_404_for_unknown(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.get("/v1/tasks/no-such-id/result")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cancel_unknown_task_404(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post("/v1/tasks/no-such-id/cancel")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_event_stream_replays_full_tape(
    server_client: httpx.AsyncClient,
) -> None:
    submit = await server_client.post("/v1/echo", json={"count": 2})
    task_id = submit.json()["task_id"]

    # Wait for terminal so the bus is closed and ndjson_lines streams
    # purely from the store (no live tail).
    import asyncio

    for _ in range(50):
        await asyncio.sleep(0.02)
        row = await server_client.get(f"/v1/tasks/{task_id}")
        if row.json()["status"] in {"succeeded", "failed", "cancelled"}:
            break

    async with server_client.stream(
        "GET", f"/v1/tasks/{task_id}/events"
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        lines = []
        async for line in resp.aiter_lines():
            line = line.strip()
            if line:
                lines.append(line)

    import json

    events = [json.loads(line) for line in lines]
    types = [e["type"] for e in events]
    assert types[0] == "task_started"
    assert types[-1] == "final"
    assert any(e["type"] == "progress" for e in events)
    assert any(e["type"] == "partial" for e in events)
