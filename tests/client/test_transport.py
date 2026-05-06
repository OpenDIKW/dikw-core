"""Transport behaviour against the in-memory ASGI server.

These tests intentionally drive the real server (via the ``asgi_client``
fixture chain) instead of mocking httpx — we want the wire contract
between :class:`Transport` and ``dikw_core.server`` exercised together
so that schema drift on either side fails loudly.
"""

from __future__ import annotations

import httpx
import pytest

from dikw_core.client.transport import ClientError, Transport


@pytest.mark.asyncio
async def test_get_status_returns_storage_counts(
    client_transport: Transport,
) -> None:
    counts = await client_transport.get_json("/v1/status")
    # Fresh wiki — every counter is zero, the keys themselves are the
    # contract we care about.
    assert isinstance(counts, dict)
    assert "documents_by_layer" in counts
    assert "chunks" in counts


@pytest.mark.asyncio
async def test_404_raises_client_error_with_code(
    client_transport: Transport,
) -> None:
    with pytest.raises(ClientError) as excinfo:
        await client_transport.get_json("/v1/base/pages/does-not-exist.md")
    err = excinfo.value
    assert err.status == 404
    assert err.code == "page_not_found"


@pytest.mark.asyncio
async def test_400_carries_server_error_code(
    client_transport: Transport,
) -> None:
    """The server's BadRequest envelope (``code='bad_request'``) must
    propagate verbatim — clients branch on ``code``, not on message
    text."""
    with pytest.raises(ClientError) as excinfo:
        await client_transport.post_json(
            "/v1/check",
            json_body={"llm_only": True, "embed_only": True},
        )
    assert excinfo.value.status == 400
    assert excinfo.value.code == "bad_request"


@pytest.mark.asyncio
async def test_ndjson_stream_yields_events_and_drops_heartbeat(
    client_transport: Transport,
) -> None:
    """Drive a query stream against the in-memory server.

    Without LLM credentials configured the engine's ``api.query`` will
    fail at LLM build time; we still expect at least the
    ``query_started`` event and a terminal ``final`` of status
    ``failed``. The point of this test is to verify NDJSON parsing
    + heartbeat suppression at the transport layer, not the engine
    success path (covered in ``tests/server/test_query_stream.py``).
    """
    types: list[str] = []
    async with client_transport.stream_ndjson(
        "POST",
        "/v1/query",
        json_body={"q": "hello", "limit": 3},
    ) as events:
        async for ev in events:
            types.append(ev["type"])
            assert ev["type"] != "heartbeat"  # transport must drop it
            if ev["type"] == "final":
                break

    # At minimum we got a query_started and a final.
    assert "query_started" in types
    assert types[-1] == "final"


@pytest.mark.asyncio
async def test_stream_4xx_surfaces_as_client_error_before_iteration(
    client_transport: Transport,
) -> None:
    """A 400 from the streaming endpoint must surface as ClientError
    *before* the iterator yields anything — partial streams from a
    failed request would be ambiguous to render."""
    with pytest.raises(ClientError) as excinfo:
        async with client_transport.stream_ndjson(
            "POST",
            "/v1/query",
            json_body={"q": "   ", "limit": 3},  # empty after strip → 400
        ) as events:
            async for _ in events:
                pytest.fail("stream should have raised before yielding")
    assert excinfo.value.status == 400
    assert excinfo.value.code == "bad_request"


@pytest.mark.asyncio
async def test_transport_wraps_request_error_as_network_client_error() -> None:
    """When httpx fails before any response (DNS, refused, dropped
    socket) we must surface a ``ClientError(status=0, code='network_error')``
    instead of leaking a raw httpx traceback. Every CLI command relies
    on this single failure-channel to render a clean message + exit."""
    # Bind to a port nobody listens on — the OS rejects the connection
    # immediately so the test is fast and deterministic.
    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:1",
        timeout=httpx.Timeout(connect=0.5, read=0.5, write=0.5, pool=0.5),
    )
    transport = Transport(client=client, token=None)
    try:
        with pytest.raises(ClientError) as excinfo:
            await transport.get_json("/v1/status")
        assert excinfo.value.status == 0
        assert excinfo.value.code == "network_error"
    finally:
        await client.aclose()
