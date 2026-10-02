"""The real ``openai`` SDK must send requests through our ``httpx`` client.

``providers/_http.py`` hands ``AsyncOpenAI`` an ``httpx.AsyncClient``. openai
3.x moved to ``httpx2`` and accepts that client only through a legacy shim —
the support anthropic 1.x dropped (#279) — so this guards the lifted
``openai<3`` cap (#285). Drive a real ``embed`` through the SDK over a
``MockTransport`` (no network), covering both construction and the request.
``test_provider_openai_codex_sse.py`` also exercises the real Responses reducer.
"""

from __future__ import annotations

import httpx
import pytest

from dikw_core.providers.openai_compat import OpenAICompatEmbeddings


async def test_embed_sends_request_through_our_httpx_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    clients: list[httpx.AsyncClient] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "m",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    def mock_client(
        timeout_seconds: float | None = None,
    ) -> tuple[httpx.Timeout, httpx.AsyncClient]:
        timeout = httpx.Timeout(10.0)
        client = httpx.AsyncClient(timeout=timeout, transport=httpx.MockTransport(handler))
        clients.append(client)
        return timeout, client

    monkeypatch.setattr(
        "dikw_core.providers.openai_compat.build_no_keepalive_async_client", mock_client
    )
    embedder = OpenAICompatEmbeddings(
        api_key_env="OPENAI_API_KEY", api_key="sk-test", base_url="http://stub.test/v1"
    )
    try:
        assert await embedder.embed(["hello"], model="m") == [[0.1, 0.2]]
        assert seen == ["/v1/embeddings"]
    finally:
        for client in clients:
            await client.aclose()
