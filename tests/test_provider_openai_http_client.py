"""The real ``openai`` SDK must send requests through our ``httpx`` client.

``providers/_http.py`` hands ``AsyncOpenAI`` an ``httpx.AsyncClient``. openai
3.x moved to ``httpx2`` and accepts that client only through a legacy shim —
the support anthropic 1.x dropped (#279). Every other openai test stubs
``AsyncOpenAI`` out, so drive a real ``embed`` through the SDK over a
``MockTransport`` (no network), covering both construction and the request.
"""

from __future__ import annotations

import httpx
import pytest

from dikw_core.providers.openai_compat import OpenAICompatEmbeddings


async def test_embed_sends_request_through_our_httpx_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

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
        transport = httpx.MockTransport(handler)
        return timeout, httpx.AsyncClient(timeout=timeout, transport=transport)

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
        await embedder._get_client().close()
