"""Provider contract: embed methods must wrap transient SDK / HTTP
exceptions as ``ProviderError``.

The embed-batch retry-skip path in ``info.embed._run_batch_with_retry``
only catches ``ProviderError``. If a provider's ``embed`` lets a raw
``openai.OpenAIError`` or ``httpx.HTTPStatusError`` propagate, a single
transient API failure (5xx, timeout, rate limit) aborts the whole
ingest / lint-apply / wisdom-write call instead of being skipped as
intended. This test pins the wrapping at the provider boundary so the
retry-skip contract is honoured for the two embedders we ship.

Codex review finding, 0.4.0.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from dikw_core.providers.base import ProviderError
from dikw_core.providers.gitee_multimodal import GiteeMultimodalEmbedding
from dikw_core.providers.openai_compat import OpenAICompatEmbeddings
from dikw_core.schemas import MultimodalInput


@pytest.mark.asyncio
async def test_openai_compat_embed_wraps_openai_error_as_provider_error() -> None:
    """``OpenAICompatEmbeddings.embed`` must surface OpenAI SDK
    exceptions as ``ProviderError`` so the retry-skip handler catches
    them.
    """
    from openai import APITimeoutError

    embedder = OpenAICompatEmbeddings(base_url="https://example.test/v1", api_key="k")
    fake_client = MagicMock()
    fake_client.embeddings = MagicMock()
    fake_client.embeddings.create = AsyncMock(
        side_effect=APITimeoutError(httpx.Request("POST", "https://example.test"))
    )
    embedder._client_cache = fake_client

    with pytest.raises(ProviderError) as ei:
        await embedder.embed(["hello"], model="text-embedding-3-small")
    assert "APITimeoutError" in str(ei.value)


@pytest.mark.asyncio
async def test_openai_compat_embed_wraps_api_status_error_as_provider_error() -> None:
    """A 5xx / 429 from the upstream must also wrap to ``ProviderError``,
    not propagate as ``openai.APIStatusError``.
    """
    from openai import APIStatusError

    embedder = OpenAICompatEmbeddings(base_url="https://example.test/v1", api_key="k")
    fake_client = MagicMock()
    fake_client.embeddings = MagicMock()
    fake_response = MagicMock(status_code=503, request=httpx.Request("POST", "u"))
    fake_client.embeddings.create = AsyncMock(
        side_effect=APIStatusError(
            "Service Unavailable", response=fake_response, body=None
        )
    )
    embedder._client_cache = fake_client

    with pytest.raises(ProviderError) as ei:
        await embedder.embed(["hello"], model="text-embedding-3-small")
    assert "APIStatusError" in str(ei.value)


@pytest.mark.asyncio
async def test_gitee_multimodal_embed_wraps_http_status_error_as_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GiteeMultimodalEmbedding.embed`` must surface ``httpx.HTTPStatusError``
    (5xx) as ``ProviderError`` so retry-skip catches it.
    """

    embedder = GiteeMultimodalEmbedding(base_url="https://example.test/v1")
    monkeypatch.setenv("DIKW_EMBEDDING_API_KEY", "k")

    fake_client = MagicMock()

    async def _post(*_: Any, **__: Any) -> Any:
        resp = MagicMock()
        resp.status_code = 503
        resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "503", request=httpx.Request("POST", "u"), response=MagicMock()
            )
        )
        return resp

    fake_client.post = AsyncMock(side_effect=_post)
    embedder._client = fake_client

    with pytest.raises(ProviderError) as ei:
        await embedder.embed(
            [MultimodalInput(text="hello")], model="Qwen3-VL-Embedding-8B"
        )
    assert "HTTPStatusError" in str(ei.value)


@pytest.mark.asyncio
async def test_gitee_multimodal_embed_wraps_transport_error_as_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection errors must also wrap to ``ProviderError`` — they are
    the most common transient failure in practice (Gitee TCP keepalive
    drops mid-batch).
    """

    embedder = GiteeMultimodalEmbedding(base_url="https://example.test/v1")
    monkeypatch.setenv("DIKW_EMBEDDING_API_KEY", "k")

    fake_client = MagicMock()
    fake_client.post = AsyncMock(side_effect=httpx.ConnectError("connection reset"))
    embedder._client = fake_client

    with pytest.raises(ProviderError) as ei:
        await embedder.embed(
            [MultimodalInput(text="hello")], model="Qwen3-VL-Embedding-8B"
        )
    assert "ConnectError" in str(ei.value)
