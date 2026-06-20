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

from dikw_core.providers.base import ProviderError, TransientProviderError
from dikw_core.providers.gitee_multimodal import GiteeMultimodalEmbedding
from dikw_core.providers.openai_compat import OpenAICompatEmbeddings
from dikw_core.schemas import MultimodalInput


@pytest.mark.asyncio
async def test_openai_compat_embed_classifies_timeout_as_transient() -> None:
    """``APITimeoutError`` is transient — must wrap as
    ``TransientProviderError`` so the retry-skip handler retries it.
    """
    from openai import APITimeoutError

    embedder = OpenAICompatEmbeddings(
        api_key_env="OPENAI_API_KEY", base_url="https://example.test/v1", api_key="k"
    )
    fake_client = MagicMock()
    fake_client.embeddings = MagicMock()
    fake_client.embeddings.create = AsyncMock(
        side_effect=APITimeoutError(httpx.Request("POST", "https://example.test"))
    )
    embedder._client_cache = fake_client

    with pytest.raises(TransientProviderError) as ei:
        await embedder.embed(["hello"], model="text-embedding-3-small")
    assert "APITimeoutError" in str(ei.value)


@pytest.mark.asyncio
async def test_openai_compat_embed_classifies_503_as_transient() -> None:
    """A 503 must wrap as ``TransientProviderError`` (retryable)."""
    from openai import APIStatusError

    embedder = OpenAICompatEmbeddings(
        api_key_env="OPENAI_API_KEY", base_url="https://example.test/v1", api_key="k"
    )
    fake_client = MagicMock()
    fake_client.embeddings = MagicMock()
    fake_response = MagicMock(status_code=503, request=httpx.Request("POST", "u"))
    fake_client.embeddings.create = AsyncMock(
        side_effect=APIStatusError(
            "Service Unavailable", response=fake_response, body=None
        )
    )
    embedder._client_cache = fake_client

    with pytest.raises(TransientProviderError) as ei:
        await embedder.embed(["hello"], model="text-embedding-3-small")
    assert "APIStatusError" in str(ei.value)
    assert "503" in str(ei.value)


@pytest.mark.asyncio
async def test_openai_compat_embed_classifies_401_as_permanent() -> None:
    """A 401 (auth) must wrap as a permanent ``ProviderError`` —
    NOT a ``TransientProviderError``. Permanent misconfig must propagate
    instead of being retried-then-skipped, otherwise the user gets
    "success, 0 vectors" silently (codex round-2 finding).
    """
    from openai import APIStatusError

    embedder = OpenAICompatEmbeddings(
        api_key_env="OPENAI_API_KEY", base_url="https://example.test/v1", api_key="k"
    )
    fake_client = MagicMock()
    fake_client.embeddings = MagicMock()
    fake_response = MagicMock(status_code=401, request=httpx.Request("POST", "u"))
    fake_client.embeddings.create = AsyncMock(
        side_effect=APIStatusError(
            "Unauthorized", response=fake_response, body=None
        )
    )
    embedder._client_cache = fake_client

    with pytest.raises(ProviderError) as ei:
        await embedder.embed(["hello"], model="text-embedding-3-small")
    assert not isinstance(ei.value, TransientProviderError)
    assert "401" in str(ei.value)


@pytest.mark.asyncio
async def test_openai_compat_embed_classifies_404_model_as_permanent() -> None:
    """A 404 (invalid model id) must wrap as a permanent ``ProviderError``
    — retrying a typo'd model name forever is silent corruption.
    """
    from openai import APIStatusError

    embedder = OpenAICompatEmbeddings(
        api_key_env="OPENAI_API_KEY", base_url="https://example.test/v1", api_key="k"
    )
    fake_client = MagicMock()
    fake_client.embeddings = MagicMock()
    fake_response = MagicMock(status_code=404, request=httpx.Request("POST", "u"))
    fake_client.embeddings.create = AsyncMock(
        side_effect=APIStatusError(
            "Not Found", response=fake_response, body=None
        )
    )
    embedder._client_cache = fake_client

    with pytest.raises(ProviderError) as ei:
        await embedder.embed(["hello"], model="bogus-model")
    assert not isinstance(ei.value, TransientProviderError)


def _gitee_http_status_post_factory(status: int) -> Any:
    async def _post(*_: Any, **__: Any) -> Any:
        resp = MagicMock()
        resp.status_code = status
        resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                f"{status}",
                request=httpx.Request("POST", "u"),
                response=MagicMock(status_code=status),
            )
        )
        return resp

    return _post


@pytest.mark.asyncio
async def test_gitee_multimodal_classifies_503_as_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gitee 5xx must wrap as ``TransientProviderError`` — retry-skip
    should retry the batch.
    """

    embedder = GiteeMultimodalEmbedding(
        api_key_env="GITEE_API_KEY", base_url="https://example.test/v1"
    )
    monkeypatch.setenv("GITEE_API_KEY", "k")

    fake_client = MagicMock()
    fake_client.post = AsyncMock(side_effect=_gitee_http_status_post_factory(503))
    embedder._client = fake_client

    with pytest.raises(TransientProviderError):
        await embedder.embed(
            [MultimodalInput(text="hello")], model="Qwen3-VL-Embedding-8B"
        )


@pytest.mark.asyncio
async def test_gitee_multimodal_classifies_401_as_permanent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 401 (bad API key) must wrap as a permanent ``ProviderError``,
    NOT a ``TransientProviderError`` — retrying a wrong key forever
    is silent corruption.
    """

    embedder = GiteeMultimodalEmbedding(
        api_key_env="GITEE_API_KEY", base_url="https://example.test/v1"
    )
    monkeypatch.setenv("GITEE_API_KEY", "k")

    fake_client = MagicMock()
    fake_client.post = AsyncMock(side_effect=_gitee_http_status_post_factory(401))
    embedder._client = fake_client

    with pytest.raises(ProviderError) as ei:
        await embedder.embed(
            [MultimodalInput(text="hello")], model="Qwen3-VL-Embedding-8B"
        )
    assert not isinstance(ei.value, TransientProviderError)


@pytest.mark.asyncio
async def test_gitee_multimodal_classifies_connection_error_as_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection drops must wrap as ``TransientProviderError`` — the
    most common Gitee failure (TCP keepalive mid-batch).
    """

    embedder = GiteeMultimodalEmbedding(
        api_key_env="GITEE_API_KEY", base_url="https://example.test/v1"
    )
    monkeypatch.setenv("GITEE_API_KEY", "k")

    fake_client = MagicMock()
    fake_client.post = AsyncMock(side_effect=httpx.ConnectError("connection reset"))
    embedder._client = fake_client

    with pytest.raises(TransientProviderError) as ei:
        await embedder.embed(
            [MultimodalInput(text="hello")], model="Qwen3-VL-Embedding-8B"
        )
    assert "ConnectError" in str(ei.value)
