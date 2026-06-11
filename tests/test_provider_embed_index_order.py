"""OpenAI-compat embeddings must be re-ordered by the response ``index``.

The OpenAI embeddings response carries an explicit ``index`` per item
precisely because list order is not part of the contract — any compatible
gateway (Ollama, vLLM, TEI, …) may return items out of order. The consumer
(``info.embed``) pairs vectors to chunks positionally and persists them into
the content-hash embedding cache, so an unsorted response silently
mis-assigns vectors AND poisons the cache. ``gitee_multimodal`` already
sorts its rows by ``index``; this pins the same defence on
``OpenAICompatEmbeddings``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dikw_core.providers.openai_compat import OpenAICompatEmbeddings


@pytest.mark.asyncio
async def test_openai_compat_embed_reorders_response_by_index() -> None:
    embedder = OpenAICompatEmbeddings(base_url="https://example.test/v1", api_key="k")
    fake_client = MagicMock()
    fake_client.embeddings = MagicMock()
    # Server returns the batch OUT OF ORDER: index 2, 0, 1.
    fake_client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(index=2, embedding=[2.0, 2.0]),
                SimpleNamespace(index=0, embedding=[0.0, 0.0]),
                SimpleNamespace(index=1, embedding=[1.0, 1.0]),
            ]
        )
    )
    embedder._client_cache = fake_client

    vectors = await embedder.embed(["zero", "one", "two"], model="m")
    assert vectors == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]


@pytest.mark.asyncio
async def test_openai_compat_embed_unindexed_rows_keep_response_position() -> None:
    """A gateway that omits ``index`` on SOME rows must not have those rows
    yanked to the front (a missing-index→0 fallback sorts them ahead of every
    real index > 0). The fallback key is the row's response position, so
    unindexed rows stay where the gateway put them."""
    embedder = OpenAICompatEmbeddings(base_url="https://example.test/v1", api_key="k")
    fake_client = MagicMock()
    fake_client.embeddings = MagicMock()
    fake_client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[1.0]),
                SimpleNamespace(embedding=[9.0]),  # no index — stays at pos 1
            ]
        )
    )
    embedder._client_cache = fake_client

    vectors = await embedder.embed(["a", "b"], model="m")
    assert vectors == [[1.0], [9.0]]


@pytest.mark.asyncio
async def test_openai_compat_embed_in_order_response_unchanged() -> None:
    embedder = OpenAICompatEmbeddings(base_url="https://example.test/v1", api_key="k")
    fake_client = MagicMock()
    fake_client.embeddings = MagicMock()
    fake_client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(index=0, embedding=[0.0]),
                SimpleNamespace(index=1, embedding=[1.0]),
            ]
        )
    )
    embedder._client_cache = fake_client

    vectors = await embedder.embed(["a", "b"], model="m")
    assert vectors == [[0.0], [1.0]]
