"""Provider factory: resolves LLM + Embedding instances from ``ProviderConfig``."""

from __future__ import annotations

from pathlib import Path

from ..config import ProviderConfig
from .anthropic_compat import AnthropicCompatLLM
from .base import (
    EmbeddingProvider,
    LLMProvider,
    LLMResponse,
    LLMStreamEvent,
    MultimodalEmbeddingProvider,
    ProviderError,
    RerankProvider,
    ToolSpec,
    TransientProviderError,
)
from .gitee_multimodal import GiteeMultimodalEmbedding
from .openai_codex import OpenAICodexLLM
from .openai_compat import OpenAICompatEmbeddings, OpenAICompatLLM
from .rerank import OpenAICompatReranker


def build_llm(config: ProviderConfig, *, base_root: Path | None = None) -> LLMProvider:
    """Build an LLM provider from config.

    ``base_root`` is required when ``config.llm == "openai_codex"`` because
    that provider stores its OAuth tokens at
    ``<base_root>/.dikw/auth.json``. Other LLMs ignore the parameter.
    Engine call sites already have the wiki root in scope (returned by
    ``api._with_storage``) so threading it down is a one-liner per site.
    """
    if config.llm == "anthropic_compat":
        return AnthropicCompatLLM(
            api_key_env=config.llm_api_key_env,
            base_url=config.llm_base_url,
            max_retries=config.llm_max_retries,
            timeout_seconds=config.llm_timeout_seconds,
        )
    if config.llm == "openai_compat":
        return OpenAICompatLLM(
            api_key_env=config.llm_api_key_env,
            base_url=config.llm_base_url,
            max_retries=config.llm_max_retries,
            timeout_seconds=config.llm_timeout_seconds,
        )
    if config.llm == "openai_codex":
        # The validator on ProviderConfig already required llm_base_url to
        # be set when llm == "openai_codex", so the assert here is a typing
        # narrow rather than runtime defence.
        assert config.llm_base_url is not None
        if base_root is None:
            raise ProviderError(
                "build_llm: base_root is required for the openai_codex provider — "
                "tokens live at <base_root>/.dikw/auth.json. Pass the wiki root "
                "from api._with_storage's `root` return value."
            )
        return OpenAICodexLLM(
            base_url=config.llm_base_url,
            base_root=base_root,
            max_retries=config.llm_max_retries,
            timeout_seconds=config.llm_timeout_seconds,
        )
    raise ProviderError(f"unknown LLM provider: {config.llm!r}")


def build_embedder(
    config: ProviderConfig, *, dim_override: int | None = None
) -> EmbeddingProvider:
    # The Anthropic protocol has no embeddings endpoint; both paths route
    # through OpenAI-compat
    # using ``embedding_base_url`` so users configure one endpoint explicitly.
    # ``dim_override`` lets query() pin to the active embed_versions row's
    # dim when cfg has drifted (yml edited but no re-ingest yet) — without
    # it the request would ship the new dim and get rejected by the old
    # vec_chunks_v<id> table.
    if config.embedding == "openai_compat":
        return OpenAICompatEmbeddings(
            api_key_env=config.embedding_api_key_env,
            base_url=config.embedding_base_url,
            default_dimensions=dim_override or config.embedding_dim,
            max_retries=config.embedding_max_retries,
            timeout_seconds=config.embedding_timeout_seconds,
        )
    raise ProviderError(f"unknown embedding provider: {config.embedding!r}")


def build_multimodal_embedder(
    provider: str,
    *,
    api_key_env: str,
    base_url: str | None = None,
    batch: int = 16,
) -> MultimodalEmbeddingProvider:
    """Build a multimodal embedder by name.

    One provider ships today: ``gitee_multimodal`` covers every multimodal
    model Gitee AI serves (Qwen3-VL-Embedding-8B, jina-clip-v2, …) — they
    share one wire shape, the model name in ``assets.multimodal.model``
    discriminates which one runs server-side. Additional vendors (Voyage,
    Cohere, Jina-direct) are easy follow-ons: drop a new file under
    ``providers/`` and add a branch here.
    """
    if provider == "gitee_multimodal":
        return GiteeMultimodalEmbedding(api_key_env=api_key_env, base_url=base_url, batch=batch)
    raise ProviderError(f"unknown multimodal embedding provider: {provider!r}")


def build_reranker(config: ProviderConfig) -> RerankProvider | None:
    """Build a reranker from config, or ``None`` when unconfigured.

    Returns ``None`` if ``config.rerank`` is unset — the base never opted into
    reranking, so the search layer runs no rerank leg (off because
    unconfigured). When set, the ``ProviderConfig`` validator has already
    guaranteed ``rerank_model`` / ``rerank_base_url`` / ``rerank_api_key_env``
    are present, so the asserts here are typing narrows, not runtime defence.
    """
    if config.rerank is None:
        return None
    if config.rerank == "openai_compat_rerank":
        assert config.rerank_base_url is not None
        assert config.rerank_api_key_env is not None
        return OpenAICompatReranker(
            api_key_env=config.rerank_api_key_env,
            base_url=config.rerank_base_url,
            timeout=config.rerank_timeout_seconds,
            batch_size=config.rerank_batch_size,
        )
    raise ProviderError(f"unknown rerank provider: {config.rerank!r}")


__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "LLMResponse",
    "LLMStreamEvent",
    "MultimodalEmbeddingProvider",
    "OpenAICompatReranker",
    "ProviderError",
    "RerankProvider",
    "ToolSpec",
    "TransientProviderError",
    "build_embedder",
    "build_llm",
    "build_multimodal_embedder",
    "build_reranker",
]
