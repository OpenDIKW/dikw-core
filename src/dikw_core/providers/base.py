"""LLM + Embedding provider abstractions.

Engine code talks only to these Protocols; concrete adapters in sibling files
wrap the official SDKs. Swapping providers is a config-only change at the
``providers/__init__.py`` factory level.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..schemas import MultimodalInput


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    text: str
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    raw: dict[str, Any] | None = None


class LLMStreamEvent(BaseModel):
    """One event in a streaming LLM completion.

    ``type == "token"``: incremental text fragment in ``delta``.
    ``type == "reasoning"``: thinking-process fragment in ``delta``. Optional;
    only emitted by reasoning-capable providers (OpenAI Codex Responses API).
    Consumers that only know ``token`` / ``done`` must tolerate this type as
    unrecognized and ignore it — that matches ``api.query``'s if/elif dispatch.
    ``type == "done"``: terminal event with the full assembled ``text`` and
    ``finish_reason``/``usage`` mirroring ``LLMResponse``. Sequence contract:
    zero or more ``token`` / ``reasoning`` events in any interleaving,
    followed by exactly one ``done``. Providers that don't support streaming
    raise ``NotImplementedError`` from ``complete_stream`` and callers fall
    back to ``complete``.
    """

    type: Literal["token", "reasoning", "done"]
    delta: str | None = None
    text: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)


class ProviderError(RuntimeError):
    """Base class for provider errors (auth, network, invalid model, etc.).

    ``ProviderError`` (the bare base) is the **permanent** failure class:
    auth failures, missing API key, invalid model id, 4xx other than
    rate-limit / request-timeout. Retry loops (``consume_embedding_stream``,
    the synth path's group retry) MUST let it propagate so misconfig
    fails fast instead of being silently retried-then-skipped.
    For retryable failures use :class:`TransientProviderError`.
    """


class TransientProviderError(ProviderError):
    """A retryable provider failure (5xx, 408, 429, timeout, connect drop, …).

    Raised by embedding / LLM adapters when the upstream call failed in
    a way that's plausibly worth retrying. The embed-batch retry-skip
    inside ``info.embed._run_batch_with_retry`` catches **only** this
    subclass — bare ``ProviderError`` propagates so permanent misconfig
    (missing key, 401, unknown model) fails the call instead of
    silently emitting an empty embedding batch.
    """


def _resolve_key(explicit: str | None, env_name: str) -> str:
    """Resolve an API key from an explicit value or the named env var.

    The env var name comes from config (``provider.llm_api_key_env`` /
    ``provider.embedding_api_key_env``) — the engine hardcodes no key var.
    LLM/embedding key separation is achieved by naming distinct vars (e.g.
    MiniMax LLM under ``MINIMAX_API_KEY`` + Gitee embeddings under
    ``GITEE_API_KEY``); point both at one var to share a single key. Raises
    :class:`ProviderError` (permanent) when the named var is unset, so
    misconfig fails loud with the var name instead of a wrong-key call.
    """
    key = explicit or os.environ.get(env_name)
    if not key:
        raise ProviderError(
            f"{env_name} is not set. Export it or pass `api_key` explicitly."
        )
    return key


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse: ...

    def complete_stream(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Stream a completion as ``LLMStreamEvent`` chunks.

        Optional capability: providers that haven't wired SDK-level
        streaming yet raise ``NotImplementedError``. The query layer's
        Phase-4 streaming path catches that and falls back to ``complete``
        + a single synthetic ``done`` event.
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]: ...


@runtime_checkable
class RerankProvider(Protocol):
    """Cross-encoder reranker over a retrieved candidate set.

    Scores each ``(query, document)`` pair and returns one relevance score
    per document **aligned to input order** — the search layer pairs scores
    back to candidate chunks positionally, so an adapter talking to an
    endpoint that returns results sorted by relevance MUST remap the response
    ``index`` before returning (mirrors ``EmbeddingProvider``'s defensive
    index handling).

    A reranker is a deterministic scoring model in the same epistemic
    category as the embedding model — it reorders the deterministically
    retrieved pool; it does not generate text or decide what to retrieve.
    It is part of *scoping*, not *reasoning*, so it is consistent with the
    engine's "LLM calls only enter at synth" invariant. See
    ``docs/adr/0006-reranker-deterministic-scoping.md``.
    """

    async def rerank(
        self, query: str, documents: list[str], *, model: str
    ) -> list[float]: ...


@runtime_checkable
class MultimodalEmbeddingProvider(Protocol):
    """Embedding provider that can encode text, images, or any combination
    into a single shared vector space.

    v1 callers use either text-only (chunks) or image-only (assets) inputs;
    the schema's ``MultimodalInput`` permits combined inputs for v1.5
    chunk-with-images joint encoding without breaking the wire contract.

    Output ordering must match input ordering so callers can pair vectors
    with their source rows. All vectors must have the same dimension —
    ``EmbeddingVersion.dim`` records that dim and the storage layer
    validates each row against it.
    """

    async def embed(
        self,
        inputs: list[MultimodalInput],
        *,
        model: str,
    ) -> list[list[float]]: ...


__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "LLMResponse",
    "LLMStreamEvent",
    "MultimodalEmbeddingProvider",
    "ProviderError",
    "RerankProvider",
    "ToolSpec",
    "TransientProviderError",
]
