"""OpenAI-compatible LLM + Embedding provider.

Covers OpenAI, Azure OpenAI, Ollama, vLLM, TEI, DeepSeek, Gemini-compat, and
anything else that speaks the OpenAI HTTP surface via ``base_url`` + ``api_key``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from ._http import build_no_keepalive_async_client
from .base import LLMResponse, LLMStreamEvent, ProviderError, ToolSpec

if TYPE_CHECKING:  # avoid importing openai at module load for envs without it
    from openai import AsyncOpenAI

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


API_KEY_ENV = "OPENAI_API_KEY"
EMBEDDING_API_KEY_ENV = "DIKW_EMBEDDING_API_KEY"


def _resolve_api_key(explicit: str | None) -> str:
    key = explicit or os.environ.get(API_KEY_ENV)
    if not key:
        raise ProviderError(
            f"{API_KEY_ENV} is not set. Export it or pass `api_key` explicitly."
        )
    return key


def _resolve_embedding_api_key(explicit: str | None) -> str:
    """Resolve the embedding-leg API key.

    The embedding provider reads only ``DIKW_EMBEDDING_API_KEY`` — never
    ``OPENAI_API_KEY``. This is deliberate: the intended deployment splits
    the LLM and embedding legs across different vendors (e.g., MiniMax LLM +
    Gitee AI embeddings), each with its own key. Conflating them via
    ``OPENAI_API_KEY`` silently cross-wires credentials and masks misconfig.
    """
    key = explicit or os.environ.get(EMBEDDING_API_KEY_ENV)
    if not key:
        raise ProviderError(
            f"{EMBEDDING_API_KEY_ENV} is not set. "
            "Export it or pass `api_key` explicitly."
        )
    return key


def _client(
    base_url: str,
    api_key: str,
    *,
    max_retries: int | None = None,
    timeout_seconds: float | None = None,
) -> AsyncOpenAI:
    from openai import AsyncOpenAI

    kwargs: dict[str, Any] = {"base_url": base_url, "api_key": api_key}
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    timeout, http_client = build_no_keepalive_async_client(timeout_seconds)
    kwargs["http_client"] = http_client
    kwargs["timeout"] = timeout
    return AsyncOpenAI(**kwargs)


class OpenAICompatLLM:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL)
        self._api_key_explicit = api_key
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._client_cache: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client_cache is None:
            self._client_cache = _client(
                self._base_url,
                _resolve_api_key(self._api_key_explicit),
                max_retries=self._max_retries,
                timeout_seconds=self._timeout_seconds,
            )
        return self._client_cache

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Tools are not translated into OpenAI-compat tool_call format yet; Phase 1
        # only needs plain text completion for query answering.
        _ = tools
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = {}
        if resp.usage is not None:
            usage = {
                "input_tokens": int(resp.usage.prompt_tokens or 0),
                "output_tokens": int(resp.usage.completion_tokens or 0),
            }
        return LLMResponse(
            text=text,
            finish_reason=choice.finish_reason,
            usage=usage,
        )

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
        # Tool-call streaming would need to interleave token + tool_use
        # events; synth doesn't use tools yet, so the stream path mirrors
        # ``complete``'s tool-free shape.
        _ = tools
        client = self._get_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        async def _gen() -> AsyncIterator[LLMStreamEvent]:
            # ``stream_options.include_usage`` asks the server to emit one
            # final chunk carrying token usage — without it the SDK only
            # surfaces usage on non-streamed responses, so a streamed call
            # would always report empty usage to the bus subscriber.
            # The SDK's TypedDict for ``stream_options`` and the literal-True
            # overload's ``messages`` typing both reject our plain dicts; the
            # values are structurally correct, so silence both at the call.
            stream = await client.chat.completions.create(  # type: ignore[call-overload]
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            parts: list[str] = []
            finish_reason: str | None = None
            usage: dict[str, int] = {}
            try:
                async for chunk in stream:
                    # Some servers (Gitee AI, vLLM) emit a usage-only
                    # chunk with no choices; gate on truthy choices.
                    if chunk.choices:
                        choice = chunk.choices[0]
                        delta_text = getattr(choice.delta, "content", None) or ""
                        if delta_text:
                            parts.append(delta_text)
                            yield LLMStreamEvent(type="token", delta=delta_text)
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason
                    if chunk.usage is not None:
                        usage = {
                            "input_tokens": int(chunk.usage.prompt_tokens or 0),
                            "output_tokens": int(chunk.usage.completion_tokens or 0),
                        }
            finally:
                # Older SDK versions expose ``aclose`` on the stream;
                # newer ones close on iteration completion. Guard both.
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    await aclose()
            yield LLMStreamEvent(
                type="done",
                text="".join(parts),
                finish_reason=finish_reason,
                usage=usage,
            )

        return _gen()


class OpenAICompatEmbeddings:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        default_dimensions: int | None = None,
        max_retries: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL)
        self._api_key_explicit = api_key
        self._default_dimensions = default_dimensions
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._client_cache: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client_cache is None:
            self._client_cache = _client(
                self._base_url,
                _resolve_embedding_api_key(self._api_key_explicit),
                max_retries=self._max_retries,
                timeout_seconds=self._timeout_seconds,
            )
        return self._client_cache

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        kwargs: dict[str, Any] = {"model": model, "input": texts}
        if self._default_dimensions is not None:
            kwargs["dimensions"] = self._default_dimensions
        # Wrap OpenAI SDK exceptions as ``ProviderError`` so the embed-
        # batch retry-skip in ``info.embed._run_batch_with_retry`` treats
        # transient API failures (timeouts, rate limits, 5xx) the same
        # way it treats local provider errors. Without this wrap, the
        # SDK raises ``openai.OpenAIError`` subclasses that the retry
        # handler doesn't catch — a single 503 then aborts the whole
        # ingest / lint-apply / wisdom-write call (codex review
        # finding, 0.4.0). ``openai`` is reached lazily at call time
        # — same lazy import the ``_client`` helper uses, so envs
        # without the SDK still load this module.
        from openai import OpenAIError
        try:
            resp = await client.embeddings.create(**kwargs)
        except OpenAIError as exc:
            raise ProviderError(
                f"OpenAI-compat embedding call failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return [list(r.embedding) for r in resp.data]
