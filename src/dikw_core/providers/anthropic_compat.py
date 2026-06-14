"""Anthropic-compatible LLM provider.

Wraps the official ``anthropic`` SDK and points at any Anthropic-protocol-
compatible endpoint via ``base_url`` (api.anthropic.com by default; MiniMax's
``https://api.minimaxi.com/anthropic`` and other gateway endpoints work too).
Prompt caching is applied to the system prompt via ``cache_control`` — the
system prompt is the near-static part across ``synthesize`` sessions, so
it benefits most. The Anthropic protocol has no embeddings endpoint;
embeddings must go through the OpenAI-compatible provider.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from ..telemetry import trace_llm_stream
from .base import (
    LLMResponse,
    LLMStreamEvent,
    ProviderError,
    ToolSpec,
    TransientProviderError,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


API_KEY_ENV = "ANTHROPIC_API_KEY"


def _resolve_api_key(explicit: str | None) -> str:
    key = explicit or os.environ.get(API_KEY_ENV)
    if not key:
        raise ProviderError(
            f"{API_KEY_ENV} is not set. Export it or pass `api_key` explicitly."
        )
    return key


class AnthropicCompatLLM:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._api_key_explicit = api_key
        self._base_url = base_url
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._client_cache: AsyncAnthropic | None = None

    def _get_client(self) -> AsyncAnthropic:
        if self._client_cache is None:
            import httpx
            from anthropic import AsyncAnthropic

            kwargs: dict[str, Any] = {
                "api_key": _resolve_api_key(self._api_key_explicit),
            }
            if self._base_url is not None:
                kwargs["base_url"] = self._base_url
            if self._max_retries is not None:
                kwargs["max_retries"] = self._max_retries
            # Default 600s timeout in the SDK lets a stale keepalive hang
            # the pipeline; bound it so a dead connection raises fast and
            # the SDK retries with a fresh socket. Disabling keepalive
            # ensures each retry establishes a new TCP connection rather
            # than looping on the same dead pooled socket — the failure
            # mode observed against Gitee AI's batch embedding endpoint
            # also happens with some Anthropic-compatible LLM proxies.
            if self._timeout_seconds is not None:
                timeout = httpx.Timeout(
                    connect=10.0,
                    read=self._timeout_seconds,
                    write=self._timeout_seconds,
                    pool=5.0,
                )
                kwargs["timeout"] = timeout
                kwargs["http_client"] = httpx.AsyncClient(
                    timeout=timeout,
                    limits=httpx.Limits(max_keepalive_connections=0),
                )
            self._client_cache = AsyncAnthropic(**kwargs)
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
        # ``complete`` is a collapse of ``complete_stream``: a reasoning model
        # (e.g. MiniMax-M3) can spend minutes on hidden chain-of-thought, and
        # a non-streaming ``messages.create`` bounds the WHOLE response by the
        # read timeout, so a long synthesis times out mid-receipt. Streaming
        # makes the read timeout apply PER SSE event (token / thinking / ping
        # keepalive) instead, so a steadily-streaming generation never trips
        # it. Iterate the event stream and read the terminal ``done`` event,
        # which already carries the assembled text, finish_reason, and usage.
        text = ""
        finish_reason: str | None = None
        usage: dict[str, int] = {}
        async for event in self.complete_stream(
            system=system,
            user=user,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
        ):
            if event.type == "done":
                text = event.text or ""  # "" is a legal zero-page synth result
                finish_reason = event.finish_reason
                usage = event.usage
        return LLMResponse(text=text, finish_reason=finish_reason, usage=usage)

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
        _ = tools
        client = self._get_client()
        # Same cache-eligible system block as ``complete`` so a streamed
        # call still benefits from prompt cache hits across query/synth
        # bursts. cache_control + streaming are orthogonal in the SDK.
        system_block: list[dict[str, Any]] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

        async def _gen() -> AsyncIterator[LLMStreamEvent]:
            # Classify SDK failures so the synth group retry loop
            # (api_synth, ``cfg.synth.provider_error_retries``) retries
            # transient ones (timeout, connect drop, 5xx/408/429) and a
            # permanent misconfig (401/403/404, bad model) fails fast instead
            # of being retried-then-skipped — mirrors ``OpenAICompatEmbeddings.
            # embed``. APITimeoutError IS-A APIConnectionError IS-A APIError,
            # so the except order (timeout/connect, then status, then base)
            # is load-bearing.
            from anthropic import (
                APIConnectionError,
                APIError,
                APIStatusError,
                APITimeoutError,
            )

            parts: list[str] = []
            usage: dict[str, int] = {}
            finish_reason: str | None = None
            try:
                async with client.messages.stream(
                    model=model,
                    system=system_block,  # type: ignore[arg-type]
                    messages=[{"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                ) as stream:
                    async for delta in stream.text_stream:
                        if delta:
                            yield LLMStreamEvent(type="token", delta=delta)
                    final = await stream.get_final_message()
                for block in final.content:
                    text = getattr(block, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
                if final.usage is not None:
                    usage = {
                        "input_tokens": int(
                            getattr(final.usage, "input_tokens", 0) or 0
                        ),
                        "output_tokens": int(
                            getattr(final.usage, "output_tokens", 0) or 0
                        ),
                        "cache_creation_input_tokens": int(
                            getattr(final.usage, "cache_creation_input_tokens", 0) or 0
                        ),
                        "cache_read_input_tokens": int(
                            getattr(final.usage, "cache_read_input_tokens", 0) or 0
                        ),
                    }
                finish_reason = final.stop_reason
            except asyncio.CancelledError:
                # BaseException — must propagate so synth's per-group cancel
                # contract holds; never reclassify a cancel as transient.
                raise
            except (APITimeoutError, APIConnectionError) as exc:
                raise TransientProviderError(
                    f"Anthropic-compat completion timed out / connection failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            except APIStatusError as exc:
                status = getattr(exc, "status_code", None)
                err = (
                    TransientProviderError
                    if status is not None and (status >= 500 or status in (408, 429))
                    else ProviderError
                )
                raise err(
                    f"Anthropic-compat completion failed with status {status}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            except APIError as exc:
                raise ProviderError(
                    f"Anthropic-compat completion failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            yield LLMStreamEvent(
                type="done",
                text="".join(parts),
                finish_reason=finish_reason,
                usage=usage,
            )

        # Wrap in a gen_ai.chat span; the done event's usage (incl. Anthropic
        # cache_read/creation tokens) lands on the span. Body is unchanged.
        return trace_llm_stream(
            _gen(),
            system="anthropic",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
