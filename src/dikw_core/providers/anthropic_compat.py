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
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from ..telemetry import trace_llm_stream
from .base import (
    LLMResponse,
    LLMStreamEvent,
    ProviderError,
    ToolSpec,
    TransientProviderError,
    _resolve_key,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from anthropic.types import TextBlockParam


class AnthropicCompatLLM:
    def __init__(
        self,
        *,
        api_key_env: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._api_key_env = api_key_env
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
                "api_key": _resolve_key(self._api_key_explicit, self._api_key_env),
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
            # These are ``httpx`` objects: anthropic>=1 moved to ``httpx2``
            # and rejects them, hence the ``anthropic<1`` cap (#279).
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
        system_block: list[TextBlockParam] = [
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
            started = False
            stopped = False
            block_types: list[str] = []
            open_blocks: set[int] = set()
            empty_block_stop = False
            try:
                # The raw SDK stream keeps transport/retries/per-event timeouts,
                # without its message accumulator indexing a nonexistent block
                # on MiniMax's empty-turn content_block_stop(index=0).
                raw_stream = await client.messages.create(
                    model=model,
                    system=system_block,
                    messages=[{"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                async with raw_stream:
                    async for event in raw_stream:
                        if event.type == "message_start":
                            if started:
                                raise _stream_error("duplicate message_start")
                            started = True
                            for block in event.message.content:
                                block_types.append(block.type)
                                open_blocks.add(len(block_types) - 1)
                                if block.type == "text":
                                    parts.append(block.text)
                            usage = {
                                name: int(getattr(event.message.usage, name, 0) or 0)
                                for name in _USAGE_FIELDS
                            }
                        elif not started:
                            raise _stream_error("event before message_start")
                        elif event.type == "content_block_start":
                            if empty_block_stop or event.index != len(block_types):
                                raise _stream_error("invalid content_block_start index")
                            block_types.append(event.content_block.type)
                            open_blocks.add(event.index)
                            if event.content_block.type == "text":
                                parts.append(event.content_block.text)
                        elif event.type == "content_block_delta":
                            if event.index not in open_blocks:
                                raise _stream_error("content_block_delta without an open block")
                            if event.delta.type == "text_delta":
                                if block_types[event.index] != "text":
                                    raise _stream_error("text_delta for a non-text block")
                                if event.delta.text:
                                    parts.append(event.delta.text)
                                    yield LLMStreamEvent(type="token", delta=event.delta.text)
                        elif event.type == "content_block_stop":
                            if event.index in open_blocks:
                                open_blocks.remove(event.index)
                            elif event.index == 0 and not block_types and not empty_block_stop:
                                # Permit only the captured empty-turn shape. A
                                # later start/delta, wrong index or duplicate stop
                                # is still malformed, and the terminal reason must
                                # confirm a normal stop before emitting done.
                                empty_block_stop = True
                            else:
                                raise _stream_error("content_block_stop without an open block")
                        elif event.type == "message_delta":
                            if event.delta.stop_reason is not None:
                                finish_reason = event.delta.stop_reason
                            for name in _USAGE_FIELDS:
                                value = getattr(event.usage, name, None)
                                if value is not None:
                                    usage[name] = int(value)
                        elif event.type == "message_stop":
                            stopped = True
                            break
                if not started or not stopped or open_blocks or finish_reason is None:
                    raise _stream_error("incomplete message")
                if empty_block_stop and finish_reason not in ("end_turn", "stop_sequence", "stop"):
                    raise _stream_error("orphan block stop without a normal message stop")
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


_USAGE_FIELDS = (
    "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
)


def _stream_error(detail: str) -> TransientProviderError:
    # The synth loop can retry the whole request, then skip a failing group.
    # Never report an incomplete/misindexed stream as a successful empty turn.
    return TransientProviderError(f"Anthropic-compat malformed stream: {detail}")
