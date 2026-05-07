"""OpenAI Codex provider — ChatGPT-backend Responses API with auto-refreshing OAuth.

Distinct from ``openai_compat.py`` despite sharing the openai SDK: Codex
speaks the **Responses API** (``client.responses.create``), authenticates
with a ChatGPT-issued OAuth ``access_token`` (resolved + refreshed via
``codex_auth``, not ``OPENAI_API_KEY``), and requires Cloudflare-mitigation
headers (``originator``, ``ChatGPT-Account-ID`` from the JWT). ``gpt-5.5``
and the rest of the codex model family live exclusively on
``https://chatgpt.com/backend-api/codex``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._http import build_no_keepalive_async_client
from .base import LLMResponse, LLMStreamEvent, ToolSpec
from .codex_auth import account_id_from_jwt, resolve_access_token

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Cloudflare requires these headers on every chatgpt.com/backend-api/codex
# request — without them the gateway returns 403 before the request hits
# the upstream model. The originator string is the literal codex CLI
# reports; matching it is the only way to satisfy the gate today.
_CODEX_BASE_HEADERS: dict[str, str] = {
    "originator": "codex_cli_rs",
    "User-Agent": "codex_cli_rs/0.1 (dikw-core)",
}

_FINISH_REASON_MAP: dict[str, str] = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "stop",
    "cancelled": "stop",
}


def _build_async_client(
    *,
    base_url: str,
    access_token: str,
    max_retries: int | None,
    timeout_seconds: float | None,
) -> AsyncOpenAI:
    """Construct a fresh ``AsyncOpenAI`` for one request lifecycle.

    We rebuild per-call rather than caching: the OAuth access_token is
    short-lived and a stale client cached across token refreshes would
    silently 401. The ``_is_expiring`` check is nanosecond-cheap; the
    rebuild costs are dominated by httpx connection setup, comparable to
    the SDK's own behaviour on cache miss.
    """
    from openai import AsyncOpenAI

    headers: dict[str, str] = dict(_CODEX_BASE_HEADERS)
    account_id = account_id_from_jwt(access_token)
    if account_id is not None:
        headers["ChatGPT-Account-ID"] = account_id

    timeout, http_client = build_no_keepalive_async_client(timeout_seconds)
    kwargs: dict[str, Any] = {
        "api_key": access_token,
        "base_url": base_url,
        "default_headers": headers,
        "timeout": timeout,
        "http_client": http_client,
    }
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return AsyncOpenAI(**kwargs)


def _extract_text_from_response(response: Any) -> str:
    """Walk ``response.output``, gather output_text from message items.

    Reasoning items, tool_call items, and any other type-tagged items are
    skipped — Codex's response.output is a heterogeneous list, only the
    ``message`` items carry user-facing text.
    """
    parts: list[str] = []
    output = getattr(response, "output", None) or []
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None) or []
        for part in content:
            if getattr(part, "type", None) == "output_text":
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def _extract_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }


def _request_kwargs(
    *, system: str, user: str, model: str, max_tokens: int, temperature: float
) -> dict[str, Any]:
    """Shared payload shape for ``responses.create`` and ``responses.stream``."""
    return {
        "model": model,
        "instructions": system,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user}],
            }
        ],
        "store": False,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }


class OpenAICodexLLM:
    def __init__(
        self,
        *,
        base_url: str,
        wiki_base: Path,
        max_retries: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        # ``wiki_base`` is the wiki root that owns the OAuth token store
        # at ``<wiki_base>/.dikw/auth.json``. Multiple wikis on the same
        # machine each carry their own credentials so a refresh in one
        # doesn't invalidate the other.
        self._base_url = base_url
        self._wiki_base = wiki_base
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[AsyncOpenAI]:
        """Resolve a fresh access_token, build a per-request AsyncOpenAI,
        guarantee close() runs even if the body raises."""
        token = await resolve_access_token(self._wiki_base)
        client = _build_async_client(
            base_url=self._base_url,
            access_token=token,
            max_retries=self._max_retries,
            timeout_seconds=self._timeout_seconds,
        )
        try:
            yield client
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                await close()

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
        # ChatGPT's codex backend rejects non-streaming Responses calls
        # with ``Stream must be set to true``, so ``complete`` is a
        # collapse of ``complete_stream``: iterate the event stream and
        # read the terminal ``done`` event, which already carries the
        # assembled text, finish_reason, and usage.
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
                text = event.text or ""
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
        kwargs = _request_kwargs(
            system=system,
            user=user,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        async def _gen() -> AsyncIterator[LLMStreamEvent]:
            parts: list[str] = []
            async with (
                self._client() as client,
                client.responses.stream(**kwargs) as stream,
            ):
                async for event in stream:
                    ev_type = getattr(event, "type", None)
                    # Responses API marks text deltas with the literal
                    # "response.output_text.delta" type. Reasoning summary
                    # deltas use "response.reasoning_summary_text.delta".
                    # Anything else (response.created, output_item.added,
                    # response.completed, …) is intentionally dropped:
                    # the LLMStreamEvent contract has no slot for them
                    # and the engine only consumes token/done today.
                    if ev_type == "response.output_text.delta":
                        delta = getattr(event, "delta", None) or ""
                        if delta:
                            parts.append(delta)
                            yield LLMStreamEvent(type="token", delta=delta)
                    elif ev_type == "response.reasoning_summary_text.delta":
                        delta = getattr(event, "delta", None) or ""
                        if delta:
                            yield LLMStreamEvent(type="reasoning", delta=delta)
                final = await stream.get_final_response()

            # Prefer the SDK's authoritative final text when present (it
            # already concatenates message items the same way ``complete``
            # does); the locally-collected ``parts`` is the fallback when
            # the final payload is missing or empty.
            final_text = _extract_text_from_response(final) or "".join(parts)
            status = str(getattr(final, "status", "") or "")
            finish_reason = _FINISH_REASON_MAP.get(status, "stop")
            usage = _extract_usage(final)
            yield LLMStreamEvent(
                type="done",
                text=final_text,
                finish_reason=finish_reason,
                usage=usage,
            )

        return _gen()
