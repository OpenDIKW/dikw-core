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

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..telemetry import trace_llm_stream
from ._http import build_no_keepalive_async_client
from .base import (
    LLMResponse,
    LLMStreamEvent,
    ToolSpec,
    TransientProviderError,
)
from .codex_auth import account_id_from_jwt, resolve_access_token

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Cloudflare requires these headers on every chatgpt.com/backend-api/codex
# request — without them the gateway returns 403 before the request hits
# the upstream model. The originator string is the literal codex CLI
# reports; matching it is the only way to satisfy the gate today.
_CODEX_BASE_HEADERS: dict[str, str] = {
    "originator": "codex_cli_rs",
    "User-Agent": "codex_cli_rs/0.1 (dikw-core)",
}

# ``failed`` / ``cancelled`` deliberately map to ``"error"`` (not
# ``"stop"``) so a hung-up or backend-rejected response is observable
# downstream: the same sentinel is used in the SDK-reducer fallback
# branch in ``complete_stream`` to mark partial-text recoveries.
_FINISH_REASON_MAP: dict[str, str] = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "error",
    "cancelled": "error",
}

# Signature of the OpenAI Python SDK's reducer failure when ChatGPT's
# codex backend ships ``response.output = None`` in its terminal
# ``response.completed`` event. We refuse to swallow any other
# TypeError/AttributeError — a real None-attribute bug in our own code
# would otherwise be silently absorbed into a fake-success done event.
_REDUCER_BUG_TYPE_ERROR_SIGNATURE = "'NoneType' object is not iterable"


_REDUCER_BUG_ATTR_ERROR_SIGNATURES: tuple[str, ...] = (
    # CPython renders missing-attribute as ``"... object has no attribute
    # 'output'"`` (single quotes); some derivative interpreters / SDK
    # wrappers use double quotes. Both pin to the exact ``output`` field
    # — substring-matching the bare token ``"output"`` would also catch
    # ``output_index`` / ``output_text`` and silently swallow unrelated
    # SDK schema failures.
    "attribute 'output'",
    'attribute "output"',
)


def _is_codex_final_response_reducer_bug(exc: BaseException) -> bool:
    """True iff ``exc`` matches the openai SDK's reducer failure on
    ``response.output = None`` from the chatgpt.com codex backend.

    The SDK reducer iterates ``response.output`` to assemble a typed
    final ``Response``; when ``output`` is None it raises
    ``TypeError("'NoneType' object is not iterable")``. Older / future
    SDK versions may surface the same root cause as an ``AttributeError``
    referencing the ``output`` field itself while introspecting the
    missing attribute; we accept that as the same bug — pinned to the
    field-name boundary so ``output_index`` / ``output_text``
    AttributeError (which would indicate a different schema problem)
    propagate. Everything else propagates — a real bug in our code
    must not be miscaptured as a successful completion.
    """
    msg = str(exc)
    if isinstance(exc, TypeError):
        return _REDUCER_BUG_TYPE_ERROR_SIGNATURE in msg
    if isinstance(exc, AttributeError):
        return any(sig in msg for sig in _REDUCER_BUG_ATTR_ERROR_SIGNATURES)
    return False


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
    """Wire payload for ``client.responses.stream(...)``.

    ChatGPT's codex backend exposes a stricter parameter whitelist than
    the public Responses API: ``max_output_tokens`` and ``temperature``
    both come back ``400 Unsupported parameter`` — generation length and
    sampling are managed server-side by the user's plan/model. The two
    knobs stay in the LLMProvider signature for protocol parity but are
    dropped on the wire; ``tests/fakes._CODEX_REJECTED_PARAMS`` keeps
    the test-side guard in lockstep with this list.
    """
    _ = max_tokens, temperature
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
    }


class OpenAICodexLLM:
    def __init__(
        self,
        *,
        base_url: str,
        base_root: Path,
        max_retries: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        # ``base_root`` is the base root that owns the OAuth token store
        # at ``<base_root>/.dikw/auth.json``. Multiple bases on the same
        # machine each carry their own credentials so a refresh in one
        # doesn't invalidate the other.
        self._base_url = base_url
        self._base_root = base_root
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[AsyncOpenAI]:
        """Resolve a fresh access_token, build a per-request AsyncOpenAI,
        guarantee close() runs even if the body raises."""
        token = await resolve_access_token(self._base_root)
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
            final: Any = None
            reducer_bug_seen = False
            async with (
                self._client() as client,
                client.responses.stream(**kwargs) as stream,
            ):
                # ChatGPT's codex backend ships ``response.output = None``
                # in its terminal ``response.completed`` payload (the
                # public Responses API ships a list). The openai SDK's
                # reducer assembles a typed final ``Response`` via
                # ``for output in response.output`` and dies with
                # ``TypeError: 'NoneType' object is not iterable``. The
                # failure can surface either inside ``async for`` (when
                # the reducer runs before the terminator yield) or from
                # ``get_final_response()`` (when the reducer is deferred);
                # we tolerate both and fall back to the locally
                # accumulated delta text. Without this, every
                # ``responses.stream`` call against chatgpt.com/backend-api/codex
                # — including ``dikw client check --llm-only`` — crashes
                # before the engine sees a single token.
                #
                # The catch is intentionally narrow: only TypeError /
                # AttributeError whose message matches the SDK reducer's
                # signature (see ``_is_codex_final_response_reducer_bug``)
                # is treated as a known compatibility quirk. Any other
                # TypeError / AttributeError — for instance a real
                # ``None.something`` slip in our own code — re-raises so
                # it does not silently surface as a fake-success done
                # event. Network / API failures live on ``__aenter__`` and
                # propagate naturally (they never enter this try block).
                try:
                    async for event in stream:
                        ev_type = getattr(event, "type", None)
                        # Responses API marks text deltas with the literal
                        # "response.output_text.delta" type. Reasoning
                        # summary deltas use
                        # "response.reasoning_summary_text.delta". Anything
                        # else (response.created, output_item.added,
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
                except (TypeError, AttributeError) as exc:
                    if not _is_codex_final_response_reducer_bug(exc):
                        raise
                    logger.warning(
                        "OpenAI Codex stream reducer failed during event "
                        "iteration; falling back to accumulated deltas "
                        "(%d chars). Underlying SDK bug: %s",
                        sum(len(p) for p in parts),
                        exc,
                        exc_info=True,
                    )
                    reducer_bug_seen = True
                    final = None
                else:
                    try:
                        final = await stream.get_final_response()
                    except (TypeError, AttributeError) as exc:
                        if not _is_codex_final_response_reducer_bug(exc):
                            raise
                        logger.warning(
                            "OpenAI Codex stream reducer failed in "
                            "get_final_response; falling back to "
                            "accumulated deltas (%d chars). Underlying "
                            "SDK bug: %s",
                            sum(len(p) for p in parts),
                            exc,
                            exc_info=True,
                        )
                        reducer_bug_seen = True
                        final = None

            # Trust the SDK's authoritative final payload when present,
            # even if its assembled text is empty — a model that emits a
            # legitimate empty-turn retraction must surface as ``text=""``,
            # not as whatever happened to stream in earlier (the previous
            # ``or "".join(parts)`` fallthrough fabricated content the
            # model authoritatively cleared). ``parts`` only carries the
            # done payload on the reducer-bug branch where ``final`` is
            # explicitly ``None``.
            #
            # EXCEPTION (issue #160): the ChatGPT codex backend ships a
            # terminal ``response.completed`` whose ``output`` is an EMPTY
            # LIST (``[]`` — zero items) even though valid
            # ``response.output_text.delta`` events already streamed the real
            # answer (reproduced live: 90 token events / 332 chars, usage
            # output_tokens=132, status "completed"). ``final`` is a well-
            # formed Response (NOT the ``output=None`` reducer bug, which
            # raises), so the code reached here and ``_extract_text_from_response``
            # returns "" from the empty list — discarding a complete
            # completion. When the final carries NO output items at all but
            # deltas DID arrive, the streamed ``parts`` are authoritative.
            # This is deliberately narrow: an explicit empty *message* item
            # (``output=[message(output_text="")]`` — a non-empty list, a real
            # cleared turn) keeps ``text=""`` because that output list is
            # truthy, so the authoritative-empty-retraction contract above
            # still holds.
            extracted = "" if final is None else _extract_text_from_response(final)
            if final is None:
                final_text = "".join(parts)
            elif extracted:
                final_text = extracted
            elif parts and not (getattr(final, "output", None) or []):
                final_text = "".join(parts)
            else:
                final_text = extracted

            # Total-loss safeguard: when the reducer bug fires before any
            # text delta has arrived, there is no partial response to
            # surface and ``final_text=""`` would slip through the engine
            # silently — synth (``domains/knowledge/synthesize.py``) reads
            # ``response.text`` only and treats empty text as
            # "model emitted zero pages", so an auth/refusal failure on
            # chatgpt.com/backend-api/codex would drop a knowledge page from
            # the source set on every reducer hit. Raise instead so the
            # failure surfaces on the NDJSON progress stream and the
            # caller can retry or skip with intent.
            if reducer_bug_seen and not parts:
                # TransientProviderError so synth's per-group retry-skip
                # (api.py group LLM retry loop) re-tries this — the
                # reducer bug is empirically transient (auth flap,
                # quota throttle, content-refusal that resolves on a
                # second attempt). Issue #134/#135 fix expected synth
                # to catch and retry this case.
                raise TransientProviderError(
                    "OpenAI codex backend returned response.output=None "
                    "and shipped zero text deltas; the SDK reducer "
                    "fallback has no partial response to surface. This "
                    "typically indicates an auth, quota, or content-"
                    "refusal failure on chatgpt.com/backend-api/codex — "
                    "check ``dikw auth status`` and the request payload."
                )

            status = str(getattr(final, "status", "") or "")
            # Reducer-bug fallback always reports ``"error"`` so callers
            # can distinguish a recovered partial response from a clean
            # completion — there is no authoritative ``status`` field in
            # that branch. Well-formed responses still go through the
            # status map (``failed`` / ``cancelled`` now also map to
            # ``"error"`` instead of being silently labeled ``"stop"``).
            finish_reason = (
                "error"
                if reducer_bug_seen
                else _FINISH_REASON_MAP.get(status, "stop")
            )
            usage = _extract_usage(final)
            yield LLMStreamEvent(
                type="done",
                text=final_text,
                finish_reason=finish_reason,
                usage=usage,
            )

        # Wrap in a gen_ai.chat span (gen_ai.system="openai"); token usage +
        # outbound httpx wire span nest under it. The generator body, including
        # the codex reducer-bug fallback, is unchanged.
        return trace_llm_stream(
            _gen(),
            system="openai",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
