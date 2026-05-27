"""``OpenAICodexLLM.complete()`` and the SDK plumbing around it.

ChatGPT's codex backend rejects non-streaming Responses calls with
``Stream must be set to true``, so ``complete()`` is implemented as a
collapse of ``complete_stream()``. Both fixtures consequently fake
``responses.stream`` (not ``responses.create``); ``captured`` runs an
empty event list so ``complete()`` reads its ``LLMResponse`` straight
from the ``final`` payload, while ``stream_captured`` lets streaming
tests script real delta events.

The fixtures monkeypatch ``codex_auth.resolve_access_token`` so the
auth path doesn't reach ``~/.codex/auth.json`` — tests that need a
specific token shape (JWT vs plain string) just mutate
``captured['access_token']`` before the call.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dikw_core.providers.base import LLMResponse, LLMStreamEvent, ProviderError
from dikw_core.providers.codex_auth import DEFAULT_CODEX_BASE_URL
from dikw_core.providers.openai_codex import OpenAICodexLLM

from .fakes import (
    CodexResponsesStreamStub,
    assert_codex_request_kwargs_clean,
    codex_create_sentinel,
    make_codex_response,
)
from .fakes import make_jwt as _make_jwt

# All tests in this module monkeypatch ``resolve_access_token`` so the
# wiki_base argument never round-trips to the file system. A single
# dummy Path keeps construction noise out of every test body.
_DUMMY_BASE = Path("dummy-wiki")


@pytest.fixture()
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "init_kwargs": None,
        "stream_kwargs": None,
        "next_response": make_codex_response(
            text="hello", input_tokens=5, output_tokens=7
        ),
        "close_calls": 0,
        "access_token": "test-token",
    }

    class FakeResponses:
        def stream(self, **kwargs: Any) -> CodexResponsesStreamStub:
            assert_codex_request_kwargs_clean(kwargs)
            rec["stream_kwargs"] = kwargs
            return CodexResponsesStreamStub([], final=rec["next_response"])

        create = codex_create_sentinel

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            rec["init_kwargs"] = kwargs
            self.responses = FakeResponses()

        async def close(self) -> None:
            rec["close_calls"] += 1

    async def _fake_resolve(_base: Path, **_kwargs: Any) -> str:
        return rec["access_token"]

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(
        "dikw_core.providers.openai_codex.resolve_access_token", _fake_resolve
    )
    return rec


# --------------------------------------------------------------------------- #
# Construction + auth header injection
# --------------------------------------------------------------------------- #


async def test_complete_passes_explicit_base_url(captured: dict[str, Any]) -> None:
    provider = OpenAICodexLLM(base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE)
    await provider.complete(system="s", user="u", model="gpt-5.5")
    assert captured["init_kwargs"]["base_url"] == DEFAULT_CODEX_BASE_URL


async def test_complete_passes_access_token_as_api_key(
    captured: dict[str, Any],
) -> None:
    captured["access_token"] = "my-secret-token"
    provider = OpenAICodexLLM(base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE)
    await provider.complete(system="s", user="u", model="gpt-5.5")
    assert captured["init_kwargs"]["api_key"] == "my-secret-token"


async def test_complete_passes_codex_cloudflare_headers(
    captured: dict[str, Any],
) -> None:
    provider = OpenAICodexLLM(base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE)
    await provider.complete(system="s", user="u", model="gpt-5.5")
    headers = captured["init_kwargs"]["default_headers"]
    assert headers["originator"] == "codex_cli_rs"
    assert headers["User-Agent"].startswith("codex_cli_rs/")


async def test_complete_includes_account_id_when_token_is_jwt(
    captured: dict[str, Any],
) -> None:
    token = _make_jwt({"chatgpt_account_id": "acc-42", "exp": 9_999_999_999})
    captured["access_token"] = token
    provider = OpenAICodexLLM(base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE)
    await provider.complete(system="s", user="u", model="gpt-5.5")
    headers = captured["init_kwargs"]["default_headers"]
    assert headers["ChatGPT-Account-ID"] == "acc-42"


async def test_complete_omits_account_id_for_non_jwt_token(
    captured: dict[str, Any],
) -> None:
    captured["access_token"] = "plain-not-jwt"
    provider = OpenAICodexLLM(base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE)
    await provider.complete(system="s", user="u", model="gpt-5.5")
    headers = captured["init_kwargs"]["default_headers"]
    assert "ChatGPT-Account-ID" not in headers


# --------------------------------------------------------------------------- #
# responses.create kwargs shape
# --------------------------------------------------------------------------- #


async def test_complete_calls_responses_stream_with_responses_api_shape(
    captured: dict[str, Any],
) -> None:
    """``complete()`` collapses ``complete_stream()``, so the SDK call
    underneath is ``responses.stream`` — the codex backend rejects the
    non-streaming variant with ``Stream must be set to true``."""
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    await provider.complete(
        system="be helpful",
        user="hello world",
        model="gpt-5.5",
        max_tokens=512,
        temperature=0.4,
    )
    kwargs = captured["stream_kwargs"]
    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["instructions"] == "be helpful"
    assert kwargs["store"] is False
    # Input is the Responses API shape — list of items with content parts.
    assert kwargs["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hello world"}],
        }
    ]


async def test_complete_does_not_pass_messages_kwarg(
    captured: dict[str, Any],
) -> None:
    """Regression: Responses API uses `instructions` + `input`, NOT `messages`."""
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    await provider.complete(system="s", user="u", model="gpt-5.5")
    assert "messages" not in captured["stream_kwargs"]


async def test_complete_does_not_pass_max_output_tokens_kwarg(
    captured: dict[str, Any],
) -> None:
    """Regression: ChatGPT codex backend rejects ``max_output_tokens``
    with a 400 ``Unsupported parameter``. ``max_tokens`` stays in the
    LLMProvider signature (other providers honor it) but never reaches
    the codex wire payload. The fixture's
    ``assert_codex_request_kwargs_clean`` makes every test in this
    module a passive regression for the same invariant."""
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    await provider.complete(
        system="s", user="u", model="gpt-5.5", max_tokens=512
    )
    assert "max_output_tokens" not in captured["stream_kwargs"]


async def test_complete_does_not_pass_temperature_kwarg(
    captured: dict[str, Any],
) -> None:
    """Regression: ChatGPT codex backend rejects ``temperature`` with a
    400 ``Unsupported parameter`` — sampling is managed server-side.
    The provider accepts the kwarg for protocol parity but drops it
    from the wire payload."""
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    await provider.complete(
        system="s", user="u", model="gpt-5.5", temperature=0.7
    )
    assert "temperature" not in captured["stream_kwargs"]


async def test_complete_uses_streaming_responses_endpoint_only(
    captured: dict[str, Any],
) -> None:
    """``complete()`` must reach the model via ``responses.stream`` —
    codex rejects non-streaming Responses. The fixture's
    ``responses.create`` is a ``codex_create_sentinel`` that fails on
    invocation, so this test passes only if the streaming path was
    taken."""
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    await provider.complete(system="s", user="u", model="gpt-5.5")
    assert captured["stream_kwargs"] is not None


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


async def test_complete_returns_text_from_output_messages(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = make_codex_response(text="hello world")
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert isinstance(resp, LLMResponse)
    assert resp.text == "hello world"


async def test_complete_concatenates_multiple_output_text_parts(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text="hello "),
                    SimpleNamespace(type="output_text", text="world"),
                ],
            )
        ],
        status="completed",
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.text == "hello world"


async def test_complete_skips_non_message_output_items(
    captured: dict[str, Any],
) -> None:
    """reasoning items and tool_call items must not bleed into ``text``."""
    captured["next_response"] = SimpleNamespace(
        output=[
            SimpleNamespace(type="reasoning", summary="thought"),
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="answer")],
            ),
        ],
        status="completed",
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.text == "answer"


async def test_complete_maps_status_completed_to_stop(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = make_codex_response(status="completed")
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.finish_reason == "stop"


async def test_complete_maps_status_incomplete_to_length(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = make_codex_response(status="incomplete")
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.finish_reason == "length"


async def test_complete_maps_status_failed_to_error(
    captured: dict[str, Any],
) -> None:
    """Backend-side ``failed`` status must surface as ``finish_reason
    = "error"`` (not ``"stop"``) so a caller can distinguish a
    backend-rejected response from a clean completion. The old map
    silently labeled both ``"stop"``, hiding hung-up calls."""
    captured["next_response"] = make_codex_response(status="failed")
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.finish_reason == "error"


async def test_complete_maps_status_cancelled_to_error(
    captured: dict[str, Any],
) -> None:
    """Mirror of ``failed`` mapping for explicit cancellation."""
    captured["next_response"] = make_codex_response(status="cancelled")
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.finish_reason == "error"


async def test_complete_extracts_usage_input_output_tokens(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = make_codex_response(input_tokens=42, output_tokens=99)
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.usage == {"input_tokens": 42, "output_tokens": 99}


async def test_complete_handles_response_without_usage(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="x")],
            )
        ],
        status="completed",
        usage=None,
    )
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.usage == {}


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


@pytest.fixture()
def stream_captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "init_kwargs": None,
        "stream_kwargs": None,
        "events": [],
        "final": make_codex_response(text="full text", input_tokens=3, output_tokens=4),
        "access_token": "test-token",
    }

    class FakeResponses:
        def stream(self, **kwargs: Any) -> CodexResponsesStreamStub:
            assert_codex_request_kwargs_clean(kwargs)
            rec["stream_kwargs"] = kwargs
            return CodexResponsesStreamStub(rec["events"], final=rec["final"])

        create = codex_create_sentinel

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            rec["init_kwargs"] = kwargs
            self.responses = FakeResponses()

        async def close(self) -> None:
            return None

    async def _fake_resolve(_base: Path, **_kwargs: Any) -> str:
        return rec["access_token"]

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(
        "dikw_core.providers.openai_codex.resolve_access_token", _fake_resolve
    )
    return rec


async def _drain(provider: OpenAICodexLLM, **kwargs: Any) -> list[LLMStreamEvent]:
    events: list[LLMStreamEvent] = []
    async for ev in provider.complete_stream(**kwargs):
        events.append(ev)
    return events


async def test_complete_stream_yields_token_for_output_text_delta(
    stream_captured: dict[str, Any],
) -> None:
    stream_captured["events"] = [
        SimpleNamespace(type="response.output_text.delta", delta="hel"),
        SimpleNamespace(type="response.output_text.delta", delta="lo"),
    ]
    stream_captured["final"] = make_codex_response(text="hello")
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    events = await _drain(provider, system="s", user="u", model="gpt-5.5")
    tokens = [e for e in events if e.type == "token"]
    assert [e.delta for e in tokens] == ["hel", "lo"]


async def test_complete_stream_yields_reasoning_for_summary_delta(
    stream_captured: dict[str, Any],
) -> None:
    stream_captured["events"] = [
        SimpleNamespace(
            type="response.reasoning_summary_text.delta", delta="thinking…"
        ),
        SimpleNamespace(type="response.output_text.delta", delta="answer"),
    ]
    stream_captured["final"] = make_codex_response(text="answer")
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    events = await _drain(provider, system="s", user="u", model="gpt-5.5")
    reasoning = [e for e in events if e.type == "reasoning"]
    tokens = [e for e in events if e.type == "token"]
    assert [e.delta for e in reasoning] == ["thinking…"]
    assert [e.delta for e in tokens] == ["answer"]


async def test_complete_stream_yields_done_with_assembled_text(
    stream_captured: dict[str, Any],
) -> None:
    stream_captured["events"] = [
        SimpleNamespace(type="response.output_text.delta", delta="hel"),
        SimpleNamespace(type="response.output_text.delta", delta="lo"),
    ]
    stream_captured["final"] = make_codex_response(
        text="hello", status="completed", input_tokens=3, output_tokens=4
    )
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    events = await _drain(provider, system="s", user="u", model="gpt-5.5")
    assert events[-1].type == "done"
    assert events[-1].text == "hello"
    assert events[-1].finish_reason == "stop"
    assert events[-1].usage == {"input_tokens": 3, "output_tokens": 4}


async def test_complete_stream_emits_exactly_one_done_event(
    stream_captured: dict[str, Any],
) -> None:
    stream_captured["events"] = [
        SimpleNamespace(type="response.output_text.delta", delta="x"),
    ]
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    events = await _drain(provider, system="s", user="u", model="gpt-5.5")
    done_events = [e for e in events if e.type == "done"]
    assert len(done_events) == 1


async def test_complete_stream_skips_unknown_event_types(
    stream_captured: dict[str, Any],
) -> None:
    stream_captured["events"] = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.output_item.added"),
        SimpleNamespace(type="response.output_text.delta", delta="x"),
        SimpleNamespace(type="response.completed"),
    ]
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    events = await _drain(provider, system="s", user="u", model="gpt-5.5")
    # Two events emitted: token + done. Unknown types are silently dropped.
    assert [e.type for e in events] == ["token", "done"]


async def test_complete_stream_skips_empty_deltas(
    stream_captured: dict[str, Any],
) -> None:
    stream_captured["events"] = [
        SimpleNamespace(type="response.output_text.delta", delta=""),
        SimpleNamespace(type="response.output_text.delta", delta=None),
        SimpleNamespace(type="response.output_text.delta", delta="real"),
    ]
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    events = await _drain(provider, system="s", user="u", model="gpt-5.5")
    tokens = [e for e in events if e.type == "token"]
    assert [e.delta for e in tokens] == ["real"]


async def test_complete_stream_passes_responses_api_shape(
    stream_captured: dict[str, Any],
) -> None:
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    await _drain(
        provider, system="be helpful", user="hello", model="gpt-5.5", max_tokens=128
    )
    kw = stream_captured["stream_kwargs"]
    assert kw["model"] == "gpt-5.5"
    assert kw["instructions"] == "be helpful"
    assert kw["store"] is False
    assert "max_output_tokens" not in kw
    assert kw["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]


async def test_complete_stream_injects_codex_headers(
    stream_captured: dict[str, Any],
) -> None:
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    await _drain(provider, system="s", user="u", model="gpt-5.5")
    headers = stream_captured["init_kwargs"]["default_headers"]
    assert headers["originator"] == "codex_cli_rs"


# --------------------------------------------------------------------------- #
# Resilience against openai-SDK reducer failures on ChatGPT codex backend
#
# The codex backend ships ``response.output = None`` in its
# ``response.completed`` payload (the public Responses API ships a list);
# the openai SDK's high-level streaming context manager runs an internal
# reducer that iterates ``response.output`` to assemble the final typed
# ``Response``. With ``output`` being None the SDK raises
# ``TypeError: 'NoneType' object is not iterable`` either from the
# terminal ``async for`` event (reducer-before-yield) or from
# ``stream.get_final_response()`` (reducer-at-terminator). Either way,
# the provider must fall back to the locally accumulated delta text and
# still emit exactly one ``done`` event instead of propagating TypeError
# to the engine. ``dikw client check --llm-only`` is enough to trigger
# this in production, so the bug surfaces before any K-layer work.
# --------------------------------------------------------------------------- #


class _StreamRaisesOnFinal:
    """Iterates events cleanly; ``get_final_response`` raises TypeError.

    Mimics the SDK code path where the reducer is deferred until the
    consumer calls ``stream.get_final_response()``.
    """

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> _StreamRaisesOnFinal:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def __aiter__(self) -> _StreamRaisesOnFinal:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None

    async def get_final_response(self) -> Any:
        raise TypeError("'NoneType' object is not iterable")


class _StreamRaisesDuringIteration:
    """Yields scripted delta events, then raises TypeError from
    ``__anext__`` instead of stopping cleanly.

    Mimics the SDK code path where the reducer runs *before* the
    terminator event is yielded — so the failure surfaces inside the
    consumer's ``async for`` loop, not from ``get_final_response``.
    """

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> _StreamRaisesDuringIteration:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def __aiter__(self) -> _StreamRaisesDuringIteration:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise TypeError("'NoneType' object is not iterable") from None

    async def get_final_response(self) -> Any:
        raise AssertionError(
            "iteration aborted with TypeError; get_final_response should "
            "not be reached"
        )


@pytest.fixture()
def failing_stream(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stream-factory fixture for SDK-reducer failure modes.

    Each test populates ``rec['stream_factory']`` with a callable that
    returns the desired failing stub; the rest of the wiring (auth,
    AsyncOpenAI, close) is fake-only.
    """
    rec: dict[str, Any] = {
        "stream_factory": lambda kwargs: CodexResponsesStreamStub(
            [], final=make_codex_response()
        ),
        "access_token": "test-token",
    }

    class FakeResponses:
        def stream(self, **kwargs: Any) -> Any:
            assert_codex_request_kwargs_clean(kwargs)
            return rec["stream_factory"](kwargs)

        create = codex_create_sentinel

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.responses = FakeResponses()

        async def close(self) -> None:
            return None

    async def _fake_resolve(_base: Path, **_kwargs: Any) -> str:
        return rec["access_token"]

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(
        "dikw_core.providers.openai_codex.resolve_access_token", _fake_resolve
    )
    return rec


async def test_complete_stream_falls_back_when_get_final_response_raises(
    failing_stream: dict[str, Any],
) -> None:
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="hel"),
        SimpleNamespace(type="response.output_text.delta", delta="lo"),
    ]
    failing_stream["stream_factory"] = lambda _kw: _StreamRaisesOnFinal(events)

    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    out = await _drain(provider, system="s", user="u", model="gpt-5.5")

    tokens = [e for e in out if e.type == "token"]
    assert [e.delta for e in tokens] == ["hel", "lo"]
    done = [e for e in out if e.type == "done"]
    assert len(done) == 1
    assert done[0].text == "hello"
    # Reducer-bug fallback surfaces as ``finish_reason="error"`` so the
    # caller can distinguish a recovered partial response from a clean
    # completion — the SDK never delivered an authoritative ``status``.
    assert done[0].finish_reason == "error"
    assert done[0].usage == {}


async def test_complete_stream_falls_back_when_event_iteration_raises(
    failing_stream: dict[str, Any],
) -> None:
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="par"),
        SimpleNamespace(type="response.output_text.delta", delta="tial"),
    ]
    failing_stream["stream_factory"] = lambda _kw: _StreamRaisesDuringIteration(
        events
    )

    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    out = await _drain(provider, system="s", user="u", model="gpt-5.5")

    tokens = [e for e in out if e.type == "token"]
    assert [e.delta for e in tokens] == ["par", "tial"]
    done = [e for e in out if e.type == "done"]
    assert len(done) == 1
    assert done[0].text == "partial"
    assert done[0].finish_reason == "error"
    assert done[0].usage == {}


async def test_complete_collapses_stream_failure_into_llm_response(
    failing_stream: dict[str, Any],
) -> None:
    """``complete()`` collapses ``complete_stream()``; the same fallback
    must surface as a normal ``LLMResponse`` instead of a propagated
    TypeError, otherwise ``dikw client check --llm-only`` (and every
    synth call) still crashes."""
    events = [SimpleNamespace(type="response.output_text.delta", delta="ok")]
    failing_stream["stream_factory"] = lambda _kw: _StreamRaisesOnFinal(events)

    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")

    assert isinstance(resp, LLMResponse)
    assert resp.text == "ok"
    assert resp.finish_reason == "error"
    assert resp.usage == {}


# --------------------------------------------------------------------------- #
# Narrow-catch + propagation guarantees
#
# The reducer-bug fallback only fires for the specific openai-SDK
# signature (``TypeError("'NoneType' object is not iterable")`` /
# AttributeError referencing ``output``). The following failure modes
# must propagate so a real bug never masquerades as a successful done
# event:
#   - Unrelated TypeError / AttributeError raised inside the stream
#     iteration (covered by ``_StreamRaisesUnrelatedTypeError`` /
#     ``_StreamRaisesUnrelatedAttributeError`` below)
#   - Unrelated exceptions raised by the deferred ``get_final_response()``
#     branch (covered by ``_StreamGetFinalRaisesUnrelated`` below)
#   - Stream-context-manager (``__aenter__``) failures — network, 401,
#     DNS, timeout — handled by ``_StreamRaisesOnEnter`` below
# --------------------------------------------------------------------------- #


class _StreamRaisesUnrelatedTypeError:
    """Yields one delta, then raises a TypeError whose message does NOT
    match the SDK reducer signature — must propagate, not fall back."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> _StreamRaisesUnrelatedTypeError:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def __aiter__(self) -> _StreamRaisesUnrelatedTypeError:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise TypeError("totally unrelated bug in our code") from None

    async def get_final_response(self) -> Any:
        raise AssertionError("not reached")


class _StreamRaisesUnrelatedAttributeError:
    """Raises an AttributeError whose message references a field that
    CONTAINS ``output`` as a prefix (``output_index`` / ``output_text``)
    but is not the reducer's actual ``output`` field — must propagate,
    not be absorbed by a substring match. Pins the field-name boundary
    that the helper now enforces.
    """

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> _StreamRaisesUnrelatedAttributeError:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def __aiter__(self) -> _StreamRaisesUnrelatedAttributeError:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise AttributeError(
                "'ResponseStreamEvent' object has no attribute 'output_index'"
            ) from None

    async def get_final_response(self) -> Any:
        raise AssertionError("not reached")


class _StreamGetFinalRaisesUnrelated:
    """``async for`` completes cleanly; ``get_final_response()`` raises
    an unrelated TypeError / AttributeError. The reducer-bug catch must
    not absorb it — pins the symmetric narrow-catch invariant on the
    deferred-reducer branch."""

    def __init__(self, events: list[Any], exc: BaseException) -> None:
        self._events = events
        self._exc = exc

    async def __aenter__(self) -> _StreamGetFinalRaisesUnrelated:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def __aiter__(self) -> _StreamGetFinalRaisesUnrelated:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None

    async def get_final_response(self) -> Any:
        raise self._exc


class _StreamRaisesOnEnter:
    """``__aenter__`` raises — mimics a real network / auth failure that
    surfaces while the stream context is being established."""

    async def __aenter__(self) -> Any:
        raise RuntimeError("stream open failed (e.g. 401, dns, timeout)")

    async def __aexit__(self, *_: Any) -> None:
        return None


async def test_complete_stream_propagates_unrelated_type_error(
    failing_stream: dict[str, Any],
) -> None:
    """Narrow-catch guard: a TypeError whose message is not the SDK
    reducer signature must propagate, not be absorbed as a fake-success
    done event. Otherwise a real ``None.attr`` slip in our own code path
    would silently surface as a partial-text completion."""
    events = [SimpleNamespace(type="response.output_text.delta", delta="x")]
    failing_stream["stream_factory"] = lambda _kw: _StreamRaisesUnrelatedTypeError(
        events
    )

    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    with pytest.raises(TypeError, match="totally unrelated bug"):
        await _drain(provider, system="s", user="u", model="gpt-5.5")


async def test_complete_stream_propagates_stream_enter_failure(
    failing_stream: dict[str, Any],
) -> None:
    """Stream-establishment errors (auth, network, timeout) live on
    ``__aenter__`` and must NOT be swallowed by the reducer-bug catch —
    that try-block sits inside ``async with``, so the propagation path
    is structural. This test pins that invariant."""
    failing_stream["stream_factory"] = lambda _kw: _StreamRaisesOnEnter()

    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    with pytest.raises(RuntimeError, match="stream open failed"):
        await _drain(provider, system="s", user="u", model="gpt-5.5")


async def test_complete_propagates_stream_enter_failure(
    failing_stream: dict[str, Any],
) -> None:
    """Same propagation guarantee through the ``complete()`` collapse."""
    failing_stream["stream_factory"] = lambda _kw: _StreamRaisesOnEnter()

    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    with pytest.raises(RuntimeError, match="stream open failed"):
        await provider.complete(system="s", user="u", model="gpt-5.5")


async def test_complete_stream_propagates_unrelated_attribute_error(
    failing_stream: dict[str, Any],
) -> None:
    """AttributeError whose message names ``output_index`` (or any other
    field prefixed by ``output``) must propagate. Catching the bare
    substring ``"output"`` would silently absorb unrelated schema bugs
    into a fake-success done event; the helper now pins
    ``"attribute 'output'"`` to the exact field boundary."""
    events = [SimpleNamespace(type="response.output_text.delta", delta="x")]
    failing_stream[
        "stream_factory"
    ] = lambda _kw: _StreamRaisesUnrelatedAttributeError(events)

    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    with pytest.raises(AttributeError, match="output_index"):
        await _drain(provider, system="s", user="u", model="gpt-5.5")


async def test_complete_stream_propagates_unrelated_get_final_type_error(
    failing_stream: dict[str, Any],
) -> None:
    """Deferred-reducer branch (``get_final_response``) must enforce the
    same narrow catch as the iteration branch: a TypeError whose message
    does not match the SDK reducer signature propagates instead of being
    absorbed into the partial-text fallback."""
    events = [SimpleNamespace(type="response.output_text.delta", delta="ok")]
    failing_stream["stream_factory"] = lambda _kw: _StreamGetFinalRaisesUnrelated(
        events,
        TypeError("totally unrelated final-response bug"),
    )

    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    with pytest.raises(TypeError, match="totally unrelated final-response bug"):
        await _drain(provider, system="s", user="u", model="gpt-5.5")


async def test_complete_stream_propagates_unrelated_get_final_attribute_error(
    failing_stream: dict[str, Any],
) -> None:
    """Mirror of the above for AttributeError. Pins that the deferred
    branch also uses ``_is_codex_final_response_reducer_bug`` for
    narrow catch — an AttributeError naming a non-target field surfaces."""
    events = [SimpleNamespace(type="response.output_text.delta", delta="ok")]
    failing_stream["stream_factory"] = lambda _kw: _StreamGetFinalRaisesUnrelated(
        events,
        AttributeError("'ResponseStreamEvent' object has no attribute 'output_text'"),
    )

    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    with pytest.raises(AttributeError, match="output_text"):
        await _drain(provider, system="s", user="u", model="gpt-5.5")


async def test_reducer_bug_with_zero_deltas_raises_provider_error(
    failing_stream: dict[str, Any],
) -> None:
    """When the SDK reducer bug fires before a single delta arrives,
    the fallback has nothing to surface and ``final_text=""`` would
    pass through synth (which reads ``response.text`` only) as
    "model emitted zero pages" — a silent source-drop on every reducer
    hit. The provider must raise ``ProviderError`` instead so the
    failure shows up on the NDJSON progress stream and the caller can
    retry or skip with intent. Mirrors auth/quota/refusal failure modes
    on chatgpt.com/backend-api/codex."""
    failing_stream["stream_factory"] = lambda _kw: _StreamRaisesOnFinal(events=[])

    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    with pytest.raises(ProviderError, match="zero text deltas"):
        await _drain(provider, system="s", user="u", model="gpt-5.5")


async def test_complete_zero_delta_reducer_bug_raises_provider_error(
    failing_stream: dict[str, Any],
) -> None:
    """``complete()`` collapse path inherits the same total-loss
    safeguard — the synth call site uses ``complete()``, not
    ``complete_stream()`` directly, so this pin matters."""
    failing_stream["stream_factory"] = lambda _kw: _StreamRaisesOnFinal(events=[])

    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    with pytest.raises(ProviderError, match="zero text deltas"):
        await provider.complete(system="s", user="u", model="gpt-5.5")


async def test_authoritative_empty_final_does_not_fall_back_to_parts(
    stream_captured: dict[str, Any],
) -> None:
    """If the SDK delivers a clean ``response.completed`` with an
    explicitly empty assistant turn (e.g. model retraction, refusal
    routed through an empty message), the provider must report
    ``text=""`` — NOT whatever happened to stream in earlier. The old
    ``_extract_text_from_response(final) or "".join(parts)`` fabricated
    content the model had authoritatively cleared."""
    stream_captured["events"] = [
        SimpleNamespace(type="response.output_text.delta", delta="streamed"),
    ]
    stream_captured["final"] = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="")],
            )
        ],
        status="completed",
        usage=SimpleNamespace(input_tokens=1, output_tokens=0),
    )
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    events = await _drain(provider, system="s", user="u", model="gpt-5.5")
    done = [e for e in events if e.type == "done"]
    assert len(done) == 1
    # Authoritative empty turn — the streamed 'streamed' must not bleed
    # into the final text.
    assert done[0].text == ""
    assert done[0].finish_reason == "stop"


async def test_reducer_bug_fallback_warning_excludes_delta_text(
    failing_stream: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression guard: the fallback warning must log only the
    accumulated character count + the SDK exception, never the delta
    text itself (which can contain user content / model output). If a
    future edit widens the format string to include ``"".join(parts)``,
    this test pins it down."""
    secret = "supersecret-token-do-not-log-this"
    events = [SimpleNamespace(type="response.output_text.delta", delta=secret)]
    failing_stream["stream_factory"] = lambda _kw: _StreamRaisesOnFinal(events)

    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, wiki_base=_DUMMY_BASE
    )
    with caplog.at_level(
        logging.WARNING, logger="dikw_core.providers.openai_codex"
    ):
        await _drain(provider, system="s", user="u", model="gpt-5.5")

    rendered = "\n".join(
        rec.getMessage() + "\n" + (rec.exc_text or "")
        for rec in caplog.records
    )
    assert secret not in rendered
    # And the warning must actually fire — silent fallback would lose
    # observability for the very compatibility quirk we are tracking.
    assert any("reducer" in rec.message for rec in caplog.records)
