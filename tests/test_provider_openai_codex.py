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

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dikw_core.providers.base import LLMResponse, LLMStreamEvent
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
