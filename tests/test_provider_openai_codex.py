"""``OpenAICodexLLM.complete()`` and the SDK plumbing around it.

Mirrors the shape of ``test_provider_openai_compat_retries.py``:
monkeypatch ``openai.AsyncOpenAI`` with a stub that captures init
kwargs + ``responses.create`` calls so we never touch the network.
The ``captured`` / ``stream_captured`` fixtures also monkeypatch
``codex_auth.resolve_access_token`` so the auth path doesn't reach
``~/.codex/auth.json`` — tests that need a specific token shape
(JWT vs plain string) just mutate ``captured['access_token']`` before
the call.
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any

import pytest

from dikw_core.providers.base import LLMResponse
from dikw_core.providers.codex_auth import DEFAULT_CODEX_BASE_URL
from dikw_core.providers.openai_codex import OpenAICodexLLM


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_jwt(claims: dict[str, Any]) -> str:
    header = _b64url(json.dumps({"alg": "none"}).encode("utf-8"))
    payload = _b64url(json.dumps(claims).encode("utf-8"))
    return f"{header}.{payload}.sig"


def _make_response(
    *,
    text: str = "hello",
    status: str = "completed",
    input_tokens: int = 5,
    output_tokens: int = 7,
    output: list[Any] | None = None,
) -> SimpleNamespace:
    """Build a SimpleNamespace shaped like ``openai.types.responses.Response``."""
    if output is None:
        output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ]
    return SimpleNamespace(
        output=output,
        status=status,
        usage=SimpleNamespace(
            input_tokens=input_tokens, output_tokens=output_tokens
        ),
    )


@pytest.fixture()
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "init_kwargs": None,
        "create_kwargs": None,
        "next_response": _make_response(),
        "close_calls": 0,
        "access_token": "test-token",
    }

    class FakeResponses:
        async def create(self, **kwargs: Any) -> Any:
            rec["create_kwargs"] = kwargs
            return rec["next_response"]

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            rec["init_kwargs"] = kwargs
            self.responses = FakeResponses()

        async def close(self) -> None:
            rec["close_calls"] += 1

    async def _fake_resolve() -> str:
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
    provider = OpenAICodexLLM(base_url=DEFAULT_CODEX_BASE_URL)
    await provider.complete(system="s", user="u", model="gpt-5.5")
    assert captured["init_kwargs"]["base_url"] == DEFAULT_CODEX_BASE_URL


async def test_complete_passes_access_token_as_api_key(
    captured: dict[str, Any],
) -> None:
    captured["access_token"] = "my-secret-token"
    provider = OpenAICodexLLM(base_url=DEFAULT_CODEX_BASE_URL)
    await provider.complete(system="s", user="u", model="gpt-5.5")
    assert captured["init_kwargs"]["api_key"] == "my-secret-token"


async def test_complete_passes_codex_cloudflare_headers(
    captured: dict[str, Any],
) -> None:
    provider = OpenAICodexLLM(base_url=DEFAULT_CODEX_BASE_URL)
    await provider.complete(system="s", user="u", model="gpt-5.5")
    headers = captured["init_kwargs"]["default_headers"]
    assert headers["originator"] == "codex_cli_rs"
    assert headers["User-Agent"].startswith("codex_cli_rs/")


async def test_complete_includes_account_id_when_token_is_jwt(
    captured: dict[str, Any],
) -> None:
    token = _make_jwt({"chatgpt_account_id": "acc-42", "exp": 9_999_999_999})
    captured["access_token"] = token
    provider = OpenAICodexLLM(base_url=DEFAULT_CODEX_BASE_URL)
    await provider.complete(system="s", user="u", model="gpt-5.5")
    headers = captured["init_kwargs"]["default_headers"]
    assert headers["ChatGPT-Account-ID"] == "acc-42"


async def test_complete_omits_account_id_for_non_jwt_token(
    captured: dict[str, Any],
) -> None:
    captured["access_token"] = "plain-not-jwt"
    provider = OpenAICodexLLM(base_url=DEFAULT_CODEX_BASE_URL)
    await provider.complete(system="s", user="u", model="gpt-5.5")
    headers = captured["init_kwargs"]["default_headers"]
    assert "ChatGPT-Account-ID" not in headers


# --------------------------------------------------------------------------- #
# responses.create kwargs shape
# --------------------------------------------------------------------------- #


async def test_complete_calls_responses_create_with_responses_api_shape(
    captured: dict[str, Any],
) -> None:
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL
    )
    await provider.complete(
        system="be helpful",
        user="hello world",
        model="gpt-5.5",
        max_tokens=512,
        temperature=0.4,
    )
    kwargs = captured["create_kwargs"]
    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["instructions"] == "be helpful"
    assert kwargs["store"] is False
    assert kwargs["max_output_tokens"] == 512
    assert kwargs["temperature"] == 0.4
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
        base_url=DEFAULT_CODEX_BASE_URL
    )
    await provider.complete(system="s", user="u", model="gpt-5.5")
    assert "messages" not in captured["create_kwargs"]


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


async def test_complete_returns_text_from_output_messages(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = _make_response(text="hello world")
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL
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
        base_url=DEFAULT_CODEX_BASE_URL
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
        base_url=DEFAULT_CODEX_BASE_URL
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.text == "answer"


async def test_complete_maps_status_completed_to_stop(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = _make_response(status="completed")
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.finish_reason == "stop"


async def test_complete_maps_status_incomplete_to_length(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = _make_response(status="incomplete")
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.finish_reason == "length"


async def test_complete_extracts_usage_input_output_tokens(
    captured: dict[str, Any],
) -> None:
    captured["next_response"] = _make_response(input_tokens=42, output_tokens=99)
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL
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
        base_url=DEFAULT_CODEX_BASE_URL
    )
    resp = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert resp.usage == {}


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


class _StubStream:
    """Mimics the async context manager openai SDK returns from
    ``responses.stream(...)``: yields events, then exposes a
    ``get_final_response()`` for the terminal payload."""

    def __init__(
        self, events: list[Any], *, final: SimpleNamespace
    ) -> None:
        self._events = events
        self._final = final

    async def __aenter__(self) -> _StubStream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def __aiter__(self) -> _StubStream:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None

    def get_final_response(self) -> SimpleNamespace:
        return self._final


@pytest.fixture()
def stream_captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "init_kwargs": None,
        "stream_kwargs": None,
        "events": [],
        "final": _make_response(text="full text", input_tokens=3, output_tokens=4),
        "access_token": "test-token",
    }

    class FakeResponses:
        def stream(self, **kwargs: Any) -> _StubStream:
            rec["stream_kwargs"] = kwargs
            return _StubStream(rec["events"], final=rec["final"])

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            rec["init_kwargs"] = kwargs
            self.responses = FakeResponses()

        async def close(self) -> None:
            return None

    async def _fake_resolve() -> str:
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


from dikw_core.providers.base import LLMStreamEvent  # noqa: E402


async def test_complete_stream_yields_token_for_output_text_delta(
    stream_captured: dict[str, Any],
) -> None:
    stream_captured["events"] = [
        SimpleNamespace(type="response.output_text.delta", delta="hel"),
        SimpleNamespace(type="response.output_text.delta", delta="lo"),
    ]
    stream_captured["final"] = _make_response(text="hello")
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL
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
    stream_captured["final"] = _make_response(text="answer")
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL
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
    stream_captured["final"] = _make_response(
        text="hello", status="completed", input_tokens=3, output_tokens=4
    )
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL
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
        base_url=DEFAULT_CODEX_BASE_URL
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
        base_url=DEFAULT_CODEX_BASE_URL
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
        base_url=DEFAULT_CODEX_BASE_URL
    )
    events = await _drain(provider, system="s", user="u", model="gpt-5.5")
    tokens = [e for e in events if e.type == "token"]
    assert [e.delta for e in tokens] == ["real"]


async def test_complete_stream_passes_responses_api_shape(
    stream_captured: dict[str, Any],
) -> None:
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL
    )
    await _drain(
        provider, system="be helpful", user="hello", model="gpt-5.5", max_tokens=128
    )
    kw = stream_captured["stream_kwargs"]
    assert kw["model"] == "gpt-5.5"
    assert kw["instructions"] == "be helpful"
    assert kw["store"] is False
    assert kw["max_output_tokens"] == 128
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
        base_url=DEFAULT_CODEX_BASE_URL
    )
    await _drain(provider, system="s", user="u", model="gpt-5.5")
    headers = stream_captured["init_kwargs"]["default_headers"]
    assert headers["originator"] == "codex_cli_rs"
