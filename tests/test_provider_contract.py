"""LLMProvider behavioral contract suite.

Parametrised over the three concrete providers (anthropic_compat,
openai_compat, openai_codex) so engine code can rely on ``LLMProvider``
making the same promises regardless of the wire protocol underneath.
Each backend has its own SDK quirks (Responses API vs chat.completions
vs Anthropic Messages, streaming vs non-streaming, JSON vs SSE); this
file pins the boundary at the Protocol so a provider that drifts from
the contract fails CI before engine code does.

Each harness fakes its provider's SDK at the call boundary and exposes
a tiny scripting API (``arrange_complete`` / ``arrange_stream``); the
contract tests below describe what every LLMProvider must deliver to
engine callers (``api.synth``, ``api.query``, ``check_providers``, …).
Adding a fourth provider means writing one more harness — no new test
cases.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import pytest

from dikw_core.providers.anthropic_compat import AnthropicCompatLLM
from dikw_core.providers.base import LLMProvider, LLMResponse
from dikw_core.providers.codex_auth import DEFAULT_CODEX_BASE_URL
from dikw_core.providers.openai_codex import _FINISH_REASON_MAP, OpenAICodexLLM
from dikw_core.providers.openai_compat import OpenAICompatLLM

from .fakes import (
    CodexResponsesStreamStub,
    assert_codex_request_kwargs_clean,
    codex_create_sentinel,
    make_codex_response,
)

# --------------------------------------------------------------------------- #
# Scripting datatypes — what every harness must be able to deliver.
# --------------------------------------------------------------------------- #


@dataclass
class _CompleteScript:
    text: str = "hi"
    finish_reason: str = "stop"
    input_tokens: int = 5
    output_tokens: int = 7


@dataclass
class _StreamScript:
    deltas: list[str] = field(default_factory=lambda: ["he", "llo"])
    # Empty default → __post_init__ fills it from joined deltas; tests
    # pass it explicitly only when the SDK's reported final differs from
    # concatenated tokens (rare).
    final_text: str = ""
    finish_reason: str = "stop"
    input_tokens: int = 3
    output_tokens: int = 4

    def __post_init__(self) -> None:
        if not self.final_text:
            self.final_text = "".join(self.deltas)


class _Harness(Protocol):
    """Test-side adapter: fakes a provider's SDK at the call boundary
    and lets contract tests script the response without knowing the
    wire format."""

    def make(self) -> LLMProvider: ...
    def arrange_complete(self, script: _CompleteScript) -> None: ...
    def arrange_stream(self, script: _StreamScript) -> None: ...


# --------------------------------------------------------------------------- #
# openai_compat harness — chat.completions.create with stream=True/False
# --------------------------------------------------------------------------- #


class _OpenAICompatHarness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._complete = _CompleteScript()
        self._stream = _StreamScript()
        harness = self

        class _FakeAsyncStream:
            def __init__(self, chunks: list[Any]) -> None:
                self._chunks = list(chunks)

            def __aiter__(self) -> _FakeAsyncStream:
                return self

            async def __anext__(self) -> Any:
                if not self._chunks:
                    raise StopAsyncIteration
                return self._chunks.pop(0)

            async def aclose(self) -> None:
                return None

        def _delta_chunk(text: str) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=text),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )

        def _finish_chunk(finish_reason: str) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None),
                        finish_reason=finish_reason,
                    )
                ],
                usage=None,
            )

        def _usage_chunk(input_tokens: int, output_tokens: int) -> SimpleNamespace:
            # OpenAI-style streams emit a final chunk with empty choices and
            # populated usage when stream_options={"include_usage": True}.
            return SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                ),
            )

        class _FakeCompletions:
            async def create(self, **kwargs: Any) -> Any:
                if kwargs.get("stream"):
                    s = harness._stream
                    chunks: list[Any] = [_delta_chunk(d) for d in s.deltas]
                    chunks.append(_finish_chunk(s.finish_reason))
                    chunks.append(_usage_chunk(s.input_tokens, s.output_tokens))
                    return _FakeAsyncStream(chunks)
                c = harness._complete
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=c.text),
                            finish_reason=c.finish_reason,
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=c.input_tokens,
                        completion_tokens=c.output_tokens,
                    ),
                )

        class _FakeAsyncOpenAI:
            def __init__(self, **_kwargs: Any) -> None:
                self.chat = SimpleNamespace(completions=_FakeCompletions())
                self.embeddings = SimpleNamespace()

        monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    def make(self) -> LLMProvider:
        return OpenAICompatLLM(base_url="http://fake.example/v1")

    def arrange_complete(self, script: _CompleteScript) -> None:
        self._complete = script

    def arrange_stream(self, script: _StreamScript) -> None:
        self._stream = script


# --------------------------------------------------------------------------- #
# anthropic_compat harness — messages.create + messages.stream
# --------------------------------------------------------------------------- #


class _AnthropicHarness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._complete = _CompleteScript()
        self._stream = _StreamScript()
        harness = self

        class _FakeMessageStream:
            def __init__(self, deltas: list[str], final: SimpleNamespace) -> None:
                self._deltas = deltas
                self._final = final

            async def __aenter__(self) -> _FakeMessageStream:
                return self

            async def __aexit__(self, *_: Any) -> None:
                return None

            @property
            def text_stream(self) -> AsyncIterator[str]:
                async def _gen() -> AsyncIterator[str]:
                    for d in self._deltas:
                        yield d

                return _gen()

            async def get_final_message(self) -> SimpleNamespace:
                return self._final

        def _make_final(
            text: str, finish_reason: str, input_tokens: int, output_tokens: int
        ) -> SimpleNamespace:
            return SimpleNamespace(
                content=[SimpleNamespace(text=text)],
                usage=SimpleNamespace(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                ),
                stop_reason=finish_reason,
            )

        class _FakeMessages:
            async def create(self, **_kwargs: Any) -> Any:
                c = harness._complete
                return _make_final(
                    c.text, c.finish_reason, c.input_tokens, c.output_tokens
                )

            def stream(self, **_kwargs: Any) -> _FakeMessageStream:
                s = harness._stream
                final = _make_final(
                    s.final_text, s.finish_reason, s.input_tokens, s.output_tokens
                )
                return _FakeMessageStream(list(s.deltas), final)

        class _FakeAsyncAnthropic:
            def __init__(self, **_kwargs: Any) -> None:
                self.messages = _FakeMessages()

        monkeypatch.setattr("anthropic.AsyncAnthropic", _FakeAsyncAnthropic)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    def make(self) -> LLMProvider:
        return AnthropicCompatLLM()

    def arrange_complete(self, script: _CompleteScript) -> None:
        self._complete = script

    def arrange_stream(self, script: _StreamScript) -> None:
        self._stream = script


# --------------------------------------------------------------------------- #
# openai_codex harness — Responses API stream-only path
# --------------------------------------------------------------------------- #


# Reverse prod's status→finish_reason map, picking the first status that
# resolves to each finish_reason as the canonical test value (e.g. "stop"
# → "completed"). Auto-syncs with the production map: if an SDK status
# string is renamed, the test stays correct or fails loudly at import.
_CODEX_FINISH_TO_STATUS: dict[str, str] = {}
for _status, _reason in _FINISH_REASON_MAP.items():
    _CODEX_FINISH_TO_STATUS.setdefault(_reason, _status)
del _status, _reason


class _CodexHarness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._complete = _CompleteScript()
        self._stream: _StreamScript | None = None
        harness = self

        def _final_for_script(
            text: str, finish_reason: str, input_tokens: int, output_tokens: int
        ) -> SimpleNamespace:
            return make_codex_response(
                text=text,
                status=_CODEX_FINISH_TO_STATUS.get(finish_reason, "completed"),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        class _FakeResponses:
            def stream(self, **kwargs: Any) -> CodexResponsesStreamStub:
                assert_codex_request_kwargs_clean(kwargs)
                if harness._stream is not None:
                    s = harness._stream
                    events: list[Any] = [
                        SimpleNamespace(type="response.output_text.delta", delta=d)
                        for d in s.deltas
                    ]
                    final = _final_for_script(
                        s.final_text,
                        s.finish_reason,
                        s.input_tokens,
                        s.output_tokens,
                    )
                else:
                    c = harness._complete
                    events = []
                    final = _final_for_script(
                        c.text, c.finish_reason, c.input_tokens, c.output_tokens
                    )
                return CodexResponsesStreamStub(events, final=final)

            create = codex_create_sentinel

        class _FakeAsyncOpenAI:
            def __init__(self, **_kwargs: Any) -> None:
                self.responses = _FakeResponses()

            async def close(self) -> None:
                return None

        async def _fake_resolve(_base: Path, **_kwargs: Any) -> str:
            return "test-token"

        monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)
        monkeypatch.setattr(
            "dikw_core.providers.openai_codex.resolve_access_token", _fake_resolve
        )

    def make(self) -> LLMProvider:
        return OpenAICodexLLM(
            base_url=DEFAULT_CODEX_BASE_URL, wiki_base=Path("dummy-wiki")
        )

    def arrange_complete(self, script: _CompleteScript) -> None:
        self._complete = script

    def arrange_stream(self, script: _StreamScript) -> None:
        self._stream = script


# --------------------------------------------------------------------------- #
# Parametrised fixture
# --------------------------------------------------------------------------- #


@pytest.fixture(
    params=[
        pytest.param("openai_compat", id="openai_compat"),
        pytest.param("anthropic_compat", id="anthropic_compat"),
        pytest.param("openai_codex", id="openai_codex"),
    ]
)
def harness(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> _Harness:
    if request.param == "openai_compat":
        return _OpenAICompatHarness(monkeypatch)
    if request.param == "anthropic_compat":
        return _AnthropicHarness(monkeypatch)
    if request.param == "openai_codex":
        return _CodexHarness(monkeypatch)
    raise RuntimeError(f"unreachable: harness {request.param}")


# --------------------------------------------------------------------------- #
# Contract: complete()
# --------------------------------------------------------------------------- #


async def test_complete_returns_llm_response_with_text(
    harness: _Harness,
) -> None:
    harness.arrange_complete(_CompleteScript(text="hello"))
    provider = harness.make()
    resp = await provider.complete(system="s", user="u", model="m")
    assert isinstance(resp, LLMResponse)
    assert resp.text == "hello"


async def test_complete_returns_finish_reason(harness: _Harness) -> None:
    harness.arrange_complete(_CompleteScript(finish_reason="stop"))
    provider = harness.make()
    resp = await provider.complete(system="s", user="u", model="m")
    assert resp.finish_reason == "stop"


async def test_complete_reports_input_output_token_usage(
    harness: _Harness,
) -> None:
    harness.arrange_complete(_CompleteScript(input_tokens=11, output_tokens=22))
    provider = harness.make()
    resp = await provider.complete(system="s", user="u", model="m")
    assert resp.usage["input_tokens"] == 11
    assert resp.usage["output_tokens"] == 22


# --------------------------------------------------------------------------- #
# Contract: complete_stream()
# --------------------------------------------------------------------------- #


async def _drain(provider: LLMProvider) -> list[Any]:
    events: list[Any] = []
    async for ev in provider.complete_stream(system="s", user="u", model="m"):
        events.append(ev)
    return events


async def test_stream_emits_token_event_per_delta(harness: _Harness) -> None:
    harness.arrange_stream(_StreamScript(deltas=["he", "llo"]))
    provider = harness.make()
    events = await _drain(provider)
    tokens = [e for e in events if e.type == "token"]
    assert [e.delta for e in tokens] == ["he", "llo"]


async def test_stream_terminates_with_exactly_one_done_event(
    harness: _Harness,
) -> None:
    harness.arrange_stream(_StreamScript())
    provider = harness.make()
    events = await _drain(provider)
    done = [e for e in events if e.type == "done"]
    assert len(done) == 1
    assert events[-1].type == "done"


async def test_stream_done_event_carries_assembled_text(
    harness: _Harness,
) -> None:
    harness.arrange_stream(_StreamScript(deltas=["he", "llo"]))
    provider = harness.make()
    events = await _drain(provider)
    assert events[-1].text == "hello"


async def test_stream_done_event_carries_finish_reason(
    harness: _Harness,
) -> None:
    harness.arrange_stream(_StreamScript(finish_reason="stop"))
    provider = harness.make()
    events = await _drain(provider)
    assert events[-1].finish_reason == "stop"


async def test_stream_done_event_carries_usage(harness: _Harness) -> None:
    harness.arrange_stream(_StreamScript(input_tokens=11, output_tokens=22))
    provider = harness.make()
    events = await _drain(provider)
    done = events[-1]
    assert done.usage["input_tokens"] == 11
    assert done.usage["output_tokens"] == 22
