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

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import httpx
import pytest

from dikw_core import telemetry
from dikw_core.providers.anthropic_compat import AnthropicCompatLLM
from dikw_core.providers.base import (
    LLMProvider,
    LLMResponse,
    ProviderError,
    TransientProviderError,
)
from dikw_core.providers.codex_auth import DEFAULT_CODEX_BASE_URL
from dikw_core.providers.gitee_multimodal import GiteeMultimodalEmbedding
from dikw_core.providers.openai_codex import _FINISH_REASON_MAP, OpenAICodexLLM
from dikw_core.providers.openai_compat import OpenAICompatEmbeddings, OpenAICompatLLM
from dikw_core.schemas import MultimodalInput

from .fakes import (
    CodexResponsesStreamStub,
    anthropic_create_sentinel,
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
        # ``None`` → ``complete()`` (collapsed over the streaming create) reads
        # the ``_complete`` script as a single-shot stream, mirroring codex.
        self._stream: _StreamScript | None = None
        # Optional fault injection: "create" raises from the create() await
        # (connection-open phase); "iteration" raises mid-chunk-stream.
        self._raise_at: str | None = None
        self._exc_factory: Callable[[], BaseException] | None = None
        harness = self

        class _FakeAsyncStream:
            def __init__(self, chunks: list[Any]) -> None:
                self._chunks = list(chunks)
                self._raised = False

            def __aiter__(self) -> _FakeAsyncStream:
                return self

            async def __anext__(self) -> Any:
                if not self._chunks:
                    if (
                        harness._raise_at == "iteration"
                        and harness._exc_factory
                        and not self._raised
                    ):
                        self._raised = True
                        raise harness._exc_factory()
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

        def _build_chunks(deltas: list[str], s: Any) -> list[Any]:
            chunks: list[Any] = [_delta_chunk(d) for d in deltas]
            chunks.append(_finish_chunk(s.finish_reason))
            chunks.append(_usage_chunk(s.input_tokens, s.output_tokens))
            return chunks

        class _FakeCompletions:
            async def create(self, **kwargs: Any) -> Any:
                # ``complete`` must collapse the streaming path — a
                # non-streaming create() is the bug this PR removes.
                if not kwargs.get("stream"):
                    pytest.fail(
                        "OpenAICompatLLM.complete must call create(stream=True); "
                        "the non-streaming path trips the whole-response timeout."
                    )
                if harness._raise_at == "create" and harness._exc_factory:
                    raise harness._exc_factory()
                if harness._stream is not None:
                    s = harness._stream
                    return _FakeAsyncStream(_build_chunks(list(s.deltas), s))
                c = harness._complete
                deltas = [c.text] if c.text else []
                return _FakeAsyncStream(_build_chunks(deltas, c))

        class _FakeAsyncOpenAI:
            def __init__(self, **_kwargs: Any) -> None:
                self.chat = SimpleNamespace(completions=_FakeCompletions())
                self.embeddings = SimpleNamespace()

        monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    def make(self) -> LLMProvider:
        return OpenAICompatLLM(api_key_env="OPENAI_API_KEY", base_url="http://fake.example/v1")

    def arrange_complete(self, script: _CompleteScript) -> None:
        self._complete = script

    def arrange_stream(self, script: _StreamScript) -> None:
        self._stream = script

    def arrange_stream_raises(
        self, exc_factory: Callable[[], BaseException], *, at: str = "iteration"
    ) -> None:
        self._exc_factory = exc_factory
        self._raise_at = at


# --------------------------------------------------------------------------- #
# anthropic_compat harness — messages.create + messages.stream
# --------------------------------------------------------------------------- #


class _AnthropicHarness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._complete = _CompleteScript()
        # ``None`` → ``complete()`` (collapsed over ``messages.stream``) reads
        # the ``_complete`` script with no token deltas, mirroring _CodexHarness.
        self._stream: _StreamScript | None = None
        # Optional fault injection for the classification / cancel contract:
        # raise ``_exc_factory()`` at "aenter" | "text_stream" | "final".
        self._raise_at: str | None = None
        self._exc_factory: Callable[[], BaseException] | None = None
        harness = self

        class _FakeMessageStream:
            def __init__(self, deltas: list[str], final: SimpleNamespace) -> None:
                self._deltas = deltas
                self._final = final

            async def __aenter__(self) -> _FakeMessageStream:
                if harness._raise_at == "aenter" and harness._exc_factory:
                    raise harness._exc_factory()
                return self

            async def __aexit__(self, *_: Any) -> None:
                return None

            @property
            def text_stream(self) -> AsyncIterator[str]:
                async def _gen() -> AsyncIterator[str]:
                    for d in self._deltas:
                        yield d
                    if harness._raise_at == "text_stream" and harness._exc_factory:
                        raise harness._exc_factory()

                return _gen()

            async def get_final_message(self) -> SimpleNamespace:
                if harness._raise_at == "final" and harness._exc_factory:
                    raise harness._exc_factory()
                return self._final

        def _make_final(
            text: str, finish_reason: str, input_tokens: int, output_tokens: int
        ) -> SimpleNamespace:
            return SimpleNamespace(
                content=[SimpleNamespace(text=text)] if text else [],
                usage=SimpleNamespace(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                ),
                stop_reason=finish_reason,
            )

        class _FakeMessages:
            # ``complete`` must collapse ``messages.stream``; calling the
            # non-streaming ``create`` fails the test loudly.
            create = anthropic_create_sentinel

            def stream(self, **_kwargs: Any) -> _FakeMessageStream:
                if harness._stream is not None:
                    s = harness._stream
                    final = _make_final(
                        s.final_text, s.finish_reason, s.input_tokens, s.output_tokens
                    )
                    return _FakeMessageStream(list(s.deltas), final)
                c = harness._complete
                final = _make_final(
                    c.text, c.finish_reason, c.input_tokens, c.output_tokens
                )
                return _FakeMessageStream([], final)

        class _FakeAsyncAnthropic:
            def __init__(self, **_kwargs: Any) -> None:
                self.messages = _FakeMessages()

        monkeypatch.setattr("anthropic.AsyncAnthropic", _FakeAsyncAnthropic)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    def make(self) -> LLMProvider:
        return AnthropicCompatLLM(api_key_env="ANTHROPIC_API_KEY")

    def arrange_complete(self, script: _CompleteScript) -> None:
        self._complete = script

    def arrange_stream(self, script: _StreamScript) -> None:
        self._stream = script

    def arrange_stream_raises(
        self, exc_factory: Callable[[], BaseException], *, at: str = "text_stream"
    ) -> None:
        self._exc_factory = exc_factory
        self._raise_at = at


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
            base_url=DEFAULT_CODEX_BASE_URL, base_root=Path("dummy-wiki")
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


# --------------------------------------------------------------------------- #
# Contract: gen_ai.* tracing span (PR2 OTel arc)
#
# Every LLMProvider's call must emit ONE gen_ai.chat span carrying the model,
# the gen_ai.system, and the token usage off the done event — the operator-side
# observability the synth/query paths rely on. One case auto-covers all three
# providers via the parametrised harness.
# --------------------------------------------------------------------------- #


async def test_complete_emits_gen_ai_chat_span_with_usage(
    harness: _Harness, span_exporter: Any
) -> None:
    harness.arrange_complete(
        _CompleteScript(finish_reason="stop", input_tokens=11, output_tokens=22)
    )
    provider = harness.make()
    await provider.complete(system="s", user="u", model="m")

    spans = [s for s in span_exporter.get_finished_spans() if s.name == "chat m"]
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[telemetry.GEN_AI_OPERATION_NAME] == "chat"
    assert attrs[telemetry.GEN_AI_REQUEST_MODEL] == "m"
    assert attrs[telemetry.GEN_AI_SYSTEM] in ("openai", "anthropic")
    assert attrs[telemetry.GEN_AI_USAGE_INPUT_TOKENS] == 11
    assert attrs[telemetry.GEN_AI_USAGE_OUTPUT_TOKENS] == 22


# The embedding providers (openai_compat + gitee_multimodal) have no shared
# harness, so the gen_ai.embeddings span is pinned per-backend here — the other
# half of the gen_ai_span call sites the chat test above does not reach.


async def test_openai_embed_emits_gen_ai_embeddings_span_with_usage(
    span_exporter: Any,
) -> None:
    embedder = OpenAICompatEmbeddings(
        api_key_env="OPENAI_API_KEY", base_url="https://example.test/v1", api_key="k"
    )

    async def _create(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])],
            usage=SimpleNamespace(prompt_tokens=7),
        )

    embedder._client_cache = SimpleNamespace(  # type: ignore[assignment]
        embeddings=SimpleNamespace(create=_create)
    )
    await embedder.embed(["hello"], model="text-embed-3")

    spans = [
        s
        for s in span_exporter.get_finished_spans()
        if s.name == "embeddings text-embed-3"
    ]
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[telemetry.GEN_AI_OPERATION_NAME] == "embeddings"
    assert attrs[telemetry.GEN_AI_SYSTEM] == "openai"
    assert attrs[telemetry.GEN_AI_REQUEST_MODEL] == "text-embed-3"
    assert attrs[telemetry.GEN_AI_USAGE_INPUT_TOKENS] == 7


async def test_gitee_embed_emits_gen_ai_embeddings_span(span_exporter: Any) -> None:
    embedder = GiteeMultimodalEmbedding(
        api_key_env="GITEE_API_KEY", base_url="https://example.test/v1", api_key="k"
    )

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"data": [{"index": 0, "embedding": [0.3, 0.4]}]}

    async def _post(_url: str, *, json: dict[str, Any]) -> _Resp:
        return _Resp()

    embedder._client = SimpleNamespace(post=_post)  # type: ignore[assignment]
    await embedder.embed([MultimodalInput(text="hi")], model="qwen-vl")

    spans = [
        s for s in span_exporter.get_finished_spans() if s.name == "embeddings qwen-vl"
    ]
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[telemetry.GEN_AI_OPERATION_NAME] == "embeddings"
    assert attrs[telemetry.GEN_AI_SYSTEM] == "gitee"
    assert attrs[telemetry.GEN_AI_REQUEST_MODEL] == "qwen-vl"
    # Gitee's response carries no usage block — no token attributes set.
    assert telemetry.GEN_AI_USAGE_INPUT_TOKENS not in attrs


# --------------------------------------------------------------------------- #
# Contract: streaming exception classification (anthropic_compat + openai_compat)
#
# A reasoning model's long synthesis can fail mid-stream; the synth group
# retry loop retries ONLY TransientProviderError and lets bare ProviderError
# fail fast. These tests pin that real SDK exceptions are classified into the
# right bucket and that a cancel is never reclassified. openai_codex's own
# classification is a documented follow-up, so it is excluded here.
# --------------------------------------------------------------------------- #


@dataclass
class _Classifier:
    harness: Any
    open_at: str  # the realistic site a connection-phase error surfaces from
    timeout: Callable[[], BaseException]
    connection: Callable[[], BaseException]
    status: Callable[[int], BaseException]
    api_error: Callable[[], BaseException]


@pytest.fixture(
    params=[
        pytest.param("openai_compat", id="openai_compat"),
        pytest.param("anthropic_compat", id="anthropic_compat"),
    ]
)
def classifier(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> _Classifier:
    req = httpx.Request("POST", "http://fake.example/v1/x")
    if request.param == "anthropic_compat":
        import anthropic

        return _Classifier(
            harness=_AnthropicHarness(monkeypatch),
            open_at="aenter",
            timeout=lambda: anthropic.APITimeoutError(request=req),
            connection=lambda: anthropic.APIConnectionError(request=req),
            status=lambda code: anthropic.APIStatusError(
                "boom", response=httpx.Response(code, request=req), body=None
            ),
            api_error=lambda: anthropic.APIError("boom", req, body=None),
        )
    import openai

    return _Classifier(
        harness=_OpenAICompatHarness(monkeypatch),
        open_at="create",
        timeout=lambda: openai.APITimeoutError(request=req),
        connection=lambda: openai.APIConnectionError(request=req),
        status=lambda code: openai.APIStatusError(
            "boom", response=httpx.Response(code, request=req), body=None
        ),
        api_error=lambda: openai.APIError("boom", req, body=None),
    )


async def test_stream_timeout_is_transient(classifier: _Classifier) -> None:
    classifier.harness.arrange_stream_raises(classifier.timeout)
    provider = classifier.harness.make()
    with pytest.raises(TransientProviderError):
        await _drain(provider)


async def test_stream_connection_error_is_transient(classifier: _Classifier) -> None:
    classifier.harness.arrange_stream_raises(classifier.connection)
    provider = classifier.harness.make()
    with pytest.raises(TransientProviderError):
        await _drain(provider)


@pytest.mark.parametrize("code", [500, 503, 408, 429])
async def test_stream_5xx_and_throttle_are_transient(
    classifier: _Classifier, code: int
) -> None:
    classifier.harness.arrange_stream_raises(lambda: classifier.status(code))
    provider = classifier.harness.make()
    with pytest.raises(TransientProviderError):
        await _drain(provider)


@pytest.mark.parametrize("code", [400, 401, 403, 404])
async def test_stream_client_errors_fail_fast(
    classifier: _Classifier, code: int
) -> None:
    # Surfaces from the connection-open site (where a real 401 lands), and
    # must be the bare permanent ProviderError — NOT the retryable subclass,
    # so misconfig is not retried-then-skipped.
    classifier.harness.arrange_stream_raises(
        lambda: classifier.status(code), at=classifier.open_at
    )
    provider = classifier.harness.make()
    with pytest.raises(ProviderError) as ei:
        await _drain(provider)
    assert not isinstance(ei.value, TransientProviderError)


async def test_stream_base_api_error_is_permanent(classifier: _Classifier) -> None:
    classifier.harness.arrange_stream_raises(classifier.api_error)
    provider = classifier.harness.make()
    with pytest.raises(ProviderError) as ei:
        await _drain(provider)
    assert not isinstance(ei.value, TransientProviderError)


async def test_stream_cancel_propagates_untouched(classifier: _Classifier) -> None:
    # asyncio.CancelledError is a BaseException — it must propagate so synth's
    # per-group cancel contract holds, never be reclassified as transient.
    classifier.harness.arrange_stream_raises(lambda: asyncio.CancelledError())
    provider = classifier.harness.make()
    with pytest.raises(asyncio.CancelledError):
        await _drain(provider)


async def test_complete_empty_text_is_legal_zero_page(harness: _Harness) -> None:
    # A reasoning model that legitimately emits zero text (synth's "this
    # candidate duplicates an existing page → zero <page> blocks" path)
    # yields no tokens and an empty final; complete() must return text="" and
    # NOT raise — the empty result is a legal signal, not a failure.
    harness.arrange_stream(_StreamScript(deltas=[], final_text=""))
    provider = harness.make()
    resp = await provider.complete(system="s", user="u", model="m")
    assert resp.text == ""
