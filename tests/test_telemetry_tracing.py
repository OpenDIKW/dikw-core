"""PR2 tracing: the gen_ai / task span helpers emit correct spans + degrade.

These cover the engine-facing span seams added in the OTel tracing arc:
``gen_ai_span`` (single-shot provider calls), ``trace_llm_stream`` (streaming
LLM calls), and the no-op fallback when the ``[otel]`` extra is absent. The
task-span integration (capture + link across the asyncio.create_task boundary)
is exercised end-to-end in ``tests/server/test_telemetry_tracing.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dikw_core import telemetry
from dikw_core.providers.base import LLMStreamEvent


def _one_span(exporter: Any) -> Any:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f"expected exactly one span, got {[s.name for s in spans]}"
    return spans[0]


def test_gen_ai_span_sets_request_and_response_attributes(span_exporter: Any) -> None:
    with telemetry.gen_ai_span(
        operation="chat", system="openai", model="gpt-x", max_tokens=128, temperature=0.2
    ) as span:
        span.set_response(finish_reason="stop", usage={"input_tokens": 5, "output_tokens": 7})

    s = _one_span(span_exporter)
    assert s.name == "chat gpt-x"
    attrs = s.attributes
    assert attrs[telemetry.GEN_AI_OPERATION_NAME] == "chat"
    assert attrs[telemetry.GEN_AI_SYSTEM] == "openai"
    assert attrs[telemetry.GEN_AI_REQUEST_MODEL] == "gpt-x"
    assert attrs[telemetry.GEN_AI_REQUEST_MAX_TOKENS] == 128
    assert attrs[telemetry.GEN_AI_REQUEST_TEMPERATURE] == pytest.approx(0.2)
    assert attrs[telemetry.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert attrs[telemetry.GEN_AI_USAGE_INPUT_TOKENS] == 5
    assert attrs[telemetry.GEN_AI_USAGE_OUTPUT_TOKENS] == 7

    from opentelemetry.trace import StatusCode

    assert s.status.status_code == StatusCode.OK


def test_gen_ai_span_records_anthropic_cache_tokens(span_exporter: Any) -> None:
    with telemetry.gen_ai_span(operation="chat", system="anthropic", model="m") as span:
        span.set_response(
            usage={
                "input_tokens": 3,
                "output_tokens": 4,
                "cache_read_input_tokens": 11,
                "cache_creation_input_tokens": 2,
            }
        )

    attrs = _one_span(span_exporter).attributes
    assert attrs[telemetry.GEN_AI_CACHE_READ_INPUT_TOKENS] == 11
    assert attrs[telemetry.GEN_AI_CACHE_CREATION_INPUT_TOKENS] == 2


def test_gen_ai_span_marks_error_status(span_exporter: Any) -> None:
    from opentelemetry.trace import StatusCode

    with pytest.raises(ValueError, match="boom"):
        with telemetry.gen_ai_span(operation="chat", system="openai", model="m"):
            raise ValueError("boom")

    s = _one_span(span_exporter)
    assert s.status.status_code == StatusCode.ERROR
    assert telemetry.DIKW_CANCELLED not in s.attributes
    # the exception was recorded as a span event
    assert any(ev.name == "exception" for ev in s.events)


def test_gen_ai_span_marks_cancel_not_error(span_exporter: Any) -> None:
    from opentelemetry.trace import StatusCode

    with pytest.raises(asyncio.CancelledError):
        with telemetry.gen_ai_span(operation="chat", system="openai", model="m"):
            raise asyncio.CancelledError

    s = _one_span(span_exporter)
    # A cancel is a graceful terminal — flagged, with the status left UNSET
    # (not ERROR, and not falsely OK).
    assert s.attributes[telemetry.DIKW_CANCELLED] is True
    assert s.status.status_code == StatusCode.UNSET


def test_capture_otel_context_returns_none_without_active_span(span_exporter: Any) -> None:
    """OTEL installed + an active provider, but no current span: the
    ``is_valid == False`` branch returns None, so a task submitted outside any
    request span gets no back-link (rather than a bogus invalid-span link)."""
    assert telemetry.capture_otel_context() is None


async def test_trace_llm_stream_passes_events_and_reads_done(span_exporter: Any) -> None:
    async def _fake_stream() -> Any:
        yield LLMStreamEvent(type="token", delta="he")
        yield LLMStreamEvent(type="token", delta="llo")
        yield LLMStreamEvent(
            type="done",
            text="hello",
            finish_reason="stop",
            usage={"input_tokens": 9, "output_tokens": 13},
        )

    seen = [
        ev
        async for ev in telemetry.trace_llm_stream(
            _fake_stream(), system="anthropic", model="claude", max_tokens=64
        )
    ]
    # Events pass through untouched.
    assert [e.type for e in seen] == ["token", "token", "done"]
    assert seen[-1].text == "hello"

    s = _one_span(span_exporter)
    assert s.name == "chat claude"
    attrs = s.attributes
    assert attrs[telemetry.GEN_AI_SYSTEM] == "anthropic"
    assert attrs[telemetry.GEN_AI_REQUEST_MODEL] == "claude"
    assert attrs[telemetry.GEN_AI_USAGE_INPUT_TOKENS] == 9
    assert attrs[telemetry.GEN_AI_USAGE_OUTPUT_TOKENS] == 13
    assert attrs[telemetry.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)


async def test_trace_llm_stream_marks_error_on_provider_failure(
    span_exporter: Any,
) -> None:
    from opentelemetry.trace import StatusCode

    async def _failing_stream() -> Any:
        yield LLMStreamEvent(type="token", delta="x")
        raise RuntimeError("upstream 500")

    with pytest.raises(RuntimeError, match="upstream 500"):
        async for _ in telemetry.trace_llm_stream(
            _failing_stream(), system="openai", model="m"
        ):
            pass

    assert _one_span(span_exporter).status.status_code == StatusCode.ERROR


async def test_trace_llm_stream_early_break_is_graceful_not_error(
    span_exporter: Any,
) -> None:
    """A consumer that stops pulling after ``done`` and closes the stream is a
    graceful early-close (GeneratorExit), not an LLM failure — the span must end
    UNSET with no recorded exception, not ERROR."""
    from opentelemetry.trace import StatusCode

    async def _fake_stream() -> Any:
        yield LLMStreamEvent(type="token", delta="he")
        yield LLMStreamEvent(
            type="done", text="he", finish_reason="stop", usage={"input_tokens": 1}
        )
        yield LLMStreamEvent(type="token", delta="never-pulled")

    stream = telemetry.trace_llm_stream(_fake_stream(), system="openai", model="m")
    async for ev in stream:
        if ev.type == "done":
            break
    # Breaking out of an ``async for`` does NOT auto-close the generator;
    # aclose() throws GeneratorExit into the suspended span body.
    await stream.aclose()

    s = _one_span(span_exporter)
    assert s.status.status_code == StatusCode.UNSET
    assert s.status.status_code != StatusCode.ERROR
    assert not any(ev.name == "exception" for ev in s.events)
    # The terminal ``done`` was processed before the break.
    assert s.attributes[telemetry.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)


def test_span_helpers_are_noops_without_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimal install (no ``[otel]`` extra): every span helper degrades to a
    usable no-op so engine/provider code can emit spans unconditionally."""
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)
    assert telemetry.capture_otel_context() is None
    with telemetry.gen_ai_span(operation="chat", system="openai", model="m") as gspan:
        gspan.set_response(finish_reason="stop", usage={"input_tokens": 1})
    with telemetry.task_span("ingest", task_id="t1", base_id="b1") as tspan:
        tspan.ok()
        tspan.cancelled()
        tspan.record_error(ValueError("x"))


async def test_trace_llm_stream_is_noop_without_otel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    async def _stream() -> Any:
        yield LLMStreamEvent(type="done", text="ok", finish_reason="stop")

    seen = [ev async for ev in telemetry.trace_llm_stream(_stream(), system="x", model="m")]
    assert [e.type for e in seen] == ["done"]


# --------------------------------------------------------------------------- #
# op_span — the generic engine op-level span helper (PR2b)
# --------------------------------------------------------------------------- #


def test_op_span_sets_open_and_posthoc_attributes_and_ok(span_exporter: Any) -> None:
    from opentelemetry.trace import StatusCode

    with telemetry.op_span(
        "dikw.ingest",
        attributes={telemetry.DIKW_LAYER: "data", telemetry.DIKW_OP: "ingest"},
    ) as span:
        span.set_attribute(telemetry.DIKW_SOURCE_PATH, "sources/a.md")

    s = _one_span(span_exporter)
    assert s.name == "dikw.ingest"
    assert s.attributes[telemetry.DIKW_LAYER] == "data"
    assert s.attributes[telemetry.DIKW_OP] == "ingest"
    assert s.attributes[telemetry.DIKW_SOURCE_PATH] == "sources/a.md"
    assert s.status.status_code == StatusCode.OK


def test_op_span_marks_error_status(span_exporter: Any) -> None:
    from opentelemetry.trace import StatusCode

    with pytest.raises(ValueError, match="boom"):
        with telemetry.op_span("dikw.synth"):
            raise ValueError("boom")

    s = _one_span(span_exporter)
    assert s.status.status_code == StatusCode.ERROR
    assert telemetry.DIKW_CANCELLED not in s.attributes
    assert any(ev.name == "exception" for ev in s.events)


def test_op_span_marks_cancel_not_error(span_exporter: Any) -> None:
    from opentelemetry.trace import StatusCode

    with pytest.raises(asyncio.CancelledError):
        with telemetry.op_span("dikw.synth"):
            raise asyncio.CancelledError

    s = _one_span(span_exporter)
    assert s.attributes[telemetry.DIKW_CANCELLED] is True
    assert s.status.status_code == StatusCode.UNSET


def test_op_span_generatorexit_is_graceful(span_exporter: Any) -> None:
    """A GeneratorExit thrown into op_span (a wrapping generator closed early —
    reachable since _traced_leg runs op_span inside a create_task'd coroutine) is
    a graceful close: status left UNSET, no exception event, not ERROR."""
    from opentelemetry.trace import StatusCode

    def _gen() -> Any:
        with telemetry.op_span("dikw.synth"):
            yield 1
            yield 2  # never pulled — consumer closes after the first

    g = _gen()
    next(g)  # enter the op_span, suspend at the first yield
    g.close()  # throws GeneratorExit into the suspended `with op_span`

    s = _one_span(span_exporter)
    assert s.status.status_code == StatusCode.UNSET
    assert telemetry.DIKW_CANCELLED not in s.attributes
    assert not any(ev.name == "exception" for ev in s.events)


def test_op_span_nests_child_under_parent(span_exporter: Any) -> None:
    with telemetry.op_span("dikw.synth"):
        with telemetry.op_span(
            "dikw.synth.source",
            attributes={telemetry.DIKW_SOURCE_PATH: "sources/a.md"},
        ):
            pass

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    parent, child = spans["dikw.synth"], spans["dikw.synth.source"]
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id


def test_op_span_is_noop_without_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)
    with telemetry.op_span(
        "dikw.ingest", attributes={telemetry.DIKW_LAYER: "data"}
    ) as span:
        span.set_attribute(telemetry.DIKW_SOURCE_PATH, "x")


async def test_traced_op_wraps_async_fn_and_nests_children(span_exporter: Any) -> None:
    from opentelemetry.trace import StatusCode

    @telemetry.traced_op(
        "dikw.ingest",
        attributes={telemetry.DIKW_LAYER: "data", telemetry.DIKW_OP: "ingest"},
    )
    async def _do(x: int) -> int:
        # A provider / nested op span opened in the body must become a child.
        with telemetry.op_span("dikw.ingest.inner"):
            pass
        return x * 2

    assert await _do(21) == 42

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    outer, inner = spans["dikw.ingest"], spans["dikw.ingest.inner"]
    assert outer.attributes[telemetry.DIKW_LAYER] == "data"
    assert outer.attributes[telemetry.DIKW_OP] == "ingest"
    assert outer.status.status_code == StatusCode.OK
    assert inner.parent is not None
    assert inner.parent.span_id == outer.context.span_id


async def test_traced_op_is_noop_without_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    @telemetry.traced_op("dikw.synth")
    async def _do() -> str:
        return "ok"

    assert await _do() == "ok"
