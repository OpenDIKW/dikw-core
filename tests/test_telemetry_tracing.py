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
    # A cancel is a graceful terminal — flagged, but NOT an error status.
    assert s.attributes[telemetry.DIKW_CANCELLED] is True
    assert s.status.status_code != StatusCode.ERROR


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
