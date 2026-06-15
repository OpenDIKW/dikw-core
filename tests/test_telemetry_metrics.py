"""PR3a: GenAI metrics — token usage + operation duration histograms.

The metrics piggy-back on the existing ``gen_ai_span`` / ``trace_llm_stream``
span seam (PR2a), so every provider's chat + embeddings call emits
``gen_ai.client.token.usage`` and ``gen_ai.client.operation.duration`` with no
provider-side code change. These tests pin that emission against an
``InMemoryMetricReader`` and prove the meter-provider bootstrap + teardown.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from dikw_core import telemetry

pytestmark = pytest.mark.skipif(
    not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra"
)

_KW = dict(  # noqa: C408 - shared kwargs for configure_telemetry calls
    endpoint=None, service_name="dikw-core", sample_ratio=1.0, version="0.0.0+test"
)


@pytest.fixture(autouse=True)
def _reset_telemetry() -> Iterator[None]:
    telemetry.reset_telemetry_for_testing()
    yield
    telemetry.reset_telemetry_for_testing()


def _install_inmemory_meter() -> Any:
    """Register a MeterProvider backed by an InMemoryMetricReader and flip the
    ``_meter_provider`` gate, mimicking what ``configure_telemetry`` wires but
    with a reader whose data the test can read back (the production path uses a
    network OTLP exporter). The autouse reset clears OTel's process-once latch
    before each test so ``set_meter_provider`` takes effect."""
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    telemetry._meter_provider = provider
    return reader


def _histogram_points(reader: Any, name: str) -> list[Any]:
    data = reader.get_metrics_data()
    points: list[Any] = []
    if data is None:
        return points
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == name:
                    points.extend(metric.data.data_points)
    return points


def _point_for(points: list[Any], **match: str) -> Any:
    for point in points:
        attrs = dict(point.attributes)
        if all(attrs.get(k) == v for k, v in match.items()):
            return point
    raise AssertionError(f"no data point matching {match} in {points}")


@dataclass
class _Event:
    type: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


def test_configure_telemetry_wires_meter_provider() -> None:
    """``configure_telemetry`` must register an SDK MeterProvider alongside the
    tracer provider so server-side GenAI + HTTP metrics flow to the OTLP
    endpoint."""
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.sdk.metrics import MeterProvider

    assert telemetry.configure_telemetry(enabled=True, **_KW) is True  # type: ignore[arg-type]
    assert isinstance(otel_metrics.get_meter_provider(), MeterProvider)
    assert telemetry._meter_provider is not None
    telemetry.shutdown_telemetry()


def test_gen_ai_span_records_token_usage_and_duration() -> None:
    """A completed provider call records two token-usage points (input/output,
    tagged ``gen_ai.token.type``) and one operation-duration point — all tagged
    with operation/system/model."""
    reader = _install_inmemory_meter()

    with telemetry.gen_ai_span(operation="chat", system="openai", model="gpt-x") as span:
        span.set_response(
            finish_reason="stop", usage={"input_tokens": 100, "output_tokens": 20}
        )

    tokens = _histogram_points(reader, "gen_ai.client.token.usage")
    in_point = _point_for(
        tokens,
        **{
            telemetry.GEN_AI_TOKEN_TYPE: "input",
            telemetry.GEN_AI_OPERATION_NAME: "chat",
            telemetry.GEN_AI_SYSTEM: "openai",
            telemetry.GEN_AI_REQUEST_MODEL: "gpt-x",
        },
    )
    assert in_point.sum == 100
    out_point = _point_for(tokens, **{telemetry.GEN_AI_TOKEN_TYPE: "output"})
    assert out_point.sum == 20

    durations = _histogram_points(reader, "gen_ai.client.operation.duration")
    dur_point = _point_for(durations, **{telemetry.GEN_AI_OPERATION_NAME: "chat"})
    assert dur_point.count == 1
    assert dur_point.sum >= 0
    assert telemetry.ERROR_TYPE not in dict(dur_point.attributes)


def test_gen_ai_span_records_error_type_on_failure() -> None:
    """A failed call tags the duration point with ``error.type`` and records NO
    token usage (there is none)."""
    reader = _install_inmemory_meter()

    with pytest.raises(ValueError), telemetry.gen_ai_span(
        operation="chat", system="openai", model="gpt-x"
    ):
        raise ValueError("boom")

    durations = _histogram_points(reader, "gen_ai.client.operation.duration")
    dur_point = _point_for(durations, **{telemetry.ERROR_TYPE: "ValueError"})
    assert dur_point.count == 1
    assert _histogram_points(reader, "gen_ai.client.token.usage") == []


def test_gen_ai_span_cancel_is_not_an_error_in_metrics() -> None:
    """A cooperative cancel is a graceful terminal: duration is recorded WITHOUT
    an ``error.type`` tag (it must not pollute the failure dashboards)."""
    import asyncio

    reader = _install_inmemory_meter()

    with pytest.raises(asyncio.CancelledError), telemetry.gen_ai_span(
        operation="chat", system="openai", model="gpt-x"
    ):
        raise asyncio.CancelledError

    durations = _histogram_points(reader, "gen_ai.client.operation.duration")
    dur_point = _point_for(durations, **{telemetry.GEN_AI_OPERATION_NAME: "chat"})
    assert telemetry.ERROR_TYPE not in dict(dur_point.attributes)


async def test_trace_llm_stream_records_metrics_from_done_event() -> None:
    """The streaming wrapper reads usage off the terminal ``done`` event and
    records the same metrics as the non-streaming path."""
    reader = _install_inmemory_meter()

    async def _events() -> AsyncIterator[_Event]:
        yield _Event(type="token")
        yield _Event(
            type="done",
            finish_reason="stop",
            usage={"input_tokens": 7, "output_tokens": 3},
        )

    collected = [
        event
        async for event in telemetry.trace_llm_stream(
            _events(), system="anthropic", model="claude-x"
        )
    ]
    assert len(collected) == 2

    tokens = _histogram_points(reader, "gen_ai.client.token.usage")
    assert _point_for(tokens, **{telemetry.GEN_AI_TOKEN_TYPE: "input"}).sum == 7
    assert _point_for(tokens, **{telemetry.GEN_AI_TOKEN_TYPE: "output"}).sum == 3


def test_no_metrics_recorded_when_meter_provider_absent() -> None:
    """With no meter provider wired (telemetry inactive), ``gen_ai_span`` must
    not create instruments or crash — it is a pure no-op on the metrics side."""
    with telemetry.gen_ai_span(operation="embeddings", system="gitee", model="m") as span:
        span.set_response(usage={"input_tokens": 5})
    assert telemetry._gen_ai_op_duration is None
    assert telemetry._gen_ai_token_usage is None


def test_shutdown_unwinds_meter_provider() -> None:
    """Shutdown must flush + drop the meter provider (stopping its exporter
    thread) and clear the cached instruments, so a fresh lifespan re-wires."""
    assert telemetry.configure_telemetry(enabled=True, **_KW) is True  # type: ignore[arg-type]
    assert telemetry._meter_provider is not None
    telemetry.shutdown_telemetry()
    assert telemetry._meter_provider is None
    assert telemetry._gen_ai_op_duration is None
    assert telemetry._gen_ai_token_usage is None
