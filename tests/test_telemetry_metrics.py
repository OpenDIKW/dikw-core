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


def _raise_in_gen_ai_span(exc: BaseException) -> None:
    """Run a ``gen_ai_span`` whose body raises ``exc``. Factored out so the
    test's ``pytest.raises`` wraps a CALL — a single ``with`` (no ruff SIM117
    nudge to combine) whose post-block stays reachable to static analysis —
    rather than a bare ``raise`` inside a combined ``with``."""
    with telemetry.gen_ai_span(operation="chat", system="openai", model="gpt-x"):
        raise exc


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


def test_anthropic_cache_tokens_get_their_own_token_type_series() -> None:
    """Anthropic prompt-cache tokens are a separate cost tier from fresh input,
    so they land on distinct ``gen_ai.token.type`` series (``cache_read`` /
    ``cache_creation``) — not folded into ``input`` — preserving both total
    volume (summable) and the cost-tier breakdown for metrics-only dashboards."""
    reader = _install_inmemory_meter()

    with telemetry.gen_ai_span(
        operation="chat", system="anthropic", model="claude-x"
    ) as span:
        span.set_response(
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 7,
            }
        )

    tokens = _histogram_points(reader, "gen_ai.client.token.usage")
    assert _point_for(tokens, **{telemetry.GEN_AI_TOKEN_TYPE: "input"}).sum == 10
    assert _point_for(tokens, **{telemetry.GEN_AI_TOKEN_TYPE: "output"}).sum == 5
    assert _point_for(tokens, **{telemetry.GEN_AI_TOKEN_TYPE: "cache_read"}).sum == 100
    assert (
        _point_for(tokens, **{telemetry.GEN_AI_TOKEN_TYPE: "cache_creation"}).sum == 7
    )


def test_gen_ai_span_records_error_type_on_failure() -> None:
    """A failed call tags the duration point with ``error.type`` and records NO
    token usage (there is none)."""
    reader = _install_inmemory_meter()

    with pytest.raises(ValueError):
        _raise_in_gen_ai_span(ValueError("boom"))

    durations = _histogram_points(reader, "gen_ai.client.operation.duration")
    dur_point = _point_for(durations, **{telemetry.ERROR_TYPE: "ValueError"})
    assert dur_point.count == 1
    assert _histogram_points(reader, "gen_ai.client.token.usage") == []


def test_gen_ai_span_cancel_records_no_metric() -> None:
    """A cooperative cancel is a graceful, partial abandonment — NOT a completed
    operation — so it records no duration (and no token) point: its cut-short
    elapsed time would skew the operation-duration latency series with a point
    indistinguishable from a real completion. The span still carries
    ``dikw.cancelled`` for trace-level visibility."""
    import asyncio

    reader = _install_inmemory_meter()

    with pytest.raises(asyncio.CancelledError):
        _raise_in_gen_ai_span(asyncio.CancelledError())

    assert _histogram_points(reader, "gen_ai.client.operation.duration") == []
    assert _histogram_points(reader, "gen_ai.client.token.usage") == []


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


def test_adopt_meter_provider_loses_race_and_degrades() -> None:
    """When a MeterProvider is already registered (process-once), dikw's bid is
    ignored: ``_adopt_meter_provider`` must shut down the orphan it built and
    report False, so the PeriodicExportingMetricReader thread doesn't leak and
    metrics never flow to a foreign provider."""
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider

    foreign = MeterProvider()
    metrics.set_meter_provider(foreign)  # wins the process-once race

    ours = MeterProvider()
    shutdowns: list[bool] = []
    original_shutdown = ours.shutdown

    def _spy_shutdown(*args: Any, **kwargs: Any) -> Any:
        shutdowns.append(True)
        return original_shutdown(*args, **kwargs)

    ours.shutdown = _spy_shutdown  # type: ignore[method-assign]

    assert telemetry._adopt_meter_provider(ours) is False
    assert shutdowns == [True]  # orphan torn down
    assert metrics.get_meter_provider() is foreign  # foreign provider untouched


def test_shutdown_unwinds_meter_provider() -> None:
    """Shutdown must flush + drop the meter provider (stopping its exporter
    thread) and clear the cached instruments, so a fresh lifespan re-wires."""
    assert telemetry.configure_telemetry(enabled=True, **_KW) is True  # type: ignore[arg-type]
    assert telemetry._meter_provider is not None
    telemetry.shutdown_telemetry()
    assert telemetry._meter_provider is None
    assert telemetry._gen_ai_op_duration is None
    assert telemetry._gen_ai_token_usage is None
