"""PR3b: dikw-domain metrics — counters + duration histograms.

The domain instruments piggy-back on the same ``_meter_provider`` gate as the
PR3a GenAI metrics: a public ``record_*`` helper per seam, each a pure no-op
until :func:`configure_telemetry` wires dikw's own meter provider, each
defensively wrapped so telemetry can never crash the engine. These tests pin
the emitted points against an ``InMemoryMetricReader`` and prove the
behaviour-preserving transparency of the one eval-gated wrapper
(``_traced_leg`` in ``domains/info/search.py``).
"""

from __future__ import annotations

from collections.abc import Iterator
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
    with a reader whose data the test can read back. The autouse reset clears
    OTel's process-once latch before each test so ``set_meter_provider`` takes
    effect."""
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    telemetry._meter_provider = provider
    return reader


def _points(reader: Any, name: str) -> list[Any]:
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


# ---- ingest counters ----------------------------------------------------


def test_record_ingest_metrics_emits_files_chunks_errors() -> None:
    reader = _install_inmemory_meter()

    telemetry.record_ingest_metrics(
        added=2,
        updated=1,
        unchanged=3,
        chunks=10,
        errors_by_kind={"parse_error": 1, "read_error": 2},
    )

    files = _points(reader, "dikw.ingest.files")
    assert _point_for(files, **{telemetry.DIKW_RESULT: "added"}).value == 2
    assert _point_for(files, **{telemetry.DIKW_RESULT: "updated"}).value == 1
    assert _point_for(files, **{telemetry.DIKW_RESULT: "unchanged"}).value == 3

    assert _points(reader, "dikw.ingest.chunks")[0].value == 10

    errors = _points(reader, "dikw.ingest.errors")
    assert _point_for(errors, **{telemetry.DIKW_ERROR_KIND: "parse_error"}).value == 1
    assert _point_for(errors, **{telemetry.DIKW_ERROR_KIND: "read_error"}).value == 2

    # Embedded-chunk volume is metered at the embed seam, NOT from the ingest
    # report — so record_ingest_metrics emits no embed.chunks point.
    assert _points(reader, "dikw.embed.chunks") == []


def test_record_ingest_metrics_skips_zero_valued_series() -> None:
    """A clean run with no added/updated/unchanged files (or no errors) must not
    emit a 0-valued point — keeps the metric cardinality honest."""
    reader = _install_inmemory_meter()

    telemetry.record_ingest_metrics(
        added=0, updated=0, unchanged=0, chunks=0, errors_by_kind={}
    )

    assert _points(reader, "dikw.ingest.files") == []
    assert _points(reader, "dikw.ingest.chunks") == []
    assert _points(reader, "dikw.ingest.errors") == []


# ---- synth counters -----------------------------------------------------


def test_record_synth_metrics_emits_pages_unresolved_persist_errors() -> None:
    reader = _install_inmemory_meter()

    telemetry.record_synth_metrics(
        created=3, updated=1, unresolved_wikilinks=5, persist_errors=2
    )

    pages = _points(reader, "dikw.synth.pages")
    assert _point_for(pages, **{telemetry.DIKW_RESULT: "created"}).value == 3
    assert _point_for(pages, **{telemetry.DIKW_RESULT: "updated"}).value == 1

    assert _points(reader, "dikw.synth.unresolved_wikilinks")[0].value == 5
    assert _points(reader, "dikw.synth.persist_errors")[0].value == 2


# ---- embed counters -----------------------------------------------------


def test_record_embed_metrics_emits_embedded_skipped_and_retries() -> None:
    reader = _install_inmemory_meter()

    telemetry.record_embed_metrics(embedded=12, chunks_skipped=4, retries=3)

    assert _points(reader, "dikw.embed.chunks")[0].value == 12
    assert _points(reader, "dikw.embed.skipped")[0].value == 4
    assert _points(reader, "dikw.embed.retries")[0].value == 3


def test_record_embed_metrics_skips_zero() -> None:
    reader = _install_inmemory_meter()

    telemetry.record_embed_metrics(embedded=0, chunks_skipped=0, retries=0)

    assert _points(reader, "dikw.embed.chunks") == []
    assert _points(reader, "dikw.embed.skipped") == []
    assert _points(reader, "dikw.embed.retries") == []


# ---- duration histograms ------------------------------------------------


def test_record_retrieve_leg_duration_tags_leg() -> None:
    reader = _install_inmemory_meter()

    telemetry.record_retrieve_leg_duration("bm25", 0.01)
    telemetry.record_retrieve_leg_duration("vector", 0.02)

    points = _points(reader, "dikw.retrieve.leg.duration")
    bm25 = _point_for(points, **{telemetry.DIKW_RETRIEVAL_LEG: "bm25"})
    assert bm25.count == 1
    assert bm25.sum >= 0
    vector = _point_for(points, **{telemetry.DIKW_RETRIEVAL_LEG: "vector"})
    assert vector.count == 1


def test_record_task_duration_tags_op_and_status() -> None:
    reader = _install_inmemory_meter()

    telemetry.record_task_duration("ingest", "ok", 1.5)
    telemetry.record_task_duration("synth", "cancelled", 0.5)

    points = _points(reader, "dikw.task.duration")
    ok = _point_for(
        points, **{telemetry.DIKW_OP: "ingest", telemetry.DIKW_STATUS: "ok"}
    )
    assert ok.count == 1
    cancelled = _point_for(
        points, **{telemetry.DIKW_OP: "synth", telemetry.DIKW_STATUS: "cancelled"}
    )
    assert cancelled.count == 1


# ---- inactive / teardown ------------------------------------------------


def test_no_domain_metrics_when_meter_provider_absent() -> None:
    """With no meter provider wired the record helpers must not create
    instruments or crash — pure no-ops on the metrics side."""
    telemetry.record_ingest_metrics(
        added=1, updated=0, unchanged=0, chunks=1, errors_by_kind={}
    )
    telemetry.record_synth_metrics(
        created=1, updated=0, unresolved_wikilinks=0, persist_errors=0
    )
    telemetry.record_embed_metrics(embedded=1, chunks_skipped=1, retries=1)
    telemetry.record_retrieve_leg_duration("bm25", 0.01)
    telemetry.record_task_duration("ingest", "ok", 1.0)
    assert telemetry._domain_instruments is None


def test_shutdown_clears_domain_instruments() -> None:
    assert telemetry.configure_telemetry(enabled=True, **_KW) is True  # type: ignore[arg-type]
    telemetry.record_synth_metrics(
        created=1, updated=0, unresolved_wikilinks=0, persist_errors=0
    )
    assert telemetry._domain_instruments is not None
    telemetry.shutdown_telemetry()
    assert telemetry._domain_instruments is None


# ---- behaviour-preservation guard (eval-gated path) ---------------------


async def test_traced_leg_is_transparent_with_metrics_on() -> None:
    """``_traced_leg`` (the one eval-gated wrapper, in domains/info/search.py)
    must return its leg coroutine's result UNCHANGED and run it exactly once,
    whether or not metrics are wired — the leg-duration instrumentation only
    times the existing await, never alters retrieval. Records a duration point
    when metrics are on."""
    from dikw_core.domains.info.search import _traced_leg

    sentinel = [object(), object()]
    calls = 0

    async def _leg() -> list[object]:
        nonlocal calls
        calls += 1
        return sentinel

    # metrics off
    assert await _traced_leg("bm25", _leg()) is sentinel
    assert calls == 1

    # metrics on — same result, leg still runs once more, duration recorded
    reader = _install_inmemory_meter()
    assert await _traced_leg("bm25", _leg()) is sentinel
    assert calls == 2
    assert _points(reader, "dikw.retrieve.leg.duration")[0].count == 1


async def test_traced_leg_records_duration_when_leg_errors_and_reraises() -> None:
    """A leg that raises a hard error (not NotSupported) is still timed exactly
    once (the single ``finally``) and the exception propagates unchanged — a
    slow-then-failing leg must not be invisible on the latency histogram."""
    from dikw_core.domains.info.search import _traced_leg

    reader = _install_inmemory_meter()

    async def _boom() -> list[object]:
        raise RuntimeError("leg blew up")

    with pytest.raises(RuntimeError, match="leg blew up"):
        await _traced_leg("vector", _boom())

    points = _points(reader, "dikw.retrieve.leg.duration")
    assert _point_for(points, **{telemetry.DIKW_RETRIEVAL_LEG: "vector"}).count == 1
