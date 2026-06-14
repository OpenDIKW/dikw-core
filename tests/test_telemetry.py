"""PR1 foundations: the telemetry seam is no-op-safe + entry-gated.

Proves the contract the rest of the OTel arc builds on: accessors are always
usable (otel installed or not), and ``configure_telemetry`` activates only on
the intended path (enabled + ``[otel]`` present + not ``OTEL_SDK_DISABLED``),
staying a zero-side-effect no-op otherwise.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any

import pytest

from dikw_core import telemetry

_KW = dict(  # noqa: C408 - shared kwargs for configure_telemetry calls
    endpoint=None, service_name="dikw-core", sample_ratio=1.0, version="0.0.0+test"
)


@pytest.fixture(autouse=True)
def _reset_telemetry() -> Iterator[None]:
    telemetry.reset_telemetry_for_testing()
    yield
    telemetry.reset_telemetry_for_testing()


def test_attribute_keys_are_dikw_namespaced() -> None:
    assert telemetry.DIKW_LAYER == "dikw.layer"
    assert telemetry.DIKW_TASK_ID == "dikw.task_id"
    assert telemetry.DIKW_RETRIEVAL_LEG == "dikw.retrieval.leg"
    for key in (
        telemetry.DIKW_LAYER,
        telemetry.DIKW_OP,
        telemetry.DIKW_TASK_ID,
        telemetry.DIKW_BASE_ID,
        telemetry.DIKW_SOURCE_PATH,
        telemetry.DIKW_CATEGORY,
        telemetry.DIKW_RETRIEVAL_LEG,
        telemetry.DIKW_EMBED_VERSION_ID,
    ):
        assert key.startswith("dikw.")


def test_get_tracer_span_is_a_usable_context_manager() -> None:
    """Engine code can open a span unconditionally — no provider required."""
    tracer = telemetry.get_tracer()
    with tracer.start_as_current_span("unit.test") as span:
        span.set_attribute(telemetry.DIKW_LAYER, "data")
    # reaching here without raising is the assertion


def test_get_meter_instruments_are_usable() -> None:
    meter = telemetry.get_meter()
    counter = meter.create_counter("dikw.test.counter")
    counter.add(1, {"k": "v"})
    hist = meter.create_histogram("dikw.test.duration")
    hist.record(0.5, {"k": "v"})


def test_telemetry_should_activate_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared gate the SDK bootstrap and the FastAPI-instrumentation
    decision both read — they must never diverge."""
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    assert telemetry.telemetry_should_activate(False) is False
    if telemetry.OTEL_AVAILABLE:
        assert telemetry.telemetry_should_activate(True) is True
        monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
        assert telemetry.telemetry_should_activate(True) is False
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)
    assert telemetry.telemetry_should_activate(True) is False


def test_configure_telemetry_noop_when_disabled() -> None:
    assert telemetry.configure_telemetry(enabled=False, **_KW) is False


def test_configure_telemetry_honours_otel_sdk_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert telemetry.configure_telemetry(enabled=True, **_KW) is False


def test_configure_telemetry_noop_when_otel_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a minimal install: enabled config still degrades to no-op and
    the accessors stay safe."""
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)
    assert telemetry.configure_telemetry(enabled=True, **_KW) is False
    with telemetry.get_tracer().start_as_current_span("x"):
        pass
    telemetry.get_meter().create_counter("c").add(1)


@pytest.mark.skipif(
    not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra"
)
def test_configure_telemetry_degrades_when_sdk_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """opentelemetry-api present but the SDK/exporter absent (a partial /
    manual install) → warn + return False, never crash. Exercises the
    ImportError arm inside configure_telemetry by blocking opentelemetry.sdk.trace."""
    blocked = "opentelemetry.sdk.trace"
    monkeypatch.delitem(sys.modules, blocked, raising=False)

    class _Blocker:
        def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
            if name == blocked:
                raise ModuleNotFoundError(blocked)
            return None

    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        assert telemetry.configure_telemetry(enabled=True, **_KW) is False
        assert telemetry._configured is False
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.pop(blocked, None)


def test_noop_shim_methods_are_all_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under a minimal install the hand-rolled no-op shim is the safety net,
    so exercise every method engine code may emit to. CI runs with [otel]
    installed, so force the otel-absent branch to reach the shim (the real
    OTel no-ops cover this path when the extra IS present)."""
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)
    tracer = telemetry.get_tracer()
    with tracer.start_as_current_span("s") as span:
        span.set_attribute(telemetry.DIKW_OP, "ingest")
        span.set_status("ok")
        span.record_exception(ValueError("boom"))
        span.add_event("evt")
        span.end()
    out_of_band = tracer.start_span("s2")
    out_of_band.set_attribute("k", "v")
    out_of_band.end()
    meter = telemetry.get_meter()
    meter.create_counter("dikw.test.counter").add(1, {"k": "v"})
    meter.create_up_down_counter("dikw.test.gauge").add(-1)
    meter.create_histogram("dikw.test.hist").record(1.5, {"k": "v"})


def test_configure_telemetry_disabled_does_not_latch() -> None:
    """A no-op (disabled) call must not block a later real activation — the
    idempotency latch flips only on actual activation."""
    assert telemetry.configure_telemetry(enabled=False, **_KW) is False
    assert telemetry._configured is False


@pytest.mark.skipif(
    not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra"
)
def test_configure_telemetry_activates_and_is_idempotent() -> None:
    first = telemetry.configure_telemetry(
        enabled=True,
        endpoint="http://localhost:4318",
        service_name="dikw-core-test",
        sample_ratio=1.0,
        version="0.0.0+test",
    )
    assert first is True
    assert telemetry._configured is True
    # second call is a no-op short-circuit, not a double-registration
    second = telemetry.configure_telemetry(
        enabled=True,
        endpoint="http://localhost:4318",
        service_name="dikw-core-test",
        sample_ratio=1.0,
        version="0.0.0+test",
    )
    assert second is True

    # an SDK provider is now globally registered
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    assert isinstance(trace.get_tracer_provider(), TracerProvider)

    # shutdown is safe and clears the latch via the reset fixture afterwards
    telemetry.shutdown_telemetry()


@pytest.mark.skipif(not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra")
def test_configure_telemetry_instruments_httpx_and_shutdown_unwinds() -> None:
    """Activation globally patches httpx (provider outbound spans + W3C
    traceparent); shutdown un-patches it so a fresh in-process lifespan
    re-activates from a clean state rather than double-wrapping."""
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    assert HTTPXClientInstrumentor()._is_instrumented_by_opentelemetry is False
    assert telemetry.configure_telemetry(enabled=True, **_KW) is True
    assert HTTPXClientInstrumentor()._is_instrumented_by_opentelemetry is True
    telemetry.shutdown_telemetry()
    assert HTTPXClientInstrumentor()._is_instrumented_by_opentelemetry is False


def test_shutdown_telemetry_safe_when_inactive() -> None:
    telemetry.shutdown_telemetry()  # no provider — must not raise


@pytest.mark.skipif(
    not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra"
)
def test_shutdown_resets_configured_latch() -> None:
    """Post-shutdown state must be honest: a stale ``_configured=True`` would
    make a fresh lifespan in the same process short-circuit and log
    ``telemetry=on`` while exporting nothing."""
    assert (
        telemetry.configure_telemetry(
            enabled=True,
            endpoint=None,
            service_name="dikw-core-test",
            sample_ratio=1.0,
            version="0.0.0+test",
        )
        is True
    )
    assert telemetry._configured is True
    telemetry.shutdown_telemetry()
    assert telemetry._configured is False
    assert telemetry._provider is None


@pytest.mark.skipif(
    not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra"
)
def test_configure_after_shutdown_reports_inactive_not_false_on() -> None:
    """OTel's set_tracer_provider is process-once. After a shutdown leaves the
    global provider stuck, a second activation can't re-register — it must
    report ``False`` (honest) rather than ``True`` with a dead provider."""
    kw = dict(  # noqa: C408
        enabled=True,
        endpoint=None,
        service_name="dikw-core-test",
        sample_ratio=1.0,
        version="0.0.0+test",
    )
    assert telemetry.configure_telemetry(**kw) is True  # type: ignore[arg-type]
    telemetry.shutdown_telemetry()  # global provider stays registered (set-once)
    # second activation: latch reset lets it proceed, but the provider install
    # is ignored by OTel → honest inactive, not a misleading telemetry=on
    assert telemetry.configure_telemetry(**kw) is False  # type: ignore[arg-type]
    assert telemetry._configured is False
    assert telemetry._provider is None
