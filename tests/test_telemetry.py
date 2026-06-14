"""PR1 foundations: the telemetry seam is no-op-safe + entry-gated.

Proves the contract the rest of the OTel arc builds on: accessors are always
usable (otel installed or not), and ``configure_telemetry`` activates only on
the intended path (enabled + ``[otel]`` present + not ``OTEL_SDK_DISABLED``),
staying a zero-side-effect no-op otherwise.
"""

from __future__ import annotations

from collections.abc import Iterator

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
