"""PR2c: env-only client OTel bootstrap + W3C traceparent propagation.

The ``dikw client`` CLI has no ``dikw.yml`` (see ``TelemetryConfig`` docstring),
so its telemetry is driven purely by the standard ``OTEL_*`` env vars. These
pin that :func:`configure_client_telemetry_from_env`:

* activates ONLY when an OTLP endpoint env is set (so a plain ``dikw client``
  invocation pays zero cost), the ``[otel]`` extra is present, and
  ``OTEL_SDK_DISABLED`` is unset;
* reads ``OTEL_SERVICE_NAME`` for the resource identity;
* globally instruments httpx so an outbound ``Transport`` request carries a
  ``traceparent`` header — the wire half of client→server trace stitching.

The ``_root`` gating (only the ``client`` subgroup bootstraps) is covered in
``tests/test_cli.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from dikw_core import telemetry
from dikw_core.client.config import ClientConfig
from dikw_core.client.transport import Transport

_ENV_VARS = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_SDK_DISABLED",
    "OTEL_SERVICE_NAME",
)


@pytest.fixture(autouse=True)
def _reset_telemetry_and_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clean OTEL_* env + telemetry latch around every test so neither the
    host environment nor a prior test leaks an endpoint / disabled flag in."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    telemetry.reset_telemetry_for_testing()
    yield
    telemetry.reset_telemetry_for_testing()


_V = "0.0.0+test"


# ---- activation predicate ----------------------------------------------


def test_noop_without_any_endpoint_env() -> None:
    """No OTEL_* endpoint set → no-op, and the latch stays clear so a real
    activation later in the same process isn't blocked."""
    assert telemetry.configure_client_telemetry_from_env(version=_V) is False
    assert telemetry._configured is False


def test_noop_when_otel_sdk_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert telemetry.configure_client_telemetry_from_env(version=_V) is False
    assert telemetry._configured is False


def test_noop_when_otel_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimal install: endpoint set but the [otel] extra is missing → no-op."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)
    assert telemetry.configure_client_telemetry_from_env(version=_V) is False
    assert telemetry._configured is False


@pytest.mark.skipif(not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra")
def test_activates_from_generic_endpoint_env_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.trace import TracerProvider

    assert HTTPXClientInstrumentor()._is_instrumented_by_opentelemetry is False
    assert telemetry.configure_client_telemetry_from_env(version=_V) is True
    assert telemetry._configured is True
    # global SDK provider registered + httpx globally patched for traceparent
    from opentelemetry import trace

    provider_before = trace.get_tracer_provider()
    assert isinstance(provider_before, TracerProvider)
    assert HTTPXClientInstrumentor()._is_instrumented_by_opentelemetry is True
    # second call short-circuits on the latch — same provider, no re-instrument
    assert telemetry.configure_client_telemetry_from_env(version=_V) is True
    assert trace.get_tracer_provider() is provider_before
    assert HTTPXClientInstrumentor()._is_instrumented_by_opentelemetry is True


@pytest.mark.skipif(not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra")
def test_reports_inactive_when_a_provider_is_already_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OTel's ``set_tracer_provider`` is process-once. If a provider is already
    globally registered (here: a prior server activation, then ``shutdown`` —
    which leaves the global provider set-once), the client bootstrap must LOSE
    the ``_adopt_provider`` race and report inactive — NOT flip the latch,
    register a provider, or instrument httpx. Client twin of
    ``test_configure_after_shutdown_reports_inactive_not_false_on``.
    """
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    assert (
        telemetry.configure_telemetry(
            enabled=True,
            endpoint=None,
            service_name="dikw-core-test",
            sample_ratio=1.0,
            version=_V,
        )
        is True
    )
    telemetry.shutdown_telemetry()  # global provider stays registered (set-once)
    assert telemetry._configured is False
    assert HTTPXClientInstrumentor()._is_instrumented_by_opentelemetry is False

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    # the adopt race is lost → honest inactive, no provider, httpx untouched
    assert telemetry.configure_client_telemetry_from_env(version=_V) is False
    assert telemetry._configured is False
    assert telemetry._provider is None
    assert HTTPXClientInstrumentor()._is_instrumented_by_opentelemetry is False


@pytest.mark.skipif(not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra")
def test_degrades_when_sdk_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """opentelemetry-api present but the SDK absent (a non-standard partial
    install) → warn + return False, never crash the CLI ``_root`` entry.
    Client twin of ``test_configure_telemetry_degrades_when_sdk_import_fails``;
    exercises the ImportError arm that ``test_noop_when_otel_absent`` (which
    returns before the try-import) never reaches."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
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
        assert telemetry.configure_client_telemetry_from_env(version=_V) is False
        assert telemetry._configured is False
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.pop(blocked, None)


@pytest.mark.skipif(not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra")
def test_activates_from_traces_specific_endpoint_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-signal ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` alone also counts as
    a configured endpoint (the SDK exporter reads it directly)."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces"
    )
    assert telemetry.configure_client_telemetry_from_env(version=_V) is True
    assert telemetry._configured is True


@pytest.mark.skipif(not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra")
def test_resource_service_name_defaults_and_honours_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "my-cli")
    assert telemetry.configure_client_telemetry_from_env(version=_V) is True
    from opentelemetry import trace

    resource = trace.get_tracer_provider().resource  # type: ignore[attr-defined]
    assert resource.attributes["service.name"] == "my-cli"
    assert resource.attributes["service.version"] == _V


@pytest.mark.skipif(not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra")
def test_default_service_name_is_dikw_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    assert telemetry.configure_client_telemetry_from_env(version=_V) is True
    from opentelemetry import trace

    resource = trace.get_tracer_provider().resource  # type: ignore[attr-defined]
    assert resource.attributes["service.name"] == "dikw-client"


# ---- W3C traceparent propagation through Transport ---------------------


async def test_transport_uses_the_instrumented_transport_class() -> None:
    """``Transport`` must build its client on httpx's default
    ``AsyncHTTPTransport`` — the exact class :func:`configure_client_telemetry_from_env`'s
    global ``HTTPXClientInstrumentor().instrument()`` patches at
    ``handle_async_request``. A custom transport here would silently bypass the
    instrumentation and drop traceparent propagation, so guard it. (Pure
    structural check — no ``[otel]`` extra needed.)
    """
    cfg = ClientConfig(server_url="http://test", token=None)
    async with Transport.from_config(cfg) as t:
        assert isinstance(t._client._transport, httpx.AsyncHTTPTransport)


@pytest.mark.skipif(not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra")
async def test_instrumented_transport_injects_traceparent_header() -> None:
    """An instrumented client issuing a request through ``Transport`` carries a
    well-formed W3C ``traceparent`` whose trace-id matches the recorded client
    span — the propagation the server's FastAPI instrumentation adopts as the
    parent.

    Drives the SAME ``_handle_async_request_wrapper`` injection path that
    :func:`configure_client_telemetry_from_env`'s global ``.instrument()`` turns
    on, but via per-instance ``instrument_client`` over a ``MockTransport`` so
    the request never hits the network and the spans land in an in-memory
    exporter (the global ``.instrument()`` patches ``AsyncHTTPTransport`` at the
    class level, which a mock transport would bypass — see the structural guard
    above that pins production onto that class).
    """
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["traceparent"] = request.headers.get("traceparent")
        return httpx.Response(200, json={"ok": True})

    cfg = ClientConfig(server_url="http://test", token=None)
    client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    HTTPXClientInstrumentor().instrument_client(client, tracer_provider=provider)
    try:
        async with Transport.from_config(cfg, client=client) as t:
            await t.get_json("/v1/health")
    finally:
        provider.shutdown()

    tp = captured["traceparent"]
    assert tp is not None, "instrumented client must inject a traceparent header"
    parts = tp.split("-")
    assert len(parts) == 4, f"malformed traceparent: {tp!r}"
    version, trace_id, span_id, flags = parts
    assert version == "00"
    assert len(trace_id) == 32
    assert len(span_id) == 16
    # sampled bit set (0x01); the byte may also carry 0x02 (W3C level-2
    # random-trace-id flag), so mask rather than compare the whole byte.
    assert int(flags, 16) & 0x01, f"traceparent not sampled: {flags!r}"

    finished = exporter.get_finished_spans()
    assert finished, "expected the auto-created httpx client span"
    assert f"{finished[-1].context.trace_id:032x}" == trace_id
