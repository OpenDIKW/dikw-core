"""OpenTelemetry seam for dikw — accessors, attribute keys, SDK bootstrap.

dikw instruments against the OTel **API** (``get_tracer`` / ``get_meter``)
and lets the operator wire the SDK at the process entry. The whole stack is
an **optional** ``[otel]`` extra: with it installed, ``get_tracer()`` returns
a real tracer that is a no-op until :func:`configure_telemetry` registers a
provider; without it installed, the accessors return hand-rolled no-ops so
engine code can emit spans/metrics unconditionally and pay ~zero cost.

Layering: this module sits at the engine root and imports only
``opentelemetry`` (optional) + stdlib — never ``server`` / FastAPI. Engine
modules call the accessors + attribute-key constants; **only the entry point**
(the server lifespan) calls :func:`configure_telemetry`, exactly like
``init_logging`` is wired from the CLI / app factory. The FastAPI
auto-instrumentation lives in ``server/app.py`` (server code may import
FastAPI), gated on :data:`OTEL_AVAILABLE`, so the web-framework instrumentation
import never leaks into the engine.

This is the operator-facing observability channel; the user-facing channel is
the ``ProgressReporter`` event stream over NDJSON. Don't confuse the two.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter
    from opentelemetry.trace import Tracer

logger = logging.getLogger(__name__)

try:
    from opentelemetry import metrics as _otel_metrics
    from opentelemetry import trace as _otel_trace

    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in the no-otel install
    OTEL_AVAILABLE = False

# Instrumentation scope name — the tracer/meter "library" identity that shows
# up on every emitted span/metric so a backend can attribute them to dikw.
_INSTRUMENTATION_NAME: Final = "dikw_core"

# ---- semantic-convention attribute keys --------------------------------
# Reuse OTel-standard keys where they exist (``gen_ai.*``, ``http.*``,
# ``service.*``); dikw-specific dimensions live under the ``dikw.*`` namespace.
# Module-level ``Final`` constants (the repo's existing convention for stable
# string keys) so call sites are grep-able and typo-proof.
DIKW_LAYER: Final = "dikw.layer"  # data | info | knowledge | wisdom
DIKW_OP: Final = "dikw.op"  # ingest | synth | retrieve | eval | ...
DIKW_TASK_ID: Final = "dikw.task_id"  # the uuid4 from TaskManager.submit
DIKW_BASE_ID: Final = "dikw.base_id"  # _base_scope_id(root)
DIKW_SOURCE_PATH: Final = "dikw.source_path"
DIKW_CATEGORY: Final = "dikw.category"
DIKW_RETRIEVAL_LEG: Final = "dikw.retrieval.leg"  # bm25 | vector | graph | rrf
DIKW_EMBED_VERSION_ID: Final = "dikw.embed.version_id"


# ---- no-op fallbacks (used only when [otel] is NOT installed) -----------
# When otel IS installed, ``trace.get_tracer`` / ``metrics.get_meter`` already
# return no-op-until-provider objects, so these classes only run in a minimal
# install. They cover exactly the surface engine code emits to.


class _NoopSpan:
    def set_attribute(self, key: str, value: object) -> None: ...
    def set_status(self, *args: object, **kwargs: object) -> None: ...
    def record_exception(self, *args: object, **kwargs: object) -> None: ...
    def add_event(self, *args: object, **kwargs: object) -> None: ...
    def end(self, *args: object, **kwargs: object) -> None: ...


class _NoopTracer:
    @contextmanager
    def start_as_current_span(
        self, name: str, *args: object, **kwargs: object
    ) -> Iterator[_NoopSpan]:
        yield _NOOP_SPAN

    def start_span(self, name: str, *args: object, **kwargs: object) -> _NoopSpan:
        return _NOOP_SPAN


class _NoopInstrument:
    def add(self, amount: float, *args: object, **kwargs: object) -> None: ...
    def record(self, amount: float, *args: object, **kwargs: object) -> None: ...


class _NoopMeter:
    def create_counter(self, *args: object, **kwargs: object) -> _NoopInstrument:
        return _NOOP_INSTRUMENT

    def create_up_down_counter(self, *args: object, **kwargs: object) -> _NoopInstrument:
        return _NOOP_INSTRUMENT

    def create_histogram(self, *args: object, **kwargs: object) -> _NoopInstrument:
        return _NOOP_INSTRUMENT


_NOOP_SPAN = _NoopSpan()
_NOOP_INSTRUMENT = _NoopInstrument()
_NOOP_TRACER: Tracer = cast("Tracer", _NoopTracer())
_NOOP_METER: Meter = cast("Meter", _NoopMeter())


def get_tracer() -> Tracer:
    """Return the dikw tracer. No-op until :func:`configure_telemetry` runs
    (otel installed) or always no-op (otel absent)."""
    if OTEL_AVAILABLE:
        return _otel_trace.get_tracer(_INSTRUMENTATION_NAME)
    return _NOOP_TRACER


def get_meter() -> Meter:
    """Return the dikw meter. Same no-op semantics as :func:`get_tracer`."""
    if OTEL_AVAILABLE:
        return _otel_metrics.get_meter(_INSTRUMENTATION_NAME)
    return _NOOP_METER


# ---- SDK bootstrap (entry-point only) ----------------------------------

_configured = False
# Holds the SDK TracerProvider once activated so the server lifespan can flush
# + shut it down cleanly. Typed loosely (object) to avoid importing the SDK
# type at module scope — the SDK lives behind the optional extra.
_provider: object | None = None


def _otel_sdk_disabled() -> bool:
    """Honour the standard ``OTEL_SDK_DISABLED`` kill-switch."""
    return os.getenv("OTEL_SDK_DISABLED", "").strip().lower() in ("1", "true", "yes")


def configure_telemetry(
    *,
    enabled: bool,
    endpoint: str | None,
    service_name: str,
    sample_ratio: float,
    version: str,
) -> bool:
    """Register the OTel SDK providers + OTLP/HTTP exporter. Idempotent.

    Returns ``True`` when telemetry is activated, ``False`` for every no-op
    path (disabled in config, ``[otel]`` not installed, or ``OTEL_SDK_DISABLED``
    set). Called once from the server lifespan after cfg load — never from
    engine code.

    ``endpoint`` is the OTLP/HTTP base URL (e.g. ``http://collector:4318``);
    the per-signal path (``/v1/traces``) is appended automatically. Left
    ``None``, the exporter falls back to the standard
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var.
    """
    global _configured, _provider
    if _configured:
        return True
    if not enabled or not OTEL_AVAILABLE or _otel_sdk_disabled():
        return False

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    except ImportError as e:
        # opentelemetry-api present but the SDK / OTLP exporter is not — a
        # non-standard partial install (the ``[otel]`` extra bundles them
        # together). Telemetry must never crash the server: degrade to no-op.
        logger.warning(
            "telemetry enabled but the OTel SDK/exporter is not installed (%s); "
            "install the [otel] extra. Continuing without telemetry.",
            e,
        )
        return False

    resource = Resource.create(
        {"service.name": service_name, "service.version": version}
    )
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(root=TraceIdRatioBased(sample_ratio)),
    )
    exporter = (
        OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")
        if endpoint
        else OTLPSpanExporter()
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    _otel_trace.set_tracer_provider(provider)

    _provider = provider
    _configured = True
    return True


def shutdown_telemetry() -> None:
    """Flush + shut down the SDK providers. Safe to call when inactive."""
    global _provider
    provider = _provider
    if provider is None:
        return
    shutdown = getattr(provider, "shutdown", None)
    if callable(shutdown):
        shutdown()
    _provider = None


def reset_telemetry_for_testing() -> None:
    """Test-only: clear the idempotency latch + reset the global provider.

    Lets a test exercise both the no-op and the activated paths in one
    process. Best-effort on the otel internals — wrapped so SDK version
    drift can only cost test isolation, never production behaviour.
    """
    global _configured, _provider
    shutdown_telemetry()
    _configured = False
    _provider = None
    if OTEL_AVAILABLE:
        try:
            _otel_trace._TRACER_PROVIDER_SET_ONCE = (
                _otel_trace._TRACER_PROVIDER_SET_ONCE.__class__()
            )
            _otel_trace._TRACER_PROVIDER = None
        except Exception:  # pragma: no cover - internal-API drift guard
            pass


__all__ = [
    "DIKW_BASE_ID",
    "DIKW_CATEGORY",
    "DIKW_EMBED_VERSION_ID",
    "DIKW_LAYER",
    "DIKW_OP",
    "DIKW_RETRIEVAL_LEG",
    "DIKW_SOURCE_PATH",
    "DIKW_TASK_ID",
    "OTEL_AVAILABLE",
    "configure_telemetry",
    "get_meter",
    "get_tracer",
    "reset_telemetry_for_testing",
    "shutdown_telemetry",
]
