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

import asyncio
import functools
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Final, TypeVar, cast

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter
    from opentelemetry.trace import Span, SpanContext, Tracer

logger = logging.getLogger(__name__)

try:
    from opentelemetry import context as _otel_context
    from opentelemetry import metrics as _otel_metrics
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import Link as _Link
    from opentelemetry.trace import Status as _Status
    from opentelemetry.trace import StatusCode as _StatusCode

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
DIKW_RETRIEVAL_LEG: Final = "dikw.retrieval.leg"  # bm25 | vector | asset | graph
DIKW_EMBED_VERSION_ID: Final = "dikw.embed.version_id"
DIKW_CANCELLED: Final = "dikw.cancelled"  # task/span ended by cooperative cancel
# Engine op-span detail attributes (PR2b — set on the per-op spans opened via
# :func:`op_span`; trace-tree structure, not aggregate metrics — those land in
# the metrics PR).
DIKW_RETRIEVE_LIMIT: Final = "dikw.retrieve.limit"
DIKW_RETRIEVE_HIT_COUNT: Final = "dikw.retrieve.hit_count"
DIKW_LEG_HIT_COUNT: Final = "dikw.retrieve.leg.hit_count"

# gen_ai.* semantic-convention keys (OTel standard — NOT under the dikw.*
# namespace). Set on the per-call provider span by :func:`gen_ai_span`.
GEN_AI_OPERATION_NAME: Final = "gen_ai.operation.name"  # chat | embeddings
GEN_AI_SYSTEM: Final = "gen_ai.system"  # openai | anthropic | gitee
GEN_AI_REQUEST_MODEL: Final = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS: Final = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE: Final = "gen_ai.request.temperature"
GEN_AI_RESPONSE_FINISH_REASONS: Final = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS: Final = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS: Final = "gen_ai.usage.output_tokens"
# Anthropic-only prompt-cache accounting (vendor-namespaced; absent elsewhere).
GEN_AI_CACHE_READ_INPUT_TOKENS: Final = "gen_ai.anthropic.cache_read_input_tokens"
GEN_AI_CACHE_CREATION_INPUT_TOKENS: Final = "gen_ai.anthropic.cache_creation_input_tokens"
# Metric-only attribute keys (set on the GenAI metric data points emitted by
# :func:`_record_gen_ai_metrics`, NOT on spans). ``gen_ai.token.type`` splits the
# token-usage histogram into input/output series; ``error.type`` (OTel standard)
# tags the duration histogram on a failed call.
GEN_AI_TOKEN_TYPE: Final = "gen_ai.token.type"  # input | output
ERROR_TYPE: Final = "error.type"  # exception class name on a failed operation

# ---- metric instrument identities (OTel GenAI semconv) -----------------
# Histogram NAMES (not attribute keys) — kept private; the meter creates these
# once in :func:`_gen_ai_instruments`. Explicit bucket boundaries follow the
# semconv's advice: the SDK default histogram buckets top out at 10k, useless
# for token counts (tens of thousands) and sub-second-to-minute LLM latencies.
_METRIC_GEN_AI_TOKEN_USAGE: Final = "gen_ai.client.token.usage"
_METRIC_GEN_AI_OP_DURATION: Final = "gen_ai.client.operation.duration"
_GEN_AI_TOKEN_BUCKETS: Final = [
    1, 4, 16, 64, 256, 1024, 4096, 16384, 65536,
    262144, 1048576, 4194304, 16777216, 67108864,
]  # fmt: skip
_GEN_AI_DURATION_BUCKETS: Final = [
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28,
    2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
]  # fmt: skip


# ---- no-op fallbacks (used only when [otel] is NOT installed) -----------
# When otel IS installed, ``trace.get_tracer`` / ``metrics.get_meter`` already
# return no-op-until-provider objects, so these classes only run in a minimal
# install. They cover exactly the surface engine code emits to.


class _NoopSpan:
    def set_attribute(self, key: str, value: object) -> None:
        pass

    def set_status(self, *args: object, **kwargs: object) -> None:
        pass

    def record_exception(self, *args: object, **kwargs: object) -> None:
        pass

    def add_event(self, *args: object, **kwargs: object) -> None:
        pass

    def end(self, *args: object, **kwargs: object) -> None:
        pass


class _NoopTracer:
    @contextmanager
    def start_as_current_span(
        self, name: str, *args: object, **kwargs: object
    ) -> Iterator[_NoopSpan]:
        yield _NOOP_SPAN

    def start_span(self, name: str, *args: object, **kwargs: object) -> _NoopSpan:
        return _NOOP_SPAN


class _NoopInstrument:
    def add(self, amount: float, *args: object, **kwargs: object) -> None:
        pass

    def record(self, amount: float, *args: object, **kwargs: object) -> None:
        pass


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


# ---- span helpers ------------------------------------------------------
# Engine (providers) + server (task subsystem) call these; the OTel-specific
# machinery (Link / Status / context, root vs child span shape) lives here so
# callers import ``telemetry`` only and never ``opentelemetry`` directly — that
# keeps the ``[otel]`` extra optional (these degrade to no-ops when it's
# absent) and the no-op + active paths sharing one gate.


class _GenAISpanHandle:
    """Lets a provider stamp response attributes on its in-flight gen_ai span.

    Wraps a real span when telemetry is active, or :data:`_NOOP_SPAN` otherwise
    (whose ``set_attribute`` is a no-op) — one class covers both paths because
    ``set_response`` only ever calls ``set_attribute``. Also stashes the reported
    ``usage`` so the enclosing :func:`gen_ai_span` can record the token-usage
    metric at span close (the metric needs the same data the span attributes do).
    """

    __slots__ = ("_span", "usage")

    def __init__(self, span: Span) -> None:
        self._span = span
        self.usage: dict[str, int] | None = None

    def set_response(
        self,
        *,
        finish_reason: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        span = self._span
        if finish_reason is not None:
            span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, (finish_reason,))
        if usage:
            self.usage = usage
            if "input_tokens" in usage:
                span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, int(usage["input_tokens"]))
            if "output_tokens" in usage:
                span.set_attribute(
                    GEN_AI_USAGE_OUTPUT_TOKENS, int(usage["output_tokens"])
                )
            # Anthropic-only; other providers never populate these keys.
            if usage.get("cache_read_input_tokens"):
                span.set_attribute(
                    GEN_AI_CACHE_READ_INPUT_TOKENS,
                    int(usage["cache_read_input_tokens"]),
                )
            if usage.get("cache_creation_input_tokens"):
                span.set_attribute(
                    GEN_AI_CACHE_CREATION_INPUT_TOKENS,
                    int(usage["cache_creation_input_tokens"]),
                )


@contextmanager
def gen_ai_span(
    *,
    operation: str,
    system: str,
    model: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> Iterator[_GenAISpanHandle]:
    """Wrap one LLM/embedding provider call in a ``gen_ai.*`` span.

    Use as ``with gen_ai_span(...) as span:`` around the provider's request,
    calling ``span.set_response(finish_reason=..., usage=...)`` before the
    terminal event. The span name follows the GenAI semconv (``<op> <model>``).
    An ``asyncio.CancelledError`` is recorded as a cancel (``dikw.cancelled``),
    NOT an error, so cancels don't pollute error-rate dashboards; a
    ``GeneratorExit`` (a consumer breaking out of / ``aclose``-ing a wrapped
    generator early) is likewise a graceful terminal — re-raised without a
    status so an abandoned stream isn't mis-reported as an LLM failure; any
    other exception sets ``StatusCode.ERROR`` + records it. No-op when telemetry
    is inactive. The underlying httpx wire call nests as a child via the active
    OTel context (httpx auto-instrumentation, wired in
    :func:`configure_telemetry`).
    """
    if not OTEL_AVAILABLE:
        yield _GenAISpanHandle(cast("Span", _NOOP_SPAN))
        return
    tracer = _otel_trace.get_tracer(_INSTRUMENTATION_NAME)
    start = time.perf_counter()
    error_type: str | None = None
    with tracer.start_as_current_span(
        f"{operation} {model}",
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, operation)
        span.set_attribute(GEN_AI_SYSTEM, system)
        span.set_attribute(GEN_AI_REQUEST_MODEL, model)
        if max_tokens is not None:
            span.set_attribute(GEN_AI_REQUEST_MAX_TOKENS, int(max_tokens))
        if temperature is not None:
            span.set_attribute(GEN_AI_REQUEST_TEMPERATURE, float(temperature))
        handle = _GenAISpanHandle(span)
        try:
            yield handle
        except asyncio.CancelledError:
            span.set_attribute(DIKW_CANCELLED, True)
            raise
        except GeneratorExit:
            # A consumer that breaks out of (or aclose()s) a wrapped generator
            # early throws GeneratorExit in at the suspended ``yield``. It is a
            # graceful close, NOT a failure — re-raise leaving the status UNSET
            # so an abandoned stream isn't recorded as an LLM error. (Must
            # precede ``except BaseException``: GeneratorExit is a BaseException
            # but not an Exception or CancelledError.)
            raise
        except BaseException as exc:
            error_type = type(exc).__name__
            span.record_exception(exc)
            span.set_status(_Status(_StatusCode.ERROR, str(exc)))
            raise
        else:
            span.set_status(_Status(_StatusCode.OK))
        finally:
            # Record the GenAI metrics from inside the still-open span so the
            # SDK can attach an exemplar back to this trace. ``error_type`` is
            # set only on the hard-error arm — a cancel / GeneratorExit is a
            # graceful terminal and records duration WITHOUT an error tag, so it
            # doesn't pollute failure-rate dashboards (mirrors the span status).
            _record_gen_ai_metrics(
                operation=operation,
                system=system,
                model=model,
                usage=handle.usage,
                duration_seconds=time.perf_counter() - start,
                error_type=error_type,
            )


async def trace_llm_stream[StreamEventT](
    events: AsyncIterator[StreamEventT],
    *,
    system: str,
    model: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> AsyncIterator[StreamEventT]:
    """Wrap a provider's ``complete_stream`` generator in a ``chat`` gen_ai span.

    Opens the span around iteration so the provider's SDK call (run lazily on
    first pull, inside this span) and its outbound httpx wire span nest under
    it, reads token usage + finish_reason off the terminal ``done`` event, and
    propagates cancel/error exactly like :func:`gen_ai_span`. Providers call
    ``return trace_llm_stream(_gen(), ...)`` — the generator body is untouched.
    Duck-typed on the event (``.type`` / ``.finish_reason`` / ``.usage``) so
    this module needs no import from ``providers`` (which would be a cycle).

    Contract: the returned generator must be driven to completion (or
    ``aclose``-d) in the SAME asyncio task that started its first ``__anext__``
    — :func:`gen_ai_span` attaches the OTel context (a ``contextvars`` Token)
    on first pull, and closing it from a different task would log
    "Failed to detach context". Every in-tree consumer is a single-coroutine
    ``async for`` drain, so this holds today.
    """
    with gen_ai_span(
        operation="chat",
        system=system,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    ) as span:
        async for event in events:
            if getattr(event, "type", None) == "done":
                span.set_response(
                    finish_reason=getattr(event, "finish_reason", None),
                    usage=getattr(event, "usage", None),
                )
            yield event


class _TaskSpanHandle:
    """Status sink for a background task span (see :func:`task_span`).

    The task runner swallows ``asyncio.CancelledError`` (a graceful terminal),
    so the span CM cannot infer the outcome from a propagating exception — the
    caller sets it explicitly via ``ok()`` / ``cancelled()`` / ``record_error()``.
    Guards the ``Status`` construction on :data:`OTEL_AVAILABLE` so the same
    class is reusable on the no-op path.
    """

    __slots__ = ("_span",)

    def __init__(self, span: Span) -> None:
        self._span = span

    def ok(self) -> None:
        if OTEL_AVAILABLE:
            self._span.set_status(_Status(_StatusCode.OK))

    def cancelled(self) -> None:
        # A user-requested cancel is a graceful terminal, not a failure —
        # mark it (queryable) but leave the status UNSET, not ERROR.
        self._span.set_attribute(DIKW_CANCELLED, True)

    def record_error(self, exc: BaseException) -> None:
        if OTEL_AVAILABLE:
            self._span.record_exception(exc)
            self._span.set_status(_Status(_StatusCode.ERROR, str(exc)))


def capture_otel_context() -> SpanContext | None:
    """Capture the current span's context for later linking from a detached
    task. Returns the ``SpanContext`` or ``None`` when there's no valid active
    span / the ``[otel]`` extra is absent.

    Called at task-submit time (inside the HTTP request span); the returned
    context is a lightweight immutable value safe to hold across the
    ``asyncio.create_task`` boundary — it does NOT keep the request span open.
    """
    if not OTEL_AVAILABLE:
        return None
    span_context = _otel_trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return None
    return span_context


@contextmanager
def task_span(
    op: str,
    *,
    task_id: str,
    link: SpanContext | None = None,
    base_id: str | None = None,
) -> Iterator[_TaskSpanHandle]:
    """Open the root span for one background task, kept current across the run.

    The task runs in a detached ``asyncio.create_task`` coroutine that outlives
    the HTTP request span, so this is a NEW ROOT span (parent ignored) LINKED
    back to the submitting request's context — the OTel idiom for
    request-triggered fire-and-forget work. Held current for the whole runner
    await so downstream engine/provider spans nest under it. No-op when
    telemetry is inactive.
    """
    if not OTEL_AVAILABLE:
        yield _TaskSpanHandle(cast("Span", _NOOP_SPAN))
        return
    tracer = _otel_trace.get_tracer(_INSTRUMENTATION_NAME)
    links = [_Link(link)] if link is not None else None
    # Force a root span: ``create_task`` copied the request context, but the
    # request span has already ended — link, don't parent.
    span = tracer.start_span(
        f"dikw.task.{op}", context=_otel_context.Context(), links=links
    )
    span.set_attribute(DIKW_OP, op)
    span.set_attribute(DIKW_TASK_ID, task_id)
    if base_id:
        span.set_attribute(DIKW_BASE_ID, base_id)
    with _otel_trace.use_span(
        span, end_on_exit=True, record_exception=False, set_status_on_exception=False
    ):
        yield _TaskSpanHandle(span)


@contextmanager
def op_span(
    name: str, *, attributes: dict[str, str | int | float | bool] | None = None
) -> Iterator[Span]:
    """Open a current engine-operation span (e.g. ``dikw.ingest``, ``dikw.synth``).

    The generic seam for the engine op-level trace tree: opened as a normal
    ``with op_span(...) as span:`` around an existing async/sync body, it stays
    current across ``await`` points in the SAME task (contextvars propagate), so
    nested ``op_span`` / provider spans become its children. Set open-time
    dimensions via ``attributes``; set post-hoc detail (hit counts, …) by calling
    ``span.set_attribute(...)`` inside the block.

    Outcome handling mirrors :func:`gen_ai_span`: an ``asyncio.CancelledError``
    is a graceful cancel (flagged ``dikw.cancelled``, status left UNSET, not
    ERROR); a ``GeneratorExit`` (early close of a wrapping generator) is likewise
    graceful; any other exception sets ``StatusCode.ERROR`` + records it; a clean
    exit sets ``OK``. No-op (zero spans) when telemetry is inactive.

    Engine op spans do NOT set ``dikw.base_id`` — they nest under the task root
    span (:func:`task_span`), which carries it, so it is inherited via the trace.
    """
    if not OTEL_AVAILABLE:
        yield cast("Span", _NOOP_SPAN)
        return
    tracer = _otel_trace.get_tracer(_INSTRUMENTATION_NAME)
    with tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except asyncio.CancelledError:
            span.set_attribute(DIKW_CANCELLED, True)
            raise
        except GeneratorExit:
            raise
        except BaseException as exc:
            span.record_exception(exc)
            span.set_status(_Status(_StatusCode.ERROR, str(exc)))
            raise
        else:
            span.set_status(_Status(_StatusCode.OK))


_AsyncFn = TypeVar("_AsyncFn", bound=Callable[..., Awaitable[Any]])


def traced_op(
    name: str, *, attributes: dict[str, str | int | float | bool] | None = None
) -> Callable[[_AsyncFn], _AsyncFn]:
    """Decorator: wrap an async engine-op entry in an :func:`op_span`.

    The zero-re-indent way to give a large facade coroutine (``ingest``,
    ``synthesize``, ``lint_propose``, …) its op-level span: the span opens
    around the whole awaited call and stays current so the op's provider /
    nested spans become its children. Cancel / error / OK outcome handling is
    :func:`op_span`'s. No-op when telemetry is inactive. For ops that also want
    a post-hoc detail attribute off the result (e.g. ``retrieve``'s hit count),
    open :func:`op_span` inline instead so the result is in scope.
    """

    def decorate(fn: _AsyncFn) -> _AsyncFn:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with op_span(name, attributes=attributes):
                return await fn(*args, **kwargs)

        return cast("_AsyncFn", wrapper)

    return decorate


# ---- SDK bootstrap (entry-point only) ----------------------------------

_configured = False
# Holds the SDK TracerProvider once activated so the server lifespan can flush
# + shut it down cleanly. Typed loosely (object) to avoid importing the SDK
# type at module scope — the SDK lives behind the optional extra.
_provider: object | None = None
# The SDK MeterProvider (server-side metrics). Set ONLY when dikw's provider
# won registration in :func:`configure_telemetry`; the client bootstrap leaves
# it None (the remote client makes no LLM/embedding calls — there's nothing to
# meter). Doubles as the gate for :func:`_gen_ai_instruments`: instruments are
# created only when our provider is the global one, so gen_ai metrics never leak
# to a foreign meter provider.
_meter_provider: object | None = None
# Lazily-created (once) GenAI metric instruments, cached so each record is a
# cheap lookup. Typed ``Any`` because the SDK Histogram type lives behind the
# optional extra. Cleared by shutdown / reset so a fresh lifespan re-binds.
_gen_ai_token_usage: Any = None
_gen_ai_op_duration: Any = None


def _gen_ai_instruments() -> tuple[Any, Any] | None:
    """Return the (token-usage, operation-duration) histograms, or ``None`` when
    metrics are inactive (no dikw meter provider wired).

    Creating an instrument is meant to be done once and reused; gate on our own
    :data:`_meter_provider` (not the global, which a foreign auto-instrumentation
    could own) so metrics flow only to dikw's exporter. The instruments bind to
    the active meter on first use, after the provider is registered.
    """
    global _gen_ai_token_usage, _gen_ai_op_duration
    if _meter_provider is None:
        return None
    if _gen_ai_op_duration is None:
        meter = get_meter()
        _gen_ai_token_usage = meter.create_histogram(
            _METRIC_GEN_AI_TOKEN_USAGE,
            unit="{token}",
            description="Number of input/output tokens used per GenAI request.",
        )
        _gen_ai_op_duration = meter.create_histogram(
            _METRIC_GEN_AI_OP_DURATION,
            unit="s",
            description="Duration of a GenAI client operation (chat / embeddings).",
        )
    return _gen_ai_token_usage, _gen_ai_op_duration


def _record_gen_ai_metrics(
    *,
    operation: str,
    system: str,
    model: str,
    usage: dict[str, int] | None,
    duration_seconds: float,
    error_type: str | None,
) -> None:
    """Emit the OTel GenAI metrics for one provider call: always the operation
    duration (tagged ``error.type`` on failure), plus an input + output token
    point when the call reported usage. No-op when metrics are inactive."""
    instruments = _gen_ai_instruments()
    if instruments is None:
        return
    token_usage, op_duration = instruments
    base = {
        GEN_AI_OPERATION_NAME: operation,
        GEN_AI_SYSTEM: system,
        GEN_AI_REQUEST_MODEL: model,
    }
    duration_attrs = base if error_type is None else {**base, ERROR_TYPE: error_type}
    op_duration.record(duration_seconds, duration_attrs)
    if usage:
        if "input_tokens" in usage:
            token_usage.record(
                int(usage["input_tokens"]), {**base, GEN_AI_TOKEN_TYPE: "input"}
            )
        if "output_tokens" in usage:
            token_usage.record(
                int(usage["output_tokens"]), {**base, GEN_AI_TOKEN_TYPE: "output"}
            )


def _otel_sdk_disabled() -> bool:
    """Honour the standard ``OTEL_SDK_DISABLED`` kill-switch."""
    return os.getenv("OTEL_SDK_DISABLED", "").strip().lower() in ("1", "true", "yes")


def telemetry_should_activate(enabled: bool) -> bool:
    """Whether telemetry should be wired: requested in config AND the ``[otel]``
    extra is installed AND not killed via ``OTEL_SDK_DISABLED``.

    Shared by :func:`configure_telemetry` (the SDK bootstrap) and the server's
    build-time decision to wire FastAPI instrumentation, so the two never
    diverge — the HTTP-span middleware must not be added when telemetry is off
    (it can't be added later in the lifespan), else a disabled server would
    still pay middleware cost and could emit spans to a foreign global provider.
    """
    return enabled and OTEL_AVAILABLE and not _otel_sdk_disabled()


def _adopt_provider(provider: Any) -> bool:
    """Register ``provider`` as the global ``TracerProvider``, honouring OTel's
    process-once ``set_tracer_provider``.

    Returns ``True`` when dikw's provider won registration. When a provider is
    already registered (a prior in-process bootstrap, or external
    auto-instrumentation), ``set_tracer_provider`` silently no-ops — so release
    the built provider's background exporter thread and return ``False`` rather
    than claim success with a provider that was never installed (spans would
    flow to the other one and dikw's exporter would receive nothing). Shared by
    the server (:func:`configure_telemetry`) and client
    (:func:`configure_client_telemetry_from_env`) bootstraps; each caller then
    instruments httpx + flips the latch only on a ``True`` return.
    """
    _otel_trace.set_tracer_provider(provider)
    if _otel_trace.get_tracer_provider() is provider:
        return True
    shutdown = getattr(provider, "shutdown", None)
    if callable(shutdown):
        shutdown()
    logger.warning(
        "telemetry not activated: a TracerProvider is already registered in "
        "this process; skipping dikw OTel bootstrap"
    )
    return False


def _adopt_meter_provider(provider: Any) -> bool:
    """Register ``provider`` as the global ``MeterProvider``, honouring OTel's
    process-once ``set_meter_provider`` — the metrics analogue of
    :func:`_adopt_provider`.

    Returns ``True`` when dikw's meter provider won registration. If one is
    already registered (``set_meter_provider`` then silently no-ops), shut down
    the orphan's ``PeriodicExportingMetricReader`` thread and return ``False`` —
    metrics degrade rather than leak to a foreign provider, while tracing (which
    won its own race above) stays up. Realistically this loses only if external
    auto-instrumentation pre-registered a meter provider, in which case the
    tracer race above would already have failed and we'd never reach here.
    """
    _otel_metrics.set_meter_provider(provider)
    if _otel_metrics.get_meter_provider() is provider:
        return True
    shutdown = getattr(provider, "shutdown", None)
    if callable(shutdown):
        shutdown()
    logger.warning(
        "metrics not activated: a MeterProvider is already registered in this "
        "process; dikw traces are still active but metrics are disabled"
    )
    return False


def configure_telemetry(
    *,
    enabled: bool,
    endpoint: str | None,
    service_name: str,
    sample_ratio: float,
    version: str,
) -> bool:
    """Register the OTel SDK providers (tracer + meter) + OTLP/HTTP exporters.
    Idempotent.

    Returns ``True`` when telemetry is activated, ``False`` for every no-op
    path (disabled in config, ``[otel]`` not installed, or ``OTEL_SDK_DISABLED``
    set). Called once from the server lifespan after cfg load — never from
    engine code.

    ``endpoint`` is the OTLP/HTTP base URL (e.g. ``http://collector:4318``).
    When set, dikw appends the per-signal ``/v1/traces`` and ``/v1/metrics``
    paths; when ``None``, the exporters are constructed bare and the SDK's own
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` env handling applies (which appends the
    paths itself). ``sample_ratio`` governs trace sampling only — metrics are
    not sampled. The tracer race is decisive: if dikw loses it the whole
    bootstrap reports inactive; a lost meter race (near-impossible once the
    tracer race is won) degrades metrics alone and keeps traces up.
    """
    global _configured, _provider, _meter_provider
    if _configured:
        return True
    if not telemetry_should_activate(enabled):
        return False

    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.metrics.view import (
            ExplicitBucketHistogramAggregation,
            View,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    except ImportError as e:
        # opentelemetry-api present but the SDK / OTLP exporter / instrumentation
        # is not — a non-standard partial install (the ``[otel]`` extra bundles
        # them together). Telemetry must never crash the server: degrade to no-op.
        logger.warning(
            "telemetry enabled but the OTel SDK/exporter/instrumentation is not "
            "installed (%s); install the [otel] extra. Continuing without telemetry.",
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
    if not _adopt_provider(provider):
        return False

    # Global httpx patch — covers every server-side provider httpx client
    # (built lazily on first synth/embed, always after this point) for outbound
    # gen_ai wire spans + W3C traceparent propagation. Pin it to dikw's provider
    # so spans never leak to a foreign global, and only AFTER the process-once
    # guard passes — if dikw didn't win provider registration it must not have
    # globally patched httpx either. Unwound in :func:`shutdown_telemetry`.
    HTTPXClientInstrumentor().instrument(tracer_provider=provider)

    # Meter provider: a PeriodicExportingMetricReader pushes to the OTLP/HTTP
    # metrics endpoint on a timer. Views pin the two GenAI histograms to the
    # semconv-advised bucket boundaries (the SDK defaults are useless for token
    # counts / LLM latencies). A lost meter race degrades metrics only.
    metric_exporter = (
        OTLPMetricExporter(endpoint=endpoint.rstrip("/") + "/v1/metrics")
        if endpoint
        else OTLPMetricExporter()
    )
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[PeriodicExportingMetricReader(metric_exporter)],
        views=[
            View(
                instrument_name=_METRIC_GEN_AI_TOKEN_USAGE,
                aggregation=ExplicitBucketHistogramAggregation(_GEN_AI_TOKEN_BUCKETS),
            ),
            View(
                instrument_name=_METRIC_GEN_AI_OP_DURATION,
                aggregation=ExplicitBucketHistogramAggregation(
                    _GEN_AI_DURATION_BUCKETS
                ),
            ),
        ],
    )
    if _adopt_meter_provider(meter_provider):
        _meter_provider = meter_provider

    _provider = provider
    _configured = True
    return True


def _otel_endpoint_configured() -> bool:
    """True when an OTLP export endpoint is set via the standard env vars — the
    generic ``OTEL_EXPORTER_OTLP_ENDPOINT`` or the per-signal
    ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` (the SDK exporter reads either)."""
    return bool(
        os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
    )


def configure_client_telemetry_from_env(*, version: str) -> bool:
    """Env-only OTel bootstrap for the ``dikw client`` CLI (no ``dikw.yml``).

    The remote client has no base config (cf. ``config.TelemetryConfig``), so its
    telemetry is driven purely by the standard ``OTEL_*`` env vars. Activates
    ONLY when an OTLP endpoint is configured (:func:`_otel_endpoint_configured`),
    the ``[otel]`` extra is installed, and ``OTEL_SDK_DISABLED`` is unset — so a
    plain ``dikw client`` invocation with no OTEL_* env pays zero cost. Wires a
    ``TracerProvider`` (``service.name`` from ``OTEL_SERVICE_NAME``, default
    ``dikw-client``; sampler + exporter endpoint/headers read from the SDK's own
    ``OTEL_*`` env; transport is fixed to OTLP/HTTP) and the global httpx
    instrumentation, so the
    client's outbound request auto-injects the W3C ``traceparent`` header and the
    ``dikw serve`` FastAPI instrumentation adopts it as the parent — one trace
    spans client → server → task → provider.

    Idempotent (process-once); shares the latch + ``_provider`` slot with
    :func:`configure_telemetry` — only one runs per process (client vs. server),
    and :func:`shutdown_telemetry` tears either down. Returns ``True`` when
    activated, ``False`` on every no-op path. The caller (the CLI entry)
    registers an ``atexit`` flush on a ``True`` return — the short-lived client
    process would otherwise exit before the ``BatchSpanProcessor`` timer flushes.
    """
    global _configured, _provider
    if _configured:
        return True
    if not OTEL_AVAILABLE or _otel_sdk_disabled() or not _otel_endpoint_configured():
        return False

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as e:
        # opentelemetry-api present but the SDK/exporter/instrumentation is not —
        # a non-standard partial install. Never crash the CLI: degrade to no-op.
        logger.warning(
            "client telemetry requested via OTEL_* env but the OTel SDK/exporter/"
            "instrumentation is not installed (%s); install the [otel] extra. "
            "Continuing without telemetry.",
            e,
        )
        return False

    service_name = os.getenv("OTEL_SERVICE_NAME", "").strip() or "dikw-client"
    resource = Resource.create(
        {"service.name": service_name, "service.version": version}
    )
    # No explicit sampler: the SDK reads OTEL_TRACES_SAMPLER (default
    # parentbased_always_on) — env-faithful for a client. Bare OTLPSpanExporter:
    # it reads OTEL_EXPORTER_OTLP_(TRACES_)ENDPOINT/HEADERS itself (the import
    # pins transport to OTLP/HTTP — OTEL_EXPORTER_OTLP_PROTOCOL is not honoured,
    # matching the server's configure_telemetry).
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    if not _adopt_provider(provider):
        return False

    # Global httpx patch (pinned to dikw's provider, only after the process-once
    # guard passes) — the client's outbound Transport request then carries a
    # traceparent header. Unwound in :func:`shutdown_telemetry`.
    HTTPXClientInstrumentor().instrument(tracer_provider=provider)

    _provider = provider
    _configured = True
    return True


def shutdown_telemetry() -> None:
    """Flush + shut down the SDK providers. Safe to call when inactive.

    Clears the idempotency latch too, so the post-shutdown state is honest
    (telemetry is no longer active) and a fresh lifespan in the same process
    re-attempts activation rather than short-circuiting on a stale ``True``.
    Note: re-registering a provider after shutdown is still bounded by OTel's
    process-once ``set_tracer_provider`` — the production path is one
    activation per process; tests get a clean slate via
    :func:`reset_telemetry_for_testing`.
    """
    global _configured, _provider, _meter_provider
    global _gen_ai_token_usage, _gen_ai_op_duration
    provider = _provider
    meter_provider = _meter_provider
    _provider = None
    _meter_provider = None
    _gen_ai_token_usage = None
    _gen_ai_op_duration = None
    _configured = False
    if provider is None and meter_provider is None:
        return
    # Symmetric unwind of the global httpx patch so a fresh in-process lifespan
    # re-runs configure_telemetry from a clean, unpatched state (without this a
    # second instrument() would double-wrap). Best-effort: a partial install or
    # SDK drift may make this raise, but teardown must never propagate into the
    # lifespan finally.
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().uninstrument()
    except Exception:  # pragma: no cover - defensive teardown guard
        pass
    # Shut down both providers — the meter provider's shutdown also stops the
    # PeriodicExportingMetricReader's background timer thread + flushes a final
    # export, so a short-lived process drops no metrics.
    for active in (provider, meter_provider):
        if active is None:
            continue
        shutdown = getattr(active, "shutdown", None)
        if callable(shutdown):
            shutdown()


def reset_telemetry_for_testing() -> None:
    """Test-only: clear the idempotency latch + reset the global provider.

    Lets a test exercise both the no-op and the activated paths in one
    process. Best-effort on the otel internals — wrapped so SDK version
    drift can only cost test isolation, never production behaviour.
    """
    global _configured, _provider, _meter_provider
    global _gen_ai_token_usage, _gen_ai_op_duration
    shutdown_telemetry()
    _configured = False
    _provider = None
    _meter_provider = None
    _gen_ai_token_usage = None
    _gen_ai_op_duration = None
    if OTEL_AVAILABLE:
        try:
            _otel_trace._TRACER_PROVIDER_SET_ONCE = (
                _otel_trace._TRACER_PROVIDER_SET_ONCE.__class__()
            )
            _otel_trace._TRACER_PROVIDER = None
        except Exception:  # pragma: no cover - internal-API drift guard
            pass
        try:
            # Metrics has its own process-once latch + global, in the internal
            # module (not re-exported at ``opentelemetry.metrics`` top level).
            from opentelemetry.metrics import _internal as _metrics_internal

            _metrics_internal._METER_PROVIDER_SET_ONCE = (
                _metrics_internal._METER_PROVIDER_SET_ONCE.__class__()
            )
            _metrics_internal._METER_PROVIDER = None
        except Exception:  # pragma: no cover - internal-API drift guard
            pass


# ``reset_telemetry_for_testing`` is intentionally NOT exported — it's a
# test-only helper that reaches into OTel internals, not public API.
__all__ = [
    "DIKW_BASE_ID",
    "DIKW_CANCELLED",
    "DIKW_CATEGORY",
    "DIKW_EMBED_VERSION_ID",
    "DIKW_LAYER",
    "DIKW_LEG_HIT_COUNT",
    "DIKW_OP",
    "DIKW_RETRIEVAL_LEG",
    "DIKW_RETRIEVE_HIT_COUNT",
    "DIKW_RETRIEVE_LIMIT",
    "DIKW_SOURCE_PATH",
    "DIKW_TASK_ID",
    "ERROR_TYPE",
    "GEN_AI_CACHE_CREATION_INPUT_TOKENS",
    "GEN_AI_CACHE_READ_INPUT_TOKENS",
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_REQUEST_MAX_TOKENS",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_REQUEST_TEMPERATURE",
    "GEN_AI_RESPONSE_FINISH_REASONS",
    "GEN_AI_SYSTEM",
    "GEN_AI_TOKEN_TYPE",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    "OTEL_AVAILABLE",
    "capture_otel_context",
    "configure_client_telemetry_from_env",
    "configure_telemetry",
    "gen_ai_span",
    "get_meter",
    "get_tracer",
    "op_span",
    "shutdown_telemetry",
    "task_span",
    "telemetry_should_activate",
    "trace_llm_stream",
    "traced_op",
]
