"""Central logging init for dikw — read ``DIKW_LOG_LEVEL`` / ``DIKW_LOG_FORMAT`` once.

CLI (``dikw …``) and ``dikw serve`` both call :func:`init_logging` from
their entry callback / app factory so every code path picks up the same
root level. Idempotent — the second call is a no-op so wiring it in
multiple places is safe.

``DIKW_LOG_FORMAT`` selects the line shape: the default ``text`` keeps the
human-readable terminal formatter byte-for-byte; ``json`` opts into one JSON
object per record (:class:`_JsonFormatter`) — the machine-readable form a log
aggregator parses and correlates to traces. When the ``[otel]`` extra is active
the trace/span ids the ``LoggingInstrumentor`` stamps on each record (wired in
``telemetry.configure_telemetry``) surface in the JSON; without it (or outside a
span) they degrade away. Both are env vars, not ``dikw.yml`` fields, because CLI
parsing happens before any base is loaded.

Logging is the operator-facing channel; the user-facing channel is the
``ProgressReporter`` event stream over NDJSON. Don't confuse the two:
events go to the UI, logs go to the terminal / file.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_initialized = False


_HANDLER_TAG = "_dikw_init_logging"

_TEXT_FORMAT = "%(asctime)s %(name)s %(levelname)s: %(message)s"
_TEXT_DATEFMT = "%H:%M:%S"

# LogRecord attributes the stdlib always sets — everything else in
# ``record.__dict__`` is a caller's ``logger.info(..., extra={...})`` field that
# the JSON formatter folds in. ``message``/``asctime`` are computed during
# formatting; ``taskName`` is 3.12+ (the repo's floor). The ``otel*`` correlation
# fields the LoggingInstrumentor injects are surfaced explicitly as
# trace_id/span_id/service, so they're filtered by the ``otel`` prefix below
# rather than enumerated here.
_STD_LOGRECORD_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
})  # fmt: skip


class _JsonFormatter(logging.Formatter):
    """One JSON object per record — the machine-readable log format an OTLP
    collector / log aggregator parses and correlates to traces.

    Always emits ``ts``/``level``/``logger``/``message``; folds in the
    ``trace_id``/``span_id`` the OTel ``LoggingInstrumentor`` stamps on the record
    (present only when the ``[otel]`` extra is active AND a span is in scope —
    absent otherwise, graceful degradation); appends ``exception`` text when the
    record carries ``exc_info``; and passes through any ``extra={...}`` fields a
    caller attached. Pure stdlib so it works in a minimal (no-``[otel]``) install.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # The instrumentor stamps "0" for "no active span" — meaningless for
        # correlation, so omit the pair rather than emit a zero id.
        trace_id = getattr(record, "otelTraceID", None)
        if trace_id and trace_id != "0":
            payload["trace_id"] = trace_id
            payload["span_id"] = getattr(record, "otelSpanID", None)
        service = getattr(record, "otelServiceName", None)
        if service:
            payload["service"] = service
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Caller ``extra={...}`` fields, last — but ``key not in payload`` guards
        # the synthesized core / correlation keys (ts/level/logger/message/
        # trace_id/span_id/service/exception) so an extra named e.g. ``service``
        # or ``trace_id`` can't clobber a field a log aggregator correlates on.
        for key, value in record.__dict__.items():
            if (
                key not in _STD_LOGRECORD_ATTRS
                and not key.startswith("otel")
                and key not in payload
            ):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


def _build_formatter() -> logging.Formatter:
    """Pick the handler formatter from ``DIKW_LOG_FORMAT`` (``text`` default /
    ``json`` opt-in). Any unrecognised value falls back to text — a junk env var
    must never break CLI startup (mirrors the ``DIKW_LOG_LEVEL`` clamp)."""
    if os.environ.get("DIKW_LOG_FORMAT", "").strip().lower() == "json":
        return _JsonFormatter()
    return logging.Formatter(fmt=_TEXT_FORMAT, datefmt=_TEXT_DATEFMT)


def init_logging() -> None:
    """Configure the root logger from ``DIKW_LOG_LEVEL`` / ``DIKW_LOG_FORMAT``.
    Idempotent."""
    global _initialized
    if _initialized:
        return
    raw = os.environ.get("DIKW_LOG_LEVEL", "INFO").upper()
    level = raw if raw in _LEVELS else "INFO"
    root = logging.getLogger()
    # Drop any prior handler we installed (re-init under test) but leave
    # foreign handlers alone so an embedding application's own logging
    # setup keeps working.
    root.handlers[:] = [
        h for h in root.handlers if not getattr(h, _HANDLER_TAG, False)
    ]
    handler = logging.StreamHandler()
    handler.setFormatter(_build_formatter())
    setattr(handler, _HANDLER_TAG, True)
    root.addHandler(handler)
    root.setLevel(level)
    # Quiet noisy third-party libs to at least WARNING (they emit one line
    # per HTTP request) but never *louder* than the root level — if the
    # operator asked for ERROR/CRITICAL we honour that. `max` works because
    # logging level numbers go up as filtering tightens (DEBUG=10 < ERROR=40).
    noisy_level = max(logging.WARNING, root.level)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(noisy_level)
    _initialized = True


__all__ = ["init_logging"]
