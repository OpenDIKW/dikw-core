"""``init_logging()`` contract: env-driven, idempotent, isolates foreign handlers."""

from __future__ import annotations

import json
import logging
import sys

import pytest

from dikw_core.logging import _HANDLER_TAG, _JsonFormatter, init_logging


@pytest.fixture(autouse=True)
def _reset_logging_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dikw_core.logging._initialized", False, raising=True)
    root = logging.getLogger()
    saved = list(root.handlers)
    saved_level = root.level
    root.handlers.clear()
    yield
    root.handlers[:] = saved
    root.setLevel(saved_level)
    monkeypatch.setattr("dikw_core.logging._initialized", False, raising=True)


def test_init_logging_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIKW_LOG_LEVEL", "DEBUG")
    init_logging()
    assert logging.getLogger().level == logging.DEBUG


def test_init_logging_default_is_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIKW_LOG_LEVEL", raising=False)
    init_logging()
    assert logging.getLogger().level == logging.INFO


def test_init_logging_invalid_level_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A junk env value must NOT crash CLI startup."""
    monkeypatch.setenv("DIKW_LOG_LEVEL", "BOGUS")
    init_logging()
    assert logging.getLogger().level == logging.INFO


def test_init_logging_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIKW_LOG_LEVEL", "INFO")
    init_logging()
    handler_count_after_first = len(logging.getLogger().handlers)
    init_logging()
    assert len(logging.getLogger().handlers) == handler_count_after_first


def test_init_logging_quiets_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIKW_LOG_LEVEL", "INFO")
    init_logging()
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING


def test_init_logging_does_not_re_amplify_quieted_loggers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``DIKW_LOG_LEVEL=ERROR`` must NOT push noisy loggers back up to
    WARNING — the operator asked for a quieter floor and unconditional
    WARNING-clamping would override it."""
    monkeypatch.setenv("DIKW_LOG_LEVEL", "ERROR")
    init_logging()
    assert logging.getLogger().level == logging.ERROR
    assert logging.getLogger("httpx").level >= logging.ERROR
    assert logging.getLogger("httpcore").level >= logging.ERROR
    assert logging.getLogger("urllib3").level >= logging.ERROR


# ---- DIKW_LOG_FORMAT: text default / json opt-in (PR4) ------------------


def _dikw_handler() -> logging.Handler:
    """The single handler ``init_logging`` installed (tagged), ignoring foreign
    handlers an embedding application may have added."""
    root = logging.getLogger()
    return next(h for h in root.handlers if getattr(h, _HANDLER_TAG, False))


def test_init_logging_text_format_is_byte_stable_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default (``DIKW_LOG_FORMAT`` unset) MUST stay the pre-PR4 plain-text
    formatter, byte-for-byte — operators and tests parse that exact line shape."""
    monkeypatch.delenv("DIKW_LOG_FORMAT", raising=False)
    init_logging()
    formatter = _dikw_handler().formatter
    assert formatter is not None
    assert not isinstance(formatter, _JsonFormatter)
    assert formatter._style._fmt == "%(asctime)s %(name)s %(levelname)s: %(message)s"
    assert formatter.datefmt == "%H:%M:%S"


def test_init_logging_json_format_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIKW_LOG_FORMAT", "json")
    init_logging()
    assert isinstance(_dikw_handler().formatter, _JsonFormatter)


def test_init_logging_json_format_is_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DIKW_LOG_FORMAT", "JSON")
    init_logging()
    assert isinstance(_dikw_handler().formatter, _JsonFormatter)


def test_init_logging_unknown_format_falls_back_to_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A junk value must not crash startup or silently pick JSON — fall back to
    the text default (mirrors the ``DIKW_LOG_LEVEL`` clamp)."""
    monkeypatch.setenv("DIKW_LOG_FORMAT", "yaml")
    init_logging()
    assert not isinstance(_dikw_handler().formatter, _JsonFormatter)


def _make_record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="dikw_core.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_emits_core_fields() -> None:
    payload = json.loads(_JsonFormatter().format(_make_record()))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "dikw_core.test"
    assert payload["message"] == "hello world"
    assert payload["ts"]  # non-empty timestamp


def test_json_formatter_surfaces_trace_ids_when_present() -> None:
    payload = json.loads(
        _JsonFormatter().format(
            _make_record(
                otelTraceID="abc123",
                otelSpanID="def456",
                otelServiceName="dikw-core",
            )
        )
    )
    assert payload["trace_id"] == "abc123"
    assert payload["span_id"] == "def456"
    assert payload["service"] == "dikw-core"


def test_json_formatter_omits_zero_trace_id() -> None:
    """``LoggingInstrumentor`` stamps ``"0"`` outside an active span — a zero id
    is meaningless for correlation, so it must not appear in the payload."""
    payload = json.loads(
        _JsonFormatter().format(_make_record(otelTraceID="0", otelSpanID="0"))
    )
    assert "trace_id" not in payload
    assert "span_id" not in payload


def test_json_formatter_degrades_without_otel_fields() -> None:
    """No ``otel*`` attributes (minimal install / no active span) → no trace
    keys, no crash."""
    payload = json.loads(_JsonFormatter().format(_make_record()))
    assert "trace_id" not in payload
    assert "service" not in payload


def test_json_formatter_passes_through_extra_and_exception() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = _make_record(base_id="b-1", exc_info=sys.exc_info())
    payload = json.loads(_JsonFormatter().format(record))
    assert payload["base_id"] == "b-1"
    assert "ValueError: boom" in payload["exception"]


def test_json_formatter_extra_cannot_clobber_correlation_fields() -> None:
    """A caller ``extra={}`` key that collides with a synthesized core /
    correlation field must NOT overwrite it — the trace id a log aggregator
    correlates on wins over a same-named caller field."""
    record = _make_record(
        otelTraceID="real-trace",
        otelServiceName="dikw-core",
        service="HIJACK",
        trace_id="HIJACK",
    )
    payload = json.loads(_JsonFormatter().format(record))
    assert payload["service"] == "dikw-core"
    assert payload["trace_id"] == "real-trace"
