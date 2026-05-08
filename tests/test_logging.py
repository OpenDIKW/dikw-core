"""Central logging init contract.

Every entry point (``dikw …`` CLI commands, ``dikw serve`` uvicorn,
direct ``build_app`` for tests) calls ``init_logging()`` once so the
root logger level is consistently driven by ``DIKW_LOG_LEVEL``. Pinning
the contract here keeps drift away — silent regressions to "logs work
in CLI but not in server" are exactly what burns operators when a
production stall doesn't print anything.
"""

from __future__ import annotations

import logging

import pytest

from dikw_core.logging import init_logging


@pytest.fixture(autouse=True)
def _reset_logging_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets a fresh init flag + clean handler list. Without
    this every test after the first sees a memoised level / handler
    set from the previous run, defeating the assertion."""
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
    """A junk env value must NOT crash CLI startup — fall back to INFO."""
    monkeypatch.setenv("DIKW_LOG_LEVEL", "BOGUS")
    init_logging()
    assert logging.getLogger().level == logging.INFO


def test_init_logging_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling init_logging() twice (CLI callback + server build_app) must
    NOT double up handlers — operators tail logs and can't tell which
    duplicate line came from which init."""
    monkeypatch.setenv("DIKW_LOG_LEVEL", "INFO")
    init_logging()
    handler_count_after_first = len(logging.getLogger().handlers)
    init_logging()
    assert len(logging.getLogger().handlers) == handler_count_after_first


def test_init_logging_quiets_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    """httpx's INFO is one line per HTTP request — too noisy at our
    default INFO. The user gets it back via DIKW_LOG_LEVEL=DEBUG."""
    monkeypatch.setenv("DIKW_LOG_LEVEL", "INFO")
    init_logging()
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING
