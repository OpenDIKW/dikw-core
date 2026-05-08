"""Central logging init for dikw — read ``DIKW_LOG_LEVEL`` once.

CLI (``dikw …``) and ``dikw serve`` both call :func:`init_logging` from
their entry callback / app factory so every code path picks up the same
root level. Idempotent — the second call is a no-op so wiring it in
multiple places is safe.

Logging is the operator-facing channel; the user-facing channel is the
``ProgressReporter`` event stream over NDJSON. Don't confuse the two:
events go to the UI, logs go to the terminal / file.
"""

from __future__ import annotations

import logging
import os

_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_initialized = False


_HANDLER_TAG = "_dikw_init_logging"


def init_logging() -> None:
    """Configure the root logger from ``DIKW_LOG_LEVEL``. Idempotent."""
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
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    setattr(handler, _HANDLER_TAG, True)
    root.addHandler(handler)
    root.setLevel(level)
    # Quiet noisy third-party libs at our default INFO; they emit one
    # line per HTTP request and drown the engine's own progress logs.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _initialized = True


__all__ = ["init_logging"]
