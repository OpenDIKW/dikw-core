"""Server-side timestamp helper.

Centralises the ISO8601-with-millis-Z format the server uses for
NDJSON events, task rows, and import responses. The format mirrors
what JavaScript's ``Date.prototype.toISOString()`` emits, so client
consumers can pass it through ``new Date(...)`` without conversion.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime


def isoformat_utc_ms(ts: float | None = None) -> str:
    """ISO8601 UTC with millisecond precision and a trailing ``Z``.

    ``ts`` defaults to wall-clock now; pass an explicit unix timestamp
    when stamping an event whose true time is older than the call site.
    """
    if ts is None:
        ts = time.time()
    return (
        datetime.fromtimestamp(ts, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
