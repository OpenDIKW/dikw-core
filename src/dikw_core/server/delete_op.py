"""Delete task wiring.

Mirrors :mod:`wisdom_op` / :mod:`ingest_op`: ``make_delete_runner`` returns
a ``TaskRunner`` closure that drives :func:`api.delete_page`. The HTTP
submit path stays thin — it validates the payload via
:class:`schemas.DeleteSubmit` and hands the primitive fields (``path`` /
``reason``) to the runner, which then opens storage, purges the document
row, and moves the on-disk file to ``<base>/trash/`` while emitting a
``delete`` phase event for NDJSON consumers.

The runner takes primitives rather than the submit object so this module
imports only ``api`` (no ``schemas.DeleteSubmit``), keeping it free of any
``routes_tasks`` coupling — same shape as ``ingest_op``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .. import api
from ..progress import ProgressReporter


def make_delete_runner(
    *,
    base_root: Path,
    path: str,
    reason: str | None = None,
    lock: asyncio.Lock | None = None,
) -> Callable[[ProgressReporter], Awaitable[dict[str, Any]]]:
    """Build a ``TaskRunner`` that drives ``api.delete_page``.

    ``lock`` (the runtime's ``ingest_lock``) serialises the delete against
    a concurrent ``dikw ingest`` on the same base — without it, an ingest
    re-creating a row from the file and a delete purging that same row
    could race. Tests that drive the runner in isolation may pass ``None``
    (an internal ``asyncio.Lock`` is used so the runner still completes).
    """

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        guard = lock if lock is not None else asyncio.Lock()
        async with guard:
            report = await api.delete_page(
                base_root, path, reason=reason, reporter=reporter
            )
            return report.model_dump(mode="json")

    return _runner


__all__ = ["make_delete_runner"]
