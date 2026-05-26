"""Wisdom-write task wiring (0.3.1).

Mirrors :mod:`lint_op`: ``make_wisdom_write_runner`` returns a
``TaskRunner`` closure that drives :func:`api.write_wisdom_page`. The
HTTP submit path stays thin — it validates the payload via
:class:`schemas.WisdomWriteSubmit` and hands the structured fields to
the runner, which then opens storage, writes the file, and emits a
single ``wisdom_write`` phase event for NDJSON consumers.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .. import api
from ..progress import ProgressReporter
from ..schemas import WisdomWriteSubmit


def make_wisdom_write_runner(
    *,
    wiki_root: Path,
    submit: WisdomWriteSubmit,
) -> Callable[[ProgressReporter], Awaitable[dict[str, Any]]]:
    """Build a ``TaskRunner`` that drives ``api.write_wisdom_page``.

    ``submit`` arrives already validated (slug/author kebab-case enforced
    by the Pydantic schema) — the runner forwards the structured fields
    through to the engine and dumps the resulting
    :class:`WisdomWriteReport` for the task's terminal payload.
    """

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        report = await api.write_wisdom_page(
            wiki_root,
            slug=submit.slug,
            title=submit.title,
            body=submit.body,
            author=submit.author,
            status=submit.status,
            tags=submit.tags,
            sources=submit.sources,
            extras=submit.extras,
            no_embed=submit.no_embed,
            reporter=reporter,
        )
        return report.model_dump(mode="json")

    return _runner


__all__ = ["make_wisdom_write_runner"]
