"""Ingest task wiring — bridge between ``api.ingest`` and the TaskManager.

Lives in its own module (rather than inside ``routes_tasks.py``) so the
TaskRunner closure is independently testable.

In the post-refactor world ingest is a pure scan-disk task: the client
uploads sources separately via ``/v1/upload/sources``, which commits
files straight into ``<base>/sources/``. ``api.ingest`` then walks
that tree, hashing for idempotency and chunking + embedding the
new/changed files.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .. import api
from ..config import CONFIG_FILENAME, load_config
from ..progress import ProgressReporter
from ..providers import build_embedder, build_multimodal_embedder
from .errors import BadRequest


def _ingest_report_to_dict(report: api.IngestReport) -> dict[str, Any]:
    return dataclasses.asdict(report)


def make_ingest_runner(
    *,
    wiki_root: Path,
    no_embed: bool,
    lock: asyncio.Lock | None = None,
) -> Callable[[ProgressReporter], Awaitable[dict[str, Any]]]:
    """Build a ``TaskRunner`` that drives ``api.ingest`` for one task.

    ``lock`` (the runtime's ``ingest_lock``) serializes overlapping
    ingest tasks so two concurrent runs can't interleave their
    storage row writes. Tests that drive the runner in isolation may
    pass ``None``.
    """

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        # The lock is held for the whole runner — concurrent ingests on
        # the same wiki would race on the storage row writes, so single-
        # threading the entire op is the only safe story.
        guard = lock if lock is not None else asyncio.Lock()
        async with guard:
            # Reload cfg from disk INSIDE the runner so the provider
            # objects we build below stay aligned with whatever
            # ``api.ingest`` reads (it does its own ``_with_storage`` →
            # ``load_config`` reload). Otherwise an operator who edits
            # ``dikw.yml`` while the long-lived server is up gets
            # vectors tagged with the FRESH ``embed_version`` but
            # produced by the STALE provider URL/model — silent corruption.
            cfg = load_config(wiki_root / CONFIG_FILENAME)

            embedder = None
            multimodal_embedder = None
            if not no_embed:
                try:
                    embedder = build_embedder(cfg.provider)
                except Exception as e:
                    # Surface as a runner-level failure so the task ends
                    # ``failed`` with a clear cause (most often a missing API
                    # key); the engine itself can't differentiate config
                    # errors from network errors mid-run.
                    raise BadRequest(
                        f"could not build embedder: {e}",
                        code="embedder_unavailable",
                    ) from e
                mm_cfg = cfg.assets.multimodal
                if mm_cfg is not None:
                    try:
                        multimodal_embedder = build_multimodal_embedder(
                            mm_cfg.provider,
                            base_url=mm_cfg.base_url,
                            batch=mm_cfg.batch,
                        )
                    except Exception as e:
                        await reporter.log(
                            "WARN",
                            f"multimodal embedder unavailable, "
                            f"falling back to text-only ingest: {e}",
                        )

            report = await api.ingest(
                wiki_root,
                embedder=embedder,
                multimodal_embedder=multimodal_embedder,
                reporter=reporter,
            )
            return _ingest_report_to_dict(report)

    return _runner


__all__ = ["make_ingest_runner"]
