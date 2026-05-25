"""Async task subsystem: TaskManager, TaskStore, event schemas.

Long-running engine operations (ingest, synth, eval) are dispatched
through a TaskManager that persists task rows + an append-only event log to a
TaskStore (independent of wiki storage). Per-task ``asyncio.Condition``
instances wake long-poll ``GET /v1/tasks/{id}/events`` handlers; cursor
pagination supports resume-by-seq.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import assert_never

from ...config import (
    DikwConfig,
    PostgresStorageConfig,
    SQLiteStorageConfig,
)
from .events import (
    ErrorEvent,
    FinalEvent,
    HeartbeatEvent,
    LogEvent,
    PartialEvent,
    ProgressEvent,
    TaskStartedEvent,
)
from .manager import TaskBusReporter, TaskManager, TaskRunner
from .store import (
    TERMINAL_STATUSES,
    TaskCounters,
    TaskNotFound,
    TaskRow,
    TaskStatus,
    TaskStore,
    TaskStoreError,
)
from .store_sqlite import SqliteTaskStore

_TASKS_DSN_ENV = "DIKW_SERVER_TASKS_DSN"
_DEFAULT_TASKS_DB_PATH = ".dikw/server-tasks.db"


def build_task_store(
    cfg: DikwConfig, root: Path, *, instance_id: str = ""
) -> TaskStore:
    """Pick a TaskStore matching the wiki's storage backend.

    Per the migration plan's Phase-0 decision:
      * SQLite wiki backend → ``<root>/.dikw/server-tasks.db``
        (a SQLite file regardless of which the wiki uses).
      * Postgres wiki backend → a separate Postgres database whose DSN
        comes from ``DIKW_SERVER_TASKS_DSN`` (intentionally distinct from
        the wiki DSN so a wiki schema wipe doesn't lose task history).

    ``instance_id`` is stamped on each task row at create-time and used
    by ``list_running()`` to scope the orphan-reaper. Pass a stable
    per-server value so a single-machine restart picks up its own
    leftover rows without touching another live ``dikw serve`` process
    sharing the same Postgres task DB. Default ``""`` is correct for
    tests and the single-server SQLite case.
    """
    storage_cfg = cfg.storage
    if isinstance(storage_cfg, PostgresStorageConfig):
        dsn = os.environ.get(_TASKS_DSN_ENV)
        if not dsn:
            raise TaskStoreError(
                f"wiki storage is postgres but {_TASKS_DSN_ENV} is not set; "
                "the server task store must live in a separate database. "
                "Export DIKW_SERVER_TASKS_DSN pointing at a Postgres DB you "
                "control."
            )
        # Lazy import keeps `import dikw_core.server.tasks` working without
        # the [postgres] extra installed (e.g. when only sqlite is configured).
        from .store_postgres import PostgresTaskStore

        return PostgresTaskStore(dsn=dsn, instance_id=instance_id)

    if isinstance(storage_cfg, SQLiteStorageConfig):
        return SqliteTaskStore(
            path=root / _DEFAULT_TASKS_DB_PATH, instance_id=instance_id
        )

    assert_never(storage_cfg)


__all__ = [
    "TERMINAL_STATUSES",
    "ErrorEvent",
    "FinalEvent",
    "HeartbeatEvent",
    "LogEvent",
    "PartialEvent",
    "ProgressEvent",
    "SqliteTaskStore",
    "TaskBusReporter",
    "TaskCounters",
    "TaskManager",
    "TaskNotFound",
    "TaskRow",
    "TaskRunner",
    "TaskStartedEvent",
    "TaskStatus",
    "TaskStore",
    "TaskStoreError",
    "build_task_store",
]
