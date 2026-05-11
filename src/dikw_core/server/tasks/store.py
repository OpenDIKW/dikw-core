"""TaskStore Protocol + persistent task row + status enum.

Every long-running task has a ``TaskRow`` row plus an append-only
``task_events`` log. Concrete stores live in sibling files
(``store_sqlite.py``, ``store_postgres.py``) and are resolved by the
``build_task_store`` factory in ``server/tasks/__init__.py``.

The store boundary is *engine-agnostic*: it only knows about generic
op names ("ingest", "synth", "echo", …) + JSON dicts. The TaskManager
on top translates between domain types and dicts.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    """Lifecycle states. The store is the source of truth; the in-memory
    TaskManager mirrors but never overrides the persisted status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)


class TaskRow(BaseModel):
    """One persisted task. ``params_digest`` is a sha256 of the canonical
    JSON params dict — used by future client tooling to dedup retries
    without storing the raw params (which may carry large embedded blobs
    once we wire import-id-driven ingest in Phase 3)."""

    task_id: str
    op: str
    status: TaskStatus
    created_at: str  # ISO8601 UTC
    started_at: str | None = None
    finished_at: str | None = None
    params_digest: str = ""
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    model_config = {"use_enum_values": False}


class TaskNotFound(LookupError):
    """Raised by ``get`` / ``update_status`` when the task_id is unknown."""


@runtime_checkable
class TaskStore(Protocol):
    """Persistent storage for tasks + event tape.

    All implementations MUST guarantee:
      * ``append_event`` is atomic and returns a strictly increasing seq
        (per task_id); the store is the source of truth for the seq
        numbering, not the bus.
      * ``list_events(task_id, from_seq=N)`` returns every event with
        seq >= N, in seq order.
      * ``update_status`` is idempotent on the same target status.
      * Concurrent appenders to *different* tasks must not block each other.
    """

    async def init(self) -> None:
        """Create tables / files / etc. Called once at server startup."""
        ...

    async def close(self) -> None:
        """Release any pooled resources. Idempotent."""
        ...

    async def create(self, row: TaskRow) -> None:
        """Insert a fresh PENDING task row."""
        ...

    async def get(self, task_id: str) -> TaskRow | None:
        ...

    async def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        op: str | None = None,
        limit: int = 100,
    ) -> list[TaskRow]:
        ...

    async def list_running(self) -> list[TaskRow]:
        """Rows currently marked PENDING or RUNNING — used at server boot
        to mark orphans as failed{server_restart}."""
        ...

    async def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        started_at: str | None = None,
        finished_at: str | None = None,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        ...

    async def append_event(
        self, task_id: str, event: dict[str, Any]
    ) -> int:
        """Persist an event dict and return the assigned seq.

        ``event`` must be the in-flight representation — the store
        injects ``seq`` and ``ts`` on its way to the wire (the same
        dict, mutated in place, is what the bus fans out)."""
        ...

    async def list_events(
        self, task_id: str, *, from_seq: int = 0
    ) -> list[dict[str, Any]]:
        """Replay events with ``seq >= from_seq``, in order."""
        ...


class TaskStoreError(RuntimeError):
    """Base class for adapter-level errors (I/O, schema, etc.)."""


class TaskCounters(BaseModel):
    """Aggregate counts surfaced by ``GET /v1/tasks?summary=1``. Defined
    here so the SQL/JSONL adapters share one shape."""

    by_status: dict[str, int] = Field(default_factory=dict)
    by_op: dict[str, int] = Field(default_factory=dict)


__all__ = [
    "TERMINAL_STATUSES",
    "TaskCounters",
    "TaskNotFound",
    "TaskRow",
    "TaskStatus",
    "TaskStore",
    "TaskStoreError",
]
