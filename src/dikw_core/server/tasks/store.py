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
# The same set as raw column values, for adapters comparing against a
# ``status`` string read straight out of SQL without re-wrapping in the enum.
TERMINAL_STATUS_VALUES: frozenset[str] = frozenset(s.value for s in TERMINAL_STATUSES)


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


def summary_row_to_task(row: tuple[Any, ...]) -> TaskRow:
    """Build a ``TaskRow`` from a summary-projection row: the 7 columns
    ``list_tasks`` SELECTs, in order ``(task_id, op, status, created_at,
    started_at, finished_at, params_digest)``. ``result``/``error`` are
    forced to ``None`` since they're never in the summary SELECT.

    Shared by both adapters so the summary column contract lives in one
    place.
    """
    task_id, op, status, created_at, started_at, finished_at, params_digest = row
    return TaskRow(
        task_id=task_id,
        op=op,
        status=TaskStatus(status),
        created_at=created_at,
        started_at=started_at,
        finished_at=finished_at,
        params_digest=params_digest or "",
        result=None,
        error=None,
    )


class TaskNotFound(LookupError):
    """Raised by ``get`` / ``update_status`` when the task_id is unknown."""


@runtime_checkable
class TaskStore(Protocol):
    """Persistent storage for tasks + event tape.

    All implementations MUST guarantee:
      * ``append_event`` is atomic and assigns a strictly increasing seq
        (per task_id) on the **append** path; the store is the source of
        truth for the seq numbering, not the bus. To uphold the event-tape
        invariant that ``final`` is always the last event (see ``events.py``),
        a NON-``final`` event appended once the task row is already terminal
        is **dropped** — no row inserted — and returns the current max seq, so
        an obsolete late write (e.g. a cancelled runner's in-flight
        ``reporter.progress`` whose ``to_thread`` DB write outlives the
        cancelled await — ``run_in_executor`` does not stop the worker thread)
        cannot land after the ``final``. The ``final`` itself is always
        appended (the runner commits the terminal status immediately before it).
      * ``list_events(task_id, from_seq=N)`` returns every event with
        seq >= N, in seq order.
      * ``update_status`` is idempotent on the same target status, and a
        **terminal** status is immutable: once a task is succeeded / failed
        / cancelled, a later ``update_status`` is a silent no-op (so a cancel
        that lands first wins over a runner's late failure). It raises
        ``TaskNotFound`` only when the ``task_id`` is unknown — never for a
        row that is merely already terminal.
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
        after_created_at: str | None = None,
        after_task_id: str | None = None,
    ) -> list[TaskRow]:
        """Summary listing of tasks, newest first.

        Returns rows ordered by ``(created_at DESC, task_id ASC)`` —
        ``task_id`` is the deterministic tie-breaker on identical
        timestamps so keyset cursors stay stable.

        ``result`` and ``error`` are **always** returned as ``None``;
        ``list_tasks`` is the summary view. Callers needing the full
        payload must use ``get(task_id)`` (or the HTTP ``/result``
        endpoint one layer up). This keeps ``GET /v1/tasks`` bandwidth
        bounded even when a synth/eval task stamped 50 KB of result
        into the row.

        Keyset paging: pass ``after_created_at`` + ``after_task_id``
        together to fetch the page strictly *after* that (timestamp,
        id) position in the sort order. Both must be set or both
        omitted — passing exactly one is undefined and adapters MAY
        treat it as "no cursor".
        """
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

        On the normal append path the store injects ``seq`` and ``ts`` into
        ``event`` in place (the same dict, mutated, is what the bus fans out).
        A NON-``final`` event arriving after the task row is terminal is
        dropped to keep ``final`` last (see the MUST-list above): nothing is
        inserted, ``event`` is left unstamped, and the current max seq is
        returned."""
        ...

    async def list_events(
        self,
        task_id: str,
        *,
        from_seq: int = 0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Replay events with ``seq >= from_seq``, in order.

        ``limit``: cap on returned rows. ``None`` returns everything
        from ``from_seq`` onwards (used by the historical replay path);
        a concrete int is the cursor-based pagination knob driving the
        long-poll ``GET /v1/tasks/{id}/events`` endpoint.
        """
        ...

    async def max_seq(self, task_id: str) -> int:
        """Highest seq currently persisted for ``task_id``, or 0 if no
        events. Single-row indexed lookup — avoids the O(N) tape scan
        the cursor ``/events`` endpoint needs to compute ``last_seq``."""
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
    "TERMINAL_STATUS_VALUES",
    "TaskCounters",
    "TaskNotFound",
    "TaskRow",
    "TaskStatus",
    "TaskStore",
    "TaskStoreError",
]
