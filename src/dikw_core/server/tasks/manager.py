"""TaskManager: dispatch async engine ops + persist events + fan out to bus.

A ``TaskRunner`` is any async callable that takes a ``ProgressReporter``
and returns a JSON-serialisable result dict. The manager wraps the
runner with three concerns the engine itself doesn't carry:

  1. Persistence: store row lifecycle (PENDING → RUNNING → terminal)
     and an append-only event tape so a reconnecting client can replay.
  2. Fanout: every event the runner emits via the reporter goes to
     ``ProgressBus.publish`` for live subscribers and ``TaskStore.append_event``
     for replay.
  3. Cancellation: the manager keeps the underlying ``asyncio.Task`` and
     the ``CancelToken`` so an HTTP cancel handler can flip both.

Server boot calls ``restart_cleanup`` to mark any leftover RUNNING /
PENDING rows as failed{server_restart} — we deliberately don't try to
resume tasks across restarts (ingest is idempotent; the user can resubmit).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import traceback
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from ...progress import CancelToken, ProgressReporter
from .._time import isoformat_utc_ms as _isoformat
from .bus import ProgressBus
from .store import TaskRow, TaskStatus, TaskStore

logger = logging.getLogger(__name__)


TaskRunner = Callable[[ProgressReporter], Awaitable[dict[str, Any]]]


def _params_digest(params: dict[str, Any] | None) -> str:
    if not params:
        return ""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class TaskBusReporter:
    """``ProgressReporter`` impl that writes through a TaskStore + bus.

    Every event method:
      1. Stamps ``seq`` (from store) and ``ts``.
      2. Persists the event in ``task_events`` so a late reconnect can
         replay everything from seq=0.
      3. Publishes the same dict to the in-memory bus for live subscribers.

    Cancellation is exposed via the shared ``CancelToken`` the manager
    creates when submitting the task; the engine main loop polls it
    between work units.
    """

    def __init__(
        self,
        *,
        task_id: str,
        store: TaskStore,
        bus: ProgressBus,
        token: CancelToken,
    ) -> None:
        self._task_id = task_id
        self._store = store
        self._bus = bus
        self._token = token

    async def progress(
        self,
        *,
        phase: str,
        current: int = 0,
        total: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        await self._emit(
            {
                "type": "progress",
                "phase": phase,
                "current": current,
                "total": total,
                "detail": detail,
            }
        )

    async def log(self, level: str, message: str) -> None:
        await self._emit({"type": "log", "level": level, "message": message})

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        await self._emit({"type": "partial", "kind": kind, "payload": payload})

    def cancel_token(self) -> CancelToken:
        return self._token

    async def emit_raw(self, event: dict[str, Any]) -> None:
        """Public escape hatch used by the manager itself for ``task_started``
        and ``final`` events — anything outside the ProgressReporter
        Protocol but still needing the same store + bus path."""
        await self._emit(event)

    async def _emit(self, event: dict[str, Any]) -> None:
        # ``append_event`` mutates ``event`` in place to inject ``seq`` + ``ts``;
        # the bus then sees the same complete dict for fanout.
        await self._store.append_event(self._task_id, event)
        await self._bus.publish(self._task_id, event)


class TaskManager:
    """Submit, query, and cancel async tasks."""

    def __init__(self, *, store: TaskStore, bus: ProgressBus) -> None:
        self._store = store
        self._bus = bus
        self._running: dict[str, tuple[asyncio.Task[Any], CancelToken]] = {}
        self._lock = asyncio.Lock()

    # ---- lifecycle ------------------------------------------------------

    async def restart_cleanup(self) -> None:
        """Mark every PENDING/RUNNING row as failed{server_restart}.

        Called once during server startup. We don't try to resume the
        underlying work; ingest is idempotent and the user can resubmit.
        Any latent NDJSON subscriber on a half-finished tape will see
        the synthetic ``final`` event we append here and unblock cleanly.
        """
        rows = await self._store.list_running()
        for row in rows:
            error = {"reason": "server_restart"}
            await self._store.update_status(
                row.task_id,
                TaskStatus.FAILED,
                finished_at=_isoformat(),
                error=error,
            )
            await self._store.append_event(
                row.task_id,
                {"type": "final", "status": "failed", "error": error},
            )
            logger.info(
                "marked orphan task %s (op=%s) as failed{server_restart}",
                row.task_id,
                row.op,
            )

    async def shutdown(self, *, timeout: float = 5.0) -> None:
        """Cancel every in-flight task and wait briefly for them to unwind.

        Called on server shutdown. Tasks that don't finish within
        ``timeout`` are abandoned to the GC; their store rows will be
        picked up by the next ``restart_cleanup`` and marked failed.
        """
        async with self._lock:
            entries = list(self._running.items())
        for _task_id, (asyncio_task, token) in entries:
            token.cancel()
            asyncio_task.cancel()
        if entries:
            await asyncio.wait(
                [t for _, (t, _tok) in entries],
                timeout=timeout,
            )

    # ---- submit / cancel / query ---------------------------------------

    async def submit(
        self,
        *,
        op: str,
        runner: TaskRunner,
        params: dict[str, Any] | None = None,
    ) -> TaskRow:
        """Insert a PENDING task row, register it on the bus, and start the
        runner under an ``asyncio.create_task`` we hold a handle to. The
        coroutine performs status transitions and emits the ``task_started``
        / ``final`` envelope events. Returns the persisted row immediately.
        """
        task_id = str(uuid.uuid4())
        now = _isoformat()
        row = TaskRow(
            task_id=task_id,
            op=op,
            status=TaskStatus.PENDING,
            created_at=now,
            params_digest=_params_digest(params),
        )
        await self._store.create(row)
        await self._bus.register(task_id)

        token = CancelToken()
        reporter = TaskBusReporter(
            task_id=task_id, store=self._store, bus=self._bus, token=token
        )

        async def _run() -> None:
            # Always update the task row to its terminal status BEFORE
            # emitting the corresponding ``final`` event. A subscriber
            # that sees ``final`` and immediately calls ``/result`` must
            # find the row terminal — the reverse order races and yields
            # ``task_not_terminal`` for a brief window. The post-emit
            # crash mode (row terminal but tape missing ``final``) is
            # easier to recover than ``restart_cleanup`` appending a
            # second terminal event over a still-RUNNING row.
            try:
                started_at = _isoformat()
                await self._store.update_status(
                    task_id, TaskStatus.RUNNING, started_at=started_at
                )
                await reporter.emit_raw(
                    {"type": "task_started", "task_id": task_id, "op": op}
                )
                result = await runner(reporter)
                finished_at = _isoformat()
                await self._store.update_status(
                    task_id,
                    TaskStatus.SUCCEEDED,
                    finished_at=finished_at,
                    result=result,
                )
                await reporter.emit_raw(
                    {"type": "final", "status": "succeeded", "result": result}
                )
            except asyncio.CancelledError:
                finished_at = _isoformat()
                await self._store.update_status(
                    task_id, TaskStatus.CANCELLED, finished_at=finished_at
                )
                await reporter.emit_raw(
                    {"type": "final", "status": "cancelled"}
                )
                # Don't re-raise: the manager owns the task lifecycle and a
                # graceful "cancelled" final is the contract.
            except Exception as e:
                finished_at = _isoformat()
                error = {
                    "code": e.__class__.__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                await self._store.update_status(
                    task_id,
                    TaskStatus.FAILED,
                    finished_at=finished_at,
                    error=error,
                )
                await reporter.emit_raw(
                    {"type": "final", "status": "failed", "error": error}
                )
            finally:
                await self._bus.close(task_id)
                async with self._lock:
                    self._running.pop(task_id, None)

        asyncio_task = asyncio.create_task(_run(), name=f"dikw-task-{task_id}")
        async with self._lock:
            self._running[task_id] = (asyncio_task, token)
        return row

    async def cancel(self, task_id: str) -> bool:
        """Request cooperative cancellation. Returns True if we found a
        live in-flight task; False if it's already finished or unknown.
        Idempotent."""
        async with self._lock:
            entry = self._running.get(task_id)
        if entry is None:
            # Already terminal — store has the final state, nothing to do.
            return False
        asyncio_task, token = entry
        token.cancel()
        asyncio_task.cancel()
        return True

    def is_running(self, task_id: str) -> bool:
        return task_id in self._running


__all__ = ["TaskBusReporter", "TaskManager", "TaskRunner"]
