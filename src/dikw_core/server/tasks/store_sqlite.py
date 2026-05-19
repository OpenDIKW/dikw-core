"""SQLite-backed TaskStore.

Pattern mirrors ``storage/sqlite.py``: stdlib ``sqlite3`` wrapped in
``asyncio.to_thread`` so the rest of the engine stays async without
pulling in aiosqlite.

Connection lifecycle: **per-call** (not shared). Each ``_xxx_sync``
opens a fresh ``sqlite3.Connection``, performs the op, and closes it.
Earlier revisions held one ``sqlite3.Connection`` per store and shared
it across ``asyncio.to_thread`` workers via ``check_same_thread=False``;
under CI's concurrent submit + long-poll + cancel mix, that produced
``InterfaceError('bad parameter or other API misuse')`` and phantom
rows (``status=NULL`` reads on a ``NOT NULL`` column). The cure was
worse than the disease — wrapping reads in ``asyncio.Lock`` deadlocked
against the per-task ``asyncio.Condition`` used by long-poll, hanging
CI for 6h. Per-call conn removes the shared state entirely; WAL +
``busy_timeout`` lets SQLite serialize writes at the engine level
with full reader parallelism.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from .._time import isoformat_utc_ms as _isoformat
from .store import (
    TaskNotFound,
    TaskRow,
    TaskStatus,
    TaskStoreError,
    summary_row_to_task,
)

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id        TEXT PRIMARY KEY,
    op             TEXT NOT NULL,
    status         TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    started_at     TEXT,
    finished_at    TEXT,
    params_digest  TEXT NOT NULL DEFAULT '',
    result         TEXT,
    error          TEXT,
    instance_id    TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS tasks_status_created_idx
    ON tasks(status, created_at DESC);
CREATE INDEX IF NOT EXISTS tasks_op_created_idx
    ON tasks(op, created_at DESC);
CREATE INDEX IF NOT EXISTS tasks_instance_status_idx
    ON tasks(instance_id, status);
-- Covers the unfiltered keyset walk (`list_tasks` with no status/op
-- filter, ORDER BY created_at DESC, task_id ASC) so `tasks list --all`
-- pages index-only instead of full-scan + filesort.
CREATE INDEX IF NOT EXISTS tasks_created_taskid_idx
    ON tasks(created_at DESC, task_id);

CREATE TABLE IF NOT EXISTS task_events (
    task_id  TEXT NOT NULL,
    seq      INTEGER NOT NULL,
    ts       TEXT NOT NULL,
    body     TEXT NOT NULL,  -- full event JSON minus seq/ts
    PRIMARY KEY (task_id, seq),
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
) WITHOUT ROWID;
"""


class SqliteTaskStore:
    """SQLite TaskStore using per-call connections.

    The DB file is created on ``init()`` if missing — callers don't need
    to pre-create the parent dir as long as they pass an absolute path
    inside the wiki root.
    """

    def __init__(self, path: str | Path, *, instance_id: str = "") -> None:
        self._path = Path(path)
        self._instance_id = instance_id

    # ---- lifecycle ------------------------------------------------------

    async def init(self) -> None:
        await asyncio.to_thread(self._init_sync)

    def _init_sync(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._open_conn()
        try:
            # ``journal_mode = WAL`` is a *persistent* PRAGMA written to
            # the DB file header — set it once here and every subsequent
            # ``_open_conn`` sees WAL automatically.
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_SCHEMA_DDL)
            # Bring older DBs up to schema. SQLite ALTER TABLE ADD COLUMN
            # got IF NOT EXISTS in 3.36; older runtimes raise OperationalError
            # on the duplicate which we swallow.
            try:
                conn.execute(
                    "ALTER TABLE tasks ADD COLUMN instance_id TEXT NOT NULL DEFAULT ''"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        finally:
            conn.close()

    async def close(self) -> None:
        # Per-call conn pattern: nothing to close at the store level.
        # Kept for Protocol compatibility + lifecycle symmetry.
        return None

    # ---- task rows ------------------------------------------------------

    async def create(self, row: TaskRow) -> None:
        await asyncio.to_thread(self._create_sync, row)

    def _create_sync(self, row: TaskRow) -> None:
        conn = self._open_conn()
        try:
            conn.execute(
                "INSERT INTO tasks(task_id, op, status, created_at, started_at, "
                "finished_at, params_digest, result, error, instance_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row.task_id,
                    row.op,
                    row.status.value,
                    row.created_at,
                    row.started_at,
                    row.finished_at,
                    row.params_digest,
                    json.dumps(row.result) if row.result is not None else None,
                    json.dumps(row.error) if row.error is not None else None,
                    self._instance_id,
                ),
            )
        finally:
            conn.close()

    async def get(self, task_id: str) -> TaskRow | None:
        return await asyncio.to_thread(self._get_sync, task_id)

    def _get_sync(self, task_id: str) -> TaskRow | None:
        # Filter by ``instance_id`` for symmetry with the Postgres store.
        # In single-server SQLite (the default) the filter matches all
        # rows; in any future shared-DB sqlite scenario it scopes reads
        # the same way Postgres does.
        conn = self._open_conn()
        try:
            cur = conn.execute(
                "SELECT task_id, op, status, created_at, started_at, finished_at, "
                "params_digest, result, error FROM tasks "
                "WHERE task_id = ? AND instance_id = ?",
                (task_id, self._instance_id),
            )
            row = cur.fetchone()
            return _row_to_task(row) if row is not None else None
        finally:
            conn.close()

    async def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        op: str | None = None,
        limit: int = 100,
        after_created_at: str | None = None,
        after_task_id: str | None = None,
    ) -> list[TaskRow]:
        return await asyncio.to_thread(
            self._list_sync,
            status,
            op,
            limit,
            after_created_at,
            after_task_id,
        )

    def _list_sync(
        self,
        status: TaskStatus | None,
        op: str | None,
        limit: int,
        after_created_at: str | None,
        after_task_id: str | None,
    ) -> list[TaskRow]:
        conn = self._open_conn()
        try:
            clauses: list[str] = ["instance_id = ?"]
            params: list[Any] = [self._instance_id]
            if status is not None:
                clauses.append("status = ?")
                params.append(status.value)
            if op is not None:
                clauses.append("op = ?")
                params.append(op)
            # Keyset cursor: "strictly after (after_created_at,
            # after_task_id)" under the (created_at DESC, task_id ASC)
            # sort order. Standard OR form so it works on SQLite < 3.15
            # which lacks row-value tuple comparison.
            if after_created_at is not None and after_task_id is not None:
                clauses.append(
                    "(created_at < ? OR (created_at = ? AND task_id > ?))"
                )
                params.extend(
                    [after_created_at, after_created_at, after_task_id]
                )
            where = " WHERE " + " AND ".join(clauses)
            params.append(int(limit))
            # Summary view — never SELECT ``result`` / ``error``. Keeps
            # ``GET /v1/tasks`` bandwidth bounded; ``get()`` still loads
            # the full payload for single-task detail.
            cur = conn.execute(
                "SELECT task_id, op, status, created_at, started_at, "
                "finished_at, params_digest FROM tasks"
                + where
                + " ORDER BY created_at DESC, task_id ASC LIMIT ?",
                params,
            )
            return [summary_row_to_task(r) for r in cur.fetchall()]
        finally:
            conn.close()

    async def list_running(self) -> list[TaskRow]:
        return await asyncio.to_thread(self._list_running_sync)

    def _list_running_sync(self) -> list[TaskRow]:
        # Filter by ``instance_id`` so the orphan-reaper at server boot
        # never touches rows owned by another live ``dikw serve``
        # instance pointed at the same task DB. With the default
        # ``instance_id=""`` (single-server / tests) this matches all
        # rows for that wiki, preserving the previous behaviour.
        conn = self._open_conn()
        try:
            cur = conn.execute(
                "SELECT task_id, op, status, created_at, started_at, finished_at, "
                "params_digest, result, error FROM tasks "
                "WHERE status IN (?, ?) AND instance_id = ?",
                (
                    TaskStatus.PENDING.value,
                    TaskStatus.RUNNING.value,
                    self._instance_id,
                ),
            )
            return [_row_to_task(r) for r in cur.fetchall()]
        finally:
            conn.close()

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
        await asyncio.to_thread(
            self._update_status_sync,
            task_id,
            status,
            started_at,
            finished_at,
            result,
            error,
        )

    def _update_status_sync(
        self,
        task_id: str,
        status: TaskStatus,
        started_at: str | None,
        finished_at: str | None,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
    ) -> None:
        conn = self._open_conn()
        try:
            sets = ["status = ?"]
            params: list[Any] = [status.value]
            if started_at is not None:
                sets.append("started_at = ?")
                params.append(started_at)
            if finished_at is not None:
                sets.append("finished_at = ?")
                params.append(finished_at)
            if result is not None:
                sets.append("result = ?")
                params.append(json.dumps(result))
            if error is not None:
                sets.append("error = ?")
                params.append(json.dumps(error))
            params.extend([task_id, self._instance_id])
            cur = conn.execute(
                f"UPDATE tasks SET {', '.join(sets)} "
                "WHERE task_id = ? AND instance_id = ?",
                params,
            )
            if cur.rowcount == 0:
                raise TaskNotFound(task_id)
        finally:
            conn.close()

    # ---- event tape -----------------------------------------------------

    async def append_event(
        self, task_id: str, event: dict[str, Any]
    ) -> int:
        return await asyncio.to_thread(self._append_event_sync, task_id, event)

    def _append_event_sync(
        self, task_id: str, event: dict[str, Any]
    ) -> int:
        conn = self._open_conn()
        # Atomic: read max(seq) + insert in a single transaction so two
        # concurrent appenders to the *same* task can't collide on seq.
        # ``BEGIN IMMEDIATE`` acquires the file-level write lock up
        # front; ``busy_timeout`` (set in ``_open_conn``) makes parallel
        # writers wait cleanly instead of spinning on SQLITE_BUSY. With
        # per-call connections this is also the cross-coroutine + cross-
        # process safety net rolled into one — no app-level lock needed.
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM task_events WHERE task_id = ?",
                    (task_id,),
                )
                (max_seq,) = cur.fetchone()
                seq = int(max_seq) + 1
                ts = event.pop("ts", None) or _isoformat()
                event["seq"] = seq
                event["ts"] = ts
                body = {k: v for k, v in event.items() if k not in ("seq", "ts")}
                conn.execute(
                    "INSERT INTO task_events(task_id, seq, ts, body) VALUES (?, ?, ?, ?)",
                    (task_id, seq, ts, json.dumps(body)),
                )
                conn.execute("COMMIT")
            except sqlite3.Error as e:
                conn.execute("ROLLBACK")
                raise TaskStoreError(f"append_event failed for {task_id}: {e}") from e
            return seq
        finally:
            conn.close()

    async def list_events(
        self,
        task_id: str,
        *,
        from_seq: int = 0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._list_events_sync, task_id, from_seq, limit
        )

    def _list_events_sync(
        self, task_id: str, from_seq: int, limit: int | None
    ) -> list[dict[str, Any]]:
        # Same instance gate as the Postgres impl — keeps the cross-store
        # contract identical so a future shared-DB sqlite scenario doesn't
        # surprise the operator.
        conn = self._open_conn()
        try:
            sql = (
                "SELECT seq, ts, body FROM task_events "
                "WHERE task_id = ? AND seq >= ? "
                "AND EXISTS (SELECT 1 FROM tasks "
                "            WHERE task_id = ? AND instance_id = ?) "
                "ORDER BY seq"
            )
            params: tuple[Any, ...] = (
                task_id,
                int(from_seq),
                task_id,
                self._instance_id,
            )
            if limit is not None:
                sql += " LIMIT ?"
                params = (*params, int(limit))
            cur = conn.execute(sql, params)
            out: list[dict[str, Any]] = []
            for seq, ts, body in cur.fetchall():
                event = json.loads(body)
                event["seq"] = int(seq)
                event["ts"] = ts
                out.append(event)
            return out
        finally:
            conn.close()

    async def max_seq(self, task_id: str) -> int:
        return await asyncio.to_thread(self._max_seq_sync, task_id)

    def _max_seq_sync(self, task_id: str) -> int:
        conn = self._open_conn()
        try:
            cur = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM task_events "
                "WHERE task_id = ? "
                "AND EXISTS (SELECT 1 FROM tasks "
                "            WHERE task_id = ? AND instance_id = ?)",
                (task_id, task_id, self._instance_id),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()

    # ---- internals ------------------------------------------------------

    def _open_conn(self) -> sqlite3.Connection:
        # Fresh conn per call. ``isolation_level=None`` means autocommit;
        # txns are managed explicitly via ``BEGIN IMMEDIATE`` / ``COMMIT``.
        # ``check_same_thread`` defaults to True — that's fine, each conn
        # is born and dies inside one ``to_thread`` worker.
        # ``busy_timeout`` makes concurrent writers wait up to 30s on the
        # file-level write lock instead of raising SQLITE_BUSY immediately.
        # ``foreign_keys`` is per-connection (default OFF in sqlite) so
        # we need to enable it every time to keep the FK cascade from
        # ``task_events`` → ``tasks`` working.
        conn = sqlite3.connect(
            str(self._path),
            isolation_level=None,
            timeout=30.0,
        )
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


def _row_to_task(row: tuple[Any, ...]) -> TaskRow:
    (
        task_id,
        op,
        status,
        created_at,
        started_at,
        finished_at,
        params_digest,
        result,
        error,
    ) = row
    return TaskRow(
        task_id=task_id,
        op=op,
        status=TaskStatus(status),
        created_at=created_at,
        started_at=started_at,
        finished_at=finished_at,
        params_digest=params_digest or "",
        result=json.loads(result) if result else None,
        error=json.loads(error) if error else None,
    )


__all__ = ["SqliteTaskStore"]
