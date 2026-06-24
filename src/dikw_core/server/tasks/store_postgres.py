"""Postgres-backed TaskStore.

Used when the wiki storage backend is itself Postgres — the user's
Phase-0 decision was to keep server task metadata in a *separate* DB so
a wipe of the wiki schema doesn't lose task history. The DSN comes from
``DIKW_SERVER_TASKS_DSN`` (set independently of the wiki DSN).

Schema lives in a dedicated ``dikw_server_tasks`` schema inside the
target DB, created on ``init()`` if missing.
"""

from __future__ import annotations

import json
import zlib
from typing import TYPE_CHECKING, Any

from .._time import isoformat_utc_ms as _isoformat
from .store import (
    TERMINAL_STATUSES,
    TaskNotFound,
    TaskRow,
    TaskStatus,
    TaskStoreError,
    summary_row_to_task,
)

if TYPE_CHECKING:  # imports happen in init() so base install works without pg deps
    from psycopg_pool import AsyncConnectionPool


_SCHEMA_DDL = """
CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.tasks (
    task_id        TEXT PRIMARY KEY,
    op             TEXT NOT NULL,
    status         TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    started_at     TEXT,
    finished_at    TEXT,
    params_digest  TEXT NOT NULL DEFAULT '',
    result         JSONB,
    error          JSONB,
    instance_id    TEXT NOT NULL DEFAULT ''
);

-- Bring older DBs up to schema (no-op when the column is already there).
ALTER TABLE {schema}.tasks ADD COLUMN IF NOT EXISTS instance_id TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS tasks_status_created_idx
    ON {schema}.tasks(status, created_at DESC);
CREATE INDEX IF NOT EXISTS tasks_op_created_idx
    ON {schema}.tasks(op, created_at DESC);
CREATE INDEX IF NOT EXISTS tasks_instance_status_idx
    ON {schema}.tasks(instance_id, status);
-- Covers the unfiltered keyset walk (``list_tasks`` with no status/op
-- filter, ORDER BY created_at DESC, task_id ASC) so ``tasks list --all``
-- pages index-only instead of seq-scan + sort.
CREATE INDEX IF NOT EXISTS tasks_created_taskid_idx
    ON {schema}.tasks(created_at DESC, task_id);

CREATE TABLE IF NOT EXISTS {schema}.task_events (
    task_id  TEXT NOT NULL,
    seq      BIGINT NOT NULL,
    ts       TEXT NOT NULL,
    body     JSONB NOT NULL,
    PRIMARY KEY (task_id, seq),
    FOREIGN KEY (task_id) REFERENCES {schema}.tasks(task_id) ON DELETE CASCADE
);
"""


class PostgresTaskStore:
    """Postgres TaskStore, gated on the ``[postgres]`` extra.

    Pool sizing is kept small (default max 5) — the task event tape is
    write-heavy but per-task; concurrency comes from running multiple
    tasks, not multiple writers per task. The asyncio.Lock-on-append
    pattern from the SQLite impl is replaced here by a per-task
    advisory lock acquired inside the txn (so cross-process appenders
    don't collide on seq).
    """

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "dikw_server_tasks",
        pool_size: int = 5,
        instance_id: str = "",
    ) -> None:
        if not _is_safe_identifier(schema):
            raise TaskStoreError(
                f"refusing schema {schema!r}: must match [A-Za-z_][A-Za-z0-9_]*"
            )
        self._dsn = dsn
        self._schema = schema
        self._pool_size = pool_size
        self._instance_id = instance_id
        self._pool: AsyncConnectionPool | None = None

    # ---- lifecycle ------------------------------------------------------

    async def init(self) -> None:
        try:
            import psycopg
            from psycopg_pool import AsyncConnectionPool
        except ImportError as e:  # pragma: no cover - only without extras
            raise TaskStoreError(
                "Postgres TaskStore requires the `postgres` extra — "
                "install via `uv pip install dikw-core[postgres]`"
            ) from e

        # Bootstrap schema + tables on a one-off connection so the pool
        # can be opened against a ready database.
        boot = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
        try:
            async with boot.cursor() as cur:
                await cur.execute(_SCHEMA_DDL.format(schema=self._schema))
        finally:
            await boot.close()

        self._pool = AsyncConnectionPool(
            conninfo=self._dsn,
            min_size=1,
            max_size=self._pool_size,
            kwargs={"autocommit": False},
            open=False,
        )
        await self._pool.open()

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ---- task rows ------------------------------------------------------

    async def create(self, row: TaskRow) -> None:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                f"INSERT INTO {self._schema}.tasks(task_id, op, status, "
                "created_at, started_at, finished_at, params_digest, "
                "result, error, instance_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)",
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
            await conn.commit()

    async def get(self, task_id: str) -> TaskRow | None:
        # Filter by ``instance_id`` so a shared task DB doesn't leak
        # tasks across servers; an unowned task_id reads as missing
        # exactly like an unknown id.
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                f"SELECT task_id, op, status, created_at, started_at, "
                f"finished_at, params_digest, result, error FROM "
                f"{self._schema}.tasks WHERE task_id = %s AND instance_id = %s",
                (task_id, self._instance_id),
            )
            row = await cur.fetchone()
            return _row_to_task(row) if row is not None else None

    async def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        op: str | None = None,
        limit: int = 100,
        after_created_at: str | None = None,
        after_task_id: str | None = None,
    ) -> list[TaskRow]:
        # Always scoped to this server's instance — operator listings
        # must not enumerate tasks owned by another wiki sharing the DSN.
        clauses: list[str] = ["instance_id = %s"]
        params: list[Any] = [self._instance_id]
        if status is not None:
            clauses.append("status = %s")
            params.append(status.value)
        if op is not None:
            clauses.append("op = %s")
            params.append(op)
        # Keyset cursor under (created_at DESC, task_id ASC). OR form
        # matches the sqlite implementation and keeps the plan stable
        # whether Postgres picks the (status, created_at) or the
        # (op, created_at) index.
        if after_created_at is not None and after_task_id is not None:
            clauses.append(
                "(created_at < %s OR (created_at = %s AND task_id > %s))"
            )
            params.extend([after_created_at, after_created_at, after_task_id])
        where = " WHERE " + " AND ".join(clauses)
        params.append(int(limit))
        async with self._acquire() as conn, conn.cursor() as cur:
            # Summary projection — never SELECT result/error here, so a
            # 50 KB synth result never crosses the wire for a list call.
            # Detail goes through ``get(task_id)`` / ``GET /v1/tasks/{id}``.
            await cur.execute(
                f"SELECT task_id, op, status, created_at, started_at, "
                f"finished_at, params_digest FROM "
                f"{self._schema}.tasks{where} "
                f"ORDER BY created_at DESC, task_id ASC LIMIT %s",
                params,
            )
            rows = await cur.fetchall()
        return [summary_row_to_task(r) for r in rows]

    async def list_running(self) -> list[TaskRow]:
        # Filter by ``instance_id`` so a restart in one ``dikw serve``
        # process never reaps another live process's tasks via the
        # shared Postgres task store. With the default ``instance_id=""``
        # (single-server / tests) this still matches every leftover
        # row, preserving the previous behaviour.
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                f"SELECT task_id, op, status, created_at, started_at, "
                f"finished_at, params_digest, result, error FROM "
                f"{self._schema}.tasks "
                "WHERE status IN (%s, %s) AND instance_id = %s",
                (
                    TaskStatus.PENDING.value,
                    TaskStatus.RUNNING.value,
                    self._instance_id,
                ),
            )
            rows = await cur.fetchall()
        return [_row_to_task(r) for r in rows]

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
        sets = ["status = %s"]
        params: list[Any] = [status.value]
        if started_at is not None:
            sets.append("started_at = %s")
            params.append(started_at)
        if finished_at is not None:
            sets.append("finished_at = %s")
            params.append(finished_at)
        if result is not None:
            sets.append("result = %s::jsonb")
            params.append(json.dumps(result))
        if error is not None:
            sets.append("error = %s::jsonb")
            params.append(json.dumps(error))
        terminal = [s.value for s in TERMINAL_STATUSES]
        placeholders = ", ".join("%s" for _ in terminal)
        params.extend([task_id, self._instance_id, *terminal])
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                f"UPDATE {self._schema}.tasks SET {', '.join(sets)} "
                f"WHERE task_id = %s AND instance_id = %s "
                f"AND status NOT IN ({placeholders})",
                params,
            )
            updated = cur.rowcount
            missing = False
            if updated == 0:
                # Zero rows: unknown task OR already terminal. Terminal rows
                # are immutable — a late write is a silent no-op (cancel-wins
                # + idempotency); only a missing task_id raises.
                await cur.execute(
                    f"SELECT 1 FROM {self._schema}.tasks "
                    "WHERE task_id = %s AND instance_id = %s",
                    (task_id, self._instance_id),
                )
                missing = (await cur.fetchone()) is None
            await conn.commit()
        if missing:
            raise TaskNotFound(task_id)

    # ---- event tape -----------------------------------------------------

    async def append_event(
        self, task_id: str, event: dict[str, Any]
    ) -> int:
        ts = event.pop("ts", None) or _isoformat()
        body = {k: v for k, v in event.items() if k not in ("seq", "ts")}
        async with self._acquire() as conn, conn.cursor() as cur:
            try:
                # Per-task advisory lock keeps cross-process appenders
                # from racing on max(seq). The lock id is a deterministic
                # hash of the task_id — Python's built-in ``hash()``
                # is randomised per process under PYTHONHASHSEED, which
                # would have two appenders pick different lock ids and
                # defeat the lock entirely.
                lock_id = _advisory_lock_id(task_id)
                await cur.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))
                await cur.execute(
                    f"SELECT COALESCE(MAX(seq), 0) FROM "
                    f"{self._schema}.task_events WHERE task_id = %s",
                    (task_id,),
                )
                row = await cur.fetchone()
                seq = int(row[0] if row is not None else 0) + 1
                await cur.execute(
                    f"INSERT INTO {self._schema}.task_events "
                    "(task_id, seq, ts, body) VALUES (%s, %s, %s, %s::jsonb)",
                    (task_id, seq, ts, json.dumps(body)),
                )
                await conn.commit()
            except Exception as e:
                await conn.rollback()
                raise TaskStoreError(
                    f"append_event failed for {task_id}: {e}"
                ) from e
        event["seq"] = seq
        event["ts"] = ts
        return seq

    async def list_events(
        self,
        task_id: str,
        *,
        from_seq: int = 0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        # Gate the read on the task belonging to this server's instance —
        # otherwise a shared-DB deployment leaks event tapes across
        # wikis. ``EXISTS`` keeps the gate cheap (PK lookup on tasks).
        sql = (
            f"SELECT seq, ts, body FROM {self._schema}.task_events "
            "WHERE task_id = %s AND seq >= %s "
            f"AND EXISTS (SELECT 1 FROM {self._schema}.tasks "
            "             WHERE task_id = %s AND instance_id = %s) "
            "ORDER BY seq"
        )
        params: tuple[Any, ...] = (
            task_id,
            int(from_seq),
            task_id,
            self._instance_id,
        )
        if limit is not None:
            sql += " LIMIT %s"
            params = (*params, int(limit))
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        out: list[dict[str, Any]] = []
        for seq, ts, body in rows:
            event = dict(body) if isinstance(body, dict) else json.loads(body)
            event["seq"] = int(seq)
            event["ts"] = ts
            out.append(event)
        return out

    async def max_seq(self, task_id: str) -> int:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                f"SELECT COALESCE(MAX(seq), 0) FROM {self._schema}.task_events "
                "WHERE task_id = %s "
                f"AND EXISTS (SELECT 1 FROM {self._schema}.tasks "
                "             WHERE task_id = %s AND instance_id = %s)",
                (task_id, task_id, self._instance_id),
            )
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    # ---- internals ------------------------------------------------------

    def _acquire(self) -> Any:
        if self._pool is None:
            raise TaskStoreError("PostgresTaskStore.init() must be called first")
        return self._pool.connection()


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
        # JSONB columns come back as dicts already; the SQLite impl
        # parses strings — the TaskStore Protocol abstracts this away.
        result=result if isinstance(result, dict) else None,
        error=error if isinstance(error, dict) else None,
    )


def _is_safe_identifier(s: str) -> bool:
    if not s or not (s[0].isalpha() or s[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in s)


def _advisory_lock_id(task_id: str) -> int:
    """Stable signed 32-bit int for ``pg_advisory_xact_lock``. Must be
    deterministic across processes — Python's built-in ``hash()`` is
    randomised per interpreter via PYTHONHASHSEED, so two ``dikw serve``
    processes appending to the same task would pick different lock ids
    and race on ``max(seq)+1``. ``zlib.crc32`` gives the same value
    everywhere (and 32 bits is plenty — collisions are harmless, they
    just briefly serialise unrelated tasks)."""
    return int(zlib.crc32(task_id.encode("utf-8")) & 0x7FFFFFFF)


__all__ = ["PostgresTaskStore"]
