"""Contract tests for TaskStore implementations.

Parametrised across the SQLite implementation (always runs) and the
Postgres implementation (runs only when ``DIKW_TEST_POSTGRES_TASKS_DSN``
is set, mirroring the wiki ``test_storage_contract.py`` pattern). Both
must pass the same behavioural assertions so the factory in
``server/tasks/__init__.py`` can swap them transparently.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from dikw_core.server.tasks import (
    SqliteTaskStore,
    TaskNotFound,
    TaskRow,
    TaskStatus,
    TaskStore,
)
from dikw_core.server.tasks.store_postgres import PostgresTaskStore

POSTGRES_DSN_ENV = "DIKW_TEST_POSTGRES_TASKS_DSN"


def _now() -> str:
    return "2026-05-02T12:00:00.000Z"


def _row(task_id: str | None = None, *, op: str = "echo") -> TaskRow:
    return TaskRow(
        task_id=task_id or str(uuid.uuid4()),
        op=op,
        status=TaskStatus.PENDING,
        created_at=_now(),
    )


@pytest.fixture(params=["sqlite", "postgres"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[TaskStore]:
    if request.param == "sqlite":
        s: TaskStore = SqliteTaskStore(path=tmp_path / "tasks.db")
        await s.init()
        try:
            yield s
        finally:
            await s.close()
        return

    dsn = os.environ.get(POSTGRES_DSN_ENV)
    if not dsn:
        pytest.skip(f"Postgres TaskStore tests require {POSTGRES_DSN_ENV}")
    schema = f"dikw_test_tasks_{uuid.uuid4().hex[:8]}"
    s = PostgresTaskStore(dsn=dsn, schema=schema)
    await s.init()
    try:
        yield s
    finally:
        await s.close()
        # Best-effort teardown of the per-test schema.
        import psycopg

        async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


# ---- task rows ----------------------------------------------------------


@pytest.mark.asyncio
async def test_create_and_get_roundtrip(store: TaskStore) -> None:
    row = _row()
    await store.create(row)
    fetched = await store.get(row.task_id)
    assert fetched is not None
    assert fetched.task_id == row.task_id
    assert fetched.op == "echo"
    assert fetched.status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_get_unknown_returns_none(store: TaskStore) -> None:
    assert await store.get("does-not-exist") is None


@pytest.mark.asyncio
async def test_update_status_and_terminal_payload(store: TaskStore) -> None:
    row = _row()
    await store.create(row)
    await store.update_status(
        row.task_id,
        TaskStatus.RUNNING,
        started_at="2026-05-02T12:00:01.000Z",
    )
    await store.update_status(
        row.task_id,
        TaskStatus.SUCCEEDED,
        finished_at="2026-05-02T12:00:02.000Z",
        result={"echoed": 5},
    )
    fetched = await store.get(row.task_id)
    assert fetched is not None
    assert fetched.status == TaskStatus.SUCCEEDED
    assert fetched.started_at == "2026-05-02T12:00:01.000Z"
    assert fetched.finished_at == "2026-05-02T12:00:02.000Z"
    assert fetched.result == {"echoed": 5}


@pytest.mark.asyncio
async def test_update_unknown_raises(store: TaskStore) -> None:
    with pytest.raises(TaskNotFound):
        await store.update_status("nope", TaskStatus.SUCCEEDED)


@pytest.mark.asyncio
async def test_terminal_status_is_immutable(store: TaskStore) -> None:
    # Once a task reaches a terminal state, a later update_status is a
    # silent no-op (NOT a TaskNotFound — the row exists): the terminal row
    # never changes. This keeps a late failure from clobbering a succeeded
    # row and makes cancel-wins deterministic.
    row = _row()
    await store.create(row)
    await store.update_status(
        row.task_id,
        TaskStatus.SUCCEEDED,
        finished_at="2026-05-02T12:00:02.000Z",
        result={"echoed": 5},
    )
    # A later FAILED write must NOT raise and must NOT change anything.
    await store.update_status(
        row.task_id,
        TaskStatus.FAILED,
        finished_at="2026-05-02T13:00:00.000Z",
        error={"type": "late"},
    )
    fetched = await store.get(row.task_id)
    assert fetched is not None
    assert fetched.status == TaskStatus.SUCCEEDED
    assert fetched.finished_at == "2026-05-02T12:00:02.000Z"
    assert fetched.result == {"echoed": 5}
    assert fetched.error is None


@pytest.mark.asyncio
async def test_cancel_wins_over_late_failure(store: TaskStore) -> None:
    row = _row()
    await store.create(row)
    await store.update_status(row.task_id, TaskStatus.CANCELLED)
    # The runner's own error-path write arrives after the cancel — dropped.
    await store.update_status(row.task_id, TaskStatus.FAILED, error={"type": "boom"})
    fetched = await store.get(row.task_id)
    assert fetched is not None
    assert fetched.status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_list_filters_and_orders(store: TaskStore) -> None:
    a = _row(op="echo")
    b = _row(op="ingest")
    await store.create(a)
    await store.create(b)
    await store.update_status(a.task_id, TaskStatus.SUCCEEDED)

    by_status = await store.list_tasks(status=TaskStatus.PENDING)
    assert {r.task_id for r in by_status} == {b.task_id}

    by_op = await store.list_tasks(op="echo")
    assert {r.task_id for r in by_op} == {a.task_id}

    every = await store.list_tasks()
    # created_at is identical in tests, so ordering by it is stable but
    # not deterministic across rows; we only assert membership.
    assert {r.task_id for r in every} == {a.task_id, b.task_id}


@pytest.mark.asyncio
async def test_list_drops_result_and_error_payload(store: TaskStore) -> None:
    """``list_tasks`` is the summary view: ``result`` and ``error`` come
    back as ``None`` regardless of what is persisted. Fetching the heavy
    payload happens via ``get(task_id)`` (or the HTTP ``/result`` endpoint
    one layer up). Keeps `GET /v1/tasks` bandwidth-bounded."""
    row = _row()
    await store.create(row)
    await store.update_status(
        row.task_id,
        TaskStatus.SUCCEEDED,
        finished_at="2026-05-19T12:00:00.000Z",
        result={"large": "x" * 10_000, "items": [1, 2, 3]},
    )

    listed = await store.list_tasks()
    assert len(listed) == 1
    assert listed[0].result is None
    assert listed[0].error is None

    full = await store.get(row.task_id)
    assert full is not None
    assert full.result == {"large": "x" * 10_000, "items": [1, 2, 3]}


@pytest.mark.asyncio
async def test_list_keyset_cursor_pagination(store: TaskStore) -> None:
    """Keyset pagination order is ``(created_at DESC, task_id ASC)`` —
    paging ``limit=2`` over 5 rows (with deliberate ``created_at`` ties)
    yields every row exactly once. The tie-break is the source of truth
    for the HTTP cursor's opaque payload, so it must be stable."""
    t_high = "2026-05-19T12:00:03.000Z"
    t_mid = "2026-05-19T12:00:02.000Z"
    t_low = "2026-05-19T12:00:01.000Z"
    rows = [
        TaskRow(task_id="a1", op="echo", status=TaskStatus.PENDING, created_at=t_high),
        TaskRow(task_id="m1", op="echo", status=TaskStatus.PENDING, created_at=t_mid),
        TaskRow(task_id="m2", op="echo", status=TaskStatus.PENDING, created_at=t_mid),
        TaskRow(task_id="z1", op="echo", status=TaskStatus.PENDING, created_at=t_low),
        TaskRow(task_id="z2", op="echo", status=TaskStatus.PENDING, created_at=t_low),
    ]
    for r in rows:
        await store.create(r)

    page1 = await store.list_tasks(limit=2)
    assert [r.task_id for r in page1] == ["a1", "m1"]

    page2 = await store.list_tasks(
        limit=2,
        after_created_at=page1[-1].created_at,
        after_task_id=page1[-1].task_id,
    )
    assert [r.task_id for r in page2] == ["m2", "z1"]

    page3 = await store.list_tasks(
        limit=2,
        after_created_at=page2[-1].created_at,
        after_task_id=page2[-1].task_id,
    )
    assert [r.task_id for r in page3] == ["z2"]

    page4 = await store.list_tasks(
        limit=2,
        after_created_at=page3[-1].created_at,
        after_task_id=page3[-1].task_id,
    )
    assert page4 == []


@pytest.mark.asyncio
async def test_list_cursor_with_status_filter(store: TaskStore) -> None:
    """Cursor advances *within* the filter result set — the rows skipped
    by ``status=`` must not consume cursor positions, otherwise paging
    through a busy queue would silently drop entries that change status
    mid-walk."""
    t1 = "2026-05-19T12:00:01.000Z"
    t2 = "2026-05-19T12:00:02.000Z"
    t3 = "2026-05-19T12:00:03.000Z"
    a = TaskRow(task_id="aaa", op="echo", status=TaskStatus.PENDING, created_at=t3)
    b = TaskRow(task_id="bbb", op="echo", status=TaskStatus.PENDING, created_at=t2)
    c = TaskRow(task_id="ccc", op="echo", status=TaskStatus.PENDING, created_at=t1)
    for r in (a, b, c):
        await store.create(r)
    await store.update_status(b.task_id, TaskStatus.SUCCEEDED)

    page1 = await store.list_tasks(status=TaskStatus.PENDING, limit=1)
    assert [r.task_id for r in page1] == ["aaa"]

    page2 = await store.list_tasks(
        status=TaskStatus.PENDING,
        limit=1,
        after_created_at=page1[-1].created_at,
        after_task_id=page1[-1].task_id,
    )
    assert [r.task_id for r in page2] == ["ccc"]

    page3 = await store.list_tasks(
        status=TaskStatus.PENDING,
        limit=1,
        after_created_at=page2[-1].created_at,
        after_task_id=page2[-1].task_id,
    )
    assert page3 == []


@pytest.mark.asyncio
async def test_list_running_excludes_terminal(store: TaskStore) -> None:
    pending = _row()
    running = _row()
    done = _row()
    for r in (pending, running, done):
        await store.create(r)
    await store.update_status(running.task_id, TaskStatus.RUNNING)
    await store.update_status(done.task_id, TaskStatus.SUCCEEDED)
    rows = await store.list_running()
    assert {r.task_id for r in rows} == {pending.task_id, running.task_id}


# ---- event tape ---------------------------------------------------------


@pytest.mark.asyncio
async def test_append_event_assigns_monotonic_seq(store: TaskStore) -> None:
    row = _row()
    await store.create(row)

    seq1 = await store.append_event(row.task_id, {"type": "task_started"})
    seq2 = await store.append_event(
        row.task_id, {"type": "progress", "phase": "x"}
    )
    seq3 = await store.append_event(row.task_id, {"type": "final"})
    assert (seq1, seq2, seq3) == (1, 2, 3)


@pytest.mark.asyncio
async def test_list_events_replays_in_seq_order(store: TaskStore) -> None:
    row = _row()
    await store.create(row)

    await store.append_event(row.task_id, {"type": "task_started"})
    await store.append_event(
        row.task_id, {"type": "progress", "phase": "p", "current": 1}
    )
    await store.append_event(row.task_id, {"type": "final"})

    events = await store.list_events(row.task_id)
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert events[0]["type"] == "task_started"
    assert events[1]["phase"] == "p"
    assert "ts" in events[0]


@pytest.mark.asyncio
async def test_append_after_terminal_dropped_keeping_final_last(
    store: TaskStore,
) -> None:
    """A NON-final event appended after the task row is terminal is dropped,
    so the ``final`` stays the last event on the tape (events.py invariant
    "final is always the last event"). Without this guard a cancelled runner's
    in-flight ``reporter.progress`` — whose ``to_thread`` DB write keeps running
    after its await was cancelled (``run_in_executor`` does not stop the worker
    thread) — could land a ``progress`` after the manager's ``final``, stranding
    every consumer that waits for ``events[-1] == 'final'`` (issue #256)."""
    row = _row()
    await store.create(row)
    await store.update_status(row.task_id, TaskStatus.RUNNING)
    await store.append_event(row.task_id, {"type": "task_started"})
    # A non-final event BEFORE the row goes terminal is appended normally.
    await store.append_event(row.task_id, {"type": "progress", "phase": "p"})
    # The manager commits the terminal status immediately before the final.
    await store.update_status(row.task_id, TaskStatus.CANCELLED)
    final_seq = await store.append_event(
        row.task_id, {"type": "final", "status": "cancelled"}
    )
    # An obsolete late progress arriving AFTER terminal + final is dropped:
    # it returns the current max seq and does not advance the tape.
    dropped = await store.append_event(
        row.task_id, {"type": "progress", "phase": "late"}
    )

    events = await store.list_events(row.task_id)
    assert [e["type"] for e in events] == ["task_started", "progress", "final"]
    assert events[-1]["type"] == "final"
    assert dropped == final_seq
    assert await store.max_seq(row.task_id) == final_seq


@pytest.mark.asyncio
async def test_list_events_from_seq_truncates(store: TaskStore) -> None:
    row = _row()
    await store.create(row)
    for i in range(5):
        await store.append_event(row.task_id, {"type": "progress", "i": i})

    tail = await store.list_events(row.task_id, from_seq=3)
    assert [e["seq"] for e in tail] == [3, 4, 5]


@pytest.mark.asyncio
async def test_event_isolation_across_tasks(store: TaskStore) -> None:
    a = _row()
    b = _row()
    await store.create(a)
    await store.create(b)
    await store.append_event(a.task_id, {"type": "x"})
    await store.append_event(a.task_id, {"type": "y"})
    await store.append_event(b.task_id, {"type": "z"})
    a_events = await store.list_events(a.task_id)
    b_events = await store.list_events(b.task_id)
    assert [e["seq"] for e in a_events] == [1, 2]
    assert [e["seq"] for e in b_events] == [1]


@pytest.mark.asyncio
async def test_list_events_with_limit_returns_at_most_n(store: TaskStore) -> None:
    row = _row()
    await store.create(row)
    for i in range(5):
        await store.append_event(row.task_id, {"type": "progress", "i": i})

    page = await store.list_events(row.task_id, limit=3)
    assert [e["seq"] for e in page] == [1, 2, 3]


@pytest.mark.asyncio
async def test_list_events_with_limit_and_from_seq(store: TaskStore) -> None:
    row = _row()
    await store.create(row)
    for i in range(5):
        await store.append_event(row.task_id, {"type": "progress", "i": i})

    page = await store.list_events(row.task_id, from_seq=2, limit=2)
    assert [e["seq"] for e in page] == [2, 3]


@pytest.mark.asyncio
async def test_list_events_no_limit_still_returns_all(store: TaskStore) -> None:
    """Backwards-safe: omitting `limit` keeps the original full-replay
    behaviour. Guards against a refactor that defaults to a tight cap."""
    row = _row()
    await store.create(row)
    for i in range(7):
        await store.append_event(row.task_id, {"type": "progress", "i": i})

    page = await store.list_events(row.task_id)
    assert [e["seq"] for e in page] == [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.asyncio
async def test_max_seq_returns_highest_event_seq(store: TaskStore) -> None:
    row = _row()
    await store.create(row)
    assert await store.max_seq(row.task_id) == 0
    await store.append_event(row.task_id, {"type": "x"})
    await store.append_event(row.task_id, {"type": "y"})
    await store.append_event(row.task_id, {"type": "z"})
    assert await store.max_seq(row.task_id) == 3


@pytest.mark.asyncio
async def test_max_seq_unknown_task_returns_zero(store: TaskStore) -> None:
    assert await store.max_seq("no-such-task") == 0
