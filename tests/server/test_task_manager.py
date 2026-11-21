"""TaskManager + ProgressBus + TaskBusReporter behavioural tests.

Drives the manager directly (no FastAPI) to cover:
  * happy-path submit → task_started → progress* → final{succeeded}
  * failure path → final{failed} with traceback in error
  * cancellation: pre-cancel + mid-flight cancel both end in final{cancelled}
  * multi-subscriber fanout via the bus
  * resume-by-seq replay through ``store.list_events``
  * restart_cleanup marks orphan rows as failed{server_restart}
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from dikw_core.progress import ProgressReporter
from dikw_core.server.tasks import (
    SqliteTaskStore,
    TaskManager,
    TaskStatus,
)


async def _wait_terminal(
    store: SqliteTaskStore, task_id: str, *, timeout: float = 30.0
) -> None:
    """Wait until the task is fully terminal — both the row's status
    is in a terminal state AND the matching ``final`` event has been
    appended. The TaskManager updates the row before emitting the
    final event (so /result is consistent the moment a follower sees
    ``final``); waiting only on the row therefore races the final
    event onto the tape, which is fine for /result-style tests but
    breaks any test that reads the event tape next."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        row = await store.get(task_id)
        if row is not None and row.status in (
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ):
            events = await store.list_events(task_id)
            if events and events[-1].get("type") == "final":
                return
        await asyncio.sleep(0.01)
    raise AssertionError(f"task {task_id} did not reach terminal state in {timeout}s")


@pytest.mark.asyncio
async def test_happy_path_emits_full_event_tape(
    manager_only: tuple[TaskManager, SqliteTaskStore],
) -> None:
    manager, store = manager_only

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        await reporter.progress(phase="step", current=1, total=2)
        await reporter.progress(phase="step", current=2, total=2)
        await reporter.partial("hello", {"world": True})
        return {"ok": True}

    row = await manager.submit(op="echo", runner=_runner, params={"a": 1})
    await _wait_terminal(store, row.task_id)
    final_row = await store.get(row.task_id)
    assert final_row is not None
    assert final_row.status == TaskStatus.SUCCEEDED
    assert final_row.result == {"ok": True}
    assert final_row.params_digest, "params_digest should be a non-empty hash"

    events = await store.list_events(row.task_id)
    types = [e["type"] for e in events]
    assert types == [
        "task_started",
        "progress",
        "progress",
        "partial",
        "final",
    ]
    assert events[0]["task_id"] == row.task_id
    assert events[0]["op"] == "echo"
    assert events[-1]["status"] == "succeeded"
    # seq strictly monotonic
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)
    assert seqs[0] == 1


@pytest.mark.asyncio
async def test_failure_path_records_traceback(
    manager_only: tuple[TaskManager, SqliteTaskStore],
) -> None:
    manager, store = manager_only

    async def _runner(_reporter: ProgressReporter) -> dict[str, Any]:
        raise ValueError("boom")

    row = await manager.submit(op="echo", runner=_runner)
    await _wait_terminal(store, row.task_id)
    final_row = await store.get(row.task_id)
    assert final_row is not None
    assert final_row.status == TaskStatus.FAILED
    assert final_row.error is not None
    assert final_row.error["code"] == "ValueError"
    assert final_row.error["message"] == "boom"
    assert "Traceback" in final_row.error["traceback"]


@pytest.mark.asyncio
async def test_pre_cancel_via_token(
    manager_only: tuple[TaskManager, SqliteTaskStore],
) -> None:
    manager, store = manager_only
    started = asyncio.Event()

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        started.set()
        # Honour cancel token at the first checkpoint.
        for i in range(10):
            reporter.cancel_token().raise_if_cancelled()
            await reporter.progress(phase="x", current=i, total=10)
            await asyncio.sleep(0.01)
        return {"done": True}

    row = await manager.submit(op="echo", runner=_runner)
    # Wait deterministically until the runner has started (``_run`` is inside
    # its try block) before cancelling, so the cancel lands where the except
    # arm records a CANCELLED terminal + ``final``. A bare ``sleep`` here only
    # *probably* schedules the task — a single ``sleep(0)`` was tight enough on
    # 3.13 + slow CI to occasionally leave the task PENDING when ``cancel()``
    # arrived, stranding it (no ``final``) → a phantom 5s-wait hang. The
    # ``started`` Event removes the race entirely (same pattern as
    # ``test_cancel_after_terminal_emits_no_contradictory_final``).
    await asyncio.wait_for(started.wait(), timeout=10.0)
    await manager.cancel(row.task_id)
    await _wait_terminal(store, row.task_id)
    final_row = await store.get(row.task_id)
    assert final_row is not None
    assert final_row.status == TaskStatus.CANCELLED
    events = await store.list_events(row.task_id)
    assert events[-1]["type"] == "final"
    assert events[-1]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_unknown_task_returns_false(
    manager_only: tuple[TaskManager, SqliteTaskStore],
) -> None:
    manager, _store = manager_only
    assert await manager.cancel("nope") is False


@pytest.mark.asyncio
async def test_cancel_after_terminal_emits_no_contradictory_final(
    manager_only: tuple[TaskManager, SqliteTaskStore],
) -> None:
    """If a run commits a terminal status in the instant before a cancel
    lands, the cancel arm's CANCELLED write no-ops (terminal is immutable)
    and it must NOT append a ``cancelled`` final that contradicts the
    persisted (succeeded) row. Reproduce the race deterministically by
    committing SUCCEEDED out-of-band while the runner is blocked, then
    cancelling."""
    manager, store = manager_only
    started = asyncio.Event()
    release = asyncio.Event()

    async def _runner(_reporter: ProgressReporter) -> dict[str, Any]:
        started.set()
        await release.wait()  # CancelledError is injected here
        return {"unreached": True}

    row = await manager.submit(op="echo", runner=_runner)
    await asyncio.wait_for(started.wait(), timeout=5.0)
    # The run reaches a terminal status just before the cancel lands.
    await store.update_status(row.task_id, TaskStatus.SUCCEEDED, result={"ok": 1})
    await manager.cancel(row.task_id)

    # Wait for _run to finish (it removes itself from _running in its finally).
    deadline = asyncio.get_event_loop().time() + 5.0
    while asyncio.get_event_loop().time() < deadline:
        async with manager._lock:
            finished = row.task_id not in manager._running
        if finished:
            break
        await asyncio.sleep(0.02)
    else:  # pragma: no cover - only on a hang
        raise AssertionError("run did not finish after cancel")

    final_row = await store.get(row.task_id)
    assert final_row is not None
    assert final_row.status == TaskStatus.SUCCEEDED  # cancel did not clobber it
    events = await store.list_events(row.task_id)
    contradictory = [
        e for e in events if e.get("type") == "final" and e.get("status") == "cancelled"
    ]
    assert contradictory == []


@pytest.mark.asyncio
async def test_resume_by_seq_via_store(
    manager_only: tuple[TaskManager, SqliteTaskStore],
) -> None:
    manager, store = manager_only

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        for i in range(5):
            await reporter.progress(phase="p", current=i, total=5)
        return {"ok": True}

    row = await manager.submit(op="echo", runner=_runner)
    await _wait_terminal(store, row.task_id)

    full = await store.list_events(row.task_id)
    assert len(full) >= 7  # task_started + 5 progress + final, possibly more
    tail = await store.list_events(row.task_id, from_seq=4)
    assert tail[0]["seq"] == 4
    assert {e["seq"] for e in tail} == {e["seq"] for e in full if e["seq"] >= 4}


# ---- per-task asyncio.Condition wakes long-poll waiters ---------------


@pytest.mark.asyncio
async def test_condition_notifies_on_append(
    manager_only: tuple[TaskManager, SqliteTaskStore],
) -> None:
    """A coroutine waiting on ``manager.condition_for(task_id)`` must
    wake within notify latency (target <100ms) when the runner emits an
    event. This is the foundation of the long-poll endpoint — without
    it the `wait=K` handler would always burn its full timeout."""
    manager, store = manager_only
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        entered.set()
        await gate.wait()
        await reporter.progress(phase="x", current=1, total=1)
        return {"done": True}

    row = await manager.submit(op="echo", runner=_runner)
    # Wait until the runner enters its body so ``task_started`` (and its notify)
    # has already completed before parking ``_waiter``.
    await asyncio.wait_for(entered.wait(), timeout=5.0)
    cond = manager.condition_for(row.task_id)

    woke = asyncio.Event()
    parked = asyncio.Event()

    async def _waiter() -> None:
        async with cond:
            parked.set()
            await cond.wait()
            woke.set()

    waiter_task = asyncio.create_task(_waiter())
    # Deterministic handoff: once ``parked`` fires inside ``async with cond``,
    # re-acquiring ``cond`` guarantees ``_waiter`` has released the lock inside
    # ``cond.wait()`` and is genuinely parked before we unblock the runner.
    await asyncio.wait_for(parked.wait(), timeout=2.0)
    async with cond:
        pass
    gate.set()

    try:
        await asyncio.wait_for(woke.wait(), timeout=1.0)
    finally:
        waiter_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await waiter_task
        await _wait_terminal(store, row.task_id)


@pytest.mark.asyncio
async def test_condition_notifies_on_terminal_transition(
    manager_only: tuple[TaskManager, SqliteTaskStore],
) -> None:
    """Even a runner that never emits a progress event must wake the
    waiter when the task transitions terminal — otherwise a quiet task
    would silently stall every long-poll handler until their wait
    timeout fires."""
    manager, store = manager_only
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def _runner(_reporter: ProgressReporter) -> dict[str, Any]:
        entered.set()
        await gate.wait()
        return {"ok": True}

    row = await manager.submit(op="echo", runner=_runner)
    await asyncio.wait_for(entered.wait(), timeout=5.0)
    cond = manager.condition_for(row.task_id)

    woke = asyncio.Event()
    parked = asyncio.Event()

    async def _waiter() -> None:
        async with cond:
            parked.set()
            await cond.wait()
            woke.set()

    waiter_task = asyncio.create_task(_waiter())
    await asyncio.wait_for(parked.wait(), timeout=2.0)
    async with cond:
        pass
    gate.set()

    try:
        await asyncio.wait_for(woke.wait(), timeout=1.0)
    finally:
        waiter_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await waiter_task
        await _wait_terminal(store, row.task_id)


@pytest.mark.asyncio
async def test_condition_per_task_isolation(
    manager_only: tuple[TaskManager, SqliteTaskStore],
) -> None:
    """Notify on task A's condition must not wake a waiter parked on
    task B's condition. Per-task isolation; conditions are not shared."""
    manager, store = manager_only
    a_entered = asyncio.Event()
    b_entered = asyncio.Event()
    gate_a = asyncio.Event()
    gate_b_finishes = asyncio.Event()

    async def _runner_a(reporter: ProgressReporter) -> dict[str, Any]:
        a_entered.set()
        await gate_a.wait()
        await reporter.progress(phase="a", current=1, total=1)
        return {"a": True}

    async def _runner_b(_reporter: ProgressReporter) -> dict[str, Any]:
        b_entered.set()
        await gate_b_finishes.wait()
        return {"b": True}

    row_a = await manager.submit(op="echo", runner=_runner_a)
    row_b = await manager.submit(op="echo", runner=_runner_b)

    # Waiting for both runners to enter their bodies guarantees TaskManager has
    # already finished appending AND notifying ``task_started`` on both tasks
    # before we park B's waiter — zero timing sleeps needed.
    await asyncio.wait_for(a_entered.wait(), timeout=5.0)
    await asyncio.wait_for(b_entered.wait(), timeout=5.0)

    cond_b = manager.condition_for(row_b.task_id)
    b_woke = asyncio.Event()
    b_parked = asyncio.Event()

    async def _waiter_b() -> None:
        async with cond_b:
            b_parked.set()
            await cond_b.wait()
            b_woke.set()

    waiter_b_task = asyncio.create_task(_waiter_b())
    await asyncio.wait_for(b_parked.wait(), timeout=2.0)
    async with cond_b:
        pass

    # Let A run all the way through progress + terminal notifies + manager
    # cleanup before asserting B's waiter stayed asleep.
    gate_a.set()
    await _wait_terminal(store, row_a.task_id)
    deadline = asyncio.get_event_loop().time() + 5.0
    while asyncio.get_event_loop().time() < deadline:
        async with manager._lock:
            done_a = row_a.task_id not in manager._running
        if done_a:
            break
        await asyncio.sleep(0.005)
    else:  # pragma: no cover - only on a hang
        raise AssertionError("task A did not finish manager cleanup")
    await asyncio.sleep(0)
    assert not b_woke.is_set(), "B's waiter woke on A's notify — conditions not per-task"

    # Cleanup: let B finish so the waiter does wake (proves the waiter
    # was correctly parked, not deadlocked).
    gate_b_finishes.set()
    try:
        await asyncio.wait_for(b_woke.wait(), timeout=1.0)
    finally:
        waiter_b_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await waiter_b_task
        await _wait_terminal(store, row_a.task_id)
        await _wait_terminal(store, row_b.task_id)


@pytest.mark.asyncio
async def test_restart_cleanup_marks_orphans_failed(
    manager_only: tuple[TaskManager, SqliteTaskStore],
) -> None:
    _manager, store = manager_only
    # Simulate a leftover row from a prior process: PENDING in storage,
    # nothing in the in-memory manager.
    from dikw_core.server.tasks import TaskRow

    leftover = TaskRow(
        task_id="leftover-1",
        op="echo",
        status=TaskStatus.RUNNING,
        created_at="2026-05-02T11:00:00.000Z",
        started_at="2026-05-02T11:00:01.000Z",
    )
    await store.create(leftover)

    # New manager on the same store mimics a fresh server boot.
    new_manager = TaskManager(store=store)
    await new_manager.restart_cleanup()

    row = await store.get("leftover-1")
    assert row is not None
    assert row.status == TaskStatus.FAILED
    assert row.error == {"reason": "server_restart"}
    events = await store.list_events("leftover-1")
    # The synthetic final event lands on the tape so any latent NDJSON
    # subscriber sees a clean terminator.
    assert events[-1]["type"] == "final"
    assert events[-1]["status"] == "failed"
