"""PR2 tracing: the background-task span links back to the request span.

``TaskManager.submit`` captures the live request span context, and the detached
``_run`` coroutine opens a NEW ROOT span linked to it (the OTel idiom for
request-triggered fire-and-forget work, since the request span ends before the
task runs). These drive the manager directly (no FastAPI) with an in-memory
span exporter to pin: root-not-child, the link target, the dikw.* attributes,
and the OK / cancelled outcomes.
"""

from __future__ import annotations

import asyncio
from typing import Any

from dikw_core import telemetry
from dikw_core.progress import ProgressReporter
from dikw_core.server.tasks import SqliteTaskStore, TaskManager


async def _wait_terminal(store: SqliteTaskStore, task_id: str, *, timeout: float = 30.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        row = await store.get(task_id)
        if row is not None and row.status.value in ("succeeded", "failed", "cancelled"):
            events = await store.list_events(task_id)
            if events and events[-1].get("type") == "final":
                return
        await asyncio.sleep(0.01)
    raise AssertionError(f"task {task_id} did not reach terminal state in {timeout}s")


async def _wait_for_span(exporter: Any, name: str, *, timeout: float = 5.0) -> Any:
    """The task span ends after the final event is emitted (on ``with`` exit),
    so it can lag ``_wait_terminal``; poll the exporter until it lands."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        for s in exporter.get_finished_spans():
            if s.name == name:
                return s
        await asyncio.sleep(0.01)
    raise AssertionError(f"span {name!r} was not exported within {timeout}s")


async def test_task_span_is_root_linked_to_request_with_attrs(
    manager_only: tuple[TaskManager, SqliteTaskStore],
    span_exporter: Any,
) -> None:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode

    manager, store = manager_only

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        await reporter.progress(phase="step", current=1, total=1)
        return {"ok": True}

    tracer = trace.get_tracer("test.request")
    with tracer.start_as_current_span("http.request") as request_span:
        request_span_id = request_span.get_span_context().span_id
        # submit() runs inside the request span — that's where it captures the
        # context the detached task links back to.
        row = await manager.submit(op="echo", runner=_runner, base_id="base-xyz")

    await _wait_terminal(store, row.task_id)
    span = await _wait_for_span(span_exporter, "dikw.task.echo")

    # Root, not a child of the (already-ended) request span.
    assert span.parent is None
    # Linked back to the request span.
    assert len(span.links) == 1
    assert span.links[0].context.span_id == request_span_id
    # dikw.* attributes.
    assert span.attributes[telemetry.DIKW_OP] == "echo"
    assert span.attributes[telemetry.DIKW_TASK_ID] == row.task_id
    assert span.attributes[telemetry.DIKW_BASE_ID] == "base-xyz"
    assert span.status.status_code == StatusCode.OK


async def test_task_span_marks_cancel_not_error(
    manager_only: tuple[TaskManager, SqliteTaskStore],
    span_exporter: Any,
) -> None:
    from opentelemetry.trace import StatusCode

    manager, store = manager_only

    async def _blocker(reporter: ProgressReporter) -> dict[str, Any]:
        while True:
            reporter.cancel_token().raise_if_cancelled()
            await asyncio.sleep(0.01)

    row = await manager.submit(op="echo", runner=_blocker, base_id="b")
    # Wait until the coroutine has actually entered the try block (status
    # RUNNING) before cancelling — cancelling a not-yet-scheduled task throws
    # CancelledError before the span/try opens, so no final event is emitted.
    for _ in range(500):
        r = await store.get(row.task_id)
        if r is not None and r.status.value == "running":
            break
        await asyncio.sleep(0.01)
    assert await manager.cancel(row.task_id) is True

    await _wait_terminal(store, row.task_id)
    span = await _wait_for_span(span_exporter, "dikw.task.echo")
    # A user cancel is a graceful terminal — flagged, not an error status.
    assert span.attributes[telemetry.DIKW_CANCELLED] is True
    assert span.status.status_code != StatusCode.ERROR
