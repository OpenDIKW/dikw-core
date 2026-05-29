"""``dikw client tasks *`` subgroup tests.

After the task-first CLI flip, the subgroup is:

* ``tasks list`` (unchanged) — list rows
* ``tasks status <id>`` (renamed from ``show``) — single-row JSON
* ``tasks events <id> --from-seq N --limit M --wait K`` (new) — single
  paged GET to the cursor endpoint, raw JSON to stdout
* ``tasks wait <id> --poll-wait K --timeout S`` (new) — long-poll loop
  until terminal, exit code mapped (0 / 1 / 130 / 124)
* ``tasks cancel <id>`` — agent-first JSON by default, ``--pretty``
  opts in to the colored human line
* ~~``tasks follow``~~ — removed; ``tasks wait`` covers it

This file owns the contract for the new commands + the removed-shape
regression coverage; ``tasks list`` already has coverage elsewhere.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from typer.testing import CliRunner

from dikw_core.cli import app
from dikw_core.server.runtime import ServerRuntime

from ..fakes import FakeEmbeddings


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(app, args)


# ---- status (renamed from show) ----------------------------------------


def test_tasks_status_returns_row_json(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dikw_core.server.ingest_op.build_embedder", lambda _cfg: FakeEmbeddings()
    )
    patch_transport_factory()
    submit = _run(["client", "ingest", "--no-embed"])
    assert submit.exit_code == 0, submit.stdout
    task_id = json.loads(submit.stdout)["task_id"]

    result = _run(["client", "tasks", "status", task_id])
    assert result.exit_code == 0, result.stdout
    row = json.loads(result.stdout)
    assert row["task_id"] == task_id
    assert row["op"] == "ingest"
    # CliRunner's per-invocation asyncio.run loop can race the
    # server-side runner: a fast runner finishes inside the first
    # invocation (-> succeeded); a slow one gets cancelled when the
    # loop tears down. Either is a valid terminal state for this
    # smoke test — we're asserting the row shape, not the task fate.
    assert row["status"] in {
        "pending", "running", "succeeded", "failed", "cancelled",
    }


def test_tasks_show_removed(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """The legacy ``show`` name is gone — Typer must report "No such
    command" so scripts blindly using the old name fail loudly instead
    of silently dropping."""
    patch_transport_factory()
    r = _run(["client", "tasks", "show", "anything"])
    assert r.exit_code != 0


# ---- events (new) ------------------------------------------------------


def test_tasks_events_single_page_json(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tasks events <id>`` is the agent's paging primitive — one HTTP
    call, raw ``EventsPage`` JSON to stdout, exit 0 regardless of task
    state. Lets agents script a follow-up cursor advance themselves
    without committing to the blocking ``tasks wait`` shape."""
    monkeypatch.setattr(
        "dikw_core.server.ingest_op.build_embedder", lambda _cfg: FakeEmbeddings()
    )
    patch_transport_factory()
    submit = _run(["client", "ingest", "--no-embed"])
    task_id = json.loads(submit.stdout)["task_id"]
    # Make sure the task has had time to land at least task_started.
    _run(["client", "tasks", "wait", task_id])

    result = _run(
        ["client", "tasks", "events", task_id, "--from-seq", "0", "--limit", "5", "--wait", "0"]
    )
    assert result.exit_code == 0, result.stdout
    page = json.loads(result.stdout)
    assert page["task_id"] == task_id
    assert isinstance(page["events"], list) and page["events"]
    assert {"next_from_seq", "has_more", "last_seq", "task_status"} <= set(page)


# ---- wait (new) --------------------------------------------------------


def test_tasks_wait_exits_with_terminal_status(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tasks wait`` blocks until terminal and maps the final status
    to the standard exit code.

    Smoke-level only: deep status/exit-code coverage lives at the
    helper level in ``test_task_follow.py`` (which runs in a single
    pytest-asyncio loop so slow tasks don't get cancelled by
    CliRunner's per-invocation ``asyncio.run`` teardown). Here we
    only assert the command exits in the success/cancelled band, not
    which one — because CliRunner's loop lifecycle can race the
    server-side runner."""
    monkeypatch.setattr(
        "dikw_core.server.ingest_op.build_embedder", lambda _cfg: FakeEmbeddings()
    )
    patch_transport_factory()
    submit = _run(["client", "ingest", "--no-embed"])
    task_id = json.loads(submit.stdout)["task_id"]

    result = _run(["client", "tasks", "wait", task_id, "--plain"])
    # 0 (succeeded) or 130 (cancelled by CliRunner loop teardown).
    # The wiring is what matters: the command resolves to a terminal
    # status and returns a stable exit code from the mapping table.
    assert result.exit_code in {0, 130}, result.stdout


def test_tasks_follow_removed(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """``follow`` is replaced by ``wait`` (which renders identically
    when a TTY); Typer must reject the old name so scripts blindly
    using it fail loudly."""
    patch_transport_factory()
    r = _run(["client", "tasks", "follow", "anything"])
    assert r.exit_code != 0


# ---- cancel ------------------------------------------------------------


def test_tasks_cancel_default_emits_json(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent-first default: ``tasks cancel`` writes the raw
    ``CancelResponse`` JSON (``{task_id, cancelled, already_terminal}``)
    to stdout so agents can parse it without stripping rich ANSI."""
    monkeypatch.setattr(
        "dikw_core.server.ingest_op.build_embedder", lambda _cfg: FakeEmbeddings()
    )
    patch_transport_factory()
    submit = _run(["client", "ingest", "--no-embed"])
    task_id = json.loads(submit.stdout)["task_id"]

    result = _run(["client", "tasks", "cancel", task_id])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["task_id"] == task_id
    assert isinstance(payload["cancelled"], bool)
    assert isinstance(payload["already_terminal"], bool)


def test_tasks_cancel_pretty_emits_human_line(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--pretty`` opts in to the colored human line — must not emit
    JSON, must mention the task id, and must distinguish the
    already-terminal no-op from a live cancel request."""
    monkeypatch.setattr(
        "dikw_core.server.ingest_op.build_embedder", lambda _cfg: FakeEmbeddings()
    )
    patch_transport_factory()
    submit = _run(["client", "ingest", "--no-embed"])
    task_id = json.loads(submit.stdout)["task_id"]

    result = _run(["client", "tasks", "cancel", task_id, "--pretty"])
    assert result.exit_code == 0, result.stdout
    # Not JSON — stripping ANSI would still leave free-form text.
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)
    assert task_id in result.stdout
    assert ("already terminal" in result.stdout) or (
        "cancel requested" in result.stdout
    )
