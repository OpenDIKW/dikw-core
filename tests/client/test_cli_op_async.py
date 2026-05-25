"""CLI op-command contract: async-by-default + ``--wait`` opt-in.

Covers the task-submitting op commands — ``ingest``, ``synth``,
``eval``, ``lint propose``, ``lint apply`` — flipped from blocking-by-
default (stream until terminal) to async-by-default (submit + print
task handle + exit 0). The blocking shape is opt-in via ``--wait`` and
the exit-code mapping under that flag is the agent contract.

Exit code mapping under ``--wait``:

* 0 — task ``succeeded``
* 1 — task ``failed``
* 130 — task ``cancelled`` (POSIX SIGINT convention)
* 124 — client-side timeout fired (POSIX timeout convention)

Tests drive the FastAPI runtime in-memory via the shared
``patch_transport_factory`` fixture (see ``tests/conftest.py``) — no
sockets, no flakes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from typer.testing import CliRunner

from dikw_core.cli import app
from dikw_core.server import synth_op
from dikw_core.server.runtime import ServerRuntime

from ..fakes import FakeEmbeddings, FakeLLM


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(app, args)


# ---- async-by-default --------------------------------------------------


def test_ingest_default_async_prints_task_handle_exit_zero(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``--wait``: submit, print the handle JSON, exit 0
    immediately. The task itself may still be running — we don't follow
    it."""
    monkeypatch.setattr("dikw_core.api.build_embedder", lambda _cfg: FakeEmbeddings())
    patch_transport_factory()
    result = _run(["client", "ingest", "--no-embed"])
    assert result.exit_code == 0, result.stdout
    handle = json.loads(result.stdout)
    assert isinstance(handle.get("task_id"), str) and handle["task_id"]
    assert handle.get("status") in {"pending", "running", "succeeded"}
    assert handle.get("events_url") == f"/v1/tasks/{handle['task_id']}/events"
    assert (
        handle.get("wait_command")
        == f"dikw client tasks wait {handle['task_id']}"
    )


def test_synth_default_async_prints_task_handle(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(synth_op, "build_llm", lambda _cfg, **_kw: FakeLLM())
    monkeypatch.setattr(synth_op, "build_embedder", lambda _cfg: FakeEmbeddings())
    patch_transport_factory()
    result = _run(["client", "synth"])
    assert result.exit_code == 0, result.stdout
    handle = json.loads(result.stdout)
    assert handle.get("task_id")
    assert "events_url" in handle


def test_lint_propose_default_async_prints_task_handle(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["client", "lint", "propose", "--rule", "broken_wikilink"])
    assert result.exit_code == 0, result.stdout
    handle = json.loads(result.stdout)
    assert handle.get("task_id")


# ---- --wait opt-in, exit-code mapping ----------------------------------


def test_ingest_wait_renders_report_exits_zero(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--wait`` follows the task to terminal, renders the
    ``IngestReport`` table, and maps ``succeeded`` to exit 0."""
    monkeypatch.setattr("dikw_core.api.build_embedder", lambda _cfg: FakeEmbeddings())
    patch_transport_factory()
    result = _run(["client", "ingest", "--no-embed", "--wait", "--plain"])
    assert result.exit_code == 0, result.stdout
    # Report table renders the standard metric labels.
    assert "scanned" in result.stdout.lower()


