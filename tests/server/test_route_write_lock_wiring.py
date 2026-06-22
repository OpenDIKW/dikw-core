"""F2 / RC-2 wiring regression: the synth + lint-apply ROUTES must thread the
runtime's shared ``ingest_lock`` into their runner factories.

``test_runner_write_lock.py`` proves the runners serialize *given* a shared
lock; it can't catch a regression where ``routes_tasks.py`` passes ``lock=None``
(or a fresh per-request lock) — that would silently defeat F2's cross-op
serialization while every runner-level test stayed green, because each of those
constructs its own lock. These tests close that gap by asserting the lock the
route hands the factory IS ``rt.ingest_lock`` (object identity).
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from dikw_core.server import routes_tasks as routes_tasks_module
from dikw_core.server.runtime import ServerRuntime

from .conftest import wait_task_terminal as _wait_terminal


def _capture_lock(captured: dict[str, Any]) -> Any:
    """A runner-factory stand-in that records the ``lock`` kwarg and returns a
    trivial runner so the submitted task terminates immediately."""

    def _factory(**kwargs: Any) -> Any:
        captured["lock"] = kwargs.get("lock")

        async def _runner(_reporter: Any) -> dict[str, Any]:
            return {}

        return _runner

    return _factory


@pytest.mark.asyncio
async def test_synth_route_wires_ingest_lock(
    server_client: httpx.AsyncClient,
    runtime: ServerRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        routes_tasks_module, "make_synth_runner", _capture_lock(captured)
    )
    resp = await server_client.post(
        "/v1/synth", json={"force_all": False, "no_embed": True}
    )
    assert resp.status_code == 200, resp.text
    assert captured.get("lock") is runtime.ingest_lock, (
        "submit_synth must pass rt.ingest_lock into make_synth_runner"
    )
    await _wait_terminal(server_client, resp.json()["task_id"], timeout=10.0)


@pytest.mark.asyncio
async def test_lint_apply_route_wires_ingest_lock(
    server_client: httpx.AsyncClient,
    runtime: ServerRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        routes_tasks_module, "make_lint_apply_runner", _capture_lock(captured)
    )
    resp = await server_client.post(
        "/v1/lint/apply", json={"proposal_task_id": "prop-x"}
    )
    assert resp.status_code == 200, resp.text
    assert captured.get("lock") is runtime.ingest_lock, (
        "submit_lint_apply must pass rt.ingest_lock into make_lint_apply_runner"
    )
    await _wait_terminal(server_client, resp.json()["task_id"], timeout=10.0)
