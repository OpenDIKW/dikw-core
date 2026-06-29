"""Shared fixtures for server-side tests.

Two flavours of test infrastructure:

  * ``server_client`` — full FastAPI app wired to a real (test) wiki via
    ``build_app``. The server runs in-process via ``ASGITransport`` so
    no socket is bound. Suits routes_sync + routes_tasks integration
    tests where the engine should actually exercise.

  * ``manager_only`` — naked ``TaskManager`` + ``SqliteTaskStore``
    pair for tests that only care about the task subsystem semantics.
    Cheap, no FastAPI overhead.

  * ``ingested_wiki`` — extends ``server_client`` by copying a small
    fixture corpus into the wiki's ``sources/`` and running ``ingest``
    via the engine API (skips the HTTP import path so test setup stays
    cheap). Several route tests need a wiki with both documents and
    embeddings populated; reuse this fixture instead of cloning the
    setup per file.
"""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from dikw_core import api as api_module
from dikw_core.server.app import build_app
from dikw_core.server.auth import AuthConfig
from dikw_core.server.runtime import ServerRuntime, build_runtime, teardown_runtime
from dikw_core.server.tasks import SqliteTaskStore, TaskManager

from ..fakes import FakeEmbeddings, init_test_base

FIXTURES_NOTES = Path(__file__).parent.parent / "fixtures" / "notes"


@pytest.fixture()
def base_root(tmp_path: Path) -> Path:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki, description="server-test wiki")
    return wiki


@pytest.fixture()
async def runtime(base_root: Path) -> AsyncIterator[ServerRuntime]:
    """Live runtime backed by a fresh tmp wiki, no auth."""
    auth = AuthConfig(host="127.0.0.1", token=None)
    rt = await build_runtime(root=base_root, auth=auth)
    try:
        yield rt
    finally:
        await teardown_runtime(rt)


def _build_test_app(rt: ServerRuntime) -> FastAPI:
    """``build_app`` with a runtime factory that hands back the already
    constructed runtime (no per-test rebuild)."""

    async def _factory() -> ServerRuntime:
        return rt

    return build_app(runtime_factory=_factory, auth=rt.auth)


@pytest.fixture()
async def server_client(
    runtime: ServerRuntime,
) -> AsyncIterator[httpx.AsyncClient]:
    """``httpx.AsyncClient`` bound to the in-memory FastAPI app."""
    app = _build_test_app(runtime)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        # Trigger the lifespan startup so app.state.runtime is set —
        # ASGITransport doesn't fire it automatically until the first
        # request, but we want failures to surface here, not at the
        # call site.
        async with app.router.lifespan_context(app):
            yield client


@pytest.fixture()
async def server_client_with_token(
    base_root: Path,
) -> AsyncIterator[tuple[httpx.AsyncClient, str]]:
    """Token-required variant. The host stays loopback (127.0.0.1) but a
    token is set, which still triggers token-required mode per
    ``AuthConfig.required``."""
    auth = AuthConfig(host="127.0.0.1", token="s3cret")
    rt = await build_runtime(root=base_root, auth=auth)
    app = _build_test_app(rt)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            async with app.router.lifespan_context(app):
                yield client, "s3cret"
    finally:
        await teardown_runtime(rt)


@pytest.fixture()
async def ingested_wiki(
    server_client: httpx.AsyncClient,
    base_root: Path,
) -> Path:
    """Wiki with the standard ``tests/fixtures/notes`` corpus ingested
    via ``api.ingest`` + ``FakeEmbeddings``. Used by query / retrieve /
    health route tests that need both documents and embeddings.
    """
    dest = base_root / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES_NOTES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    await api_module.ingest(base_root, embedder=FakeEmbeddings())
    _ = server_client  # ensure runtime lifespan is up before we ingest
    return base_root


@pytest.fixture()
async def manager_only(
    tmp_path: Path,
) -> AsyncIterator[tuple[TaskManager, SqliteTaskStore]]:
    """``TaskManager`` + fresh SQLite store, no FastAPI."""
    store = SqliteTaskStore(path=tmp_path / "tasks.db")
    await store.init()
    manager = TaskManager(store=store)
    try:
        yield manager, store
    finally:
        await manager.shutdown()
        await store.close()


async def wait_task_terminal(
    client: httpx.AsyncClient, task_id: str, *, timeout: float = 10.0
) -> dict[str, Any]:
    """Poll ``GET /v1/tasks/{id}`` until status is terminal; return the row.

    Shared by every HTTP-level task test that needs to wait for a
    submitted runner to finish before asserting on the ``/result`` payload.
    The status row is terminal the instant ``/result`` is consistent, but
    NOT a sound proxy for "the ``final`` event is on the tape" — a test that
    reads the event tape and asserts on its last entry must use
    ``wait_event_tape_final`` instead (see its docstring for the race).
    Default 10s timeout — synth/eval paths that need more should pass an
    explicit value."""
    import asyncio as _asyncio

    deadline = _asyncio.get_event_loop().time() + timeout
    while _asyncio.get_event_loop().time() < deadline:
        r = await client.get(f"/v1/tasks/{task_id}")
        if r.status_code == 200 and r.json()["status"] in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            return r.json()
        await _asyncio.sleep(0.05)
    raise AssertionError(f"task {task_id} never reached a terminal state")


async def wait_event_tape_final(
    client: httpx.AsyncClient, task_id: str, *, timeout: float = 10.0
) -> list[dict[str, Any]]:
    """Poll ``GET /v1/tasks/{id}/events`` until the *complete* tape's last
    event is the terminal ``final`` envelope; return the full event list.

    ``TaskManager._run`` flips the task status row to its terminal state
    *before* appending the ``final`` event to the tape (so a follower that
    sees ``final`` and immediately calls ``/result`` always finds the row
    terminal — the reverse order races ``task_not_terminal``). A test that
    trusts ``wait_task_terminal`` (the status-row proxy) and then reads the
    tape with ``wait=0`` therefore races the trailing ``progress`` event onto
    the last slot, intermittently seeing ``'progress'`` where it expects
    ``'final'``. Such tests must wait for the signal they actually depend on —
    the ``final`` event itself. This is the HTTP-layer analogue of the
    store-level waiter in ``test_task_manager.py`` / ``test_telemetry_tracing.py``.

    A ``wait>0`` long-poll cannot stand in here: the events endpoint only
    enters its wait loop when the ``from_seq`` slice is empty, but these tests
    read from ``from_seq=0`` where the tape already carries ``task_started`` +
    ``progress`` events, so the handler returns immediately with whatever is on
    the tape — possibly still missing ``final``. Polling the tape is the fix.

    Each poll reads the *whole* tape by following ``has_more`` / ``next_from_seq``
    (the endpoint caps a single page at 1000 events), so the trailing ``final``
    envelope is found even on a tape longer than one page rather than paging off
    the end. A non-200 from the endpoint is a real failure (e.g. a 500 from a
    store-read regression) and is surfaced immediately rather than masked as a
    misleading "never reached final" timeout.
    """
    import asyncio as _asyncio

    deadline = _asyncio.get_event_loop().time() + timeout
    while _asyncio.get_event_loop().time() < deadline:
        events: list[dict[str, Any]] = []
        from_seq = 0
        while True:
            r = await client.get(
                f"/v1/tasks/{task_id}/events",
                params={"from_seq": from_seq, "limit": 1000, "wait": 0},
            )
            if r.status_code != 200:
                raise AssertionError(
                    f"GET /v1/tasks/{task_id}/events returned "
                    f"{r.status_code}: {r.text}"
                )
            page = r.json()
            events.extend(page["events"])
            if not page["has_more"]:
                break
            from_seq = page["next_from_seq"]
        if events and events[-1]["type"] == "final":
            return events
        await _asyncio.sleep(0.05)
    raise AssertionError(
        f"task {task_id} event tape never ended with a 'final' event"
    )
