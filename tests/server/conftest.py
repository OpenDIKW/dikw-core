"""Shared fixtures for server-side tests.

Two flavours of test infrastructure:

  * ``server_client`` — full FastAPI app wired to a real (test) wiki via
    ``build_app``. The server runs in-process via ``ASGITransport`` so
    no socket is bound. Suits routes_sync + routes_tasks integration
    tests where the engine should actually exercise.

  * ``manager_only`` — naked ``TaskManager`` + ``ProgressBus`` +
    ``SqliteTaskStore`` triplet for tests that only care about the
    task subsystem semantics. Cheap, no FastAPI overhead.

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

import httpx
import pytest
from fastapi import FastAPI

from dikw_core import api as api_module
from dikw_core.server.app import build_app
from dikw_core.server.auth import AuthConfig
from dikw_core.server.runtime import ServerRuntime, build_runtime, teardown_runtime
from dikw_core.server.tasks import ProgressBus, SqliteTaskStore, TaskManager

from ..fakes import FakeEmbeddings, init_test_wiki

FIXTURES_NOTES = Path(__file__).parent.parent / "fixtures" / "notes"


@pytest.fixture()
def wiki_root(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki, description="server-test wiki")
    return wiki


@pytest.fixture()
async def runtime(wiki_root: Path) -> AsyncIterator[ServerRuntime]:
    """Live runtime backed by a fresh tmp wiki, no auth."""
    auth = AuthConfig(host="127.0.0.1", token=None)
    rt = await build_runtime(root=wiki_root, auth=auth)
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
    wiki_root: Path,
) -> AsyncIterator[tuple[httpx.AsyncClient, str]]:
    """Token-required variant. The host stays loopback (127.0.0.1) but a
    token is set, which still triggers token-required mode per
    ``AuthConfig.required``."""
    auth = AuthConfig(host="127.0.0.1", token="s3cret")
    rt = await build_runtime(root=wiki_root, auth=auth)
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
    wiki_root: Path,
) -> Path:
    """Wiki with the standard ``tests/fixtures/notes`` corpus ingested
    via ``api.ingest`` + ``FakeEmbeddings``. Used by query / retrieve /
    health route tests that need both documents and embeddings.
    """
    dest = wiki_root / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES_NOTES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    await api_module.ingest(wiki_root, embedder=FakeEmbeddings())
    _ = server_client  # ensure runtime lifespan is up before we ingest
    return wiki_root


@pytest.fixture()
async def manager_only(
    tmp_path: Path,
) -> AsyncIterator[tuple[TaskManager, SqliteTaskStore, ProgressBus]]:
    """``TaskManager`` + ``ProgressBus`` + fresh SQLite store, no FastAPI."""
    store = SqliteTaskStore(path=tmp_path / "tasks.db")
    await store.init()
    bus = ProgressBus()
    manager = TaskManager(store=store, bus=bus)
    try:
        yield manager, store, bus
    finally:
        await manager.shutdown()
        await store.close()
