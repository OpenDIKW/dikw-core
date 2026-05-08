"""Pytest session bootstrap + cross-suite fixtures.

Windows ships ``asyncio.ProactorEventLoop`` as the default, which
psycopg's async client refuses to run under (raises ``InterfaceError``
at connect). Switch the policy to ``WindowsSelectorEventLoopPolicy`` so
the Postgres storage-contract tests run locally; Linux / macOS / CI
are unaffected.

The ``asgi_client``, ``client_transport``, and ``patch_transport_factory``
fixtures live here (rather than in ``tests/client/conftest.py``) so the
top-level ``tests/test_eval_cli.py`` — which is a CLI-routed test, not
a pure client unit test — can pull them in too.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from dikw_core.client.config import ClientConfig
from dikw_core.client.transport import Transport
from dikw_core.providers.codex_auth import dikw_auth_path
from dikw_core.server.app import build_app
from dikw_core.server.auth import AuthConfig
from dikw_core.server.runtime import ServerRuntime, build_runtime, teardown_runtime
from dikw_core.storage.base import Storage
from dikw_core.storage.sqlite import SQLiteStorage

from .fakes import init_test_wiki

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture(
    params=[
        pytest.param("sqlite", id="sqlite"),
        pytest.param(
            "postgres",
            id="postgres",
            marks=pytest.mark.skipif(
                not os.environ.get("DIKW_TEST_POSTGRES_DSN"),
                reason="Postgres adapter tests require DIKW_TEST_POSTGRES_DSN",
            ),
        ),
    ]
)
async def parametrized_storage(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[Storage]:
    """Yield a connected, migrated Storage instance per backend.

    Parameterised over SQLite + Postgres so adapter-spanning tests
    (storage contract, hybrid-search graph leg) auto-run on both.
    Postgres parametrisation skips when ``DIKW_TEST_POSTGRES_DSN`` is
    unset; CI provides one. Schema name is derived from ``tmp_path``
    so parallel runs don't collide.
    """
    backend = request.param
    schema: str | None = None
    if backend == "sqlite":
        s: Storage = SQLiteStorage(tmp_path / "test.sqlite", cjk_tokenizer="jieba")
    elif backend == "postgres":
        from dikw_core.storage.postgres import PostgresStorage

        dsn = os.environ["DIKW_TEST_POSTGRES_DSN"]
        schema = f"dikw_test_{abs(hash(str(tmp_path))) % 10_000_000:07d}"
        s = PostgresStorage(dsn, schema=schema, pool_size=2, cjk_tokenizer="jieba")
    else:
        raise RuntimeError(f"unreachable: adapter {backend}")

    await s.connect()
    await s.migrate()
    try:
        yield s
    finally:
        if backend == "postgres":
            from psycopg import AsyncConnection

            conn = await AsyncConnection.connect(os.environ["DIKW_TEST_POSTGRES_DSN"])
            try:
                async with conn.cursor() as cur:
                    await cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
                await conn.commit()
            finally:
                await conn.close()
        await s.close()


# --------------------------------------------------------------------------- #
# Codex auth fixtures — every codex test gets an isolated wiki base, with
# its own ``<base>/.dikw/auth.json``. ``CODEX_HOME`` is also redirected to a
# scratch dir so the lazy migration / ``dikw auth import`` paths can't
# accidentally read the developer's real ``~/.codex/auth.json``.
# --------------------------------------------------------------------------- #


@pytest.fixture()
def dikw_base(tmp_path: Path) -> Path:
    """Return a tmp path role-playing as a wiki base. ``.dikw/`` is
    pre-created so the auth store can be written without bootstrapping a
    full wiki."""
    base = tmp_path / "wiki"
    (base / ".dikw").mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture(autouse=True)
def _isolated_codex_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """Redirect ``$CODEX_HOME`` to an empty scratch dir for every test.

    Tests that exercise the codex CLI import path opt in by writing into
    the returned path; everything else stays oblivious. Without this the
    developer's real ``~/.codex/auth.json`` would influence test outcomes
    on Windows boxes that have it (lazy migration would silently kick in
    during tests that don't expect it).
    """
    home = tmp_path / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CODEX_HOME", str(home))
    return home


def make_dikw_auth_store(
    base: Path,
    *,
    access_token: str,
    refresh_token: str,
    last_refresh: str = "2026-05-06T03:14:22Z",
    auth_mode: str = "chatgpt",
    extra_providers: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Helper for tests: write a v1 dikw auth store with the given codex tokens.

    ``extra_providers`` (e.g. ``{"anthropic": {...}}``) is preserved
    verbatim so tests can verify the multi-provider read-modify-write
    contract.
    """
    auth_path = dikw_auth_path(base)
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    providers: dict[str, dict[str, Any]] = dict(extra_providers or {})
    providers["openai-codex"] = {
        "tokens": {"access_token": access_token, "refresh_token": refresh_token},
        "last_refresh": last_refresh,
        "auth_mode": auth_mode,
    }
    auth_path.write_text(
        json.dumps({"version": 1, "providers": providers}, indent=2),
        encoding="utf-8",
    )
    return auth_path


def make_codex_cli_auth_store(
    home: Path,
    *,
    access_token: str,
    refresh_token: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write codex CLI's flat ``auth.json`` schema for import-path tests."""
    home.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(extra or {})
    payload["tokens"] = {"access_token": access_token, "refresh_token": refresh_token}
    auth_path = home / "auth.json"
    auth_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return auth_path


@pytest.fixture()
def client_wiki(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki, description="client-test wiki")
    return wiki


@pytest.fixture()
async def asgi_client(
    client_wiki: Path,
) -> AsyncIterator[tuple[httpx.AsyncClient, ServerRuntime]]:
    """Build the FastAPI app + an ``httpx.AsyncClient`` bound to it.

    The runtime is built once per test from a fresh tmp wiki and torn
    down on exit. The fixture yields a (client, runtime) pair so tests
    can poke runtime state directly when needed (engine ingest before
    a CLI command, etc.).
    """
    auth = AuthConfig(host="127.0.0.1", token=None)
    rt = await build_runtime(root=client_wiki, auth=auth)

    async def _factory() -> ServerRuntime:
        return rt

    app = build_app(runtime_factory=_factory, auth=auth)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            async with app.router.lifespan_context(app):
                yield client, rt
    finally:
        await teardown_runtime(rt)


@pytest.fixture()
async def client_transport(
    asgi_client: tuple[httpx.AsyncClient, ServerRuntime],
) -> AsyncIterator[Transport]:
    httpx_client, _rt = asgi_client
    cfg = ClientConfig(server_url="http://test", token=None)
    t = Transport.from_config(cfg, client=httpx_client)
    try:
        yield t
    finally:
        # The asgi_client fixture owns the underlying httpx client's
        # lifecycle; closing it here would race the lifespan teardown.
        pass


@pytest.fixture()
def patch_transport_factory(
    asgi_client: tuple[httpx.AsyncClient, ServerRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], None]:
    """Returns a hook that monkeypatches ``Transport.from_config`` to
    always return a transport bound to the in-memory ASGI client.

    Tests using the Typer ``CliRunner`` call this once at the top of
    the test body. Without it, ``dikw client *`` commands try to open a
    real httpx client against ``http://127.0.0.1:8765`` and the test
    hangs.
    """

    httpx_client, _rt = asgi_client

    def _hook() -> None:
        def fake_from_config(
            _cfg: ClientConfig, *, client: httpx.AsyncClient | None = None
        ) -> Transport:
            del client
            return Transport(client=httpx_client, token=None)

        monkeypatch.setattr(
            "dikw_core.client.cli_app.Transport.from_config",
            staticmethod(fake_from_config),
        )

        async def _no_aexit(self: Transport, *_: Any) -> None:
            del self
            return None

        monkeypatch.setattr(Transport, "__aexit__", _no_aexit)

    return _hook
