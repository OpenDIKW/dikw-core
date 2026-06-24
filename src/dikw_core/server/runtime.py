"""Per-process engine handle: cfg / base root / storage / task subsystem.

One ``ServerRuntime`` is built at server startup and torn down at shutdown;
route handlers reach it via FastAPI dependencies. We keep this small —
just the long-lived state — because the engine itself is largely stateless
(LLM / embedding providers are built per-request as today).

Lifecycle:
  startup  →  load cfg, build storage, connect, migrate,
              build task store + manager, restart_cleanup
  shutdown →  manager.shutdown, store.close, task_store.close
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI

from .. import __version__
from ..api import _assert_base_upgraded
from ..config import CONFIG_FILENAME, DikwConfig, load_config
from ..storage import Storage, build_storage
from ..telemetry import configure_telemetry, shutdown_telemetry
from .auth import AuthConfig
from .tasks import (
    SqliteTaskStore,
    TaskManager,
    TaskStore,
    build_task_store,
)

logger = logging.getLogger(__name__)


_BASE_ID_FILENAME = "base_id"


def _base_scope_id(root: Path) -> str:
    """Stable identifier for the base this server is bound to.

    Used by the task store to scope every read + write so a shared
    Postgres task DB does not leak rows across bases, AND so multiple
    replicas of the same base share state (a follow/cancel routed to
    replica B must find the task submitted via replica A).

    Resolution order:
      1. ``DIKW_BASE_INSTANCE_ID`` env var — operator override for
         exotic deployments (e.g. multiple bases intentionally pooled
         under one task ID).
      2. ``<root>/.dikw/base_id`` — a UUID4 generated on first run and
         persisted to the base tree. Survives the base being mounted
         at different paths in different containers, which a
         path-hash scheme cannot.
      3. Generate a fresh UUID4, write it to (2), return it.

    A path-based hash was the previous scheme but broke whenever two
    replicas mounted the same base under different filesystem paths —
    every cross-replica read filtered under a different scope and the
    public task APIs silently stopped working.
    """
    env_override = os.getenv("DIKW_BASE_INSTANCE_ID", "").strip()
    if env_override:
        return env_override

    dikw_dir = root / ".dikw"
    dikw_dir.mkdir(parents=True, exist_ok=True)
    id_path = dikw_dir / _BASE_ID_FILENAME
    # Fast path: an id already persisted by an earlier run.
    try:
        existing = id_path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except FileNotFoundError:
        pass
    # First run for this base. Create the id file *exclusively* so two
    # processes cold-starting the same base converge on one winner instead
    # of each minting a different UUID (which would silently split the
    # task-store scope — cross-replica follow/cancel would stop finding each
    # other's tasks). ``open(..., "x")`` lets exactly one creator win.
    try:
        with open(id_path, "x", encoding="utf-8") as fh:
            new_id = uuid.uuid4().hex
            fh.write(new_id + "\n")
            return new_id
    except FileExistsError:
        # Lost the create race. The winner writes its id immediately after
        # creating the file, so poll briefly until the content lands rather
        # than racing the empty-file window and returning a divergent id.
        for _ in range(200):
            existing = id_path.read_text(encoding="utf-8").strip()
            if existing:
                return existing
            time.sleep(0.005)
    raise RuntimeError(
        f"base id file {id_path} exists but never received an id; remove it "
        "or pin DIKW_BASE_INSTANCE_ID"
    )


@dataclass
class ServerRuntime:
    """All server-wide state. Held under ``app.state.runtime``."""

    cfg: DikwConfig
    root: Path
    storage: Storage
    task_store: TaskStore
    manager: TaskManager
    auth: AuthConfig
    # Stable id for the base this server is bound to (``_base_scope_id``);
    # stamped on task spans as ``dikw.base_id``. Resolved once at build time
    # (it does file I/O) and reused for the task-store scope.
    base_id: str = ""
    # Single base-level write mutex. Every base-mutating task acquires it so
    # two writers can't interleave their (un-enclosed-by-a-transaction)
    # storage-row + on-disk writes on the same base: ingest, import, wisdom
    # write, delete, synth, and lint apply. Named ``ingest_lock`` for
    # history; it now guards the whole K/W/D write surface. Read paths
    # (retrieve, list, read_page) do NOT take it — the SQLite adapter
    # serializes its own connection internally and reads tolerate the
    # eventual-consistency window. Held for each writer's storage-mutating
    # span; concurrent writes to one base are a degenerate case anyway.
    ingest_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def build_runtime(
    *, root: Path, auth: AuthConfig
) -> ServerRuntime:
    """Resolve cfg + open every long-lived handle. Caller owns teardown."""
    root = root.resolve()
    cfg_path = root / CONFIG_FILENAME
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"no {CONFIG_FILENAME} at {root} — initialise the base first "
            "or point `dikw serve --base` at an existing dikw base"
        )
    _assert_base_upgraded(root)
    cfg = load_config(cfg_path)

    storage = build_storage(
        cfg.storage,
        root=root,
        cjk_tokenizer=cfg.retrieval.cjk_tokenizer,
    )
    await storage.connect()
    await storage.migrate()

    base_id = _base_scope_id(root)
    task_store = build_task_store(cfg, root=root, instance_id=base_id)
    await task_store.init()

    manager = TaskManager(store=task_store)
    # Auto-cleanup is safe only when this process owns the task store
    # exclusively — i.e. the per-base sqlite file. With a shared Postgres
    # task DB another live replica of *the same base* may have in-flight
    # tasks that belong to its own asyncio loop; cancelling them here
    # would mark a healthy peer's work as failed{server_restart}.
    #
    # Single-server Postgres deployments (the common case) lose orphan
    # cleanup unless they opt in via ``DIKW_TASK_REAP_ON_START=1``. Set
    # that env var only when you're SURE no other ``dikw serve`` instance
    # shares the task DSN — multi-replica deployments must leave it unset.
    force_reap = os.getenv("DIKW_TASK_REAP_ON_START", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    owns_task_store = isinstance(task_store, SqliteTaskStore) or force_reap
    if owns_task_store:
        await manager.restart_cleanup()
        # Same gate as restart_cleanup: a server killed mid-import leaves a
        # staging dir behind (the import endpoint always rmtrees its own
        # staging on return, so a leftover means the process died first).
        # Wipe it only when this process owns the base exclusively — a
        # replica sharing a Postgres task store must not delete a live
        # peer's in-flight staging.
        _cleanup_orphan_staging(root)
    else:
        logger.info(
            "skipping restart_cleanup + orphan-staging cleanup for shared "
            "task store (%s); stuck rows / orphan staging from a previous "
            "incarnation must be reaped out-of-band, or set "
            "DIKW_TASK_REAP_ON_START=1 if this is the only server bound to "
            "the task DSN",
            type(task_store).__name__,
        )

    return ServerRuntime(
        cfg=cfg,
        root=root,
        storage=storage,
        task_store=task_store,
        manager=manager,
        auth=auth,
        base_id=base_id,
    )


async def teardown_runtime(rt: ServerRuntime) -> None:
    await rt.manager.shutdown()
    await rt.storage.close()
    await rt.task_store.close()


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """FastAPI ``lifespan`` hook. Pulls a pre-prepared ``ServerRuntime`` off
    ``app.state.runtime_factory`` (a callable set by the app builder so
    tests can inject without spinning a real engine)."""
    factory = getattr(app.state, "runtime_factory", None)
    if factory is None:
        raise RuntimeError(
            "app.state.runtime_factory is unset; build the app via "
            "server.app.build_app(...)"
        )
    rt: ServerRuntime = await factory()
    app.state.runtime = rt
    # SDK bootstrap is an entry-point concern (cf. ``init_logging``); do it here
    # — after cfg load so the ``telemetry:`` section is available — rather than
    # in the engine. No-op unless ``telemetry.enabled`` and the ``[otel]`` extra
    # is installed. FastAPI auto-instrumentation (wired in ``build_app``) then
    # starts producing real server spans for subsequent requests.
    telemetry_on = configure_telemetry(
        enabled=rt.cfg.telemetry.enabled,
        endpoint=rt.cfg.telemetry.endpoint,
        service_name=rt.cfg.telemetry.service_name,
        sample_ratio=rt.cfg.telemetry.sample_ratio,
        version=__version__,
    )
    logger.info(
        "dikw server ready  base=%s storage=%s auth=%s telemetry=%s",
        rt.root,
        rt.cfg.storage.backend,
        "token" if rt.auth.required else "off",
        "on" if telemetry_on else "off",
    )
    try:
        yield
    finally:
        # shutdown_telemetry must run even if teardown raises — otherwise a
        # storage/manager close error would leak the BatchSpanProcessor export
        # thread and strand the _configured latch for any in-process restart.
        try:
            await teardown_runtime(rt)
        finally:
            shutdown_telemetry()


def get_runtime(app: FastAPI) -> ServerRuntime:
    """Access the runtime from any route handler via ``request.app``."""
    rt = getattr(app.state, "runtime", None)
    if rt is None:
        raise RuntimeError("server runtime is not initialised")
    return rt  # type: ignore[no-any-return]


def _cleanup_orphan_staging(root: Path) -> None:
    """Wipe ``<root>/.dikw/staging/`` on startup.

    Successful + failed imports always rmtree their own per-id
    subdirectory in a ``finally`` block; anything left here means a
    process died mid-import (SIGKILL, OOM). The contents are pure
    transient state — no commit happened, no client is waiting on
    them — so wipe wholesale rather than per-id.

    Also rmtrees the legacy ``.dikw/upload-staging/`` directory if it
    survives from a pre-rename install — it holds the same kind of
    transient bytes and the new server would otherwise leave it
    lingering forever.

    String-literal path (rather than ``from .routes_import import
    STAGING_DIRNAME``) avoids a circular import — routes_import
    imports ``ServerRuntime`` from this module."""
    legacy = root / ".dikw" / "upload-staging"
    staging = root / ".dikw" / "staging"
    try:
        shutil.rmtree(legacy, ignore_errors=True)
        if staging.exists():
            for entry in staging.iterdir():
                shutil.rmtree(entry, ignore_errors=True)
    except OSError as e:
        logger.warning("orphan staging cleanup skipped: %s", e)


__all__ = [
    "ServerRuntime",
    "build_runtime",
    "get_runtime",
    "lifespan",
    "teardown_runtime",
]
