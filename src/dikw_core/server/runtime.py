"""Per-process engine handle: cfg / wiki root / storage / task subsystem.

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
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI

from ..config import CONFIG_FILENAME, DikwConfig, load_config
from ..storage import Storage, build_storage
from .auth import AuthConfig
from .tasks import (
    ProgressBus,
    SqliteTaskStore,
    TaskManager,
    TaskStore,
    build_task_store,
)

logger = logging.getLogger(__name__)


_WIKI_ID_FILENAME = "wiki_id"


def _wiki_scope_id(root: Path) -> str:
    """Stable identifier for the wiki this server is bound to.

    Used by the task store to scope every read + write so a shared
    Postgres task DB does not leak rows across wikis, AND so multiple
    replicas of the same wiki share state (a follow/cancel routed to
    replica B must find the task submitted via replica A).

    Resolution order:
      1. ``DIKW_WIKI_INSTANCE_ID`` env var — operator override for
         exotic deployments (e.g. multiple wikis intentionally pooled
         under one task ID).
      2. ``<root>/.dikw/wiki_id`` — a UUID4 generated on first run and
         persisted to the wiki tree. Survives the wiki being mounted
         at different paths in different containers, which a
         path-hash scheme cannot.
      3. Generate a fresh UUID4, write it to (2), return it.

    A path-based hash was the previous scheme but broke whenever two
    replicas mounted the same wiki under different filesystem paths —
    every cross-replica read filtered under a different scope and the
    public task APIs silently stopped working.
    """
    env_override = os.getenv("DIKW_WIKI_INSTANCE_ID", "").strip()
    if env_override:
        return env_override

    dikw_dir = root / ".dikw"
    dikw_dir.mkdir(parents=True, exist_ok=True)
    id_path = dikw_dir / _WIKI_ID_FILENAME
    try:
        existing = id_path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except FileNotFoundError:
        pass
    new_id = uuid.uuid4().hex
    # ``write_text`` is sufficient — concurrent first-runs of the same
    # wiki against the same volume would race here, but the file is
    # tiny + atomic at the FS layer (POSIX) and read-on-startup means
    # a brief mismatch only loses already-orphaned tasks. Operators
    # that fan out from cold should pre-seed the file.
    id_path.write_text(new_id + "\n", encoding="utf-8")
    return new_id


@dataclass
class ServerRuntime:
    """All server-wide state. Held under ``app.state.runtime``."""

    cfg: DikwConfig
    root: Path
    storage: Storage
    task_store: TaskStore
    bus: ProgressBus
    manager: TaskManager
    auth: AuthConfig
    # Serializes wiki-mutating ops (currently ingest) so two concurrent
    # tasks can't interleave their staging-commit + on-disk writes and
    # leave the sources/ tree as a mix of both. Held for the entire ingest
    # runner — concurrent ingests on the same wiki are a degenerate case.
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
    cfg = load_config(cfg_path)

    # Crash-recovery: a server killed mid-import leaves a staging dir
    # behind. The import endpoint always rmtrees its own staging when
    # the request returns, so any leftover here means a process died
    # before that block ran. Wipe wholesale — staging is purely
    # transient.
    _cleanup_orphan_staging(root)

    storage = build_storage(
        cfg.storage,
        root=root,
        cjk_tokenizer=cfg.retrieval.cjk_tokenizer,
    )
    await storage.connect()
    await storage.migrate()

    task_store = build_task_store(
        cfg, root=root, instance_id=_wiki_scope_id(root)
    )
    await task_store.init()

    bus = ProgressBus()
    manager = TaskManager(store=task_store, bus=bus)
    # Auto-cleanup is safe only when this process owns the task store
    # exclusively — i.e. the per-wiki sqlite file. With a shared Postgres
    # task DB another live replica of *the same wiki* may have in-flight
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
    if isinstance(task_store, SqliteTaskStore) or force_reap:
        await manager.restart_cleanup()
    else:
        logger.info(
            "skipping restart_cleanup for shared task store (%s); "
            "stuck rows from a previous incarnation must be reaped "
            "out-of-band, or set DIKW_TASK_REAP_ON_START=1 if this is "
            "the only server bound to the task DSN",
            type(task_store).__name__,
        )

    return ServerRuntime(
        cfg=cfg,
        root=root,
        storage=storage,
        task_store=task_store,
        bus=bus,
        manager=manager,
        auth=auth,
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
    logger.info(
        "dikw server ready  base=%s storage=%s auth=%s",
        rt.root,
        rt.cfg.storage.backend,
        "token" if rt.auth.required else "off",
    )
    try:
        yield
    finally:
        await teardown_runtime(rt)


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
