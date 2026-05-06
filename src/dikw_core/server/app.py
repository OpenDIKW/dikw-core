"""FastAPI app factory.

Single entry point — ``build_app`` — composes the runtime factory, auth
dependency, sync routes, task routes, and error handlers. The CLI's
``dikw serve`` command instantiates this and hands it to uvicorn; tests
build it with an injected runtime factory so they don't need a real
storage backend.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI

from .auth import AuthConfig, load_auth_config, make_dependency
from .errors import install_handlers
from .routes_pages import make_router as make_pages_router
from .routes_query import make_router as make_query_router
from .routes_retrieve import make_router as make_retrieve_router
from .routes_sync import make_router as make_sync_router
from .routes_tasks import make_router as make_tasks_router
from .routes_upload import make_router as make_upload_router
from .runtime import ServerRuntime, build_runtime, lifespan


def build_app(
    *,
    runtime_factory: Callable[[], Awaitable[ServerRuntime]],
    auth: AuthConfig,
) -> FastAPI:
    """Assemble the FastAPI app around an already-resolved auth config.

    The runtime is built lazily inside the lifespan hook so a uvicorn
    reload picks up cfg changes without manual orchestration; tests
    wanting an in-memory engine pass a factory that returns a
    pre-stubbed ``ServerRuntime``.
    """
    app = FastAPI(
        title="dikw-core",
        version="0.1",  # bump when the wire contract changes
        lifespan=lifespan,
    )
    app.state.runtime_factory = runtime_factory

    install_handlers(app)
    auth_dep = make_dependency(auth)
    app.include_router(make_sync_router(auth_dep=auth_dep))
    app.include_router(make_tasks_router(auth_dep=auth_dep))
    app.include_router(make_upload_router(auth_dep=auth_dep))
    app.include_router(make_query_router(auth_dep=auth_dep))
    app.include_router(make_retrieve_router(auth_dep=auth_dep))
    app.include_router(make_pages_router(auth_dep=auth_dep))
    return app


def build_app_from_disk(
    *,
    wiki_root: Path,
    host: str,
    token_override: str | None = None,
) -> FastAPI:
    """Convenience: resolve auth from env, build a runtime that loads the
    wiki at ``wiki_root``, return the wired FastAPI app. Used by
    ``dikw serve``."""
    auth = load_auth_config(host=host, token_override=token_override)

    async def _factory() -> ServerRuntime:
        return await build_runtime(root=wiki_root, auth=auth)

    return build_app(runtime_factory=_factory, auth=auth)


__all__ = ["build_app", "build_app_from_disk"]
