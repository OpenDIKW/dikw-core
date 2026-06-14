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

from ..config import CONFIG_FILENAME, load_config
from ..logging import init_logging
from ..telemetry import telemetry_should_activate
from .auth import AuthConfig, load_auth_config, make_dependency
from .errors import install_handlers
from .routes_assets import make_router as make_assets_router
from .routes_graph import make_router as make_graph_router
from .routes_import import make_router as make_import_router
from .routes_pages import make_router as make_pages_router
from .routes_retrieve import make_router as make_retrieve_router
from .routes_sync import make_router as make_sync_router
from .routes_tasks import make_router as make_tasks_router
from .runtime import ServerRuntime, build_runtime, lifespan


def build_app(
    *,
    runtime_factory: Callable[[], Awaitable[ServerRuntime]],
    auth: AuthConfig,
    instrument_telemetry: bool = False,
) -> FastAPI:
    """Assemble the FastAPI app around an already-resolved auth config.

    The runtime is built lazily inside the lifespan hook so a uvicorn
    reload picks up cfg changes without manual orchestration; tests
    wanting an in-memory engine pass a factory that returns a
    pre-stubbed ``ServerRuntime``.

    ``instrument_telemetry`` wires OTel HTTP server-span instrumentation.
    ``build_app_from_disk`` resolves it from the base's ``telemetry:`` config;
    it defaults off so test apps (and a disabled server) carry no middleware.
    """
    init_logging()
    app = FastAPI(
        title="dikw-core",
        version="0.1",  # bump when the wire contract changes
        lifespan=lifespan,
    )
    app.state.runtime_factory = runtime_factory

    # OTel HTTP server-span instrumentation. Lives here (server code may import
    # FastAPI) not in ``telemetry.py`` (engine root, must not depend on the web
    # framework). Gated on the *resolved telemetry setting*, not just package
    # availability: a disabled server must carry no middleware — otherwise it
    # pays per-request cost and could emit spans to a foreign global provider
    # (e.g. external auto-instrumentation) despite telemetry being configured
    # off. Resolved at build time because the middleware can't be added later in
    # the lifespan (the app's middleware stack is already built by then). The
    # inner guard handles a partial install (api present, instrumentation-fastapi
    # absent) — still serve, with HTTP spans degraded off, mirroring
    # ``configure_telemetry``.
    if instrument_telemetry:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        except ImportError:
            pass  # partial OTel install; install the [otel] extra for HTTP spans
        else:
            FastAPIInstrumentor.instrument_app(app)

    install_handlers(app)
    auth_dep = make_dependency(auth)
    app.include_router(make_sync_router(auth_dep=auth_dep))
    app.include_router(make_tasks_router(auth_dep=auth_dep))
    app.include_router(make_import_router(auth_dep=auth_dep))
    app.include_router(make_retrieve_router(auth_dep=auth_dep))
    app.include_router(make_pages_router(auth_dep=auth_dep))
    app.include_router(make_assets_router(auth_dep=auth_dep))
    app.include_router(make_graph_router(auth_dep=auth_dep))
    return app


def build_app_from_disk(
    *,
    base_root: Path,
    host: str,
    token_override: str | None = None,
) -> FastAPI:
    """Convenience: resolve auth from env, build a runtime that loads the
    wiki at ``base_root``, return the wired FastAPI app. Used by
    ``dikw serve``."""
    auth = load_auth_config(host=host, token_override=token_override)

    async def _factory() -> ServerRuntime:
        return await build_runtime(root=base_root, auth=auth)

    # Resolve the telemetry-instrumentation decision now (build time) from the
    # base config — FastAPI middleware can't be added once the app's stack is
    # built (i.e. not in the lifespan).
    try:
        cfg = load_config(base_root / CONFIG_FILENAME)
        instrument_telemetry = telemetry_should_activate(cfg.telemetry.enabled)
    except Exception:
        # Best-effort only: a missing/invalid config (or version mismatch) is
        # surfaced with the proper error by build_runtime when the lifespan
        # starts. Here we just need the instrument flag, so any failure simply
        # leaves it off rather than crashing app construction.
        instrument_telemetry = False

    return build_app(
        runtime_factory=_factory,
        auth=auth,
        instrument_telemetry=instrument_telemetry,
    )


__all__ = ["build_app", "build_app_from_disk"]
