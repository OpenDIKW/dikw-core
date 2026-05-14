"""``GET /v1/base/graph`` — full base graph for web + agent consumers.

Issue #89: ``dikw-web``'s Knowledge Graph used to loop
``GET /v1/base/pages/{path}`` for every page and re-parse wikilinks in
the browser. This endpoint returns the whole graph in one read-only
JSON response — nodes (every active doc), edges (every resolvable
wikilink / cross-page markdown link), and ``unresolved`` (broken
wikilinks, surfaced without ghost nodes per issue v1 stance).

The handler is a thin wrapper over ``api.list_graph``; the engine owns
all the K-layer link semantics (exact title → fuzzy normalize →
collision-refuse) so each client doesn't reinvent them.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from .. import api
from ..schemas import GraphResult
from .runtime import ServerRuntime, get_runtime


def make_router(*, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/v1", dependencies=[Depends(auth_dep)])

    @router.get("/base/graph", response_model=GraphResult)
    async def get_graph(
        request: Request,
        active: bool | None = Query(default=True),
    ) -> GraphResult:
        rt: ServerRuntime = get_runtime(request.app)
        return await api.list_graph(rt.root, active=active)

    return router


__all__ = ["make_router"]
