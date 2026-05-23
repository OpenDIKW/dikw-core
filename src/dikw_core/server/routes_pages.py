"""``/v1/base/pages*`` — cross-layer page read with chunk anchors.

Companion to ``POST /v1/retrieve``: an agent that hits a chunk via
retrieve calls ``GET /v1/base/pages/{path}`` to read the full page body
plus per-chunk ``anchors[]`` so it can join hit chunks back onto the
unchunked body for prompt assembly or rendering.
``GET /v1/base/pages/{path}/links`` is the K-layer graph hop: returns
outgoing / incoming edges so an agent can walk neighbours without
re-parsing wiki bodies.
``GET /v1/base/pages/{path}/provenance`` is the K↔D attribution edge:
returns forward sources (a K-page's ``derived_from``) plus reverse
derived pages (a D-source's ``derived_pages``). Distinct from
``/links`` — different table, different semantics — see
``docs/adr/0001-provenance-as-separate-edge.md``.

Path safety is index-driven — paths absent from the ``documents`` table
return 404, which transparently covers ``..`` traversal and unindexed
files (``dikw.yml``, etc.) without a separate sandbox check.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request

from .. import api
from ..schemas import (
    DocumentRecord,
    Layer,
    LinkDirection,
    PageLinksResult,
    PageProvenanceResult,
    PageReadResult,
    ProvenanceDirection,
)
from .errors import NotFoundError
from .runtime import ServerRuntime, get_runtime


def make_router(*, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/v1", dependencies=[Depends(auth_dep)])

    @router.get("/base/pages", response_model=list[DocumentRecord])
    async def list_pages(
        request: Request,
        layer: Layer | None = Query(default=None),
        active: bool | None = Query(default=True),
        since_ts: float | None = Query(default=None),
    ) -> list[DocumentRecord]:
        rt: ServerRuntime = get_runtime(request.app)
        return await api.list_pages(
            rt.root, layer=layer, active=active, since_ts=since_ts
        )

    # Order matters: the ``/links`` and ``/provenance`` sub-routes must
    # be declared BEFORE the catch-all ``{path:path}`` get_page handler
    # so FastAPI doesn't greedily match e.g. ``wiki/foo.md/links``
    # against ``{path:path}``.
    @router.get(
        "/base/pages/{path:path}/links", response_model=PageLinksResult
    )
    async def get_page_links(
        request: Request,
        path: str,
        direction: Annotated[LinkDirection, Query()] = "both",
        limit: Annotated[int | None, Query(ge=0)] = None,
    ) -> PageLinksResult:
        rt: ServerRuntime = get_runtime(request.app)
        try:
            return await api.list_links(
                rt.root, path, direction=direction, limit=limit
            )
        except api.PageNotFound as e:
            raise NotFoundError(
                f"page not found: {path!r}", code="page_not_found"
            ) from e

    @router.get(
        "/base/pages/{path:path}/provenance",
        response_model=PageProvenanceResult,
    )
    async def get_page_provenance(
        request: Request,
        path: str,
        direction: Annotated[ProvenanceDirection, Query()] = "both",
        limit: Annotated[int | None, Query(ge=0)] = None,
    ) -> PageProvenanceResult:
        rt: ServerRuntime = get_runtime(request.app)
        try:
            return await api.read_provenance(
                rt.root, path, direction=direction, limit=limit
            )
        except api.PageNotFound as e:
            raise NotFoundError(
                f"page not found: {path!r}", code="page_not_found"
            ) from e

    @router.get("/base/pages/{path:path}", response_model=PageReadResult)
    async def get_page(request: Request, path: str) -> PageReadResult:
        rt: ServerRuntime = get_runtime(request.app)
        try:
            return await api.read_page(rt.root, path)
        except api.PageNotFound as e:
            raise NotFoundError(
                f"page not found: {path!r}", code="page_not_found"
            ) from e

    return router


__all__ = ["make_router"]
