"""``/v1/assets/{asset_id}`` — stream content-addressed asset bytes.

Assets are content-addressed by SHA-256, so the response is immutable;
the route emits ``ETag`` + long ``max-age`` + ``immutable`` so caches
can revalidate with a single ``If-None-Match``.

Failure surface is uniform 404 (unknown id, malformed id, file gone,
out-of-bounds ``stored_path``) so the route can't be used to probe
which ids exist.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse

from .. import api
from ..schemas import ASSET_URL_TEMPLATE
from .errors import NotFoundError
from .runtime import ServerRuntime, get_runtime

# The route this module owns; ``ASSET_URL_TEMPLATE`` is the DTO-side
# wire contract that :class:`schemas.PageAsset.url` is built against.
assert ASSET_URL_TEMPLATE == "/v1/assets/{asset_id}", (
    "ASSET_URL_TEMPLATE drifted from the route declared below"
)

_ASSET_ID_RE = re.compile(r"^[0-9a-f]{64}$")


def make_router(*, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/v1", dependencies=[Depends(auth_dep)])

    @router.get("/assets/{asset_id}")
    async def get_asset(request: Request, asset_id: str) -> FileResponse:
        rt: ServerRuntime = get_runtime(request.app)
        try:
            if not _ASSET_ID_RE.match(asset_id):
                raise api.AssetNotFound(asset_id)
            abs_path, record = await api.read_asset(rt.root, asset_id)
        except api.AssetNotFound as e:
            raise NotFoundError(
                f"asset not found: {asset_id!r}", code="asset_not_found"
            ) from e
        headers = {
            "ETag": f'"{record.asset_id}"',
            "Cache-Control": "public, max-age=31536000, immutable",
        }
        return FileResponse(
            path=abs_path, media_type=record.mime, headers=headers
        )

    return router


__all__ = ["make_router"]
