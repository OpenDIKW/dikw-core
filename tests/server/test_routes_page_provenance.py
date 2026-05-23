"""HTTP-level tests for ``GET /v1/base/pages/{path}/provenance``.

Companion endpoint to ``GET /v1/base/pages/{path}/links``: same shape,
different edge. Links is the body-derived ``[[wikilink]]`` graph;
provenance is the frontmatter-derived ``sources:`` attribution
(``derived_from`` forward, ``derived_pages`` reverse). See
``docs/adr/0001-provenance-as-separate-edge.md`` for why the two live
in different tables.

These tests lock the wire shape, the ``direction`` / ``limit`` query
params, the route-ordering invariant (must beat the ``{path:path}``
catch-all), the dangling-source surfacing contract
(``resolved=False`` instead of silent drop), and the 404-on-unknown-
path policy.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from dikw_core.api import _doc_id_for, _with_storage
from dikw_core.schemas import DocumentRecord, Layer


def _doc(
    path: str,
    layer: Layer = Layer.WIKI,
    *,
    title: str | None = None,
) -> DocumentRecord:
    return DocumentRecord(
        doc_id=_doc_id_for(layer, path),
        path=path,
        title=title or path.rsplit("/", 1)[-1],
        hash="0" * 64,
        mtime=0.0,
        layer=layer,
        active=True,
    )


async def _seed_basic(root: Path) -> tuple[str, str, str]:
    """Seed: one source ``src`` claimed by two K-pages (``a``, ``b``);
    ``a`` additionally claims a ghost (unindexed) source so the
    dangling case is covered too. Returns ``(src_path, a_path,
    b_path)`` for assertions."""
    src_path = "sources/src.md"
    ghost_path = "sources/ghost.md"
    a_path = "wiki/a.md"
    b_path = "wiki/b.md"
    cfg, _root, storage = await _with_storage(root)
    del cfg
    try:
        await storage.upsert_document(_doc(src_path, layer=Layer.SOURCE))
        for p in (a_path, b_path):
            await storage.upsert_document(_doc(p, layer=Layer.WIKI))
        await storage.replace_provenance_from(
            _doc_id_for(Layer.WIKI, a_path), [src_path, ghost_path]
        )
        await storage.replace_provenance_from(
            _doc_id_for(Layer.WIKI, b_path), [src_path]
        )
    finally:
        await storage.close()
    return src_path, a_path, b_path


@pytest.mark.asyncio
async def test_get_provenance_returns_page_provenance_result(
    server_client: httpx.AsyncClient, wiki_root: Path
) -> None:
    """Wire shape: K-page → forward sources populated, reverse empty;
    response carries ``path`` + ``derived_from`` + ``derived_pages``."""
    _src, a_path, _b = await _seed_basic(wiki_root)
    resp = await server_client.get(f"/v1/base/pages/{a_path}/provenance")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["path"] == a_path
    assert body["derived_pages"] == []
    by_path = {s["source_path"]: s for s in body["derived_from"]}
    assert by_path["sources/src.md"]["resolved"] is True
    assert by_path["sources/src.md"]["doc_id"] == _doc_id_for(
        Layer.SOURCE, "sources/src.md"
    )
    # Dangling surfaced faithfully (NOT silently dropped).
    assert by_path["sources/ghost.md"]["resolved"] is False
    assert by_path["sources/ghost.md"]["doc_id"] is None


@pytest.mark.asyncio
async def test_get_provenance_reverse_for_source_returns_derived_pages(
    server_client: httpx.AsyncClient, wiki_root: Path
) -> None:
    """A SOURCE-layer path returns the K-pages whose frontmatter
    ``sources:`` claims it — the answer to the "which pages reference
    this source?" question this feature exists for."""
    src_path, a_path, b_path = await _seed_basic(wiki_root)
    resp = await server_client.get(f"/v1/base/pages/{src_path}/provenance")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["path"] == src_path
    assert body["derived_from"] == []  # SOURCE rows have no forward edges
    assert sorted(dp["path"] for dp in body["derived_pages"]) == sorted(
        [a_path, b_path]
    )


@pytest.mark.asyncio
async def test_get_provenance_direction_filter(
    server_client: httpx.AsyncClient, wiki_root: Path
) -> None:
    """``direction=in`` vs ``out`` populates only the matching list.
    Same semantics as ``/links?direction``."""
    src_path, a_path, _b = await _seed_basic(wiki_root)

    out_only = await server_client.get(
        f"/v1/base/pages/{a_path}/provenance", params={"direction": "out"}
    )
    assert out_only.status_code == 200
    body = out_only.json()
    assert body["derived_from"] and body["derived_pages"] == []

    in_only = await server_client.get(
        f"/v1/base/pages/{src_path}/provenance", params={"direction": "in"}
    )
    assert in_only.status_code == 200
    body = in_only.json()
    assert body["derived_pages"] and body["derived_from"] == []


@pytest.mark.asyncio
async def test_get_provenance_unknown_path_404(
    server_client: httpx.AsyncClient, wiki_root: Path
) -> None:
    resp = await server_client.get("/v1/base/pages/wiki/missing.md/provenance")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "page_not_found"


@pytest.mark.asyncio
async def test_get_provenance_rejects_bad_direction(
    server_client: httpx.AsyncClient, wiki_root: Path
) -> None:
    _src, a_path, _b = await _seed_basic(wiki_root)
    resp = await server_client.get(
        f"/v1/base/pages/{a_path}/provenance",
        params={"direction": "sideways"},
    )
    assert resp.status_code in (400, 422)


@pytest.mark.asyncio
async def test_get_provenance_rejects_negative_limit(
    server_client: httpx.AsyncClient, wiki_root: Path
) -> None:
    """``limit < 0`` is rejected by FastAPI's ``Query(ge=0)`` before
    the handler runs — matches the ``/links`` route's clamp policy."""
    _src, a_path, _b = await _seed_basic(wiki_root)
    resp = await server_client.get(
        f"/v1/base/pages/{a_path}/provenance", params={"limit": -1}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_provenance_limit_caps_each_side(
    server_client: httpx.AsyncClient, wiki_root: Path
) -> None:
    """``limit`` caps ``derived_from`` (and would cap ``derived_pages``
    on a SOURCE-layer query — symmetric)."""
    page_path = "wiki/hub.md"
    src_paths = [f"sources/s{i}.md" for i in range(5)]
    cfg, _root, storage = await _with_storage(wiki_root)
    del cfg
    try:
        await storage.upsert_document(_doc(page_path, layer=Layer.WIKI))
        for sp in src_paths:
            await storage.upsert_document(_doc(sp, layer=Layer.SOURCE))
        await storage.replace_provenance_from(
            _doc_id_for(Layer.WIKI, page_path), src_paths
        )
    finally:
        await storage.close()

    resp = await server_client.get(
        f"/v1/base/pages/{page_path}/provenance", params={"limit": 2}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["derived_from"]) == 2


@pytest.mark.asyncio
async def test_get_page_route_still_works_after_provenance_route_added(
    server_client: httpx.AsyncClient, wiki_root: Path
) -> None:
    """Route-ordering invariant: ``/provenance`` is declared BEFORE the
    catch-all ``{path:path}`` get_page handler, but a plain page-read
    against ``/v1/base/pages/wiki/foo.md`` (no /provenance suffix) must
    still resolve to ``get_page`` (404 here is fine — what we're guarding
    against is the route silently re-routing into ``get_page_provenance``
    and returning the wrong shape)."""
    resp = await server_client.get("/v1/base/pages/wiki/does-not-exist.md")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "page_not_found"
