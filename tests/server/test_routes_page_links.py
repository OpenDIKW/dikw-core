"""HTTP-level tests for ``GET /v1/base/pages/{path}/links``.

Companion endpoint to ``GET /v1/base/pages/{path}``: lets an agent walk
the K-layer link graph without re-parsing wiki bodies. Tests lock the
wire shape, the ``direction`` filter, the ``limit`` cap, and the
404-on-unknown-path policy.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from dikw_core.api import _doc_id_for, _with_storage
from dikw_core.schemas import DocumentRecord, Layer, LinkRecord, LinkType


def _doc(path: str) -> DocumentRecord:
    return DocumentRecord(
        doc_id=_doc_id_for(Layer.KNOWLEDGE, path),
        path=path,
        hash="0" * 64,
        mtime=0.0,
        layer=Layer.KNOWLEDGE,
        active=True,
    )


async def _seed_triangle(root: Path) -> tuple[str, str, str]:
    a_path = "knowledge/a.md"
    b_path = "knowledge/b.md"
    c_path = "knowledge/c.md"
    cfg, _root, storage = await _with_storage(root)
    del cfg
    try:
        for p in (a_path, b_path, c_path):
            await storage.upsert_document(_doc(p))
        await storage.upsert_link(
            LinkRecord(
                src_doc_id=_doc_id_for(Layer.KNOWLEDGE, a_path),
                dst_path=b_path,
                link_type=LinkType.WIKILINK,
                anchor=None,
                line=5,
            )
        )
        await storage.upsert_link(
            LinkRecord(
                src_doc_id=_doc_id_for(Layer.KNOWLEDGE, b_path),
                dst_path=c_path,
                link_type=LinkType.WIKILINK,
                anchor="Section",
                line=7,
            )
        )
    finally:
        await storage.close()
    return a_path, b_path, c_path


@pytest.mark.asyncio
async def test_get_page_links_both_returns_in_and_out(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    a_path, b_path, c_path = await _seed_triangle(base_root)
    resp = await server_client.get(f"/v1/base/pages/{b_path}/links")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["path"] == b_path
    assert len(body["outgoing"]) == 1
    out = body["outgoing"][0]
    assert out["dst_path"] == c_path
    assert out["link_type"] == "wikilink"
    assert out["line"] == 7
    assert out["anchor"] == "Section"
    assert len(body["incoming"]) == 1
    inb = body["incoming"][0]
    assert inb["src_path"] == a_path
    assert inb["src_doc_id"] == _doc_id_for(Layer.KNOWLEDGE, a_path)
    assert inb["link_type"] == "wikilink"
    assert inb["line"] == 5


@pytest.mark.asyncio
async def test_get_page_links_out_only(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    _, b_path, _ = await _seed_triangle(base_root)
    resp = await server_client.get(
        f"/v1/base/pages/{b_path}/links", params={"direction": "out"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["outgoing"] and body["incoming"] == []


@pytest.mark.asyncio
async def test_get_page_links_in_only(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    _, b_path, _ = await _seed_triangle(base_root)
    resp = await server_client.get(
        f"/v1/base/pages/{b_path}/links", params={"direction": "in"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["incoming"] and body["outgoing"] == []


@pytest.mark.asyncio
async def test_get_page_links_unknown_path_404(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    resp = await server_client.get("/v1/base/pages/knowledge/missing.md/links")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "page_not_found"


@pytest.mark.asyncio
async def test_get_page_links_rejects_bad_direction(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    _, b_path, _ = await _seed_triangle(base_root)
    resp = await server_client.get(
        f"/v1/base/pages/{b_path}/links", params={"direction": "sideways"}
    )
    # FastAPI enum/literal validation → 422.
    assert resp.status_code in (400, 422)


@pytest.mark.asyncio
async def test_get_page_links_limit_caps_each_list(
    server_client: httpx.AsyncClient, base_root: Path
) -> None:
    hub_path = "knowledge/hub.md"
    cfg, _root, storage = await _with_storage(base_root)
    del cfg
    try:
        await storage.upsert_document(_doc(hub_path))
        for i, dst in enumerate(("knowledge/x.md", "knowledge/y.md", "knowledge/z.md"), start=1):
            await storage.upsert_document(_doc(dst))
            await storage.upsert_link(
                LinkRecord(
                    src_doc_id=_doc_id_for(Layer.KNOWLEDGE, hub_path),
                    dst_path=dst,
                    link_type=LinkType.WIKILINK,
                    anchor=None,
                    line=i,
                )
            )
        for i, src in enumerate(("knowledge/p.md", "knowledge/q.md", "knowledge/r.md"), start=1):
            await storage.upsert_document(_doc(src))
            await storage.upsert_link(
                LinkRecord(
                    src_doc_id=_doc_id_for(Layer.KNOWLEDGE, src),
                    dst_path=hub_path,
                    link_type=LinkType.WIKILINK,
                    anchor=None,
                    line=10 + i,
                )
            )
    finally:
        await storage.close()

    resp = await server_client.get(
        f"/v1/base/pages/{hub_path}/links", params={"limit": 2}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["outgoing"]) == 2
    assert len(body["incoming"]) == 2
