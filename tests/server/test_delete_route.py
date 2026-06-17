"""HTTP-level tests for ``POST /v1/base/delete``.

Covers the submit → task SUCCEEDED loop against the in-memory ASGI
runtime: the purge + trash side-effects, the audit ``reason`` passthrough,
the failed-task path for an unknown document, and the 422 boundary reject
for a blank path. Documents are seeded straight into the ``documents``
table via :func:`seed_doc` (no embedder needed — delete touches no
vectors).
"""

from __future__ import annotations

from pathlib import Path

import frontmatter
import httpx
import pytest

from dikw_core.schemas import Layer

from ..fakes import seed_doc
from .conftest import wait_task_terminal


@pytest.mark.asyncio
async def test_delete_route_purges_and_trashes(
    server_client: httpx.AsyncClient, base_root: Path,
) -> None:
    path = "knowledge/concepts/dead.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=path, body="# Dead\n", title="Dead"
    )

    resp = await server_client.post("/v1/base/delete", json={"path": path})
    assert resp.status_code == 200, resp.text
    handle = resp.json()
    assert handle["op"] == "delete"
    task_id = handle["task_id"]

    row = await wait_task_terminal(server_client, task_id)
    assert row["status"] == "succeeded", row

    result = (await server_client.get(f"/v1/tasks/{task_id}/result")).json()
    payload = result["result"]
    assert payload["path"] == path
    assert payload["layer"] == "knowledge"
    assert payload["trashed_to"] == "trash/knowledge/concepts/dead.md"

    assert (base_root / "trash" / "knowledge" / "concepts" / "dead.md").is_file()
    assert not (base_root / path).exists()


@pytest.mark.asyncio
async def test_delete_route_unknown_path_task_fails(
    server_client: httpx.AsyncClient, base_root: Path,
) -> None:
    """An unregistered path can only be detected by the engine (storage
    probe), so it surfaces as a FAILED task — not a submit-time 422."""
    resp = await server_client.post(
        "/v1/base/delete", json={"path": "knowledge/never-existed.md"}
    )
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]
    row = await wait_task_terminal(server_client, task_id)
    assert row["status"] == "failed", row
    _ = base_root


@pytest.mark.asyncio
async def test_delete_route_rejects_blank_path(
    server_client: httpx.AsyncClient, base_root: Path,
) -> None:
    resp = await server_client.post("/v1/base/delete", json={"path": "   "})
    assert resp.status_code == 422, resp.text
    _ = base_root


@pytest.mark.asyncio
async def test_delete_route_reason_in_trash_audit(
    server_client: httpx.AsyncClient, base_root: Path,
) -> None:
    path = "wisdom/scratch.md"
    await seed_doc(
        base_root, layer=Layer.WISDOM, path=path, body="# Scratch\n", title="Scratch"
    )

    resp = await server_client.post(
        "/v1/base/delete", json={"path": path, "reason": "obsolete"}
    )
    assert resp.status_code == 200
    row = await wait_task_terminal(server_client, resp.json()["task_id"])
    assert row["status"] == "succeeded", row

    trashed = frontmatter.loads(
        (base_root / "trash" / "wisdom" / "scratch.md").read_text(encoding="utf-8")
    ).metadata.get("trashed")
    assert isinstance(trashed, dict)
    assert trashed.get("reason") == "obsolete"
