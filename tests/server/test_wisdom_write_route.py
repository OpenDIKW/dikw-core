"""HTTP-level tests for ``POST /v1/base/wisdom``.

Covers the submit → task SUCCEEDED loop end-to-end against the
in-memory ASGI runtime: payload schema validation, the upsert
semantics, the cross-file ``[[wikilink]]`` resolve through the
shared title index, and the failure modes the route surfaces (400
for non-kebab inputs, 422 for missing required fields).

The route runs with ``no_embed: True`` in all tests so the server
doesn't need a real embedding provider — the engine-layer tests
(``tests/test_write_wisdom_page.py``) cover the embedding branch
against a ``FakeEmbeddings`` injection.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from .conftest import wait_task_terminal


@pytest.mark.asyncio
async def test_wisdom_write_creates_file_and_task_succeeds(
    server_client: httpx.AsyncClient, base_root: Path,
) -> None:
    resp = await server_client.post(
        "/v1/base/wisdom",
        json={
            "slug": "first-principles",
            "author": "elon-musk",
            "title": "First Principles",
            "body": "Reason from physics.\n",
            "no_embed": True,
        },
    )
    assert resp.status_code == 200, resp.text
    handle = resp.json()
    assert handle["op"] == "wisdom.write"
    task_id = handle["task_id"]

    row = await wait_task_terminal(server_client, task_id)
    assert row["status"] == "succeeded", row

    result = (await server_client.get(f"/v1/tasks/{task_id}/result")).json()
    assert result["status"] == "succeeded"
    payload = result["result"]
    assert payload["path"] == "wisdom/elon-musk/first-principles.md"
    assert payload["created"] is True
    assert payload["embedded"] == 0  # no_embed=True

    abs_path = base_root / "wisdom" / "elon-musk" / "first-principles.md"
    assert abs_path.is_file()


@pytest.mark.asyncio
async def test_wisdom_write_upsert_marks_updated(
    server_client: httpx.AsyncClient, base_root: Path,
) -> None:
    body1 = {"slug": "x", "title": "X", "body": "first.\n", "no_embed": True}
    r1 = await server_client.post("/v1/base/wisdom", json=body1)
    assert r1.status_code == 200
    row1 = await wait_task_terminal(server_client, r1.json()["task_id"])
    assert row1["status"] == "succeeded"
    result1 = (await server_client.get(f"/v1/tasks/{r1.json()['task_id']}/result")).json()
    assert result1["result"]["created"] is True

    body2 = {"slug": "x", "title": "X", "body": "second body.\n", "no_embed": True}
    r2 = await server_client.post("/v1/base/wisdom", json=body2)
    assert r2.status_code == 200
    row2 = await wait_task_terminal(server_client, r2.json()["task_id"])
    assert row2["status"] == "succeeded"
    result2 = (await server_client.get(f"/v1/tasks/{r2.json()['task_id']}/result")).json()
    assert result2["result"]["created"] is False
    _ = base_root


@pytest.mark.asyncio
async def test_wisdom_write_rejects_non_kebab_slug(
    server_client: httpx.AsyncClient, base_root: Path,
) -> None:
    resp = await server_client.post(
        "/v1/base/wisdom",
        json={
            "slug": "Foo Bar",  # space + uppercase
            "title": "Foo",
            "body": "b.\n",
            "no_embed": True,
        },
    )
    # Pydantic field_validator → 422
    assert resp.status_code == 422, resp.text
    _ = base_root


@pytest.mark.asyncio
async def test_wisdom_write_rejects_non_kebab_author(
    server_client: httpx.AsyncClient, base_root: Path,
) -> None:
    resp = await server_client.post(
        "/v1/base/wisdom",
        json={
            "slug": "good-slug",
            "author": "ElonMusk",
            "title": "Foo",
            "body": "b.\n",
            "no_embed": True,
        },
    )
    assert resp.status_code == 422, resp.text
    _ = base_root


@pytest.mark.asyncio
async def test_wisdom_write_rejects_missing_title(
    server_client: httpx.AsyncClient, base_root: Path,
) -> None:
    resp = await server_client.post(
        "/v1/base/wisdom",
        json={"slug": "ok", "body": "b.\n", "no_embed": True},
    )
    assert resp.status_code == 422
    _ = base_root


@pytest.mark.asyncio
async def test_wisdom_write_emits_progress_event(
    server_client: httpx.AsyncClient, base_root: Path,
) -> None:
    """The runner must emit at least one ``wisdom_write`` phase event so
    NDJSON consumers (UI / agent observability) can see progress."""
    resp = await server_client.post(
        "/v1/base/wisdom",
        json={
            "slug": "tracked",
            "title": "Tracked",
            "body": "body.\n",
            "no_embed": True,
        },
    )
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]
    row = await wait_task_terminal(server_client, task_id)
    assert row["status"] == "succeeded", row

    events_resp = await server_client.get(f"/v1/tasks/{task_id}/events?wait=0")
    assert events_resp.status_code == 200
    events = events_resp.json()["events"]
    assert any(
        e.get("phase") == "wisdom_write" for e in events
    ), [e.get("phase") for e in events]
    _ = base_root
