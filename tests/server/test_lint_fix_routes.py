"""HTTP-level tests for ``POST /v1/lint/propose`` + ``POST /v1/lint/apply``.

Covers the full submit → propose-task SUCCEEDED → apply-task SUCCEEDED
loop end-to-end against the in-memory ASGI runtime, plus the apply
runner's failure modes (bad ``proposal_task_id``, non-terminal source
task) so the wire contract stays explicit.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from dikw_core import api as api_module
from dikw_core.schemas import DocumentRecord, Layer


def _wiki_doc_id(path: str) -> str:
    from dikw_core.domains.data.path_norm import normalize_path

    return f"knowledge:{normalize_path(path)}"


async def _wait_terminal(
    client: httpx.AsyncClient, task_id: str, *, timeout: float = 15.0
) -> dict[str, Any]:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        r = await client.get(f"/v1/tasks/{task_id}")
        if r.status_code == 200 and r.json()["status"] in {
            "succeeded", "failed", "cancelled",
        }:
            return r.json()
        await asyncio.sleep(0.05)
    raise AssertionError(f"task {task_id} never terminated")


async def _seed_broken_link_pages(base_root: Path) -> None:
    """Drop two pages on disk + register them via the engine API
    (independent of HTTP, so propose has work to do when called)."""
    target = base_root / "knowledge/concepts/foo-bar.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "---\nid: K-foobar\ntype: concept\ntitle: Foo Bar\n"
        "created: 2026-05-09T00:00:00+00:00\n"
        "updated: 2026-05-09T00:00:00+00:00\n---\n\n"
        "# Foo Bar\n\nbody\n",
        encoding="utf-8",
    )
    src = base_root / "knowledge/concepts/source.md"
    src.write_text(
        "---\nid: K-source\ntype: concept\ntitle: Source\n"
        "created: 2026-05-09T00:00:00+00:00\n"
        "updated: 2026-05-09T00:00:00+00:00\n---\n\n"
        "# Source\n\nSee [[fooo bar]] for context.\n",
        encoding="utf-8",
    )
    _cfg, _root, storage = await api_module._with_storage(base_root)
    try:
        for path, title in [
            ("knowledge/concepts/foo-bar.md", "Foo Bar"),
            ("knowledge/concepts/source.md", "Source"),
        ]:
            await storage.upsert_document(
                DocumentRecord(
                    doc_id=_wiki_doc_id(path), path=path, title=title,
                    hash=f"hash-{path}", mtime=0.0,
                    layer=Layer.KNOWLEDGE, active=True,
                )
            )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_lint_propose_then_apply_full_flow(
    server_client: httpx.AsyncClient, base_root: Path,
) -> None:
    await _seed_broken_link_pages(base_root)

    # 1. submit propose.
    resp = await server_client.post(
        "/v1/lint/propose",
        json={"rule": "broken_wikilink", "limit": 10},
    )
    assert resp.status_code == 200, resp.text
    propose_handle = resp.json()
    propose_id = propose_handle["task_id"]
    assert propose_handle["op"] == "lint.propose"

    # 2. wait for propose to terminate; result holds 1 proposal.
    propose_row = await _wait_terminal(server_client, propose_id)
    assert propose_row["status"] == "succeeded"

    res = await server_client.get(f"/v1/tasks/{propose_id}/result")
    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "succeeded"
    proposals = payload["result"]["proposals"]
    assert len(proposals) == 1
    assert proposals[0]["operations"][0]["kind"] == "update_page"

    # 3. submit apply referencing the propose task.
    resp = await server_client.post(
        "/v1/lint/apply",
        json={"proposal_task_id": propose_id},
    )
    assert resp.status_code == 200, resp.text
    apply_handle = resp.json()
    apply_id = apply_handle["task_id"]

    apply_row = await _wait_terminal(server_client, apply_id)
    assert apply_row["status"] == "succeeded", apply_row

    apply_res = (
        await server_client.get(f"/v1/tasks/{apply_id}/result")
    ).json()
    applied = apply_res["result"]["applied"]
    assert len(applied) == 1
    # 4. on disk: the source page now references the resolved target.
    rewritten = (base_root / "knowledge/concepts/source.md").read_text(
        encoding="utf-8"
    )
    assert "[[Foo Bar]]" in rewritten
    assert "[[fooo bar]]" not in rewritten


@pytest.mark.asyncio
async def test_lint_apply_unknown_propose_id_fails_with_clear_cause(
    server_client: httpx.AsyncClient, base_root: Path,
) -> None:
    _ = base_root  # runtime lifespan needs the fixture in scope
    resp = await server_client.post(
        "/v1/lint/apply",
        json={"proposal_task_id": "no-such-task"},
    )
    # The submit succeeds (task is registered); the runner raises
    # NotFoundError, which the manager surfaces as a FAILED final.
    assert resp.status_code == 200, resp.text
    apply_id = resp.json()["task_id"]
    row = await _wait_terminal(server_client, apply_id)
    assert row["status"] == "failed"
    err_payload = (
        await server_client.get(f"/v1/tasks/{apply_id}/result")
    ).json()
    assert err_payload["status"] == "failed"
    assert "no-such-task" in str(err_payload["error"])


@pytest.mark.asyncio
async def test_lint_apply_rejects_non_propose_task_id(
    server_client: httpx.AsyncClient, base_root: Path,
) -> None:
    """If the caller passes a SUCCEEDED task from a different op (e.g.
    ``echo``), apply must reject with a ``proposal_wrong_op`` error
    rather than silently treating an unrelated result dict as an empty
    ``FixProposalReport``."""
    _ = base_root
    echo = await server_client.post("/v1/echo", json={"count": 1})
    assert echo.status_code == 200, echo.text
    echo_id = echo.json()["task_id"]
    await _wait_terminal(server_client, echo_id)

    apply_resp = await server_client.post(
        "/v1/lint/apply", json={"proposal_task_id": echo_id}
    )
    assert apply_resp.status_code == 200, apply_resp.text
    apply_id = apply_resp.json()["task_id"]
    row = await _wait_terminal(server_client, apply_id)
    assert row["status"] == "failed"
    err = (await server_client.get(f"/v1/tasks/{apply_id}/result")).json()
    assert err["status"] == "failed"
    assert "lint.propose" in str(err["error"])


@pytest.mark.asyncio
async def test_lint_propose_rejects_invalid_limit(
    server_client: httpx.AsyncClient, base_root: Path,
) -> None:
    _ = base_root
    resp = await server_client.post(
        "/v1/lint/propose",
        json={"limit": 0},
    )
    # pydantic Field(ge=1) rejects with 422 unprocessable.
    assert resp.status_code == 422
    assert "limit" in resp.text.lower()
