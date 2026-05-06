"""End-to-end ingest task tests through the FastAPI app.

Drives the full ``upload → ingest`` loop with the in-memory ASGI
transport and a fake provider config (so ``no_embed=True`` keeps the
test off any real network). Asserts:

  * Upload-id-driven ingest commits the staged tree onto ``wiki/sources``
    and the resulting ``IngestReport`` lands in the task ``final``.
  * Without ``upload_id``, ingest still runs against on-disk sources.
  * Replaying ``GET /v1/tasks/{id}/events`` after terminal returns the
    full tape (including the final event).
  * ``GET /v1/tasks/{id}/events?from_seq=N`` truncates correctly.
  * Unknown ``upload_id`` surfaces as a task ``failed`` (not 404 on the
    submit) — the validation runs in the runner so the task tape carries
    the failure reason.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import httpx
import pytest


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path, body in files.items():
            ti = tarfile.TarInfo(path)
            ti.size = len(body)
            tf.addfile(ti, io.BytesIO(body))
    return buf.getvalue()


def _manifest_for(files: dict[str, bytes]) -> dict[str, Any]:
    return {
        "files": [
            {
                "path": p,
                "size": len(b),
                "sha256": hashlib.sha256(b).hexdigest(),
            }
            for p, b in files.items()
        ],
        "total_bytes": sum(len(b) for b in files.values()),
    }


async def _wait_terminal(
    client: httpx.AsyncClient, task_id: str, *, timeout: float = 10.0
) -> dict[str, Any]:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        r = await client.get(f"/v1/tasks/{task_id}")
        if r.status_code == 200 and r.json()["status"] in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            return r.json()
        await asyncio.sleep(0.05)
    raise AssertionError(f"task {task_id} never reached a terminal state")


# ---- happy path ---------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_then_ingest_round_trip(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    files = {
        "sources/notes/alpha.md": b"# Alpha\n\nFirst note.\n",
        "sources/notes/beta.md": b"# Beta\n\nSecond note.\n",
    }
    upload = await server_client.post(
        "/v1/upload/sources",
        files={
            "payload": ("u.tar.gz", _tar_bytes(files), "application/gzip"),
        },
        data={"manifest": json.dumps(_manifest_for(files))},
    )
    assert upload.status_code == 200, upload.text
    upload_id = upload.json()["upload_id"]

    submit = await server_client.post(
        "/v1/ingest", json={"upload_id": upload_id, "no_embed": True}
    )
    assert submit.status_code == 200, submit.text
    handle = submit.json()
    assert handle["op"] == "ingest"
    task_id = handle["task_id"]

    row = await _wait_terminal(server_client, task_id)
    assert row["status"] == "succeeded", row
    result_resp = await server_client.get(f"/v1/tasks/{task_id}/result")
    assert result_resp.status_code == 200
    result = result_resp.json()["result"]
    assert result["scanned"] == 2
    assert result["added"] == 2
    assert result["updated"] == 0
    assert result["unchanged"] == 0
    assert result["chunks"] >= 2
    # ``embedded`` stays 0 because no_embed=True.
    assert result["embedded"] == 0
    # upload_commit is recorded so the client can verify the staging
    # tree was applied as expected.
    assert result["upload_commit"] == {"sources": 2, "assets": 0}

    # Files actually on disk in the wiki root.
    assert (wiki_root / "sources" / "notes" / "alpha.md").read_bytes() == files[
        "sources/notes/alpha.md"
    ]
    # Staging tree is gone.
    assert not (wiki_root / ".dikw" / "upload-staging" / upload_id).exists()


@pytest.mark.asyncio
async def test_ingest_without_upload_uses_existing_sources(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    src_dir = wiki_root / "sources" / "notes"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "preexisting.md").write_text("# Pre\nbody\n", encoding="utf-8")

    submit = await server_client.post(
        "/v1/ingest", json={"no_embed": True}
    )
    assert submit.status_code == 200
    task_id = submit.json()["task_id"]
    row = await _wait_terminal(server_client, task_id)
    assert row["status"] == "succeeded"

    result = (await server_client.get(f"/v1/tasks/{task_id}/result")).json()[
        "result"
    ]
    assert result["scanned"] == 1
    assert result["added"] == 1
    # No upload_commit field when ingest ran without an upload_id.
    assert "upload_commit" not in result


# ---- event tape replay --------------------------------------------------


@pytest.mark.asyncio
async def test_event_tape_replay_after_terminal(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    src_dir = wiki_root / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "x.md").write_text("# X\n", encoding="utf-8")

    submit = await server_client.post(
        "/v1/ingest", json={"no_embed": True}
    )
    task_id = submit.json()["task_id"]
    await _wait_terminal(server_client, task_id)

    async with server_client.stream(
        "GET", f"/v1/tasks/{task_id}/events"
    ) as resp:
        assert resp.status_code == 200
        lines = [
            line.strip() for line in [
                ln async for ln in resp.aiter_lines()
            ] if line.strip()
        ]
    events = [json.loads(line) for line in lines]
    assert events[0]["type"] == "task_started"
    assert events[0]["op"] == "ingest"
    assert events[-1]["type"] == "final"
    assert events[-1]["status"] == "succeeded"
    # ``scan`` phase fires at least once (initial, plus per-file).
    assert any(
        e["type"] == "progress" and e["phase"] == "scan" for e in events
    )


@pytest.mark.asyncio
async def test_resume_from_seq_returns_tail_only(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    src_dir = wiki_root / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "x.md").write_text("# X\n", encoding="utf-8")

    submit = await server_client.post(
        "/v1/ingest", json={"no_embed": True}
    )
    task_id = submit.json()["task_id"]
    await _wait_terminal(server_client, task_id)

    # First read the full tape to learn the seq range.
    async with server_client.stream(
        "GET", f"/v1/tasks/{task_id}/events"
    ) as resp:
        full = [
            json.loads(ln)
            for ln in [line async for line in resp.aiter_lines()]
            if ln.strip()
        ]
    last_seq = full[-1]["seq"]

    # Resume from the middle.
    cutoff = last_seq // 2 + 1
    async with server_client.stream(
        "GET", f"/v1/tasks/{task_id}/events?from_seq={cutoff}"
    ) as resp:
        tail = [
            json.loads(ln)
            for ln in [line async for line in resp.aiter_lines()]
            if ln.strip()
        ]
    assert tail, "tail should not be empty when from_seq < last_seq"
    assert tail[0]["seq"] >= cutoff
    assert tail[-1]["type"] == "final"


# ---- per-file error surface --------------------------------------------


@pytest.mark.asyncio
async def test_file_error_event_lands_on_event_tape(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    """Per-file failures during ingest must surface on the event tape
    as ``partial`` events with ``kind=file_error`` so a client tailing
    the NDJSON stream sees the failure live, and must also land on
    ``IngestReport.errors`` in the final result so a non-streaming
    poller sees the same information."""
    src_dir = wiki_root / "sources" / "notes"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "good.md").write_text("# Good\n\nbody.\n", encoding="utf-8")
    # Broken YAML front-matter — frontmatter.loads → yaml.YAMLError →
    # caught by the engine's parse_error branch.
    (src_dir / "broken.md").write_text(
        "---\nbroken: : :\n---\n# T\n", encoding="utf-8"
    )

    submit = await server_client.post(
        "/v1/ingest", json={"no_embed": True}
    )
    task_id = submit.json()["task_id"]
    row = await _wait_terminal(server_client, task_id)
    # The run as a whole succeeds — per-file errors are non-fatal.
    assert row["status"] == "succeeded"

    # Wire-event coverage.
    async with server_client.stream(
        "GET", f"/v1/tasks/{task_id}/events"
    ) as resp:
        events = [
            json.loads(ln)
            for ln in [line async for line in resp.aiter_lines()]
            if ln.strip()
        ]
    file_error_events = [
        e for e in events
        if e["type"] == "partial" and e.get("kind") == "file_error"
    ]
    assert len(file_error_events) == 1, file_error_events
    payload = file_error_events[0]["payload"]
    assert payload["kind"] == "parse_error"
    assert payload["path"].endswith("broken.md")
    assert payload["message"]

    # Final-report coverage.
    result = (await server_client.get(f"/v1/tasks/{task_id}/result")).json()[
        "result"
    ]
    assert isinstance(result["errors"], list) and len(result["errors"]) == 1
    err = result["errors"][0]
    assert err["kind"] == "parse_error"
    assert err["path"].endswith("broken.md")
    # ``good.md`` still ingested cleanly.
    assert result["added"] == 1


# ---- failure paths ------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_upload_id_makes_task_fail(
    server_client: httpx.AsyncClient,
) -> None:
    submit = await server_client.post(
        "/v1/ingest",
        json={"upload_id": "deadbeef0000", "no_embed": True},
    )
    # Submit succeeds — the runner is what catches the missing upload.
    assert submit.status_code == 200
    task_id = submit.json()["task_id"]
    row = await _wait_terminal(server_client, task_id)
    assert row["status"] == "failed"

    result = (await server_client.get(f"/v1/tasks/{task_id}/result")).json()
    assert result["status"] == "failed"
    assert result["error"] is not None
    # ApiError → ``code`` field on the persisted error dict.
    assert "deadbeef0000" in result["error"]["message"]
