"""End-to-end ingest task tests through the FastAPI app.

In the post-refactor world, ``/v1/ingest`` is a pure scan-disk task:
the client imports sources separately via ``/v1/import`` (which
commits straight into ``<base>/sources/``), then calls ingest to
chunk + embed whatever lives on disk. The previous ``upload_id``
parameter is gone — see ``test_import_packages.py`` for the import
side of the contract.

Asserts:

  * Ingest scans ``<base>/sources/`` and reports the right counts.
  * ``GET /v1/tasks/{id}/events`` after terminal returns the full tape.
  * ``GET /v1/tasks/{id}/events?from_seq=N`` truncates correctly.
  * Per-file parse errors surface on the event tape AND in the final
    ``IngestReport.errors`` list (non-fatal).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from .conftest import wait_event_tape_final as _wait_tape_final
from .conftest import wait_task_terminal as _wait_terminal

# ---- happy path ---------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_scans_existing_sources(
    server_client: httpx.AsyncClient,
    base_root: Path,
) -> None:
    src_dir = base_root / "sources" / "notes"
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


@pytest.mark.asyncio
async def test_ingest_submit_rejects_unknown_fields(
    server_client: httpx.AsyncClient,
) -> None:
    """``IngestSubmit`` has ``extra: 'forbid'``; a request body carrying
    fields the schema doesn't know must yield 422 instead of silently
    succeeding (which would let typos like ``no_embeed`` look fine)."""
    submit = await server_client.post(
        "/v1/ingest", json={"unknown_extra_field": "x", "no_embed": True}
    )
    assert submit.status_code == 422, submit.text


# ---- event tape replay --------------------------------------------------


@pytest.mark.asyncio
async def test_event_tape_replay_after_terminal(
    server_client: httpx.AsyncClient,
    base_root: Path,
) -> None:
    src_dir = base_root / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "x.md").write_text("# X\n", encoding="utf-8")

    submit = await server_client.post(
        "/v1/ingest", json={"no_embed": True}
    )
    task_id = submit.json()["task_id"]
    # Wait for the ``final`` event to land on the tape, not just the status
    # row — the manager flips the row terminal *before* appending ``final``,
    # so a bare ``wait=0`` read races the trailing ``progress`` event onto
    # the last slot (see ``wait_event_tape_final``).
    events = await _wait_tape_final(server_client, task_id)
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
    base_root: Path,
) -> None:
    src_dir = base_root / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "x.md").write_text("# X\n", encoding="utf-8")

    submit = await server_client.post(
        "/v1/ingest", json={"no_embed": True}
    )
    task_id = submit.json()["task_id"]
    # Read the full tape once ``final`` has landed, to learn the seq range
    # without racing the status-row/tape ordering (see ``wait_event_tape_final``).
    full = await _wait_tape_final(server_client, task_id)
    last_seq = full[-1]["seq"]

    # Resume from the middle.
    cutoff = last_seq // 2 + 1
    tail_resp = await server_client.get(
        f"/v1/tasks/{task_id}/events",
        params={"from_seq": cutoff, "limit": 1000, "wait": 0},
    )
    tail = tail_resp.json()["events"]
    assert tail, "tail should not be empty when from_seq < last_seq"
    assert tail[0]["seq"] >= cutoff
    assert tail[-1]["type"] == "final"


# ---- per-file error surface --------------------------------------------


@pytest.mark.asyncio
async def test_file_error_event_lands_on_event_tape(
    server_client: httpx.AsyncClient,
    base_root: Path,
) -> None:
    """Per-file failures during ingest must surface on the event tape
    as ``partial`` events with ``kind=file_error`` so a client tailing
    the NDJSON stream sees the failure live, and must also land on
    ``IngestReport.errors`` in the final result so a non-streaming
    poller sees the same information."""
    src_dir = base_root / "sources" / "notes"
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
    resp = await server_client.get(
        f"/v1/tasks/{task_id}/events",
        params={"from_seq": 0, "limit": 1000, "wait": 0},
    )
    events = resp.json()["events"]
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
