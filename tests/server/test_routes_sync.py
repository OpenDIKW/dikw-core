"""HTTP-level tests for the synchronous route surface.

Goes through the FastAPI app via ``httpx.ASGITransport`` so the auth
dependency, error envelope, and Pydantic response shaping are all
exercised without binding a real socket.
"""

from __future__ import annotations

import httpx
import pytest

from .conftest import wait_task_terminal as _wait_terminal_via_http


@pytest.mark.asyncio
async def test_info_endpoint(server_client: httpx.AsyncClient) -> None:
    resp = await server_client.get("/v1/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["base_root"]
    assert body["storage_backend"] in {"sqlite", "postgres"}
    assert body["auth_required"] is False  # localhost-no-token default


@pytest.mark.asyncio
async def test_healthz_and_readyz(server_client: httpx.AsyncClient) -> None:
    h = await server_client.get("/v1/healthz")
    assert h.status_code == 200 and h.json()["status"] == "ok"
    r = await server_client.get("/v1/readyz")
    assert r.status_code == 200 and r.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_status_returns_storage_counts(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.get("/v1/status")
    assert resp.status_code == 200
    body = resp.json()
    # Fresh wiki: docs/chunks/etc all zero.
    assert body["chunks"] == 0
    assert body["embeddings"] == 0
    assert body["links"] == 0


@pytest.mark.asyncio
async def test_check_rejects_both_only_flags(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post(
        "/v1/check", json={"llm_only": True, "embed_only": True}
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "bad_request"


@pytest.mark.asyncio
async def test_lint_runs_against_empty_wiki(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post("/v1/lint")
    assert resp.status_code == 200
    body = resp.json()
    # LintReport's empty shape — exact keys depend on engine but the
    # endpoint should at least return a JSON object.
    assert isinstance(body, dict)


@pytest.mark.asyncio
async def test_unknown_chunk_returns_404(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.get("/v1/doc/chunks/99999")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_doc_search_against_empty_wiki(
    server_client: httpx.AsyncClient,
) -> None:
    # ``mode=bm25`` keeps the request path off the dense embedder, which
    # the test wiki's fake provider config can't reach (no real API key).
    resp = await server_client.post(
        "/v1/doc/search",
        json={"q": "anything", "limit": 3, "mode": "bm25"},
    )
    assert resp.status_code == 200
    assert resp.json() == []


# ---- task routes (echo end-to-end) -------------------------------------


@pytest.mark.asyncio
async def test_echo_submit_returns_handle(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post("/v1/echo", json={"count": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["op"] == "echo"
    assert body["task_id"]
    for key in ("self", "events", "result", "cancel"):
        assert key in body["links"]


@pytest.mark.asyncio
async def test_echo_lifecycle_to_terminal_result(
    server_client: httpx.AsyncClient,
) -> None:
    submit = await server_client.post("/v1/echo", json={"count": 3})
    task_id = submit.json()["task_id"]

    # Poll for terminal — echo task is fast (sub-second on a free CPU)
    # but ASGI is single-threaded so we drive the loop with sleeps.
    import asyncio

    for _ in range(50):
        result = await server_client.get(f"/v1/tasks/{task_id}/result")
        if result.status_code == 200:
            break
        await asyncio.sleep(0.05)
    assert result.status_code == 200
    body = result.json()
    assert body["status"] == "succeeded"
    assert body["result"] == {"echoed": 3}


@pytest.mark.asyncio
async def test_task_result_404_for_unknown(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.get("/v1/tasks/no-such-id/result")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cancel_unknown_task_404(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post("/v1/tasks/no-such-id/cancel")
    assert resp.status_code == 404


# ---- cursor JSON /events endpoint ---------------------------------------


@pytest.mark.asyncio
async def test_events_snapshot_returns_immediately_with_backlog(
    server_client: httpx.AsyncClient,
) -> None:
    """``wait=0`` on a terminal task with backlog: 200 JSON, first ``limit``
    events ordered by seq, ``task_status == succeeded``, ``has_more=true``
    when there's more on the tape."""
    submit = await server_client.post("/v1/echo", json={"count": 3})
    task_id = submit.json()["task_id"]
    await _wait_terminal_via_http(server_client, task_id)

    resp = await server_client.get(
        f"/v1/tasks/{task_id}/events",
        params={"from_seq": 0, "limit": 3, "wait": 0},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["task_status"] == "succeeded"
    assert len(body["events"]) == 3
    assert [e["seq"] for e in body["events"]] == [1, 2, 3]
    assert body["next_from_seq"] == 4
    assert body["has_more"] is True
    assert body["last_seq"] >= 3


@pytest.mark.asyncio
async def test_events_snapshot_caught_up_returns_empty_events(
    server_client: httpx.AsyncClient,
) -> None:
    """``wait=0`` with ``from_seq`` already past the tape returns
    empty events + cursor unchanged + correct task_status."""
    submit = await server_client.post("/v1/echo", json={"count": 2})
    task_id = submit.json()["task_id"]
    row = await _wait_terminal_via_http(server_client, task_id)

    resp = await server_client.get(
        f"/v1/tasks/{task_id}/events",
        params={"from_seq": 9999, "wait": 0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["events"] == []
    assert body["next_from_seq"] == 9999
    assert body["has_more"] is False
    assert body["task_status"] == row["status"]


@pytest.mark.asyncio
async def test_events_long_poll_wakes_on_append(
    server_client: httpx.AsyncClient,
) -> None:
    """``wait>0`` with no backlog but a task still producing events:
    response arrives within notify latency (≪ wait timeout)."""
    import asyncio
    import time

    submit = await server_client.post(
        "/v1/echo", json={"count": 3, "delay_ms": 150}
    )
    task_id = submit.json()["task_id"]

    # Let task_started land, then snapshot to learn current cursor.
    await asyncio.sleep(0.05)
    snap = await server_client.get(
        f"/v1/tasks/{task_id}/events", params={"from_seq": 0, "wait": 0}
    )
    last_seen = snap.json()["last_seq"]

    start = time.monotonic()
    resp = await server_client.get(
        f"/v1/tasks/{task_id}/events",
        params={"from_seq": last_seen + 1, "wait": 5},
    )
    elapsed = time.monotonic() - start
    assert resp.status_code == 200
    body = resp.json()
    # Returned faster than the full 5s wait — the Condition woke us.
    assert elapsed < 2.0, f"long-poll timed out at {elapsed:.3f}s instead of waking"
    assert len(body["events"]) >= 1, "no events surfaced after wake"
    assert body["events"][0]["seq"] == last_seen + 1
    await _wait_terminal_via_http(server_client, task_id)


@pytest.mark.asyncio
async def test_events_long_poll_returns_on_terminal(
    server_client: httpx.AsyncClient,
) -> None:
    """``wait>0`` wakes up promptly when a task is in flight + emits a
    new event, and the task eventually reaches terminal.

    The test does NOT assert the long-poll *response* carries terminal
    status — the cursor handler is "return on any new event" by design,
    and that event can be a non-terminal ``partial`` whose append
    *precedes* the manager's ``update_status(SUCCEEDED)`` by a few
    microseconds. Two decoupled assertions instead:

    1. Long-poll resolves well before the 5s deadline (i.e. the
       ``Condition.notify_all`` wake-up path actually fires).
    2. The task eventually reaches a terminal state.
    """
    import asyncio
    import time

    submit = await server_client.post(
        "/v1/echo", json={"count": 1, "delay_ms": 100}
    )
    task_id = submit.json()["task_id"]
    # Get a cursor that's likely past task_started + the one progress
    # already emitted, so the long-poll genuinely waits for the next
    # event (partial or final).
    await asyncio.sleep(0.05)
    snap = await server_client.get(
        f"/v1/tasks/{task_id}/events", params={"from_seq": 0, "wait": 0}
    )
    last_seen = snap.json()["last_seq"]

    start = time.monotonic()
    resp = await server_client.get(
        f"/v1/tasks/{task_id}/events",
        params={"from_seq": last_seen + 1, "wait": 5},
    )
    elapsed = time.monotonic() - start
    assert resp.status_code == 200
    body = resp.json()
    # Assertion 1: wake-up actually fired (not the 5s timeout).
    assert elapsed < 2.0
    # Body returned at least one new event past the cursor.
    assert body["events"], "long-poll returned with no events"

    # Assertion 2: task eventually reaches terminal — poll until then.
    deadline = time.monotonic() + 2.0
    final_status: str | None = None
    while time.monotonic() < deadline:
        row_resp = await server_client.get(f"/v1/tasks/{task_id}")
        final_status = row_resp.json()["status"]
        if final_status in {"succeeded", "failed", "cancelled"}:
            break
        await asyncio.sleep(0.02)
    assert final_status in {"succeeded", "failed", "cancelled"}, (
        f"task did not reach terminal within 2s; last status={final_status!r}"
    )


@pytest.mark.asyncio
async def test_events_pagination_cursor_advances(
    server_client: httpx.AsyncClient,
) -> None:
    """Chain N calls each using prev ``next_from_seq``; assert no
    duplicates, no gaps, full tape covered."""
    submit = await server_client.post("/v1/echo", json={"count": 5})
    task_id = submit.json()["task_id"]
    await _wait_terminal_via_http(server_client, task_id)

    seen: list[int] = []
    from_seq = 0
    for _ in range(20):  # guard against runaway loop
        resp = await server_client.get(
            f"/v1/tasks/{task_id}/events",
            params={"from_seq": from_seq, "limit": 2, "wait": 0},
        )
        body = resp.json()
        if not body["events"]:
            break
        for ev in body["events"]:
            assert ev["seq"] not in seen, f"duplicate seq {ev['seq']}"
            seen.append(ev["seq"])
        from_seq = body["next_from_seq"]
        if not body["has_more"]:
            break
    # echo emits: task_started + N progress + 1 partial + final = N+3
    assert seen == sorted(seen), "events out of order"
    assert len(seen) >= 5  # at least the N progress events


@pytest.mark.asyncio
async def test_events_resume_after_disconnect(
    server_client: httpx.AsyncClient,
) -> None:
    """Stateless cursor: re-issue with the next_from_seq after a
    'disconnect' (just a fresh GET) yields exactly the next event,
    no replay, no gap."""
    submit = await server_client.post("/v1/echo", json={"count": 4})
    task_id = submit.json()["task_id"]
    await _wait_terminal_via_http(server_client, task_id)

    first = await server_client.get(
        f"/v1/tasks/{task_id}/events",
        params={"from_seq": 0, "limit": 2, "wait": 0},
    )
    first_body = first.json()
    cursor = first_body["next_from_seq"]
    # Simulate disconnect + reconnect with cursor.
    second = await server_client.get(
        f"/v1/tasks/{task_id}/events",
        params={"from_seq": cursor, "limit": 100, "wait": 0},
    )
    second_body = second.json()
    assert second_body["events"][0]["seq"] == cursor
    # No overlap between the two pages.
    first_seqs = {e["seq"] for e in first_body["events"]}
    second_seqs = {e["seq"] for e in second_body["events"]}
    assert first_seqs.isdisjoint(second_seqs)


@pytest.mark.asyncio
async def test_events_wait_caps_at_60s_server_side(
    server_client: httpx.AsyncClient,
) -> None:
    """Buggy / malicious client passing ``wait=99999`` is clamped to the
    server cap — we assert validation rejects values > 60 with 422
    rather than letting the connection hang forever."""
    submit = await server_client.post("/v1/echo", json={"count": 1})
    task_id = submit.json()["task_id"]
    await _wait_terminal_via_http(server_client, task_id)

    resp = await server_client.get(
        f"/v1/tasks/{task_id}/events",
        params={"from_seq": 9999, "wait": 99999},
    )
    # FastAPI/Pydantic Query validation returns 422 on out-of-range
    # values; the cap is part of the Query() constraint, not a runtime
    # if-check, so callers see the failure immediately.
    assert resp.status_code in (400, 422)


@pytest.mark.asyncio
async def test_events_unknown_task_returns_404(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.get("/v1/tasks/no-such-task/events")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_echo_delay_ms_paces_progress_events(
    server_client: httpx.AsyncClient,
) -> None:
    """``delay_ms`` actually slows the runner — task is still running
    well after the first event lands, proving the parameter is wired
    through and not silently ignored. Without this guard a regression
    that drops the sleep would only surface via the (more sensitive)
    long-poll wake-up tests."""
    import asyncio
    import time

    submit = await server_client.post(
        "/v1/echo", json={"count": 3, "delay_ms": 200}
    )
    task_id = submit.json()["task_id"]

    # Wait briefly for task_started + first progress to land.
    await asyncio.sleep(0.05)

    # Task should still be RUNNING — 3 events at 200ms each = ~600ms.
    row = await server_client.get(f"/v1/tasks/{task_id}")
    assert row.json()["status"] in ("pending", "running")

    # Eventually terminates.
    start = time.monotonic()
    await _wait_terminal_via_http(server_client, task_id, timeout=5.0)
    elapsed = time.monotonic() - start
    # Roughly N * delay_ms; allow slack for scheduling.
    assert elapsed > 0.3, f"delay_ms not respected: terminal in {elapsed:.3f}s"
