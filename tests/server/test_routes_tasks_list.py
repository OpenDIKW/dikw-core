"""HTTP contract for ``GET /v1/tasks`` after the 0.2.0 cursor refactor.

Three behaviours pinned here:

  * Response is a ``TaskListPage`` envelope (``{tasks, next_cursor,
    has_more}``), no longer a bare array.
  * Each row in ``tasks`` is the summary projection — ``result`` and
    ``error`` are not exposed; callers must use ``GET /v1/tasks/{id}/result``
    or ``GET /v1/tasks/{id}`` for full detail.
  * ``cursor`` query param round-trips across pages using a stable
    keyset over ``(created_at DESC, task_id ASC)``.
"""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest

from .conftest import wait_task_terminal as _wait_terminal


async def _submit_echo(client: httpx.AsyncClient, *, delay_ms: int = 0) -> str:
    r = await client.post("/v1/echo", json={"count": 1, "delay_ms": delay_ms})
    assert r.status_code == 200, r.text
    return str(r.json()["task_id"])


async def _submit_n_echo_terminal(
    client: httpx.AsyncClient, n: int
) -> list[str]:
    """Submit ``n`` echo tasks sequentially, wait each to terminal,
    return their ids in submission order (=oldest first)."""
    ids: list[str] = []
    for _ in range(n):
        tid = await _submit_echo(client)
        await _wait_terminal(client, tid)
        ids.append(tid)
        # Tiny sleep so the millisecond-resolution ``created_at`` differs
        # between submissions — keyset paging needs distinct keys to
        # round-trip deterministically in the assertion.
        await asyncio.sleep(0.005)
    return ids


# ---- envelope shape -----------------------------------------------------


@pytest.mark.asyncio
async def test_list_returns_envelope_shape(
    server_client: httpx.AsyncClient,
) -> None:
    """Response is `{tasks: [...], next_cursor: str|null, has_more: bool}`.
    The previous bare-array shape is no longer valid — 0.2.0 breaking
    change recorded in CHANGELOG."""
    ids = await _submit_n_echo_terminal(server_client, 3)

    r = await server_client.get("/v1/tasks")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, dict), f"want envelope dict, got {type(body)}"
    assert set(body.keys()) == {"tasks", "next_cursor", "has_more"}
    assert isinstance(body["tasks"], list)
    assert {t["task_id"] for t in body["tasks"]} >= set(ids)
    # Three submitted tasks fit comfortably under the default limit, so
    # the envelope must report no more pages waiting.
    assert body["has_more"] is False
    assert body["next_cursor"] is None


@pytest.mark.asyncio
async def test_list_omits_result_and_error(
    server_client: httpx.AsyncClient,
) -> None:
    """``result`` and ``error`` are never sent on the list endpoint —
    consumers must hit ``GET /v1/tasks/{id}/result`` or
    ``GET /v1/tasks/{id}`` for full detail. Echo's terminal ``result``
    payload is small but the rule is universal regardless of size."""
    tid = await _submit_echo(server_client)
    await _wait_terminal(server_client, tid)

    # Sanity: the single-task endpoint still surfaces ``result``.
    detail = (await server_client.get(f"/v1/tasks/{tid}/result")).json()
    assert detail["result"], "echo runner should stamp a result payload"

    # List view must NOT carry it.
    body = (await server_client.get("/v1/tasks")).json()
    row = next(t for t in body["tasks"] if t["task_id"] == tid)
    # The pydantic dump may still emit the key as null, but it must not
    # contain the actual payload — accept either "absent" or "null".
    assert row.get("result") in (None, {}, []), row
    assert row.get("error") in (None, {}, []), row


# ---- cursor pagination round-trip ---------------------------------------


@pytest.mark.asyncio
async def test_list_cursor_round_trips(
    server_client: httpx.AsyncClient,
) -> None:
    """12 tasks + ``limit=5`` walks in three pages (5 + 5 + 2). Each
    intermediate page exposes ``next_cursor`` and ``has_more=true``;
    the final page reports ``next_cursor=null, has_more=false``."""
    submitted = await _submit_n_echo_terminal(server_client, 12)

    p1 = (await server_client.get("/v1/tasks", params={"limit": 5})).json()
    assert len(p1["tasks"]) == 5
    assert p1["has_more"] is True
    assert isinstance(p1["next_cursor"], str) and p1["next_cursor"]

    p2 = (
        await server_client.get(
            "/v1/tasks", params={"limit": 5, "cursor": p1["next_cursor"]}
        )
    ).json()
    assert len(p2["tasks"]) == 5
    assert p2["has_more"] is True

    p3 = (
        await server_client.get(
            "/v1/tasks", params={"limit": 5, "cursor": p2["next_cursor"]}
        )
    ).json()
    assert len(p3["tasks"]) == 2
    assert p3["has_more"] is False
    assert p3["next_cursor"] is None

    # No row appears twice across the walk and the union equals what we
    # submitted (the fixture wiki starts empty, so there are no
    # background rows mixed in).
    walked = [t["task_id"] for page in (p1, p2, p3) for t in page["tasks"]]
    assert len(walked) == len(set(walked)) == 12
    assert set(walked) == set(submitted)


@pytest.mark.asyncio
async def test_list_cursor_handles_same_timestamp_ties(
    server_client: httpx.AsyncClient,
) -> None:
    """Submitting bursts without a sleep can collide at millisecond
    resolution. The keyset must still walk every row exactly once
    using ``task_id`` as the tie-breaker."""
    # Submit 6 tasks back-to-back without the per-iter sleep in
    # ``_submit_n_echo_terminal`` so several land in the same ms.
    ids: list[str] = []
    for _ in range(6):
        tid = await _submit_echo(server_client)
        ids.append(tid)
    for tid in ids:
        await _wait_terminal(server_client, tid)

    pages: list[dict[str, object]] = []
    cursor: str | None = None
    while True:
        params: dict[str, object] = {"limit": 2}
        if cursor is not None:
            params["cursor"] = cursor
        body = (await server_client.get("/v1/tasks", params=params)).json()
        pages.append(body)
        if not body["has_more"]:
            break
        cursor = str(body["next_cursor"])
        assert len(pages) < 10, "guard against runaway pagination loop"

    walked = [t["task_id"] for p in pages for t in p["tasks"]]
    assert len(walked) == len(set(walked)) == 6
    assert set(walked) == set(ids)


# ---- error handling -----------------------------------------------------


@pytest.mark.asyncio
async def test_list_invalid_cursor_returns_400(
    server_client: httpx.AsyncClient,
) -> None:
    """An opaque cursor that can't be decoded MUST surface as a 400 with
    a stable ``error.code`` — agents need to distinguish a malformed
    cursor (their bug) from server faults (transient)."""
    r = await server_client.get("/v1/tasks", params={"cursor": "not-base64!!"})
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"]["code"] == "invalid_cursor"


@pytest.mark.asyncio
async def test_list_cursor_with_valid_b64_wrong_payload_returns_400(
    server_client: httpx.AsyncClient,
) -> None:
    """Decoding succeeds but the payload doesn't carry the expected
    keyset fields — still ``invalid_cursor``, not a 500."""
    bogus = base64.urlsafe_b64encode(json.dumps({"foo": 1}).encode()).decode()
    r = await server_client.get("/v1/tasks", params={"cursor": bogus})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_cursor"


@pytest.mark.asyncio
async def test_list_cursor_with_non_ascii_returns_400(
    server_client: httpx.AsyncClient,
) -> None:
    """A cursor token carrying non-ASCII bytes (e.g. URL-decoded ``é``) must
    surface as ``400 invalid_cursor``, not crash into a 500.

    Such a token makes ``padded.encode("ascii")`` raise
    ``UnicodeEncodeError``, which is a ``ValueError`` subclass and so is
    caught by the decoder's ``except (binascii.Error, ValueError)``. This
    pins that invariant: narrowing the ``except`` to just ``binascii.Error``
    would let it escape as a 500."""
    r = await server_client.get("/v1/tasks", params={"cursor": "café"})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_cursor"


# ---- filter + cursor combination ----------------------------------------


@pytest.mark.asyncio
async def test_list_filter_with_cursor(
    server_client: httpx.AsyncClient,
) -> None:
    """``status=`` + ``cursor=`` compose: cursor positions only count
    inside the post-filter result set."""
    ids = await _submit_n_echo_terminal(server_client, 4)

    # All four echoes are succeeded by now. Walk them with status filter
    # + limit=2 + cursor and validate the union equals every id.
    p1 = (
        await server_client.get(
            "/v1/tasks", params={"limit": 2, "status": "succeeded"}
        )
    ).json()
    assert len(p1["tasks"]) == 2
    assert p1["has_more"] is True

    p2 = (
        await server_client.get(
            "/v1/tasks",
            params={
                "limit": 2,
                "status": "succeeded",
                "cursor": p1["next_cursor"],
            },
        )
    ).json()
    assert len(p2["tasks"]) == 2
    assert p2["has_more"] is False
    walked = [t["task_id"] for p in (p1, p2) for t in p["tasks"]]
    assert set(walked) == set(ids)
