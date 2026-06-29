"""Unit tests for the ``wait_event_tape_final`` conftest waiter.

The waiter de-flakes HTTP-level tape-tail asserts (issue #256). These tests
pin the three behaviours the integration callers exercise only by timing /
not at all, against a minimal fake client so each branch is deterministic:

  * surfaces a non-200 from ``/events`` immediately (a real 500 must not be
    masked as a 10s "never reached final" timeout);
  * pages through a multi-page tape via ``has_more`` / ``next_from_seq`` so the
    trailing ``final`` is found even past the 1000-event page cap;
  * keeps polling until the ``final`` envelope actually lands, then returns the
    full tape.
"""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest

from .conftest import wait_event_tape_final


class _FakeResp:
    def __init__(
        self, status_code: int, payload: dict[str, Any] | None = None, text: str = ""
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    """Serves a scripted list of responses in ``get`` call order.

    The waiter's call sequence is deterministic, so scripting responses by
    order is enough; an exhausted script clamps to the last response.
    """

    def __init__(self, responses: list[_FakeResp]) -> None:
        self._responses = responses
        self.calls = 0

    async def get(
        self, url: str, params: dict[str, Any] | None = None
    ) -> _FakeResp:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


def _ev(seq: int, type_: str, **extra: Any) -> dict[str, Any]:
    return {"seq": seq, "type": type_, **extra}


@pytest.mark.asyncio
async def test_surfaces_non_200_immediately() -> None:
    client = _FakeClient([_FakeResp(500, text="boom")])
    with pytest.raises(AssertionError, match="returned 500: boom"):
        await wait_event_tape_final(
            cast(httpx.AsyncClient, client), "t1", timeout=2.0
        )
    # Failed on the very first fetch — did not busy-poll to the deadline.
    assert client.calls == 1


@pytest.mark.asyncio
async def test_pages_past_one_page_to_find_final() -> None:
    page1 = _FakeResp(
        200,
        {
            "events": [_ev(1, "task_started"), _ev(2, "progress")],
            "has_more": True,
            "next_from_seq": 3,
        },
    )
    page2 = _FakeResp(
        200,
        {
            "events": [_ev(3, "progress"), _ev(4, "final", status="succeeded")],
            "has_more": False,
            "next_from_seq": 5,
        },
    )
    client = _FakeClient([page1, page2])
    events = await wait_event_tape_final(
        cast(httpx.AsyncClient, client), "t2", timeout=2.0
    )
    # Concatenated both pages and saw the trailing ``final``.
    assert [e["type"] for e in events] == [
        "task_started",
        "progress",
        "progress",
        "final",
    ]
    assert client.calls == 2


@pytest.mark.asyncio
async def test_waits_until_final_lands() -> None:
    not_yet = _FakeResp(
        200,
        {
            "events": [_ev(1, "task_started"), _ev(2, "progress")],
            "has_more": False,
            "next_from_seq": 3,
        },
    )
    done = _FakeResp(
        200,
        {
            "events": [
                _ev(1, "task_started"),
                _ev(2, "progress"),
                _ev(3, "final", status="succeeded"),
            ],
            "has_more": False,
            "next_from_seq": 4,
        },
    )
    client = _FakeClient([not_yet, done])
    events = await wait_event_tape_final(
        cast(httpx.AsyncClient, client), "t3", timeout=2.0
    )
    assert events[-1]["type"] == "final"
    # One poll saw only progress, the next saw final.
    assert client.calls == 2
