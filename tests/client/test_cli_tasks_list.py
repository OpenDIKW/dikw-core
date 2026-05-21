"""``dikw client tasks list`` 0.2.0 envelope + ``--all`` paging.

  * Default (single-page) ``--format json`` emits the server envelope
    ``{tasks, next_cursor, has_more}`` verbatim — agents that want to
    do their own paging see the cursor.
  * ``--all --format json`` drains the cursor and emits a flat
    ``[{...}, ...]`` array — matches the in-plan UX preview.
  * Table mode keeps the four-column display and still says
    "no tasks" on an empty page.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from typer.testing import CliRunner

from dikw_core.cli import app
from dikw_core.server.runtime import ServerRuntime

from ..fakes import FakeEmbeddings


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(app, args)


def _seed_n_tasks(n: int) -> list[str]:
    """Submit ``n`` ingest tasks (no-embed, instant scan) and return
    their task_ids. Each ``_run`` is its own CliRunner.invoke +
    asyncio.run loop — the server's manager handles the runner
    asynchronously, so the rows land in the task store regardless of
    whether the CLI loop sees them transition to terminal."""
    ids: list[str] = []
    for _ in range(n):
        r = _run(["client", "ingest", "--no-embed"])
        assert r.exit_code == 0, r.stdout
        handle = json.loads(r.stdout)
        ids.append(handle["task_id"])
    return ids


def test_tasks_list_default_outputs_envelope(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """``--format json`` default: single-page envelope dict, even when
    empty. Agents that follow the cursor pattern need ``next_cursor`` /
    ``has_more`` to be present in the response shape unconditionally."""
    patch_transport_factory()
    result = _run(["client", "tasks", "list", "--format", "json"])
    assert result.exit_code == 0, result.stdout
    body = json.loads(result.stdout)
    assert isinstance(body, dict), body
    assert set(body.keys()) == {"tasks", "next_cursor", "has_more"}
    assert body["tasks"] == []
    assert body["next_cursor"] is None
    assert body["has_more"] is False


def test_tasks_list_all_outputs_flat_array(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--all --format json`` drains the cursor and emits a JSON array
    of summary rows — matches the UX preview the user picked when
    approving the plan."""
    monkeypatch.setattr(
        "dikw_core.api.build_embedder", lambda _cfg: FakeEmbeddings()
    )
    patch_transport_factory()
    ids = _seed_n_tasks(3)

    result = _run(["client", "tasks", "list", "--all", "--format", "json"])
    assert result.exit_code == 0, result.stdout
    body = json.loads(result.stdout)
    assert isinstance(body, list), body
    walked = {row["task_id"] for row in body}
    assert walked >= set(ids), (walked, ids)


def test_tasks_list_all_paginates_when_page_smaller_than_total(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--all`` with ``--limit`` smaller than the row count must keep
    walking the cursor until ``has_more=false``. Guards the
    cursor-follow loop in ``_drain_task_list`` from a regression that
    only returns the first page."""
    monkeypatch.setattr(
        "dikw_core.api.build_embedder", lambda _cfg: FakeEmbeddings()
    )
    patch_transport_factory()
    ids = _seed_n_tasks(5)

    result = _run(
        ["client", "tasks", "list", "--all", "--limit", "2", "--format", "json"]
    )
    assert result.exit_code == 0, result.stdout
    body = json.loads(result.stdout)
    assert isinstance(body, list), body
    walked = {row["task_id"] for row in body}
    # Cursor walked past the first page of two — every seeded id surfaces.
    assert walked >= set(ids), (walked, ids)


def test_tasks_list_table_mode_empty_still_prints_no_tasks(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """Table mode (opt-in via ``--format table``) — the empty-state hint
    must survive the envelope refactor."""
    patch_transport_factory()
    result = _run(["client", "tasks", "list", "--format", "table"])
    assert result.exit_code == 0, result.stdout
    assert "no tasks" in result.stdout.lower()


@pytest.mark.asyncio
async def test_drain_task_list_raises_when_page_guard_exhausted() -> None:
    """The drain helper must NOT silently truncate.

    If the server keeps returning ``has_more=true`` past the 200-page
    safety guard (cursor bug, runaway dataset), continuing to swallow
    pages and report success would let ``--all`` / ``lint proposals``
    pretend the result set is complete when it isn't. Fail loud
    instead: raise a typed error so the CLI exit-code mapper can
    surface it to the user."""
    from dikw_core.client.cli_app import (
        _DRAIN_PAGE_GUARD,
        DrainPageGuardError,
        _drain_task_list,
    )

    class _StuckTransport:
        """get_json always returns ``has_more=true`` with a fresh cursor."""

        def __init__(self) -> None:
            self.calls = 0

        async def get_json(
            self, path: str, *, params: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            self.calls += 1
            return {
                "tasks": [{"task_id": f"t{self.calls}", "op": "echo", "status": "succeeded"}],
                "next_cursor": f"c{self.calls}",
                "has_more": True,
            }

    transport = _StuckTransport()
    with pytest.raises(DrainPageGuardError) as exc_info:
        await _drain_task_list(transport)  # type: ignore[arg-type]
    # The helper exhausted exactly the guard ceiling, didn't keep going.
    assert transport.calls == _DRAIN_PAGE_GUARD
    # The error must carry enough context for the user to resume manually.
    assert exc_info.value.pages == _DRAIN_PAGE_GUARD
    assert exc_info.value.last_cursor == f"c{_DRAIN_PAGE_GUARD}"


@pytest.mark.asyncio
async def test_drain_task_list_stops_on_non_dict_body() -> None:
    """A malformed (non-dict) page body ends the walk gracefully with
    whatever was collected so far, rather than crashing the loop."""
    from dikw_core.client.cli_app import _drain_task_list

    class _BadShape:
        def __init__(self) -> None:
            self.calls = 0

        async def get_json(
            self, path: str, *, params: dict[str, Any] | None = None
        ) -> Any:
            self.calls += 1
            return ["not", "a", "dict"]

    t = _BadShape()
    rows = await _drain_task_list(t)  # type: ignore[arg-type]
    assert rows == []
    assert t.calls == 1


@pytest.mark.asyncio
async def test_drain_task_list_stops_when_cursor_absent() -> None:
    """``has_more=true`` with no usable ``next_cursor`` can't be followed —
    return the rows collected so far instead of spinning the loop."""
    from dikw_core.client.cli_app import _drain_task_list

    class _NoCursor:
        async def get_json(
            self, path: str, *, params: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            return {"tasks": [{"task_id": "t1"}], "has_more": True}

    rows = await _drain_task_list(_NoCursor())  # type: ignore[arg-type]
    assert [r["task_id"] for r in rows] == ["t1"]


def test_run_maps_drain_page_guard_to_exit_1() -> None:
    """``_run`` renders the page-guard failure and exits 1 rather than
    leaking the raw traceback to the user."""
    import typer

    from dikw_core.client.cli_app import DrainPageGuardError, _run

    async def _boom() -> None:
        raise DrainPageGuardError(pages=200, rows_collected=3, last_cursor="c200")

    with pytest.raises(typer.Exit) as exc:
        _run(_boom())
    assert exc.value.exit_code == 1
