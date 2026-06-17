"""``dikw client delete <path>`` CLI tests.

End-to-end against the in-memory ASGI server via ``patch_transport_factory``:
each test creates a page through the CLI (``wisdom write``), then deletes
it through the CLI and asserts the file moved to ``trash/`` and the report
rendered. Mirrors ``tests/client/test_cli_wisdom_write.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import frontmatter
from typer.testing import CliRunner

from dikw_core.cli import app
from dikw_core.server.runtime import ServerRuntime


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(app, args)


def _seed_via_cli(slug: str) -> Any:
    return _run([
        "client", "wisdom", "write",
        "--slug", slug,
        "--title", slug.title(),
        "--body", "scratch body.\n",
        "--no-embed",
        "--plain",
    ])


def test_delete_basic(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    client_base: Path,
) -> None:
    patch_transport_factory()
    assert _seed_via_cli("scratch").exit_code == 0
    assert (client_base / "wisdom" / "scratch.md").is_file()

    r = _run(["client", "delete", "wisdom/scratch.md", "--plain"])
    assert r.exit_code == 0, r.stdout
    # ``--wait`` is the default, so the report renders inline.
    assert "wisdom/scratch.md" in r.stdout
    assert "deleted" in r.stdout.lower()

    # File moved out of wisdom/ and into trash/.
    assert not (client_base / "wisdom" / "scratch.md").exists()
    assert (client_base / "trash" / "wisdom" / "scratch.md").is_file()
    _ = asgi_client


def test_delete_unknown_path_exits_nonzero(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    client_base: Path,
) -> None:
    patch_transport_factory()
    r = _run(["client", "delete", "knowledge/never-existed.md", "--plain"])
    # The task fails (PageNotFound) → --wait maps failed → exit 1.
    assert r.exit_code != 0
    _ = (asgi_client, client_base)


def test_delete_no_wait_returns_task_handle(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    client_base: Path,
) -> None:
    patch_transport_factory()
    assert _seed_via_cli("async-del").exit_code == 0

    r = _run(["client", "delete", "wisdom/async-del.md", "--no-wait", "--plain"])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert "task_id" in payload
    assert payload.get("status") in {"pending", "running", "succeeded"}
    _ = (asgi_client, client_base)


def test_delete_reason_stamped_in_trash(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    client_base: Path,
) -> None:
    patch_transport_factory()
    assert _seed_via_cli("dup").exit_code == 0

    r = _run(["client", "delete", "wisdom/dup.md", "--reason", "obsolete", "--plain"])
    assert r.exit_code == 0, r.stdout

    trashed = frontmatter.loads(
        (client_base / "trash" / "wisdom" / "dup.md").read_text(encoding="utf-8")
    ).metadata.get("trashed")
    assert isinstance(trashed, dict)
    assert trashed.get("reason") == "obsolete"
    _ = asgi_client
