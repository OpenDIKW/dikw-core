"""``dikw client wisdom write`` CLI tests.

End-to-end against the in-memory ASGI server: the CLI uses
``patch_transport_factory`` to route through the same ASGI client that
serves the wisdom write tasks, so the assertions cover the full
submit → events → result render path.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from dikw_core.cli import app
from dikw_core.server.runtime import ServerRuntime


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(app, args)


def test_wisdom_write_basic(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    client_base: Path,
) -> None:
    patch_transport_factory()

    r = _run([
        "client", "wisdom", "write",
        "--slug", "first-principles",
        "--title", "First Principles",
        "--body", "Reason from physics.\n",
        "--no-embed",
        "--plain",
    ])
    assert r.exit_code == 0, r.stdout
    # ``--wait`` is the default, so the report renders inline.
    assert "first-principles" in r.stdout
    assert "created" in r.stdout.lower()

    # File landed on disk.
    assert (client_base / "wisdom" / "first-principles.md").is_file()
    _ = asgi_client


def test_wisdom_write_with_author_and_status(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    client_base: Path,
) -> None:
    patch_transport_factory()

    r = _run([
        "client", "wisdom", "write",
        "--slug", "first-principles",
        "--author", "elon-musk",
        "--title", "First Principles",
        "--body", "body.\n",
        "--status", "draft",
        "--no-embed",
        "--plain",
    ])
    assert r.exit_code == 0, r.stdout

    abs_path = client_base / "wisdom" / "elon-musk" / "first-principles.md"
    assert abs_path.is_file()
    text = abs_path.read_text(encoding="utf-8")
    assert "status: draft" in text
    _ = asgi_client


def test_wisdom_write_body_from_file(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    client_base: Path,
    tmp_path: Path,
) -> None:
    patch_transport_factory()
    body_file = tmp_path / "body.md"
    body_file.write_text("body from file.\n", encoding="utf-8")

    r = _run([
        "client", "wisdom", "write",
        "--slug", "from-file",
        "--title", "From File",
        "--body-file", str(body_file),
        "--no-embed",
        "--plain",
    ])
    assert r.exit_code == 0, r.stdout
    written = (client_base / "wisdom" / "from-file.md").read_text(encoding="utf-8")
    assert "body from file." in written
    _ = asgi_client


def test_wisdom_write_no_wait_returns_task_handle(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    client_base: Path,
) -> None:
    patch_transport_factory()

    r = _run([
        "client", "wisdom", "write",
        "--slug", "async",
        "--title", "Async",
        "--body", "b.\n",
        "--no-embed",
        "--no-wait",
        "--plain",
    ])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert "task_id" in payload
    assert payload.get("status") in {"pending", "running", "succeeded"}
    _ = (asgi_client, client_base)


def test_wisdom_write_rejects_both_body_and_body_file(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    client_base: Path,
    tmp_path: Path,
) -> None:
    patch_transport_factory()
    body_file = tmp_path / "body.md"
    body_file.write_text("from file.\n", encoding="utf-8")

    r = _run([
        "client", "wisdom", "write",
        "--slug", "ambiguous",
        "--title", "Ambiguous",
        "--body", "inline.\n",
        "--body-file", str(body_file),
        "--no-embed",
        "--plain",
    ])
    # Typer's BadParameter exits with code 2 and writes to stderr,
    # which CliRunner doesn't always capture into ``stdout``; assert on
    # the exit code alone — the parameter check itself is verified by
    # the engine-layer kebab-case tests + the no-body test below.
    assert r.exit_code != 0
    _ = (asgi_client, client_base)


def test_wisdom_write_rejects_no_body_input(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    client_base: Path,
) -> None:
    patch_transport_factory()

    r = _run([
        "client", "wisdom", "write",
        "--slug", "empty",
        "--title", "Empty",
        "--no-embed",
        "--plain",
    ])
    assert r.exit_code != 0, r.stdout
    _ = (asgi_client, client_base)


def test_wisdom_write_tags_and_sources_repeatable(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    client_base: Path,
) -> None:
    patch_transport_factory()
    # Pre-create a referenced source so provenance edges are populated.
    src_dir = client_base / "sources" / "notes"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "x.md").write_text("# x\n", encoding="utf-8")

    r = _run([
        "client", "wisdom", "write",
        "--slug", "tagged",
        "--title", "Tagged",
        "--body", "b.\n",
        "--tag", "mental-model",
        "--tag", "physics",
        "--source", "sources/notes/x.md",
        "--no-embed",
        "--plain",
    ])
    assert r.exit_code == 0, r.stdout
    text = (client_base / "wisdom" / "tagged.md").read_text(encoding="utf-8")
    assert "mental-model" in text
    assert "physics" in text
    assert "sources/notes/x.md" in text
    _ = asgi_client


def test_wisdom_write_rejects_non_kebab_slug(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    client_base: Path,
) -> None:
    patch_transport_factory()

    r = _run([
        "client", "wisdom", "write",
        "--slug", "Bad Slug",
        "--title", "Bad",
        "--body", "b.\n",
        "--no-embed",
        "--plain",
    ])
    # Server returns 422 → CLI exits non-zero.
    assert r.exit_code != 0
    _ = (asgi_client, client_base)
