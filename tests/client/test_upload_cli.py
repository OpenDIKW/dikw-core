"""End-to-end CLI tests for ``dikw client upload <path>``.

These tests drive the full Typer command surface against the in-memory
ASGI server fixture: client packs a local input dir, ships it via
multipart upload, server validates per-package + commits to
``<base>/sources/``. Both happy paths and pre-flight rejection paths
are covered.

Pre-flight rejection (frontmatter_error / asset_missing / empty_body)
exits 2 — Unix convention for "user supplied bad input." Server-side
per-package rejection still exits 0 because the upload itself
succeeded; the rejected list is rendered for the user to retry.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from dikw_core.cli import app
from dikw_core.server.runtime import ServerRuntime


def _run(args: list[str]) -> Any:
    # mix_stderr default differs across click/typer versions; force a
    # combined buffer so assertions can grep the whole output.
    return CliRunner().invoke(app, args)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


# ---- happy paths -------------------------------------------------------


def test_upload_single_md_file(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    tmp_path: Path,
) -> None:
    """``dikw client upload <one.md>`` — single file path, no assets."""
    patch_transport_factory()
    note = tmp_path / "alpha.md"
    note.write_text("# Alpha\nbody\n", encoding="utf-8")

    result = _run(["client", "upload", str(note)])
    assert result.exit_code == 0, result.stdout

    _, rt = asgi_client
    assert (rt.root / "sources" / "alpha.md").read_text(
        encoding="utf-8"
    ) == "# Alpha\nbody\n"


def test_upload_directory_recursive(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    tmp_path: Path,
) -> None:
    """``dikw client upload <dir>`` — every ``**/*.md`` becomes a package."""
    patch_transport_factory()
    src = tmp_path / "inbox"
    _write(src / "a.md", "# A\nbody\n")
    _write(src / "sub" / "b.md", "# B\nbody\n")

    result = _run(["client", "upload", str(src)])
    assert result.exit_code == 0, result.stdout

    _, rt = asgi_client
    assert (rt.root / "sources" / "a.md").exists()
    assert (rt.root / "sources" / "sub" / "b.md").exists()


def test_upload_md_with_sibling_asset(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    tmp_path: Path,
) -> None:
    """A note that embeds ``diagram.png`` ships both in one package;
    server commits both to ``sources/`` preserving the sibling layout."""
    patch_transport_factory()
    src = tmp_path / "inbox"
    _write(src / "note.md", "# n\n![](diagram.png)\n")
    (src / "diagram.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    result = _run(["client", "upload", str(src)])
    assert result.exit_code == 0, result.stdout

    _, rt = asgi_client
    assert (rt.root / "sources" / "note.md").exists()
    assert (rt.root / "sources" / "diagram.png").exists()


def test_upload_cross_directory_asset(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    tmp_path: Path,
) -> None:
    """``![](../shared/logo.png)`` resolves via project_root fallback;
    archive path puts the asset at ``sources/shared/logo.png`` and
    server commits it preserving that layout."""
    patch_transport_factory()
    src = tmp_path / "inbox"
    _write(src / "sub" / "note.md", "# n\n![](../shared/logo.png)\n")
    (src / "shared").mkdir(parents=True)
    (src / "shared" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    result = _run(["client", "upload", str(src)])
    assert result.exit_code == 0, result.stdout

    _, rt = asgi_client
    assert (rt.root / "sources" / "sub" / "note.md").exists()
    assert (rt.root / "sources" / "shared" / "logo.png").exists()


def test_top_level_alias_dikw_upload(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    tmp_path: Path,
) -> None:
    """``dikw upload`` (no ``client`` prefix) must resolve via the splice
    in ``cli.py`` — same machinery as ``dikw status``, ``dikw query``."""
    patch_transport_factory()
    note = tmp_path / "alpha.md"
    note.write_text("# A\nbody\n", encoding="utf-8")

    result = _run(["upload", str(note)])
    assert result.exit_code == 0, result.stdout


# ---- pre-flight rejection (exit 2) -------------------------------------


def test_upload_pre_flight_frontmatter_error(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    tmp_path: Path,
) -> None:
    """Malformed YAML between ``---`` fences is caught client-side; no
    upload request leaves the machine."""
    patch_transport_factory()
    note = tmp_path / "bad.md"
    note.write_text(
        "---\nfoo: : bar\n---\n# x\nbody\n", encoding="utf-8"
    )

    result = _run(["client", "upload", str(note)])
    assert result.exit_code == 2, result.stdout
    assert "frontmatter" in result.stdout.lower()

    _, rt = asgi_client
    assert not (rt.root / "sources" / "bad.md").exists()


def test_upload_pre_flight_asset_missing(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    tmp_path: Path,
) -> None:
    """Reference to a nonexistent file is rejected before packaging.
    Error output names the missing file so the user can fix it."""
    patch_transport_factory()
    note = tmp_path / "n.md"
    note.write_text("# x\n![](ghost.png)\nbody\n", encoding="utf-8")

    result = _run(["client", "upload", str(note)])
    assert result.exit_code == 2, result.stdout
    assert "ghost.png" in result.stdout

    _, rt = asgi_client
    assert not (rt.root / "sources" / "n.md").exists()


def test_upload_pre_flight_empty_body(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    tmp_path: Path,
) -> None:
    """Front-matter-only file (no real content) gets caught."""
    patch_transport_factory()
    note = tmp_path / "empty.md"
    note.write_text("---\ntitle: empty\n---\n   \n", encoding="utf-8")

    result = _run(["client", "upload", str(note)])
    assert result.exit_code == 2, result.stdout
    assert "empty" in result.stdout.lower()


def test_upload_orphan_asset_rejected(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    tmp_path: Path,
) -> None:
    """A png in the input dir not referenced by any md is an orphan;
    pre-flight rejects rather than silently dropping it."""
    patch_transport_factory()
    src = tmp_path / "inbox"
    _write(src / "note.md", "# n\nbody\n")  # no asset reference
    (src / "stray.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    result = _run(["client", "upload", str(src)])
    assert result.exit_code == 2, result.stdout
    assert "orphan" in result.stdout.lower() or "stray.png" in result.stdout


def test_upload_aggregates_multiple_pre_flight_issues(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    tmp_path: Path,
) -> None:
    """Two bad files in one dir → both reported, not just the first."""
    patch_transport_factory()
    src = tmp_path / "inbox"
    _write(src / "a.md", "# A\n![](missing-a.png)\n")
    _write(src / "b.md", "# B\n![](missing-b.png)\n")

    result = _run(["client", "upload", str(src)])
    assert result.exit_code == 2, result.stdout
    assert "missing-a.png" in result.stdout
    assert "missing-b.png" in result.stdout


# ---- bad CLI args ------------------------------------------------------


def test_upload_path_argument_required(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["client", "upload"])
    assert result.exit_code != 0


def test_upload_path_must_exist(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    tmp_path: Path,
) -> None:
    patch_transport_factory()
    bogus = tmp_path / "does-not-exist"
    result = _run(["client", "upload", str(bogus)])
    assert result.exit_code != 0
    assert (
        "not exist" in result.stdout.lower()
        or "no such" in result.stdout.lower()
        or "does not" in result.stdout.lower()
    )
