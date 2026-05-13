"""CLI tests for ``dikw client assets get``.

Drives the Typer ``CliRunner`` against the in-memory ASGI server: seed
an asset on the runtime, invoke the CLI like the user would, then
assert the file lands on disk + stdout carries a parseable JSON
envelope.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from dikw_core.cli import app
from dikw_core.server.runtime import ServerRuntime

from ..fakes import png_with_dims, seed_asset


def _seed(rt: ServerRuntime) -> tuple[str, bytes]:
    """Sync wrapper around :func:`seed_asset` for use inside Typer
    ``CliRunner`` tests, which can't ``await`` in the test body."""
    payload = png_with_dims(1, 1)
    asset_id = hashlib.sha256(payload).hexdigest()
    rel = f"assets/{asset_id[:2]}/{asset_id[:8]}-x.png"
    asyncio.run(
        seed_asset(
            rt.root, asset_id=asset_id, stored_path=rel, payload=payload
        )
    )
    return asset_id, payload


def test_assets_get_writes_file_and_prints_json(
    patch_transport_factory: Callable[[], None],
    asgi_client: tuple[Any, ServerRuntime],
    tmp_path: Path,
) -> None:
    """Binary lands at ``--output``; stdout is a JSON envelope so a
    downstream agent can script the call without parsing free-form
    text."""
    patch_transport_factory()
    _, rt = asgi_client
    asset_id, payload = _seed(rt)
    out = tmp_path / "downloaded.png"

    result = CliRunner().invoke(
        app, ["client", "assets", "get", asset_id, "--output", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert out.read_bytes() == payload
    envelope = json.loads(result.output)
    assert envelope["asset_id"] == asset_id
    assert Path(envelope["path"]) == out
    assert envelope["bytes"] == len(payload)


def test_assets_get_404_exits_nonzero(
    patch_transport_factory: Callable[[], None],
    asgi_client: tuple[Any, ServerRuntime],
    tmp_path: Path,
) -> None:
    patch_transport_factory()
    _ = asgi_client
    out = tmp_path / "should-not-exist.png"
    result = CliRunner().invoke(
        app, ["client", "assets", "get", "0" * 64, "--output", str(out)]
    )

    assert result.exit_code != 0
    assert not out.exists()
