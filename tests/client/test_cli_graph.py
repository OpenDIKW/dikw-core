"""CLI tests for ``dikw client graph get``.

Drives the Typer ``CliRunner`` against the in-memory ASGI server: seed
a few docs on the runtime, invoke the CLI like the user would, and
assert that stdout carries a parseable JSON envelope shaped like
``GraphResult``. Agent-first default per CLAUDE.md memory: JSON to
stdout, no human-only flags muddying the contract.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from typer.testing import CliRunner

from dikw_core.cli import app
from dikw_core.schemas import Layer
from dikw_core.server.runtime import ServerRuntime

from ..fakes import seed_doc


def _seed_sync(
    rt: ServerRuntime, *, layer: Layer, path: str, body: str, title: str,
    active: bool = True,
) -> None:
    """``CliRunner`` test bodies can't ``await`` — wrap the async
    seed helper for use inside Typer CLI tests, mirroring
    :func:`tests.client.test_cli_assets._seed`."""
    asyncio.run(
        seed_doc(
            rt.root,
            layer=layer,
            path=path,
            body=body,
            title=title,
            active=active,
        )
    )


def test_graph_get_emits_json_envelope(
    patch_transport_factory: Callable[[], None],
    asgi_client: tuple[Any, ServerRuntime],
) -> None:
    """Default invocation prints a JSON envelope with the documented
    GraphResult keys — a downstream agent can pipe stdout straight into
    ``jq`` without any text-stripping."""
    patch_transport_factory()
    _, rt = asgi_client
    _seed_sync(rt, layer=Layer.WIKI, path="wiki/A.md", title="A", body="[[B]]\n")
    _seed_sync(rt, layer=Layer.WIKI, path="wiki/B.md", title="B", body="# B\n")

    result = CliRunner().invoke(app, ["client", "graph", "get"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {
        "base_revision",
        "generated_at",
        "nodes",
        "edges",
        "unresolved",
        "stats",
    } <= payload.keys()
    assert {n["path"] for n in payload["nodes"]} == {"wiki/A.md", "wiki/B.md"}
    assert payload["stats"]["edge_count"] == 1


def test_graph_get_active_flag_propagates_to_wire(
    patch_transport_factory: Callable[[], None],
    asgi_client: tuple[Any, ServerRuntime],
) -> None:
    """``--no-active`` (Typer's negation of a bool ``--active`` flag)
    drives the server's ``active=false`` filter — proves the CLI actually
    forwards the param rather than silently using its default."""
    patch_transport_factory()
    _, rt = asgi_client
    _seed_sync(rt, layer=Layer.WIKI, path="wiki/A.md", title="A", body="# A\n")
    _seed_sync(
        rt, layer=Layer.WIKI, path="wiki/B.md", title="B", body="# B\n", active=False,
    )

    # Default → only active.
    default = CliRunner().invoke(app, ["client", "graph", "get"])
    assert default.exit_code == 0, default.output
    assert {n["path"] for n in json.loads(default.output)["nodes"]} == {"wiki/A.md"}

    # --no-active → only deactivated (mirrors `?active=false` route semantics).
    inactive = CliRunner().invoke(app, ["client", "graph", "get", "--no-active"])
    assert inactive.exit_code == 0, inactive.output
    assert {n["path"] for n in json.loads(inactive.output)["nodes"]} == {"wiki/B.md"}
