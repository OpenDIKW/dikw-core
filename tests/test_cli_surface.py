"""CLI command-tree surface guard.

Two checks the ``dikw --help`` exit-0 smoke in CI can't make:

1. The top level stays local-only (``version`` / ``init`` / ``serve`` / ``auth`` /
   ``client``). CLAUDE.md forbids top-level HTTP aliases — every HTTP-bound verb
   lives under ``dikw client <verb>``. A re-introduced alias fails here.
2. A committed golden snapshot of the whole command tree, so an accidental rename
   or removed subcommand is caught and forces a deliberate golden update + doc sync
   (cf. the repeated CLI-spelling-drift incidents the project has hit).

Regenerate the golden after an intentional CLI change:

    DIKW_REGEN_CLI_GOLDEN=1 uv run pytest tests/test_cli_surface.py
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import pytest
from typer.main import get_command

from dikw_core.cli import app

_GOLDEN = Path(__file__).parent / "cli_command_tree.golden.txt"


def _walk(cmd: click.Command, prefix: str) -> list[str]:
    lines: list[str] = []
    if isinstance(cmd, click.Group):
        for name in sorted(cmd.commands):
            sub = cmd.commands[name]
            path = f"{prefix} {name}"
            kind = "group" if isinstance(sub, click.Group) else "command"
            lines.append(f"{path} [{kind}]")
            lines.extend(_walk(sub, path))
    return lines


def render_command_tree() -> str:
    root = get_command(app)
    return "\n".join(_walk(root, "dikw")) + "\n"


def test_top_level_commands_are_local_only() -> None:
    root = get_command(app)
    assert isinstance(root, click.Group)
    top = set(root.commands)
    assert top == {"version", "init", "serve", "auth", "client"}, (
        "Top-level CLI must stay local-only (version/init/serve/auth/client). "
        "HTTP-bound verbs live under `dikw client <verb>` with no top-level "
        f"aliases (CLAUDE.md). Unexpected top-level set: {sorted(top)}"
    )


def test_cli_command_tree_matches_golden() -> None:
    actual = render_command_tree()
    if os.environ.get("DIKW_REGEN_CLI_GOLDEN"):
        _GOLDEN.write_text(actual, encoding="utf-8")
        pytest.skip("regenerated CLI command-tree golden")
    assert _GOLDEN.exists(), (
        "missing golden; generate with "
        "`DIKW_REGEN_CLI_GOLDEN=1 uv run pytest tests/test_cli_surface.py`"
    )
    assert _GOLDEN.read_text(encoding="utf-8") == actual, (
        "CLI command tree changed vs the committed golden. If intentional, "
        "regenerate with `DIKW_REGEN_CLI_GOLDEN=1 uv run pytest "
        "tests/test_cli_surface.py` and sync any docs referencing the command."
    )
