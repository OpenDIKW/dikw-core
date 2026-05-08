"""``dikw`` CLI.

Top-level commands fall into two groups:

* **Local** — ``version``, ``init``, ``serve``. These run entirely in
  this process; no server connection required. ``init`` scaffolds a
  fresh wiki on disk so you can run it before any server exists;
  ``serve`` starts the HTTP server itself.
* **Remote** (``dikw client *``) — every other operation talks to a
  running ``dikw serve`` instance over HTTP + NDJSON. The full surface
  lives under the ``client`` subcommand group, and the most common
  commands are also exposed as top-level aliases so existing muscle
  memory (``dikw status``, ``dikw query "…"``) keeps working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from . import __version__, api
from .auth_cli import app as auth_app
from .client.cli_app import app as client_app
from .logging import init_logging

app = typer.Typer(
    name="dikw",
    help="AI-native knowledge engine — Data · Information · Knowledge · Wisdom",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback()
def _root(ctx: typer.Context) -> None:
    """Configure logging from DIKW_LOG_LEVEL before any subcommand runs."""
    _ = ctx
    init_logging()


# ---- local-only commands ------------------------------------------------


@app.command("version")
def version_cmd() -> None:
    """Print the dikw-core package version."""
    console.print(__version__)


@app.command(
    "init",
    epilog=(
        "Examples:\n\n"
        "  dikw init\n\n"
        "  dikw init my-base -d \"my notes\""
    ),
)
def init_cmd(
    path: Annotated[
        Path,
        typer.Argument(
            help="Directory to scaffold the dikw base into. Created if it doesn't exist."
        ),
    ] = Path("."),
    description: Annotated[
        str,
        typer.Option("--description", "-d", help="One-line description for dikw.yml."),
    ] = "",
) -> None:
    """Scaffold a new dikw base at PATH (no server required)."""
    try:
        root = api.init_wiki(path, description=description or None)
    except FileExistsError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e
    console.print(f"[green]initialized[/green] dikw base at [bold]{root}[/bold]")
    console.print(
        "Next: add markdown under [cyan]sources/[/cyan], "
        "run [cyan]dikw serve --base .[/cyan] in another terminal, "
        "then [cyan]dikw status[/cyan]."
    )


@app.command(
    "serve",
    epilog=(
        "Examples:\n\n"
        "  dikw serve\n\n"
        "  dikw serve --base ./my-base\n\n"
        "  dikw serve --host 0.0.0.0 --token $DIKW_SERVER_TOKEN"
    ),
)
def serve_cmd(
    base: Annotated[
        Path,
        typer.Option(
            "--base",
            "-b",
            help="Path to the dikw base (must contain dikw.yml). Defaults to cwd.",
        ),
    ] = Path("."),
    host: Annotated[
        str,
        typer.Option(
            "--host",
            "-H",
            help="Interface to bind. 0.0.0.0 requires DIKW_SERVER_TOKEN to be set.",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="TCP port to bind."),
    ] = 8765,
    token: Annotated[
        str | None,
        typer.Option(
            "--token",
            help=(
                "Bearer token clients must present. Overrides "
                "DIKW_SERVER_TOKEN. Required when --host is non-loopback."
            ),
        ),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="uvicorn log level."),
    ] = "info",
) -> None:
    """Start the dikw HTTP server (FastAPI + NDJSON) on this host."""
    import uvicorn

    from .server.app import build_app_from_disk
    from .server.auth import (
        ensure_auth_invariant,
        load_auth_config,
    )

    base_root = base.resolve()
    auth_cfg = load_auth_config(host=host, token_override=token)
    try:
        ensure_auth_invariant(auth_cfg)
    except RuntimeError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=2) from e

    fastapi_app = build_app_from_disk(
        wiki_root=base_root,
        host=host,
        token_override=token,
    )
    posture = "token" if auth_cfg.required else "open (localhost only, no token)"
    console.print(
        f"[green]dikw serve[/green]  base=[cyan]{base_root}[/cyan]  "
        f"bind=[cyan]http://{host}:{port}[/cyan]  auth=[cyan]{posture}[/cyan]",
        highlight=False,
    )
    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        log_level=log_level,
    )


# ---- mount local auth subcommands --------------------------------------

# ``dikw auth login|import|status|list|logout`` — manage the local OAuth
# token store at ``<wiki>/.dikw/auth.json``. Local-only (does not talk
# to ``dikw serve``); must run on the same host as the server process.
app.add_typer(auth_app, name="auth")


# ---- mount remote CLI commands -----------------------------------------


# Full ``dikw client *`` surface.
app.add_typer(client_app, name="client")

# Top-level aliases for muscle memory: ``dikw status`` ≡ ``dikw client status``.
# We splice the client app's already-registered commands and subgroups onto
# the parent so the same callable surfaces under both prefixes without
# duplicating definitions. Names that already exist at the top level
# (``init``, currently — the local scaffold should always win because it
# runs before any server exists) are skipped here so Typer doesn't end up
# with two same-name entries.
_existing_command_names = {c.name for c in app.registered_commands}
_existing_group_names = {g.name for g in app.registered_groups}
for cmd in client_app.registered_commands:
    if cmd.name not in _existing_command_names:
        app.registered_commands.append(cmd)
for group in client_app.registered_groups:
    if group.name not in _existing_group_names:
        app.registered_groups.append(group)


def main() -> None:  # pragma: no cover - entry point shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
