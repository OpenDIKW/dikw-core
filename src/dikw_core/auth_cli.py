"""``dikw auth *`` — local commands for the dikw OAuth credential store.

Tokens live at ``<base>/.dikw/auth.json``. Each base owns its own copy so
the codex CLI's rotating refresh_token in ``~/.codex/auth.json`` doesn't
collide with dikw's. Today only ``openai-codex`` is supported; the schema
under ``providers`` is multi-provider so adding anthropic-OAuth later is
just another match arm.

These commands are local (they don't talk to ``dikw serve``) and must run
on the same machine as the server process — both read the same on-disk
auth store.
"""

from __future__ import annotations

import contextlib
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .providers.codex_auth import (
    SUPPORTED_PROVIDERS,
    AuthStatus,
    CodexAuthError,
    DeviceCodeChallenge,
    auth_status,
    codex_home,
    device_code_login,
    dikw_auth_path,
    import_from_codex_cli,
    list_providers,
)
from .providers.codex_auth import (
    logout as logout_provider,
)

app = typer.Typer(
    name="auth",
    help="Manage OAuth credentials for LLM providers (token store: <base>/.dikw/auth.json).",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _resolve_base_root(base: Path) -> Path:
    """Resolve ``--base`` to an absolute path. Mirrors ``dikw serve`` —
    we don't require ``dikw.yml`` to exist so a brand-new user can run
    ``dikw auth login`` before ``dikw init`` if they prefer; the .dikw/
    directory just gets created on first save."""
    return base.resolve()


def _ensure_supported(provider: str) -> None:
    if provider not in SUPPORTED_PROVIDERS:
        console.print(
            f"[red]error:[/red] provider {provider!r} is not supported yet. "
            f"This build only handles: {', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )
        raise typer.Exit(code=2)


def _format_expires_in(seconds: int | None) -> str:
    if seconds is None:
        return "(unknown)"
    if seconds <= 0:
        return "[red]expired[/red]"
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_last_refresh(value: str | None) -> str:
    if not value:
        return "(none)"
    # Best-effort prettify; if we can't parse it, surface the raw string.
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _print_status_row(status: AuthStatus) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("provider")
    table.add_column("status")
    table.add_column("expires in")
    table.add_column("last refresh")
    table.add_column("account")
    if not status.exists:
        table.add_row(status.provider, "[yellow]not configured[/yellow]", "—", "—", "—")
    else:
        if status.expires_in_seconds is None:
            state = "[yellow]unknown[/yellow]"
        elif status.expiring_soon:
            state = "[yellow]refresh needed[/yellow]"
        else:
            state = "[green]active[/green]"
        table.add_row(
            status.provider,
            state,
            _format_expires_in(status.expires_in_seconds),
            _format_last_refresh(status.last_refresh),
            status.account_id or "(unknown)",
        )
    console.print(table)


# --------------------------------------------------------------------------- #
# login — full device-code OAuth flow
# --------------------------------------------------------------------------- #


@app.command("login")
def login_cmd(
    provider: Annotated[
        str, typer.Argument(help="Provider name. Today only `openai-codex` is supported.")
    ],
    base: Annotated[
        Path,
        typer.Option(
            "--base", "-b",
            help="Base root that owns the token store. Defaults to current directory.",
        ),
    ] = Path("."),
    no_browser: Annotated[
        bool,
        typer.Option(
            "--no-browser",
            help="Don't try to open the verification URL in a browser.",
        ),
    ] = False,
) -> None:
    """Authenticate via OpenAI's device-code OAuth flow."""
    _ensure_supported(provider)
    base_root = _resolve_base_root(base)

    def _on_challenge(challenge: DeviceCodeChallenge) -> None:
        console.print()
        console.print("[bold]To authorize dikw, follow these steps:[/bold]")
        console.print(f"  1. Open this URL in a browser: [cyan]{challenge.verification_uri}[/cyan]")
        console.print(f"  2. Enter this code: [bold cyan]{challenge.user_code}[/bold cyan]")
        console.print()
        console.print(
            f"Waiting for authorization... (poll every {challenge.poll_interval_seconds}s, "
            "press Ctrl+C to cancel)"
        )
        if not no_browser:
            with contextlib.suppress(Exception):
                # User just enters the code manually if open() throws —
                # not worth surfacing when the URL is already on screen.
                webbrowser.open(challenge.verification_uri, new=2)

    try:
        result = device_code_login(base_root, on_challenge=_on_challenge)
    except KeyboardInterrupt:
        console.print("\n[yellow]login cancelled[/yellow]")
        raise typer.Exit(code=130) from None
    except CodexAuthError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    console.print()
    console.print(
        f"[green]logged in[/green] as account "
        f"[bold]{result.account_id or '(unknown)'}[/bold]"
    )
    console.print(f"tokens written to [cyan]{result.dest_path}[/cyan]")


# --------------------------------------------------------------------------- #
# import — copy tokens from codex CLI's auth.json into the dikw store
# --------------------------------------------------------------------------- #


@app.command("import")
def import_cmd(
    provider: Annotated[
        str,
        typer.Argument(help="Provider name. Today only `openai-codex` is supported."),
    ] = "openai-codex",
    base: Annotated[
        Path,
        typer.Option(
            "--base", "-b",
            help="Base root that owns the token store. Defaults to current directory.",
        ),
    ] = Path("."),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help=(
                "Import even if the codex CLI access_token is already expired "
                "(a refresh attempt will run on next use)."
            ),
        ),
    ] = False,
) -> None:
    """Copy tokens from codex CLI's ``~/.codex/auth.json`` into the dikw store."""
    _ensure_supported(provider)
    base_root = _resolve_base_root(base)
    try:
        result = import_from_codex_cli(base_root, force=force)
    except CodexAuthError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e
    console.print(
        f"[green]imported[/green] codex tokens from [cyan]{result.source_path}[/cyan] "
        f"to [cyan]{result.dest_path}[/cyan]"
    )
    if result.account_id:
        console.print(f"account: [bold]{result.account_id}[/bold]")
    if result.expires_at is not None:
        remaining = max(0, int(result.expires_at - time.time()))
        console.print(f"access_token expires in {_format_expires_in(remaining)}")
    # The imported refresh_token is still shared with codex CLI until the
    # next rotation. Be honest about that — silent assumptions here are how
    # users get unexpectedly logged out of one side.
    console.print(
        "[yellow]note:[/yellow] the imported refresh_token is shared with codex CLI "
        "until either side refreshes; the rotation will invalidate the other copy. "
        "Run [cyan]dikw auth login openai-codex[/cyan] for fully independent credentials."
    )


# --------------------------------------------------------------------------- #
# status — print one provider's snapshot
# --------------------------------------------------------------------------- #


@app.command("status")
def status_cmd(
    provider: Annotated[
        str,
        typer.Argument(help="Provider name."),
    ] = "openai-codex",
    base: Annotated[
        Path,
        typer.Option(
            "--base", "-b",
            help="Base root that owns the token store. Defaults to current directory.",
        ),
    ] = Path("."),
) -> None:
    """Show the dikw auth store entry for a provider."""
    _ensure_supported(provider)
    base_root = _resolve_base_root(base)
    status = auth_status(base_root, provider=provider)
    console.print(f"store: [cyan]{status.path}[/cyan]")
    if not status.exists:
        console.print(
            f"[yellow]no credentials for {provider!r}.[/yellow] "
            f"Run [cyan]dikw auth login {provider}[/cyan] to authenticate, "
            f"or [cyan]dikw auth import {provider}[/cyan] to import "
            f"from codex CLI ({codex_home() / 'auth.json'})."
        )
        raise typer.Exit(code=1)
    _print_status_row(status)


# --------------------------------------------------------------------------- #
# list — enumerate providers in the store
# --------------------------------------------------------------------------- #


@app.command("list")
def list_cmd(
    base: Annotated[
        Path,
        typer.Option(
            "--base", "-b",
            help="Base root that owns the token store. Defaults to current directory.",
        ),
    ] = Path("."),
) -> None:
    """List providers with credentials in the dikw auth store."""
    base_root = _resolve_base_root(base)
    path = dikw_auth_path(base_root)
    providers = list_providers(base_root)
    if not providers:
        console.print(f"store: [cyan]{path}[/cyan]")
        console.print("[yellow]no providers configured.[/yellow]")
        return
    console.print(f"store: [cyan]{path}[/cyan]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("provider")
    table.add_column("status")
    table.add_column("expires in")
    table.add_column("account")
    for name in providers:
        status = auth_status(base_root, provider=name)
        if status.expires_in_seconds is None:
            state = "[yellow]unknown[/yellow]"
        elif status.expiring_soon:
            state = "[yellow]refresh needed[/yellow]"
        else:
            state = "[green]active[/green]"
        table.add_row(
            name,
            state,
            _format_expires_in(status.expires_in_seconds),
            status.account_id or "(unknown)",
        )
    console.print(table)


# --------------------------------------------------------------------------- #
# logout — drop one provider's tokens
# --------------------------------------------------------------------------- #


@app.command("logout")
def logout_cmd(
    provider: Annotated[
        str,
        typer.Argument(help="Provider name."),
    ],
    base: Annotated[
        Path,
        typer.Option(
            "--base", "-b",
            help="Base root that owns the token store. Defaults to current directory.",
        ),
    ] = Path("."),
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt."),
    ] = False,
) -> None:
    """Remove a provider's tokens from the dikw auth store."""
    _ensure_supported(provider)
    base_root = _resolve_base_root(base)
    if not yes:
        confirm = typer.confirm(
            f"Remove tokens for {provider!r} from {dikw_auth_path(base_root)}?",
            default=False,
        )
        if not confirm:
            console.print("[yellow]aborted[/yellow]")
            raise typer.Exit(code=1)
    removed = logout_provider(base_root, provider=provider)
    if removed:
        console.print(f"[green]removed[/green] {provider!r} from the dikw auth store")
    else:
        console.print(f"[yellow]no credentials for {provider!r} were present[/yellow]")
        raise typer.Exit(code=1)
