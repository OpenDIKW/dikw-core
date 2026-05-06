"""``dikw auth *`` CLI surface — login / import / status / list / logout.

Drives the typer app through ``CliRunner``. Real HTTP / device-flow
behaviour is unit-tested elsewhere; here we just pin command names,
exit codes, and the high-level wiring (does ``import`` actually call
``import_from_codex_cli``? does ``login`` call ``device_code_login``?).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dikw_core.auth_cli import app
from dikw_core.providers.codex_auth import (
    AuthStatus,
    DeviceCodeChallenge,
    ImportResult,
    dikw_auth_path,
)

from .conftest import make_dikw_auth_store
from .fakes import make_jwt

runner = CliRunner()


def _fresh_jwt(account_id: str = "acc-cli") -> str:
    return make_jwt({"exp": int(time.time()) + 3600, "chatgpt_account_id": account_id})


# --------------------------------------------------------------------------- #
# login — wires the device flow + persists tokens
# --------------------------------------------------------------------------- #


def test_login_invokes_device_code_login_and_reports_account(
    dikw_base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_login(
        base: Path,
        *,
        on_challenge: Any | None = None,
        timeout_seconds: int = 0,
    ) -> ImportResult:
        del timeout_seconds
        captured["base"] = base
        if on_challenge is not None:
            on_challenge(
                DeviceCodeChallenge(
                    user_code="QWER-ASDF",
                    verification_uri="https://example.test/codex/device",
                    device_auth_id="d-1",
                    poll_interval_seconds=5,
                )
            )
        return ImportResult(
            source_path=Path("(device-code login)"),
            dest_path=dikw_auth_path(base),
            account_id="acc-77",
            expires_at=int(time.time()) + 3600,
        )

    monkeypatch.setattr("dikw_core.auth_cli.device_code_login", fake_login)
    monkeypatch.setattr("dikw_core.auth_cli.webbrowser.open", lambda *a, **kw: None)

    result = runner.invoke(
        app, ["login", "openai-codex", "--wiki", str(dikw_base), "--no-browser"]
    )
    assert result.exit_code == 0, result.output
    assert "QWER-ASDF" in result.output
    assert "acc-77" in result.output
    assert captured["base"] == dikw_base.resolve()


def test_login_rejects_unknown_provider(dikw_base: Path) -> None:
    result = runner.invoke(
        app, ["login", "anthropic", "--wiki", str(dikw_base)]
    )
    assert result.exit_code == 2
    assert "not supported" in result.output


# --------------------------------------------------------------------------- #
# import — wires import_from_codex_cli with the right flags
# --------------------------------------------------------------------------- #


def test_import_invokes_import_from_codex_cli_with_force_flag(
    dikw_base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_import(base: Path, *, force: bool = False) -> ImportResult:
        captured["base"] = base
        captured["force"] = force
        return ImportResult(
            source_path=Path("/fake/.codex/auth.json"),
            dest_path=dikw_auth_path(base),
            account_id="acc-import",
            expires_at=int(time.time()) + 1200,
        )

    monkeypatch.setattr("dikw_core.auth_cli.import_from_codex_cli", fake_import)

    result = runner.invoke(
        app, ["import", "openai-codex", "--wiki", str(dikw_base), "--force"]
    )
    assert result.exit_code == 0, result.output
    assert captured["force"] is True
    assert captured["base"] == dikw_base.resolve()
    assert "acc-import" in result.output


def test_import_defaults_provider_to_openai_codex(
    dikw_base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Today only one provider is supported, so the argument is optional."""
    monkeypatch.setattr(
        "dikw_core.auth_cli.import_from_codex_cli",
        lambda base, *, force=False: ImportResult(
            source_path=Path("/fake/.codex/auth.json"),
            dest_path=dikw_auth_path(base),
            account_id=None,
            expires_at=None,
        ),
    )
    result = runner.invoke(app, ["import", "--wiki", str(dikw_base)])
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# status — formatted snapshot
# --------------------------------------------------------------------------- #


def test_status_reports_active_when_token_fresh(
    dikw_base: Path,
) -> None:
    """End-to-end through the real ``auth_status`` — we only mock the
    file (via the dikw_base fixture)."""
    fresh = _fresh_jwt(account_id="acc-fresh")
    make_dikw_auth_store(dikw_base, access_token=fresh, refresh_token="rt-1")
    result = runner.invoke(
        app, ["status", "openai-codex", "--wiki", str(dikw_base)]
    )
    assert result.exit_code == 0, result.output
    assert "active" in result.output
    assert "acc-fresh" in result.output


def test_status_exits_nonzero_when_no_credentials(
    dikw_base: Path,
) -> None:
    """A user who never logged in should see a non-zero exit so shell
    pipelines can branch on it."""
    result = runner.invoke(
        app, ["status", "openai-codex", "--wiki", str(dikw_base)]
    )
    assert result.exit_code == 1
    assert "no credentials" in result.output.lower()
    assert "dikw auth login" in result.output


# --------------------------------------------------------------------------- #
# list — enumerates configured providers
# --------------------------------------------------------------------------- #


def test_list_shows_no_providers_when_store_empty(
    dikw_base: Path,
) -> None:
    result = runner.invoke(app, ["list", "--wiki", str(dikw_base)])
    assert result.exit_code == 0
    assert "no providers" in result.output.lower()


def test_list_shows_configured_provider(
    dikw_base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dikw_core.auth_cli.list_providers", lambda _: ["openai-codex"]
    )
    monkeypatch.setattr(
        "dikw_core.auth_cli.auth_status",
        lambda _, *, provider: AuthStatus(
            provider=provider,
            path=dikw_auth_path(_),
            exists=True,
            expires_in_seconds=1500,
            last_refresh="2026-05-06T03:14:22Z",
            auth_mode="chatgpt",
            account_id="acc-list",
        ),
    )
    result = runner.invoke(app, ["list", "--wiki", str(dikw_base)])
    assert result.exit_code == 0
    assert "openai-codex" in result.output
    assert "acc-list" in result.output


# --------------------------------------------------------------------------- #
# logout — confirmation + actual removal
# --------------------------------------------------------------------------- #


def test_logout_with_yes_flag_removes_provider(
    dikw_base: Path,
) -> None:
    make_dikw_auth_store(dikw_base, access_token=_fresh_jwt(), refresh_token="rt-1")
    result = runner.invoke(
        app,
        ["logout", "openai-codex", "--wiki", str(dikw_base), "--yes"],
    )
    assert result.exit_code == 0
    assert "removed" in result.output.lower()


def test_logout_without_yes_aborts_on_no_input(
    dikw_base: Path,
) -> None:
    """Default confirmation is ``no``; an empty stdin response aborts."""
    make_dikw_auth_store(dikw_base, access_token=_fresh_jwt(), refresh_token="rt-1")
    result = runner.invoke(
        app,
        ["logout", "openai-codex", "--wiki", str(dikw_base)],
        input="\n",
    )
    assert result.exit_code == 1
    assert "aborted" in result.output.lower()


def test_logout_when_no_credentials_present(
    dikw_base: Path,
) -> None:
    result = runner.invoke(
        app,
        ["logout", "openai-codex", "--wiki", str(dikw_base), "--yes"],
    )
    assert result.exit_code == 1
    assert "no credentials" in result.output.lower()


# --------------------------------------------------------------------------- #
# Top-level mount — `dikw auth ...` reaches the subapp
# --------------------------------------------------------------------------- #


def test_auth_app_top_level_help_lists_subcommands() -> None:
    """The auth Typer app advertises login / import / status / logout / list."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("login", "import", "status", "list", "logout"):
        assert cmd in result.output


def test_dikw_cli_root_includes_auth_subgroup() -> None:
    """The top-level dikw CLI must surface ``auth`` so muscle memory works."""
    from dikw_core.cli import app as root_app

    result = runner.invoke(root_app, ["auth", "--help"])
    assert result.exit_code == 0
    assert "login" in result.output
