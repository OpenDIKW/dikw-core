"""Lazy migration + ``import_from_codex_cli`` + status / list / logout.

Covers the boundary where dikw used to write ``~/.codex/auth.json`` and
now writes ``<base>/.dikw/auth.json``:

* The transparent first-run import on existing installs.
* The explicit ``dikw auth import openai-codex`` path (success, missing,
  expired, force-import).
* Multi-provider preservation: future anthropic OAuth entries shouldn't
  be clobbered when codex tokens change.
* ``auth_status``, ``list_providers``, ``logout`` round-trips.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from dikw_core.providers.codex_auth import (
    CodexAuthError,
    auth_status,
    codex_home,
    dikw_auth_path,
    import_from_codex_cli,
    list_providers,
    logout,
    read_codex_tokens,
    resolve_access_token,
)

from .conftest import make_codex_cli_auth_store, make_dikw_auth_store
from .fakes import make_jwt


def _fresh_jwt(*, account_id: str = "acc-1") -> str:
    return make_jwt(
        {"exp": int(time.time()) + 3600, "chatgpt_account_id": account_id}
    )


def _expired_jwt() -> str:
    return make_jwt({"exp": int(time.time()) - 3600})


# --------------------------------------------------------------------------- #
# Lazy migration — automatic on first ``resolve_access_token`` call
# --------------------------------------------------------------------------- #


async def test_lazy_migration_imports_codex_cli_tokens_on_first_use(
    dikw_base: Path,
    _isolated_codex_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the dikw store is empty but codex CLI has fresh tokens, the
    first ``resolve_access_token`` call transparently copies them over
    so existing users keep working through the upgrade boundary."""
    fresh = _fresh_jwt()
    make_codex_cli_auth_store(
        _isolated_codex_home, access_token=fresh, refresh_token="rt-1"
    )

    token = await resolve_access_token(dikw_base)
    assert token == fresh
    # Tokens are now in the dikw store, not the codex CLI store.
    on_disk = json.loads(dikw_auth_path(dikw_base).read_text(encoding="utf-8"))
    assert on_disk["providers"]["openai-codex"]["tokens"]["access_token"] == fresh

    # User-facing migration message lands on stderr exactly once.
    captured = capsys.readouterr()
    assert "Imported codex tokens" in captured.err
    assert str(_isolated_codex_home / "auth.json") in captured.err


async def test_lazy_migration_does_not_overwrite_existing_dikw_store(
    dikw_base: Path,
    _isolated_codex_home: Path,
) -> None:
    """If the dikw store already has tokens (even just an empty file),
    don't second-guess it by re-importing from codex CLI — the user
    already chose their source."""
    fresh_dikw = _fresh_jwt(account_id="acc-dikw")
    fresh_cli = _fresh_jwt(account_id="acc-cli")
    make_dikw_auth_store(
        dikw_base, access_token=fresh_dikw, refresh_token="rt-dikw"
    )
    make_codex_cli_auth_store(
        _isolated_codex_home, access_token=fresh_cli, refresh_token="rt-cli"
    )

    token = await resolve_access_token(dikw_base)
    assert token == fresh_dikw  # dikw value wins, codex CLI was not read.


async def test_lazy_migration_skipped_when_codex_cli_tokens_expired(
    dikw_base: Path,
    _isolated_codex_home: Path,
) -> None:
    """An expired codex CLI access_token is NOT auto-imported — we want
    the user to see the missing-credential error and run an explicit
    ``dikw auth import --force`` or fresh login."""
    make_codex_cli_auth_store(
        _isolated_codex_home,
        access_token=_expired_jwt(),
        refresh_token="rt-stale",
    )
    with pytest.raises(CodexAuthError) as excinfo:
        await resolve_access_token(dikw_base)
    assert excinfo.value.code == "codex_auth_missing"


async def test_lazy_migration_skipped_when_no_codex_cli_tokens(
    dikw_base: Path,
) -> None:
    """No dikw store and no codex CLI tokens — surface the standard
    missing-credential error pointing at the bootstrap commands."""
    with pytest.raises(CodexAuthError) as excinfo:
        await resolve_access_token(dikw_base)
    assert excinfo.value.code == "codex_auth_missing"


async def test_lazy_migration_does_not_replay_on_subsequent_calls(
    dikw_base: Path,
    _isolated_codex_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Import message fires once, never again — after the first import the
    dikw store exists, so the migration check short-circuits."""
    fresh = _fresh_jwt()
    make_codex_cli_auth_store(
        _isolated_codex_home, access_token=fresh, refresh_token="rt-1"
    )
    await resolve_access_token(dikw_base)
    capsys.readouterr()  # drain the first warning

    await resolve_access_token(dikw_base)
    captured = capsys.readouterr()
    assert "Imported codex tokens" not in captured.err


# --------------------------------------------------------------------------- #
# import_from_codex_cli — explicit CLI entrypoint
# --------------------------------------------------------------------------- #


def test_import_from_codex_cli_round_trip(
    dikw_base: Path, _isolated_codex_home: Path
) -> None:
    fresh = _fresh_jwt(account_id="acc-77")
    make_codex_cli_auth_store(
        _isolated_codex_home, access_token=fresh, refresh_token="rt-1"
    )
    result = import_from_codex_cli(dikw_base)
    assert result.account_id == "acc-77"
    assert result.dest_path == dikw_auth_path(dikw_base)
    assert result.source_path == _isolated_codex_home / "auth.json"
    assert read_codex_tokens(dikw_base) == {
        "access_token": fresh,
        "refresh_token": "rt-1",
    }


def test_import_from_codex_cli_missing_source_raises(dikw_base: Path) -> None:
    """No ``codex_home()/auth.json`` → tell the user how to log in."""
    with pytest.raises(CodexAuthError) as excinfo:
        import_from_codex_cli(dikw_base)
    assert excinfo.value.code == "codex_cli_auth_missing"
    assert "dikw auth login" in str(excinfo.value)


def test_import_from_codex_cli_rejects_expired_without_force(
    dikw_base: Path, _isolated_codex_home: Path
) -> None:
    make_codex_cli_auth_store(
        _isolated_codex_home,
        access_token=_expired_jwt(),
        refresh_token="rt-stale",
    )
    with pytest.raises(CodexAuthError) as excinfo:
        import_from_codex_cli(dikw_base, force=False)
    assert excinfo.value.code == "codex_cli_auth_expired"


def test_import_from_codex_cli_force_accepts_expired_access_token(
    dikw_base: Path, _isolated_codex_home: Path
) -> None:
    """``--force`` is the escape hatch for the case where access_token is
    expired but refresh_token is still good — refresh runs on next use."""
    expired = _expired_jwt()
    make_codex_cli_auth_store(
        _isolated_codex_home, access_token=expired, refresh_token="rt-still-good"
    )
    result = import_from_codex_cli(dikw_base, force=True)
    assert result.dest_path.is_file()
    on_disk = read_codex_tokens(dikw_base)
    assert on_disk["refresh_token"] == "rt-still-good"


def test_import_from_codex_cli_preserves_other_provider_entries(
    dikw_base: Path, _isolated_codex_home: Path
) -> None:
    """Multi-provider read-modify-write: an existing anthropic entry must
    survive an import that only touches openai-codex."""
    make_dikw_auth_store(
        dikw_base,
        access_token="placeholder-at",
        refresh_token="placeholder-rt",
        extra_providers={
            "anthropic": {
                "tokens": {"access_token": "anth-at", "refresh_token": "anth-rt"},
                "last_refresh": "2026-04-01T00:00:00Z",
                "auth_mode": "claude",
            }
        },
    )
    fresh = _fresh_jwt(account_id="acc-x")
    make_codex_cli_auth_store(
        _isolated_codex_home, access_token=fresh, refresh_token="rt-x"
    )
    import_from_codex_cli(dikw_base)
    on_disk = json.loads(dikw_auth_path(dikw_base).read_text(encoding="utf-8"))
    assert on_disk["providers"]["openai-codex"]["tokens"]["access_token"] == fresh
    assert on_disk["providers"]["anthropic"]["tokens"]["access_token"] == "anth-at"


def test_import_from_codex_cli_resolves_codex_home_via_env(
    dikw_base: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``$CODEX_HOME`` is the documented override; importer respects it."""
    custom_home = tmp_path / "custom" / "codex"
    custom_home.mkdir(parents=True)
    monkeypatch.setenv("CODEX_HOME", str(custom_home))
    assert codex_home() == custom_home
    fresh = _fresh_jwt()
    make_codex_cli_auth_store(
        custom_home, access_token=fresh, refresh_token="rt-from-env"
    )
    import_from_codex_cli(dikw_base)
    assert read_codex_tokens(dikw_base)["refresh_token"] == "rt-from-env"


# --------------------------------------------------------------------------- #
# auth_status / list_providers / logout
# --------------------------------------------------------------------------- #


def test_auth_status_reports_missing_when_no_store(dikw_base: Path) -> None:
    status = auth_status(dikw_base)
    assert status.exists is False
    assert status.expires_in_seconds is None
    assert status.account_id is None


def test_auth_status_reports_active_for_fresh_jwt(dikw_base: Path) -> None:
    fresh = _fresh_jwt(account_id="acc-status")
    make_dikw_auth_store(dikw_base, access_token=fresh, refresh_token="rt-1")
    status = auth_status(dikw_base)
    assert status.exists is True
    assert status.expiring_soon is False
    assert status.account_id == "acc-status"
    # Within ~1s tolerance of the 3600s fixture window.
    assert status.expires_in_seconds is not None
    assert 3500 < status.expires_in_seconds <= 3600


def test_auth_status_marks_expiring_soon_for_short_lived(
    dikw_base: Path,
) -> None:
    short = make_jwt({"exp": int(time.time()) + 30})
    make_dikw_auth_store(dikw_base, access_token=short, refresh_token="rt-1")
    status = auth_status(dikw_base)
    assert status.expiring_soon is True


def test_auth_status_handles_unknown_exp(dikw_base: Path) -> None:
    """A token without an ``exp`` claim → status reports unknown but not crashing."""
    no_exp = make_jwt({"chatgpt_account_id": "acc-?"})
    make_dikw_auth_store(dikw_base, access_token=no_exp, refresh_token="rt-1")
    status = auth_status(dikw_base)
    assert status.expires_in_seconds is None
    assert status.expiring_soon is True


def test_list_providers_returns_sorted_provider_names(dikw_base: Path) -> None:
    make_dikw_auth_store(
        dikw_base,
        access_token=_fresh_jwt(),
        refresh_token="rt-1",
        extra_providers={
            "anthropic": {
                "tokens": {"access_token": "a", "refresh_token": "b"},
                "last_refresh": None,
                "auth_mode": "claude",
            }
        },
    )
    assert list_providers(dikw_base) == ["anthropic", "openai-codex"]


def test_list_providers_skips_provider_with_empty_tokens(
    dikw_base: Path,
) -> None:
    """Don't surface a provider that's been logged out (has the entry but
    no tokens) as if it were configured."""
    auth_path = dikw_auth_path(dikw_base)
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "openai-codex": {
                        "tokens": {"access_token": "", "refresh_token": ""},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert list_providers(dikw_base) == []


def test_logout_removes_provider_node(dikw_base: Path) -> None:
    make_dikw_auth_store(
        dikw_base, access_token=_fresh_jwt(), refresh_token="rt-1"
    )
    assert logout(dikw_base) is True
    on_disk = json.loads(dikw_auth_path(dikw_base).read_text(encoding="utf-8"))
    assert "openai-codex" not in on_disk["providers"]


def test_logout_preserves_other_providers(dikw_base: Path) -> None:
    make_dikw_auth_store(
        dikw_base,
        access_token=_fresh_jwt(),
        refresh_token="rt-1",
        extra_providers={
            "anthropic": {
                "tokens": {"access_token": "a", "refresh_token": "b"},
                "last_refresh": None,
                "auth_mode": "claude",
            }
        },
    )
    assert logout(dikw_base) is True
    on_disk = json.loads(dikw_auth_path(dikw_base).read_text(encoding="utf-8"))
    assert "anthropic" in on_disk["providers"]
    assert "openai-codex" not in on_disk["providers"]


def test_logout_returns_false_when_provider_absent(dikw_base: Path) -> None:
    """Distinct exit behaviour from "removed something" — the CLI surfaces
    a different message ("nothing to remove")."""
    auth_path = dikw_auth_path(dikw_base)
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(
        json.dumps({"version": 1, "providers": {}}), encoding="utf-8"
    )
    assert logout(dikw_base) is False


def test_logout_returns_false_when_store_missing(dikw_base: Path) -> None:
    assert logout(dikw_base) is False


async def test_logout_blocks_lazy_migration(
    dikw_base: Path,
    _isolated_codex_home: Path,
) -> None:
    """After explicit logout, the dikw store still exists (with codex
    removed). The next ``resolve_access_token`` must NOT silently
    re-import from codex CLI — that would undo the user's logout."""
    make_dikw_auth_store(
        dikw_base, access_token=_fresh_jwt(), refresh_token="rt-1"
    )
    make_codex_cli_auth_store(
        _isolated_codex_home, access_token=_fresh_jwt(), refresh_token="rt-cli"
    )
    logout(dikw_base)

    with pytest.raises(CodexAuthError) as excinfo:
        await resolve_access_token(dikw_base)
    assert excinfo.value.code == "codex_auth_missing"


# --------------------------------------------------------------------------- #
# Defensive: corrupted store doesn't crash status / list
# --------------------------------------------------------------------------- #


def test_auth_status_corrupted_store_marks_expiring(dikw_base: Path) -> None:
    auth_path = dikw_auth_path(dikw_base)
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text("{not-json", encoding="utf-8")
    status = auth_status(dikw_base)
    assert status.exists is True
    assert status.expiring_soon is True


def test_list_providers_ignores_corrupted_store(dikw_base: Path) -> None:
    auth_path = dikw_auth_path(dikw_base)
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text("{not-json", encoding="utf-8")
    assert list_providers(dikw_base) == []


# --------------------------------------------------------------------------- #
# Schema acceptance — _load_store accepts well-formed v1 with empty providers
# --------------------------------------------------------------------------- #


def test_schema_v1_with_empty_providers_yields_missing_error(
    dikw_base: Path,
) -> None:
    """A freshly-created (empty) auth store is valid; reading codex from
    it surfaces the missing-credential error rather than a shape error."""
    auth_path = dikw_auth_path(dikw_base)
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(
        json.dumps({"version": 1, "providers": {}}), encoding="utf-8"
    )
    with pytest.raises(CodexAuthError) as excinfo:
        read_codex_tokens(dikw_base)
    assert excinfo.value.code == "codex_auth_missing"


async def test_schema_unsupported_version_raises_for_resolve(
    dikw_base: Path,
) -> None:
    auth_path = dikw_auth_path(dikw_base)
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(
        json.dumps(
            {
                "version": 99,
                "providers": {
                    "openai-codex": {
                        "tokens": {"access_token": "at", "refresh_token": "rt"}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CodexAuthError) as excinfo:
        await resolve_access_token(dikw_base)
    assert excinfo.value.code == "codex_auth_unsupported_version"
