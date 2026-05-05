"""Codex OAuth credential resolution — pure-function + file-IO layer.

Pure helpers (JWT decoding, expiry check, account-id extraction,
codex_home resolution) live alongside file-IO + locking tests. The OAuth
HTTP refresh + ``resolve_access_token`` orchestration live in
test_codex_auth_refresh.py.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from dikw_core.providers.codex_auth import (
    CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    CodexAuthError,
    _auth_file_lock,
    _decode_jwt_claims,
    _is_expiring,
    account_id_from_jwt,
    codex_home,
    read_codex_tokens,
    save_codex_tokens,
)

from .fakes import make_jwt as _make_jwt


def _b64url(data: bytes) -> str:
    """One-off helper for the padding-tolerance test below — every other
    JWT-shaped value comes from ``make_jwt`` (in tests/fakes.py)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


# ---------------------------- codex_home --------------------------------- #


def test_codex_home_default_is_dot_codex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEX_HOME", raising=False)
    assert codex_home() == Path.home() / ".codex"


def test_codex_home_respects_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert codex_home() == tmp_path


def test_codex_home_treats_blank_env_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_HOME", "   ")
    assert codex_home() == Path.home() / ".codex"


# ---------------------------- _decode_jwt_claims ------------------------- #


def test_decode_jwt_claims_returns_payload() -> None:
    token = _make_jwt({"exp": 1234567890, "chatgpt_account_id": "user-abc"})
    claims = _decode_jwt_claims(token)
    assert claims["exp"] == 1234567890
    assert claims["chatgpt_account_id"] == "user-abc"


def test_decode_jwt_claims_handles_payload_without_padding() -> None:
    # Payload whose base64url encoding lacks the trailing '=' — the decoder
    # must add padding before decoding. A 1-byte payload triggers it.
    payload_bytes = b'{"x":1}'
    token = "h." + _b64url(payload_bytes) + ".s"
    assert _decode_jwt_claims(token) == {"x": 1}


def test_decode_jwt_claims_returns_empty_for_non_jwt() -> None:
    assert _decode_jwt_claims("plain-string-not-a-jwt") == {}
    assert _decode_jwt_claims("only.two") == {}
    assert _decode_jwt_claims("") == {}


def test_decode_jwt_claims_returns_empty_for_garbage_payload() -> None:
    # Three-part shape but middle segment isn't valid base64url JSON.
    assert _decode_jwt_claims("h.@@@.s") == {}


# ---------------------------- _is_expiring ------------------------------- #


def test_is_expiring_false_when_fresh() -> None:
    far_future = int(time.time()) + 3600
    token = _make_jwt({"exp": far_future})
    assert _is_expiring(token, skew_seconds=120) is False


def test_is_expiring_true_when_within_skew() -> None:
    soon = int(time.time()) + 30
    token = _make_jwt({"exp": soon})
    assert _is_expiring(token, skew_seconds=120) is True


def test_is_expiring_true_when_already_expired() -> None:
    past = int(time.time()) - 60
    token = _make_jwt({"exp": past})
    assert _is_expiring(token, skew_seconds=0) is True


def test_is_expiring_true_when_no_exp_claim() -> None:
    token = _make_jwt({"chatgpt_account_id": "user-abc"})
    # No exp claim → conservative: treat as expiring so the caller refreshes.
    assert _is_expiring(token, skew_seconds=120) is True


def test_is_expiring_true_for_non_jwt() -> None:
    assert _is_expiring("not-a-jwt", skew_seconds=120) is True
    assert _is_expiring("", skew_seconds=120) is True


def test_default_skew_constant_is_two_minutes() -> None:
    # Sanity-pin the shipped default so a future tweak surfaces in review.
    assert CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS == 120


# ---------------------------- account_id_from_jwt ------------------------ #


def test_account_id_from_jwt_extracts_chatgpt_account_id() -> None:
    token = _make_jwt({"chatgpt_account_id": "acc-12345", "exp": 999})
    assert account_id_from_jwt(token) == "acc-12345"


def test_account_id_from_jwt_returns_none_for_plain_string() -> None:
    assert account_id_from_jwt("plain-token-not-jwt") is None


def test_account_id_from_jwt_returns_none_when_claim_missing() -> None:
    token = _make_jwt({"exp": 999})
    assert account_id_from_jwt(token) is None


def test_account_id_from_jwt_returns_none_when_claim_not_string() -> None:
    token = _make_jwt({"chatgpt_account_id": 12345})
    assert account_id_from_jwt(token) is None


# ---------------------------- read_codex_tokens -------------------------- #


@pytest.fixture()
def codex_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate every test's $CODEX_HOME to a fresh tmp_path."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    return tmp_path


def _write_auth_json(home: Path, payload: dict[str, Any]) -> Path:
    auth_path = home / "auth.json"
    auth_path.write_text(json.dumps(payload), encoding="utf-8")
    return auth_path


def test_read_codex_tokens_missing_file_raises_with_relogin_hint(
    codex_dir: Path,
) -> None:
    with pytest.raises(CodexAuthError) as excinfo:
        read_codex_tokens()
    err = excinfo.value
    assert err.code == "codex_auth_missing"
    assert err.relogin_required is True
    # Error message tells the user how to recover.
    assert "codex" in str(err).lower()


def test_read_codex_tokens_returns_access_and_refresh(
    codex_dir: Path,
) -> None:
    _write_auth_json(
        codex_dir,
        {"tokens": {"access_token": "at-1", "refresh_token": "rt-1"}},
    )
    tokens = read_codex_tokens()
    assert tokens["access_token"] == "at-1"
    assert tokens["refresh_token"] == "rt-1"


def test_read_codex_tokens_malformed_json_raises(codex_dir: Path) -> None:
    (codex_dir / "auth.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises(CodexAuthError) as excinfo:
        read_codex_tokens()
    assert excinfo.value.code == "codex_auth_invalid_json"


def test_read_codex_tokens_missing_access_token_raises(
    codex_dir: Path,
) -> None:
    _write_auth_json(codex_dir, {"tokens": {"refresh_token": "rt-1"}})
    with pytest.raises(CodexAuthError) as excinfo:
        read_codex_tokens()
    assert excinfo.value.code == "codex_auth_missing_access_token"
    assert excinfo.value.relogin_required is True


def test_read_codex_tokens_missing_refresh_token_raises(
    codex_dir: Path,
) -> None:
    _write_auth_json(codex_dir, {"tokens": {"access_token": "at-1"}})
    with pytest.raises(CodexAuthError) as excinfo:
        read_codex_tokens()
    assert excinfo.value.code == "codex_auth_missing_refresh_token"
    assert excinfo.value.relogin_required is True


def test_read_codex_tokens_missing_tokens_block_raises(codex_dir: Path) -> None:
    _write_auth_json(codex_dir, {"some_other_key": True})
    with pytest.raises(CodexAuthError) as excinfo:
        read_codex_tokens()
    assert excinfo.value.code == "codex_auth_invalid_shape"


# ---------------------------- save_codex_tokens -------------------------- #


def test_save_codex_tokens_round_trip(codex_dir: Path) -> None:
    save_codex_tokens({"access_token": "at-2", "refresh_token": "rt-2"})
    tokens = read_codex_tokens()
    assert tokens == {"access_token": "at-2", "refresh_token": "rt-2"}


def test_save_codex_tokens_preserves_unrelated_top_level_keys(
    codex_dir: Path,
) -> None:
    """codex CLI writes additional top-level fields (last_refresh,
    OPENAI_API_KEY shim, …) — replacing tokens must not wipe them."""
    _write_auth_json(
        codex_dir,
        {
            "tokens": {"access_token": "old", "refresh_token": "rt-old"},
            "OPENAI_API_KEY": "sk-shim",
            "last_refresh": "2026-04-01T00:00:00Z",
        },
    )
    save_codex_tokens({"access_token": "at-new", "refresh_token": "rt-new"})

    on_disk = json.loads((codex_dir / "auth.json").read_text(encoding="utf-8"))
    assert on_disk["tokens"]["access_token"] == "at-new"
    assert on_disk["tokens"]["refresh_token"] == "rt-new"
    assert on_disk["OPENAI_API_KEY"] == "sk-shim"
    # last_refresh is bumped on write so we don't pin its value, just its
    # presence and shape.
    assert isinstance(on_disk["last_refresh"], str)
    assert on_disk["last_refresh"] != "2026-04-01T00:00:00Z"


def test_save_codex_tokens_creates_codex_home_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "nested" / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(target))
    save_codex_tokens({"access_token": "at-3", "refresh_token": "rt-3"})
    assert (target / "auth.json").is_file()


# ---------------------------- _auth_file_lock --------------------------- #


def test_auth_file_lock_serializes_concurrent_writers(codex_dir: Path) -> None:
    """Two threads both try to enter the lock while holding it for a brief
    sleep; their critical sections must not overlap."""
    lock_path = codex_dir / "auth.lock"
    in_section: list[bool] = [False]
    overlap_observed: list[bool] = [False]
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        with _auth_file_lock(lock_path, timeout=10.0):
            if in_section[0]:
                overlap_observed[0] = True
            in_section[0] = True
            time.sleep(0.05)
            in_section[0] = False

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "worker stuck — lock acquisition deadlocked"

    assert not overlap_observed[0]


def test_auth_file_lock_is_reentrant_on_same_thread(codex_dir: Path) -> None:
    """A single thread can re-enter the lock without deadlocking — needed
    so resolve_access_token() can call save_codex_tokens() under its own
    outer lock without recursive blocking."""
    lock_path = codex_dir / "auth.lock"
    with _auth_file_lock(lock_path, timeout=2.0):
        with _auth_file_lock(lock_path, timeout=2.0):
            pass
