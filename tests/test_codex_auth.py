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
    dikw_auth_dir,
    dikw_auth_path,
    read_codex_tokens,
    save_codex_tokens,
)

from .conftest import make_dikw_auth_store
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


def _write_dikw_store(base: Path, payload: dict[str, Any]) -> Path:
    """Write an arbitrary store payload directly — used by the malformed /
    missing-block tests below to skip ``make_dikw_auth_store``'s well-formed
    contract."""
    auth_path = dikw_auth_path(base)
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps(payload), encoding="utf-8")
    return auth_path


def test_read_codex_tokens_missing_file_raises_with_relogin_hint(
    dikw_base: Path,
) -> None:
    with pytest.raises(CodexAuthError) as excinfo:
        read_codex_tokens(dikw_base)
    err = excinfo.value
    assert err.code == "codex_auth_missing"
    assert err.relogin_required is True
    # Error message tells the user how to recover.
    assert "dikw auth login" in str(err).lower() or "dikw auth import" in str(err).lower()


def test_read_codex_tokens_returns_access_and_refresh(dikw_base: Path) -> None:
    make_dikw_auth_store(dikw_base, access_token="at-1", refresh_token="rt-1")
    tokens = read_codex_tokens(dikw_base)
    assert tokens["access_token"] == "at-1"
    assert tokens["refresh_token"] == "rt-1"


def test_read_codex_tokens_malformed_json_raises(dikw_base: Path) -> None:
    auth_path = dikw_auth_path(dikw_base)
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(CodexAuthError) as excinfo:
        read_codex_tokens(dikw_base)
    assert excinfo.value.code == "codex_auth_invalid_json"


def test_read_codex_tokens_unsupported_version_raises(dikw_base: Path) -> None:
    """A future-version store written by a newer dikw must error rather
    than silently misread; readers refuse versions they don't recognise."""
    _write_dikw_store(
        dikw_base,
        {
            "version": 99,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "at", "refresh_token": "rt"}
                }
            },
        },
    )
    with pytest.raises(CodexAuthError) as excinfo:
        read_codex_tokens(dikw_base)
    assert excinfo.value.code == "codex_auth_unsupported_version"


def test_read_codex_tokens_missing_provider_node_raises(dikw_base: Path) -> None:
    """Auth store exists but no openai-codex entry → codex_auth_missing
    so the CLI surfaces the right ``dikw auth login`` hint."""
    _write_dikw_store(dikw_base, {"version": 1, "providers": {}})
    with pytest.raises(CodexAuthError) as excinfo:
        read_codex_tokens(dikw_base)
    assert excinfo.value.code == "codex_auth_missing"


def test_read_codex_tokens_missing_access_token_raises(dikw_base: Path) -> None:
    _write_dikw_store(
        dikw_base,
        {
            "version": 1,
            "providers": {"openai-codex": {"tokens": {"refresh_token": "rt-1"}}},
        },
    )
    with pytest.raises(CodexAuthError) as excinfo:
        read_codex_tokens(dikw_base)
    assert excinfo.value.code == "codex_auth_missing_access_token"
    assert excinfo.value.relogin_required is True


def test_read_codex_tokens_missing_refresh_token_raises(dikw_base: Path) -> None:
    _write_dikw_store(
        dikw_base,
        {
            "version": 1,
            "providers": {"openai-codex": {"tokens": {"access_token": "at-1"}}},
        },
    )
    with pytest.raises(CodexAuthError) as excinfo:
        read_codex_tokens(dikw_base)
    assert excinfo.value.code == "codex_auth_missing_refresh_token"
    assert excinfo.value.relogin_required is True


def test_read_codex_tokens_missing_tokens_block_raises(dikw_base: Path) -> None:
    _write_dikw_store(
        dikw_base,
        {"version": 1, "providers": {"openai-codex": {"some_other_key": True}}},
    )
    with pytest.raises(CodexAuthError) as excinfo:
        read_codex_tokens(dikw_base)
    assert excinfo.value.code == "codex_auth_invalid_shape"


# ---------------------------- save_codex_tokens -------------------------- #


def test_save_codex_tokens_round_trip(dikw_base: Path) -> None:
    save_codex_tokens(dikw_base, {"access_token": "at-2", "refresh_token": "rt-2"})
    tokens = read_codex_tokens(dikw_base)
    assert tokens == {"access_token": "at-2", "refresh_token": "rt-2"}


def test_save_codex_tokens_preserves_other_provider_entries(
    dikw_base: Path,
) -> None:
    """Multi-provider read-modify-write: anthropic / future provider nodes
    co-exist with openai-codex and must survive a refresh-write."""
    make_dikw_auth_store(
        dikw_base,
        access_token="old",
        refresh_token="rt-old",
        extra_providers={
            "anthropic": {
                "tokens": {"access_token": "anth-at", "refresh_token": "anth-rt"},
                "last_refresh": "2026-04-01T00:00:00Z",
                "auth_mode": "claude",
            }
        },
    )
    save_codex_tokens(dikw_base, {"access_token": "at-new", "refresh_token": "rt-new"})

    on_disk = json.loads(dikw_auth_path(dikw_base).read_text(encoding="utf-8"))
    assert on_disk["version"] == 1
    assert on_disk["providers"]["openai-codex"]["tokens"]["access_token"] == "at-new"
    assert on_disk["providers"]["openai-codex"]["tokens"]["refresh_token"] == "rt-new"
    # Other provider entry is untouched.
    assert on_disk["providers"]["anthropic"]["tokens"]["access_token"] == "anth-at"
    assert on_disk["providers"]["anthropic"]["last_refresh"] == "2026-04-01T00:00:00Z"
    # last_refresh on the openai-codex node is bumped on write.
    assert isinstance(on_disk["providers"]["openai-codex"]["last_refresh"], str)


def test_save_codex_tokens_creates_dikw_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "knowledge"
    save_codex_tokens(target, {"access_token": "at-3", "refresh_token": "rt-3"})
    assert dikw_auth_path(target).is_file()
    assert dikw_auth_dir(target).is_dir()


# ---------------------------- _auth_file_lock --------------------------- #


def test_auth_file_lock_serializes_concurrent_writers(dikw_base: Path) -> None:
    """Two threads both try to enter the lock while holding it for a brief
    sleep; their critical sections must not overlap."""
    lock_path = dikw_base / ".dikw" / "auth.lock"
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


def test_auth_file_lock_is_strictly_os_level_no_reentry(dikw_base: Path) -> None:
    """Reentrancy was removed because asyncio's ``threading.local``-based
    depth counter would let two coroutines on the same event loop bypass
    the OS lock and refresh the same OAuth refresh_token concurrently.
    Nested acquisition on the same thread must time out instead of
    silently re-entering."""
    lock_path = dikw_base / ".dikw" / "auth.lock"
    with _auth_file_lock(lock_path, timeout=10.0):
        with pytest.raises(TimeoutError):
            with _auth_file_lock(lock_path, timeout=1.0):
                pytest.fail("nested acquire should have timed out")


def test_save_codex_tokens_writes_atomically(dikw_base: Path) -> None:
    """Replace-via-rename leaves no observable partial state. After a
    successful save the .tmp scratch file must be gone."""
    save_codex_tokens(dikw_base, {"access_token": "at-1", "refresh_token": "rt-1"})
    assert dikw_auth_path(dikw_base).is_file()
    assert not (dikw_base / ".dikw" / "auth.json.tmp").exists()


def test_save_codex_tokens_writes_with_owner_only_permissions(
    dikw_base: Path,
) -> None:
    """OAuth tokens must not leak to other local users. The atomic-write
    helper opens auth.json.tmp with mode 0o600, so even on POSIX with a
    permissive umask (022) the resulting auth.json is owner-only.

    On Windows ``stat().st_mode`` doesn't reflect NTFS ACLs, so this
    check is a no-op there — we still assert the call succeeded."""
    import os
    import stat
    import sys

    save_codex_tokens(dikw_base, {"access_token": "at-secret", "refresh_token": "rt-secret"})
    auth_path = dikw_auth_path(dikw_base)
    assert auth_path.is_file()
    if sys.platform != "win32":
        mode = stat.S_IMODE(os.stat(auth_path).st_mode)
        assert mode == 0o600, f"expected 0o600, got {mode:o}"


def test_save_codex_tokens_overrides_leftover_tmp_permissions(
    dikw_base: Path,
) -> None:
    """A leftover ``auth.json.tmp`` from a previous crashed writer (or a
    manual touch) used to be reused by O_CREAT|O_TRUNC, which silently
    carried its old permissions onto auth.json after os.replace. The
    write path now unlinks any pre-existing tmp first, so the next
    save lands with mode 0o600 regardless of the leftover."""
    import os
    import stat
    import sys

    leftover = dikw_base / ".dikw" / "auth.json.tmp"
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_text("{}", encoding="utf-8")
    if sys.platform != "win32":
        os.chmod(leftover, 0o644)

    save_codex_tokens(dikw_base, {"access_token": "at", "refresh_token": "rt"})
    auth_path = dikw_auth_path(dikw_base)
    assert auth_path.is_file()
    assert not leftover.exists()
    if sys.platform != "win32":
        mode = stat.S_IMODE(os.stat(auth_path).st_mode)
        assert mode == 0o600
