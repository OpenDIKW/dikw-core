"""Codex OAuth credential resolution — pure-function layer.

This file covers the in-memory helpers (JWT decoding, expiry check,
account-id extraction, codex_home resolution). File I/O / refresh are
covered in test_codex_auth_io.py and test_codex_auth_refresh.py.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import pytest

from dikw_core.providers.codex_auth import (
    CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    _decode_jwt_claims,
    _is_expiring,
    account_id_from_jwt,
    codex_home,
)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_jwt(claims: dict) -> str:
    header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode("utf-8"))
    payload = _b64url(json.dumps(claims).encode("utf-8"))
    # Signature segment is mandatory for the 3-part shape but the helpers
    # under test never verify it.
    return f"{header}.{payload}.signature-not-checked"


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
