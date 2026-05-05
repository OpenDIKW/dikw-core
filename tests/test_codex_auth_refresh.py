"""Codex OAuth refresh + resolve_access_token orchestration.

Mocks ``httpx.Client.post`` so no real network call ever fires; uses
``tmp_path`` + ``monkeypatch`` ``CODEX_HOME`` so no real ``~/.codex/auth.json``
is touched. Covers:

* The POST shape (URL, content-type, form data, client_id)
* refresh_token rotation (new vs. reused)
* Error-code → ``relogin_required`` mapping
* The full ``resolve_access_token`` flow (fresh / expiring / write-back)
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from dikw_core.providers.codex_auth import (
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_TOKEN_URL,
    CodexAuthError,
    refresh_codex_tokens,
    resolve_access_token,
)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_jwt(claims: dict[str, Any]) -> str:
    header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode("utf-8"))
    payload = _b64url(json.dumps(claims).encode("utf-8"))
    return f"{header}.{payload}.sig"


def _fresh_jwt() -> str:
    return _make_jwt({"exp": int(time.time()) + 3600})


def _expiring_jwt() -> str:
    return _make_jwt({"exp": int(time.time()) + 30})


@pytest.fixture()
def codex_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    return tmp_path


def _write_auth_json(home: Path, payload: dict[str, Any]) -> None:
    (home / "auth.json").write_text(json.dumps(payload), encoding="utf-8")


def _read_auth_json(home: Path) -> dict[str, Any]:
    return json.loads((home / "auth.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# httpx.Client mocking
# --------------------------------------------------------------------------- #


class _StubResponse:
    def __init__(
        self, *, status_code: int = 200, json_body: dict[str, Any] | None = None
    ) -> None:
        self.status_code = status_code
        self._json = json_body or {}

    def json(self) -> dict[str, Any]:
        return self._json


class _StubAsyncClient:
    def __init__(self, *, response: _StubResponse, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self._response = response
        self.post_calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> _StubAsyncClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> _StubResponse:
        self.post_calls.append({"url": url, **kwargs})
        return self._response


@pytest.fixture()
def patched_http(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Patch ``httpx.AsyncClient`` to return a configurable stub. Tests set
    ``rec['next_response']`` before calling refresh_codex_tokens."""
    rec: dict[str, Any] = {
        "next_response": _StubResponse(
            status_code=200,
            json_body={"access_token": "at-new", "refresh_token": "rt-new"},
        ),
        "last_client": None,
    }

    def _factory(**kwargs: Any) -> _StubAsyncClient:
        client = _StubAsyncClient(response=rec["next_response"], **kwargs)
        rec["last_client"] = client
        return client

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    return rec


# --------------------------------------------------------------------------- #
# refresh_codex_tokens — request shape
# --------------------------------------------------------------------------- #


async def test_refresh_posts_to_openai_oauth_token_url(patched_http: dict[str, Any]) -> None:
    await refresh_codex_tokens(refresh_token="rt-old")
    client = patched_http["last_client"]
    assert client is not None
    assert len(client.post_calls) == 1
    assert client.post_calls[0]["url"] == CODEX_OAUTH_TOKEN_URL
    assert CODEX_OAUTH_TOKEN_URL == "https://auth.openai.com/oauth/token"


async def test_refresh_sends_form_encoded_grant_with_codex_client_id(
    patched_http: dict[str, Any],
) -> None:
    await refresh_codex_tokens(refresh_token="rt-old")
    call = patched_http["last_client"].post_calls[0]
    headers = call.get("headers") or {}
    assert headers.get("Content-Type") == "application/x-www-form-urlencoded"
    data = call.get("data") or {}
    assert data["grant_type"] == "refresh_token"
    assert data["refresh_token"] == "rt-old"
    assert data["client_id"] == CODEX_OAUTH_CLIENT_ID


async def test_refresh_returns_new_access_and_refresh_tokens(
    patched_http: dict[str, Any],
) -> None:
    patched_http["next_response"] = _StubResponse(
        status_code=200,
        json_body={"access_token": "at-rotated", "refresh_token": "rt-rotated"},
    )
    new = await refresh_codex_tokens(refresh_token="rt-old")
    assert new["access_token"] == "at-rotated"
    assert new["refresh_token"] == "rt-rotated"


async def test_refresh_keeps_existing_refresh_when_response_omits_one(
    patched_http: dict[str, Any],
) -> None:
    patched_http["next_response"] = _StubResponse(
        status_code=200, json_body={"access_token": "at-rotated"}
    )
    new = await refresh_codex_tokens(refresh_token="rt-keep")
    assert new["access_token"] == "at-rotated"
    assert new["refresh_token"] == "rt-keep"


# --------------------------------------------------------------------------- #
# refresh_codex_tokens — error mapping
# --------------------------------------------------------------------------- #


async def test_refresh_invalid_grant_marks_relogin(patched_http: dict[str, Any]) -> None:
    patched_http["next_response"] = _StubResponse(
        status_code=400,
        json_body={
            "error": "invalid_grant",
            "error_description": "Refresh token expired.",
        },
    )
    with pytest.raises(CodexAuthError) as excinfo:
        await refresh_codex_tokens(refresh_token="rt-bad")
    err = excinfo.value
    assert err.code == "invalid_grant"
    assert err.relogin_required is True


async def test_refresh_token_reused_marks_relogin(patched_http: dict[str, Any]) -> None:
    patched_http["next_response"] = _StubResponse(
        status_code=400,
        json_body={"error": "refresh_token_reused"},
    )
    with pytest.raises(CodexAuthError) as excinfo:
        await refresh_codex_tokens(refresh_token="rt-stale")
    err = excinfo.value
    assert err.code == "refresh_token_reused"
    assert err.relogin_required is True
    # Message tells the user they need to re-run codex.
    assert "codex" in str(err).lower()


async def test_refresh_401_forces_relogin_even_without_known_code(
    patched_http: dict[str, Any],
) -> None:
    patched_http["next_response"] = _StubResponse(
        status_code=401, json_body={"error": "unauthorized_client"}
    )
    with pytest.raises(CodexAuthError) as excinfo:
        await refresh_codex_tokens(refresh_token="rt-x")
    assert excinfo.value.relogin_required is True


async def test_refresh_500_does_not_mark_relogin(patched_http: dict[str, Any]) -> None:
    """5xx is transient (server-side); the user shouldn't be told to relogin
    because of a flaky upstream."""
    patched_http["next_response"] = _StubResponse(status_code=500, json_body={})
    with pytest.raises(CodexAuthError) as excinfo:
        await refresh_codex_tokens(refresh_token="rt-x")
    assert excinfo.value.relogin_required is False


async def test_refresh_missing_access_token_in_response_raises(
    patched_http: dict[str, Any],
) -> None:
    patched_http["next_response"] = _StubResponse(
        status_code=200, json_body={"refresh_token": "rt-only"}
    )
    with pytest.raises(CodexAuthError) as excinfo:
        await refresh_codex_tokens(refresh_token="rt-old")
    assert excinfo.value.code == "codex_refresh_missing_access_token"
    assert excinfo.value.relogin_required is True


# --------------------------------------------------------------------------- #
# resolve_access_token
# --------------------------------------------------------------------------- #


async def test_resolve_returns_existing_when_fresh(
    codex_dir: Path, patched_http: dict[str, Any]
) -> None:
    fresh = _fresh_jwt()
    _write_auth_json(
        codex_dir, {"tokens": {"access_token": fresh, "refresh_token": "rt-1"}}
    )

    token = await resolve_access_token()
    assert token == fresh
    # No HTTP call should have fired.
    assert patched_http["last_client"] is None


async def test_resolve_refreshes_when_expiring_and_writes_back(
    codex_dir: Path, patched_http: dict[str, Any]
) -> None:
    new_fresh = _fresh_jwt()
    patched_http["next_response"] = _StubResponse(
        status_code=200,
        json_body={"access_token": new_fresh, "refresh_token": "rt-rotated"},
    )

    _write_auth_json(
        codex_dir,
        {"tokens": {"access_token": _expiring_jwt(), "refresh_token": "rt-old"}},
    )

    token = await resolve_access_token()
    assert token == new_fresh

    on_disk = _read_auth_json(codex_dir)
    assert on_disk["tokens"]["access_token"] == new_fresh
    assert on_disk["tokens"]["refresh_token"] == "rt-rotated"
    # POST was called exactly once with the OLD refresh_token.
    call = patched_http["last_client"].post_calls[0]
    assert call["data"]["refresh_token"] == "rt-old"


async def test_resolve_propagates_relogin_required_on_invalid_grant(
    codex_dir: Path, patched_http: dict[str, Any]
) -> None:
    patched_http["next_response"] = _StubResponse(
        status_code=400, json_body={"error": "invalid_grant"}
    )
    _write_auth_json(
        codex_dir,
        {"tokens": {"access_token": _expiring_jwt(), "refresh_token": "rt-stale"}},
    )

    with pytest.raises(CodexAuthError) as excinfo:
        await resolve_access_token()
    assert excinfo.value.relogin_required is True


async def test_resolve_raises_codex_auth_missing_when_file_absent(
    codex_dir: Path, patched_http: dict[str, Any]
) -> None:
    with pytest.raises(CodexAuthError) as excinfo:
        await resolve_access_token()
    assert excinfo.value.code == "codex_auth_missing"
