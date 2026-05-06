"""OAuth device-code flow — ``request_device_code`` + ``device_code_login``.

Mocks ``httpx.Client`` so we never touch the real OpenAI auth endpoint.
The flow has three stages:

1. POST ``/api/accounts/deviceauth/usercode``  → user_code + device_auth_id
2. Poll ``/api/accounts/deviceauth/token``     → authorization_code + verifier
3. POST ``/oauth/token``                        → access + refresh token

Pending authorisation responds 403/404; the polling loop keeps trying
until a 200 lands or the deadline passes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from dikw_core.providers.codex_auth import (
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_DEVICE_REDIRECT_URI,
    CODEX_OAUTH_DEVICE_TOKEN_URL,
    CODEX_OAUTH_DEVICE_USERCODE_URL,
    CODEX_OAUTH_TOKEN_URL,
    CodexAuthError,
    DeviceCodeChallenge,
    auth_status,
    device_code_login,
    request_device_code,
)


class _StubResponse:
    def __init__(
        self, *, status_code: int = 200, json_body: dict[str, Any] | None = None
    ) -> None:
        self.status_code = status_code
        self._json = json_body or {}

    def json(self) -> dict[str, Any]:
        return self._json


class _ScriptedClient:
    """Sync stand-in for ``httpx.Client`` driven by a queue of responses
    keyed by URL. Each entry is a list popped from the front, so a
    polling loop sees a sequence of pending → success."""

    def __init__(
        self, *, scripts: dict[str, list[_StubResponse]], **kwargs: Any
    ) -> None:
        self.kwargs = kwargs
        self._scripts = {url: list(responses) for url, responses in scripts.items()}
        self.calls: list[dict[str, Any]] = []

    def __enter__(self) -> _ScriptedClient:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def post(self, url: str, **kwargs: Any) -> _StubResponse:
        self.calls.append({"url": url, **kwargs})
        queue = self._scripts.get(url)
        if not queue:
            raise AssertionError(f"unscripted POST to {url}")
        return queue.pop(0)


@pytest.fixture()
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop ``time.sleep`` in the polling loop so 5s intervals don't make
    the test suite glacial. The codex_auth module imports ``time`` at
    top level, so we patch its ``sleep``."""
    import dikw_core.providers.codex_auth as codex_auth

    monkeypatch.setattr(codex_auth.time, "sleep", lambda _: None)


# --------------------------------------------------------------------------- #
# request_device_code (step 1)
# --------------------------------------------------------------------------- #


def test_request_device_code_posts_client_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    scripts = {
        CODEX_OAUTH_DEVICE_USERCODE_URL: [
            _StubResponse(
                json_body={
                    "user_code": "ABCD-EFGH",
                    "device_auth_id": "dev-1",
                    "interval": 5,
                }
            )
        ]
    }

    def factory(**kwargs: Any) -> _ScriptedClient:
        client = _ScriptedClient(scripts=scripts, **kwargs)
        captured["client"] = client
        return client

    monkeypatch.setattr(httpx, "Client", factory)
    challenge = request_device_code()
    assert isinstance(challenge, DeviceCodeChallenge)
    assert challenge.user_code == "ABCD-EFGH"
    assert challenge.device_auth_id == "dev-1"
    assert challenge.poll_interval_seconds >= 3
    call = captured["client"].calls[0]
    assert call["url"] == CODEX_OAUTH_DEVICE_USERCODE_URL
    assert call["json"] == {"client_id": CODEX_OAUTH_CLIENT_ID}


def test_request_device_code_clamps_interval_to_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An issuer that returns interval=1 mustn't make us hammer the
    polling endpoint — clamp to the floor."""
    scripts = {
        CODEX_OAUTH_DEVICE_USERCODE_URL: [
            _StubResponse(
                json_body={
                    "user_code": "X",
                    "device_auth_id": "d",
                    "interval": 1,
                }
            )
        ]
    }
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: _ScriptedClient(scripts=scripts, **kw)
    )
    challenge = request_device_code()
    assert challenge.poll_interval_seconds >= 3


def test_request_device_code_handles_missing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issuer returns 200 but body is missing user_code → structured error."""
    scripts = {
        CODEX_OAUTH_DEVICE_USERCODE_URL: [
            _StubResponse(json_body={"device_auth_id": "d"})
        ]
    }
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: _ScriptedClient(scripts=scripts, **kw)
    )
    with pytest.raises(CodexAuthError) as excinfo:
        request_device_code()
    assert excinfo.value.code == "device_code_incomplete"


def test_request_device_code_non_200_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = {
        CODEX_OAUTH_DEVICE_USERCODE_URL: [_StubResponse(status_code=503, json_body={})]
    }
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: _ScriptedClient(scripts=scripts, **kw)
    )
    with pytest.raises(CodexAuthError) as excinfo:
        request_device_code()
    assert excinfo.value.code == "device_code_request_error"


# --------------------------------------------------------------------------- #
# device_code_login — happy path through all three stages
# --------------------------------------------------------------------------- #


def _happy_scripts() -> dict[str, list[_StubResponse]]:
    return {
        CODEX_OAUTH_DEVICE_USERCODE_URL: [
            _StubResponse(
                json_body={
                    "user_code": "ABCD-EFGH",
                    "device_auth_id": "dev-1",
                    "interval": 5,
                }
            )
        ],
        CODEX_OAUTH_DEVICE_TOKEN_URL: [
            # First poll: still pending.
            _StubResponse(status_code=403, json_body={}),
            # Second poll: user authorised.
            _StubResponse(
                json_body={
                    "authorization_code": "auth-code-1",
                    "code_verifier": "verifier-1",
                }
            ),
        ],
        CODEX_OAUTH_TOKEN_URL: [
            _StubResponse(
                json_body={
                    "access_token": "at-final",
                    "refresh_token": "rt-final",
                }
            )
        ],
    }


def test_device_code_login_persists_tokens_to_dikw_store(
    dikw_base: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_polling: None,
) -> None:
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: _ScriptedClient(scripts=_happy_scripts(), **kw)
    )
    callbacks: list[DeviceCodeChallenge] = []
    result = device_code_login(
        dikw_base, on_challenge=callbacks.append, timeout_seconds=60
    )
    assert result.dest_path.is_file()
    assert callbacks[0].user_code == "ABCD-EFGH"

    # Status reflects the freshly written tokens.
    status = auth_status(dikw_base)
    assert status.exists is True
    assert status.account_id is None  # plain (non-JWT) tokens in this fixture


def test_device_code_login_token_exchange_uses_authorization_code_grant(
    dikw_base: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_polling: None,
) -> None:
    """Verify the ``/oauth/token`` POST shape — grant_type, redirect_uri,
    PKCE verifier — exactly matches what OpenAI expects."""
    captured_clients: list[_ScriptedClient] = []
    scripts = _happy_scripts()

    def factory(**kw: Any) -> _ScriptedClient:
        c = _ScriptedClient(scripts=scripts, **kw)
        captured_clients.append(c)
        return c

    monkeypatch.setattr(httpx, "Client", factory)
    device_code_login(dikw_base, on_challenge=None, timeout_seconds=60)

    # Find the call to /oauth/token across all clients.
    token_call = None
    for client in captured_clients:
        for call in client.calls:
            if call["url"] == CODEX_OAUTH_TOKEN_URL:
                token_call = call
                break
    assert token_call is not None, "token exchange POST never fired"
    data = token_call["data"]
    assert data["grant_type"] == "authorization_code"
    assert data["code"] == "auth-code-1"
    assert data["code_verifier"] == "verifier-1"
    assert data["redirect_uri"] == CODEX_OAUTH_DEVICE_REDIRECT_URI
    assert data["client_id"] == CODEX_OAUTH_CLIENT_ID
    headers = token_call.get("headers") or {}
    assert headers.get("Content-Type") == "application/x-www-form-urlencoded"


def test_device_code_login_polls_with_device_auth_id(
    dikw_base: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_polling: None,
) -> None:
    captured_clients: list[_ScriptedClient] = []
    scripts = _happy_scripts()

    def factory(**kw: Any) -> _ScriptedClient:
        c = _ScriptedClient(scripts=scripts, **kw)
        captured_clients.append(c)
        return c

    monkeypatch.setattr(httpx, "Client", factory)
    device_code_login(dikw_base, on_challenge=None, timeout_seconds=60)

    poll_calls = [
        call
        for client in captured_clients
        for call in client.calls
        if call["url"] == CODEX_OAUTH_DEVICE_TOKEN_URL
    ]
    # First call is 403 (pending), second is 200 (success).
    assert len(poll_calls) == 2
    assert poll_calls[0]["json"] == {
        "device_auth_id": "dev-1",
        "user_code": "ABCD-EFGH",
    }


def test_device_code_login_times_out_when_user_never_authorises(
    dikw_base: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_polling: None,
) -> None:
    """All polls keep returning 403 — the function should raise
    ``device_code_timeout`` after the deadline passes rather than
    looping forever."""
    scripts: dict[str, list[_StubResponse]] = {
        CODEX_OAUTH_DEVICE_USERCODE_URL: [
            _StubResponse(
                json_body={
                    "user_code": "X",
                    "device_auth_id": "d",
                    "interval": 5,
                }
            )
        ],
        # Many pending responses — the loop terminates when the timeout
        # elapses, so we just need the queue to be deeper than the loop
        # iterations could plausibly need.
        CODEX_OAUTH_DEVICE_TOKEN_URL: [
            _StubResponse(status_code=403, json_body={}) for _ in range(50)
        ],
    }
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: _ScriptedClient(scripts=scripts, **kw)
    )
    # Drive monotonic time forward so the loop trips the deadline quickly.
    base_time = time.monotonic()
    elapsed = [0.0]

    def fake_monotonic() -> float:
        # Each call advances 10s — the loop sleeps and re-checks.
        elapsed[0] += 10.0
        return base_time + elapsed[0]

    monkeypatch.setattr(
        "dikw_core.providers.codex_auth.time.monotonic", fake_monotonic
    )
    with pytest.raises(CodexAuthError) as excinfo:
        device_code_login(dikw_base, on_challenge=None, timeout_seconds=15)
    assert excinfo.value.code == "device_code_timeout"


def test_device_code_login_propagates_token_exchange_failure(
    dikw_base: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_polling: None,
) -> None:
    """If the final ``/oauth/token`` call returns non-200, surface a
    structured error — don't write a half-broken store."""
    scripts = _happy_scripts()
    scripts[CODEX_OAUTH_TOKEN_URL] = [_StubResponse(status_code=400, json_body={})]
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: _ScriptedClient(scripts=scripts, **kw)
    )
    with pytest.raises(CodexAuthError) as excinfo:
        device_code_login(dikw_base, on_challenge=None, timeout_seconds=60)
    assert excinfo.value.code == "token_exchange_error"


def test_device_code_login_fails_when_authorization_code_missing(
    dikw_base: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_polling: None,
) -> None:
    scripts = _happy_scripts()
    scripts[CODEX_OAUTH_DEVICE_TOKEN_URL] = [
        _StubResponse(json_body={"code_verifier": "v"})
    ]
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: _ScriptedClient(scripts=scripts, **kw)
    )
    with pytest.raises(CodexAuthError) as excinfo:
        device_code_login(dikw_base, on_challenge=None, timeout_seconds=60)
    assert excinfo.value.code == "device_code_incomplete_exchange"


def test_device_code_login_calls_on_challenge_before_polling(
    dikw_base: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_polling: None,
) -> None:
    """The CLI hook for rendering the user code must fire before the
    blocking poll — otherwise the user sees nothing while the loop runs."""
    scripts = _happy_scripts()
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: _ScriptedClient(scripts=scripts, **kw)
    )
    events: list[str] = []

    def hook(challenge: DeviceCodeChallenge) -> None:
        events.append(f"hook:{challenge.user_code}")

    # Patch sleep so we can detect ordering: polling sleep is the only
    # sleep in the path (after the hook fires).
    import dikw_core.providers.codex_auth as codex_auth

    def tracked_sleep(_: float) -> None:
        events.append("sleep")

    monkeypatch.setattr(codex_auth.time, "sleep", tracked_sleep)
    device_code_login(dikw_base, on_challenge=hook, timeout_seconds=60)
    # Hook fires before any polling sleep.
    assert events.index("hook:ABCD-EFGH") < events.index("sleep")
