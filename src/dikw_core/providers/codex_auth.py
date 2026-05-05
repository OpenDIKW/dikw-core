"""Codex OAuth credential resolution.

Reads ``~/.codex/auth.json`` (Codex CLI's standard location, override path
via ``$CODEX_HOME``), checks the access_token's JWT ``exp`` claim, and
refreshes through ``https://auth.openai.com/oauth/token`` when the token is
within ``CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS`` of expiry. Writes
refreshed tokens back to the same file under a cross-process advisory lock.

OAuth client_id is the public identifier of the **codex CLI application**
itself (not a per-user secret) — the same value every codex CLI install
uses globally. ChatGPT's OAuth issuer pins refresh_tokens to the client_id
that minted them, so refreshing a token written by codex CLI requires
sending codex CLI's client_id back. Sourced from the codex CLI repo and
mirrored by hermes-agent (hermes_cli/auth.py:74-91).
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .base import ProviderError

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover — Windows
    _fcntl = None  # type: ignore[assignment]

try:
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover — POSIX
    _msvcrt = None  # type: ignore[assignment]

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120
CODEX_AUTH_LOCK_TIMEOUT_SECONDS = 30.0


class CodexAuthError(ProviderError):
    """OAuth-specific failure with a structured ``code`` for diagnostics.

    ``relogin_required=True`` signals the user must run ``codex`` again to
    mint a fresh refresh_token (e.g., the existing one was rotated by
    another client and we got ``invalid_grant`` from the token endpoint).
    """

    def __init__(
        self, message: str, *, code: str, relogin_required: bool = False
    ) -> None:
        super().__init__(message)
        self.code = code
        self.relogin_required = relogin_required


_auth_lock_holder = threading.local()


def codex_home() -> Path:
    """Resolve ``$CODEX_HOME`` or fall back to ``~/.codex`` (codex CLI default).

    A blank env value is treated as unset so users can opt back into the
    default by clearing the variable rather than unsetting it.
    """
    raw = os.environ.get("CODEX_HOME", "").strip()
    if not raw:
        return Path.home() / ".codex"
    return Path(raw).expanduser()


def _decode_jwt_claims(token: str) -> dict[str, Any]:
    """base64url-decode the JWT payload segment.

    Returns ``{}`` for any input that isn't a parseable 3-segment JWT — the
    helpers built on top of this function (``_is_expiring``,
    ``account_id_from_jwt``) all default to a safe behaviour when the token
    isn't a JWT (refresh, drop the header), so silently returning empty
    keeps the policy in one place.
    """
    if not isinstance(token, str) or token.count(".") != 2:
        return {}
    payload_segment = token.split(".", 2)[1]
    if not payload_segment:
        return {}
    # base64url without padding — pad to a multiple of 4 before decoding.
    padded = payload_segment + "=" * (-len(payload_segment) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        claims = json.loads(raw)
    except Exception:
        return {}
    return claims if isinstance(claims, dict) else {}


def _is_expiring(token: str, *, skew_seconds: int) -> bool:
    """True if the token has < ``skew_seconds`` left, or isn't a JWT.

    Conservative on the unknown side: a non-JWT or a JWT without an ``exp``
    claim is treated as expiring so the caller refreshes. Better one extra
    network call than a 401 mid-pipeline.
    """
    claims = _decode_jwt_claims(token)
    exp = claims.get("exp")
    if not isinstance(exp, int | float):
        return True
    return float(exp) <= (time.time() + max(0, int(skew_seconds)))


def account_id_from_jwt(token: str) -> str | None:
    """Extract the ``chatgpt_account_id`` claim for the
    ``ChatGPT-Account-ID`` Cloudflare header.

    Returns ``None`` when the token isn't a JWT or the claim is absent /
    not a string. Callers omit the header in that case rather than
    sending a malformed value.
    """
    claims = _decode_jwt_claims(token)
    value = claims.get("chatgpt_account_id")
    if isinstance(value, str) and value:
        return value
    return None


# --------------------------------------------------------------------------- #
# Cross-process advisory file lock — fcntl on POSIX, msvcrt on Windows
# (style ported from hermes_cli/auth.py:756-810). Reentrant within one
# thread so resolve_access_token() can call save_codex_tokens() under its
# own outer lock without deadlocking.
# --------------------------------------------------------------------------- #


@contextmanager
def _auth_file_lock(path: Path, *, timeout: float) -> Iterator[None]:
    depth = getattr(_auth_lock_holder, "depth", 0)
    if depth > 0:
        _auth_lock_holder.depth = depth + 1
        try:
            yield
        finally:
            _auth_lock_holder.depth -= 1
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    if _fcntl is None and _msvcrt is None:  # pragma: no cover — defensive
        # No native lock primitives — degrade to thread-only mutual exclusion.
        _auth_lock_holder.depth = 1
        try:
            yield
        finally:
            _auth_lock_holder.depth = 0
        return

    # Windows ``msvcrt.locking`` requires the file to have at least 1 byte
    # at offset 0. Pre-seed the lock file when needed — using append mode
    # so a concurrent worker holding an open handle on it isn't blocked
    # (write_text would call open("w") which collides with the other
    # worker's r+ handle and raises PermissionError on Windows).
    if _msvcrt is not None:
        try:
            if not path.exists() or path.stat().st_size == 0:
                with path.open("a", encoding="utf-8") as seed:
                    seed.write(" ")
        except (PermissionError, OSError):
            # Another worker raced us and already seeded the file — that's
            # the desired end state, just continue to the lock acquire.
            pass

    open_mode = "r+" if _msvcrt else "a+"
    with path.open(open_mode) as lock_file:
        deadline = time.time() + max(1.0, timeout)
        while True:
            try:
                if _fcntl is not None:
                    _fcntl.flock(  # type: ignore[attr-defined]
                        lock_file.fileno(),
                        _fcntl.LOCK_EX | _fcntl.LOCK_NB,  # type: ignore[attr-defined]
                    )
                else:
                    assert _msvcrt is not None
                    lock_file.seek(0)
                    _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_NBLCK, 1)
                break
            except (BlockingIOError, OSError, PermissionError):
                if time.time() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for codex auth lock at {path}"
                    ) from None
                time.sleep(0.05)

        _auth_lock_holder.depth = 1
        try:
            yield
        finally:
            _auth_lock_holder.depth = 0
            if _fcntl is not None:
                _fcntl.flock(  # type: ignore[attr-defined]
                    lock_file.fileno(),
                    _fcntl.LOCK_UN,  # type: ignore[attr-defined]
                )
            elif _msvcrt is not None:
                try:
                    lock_file.seek(0)
                    _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_UNLCK, 1)
                except OSError:  # pragma: no cover — release best-effort
                    pass


# --------------------------------------------------------------------------- #
# auth.json read / write
# --------------------------------------------------------------------------- #


def _auth_file_path() -> Path:
    return codex_home() / "auth.json"


def _auth_lock_path() -> Path:
    return codex_home() / "auth.json.lock"


def read_codex_tokens() -> dict[str, str]:
    """Load access_token + refresh_token from ``<CODEX_HOME>/auth.json``.

    Raises ``CodexAuthError`` with a structured ``code`` on every error
    path so callers can route on it (e.g., ``relogin_required`` triggers
    a user-facing "run codex again" message).
    """
    path = _auth_file_path()
    if not path.is_file():
        raise CodexAuthError(
            f"No Codex credentials at {path}. "
            "Run `codex` once in your terminal to authenticate.",
            code="codex_auth_missing",
            relogin_required=True,
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CodexAuthError(
            f"Codex auth file at {path} is not valid JSON. "
            "Re-run `codex` to repair.",
            code="codex_auth_invalid_json",
            relogin_required=True,
        ) from exc

    tokens = raw.get("tokens") if isinstance(raw, dict) else None
    if not isinstance(tokens, dict):
        raise CodexAuthError(
            f"Codex auth file at {path} is missing the `tokens` block. "
            "Re-run `codex` to repair.",
            code="codex_auth_invalid_shape",
            relogin_required=True,
        )

    access = tokens.get("access_token")
    if not isinstance(access, str) or not access.strip():
        raise CodexAuthError(
            f"Codex auth at {path} is missing access_token. Re-run `codex`.",
            code="codex_auth_missing_access_token",
            relogin_required=True,
        )
    refresh = tokens.get("refresh_token")
    if not isinstance(refresh, str) or not refresh.strip():
        raise CodexAuthError(
            f"Codex auth at {path} is missing refresh_token. Re-run `codex`.",
            code="codex_auth_missing_refresh_token",
            relogin_required=True,
        )
    return {"access_token": access.strip(), "refresh_token": refresh.strip()}


def save_codex_tokens(tokens: dict[str, str]) -> None:
    """Write fresh tokens to ``<CODEX_HOME>/auth.json`` under lock.

    Preserves any unrelated top-level keys codex CLI may have written
    (``OPENAI_API_KEY`` shim, custom config blocks). Bumps ``last_refresh``
    so callers can audit when this writer last touched the file.
    """
    home = codex_home()
    home.mkdir(parents=True, exist_ok=True)
    path = home / "auth.json"

    with _auth_file_lock(_auth_lock_path(), timeout=CODEX_AUTH_LOCK_TIMEOUT_SECONDS):
        existing: dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                # Corrupt file — overwrite. The lock above already
                # blocks any racing writer.
                existing = {}

        existing["tokens"] = {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
        }
        existing["last_refresh"] = (
            datetime.now(UTC).isoformat().replace("+00:00", "Z")
        )
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# OAuth refresh (HTTP) + resolve_access_token orchestration
# --------------------------------------------------------------------------- #


_RELOGIN_ERROR_CODES = frozenset({"invalid_grant", "invalid_token", "invalid_request"})


def _extract_oauth_error(payload: Any) -> tuple[str, str | None]:
    """Pull (code, description) out of an OAuth token-endpoint error body.

    Handles both the spec shape ``{"error":"code","error_description":"..."}``
    and OpenAI's nested ``{"error":{"code":"...","message":"..."}}``.
    """
    if not isinstance(payload, dict):
        return "codex_refresh_failed", None
    err = payload.get("error")
    if isinstance(err, dict):
        code = err.get("code") or err.get("type") or "codex_refresh_failed"
        desc = err.get("message")
        return (
            str(code) if isinstance(code, str) else "codex_refresh_failed",
            desc if isinstance(desc, str) and desc.strip() else None,
        )
    if isinstance(err, str) and err.strip():
        desc = payload.get("error_description") or payload.get("message")
        return (
            err.strip(),
            desc if isinstance(desc, str) and desc.strip() else None,
        )
    return "codex_refresh_failed", None


def refresh_codex_tokens(
    *, refresh_token: str, timeout_seconds: float = 20.0
) -> dict[str, str]:
    """Exchange a refresh_token for a fresh access_token at the OpenAI
    OAuth token endpoint.

    Returns ``{access_token, refresh_token}`` — when the response carries a
    rotated refresh_token we use it, otherwise the input is preserved
    (some token endpoints omit the field on no-rotation refreshes).

    Raises ``CodexAuthError`` on failure with ``relogin_required=True`` for
    the codes that mean "the refresh_token can never recover" (invalid_grant
    / refresh_token_reused / 401-without-known-code), and
    ``relogin_required=False`` for transient 5xx so the user isn't told to
    re-login because of an upstream blip.
    """
    # Lazy import: httpx is already a project dep, but isolating it here
    # keeps the rest of the module importable in environments that stub it.
    import httpx

    timeout = httpx.Timeout(max(5.0, float(timeout_seconds)))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        response = client.post(
            CODEX_OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
        )

    if response.status_code != 200:
        try:
            payload = response.json()
        except Exception:
            payload = None
        code, description = _extract_oauth_error(payload)
        relogin = code in _RELOGIN_ERROR_CODES
        if code == "refresh_token_reused":
            relogin = True
            message = (
                "Codex refresh token was already consumed by another client "
                "(e.g. codex CLI or another dikw process). Run `codex` to "
                "generate fresh tokens."
            )
        elif description:
            message = f"Codex token refresh failed: {description}"
        else:
            message = (
                f"Codex token refresh failed with status {response.status_code}."
            )
        # 401/403 from the OAuth endpoint always means the refresh token
        # is bad — force relogin even if the body's error code wasn't one
        # of the known relogin strings.
        if response.status_code in (401, 403):
            relogin = True
        raise CodexAuthError(message, code=code, relogin_required=relogin)

    try:
        body = response.json()
    except Exception as exc:  # pragma: no cover — JSON parse fault
        raise CodexAuthError(
            "Codex token refresh returned invalid JSON.",
            code="codex_refresh_invalid_json",
            relogin_required=True,
        ) from exc

    new_access = body.get("access_token") if isinstance(body, dict) else None
    if not isinstance(new_access, str) or not new_access.strip():
        raise CodexAuthError(
            "Codex token refresh response was missing access_token.",
            code="codex_refresh_missing_access_token",
            relogin_required=True,
        )
    new_refresh = body.get("refresh_token") if isinstance(body, dict) else None
    if isinstance(new_refresh, str) and new_refresh.strip():
        rotated = new_refresh.strip()
    else:
        # Some token endpoints don't rotate on every call — keep the old
        # refresh_token rather than nulling out the long-term credential.
        rotated = refresh_token
    return {"access_token": new_access.strip(), "refresh_token": rotated}


def resolve_access_token(
    *,
    refresh_skew_seconds: int = CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    refresh_timeout_seconds: float = 20.0,
) -> str:
    """Single entrypoint for the LLM provider: load tokens, refresh if
    expiring, write back the fresh pair, return the active access_token.

    Re-reads the file under lock before refreshing so two parallel dikw
    workers each seeing a near-expiring token will only fire one network
    refresh — the second worker grabs the lock, re-reads, and finds the
    first worker's freshly-written token already valid.
    """
    tokens = read_codex_tokens()
    if not _is_expiring(tokens["access_token"], skew_seconds=refresh_skew_seconds):
        return tokens["access_token"]

    with _auth_file_lock(_auth_lock_path(), timeout=CODEX_AUTH_LOCK_TIMEOUT_SECONDS):
        # Re-check under lock: the holder of this lock right before us may
        # have just refreshed.
        tokens = read_codex_tokens()
        if not _is_expiring(
            tokens["access_token"], skew_seconds=refresh_skew_seconds
        ):
            return tokens["access_token"]
        refreshed = refresh_codex_tokens(
            refresh_token=tokens["refresh_token"],
            timeout_seconds=refresh_timeout_seconds,
        )
        save_codex_tokens(refreshed)
        return refreshed["access_token"]
