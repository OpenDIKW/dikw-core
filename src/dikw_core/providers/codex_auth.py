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

import asyncio
import base64
import contextlib
import json
import os
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .base import ProviderError

# Annotate as Any|None so mypy treats attribute access uniformly across
# platforms — fcntl is POSIX-only and msvcrt is Windows-only, so leaving
# the import-as type would force per-platform `# type: ignore[attr-defined]`
# and per-platform `unused-ignore` churn. The runtime ``is not None`` checks
# below carry the actual platform dispatch.
_fcntl: Any | None = None
_msvcrt: Any | None = None

try:
    import fcntl
    _fcntl = fcntl
except ImportError:  # pragma: no cover — Windows
    pass

try:
    import msvcrt
    _msvcrt = msvcrt
except ImportError:  # pragma: no cover — POSIX
    pass

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
# Cross-process advisory file lock — fcntl on POSIX, msvcrt on Windows.
# Strictly OS-level: no in-process reentrancy. An earlier ``threading.local``
# depth counter let nested ``with _auth_file_lock(): ...`` skip the OS lock,
# which is correct on a sync stack but unsafe under asyncio: two coroutines
# on the same event loop share the same thread, so the second one would see
# the first's depth>0 and skip locking even though they're independent
# tasks — leading to two concurrent OAuth refreshes that mutually invalidate
# each other's refresh_token. Callers that need to do work under the lock
# now hold it directly via this contextmanager and must not call into other
# functions that re-acquire it.
# --------------------------------------------------------------------------- #


def _seed_lock_file_if_needed(path: Path) -> None:
    """Windows ``msvcrt.locking`` requires the file to have ≥1 byte at
    offset 0; pre-seed via append mode so a concurrent worker holding an
    r+ handle isn't blocked by a write_text/open("w") collision."""
    if _msvcrt is None:
        return
    try:
        if not path.exists() or path.stat().st_size == 0:
            with path.open("a", encoding="utf-8") as seed:
                seed.write(" ")
    except (PermissionError, OSError):
        # Another worker already seeded the file — desired end state.
        pass


def _try_os_lock_acquire(lock_file: Any) -> None:
    """Single non-blocking acquire attempt. Raises BlockingIOError /
    OSError / PermissionError on contention; returns silently on success."""
    if _fcntl is not None:
        _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        return
    assert _msvcrt is not None
    lock_file.seek(0)
    _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_NBLCK, 1)


def _release_os_lock(lock_file: Any) -> None:
    if _fcntl is not None:
        _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_UN)
        return
    if _msvcrt is not None:
        try:
            lock_file.seek(0)
            _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_UNLCK, 1)
        except OSError:  # pragma: no cover — release best-effort
            pass


@contextmanager
def _auth_file_lock(path: Path, *, timeout: float) -> Iterator[None]:
    """Sync flavour — used by ``save_codex_tokens`` (the public sync API).

    The async variant ``_async_auth_file_lock`` mirrors this except the
    retry sleep yields back to the event loop instead of blocking it; use
    that one from any code path running under asyncio.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if _fcntl is None and _msvcrt is None:  # pragma: no cover — defensive
        yield
        return
    _seed_lock_file_if_needed(path)
    open_mode = "r+" if _msvcrt else "a+"
    with path.open(open_mode) as lock_file:
        deadline = time.time() + max(1.0, timeout)
        while True:
            try:
                _try_os_lock_acquire(lock_file)
                break
            except (BlockingIOError, OSError, PermissionError):
                if time.time() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for codex auth lock at {path}"
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            _release_os_lock(lock_file)


@asynccontextmanager
async def _async_auth_file_lock(
    path: Path, *, timeout: float
) -> AsyncIterator[None]:
    """Async flavour of ``_auth_file_lock``.

    Used by ``resolve_access_token`` so two coroutines on the same event
    loop racing for the same expiring token don't end up with the second
    one blocking the loop in ``time.sleep`` while the first awaits the
    OAuth refresh — the second instead yields via ``asyncio.sleep`` and
    re-checks once the first releases. OS-level lock semantics are
    identical to the sync version (cross-process safety is preserved
    because both flavours use the same file).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if _fcntl is None and _msvcrt is None:  # pragma: no cover — defensive
        yield
        return
    _seed_lock_file_if_needed(path)
    open_mode = "r+" if _msvcrt else "a+"
    with path.open(open_mode) as lock_file:
        deadline = time.time() + max(1.0, timeout)
        while True:
            try:
                _try_os_lock_acquire(lock_file)
                break
            except (BlockingIOError, OSError, PermissionError):
                if time.time() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for codex auth lock at {path}"
                    ) from None
                await asyncio.sleep(0.05)
        try:
            yield
        finally:
            _release_os_lock(lock_file)


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


def _write_tokens_unlocked(home: Path, tokens: dict[str, str]) -> None:
    """Atomic, in-place token write. Caller must hold ``_auth_file_lock``.

    Preserves any unrelated top-level keys codex CLI may have written
    (``OPENAI_API_KEY`` shim, custom config blocks). Bumps ``last_refresh``
    so callers can audit when this writer last touched the file. Writes to
    ``auth.json.tmp`` then ``os.replace`` onto ``auth.json`` so cross-process
    readers (which don't hold the advisory lock) never observe a partially
    written file — ``os.replace`` is atomic on both POSIX and Windows.
    """
    path = home / "auth.json"
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            # Corrupt file — overwrite. The lock the caller holds blocks
            # racing writers, but cross-process readers may have seen the
            # corruption from a previous crash mid-write; reset deliberately.
            existing = {}

    existing["tokens"] = {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
    }
    existing["last_refresh"] = (
        datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    tmp_path = path.with_name(path.name + ".tmp")
    # Open with O_CREAT|O_WRONLY|0o600 so a permissive umask (e.g. 022)
    # doesn't leave OAuth tokens world-readable on POSIX. mode is
    # effectively a no-op on Windows where NTFS ACLs handle privacy at
    # the user level. **First unlink any leftover .tmp** — POSIX honours
    # the mode argument only on creation; reusing a pre-existing file
    # (e.g. from a crashed writer or a manual touch) would silently
    # carry that file's permissions onto auth.json after os.replace.
    tmp_path.unlink(missing_ok=True)
    payload = json.dumps(existing, indent=2).encode("utf-8")
    flags = os.O_CREAT | os.O_WRONLY | os.O_EXCL
    fd = os.open(tmp_path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
    except BaseException:
        # fdopen consumes fd on success; on failure ensure no leak before
        # re-raising so the temp file gets cleaned up below.
        with contextlib.suppress(OSError):
            os.close(fd)
        raise
    os.replace(tmp_path, path)


def save_codex_tokens(tokens: dict[str, str]) -> None:
    """Public ``save_codex_tokens`` — acquires the advisory lock, then
    atomic-writes. ``resolve_access_token`` does the unlocked write
    directly because it already holds the lock for double-checked
    refresh."""
    home = codex_home()
    home.mkdir(parents=True, exist_ok=True)
    with _auth_file_lock(_auth_lock_path(), timeout=CODEX_AUTH_LOCK_TIMEOUT_SECONDS):
        _write_tokens_unlocked(home, tokens)


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


async def refresh_codex_tokens(
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

    Async because the only caller (resolve_access_token) runs inside the
    LLM provider's async path — a sync httpx.Client.post would block the
    asyncio event loop for the whole OAuth round-trip (200-800ms typical),
    stalling every other in-flight task.
    """
    import httpx

    timeout = httpx.Timeout(max(5.0, float(timeout_seconds)))
    async with httpx.AsyncClient(
        timeout=timeout, headers={"Accept": "application/json"}
    ) as client:
        response = await client.post(
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


async def resolve_access_token(
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

    home = codex_home()
    home.mkdir(parents=True, exist_ok=True)
    async with _async_auth_file_lock(
        _auth_lock_path(), timeout=CODEX_AUTH_LOCK_TIMEOUT_SECONDS
    ):
        # Re-check under lock: the holder of this lock right before us may
        # have just refreshed.
        tokens = read_codex_tokens()
        if not _is_expiring(
            tokens["access_token"], skew_seconds=refresh_skew_seconds
        ):
            return tokens["access_token"]
        refreshed = await refresh_codex_tokens(
            refresh_token=tokens["refresh_token"],
            timeout_seconds=refresh_timeout_seconds,
        )
        # Direct unlocked write — we already hold the lock. Calling
        # save_codex_tokens() here would re-acquire it, which used to be
        # silently allowed by a threading.local depth counter but is unsafe
        # on an asyncio event loop where two tasks share one thread.
        _write_tokens_unlocked(home, refreshed)
        return refreshed["access_token"]
