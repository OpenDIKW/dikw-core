"""Codex OAuth credential resolution — dikw self-managed token store.

Tokens live at ``<base>/.dikw/auth.json`` (the dikw "auth store"), separate
from codex CLI's ``~/.codex/auth.json``. Each base owns its own copy of the
credentials. Why a separate store: OpenAI's ChatGPT OAuth issuer rotates
the refresh_token on every refresh; if multiple clients (codex CLI,
hermes-agent, dikw) write the same file, whichever client refreshes second
is silently logged out because its refresh_token has just been invalidated
by the first. Each client therefore needs its own persisted refresh_token.

Bootstrap paths into the dikw auth store:
  * ``device_code_login(base)`` runs the OpenAI device-code flow itself —
    no dependency on codex CLI.
  * ``import_from_codex_cli(base)`` reads codex CLI's existing
    ``~/.codex/auth.json`` (override via ``$CODEX_HOME``) once and copies
    the tokens; codex CLI's file is never written by dikw afterwards.
  * ``_maybe_migrate_from_codex_cli(base)`` is a lazy in-process variant
    of the above — fires automatically the first time dikw needs a token
    and the dikw store is missing while codex CLI's file is valid. This
    keeps existing users unblocked across the upgrade boundary.

OAuth client_id is the public identifier of the **codex CLI application**
itself (not a per-user secret) — the same value every codex CLI install
uses globally. ChatGPT's OAuth issuer pins refresh_tokens to the client_id
that minted them, so refreshing a token written by codex CLI requires
sending codex CLI's client_id back. Sourced from the codex CLI repo and
mirrored by hermes-agent (hermes_cli/auth.py).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import sys
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
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
CODEX_OAUTH_ISSUER = "https://auth.openai.com"
CODEX_OAUTH_TOKEN_URL = f"{CODEX_OAUTH_ISSUER}/oauth/token"
CODEX_OAUTH_DEVICE_USERCODE_URL = f"{CODEX_OAUTH_ISSUER}/api/accounts/deviceauth/usercode"
CODEX_OAUTH_DEVICE_TOKEN_URL = f"{CODEX_OAUTH_ISSUER}/api/accounts/deviceauth/token"
CODEX_OAUTH_DEVICE_VERIFICATION_URL = f"{CODEX_OAUTH_ISSUER}/codex/device"
CODEX_OAUTH_DEVICE_REDIRECT_URI = f"{CODEX_OAUTH_ISSUER}/deviceauth/callback"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120
CODEX_AUTH_LOCK_TIMEOUT_SECONDS = 30.0
CODEX_DEVICE_LOGIN_TIMEOUT_SECONDS = 15 * 60  # 15 min — matches codex CLI
CODEX_DEVICE_POLL_MIN_INTERVAL_SECONDS = 3

# Auth store schema constants — bumping ``_AUTH_STORE_VERSION`` is a hard
# break; readers refuse versions they don't recognise.
_AUTH_STORE_VERSION = 1
_PROVIDER_KEY = "openai-codex"

# Providers the dikw auth store knows how to authenticate today. The
# ``auth_cli`` layer reuses this so the supported set stays single-sourced
# when (e.g.) anthropic OAuth lands.
SUPPORTED_PROVIDERS: frozenset[str] = frozenset({_PROVIDER_KEY})


class CodexAuthError(ProviderError):
    """OAuth-specific failure with a structured ``code`` for diagnostics.

    ``relogin_required=True`` signals the user must run the device-code
    login again (or import fresh tokens) to mint a new refresh_token —
    e.g., the existing one was rotated by another client and we got
    ``invalid_grant`` from the token endpoint.
    """

    def __init__(
        self, message: str, *, code: str, relogin_required: bool = False
    ) -> None:
        super().__init__(message)
        self.code = code
        self.relogin_required = relogin_required


# --------------------------------------------------------------------------- #
# Path resolution — dikw auth store (writes go here) and codex CLI source
# (reads via $CODEX_HOME, only for one-shot import)
# --------------------------------------------------------------------------- #


def codex_home() -> Path:
    """Resolve ``$CODEX_HOME`` or fall back to ``~/.codex`` (codex CLI default).

    A blank env value is treated as unset so users can opt back into the
    default by clearing the variable rather than unsetting it.

    **Used only by ``import_from_codex_cli``** as the read-only source path
    — dikw never writes to this location. All dikw writes target
    ``dikw_auth_path(base)`` instead.
    """
    raw = os.environ.get("CODEX_HOME", "").strip()
    if not raw:
        return Path.home() / ".codex"
    return Path(raw).expanduser()


def dikw_auth_dir(base: Path) -> Path:
    """The ``.dikw/`` state directory inside a wiki base."""
    return base / ".dikw"


def dikw_auth_path(base: Path) -> Path:
    """Resolve the dikw auth store file for a given base — single source of truth."""
    return dikw_auth_dir(base) / "auth.json"


def dikw_auth_lock_path(base: Path) -> Path:
    return dikw_auth_dir(base) / "auth.json.lock"


# --------------------------------------------------------------------------- #
# JWT inspection (pure functions)
# --------------------------------------------------------------------------- #


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
# Auth store read / write — nested multi-provider schema:
#
#   {
#     "version": 1,
#     "providers": {
#       "openai-codex": {
#         "tokens": {"access_token": "...", "refresh_token": "..."},
#         "last_refresh": "2026-05-06T03:14:22Z",
#         "auth_mode": "chatgpt"
#       }
#     }
#   }
#
# The schema mirrors hermes-agent's ``~/.hermes/auth.json`` so users with
# both tools have one mental model. Future OAuth providers (anthropic,
# etc.) will get sibling entries under ``providers``.
# --------------------------------------------------------------------------- #


def _load_store(path: Path) -> dict[str, Any]:
    """Load the auth store, validate version, return mutable dict.

    Returns an empty store skeleton if the file is missing. Raises
    ``CodexAuthError`` for unparseable JSON or unsupported version so
    callers route on it.
    """
    if not path.is_file():
        return {"version": _AUTH_STORE_VERSION, "providers": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CodexAuthError(
            f"dikw auth store at {path} is not valid JSON. "
            "Re-run `dikw auth login openai-codex` to repair.",
            code="codex_auth_invalid_json",
            relogin_required=True,
        ) from exc
    if not isinstance(raw, dict):
        raise CodexAuthError(
            f"dikw auth store at {path} has an unexpected top-level shape.",
            code="codex_auth_invalid_shape",
            relogin_required=True,
        )
    version = raw.get("version")
    if version != _AUTH_STORE_VERSION:
        raise CodexAuthError(
            f"dikw auth store at {path} has unsupported version {version!r}; "
            f"this build expects version {_AUTH_STORE_VERSION}.",
            code="codex_auth_unsupported_version",
            relogin_required=True,
        )
    providers = raw.get("providers")
    if not isinstance(providers, dict):
        # Tolerate a missing/null providers map by normalising — the user
        # can still log in to fill it.
        raw["providers"] = {}
    return raw


def read_codex_tokens(base: Path) -> dict[str, str]:
    """Load access_token + refresh_token for ``openai-codex`` from the
    dikw auth store at ``<base>/.dikw/auth.json``.

    Raises ``CodexAuthError`` with a structured ``code`` on every error
    path so callers can route on it (e.g., ``relogin_required`` triggers
    a user-facing "run dikw auth login" message).
    """
    path = dikw_auth_path(base)
    # _load_store returns an empty skeleton when the file is missing, so
    # the missing-file case collapses into the missing-provider-node case
    # below — same error code, same recovery hint, one branch.
    store = _load_store(path)
    provider_node = store.get("providers", {}).get(_PROVIDER_KEY)
    if not isinstance(provider_node, dict):
        raise CodexAuthError(
            f"No dikw codex credentials at {path}. "
            "Run `dikw auth login openai-codex` to authenticate, "
            "or `dikw auth import openai-codex` to import from codex CLI.",
            code="codex_auth_missing",
            relogin_required=True,
        )

    tokens = provider_node.get("tokens")
    if not isinstance(tokens, dict):
        raise CodexAuthError(
            f"dikw auth store at {path} is missing the openai-codex `tokens` block. "
            "Re-run `dikw auth login openai-codex` to repair.",
            code="codex_auth_invalid_shape",
            relogin_required=True,
        )

    access = tokens.get("access_token")
    if not isinstance(access, str) or not access.strip():
        raise CodexAuthError(
            f"dikw auth store at {path} is missing access_token. "
            "Re-run `dikw auth login openai-codex`.",
            code="codex_auth_missing_access_token",
            relogin_required=True,
        )
    refresh = tokens.get("refresh_token")
    if not isinstance(refresh, str) or not refresh.strip():
        raise CodexAuthError(
            f"dikw auth store at {path} is missing refresh_token. "
            "Re-run `dikw auth login openai-codex`.",
            code="codex_auth_missing_refresh_token",
            relogin_required=True,
        )
    return {"access_token": access.strip(), "refresh_token": refresh.strip()}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _atomic_write_store(auth_path: Path, store: dict[str, Any]) -> None:
    """Atomic 0o600 JSON write — caller must hold the advisory lock.

    Writes ``auth.json.tmp`` then ``os.replace`` onto ``auth.json`` so
    cross-process readers (which don't hold the advisory lock) never
    observe a partially-written file. ``os.replace`` is atomic on both
    POSIX and Windows.

    Mode 0o600 keeps OAuth tokens off other local users' eyes on POSIX
    even with a permissive umask (Windows uses NTFS ACLs and ignores the
    mode bits). The ``unlink + O_EXCL`` dance is what enforces it: POSIX
    honours the mode argument only on creation, so reusing a pre-existing
    .tmp (e.g. from a crashed writer) would silently carry its old
    permissions onto auth.json after os.replace.
    """
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = auth_path.with_name(auth_path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)
    payload = json.dumps(store, indent=2).encode("utf-8")
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
    os.replace(tmp_path, auth_path)


def _write_tokens_unlocked(
    auth_path: Path,
    tokens: dict[str, str],
    *,
    last_refresh: str | None = None,
    auth_mode: str = "chatgpt",
) -> None:
    """Atomic, in-place token write. Caller must hold ``_auth_file_lock``.

    Read-modify-write on the full store: any other provider entry already
    on disk is preserved untouched.
    """
    try:
        store = _load_store(auth_path)
    except CodexAuthError:
        # Corrupt or unsupported file — overwrite. The lock the caller
        # holds blocks racing writers, but cross-process readers may have
        # seen the corruption from a previous crash mid-write; reset
        # deliberately rather than refusing to make progress.
        store = {"version": _AUTH_STORE_VERSION, "providers": {}}

    providers = store.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        store["providers"] = providers
    providers[_PROVIDER_KEY] = {
        "tokens": {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
        },
        "last_refresh": last_refresh or _now_iso(),
        "auth_mode": auth_mode,
    }
    _atomic_write_store(auth_path, store)


def save_codex_tokens(
    base: Path, tokens: dict[str, str], *, auth_mode: str = "chatgpt"
) -> None:
    """Public sync save — acquires the advisory lock, then atomic-writes.

    ``resolve_access_token`` does the unlocked write directly because it
    already holds the lock for double-checked refresh.
    """
    dikw_auth_dir(base).mkdir(parents=True, exist_ok=True)
    with _auth_file_lock(
        dikw_auth_lock_path(base), timeout=CODEX_AUTH_LOCK_TIMEOUT_SECONDS
    ):
        _write_tokens_unlocked(dikw_auth_path(base), tokens, auth_mode=auth_mode)


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
                "(e.g. codex CLI or another dikw process). "
                "Re-authenticate by running `dikw auth login openai-codex`."
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
    base: Path,
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

    On the very first call after upgrade, transparently imports tokens
    from codex CLI's ``~/.codex/auth.json`` if the dikw auth store
    doesn't exist yet — see ``_maybe_migrate_from_codex_cli``.
    """
    _maybe_migrate_from_codex_cli(base)

    tokens = read_codex_tokens(base)
    if not _is_expiring(tokens["access_token"], skew_seconds=refresh_skew_seconds):
        return tokens["access_token"]

    dikw_auth_dir(base).mkdir(parents=True, exist_ok=True)
    async with _async_auth_file_lock(
        dikw_auth_lock_path(base), timeout=CODEX_AUTH_LOCK_TIMEOUT_SECONDS
    ):
        # Re-check under lock: the holder of this lock right before us may
        # have just refreshed.
        tokens = read_codex_tokens(base)
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
        _write_tokens_unlocked(dikw_auth_path(base), refreshed)
        return refreshed["access_token"]


# --------------------------------------------------------------------------- #
# Lazy migration from codex CLI's auth.json on first use
# --------------------------------------------------------------------------- #


def _read_codex_cli_tokens_if_valid() -> dict[str, str] | None:
    """Read codex CLI's flat ``auth.json`` and return tokens iff non-expired.

    Returns ``None`` for any failure (file missing, malformed, expired) —
    the caller treats that as "no migration possible, proceed to the
    normal codex_auth_missing error path".
    """
    src = codex_home() / "auth.json"
    if not src.is_file():
        return None
    try:
        raw = json.loads(src.read_text(encoding="utf-8"))
    except Exception:
        return None
    tokens = raw.get("tokens") if isinstance(raw, dict) else None
    if not isinstance(tokens, dict):
        return None
    access = tokens.get("access_token")
    refresh = tokens.get("refresh_token")
    if not isinstance(access, str) or not access.strip():
        return None
    if not isinstance(refresh, str) or not refresh.strip():
        return None
    if _is_expiring(access, skew_seconds=0):
        # Don't migrate already-expired tokens — refresh_token may still be
        # valid but we want the user to do an explicit `dikw auth import`
        # so they see the lazy migration boundary clearly.
        return None
    return {"access_token": access.strip(), "refresh_token": refresh.strip()}


def _maybe_migrate_from_codex_cli(base: Path) -> None:
    """Populate the dikw auth store from codex CLI's file on first use.

    Runs only when the dikw store **file** is missing entirely. After
    ``dikw auth logout openai-codex`` the store still exists (with the
    openai-codex node removed), so this won't auto-undo a deliberate
    logout — the user must explicitly re-import or re-login.
    """
    dest = dikw_auth_path(base)
    if dest.exists():
        return
    src_tokens = _read_codex_cli_tokens_if_valid()
    if src_tokens is None:
        return
    src_path = codex_home() / "auth.json"
    save_codex_tokens(base, src_tokens)
    sys.stderr.write(
        f"[dikw] Imported codex tokens from {src_path} to {dest}.\n"
        f"[dikw] dikw will no longer write to {src_path}, but the imported "
        "refresh_token is still shared with codex CLI until the next refresh "
        "on either side rotates it (the other side's copy will then be "
        "invalidated). For fully independent credentials run "
        "`dikw auth login openai-codex`.\n"
    )
    sys.stderr.flush()


# --------------------------------------------------------------------------- #
# Explicit import (CLI: dikw auth import openai-codex)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ImportResult:
    """Outcome of an import / login operation, surfaced to the CLI layer."""

    source_path: Path
    dest_path: Path
    account_id: str | None
    expires_at: int | None  # epoch seconds, from JWT exp claim


def _import_result_for(base: Path, src: Path, tokens: dict[str, str]) -> ImportResult:
    access = tokens["access_token"]
    claims = _decode_jwt_claims(access)
    exp = claims.get("exp")
    return ImportResult(
        source_path=src,
        dest_path=dikw_auth_path(base),
        account_id=account_id_from_jwt(access),
        expires_at=int(exp) if isinstance(exp, int | float) else None,
    )


def import_from_codex_cli(base: Path, *, force: bool = False) -> ImportResult:
    """Copy tokens from ``codex_home()/auth.json`` into the dikw auth store.

    Refuses already-expired tokens unless ``force=True`` (in which case
    the next ``resolve_access_token`` will trigger a refresh; useful when
    only the access_token is expired but refresh_token is still good).
    """
    src = codex_home() / "auth.json"
    if not src.is_file():
        raise CodexAuthError(
            f"No codex CLI credentials at {src}. "
            "Run `codex` once to authenticate, or use "
            "`dikw auth login openai-codex` to skip codex CLI entirely.",
            code="codex_cli_auth_missing",
            relogin_required=True,
        )
    try:
        raw = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CodexAuthError(
            f"Codex CLI auth file at {src} is not valid JSON.",
            code="codex_cli_auth_invalid_json",
            relogin_required=True,
        ) from exc
    tokens_block = raw.get("tokens") if isinstance(raw, dict) else None
    if not isinstance(tokens_block, dict):
        raise CodexAuthError(
            f"Codex CLI auth file at {src} is missing the `tokens` block.",
            code="codex_cli_auth_invalid_shape",
            relogin_required=True,
        )
    access = tokens_block.get("access_token")
    refresh = tokens_block.get("refresh_token")
    if not isinstance(access, str) or not access.strip():
        raise CodexAuthError(
            f"Codex CLI auth file at {src} is missing access_token.",
            code="codex_cli_auth_missing_access_token",
            relogin_required=True,
        )
    if not isinstance(refresh, str) or not refresh.strip():
        raise CodexAuthError(
            f"Codex CLI auth file at {src} is missing refresh_token.",
            code="codex_cli_auth_missing_refresh_token",
            relogin_required=True,
        )
    tokens = {"access_token": access.strip(), "refresh_token": refresh.strip()}
    if not force and _is_expiring(tokens["access_token"], skew_seconds=0):
        raise CodexAuthError(
            f"Codex CLI access_token at {src} has already expired. "
            "Run `codex` to refresh it, then retry, or pass --force to "
            "import anyway (a refresh attempt will run on next use).",
            code="codex_cli_auth_expired",
            relogin_required=True,
        )
    save_codex_tokens(base, tokens)
    return _import_result_for(base, src, tokens)


# --------------------------------------------------------------------------- #
# Device code login (CLI: dikw auth login openai-codex)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DeviceCodeChallenge:
    """User-visible state from step 1 of the device flow."""

    user_code: str
    verification_uri: str
    device_auth_id: str
    poll_interval_seconds: int


def request_device_code() -> DeviceCodeChallenge:
    """Step 1 of the device flow — ask the OAuth issuer for a user code.

    Split out from ``device_code_login`` so the CLI can render the
    challenge before blocking on the polling loop and tests can mock the
    two halves separately.
    """
    import httpx

    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            response = client.post(
                CODEX_OAUTH_DEVICE_USERCODE_URL,
                json={"client_id": CODEX_OAUTH_CLIENT_ID},
                headers={"Content-Type": "application/json"},
            )
    except Exception as exc:
        raise CodexAuthError(
            f"Failed to request device code: {exc}",
            code="device_code_request_failed",
            relogin_required=False,
        ) from exc
    if response.status_code != 200:
        raise CodexAuthError(
            f"Device code request returned status {response.status_code}.",
            code="device_code_request_error",
            relogin_required=False,
        )
    try:
        data = response.json()
    except Exception as exc:
        raise CodexAuthError(
            "Device code response was not valid JSON.",
            code="device_code_invalid_json",
            relogin_required=False,
        ) from exc
    user_code = data.get("user_code")
    device_auth_id = data.get("device_auth_id")
    if not isinstance(user_code, str) or not user_code:
        raise CodexAuthError(
            "Device code response missing user_code.",
            code="device_code_incomplete",
            relogin_required=False,
        )
    if not isinstance(device_auth_id, str) or not device_auth_id:
        raise CodexAuthError(
            "Device code response missing device_auth_id.",
            code="device_code_incomplete",
            relogin_required=False,
        )
    raw_interval = data.get("interval", 5)
    try:
        interval = max(CODEX_DEVICE_POLL_MIN_INTERVAL_SECONDS, int(raw_interval))
    except (TypeError, ValueError):
        interval = CODEX_DEVICE_POLL_MIN_INTERVAL_SECONDS
    return DeviceCodeChallenge(
        user_code=user_code,
        verification_uri=CODEX_OAUTH_DEVICE_VERIFICATION_URL,
        device_auth_id=device_auth_id,
        poll_interval_seconds=interval,
    )


def _poll_for_authorization_code(
    challenge: DeviceCodeChallenge, *, timeout_seconds: int
) -> dict[str, str]:
    """Step 2 — block until the user authorises in the browser.

    Returns ``{"authorization_code": ..., "code_verifier": ...}`` on
    success. Issuer returns 200 + body when the user has completed login;
    403/404 mean "still pending"; anything else is a hard error.
    """
    import httpx

    deadline = time.monotonic() + max(5, timeout_seconds)
    interval = challenge.poll_interval_seconds
    last_error: CodexAuthError | None = None
    with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
        while time.monotonic() < deadline:
            time.sleep(interval)
            try:
                resp = client.post(
                    CODEX_OAUTH_DEVICE_TOKEN_URL,
                    json={
                        "device_auth_id": challenge.device_auth_id,
                        "user_code": challenge.user_code,
                    },
                    headers={"Content-Type": "application/json"},
                )
            except Exception as exc:
                last_error = CodexAuthError(
                    f"Device auth polling network error: {exc}",
                    code="device_code_poll_network",
                    relogin_required=False,
                )
                continue
            if resp.status_code == 200:
                try:
                    body = resp.json()
                except Exception as exc:
                    raise CodexAuthError(
                        "Device auth poll returned invalid JSON.",
                        code="device_code_poll_invalid_json",
                        relogin_required=False,
                    ) from exc
                authorization_code = body.get("authorization_code")
                code_verifier = body.get("code_verifier")
                if not isinstance(authorization_code, str) or not authorization_code:
                    raise CodexAuthError(
                        "Device auth response missing authorization_code.",
                        code="device_code_incomplete_exchange",
                        relogin_required=False,
                    )
                if not isinstance(code_verifier, str) or not code_verifier:
                    raise CodexAuthError(
                        "Device auth response missing code_verifier.",
                        code="device_code_incomplete_exchange",
                        relogin_required=False,
                    )
                return {
                    "authorization_code": authorization_code,
                    "code_verifier": code_verifier,
                }
            if resp.status_code in (403, 404):
                continue
            raise CodexAuthError(
                f"Device auth polling returned status {resp.status_code}.",
                code="device_code_poll_error",
                relogin_required=False,
            )
    if last_error is not None:
        raise last_error
    raise CodexAuthError(
        f"Device login timed out after {timeout_seconds}s.",
        code="device_code_timeout",
        relogin_required=False,
    )


def _exchange_authorization_code(
    authorization_code: str, code_verifier: str
) -> dict[str, str]:
    """Step 3 — swap authorization_code for the access/refresh token pair."""
    import httpx

    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            resp = client.post(
                CODEX_OAUTH_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": authorization_code,
                    "redirect_uri": CODEX_OAUTH_DEVICE_REDIRECT_URI,
                    "client_id": CODEX_OAUTH_CLIENT_ID,
                    "code_verifier": code_verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
    except Exception as exc:
        raise CodexAuthError(
            f"Token exchange request failed: {exc}",
            code="token_exchange_request_failed",
            relogin_required=False,
        ) from exc
    if resp.status_code != 200:
        raise CodexAuthError(
            f"Token exchange returned status {resp.status_code}.",
            code="token_exchange_error",
            relogin_required=False,
        )
    try:
        body = resp.json()
    except Exception as exc:
        raise CodexAuthError(
            "Token exchange response was not valid JSON.",
            code="token_exchange_invalid_json",
            relogin_required=False,
        ) from exc
    access = body.get("access_token") if isinstance(body, dict) else None
    refresh = body.get("refresh_token") if isinstance(body, dict) else None
    if not isinstance(access, str) or not access.strip():
        raise CodexAuthError(
            "Token exchange did not return an access_token.",
            code="token_exchange_no_access_token",
            relogin_required=False,
        )
    if not isinstance(refresh, str) or not refresh.strip():
        raise CodexAuthError(
            "Token exchange did not return a refresh_token.",
            code="token_exchange_no_refresh_token",
            relogin_required=False,
        )
    return {"access_token": access.strip(), "refresh_token": refresh.strip()}


def device_code_login(
    base: Path,
    *,
    on_challenge: Any | None = None,
    timeout_seconds: int = CODEX_DEVICE_LOGIN_TIMEOUT_SECONDS,
) -> ImportResult:
    """Run the full OpenAI device-code OAuth flow and persist tokens.

    ``on_challenge`` (optional callable) is invoked with the
    ``DeviceCodeChallenge`` after step 1 so the CLI layer can render the
    user code + URL before this function blocks in the polling loop.
    Tests pass a no-op to suppress side effects.
    """
    challenge = request_device_code()
    if on_challenge is not None:
        on_challenge(challenge)
    code_pair = _poll_for_authorization_code(challenge, timeout_seconds=timeout_seconds)
    tokens = _exchange_authorization_code(
        code_pair["authorization_code"], code_pair["code_verifier"]
    )
    save_codex_tokens(base, tokens)
    src = Path("(device-code login)")
    return _import_result_for(base, src, tokens)


# --------------------------------------------------------------------------- #
# Status / list / logout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AuthStatus:
    """Snapshot of one provider's credential state for the CLI."""

    provider: str
    path: Path
    exists: bool
    expires_in_seconds: int | None
    last_refresh: str | None
    auth_mode: str | None
    account_id: str | None

    @property
    def expiring_soon(self) -> bool:
        """True when the access_token is within the refresh skew window
        (or its lifetime is unknown). Derived from ``expires_in_seconds``
        so it can never disagree with the displayed countdown."""
        if self.expires_in_seconds is None:
            return True
        return self.expires_in_seconds <= CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS


def _empty_status(provider: str, path: Path, *, exists: bool) -> AuthStatus:
    return AuthStatus(
        provider=provider,
        path=path,
        exists=exists,
        expires_in_seconds=None,
        last_refresh=None,
        auth_mode=None,
        account_id=None,
    )


def auth_status(base: Path, *, provider: str = _PROVIDER_KEY) -> AuthStatus:
    """Snapshot the auth state for a provider in the dikw store."""
    path = dikw_auth_path(base)
    try:
        store = _load_store(path)
    except CodexAuthError:
        # Corrupt / unsupported store — file exists but unusable. The
        # ``expiring_soon`` property already reports True for unknown
        # expiry, so no extra signalling needed here.
        return _empty_status(provider, path, exists=True)

    node = store.get("providers", {}).get(provider)
    if not isinstance(node, dict):
        # File may exist (e.g. after logout, or when only another provider
        # has credentials), but for *this* provider there's nothing
        # configured — surface that to the CLI as a missing-credentials
        # state so scripts can branch on `dikw auth status` exit code.
        return _empty_status(provider, path, exists=False)

    tokens = node.get("tokens")
    access = tokens.get("access_token") if isinstance(tokens, dict) else None
    expires_in: int | None = None
    account: str | None = None
    if isinstance(access, str) and access.strip():
        claims = _decode_jwt_claims(access)
        exp = claims.get("exp")
        if isinstance(exp, int | float):
            expires_in = max(0, int(float(exp) - time.time()))
        account_value = claims.get("chatgpt_account_id")
        if isinstance(account_value, str) and account_value:
            account = account_value
    last_refresh = node.get("last_refresh")
    auth_mode = node.get("auth_mode")
    return AuthStatus(
        provider=provider,
        path=path,
        exists=True,
        expires_in_seconds=expires_in,
        last_refresh=last_refresh if isinstance(last_refresh, str) else None,
        auth_mode=auth_mode if isinstance(auth_mode, str) else None,
        account_id=account,
    )


def list_providers(base: Path) -> list[str]:
    """Return the list of providers with non-empty token blocks in the
    dikw auth store, sorted alphabetically."""
    path = dikw_auth_path(base)
    if not path.is_file():
        return []
    try:
        store = _load_store(path)
    except CodexAuthError:
        return []
    providers = store.get("providers", {})
    if not isinstance(providers, dict):
        return []
    out: list[str] = []
    for name, node in providers.items():
        if not isinstance(node, dict):
            continue
        tokens = node.get("tokens")
        if not isinstance(tokens, dict):
            continue
        if not tokens.get("access_token") or not tokens.get("refresh_token"):
            continue
        out.append(str(name))
    return sorted(out)


def logout(base: Path, *, provider: str = _PROVIDER_KEY) -> bool:
    """Remove ``provider``'s entry from the dikw auth store.

    Returns ``True`` when an entry was actually removed, ``False`` when
    nothing was there. Other providers' entries are preserved. The auth
    store file itself is left in place even if it ends up empty — that
    keeps lazy migration from re-importing codex CLI tokens behind the
    user's back after an explicit logout.
    """
    path = dikw_auth_path(base)
    if not path.is_file():
        return False
    with _auth_file_lock(
        dikw_auth_lock_path(base), timeout=CODEX_AUTH_LOCK_TIMEOUT_SECONDS
    ):
        try:
            store = _load_store(path)
        except CodexAuthError:
            return False
        providers = store.get("providers")
        if not isinstance(providers, dict) or provider not in providers:
            return False
        del providers[provider]
        _atomic_write_store(path, store)
    return True
