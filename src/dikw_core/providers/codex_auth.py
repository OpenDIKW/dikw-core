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
import time
from pathlib import Path
from typing import Any

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120


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
