"""Shared httpx client configuration for the LLM providers.

All three providers (anthropic_compat, openai_compat, openai_codex) face
the same wire-level failure mode and apply the same workaround: hand the
vendor SDK a custom ``httpx.AsyncClient`` with connection keepalive
disabled and per-leg bounded timeouts. Provider endpoints commonly used
for batch embedding (Gitee AI's Qwen3-* family in particular) silently
drop idle TCP keepalives mid-batch; the SDK's retry path then loops on
the same dead socket from the pool until the read timeout fires N+1
times. Forcing a fresh connection per request adds ~50ms TLS handshake
overhead but eliminates the multi-minute-per-batch retry storm. The
same pattern protects the codex provider from Cloudflare's idle drops.
"""

from __future__ import annotations

import httpx

DEFAULT_TIMEOUT_SECONDS = 60.0


def build_no_keepalive_async_client(
    timeout_seconds: float | None = None,
    *,
    default_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[httpx.Timeout, httpx.AsyncClient]:
    """Return ``(timeout, async_client)`` configured for vendor LLM SDKs.

    Both values are passed to the SDK constructor: ``timeout`` so the
    SDK applies its own retry policy on a per-leg deadline, and
    ``http_client`` so the SDK reuses our keepalive-disabled client
    instead of allocating its own.
    """
    seconds = timeout_seconds if timeout_seconds is not None else default_seconds
    timeout = httpx.Timeout(connect=10.0, read=seconds, write=seconds, pool=5.0)
    client = httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(max_keepalive_connections=0),
    )
    return timeout, client
