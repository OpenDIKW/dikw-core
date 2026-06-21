"""Thin httpx wrapper used by every ``dikw client`` subcommand.

Responsibilities:

* Apply the bearer token to every request.
* Map server error envelopes (``{"error": {"code", "message", "detail"}}``)
  to a single :class:`ClientError` exception so callers can branch on
  ``code`` without parsing JSON themselves.
* Stream NDJSON responses as an async iterator of decoded events,
  swallowing the server's heartbeat events so the renderer doesn't have
  to (heartbeat is purely transport-keepalive, never carries state).

Things this module deliberately does NOT do:

* No retries — the engine endpoints are either fast (sync) or already
  idempotent + resumable via task event ``from_seq``. A retry layer here
  would only mask real network or auth issues.
* No streaming uploads — the multipart upload helper lives in
  ``client/importer.py`` because it owns the manifest + tar.gz packing
  logic; the transport just wraps the bytes.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from types import TracebackType
from typing import Any, cast

import httpx

from .config import ClientConfig

# Match the server's heartbeat event type name; we drop these silently
# rather than handing them to the renderer because they carry no state
# beyond "the connection is still alive."
_HEARTBEAT_TYPE = "heartbeat"
# Escape hatch for the version handshake: set truthy to downgrade a
# detected client/server version mismatch from a hard fail to a one-line
# stderr warning (deliberate mixed-version debugging).
_SKEW_ALLOW_ENV = "DIKW_ALLOW_VERSION_SKEW"
# Long enough to cover slow first-byte from the server's task setup
# (engine boot + storage migration on cold start) but short enough that
# a wedged endpoint doesn't hang the CLI for minutes.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=5.0)


def _installed_version() -> str | None:
    """The ``dikw-core`` version this client is installed at, or ``None``
    when running from an uninstalled source checkout. Layering-clean —
    reads package metadata via stdlib, never imports the engine."""
    try:
        return _pkg_version("dikw-core")
    except PackageNotFoundError:
        return None


def _skew_allowed() -> bool:
    return os.environ.get(_SKEW_ALLOW_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class ClientError(Exception):
    """Raised for any non-2xx response from the server.

    ``code`` is the server's stable error code (e.g. ``not_found``,
    ``unauthorized``, ``bad_request``); CLI code branches on it without
    parsing the message. ``status`` is the HTTP status code, useful for
    transport-layer decisions (auth vs. validation vs. server bug).
    """

    def __init__(
        self,
        *,
        status: int,
        code: str,
        message: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail


class Transport:
    """One-per-CLI-command wrapper around ``httpx.AsyncClient``.

    Construct via ``Transport.from_config(...)`` (or pass a custom
    ``httpx.AsyncClient`` for tests using ``ASGITransport``). Use as an
    async context manager so the underlying connection pool is closed
    even on error.
    """

    def __init__(self, *, client: httpx.AsyncClient, token: str | None) -> None:
        self._client = client
        self._token = token
        # The version handshake runs at most once per instance, before the
        # first request actually reaches the server (see _ensure_version_compat).
        # The lock serializes concurrent first-callers (e.g. the asyncio.gather
        # in _gather_task_results, which shares one Transport) so none slips its
        # real request past an in-flight probe — without it a skew refusal could
        # land after the racing request already reached the skewed server.
        self._version_checked = False
        self._version_lock = asyncio.Lock()

    @classmethod
    def from_config(
        cls,
        cfg: ClientConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> Transport:
        if client is None:
            client = httpx.AsyncClient(
                base_url=cfg.server_url,
                timeout=_DEFAULT_TIMEOUT,
            )
        return cls(client=client, token=cfg.token)

    async def __aenter__(self) -> Transport:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    # ---- version handshake --------------------------------------------

    async def _ensure_version_compat(self) -> None:
        """Probe ``GET /v1/info`` once and compare the server's
        ``engine_version`` to this client's installed version.

        Karpathy's rule applies to the failure policy: a *positive*
        mismatch is real drift worth stopping for (dikw-core is alpha —
        breaking changes land in any minor, so a skewed client/server pair
        can silently misbehave), but anything *ambiguous* — server
        unreachable, ``/v1/info`` non-200, field missing, or the client
        running from an uninstalled checkout — must NOT raise a false
        skew. We skip silently in every ambiguous case and let the real
        request surface its own error through the normal channel.

        Hard-fails on a confirmed mismatch unless ``DIKW_ALLOW_VERSION_SKEW``
        is set, which downgrades it to a one-line stderr warning.

        Concurrency: the probe runs under ``self._version_lock`` so that a
        burst of concurrent first-requests on one Transport (e.g.
        ``_gather_task_results``' ``asyncio.gather``) all block on the
        verdict instead of the second caller seeing a half-set flag and
        racing its real request through. ``_version_checked`` is set only
        after a non-raising probe, so on a skew refusal the flag stays
        False and every queued caller is likewise refused (each re-probes;
        acceptable on the terminal failure path).
        """
        if self._version_checked:
            return
        async with self._version_lock:
            if self._version_checked:
                return
            await self._probe_version_compat()
            self._version_checked = True

    async def _probe_version_compat(self) -> None:
        """Single ``/v1/info`` probe + comparison. Returns on every
        ambiguous case (so the handshake never raises a false skew);
        raises ``ClientError(version_skew)`` only on a confirmed mismatch
        without the allow-env override. Always called while holding
        ``self._version_lock``."""
        client_ver = _installed_version()
        if client_ver is None:
            return
        try:
            resp = await self._client.get("/v1/info", headers=self._headers())
        except httpx.RequestError:
            return  # unreachable — let the real request raise network_error
        if resp.status_code != 200:
            return
        try:
            body = resp.json()
        except json.JSONDecodeError:
            return
        server_ver = body.get("engine_version") if isinstance(body, dict) else None
        if not isinstance(server_ver, str) or not server_ver:
            return
        if server_ver == client_ver:
            return

        message = (
            f"dikw client is {client_ver} but the server at this URL is "
            f"{server_ver}; their wire contract may have drifted (dikw-core "
            f"is alpha — breaking changes land in any minor). Pin both to the "
            f"same release, or set {_SKEW_ALLOW_ENV}=1 to proceed anyway."
        )
        if _skew_allowed():
            print(f"warning: version skew — {message}", file=sys.stderr)
            return
        raise ClientError(status=0, code="version_skew", message=message)

    # ---- request primitives -------------------------------------------

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if extra:
            headers.update(extra)
        return headers

    async def get_json(
        self, path: str, *, params: Mapping[str, Any] | None = None
    ) -> Any:
        await self._ensure_version_compat()
        try:
            resp = await self._client.get(
                path, params=params, headers=self._headers()
            )
        except httpx.RequestError as e:
            raise _network_error(e) from e
        return _parse_json_response(resp)

    async def get_bytes(
        self, path: str, *, params: Mapping[str, Any] | None = None
    ) -> bytes:
        """Fetch a binary response body — buffers fully in memory.

        Binary counterpart to :meth:`get_json`. Server error envelopes
        flow through :class:`ClientError` exactly like the JSON path,
        so callers branch on ``code``.
        """
        await self._ensure_version_compat()
        try:
            resp = await self._client.get(
                path, params=params, headers=self._headers()
            )
        except httpx.RequestError as e:
            raise _network_error(e) from e
        if resp.status_code >= 400:
            _raise_for_error(resp)
        return resp.content

    async def post_json(
        self,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        await self._ensure_version_compat()
        try:
            resp = await self._client.post(
                path,
                json=dict(json_body) if json_body is not None else None,
                params=params,
                headers=self._headers(),
            )
        except httpx.RequestError as e:
            raise _network_error(e) from e
        return _parse_json_response(resp)

    async def post_multipart(
        self,
        path: str,
        *,
        files: Mapping[str, tuple[str, Any, str]],
        data: Mapping[str, str] | None = None,
    ) -> Any:
        """Send a multipart POST (sources import).

        ``files`` matches httpx's ``files=`` shape: ``name -> (filename,
        fileobj, content_type)``. The transport doesn't own the file
        objects — caller is responsible for closing them after the call
        returns.
        """
        await self._ensure_version_compat()
        try:
            resp = await self._client.post(
                path,
                files=cast(Any, files),
                data=cast(Any, dict(data)) if data is not None else None,
                headers=self._headers(),
            )
        except httpx.RequestError as e:
            raise _network_error(e) from e
        return _parse_json_response(resp)

    async def get_task_events_page(
        self,
        task_id: str,
        *,
        from_seq: int = 0,
        limit: int = 100,
        wait: int = 0,
    ) -> dict[str, Any]:
        """Cursor page from ``GET /v1/tasks/{task_id}/events``.

        Returns the server's ``EventsPage`` JSON as a plain dict — the
        agent paging primitive used by ``dikw client tasks events`` and
        the building block for :func:`follow_to_terminal`. ``wait=0`` is
        a snapshot; ``wait>0`` is a server-side long-poll (server caps
        at 60s). Errors flow through the standard ``ClientError`` channel.

        The httpx read timeout is widened to ``wait + 15s`` for this
        request so the server's full long-poll hold doesn't race the
        client's normal 60s read timeout (a ``wait=60`` call with even
        a few hundred ms of scheduling overhead would otherwise raise
        ``network_error`` instead of returning an empty page).
        """
        await self._ensure_version_compat()
        params = {"from_seq": from_seq, "limit": limit, "wait": wait}
        timeout: httpx.Timeout | None = None
        if wait > 0:
            timeout = httpx.Timeout(
                connect=5.0,
                read=float(wait) + 15.0,
                write=60.0,
                pool=5.0,
            )
        try:
            resp = await self._client.get(
                f"/v1/tasks/{task_id}/events",
                params=params,
                headers=self._headers(),
                timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
            )
        except httpx.RequestError as e:
            raise _network_error(e) from e
        page = _parse_json_response(resp)
        return cast(dict[str, Any], page)

    @asynccontextmanager
    async def stream_ndjson(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[AsyncIterator[dict[str, Any]]]:
        """Open an NDJSON stream and yield decoded events.

        Heartbeat events are dropped. If the server returns 4xx/5xx, the
        body is parsed eagerly and re-raised as ``ClientError`` before
        the iterator yields anything — that way the caller's renderer
        never sees a partial stream from a failed request.

        Network failures at every stage — connect (``__aenter__``),
        first-byte (``aiter_lines`` lazy connect), or mid-stream socket
        drop — are funnelled through ``_network_error`` so streaming
        commands like ``query``/``ingest``/``synth``/``tasks follow``
        never leak a raw httpx traceback to the operator.
        """
        await self._ensure_version_compat()
        try:
            async with self._client.stream(
                method,
                path,
                json=dict(json_body) if json_body is not None else None,
                params=params,
                headers=self._headers(),
            ) as resp:
                if resp.status_code >= 400:
                    # Drain the body so we can include the server's error
                    # envelope; ``aread`` materialises it from the streaming
                    # response without leaving the connection half-read.
                    await resp.aread()
                    _raise_for_error(resp)

                async def _iter() -> AsyncIterator[dict[str, Any]]:
                    try:
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                # The server only ever emits well-formed JSON
                                # lines; anything else is a transport-layer
                                # corruption (e.g. a reverse proxy injecting
                                # text). Surface it as a ClientError instead of
                                # silently dropping it.
                                raise ClientError(
                                    status=resp.status_code,
                                    code="invalid_ndjson",
                                    message=f"non-JSON line in stream: {line!r}",
                                ) from None
                            if (
                                isinstance(event, dict)
                                and event.get("type") == _HEARTBEAT_TYPE
                            ):
                                continue
                            if not isinstance(event, dict):
                                raise ClientError(
                                    status=resp.status_code,
                                    code="invalid_ndjson",
                                    message=f"non-object NDJSON event: {event!r}",
                                )
                            yield event
                    except httpx.RequestError as e:
                        raise _network_error(e) from e

                yield _iter()
        except httpx.RequestError as e:
            raise _network_error(e) from e


def _parse_json_response(resp: httpx.Response) -> Any:
    """Decode a JSON response, mapping server errors to :class:`ClientError`.

    Empty 204 bodies are returned as ``None`` so callers can treat them
    uniformly without checking the status code.
    """
    if resp.status_code >= 400:
        _raise_for_error(resp)
    if resp.status_code == 204 or not resp.content:
        return None
    try:
        return resp.json()
    except json.JSONDecodeError as e:
        raise ClientError(
            status=resp.status_code,
            code="invalid_response",
            message=f"server returned non-JSON: {resp.text[:200]!r}",
        ) from e


def _network_error(exc: httpx.RequestError) -> ClientError:
    """Wrap a transport-level failure (DNS, refused, timeout, dropped
    socket) so every CLI command surfaces it through the same
    ``ClientError`` channel as a server-side error envelope, instead of
    leaking an httpx traceback to stderr. ``status=0`` distinguishes
    network errors from any real HTTP status; the code is stable enough
    for shell scripts to branch on without parsing the message.
    """
    return ClientError(
        status=0,
        code="network_error",
        message=f"could not reach server: {exc.__class__.__name__}: {exc}",
    )


def _raise_for_error(resp: httpx.Response) -> None:
    """Translate the server's ``{"error": {...}}`` envelope to ClientError.

    The envelope shape is fixed by ``dikw_core.server.errors``, so we
    can decode it without a generic fallback path. Bodies that don't
    match (e.g. uvicorn's bare 404 before the app loads) still produce a
    ClientError — just with a synthetic ``code`` derived from the
    status.
    """
    try:
        body = resp.json()
    except json.JSONDecodeError:
        body = None
    err: dict[str, Any] | None = None
    if isinstance(body, dict):
        candidate = body.get("error")
        if isinstance(candidate, dict):
            err = candidate
    if err is None:
        raise ClientError(
            status=resp.status_code,
            code=f"http_{resp.status_code}",
            message=resp.text[:200] or f"HTTP {resp.status_code}",
        )
    detail_obj = err.get("detail")
    detail = detail_obj if isinstance(detail_obj, dict) else None
    raise ClientError(
        status=resp.status_code,
        code=str(err.get("code") or f"http_{resp.status_code}"),
        message=str(err.get("message") or resp.text[:200]),
        detail=detail,
    )


__all__ = ["ClientError", "Transport"]
