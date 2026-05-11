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

import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, cast

import httpx

from .config import ClientConfig

# Match the server's heartbeat event type name; we drop these silently
# rather than handing them to the renderer because they carry no state
# beyond "the connection is still alive."
_HEARTBEAT_TYPE = "heartbeat"
# Long enough to cover slow first-byte from the server's task setup
# (engine boot + storage migration on cold start) but short enough that
# a wedged endpoint doesn't hang the CLI for minutes.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=5.0)


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
        try:
            resp = await self._client.get(
                path, params=params, headers=self._headers()
            )
        except httpx.RequestError as e:
            raise _network_error(e) from e
        return _parse_json_response(resp)

    async def post_json(
        self,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
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
