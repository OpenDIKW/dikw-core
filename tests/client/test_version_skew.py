"""Client/server version-handshake behaviour at the transport layer.

The transport probes ``GET /v1/info`` once per :class:`Transport` instance
and compares the server's ``engine_version`` to the client's own installed
``dikw-core`` version. A positive mismatch hard-fails the command (so a
downstream debugging silent wire drift sees it immediately); anything
ambiguous (server unreachable, ``/v1/info`` non-200, field missing) is
skipped so we never raise a *false* skew. ``DIKW_ALLOW_VERSION_SKEW=1``
downgrades the hard-fail to a one-line stderr warning for deliberate
mixed-version debugging.

These tests drive a ``httpx.MockTransport`` so both versions are fully
controlled — the real in-memory server always reports the same version
the client is installed at, which is exactly the no-skew path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from dikw_core.cli import app
from dikw_core.client import transport as transport_mod
from dikw_core.client.transport import ClientError, Transport
from dikw_core.server.runtime import ServerRuntime

_ALLOW_ENV = "DIKW_ALLOW_VERSION_SKEW"


def _mock_transport(
    *,
    server_version: str | None,
    info_status: int = 200,
    info_raises: bool = False,
    info_non_json: bool = False,
    record: list[str] | None = None,
) -> httpx.MockTransport:
    """A mock backend: ``/v1/info`` reports ``server_version`` (or errors),
    ``/v1/status`` always succeeds with a sentinel body."""

    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request.url.path)
        if request.url.path == "/v1/info":
            if info_raises:
                raise httpx.ConnectError("boom", request=request)
            if info_non_json:
                return httpx.Response(
                    200, content=b"<html>not json</html>", headers={}
                )
            if info_status != 200:
                return httpx.Response(
                    info_status,
                    json={"error": {"code": "boom", "message": "down"}},
                )
            body: dict[str, object] = {}
            if server_version is not None:
                body["engine_version"] = server_version
            return httpx.Response(200, json=body)
        if request.url.path == "/v1/status":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(
            404, json={"error": {"code": "not_found", "message": "nope"}}
        )

    return httpx.MockTransport(handler)


def _transport(mock: httpx.MockTransport) -> Transport:
    client = httpx.AsyncClient(base_url="http://srv", transport=mock)
    return Transport(client=client, token="t")


@pytest.mark.asyncio
async def test_version_skew_hard_fails_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transport_mod, "_installed_version", lambda: "0.5.0")
    monkeypatch.delenv(_ALLOW_ENV, raising=False)
    t = _transport(_mock_transport(server_version="0.6.0"))
    try:
        with pytest.raises(ClientError) as excinfo:
            await t.get_json("/v1/status")
        assert excinfo.value.code == "version_skew"
        # The message must name BOTH versions — that pair IS the diagnostic.
        assert "0.5.0" in excinfo.value.message
        assert "0.6.0" in excinfo.value.message
        assert _ALLOW_ENV in excinfo.value.message
    finally:
        await t.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_version_skew_allowed_warns_and_proceeds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(transport_mod, "_installed_version", lambda: "0.5.0")
    monkeypatch.setenv(_ALLOW_ENV, "1")
    t = _transport(_mock_transport(server_version="0.6.0"))
    try:
        got = await t.get_json("/v1/status")
        assert got == {"ok": True}  # request proceeded despite skew
        assert "skew" in capsys.readouterr().err.lower()
    finally:
        await t.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_matching_versions_are_silent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(transport_mod, "_installed_version", lambda: "0.6.0")
    monkeypatch.delenv(_ALLOW_ENV, raising=False)
    t = _transport(_mock_transport(server_version="0.6.0"))
    try:
        got = await t.get_json("/v1/status")
        assert got == {"ok": True}
        assert capsys.readouterr().err == ""
    finally:
        await t.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_info_unreachable_skips_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``/v1/info`` can't be reached the check is skipped — the real
    request surfaces its own error, never a spurious ``version_skew``."""
    monkeypatch.setattr(transport_mod, "_installed_version", lambda: "0.5.0")
    monkeypatch.delenv(_ALLOW_ENV, raising=False)
    t = _transport(_mock_transport(server_version="0.6.0", info_raises=True))
    try:
        got = await t.get_json("/v1/status")
        assert got == {"ok": True}  # check skipped, request went through
    finally:
        await t.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_info_non_200_skips_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transport_mod, "_installed_version", lambda: "0.5.0")
    monkeypatch.delenv(_ALLOW_ENV, raising=False)
    t = _transport(_mock_transport(server_version="0.6.0", info_status=503))
    try:
        got = await t.get_json("/v1/status")
        assert got == {"ok": True}
    finally:
        await t.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_missing_engine_version_field_skips_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transport_mod, "_installed_version", lambda: "0.5.0")
    monkeypatch.delenv(_ALLOW_ENV, raising=False)
    t = _transport(_mock_transport(server_version=None))  # no engine_version key
    try:
        got = await t.get_json("/v1/status")
        assert got == {"ok": True}
    finally:
        await t.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_non_json_info_body_skips_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 200 ``/v1/info`` whose body isn't JSON (a misbehaving reverse
    proxy) is ambiguous — skip rather than crash or raise a false skew."""
    monkeypatch.setattr(transport_mod, "_installed_version", lambda: "0.5.0")
    monkeypatch.delenv(_ALLOW_ENV, raising=False)
    t = _transport(_mock_transport(server_version="0.6.0", info_non_json=True))
    try:
        got = await t.get_json("/v1/status")
        assert got == {"ok": True}
    finally:
        await t.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_check_runs_once_per_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two requests on one Transport must probe ``/v1/info`` exactly once."""
    monkeypatch.setattr(transport_mod, "_installed_version", lambda: "0.6.0")
    monkeypatch.delenv(_ALLOW_ENV, raising=False)
    record: list[str] = []
    t = _transport(_mock_transport(server_version="0.6.0", record=record))
    try:
        await t.get_json("/v1/status")
        await t.get_json("/v1/status")
        assert record.count("/v1/info") == 1
        assert record.count("/v1/status") == 2
    finally:
        await t.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_concurrent_first_requests_all_refused_on_skew(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first-requests on one Transport (the ``asyncio.gather``
    shape in ``_gather_task_results``) must ALL be refused on skew — none
    may slip its real request to the skewed server before the verdict."""
    monkeypatch.setattr(transport_mod, "_installed_version", lambda: "0.5.0")
    monkeypatch.delenv(_ALLOW_ENV, raising=False)
    record: list[str] = []
    t = _transport(_mock_transport(server_version="0.6.0", record=record))
    try:
        results = await asyncio.gather(
            t.get_json("/v1/status"),
            t.get_json("/v1/status"),
            t.get_json("/v1/status"),
            return_exceptions=True,
        )
        assert all(
            isinstance(r, ClientError) and r.code == "version_skew"
            for r in results
        ), results
        # The skewed server was probed but NO real /v1/status ever landed.
        assert "/v1/status" not in record
    finally:
        await t.__aexit__(None, None, None)


def test_installed_version_none_when_package_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uninstalled source checkout (no distribution metadata) yields
    ``None`` so the handshake skips rather than crashing."""

    def _raise(_name: str) -> str:
        raise PackageNotFoundError("dikw-core")

    monkeypatch.setattr(transport_mod, "_pkg_version", _raise)
    assert transport_mod._installed_version() is None


@pytest.mark.asyncio
async def test_none_client_version_skips_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the client can't determine its own version the check is
    skipped — a server on any version is accepted, no false skew."""
    monkeypatch.setattr(transport_mod, "_installed_version", lambda: None)
    monkeypatch.delenv(_ALLOW_ENV, raising=False)
    record: list[str] = []
    t = _transport(_mock_transport(server_version="0.6.0", record=record))
    try:
        got = await t.get_json("/v1/status")
        assert got == {"ok": True}
        # client version unknown → we don't even probe /v1/info.
        assert "/v1/info" not in record
    finally:
        await t.__aexit__(None, None, None)


def test_on_error_labels_version_skew_distinctly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``_on_error`` must label a ``version_skew`` (also ``status==0``)
    distinctly from a plain ``network_error`` so the operator isn't told
    their server is unreachable when it's really a version mismatch."""
    from dikw_core.client import cli_app

    cli_app._on_error(ClientError(status=0, code="network_error", message="down"))
    assert "network error" in capsys.readouterr().out.lower()

    cli_app._on_error(ClientError(status=0, code="version_skew", message="drift"))
    out = capsys.readouterr().out.lower()
    assert "version skew" in out
    assert "network error" not in out


def test_cli_command_hard_fails_on_skew(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a ``dikw client`` command against a server whose
    version differs from the client's exits non-zero with a clear
    ``version skew`` message (covers the transport raise + the
    ``_on_error`` rendering branch together)."""
    patch_transport_factory()
    # The in-memory server reports the real installed version; force a
    # mismatch from the client side with a sentinel that never equals it.
    monkeypatch.setattr(
        transport_mod, "_installed_version", lambda: "0.0.0-skew-test"
    )
    monkeypatch.delenv(_ALLOW_ENV, raising=False)
    result = CliRunner().invoke(app, ["client", "status"])
    assert result.exit_code != 0, result.output
    assert "version skew" in result.output.lower()
