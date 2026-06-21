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

import httpx
import pytest

from dikw_core.client import transport as transport_mod
from dikw_core.client.transport import ClientError, Transport

_ALLOW_ENV = "DIKW_ALLOW_VERSION_SKEW"


def _mock_transport(
    *,
    server_version: str | None,
    info_status: int = 200,
    info_raises: bool = False,
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
