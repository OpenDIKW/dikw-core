"""build_app must tolerate a partial OTel install.

``OTEL_AVAILABLE`` only checks the OTel API. When the API is installed but
``opentelemetry-instrumentation-fastapi`` is not (a partial / manual install),
``build_app`` must degrade HTTP-span instrumentation off rather than raise
``ModuleNotFoundError`` — the same graceful posture ``configure_telemetry``
takes for the SDK/exporter.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from dikw_core import telemetry
from dikw_core.server.app import build_app
from dikw_core.server.auth import AuthConfig
from dikw_core.server.runtime import ServerRuntime

_BLOCKED = "opentelemetry.instrumentation.fastapi"


class _Blocker:
    """meta_path finder that makes one module import-fail."""

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name == _BLOCKED:
            raise ModuleNotFoundError(_BLOCKED)
        return None


async def _no_factory() -> ServerRuntime:
    raise AssertionError("runtime factory must not run during build")


@pytest.mark.skipif(not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra")
def test_build_app_survives_missing_fastapi_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, _BLOCKED, raising=False)
    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        # instrument_telemetry=True forces the import path so the guard is
        # actually exercised (not skipped because telemetry is off).
        app = build_app(
            runtime_factory=_no_factory,
            auth=AuthConfig(host="127.0.0.1", token=None),
            instrument_telemetry=True,
        )
        assert app is not None
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.pop(_BLOCKED, None)


@pytest.mark.skipif(not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra")
def test_build_app_instruments_only_when_telemetry_enabled() -> None:
    """The FastAPI HTTP-span middleware is wired only when telemetry is on, so a
    disabled server carries no middleware (no per-request cost, no spans to a
    foreign provider)."""
    auth = AuthConfig(host="127.0.0.1", token=None)
    off = build_app(
        runtime_factory=_no_factory, auth=auth, instrument_telemetry=False
    )
    assert getattr(off, "_is_instrumented_by_opentelemetry", False) is False
    on = build_app(
        runtime_factory=_no_factory, auth=auth, instrument_telemetry=True
    )
    assert getattr(on, "_is_instrumented_by_opentelemetry", False) is True
