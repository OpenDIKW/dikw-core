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


@pytest.mark.skipif(not telemetry.OTEL_AVAILABLE, reason="requires the [otel] extra")
def test_build_app_survives_missing_fastapi_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, _BLOCKED, raising=False)
    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        async def _factory() -> ServerRuntime:
            raise AssertionError("runtime factory must not run during build")

        app = build_app(
            runtime_factory=_factory,
            auth=AuthConfig(host="127.0.0.1", token=None),
        )
        assert app is not None
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.pop(_BLOCKED, None)
