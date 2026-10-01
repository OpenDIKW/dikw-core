"""Regression guard for #279: the real ``AsyncAnthropic`` must accept our client.

``build_llm`` always threads ``llm_timeout_seconds`` (default 120), so
``_get_client`` always hands the SDK an ``httpx.AsyncClient``. ``anthropic``
1.x moved to ``httpx2`` and raises ``TypeError`` on it. CI missed that because
``uv.lock`` pinned 0.x while the published wheel's open range resolved 1.x;
``pyproject.toml`` now caps ``anthropic<1``. This goes red once the lock
resolves ``anthropic>=1`` — so lifting the cap needs both
``uv lock --upgrade-package anthropic`` and moving that client off ``httpx``.
"""

from __future__ import annotations

import pytest
from anthropic import AsyncAnthropic

from dikw_core.providers import build_llm
from dikw_core.providers.anthropic_compat import AnthropicCompatLLM

from .fakes import make_provider_cfg


async def test_default_config_builds_real_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    cfg = make_provider_cfg(llm="anthropic_compat")

    llm = build_llm(cfg)
    assert isinstance(llm, AnthropicCompatLLM)
    client = llm._get_client()
    try:
        assert isinstance(client, AsyncAnthropic)
        assert client.timeout.read == cfg.llm_timeout_seconds
    finally:
        await client.close()
