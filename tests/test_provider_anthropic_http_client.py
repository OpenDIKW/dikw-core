"""Build the REAL ``anthropic.AsyncAnthropic`` client — no SDK stub.

The other ``test_provider_anthropic_*`` files swap ``AsyncAnthropic`` for a
kwargs-recording stub, so they can't see the SDK reject what we hand it.
That's how #279 shipped: ``anthropic`` 1.x moved its transport to ``httpx2``
and raises ``TypeError`` on the ``httpx.AsyncClient`` that ``_get_client``
passes whenever ``timeout_seconds`` is set — i.e. on every default config
(``ProviderConfig.llm_timeout_seconds`` defaults to 120). ``pyproject.toml``
caps ``anthropic<1``; this fails loudly if the cap is lifted without
migrating the client construction.
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
    # The default timeout is what routes construction through ``http_client``.
    assert cfg.llm_timeout_seconds is not None

    llm = build_llm(cfg)
    assert isinstance(llm, AnthropicCompatLLM)
    client = llm._get_client()
    try:
        assert isinstance(client, AsyncAnthropic)
        assert client.timeout.read == cfg.llm_timeout_seconds
    finally:
        await client.close()
