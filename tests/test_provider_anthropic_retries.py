"""Verify the Anthropic-compat provider threads ``max_retries`` to the SDK.

The Anthropic SDK takes ``max_retries`` at client construction (not per call)
and retries on 408/409/429/5xx with exponential backoff + jitter. We surface
this as ``ProviderConfig.llm_max_retries`` so vendors with occasional
overloads (MiniMax 529) get more headroom than the SDK default of 2 without
pulling in a third-party retry layer.
"""

from __future__ import annotations

import pytest

from dikw_core.providers import build_llm
from dikw_core.providers.anthropic_compat import AnthropicCompatLLM

from .fakes import make_provider_cfg


@pytest.fixture()
def captured_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setattr("anthropic.AsyncAnthropic", FakeAsyncAnthropic)
    return captured


def test_anthropic_client_passes_max_retries_when_set(
    captured_kwargs: dict[str, object],
) -> None:
    llm = AnthropicCompatLLM(
        api_key_env="ANTHROPIC_API_KEY", api_key="sk-explicit", max_retries=7
    )
    llm._get_client()
    kwargs = captured_kwargs["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("max_retries") == 7


def test_anthropic_client_omits_max_retries_when_none(
    captured_kwargs: dict[str, object],
) -> None:
    """If the caller doesn't override, let the SDK pick its default.

    The SDK's default (2) stays in control; we only override when the user
    has explicitly asked for it through ``ProviderConfig``.
    """
    llm = AnthropicCompatLLM(api_key_env="ANTHROPIC_API_KEY", api_key="sk-explicit")
    llm._get_client()
    kwargs = captured_kwargs["kwargs"]
    assert isinstance(kwargs, dict)
    assert "max_retries" not in kwargs


def test_build_llm_wires_max_retries_from_config(
    captured_kwargs: dict[str, object],
) -> None:
    cfg = make_provider_cfg(llm="anthropic_compat", llm_max_retries=4)
    llm = build_llm(cfg)
    assert isinstance(llm, AnthropicCompatLLM)
    llm._get_client()
    kwargs = captured_kwargs["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("max_retries") == 4
