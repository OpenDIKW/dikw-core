"""Verify the openai_compat providers thread ``max_retries`` per leg.

LLM and embedding legs carry **independent** retry budgets because they
frequently point at different vendors (MiniMax LLM + Gitee AI embeddings,
Anthropic LLM + OpenAI embeddings). A single shared value would force a
stricter-than-needed cap on whichever vendor was less flaky, or conversely
waste retries against the flakier one.
"""

from __future__ import annotations

from typing import Any

import pytest

from dikw_core.providers import build_embedder, build_llm
from dikw_core.providers.openai_compat import OpenAICompatEmbeddings, OpenAICompatLLM

from .fakes import make_provider_cfg


@pytest.fixture()
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    rec: dict[str, Any] = {"init_kwargs": None}

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> Any:  # pragma: no cover - unused
            return type("Resp", (), {"choices": [], "usage": None})()

    class FakeEmbeddings:
        async def create(self, **kwargs: Any) -> Any:
            n = len(kwargs.get("input", []) or [])
            return type(
                "Resp",
                (),
                {"data": [type("Row", (), {"embedding": [0.0, 1.0]})() for _ in range(n)]},
            )()

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            rec["init_kwargs"] = kwargs
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    return rec


async def test_llm_client_passes_max_retries_when_set(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAICompatLLM(
        api_key_env="OPENAI_API_KEY", base_url="http://fake.example/v1", max_retries=6
    )
    provider._get_client()
    init = captured["init_kwargs"]
    assert init is not None
    assert init.get("max_retries") == 6


async def test_llm_client_omits_max_retries_when_none(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAICompatLLM(
        api_key_env="OPENAI_API_KEY", base_url="http://fake.example/v1"
    )
    provider._get_client()
    init = captured["init_kwargs"]
    assert init is not None
    assert "max_retries" not in init


async def test_embeddings_client_passes_max_retries_when_set(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITEE_API_KEY", "sk-test")
    provider = OpenAICompatEmbeddings(
        api_key_env="GITEE_API_KEY", base_url="http://gitee.example/v1", max_retries=8
    )
    await provider.embed(["ping"], model="Qwen3-Embedding-8B")
    init = captured["init_kwargs"]
    assert init is not None
    assert init.get("max_retries") == 8


async def test_build_llm_wires_llm_max_retries_from_config(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = make_provider_cfg(llm="openai_compat", llm_max_retries=3)
    provider = build_llm(cfg)
    assert isinstance(provider, OpenAICompatLLM)
    provider._get_client()
    init = captured["init_kwargs"]
    assert init is not None
    assert init.get("max_retries") == 3


async def test_build_embedder_wires_embedding_max_retries_from_config(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITEE_API_KEY", "sk-test")
    cfg = make_provider_cfg(
        embedding_base_url="http://gitee.example/v1",
        embedding_max_retries=9,
        embedding_api_key_env="GITEE_API_KEY",
    )
    provider = build_embedder(cfg)
    await provider.embed(["ping"], model="Qwen3-Embedding-8B")
    init = captured["init_kwargs"]
    assert init is not None
    assert init.get("max_retries") == 9


async def test_legs_carry_independent_retry_budgets(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM and embedding legs can target different retry budgets in the same config."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.setenv("GITEE_API_KEY", "sk-embed")
    cfg = make_provider_cfg(
        llm="openai_compat",
        llm_max_retries=2,
        embedding_max_retries=10,
        embedding_api_key_env="GITEE_API_KEY",
    )

    llm = build_llm(cfg)
    assert isinstance(llm, OpenAICompatLLM)
    llm._get_client()
    llm_init = captured["init_kwargs"]
    assert llm_init is not None
    assert llm_init.get("max_retries") == 2

    # Reset for the second client so we can inspect embedding-leg kwargs.
    captured["init_kwargs"] = None
    embedder = build_embedder(cfg)
    await embedder.embed(["ping"], model="any")
    embed_init = captured["init_kwargs"]
    assert embed_init is not None
    assert embed_init.get("max_retries") == 10
