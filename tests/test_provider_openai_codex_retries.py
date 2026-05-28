"""Verify the openai_codex provider threads ``max_retries`` / ``timeout``.

Mirrors ``test_provider_openai_compat_retries.py`` shape: monkey-patch
``openai.AsyncOpenAI`` with a stub that captures init kwargs, then
assert ``OpenAICodexLLM`` and ``build_llm(cfg)`` plumb both knobs all
the way to the SDK constructor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from dikw_core.providers import build_llm
from dikw_core.providers.codex_auth import DEFAULT_CODEX_BASE_URL
from dikw_core.providers.openai_codex import OpenAICodexLLM

from .fakes import (
    CodexResponsesStreamStub,
    codex_create_sentinel,
    make_codex_response,
    make_provider_cfg,
)

_DUMMY_BASE = Path("dummy-wiki")
_EMPTY_FINAL = make_codex_response()


@pytest.fixture()
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    rec: dict[str, Any] = {"init_kwargs": None}

    class FakeResponses:
        def stream(self, **_kwargs: Any) -> CodexResponsesStreamStub:
            return CodexResponsesStreamStub([], final=_EMPTY_FINAL)

        create = codex_create_sentinel

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            rec["init_kwargs"] = kwargs
            self.responses = FakeResponses()

        async def close(self) -> None:
            return None

    async def _fake_resolve(_base: Path, **_kwargs: Any) -> str:
        return "test-token"

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(
        "dikw_core.providers.openai_codex.resolve_access_token", _fake_resolve
    )
    return rec


async def _exercise(provider: OpenAICodexLLM) -> None:
    """Trigger a real complete() call so the lazy AsyncOpenAI build runs."""
    await provider.complete(system="s", user="u", model="gpt-5.5")


async def test_codex_client_passes_max_retries_when_set(
    captured: dict[str, Any],
) -> None:
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, base_root=_DUMMY_BASE, max_retries=6
    )
    await _exercise(provider)
    assert captured["init_kwargs"]["max_retries"] == 6


async def test_codex_client_omits_max_retries_when_none(
    captured: dict[str, Any],
) -> None:
    provider = OpenAICodexLLM(base_url=DEFAULT_CODEX_BASE_URL, base_root=_DUMMY_BASE)
    await _exercise(provider)
    assert "max_retries" not in captured["init_kwargs"]


async def test_codex_client_passes_timeout(captured: dict[str, Any]) -> None:
    provider = OpenAICodexLLM(
        base_url=DEFAULT_CODEX_BASE_URL, base_root=_DUMMY_BASE, timeout_seconds=42.0
    )
    await _exercise(provider)
    timeout = captured["init_kwargs"]["timeout"]
    # Timeout is an httpx.Timeout — confirm the read leg pinned to our value.
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == 42.0


async def test_build_llm_wires_max_retries_from_config(
    captured: dict[str, Any],
) -> None:
    cfg = make_provider_cfg(
        llm="openai_codex",
        llm_base_url=DEFAULT_CODEX_BASE_URL,
        llm_max_retries=3,
    )
    provider = build_llm(cfg, base_root=_DUMMY_BASE)
    assert isinstance(provider, OpenAICodexLLM)
    await _exercise(provider)
    assert captured["init_kwargs"]["max_retries"] == 3


async def test_build_llm_wires_timeout_from_config(
    captured: dict[str, Any],
) -> None:
    cfg = make_provider_cfg(
        llm="openai_codex",
        llm_base_url=DEFAULT_CODEX_BASE_URL,
        llm_timeout_seconds=99.0,
    )
    provider = build_llm(cfg, base_root=_DUMMY_BASE)
    assert isinstance(provider, OpenAICodexLLM)
    await _exercise(provider)
    timeout = captured["init_kwargs"]["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == 99.0
