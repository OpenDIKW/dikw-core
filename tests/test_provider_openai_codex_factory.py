"""``build_llm`` resolves ``llm: openai_codex`` to ``OpenAICodexLLM``.

The base_url / max_retries / timeout flow from cfg → SDK constructor is
covered end-to-end in ``test_provider_openai_codex_retries.py`` (which
mocks ``openai.AsyncOpenAI``); this file just pins the dispatch table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core.providers import build_llm
from dikw_core.providers.base import ProviderError
from dikw_core.providers.codex_auth import DEFAULT_CODEX_BASE_URL
from dikw_core.providers.openai_codex import OpenAICodexLLM

from .fakes import make_provider_cfg


def test_build_llm_returns_openai_codex_instance() -> None:
    cfg = make_provider_cfg(
        llm="openai_codex", llm_base_url=DEFAULT_CODEX_BASE_URL
    )
    provider = build_llm(cfg, base_root=Path("dummy-wiki"))
    assert isinstance(provider, OpenAICodexLLM)


def test_build_llm_requires_base_root_for_openai_codex() -> None:
    """``openai_codex`` stores its OAuth tokens at
    ``<base_root>/.dikw/auth.json`` so the factory cannot build a working
    instance without one. The error message tells the engine where to
    plumb the wiki root from."""
    cfg = make_provider_cfg(
        llm="openai_codex", llm_base_url=DEFAULT_CODEX_BASE_URL
    )
    with pytest.raises(ProviderError) as excinfo:
        build_llm(cfg)
    assert "base_root" in str(excinfo.value)
