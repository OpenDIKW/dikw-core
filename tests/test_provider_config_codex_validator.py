"""``ProviderConfig`` rejects ``openai_codex`` without an explicit base URL.

The Codex protocol is hard-bound to ``https://chatgpt.com/backend-api/codex``;
unlike ``openai_compat`` / ``anthropic_compat`` it does not have a sensible
SDK default. The validator surfaces this at config-load time so ``dikw check``
fails before any LLM call is attempted, with a message that tells the user
what to paste into ``dikw.yml``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dikw_core.config import ProviderConfig

from .fakes import make_provider_cfg


def test_openai_codex_requires_explicit_llm_base_url() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_provider_cfg(llm="openai_codex")
    msg = str(excinfo.value)
    assert "openai_codex" in msg
    assert "llm_base_url" in msg
    assert "https://chatgpt.com/backend-api/codex" in msg


def test_openai_codex_rejects_empty_string_base_url() -> None:
    with pytest.raises(ValidationError):
        make_provider_cfg(llm="openai_codex", llm_base_url="")


def test_openai_codex_rejects_whitespace_base_url() -> None:
    with pytest.raises(ValidationError):
        make_provider_cfg(llm="openai_codex", llm_base_url="   ")


def test_openai_codex_with_explicit_base_url_validates() -> None:
    cfg = make_provider_cfg(
        llm="openai_codex",
        llm_base_url="https://chatgpt.com/backend-api/codex",
    )
    assert cfg.llm == "openai_codex"
    assert cfg.llm_base_url == "https://chatgpt.com/backend-api/codex"


def test_openai_compat_no_base_url_still_valid() -> None:
    """Existing protocols keep working with llm_base_url=None — the validator
    must not over-reach beyond openai_codex."""
    cfg = make_provider_cfg(llm="openai_compat")
    assert cfg.llm == "openai_compat"
    assert cfg.llm_base_url is None


def test_anthropic_compat_no_base_url_still_valid() -> None:
    cfg = make_provider_cfg(llm="anthropic_compat")
    assert cfg.llm == "anthropic_compat"
    assert cfg.llm_base_url is None


def test_provider_config_default_llm_unchanged() -> None:
    """Sanity: the new Literal value is opt-in, default still anthropic_compat."""
    cfg = ProviderConfig(
        embedding_dim=1024,
        embedding_revision="",
        embedding_normalize=True,
        embedding_distance="cosine",
    )
    assert cfg.llm == "anthropic_compat"
