"""Provider API-key env var is config-driven (``llm_api_key_env`` /
``embedding_api_key_env``).

Pins the change that replaced the hardcoded ``ANTHROPIC_API_KEY`` /
``OPENAI_API_KEY`` / ``DIKW_EMBEDDING_API_KEY`` constants with **required**
per-base config fields naming which env var holds each leg's key. No fallback:
a config must name its key var, and the provider reads exactly that var. This
is what lets two ``anthropic_compat`` vendors (DeepSeek + MiniMax) coexist in
one ``.env`` under distinct vendor-canonical names.
"""

from __future__ import annotations

import pytest

from dikw_core.api_lint import lint_apply
from dikw_core.config import ProviderConfig, dump_config_yaml, load_config
from dikw_core.domains.knowledge.lint_fix import FixProposalReport
from dikw_core.eval.fake_embedder import FakeEmbeddings
from dikw_core.providers import build_embedder, build_llm
from dikw_core.providers.base import ProviderError

from .fakes import init_test_base, make_provider_cfg

_IDENTITY = {
    "embedding_dim": 1024,
    "embedding_revision": "",
    "embedding_normalize": True,
    "embedding_distance": "cosine",
}


def test_provider_config_requires_llm_api_key_env() -> None:
    """``llm_api_key_env`` is a required field — omitting it fails config load."""
    with pytest.raises(Exception) as exc:
        ProviderConfig(**_IDENTITY, embedding_api_key_env="OPENAI_API_KEY")
    assert "llm_api_key_env" in str(exc.value)


def test_provider_config_requires_embedding_api_key_env() -> None:
    """``embedding_api_key_env`` is a required field too."""
    with pytest.raises(Exception) as exc:
        ProviderConfig(**_IDENTITY, llm_api_key_env="ANTHROPIC_API_KEY")
    assert "embedding_api_key_env" in str(exc.value)


def test_anthropic_llm_reads_configured_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """``anthropic_compat`` resolves its key from the configured env var name —
    not the historical hardcoded ``ANTHROPIC_API_KEY``."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    cfg = make_provider_cfg(
        llm="anthropic_compat",
        llm_base_url="https://api.deepseek.com/anthropic",
        llm_api_key_env="DEEPSEEK_API_KEY",
    )
    llm = build_llm(cfg)
    assert llm._get_client().api_key == "sk-deepseek"  # type: ignore[attr-defined]


def test_anthropic_llm_missing_configured_env_var_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing key surfaces a ``ProviderError`` naming the *configured* var."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = make_provider_cfg(llm="anthropic_compat", llm_api_key_env="DEEPSEEK_API_KEY")
    llm = build_llm(cfg)
    with pytest.raises(ProviderError, match="DEEPSEEK_API_KEY"):
        llm._get_client()  # type: ignore[attr-defined]


def test_openai_llm_reads_configured_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """``openai_compat`` LLM resolves its key from the configured env var name."""
    monkeypatch.setenv("MYLLM_KEY", "sk-openai-llm")
    cfg = make_provider_cfg(llm="openai_compat", llm_api_key_env="MYLLM_KEY")
    llm = build_llm(cfg)
    assert llm._get_client().api_key == "sk-openai-llm"  # type: ignore[attr-defined]


def test_embedder_reads_configured_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """The embedder resolves its key from the configured ``embedding_api_key_env``
    name — not the removed ``DIKW_EMBEDDING_API_KEY``."""
    monkeypatch.setenv("GITEE_API_KEY", "sk-gitee")
    cfg = make_provider_cfg(
        embedding_base_url="https://ai.gitee.com/v1",
        embedding_api_key_env="GITEE_API_KEY",
    )
    emb = build_embedder(cfg)
    assert emb._get_client().api_key == "sk-gitee"  # type: ignore[attr-defined]


def test_embedder_missing_configured_env_var_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITEE_API_KEY", raising=False)
    cfg = make_provider_cfg(embedding_api_key_env="GITEE_API_KEY")
    emb = build_embedder(cfg)
    with pytest.raises(ProviderError, match="GITEE_API_KEY"):
        emb._get_client()  # type: ignore[attr-defined]


async def _run_lint_gate(base: object) -> None:
    await lint_apply(base, proposal_report=FixProposalReport(proposals=[], skipped=[]))  # type: ignore[arg-type]


async def test_lint_apply_embed_gate_reads_configured_env(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lint-apply inline-embed gate keys off ``cfg.provider.embedding_api_key_env``,
    not the removed hardcoded ``DIKW_EMBEDDING_API_KEY``."""
    from pathlib import Path

    base = Path(str(tmp_path)) / "kb"
    init_test_base(base, description="gate test")
    cfg_path = base / "dikw.yml"
    cfg = load_config(cfg_path)
    cfg.provider.embedding_api_key_env = "GITEE_API_KEY"
    cfg_path.write_text(dump_config_yaml(cfg), encoding="utf-8")

    monkeypatch.setenv("GITEE_API_KEY", "sk-gitee")
    calls: list[int] = []

    def _spy(_provider_cfg: object, **_kw: object) -> FakeEmbeddings:
        calls.append(1)
        return FakeEmbeddings()

    async def _defer(*_a: object, **_k: object) -> None:
        return None

    monkeypatch.setattr("dikw_core.api_lint.build_embedder", _spy)
    monkeypatch.setattr(
        "dikw_core.api_lint._resolve_active_text_version_for_inline_embed", _defer
    )
    await _run_lint_gate(base)
    assert calls == [1]


async def test_lint_apply_embed_gate_off_when_env_unset(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    base = Path(str(tmp_path)) / "kb"
    init_test_base(base, description="gate test")
    cfg_path = base / "dikw.yml"
    cfg = load_config(cfg_path)
    cfg.provider.embedding_api_key_env = "GITEE_API_KEY"
    cfg_path.write_text(dump_config_yaml(cfg), encoding="utf-8")

    monkeypatch.delenv("GITEE_API_KEY", raising=False)
    calls: list[int] = []

    def _spy(_provider_cfg: object, **_kw: object) -> FakeEmbeddings:
        calls.append(1)
        return FakeEmbeddings()

    monkeypatch.setattr("dikw_core.api_lint.build_embedder", _spy)
    await _run_lint_gate(base)
    assert calls == []
