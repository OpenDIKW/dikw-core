"""Verify the openai_compat embedding provider threads ctor args to the SDK.

Mirrors the shape of ``tests/test_provider_anthropic_base_url.py`` — we
monkey-patch ``openai.AsyncOpenAI`` with a stub whose ``embeddings.create``
captures the kwargs it's called with. This lets us assert the provider
plumbs ``base_url`` / ``api_key`` / ``dimensions`` end-to-end without
making any real HTTP calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dikw_core.providers import build_embedder
from dikw_core.providers.base import ProviderError
from dikw_core.providers.openai_compat import OpenAICompatEmbeddings

from .fakes import make_provider_cfg


@pytest.fixture()
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``openai.AsyncOpenAI`` with a stub that records init + call kwargs."""
    rec: dict[str, Any] = {"init_kwargs": None, "embed_kwargs": None}

    class FakeEmbeddings:
        async def create(self, **kwargs: Any) -> Any:
            rec["embed_kwargs"] = kwargs
            n = len(kwargs.get("input", []) or [])
            return type(
                "Resp",
                (),
                {"data": [type("Row", (), {"embedding": [0.0, 1.0]})() for _ in range(n)]},
            )()

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            rec["init_kwargs"] = kwargs
            self.embeddings = FakeEmbeddings()

    # OpenAICompatEmbeddings imports AsyncOpenAI lazily inside _client(); patch the
    # module-level attribute the helper will look up.
    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    # Strip any ambient env vars so these tests are hermetic.
    monkeypatch.delenv("DIKW_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    return rec


# ---- Step 1: default_dimensions + base_url plumbing --------------------


async def test_embeddings_client_uses_base_url_when_provided(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIKW_EMBEDDING_API_KEY", "sk-test")
    provider = OpenAICompatEmbeddings(base_url="http://gitee.example/v1")
    await provider.embed(["ping"], model="Qwen3-Embedding-8B")
    init = captured["init_kwargs"]
    assert init is not None
    assert init["base_url"] == "http://gitee.example/v1"


async def test_embeddings_passes_dimensions_when_default_set(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIKW_EMBEDDING_API_KEY", "sk-test")
    provider = OpenAICompatEmbeddings(
        base_url="http://gitee.example/v1", default_dimensions=1024
    )
    await provider.embed(["ping"], model="Qwen3-Embedding-8B")
    call = captured["embed_kwargs"]
    assert call is not None
    assert call.get("dimensions") == 1024
    assert call.get("model") == "Qwen3-Embedding-8B"
    assert call.get("input") == ["ping"]


async def test_embeddings_omits_dimensions_when_default_none(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIKW_EMBEDDING_API_KEY", "sk-test")
    provider = OpenAICompatEmbeddings(base_url="http://gitee.example/v1")
    await provider.embed(["ping"], model="Qwen3-Embedding-8B")
    call = captured["embed_kwargs"]
    assert call is not None
    assert "dimensions" not in call


# ---- Step 2: DIKW_EMBEDDING_API_KEY only --------------------------------


async def test_embeddings_reads_dikw_embedding_api_key(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIKW_EMBEDDING_API_KEY", "sk-gitee")
    provider = OpenAICompatEmbeddings(base_url="http://gitee.example/v1")
    await provider.embed(["ping"], model="Qwen3-Embedding-8B")
    init = captured["init_kwargs"]
    assert init is not None
    assert init["api_key"] == "sk-gitee"


async def test_embeddings_ignores_openai_api_key(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only the old OpenAI env var is set — the new embedding leg must not fall
    # back to it, and must raise a ProviderError pointing at the new var.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-be-ignored")
    provider = OpenAICompatEmbeddings(base_url="http://gitee.example/v1")
    with pytest.raises(ProviderError) as excinfo:
        await provider.embed(["ping"], model="Qwen3-Embedding-8B")
    assert "DIKW_EMBEDDING_API_KEY" in str(excinfo.value)


async def test_embeddings_explicit_api_key_wins(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIKW_EMBEDDING_API_KEY", "sk-env")
    provider = OpenAICompatEmbeddings(
        base_url="http://gitee.example/v1", api_key="sk-explicit"
    )
    await provider.embed(["ping"], model="Qwen3-Embedding-8B")
    init = captured["init_kwargs"]
    assert init is not None
    assert init["api_key"] == "sk-explicit"


# ---- Step 3: build_embedder wires config.embedding_dim ----------


async def test_build_embedder_passes_dimensions_from_config(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIKW_EMBEDDING_API_KEY", "sk-test")
    cfg = make_provider_cfg(
        embedding_base_url="http://gitee.example/v1",
        embedding_dim=512,
    )
    provider = build_embedder(cfg)
    assert isinstance(provider, OpenAICompatEmbeddings)
    await provider.embed(["ping"], model="Qwen3-Embedding-8B")
    call = captured["embed_kwargs"]
    assert call is not None
    assert call.get("dimensions") == 512


def test_provider_config_round_trips_embedding_dim() -> None:
    import yaml

    from dikw_core.config import DikwConfig, dump_config_yaml

    cfg = DikwConfig(provider=make_provider_cfg(embedding_dim=1024))
    yaml_text = dump_config_yaml(cfg)
    reparsed = DikwConfig.model_validate(yaml.safe_load(yaml_text))
    assert reparsed.provider.embedding_dim == 1024


def test_provider_config_round_trips_embedding_provider_label() -> None:
    """Display label for ``dikw client check`` travels with the wiki, not env.

    Before this refactor the label was read from ``DIKW_EMBEDDING_PROVIDER``
    at probe time. That meant "what embedding vendor am I pointed at?" was
    shell state, not config — inconsistent with base_url/model/dims which
    already live in ``dikw.yml``. This round-trip locks the new location.
    """
    import yaml

    from dikw_core.config import DikwConfig, dump_config_yaml

    cfg = DikwConfig(provider=make_provider_cfg(embedding_provider_label="gitee-ai"))
    yaml_text = dump_config_yaml(cfg)
    reparsed = DikwConfig.model_validate(yaml.safe_load(yaml_text))
    assert reparsed.provider.embedding_provider_label == "gitee-ai"


def test_provider_config_round_trips_embedding_batch_size() -> None:
    """Gitee AI caps embedding batches at ~25 items; users override here.

    OpenAI's native cap is ~2048, so ``64`` is a safe default. Gitee users
    set ``embedding_batch_size: 16`` (or similar) in their dikw.yml to fit
    the stricter backend limit without code changes.
    """
    import yaml

    from dikw_core.config import DikwConfig, dump_config_yaml

    cfg = DikwConfig(provider=make_provider_cfg(embedding_batch_size=16))
    yaml_text = dump_config_yaml(cfg)
    reparsed = DikwConfig.model_validate(yaml.safe_load(yaml_text))
    assert reparsed.provider.embedding_batch_size == 16


async def test_api_ingest_honours_embedding_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``api.ingest`` must pass ``cfg.provider.embedding_batch_size`` to ``embed_chunks``.

    Covers the Gitee AI case: a stricter backend limit surfaces via config
    without changing call sites.
    """
    from dikw_core import api

    monkeypatch.setenv("DIKW_EMBEDDING_API_KEY", "sk-test")

    wiki = tmp_path / "knowledge"
    api.init_base(wiki, description="batch-size smoke test")
    (wiki / "dikw.yml").write_text(
        "provider:\n"
        "  llm: anthropic_compat\n"
        "  embedding: openai_compat\n"
        "  embedding_model: test-embed\n"
        "  embedding_base_url: https://fake.example/v1\n"
        "  embedding_dim: 2\n"
        "  embedding_revision: ''\n"
        "  embedding_normalize: true\n"
        "  embedding_distance: cosine\n"
        "  embedding_batch_size: 7\n"
        "storage:\n"
        "  backend: sqlite\n"
        "  path: .dikw/index.sqlite\n"
        "schema:\n"
        "  description: test\n"
        "sources:\n"
        "  - path: ./sources\n"
        "    pattern: '**/*.md'\n",
        encoding="utf-8",
    )
    src_dir = wiki / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    # 15 tiny docs → with batch_size=7 the embedder sees batches of 7, 7, 1.
    for i in range(15):
        (src_dir / f"doc{i}.md").write_text(f"# doc {i}\n\nbody {i}\n", encoding="utf-8")

    batch_sizes: list[int] = []

    class BatchCapturingEmbedder:
        async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
            batch_sizes.append(len(texts))
            return [[0.0, 1.0] for _ in texts]

    report = await api.ingest(wiki, embedder=BatchCapturingEmbedder())
    assert report.embedded > 0
    assert batch_sizes, "expected at least one batch to be embedded"
    assert max(batch_sizes) <= 7, (
        f"batch_size cap was 7 but saw batches: {batch_sizes}"
    )
