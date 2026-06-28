"""End-to-end wiring of the rerank leg through ``api.retrieve``.

The unit tests in ``test_search.py`` drive ``HybridSearcher`` with a reranker
directly; this pins the api layer's plumbing: ``_retrieve_inner`` builds the
reranker via ``build_reranker(cfg.provider)`` only when configured + enabled,
threads it into ``HybridSearcher.from_config``, and closes it in its ``finally``
on every exit. ``build_reranker`` is monkeypatched to a ``FakeReranker`` so the
test needs no rerank vendor / network.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.config import dump_config_yaml, load_config

from .fakes import FakeEmbeddings, FakeReranker, init_test_base


def _configure_rerank(wiki: Path) -> None:
    cfg_path = wiki / "dikw.yml"
    cfg = load_config(cfg_path)
    cfg.provider.rerank = "openai_compat_rerank"
    cfg.provider.rerank_model = "bge-reranker-v2-m3"
    cfg.provider.rerank_base_url = "https://ai.gitee.com/v1"
    cfg.provider.rerank_api_key_env = "GITEE_API_KEY"
    cfg_path.write_text(dump_config_yaml(cfg), encoding="utf-8")


@pytest.mark.asyncio
async def test_retrieve_builds_and_closes_reranker_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    _configure_rerank(wiki)

    src = wiki / "sources"
    src.mkdir(parents=True, exist_ok=True)
    (src / "a.md").write_text(
        "# Alpha\n\nReciprocal rank fusion blends BM25 and vector retrieval.\n",
        encoding="utf-8",
    )
    (src / "b.md").write_text(
        "# Beta\n\nCross-encoder reranking reorders the fused candidate set.\n",
        encoding="utf-8",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    fake = FakeReranker()
    # api_retrieve binds ``build_reranker`` into its own namespace.
    monkeypatch.setattr(
        "dikw_core.api_retrieve.build_reranker", lambda _cfg: fake
    )

    result = await api.retrieve(
        "rank fusion reranking", wiki, limit=3, embedder=FakeEmbeddings()
    )

    assert result.chunks, "expected hits"
    assert fake.call_count == 1, "the configured reranker must be invoked once"
    assert fake.closed is True, "the reranker must be closed in _retrieve_inner's finally"


@pytest.mark.asyncio
async def test_retrieve_skips_reranker_when_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``provider.rerank`` → ``build_reranker`` returns ``None`` → no rerank
    leg; retrieve still returns hits."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)  # no rerank configured

    src = wiki / "sources"
    src.mkdir(parents=True, exist_ok=True)
    (src / "a.md").write_text("# Alpha\n\nVector retrieval over chunks.\n", encoding="utf-8")
    await api.ingest(wiki, embedder=FakeEmbeddings())

    fake = FakeReranker()
    # Even if a fake is available, an unconfigured base must not reach for it.
    monkeypatch.setattr("dikw_core.api_retrieve.build_reranker", lambda _cfg: None)

    result = await api.retrieve(
        "vector retrieval", wiki, limit=3, embedder=FakeEmbeddings()
    )
    assert result.chunks
    assert fake.call_count == 0


@pytest.mark.asyncio
async def test_retrieve_warns_when_rerank_enabled_but_unconfigured(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``retrieval.rerank_enabled`` (default True) + no ``provider.rerank`` is an
    enabled-but-unconfigured contradiction: warn once per retrieve so the
    operator notices the rerank leg is silently off. Retrieve still returns
    hits (a missing reranker degrades, never blocks). A base that genuinely
    doesn't want rerank silences this with ``rerank_enabled: false``."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)  # clears the default reranker; leaves rerank_enabled=True

    src = wiki / "sources"
    src.mkdir(parents=True, exist_ok=True)
    (src / "a.md").write_text("# Alpha\n\nVector retrieval over chunks.\n", encoding="utf-8")
    await api.ingest(wiki, embedder=FakeEmbeddings())

    with caplog.at_level(logging.WARNING, logger="dikw_core.api_retrieve"):
        result = await api.retrieve(
            "vector retrieval", wiki, limit=3, embedder=FakeEmbeddings()
        )

    assert result.chunks
    assert any(
        r.levelno == logging.WARNING and "rerank" in r.getMessage().lower()
        for r in caplog.records
    ), "an enabled-but-unconfigured reranker must warn"
