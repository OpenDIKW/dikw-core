"""Live end-to-end smoke test: MiniMax LLM + Gitee AI embeddings.

Gated on two env secrets — ``ANTHROPIC_API_KEY`` (the MiniMax key, since
MiniMax exposes an Anthropic-compatible endpoint) and
``DIKW_EMBEDDING_API_KEY`` (the Gitee AI key, since MiniMax has no
embedding endpoint). When either is missing the test skips, so the
default `pytest` run stays hermetic.

All non-secret config (base URLs, model names, dimensions, batch size,
display label) lives in the committed fixture
``tests/fixtures/live-minimax-gitee.dikw.yml``. The fixture copies that
yaml verbatim to the temp wiki; the test never stitches config from env.
This matches how a CLI user would run: put secrets in ``.env``, edit
``dikw.yml``, run ``dikw``.

Run manually:

    # fill ANTHROPIC_API_KEY + DIKW_EMBEDDING_API_KEY in .env (gitignored),
    # then either rely on pytest-dotenv (auto-loaded):
    uv run pytest tests/test_mvp_e2e.py -v -s
    # …or export them yourself before invoking pytest.

The test ingests the packaged ``evals/datasets/mvp/corpus/`` into a temp wiki, then
asks one Karpathy question and asserts the returned citations include a
karpathy source. It does not assert a specific answer string — live LLM
output is non-deterministic — only that the grounding pipeline
round-trips end to end.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.eval.dataset import datasets_root
from dikw_core.providers import build_embedder

CORPUS_DIR = datasets_root() / "mvp" / "corpus"
CONFIG_FIXTURE = Path(__file__).parent / "fixtures" / "live-minimax-gitee.dikw.yml"

REQUIRED_ENV = ("ANTHROPIC_API_KEY", "DIKW_EMBEDDING_API_KEY")


def _missing_env() -> list[str]:
    return [name for name in REQUIRED_ENV if not os.environ.get(name)]


pytestmark = pytest.mark.skipif(
    bool(_missing_env()),
    reason=(
        "live MiniMax+Gitee smoke test — set "
        + " and ".join(REQUIRED_ENV)
        + " (missing: "
        + ", ".join(_missing_env() or ["(none)"])
        + "); see tests/fixtures/live-minimax-gitee.dikw.yml for the "
        "committed provider config."
    ),
)


@pytest.fixture()
def mvp_wiki(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    api.init_wiki(wiki, description="mvp e2e smoke test wiki")

    # The committed fixture is the source of truth for provider config —
    # drop it in wholesale so test and CLI see the same dikw.yml shape.
    shutil.copy(CONFIG_FIXTURE, wiki / "dikw.yml")

    dest = wiki / "sources" / "corpus"
    dest.mkdir(parents=True, exist_ok=True)
    for src in CORPUS_DIR.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    return wiki


@pytest.mark.asyncio
async def test_check_connectivity_roundtrips(mvp_wiki: Path) -> None:
    """Verify provider config before we ingest — fail fast on bad credentials."""
    report = await api.check_providers(mvp_wiki)
    assert report.llm is not None and report.llm.ok, (
        f"LLM probe failed: {report.llm.detail if report.llm else '(no probe)'}"
    )
    assert report.embed is not None and report.embed.ok, (
        f"Embedding probe failed: {report.embed.detail if report.embed else '(no probe)'}"
    )
    # The provider label from the committed yaml should flow into the probe
    # detail — cheap cross-check that dikw.yml → ProviderConfig → probe is wired.
    assert "gitee-ai" in report.embed.detail


@pytest.mark.asyncio
async def test_ingest_then_query_returns_karpathy_citation(
    mvp_wiki: Path,
) -> None:
    # ``api.ingest`` only embeds when an embedder is passed (matching the
    # CLI's ``--no-embed``-aware wiring); ``api.retrieve`` doesn't
    # auto-build one either when one isn't passed explicitly. Mirror the
    # CLI here so the smoke test exercises the full ingest → embed →
    # hybrid-search path against live providers. dikw-core no longer
    # owns the LLM-synthesis step (PR-1 removed ``query``) — agents
    # compose retrieve output with their own LLM.
    cfg, _ = api.load_wiki(mvp_wiki)
    embedder = build_embedder(cfg.provider)

    ingest_report = await api.ingest(mvp_wiki, embedder=embedder)
    assert ingest_report.added > 0, "expected some docs to be added"
    assert ingest_report.embedded > 0, "expected some chunks to be embedded"

    result = await api.retrieve(
        "What does Karpathy mean by deterministic scoping versus probabilistic reasoning?",
        mvp_wiki,
        limit=5,
        embedder=embedder,
    )
    assert result.chunks, "expected at least one chunk hit from live retrieve"
    assert any("karpathy" in hit.path.lower() for hit in result.chunks), (
        "expected a karpathy chunk, got: "
        f"{[hit.path for hit in result.chunks]}"
    )
