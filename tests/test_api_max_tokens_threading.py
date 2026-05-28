"""Verify ``api.synthesize`` threads the per-op ``llm_max_tokens_*`` value
from ``ProviderConfig`` into ``LLMProvider.complete`` instead of a
pre-refactor hardcoded constant.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dikw_core import api

from .fakes import FakeEmbeddings, FakeLLM

FIXTURES = Path(__file__).parent / "fixtures" / "notes"


def _write_wiki(
    tmp_path: Path,
    *,
    llm_max_tokens_synth: int,
) -> Path:
    wiki = tmp_path / "knowledge"
    api.init_base(wiki)
    # Overwrite the auto-generated dikw.yml with per-op overrides. Fake
    # embeddings need dim=64 to match ``FakeEmbeddings``.
    (wiki / "dikw.yml").write_text(
        f"""\
provider:
  llm: anthropic_compat
  llm_model: stub-model
  embedding: openai_compat
  embedding_model: stub-embed
  embedding_dim: 64
  embedding_revision: ''
  embedding_normalize: true
  embedding_distance: cosine
  llm_max_tokens_synth: {llm_max_tokens_synth}
storage:
  backend: sqlite
  path: .dikw/index.sqlite
schema:
  description: max_tokens threading test wiki
sources:
  - path: ./sources
    pattern: "**/*.md"
""",
        encoding="utf-8",
    )
    dest = wiki / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    return wiki


@pytest.mark.asyncio
async def test_synthesize_threads_max_tokens_from_config(tmp_path: Path) -> None:
    wiki = _write_wiki(
        tmp_path,
        llm_max_tokens_synth=888,
    )
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    # FakeLLM returns a STUB string that won't parse as <page>; synthesize
    # will record an error, but the call to complete() happens first and
    # captures the max_tokens we care about.
    llm = FakeLLM(response_text="STUB: not a page")
    await api.synthesize(wiki, llm=llm, embedder=embedder)

    assert llm.last_max_tokens == 888
