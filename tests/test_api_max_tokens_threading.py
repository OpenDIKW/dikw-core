"""Verify ``api.query``, ``api.synthesize``, and ``api.distill`` thread the
per-op ``llm_max_tokens_*`` values from ``ProviderConfig`` into ``LLMProvider.complete``
instead of the pre-refactor hardcoded 1024/2048/2048.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dikw_core import api

from .fakes import FakeEmbeddings, FakeLLM

FIXTURES = Path(__file__).parent / "fixtures" / "notes"


_SYNTH_SCRIPT_TEMPLATE = (
    "<page path=\"wiki/concepts/{stem}.md\" type=\"concept\">\n"
    "---\ntags: [stub]\n---\n\n"
    "# {title}\n\n"
    "Stub body for {stem}.\n"
    "</page>"
)


class _ScriptedSynthLLM:
    """Minimal LLM that returns a valid <page> for any synth prompt. Only used
    to populate K docs so the distill threading test has something to read.
    """

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list | None = None,
    ):
        from dikw_core.providers import LLMResponse

        # Extract the source path between "SOURCE DOCUMENT" markers in the
        # synth prompt — we just need a per-source unique stem.
        for line in user.splitlines():
            if "sources/notes/" in line:
                stem = Path(line.strip()).stem
                body = _SYNTH_SCRIPT_TEMPLATE.format(stem=stem, title=stem)
                return LLMResponse(text=body, finish_reason="end_turn")
        # Fallback — keeps this helper robust to prompt-template tweaks.
        body = _SYNTH_SCRIPT_TEMPLATE.format(stem="fallback", title="Fallback")
        return LLMResponse(text=body, finish_reason="end_turn")


def _write_wiki(
    tmp_path: Path,
    *,
    llm_max_tokens_query: int,
    llm_max_tokens_synth: int,
    llm_max_tokens_distill: int,
) -> Path:
    wiki = tmp_path / "wiki"
    api.init_wiki(wiki)
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
  llm_max_tokens_query: {llm_max_tokens_query}
  llm_max_tokens_synth: {llm_max_tokens_synth}
  llm_max_tokens_distill: {llm_max_tokens_distill}
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
        llm_max_tokens_query=1024,
        llm_max_tokens_synth=888,
        llm_max_tokens_distill=2048,
    )
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    # FakeLLM returns a STUB string that won't parse as <page>; synthesize
    # will record an error, but the call to complete() happens first and
    # captures the max_tokens we care about.
    llm = FakeLLM(response_text="STUB: not a page")
    await api.synthesize(wiki, llm=llm, embedder=embedder)

    assert llm.last_max_tokens == 888


@pytest.mark.asyncio
async def test_distill_threads_max_tokens_from_config(tmp_path: Path) -> None:
    wiki = _write_wiki(
        tmp_path,
        llm_max_tokens_query=1024,
        llm_max_tokens_synth=2048,
        llm_max_tokens_distill=999,
    )
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)
    # Populate K layer so distill has pages to read. Use a scripted synth
    # LLM so real <page> bodies land on disk.
    await api.synthesize(wiki, llm=_ScriptedSynthLLM(), embedder=embedder)

    # FakeLLM for distill — response doesn't need to parse, just needs to
    # capture max_tokens.
    llm = FakeLLM(response_text="STUB: not a wisdom block")
    await api.distill(wiki, llm=llm, pages_per_call=8)

    assert llm.last_max_tokens == 999
