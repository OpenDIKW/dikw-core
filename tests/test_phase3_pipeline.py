"""End-to-end Phase 3 flow: distill -> candidates on disk -> approve -> aggregate.

Uses a ScriptedLLM for synth and a separate ScriptedLLM for distill so the
test stays fully offline.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.domains.wisdom.io import aggregate_path, candidate_path
from dikw_core.schemas import WisdomKind, WisdomStatus

from .fakes import FakeEmbeddings, init_test_wiki

FIXTURES = Path(__file__).parent / "fixtures" / "notes"

_SYNTH_SCRIPT = {
    "sources/notes/dikw.md": (
        "<page path=\"wiki/concepts/dikw-pyramid.md\" type=\"concept\">\n"
        "---\ntags: [dikw]\n---\n\n"
        "# DIKW pyramid\n\n"
        "Organises raw data into four layers. See [[Karpathy LLM Wiki]].\n"
        "</page>"
    ),
    "sources/notes/karpathy-wiki.md": (
        "<page path=\"wiki/concepts/karpathy-llm-wiki.md\" type=\"concept\">\n"
        "---\ntags: [pattern]\n---\n\n"
        "# Karpathy LLM Wiki\n\n"
        "Karpathy argues scoping should be deterministic. Complements the [[DIKW pyramid]].\n"
        "</page>"
    ),
    "sources/notes/retrieval.md": (
        "<page path=\"wiki/concepts/hybrid-retrieval.md\" type=\"concept\">\n"
        "---\ntags: [search]\n---\n\n"
        "# Hybrid retrieval\n\n"
        "RRF ignores raw scores; only rank order matters. Useful with the [[DIKW pyramid]].\n"
        "</page>"
    ),
}

_DISTILL_RESPONSE = """
<wisdom kind="principle">
---
confidence: 0.85
evidence:
  - doc: wiki/concepts/karpathy-llm-wiki.md
    line: 3
    excerpt: "Karpathy argues scoping should be deterministic."
  - doc: wiki/concepts/hybrid-retrieval.md
    line: 3
    excerpt: "RRF ignores raw scores; only rank order matters."
---

# Prefer deterministic scoping over probabilistic retrieval

Use deterministic structure (index.md, link graph, FTS) to narrow scope; only
invoke probabilistic retrieval once the candidate set is small [#1][#2].
</wisdom>
"""


class RoutedLLM:
    """Scripted LLM that routes by a marker string in the prompt."""

    def __init__(
        self,
        *,
        synth_script: dict[str, str],
        distill_response: str,
    ) -> None:
        self.synth_script = synth_script
        self.distill_response = distill_response
        self.last_user: str | None = None

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

        self.last_user = user
        if "SOURCE DOCUMENT" in user:
            for src, resp in self.synth_script.items():
                if src in user:
                    return LLMResponse(text=resp, finish_reason="end_turn")
            raise AssertionError("no synth script matched")
        if "WIKI PAGES" in user:
            return LLMResponse(text=self.distill_response, finish_reason="end_turn")
        return LLMResponse(text="STUB", finish_reason="end_turn")


@pytest.fixture()
def wiki(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    dest = wiki / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    return wiki


@pytest.mark.asyncio
async def test_distill_then_approve_then_wisdom_applied_in_query(wiki: Path) -> None:
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    llm = RoutedLLM(synth_script=_SYNTH_SCRIPT, distill_response=_DISTILL_RESPONSE)
    await api.synthesize(wiki, llm=llm, embedder=embedder)

    # Distill should produce exactly one candidate across the batch.
    report = await api.distill(wiki, llm=llm, pages_per_call=8)
    assert report.candidates_added == 1
    assert report.errors == 0

    candidates = await api.list_candidates(wiki)
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.kind is WisdomKind.PRINCIPLE

    # Candidate file exists on disk.
    expected_cand_path = wiki / candidate_path(cand.kind, cand.title)
    assert expected_cand_path.is_file()

    # Approve — candidate file should disappear, aggregate should appear with entry.
    result = await api.approve_wisdom(cand.item_id, wiki)
    assert result.new_status is WisdomStatus.APPROVED
    assert not expected_cand_path.exists()

    aggregate = wiki / aggregate_path(WisdomKind.PRINCIPLE)
    assert aggregate.is_file()
    agg_text = aggregate.read_text(encoding="utf-8")
    assert cand.title in agg_text
    assert "Evidence" in agg_text

    # PR-1 removed the in-engine query verb. Wisdom application is now
    # an explicit agent-side step exposed via ``/v1/wisdom/applicable``
    # (PR-5). The candidate→approved transition + aggregate file
    # bookkeeping are the invariants this test guards; the
    # "wisdom-shapes-the-answer" behaviour will get its own test once
    # PR-5 lands.


@pytest.mark.asyncio
async def test_reject_drops_candidate_and_archives(wiki: Path) -> None:
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)
    llm = RoutedLLM(synth_script=_SYNTH_SCRIPT, distill_response=_DISTILL_RESPONSE)
    await api.synthesize(wiki, llm=llm, embedder=embedder)
    await api.distill(wiki, llm=llm, pages_per_call=8)

    candidates = await api.list_candidates(wiki)
    assert len(candidates) == 1
    cand = candidates[0]

    result = await api.reject_wisdom(cand.item_id, wiki)
    assert result.new_status is WisdomStatus.ARCHIVED
    assert not (wiki / candidate_path(cand.kind, cand.title)).exists()
    # after reject, no candidates should remain
    assert await api.list_candidates(wiki) == []


@pytest.mark.asyncio
async def test_distill_is_idempotent(wiki: Path) -> None:
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)
    llm = RoutedLLM(synth_script=_SYNTH_SCRIPT, distill_response=_DISTILL_RESPONSE)
    await api.synthesize(wiki, llm=llm, embedder=embedder)

    first = await api.distill(wiki, llm=llm, pages_per_call=8)
    assert first.candidates_added == 1

    second = await api.distill(wiki, llm=llm, pages_per_call=8)
    # Same item_id, so the seen-set dedups it.
    assert second.candidates_added == 0
