"""Synth-time existing-pages awareness (PR2).

Each synth LLM call must receive two prompt sections so the model can
detect semantic duplicates against pages already in the wiki AND pages
just emitted by an earlier group within the same source:

* ``### Already created in this batch (MUST reference, do NOT regenerate)``
  — per-source accumulator
* ``### Existing knowledge pages (reference via [[Title]] when relevant)``
  — full snapshot up to a byte threshold, switching to a vec_search-gated
  top-K beyond that

Without these sections the LLM happily regenerates pages it cannot see,
inflating broken-wikilink counts and polluting the wiki with semantic
duplicates that PR1's fuzzy resolver cannot absorb.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from dikw_core import api
from dikw_core.config import dump_config_yaml, load_config
from dikw_core.domains.knowledge.page import build_page, write_page
from dikw_core.providers import LLMResponse

from .fakes import FakeEmbeddings, init_test_base


class CapturingLLM:
    """Records every user prompt the synth pipeline issues, in call order.

    Returns a single empty-page response per call so synth proceeds
    cleanly. Tests inspect ``calls`` to assert prompt content per group.
    """

    def __init__(self, response: str = "(no page worth writing)") -> None:
        self._response = response
        self.calls: list[str] = []

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list | None = None,
    ) -> LLMResponse:
        self.calls.append(user)
        return LLMResponse(text=self._response, finish_reason="end_turn")


class GroupKeyedLLM:
    """Returns a distinct ``<page>`` block per call, keyed by call index.

    Lets the multi-group test verify that group N's prompt sees the
    pages emitted by groups 0..N-1.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list | None = None,
    ) -> LLMResponse:
        idx = len(self.calls)
        self.calls.append(user)
        text = (
            f'<page category="concept" slug="group-{idx}-page">\n'
            f"---\ntags: [synthetic]\n---\n\n"
            f"# Group {idx} page\n\n"
            f"Synthetic page emitted by group {idx}.\n"
            f"</page>"
        )
        return LLMResponse(text=text, finish_reason="end_turn")


def _write_source(wiki: Path, name: str, body: str) -> None:
    src_dir = wiki / "sources" / "notes"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / name).write_text(body, encoding="utf-8")


def _override_synth_cfg(wiki: Path, **kwargs: Any) -> None:
    """Patch ``dikw.yml`` ``synth`` block in place."""
    cfg_path = wiki / "dikw.yml"
    cfg = load_config(cfg_path)
    for k, v in kwargs.items():
        setattr(cfg.synth, k, v)
    cfg_path.write_text(dump_config_yaml(cfg), encoding="utf-8")


async def _seed_wiki_page(
    wiki: Path, *, title: str, category: str, body: str | None = None
) -> None:
    """Create one K-layer page directly via ``_persist_knowledge_page`` so a
    later synth invocation sees it as an "existing" page in storage.
    """
    _cfg, root, storage = await api._with_storage(wiki)
    try:
        page = build_page(
            title=title,
            body=body or f"# {title}\n\nSeeded fixture page.\n",
            category=category,
        )
        write_page(root, page)
        await api._persist_knowledge_page(
            storage=storage,
            root=root,
            page=page,
            embedder=None,
            embedding_model="fake",
            text_version_id=None,
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_synth_prompt_includes_existing_pages_section(tmp_path: Path) -> None:
    """An existing K-layer page renders into the prompt's
    ``### Existing knowledge pages`` section as ``- Title (category)``."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    await _seed_wiki_page(wiki, title="Tesla", category="entity")

    _write_source(
        wiki,
        "fresh.md",
        "# Fresh source\n\nA short note that mentions Tesla in passing.\n",
    )

    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    llm = CapturingLLM()
    await api.synthesize(wiki, llm=llm, embedder=embedder)

    assert llm.calls, "synth must invoke the LLM at least once"
    prompt = llm.calls[0]
    assert "### Existing knowledge pages" in prompt, (
        "fresh-base synth must render the existing-pages section header (H3, "
        "nested under the template's ## Knowledge-base context heading)"
    )
    # End-to-end nesting: the assembled prompt must place the H3 sub-section
    # AFTER the template's H2 container — the renderer-side H3 literal and the
    # template-side H2 are two halves of one contract, and only the assembled
    # prompt proves they pair up.
    assert "\n## Knowledge-base context" in prompt
    assert prompt.index("## Knowledge-base context") < prompt.index(
        "### Existing knowledge pages"
    )
    assert "- Tesla [tesla] (entity)" in prompt, (
        "the seeded page must appear as a 'Title [slug] (category)' bullet"
    )


@pytest.mark.asyncio
async def test_synth_prompt_includes_batch_accumulator_after_first_group(
    tmp_path: Path,
) -> None:
    """Group 2 of the same source must see the page group 1 emitted via
    a ``### Already created in this batch`` section. Stage A 1:N fan-out
    runs groups serially against the same source — the second group
    needs to know what the first one already wrote so it can reference
    via ``[[Title]]`` instead of regenerating."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    # Force at least two groups: shrink target_tokens_per_group hard.
    _override_synth_cfg(wiki, target_tokens_per_group=80)

    long_body = "# Long source\n\n"
    for i in range(6):
        long_body += f"## Chapter {i}\n\n"
        for _ in range(20):
            long_body += (
                f"Paragraph in chapter {i} with enough words to push the chunk "
                "budget past the very low per-group token target set by this "
                "test, forcing the synth pipeline to emit at least two groups.\n\n"
            )
    _write_source(wiki, "long.md", long_body)

    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    llm = GroupKeyedLLM()
    await api.synthesize(wiki, llm=llm, embedder=embedder)

    assert len(llm.calls) >= 2, (
        f"expected long source to fan out into >=2 groups, "
        f"got {len(llm.calls)} calls"
    )
    second_prompt = llm.calls[1]
    assert "### Already created in this batch" in second_prompt, (
        "group 2's prompt must surface the batch accumulator section"
    )
    # Group 0 emitted a page titled "Group 0 page" of type "concept".
    assert "- Group 0 page [group-0-page] (concept)" in second_prompt, (
        "the page emitted by group 0 must appear in group 1's batch section"
    )


@pytest.mark.asyncio
async def test_synth_existing_pages_truncates_to_retrieval_top_k(
    tmp_path: Path,
) -> None:
    """When the full existing-pages render exceeds
    ``synth.existing_pages_max_bytes``, the section truncates to a
    vec_search-gated top-K (``synth.existing_pages_top_k``). Without
    truncation the prompt would balloon as the wiki grows, eventually
    overflowing the model's context window.
    """
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    # Set thresholds tight enough that a handful of pages overflow and
    # only top-K=3 survive the retrieval gate.
    _override_synth_cfg(
        wiki, existing_pages_max_bytes=200, existing_pages_top_k=3
    )

    # Seed enough pages that the full Title (category) render exceeds 200 B.
    # ~30 bytes per line, 12 pages -> ~360 bytes > 200.
    for i in range(12):
        await _seed_wiki_page(
            wiki, title=f"Seeded page {i}", category="concept"
        )

    _write_source(
        wiki,
        "fresh.md",
        "# Fresh source\n\nA short note for a fresh synth call.\n",
    )

    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    llm = CapturingLLM()
    await api.synthesize(wiki, llm=llm, embedder=embedder)

    assert llm.calls, "synth must invoke the LLM at least once"
    prompt = llm.calls[0]
    # Count "- Seeded page N (concept)" bullet lines under the existing-pages
    # section header. Must be at most top_k = 3 — proves the truncation fired.
    bullet_count = sum(
        1 for line in prompt.splitlines()
        if line.startswith("- Seeded page ")
    )
    assert 0 < bullet_count <= 3, (
        f"expected retrieval-gated truncation to <=3 bullets, "
        f"got {bullet_count} (full render had 12 pages)"
    )
    # The retrieval-gated branch must also carry the [slug] (a count-only
    # assertion would pass even if the slug were dropped from this branch).
    assert any(
        re.match(r"- Seeded page \d+ \[seeded-page-\d+\] \(concept\)", line)
        for line in prompt.splitlines()
    ), "retrieval-gated existing-pages bullets must render the [slug] shape"


@pytest.mark.asyncio
async def test_synth_force_all_skips_existing_pages_section(tmp_path: Path) -> None:
    """``dikw client synth --all`` is the documented "regenerate after a
    prompt/model change" path. Showing the LLM the OLD output of the
    SAME source would, combined with the zero-block-on-duplicate rule,
    cause the model to skip the regeneration the user explicitly
    requested. force_all must NOT render the base existing-pages
    section. (The in-batch accumulator still runs so groups within the
    same source coordinate.)"""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    await _seed_wiki_page(wiki, title="Tesla", category="entity")

    _write_source(
        wiki,
        "fresh.md",
        "# Fresh source\n\nA short note that mentions Tesla in passing.\n",
    )

    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    llm = CapturingLLM()
    await api.synthesize(wiki, llm=llm, embedder=embedder, force_all=True)

    assert llm.calls
    prompt = llm.calls[0]
    # ``## `` substring form catches a leak at either heading level.
    assert "## Existing knowledge pages" not in prompt, (
        "force_all=True must not surface the existing-pages section; "
        "regeneration would otherwise be suppressed by the duplicate rule"
    )
    assert "- Tesla [tesla] (entity)" not in prompt


@pytest.mark.asyncio
async def test_synth_existing_pages_falls_back_when_knowledge_unembedded(
    tmp_path: Path,
) -> None:
    """``--no-embed`` wikis (or wikis whose K-layer pages predate the
    active text version) have no WIKI vectors. Above the byte threshold
    the retrieval-gated branch's vec_search returns nothing for every
    chunk — without a fallback the LLM would see ``(no existing pages
    — fresh wiki)`` and be told to generate freely, dropping all
    duplicate-avoidance signal exactly when the wiki has the most to
    offer it. We fall back to a bounded prefix of the snapshot."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    _override_synth_cfg(
        wiki, existing_pages_max_bytes=200, existing_pages_top_k=3
    )

    # Seed enough pages that the full render exceeds 200 B; ``embedder=None``
    # in ``_seed_wiki_page`` means these pages have NO WIKI vectors.
    for i in range(12):
        await _seed_wiki_page(
            wiki, title=f"Seeded page {i}", category="concept"
        )

    _write_source(wiki, "fresh.md", "# Fresh\n\nSource body.\n")

    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    llm = CapturingLLM()
    await api.synthesize(wiki, llm=llm, embedder=embedder)

    prompt = llm.calls[0]
    assert "### Existing knowledge pages" in prompt, (
        "fallback must surface the existing-pages section, not the "
        "fresh-wiki sentinel"
    )
    bullet_count = sum(
        1 for line in prompt.splitlines()
        if line.startswith("- Seeded page ")
    )
    assert 0 < bullet_count <= 3, (
        f"unembedded-wiki fallback should emit <=top_k=3 bullets, "
        f"got {bullet_count}"
    )
