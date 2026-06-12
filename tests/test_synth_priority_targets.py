"""Phase 2 deterministic-pipeline tuning: existing-pages slug + priority-create.

Two surgical, deterministic-scoping additions to the synth fan-out
prompt assembly:

* **existing-pages with slug** — each existing/batch page bullet renders
  as ``- Title [slug] (category)`` instead of ``- Title (category)``.
  The slug is the deterministic kebab-case file identifier; surfacing it
  lets the LLM disambiguate two same-titled pages without guessing.

* **priority-create feedback loop** — wikilink targets that an earlier
  group of THIS source referenced but that resolve to no page (existing
  snapshot OR batch) are accumulated and surfaced to later groups under
  a ``### Priority targets (create if relevant)`` section, so a group
  whose content covers one creates it at the right title instead of
  leaving the graph broken. Resolution uses the SAME exact -> fuzzy ->
  collision rules ``resolve_links`` applies at persist time, so the
  signal never counts a target someone already authored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.api_synth import (
    _PRIORITY_SECTION_HEADER,
    _build_known_title_index,
    _render_existing_section,
    _render_priority_targets,
    _unresolved_wikilink_targets,
    _wikilink_resolves,
)
from dikw_core.domains.knowledge.links import build_fuzzy_index
from dikw_core.providers import LLMResponse

from .fakes import FakeEmbeddings, init_test_base

# --- #3 existing-pages slug rendering (pure) ---------------------------


def test_render_existing_section_includes_slug() -> None:
    """``_render_existing_section`` renders ``- Title [slug] (category)``."""
    rendered = _render_existing_section(
        [("Neural Network", "neural-network", "concept")],
        "Existing knowledge pages",
    )
    assert "- Neural Network [neural-network] (concept)" in rendered
    # H3, not H2 — the section nests under the template's
    # ``## Knowledge-base context`` heading instead of competing with it.
    assert rendered.startswith("### Existing knowledge pages")


def test_render_existing_section_empty_is_blank() -> None:
    assert _render_existing_section([], "Existing knowledge pages") == ""


# --- #4 unresolved-target detection (pure, mirrors resolve_links) ------


def test_unresolved_targets_exact_and_fuzzy_resolve_are_excluded() -> None:
    title_to_path = {
        "Tesla": "knowledge/entity/tesla.md",
        "Neural Network": "knowledge/concept/neural-network.md",
    }
    fuzzy = build_fuzzy_index(title_to_path)
    # Tesla: exact hit. Neural Networks: fuzzy plural-stem hit. SpaceX: miss.
    body = "Built by [[Tesla]] using [[Neural Networks]] and [[SpaceX]]."
    assert _unresolved_wikilink_targets(
        body, title_to_path=title_to_path, fuzzy_index=fuzzy
    ) == ["SpaceX"]


def test_unresolved_targets_ambiguous_collision_stays_unresolved() -> None:
    """A target that fuzzy-maps to >=2 distinct paths must NOT resolve —
    mirroring ``resolve_links``' refuse-to-guess rule — so it still
    surfaces as a (genuinely ambiguous) priority target."""
    title_to_path = {
        "Apple Inc": "knowledge/entity/apple-inc.md",
        "Apple Inc.": "knowledge/entity/apple-inc-2.md",
    }
    fuzzy = build_fuzzy_index(title_to_path)
    # ``apple inc`` is not an exact key (case differs) and fuzzy-maps to
    # both paths -> 2 candidates -> unresolved.
    out = _unresolved_wikilink_targets(
        "See [[apple inc]] here.", title_to_path=title_to_path, fuzzy_index=fuzzy
    )
    assert out == ["apple inc"]


def test_unresolved_targets_ignores_non_wikilinks() -> None:
    body = "A [markdown](other.md) link and https://example.com are not wikilinks."
    assert _unresolved_wikilink_targets(
        body, title_to_path={}, fuzzy_index={}
    ) == []


def test_unresolved_targets_skips_empty_and_punctuation_targets() -> None:
    """An anchor-only / blank / punctuation-only wikilink strips to an empty or
    keyless target; it must NOT surface as a create directive (no usable fuzzy
    key → uncreatable). Otherwise the priority section emits ``- [[]] (...)``."""
    body = "See [[#section]] and [[ ]] here. Also [[...]] and [[,]]."
    assert _unresolved_wikilink_targets(
        body, title_to_path={}, fuzzy_index={}
    ) == []


def test_build_known_index_lets_batch_satisfy_fuzzy_target() -> None:
    """A plural target left unresolved by an early group must be considered
    RESOLVED once a later batch page provides the singular — so priority-create
    drops it instead of nudging a duplicate. ``_build_known_title_index`` +
    ``_wikilink_resolves`` model exactly that filter."""
    snapshot: dict[str, str] = {}
    batch = [("Neural Network", "neural-network", "concept")]
    known = _build_known_title_index(snapshot, batch)
    fuzzy = build_fuzzy_index(known)
    # The accumulated (plural) target now fuzzy-resolves to the batch page.
    assert _wikilink_resolves(
        "Neural Networks", title_to_path=known, fuzzy_index=fuzzy
    )
    # An unrelated target stays unresolved.
    assert not _wikilink_resolves(
        "SpaceX", title_to_path=known, fuzzy_index=fuzzy
    )


# --- #4 priority section rendering (pure) ------------------------------


def test_render_priority_targets_empty_is_blank() -> None:
    assert _render_priority_targets([]) == ""


def test_render_priority_targets_lists_targets_with_counts() -> None:
    rendered = _render_priority_targets([("SpaceX", 3), ("Mars", 1)])
    # H3 for the same nesting reason as ``_render_existing_section``.
    assert rendered.startswith(f"### {_PRIORITY_SECTION_HEADER}")
    assert "- [[SpaceX]] (3 prior references)" in rendered
    assert "- [[Mars]] (1 prior reference)" in rendered  # singular


# --- #4 integration: priority surfaces for later groups ----------------


def _page_block(slug: str, title: str, body: str, *, category: str = "concept") -> str:
    return (
        f'<page category="{category}" slug="{slug}">\n'
        f"---\ntags: [{category}]\n---\n\n"
        f"# {title}\n\n{body}\n</page>"
    )


class _ScriptedLLM:
    """Emits a scripted ``<page>`` block per call index; later/unspecified
    calls emit nothing. Records every prompt so a test can assert per-group
    prompt content."""

    def __init__(self, scripts: dict[int, str]) -> None:
        self._scripts = scripts
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
        text = self._scripts.get(idx, "(no page worth writing)")
        return LLMResponse(text=text, finish_reason="end_turn")


def _write_source(wiki: Path, name: str, body: str) -> None:
    src_dir = wiki / "sources" / "notes"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / name).write_text(body, encoding="utf-8")


def _multi_group_body() -> str:
    body = "# Long source\n\n"
    for i in range(6):
        body += f"## Chapter {i}\n\n"
        for _ in range(20):
            body += (
                f"Paragraph in chapter {i} with enough words to push the chunk "
                "budget past the very low per-group token target set by this "
                "test, forcing the synth pipeline to emit at least two groups.\n\n"
            )
    return body


def _force_multi_group(wiki: Path) -> None:
    """Shrink target_tokens_per_group so a long source fans into many groups."""
    from dikw_core.config import dump_config_yaml, load_config

    cfg_path = wiki / "dikw.yml"
    cfg = load_config(cfg_path)
    cfg.synth.target_tokens_per_group = 80
    cfg_path.write_text(dump_config_yaml(cfg), encoding="utf-8")


@pytest.mark.asyncio
async def test_priority_section_surfaces_unresolved_for_later_group(
    tmp_path: Path,
) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    _force_multi_group(wiki)
    _write_source(wiki, "long.md", _multi_group_body())

    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    # Group 0 references an unauthored [[SpaceX]]; later groups emit nothing.
    llm = _ScriptedLLM(
        {0: _page_block("rockets", "Rockets", "Orbital rockets by [[SpaceX]].")}
    )
    await api.synthesize(wiki, llm=llm, embedder=embedder)

    assert len(llm.calls) >= 2, f"expected >=2 groups, got {len(llm.calls)} calls"
    # Group 0 (first prompt) has no prior groups -> no priority section.
    # The ``## `` form is a substring of the ``### `` render, so this negative
    # catches a leak at EITHER heading level (a future H2 revert included).
    assert f"## {_PRIORITY_SECTION_HEADER}" not in llm.calls[0]
    # Group 1 must surface the [[SpaceX]] target group 0 left unresolved,
    # rendered once with the singular count.
    assert f"### {_PRIORITY_SECTION_HEADER}" in llm.calls[1]
    assert "- [[SpaceX]] (1 prior reference)" in llm.calls[1]
    # Group 0 created a page, so group 1's batch section MUST render — assert
    # it outright (a conditional gate here would go vacuous if the batch
    # accumulator ever broke) and pin the priority-block-first ordering.
    assert "### Already created in this batch" in llm.calls[1]
    assert llm.calls[1].index(f"### {_PRIORITY_SECTION_HEADER}") < llm.calls[1].index(
        "### Already created in this batch"
    )


@pytest.mark.asyncio
async def test_priority_target_dropped_once_satisfied_by_later_group(
    tmp_path: Path,
) -> None:
    """A target an early group referenced, that a middle group then CREATES,
    must NOT reappear in a still-later group's priority section."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    _force_multi_group(wiki)
    _write_source(wiki, "long.md", _multi_group_body())

    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    # Group 0 references [[Foo]] (unresolved); group 1 authors the Foo page.
    llm = _ScriptedLLM(
        {
            0: _page_block("rockets", "Rockets", "Built using [[Foo]] tech."),
            1: _page_block("foo", "Foo", "Foo is a thing."),
        }
    )
    await api.synthesize(wiki, llm=llm, embedder=embedder)

    assert len(llm.calls) >= 3, f"expected >=3 groups, got {len(llm.calls)}"
    # Group 1 still sees Foo as a priority (group 0 left it unresolved).
    assert "[[Foo]]" in llm.calls[1]
    # Group 2: Foo now exists in the batch -> dropped, nothing else pending.
    # ``## `` substring form catches a leak at either heading level.
    assert f"## {_PRIORITY_SECTION_HEADER}" not in llm.calls[2]


@pytest.mark.asyncio
async def test_priority_create_runs_under_force_all(tmp_path: Path) -> None:
    """force_all skips the existing-pages snapshot (snapshot=None) but the
    priority loop still surfaces targets unresolved against the in-batch set —
    consistent with force_all's regenerate-everything intent."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    _force_multi_group(wiki)
    _write_source(wiki, "long.md", _multi_group_body())

    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    llm = _ScriptedLLM(
        {0: _page_block("rockets", "Rockets", "Orbital rockets by [[SpaceX]].")}
    )
    await api.synthesize(wiki, llm=llm, embedder=embedder, force_all=True)

    assert len(llm.calls) >= 2
    assert f"### {_PRIORITY_SECTION_HEADER}" in llm.calls[1]
    assert "[[SpaceX]]" in llm.calls[1]


@pytest.mark.asyncio
async def test_priority_create_records_knowledge_log_note(tmp_path: Path) -> None:
    """Surfacing a priority target appends a ``priority-create`` note that the
    outer synthesize loop persists into the knowledge_log table."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    _force_multi_group(wiki)
    _write_source(wiki, "long.md", _multi_group_body())

    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    llm = _ScriptedLLM(
        {0: _page_block("rockets", "Rockets", "Orbital rockets by [[SpaceX]].")}
    )
    await api.synthesize(wiki, llm=llm, embedder=embedder)

    _cfg, _root, storage = await api._with_storage(wiki)
    try:
        notes = [e.note or "" for e in await storage.list_knowledge_log()]
    finally:
        await storage.close()
    assert any("priority-create" in n for n in notes), (
        f"expected a priority-create knowledge_log note, got {notes}"
    )
