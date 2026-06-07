"""``dikw client synth --verify`` — post-synth self-check over THIS run's pages.

Phase 1 flagship (verification roadmap 1.1). After synth writes K pages,
``synthesize(verify=True)`` runs a deterministic, no-extra-LLM体检 scoped to the
pages this run created/updated and attaches a :class:`SynthVerifyReport` to the
returned :class:`SynthReport`. The verdict is the "open the vault and click
around" pass made automatic — so a synth user doesn't have to remember to run
``dikw client lint`` afterwards.

The change is purely additive: it READS synth output, never alters it. These
tests pin the three gated legs (persist / scoped-lint / semantic-duplicate), the
two informational surfaces (orphan_page / unresolved_wikilinks), and the
loud-skip of the duplicate leg when no embedder is wired (0.6 contract: a green
verify must never imply "no duplicates" when the check never ran).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dikw_core import api, api_synth
from dikw_core.providers import LLMResponse

from .fakes import FakeEmbeddings, init_test_base

FIXTURES = Path(__file__).parent / "fixtures" / "notes"


class _ScriptedLLM:
    """Returns a canned ``<page>`` response keyed by which source is in the prompt."""

    def __init__(self, script: dict[str, str]) -> None:
        self._script = script

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
        _ = (system, model, max_tokens, temperature, tools)
        for src_path, resp in self._script.items():
            if src_path in user:
                return LLMResponse(text=resp, finish_reason="end_turn")
        raise AssertionError(f"no script entry matched prompt: {user[:200]}")


def _page(slug: str, title: str, body: str, *, tags: str = "sample") -> str:
    return (
        f'<page category="concept" slug="{slug}">\n'
        f"---\ntags: [{tags}]\n---\n\n"
        f"# {title}\n\n{body}\n"
        "</page>"
    )


# Three cross-linked pages: no orphans, no broken links, atomic, categorized.
# dikw → karpathy + retrieval; karpathy → dikw; retrieval → dikw. Every page has
# at least one inbound wikilink, so a clean run produces zero gated lint issues
# AND zero orphans.
_CLEAN_SCRIPT = {
    "sources/notes/dikw.md": _page(
        "dikw-pyramid",
        "DIKW pyramid",
        "The DIKW pyramid organises raw data into four layers. "
        "See [[Karpathy LLM Wiki]] and [[Hybrid retrieval]].",
    ),
    "sources/notes/karpathy-wiki.md": _page(
        "karpathy-llm-wiki",
        "Karpathy LLM Wiki",
        "Karpathy's pattern builds a wiki from sources. "
        "It complements the [[DIKW pyramid]] model.",
    ),
    "sources/notes/retrieval.md": _page(
        "hybrid-retrieval",
        "Hybrid retrieval",
        "BM25 + dense vectors fused with RRF, useful background for the "
        "[[DIKW pyramid]] engine.",
    ),
}


def _seed(tmp_path: Path) -> Path:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    dest = wiki / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    return wiki


# ---- the happy path: clean output verifies clean -------------------------


@pytest.mark.asyncio
async def test_clean_run_verify_passes(tmp_path: Path) -> None:
    """A fully-settled, cross-linked vault verifies completely clean.

    Two synth passes: the first persists each page but forward references to
    pages authored later in the SAME run stay unresolved (the documented
    out-of-order-write limitation — K has no scan-based reindex), so the last
    pages are still orphan. A second ``force_all`` pass re-resolves every link
    against the now-complete base, so the verify pass sees zero orphans, zero
    broken links, and zero near-duplicates."""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    await api.synthesize(wiki, llm=_ScriptedLLM(_CLEAN_SCRIPT), embedder=embedder)
    report = await api.synthesize(
        wiki,
        llm=_ScriptedLLM(_CLEAN_SCRIPT),
        embedder=embedder,
        force_all=True,
        verify=True,
    )

    assert report.updated == 3
    v = report.verify
    assert v is not None
    assert v.passed is True
    assert v.persist_ok is True
    assert v.lint_ok is True
    assert v.duplicate_ok is True
    assert v.pages_checked == 3
    assert v.lint_findings == ()
    assert v.orphan_pages == ()
    # The duplicate leg ran (embedder wired) and found nothing near-duplicate.
    assert v.duplicate_checked is True
    assert v.duplicate_ratio is not None and v.duplicate_ratio <= v.max_duplicate_ratio


@pytest.mark.asyncio
async def test_verify_off_by_default(tmp_path: Path) -> None:
    """Without ``verify=True`` the report carries no verify section — the
    post-pass is strictly opt-in (it costs an extra lint scan + embed call)."""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    report = await api.synthesize(
        wiki, llm=_ScriptedLLM(_CLEAN_SCRIPT), embedder=embedder
    )
    assert report.verify is None


# ---- gated leg: scoped lint (broken wikilink) ----------------------------


@pytest.mark.asyncio
async def test_broken_wikilink_fails_verify(tmp_path: Path) -> None:
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    script = dict(_CLEAN_SCRIPT)
    # dikw now references a page nobody authors → broken_wikilink on dikw.
    script["sources/notes/dikw.md"] = _page(
        "dikw-pyramid",
        "DIKW pyramid",
        "The DIKW pyramid has four layers. See [[Karpathy LLM Wiki]], "
        "[[Hybrid retrieval]] and [[Nonexistent Phantom Page]].",
    )

    report = await api.synthesize(
        wiki, llm=_ScriptedLLM(script), embedder=embedder, verify=True
    )

    v = report.verify
    assert v is not None
    assert v.passed is False
    assert v.lint_ok is False
    kinds = {f.kind for f in v.lint_findings}
    assert "broken_wikilink" in kinds
    assert any(
        f.path == "knowledge/concept/dikw-pyramid.md" for f in v.lint_findings
    )


# ---- gated leg: semantic duplicate ---------------------------------------


@pytest.mark.asyncio
async def test_near_duplicate_pages_fail_verify(tmp_path: Path) -> None:
    """Two pages with near-identical bodies (distinct slugs, so
    ``dedup_pages_by_slug`` does NOT merge them) trip the semantic duplicate
    gate that the exact ``duplicate_title`` lint can never catch. FakeEmbeddings
    is a lexical bag-of-words, so identical bodies → cosine ~1.0 ≥ tau."""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    shared = (
        "Reinforcement learning trains an agent through reward signals across "
        "many episodes of trial and error in a Markov decision process."
    )
    # Two distinct slugs/titles that cross-link (no orphans, no broken links),
    # but whose prose bodies are byte-identical → maximal duplicate ratio.
    script = {
        "sources/notes/dikw.md": _page(
            "reinforcement-learning",
            "Reinforcement learning",
            f"{shared} See [[Reward shaping]].",
        ),
        "sources/notes/karpathy-wiki.md": _page(
            "reward-shaping",
            "Reward shaping",
            f"{shared} See [[Reinforcement learning]].",
        ),
        # third source emits nothing extra to keep the pair isolated
        "sources/notes/retrieval.md": _page(
            "hybrid-retrieval",
            "Hybrid retrieval",
            "BM25 fused with dense vectors via RRF. See [[Reinforcement learning]].",
        ),
    }

    report = await api.synthesize(
        wiki, llm=_ScriptedLLM(script), embedder=embedder, verify=True
    )

    v = report.verify
    assert v is not None
    assert v.duplicate_checked is True
    assert v.duplicate_ratio is not None
    assert v.duplicate_ratio > v.max_duplicate_ratio, v.duplicate_ratio
    assert v.duplicate_ok is False
    assert v.passed is False


# ---- loud-skip: no embedder means the duplicate leg does NOT run ----------


@pytest.mark.asyncio
async def test_no_embedder_skips_duplicate_leg_loudly(tmp_path: Path) -> None:
    """With no active embed version the duplicate leg is SKIPPED — not silently
    passed. ``duplicate_checked`` is False and ``duplicate_ratio`` is None so a
    consumer can tell "no duplicates" apart from "never measured" (0.6 contract).
    persist + lint still run, so verify still passes a clean run."""
    wiki = _seed(tmp_path)
    # Ingest without an embedder → no active text version → synth drops its
    # embedder → the duplicate leg cannot run.
    await api.ingest(wiki)

    report = await api.synthesize(
        wiki, llm=_ScriptedLLM(_CLEAN_SCRIPT), verify=True
    )

    v = report.verify
    assert v is not None
    assert v.duplicate_checked is False
    assert v.duplicate_ratio is None
    # A skipped duplicate leg is not a failure — the clean run still passes.
    assert v.duplicate_ok is True
    assert v.passed is True


# ---- gated leg: persist failure ------------------------------------------


@pytest.mark.asyncio
async def test_persist_failure_fails_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page deactivated by a hard persist failure must fail verify — a
    half-written ``active=False`` page is never clean output."""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    fail_path = "knowledge/concept/hybrid-retrieval.md"
    fail_doc_id = api._doc_id_for(api.Layer.KNOWLEDGE, fail_path)

    original_rlf = None

    async def _boom(doc_id: object, resolved: object) -> object:
        if doc_id == fail_doc_id:
            raise RuntimeError("simulated link reconcile outage")
        assert original_rlf is not None
        return await original_rlf(doc_id, resolved)  # type: ignore[arg-type]

    original_with_storage = api_synth._with_storage

    async def _patched(path: object) -> object:
        nonlocal original_rlf
        cfg, root, storage = await original_with_storage(path)  # type: ignore[arg-type]
        original_rlf = storage.replace_links_from
        storage.replace_links_from = _boom  # type: ignore[method-assign]
        return cfg, root, storage

    monkeypatch.setattr(api_synth, "_with_storage", _patched)

    report = await api.synthesize(
        wiki, llm=_ScriptedLLM(_CLEAN_SCRIPT), embedder=embedder, verify=True
    )

    assert [e.path for e in report.persist_errors] == [fail_path]
    v = report.verify
    assert v is not None
    assert v.persist_ok is False
    assert v.persist_error_count == 1
    assert v.passed is False


# ---- informational: orphan pages are surfaced, not gated -----------------


@pytest.mark.asyncio
async def test_orphan_pages_surfaced_but_not_gated(tmp_path: Path) -> None:
    """Three standalone pages with no wikilinks at all → every page is orphan
    (nothing links in). Orphans are SURFACED on ``orphan_pages`` but must NOT
    fail verify: a freshly synthesised page is legitimately orphan until later
    pages cite it, so gating it would make ``--verify`` perpetually red."""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    script = {
        "sources/notes/dikw.md": _page(
            "dikw-pyramid", "DIKW pyramid", "Four stacked layers of meaning."
        ),
        "sources/notes/karpathy-wiki.md": _page(
            "karpathy-llm-wiki", "Karpathy LLM Wiki", "A wiki built from sources."
        ),
        "sources/notes/retrieval.md": _page(
            "hybrid-retrieval", "Hybrid retrieval", "BM25 fused with dense vectors."
        ),
    }

    report = await api.synthesize(
        wiki, llm=_ScriptedLLM(script), embedder=embedder, verify=True
    )

    v = report.verify
    assert v is not None
    assert len(v.orphan_pages) == 3
    assert v.lint_findings == ()
    assert v.lint_ok is True
    assert v.passed is True
