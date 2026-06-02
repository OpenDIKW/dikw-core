"""K-layer synth persist fault tolerance — deactivate-on-failure parity with D/W.

Synth's per-page persist loop (api_synth.py) historically had NO try/except:
a hard storage failure mid-``persist_knowledge`` (e.g. ``replace_links_from``
raising after ``upsert_document`` + ``replace_chunks`` already committed)
propagated uncaught, aborting the whole synth run and leaving a
half-written-but-``active=True`` knowledge page that still surfaced in
retrieval. D (``api.ingest``) and W (``write_wisdom_page``) already
deactivate-on-failure; these tests pin the same contract for K.

A persist exception must NOT be confused with deferred embedding: a flaky
embedder is retry-skipped into ``chunks_pending_embedding`` without raising
(test_persist_knowledge.py), so these tests inject a HARD storage failure.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from dikw_core import api, api_synth
from dikw_core.providers import LLMResponse

from .fakes import FakeEmbeddings, init_test_base

FIXTURES = Path(__file__).parent / "fixtures" / "notes"

# One <page> per source; slugs map to knowledge/concept/<slug>.md.
_SCRIPT = {
    "sources/notes/dikw.md": (
        '<page category="concept" slug="dikw-pyramid">\n'
        "---\ntags: [dikw, pyramid]\n---\n\n"
        "# DIKW pyramid\n\n"
        "The DIKW pyramid organises raw data into four layers. "
        "See [[Karpathy LLM Wiki]] for a related pattern.\n"
        "</page>"
    ),
    "sources/notes/karpathy-wiki.md": (
        '<page category="concept" slug="karpathy-llm-wiki">\n'
        "---\ntags: [pattern, llm]\n---\n\n"
        "# Karpathy LLM Wiki\n\n"
        "Karpathy's pattern defines a wiki built from source documents. "
        "It complements the [[DIKW pyramid]] model.\n"
        "</page>"
    ),
    "sources/notes/retrieval.md": (
        '<page category="concept" slug="hybrid-retrieval">\n'
        "---\ntags: [search]\n---\n\n"
        "# Hybrid retrieval\n\n"
        "BM25 + dense vectors fused with RRF. Useful background for the "
        "[[DIKW pyramid]] engine.\n"
        "</page>"
    ),
}

# The page whose persist we force to fail mid-pipeline.
_FAIL_PATH = "knowledge/concept/hybrid-retrieval.md"
_OK_PATH = "knowledge/concept/dikw-pyramid.md"


class ScriptedLLM:
    """Returns a canned <page> response keyed by which source is in the prompt."""

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


def _seed(tmp_path: Path) -> Path:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    dest = wiki / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    return wiki


def _patch_persist_failure(
    monkeypatch: pytest.MonkeyPatch, fail_doc_id: str, exc: BaseException
):
    """Wrap synth's storage so ``replace_links_from`` raises ``exc`` for one
    doc_id — a hard failure AFTER upsert_document + replace_chunks committed.
    Returns the original ``_with_storage`` for re-opening a clean storage.
    """
    original = api_synth._with_storage

    async def patched(path: object) -> object:
        cfg, root, storage = await original(path)  # type: ignore[arg-type]
        original_rlf = storage.replace_links_from

        async def maybe_boom(doc_id: object, resolved: object) -> object:
            if doc_id == fail_doc_id:
                raise exc
            return await original_rlf(doc_id, resolved)  # type: ignore[arg-type]

        storage.replace_links_from = maybe_boom  # type: ignore[method-assign]
        return cfg, root, storage

    monkeypatch.setattr(api_synth, "_with_storage", patched)
    return original


@pytest.mark.asyncio
async def test_synth_persist_failure_deactivates_page_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hard persist failure on one page deactivates that page, records it
    in ``SynthReport.persist_errors``, and lets the run continue persisting
    the remaining pages — instead of aborting and leaving a half-written
    active page in retrieval."""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    fail_doc_id = api._doc_id_for(api.Layer.KNOWLEDGE, _FAIL_PATH)
    original = _patch_persist_failure(
        monkeypatch, fail_doc_id, RuntimeError("simulated link reconcile outage")
    )

    llm = ScriptedLLM(_SCRIPT)
    # Must NOT raise — the failing page is recorded and the run continues.
    report = await api.synthesize(wiki, llm=llm, embedder=embedder)

    assert report.created == 2, "the two healthy pages must still be created"
    assert [e.path for e in report.persist_errors] == [_FAIL_PATH]
    assert "simulated link reconcile outage" in report.persist_errors[0].message

    # Storage: failed page parked inactive, healthy page active.
    cfg, _root, storage = await original(wiki)  # type: ignore[misc]
    del cfg
    try:
        failed = await storage.get_document(fail_doc_id)
        assert failed is not None and failed.active is False, (
            "a page whose persist failed must be deactivated so it is not "
            "retrievable"
        )
        ok = await storage.get_document(
            api._doc_id_for(api.Layer.KNOWLEDGE, _OK_PATH)
        )
        assert ok is not None and ok.active is True
    finally:
        await storage.close()

    # The failed page must not be reachable via the read-by-path seam.
    with pytest.raises(api.PageNotFound):
        await api.read_page(wiki, _FAIL_PATH)


@pytest.mark.asyncio
async def test_failed_source_not_marked_done_so_resynth_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source with a failed page must NOT be marked ``synth_source_done``,
    so the next default synth re-processes it and the deactivated page is
    rebuilt to ``active=True``. This is K's recovery path (K has no
    scan-based reindex): re-running synth restores a page parked inactive by
    a transient failure."""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    fail_doc_id = api._doc_id_for(api.Layer.KNOWLEDGE, _FAIL_PATH)
    _patch_persist_failure(monkeypatch, fail_doc_id, RuntimeError("boom"))

    llm = ScriptedLLM(_SCRIPT)
    first = await api.synthesize(wiki, llm=llm, embedder=embedder)
    assert [e.path for e in first.persist_errors] == [_FAIL_PATH]

    # Lift the failure (restore the real _with_storage) and re-run DEFAULT
    # synth — the two healthy sources are skipped (done), but the source that
    # produced the failed page must be retried.
    monkeypatch.undo()
    second = await api.synthesize(wiki, llm=llm, embedder=embedder)
    assert second.persist_errors == ()
    assert second.skipped == 2, "the two healthy sources stay marked done"
    assert second.created + second.updated == 1, "the failed source is retried"

    cfg, _root, storage = await api_synth._with_storage(wiki)
    del cfg
    try:
        recovered = await storage.get_document(fail_doc_id)
        assert recovered is not None and recovered.active is True, (
            "re-running synth must rebuild the page deactivated by the prior "
            "failure"
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_synth_persist_cancelled_deactivates_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``asyncio.CancelledError`` inherits from ``BaseException``, so a bare
    ``except Exception`` misses it. A cancellation mid-persist must still
    deactivate the in-flight page (so it isn't left half-written-but-active)
    and then re-raise to abort the run — matching W's handling."""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    fail_doc_id = api._doc_id_for(api.Layer.KNOWLEDGE, _FAIL_PATH)
    original = _patch_persist_failure(
        monkeypatch, fail_doc_id, asyncio.CancelledError()
    )

    llm = ScriptedLLM(_SCRIPT)
    with pytest.raises(asyncio.CancelledError):
        await api.synthesize(wiki, llm=llm, embedder=embedder)

    cfg, _root, storage = await original(wiki)  # type: ignore[misc]
    del cfg
    try:
        doc = await storage.get_document(fail_doc_id)
        assert doc is not None and doc.active is False, (
            "cancellation mid-persist must still deactivate the in-flight page"
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_forced_resynth_persist_failure_invalidates_prior_done_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source synthesized successfully earlier (so it carries a
    ``synth_source_done`` marker) and then re-synthesized with
    ``force_all=True`` where its page fails persist must NOT stay marked done.
    Withholding the *new* marker is not enough — the *prior* marker would
    still make the next default synth skip the source, stranding the
    deactivated page. The failure must invalidate the stale done marker so
    the recovery path actually fires. (codex review P1.)"""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)
    llm = ScriptedLLM(_SCRIPT)

    # 1. Clean synth → every source gets a synth_source_done marker.
    first = await api.synthesize(wiki, llm=llm, embedder=embedder)
    assert first.created == 3

    # 2. Forced re-synth where the hybrid-retrieval page fails persist.
    fail_doc_id = api._doc_id_for(api.Layer.KNOWLEDGE, _FAIL_PATH)
    _patch_persist_failure(monkeypatch, fail_doc_id, RuntimeError("boom"))
    forced = await api.synthesize(
        wiki, llm=llm, embedder=embedder, force_all=True
    )
    assert [e.path for e in forced.persist_errors] == [_FAIL_PATH]

    # 3. Next DEFAULT synth must re-process the failed source (its stale done
    #    marker is invalidated) and rebuild the deactivated page.
    monkeypatch.undo()
    third = await api.synthesize(wiki, llm=llm, embedder=embedder)
    assert third.created + third.updated == 1, (
        "the failed source must be retried by default synth, not skipped"
    )

    cfg, _root, storage = await api_synth._with_storage(wiki)
    del cfg
    try:
        recovered = await storage.get_document(fail_doc_id)
        assert recovered is not None and recovered.active is True
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_cancelled_forced_resynth_invalidates_prior_done_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation mid-persist during a ``force_all`` re-synth must ALSO
    invalidate the source's prior ``synth_source_done`` (write a
    ``synth_source_failed`` marker before re-raising), so the deactivated page
    is recoverable by the next default synth instead of stranded behind the
    stale done marker. (codex review round 2 P1.)"""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)
    llm = ScriptedLLM(_SCRIPT)

    first = await api.synthesize(wiki, llm=llm, embedder=embedder)
    assert first.created == 3

    fail_doc_id = api._doc_id_for(api.Layer.KNOWLEDGE, _FAIL_PATH)
    _patch_persist_failure(monkeypatch, fail_doc_id, asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await api.synthesize(wiki, llm=llm, embedder=embedder, force_all=True)

    monkeypatch.undo()
    await api.synthesize(wiki, llm=llm, embedder=embedder)

    cfg, _root, storage = await api_synth._with_storage(wiki)
    del cfg
    try:
        recovered = await storage.get_document(fail_doc_id)
        assert recovered is not None and recovered.active is True, (
            "a cancelled --all re-synth must not strand the deactivated page"
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_legacy_backfill_preserves_failed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a legacy base (old per-page ``synth`` rows, no backfill sentinel), a
    ``synth_source_failed`` marker must survive the one-time legacy backfill:
    the backfill must NOT re-add the failed source to ``already`` via
    ``legacy_dst_sources``, or the deactivated page is stranded. (codex review
    round 2 P1.)"""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)
    llm = ScriptedLLM(_SCRIPT)

    # Simulate a legacy base: a per-page ``synth`` row for the source whose
    # page we will fail, with NO ``synth_source_done`` and NO backfill
    # sentinel (the pre-fan-out shape the backfill block migrates).
    from dikw_core.schemas import KnowledgeLogEntry

    cfg, _root, storage = await api_synth._with_storage(wiki)
    del cfg
    try:
        await storage.append_knowledge_log(
            KnowledgeLogEntry(
                ts=1.0,
                action="synth",
                src="sources/notes/retrieval.md",
                dst=_FAIL_PATH,
            )
        )
    finally:
        await storage.close()

    # First new-pipeline run is force_all; the hybrid-retrieval page fails.
    fail_doc_id = api._doc_id_for(api.Layer.KNOWLEDGE, _FAIL_PATH)
    _patch_persist_failure(monkeypatch, fail_doc_id, RuntimeError("boom"))
    forced = await api.synthesize(
        wiki, llm=llm, embedder=embedder, force_all=True
    )
    assert [e.path for e in forced.persist_errors] == [_FAIL_PATH]

    # Next default synth runs the legacy backfill. The failed source must NOT
    # be marked done by it — it must be re-processed and the page rebuilt.
    monkeypatch.undo()
    await api.synthesize(wiki, llm=llm, embedder=embedder)

    cfg, _root, storage = await api_synth._with_storage(wiki)
    del cfg
    try:
        recovered = await storage.get_document(fail_doc_id)
        assert recovered is not None and recovered.active is True, (
            "legacy backfill must not resurrect a source with a "
            "synth_source_failed marker"
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_synth_without_embedder_keeps_pages_active(tmp_path: Path) -> None:
    """Regression guard for the pending≠failure boundary: a synth run with no
    active embed version defers embedding (pages land with no vectors) but
    must keep every page ``active=True`` — deferred embedding is NOT a persist
    failure and must NOT trip the new deactivate-on-failure path."""
    wiki = _seed(tmp_path)
    # Ingest with no embedder → no active text version, so synth drops its
    # embedder and every page's chunks are pending.
    await api.ingest(wiki)

    llm = ScriptedLLM(_SCRIPT)
    report = await api.synthesize(wiki, llm=llm)
    assert report.created == 3
    assert report.persist_errors == ()

    cfg, _root, storage = await api_synth._with_storage(wiki)
    del cfg
    try:
        for slug in ("dikw-pyramid", "karpathy-llm-wiki", "hybrid-retrieval"):
            doc = await storage.get_document(
                api._doc_id_for(api.Layer.KNOWLEDGE, f"knowledge/concept/{slug}.md")
            )
            assert doc is not None and doc.active is True
        # And the deferred page is FTS-retrievable (no vectors required).
        fail_doc_id = api._doc_id_for(api.Layer.KNOWLEDGE, _FAIL_PATH)
        hits = await storage.fts_search(
            "hybrid retrieval", limit=5, layer=api.Layer.KNOWLEDGE
        )
        assert any(h.doc_id == fail_doc_id for h in hits)
    finally:
        await storage.close()
