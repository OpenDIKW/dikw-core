"""Tests for the ``missing_file`` drift lint kind + ``MissingFileFixer``.

``missing_file`` closes the "delete the source file, the active document
row is stuck forever" gap (ADR-0005 PR2). It spans **all three** DIKW
layers: a ``documents`` row (active) whose backing file is gone from disk
is an orphan, whether it's a ``sources/`` (D), ``knowledge/`` (K), or
``wisdom/`` (W) file. The deterministic ``MissingFileFixer`` proposes a
single ``purge_document`` op that calls ``Storage.delete_document`` —
purging the row + its outgoing edges, leaving inbound edges to surface as
``broken_wikilink`` (the D5 "expose, never silently rewrite" rule).

Unlike the page-mutating lint ops, ``missing_file`` can't be suppressed
via ``lint: {skip}`` frontmatter — the file is gone, there is nowhere to
put the annotation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.domains.data.path_norm import doc_id_for
from dikw_core.domains.knowledge.lint import LintIssue, run_lint
from dikw_core.domains.knowledge.lint_fix import FixerContext
from dikw_core.schemas import Layer, LinkRecord, LinkType

from .fakes import init_test_base, seed_doc


@pytest.fixture()
def base_root(tmp_path: Path) -> Path:
    root = tmp_path / "base"
    init_test_base(root)
    return root


async def _run_lint(base: Path):
    _cfg, root, storage = await api._with_storage(base)
    try:
        return await run_lint(storage, root=root)
    finally:
        await storage.close()


# ---- Detection -----------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_file_detects_deleted_source(base_root: Path) -> None:
    """A D-layer ``sources/`` row whose file was deleted outside dikw → one
    ``missing_file`` issue. This is the canonical "delete source → orphan
    D row" gap the kind exists to close — D rows were never scanned by
    ``run_lint`` before."""
    path = "sources/notes/raw.md"
    await seed_doc(base_root, layer=Layer.SOURCE, path=path, body="# Raw\n", title="Raw")
    (base_root / path).unlink()

    report = await _run_lint(base_root)
    mf = [i for i in report.issues if i.kind == "missing_file"]
    assert [i.path for i in mf] == [path]


@pytest.mark.asyncio
async def test_missing_file_detects_deleted_knowledge_page(base_root: Path) -> None:
    """A K-page row whose file is gone surfaces as ``missing_file`` instead
    of being silently skipped — the old ``run_lint`` ``continue`` swallowed
    these."""
    path = "knowledge/concepts/dead.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=path, body="# Dead\n", title="Dead"
    )
    (base_root / path).unlink()

    report = await _run_lint(base_root)
    assert any(i.kind == "missing_file" and i.path == path for i in report.issues)


@pytest.mark.asyncio
async def test_missing_file_detects_deleted_wisdom_page(base_root: Path) -> None:
    path = "wisdom/holo/never.md"
    await seed_doc(
        base_root, layer=Layer.WISDOM, path=path, body="# Never\n", title="Never"
    )
    (base_root / path).unlink()

    report = await _run_lint(base_root)
    assert any(i.kind == "missing_file" and i.path == path for i in report.issues)


@pytest.mark.asyncio
async def test_missing_file_clean_when_files_present(base_root: Path) -> None:
    """All three layers' files present → zero ``missing_file``. Pins the
    "clean state" so a regression that always emits the kind would fail."""
    await seed_doc(
        base_root, layer=Layer.SOURCE, path="sources/a.md", body="# A\n", title="A"
    )
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path="knowledge/concepts/b.md",
        body="# B\n", title="B",
    )
    await seed_doc(
        base_root, layer=Layer.WISDOM, path="wisdom/holo/c.md", body="# C\n", title="C"
    )

    report = await _run_lint(base_root)
    assert not any(i.kind == "missing_file" for i in report.issues)


@pytest.mark.asyncio
async def test_missing_file_ignores_inactive_rows(base_root: Path) -> None:
    """``missing_file`` only targets *active* rows — the visible projection.
    An ``active=False`` row (e.g. deactivated after a failed persist) whose
    file is gone is already hidden from retrieval and is out of scope."""
    path = "sources/inactive.md"
    await seed_doc(
        base_root, layer=Layer.SOURCE, path=path, body="# X\n", title="X", active=False
    )
    (base_root / path).unlink()

    report = await _run_lint(base_root)
    assert not any(i.kind == "missing_file" for i in report.issues)


# ---- Fixer + apply -------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_file_propose_emits_purge_op(base_root: Path) -> None:
    """``lint propose --rule missing_file`` emits one ``purge_document`` op
    carrying the resolved layer; ``expected_hash`` is ``None`` (the safety
    invariant is "file absent", not "file bytes unchanged")."""
    path = "knowledge/concepts/dead.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=path, body="# Dead\n", title="Dead"
    )
    (base_root / path).unlink()

    proposal_report = await api.lint_propose(base_root, rule="missing_file", limit=10)
    assert len(proposal_report.proposals) == 1
    proposal = proposal_report.proposals[0]
    assert proposal.source == "heuristic"
    assert proposal.issue_path == path
    op = proposal.operations[0]
    assert op.kind == "purge_document"
    assert op.path == path
    assert op.layer == Layer.KNOWLEDGE
    assert op.expected_hash is None


@pytest.mark.asyncio
async def test_missing_file_apply_purges_row(base_root: Path) -> None:
    """Apply purges the orphaned row (``get_document`` → ``None``), lists it
    under ``purged_documents``, and the next lint pass is clean of
    ``missing_file``."""
    path = "wisdom/holo/gone.md"
    await seed_doc(
        base_root, layer=Layer.WISDOM, path=path, body="# Gone\n", title="Gone"
    )
    (base_root / path).unlink()
    doc_id = doc_id_for(Layer.WISDOM, path)

    proposal_report = await api.lint_propose(base_root, rule="missing_file", limit=10)
    apply_report = await api.lint_apply(base_root, proposal_report=proposal_report)

    assert len(apply_report.applied) == 1
    assert apply_report.applied[0].kind == "purge_document"
    assert apply_report.skipped == []
    assert apply_report.purged_documents == [path]
    # The purge did not masquerade as a content edit.
    assert path not in apply_report.knowledge_paths_changed

    _cfg, _root, storage = await api._with_storage(base_root)
    try:
        assert await storage.get_document(doc_id) is None
    finally:
        await storage.close()

    post = await _run_lint(base_root)
    assert not any(i.kind == "missing_file" for i in post.issues)


@pytest.mark.asyncio
async def test_missing_file_apply_leaves_inbound_links_as_broken_wikilink(
    base_root: Path,
) -> None:
    """D5 governance: a vanished K-page's referrer is never rewritten. A live
    page A links ``[[B]]``; once B's file is gone (disk is authoritative,
    ADR-0005), B is treated as gone, so A's link surfaces as
    ``broken_wikilink`` immediately — and stays broken after B's orphan row is
    purged. The purge never touches A's body."""
    a_path = "knowledge/concepts/a.md"
    b_path = "knowledge/concepts/b.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=a_path,
        body="# A\n\nSee [[B]].\n", title="A",
    )
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=b_path, body="# B\n\nbody\n", title="B"
    )
    a_doc_id = doc_id_for(Layer.KNOWLEDGE, a_path)
    a_bytes_before = (base_root / a_path).read_bytes()
    # Record the resolved A→B edge so it's a real stored link before purge.
    _cfg, _root, storage = await api._with_storage(base_root)
    try:
        await storage.replace_links_from(
            a_doc_id,
            [
                LinkRecord(
                    src_doc_id=a_doc_id, dst_path=b_path,
                    link_type=LinkType.WIKILINK, line=3,
                )
            ],
        )
    finally:
        await storage.close()

    # B's file vanishes; the row lingers.
    (base_root / b_path).unlink()
    pre = await _run_lint(base_root)
    assert any(i.kind == "missing_file" and i.path == b_path for i in pre.issues)
    # B's file is gone → B is treated as gone → A's [[B]] is already broken.
    assert any(i.kind == "broken_wikilink" and i.path == a_path for i in pre.issues)

    proposal_report = await api.lint_propose(base_root, rule="missing_file", limit=10)
    await api.lint_apply(base_root, proposal_report=proposal_report)

    post = await _run_lint(base_root)
    assert not any(i.kind == "missing_file" for i in post.issues)
    assert any(
        i.kind == "broken_wikilink" and i.path == a_path for i in post.issues
    ), "A's [[B]] stays broken_wikilink after B's row is purged"
    # The purge never rewrote A's body (D5: expose, never silently edit).
    assert (base_root / a_path).read_bytes() == a_bytes_before


@pytest.mark.asyncio
async def test_missing_file_is_the_only_kind_for_a_vanished_page(
    base_root: Path,
) -> None:
    """A vanished K-page surfaces ONLY as ``missing_file`` — never *also* as
    ``orphan_page`` / ``uncategorized`` / ``duplicate_title``. Disk is
    authoritative (ADR-0005), so a gone page is excluded from every other
    pass; those remediations ("re-file", "merge/link/leaf", "rename") all
    contradict the single actionable signal: purge the orphaned row.

    Set up every co-emission trap at once: the page sits in the fallback
    category (uncategorized bait), has no inbound links (orphan bait), and
    shares its title with a *live* page (duplicate_title bait — which must
    not misdirect the live page at a now-deleted path either)."""
    gone = "knowledge/未分类/dup.md"
    live = "knowledge/concepts/dup.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=gone, body="# Dup\n", title="Dup"
    )
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=live, body="# Dup\n\nlive\n", title="Dup"
    )
    (base_root / gone).unlink()

    report = await _run_lint(base_root)
    gone_kinds = sorted({i.kind for i in report.issues if i.path == gone})
    assert gone_kinds == ["missing_file"], (
        f"a vanished page must surface only as missing_file; got {gone_kinds}"
    )
    # The live twin is not dragged into a duplicate_title pointing at the
    # deleted path (its title is unique among live pages now).
    assert not any(
        i.kind == "duplicate_title" and i.path == live for i in report.issues
    )


@pytest.mark.asyncio
async def test_missing_file_apply_skips_when_file_reappears(base_root: Path) -> None:
    """Concurrent-restore safety: if the user restores the file between
    propose and apply, the purge is skipped (the row is valid again) and
    the row survives. Mirrors the ``expected_hash`` two-stage guard."""
    path = "knowledge/concepts/restored.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=path, body="# R\n", title="R"
    )
    (base_root / path).unlink()
    doc_id = doc_id_for(Layer.KNOWLEDGE, path)

    proposal_report = await api.lint_propose(base_root, rule="missing_file", limit=10)
    assert len(proposal_report.proposals) == 1

    # File comes back before apply.
    (base_root / path).write_text("# R\n\nrestored\n", encoding="utf-8")

    apply_report = await api.lint_apply(base_root, proposal_report=proposal_report)
    assert apply_report.applied == []
    assert len(apply_report.skipped) == 1
    assert "reappeared" in apply_report.skipped[0]["reason"]

    _cfg, _root, storage = await api._with_storage(base_root)
    try:
        assert await storage.get_document(doc_id) is not None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_missing_file_apply_cross_layer(base_root: Path) -> None:
    """A D source and a K page both missing → both purged in one apply pass.
    Confirms the op is not confined to the ``knowledge/`` sandbox the
    page-mutating ops use."""
    src = "sources/dead.md"
    kp = "knowledge/concepts/dead.md"
    await seed_doc(base_root, layer=Layer.SOURCE, path=src, body="# S\n", title="S")
    await seed_doc(base_root, layer=Layer.KNOWLEDGE, path=kp, body="# K\n", title="K")
    (base_root / src).unlink()
    (base_root / kp).unlink()

    proposal_report = await api.lint_propose(base_root, rule="missing_file", limit=10)
    assert len(proposal_report.proposals) == 2
    apply_report = await api.lint_apply(base_root, proposal_report=proposal_report)

    assert sorted(apply_report.purged_documents) == sorted([kp, src])
    _cfg, _root, storage = await api._with_storage(base_root)
    try:
        assert await storage.get_document(doc_id_for(Layer.SOURCE, src)) is None
        assert await storage.get_document(doc_id_for(Layer.KNOWLEDGE, kp)) is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_missing_file_apply_skips_when_row_already_purged(
    base_root: Path,
) -> None:
    """If the row is purged between propose and apply (a parallel apply, or
    a prior pass), the second apply skips honestly rather than reporting a
    no-op ``delete_document`` as a successful purge."""
    path = "sources/dead.md"
    await seed_doc(base_root, layer=Layer.SOURCE, path=path, body="# S\n", title="S")
    (base_root / path).unlink()
    doc_id = doc_id_for(Layer.SOURCE, path)

    proposal_report = await api.lint_propose(base_root, rule="missing_file", limit=10)

    # Purge the row out from under the proposal.
    _cfg, _root, storage = await api._with_storage(base_root)
    try:
        await storage.delete_document(doc_id)
    finally:
        await storage.close()

    apply_report = await api.lint_apply(base_root, proposal_report=proposal_report)
    assert apply_report.applied == []
    assert apply_report.purged_documents == []
    assert len(apply_report.skipped) == 1
    assert "no document row" in apply_report.skipped[0]["reason"]


# ---- Fixer unit ----------------------------------------------------------


@pytest.mark.asyncio
async def test_fixer_returns_none_when_row_absent(base_root: Path) -> None:
    """Direct unit on ``MissingFileFixer.propose``: an issue whose row no
    longer exists in storage (purged since scan) → ``None`` so the
    orchestrator records a skip instead of proposing a phantom purge."""
    from dikw_core.domains.knowledge.lint_fixers import MissingFileFixer

    _cfg, _root, storage = await api._with_storage(base_root)
    try:
        ctx = FixerContext(
            storage=storage, llm=None, embedding=None,
            base_root=base_root, all_pages=[], enable_llm=False,
        )
        issue = LintIssue(
            kind="missing_file", path="sources/ghost.md", detail="phantom"
        )

        class _Noop:
            pass

        proposal = await MissingFileFixer().propose(issue, ctx, _Noop())
        assert proposal is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_fixer_returns_none_when_file_present(base_root: Path) -> None:
    """If the file is back on disk by propose time, the fixer bails before
    touching storage — the row is valid again, nothing to purge."""
    from dikw_core.domains.knowledge.lint_fixers import MissingFileFixer

    path = "sources/present.md"
    await seed_doc(base_root, layer=Layer.SOURCE, path=path, body="# P\n", title="P")
    # File is present; storage=None proves the file check short-circuits first.
    ctx = FixerContext(
        storage=None, llm=None, embedding=None,
        base_root=base_root, all_pages=[], enable_llm=False,
    )
    issue = LintIssue(kind="missing_file", path=path, detail="present")

    class _Noop:
        pass

    proposal = await MissingFileFixer().propose(issue, ctx, _Noop())
    assert proposal is None
