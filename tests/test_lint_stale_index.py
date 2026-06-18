"""Tests for the ``stale_index`` drift lint kind + reindex fixer (ADR-0005 PR3).

``stale_index`` closes the "hand-edit a K/W file on disk, the storage
projection silently drifts" gap. An *active* ``documents`` row whose
backing file's body hash disagrees with the stored ``hash`` is stale: the
file is the source of truth (ADR-0005), the row + its chunks / links /
provenance are the lagging projection. The deterministic reindex fixer
emits a single ``reindex_page`` op that re-projects the *current* disk
bytes through ``persist_knowledge`` / ``persist_wisdom`` — re-chunk,
re-link, re-provenance, (inline-or-deferred) re-embed — **never** rewriting
the user's file and **never** re-running synth.

Spans K + W only; D-layer adds/edits are owned by ``ingest`` (zero overlap).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.domains.data.path_norm import doc_id_for
from dikw_core.domains.knowledge.lint import LintIssue, run_lint
from dikw_core.domains.knowledge.lint_fix import FixerContext
from dikw_core.schemas import Layer

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
async def test_stale_index_detects_body_edit(base_root: Path) -> None:
    """A K page whose on-disk body was hand-edited after indexing → one
    ``stale_index`` issue. ``seed_doc`` stores the *original* body hash; the
    overwrite changes the body so the stored hash no longer matches disk."""
    path = "knowledge/concepts/topic.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=path,
        body="# Topic\n\noriginal body\n", title="Topic",
    )
    (base_root / path).write_text("# Topic\n\nhand-edited body\n", encoding="utf-8")

    report = await _run_lint(base_root)
    stale = [i for i in report.issues if i.kind == "stale_index"]
    assert [i.path for i in stale] == [path]


@pytest.mark.asyncio
async def test_stale_index_detects_wisdom_edit(base_root: Path) -> None:
    path = "wisdom/holo/essay.md"
    await seed_doc(
        base_root, layer=Layer.WISDOM, path=path,
        body="# Essay\n\nv1\n", title="Essay",
    )
    (base_root / path).write_text("# Essay\n\nv2 edited\n", encoding="utf-8")

    report = await _run_lint(base_root)
    assert any(i.kind == "stale_index" and i.path == path for i in report.issues)


@pytest.mark.asyncio
async def test_stale_index_clean_when_unchanged(base_root: Path) -> None:
    """A consistent row (``seed_doc`` stores the real disk hash) → zero
    ``stale_index``. Pins that a freshly-indexed page never false-fires."""
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path="knowledge/concepts/a.md",
        body="# A\n\nbody\n", title="A",
    )
    await seed_doc(
        base_root, layer=Layer.WISDOM, path="wisdom/holo/b.md",
        body="# B\n\nbody\n", title="B",
    )

    report = await _run_lint(base_root)
    assert not any(i.kind == "stale_index" for i in report.issues)


@pytest.mark.asyncio
async def test_stale_index_not_fired_on_mtime_only_touch(base_root: Path) -> None:
    """The hash, not mtime, is authoritative: re-writing the *same* bytes
    (a touch with no content change) must NOT flag ``stale_index``. Guards
    against a regression that compares mtime instead of body hash."""
    import os

    path = "knowledge/concepts/touched.md"
    body = "# Touched\n\nstable body\n"
    await seed_doc(base_root, layer=Layer.KNOWLEDGE, path=path, body=body, title="Touched")
    # Rewrite identical bytes and bump mtime forward — content hash unchanged.
    abs_path = base_root / path
    abs_path.write_text(body, encoding="utf-8")
    st = abs_path.stat()
    os.utime(abs_path, (st.st_mtime + 1000, st.st_mtime + 1000))

    report = await _run_lint(base_root)
    assert not any(i.kind == "stale_index" for i in report.issues)


@pytest.mark.asyncio
async def test_stale_index_suppressed_by_frontmatter(base_root: Path) -> None:
    """``lint: {skip: [stale_index]}`` in a page's frontmatter suppresses the
    kind even when the body drifts (the file is present, so it can carry the
    annotation — unlike ``missing_file``)."""
    path = "knowledge/concepts/draft.md"
    # Seed with the skip block already in frontmatter, then drift the body.
    original = (
        "---\nid: K-draft\ncategory: concepts\ntitle: Draft\n"
        "lint:\n  skip:\n    - stale_index\n---\n\n# Draft\n\nv1\n"
    )
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=path, body=original, title="Draft"
    )
    edited = (
        "---\nid: K-draft\ncategory: concepts\ntitle: Draft\n"
        "lint:\n  skip:\n    - stale_index\n---\n\n# Draft\n\nv2 drifted\n"
    )
    (base_root / path).write_text(edited, encoding="utf-8")

    report = await _run_lint(base_root)
    assert not any(
        i.kind == "stale_index" and i.path == path for i in report.issues
    )
    # The page is acknowledged as an opt-out leaf.
    assert path in report.acknowledged_leaves


# ---- Fixer + apply -------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_index_propose_emits_reindex_op(base_root: Path) -> None:
    """``lint propose --rule stale_index`` emits one ``reindex_page`` op
    carrying the resolved layer; deterministic (``source == "heuristic"``)."""
    path = "knowledge/concepts/edited.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=path, body="# E\n\nv1\n", title="E"
    )
    (base_root / path).write_text("# E\n\nv2\n", encoding="utf-8")

    proposal_report = await api.lint_propose(base_root, rule="stale_index", limit=10)
    assert len(proposal_report.proposals) == 1
    proposal = proposal_report.proposals[0]
    assert proposal.source == "heuristic"
    assert proposal.issue_path == path
    op = proposal.operations[0]
    assert op.kind == "reindex_page"
    assert op.path == path
    assert op.layer == Layer.KNOWLEDGE


@pytest.mark.asyncio
async def test_stale_index_apply_reprojects_and_preserves_handedits(
    base_root: Path,
) -> None:
    """Apply re-projects the *current* disk bytes: the stored hash catches up
    (next lint clean), the file is byte-for-byte untouched (hand-edit
    preserved, synth never re-run), and the page lands in
    ``reindexed_documents``. Crucially the re-projection picks up a wikilink
    the hand-edit ADDED — proving chunks/links are rebuilt, not just the hash
    bumped."""
    a_path = "knowledge/concepts/a.md"
    target_path = "knowledge/concepts/target.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=a_path,
        body="# A\n\nno links yet\n", title="A",
    )
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=target_path,
        body="# Target\n\nbody\n", title="Target",
    )
    a_doc_id = doc_id_for(Layer.KNOWLEDGE, a_path)

    # Hand-edit A to add a [[Target]] wikilink — body drift + a new edge that
    # storage doesn't know about yet.
    edited = "# A\n\nNow see [[Target]].\n"
    (base_root / a_path).write_text(edited, encoding="utf-8")
    edited_bytes = (base_root / a_path).read_bytes()

    pre = await _run_lint(base_root)
    assert any(i.kind == "stale_index" and i.path == a_path for i in pre.issues)

    proposal_report = await api.lint_propose(base_root, rule="stale_index", limit=10)
    apply_report = await api.lint_apply(base_root, proposal_report=proposal_report)

    assert apply_report.reindexed_documents == [a_path]
    assert apply_report.persist_errors == []
    # The file was never rewritten by the fix (D5: disk is authoritative).
    assert (base_root / a_path).read_bytes() == edited_bytes

    _cfg, _root, storage = await api._with_storage(base_root)
    try:
        # The re-projection rebuilt A's outgoing links from the edited body.
        a_links = await storage.links_from(a_doc_id)
        assert [link.dst_path for link in a_links] == [target_path], (
            f"reindex must rebuild A→Target edge from the edited body; got {a_links}"
        )
    finally:
        await storage.close()

    post = await _run_lint(base_root)
    assert not any(
        i.kind == "stale_index" and i.path == a_path for i in post.issues
    ), "the stored hash must catch up to disk after reindex"


@pytest.mark.asyncio
async def test_stale_index_apply_reprojects_wisdom(base_root: Path) -> None:
    """W-layer drift re-projects through ``persist_wisdom`` — confirms reindex
    is not confined to the K layer, AND that it actually re-links (not just
    bumps the hash): the edited body adds a cross-layer ``[[wikilink]]`` to a
    tracked K page, which must become a stored W→K edge after reindex."""
    target = "knowledge/concepts/target.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=target,
        body="# Target\n\nbody\n", title="Target",
    )
    path = "wisdom/holo/note.md"
    await seed_doc(
        base_root, layer=Layer.WISDOM, path=path, body="# Note\n\nv1\n", title="Note"
    )
    (base_root / path).write_text(
        "# Note\n\nv2 revised, see [[Target]].\n", encoding="utf-8"
    )
    note_doc_id = doc_id_for(Layer.WISDOM, path)

    proposal_report = await api.lint_propose(base_root, rule="stale_index", limit=10)
    assert proposal_report.proposals[0].operations[0].layer == Layer.WISDOM
    apply_report = await api.lint_apply(base_root, proposal_report=proposal_report)
    assert apply_report.reindexed_documents == [path]

    _cfg, _root, storage = await api._with_storage(base_root)
    try:
        links = await storage.links_from(note_doc_id)
        assert [link.dst_path for link in links] == [target], (
            f"W reindex must rebuild the cross-layer W→K edge; got {links}"
        )
    finally:
        await storage.close()

    post = await _run_lint(base_root)
    assert not any(i.kind == "stale_index" and i.path == path for i in post.issues)


@pytest.mark.asyncio
async def test_stale_index_apply_deactivates_on_persist_failure(
    base_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hard persist failure during reindex deactivates the page
    (``documents.active`` is the commit marker), records it under
    ``persist_errors``, and excludes it from ``reindexed_documents`` — parity
    with the synth / create-update failure path."""
    from dikw_core.domains.knowledge import lint_fix

    path = "knowledge/concepts/boom.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=path, body="# Boom\n\nv1\n", title="Boom"
    )
    (base_root / path).write_text("# Boom\n\nv2 edited\n", encoding="utf-8")
    doc_id = doc_id_for(Layer.KNOWLEDGE, path)

    async def _boom(**_kwargs: object) -> object:
        raise RuntimeError("simulated persist failure")

    monkeypatch.setattr(lint_fix, "persist_knowledge", _boom)

    proposal_report = await api.lint_propose(base_root, rule="stale_index", limit=10)
    apply_report = await api.lint_apply(base_root, proposal_report=proposal_report)

    assert apply_report.reindexed_documents == []
    assert any(e["path"] == path for e in apply_report.persist_errors)

    _cfg, _root, storage = await api._with_storage(base_root)
    try:
        doc = await storage.get_document(doc_id)
        assert doc is not None and doc.active is False, (
            "a reindex that failed mid-persist must be deactivated"
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_stale_index_apply_skips_when_file_vanishes(base_root: Path) -> None:
    """If the drifted file is deleted between propose and apply, the reindex
    is skipped (there is nothing to re-project) and the row is left for the
    next lint to surface as ``missing_file``."""
    path = "knowledge/concepts/vanishing.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=path, body="# V\n\nv1\n", title="V"
    )
    (base_root / path).write_text("# V\n\nv2\n", encoding="utf-8")

    proposal_report = await api.lint_propose(base_root, rule="stale_index", limit=10)
    assert len(proposal_report.proposals) == 1

    (base_root / path).unlink()

    apply_report = await api.lint_apply(base_root, proposal_report=proposal_report)
    assert apply_report.reindexed_documents == []
    assert len(apply_report.skipped) == 1
    assert "absent" in apply_report.skipped[0]["reason"].lower() or (
        "not found" in apply_report.skipped[0]["reason"].lower()
    )


# ---- Fixer unit ----------------------------------------------------------


@pytest.mark.asyncio
async def test_reindex_fixer_returns_none_when_file_absent(base_root: Path) -> None:
    """Direct unit: an issue whose file no longer exists → ``None`` so the
    orchestrator records a skip instead of proposing a reindex of nothing."""
    from dikw_core.domains.knowledge.lint_fixers import ReindexPageFixer

    ctx = FixerContext(
        storage=None, llm=None, embedding=None,
        base_root=base_root, all_pages=[], enable_llm=False,
    )
    issue = LintIssue(
        kind="stale_index", path="knowledge/concepts/ghost.md", detail="gone"
    )

    class _Noop:
        pass

    proposal = await ReindexPageFixer().propose(issue, ctx, _Noop())
    assert proposal is None
