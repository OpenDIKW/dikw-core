"""Tests for the ``untracked_file`` drift lint kind + reindex fixer (ADR-0005 PR3).

``untracked_file`` closes the "hand-write a K/W markdown file, the engine
never indexes it" gap and unlocks hand-authored knowledge pages as
first-class citizens. A ``.md`` / ``.markdown`` file on disk under
``knowledge/`` or ``wisdom/`` with **no active ``documents`` row** is
untracked: disk is the source of truth (ADR-0005), so the missing row is
the lagging side. The deterministic reindex fixer emits one
``reindex_page`` op that indexes the file through
``persist_knowledge`` / ``persist_wisdom`` (the same path ``stale_index``
re-projection uses) — title/category derived from the file, chunks +
links + provenance built, (inline-or-deferred) embed.

Spans K + W only; D-layer source discovery is owned by ``ingest``.
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


def _write(base: Path, rel: str, text: str) -> None:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# ---- Detection -----------------------------------------------------------


@pytest.mark.asyncio
async def test_untracked_detects_handwritten_knowledge_page(base_root: Path) -> None:
    """A hand-written ``.md`` under ``knowledge/`` with no row → one
    ``untracked_file`` issue."""
    path = "knowledge/concepts/handwritten.md"
    _write(base_root, path, "# Handwritten\n\nbody\n")

    report = await _run_lint(base_root)
    untracked = [i for i in report.issues if i.kind == "untracked_file"]
    assert [i.path for i in untracked] == [path]


@pytest.mark.asyncio
async def test_untracked_detects_handwritten_wisdom_page(base_root: Path) -> None:
    path = "wisdom/holo/thought.md"
    _write(base_root, path, "# Thought\n\nbody\n")

    report = await _run_lint(base_root)
    assert any(i.kind == "untracked_file" and i.path == path for i in report.issues)


@pytest.mark.asyncio
async def test_untracked_clean_when_all_tracked(base_root: Path) -> None:
    """A page with an active row is tracked → no ``untracked_file``."""
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path="knowledge/concepts/tracked.md",
        body="# Tracked\n\nbody\n", title="Tracked",
    )
    report = await _run_lint(base_root)
    assert not any(i.kind == "untracked_file" for i in report.issues)


@pytest.mark.asyncio
async def test_untracked_ignores_non_markdown_files(base_root: Path) -> None:
    """Only ``.md`` / ``.markdown`` are documents — a ``.gitkeep`` placeholder
    or a stray ``.txt`` under ``knowledge/`` is never flagged."""
    _write(base_root, "knowledge/concepts/.gitkeep", "")
    _write(base_root, "knowledge/concepts/notes.txt", "scratch\n")
    _write(base_root, "knowledge/concepts/image.png", "not really a png")

    report = await _run_lint(base_root)
    assert not any(i.kind == "untracked_file" for i in report.issues)


@pytest.mark.asyncio
async def test_untracked_ignores_trash_tree(base_root: Path) -> None:
    """A soft-deleted file under ``<base>/trash/`` is outside the K/W walk —
    the walk roots at ``knowledge/`` + ``wisdom/`` so ``trash/`` is excluded."""
    _write(base_root, "trash/knowledge/concepts/deleted.md", "# Deleted\n\nbody\n")
    report = await _run_lint(base_root)
    assert not any(i.kind == "untracked_file" for i in report.issues)


@pytest.mark.asyncio
async def test_untracked_ignores_inactive_only_when_active_row_exists(
    base_root: Path,
) -> None:
    """A file with an active row is tracked even if other rows exist; the
    detector keys on the *active* projection."""
    path = "knowledge/concepts/active.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=path, body="# A\n\nbody\n", title="A"
    )
    report = await _run_lint(base_root)
    assert not any(
        i.kind == "untracked_file" and i.path == path for i in report.issues
    )


# ---- Fixer + apply -------------------------------------------------------


@pytest.mark.asyncio
async def test_untracked_propose_emits_reindex_op(base_root: Path) -> None:
    path = "knowledge/concepts/new.md"
    _write(base_root, path, "# New\n\nbody\n")

    proposal_report = await api.lint_propose(base_root, rule="untracked_file", limit=10)
    assert len(proposal_report.proposals) == 1
    op = proposal_report.proposals[0].operations[0]
    assert op.kind == "reindex_page"
    assert op.path == path
    assert op.layer == Layer.KNOWLEDGE


@pytest.mark.asyncio
async def test_untracked_apply_indexes_knowledge_page(base_root: Path) -> None:
    """Apply indexes the hand-written page: an active row appears with the
    title parsed from the file, the page lands in ``reindexed_documents``, the
    file is byte-for-byte untouched, and the next lint is clean of
    ``untracked_file``."""
    path = "knowledge/concepts/handmade.md"
    text = (
        "---\ntitle: Hand Made\ncategory: concepts\n---\n\n"
        "# Hand Made\n\nfirst-class hand-authored page\n"
    )
    _write(base_root, path, text)
    file_bytes = (base_root / path).read_bytes()
    doc_id = doc_id_for(Layer.KNOWLEDGE, path)

    proposal_report = await api.lint_propose(base_root, rule="untracked_file", limit=10)
    apply_report = await api.lint_apply(base_root, proposal_report=proposal_report)

    assert apply_report.reindexed_documents == [path]
    assert apply_report.persist_errors == []
    assert (base_root / path).read_bytes() == file_bytes  # file untouched

    _cfg, _root, storage = await api._with_storage(base_root)
    try:
        doc = await storage.get_document(doc_id)
        assert doc is not None and doc.active
        assert doc.title == "Hand Made"
        assert doc.layer == Layer.KNOWLEDGE
    finally:
        await storage.close()

    post = await _run_lint(base_root)
    assert not any(i.kind == "untracked_file" for i in post.issues)


@pytest.mark.asyncio
async def test_untracked_apply_indexes_wisdom_page(base_root: Path) -> None:
    """A hand-written wisdom file indexes through ``persist_wisdom`` — the
    ``status`` frontmatter flows through (W-only field)."""
    path = "wisdom/holo/manifesto.md"
    text = (
        "---\ntitle: Manifesto\nstatus: published\n---\n\n"
        "# Manifesto\n\nhand-written wisdom\n"
    )
    _write(base_root, path, text)
    doc_id = doc_id_for(Layer.WISDOM, path)

    proposal_report = await api.lint_propose(base_root, rule="untracked_file", limit=10)
    op = proposal_report.proposals[0].operations[0]
    assert op.layer == Layer.WISDOM
    apply_report = await api.lint_apply(base_root, proposal_report=proposal_report)
    assert apply_report.reindexed_documents == [path]

    _cfg, _root, storage = await api._with_storage(base_root)
    try:
        doc = await storage.get_document(doc_id)
        assert doc is not None and doc.active and doc.layer == Layer.WISDOM
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_untracked_apply_resolves_wikilinks(base_root: Path) -> None:
    """Indexing a hand-written page rebuilds its outgoing links — a
    ``[[Target]]`` to an already-tracked page resolves to a stored edge."""
    target_path = "knowledge/concepts/target.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=target_path,
        body="# Target\n\nbody\n", title="Target",
    )
    new_path = "knowledge/concepts/referrer.md"
    _write(base_root, new_path, "# Referrer\n\nSee [[Target]].\n")
    new_doc_id = doc_id_for(Layer.KNOWLEDGE, new_path)

    proposal_report = await api.lint_propose(base_root, rule="untracked_file", limit=10)
    await api.lint_apply(base_root, proposal_report=proposal_report)

    _cfg, _root, storage = await api._with_storage(base_root)
    try:
        links = await storage.links_from(new_doc_id)
        assert [link.dst_path for link in links] == [target_path]
    finally:
        await storage.close()


# ---- Fixer unit ----------------------------------------------------------


@pytest.mark.asyncio
async def test_reindex_fixer_returns_none_for_non_kw_path(base_root: Path) -> None:
    """Direct unit: the reindex fixer can't place a path outside knowledge/ or
    wisdom/ (the detector never emits such an issue, but the fixer guards its
    own input) → ``None``."""
    from dikw_core.domains.knowledge.lint_fixers import ReindexPageFixer

    path = "sources/stray.md"
    _write(base_root, path, "# Stray\n\nbody\n")
    ctx = FixerContext(
        storage=None, llm=None, embedding=None,
        base_root=base_root, all_pages=[], enable_llm=False,
    )
    issue = LintIssue(kind="untracked_file", path=path, detail="stray")

    class _Noop:
        pass

    proposal = await ReindexPageFixer().propose(issue, ctx, _Noop())
    assert proposal is None
