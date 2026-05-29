"""``_persist_knowledge_page`` link-set reconciliation contract.

Editing a knowledge page to drop a ``[[wikilink]]`` must remove the
corresponding edge from storage. Without this, the ``links`` table
accumulates ghost edges as users edit pages — polluting the
graph-leg retrieval channel (``neighbor_chunks_via_links``) and
quietly breaking ``orphan_page`` / ``broken_wikilink`` lint
detection.

``test_links.py`` covers the parser; this file pins the
persistence-layer round-trip so the engine-side reconciliation
invariant can't regress silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.domains.data.path_norm import normalize_path
from dikw_core.domains.knowledge.page import build_page, write_page
from dikw_core.schemas import Layer, LinkType

from .fakes import init_test_base


async def _persist(storage, root: Path, page) -> int:
    """Thin wrapper — embedder/version pinned to None so the test
    doesn't drag the whole embed pipeline in. ``_persist_knowledge_page``
    skips embedding when either is None.

    Returns the count of broken wikilinks (non-resolving ``[[Target]]``
    references) so observability tests can pin it.
    """
    return await api._persist_knowledge_page(
        storage=storage,
        root=root,
        page=page,
        embedder=None,
        embedding_model="fake",
        text_version_id=None,
    )


@pytest.mark.asyncio
async def test_persist_knowledge_page_drops_removed_wikilinks(tmp_path: Path) -> None:
    """Rewriting a page to swap ``[[Target A]]`` for ``[[Target B]]``
    must leave only the B edge in storage. Pre-fix this assertion
    fails because the A edge was never deleted."""
    base_root = tmp_path / "knowledge"
    init_test_base(base_root)
    _cfg, root, storage = await api._with_storage(base_root)
    try:
        target_a = build_page(title="Target A", body="A body.\n", type_="concept")
        target_b = build_page(title="Target B", body="B body.\n", type_="concept")
        for p in (target_a, target_b):
            write_page(root, p)
            await _persist(storage, root, p)

        src_v1 = build_page(
            title="Src",
            body="See [[Target A]] for details.\n",
            type_="concept",
            path="knowledge/src.md",
        )
        write_page(root, src_v1)
        await _persist(storage, root, src_v1)

        src_doc_id = api._doc_id_for(Layer.KNOWLEDGE, "knowledge/src.md")
        wikilinks_v1 = [
            link for link in await storage.links_from(src_doc_id)
            if link.link_type == LinkType.WIKILINK
        ]
        assert {link.dst_path for link in wikilinks_v1} == {target_a.path}

        # Rewrite the same page so the [[Target A]] reference is gone.
        src_v2 = build_page(
            title="Src",
            body="See [[Target B]] instead.\n",
            type_="concept",
            path="knowledge/src.md",
        )
        write_page(root, src_v2)
        await _persist(storage, root, src_v2)

        wikilinks_v2 = [
            link for link in await storage.links_from(src_doc_id)
            if link.link_type == LinkType.WIKILINK
        ]
        # Without reconciliation, target_a still shows up here as a
        # ghost edge — that's the bug this test pins.
        assert {link.dst_path for link in wikilinks_v2} == {target_b.path}
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_persist_knowledge_page_drops_all_wikilinks_when_body_loses_them(
    tmp_path: Path,
) -> None:
    """Rewriting a page to remove every ``[[wikilink]]`` must leave
    zero outgoing wikilink edges. Catches the edge case where a
    ``DELETE … WHERE src_doc_id = ?`` works for "swap A for B" but
    a different implementation (e.g. "delete edges that no longer
    appear in the new body") might miss the empty-target set."""
    base_root = tmp_path / "knowledge"
    init_test_base(base_root)
    _cfg, root, storage = await api._with_storage(base_root)
    try:
        target = build_page(title="Target", body="t body.\n", type_="concept")
        write_page(root, target)
        await _persist(storage, root, target)

        src_v1 = build_page(
            title="Src",
            body="See [[Target]] for context.\n",
            type_="concept",
            path="knowledge/src.md",
        )
        write_page(root, src_v1)
        await _persist(storage, root, src_v1)

        src_doc_id = api._doc_id_for(Layer.KNOWLEDGE, "knowledge/src.md")
        wikilinks_v1 = [
            link for link in await storage.links_from(src_doc_id)
            if link.link_type == LinkType.WIKILINK
        ]
        assert len(wikilinks_v1) == 1

        src_v2 = build_page(
            title="Src",
            body="Plain prose with no links anymore.\n",
            type_="concept",
            path="knowledge/src.md",
        )
        write_page(root, src_v2)
        await _persist(storage, root, src_v2)

        wikilinks_v2 = [
            link for link in await storage.links_from(src_doc_id)
            if link.link_type == LinkType.WIKILINK
        ]
        assert wikilinks_v2 == []
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_persist_knowledge_page_returns_unresolved_wikilink_count(
    tmp_path: Path,
) -> None:
    """``_persist_knowledge_page`` returns the count of unresolved ``[[wikilinks]]``
    so the synth caller can fold it into ``SynthReport.unresolved_wikilinks``.

    Without this signal, broken wikilinks are only visible after a
    separate ``dikw client lint`` pass; surfacing the count at synth time gives
    users an immediate "did the LLM emit references that don't land
    anywhere?" health check.
    """
    base_root = tmp_path / "knowledge"
    init_test_base(base_root)
    _cfg, root, storage = await api._with_storage(base_root)
    try:
        page = build_page(
            title="Src",
            body="See [[Unknown One]] and [[Unknown Two]] and [[Unknown Three]].\n",
            type_="concept",
            path="knowledge/src.md",
        )
        write_page(root, page)
        unresolved = await _persist(storage, root, page)
        assert unresolved == 3
    finally:
        await storage.close()


def test_synth_report_carries_unresolved_wikilinks_field() -> None:
    """``SynthReport`` exposes ``unresolved_wikilinks`` so the synth caller
    can aggregate per-page counts and the CLI can surface the total."""
    report = api.SynthReport()
    assert report.unresolved_wikilinks == 0
    bumped = api._sr_replace(report, unresolved_wikilinks=7)
    assert bumped.unresolved_wikilinks == 7


# ---- Provenance reconcile -------------------------------------------------
#
# Same invariant as the wikilink reconcile above but for the K-page →
# D-source attribution edge: ``_persist_knowledge_page`` must reconcile the
# ``provenance`` table from the page's ``sources:`` frontmatter on every
# call. Without this, hand-editing ``sources:`` (or re-synth that drops a
# source) would leave ghost rows that ``api.read_provenance`` still
# surfaces — the same class of bug the wikilink tests above guard
# against, but for the page-source attribution graph.


@pytest.mark.asyncio
async def test_persist_knowledge_page_writes_provenance_from_frontmatter(
    tmp_path: Path,
) -> None:
    """A fresh page with ``sources: [A, B]`` lands two provenance rows
    keyed by the page's doc_id. The reverse-lookup index also sees them
    immediately — proves the call hit both the table and the index."""
    base_root = tmp_path / "knowledge"
    init_test_base(base_root)
    _cfg, root, storage = await api._with_storage(base_root)
    try:
        page = build_page(
            title="Src",
            body="Body.\n",
            type_="concept",
            path="knowledge/src.md",
            sources=["sources/foo.md", "sources/bar.md"],
        )
        write_page(root, page)
        await _persist(storage, root, page)

        doc_id = api._doc_id_for(Layer.KNOWLEDGE, "knowledge/src.md")
        rows = await storage.provenance_from(doc_id)
        assert {r.source_path for r in rows} == {
            "sources/foo.md",
            "sources/bar.md",
        }
        # Reverse lookup wired.
        reverse = await storage.provenance_to(
            normalize_path("sources/foo.md")
        )
        assert [r.src_doc_id for r in reverse] == [doc_id]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_persist_knowledge_page_removes_stale_provenance_when_frontmatter_changes(
    tmp_path: Path,
) -> None:
    """Editing ``sources:`` to drop an entry must remove the matching
    provenance row — symmetric to the wikilink reconcile above. Without
    the reconcile call, ``sources: [old]`` → ``sources: [new]`` leaves
    a ghost row pointing at ``old``."""
    base_root = tmp_path / "knowledge"
    init_test_base(base_root)
    _cfg, root, storage = await api._with_storage(base_root)
    try:
        page_v1 = build_page(
            title="Src",
            body="Body.\n",
            type_="concept",
            path="knowledge/src.md",
            sources=["sources/old.md"],
        )
        write_page(root, page_v1)
        await _persist(storage, root, page_v1)

        doc_id = api._doc_id_for(Layer.KNOWLEDGE, "knowledge/src.md")
        before = await storage.provenance_from(doc_id)
        assert {r.source_path for r in before} == {"sources/old.md"}

        # Rewrite the same page with a different sources list.
        page_v2 = build_page(
            title="Src",
            body="Body.\n",
            type_="concept",
            path="knowledge/src.md",
            sources=["sources/new.md"],
        )
        write_page(root, page_v2)
        await _persist(storage, root, page_v2)

        after = await storage.provenance_from(doc_id)
        assert {r.source_path for r in after} == {"sources/new.md"}
        # Reverse lookup on the dropped source returns no rows.
        assert (
            await storage.provenance_to(normalize_path("sources/old.md")) == []
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_persist_knowledge_page_with_no_sources_frontmatter_leaves_provenance_empty(
    tmp_path: Path,
) -> None:
    """A page whose frontmatter has no ``sources:`` key (or an empty
    list) gets zero provenance rows. The reconcile call is unconditional
    so the leading DELETE handles the "list previously non-empty,
    user cleared it" case as well."""
    base_root = tmp_path / "knowledge"
    init_test_base(base_root)
    _cfg, root, storage = await api._with_storage(base_root)
    try:
        page = build_page(
            title="Src",
            body="Body.\n",
            type_="concept",
            path="knowledge/src.md",
            # No sources= argument → page.sources == [] → write_page
            # omits the frontmatter key entirely (see wiki.py:140).
        )
        write_page(root, page)
        await _persist(storage, root, page)

        doc_id = api._doc_id_for(Layer.KNOWLEDGE, "knowledge/src.md")
        assert await storage.provenance_from(doc_id) == []
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_persist_knowledge_page_treats_scalar_sources_as_no_op(
    tmp_path: Path,
) -> None:
    """A hand-edited frontmatter with a YAML *scalar* ``sources:`` value
    (e.g. ``sources: sources/a.md`` instead of ``sources: [sources/a.md]``)
    must NOT iterate the string character-by-character. Without the
    isinstance(list) guard, ``_persist_knowledge_page`` would insert one
    provenance row per character — 14 ghost rows for ``sources/foo.md``.

    The fix mirrors ``run_lint``'s frontmatter-meta extraction
    (``isinstance(raw_sources, list)`` short-circuits before iteration)
    so both the persist path and the lint detector treat a malformed
    scalar as "no sources declared". A future feature could promote
    scalars to single-element lists, but today's contract is symmetric:
    invalid shape → empty.
    """
    base_root = tmp_path / "knowledge"
    init_test_base(base_root)
    # build_page always wraps in list(), so we write the malformed
    # frontmatter directly to simulate a user hand-edit. The body still
    # needs valid frontmatter form so ``parse_any`` returns it intact.
    page_path = base_root / "knowledge" / "src.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(
        "---\n"
        "id: src-id\n"
        "type: concept\n"
        "title: Src\n"
        "sources: sources/foo.md\n"  # scalar, not list
        "---\n\n"
        "Body.\n",
        encoding="utf-8",
    )

    # Build a KnowledgePage just for the persist call's required arg — the
    # function re-parses the on-disk file so ``page.sources`` doesn't
    # influence the provenance write.
    page = build_page(
        title="Src",
        body="Body.\n",
        type_="concept",
        path="knowledge/src.md",
    )

    _cfg, root, storage = await api._with_storage(base_root)
    try:
        await _persist(storage, root, page)
        doc_id = api._doc_id_for(Layer.KNOWLEDGE, "knowledge/src.md")
        rows = await storage.provenance_from(doc_id)
        # Pre-fix: 14 rows (one per char in "sources/foo.md"). Fix:
        # scalar treated as malformed → zero rows.
        assert rows == [], (
            f"scalar sources should produce no provenance rows; got "
            f"{len(rows)} row(s): {[r.source_path for r in rows]}"
        )
    finally:
        await storage.close()
