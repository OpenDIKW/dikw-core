"""Engine-side unit tests for ``api.list_graph``.

The HTTP-layer surface lives in ``tests/server/test_routes_graph.py``;
this file exercises the pure helper that produces a ``GraphResult`` from
``(root,)`` so the seam (active filter, weight aggregation, distinct
inbound/outbound counts, unresolved no-ghost-nodes, deterministic
ordering) stays guarded without booting a server.

Seed pattern is deliberately direct: write markdown to disk + upsert
``DocumentRecord`` rows, no full ingest. ``list_graph`` re-parses bodies
on every call so it doesn't need chunks / embeddings; this keeps tests
fast and the failure modes localized to the graph code path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dikw_core import api
from dikw_core.schemas import Layer

from .fakes import init_test_base
from .fakes import seed_doc as _seed_doc


def _node_by_path(graph: Any, path: str) -> Any:
    for n in graph.nodes:
        if n.path == path:
            return n
    raise AssertionError(f"node {path!r} not in graph (have {[n.path for n in graph.nodes]})")


@pytest.mark.asyncio
async def test_empty_base_returns_empty_graph(tmp_path: Path) -> None:
    init_test_base(tmp_path)
    g = await api.list_graph(tmp_path)
    assert g.nodes == []
    assert g.edges == []
    assert g.unresolved == []
    assert g.stats.node_count == 0
    assert g.stats.edge_count == 0
    assert g.stats.unresolved_count == 0


@pytest.mark.asyncio
async def test_three_pages_one_edge_each(tmp_path: Path) -> None:
    """A→B, A→C, B→C → 3 nodes + 3 edges."""
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/A.md", title="A",
        body="# A\n\nLinks to [[B]] and [[C]].\n",
    )
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/B.md", title="B",
        body="# B\n\nLinks to [[C]].\n",
    )
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/C.md", title="C",
        body="# C\n\nLeaf.\n",
    )

    g = await api.list_graph(tmp_path)
    paths = {n.path for n in g.nodes}
    assert paths == {"knowledge/A.md", "knowledge/B.md", "knowledge/C.md"}
    assert g.stats.node_count == 3

    edges = {(e.source, e.target) for e in g.edges}
    assert edges == {
        ("knowledge/A.md", "knowledge/B.md"),
        ("knowledge/A.md", "knowledge/C.md"),
        ("knowledge/B.md", "knowledge/C.md"),
    }
    assert g.stats.edge_count == 3
    assert g.stats.unresolved_count == 0


@pytest.mark.asyncio
async def test_repeated_link_aggregates_weight(tmp_path: Path) -> None:
    """A → B twice → one edge with weight=2."""
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/A.md", title="A",
        body="See [[B]] and again [[B]].\n",
    )
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/B.md", title="B", body="# B\n",
    )

    g = await api.list_graph(tmp_path)
    a_to_b = [e for e in g.edges if e.source == "knowledge/A.md" and e.target == "knowledge/B.md"]
    assert len(a_to_b) == 1
    assert a_to_b[0].weight == 2
    # `id` is deterministic on (source, target).
    assert a_to_b[0].id == "knowledge/A.md->knowledge/B.md"


@pytest.mark.asyncio
async def test_inbound_outbound_distinct_pages(tmp_path: Path) -> None:
    """A→B twice + A→C once → A.outbound = 2 (distinct B/C), B.inbound = 1.

    Issue #89: counts are over distinct *connected pages*, not raw link
    occurrences.
    """
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/A.md", title="A",
        body="[[B]] [[B]] [[C]]\n",
    )
    await _seed_doc(tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/B.md", title="B", body="# B\n")
    await _seed_doc(tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/C.md", title="C", body="# C\n")

    g = await api.list_graph(tmp_path)
    a = _node_by_path(g, "knowledge/A.md")
    b = _node_by_path(g, "knowledge/B.md")
    c = _node_by_path(g, "knowledge/C.md")
    assert a.outbound == 2  # distinct B, C — NOT 3
    assert a.inbound == 0
    assert b.inbound == 1
    assert c.inbound == 1


@pytest.mark.asyncio
async def test_unresolved_wikilinks_no_ghost_nodes(tmp_path: Path) -> None:
    """``[[Unknown Page]]`` → entry in ``unresolved``, no ghost node.

    Issue #89 v1: "should not create ghost nodes in the first version."
    """
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/A.md", title="A",
        body="[[Unknown Page]] and [[Also Missing]]\n",
    )

    g = await api.list_graph(tmp_path)
    assert {n.path for n in g.nodes} == {"knowledge/A.md"}
    assert g.edges == []
    assert g.stats.unresolved_count == 2
    assert {(u.source, u.target_text) for u in g.unresolved} == {
        ("knowledge/A.md", "Unknown Page"),
        ("knowledge/A.md", "Also Missing"),
    }


@pytest.mark.asyncio
async def test_active_filter_default_excludes_deactivated(tmp_path: Path) -> None:
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/A.md", title="A",
        body="[[B]]\n", active=True,
    )
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/B.md", title="B",
        body="# B\n", active=False,
    )

    g = await api.list_graph(tmp_path)  # active=True default
    assert {n.path for n in g.nodes} == {"knowledge/A.md"}
    # Edge to B is filtered because B is not in node set.
    assert g.edges == []
    # And [[B]] surfaces as unresolved-from-A's-POV (resolution index
    # is built from the active subset only — a deactivated target is
    # invisible, so its title doesn't resolve).
    assert g.stats.unresolved_count == 1


@pytest.mark.asyncio
async def test_active_none_returns_all_docs(tmp_path: Path) -> None:
    """``active=None`` matches the ``list_pages`` convention: no
    active-flag filter, both active and deactivated docs land in the
    node set, and edges between them resolve normally."""
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/A.md", title="A",
        body="[[B]]\n", active=True,
    )
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/B.md", title="B",
        body="# B\n", active=False,
    )

    g = await api.list_graph(tmp_path, active=None)
    assert {n.path for n in g.nodes} == {"knowledge/A.md", "knowledge/B.md"}
    assert {(e.source, e.target) for e in g.edges} == {("knowledge/A.md", "knowledge/B.md")}
    # Deactivated node retains its ``active=False`` flag in the wire
    # payload so a renderer can grey it out.
    assert _node_by_path(g, "knowledge/B.md").active is False
    assert _node_by_path(g, "knowledge/A.md").active is True


@pytest.mark.asyncio
async def test_active_false_returns_only_deactivated(tmp_path: Path) -> None:
    """Symmetric with ``list_pages(active=False)``: returns only the
    deactivated subset. Rare use case, but the convention has to stay
    consistent so a client that already speaks the ``/v1/base/pages``
    contract doesn't need a graph-specific exception."""
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/A.md", title="A",
        body="# A\n", active=True,
    )
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/B.md", title="B",
        body="# B\n", active=False,
    )

    g = await api.list_graph(tmp_path, active=False)
    assert {n.path for n in g.nodes} == {"knowledge/B.md"}


@pytest.mark.asyncio
async def test_cross_layer_edges_included(tmp_path: Path) -> None:
    """A knowledge page's ``[[Source Doc Title]]`` resolves to a source-layer
    page → edge with type=wikilink, both layers in the node set."""
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/note.md", title="Note",
        body="See [[Source Doc Title]] for the raw input.\n",
    )
    await _seed_doc(
        tmp_path, layer=Layer.SOURCE, path="sources/raw.md",
        title="Source Doc Title", body="# Raw\n",
    )

    g = await api.list_graph(tmp_path)
    assert {n.path for n in g.nodes} == {"knowledge/note.md", "sources/raw.md"}
    assert {(e.source, e.target) for e in g.edges} == {
        ("knowledge/note.md", "sources/raw.md")
    }
    edge = next(iter(g.edges))
    assert edge.type == "wikilink"
    # Layers are exposed on nodes so a client renderer can color by layer.
    assert _node_by_path(g, "knowledge/note.md").layer == "knowledge"
    assert _node_by_path(g, "sources/raw.md").layer == "source"


@pytest.mark.asyncio
async def test_markdown_link_edge(tmp_path: Path) -> None:
    """A markdown ``[X](other.md)`` link whose href matches a node path
    counts as an edge with ``type=markdown``."""
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.SOURCE, path="sources/a.md", title="A",
        body="See [the other](sources/b.md) for details.\n",
    )
    await _seed_doc(
        tmp_path, layer=Layer.SOURCE, path="sources/b.md", title="B",
        body="# B\n",
    )

    g = await api.list_graph(tmp_path)
    edges = list(g.edges)
    assert len(edges) == 1
    assert edges[0].source == "sources/a.md"
    assert edges[0].target == "sources/b.md"
    assert edges[0].type == "markdown"


@pytest.mark.asyncio
async def test_url_link_not_in_edges_or_unresolved(tmp_path: Path) -> None:
    """``[Anthropic](https://anthropic.com)`` and bare URLs neither
    create edges nor count as unresolved (URLs are out-of-graph)."""
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/A.md", title="A",
        body="See [Anthropic](https://anthropic.com) and https://example.com.\n",
    )

    g = await api.list_graph(tmp_path)
    assert g.edges == []
    assert g.unresolved == []
    assert g.stats.unresolved_count == 0


@pytest.mark.asyncio
async def test_deterministic_ordering(tmp_path: Path) -> None:
    """Two consecutive calls return byte-identical node + edge + unresolved
    sequences. The only field allowed to differ is ``generated_at``."""
    init_test_base(tmp_path)
    # Seed in scrambled creation order to make sure sort is path-driven,
    # not insertion-driven.
    for path, title, body in (
        ("knowledge/zeta.md", "Zeta", "[[Alpha]]\n"),
        ("knowledge/alpha.md", "Alpha", "[[Beta]] [[Gamma]]\n"),
        ("knowledge/beta.md", "Beta", "[[Alpha]]\n"),
        ("knowledge/gamma.md", "Gamma", "[[Missing]]\n"),
    ):
        await _seed_doc(tmp_path, layer=Layer.KNOWLEDGE, path=path, title=title, body=body)

    g1 = await api.list_graph(tmp_path)
    g2 = await api.list_graph(tmp_path)

    assert [n.path for n in g1.nodes] == [n.path for n in g2.nodes]
    assert [(e.source, e.target, e.target_text) for e in g1.edges] == [
        (e.source, e.target, e.target_text) for e in g2.edges
    ]
    assert [(u.source, u.target_text) for u in g1.unresolved] == [
        (u.source, u.target_text) for u in g2.unresolved
    ]
    # base_revision is content-addressed, so it must match between
    # back-to-back calls on an unchanged base.
    assert g1.base_revision == g2.base_revision


@pytest.mark.asyncio
async def test_anchor_preserved_on_edge(tmp_path: Path) -> None:
    """``[[B#section]]`` → edge with anchor='section'."""
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/A.md", title="A",
        body="See [[B#section]].\n",
    )
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/B.md", title="B",
        body="# B\n## section\n",
    )

    g = await api.list_graph(tmp_path)
    edge = next(e for e in g.edges if e.source == "knowledge/A.md")
    assert edge.target == "knowledge/B.md"
    assert edge.anchor == "section"


@pytest.mark.asyncio
async def test_duplicate_titles_collision_refuse(tmp_path: Path) -> None:
    """Source-layer ``Foo.md`` titled ``Foo`` + synth-layer knowledge page
    titled ``Foo`` create an ambiguous resolution target. Per
    Karpathy's rule (wrong-merge irreversible, missed-resolve
    fixable), ``[[Foo]]`` from a third page must NOT silently bind to
    one of them — it must surface as unresolved so a user / lint can
    triage.

    This is more frequent in graph mode than in K-layer synth because
    graph widens the resolution universe to include source-layer
    titles (often = filename stem), which routinely collide with the
    knowledge page synthesized from that very source.
    """
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.SOURCE, path="sources/foo.md", title="Foo",
        body="# Foo source\n",
    )
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/Foo.md", title="Foo",
        body="# Foo wiki\n",
    )
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/notes/refers.md",
        title="Refers", body="See [[Foo]] for context.\n",
    )

    g = await api.list_graph(tmp_path)
    # The [[Foo]] reference must NOT create an edge (would be wrong
    # half the time).
    assert all(e.source != "knowledge/notes/refers.md" for e in g.edges), (
        f"expected no edge from refers.md (ambiguous Foo); got {g.edges}"
    )
    # And it MUST be reported as unresolved so a user can triage.
    assert any(
        u.source == "knowledge/notes/refers.md" and u.target_text == "Foo"
        for u in g.unresolved
    ), f"expected ambiguous [[Foo]] in unresolved; got {g.unresolved}"


@pytest.mark.asyncio
async def test_base_revision_changes_when_body_edited(tmp_path: Path) -> None:
    """If a user edits a markdown file on disk without rerunning
    ingest, ``list_graph`` reparses the new body (so edges may
    change) — ``base_revision`` MUST also change so a client cache
    keyed on it doesn't serve a stale graph. Storing only
    ``DocumentRecord.mtime`` (set at ingest time) misses on-disk
    edits; the digest must observe current body content."""
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/A.md", title="A",
        body="[[B]]\n",
    )
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/B.md", title="B",
        body="# B\n",
    )

    rev_before = (await api.list_graph(tmp_path)).base_revision

    # Edit on disk, no ingest. Stored mtime stays at 0.0 (the seed value).
    (tmp_path / "knowledge/A.md").write_text(
        "[[B]] [[B]]\n",  # added a duplicate link → graph weight changes
        encoding="utf-8",
    )

    rev_after = (await api.list_graph(tmp_path)).base_revision
    assert rev_after != rev_before, (
        "base_revision must observe current on-disk body; "
        "stored mtime alone misses post-ingest edits"
    )


@pytest.mark.asyncio
async def test_base_revision_changes_when_title_metadata_changes(
    tmp_path: Path,
) -> None:
    """``GraphNode.title`` is part of the response and ``title_to_paths``
    drives wikilink resolution — a title-only change (e.g. user edits
    front-matter, reingest persists the new title without changing
    the body bytes) shifts node labels and edge resolution. A digest
    that observed only body sha256 would let a client cache serve
    stale node labels. The digest must observe title too."""
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/A.md", title="Alpha",
        body="# A\n",
    )

    rev_before = (await api.list_graph(tmp_path)).base_revision

    # Re-seed same path + same body, different title — what a re-ingest
    # of an edited front-matter does.
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/A.md", title="Alpha Renamed",
        body="# A\n",
    )

    rev_after = (await api.list_graph(tmp_path)).base_revision
    assert rev_after != rev_before, (
        "base_revision must observe title metadata; otherwise a "
        "client cache misses node-label changes"
    )


@pytest.mark.asyncio
async def test_path_escape_doc_is_filtered(tmp_path: Path) -> None:
    """If the documents table holds a path that resolves outside the
    base (corrupted DB / migration drift / hostile import), the graph
    reader must NOT touch the file — defence-in-depth mirroring
    ``read_page``'s ``_assert_within`` guard. The doc is silently
    dropped from the graph; we never hash arbitrary off-base files."""
    from dikw_core.api import _doc_id_for, _with_storage
    from dikw_core.schemas import DocumentRecord

    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/legit.md", title="Legit",
        body="# Legit\n",
    )

    # Drop a victim file outside the base + a docs row pointing at it
    # via a relative-escape path. Use the parent of tmp_path as a
    # distinct outside location.
    victim = tmp_path.parent / "outside.md"
    victim.write_text("# OUTSIDE\nSecret content.\n", encoding="utf-8")
    escape_path = "../outside.md"
    cfg, _root, storage = await _with_storage(tmp_path)
    del cfg
    try:
        await storage.upsert_document(
            DocumentRecord(
                doc_id=_doc_id_for(Layer.KNOWLEDGE, escape_path),
                path=escape_path,
                title="Escape",
                hash="0" * 64,
                mtime=0.0,
                layer=Layer.KNOWLEDGE,
                active=True,
            )
        )
    finally:
        await storage.close()

    g = await api.list_graph(tmp_path)
    # Legit doc is in the graph; the escape doc is filtered out.
    assert "knowledge/legit.md" in {n.path for n in g.nodes}
    assert escape_path not in {n.path for n in g.nodes}


@pytest.mark.asyncio
async def test_markdown_link_target_normalized_for_lookup(tmp_path: Path) -> None:
    """Markdown links can spell the same registered path with
    different case / Unicode form on case-insensitive or NFC-normalizing
    filesystems. The engine's K-layer storage already normalizes via
    ``path_key`` (NFC + casefold), so a markdown ``[B](Wiki/Foo.md)``
    link against a node registered as ``knowledge/foo.md`` MUST count as
    an edge — otherwise valid edges are silently dropped on macOS /
    Windows."""
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.SOURCE, path="sources/a.md", title="A",
        body="See [the other](Sources/B.md) for details.\n",
    )
    await _seed_doc(
        tmp_path, layer=Layer.SOURCE, path="sources/b.md", title="B",
        body="# B\n",
    )

    g = await api.list_graph(tmp_path)
    md_edges = [e for e in g.edges if e.type == "markdown"]
    assert len(md_edges) == 1, (
        f"expected one markdown edge after path normalization; got {md_edges}"
    )
    assert md_edges[0].source == "sources/a.md"
    assert md_edges[0].target == "sources/b.md"


@pytest.mark.asyncio
async def test_fuzzy_collision_includes_ambiguous_titles(tmp_path: Path) -> None:
    """When multiple docs share an exact title (collision-refuse),
    the fuzzy index must STILL include all of those paths under the
    normalized key. Otherwise a third doc whose title normalizes to
    the same key can become the sole fuzzy candidate, and a wikilink
    that fuzz-matches gets bound to the wrong target.

    Setup: two docs titled exactly ``Foo`` (collision pair) + one doc
    titled ``Foo!`` (also normalizes to ``foo`` after boundary-punct
    strip) + a referrer using ``[[Foos]]`` (plural-stems to ``foo``).
    Correct behavior: ``[[Foos]]`` is unresolved (3-way fuzzy
    collision), NOT silently bound to ``Foo!``.
    """
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.SOURCE, path="sources/foo-1.md", title="Foo",
        body="# Foo source one\n",
    )
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/Foo.md", title="Foo",
        body="# Foo wiki\n",
    )
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/Foo-bang.md", title="Foo!",
        body="# Foo bang\n",
    )
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/refers.md", title="Refers",
        body="See [[Foos]].\n",
    )

    g = await api.list_graph(tmp_path)
    assert all(
        e.target != "knowledge/Foo-bang.md" or e.source != "knowledge/refers.md"
        for e in g.edges
    ), (
        f"[[Foos]] must NOT silently bind to Foo! when ambiguous Foo "
        f"paths exist; got edges {g.edges!r}"
    )
    assert any(
        u.source == "knowledge/refers.md" and u.target_text == "Foos"
        for u in g.unresolved
    ), f"expected [[Foos]] in unresolved (3-way fuzzy collision); got {g.unresolved}"


@pytest.mark.asyncio
async def test_does_not_trigger_writes(tmp_path: Path) -> None:
    """``list_graph`` is read-only — no documents/links/chunks rows
    appear or change as a side effect of calling it."""
    init_test_base(tmp_path)
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/A.md", title="A",
        body="[[B]]\n",
    )
    await _seed_doc(
        tmp_path, layer=Layer.KNOWLEDGE, path="knowledge/B.md", title="B",
        body="# B\n",
    )

    counts_before = await api.status(tmp_path)
    await api.list_graph(tmp_path)
    counts_after = await api.status(tmp_path)
    # Compare every numeric field — if list_graph ever upserts links or
    # writes anything, these dicts diverge.
    assert counts_before.model_dump() == counts_after.model_dump()
