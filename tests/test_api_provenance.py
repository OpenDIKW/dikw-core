"""Engine-side unit tests for ``api.read_provenance``.

Companion to ``test_api_links.py``: same shape, different edge. The HTTP
surface lives in ``tests/server/test_routes_pages.py``; this file pins
the pure helper that produces a ``PageProvenanceResult`` from
``(root, path, direction, limit)``.

Provenance is the K-page → D-source attribution edge — see
``docs/adr/0001-provenance-as-separate-edge.md`` for why it lives in a
dedicated ``provenance`` table separate from ``links``.

What we guard:

* path-not-registered → ``PageNotFound`` (same shape as ``list_links``)
* path resolution probes ``Layer.SOURCE`` first, then ``Layer.WIKI``
* forward leg surfaces dangling sources faithfully (``resolved=False``)
* reverse leg drops inactive K-pages
* ``direction="in"|"out"|"both"`` filter populates the right lists
* ``limit`` caps each list independently
* malformed ``limit`` (negative) raises ``ValueError`` before any I/O
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.api import _doc_id_for, _with_storage
from dikw_core.schemas import DocumentRecord, Layer

from .fakes import init_test_wiki


def _doc(
    path: str, layer: Layer = Layer.WIKI, *, title: str | None = None
) -> DocumentRecord:
    return DocumentRecord(
        doc_id=_doc_id_for(layer, path),
        path=path,
        title=title or path.rsplit("/", 1)[-1],
        hash="0" * 64,
        mtime=0.0,
        layer=layer,
        active=True,
    )


@pytest.mark.asyncio
async def test_read_provenance_forward_marks_dangling_when_source_missing(
    tmp_path: Path,
) -> None:
    """The forward leg returns every frontmatter entry, including ones
    whose ``source_path`` doesn't resolve to an active
    ``Layer.SOURCE`` document. Dangling entries carry ``resolved=False``
    + null ``doc_id``/``title`` so agents can detect drift; storage
    deliberately doesn't filter."""
    init_test_wiki(tmp_path)
    page_path = "wiki/page.md"
    real_src = "sources/real.md"
    ghost_src = "sources/ghost.md"

    cfg, _root, storage = await _with_storage(tmp_path)
    del cfg
    try:
        await storage.upsert_document(_doc(page_path, layer=Layer.WIKI))
        await storage.upsert_document(
            _doc(real_src, layer=Layer.SOURCE, title="Real")
        )
        # ghost_src deliberately NOT upserted.
        await storage.replace_provenance_from(
            _doc_id_for(Layer.WIKI, page_path),
            [real_src, ghost_src],
        )
    finally:
        await storage.close()

    result = await api.read_provenance(tmp_path, page_path, direction="out")
    assert result.path == page_path
    assert result.derived_pages == []
    by_path = {s.source_path: s for s in result.derived_from}
    assert by_path[real_src].resolved is True
    assert by_path[real_src].doc_id == _doc_id_for(Layer.SOURCE, real_src)
    assert by_path[real_src].title == "Real"
    assert by_path[ghost_src].resolved is False
    assert by_path[ghost_src].doc_id is None
    assert by_path[ghost_src].title is None


@pytest.mark.asyncio
async def test_read_provenance_reverse_filters_inactive_pages(
    tmp_path: Path,
) -> None:
    """The reverse leg drops K-pages that have been deactivated — agents
    can't follow them anyway, and surfacing them would mislead about
    which live pages claim a source. ``delete_document`` cascades
    provenance, so this guards the deactivate-without-delete path."""
    init_test_wiki(tmp_path)
    src_path = "sources/shared.md"
    active_page = "wiki/active.md"
    inactive_page = "wiki/inactive.md"

    cfg, _root, storage = await _with_storage(tmp_path)
    del cfg
    try:
        await storage.upsert_document(_doc(src_path, layer=Layer.SOURCE))
        await storage.upsert_document(_doc(active_page, layer=Layer.WIKI))
        await storage.upsert_document(_doc(inactive_page, layer=Layer.WIKI))
        for page in (active_page, inactive_page):
            await storage.replace_provenance_from(
                _doc_id_for(Layer.WIKI, page), [src_path]
            )
        # Soft-deactivate one of them. Provenance row survives the soft
        # delete (only delete_document hard-deletes provenance); the
        # API must filter.
        await storage.deactivate_document(_doc_id_for(Layer.WIKI, inactive_page))
    finally:
        await storage.close()

    result = await api.read_provenance(tmp_path, src_path, direction="in")
    assert result.path == src_path
    assert result.derived_from == []
    assert [dp.path for dp in result.derived_pages] == [active_page]


@pytest.mark.asyncio
async def test_read_provenance_both_direction_populates_both_lists(
    tmp_path: Path,
) -> None:
    """``direction='both'`` populates ``derived_from`` (forward) AND
    ``derived_pages`` (reverse) for the queried path. For a SOURCE-layer
    path, forward is always empty (no provenance row has SOURCE on the
    src side); for a WIKI-layer path, reverse is always empty (nothing
    claims a K-page as its own source). The empty list IS the answer."""
    init_test_wiki(tmp_path)
    wiki_page = "wiki/page.md"
    src_path = "sources/src.md"
    consumer_page = "wiki/consumer.md"

    cfg, _root, storage = await _with_storage(tmp_path)
    del cfg
    try:
        await storage.upsert_document(_doc(src_path, layer=Layer.SOURCE))
        await storage.upsert_document(_doc(wiki_page, layer=Layer.WIKI))
        await storage.upsert_document(_doc(consumer_page, layer=Layer.WIKI))
        # wiki_page claims src_path as its source.
        await storage.replace_provenance_from(
            _doc_id_for(Layer.WIKI, wiki_page), [src_path]
        )
        # consumer_page also claims src_path.
        await storage.replace_provenance_from(
            _doc_id_for(Layer.WIKI, consumer_page), [src_path]
        )
    finally:
        await storage.close()

    # WIKI-layer path: forward has one entry, reverse is empty.
    wiki_result = await api.read_provenance(
        tmp_path, wiki_page, direction="both"
    )
    assert [s.source_path for s in wiki_result.derived_from] == [src_path]
    assert wiki_result.derived_pages == []

    # SOURCE-layer path: forward empty, reverse has both consumers.
    src_result = await api.read_provenance(
        tmp_path, src_path, direction="both"
    )
    assert src_result.derived_from == []
    assert sorted(dp.path for dp in src_result.derived_pages) == sorted(
        [wiki_page, consumer_page]
    )


@pytest.mark.asyncio
async def test_read_provenance_limit_caps_each_side_independently(
    tmp_path: Path,
) -> None:
    """``limit`` caps ``derived_from`` and ``derived_pages`` independently
    — a hub source with many incoming K-pages must not starve the
    forward side of a separately-queried K-page. Mirrors ``list_links``
    semantics."""
    init_test_wiki(tmp_path)
    src_paths = [f"sources/s{i}.md" for i in range(5)]
    page_path = "wiki/page.md"

    cfg, _root, storage = await _with_storage(tmp_path)
    del cfg
    try:
        for sp in src_paths:
            await storage.upsert_document(_doc(sp, layer=Layer.SOURCE))
        await storage.upsert_document(_doc(page_path, layer=Layer.WIKI))
        await storage.replace_provenance_from(
            _doc_id_for(Layer.WIKI, page_path), src_paths
        )
    finally:
        await storage.close()

    result = await api.read_provenance(
        tmp_path, page_path, direction="both", limit=2
    )
    assert len(result.derived_from) == 2
    # Reverse list is empty for a WIKI path anyway → limit is a no-op
    # there; the cap on the forward side is what matters.
    assert result.derived_pages == []


@pytest.mark.asyncio
async def test_read_provenance_reverse_limit_caps_derived_pages(
    tmp_path: Path,
) -> None:
    """Reverse-leg ``limit`` truncates ``derived_pages`` for a SOURCE-layer
    path. Companion to the forward-leg coverage above; both sides go
    through symmetric ``[:limit]`` slicing and the asymmetric layer
    gate means only this test exercises the reverse branch."""
    init_test_wiki(tmp_path)
    src_path = "sources/hub.md"
    consumer_paths = [f"wiki/consumer-{i}.md" for i in range(5)]

    cfg, _root, storage = await _with_storage(tmp_path)
    del cfg
    try:
        await storage.upsert_document(_doc(src_path, layer=Layer.SOURCE))
        for cp in consumer_paths:
            await storage.upsert_document(_doc(cp, layer=Layer.WIKI))
            await storage.replace_provenance_from(
                _doc_id_for(Layer.WIKI, cp), [src_path]
            )
    finally:
        await storage.close()

    result = await api.read_provenance(
        tmp_path, src_path, direction="in", limit=2
    )
    assert len(result.derived_pages) == 2
    assert result.derived_from == []


@pytest.mark.asyncio
async def test_read_provenance_path_not_found_raises_page_not_found(
    tmp_path: Path,
) -> None:
    init_test_wiki(tmp_path)
    with pytest.raises(api.PageNotFound):
        await api.read_provenance(tmp_path, "wiki/missing.md")


@pytest.mark.asyncio
async def test_read_provenance_rejects_malformed_path(
    tmp_path: Path,
) -> None:
    """Empty / whitespace / NUL-containing paths raise ``PageNotFound``
    before any storage I/O — same shape as ``list_links`` /
    ``read_page``. Pins the early-rejection branch so storage stays
    out of the malformed-input blast radius."""
    init_test_wiki(tmp_path)
    for bad in ("", "   ", "wiki/foo\x00.md"):
        with pytest.raises(api.PageNotFound):
            await api.read_provenance(tmp_path, bad)


@pytest.mark.asyncio
async def test_read_provenance_negative_limit_raises(tmp_path: Path) -> None:
    """``limit < 0`` is a programmer error — raise ValueError before
    opening storage. Mirrors ``list_links``."""
    init_test_wiki(tmp_path)
    with pytest.raises(ValueError):
        await api.read_provenance(tmp_path, "wiki/page.md", limit=-1)


@pytest.mark.asyncio
async def test_read_provenance_wiki_layer_path_returns_only_forward(
    tmp_path: Path,
) -> None:
    """A path that resolves to ``Layer.WIKI`` never has reverse rows by
    construction (no K-page claims another K-page as its source), so
    ``direction='in'`` returns an empty list and ``direction='both'``
    returns only the forward attribution. Symmetric to the SOURCE-side
    coverage in ``test_read_provenance_both_direction_populates_both_lists``;
    pins the layer-asymmetric shape so a regression that, e.g., stops
    filtering by layer becomes a test failure rather than a quiet
    semantic drift."""
    init_test_wiki(tmp_path)
    wiki_page = "wiki/page.md"
    src_path = "sources/src.md"

    cfg, _root, storage = await _with_storage(tmp_path)
    del cfg
    try:
        await storage.upsert_document(_doc(src_path, layer=Layer.SOURCE))
        await storage.upsert_document(_doc(wiki_page, layer=Layer.WIKI))
        await storage.replace_provenance_from(
            _doc_id_for(Layer.WIKI, wiki_page), [src_path]
        )
    finally:
        await storage.close()

    in_only = await api.read_provenance(
        tmp_path, wiki_page, direction="in"
    )
    assert in_only.derived_from == []
    assert in_only.derived_pages == []


@pytest.mark.asyncio
async def test_read_provenance_wiki_path_reverse_ignores_malformed_wiki_sources(
    tmp_path: Path,
) -> None:
    """Defence-in-depth: even when a K-page's ``sources:`` accidentally
    lists another K-page path (malformed frontmatter), the reverse leg
    for that WIKI target must still return empty. The forward query
    marks the entry ``resolved=False`` (because only ``Layer.SOURCE``
    is resolved), but ``storage.provenance_to`` is layer-agnostic and
    keyed by ``source_path_key`` — without a layer gate, the reverse
    lookup would return the offender as a ``derived_pages`` entry,
    violating the documented "WIKI paths have empty reverse provenance"
    contract and letting agents treat a K-page as if it were a D-source.

    The gate (``if match.layer == Layer.SOURCE`` around the reverse
    branch) keeps the contract honest regardless of malformed inputs.
    """
    init_test_wiki(tmp_path)
    real_wiki = "wiki/real.md"
    offender_wiki = "wiki/offender.md"

    cfg, _root, storage = await _with_storage(tmp_path)
    del cfg
    try:
        await storage.upsert_document(_doc(real_wiki, layer=Layer.WIKI))
        await storage.upsert_document(
            _doc(offender_wiki, layer=Layer.WIKI)
        )
        # Offender lists the WIKI path as if it were a source — malformed
        # but possible (frontmatter is user-editable).
        await storage.replace_provenance_from(
            _doc_id_for(Layer.WIKI, offender_wiki), [real_wiki]
        )
    finally:
        await storage.close()

    # Querying the WIKI target's reverse leg must NOT surface the
    # offender as a derived page — that would imply real_wiki is a
    # D-source, which it isn't.
    result = await api.read_provenance(
        tmp_path, real_wiki, direction="both"
    )
    assert result.derived_pages == [], (
        f"WIKI path must have empty reverse provenance regardless of "
        f"malformed input; got {result.derived_pages}"
    )
