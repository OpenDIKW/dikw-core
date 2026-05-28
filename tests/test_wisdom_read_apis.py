"""HTTP/CLI page-API coverage for the W-layer document pipeline.

``read_page``, ``list_links``, and ``read_provenance`` must surface
``Layer.WISDOM`` rows alongside K-layer rows — a user who authored
``wisdom/<author>/*.md`` via the W-layer write entry must be able to
fetch what they wrote, and wikilinks pointing at wisdom pages from
other layers must resolve at read time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.api import PageNotFound
from dikw_core.schemas import Layer

from .fakes import FakeEmbeddings, ingest_wisdom_files, init_test_base, seed_doc


def _drop_wisdom(wiki: Path, rel: str, body: str) -> None:
    p = wiki / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


@pytest.mark.asyncio
async def test_read_page_returns_wisdom_layer(tmp_path: Path) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/first-principles.md",
        "---\ntitle: First Principles\n---\n# First Principles\n\nbody.\n",
    )
    await ingest_wisdom_files(
        wiki,
        ["wisdom/elon-musk/first-principles.md"],
        embedder=FakeEmbeddings(),
    )

    page = await api.read_page(wiki, "wisdom/elon-musk/first-principles.md")
    assert page.path == "wisdom/elon-musk/first-principles.md"
    assert page.title == "First Principles"
    assert page.body.startswith("# First Principles")


@pytest.mark.asyncio
async def test_read_page_404_for_unindexed_wisdom_path(tmp_path: Path) -> None:
    """An unindexed wisdom path must still 404 — wisdom inclusion in
    the read path must not weaken the index-driven safety guard."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    with pytest.raises(PageNotFound):
        await api.read_page(wiki, "wisdom/ghost/never-written.md")


@pytest.mark.asyncio
async def test_list_links_returns_outgoing_from_wisdom_to_wiki(
    tmp_path: Path,
) -> None:
    """A wisdom page links to a knowledge page via ``[[Tesla]]``. ``list_links``
    must (a) accept the wisdom source path, and (b) resolve the wiki dst
    candidate doc_id so the outgoing edge surfaces in the response."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    await seed_doc(
        wiki,
        layer=Layer.KNOWLEDGE,
        path="knowledge/concepts/tesla.md",
        body="---\ntitle: Tesla\n---\n# Tesla\n",
        title="Tesla",
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/note.md",
        "# Note\n\nSee [[Tesla]] for context.\n",
    )
    await ingest_wisdom_files(
        wiki, ["wisdom/elon-musk/note.md"], embedder=FakeEmbeddings()
    )

    links = await api.list_links(
        wiki, "wisdom/elon-musk/note.md", direction="out"
    )
    out_paths = [e.dst_path for e in links.outgoing]
    assert "knowledge/concepts/tesla.md" in out_paths


@pytest.mark.asyncio
async def test_list_links_incoming_to_knowledge_credits_wisdom_backlinks(
    tmp_path: Path,
) -> None:
    """A knowledge page's incoming links must include wikilinks from wisdom
    pages — without this the knowledge page's backlink view loses the
    wisdom-side conversations the user actually wrote."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    await seed_doc(
        wiki,
        layer=Layer.KNOWLEDGE,
        path="knowledge/concepts/tesla.md",
        body="---\ntitle: Tesla\n---\n# Tesla\n",
        title="Tesla",
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/note.md",
        "# Note\n\nSee [[Tesla]].\n",
    )
    await ingest_wisdom_files(
        wiki, ["wisdom/elon-musk/note.md"], embedder=FakeEmbeddings()
    )

    links = await api.list_links(wiki, "knowledge/concepts/tesla.md", direction="in")
    in_paths = [e.src_path for e in links.incoming]
    assert "wisdom/elon-musk/note.md" in in_paths


@pytest.mark.asyncio
async def test_read_provenance_forward_returns_wisdom_sources(
    tmp_path: Path,
) -> None:
    """A wisdom page with ``sources:`` frontmatter must expose its
    forward provenance (``derived_from``) via the read API."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    src_dir = wiki / "sources" / "notes"
    src_dir.mkdir(parents=True)
    (src_dir / "musk-bio.md").write_text("# Bio\n\nfacts.\n", encoding="utf-8")
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/from-bio.md",
        "---\nsources:\n  - sources/notes/musk-bio.md\n---\n# From Bio\n\nbody.\n",
    )
    await ingest_wisdom_files(
        wiki, ["wisdom/elon-musk/from-bio.md"], embedder=FakeEmbeddings()
    )

    prov = await api.read_provenance(
        wiki, "wisdom/elon-musk/from-bio.md", direction="out"
    )
    sources = [p.source_path for p in prov.derived_from]
    assert sources == ["sources/notes/musk-bio.md"]
    assert prov.derived_pages == []  # forward-only request


@pytest.mark.asyncio
async def test_read_provenance_reverse_includes_wisdom_in_derived_pages(
    tmp_path: Path,
) -> None:
    """The reverse leg (``direction=in``) reports K-layer pages whose
    ``sources:`` claim this source path. Wisdom pages legitimately
    cite sources too, so they SHOULD surface as ``DerivedPage`` rows
    alongside knowledge pages — anything cited via ``sources:`` is a
    legitimate derived page regardless of layer."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    src_dir = wiki / "sources" / "notes"
    src_dir.mkdir(parents=True)
    (src_dir / "musk-bio.md").write_text("# Bio\n\nfacts.\n", encoding="utf-8")
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/from-bio.md",
        "---\nsources:\n  - sources/notes/musk-bio.md\n---\n# From Bio\n\nbody.\n",
    )
    # Source layer is indexed by ``api.ingest``; the reverse provenance
    # leg reads from the source path so the D-layer doc must exist.
    await api.ingest(wiki, embedder=FakeEmbeddings())
    await ingest_wisdom_files(
        wiki, ["wisdom/elon-musk/from-bio.md"], embedder=FakeEmbeddings()
    )

    prov = await api.read_provenance(
        wiki, "sources/notes/musk-bio.md", direction="in"
    )
    derived = sorted(p.path for p in prov.derived_pages)
    assert "wisdom/elon-musk/from-bio.md" in derived
