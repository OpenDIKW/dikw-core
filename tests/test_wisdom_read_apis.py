"""HTTP/CLI page-API coverage for the 0.3.0 PR3 wisdom layer.

PR2 wrote wisdom files into ``documents`` but ``read_page``,
``list_links``, and ``read_provenance`` still iterated
``(Layer.SOURCE, Layer.WIKI)``. PR3 extends them to ``Layer.WISDOM`` so
a user who ingested ``wisdom/elon-musk/*.md`` can actually fetch what
they wrote — and so wikilinks pointing at wisdom pages from other
layers resolve at read time, not just persist time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.api import PageNotFound
from dikw_core.schemas import Layer

from .fakes import FakeEmbeddings, init_test_wiki, seed_doc


def _drop_wisdom(wiki: Path, rel: str, body: str) -> None:
    p = wiki / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


@pytest.mark.asyncio
async def test_read_page_returns_wisdom_layer(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/first-principles.md",
        "---\ntitle: First Principles\n---\n# First Principles\n\nbody.\n",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    page = await api.read_page(wiki, "wisdom/elon-musk/first-principles.md")
    assert page.path == "wisdom/elon-musk/first-principles.md"
    assert page.title == "First Principles"
    assert page.body.startswith("# First Principles")


@pytest.mark.asyncio
async def test_read_page_404_for_unindexed_wisdom_path(tmp_path: Path) -> None:
    """An unindexed wisdom path must still 404 — wisdom inclusion in
    the read path must not weaken the index-driven safety guard."""
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    with pytest.raises(PageNotFound):
        await api.read_page(wiki, "wisdom/ghost/never-written.md")


@pytest.mark.asyncio
async def test_list_links_returns_outgoing_from_wisdom_to_wiki(
    tmp_path: Path,
) -> None:
    """A wisdom page links to a wiki page via ``[[Tesla]]``. ``list_links``
    must (a) accept the wisdom source path, and (b) resolve the wiki dst
    candidate doc_id so the outgoing edge surfaces in the response."""
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    await seed_doc(
        wiki,
        layer=Layer.WIKI,
        path="wiki/concepts/tesla.md",
        body="---\ntitle: Tesla\n---\n# Tesla\n",
        title="Tesla",
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/note.md",
        "# Note\n\nSee [[Tesla]] for context.\n",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    links = await api.list_links(
        wiki, "wisdom/elon-musk/note.md", direction="out"
    )
    out_paths = [e.dst_path for e in links.outgoing]
    assert "wiki/concepts/tesla.md" in out_paths


@pytest.mark.asyncio
async def test_list_links_incoming_to_wiki_credits_wisdom_backlinks(
    tmp_path: Path,
) -> None:
    """A wiki page's incoming links must include wikilinks from wisdom
    pages — without this the wiki page's backlink view loses the
    wisdom-side conversations the user actually wrote."""
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    await seed_doc(
        wiki,
        layer=Layer.WIKI,
        path="wiki/concepts/tesla.md",
        body="---\ntitle: Tesla\n---\n# Tesla\n",
        title="Tesla",
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/note.md",
        "# Note\n\nSee [[Tesla]].\n",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    links = await api.list_links(wiki, "wiki/concepts/tesla.md", direction="in")
    in_paths = [e.src_path for e in links.incoming]
    assert "wisdom/elon-musk/note.md" in in_paths


@pytest.mark.asyncio
async def test_read_provenance_forward_returns_wisdom_sources(
    tmp_path: Path,
) -> None:
    """A wisdom page with ``sources:`` frontmatter must expose its
    forward provenance (``derived_from``) via the read API. Without
    extending the layer probe, the call 404s even though
    ``persist_page`` wrote the provenance row."""
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    src_dir = wiki / "sources" / "notes"
    src_dir.mkdir(parents=True)
    (src_dir / "musk-bio.md").write_text("# Bio\n\nfacts.\n", encoding="utf-8")
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/from-bio.md",
        "---\nsources:\n  - sources/notes/musk-bio.md\n---\n# From Bio\n\nbody.\n",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    prov = await api.read_provenance(
        wiki, "wisdom/elon-musk/from-bio.md", direction="out"
    )
    sources = [p.source_path for p in prov.derived_from]
    assert sources == ["sources/notes/musk-bio.md"]
    assert prov.derived_pages == []  # forward-only request


@pytest.mark.asyncio
async def test_read_provenance_reverse_excludes_wisdom_from_derived_pages(
    tmp_path: Path,
) -> None:
    """The reverse leg (``direction=in``) reports K-layer pages whose
    ``sources:`` claim this source path. Wisdom pages legitimately
    cite sources too, so they SHOULD surface as ``DerivedPage`` rows —
    the existing SOURCE-only gate was a wiki-only artifact."""
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    src_dir = wiki / "sources" / "notes"
    src_dir.mkdir(parents=True)
    (src_dir / "musk-bio.md").write_text("# Bio\n\nfacts.\n", encoding="utf-8")
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/from-bio.md",
        "---\nsources:\n  - sources/notes/musk-bio.md\n---\n# From Bio\n\nbody.\n",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    prov = await api.read_provenance(
        wiki, "sources/notes/musk-bio.md", direction="in"
    )
    derived = sorted(p.path for p in prov.derived_pages)
    assert "wisdom/elon-musk/from-bio.md" in derived
