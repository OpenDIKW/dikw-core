"""End-to-end tests for the 0.3.0 PR2 wisdom-as-documents pipeline.

`dikw ingest` now scans ``<root>/wisdom/<author>/**/*.md`` after the
``sources:`` scan and runs each file through the same ``persist_page``
pipeline as wiki pages: a ``documents`` row at ``layer = WISDOM``,
chunks, embeddings, outgoing ``[[wikilinks]]``, and ``provenance``
edges from ``sources:`` frontmatter. The wisdom-only
``DocumentRecord.status`` column carries the page's frontmatter
``status: draft|published|favorite|archived`` enum value; wiki/source
rows force ``status = None`` at the application layer regardless of
frontmatter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.config import load_config
from dikw_core.domains.data.path_norm import doc_id_for
from dikw_core.schemas import Layer, WisdomStatus
from dikw_core.storage import build_storage

from .fakes import FakeEmbeddings, init_test_wiki, seed_doc


def _drop_wisdom(wiki: Path, rel: str, body: str) -> None:
    p = wiki / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


async def _open_storage(wiki: Path):  # type: ignore[no-untyped-def]
    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    return storage


@pytest.mark.asyncio
async def test_wisdom_page_persists_as_document(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/first-principles.md",
        "---\ntitle: First Principles\n---\n# First Principles\n\nReason from physics, not analogy.\n",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = await _open_storage(wiki)
    try:
        doc = await storage.get_document(
            doc_id_for(Layer.WISDOM, "wisdom/elon-musk/first-principles.md")
        )
    finally:
        await storage.close()
    assert doc is not None
    assert doc.layer == Layer.WISDOM
    assert doc.path == "wisdom/elon-musk/first-principles.md"
    assert doc.title == "First Principles"
    assert doc.status is None  # frontmatter omitted ≡ published, stored as NULL


@pytest.mark.asyncio
async def test_wisdom_ingest_idempotent_on_unchanged_hash(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    _drop_wisdom(wiki, "wisdom/elon-musk/x.md", "# X\n\nbody.\n")

    r1 = await api.ingest(wiki, embedder=FakeEmbeddings())
    r2 = await api.ingest(wiki, embedder=FakeEmbeddings())
    # First pass added the wisdom page; second pass classifies it as unchanged.
    assert r1.scanned >= 1
    assert r2.unchanged >= 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_value,expected",
    [
        ("draft", WisdomStatus.DRAFT),
        ("published", WisdomStatus.PUBLISHED),
        ("favorite", WisdomStatus.FAVORITE),
        ("archived", WisdomStatus.ARCHIVED),
    ],
)
async def test_wisdom_status_frontmatter_persists(
    tmp_path: Path, status_value: str, expected: WisdomStatus
) -> None:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    _drop_wisdom(
        wiki,
        f"wisdom/elon-musk/{status_value}.md",
        f"---\nstatus: {status_value}\n---\n# {status_value}\n\nbody.\n",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = await _open_storage(wiki)
    try:
        doc = await storage.get_document(
            doc_id_for(Layer.WISDOM, f"wisdom/elon-musk/{status_value}.md")
        )
    finally:
        await storage.close()
    assert doc is not None
    assert doc.status == expected


@pytest.mark.asyncio
async def test_wisdom_no_status_frontmatter_persists_as_null(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    _drop_wisdom(wiki, "wisdom/elon-musk/no-status.md", "# Plain\n\nbody.\n")
    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = await _open_storage(wiki)
    try:
        doc = await storage.get_document(
            doc_id_for(Layer.WISDOM, "wisdom/elon-musk/no-status.md")
        )
    finally:
        await storage.close()
    assert doc is not None
    assert doc.status is None


@pytest.mark.asyncio
async def test_wisdom_unknown_status_treated_as_null(tmp_path: Path) -> None:
    """A status value outside the enum must NOT block ingest. The
    parser hands ``None`` to the document layer and the
    ``invalid_wisdom_status`` lint kind surfaces the warning later.
    """
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/weird.md",
        "---\nstatus: weird_value\n---\n# Weird\n\nbody.\n",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())  # must not raise

    storage = await _open_storage(wiki)
    try:
        doc = await storage.get_document(
            doc_id_for(Layer.WISDOM, "wisdom/elon-musk/weird.md")
        )
    finally:
        await storage.close()
    assert doc is not None
    assert doc.status is None


@pytest.mark.asyncio
async def test_wiki_page_status_frontmatter_forced_to_null(tmp_path: Path) -> None:
    """Application-level invariant: wiki/source layer documents never
    carry a non-NULL status, even if the frontmatter happens to declare
    one. (The CHECK constraint accepts NULL anywhere; this guard lives
    in the engine to keep ``status`` semantically wisdom-only.)
    """
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    src_dir = wiki / "sources" / "notes"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "with-status.md").write_text(
        "---\nstatus: favorite\n---\n# Note\n\nbody.\n", encoding="utf-8"
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = await _open_storage(wiki)
    try:
        doc = await storage.get_document(
            doc_id_for(Layer.SOURCE, "sources/notes/with-status.md")
        )
    finally:
        await storage.close()
    assert doc is not None
    assert doc.layer == Layer.SOURCE
    assert doc.status is None


@pytest.mark.asyncio
async def test_ingest_no_wisdom_directory_is_noop(tmp_path: Path) -> None:
    """A base without a ``wisdom/`` tree must ingest cleanly — the
    fresh ``init_test_wiki`` scaffold leaves wisdom/ empty (or absent)
    and that path is the most common case for a brand-new base.
    """
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    # Even with a wisdom directory that contains only a .gitkeep, no
    # wisdom documents should land — gitkeep isn't markdown.
    report = await api.ingest(wiki, embedder=FakeEmbeddings())
    storage = await _open_storage(wiki)
    try:
        wisdom_docs = list(
            await storage.list_documents(layer=Layer.WISDOM, active=True)
        )
    finally:
        await storage.close()
    assert wisdom_docs == []
    assert report.errors == ()


@pytest.mark.asyncio
async def test_wisdom_candidates_subdir_is_skipped(tmp_path: Path) -> None:
    """Bases upgrading from 0.2.x may still carry a ``wisdom/_candidates/``
    directory. The ingest scanner must skip it so drained queue
    artifacts don't show up as freshly indexed wisdom documents.
    """
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    _drop_wisdom(
        wiki,
        "wisdom/_candidates/old-candidate.md",
        "# old candidate\n\nbody.\n",
    )
    _drop_wisdom(wiki, "wisdom/elon-musk/real.md", "# real wisdom\n\nbody.\n")
    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = await _open_storage(wiki)
    try:
        docs = list(await storage.list_documents(layer=Layer.WISDOM, active=True))
    finally:
        await storage.close()
    paths = sorted(d.path for d in docs)
    assert paths == ["wisdom/elon-musk/real.md"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename", ["principles.md", "lessons.md", "patterns.md"]
)
async def test_wisdom_aggregate_files_are_skipped(
    tmp_path: Path, filename: str
) -> None:
    """The legacy 0.2.x aggregate files at ``wisdom/{principles,lessons,
    patterns}.md`` are hardcoded skips, so an upgrading base that hasn't
    deleted them yet doesn't accidentally index drained aggregations as
    first-class wisdom documents."""
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    _drop_wisdom(wiki, f"wisdom/{filename}", f"# aggregate {filename}\n\nbody.\n")
    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = await _open_storage(wiki)
    try:
        docs = list(await storage.list_documents(layer=Layer.WISDOM, active=True))
    finally:
        await storage.close()
    assert docs == []


@pytest.mark.asyncio
async def test_wisdom_wikilink_resolves_to_wiki_page(tmp_path: Path) -> None:
    """A ``[[wikilink]]`` in a wisdom page must resolve to an existing
    wiki page via the shared cross-layer title index — wisdom pages
    cite wiki pages just like wiki pages cite wiki pages.

    Wiki pages don't enter through ``api.ingest`` (they're synth-written),
    so the test seeds the K-layer ``documents`` row + file directly via
    ``seed_doc``, then runs wisdom ingest and asserts the link resolves
    against the seeded title.
    """
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    await seed_doc(
        wiki,
        layer=Layer.WIKI,
        path="wiki/concepts/tesla.md",
        body="---\ntitle: Tesla\n---\n# Tesla\n\nthe company.\n",
        title="Tesla",
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/never-sell.md",
        "# Never Sell\n\nSee [[Tesla]] for context.\n",
    )

    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = await _open_storage(wiki)
    try:
        wisdom_id = doc_id_for(Layer.WISDOM, "wisdom/elon-musk/never-sell.md")
        edges = await storage.links_from(wisdom_id)
    finally:
        await storage.close()
    resolved = [e.dst_path for e in edges]
    assert any(p.endswith("wiki/concepts/tesla.md") for p in resolved), resolved


@pytest.mark.asyncio
async def test_wisdom_sources_become_provenance(tmp_path: Path) -> None:
    """``sources:`` frontmatter on a wisdom page populates the
    ``provenance`` table just like it does on wiki pages."""
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)

    src_dir = wiki / "sources" / "notes"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "musk-bio.md").write_text(
        "# Musk Bio\n\nfacts.\n", encoding="utf-8"
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/from-bio.md",
        "---\nsources:\n  - sources/notes/musk-bio.md\n---\n# From Bio\n\nbody.\n",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = await _open_storage(wiki)
    try:
        wisdom_id = doc_id_for(Layer.WISDOM, "wisdom/elon-musk/from-bio.md")
        edges = await storage.provenance_from(wisdom_id)
    finally:
        await storage.close()
    assert [e.source_path for e in edges] == ["sources/notes/musk-bio.md"]
