"""End-to-end tests for the 0.3.0 PR2 wisdom-as-documents pipeline.

`dikw ingest` now scans ``<root>/wisdom/<author>/**/*.md`` after the
``sources:`` scan and runs each file through the same ``persist_page``
pipeline as knowledge pages: a ``documents`` row at ``layer = WISDOM``,
chunks, embeddings, outgoing ``[[wikilinks]]``, and ``provenance``
edges from ``sources:`` frontmatter. The wisdom-only
``DocumentRecord.status`` column carries the page's frontmatter
``status: draft|published|favorite|archived`` enum value; knowledge/source
rows force ``status = None`` at the application layer regardless of
frontmatter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.config import load_config
from dikw_core.domains.data.path_norm import doc_id_for
from dikw_core.domains.data.sources import iter_source_files
from dikw_core.schemas import Layer, WisdomStatus
from dikw_core.storage import Storage, build_storage

from .fakes import FakeEmbeddings, init_test_base, seed_doc


def _drop_wisdom(wiki: Path, rel: str, body: str) -> None:
    p = wiki / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


async def _open_storage(wiki: Path) -> Storage:
    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    return storage


@pytest.mark.asyncio
async def test_wisdom_page_persists_as_document(tmp_path: Path) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
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
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
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
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
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
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
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
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
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
async def test_knowledge_page_status_frontmatter_forced_to_null(tmp_path: Path) -> None:
    """Application-level invariant: knowledge/source layer documents never
    carry a non-NULL status, even if the frontmatter happens to declare
    one. (The CHECK constraint accepts NULL anywhere; this guard lives
    in the engine to keep ``status`` semantically wisdom-only.)
    """
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
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
    fresh ``init_test_base`` scaffold leaves wisdom/ empty (or absent)
    and that path is the most common case for a brand-new base.
    """
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
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
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
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
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    _drop_wisdom(wiki, f"wisdom/{filename}", f"# aggregate {filename}\n\nbody.\n")
    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = await _open_storage(wiki)
    try:
        docs = list(await storage.list_documents(layer=Layer.WISDOM, active=True))
    finally:
        await storage.close()
    assert docs == []


@pytest.mark.asyncio
async def test_wisdom_wikilink_resolves_to_knowledge_page(tmp_path: Path) -> None:
    """A ``[[wikilink]]`` in a wisdom page must resolve to an existing
    knowledge page via the shared cross-layer title index — wisdom pages
    cite knowledge pages just like knowledge pages cite knowledge pages.

    Knowledge pages don't enter through ``api.ingest`` (they're synth-written),
    so the test seeds the K-layer ``documents`` row + file directly via
    ``seed_doc``, then runs wisdom ingest and asserts the link resolves
    against the seeded title.
    """
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    await seed_doc(
        wiki,
        layer=Layer.KNOWLEDGE,
        path="knowledge/concepts/tesla.md",
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
    assert any(p.endswith("knowledge/concepts/tesla.md") for p in resolved), resolved


@pytest.mark.asyncio
async def test_wisdom_status_frontmatter_only_edit_propagates(tmp_path: Path) -> None:
    """Editing ONLY the ``status:`` frontmatter (body unchanged) must
    still update ``documents.status`` on the next ingest. PR2's body-only
    content hash short-circuits when ``existing.hash == parsed.hash``;
    without an additional status check, a user flipping ``draft`` →
    ``published`` in Obsidian sees zero effect on storage and the new
    column silently desyncs from the file the user authored.
    """
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    rel = "wisdom/elon-musk/edit-me.md"
    _drop_wisdom(wiki, rel, "---\nstatus: draft\n---\n# Edit Me\n\nstable body.\n")
    await api.ingest(wiki, embedder=FakeEmbeddings())

    # Flip ONLY status; body bytes (including the heading + paragraph)
    # are byte-identical to the first pass.
    _drop_wisdom(
        wiki, rel, "---\nstatus: published\n---\n# Edit Me\n\nstable body.\n"
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = await _open_storage(wiki)
    try:
        doc = await storage.get_document(doc_id_for(Layer.WISDOM, rel))
    finally:
        await storage.close()
    assert doc is not None
    assert doc.status == WisdomStatus.PUBLISHED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename",
    ["Principles.md", "LESSONS.MD", "Patterns.Md"],
)
async def test_wisdom_legacy_aggregate_skip_is_case_insensitive(
    tmp_path: Path, filename: str
) -> None:
    """On NTFS / APFS / HFS+ the legacy aggregate ``Principles.md`` may
    exist with capitalized spelling. The skip-set must match
    case-insensitively or the legacy file leaks in as a first-class
    wisdom document on upgrading bases.
    """
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    _drop_wisdom(wiki, f"wisdom/{filename}", "# aggregate\n\nbody.\n")
    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = await _open_storage(wiki)
    try:
        docs = list(await storage.list_documents(layer=Layer.WISDOM, active=True))
    finally:
        await storage.close()
    assert docs == []


@pytest.mark.asyncio
async def test_wisdom_to_wisdom_link_resolves_within_same_ingest(
    tmp_path: Path,
) -> None:
    """Two new wisdom pages in the SAME ingest: page A links to page B
    by title. Without a hoisted, batch-aware title index, the wisdom
    loop resolves each file against whatever is already in storage —
    so a forward reference stays unresolved until a second ingest.
    PR2 must pre-seed so all wisdom titles in the batch are
    addressable from each other on the first pass.
    """
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/a-page.md",
        "---\ntitle: A Page\n---\n# A Page\n\nSee [[B Page]].\n",
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/b-page.md",
        "---\ntitle: B Page\n---\n# B Page\n\nthe second page.\n",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = await _open_storage(wiki)
    try:
        a_id = doc_id_for(Layer.WISDOM, "wisdom/elon-musk/a-page.md")
        edges = await storage.links_from(a_id)
    finally:
        await storage.close()
    resolved = [e.dst_path for e in edges]
    assert any(p.endswith("wisdom/elon-musk/b-page.md") for p in resolved), resolved


@pytest.mark.asyncio
async def test_knowledge_wisdom_title_collision_refuses_exact_resolve(
    tmp_path: Path,
) -> None:
    """When a knowledge page and a wisdom page share the same exact title,
    ``[[Title]]`` from a third wisdom page must NOT silently bind to
    either — the refuse-to-resolve invariant (cf. design.md) requires
    the wikilink to stay broken so ``dikw lint`` surfaces the
    ambiguity, rather than the dict-merge picking whichever layer
    iterates first.
    """
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    # Seed a knowledge page titled "Tesla".
    await seed_doc(
        wiki,
        layer=Layer.KNOWLEDGE,
        path="knowledge/concepts/tesla.md",
        body="---\ntitle: Tesla\n---\n# Tesla\n\nthe company.\n",
        title="Tesla",
    )
    # User authors a wisdom page also titled "Tesla".
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/tesla.md",
        "---\ntitle: Tesla\n---\n# Tesla\n\npersonal note.\n",
    )
    # A third wisdom page references the ambiguous title.
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/musings.md",
        "# Musings\n\nSee [[Tesla]] for context.\n",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = await _open_storage(wiki)
    try:
        musings_id = doc_id_for(Layer.WISDOM, "wisdom/elon-musk/musings.md")
        edges = await storage.links_from(musings_id)
    finally:
        await storage.close()
    # The wikilink edge must NOT have been written — neither layer should
    # win the exact-match resolve when both carry the title.
    wikilink_edges = [e for e in edges if e.link_type.value == "wikilink"]
    assert wikilink_edges == [], (
        "title collision must refuse to resolve, got: "
        f"{[(e.dst_path, e.link_type.value) for e in edges]}"
    )


def test_iter_source_files_excludes_wisdom_prefix(tmp_path: Path) -> None:
    """``iter_source_files`` is the D-layer scan entry. ``wisdom/`` is a
    reserved first-class layer (own ingest branch), so even a broad
    user config like ``sources: [{path: '.', pattern: '**/*.md'}]``
    must not double-yield wisdom files as source rows — that would
    produce a duplicate ``source:`` doc-id, double chunks, and double
    embedding spend for the same on-disk file.
    """
    root = tmp_path / "knowledge"
    root.mkdir()
    # A wisdom file under wisdom/...
    (root / "wisdom" / "elon-musk").mkdir(parents=True)
    (root / "wisdom" / "elon-musk" / "note.md").write_text(
        "# Wisdom\n", encoding="utf-8"
    )
    # And a legit source file at the root.
    (root / "real-source.md").write_text("# Source\n", encoding="utf-8")

    from dikw_core.config import SourceConfig

    yielded = list(iter_source_files([SourceConfig(path=".", pattern="**/*.md")], root=root))
    logical_paths = [logical for _, logical in yielded]
    assert "real-source.md" in logical_paths
    assert not any(p.startswith("wisdom/") for p in logical_paths), logical_paths


@pytest.mark.asyncio
async def test_wisdom_sources_become_provenance(tmp_path: Path) -> None:
    """``sources:`` frontmatter on a wisdom page populates the
    ``provenance`` table just like it does on knowledge pages."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

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
