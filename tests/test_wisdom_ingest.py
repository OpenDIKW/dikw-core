"""End-to-end tests for the W-layer document pipeline.

0.4.0 reshape: ``dikw client ingest`` no longer scans
``<base>/wisdom/`` (BREAKING). Wisdom is indexed exclusively when a
file is dropped on disk and driven through ``persist_wisdom`` — in
production via ``api.write_wisdom_page`` (typed input); in these
tests via the ``ingest_wisdom_files`` helper, which mirrors what the
old W-layer ingest loop did for a list of on-disk files. The
wisdom-only ``DocumentRecord.status`` column carries the page's
frontmatter ``status: draft|published|favorite|archived`` enum
value; knowledge/source rows force ``status = None`` at the
application layer regardless of frontmatter.
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

from .fakes import (
    FakeEmbeddings,
    ingest_wisdom_files,
    init_test_base,
    seed_doc,
)


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
    await ingest_wisdom_files(
        wiki,
        ["wisdom/elon-musk/first-principles.md"],
        embedder=FakeEmbeddings(),
    )

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
    await ingest_wisdom_files(
        wiki,
        [f"wisdom/elon-musk/{status_value}.md"],
        embedder=FakeEmbeddings(),
    )

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
    await ingest_wisdom_files(
        wiki, ["wisdom/elon-musk/no-status.md"], embedder=FakeEmbeddings()
    )

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
    """A status value outside the enum must NOT block ``persist_wisdom``. The
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
    await ingest_wisdom_files(
        wiki, ["wisdom/elon-musk/weird.md"], embedder=FakeEmbeddings()
    )  # must not raise

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
async def test_ingest_does_not_scan_wisdom_directory(tmp_path: Path) -> None:
    """0.4.0 contract: a wisdom file dropped on disk is NOT indexed by
    ``dikw client ingest``. The wisdom row is created exclusively via
    ``write_wisdom_page`` / ``ingest_wisdom_files``. This is the
    BREAKING behavior change from 0.3.x — recorded as positive
    coverage so a future regression that re-introduces the W-layer
    scan branch fails this test.
    """
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/hand-written.md",
        "# Hand written\n\nuser put this here without write_wisdom_page.\n",
    )

    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = await _open_storage(wiki)
    try:
        wisdom_docs = list(
            await storage.list_documents(layer=Layer.WISDOM, active=True)
        )
    finally:
        await storage.close()
    assert wisdom_docs == []


@pytest.mark.asyncio
async def test_wisdom_wikilink_resolves_to_knowledge_page(tmp_path: Path) -> None:
    """A ``[[wikilink]]`` in a wisdom page must resolve to an existing
    knowledge page via the shared cross-layer title index — wisdom pages
    cite knowledge pages just like knowledge pages cite knowledge pages.
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

    await ingest_wisdom_files(
        wiki, ["wisdom/elon-musk/never-sell.md"], embedder=FakeEmbeddings()
    )

    storage = await _open_storage(wiki)
    try:
        wisdom_id = doc_id_for(Layer.WISDOM, "wisdom/elon-musk/never-sell.md")
        edges = await storage.links_from(wisdom_id)
    finally:
        await storage.close()
    resolved = [e.dst_path for e in edges]
    assert any(p.endswith("knowledge/concepts/tesla.md") for p in resolved), resolved


@pytest.mark.asyncio
async def test_wisdom_to_wisdom_link_resolves_after_target_indexed(
    tmp_path: Path,
) -> None:
    """A wisdom page links to another wisdom page by title. With the
    0.4.0 per-file write contract, the caller indexes B first, then A —
    A's ``[[B Page]]`` resolves on its own write because B is already
    in storage. ``ingest_wisdom_files`` drives the writes sequentially
    so this contract holds.
    """
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/b-page.md",
        "---\ntitle: B Page\n---\n# B Page\n\nthe second page.\n",
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/a-page.md",
        "---\ntitle: A Page\n---\n# A Page\n\nSee [[B Page]].\n",
    )
    await ingest_wisdom_files(
        wiki,
        [
            "wisdom/elon-musk/b-page.md",
            "wisdom/elon-musk/a-page.md",
        ],
        embedder=FakeEmbeddings(),
    )

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
    # User authors a wisdom page also titled "Tesla", then references it.
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/tesla.md",
        "---\ntitle: Tesla\n---\n# Tesla\n\npersonal note.\n",
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/musings.md",
        "# Musings\n\nSee [[Tesla]] for context.\n",
    )
    await ingest_wisdom_files(
        wiki,
        [
            "wisdom/elon-musk/tesla.md",
            "wisdom/elon-musk/musings.md",
        ],
        embedder=FakeEmbeddings(),
    )

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
    reserved first-class layer (own write entry), so even a broad
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


def test_iter_source_files_rejects_relative_escape(tmp_path: Path) -> None:
    """A ``sources[].path`` that escapes the base via ``../`` is a config
    error — ``sources`` is a managed tree under the base, not a license to
    read + index arbitrary files. The whole scan fails before any yield.
    """
    base = tmp_path / "base"
    (base / "sources").mkdir(parents=True)
    (base / "sources" / "ok.md").write_text("# ok\n", encoding="utf-8")
    # A real file OUTSIDE the base that a `../` config would otherwise slurp.
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.md").write_text("# secret\n", encoding="utf-8")

    from dikw_core.config import SourceConfig

    with pytest.raises(ValueError, match="outside the base"):
        list(
            iter_source_files(
                [SourceConfig(path="../outside", pattern="**/*.md")], root=base
            )
        )


def test_iter_source_files_rejects_absolute_escape(tmp_path: Path) -> None:
    """An absolute ``sources[].path`` pointing outside the base is rejected too."""
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("# secret\n", encoding="utf-8")

    from dikw_core.config import SourceConfig

    with pytest.raises(ValueError, match="outside the base"):
        list(
            iter_source_files(
                [SourceConfig(path=str(outside), pattern="**/*.md")], root=base
            )
        )


def test_iter_source_files_allows_absolute_path_inside_base(tmp_path: Path) -> None:
    """An absolute path that stays UNDER the base is fine — locks the boundary
    so the containment guard doesn't reject a legitimate absolute config.
    """
    base = tmp_path / "base"
    (base / "sources").mkdir(parents=True)
    (base / "sources" / "ok.md").write_text("# ok\n", encoding="utf-8")

    from dikw_core.config import SourceConfig

    yielded = list(
        iter_source_files(
            [SourceConfig(path=str((base / "sources").resolve()), pattern="**/*.md")],
            root=base,
        )
    )
    assert len(yielded) == 1
    assert yielded[0][0].name == "ok.md"


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
    await ingest_wisdom_files(
        wiki, ["wisdom/elon-musk/from-bio.md"], embedder=FakeEmbeddings()
    )

    storage = await _open_storage(wiki)
    try:
        wisdom_id = doc_id_for(Layer.WISDOM, "wisdom/elon-musk/from-bio.md")
        edges = await storage.provenance_from(wisdom_id)
    finally:
        await storage.close()
    assert [e.source_path for e in edges] == ["sources/notes/musk-bio.md"]
