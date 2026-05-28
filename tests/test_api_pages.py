"""Engine-side unit tests for ``api.read_page``.

The HTTP-layer surface lives in ``tests/server/test_routes_pages.py``;
this file exercises the pure helper that produces a ``PageReadResult``
from ``(root, path)`` so the seam (path-not-registered → ``PageNotFound``,
anchors land in seq order, body matches the on-disk file) stays guarded
without booting a server.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.schemas import Layer

from .fakes import FakeEmbeddings, init_test_wiki, png_with_dims

FIXTURES = Path(__file__).parent / "fixtures" / "notes"


def _bootstrap_wiki_with_fixture(tmp_path: Path) -> tuple[Path, str]:
    """Init a wiki, drop one fixture markdown into ``sources/demo/``,
    return ``(root, sources_relative_path)``. Caller still has to ingest."""
    init_test_wiki(tmp_path)
    src_dir = tmp_path / "sources" / "demo"
    src_dir.mkdir(parents=True, exist_ok=True)
    fixture = next(FIXTURES.glob("*.md"))
    shutil.copy2(fixture, src_dir / fixture.name)
    return tmp_path, f"sources/demo/{fixture.name}"


@pytest.mark.asyncio
async def test_read_page_returns_body_and_anchors(tmp_path: Path) -> None:
    root, rel = _bootstrap_wiki_with_fixture(tmp_path)
    await api.ingest(root, embedder=FakeEmbeddings())

    page = await api.read_page(root, rel)
    assert page.path == rel
    assert page.layer == Layer.SOURCE
    raw = (root / rel).read_text(encoding="utf-8")
    # ``body`` is the parsed body (front-matter stripped); a fixture
    # carrying ``---``-fenced front-matter must not roundtrip through
    # raw. We check that the parsed body is a suffix-substring of raw.
    assert page.body in raw
    assert "title:" not in page.body or page.body.find("title:") > raw.find("title:")
    assert page.doc_id  # non-empty
    # Ingest produced ≥ 1 chunk for any reasonable fixture.
    assert page.anchors, "expected at least one chunk anchor"
    # Anchors arrive in seq order.
    seqs = [a.seq for a in page.anchors]
    assert seqs == sorted(seqs)
    # Each anchor's [start, end) is a valid slice of body.
    for anchor in page.anchors:
        assert 0 <= anchor.start <= anchor.end <= len(page.body)
        assert anchor.chunk_id > 0  # adapter assigned a real id


@pytest.mark.asyncio
async def test_read_page_unknown_path_raises(tmp_path: Path) -> None:
    init_test_wiki(tmp_path)
    with pytest.raises(api.PageNotFound):
        await api.read_page(tmp_path, "sources/does-not-exist.md")


@pytest.mark.asyncio
async def test_read_page_path_escape_attempt_raises(tmp_path: Path) -> None:
    """``../etc/passwd``-style paths are not in the documents table, so
    they get the same ``PageNotFound`` — no special-case sandbox check
    needed because the lookup is index-driven."""
    init_test_wiki(tmp_path)
    with pytest.raises(api.PageNotFound):
        await api.read_page(tmp_path, "../etc/passwd")


@pytest.mark.asyncio
async def test_read_page_k_layer_anchors_survive_frontmatter_roundtrip(
    tmp_path: Path,
) -> None:
    """K-layer (synthesised wiki) pages go through ``write_page`` →
    ``frontmatter.dumps`` on write, then ``parse_any`` on read. The
    library's serialise→parse cycle isn't always byte-stable on the
    body, so hashing the in-memory ``page.body`` at synthesise time
    while comparing against the on-disk parsed body at read time would
    falsely flag every K-layer page as stale (anchors=[]).

    This test directly drives ``_persist_wiki_page`` (no real LLM
    needed) and verifies anchors survive the serialise→parse cycle.
    """
    from dikw_core.api import _persist_wiki_page
    from dikw_core.domains.knowledge.wiki import build_page, write_page

    init_test_wiki(tmp_path)
    page = build_page(
        title="Rountrip Test",
        body=(
            "# Rountrip Test\n\n"
            "Body paragraph one — the chunker will see this after\n"
            "frontmatter.dumps + frontmatter.loads roundtrip.\n\n"
            "## Subsection\n\nBody paragraph two — also chunkable.\n"
        ),
        tags=["test"],
        sources=["sources/whatever.md"],
    )
    write_page(tmp_path, page)

    cfg, _root, storage = await api._with_storage(tmp_path)
    try:
        await _persist_wiki_page(
            storage=storage,
            root=tmp_path,
            page=page,
            embedder=None,  # no embed leg needed for the anchor check
            embedding_model="fake",
            text_version_id=None,
            cjk_tokenizer=cfg.retrieval.cjk_tokenizer,
        )
    finally:
        await storage.close()

    result = await api.read_page(tmp_path, page.path)
    assert result.layer == Layer.WIKI
    assert result.anchors, (
        "K-layer anchors got dropped — hash mismatch between "
        "synthesise-time and read-time body coordinate space"
    )
    # Slice-correctness: the same anchor↔chunk text alignment we lock
    # for source pages must hold for K-layer pages too.
    cfg, _root, storage = await api._with_storage(tmp_path)
    del cfg
    try:
        chunks = await storage.list_chunks(result.doc_id)
    finally:
        await storage.close()
    by_seq = {c.seq: c for c in chunks}
    for anchor in result.anchors:
        chunk = by_seq[anchor.seq]
        assert result.body[anchor.start : anchor.end] == chunk.text


@pytest.mark.asyncio
async def test_read_page_returns_raw_body_on_parse_failure(
    tmp_path: Path,
) -> None:
    """If the on-disk file is corrupt (e.g. user broke the YAML
    front-matter externally), ``read_page`` must still serve the raw
    text and return empty anchors instead of 500-ing. The user gets
    "I can read it but anchors are gone" — same UX as the modified-
    since-ingest case."""
    root, rel = _bootstrap_wiki_with_fixture(tmp_path)
    await api.ingest(root, embedder=FakeEmbeddings())

    # Externally corrupt the front-matter.
    (root / rel).write_text(
        "---\n"
        "title: : :  # invalid YAML — bare colons\n"
        " - bad: indent\n"
        "---\n\n"
        "Body still readable.\n",
        encoding="utf-8",
    )

    page = await api.read_page(root, rel)
    assert "Body still readable" in page.body
    assert page.anchors == []
    # Parse-failure fallback returns the same ``{}`` as a metadata-free
    # file. Callers can disambiguate via the empty ``anchors`` signal.
    assert page.frontmatter == {}


@pytest.mark.asyncio
async def test_read_page_returns_frontmatter(tmp_path: Path) -> None:
    """``read_page`` must surface the parsed YAML front-matter so callers
    can read ``tags`` / ``sources`` / ``status`` / custom keys without
    re-reading the on-disk file themselves. Fixture ``dikw.md`` carries
    ``title: DIKW pyramid`` + ``tags: [dikw, concept]`` — both must come
    through verbatim."""
    init_test_wiki(tmp_path)
    src_dir = tmp_path / "sources" / "demo"
    src_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIXTURES / "dikw.md", src_dir / "dikw.md")
    rel = "sources/demo/dikw.md"
    await api.ingest(tmp_path, embedder=FakeEmbeddings())

    page = await api.read_page(tmp_path, rel)
    assert page.frontmatter.get("title") == "DIKW pyramid"
    assert page.frontmatter.get("tags") == ["dikw", "concept"]


@pytest.mark.asyncio
async def test_read_page_frontmatter_empty_when_absent(tmp_path: Path) -> None:
    """A markdown file with no ``---`` fence returns ``frontmatter == {}``
    (not ``None``) so callers can ``page.frontmatter.get(k)`` unconditionally."""
    init_test_wiki(tmp_path)
    src_dir = tmp_path / "sources" / "demo"
    src_dir.mkdir(parents=True, exist_ok=True)
    rel = "sources/demo/no-fm.md"
    (tmp_path / rel).write_text(
        "# Just a heading\n\nNo front-matter at all.\n",
        encoding="utf-8",
    )
    await api.ingest(tmp_path, embedder=FakeEmbeddings())

    page = await api.read_page(tmp_path, rel)
    assert page.frontmatter == {}


@pytest.mark.asyncio
async def test_read_page_k_layer_frontmatter_roundtrip(tmp_path: Path) -> None:
    """K-layer pages go through ``write_page`` (frontmatter.dumps) on the
    way to disk and back through ``parse_any`` on read. ``tags`` /
    ``sources`` must survive that roundtrip in the ``frontmatter`` dict
    so agents that drive ``synth`` can inspect provenance + tags via the
    same endpoint they already use for body."""
    from dikw_core.api import _persist_wiki_page
    from dikw_core.domains.knowledge.wiki import build_page, write_page

    init_test_wiki(tmp_path)
    page = build_page(
        title="Frontmatter Roundtrip",
        body="# Frontmatter Roundtrip\n\nBody.\n",
        tags=["roundtrip", "test"],
        sources=["sources/whatever.md"],
    )
    write_page(tmp_path, page)

    cfg, _root, storage = await api._with_storage(tmp_path)
    try:
        await _persist_wiki_page(
            storage=storage,
            root=tmp_path,
            page=page,
            embedder=None,
            embedding_model="fake",
            text_version_id=None,
            cjk_tokenizer=cfg.retrieval.cjk_tokenizer,
        )
    finally:
        await storage.close()

    result = await api.read_page(tmp_path, page.path)
    assert result.layer == Layer.WIKI
    assert result.frontmatter.get("tags") == ["roundtrip", "test"]
    assert result.frontmatter.get("sources") == ["sources/whatever.md"]


@pytest.mark.asyncio
async def test_read_page_anchors_align_with_parsed_body(tmp_path: Path) -> None:
    """The chunker runs on the front-matter-stripped body, so chunk
    ``start`` / ``end`` are offsets into that stripped body — not the
    raw on-disk file. ``read_page`` must return the parsed body so
    ``body[anchor.start:anchor.end]`` slices to the same text the
    chunker produced. Otherwise a YAML front-matter doc would slide
    every anchor by the front-matter width.
    """
    init_test_wiki(tmp_path)
    src_dir = tmp_path / "sources" / "demo"
    src_dir.mkdir(parents=True, exist_ok=True)
    rel = "sources/demo/with-frontmatter.md"
    raw = (
        "---\n"
        "title: Front-matter test\n"
        "tags: [test]\n"
        "---\n"
        "\n"
        "# Heading One\n"
        "\n"
        "This document carries YAML front-matter that the markdown\n"
        "backend strips before chunking; if read_page returned the raw\n"
        "file, anchor offsets would land in the wrong half.\n"
        "\n"
        "## Heading Two\n"
        "\n"
        "Second paragraph for chunker fodder.\n"
    )
    (tmp_path / rel).write_text(raw, encoding="utf-8")
    await api.ingest(tmp_path, embedder=FakeEmbeddings())

    page = await api.read_page(tmp_path, rel)
    assert page.body != raw, "front-matter must be stripped from body"
    assert "title:" not in page.body, "front-matter leaked into body"
    assert page.anchors, "expected at least one chunk anchor"

    # Cross-check: every anchor's slice of body must match what was
    # stored in chunks.text at ingest. We can't read chunks.text from
    # the public API directly, but the seq order + bounds is enough
    # to prove the coordinate space is consistent.
    cfg, _root, storage = await api._with_storage(tmp_path)
    del cfg
    try:
        chunks = await storage.list_chunks(page.doc_id)
    finally:
        await storage.close()
    by_seq = {c.seq: c for c in chunks}
    for anchor in page.anchors:
        chunk = by_seq[anchor.seq]
        assert page.body[anchor.start : anchor.end] == chunk.text, (
            f"anchor seq={anchor.seq} sliced {page.body[anchor.start:anchor.end]!r} "
            f"but stored chunk text was {chunk.text!r}"
        )


@pytest.mark.asyncio
async def test_read_page_drops_anchors_when_file_modified(tmp_path: Path) -> None:
    """If the on-disk file changed since ingest, the indexed chunk
    offsets no longer line up — read_page must drop the anchors so
    callers don't slice stale text. Body still serves so the user can
    at least read the current content."""
    root, rel = _bootstrap_wiki_with_fixture(tmp_path)
    await api.ingest(root, embedder=FakeEmbeddings())

    # Sanity: anchors land before the edit.
    before = await api.read_page(root, rel)
    assert before.anchors

    (root / rel).write_text(
        "Completely different content the chunker never saw.\n",
        encoding="utf-8",
    )
    after = await api.read_page(root, rel)
    assert after.body.startswith("Completely different")
    assert after.anchors == [], (
        "stale anchors leaked through after file modification"
    )


@pytest.mark.asyncio
async def test_read_page_skips_deactivated_doc(tmp_path: Path) -> None:
    """A doc whose ``active=False`` was set after ingest must NOT be
    readable through ``/v1/base/pages/{path}`` — the policy needs to
    match the list endpoint, which defaults to ``active=True`` and
    silently drops deactivated rows."""
    root, rel = _bootstrap_wiki_with_fixture(tmp_path)
    await api.ingest(root, embedder=FakeEmbeddings())

    # Sanity: it reads fine while active.
    page = await api.read_page(root, rel)
    doc_id = page.doc_id

    # Flip the ``active`` flag through a direct storage handle, mirroring
    # what ``deactivate_document`` does in the real ingest cleanup path.
    cfg, _root, storage = await api._with_storage(root)
    del cfg
    try:
        await storage.deactivate_document(doc_id)
    finally:
        await storage.close()

    with pytest.raises(api.PageNotFound):
        await api.read_page(root, rel)


@pytest.mark.asyncio
async def test_read_page_unindexed_file_raises(tmp_path: Path) -> None:
    """A file that exists on disk under the base root but isn't a
    registered document (``dikw.yml``, raw ``sources/`` files dropped in
    after ingest, etc.) must NOT be readable through this endpoint —
    only indexed pages are addressable."""
    init_test_wiki(tmp_path)
    # ``dikw.yml`` exists after init but is not a DocumentRecord.
    assert (tmp_path / "dikw.yml").is_file()
    with pytest.raises(api.PageNotFound):
        await api.read_page(tmp_path, "dikw.yml")


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "   ",  # whitespace-only
        "foo\x00bar.md",  # null byte — Path() raises ValueError on Linux
    ],
)
@pytest.mark.asyncio
async def test_read_page_rejects_malformed_path(
    tmp_path: Path, bad_path: str
) -> None:
    """Empty / whitespace-only / null-byte paths must surface as
    ``PageNotFound``, NOT as a 500 from a deeper Path/storage error."""
    init_test_wiki(tmp_path)
    with pytest.raises(api.PageNotFound):
        await api.read_page(tmp_path, bad_path)


@pytest.mark.asyncio
async def test_read_page_relative_to_guard_traps_corrupt_doc(
    tmp_path: Path,
) -> None:
    """Defence in depth: if the documents table somehow ended up with a
    row whose ``path`` resolves outside the base root (corruption, a
    direct DB write, a future ingest bug), :func:`read_page` MUST refuse
    to read the file — the ``relative_to`` check is the last line.

    The HTTP-layer ``..``-traversal test in
    ``tests/server/test_routes_pages.py`` doesn't actually reach this
    code path because httpx normalises ``..`` segments client-side; this
    test plants the corrupt doc directly into storage instead.
    """
    init_test_wiki(tmp_path)

    cfg, _root, storage = await api._with_storage(tmp_path)
    del cfg
    try:
        from dikw_core.schemas import DocumentRecord, Layer

        bad_path = "../etc/secret.md"
        await storage.upsert_document(
            DocumentRecord(
                doc_id=api._doc_id_for(Layer.SOURCE, bad_path),
                path=bad_path,
                hash="0" * 64,
                mtime=0.0,
                layer=Layer.SOURCE,
                active=True,
            )
        )
    finally:
        await storage.close()

    with pytest.raises(api.PageNotFound) as excinfo:
        await api.read_page(tmp_path, bad_path)

    # Lock that PageNotFound was raised by the ``relative_to`` guard
    # (not by the missing-file fallback) — without this, removing the
    # guard would still let the test pass when the outside file
    # happened to not exist.
    cause = excinfo.value.__cause__
    assert isinstance(cause, ValueError), (
        f"expected ValueError from relative_to, got {type(cause).__name__}"
    )


# ---- assets attached to read_page ----------------------------------------


@pytest.mark.asyncio
async def test_read_page_includes_assets(tmp_path: Path) -> None:
    """A page whose markdown references local images must surface those
    assets in :class:`PageReadResult.assets` after ingest, each entry
    carrying enough info for a remote client to fetch the bytes via
    ``GET /v1/assets/{asset_id}``.
    """
    init_test_wiki(tmp_path)
    src_dir = tmp_path / "sources" / "demo"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "arch.png").write_bytes(png_with_dims(640, 480))
    (src_dir / "tiny.png").write_bytes(png_with_dims(10, 10))
    rel = "sources/demo/with-images.md"
    (tmp_path / rel).write_text(
        "# Architecture\n\n"
        "First the wide diagram: ![arch](./arch.png)\n\n"
        "Then a small one: ![tiny](./tiny.png)\n\n"
        "Some more body text so the chunker has fodder.\n",
        encoding="utf-8",
    )
    await api.ingest(tmp_path, embedder=FakeEmbeddings())

    page = await api.read_page(tmp_path, rel)
    assert len(page.assets) == 2, (
        f"expected both images to surface on the page, got {page.assets!r}"
    )
    # Every record carries the fields a remote client needs to render.
    for asset in page.assets:
        assert asset.asset_id and len(asset.asset_id) == 64, "sha256 hex id"
        assert asset.mime == "image/png"
        assert asset.bytes > 0
        assert asset.url == f"/v1/assets/{asset.asset_id}", (
            "url must be a stable server-relative path so clients zero-pase"
        )
        # ``original_paths`` is what the user typed in markdown — needed
        # so the client can map the path back to the asset for display.
        assert asset.original_paths, "original_paths must round-trip"
    # The two assets must be distinct (different content → different ids).
    ids = {a.asset_id for a in page.assets}
    assert len(ids) == 2


@pytest.mark.asyncio
async def test_read_page_no_assets_when_no_image_refs(tmp_path: Path) -> None:
    """A plain-text page (no image refs) returns an empty assets list,
    not ``None`` — keeps the wire shape uniform so clients don't need a
    null check."""
    root, rel = _bootstrap_wiki_with_fixture(tmp_path)
    await api.ingest(root, embedder=FakeEmbeddings())
    page = await api.read_page(root, rel)
    # Fixture markdowns under tests/fixtures/notes are text-only.
    assert page.assets == []


@pytest.mark.asyncio
async def test_read_page_dedups_assets_referenced_twice(tmp_path: Path) -> None:
    """If the same image is referenced from multiple chunks (or twice in
    one chunk), the page-level ``assets`` list must dedupe by
    ``asset_id`` — content-addressed assets have one canonical entry."""
    init_test_wiki(tmp_path)
    src_dir = tmp_path / "sources" / "demo"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "same.png").write_bytes(png_with_dims(100, 100))
    rel = "sources/demo/repeated.md"
    body = (
        "# Repeats\n\n"
        "![same](./same.png)\n\n"
        "Some paragraph between the two refs to force separate chunks.\n\n"
        "## Second section\n\n"
        "![same again](./same.png)\n"
    )
    (tmp_path / rel).write_text(body, encoding="utf-8")
    await api.ingest(tmp_path, embedder=FakeEmbeddings())

    page = await api.read_page(tmp_path, rel)
    assert len(page.assets) == 1, (
        f"expected dedup by asset_id, got {[a.asset_id for a in page.assets]!r}"
    )
