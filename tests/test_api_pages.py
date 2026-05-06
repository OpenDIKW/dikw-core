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

from .fakes import FakeEmbeddings, init_test_wiki

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
