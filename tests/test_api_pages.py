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
    assert page.body == (root / rel).read_text(encoding="utf-8")
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

    with pytest.raises(api.PageNotFound):
        await api.read_page(tmp_path, bad_path)
