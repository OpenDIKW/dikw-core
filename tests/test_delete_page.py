"""Engine-layer tests for :func:`api.delete_page` — the immediate, single
-document delete verb spanning Data / Knowledge / Wisdom.

``delete_page`` resolves which layer a path lives in (storage probe),
purges the document row + its outgoing edges (``delete_document``), then
moves the on-disk file to ``<base>/trash/<rel>`` with an audit stamp. It
is symmetric with :func:`api.write_wisdom_page`: an explicitly-targeted
single-document write/delete, immediate (no propose/apply), with ``trash/``
as the recovery safety net.

Inbound edges from *live* pages are deliberately left dangling (they
surface as ``broken_wikilink`` on the next lint) — the verb never rewrites
another page's body. That non-cascade invariant is asserted here at the
verb boundary.
"""

from __future__ import annotations

from pathlib import Path

import frontmatter
import pytest

from dikw_core import api
from dikw_core.config import load_config
from dikw_core.domains.data.path_norm import doc_id_for
from dikw_core.schemas import Layer, LinkRecord, LinkType
from dikw_core.storage import Storage, build_storage

from .fakes import init_test_base, seed_doc


async def _open_storage(wiki: Path) -> Storage:
    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    return storage


@pytest.mark.asyncio
async def test_delete_knowledge_page_purges_row_and_trashes_file(
    tmp_path: Path,
) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    path = "knowledge/concepts/dead.md"
    await seed_doc(
        wiki, layer=Layer.KNOWLEDGE, path=path, body="# Dead\n\nbody\n", title="Dead"
    )

    report = await api.delete_page(wiki, path)

    assert report.path == path
    assert report.layer == Layer.KNOWLEDGE
    assert report.trashed_to == "trash/knowledge/concepts/dead.md"

    # Original path emptied; file moved under trash/ preserving its layout.
    assert not (wiki / path).exists()
    trash_file = wiki / "trash" / "knowledge" / "concepts" / "dead.md"
    assert trash_file.is_file()

    # Audit ``trashed:`` block: reason + at, no proposal_id (manual delete).
    trashed = frontmatter.loads(
        trash_file.read_text(encoding="utf-8")
    ).metadata.get("trashed")
    assert isinstance(trashed, dict)
    assert trashed.get("reason") == "delete"
    assert isinstance(trashed.get("at"), str) and trashed["at"]
    assert "proposal_id" not in trashed

    # Storage row fully purged (not just deactivated).
    storage = await _open_storage(wiki)
    try:
        assert (
            await storage.get_document(doc_id_for(Layer.KNOWLEDGE, path))
        ) is None
    finally:
        await storage.close()


@pytest.mark.parametrize(
    "layer,path",
    [
        (Layer.SOURCE, "sources/notes/raw.md"),
        (Layer.WISDOM, "wisdom/elon-musk/never-sell.md"),
    ],
)
@pytest.mark.asyncio
async def test_delete_spans_data_and_wisdom_layers(
    tmp_path: Path, layer: Layer, path: str
) -> None:
    """The verb is layer-agnostic: a D source and a W page both delete +
    trash under their own ``trash/<layer>/...`` subtree."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    await seed_doc(wiki, layer=layer, path=path, body="# T\n\nbody\n", title="T")

    report = await api.delete_page(wiki, path)

    assert report.layer == layer
    assert report.trashed_to == f"trash/{path}"
    assert (wiki / "trash" / path).is_file()
    assert not (wiki / path).exists()

    storage = await _open_storage(wiki)
    try:
        assert await storage.get_document(doc_id_for(layer, path)) is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_delete_unknown_path_raises_page_not_found(tmp_path: Path) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    with pytest.raises(api.PageNotFound):
        await api.delete_page(wiki, "knowledge/never-existed.md")


@pytest.mark.parametrize("bad", ["", "   ", "knowledge/\x00evil.md"])
@pytest.mark.asyncio
async def test_delete_malformed_path_raises_page_not_found(
    tmp_path: Path, bad: str
) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    with pytest.raises(api.PageNotFound):
        await api.delete_page(wiki, bad)


@pytest.mark.asyncio
async def test_delete_missing_file_purges_row_reports_no_trash(
    tmp_path: Path,
) -> None:
    """A row whose backing file is already gone (the ``missing_file`` drift
    case) still purges cleanly: the row is what we delete, and there is
    nothing to trash → ``trashed_to`` is None, no trash file appears."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    path = "knowledge/gone.md"
    await seed_doc(wiki, layer=Layer.KNOWLEDGE, path=path, body="# Gone\n", title="Gone")
    # Remove the on-disk file out from under the row.
    (wiki / path).unlink()

    report = await api.delete_page(wiki, path)

    assert report.path == path
    assert report.trashed_to is None
    assert not (wiki / "trash" / path).exists()
    storage = await _open_storage(wiki)
    try:
        assert await storage.get_document(doc_id_for(Layer.KNOWLEDGE, path)) is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_delete_file_vanished_mid_move_is_graceful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the file vanishes between the is_file() check and the trash move
    (a lost TOCTOU race), the FileNotFoundError is treated as 'already
    gone': the row is still purged and trashed_to is None — no crash."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    path = "knowledge/racy.md"
    await seed_doc(wiki, layer=Layer.KNOWLEDGE, path=path, body="# Racy\n", title="Racy")

    def _vanish(**_kwargs: object) -> Path:
        raise FileNotFoundError("file removed mid-delete")

    monkeypatch.setattr("dikw_core.api_delete.move_to_trash", _vanish)

    report = await api.delete_page(wiki, path)
    assert report.trashed_to is None
    storage = await _open_storage(wiki)
    try:
        assert await storage.get_document(doc_id_for(Layer.KNOWLEDGE, path)) is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_delete_trash_oserror_propagates_after_purge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine filesystem failure during the trash move (disk full,
    permission) is NOT swallowed — it propagates as a failed delete. The
    row is already purged (purge-first ordering), leaving the file at its
    original path for recovery."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    path = "knowledge/doomed.md"
    await seed_doc(wiki, layer=Layer.KNOWLEDGE, path=path, body="# Doomed\n", title="Doomed")

    def _disk_full(**_kwargs: object) -> Path:
        raise OSError("simulated disk full")

    monkeypatch.setattr("dikw_core.api_delete.move_to_trash", _disk_full)

    with pytest.raises(OSError, match="disk full"):
        await api.delete_page(wiki, path)

    storage = await _open_storage(wiki)
    try:
        assert await storage.get_document(doc_id_for(Layer.KNOWLEDGE, path)) is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_delete_leaves_inbound_links_intact(tmp_path: Path) -> None:
    """Deleting ``B`` purges B's row + B's *outgoing* edges, but a live
    page ``A`` that links ``[[B]]`` keeps its outgoing edge — it now
    dangles and surfaces as ``broken_wikilink`` on the next lint. The verb
    must never silently rewrite A's body."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    await seed_doc(
        wiki, layer=Layer.KNOWLEDGE, path="knowledge/a.md",
        body="# A\n\n[[B]]\n", title="A",
    )
    await seed_doc(
        wiki, layer=Layer.KNOWLEDGE, path="knowledge/b.md", body="# B\n", title="B"
    )
    doc_a = doc_id_for(Layer.KNOWLEDGE, "knowledge/a.md")
    doc_b = doc_id_for(Layer.KNOWLEDGE, "knowledge/b.md")

    storage = await _open_storage(wiki)
    try:
        await storage.replace_links_from(
            doc_a,
            [
                LinkRecord(
                    src_doc_id=doc_a,
                    dst_path="knowledge/b.md",
                    link_type=LinkType.WIKILINK,
                    line=3,
                )
            ],
        )
    finally:
        await storage.close()

    report = await api.delete_page(wiki, "knowledge/b.md")
    # The report surfaces the blast radius inline: A's [[B]] just dangled.
    assert report.inbound_broken == 1

    storage = await _open_storage(wiki)
    try:
        assert await storage.get_document(doc_b) is None
        a_links = await storage.links_from(doc_a)
        assert any(link.dst_path == "knowledge/b.md" for link in a_links), (
            "A's outgoing [[B]] edge must survive B's deletion (becomes "
            "broken_wikilink), not be cascade-cleaned"
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_delete_inbound_broken_counts_distinct_pages(tmp_path: Path) -> None:
    """``inbound_broken`` counts distinct *live* referring pages — multiple
    edges from one page count once, and a self-link (purged with the doc)
    is excluded."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    for name in ("a", "b", "hub"):
        await seed_doc(
            wiki, layer=Layer.KNOWLEDGE, path=f"knowledge/{name}.md",
            body=f"# {name}\n", title=name.upper(),
        )
    hub = doc_id_for(Layer.KNOWLEDGE, "knowledge/hub.md")
    doc_a = doc_id_for(Layer.KNOWLEDGE, "knowledge/a.md")
    doc_b = doc_id_for(Layer.KNOWLEDGE, "knowledge/b.md")

    storage = await _open_storage(wiki)
    try:
        # a -> hub (twice, two lines), b -> hub, hub -> hub (self).
        await storage.replace_links_from(
            doc_a,
            [
                LinkRecord(src_doc_id=doc_a, dst_path="knowledge/hub.md",
                           link_type=LinkType.WIKILINK, line=1),
                LinkRecord(src_doc_id=doc_a, dst_path="knowledge/hub.md",
                           link_type=LinkType.WIKILINK, line=5),
            ],
        )
        await storage.replace_links_from(
            doc_b,
            [LinkRecord(src_doc_id=doc_b, dst_path="knowledge/hub.md",
                        link_type=LinkType.WIKILINK, line=2)],
        )
        await storage.replace_links_from(
            hub,
            [LinkRecord(src_doc_id=hub, dst_path="knowledge/hub.md",
                        link_type=LinkType.WIKILINK, line=9)],
        )
    finally:
        await storage.close()

    report = await api.delete_page(wiki, "knowledge/hub.md")
    # a and b are the distinct live referrers; the hub self-link is excluded.
    assert report.inbound_broken == 2


@pytest.mark.asyncio
async def test_delete_custom_reason_and_appends_knowledge_log(
    tmp_path: Path,
) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    path = "knowledge/dup.md"
    await seed_doc(wiki, layer=Layer.KNOWLEDGE, path=path, body="# Dup\n", title="Dup")

    await api.delete_page(wiki, path, reason="duplicate of Canonical")

    trash_file = wiki / "trash" / path
    trashed = frontmatter.loads(
        trash_file.read_text(encoding="utf-8")
    ).metadata.get("trashed")
    assert isinstance(trashed, dict)
    assert trashed.get("reason") == "duplicate of Canonical"

    storage = await _open_storage(wiki)
    try:
        log = await storage.list_knowledge_log()
    finally:
        await storage.close()
    assert any(e.action == "delete" and e.src == path for e in log)


@pytest.mark.asyncio
async def test_delete_inactive_doc_is_deletable(tmp_path: Path) -> None:
    """A half-written (``active=False``) row is still a row the user can
    delete — unlike ``read_page``, the delete probe matches regardless of
    ``active``."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    path = "knowledge/half.md"
    await seed_doc(
        wiki, layer=Layer.KNOWLEDGE, path=path, body="# Half\n",
        title="Half", active=False,
    )

    report = await api.delete_page(wiki, path)

    assert report.path == path
    assert report.trashed_to == f"trash/{path}"
    storage = await _open_storage(wiki)
    try:
        assert await storage.get_document(doc_id_for(Layer.KNOWLEDGE, path)) is None
    finally:
        await storage.close()
