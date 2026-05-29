"""Tests for ``persist_knowledge`` — the public K-layer indexing function
shared between synth and lint-apply.

The synth path is already covered by ``test_persist_knowledge_page.py``
(routed through ``api._persist_knowledge_page``); this file pins the
provider-free path lint-apply uses (``embedder=None``) so the indexing
contract — document upsert + chunks + outgoing-link reconciliation —
holds without an embedder configured.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core.domains.data.path_norm import doc_id_for
from dikw_core.domains.knowledge.page import build_page, write_page
from dikw_core.domains.knowledge.page_index import persist_knowledge
from dikw_core.schemas import DocumentRecord, Layer, LinkType
from dikw_core.storage.base import Storage


@pytest.mark.asyncio
async def test_persist_knowledge_writes_document_chunks_and_links(
    parametrized_storage: Storage, tmp_path: Path,
) -> None:
    """End-to-end: write a page that links to a pre-seeded target and
    verify that document, chunks, and outgoing wikilink edge all land
    in storage. ``embedder=None`` exercises the lint-apply path."""
    storage = parametrized_storage

    target = build_page(title="Foo", body="# Foo\nbody.\n", type_="concept")
    write_page(tmp_path, target)
    await persist_knowledge(
        storage=storage, root=tmp_path, path=target.path, title=target.title,
        embedder=None, embedding_model="", text_version_id=None,
    )

    src = build_page(
        title="Source",
        body="# Source\n\nSee [[Foo]] here.\n",
        type_="concept",
        path="knowledge/concepts/source.md",
    )
    write_page(tmp_path, src)
    result = await persist_knowledge(
        storage=storage, root=tmp_path, path=src.path, title=src.title,
        embedder=None, embedding_model="", text_version_id=None,
    )

    assert result.unresolved_wikilinks == 0
    docs = list(await storage.list_documents(layer=Layer.KNOWLEDGE, active=True))
    src_doc = next(d for d in docs if d.path == src.path)
    assert src_doc.title == "Source"

    src_id = doc_id_for(Layer.KNOWLEDGE, src.path)
    chunks = await storage.list_chunks(src_id)
    assert len(chunks) >= 1

    links = await storage.links_from(src_id)
    assert any(
        link.link_type == LinkType.WIKILINK and link.dst_path == target.path
        for link in links
    )


@pytest.mark.asyncio
async def test_persist_knowledge_skips_embedding_when_embedder_none(
    parametrized_storage: Storage, tmp_path: Path,
) -> None:
    """Lint-apply caller passes embedder=None to keep apply provider-free.
    Document + chunks must still land; the next ``dikw client ingest`` will
    reconcile embeddings via ``doc.hash`` drift."""
    storage = parametrized_storage
    page = build_page(title="X", body="body\n", type_="concept")
    write_page(tmp_path, page)

    await persist_knowledge(
        storage=storage, root=tmp_path, path=page.path,
        embedder=None, embedding_model="", text_version_id=None,
    )

    doc_id = doc_id_for(Layer.KNOWLEDGE, page.path)
    chunks = await storage.list_chunks(doc_id)
    assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_persist_knowledge_reports_unresolved_count(
    parametrized_storage: Storage, tmp_path: Path,
) -> None:
    """An unresolved ``[[Missing]]`` should bump the returned count so
    callers can fold it into reports (synth: SynthReport, lint apply: skip)."""
    storage = parametrized_storage
    page = build_page(
        title="Source",
        body="See [[Missing]] which has no target.\n",
        type_="concept",
        path="knowledge/concepts/src.md",
    )
    write_page(tmp_path, page)
    # Seed a placeholder document row so resolve_links can self-exclude.
    # The hash/mtime placeholders below are overwritten by persist_knowledge
    # — only the row's existence at this path matters for the test.
    src_id = doc_id_for(Layer.KNOWLEDGE, page.path)
    await storage.upsert_document(DocumentRecord(
        doc_id=src_id, path=page.path, title=page.title,
        hash="placeholder-overwritten-by-persist", mtime=0.0,
        layer=Layer.KNOWLEDGE, active=True,
    ))

    result = await persist_knowledge(
        storage=storage, root=tmp_path, path=page.path, title=page.title,
        embedder=None, embedding_model="", text_version_id=None,
    )

    assert result.unresolved_wikilinks == 1
