"""Unit tests for ``persist_knowledge`` — the K-layer persist entry.

Covers the public contract of the 0.4.0 K-layer persist function:
upsert_document + chunk + FTS + (inline-or-deferred embed) +
replace_links_from + replace_provenance_from. The status field is
hard-clamped to None (K-layer invariant — status is wisdom-only).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dikw_core.domains.knowledge.page_index import persist_knowledge
from dikw_core.schemas import KnowledgePersistResult, Layer
from dikw_core.storage.sqlite import SQLiteStorage

from .fakes import FakeEmbeddings, FlakyEmbedder, init_test_base, register_text_version


def _write_k_page(
    root: Path,
    rel_path: str,
    *,
    title: str,
    body: str,
    sources: list[str] | None = None,
) -> None:
    fm: dict[str, object] = {"title": title}
    if sources is not None:
        fm["sources"] = sources
    abs_path = root / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    text = "---\n" + yaml.safe_dump(fm, sort_keys=False) + "---\n\n" + body
    abs_path.write_text(text, encoding="utf-8")


async def _new_storage_in_base(tmp_path: Path) -> tuple[Path, SQLiteStorage]:
    root = tmp_path / "base"
    init_test_base(root, description="persist_knowledge test base")
    storage = SQLiteStorage(root / ".dikw" / "index.sqlite")
    await storage.connect()
    await storage.migrate()
    return root, storage


async def test_persist_knowledge_inline_embeds_with_embedder(
    tmp_path: Path,
) -> None:
    """With ``embedder`` + ``text_version_id``, chunks are inline-embedded."""
    root, storage = await _new_storage_in_base(tmp_path)
    try:
        _write_k_page(root, "knowledge/topic-x.md", title="Topic X", body="Alpha.")
        version_id = await register_text_version(storage)
        embedder = FakeEmbeddings()

        result = await persist_knowledge(
            storage=storage,
            root=root,
            path="knowledge/topic-x.md",
            embedder=embedder,
            embedding_model="fake",
            text_version_id=version_id,
        )

        assert isinstance(result, KnowledgePersistResult)
        assert result.chunks_embedded > 0
        assert result.chunks_pending_embedding == 0
        assert result.resolved_title == "Topic X"
        # Doc landed with K-layer + status=None (clamped).
        doc = await storage.get_document("knowledge:knowledge/topic-x.md")
        assert doc is not None
        assert doc.layer is Layer.KNOWLEDGE
        assert doc.status is None
        # Embeddings landed in the per-version vec table.
        vecs = await storage.get_chunk_embeddings(
            result.chunk_ids, version_id=version_id
        )
        assert len(vecs) == len(result.chunk_ids)
    finally:
        await storage.close()


async def test_persist_knowledge_defers_embedding_without_embedder(
    tmp_path: Path,
) -> None:
    """Without embedder, chunks land in storage + FTS but vectors do not.

    ``chunks_pending_embedding`` captures the deferred count so the
    caller (lint apply) can surface it. ``list_chunks_missing_embedding``
    picks them up on the next ingest's resume scan.
    """
    root, storage = await _new_storage_in_base(tmp_path)
    try:
        _write_k_page(root, "knowledge/deferred.md", title="Deferred", body="Bravo.")
        version_id = await register_text_version(storage)

        result = await persist_knowledge(
            storage=storage,
            root=root,
            path="knowledge/deferred.md",
            embedder=None,
            text_version_id=None,
        )

        assert result.chunks_embedded == 0
        assert result.chunks_pending_embedding > 0
        # Chunks exist in chunks table — FTS too — but vec table has no rows.
        vecs = await storage.get_chunk_embeddings(
            result.chunk_ids, version_id=version_id
        )
        assert vecs == {}
        # Resume scan would pick them up.
        missing = await storage.list_chunks_missing_embedding(version_id=version_id)
        missing_ids = {c.chunk_id for c in missing}
        assert set(result.chunk_ids).issubset(missing_ids)
    finally:
        await storage.close()


async def test_persist_knowledge_replaces_links_and_provenance(
    tmp_path: Path,
) -> None:
    """Outgoing wikilinks + ``sources:`` frontmatter are reconciled atomically."""
    root, storage = await _new_storage_in_base(tmp_path)
    try:
        # Seed a target K page so the wikilink resolves.
        _write_k_page(root, "knowledge/target.md", title="Target", body="target body")
        await persist_knowledge(
            storage=storage, root=root, path="knowledge/target.md"
        )

        # Source K page with a wikilink + a sources frontmatter entry.
        _write_k_page(
            root,
            "knowledge/source.md",
            title="Source",
            body="Refers to [[Target]].",
            sources=["sources/a.md"],
        )
        result = await persist_knowledge(
            storage=storage, root=root, path="knowledge/source.md"
        )
        assert result.unresolved_wikilinks == 0

        # Outgoing links: source → target (resolved). ``LinkRecord``
        # stores the destination by path; the resolver fills it in when
        # the target is found via the title index.
        links = await storage.links_from("knowledge:knowledge/source.md")
        assert any(link.dst_path == "knowledge/target.md" for link in links)

        # Provenance edge to sources/a.md (unresolved since no D-layer doc
        # exists yet, but the edge is recorded).
        prov = await storage.provenance_from("knowledge:knowledge/source.md")
        prov_keys = {p.source_path_key for p in prov}
        assert "sources/a.md" in prov_keys
    finally:
        await storage.close()


async def test_persist_knowledge_embed_retry_skip_falls_back_to_pending(
    tmp_path: Path,
) -> None:
    """A flaky provider that exhausts retries leaves chunks pending.

    The function does NOT raise — fix is durable, embedding pending.
    """
    root, storage = await _new_storage_in_base(tmp_path)
    try:
        _write_k_page(root, "knowledge/flaky.md", title="Flaky", body="charlie delta.")
        version_id = await register_text_version(storage)
        # ALWAYS-failing embedder — every call raises ProviderError.
        embedder = FlakyEmbedder(raise_on_calls=set(range(50)))

        result = await persist_knowledge(
            storage=storage,
            root=root,
            path="knowledge/flaky.md",
            embedder=embedder,
            embedding_model="fake",
            text_version_id=version_id,
            retries=1,  # 2 attempts then skip
            backoff_seconds=0.0,
        )

        # Chunks landed + FTS landed; embedding deferred (pending count > 0).
        assert result.chunks_embedded == 0
        assert result.chunks_pending_embedding > 0
        # Doc is searchable via FTS even without vectors.
        fts_hits = await storage.fts_search("charlie", limit=5)
        assert any(
            h.doc_id == "knowledge:knowledge/flaky.md" for h in fts_hits
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_persist_knowledge_clamps_status_to_none(tmp_path: Path) -> None:
    """Even if a K page has a ``status:`` in frontmatter, it gets clamped.

    Status is wisdom-only; K-layer reads must always store status=None
    so retrieval filters never see drafted/archived K pages by accident.
    """
    root, storage = await _new_storage_in_base(tmp_path)
    try:
        # Write a K page with a stray status field — should be ignored.
        abs_path = root / "knowledge" / "status-stray.md"
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(
            "---\ntitle: Stray\nstatus: draft\n---\n\nbody\n", encoding="utf-8"
        )
        await persist_knowledge(
            storage=storage, root=root, path="knowledge/status-stray.md"
        )
        doc = await storage.get_document("knowledge:knowledge/status-stray.md")
        assert doc is not None
        assert doc.status is None
    finally:
        await storage.close()


async def test_persist_knowledge_rejects_path_escape(tmp_path: Path) -> None:
    """The shared persist leg refuses a path that escapes the base.

    ``persist_knowledge`` delegates to ``_persist_layered_page``, which
    resolves ``root / path`` and reads it via ``parse_any``. A path with a
    ``..`` segment must be rejected before that read so a malformed caller
    can't index a file from outside the base into storage.
    """
    root, storage = await _new_storage_in_base(tmp_path)
    try:
        with pytest.raises(ValueError, match="outside base"):
            await persist_knowledge(
                storage=storage,
                root=root,
                path="knowledge/../../escaped.md",
            )
    finally:
        await storage.close()
