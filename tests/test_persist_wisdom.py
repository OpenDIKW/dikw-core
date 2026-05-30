"""Unit tests for ``persist_wisdom`` — the W-layer persist entry.

Covers the 0.4.0 W-layer persist function contract: upsert_document
(status flows from parsed frontmatter — WISDOM-only field) + chunk +
FTS + (inline-or-deferred embed) + replace_links_from +
replace_provenance_from. Sole engine caller in production code is
``api.write_wisdom_page``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dikw_core.domains.wisdom import persist_wisdom
from dikw_core.schemas import Layer, WisdomPersistResult, WisdomStatus
from dikw_core.storage.sqlite import SQLiteStorage

from .fakes import FakeEmbeddings, FlakyEmbedder, init_test_base, register_text_version


def _write_w_page(
    root: Path,
    rel_path: str,
    *,
    title: str,
    body: str,
    status: str | None = None,
) -> None:
    fm: dict[str, object] = {"title": title}
    if status is not None:
        fm["status"] = status
    abs_path = root / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    text = "---\n" + yaml.safe_dump(fm, sort_keys=False) + "---\n\n" + body
    abs_path.write_text(text, encoding="utf-8")


async def _new_storage_in_base(tmp_path: Path) -> tuple[Path, SQLiteStorage]:
    root = tmp_path / "base"
    init_test_base(root, description="persist_wisdom test base")
    storage = SQLiteStorage(root / ".dikw" / "index.sqlite")
    await storage.connect()
    await storage.migrate()
    return root, storage


async def test_persist_wisdom_preserves_status_from_frontmatter(
    tmp_path: Path,
) -> None:
    """W-layer status (draft/published/archived) flows through from frontmatter."""
    root, storage = await _new_storage_in_base(tmp_path)
    try:
        _write_w_page(
            root,
            "wisdom/alice/published.md",
            title="Published",
            body="published body",
            status="published",
        )
        result = await persist_wisdom(
            storage=storage,
            root=root,
            path="wisdom/alice/published.md",
        )
        assert isinstance(result, WisdomPersistResult)
        assert result.resolved_title == "Published"
        doc = await storage.get_document("wisdom:wisdom/alice/published.md")
        assert doc is not None
        assert doc.layer is Layer.WISDOM
        assert doc.status is WisdomStatus.PUBLISHED
    finally:
        await storage.close()


async def test_persist_wisdom_inline_embed_when_provider_configured(
    tmp_path: Path,
) -> None:
    """With embedder + version_id, W chunks land in the vec table inline."""
    root, storage = await _new_storage_in_base(tmp_path)
    try:
        _write_w_page(
            root, "wisdom/bob/note.md", title="Bob's Note", body="golf hotel."
        )
        version_id = await register_text_version(storage)
        embedder = FakeEmbeddings()

        result = await persist_wisdom(
            storage=storage,
            root=root,
            path="wisdom/bob/note.md",
            embedder=embedder,
            embedding_model="fake",
            text_version_id=version_id,
        )
        assert result.chunks_embedded > 0
        assert result.chunks_pending_embedding == 0
        vecs = await storage.get_chunk_embeddings(
            result.chunk_ids, version_id=version_id
        )
        assert len(vecs) == len(result.chunk_ids)
    finally:
        await storage.close()


async def test_persist_wisdom_defers_when_no_embedder(tmp_path: Path) -> None:
    """``--no-embed`` write path: chunks + FTS land, vectors deferred."""
    root, storage = await _new_storage_in_base(tmp_path)
    try:
        _write_w_page(
            root, "wisdom/carol/draft.md", title="Draft", body="india juliet"
        )
        version_id = await register_text_version(storage)

        result = await persist_wisdom(
            storage=storage,
            root=root,
            path="wisdom/carol/draft.md",
            embedder=None,
            text_version_id=None,
        )
        assert result.chunks_embedded == 0
        assert result.chunks_pending_embedding > 0
        # Resume scan would pick these chunks up on the next ingest.
        missing = await storage.list_chunks_missing_embedding(version_id=version_id)
        assert set(result.chunk_ids).issubset({c.chunk_id for c in missing})
    finally:
        await storage.close()


async def test_persist_wisdom_embed_retry_skip_falls_back_to_pending(
    tmp_path: Path,
) -> None:
    """Flaky embedder → wisdom write survives, embedding deferred."""
    root, storage = await _new_storage_in_base(tmp_path)
    try:
        _write_w_page(
            root, "wisdom/dan/resilient.md", title="Resilient", body="kilo lima."
        )
        version_id = await register_text_version(storage)
        embedder = FlakyEmbedder(raise_on_calls=set(range(50)))

        result = await persist_wisdom(
            storage=storage,
            root=root,
            path="wisdom/dan/resilient.md",
            embedder=embedder,
            embedding_model="fake",
            text_version_id=version_id,
            retries=1,
            backoff_seconds=0.0,
        )
        assert result.chunks_embedded == 0
        assert result.chunks_pending_embedding > 0
        # Doc remains FTS-searchable.
        fts_hits = await storage.fts_search("kilo", limit=5)
        assert any(
            h.doc_id == "wisdom:wisdom/dan/resilient.md" for h in fts_hits
        )
    finally:
        await storage.close()


async def test_persist_wisdom_rejects_path_escape(tmp_path: Path) -> None:
    """The shared persist leg refuses a logical path that escapes the base.

    ``persist_wisdom`` delegates to ``_persist_layered_page``, which
    resolves ``root / path`` and reads it via ``parse_any``. A path with a
    ``..`` segment must be rejected before that read so a malformed caller
    can't index a file from outside the base into storage.
    """
    root, storage = await _new_storage_in_base(tmp_path)
    try:
        with pytest.raises(ValueError, match="outside base"):
            await persist_wisdom(
                storage=storage,
                root=root,
                path="wisdom/../../escaped.md",
            )
    finally:
        await storage.close()
