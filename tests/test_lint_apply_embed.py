"""Tests for the 0.4.0 ``lint apply`` inline-embed behavior.

When an embedder + ``text_version_id`` are wired through
``run_lint_apply``, Phase 1 re-chunks every changed page via
``persist_knowledge`` with the embedder attached, so vectors land in
the per-version vec table on return — the fixed page is retrievable
immediately. Without an embedder, the chunks remain pending and the
next ``dikw client ingest``'s missing-embedding resume scan
reconciles them.

Embed failure (transient ``ProviderError``) must NOT abort the apply
pipeline: the per-batch retry-skip inside ``persist_knowledge``
leaves failing batches pending, surfaced via
``ApplyReport.chunks_pending_embedding``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dikw_core.domains.knowledge.lint_fix import (
    FixOperation,
    FixProposal,
    FixProposalReport,
    file_sha256,
    run_lint_apply,
)
from dikw_core.schemas import DocumentRecord, Layer
from dikw_core.storage.sqlite import SQLiteStorage

from .fakes import FakeEmbeddings, FlakyEmbedder, init_test_base, register_text_version


@dataclass
class _NullReporter:
    token: Any = None

    async def progress(self, **_: Any) -> None:
        return None

    async def log(self, level: str, message: str) -> None:
        return None

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        return None

    def cancel_token(self) -> Any:
        from dikw_core.progress import CancelToken
        if self.token is None:
            self.token = CancelToken()
        return self.token


def _wiki_doc_id(path: str) -> str:
    from dikw_core.domains.data.path_norm import doc_id_for
    return doc_id_for(Layer.KNOWLEDGE, path)


async def _new_storage_in_base(tmp_path: Path) -> tuple[Path, SQLiteStorage]:
    root = tmp_path / "base"
    init_test_base(root, description="lint apply embed test base")
    storage = SQLiteStorage(root / ".dikw" / "index.sqlite")
    await storage.connect()
    await storage.migrate()
    return root, storage


async def _seed_page(
    *,
    storage: SQLiteStorage,
    base_root: Path,
    path: str,
    title: str,
    body: str,
) -> str:
    abs_path = base_root / path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(f"---\ntitle: {title}\n---\n\n{body}", encoding="utf-8")
    doc_id = _wiki_doc_id(path)
    await storage.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            path=path,
            title=title,
            hash=f"hash-{path}",
            mtime=0.0,
            layer=Layer.KNOWLEDGE,
            active=True,
        )
    )
    return doc_id


def _update_proposal(*, path: str, new_body: str, expected_hash: str) -> FixProposal:
    return FixProposal(
        proposal_id="p1",
        issue_kind="broken_wikilink",
        issue_path=path,
        issue_detail="rewrite",
        issue_line=3,
        operations=[
            FixOperation(
                kind="update_page",
                path=path,
                new_frontmatter={"title": "Source"},
                new_body=new_body,
                expected_hash=expected_hash,
            )
        ],
        rationale="test fixture",
        source="heuristic",
    )


@pytest.mark.asyncio
async def test_lint_apply_inline_embeds_when_embedder_configured(tmp_path: Path) -> None:
    """With an embedder + text_version_id, rebuilt chunks land in the
    vec table inline — ``chunks_embedded`` reflects the count and
    ``chunks_pending_embedding`` stays zero.
    """
    base_root, storage = await _new_storage_in_base(tmp_path)
    try:
        await _seed_page(
            storage=storage,
            base_root=base_root,
            path="knowledge/concepts/source.md",
            title="Source",
            body="# Source\n\nSee [[foo  bar]] for context.\n",
        )
        abs_src = base_root / "knowledge/concepts/source.md"
        expected_hash = file_sha256(abs_src)
        version_id = await register_text_version(storage)
        embedder = FakeEmbeddings()

        report = await run_lint_apply(
            proposal_report=FixProposalReport(
                proposals=[_update_proposal(
                    path="knowledge/concepts/source.md",
                    new_body="# Source\n\nSee [[Foo Bar]] for context.\n",
                    expected_hash=expected_hash,
                )]
            ),
            storage=storage,
            base_root=base_root,
            reporter=_NullReporter(),
            embedder=embedder,
            embedding_model="fake",
            text_version_id=version_id,
        )

        assert report.chunks_embedded > 0
        assert report.chunks_pending_embedding == 0

        # Vectors landed in the per-version vec table.
        doc_id = _wiki_doc_id("knowledge/concepts/source.md")
        chunks = await storage.list_chunks(doc_id)
        chunk_ids = [c.chunk_id for c in chunks]
        vecs = await storage.get_chunk_embeddings(chunk_ids, version_id=version_id)
        assert len(vecs) == len(chunk_ids)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_lint_apply_defers_embedding_without_embedder(tmp_path: Path) -> None:
    """Apply without an embedder records chunks_pending_embedding > 0 so
    the user knows the resume scan needs to run.
    """
    base_root, storage = await _new_storage_in_base(tmp_path)
    try:
        await _seed_page(
            storage=storage,
            base_root=base_root,
            path="knowledge/concepts/source.md",
            title="Source",
            body="# Source\n\nSee [[foo  bar]] for context.\n",
        )
        abs_src = base_root / "knowledge/concepts/source.md"
        expected_hash = file_sha256(abs_src)

        report = await run_lint_apply(
            proposal_report=FixProposalReport(
                proposals=[_update_proposal(
                    path="knowledge/concepts/source.md",
                    new_body="# Source\n\nSee [[Foo Bar]] for context.\n",
                    expected_hash=expected_hash,
                )]
            ),
            storage=storage,
            base_root=base_root,
            reporter=_NullReporter(),
            embedder=None,
        )

        assert report.chunks_embedded == 0
        assert report.chunks_pending_embedding > 0
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_lint_apply_survives_provider_error_via_retry_skip(
    tmp_path: Path,
) -> None:
    """A flaky embedder must not abort apply — the per-batch retry-skip
    inside ``persist_knowledge`` swallows ``ProviderError`` and the
    fix still lands. Chunks fall under ``chunks_pending_embedding``
    so the resume scan picks them up later.
    """
    base_root, storage = await _new_storage_in_base(tmp_path)
    try:
        await _seed_page(
            storage=storage,
            base_root=base_root,
            path="knowledge/concepts/source.md",
            title="Source",
            body="# Source\n\nSee [[foo  bar]] for context.\n",
        )
        abs_src = base_root / "knowledge/concepts/source.md"
        expected_hash = file_sha256(abs_src)
        version_id = await register_text_version(storage)
        embedder = FlakyEmbedder(raise_on_calls=set(range(50)))

        report = await run_lint_apply(
            proposal_report=FixProposalReport(
                proposals=[_update_proposal(
                    path="knowledge/concepts/source.md",
                    new_body="# Source\n\nSee [[Foo Bar]] for context.\n",
                    expected_hash=expected_hash,
                )]
            ),
            storage=storage,
            base_root=base_root,
            reporter=_NullReporter(),
            embedder=embedder,
            embedding_model="fake",
            text_version_id=version_id,
            embedding_error_retries=1,
            embedding_error_retry_backoff_seconds=0.0,
        )

        # Apply succeeded — file was rewritten + the op landed.
        assert len(report.applied) == 1
        rewritten = abs_src.read_text(encoding="utf-8")
        assert "[[Foo Bar]]" in rewritten

        # All chunks are pending embedding — retry-skip absorbed the error.
        assert report.chunks_embedded == 0
        assert report.chunks_pending_embedding > 0
    finally:
        await storage.close()
