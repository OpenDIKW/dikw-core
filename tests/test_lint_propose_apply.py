"""End-to-end test: api.lint_propose → api.lint_apply → re-lint clean.

Parametrised over sqlite + postgres so the full propose/apply loop
exercises both adapters. The flow:

1. Build a minimal wiki with a known target page (``Foo Bar``) and a
   source page that links to a broken alias ``[[foo  bar]]``.
2. Call ``api.lint`` and confirm the broken_wikilink issue surfaces.
3. Call ``api.lint_propose`` and confirm a fix proposal is produced.
4. Call ``api.lint_apply`` and confirm the file is rewritten.
5. Re-run ``api.lint`` and confirm the issue is gone.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.domains.knowledge.lint_fix import FixProposalReport
from dikw_core.schemas import DocumentRecord, Layer

from .fakes import init_test_base


def _wiki_doc_id(path: str) -> str:
    from dikw_core.domains.data.path_norm import normalize_path

    return f"knowledge:{normalize_path(path)}"


@pytest.fixture
def populated_wiki(tmp_path: Path) -> Path:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    # Seed two pages: a target (Foo Bar) and a source with a broken alias.
    target = wiki / "knowledge/concepts/foo-bar.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "---\nid: K-foobar\ntype: concept\ntitle: Foo Bar\n"
        "created: 2026-05-09T00:00:00+00:00\n"
        "updated: 2026-05-09T00:00:00+00:00\n---\n\n"
        "# Foo Bar\n\nbody\n",
        encoding="utf-8",
    )
    src = wiki / "knowledge/concepts/source.md"
    src.write_text(
        "---\nid: K-source\ntype: concept\ntitle: Source\n"
        "created: 2026-05-09T00:00:00+00:00\n"
        "updated: 2026-05-09T00:00:00+00:00\n---\n\n"
        "# Source\n\nSee [[fooo bar]] for context.\n",
        encoding="utf-8",
    )
    return wiki


@pytest.mark.asyncio
async def test_propose_apply_relint_clean_e2e(populated_wiki: Path) -> None:
    """Smoke the full propose/apply flow against an in-process SQLite wiki."""
    # Register the seeded pages with storage so lint can see them.
    _cfg, _root, storage = await api._with_storage(populated_wiki)
    try:
        for path, title in [
            ("knowledge/concepts/foo-bar.md", "Foo Bar"),
            ("knowledge/concepts/source.md", "Source"),
        ]:
            await storage.upsert_document(
                DocumentRecord(
                    doc_id=_wiki_doc_id(path), path=path, title=title,
                    hash=f"hash-{path}", mtime=0.0,
                    layer=Layer.KNOWLEDGE, active=True,
                )
            )
    finally:
        await storage.close()

    # 1. lint sees the broken link.
    pre = await api.lint(populated_wiki)
    assert any(i.kind == "broken_wikilink" for i in pre.issues), (
        f"expected broken_wikilink in lint output, got {pre.issues!r}"
    )

    # 2. propose -> 1 proposal.
    proposal_report = await api.lint_propose(
        populated_wiki, rule="broken_wikilink", limit=10
    )
    assert isinstance(proposal_report, FixProposalReport)
    assert len(proposal_report.proposals) == 1
    proposal = proposal_report.proposals[0]
    assert proposal.source == "heuristic"
    assert proposal.operations[0].kind == "update_page"

    # 3. apply.
    apply_report = await api.lint_apply(
        populated_wiki, proposal_report=proposal_report
    )
    assert len(apply_report.applied) == 1
    assert apply_report.skipped == []
    assert "knowledge/concepts/source.md" in apply_report.knowledge_paths_changed

    # 4. file content updated on disk.
    rewritten = (populated_wiki / "knowledge/concepts/source.md").read_text(
        encoding="utf-8"
    )
    assert "[[Foo Bar]]" in rewritten
    assert "[[foo  bar]]" not in rewritten

    # 5. re-lint: broken_wikilink gone.
    post = await api.lint(populated_wiki)
    assert not any(i.kind == "broken_wikilink" for i in post.issues), (
        f"expected no broken_wikilink after apply, got {post.issues!r}"
    )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("DIKW_TEST_POSTGRES_DSN"),
    reason="Postgres adapter test requires DIKW_TEST_POSTGRES_DSN",
)
async def test_propose_apply_against_postgres(populated_wiki: Path) -> None:
    """Mirror of the SQLite e2e but routed through Postgres so the
    apply path's ``replace_links_from`` + ``deactivate_document`` are
    exercised against the real adapter (not just the SQLite stub)."""
    # Override the wiki's storage backend by patching its dikw.yml.
    cfg_path = populated_wiki / "dikw.yml"
    cfg_text = cfg_path.read_text(encoding="utf-8")
    dsn = os.environ["DIKW_TEST_POSTGRES_DSN"]
    schema = f"dikw_test_e2e_{abs(hash(str(populated_wiki))) % 10_000_000:07d}"
    pg_block = (
        f"\nstorage:\n  backend: postgres\n  dsn: {dsn}\n  schema: {schema}\n"
    )
    if "storage:" not in cfg_text:
        cfg_path.write_text(cfg_text + pg_block, encoding="utf-8")
    else:
        # Replace existing storage block.
        import re
        cfg_path.write_text(
            re.sub(r"\nstorage:.*?(?=\n\w|\Z)", pg_block, cfg_text, count=1, flags=re.DOTALL),
            encoding="utf-8",
        )

    try:
        _cfg, _root, storage = await api._with_storage(populated_wiki)
        try:
            for path, title in [
                ("knowledge/concepts/foo-bar.md", "Foo Bar"),
                ("knowledge/concepts/source.md", "Source"),
            ]:
                await storage.upsert_document(
                    DocumentRecord(
                        doc_id=_wiki_doc_id(path), path=path, title=title,
                        hash=f"hash-{path}", mtime=0.0,
                        layer=Layer.KNOWLEDGE, active=True,
                    )
                )
        finally:
            await storage.close()

        proposal_report = await api.lint_propose(
            populated_wiki, rule="broken_wikilink", limit=10
        )
        assert len(proposal_report.proposals) == 1
        apply_report = await api.lint_apply(
            populated_wiki, proposal_report=proposal_report
        )
        assert len(apply_report.applied) == 1

        rewritten = (populated_wiki / "knowledge/concepts/source.md").read_text(
            encoding="utf-8"
        )
        assert "[[Foo Bar]]" in rewritten

        post = await api.lint(populated_wiki)
        assert not any(i.kind == "broken_wikilink" for i in post.issues)
    finally:
        # Drop the test schema so re-runs don't accumulate.
        from psycopg import AsyncConnection
        conn = await AsyncConnection.connect(dsn)
        try:
            async with conn.cursor() as cur:
                await cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            await conn.commit()
        finally:
            await conn.close()
