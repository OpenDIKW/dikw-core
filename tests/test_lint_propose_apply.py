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


@pytest.mark.asyncio
async def test_lint_apply_persist_failure_deactivates_page(
    populated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hard storage failure during lint-apply Phase 1 must deactivate the
    in-flight page and record it in ``ApplyReport.persist_errors`` instead of
    aborting the apply with a half-written-but-active page — parity with the
    synth path and with D/W."""
    from dikw_core import api_lint

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

    fail_doc_id = _wiki_doc_id("knowledge/concepts/source.md")
    original = api_lint._with_storage

    async def patched(path: object) -> object:
        cfg, root, storage = await original(path)  # type: ignore[arg-type]
        orig_rlf = storage.replace_links_from

        async def maybe_boom(doc_id: object, resolved: object) -> object:
            if doc_id == fail_doc_id:
                raise RuntimeError("simulated link reconcile outage")
            return await orig_rlf(doc_id, resolved)  # type: ignore[arg-type]

        storage.replace_links_from = maybe_boom  # type: ignore[method-assign]
        return cfg, root, storage

    monkeypatch.setattr(api_lint, "_with_storage", patched)

    # Must NOT raise — the failing page is deactivated + recorded.
    report = await api.lint_apply(populated_wiki, proposal_report=proposal_report)
    assert [e["path"] for e in report.persist_errors] == [
        "knowledge/concepts/source.md"
    ]
    assert "simulated link reconcile outage" in report.persist_errors[0]["message"]

    monkeypatch.undo()
    _cfg, _root, storage = await original(populated_wiki)  # type: ignore[misc]
    try:
        doc = await storage.get_document(fail_doc_id)
        assert doc is not None and doc.active is False, (
            "a page whose lint-apply persist failed must be deactivated"
        )
    finally:
        await storage.close()
