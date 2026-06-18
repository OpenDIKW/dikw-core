"""Tests for the ``dangling_provenance`` drift lint kind (ADR-0005 PR4).

``dangling_provenance`` closes the last fs↔DB gap: a K/W page's ``sources:``
frontmatter records a **provenance** edge (page → D-source attribution), but
the target source file is gone from disk. Disk is the source of truth
(ADR-0005), so a citation whose backing file no longer exists is drift.

It is **read-only — surfaced, never auto-repaired**: the frontmatter is the
user's to edit (consistent with ADR-0001's non-cascade design and D5 of the
plan). So there is *no* fixer; ``lint propose`` reports it for human triage
exactly like ``duplicate_title``.

Disk-authoritative subtlety (the reason it checks the *file*, not the D row):
a source file present on disk but not yet ``ingest``-ed (no active D row) is
**not** dangling — the fix there is ``ingest``, not editing frontmatter. So
the detector stats the file, never the ``documents`` projection.

Spans K + W (both carry ``sources:`` provenance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from dikw_core import api
from dikw_core.domains.data.path_norm import doc_id_for
from dikw_core.domains.knowledge.lint import LintIssue, LintReport, run_lint
from dikw_core.progress import CancelToken
from dikw_core.schemas import Layer

from .fakes import init_test_base, seed_doc


@dataclass
class _ListReporter:
    """Minimal ``ProgressReporter`` for propose-loop tests."""

    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    token: CancelToken = field(default_factory=CancelToken)

    async def progress(
        self, *, phase: str, current: int = 0, total: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(("progress", {"phase": phase, "detail": detail or {}}))

    async def log(self, level: str, message: str) -> None:
        self.events.append(("log", {"level": level, "message": message}))

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        self.events.append(("partial", {"kind": kind, "payload": payload}))

    def cancel_token(self) -> CancelToken:
        return self.token


@pytest.fixture()
def base_root(tmp_path: Path) -> Path:
    root = tmp_path / "base"
    init_test_base(root)
    return root


async def _run_lint(base: Path):
    _cfg, root, storage = await api._with_storage(base)
    try:
        return await run_lint(storage, root=root)
    finally:
        await storage.close()


async def _seed_provenance(base: Path, *, doc_id: str, source_paths: list[str]) -> None:
    _cfg, _root, storage = await api._with_storage(base)
    try:
        await storage.replace_provenance_from(doc_id, source_paths)
    finally:
        await storage.close()


# ---- Detection -----------------------------------------------------------


@pytest.mark.asyncio
async def test_dangling_provenance_detects_deleted_source_k(base_root: Path) -> None:
    """A K page whose provenance edge points at a now-deleted source file →
    one ``dangling_provenance`` issue keyed on the *page* path, naming the
    missing source in the detail."""
    page = "knowledge/concepts/topic.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=page,
        body="---\nsources:\n  - sources/gone.md\n---\n# Topic\n\nbody\n",
        title="Topic",
    )
    await _seed_provenance(
        base_root, doc_id=doc_id_for(Layer.KNOWLEDGE, page),
        source_paths=["sources/gone.md"],
    )
    # No sources/gone.md on disk → dangling.

    report = await _run_lint(base_root)
    dangling = [i for i in report.issues if i.kind == "dangling_provenance"]
    assert [i.path for i in dangling] == [page]
    assert "sources/gone.md" in dangling[0].detail


@pytest.mark.asyncio
async def test_dangling_provenance_detects_deleted_source_w(base_root: Path) -> None:
    """Wisdom pages carry provenance too — the kind spans K/W."""
    page = "wisdom/holo/essay.md"
    await seed_doc(
        base_root, layer=Layer.WISDOM, path=page,
        body="---\nsources:\n  - sources/missing.md\n---\n# Essay\n\nbody\n",
        title="Essay",
    )
    await _seed_provenance(
        base_root, doc_id=doc_id_for(Layer.WISDOM, page),
        source_paths=["sources/missing.md"],
    )

    report = await _run_lint(base_root)
    assert any(
        i.kind == "dangling_provenance" and i.path == page for i in report.issues
    )


@pytest.mark.asyncio
async def test_dangling_provenance_clean_when_source_present(base_root: Path) -> None:
    """Source file on disk → zero ``dangling_provenance`` (the common case;
    pins no false-fire on a well-attributed page)."""
    page = "knowledge/concepts/topic.md"
    (base_root / "sources").mkdir(parents=True, exist_ok=True)
    (base_root / "sources/foo.md").write_text("# Foo\n\nsource body\n", encoding="utf-8")
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=page,
        body="---\nsources:\n  - sources/foo.md\n---\n# Topic\n\nbody\n",
        title="Topic",
    )
    await _seed_provenance(
        base_root, doc_id=doc_id_for(Layer.KNOWLEDGE, page),
        source_paths=["sources/foo.md"],
    )

    report = await _run_lint(base_root)
    assert not [i for i in report.issues if i.kind == "dangling_provenance"]


@pytest.mark.asyncio
async def test_dangling_provenance_no_false_positive_when_unindexed_source(
    base_root: Path,
) -> None:
    """Disk-authoritative distinction: a source file present on disk but with
    NO active D ``documents`` row (never ingested) is NOT dangling — the fix is
    ``ingest``, not editing frontmatter. Checking ``resolved=False`` (no active
    row) instead of the file would false-fire here."""
    page = "knowledge/concepts/topic.md"
    (base_root / "sources").mkdir(parents=True, exist_ok=True)
    (base_root / "sources/unindexed.md").write_text("# U\n\nbody\n", encoding="utf-8")
    # Deliberately NO seed_doc for the source → no active D row, file present.
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=page,
        body="---\nsources:\n  - sources/unindexed.md\n---\n# Topic\n\nbody\n",
        title="Topic",
    )
    await _seed_provenance(
        base_root, doc_id=doc_id_for(Layer.KNOWLEDGE, page),
        source_paths=["sources/unindexed.md"],
    )

    report = await _run_lint(base_root)
    assert not [i for i in report.issues if i.kind == "dangling_provenance"]


@pytest.mark.asyncio
async def test_dangling_provenance_clean_when_source_case_form_drifts(
    base_root: Path,
) -> None:
    """A cited source whose on-disk spelling differs from the frontmatter only
    by case / Unicode form is NOT dangling — the engine's source identity is
    the normalized path key (the key ``read_provenance`` resolves through), and
    the detector matches an active source doc via that key. Pins the no-false-
    positive contract on a case-sensitive filesystem (Codex review P2)."""
    page = "knowledge/concepts/topic.md"
    # On-disk source spelled with capitals; provenance cites the lowercase form.
    await seed_doc(
        base_root, layer=Layer.SOURCE, path="sources/Foo.md",
        body="# Foo\n\nsource body\n", title="Foo",
    )
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=page,
        body="# Topic\n\nbody\n", title="Topic",
    )
    await _seed_provenance(
        base_root, doc_id=doc_id_for(Layer.KNOWLEDGE, page),
        source_paths=["sources/foo.md"],
    )

    report = await _run_lint(base_root)
    assert not [i for i in report.issues if i.kind == "dangling_provenance"]


@pytest.mark.asyncio
async def test_dangling_provenance_suppressed_by_lint_skip(base_root: Path) -> None:
    """``lint: {skip: [dangling_provenance]}`` in frontmatter suppresses it —
    the page file exists so it can carry the annotation (unlike missing_file /
    untracked_file), and the detector already parses the frontmatter."""
    page = "knowledge/concepts/topic.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=page,
        body=(
            "---\n"
            "sources:\n  - sources/gone.md\n"
            "lint:\n  skip:\n    - dangling_provenance\n"
            "---\n# Topic\n\nbody\n"
        ),
        title="Topic",
    )
    await _seed_provenance(
        base_root, doc_id=doc_id_for(Layer.KNOWLEDGE, page),
        source_paths=["sources/gone.md"],
    )

    report = await _run_lint(base_root)
    assert not [i for i in report.issues if i.kind == "dangling_provenance"]
    assert page in report.acknowledged_leaves


@pytest.mark.asyncio
async def test_dangling_provenance_multiple_edges_sorted(base_root: Path) -> None:
    """Multiple gone sources on one page → one issue each, deterministically
    ordered by source path (so ``lint propose --limit`` is reproducible)."""
    page = "knowledge/concepts/topic.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=page,
        body="---\nsources:\n  - sources/b.md\n  - sources/a.md\n---\n# Topic\n\nbody\n",
        title="Topic",
    )
    await _seed_provenance(
        base_root, doc_id=doc_id_for(Layer.KNOWLEDGE, page),
        source_paths=["sources/b.md", "sources/a.md"],
    )

    report = await _run_lint(base_root)
    dangling = [i for i in report.issues if i.kind == "dangling_provenance"]
    assert len(dangling) == 2
    # Both keyed on the page; details name the sources in sorted order.
    assert all(i.path == page for i in dangling)
    assert "sources/a.md" in dangling[0].detail
    assert "sources/b.md" in dangling[1].detail


@pytest.mark.asyncio
async def test_dangling_provenance_partial_some_present(base_root: Path) -> None:
    """A page citing one present + one deleted source → exactly one dangling
    issue (for the gone one), not the present one."""
    page = "knowledge/concepts/topic.md"
    (base_root / "sources").mkdir(parents=True, exist_ok=True)
    (base_root / "sources/here.md").write_text("# Here\n\nbody\n", encoding="utf-8")
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=page,
        body="---\nsources:\n  - sources/here.md\n  - sources/gone.md\n---\n# T\n\nb\n",
        title="T",
    )
    await _seed_provenance(
        base_root, doc_id=doc_id_for(Layer.KNOWLEDGE, page),
        source_paths=["sources/here.md", "sources/gone.md"],
    )

    report = await _run_lint(base_root)
    dangling = [i for i in report.issues if i.kind == "dangling_provenance"]
    # Exactly the gone source is flagged; the present one produces no issue at
    # all (count pins it) and its path appears in no dangling detail.
    assert len(dangling) == 1
    assert dangling[0].path == page
    assert "sources/gone.md" in dangling[0].detail
    assert all("here.md" not in i.detail for i in dangling)


@pytest.mark.parametrize("escaping", ["../outside.md", "/etc/passwd"])
@pytest.mark.asyncio
async def test_dangling_provenance_out_of_base_path_is_dangling(
    base_root: Path, escaping: str
) -> None:
    """A provenance edge whose path escapes the base (absolute or ``../``)
    can never resolve to an in-base source → dangling. The escaping target's
    *content* is never ``is_file``-stat-ed (containment short-circuits); an
    escaping path is dangling regardless of whether something exists there."""
    page = "knowledge/concepts/topic.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=page,
        body=f"---\nsources:\n  - {escaping}\n---\n# Topic\n\nbody\n",
        title="Topic",
    )
    # Create the relative-escape target so a naive ``is_file`` (without
    # containment) would wrongly call it present. (The absolute /etc/passwd
    # case relies on the existing system file but is still rejected by
    # containment, not by absence.)
    if escaping == "../outside.md":
        (base_root.parent / "outside.md").write_text("x\n", encoding="utf-8")
    await _seed_provenance(
        base_root, doc_id=doc_id_for(Layer.KNOWLEDGE, page),
        source_paths=[escaping],
    )

    report = await _run_lint(base_root)
    assert any(
        i.kind == "dangling_provenance" and i.path == page for i in report.issues
    )


@pytest.mark.asyncio
async def test_dangling_provenance_normalizes_backslash_separators(
    base_root: Path,
) -> None:
    """A Windows-style backslash-spelled source entry whose real file exists at
    the forward-slash path is NOT dangling — the detector normalizes separators
    before the disk join, so the verdict is identical on every platform (no
    Windows-clean / Linux-dirty divergence)."""
    page = "knowledge/concepts/topic.md"
    (base_root / "sources").mkdir(parents=True, exist_ok=True)
    (base_root / "sources/foo.md").write_text("# Foo\n\nbody\n", encoding="utf-8")
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=page,
        body="# Topic\n\nbody\n", title="Topic",
    )
    # Raw frontmatter spelling uses a backslash; the file is at sources/foo.md.
    await _seed_provenance(
        base_root, doc_id=doc_id_for(Layer.KNOWLEDGE, page),
        source_paths=["sources\\foo.md"],
    )

    report = await _run_lint(base_root)
    assert not [i for i in report.issues if i.kind == "dangling_provenance"]


@pytest.mark.asyncio
async def test_dangling_provenance_independent_of_missing_provenance_skip(
    base_root: Path,
) -> None:
    """Suppressing ``missing_provenance`` must not suppress
    ``dangling_provenance`` — the provenance edges are loaded for both checks,
    not gated solely on the missing_provenance branch."""
    page = "knowledge/concepts/topic.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=page,
        body=(
            "---\n"
            "sources:\n  - sources/gone.md\n"
            "lint:\n  skip:\n    - missing_provenance\n"
            "---\n# Topic\n\nbody\n"
        ),
        title="Topic",
    )
    await _seed_provenance(
        base_root, doc_id=doc_id_for(Layer.KNOWLEDGE, page),
        source_paths=["sources/gone.md"],
    )

    report = await _run_lint(base_root)
    assert any(i.kind == "dangling_provenance" for i in report.issues)
    assert not any(i.kind == "missing_provenance" for i in report.issues)


@pytest.mark.asyncio
async def test_dangling_provenance_silent_when_table_unreconciled(
    base_root: Path,
) -> None:
    """``dangling_provenance`` keys on the provenance *table* (the reconciled
    edge), not the raw frontmatter list. A page that cites a gone source but
    whose table is empty (a legacy / never-reconciled base) surfaces as
    ``missing_provenance`` first — NOT ``dangling_provenance``. Pins the
    documented precondition (reconcile, then the edge lands in the dangling
    pass) and the no-mask boundary."""
    page = "knowledge/concepts/topic.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=page,
        body="---\nsources:\n  - sources/gone.md\n---\n# Topic\n\nbody\n",
        title="Topic",
    )
    # Deliberately NO _seed_provenance → the provenance table is empty.

    report = await _run_lint(base_root)
    assert any(i.kind == "missing_provenance" for i in report.issues)
    assert not any(i.kind == "dangling_provenance" for i in report.issues)


@pytest.mark.asyncio
async def test_dangling_provenance_cofires_with_missing_provenance(
    base_root: Path,
) -> None:
    """When a page's table edge points at a gone source AND the table drifts
    from frontmatter, BOTH kinds fire independently in one pass — neither masks
    the other. Frontmatter cites ``new.md`` (drift vs the table) while the
    stored edge is the gone ``old.md`` (dangling); dangling reflects the TABLE
    edge, confirming it is table-driven."""
    page = "knowledge/concepts/topic.md"
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=page,
        body="---\nsources:\n  - sources/new.md\n---\n# Topic\n\nbody\n",
        title="Topic",
    )
    await _seed_provenance(
        base_root, doc_id=doc_id_for(Layer.KNOWLEDGE, page),
        source_paths=["sources/old.md"],  # stale row, neither file on disk
    )

    report = await _run_lint(base_root)
    assert any(i.kind == "missing_provenance" and i.path == page for i in report.issues)
    dangling = [i for i in report.issues if i.kind == "dangling_provenance"]
    assert len(dangling) == 1
    assert "sources/old.md" in dangling[0].detail


@pytest.mark.asyncio
async def test_dangling_provenance_cofires_with_missing_file(base_root: Path) -> None:
    """A deleted source whose D row is still active (``missing_file`` not yet
    applied) surfaces under BOTH kinds for the same root cause — ``missing_file``
    on the source row (purge it) AND ``dangling_provenance`` on the citing page
    (edit the frontmatter). Neither suppresses the other; disk authority, not
    the lingering row, decides dangling."""
    src = "sources/foo.md"
    page = "knowledge/concepts/topic.md"
    await seed_doc(
        base_root, layer=Layer.SOURCE, path=src, body="# Foo\n\nbody\n", title="Foo",
    )
    await seed_doc(
        base_root, layer=Layer.KNOWLEDGE, path=page,
        body="---\nsources:\n  - sources/foo.md\n---\n# Topic\n\nbody\n",
        title="Topic",
    )
    await _seed_provenance(
        base_root, doc_id=doc_id_for(Layer.KNOWLEDGE, page), source_paths=[src],
    )
    # Delete the source file but leave its (now-stale) active D row in place.
    (base_root / src).unlink()

    report = await _run_lint(base_root)
    assert any(i.kind == "missing_file" and i.path == src for i in report.issues)
    assert any(
        i.kind == "dangling_provenance" and i.path == page for i in report.issues
    )


# ---- No fixer (read-only kind) -------------------------------------------


def test_dangling_provenance_has_no_fixer() -> None:
    """Read-only kind: no entry in the fixer registry (like ``duplicate_title``)."""
    from dikw_core.domains.knowledge.lint_fixers import FIXER_REGISTRY

    assert "dangling_provenance" not in FIXER_REGISTRY


@pytest.mark.asyncio
async def test_lint_propose_skips_dangling_provenance(tmp_path: Path) -> None:
    """End-to-end: ``lint propose`` reports a dangling_provenance issue as a
    skip ("no fixer registered") against the real registry, never crashing —
    human-triage only (the read-only kind, like ``duplicate_title``)."""
    from dikw_core.domains.knowledge.lint_fix import FixerContext, run_lint_propose

    report = LintReport(
        issues=[
            LintIssue(
                kind="dangling_provenance",
                path="knowledge/concepts/topic.md",
                detail="declared source 'sources/gone.md' has no file on disk",
            )
        ]
    )
    ctx = FixerContext(
        storage=None, llm=None, embedding=None, base_root=tmp_path, all_pages=[],
    )
    proposal_report = await run_lint_propose(
        report=report,
        rule="dangling_provenance",
        limit=100,
        ctx=ctx,
        reporter=_ListReporter(),
    )

    assert proposal_report.proposals == []
    assert any(
        s["issue_kind"] == "dangling_provenance"
        and "no fixer registered" in s["reason"]
        for s in proposal_report.skipped
    )
