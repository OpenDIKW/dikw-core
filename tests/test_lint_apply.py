"""Tests for ``run_lint_apply`` — the executor that mutates ``knowledge/``
based on a :class:`FixProposalReport`.

PR1 design: apply only touches disk + outgoing-link reconciliation. It
intentionally skips re-chunking / re-embedding so apply stays
provider-free; a follow-up ``dikw client ingest`` reconciles ``doc.hash``
and chunks/embeddings on the next pass.

Storage-touching tests are parametrised over the ``parametrized_storage``
fixture so they auto-run against Postgres when ``DIKW_TEST_POSTGRES_DSN``
is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dikw_core.domains.knowledge.lint_fix import (
    ApplyReport,
    FixOperation,
    FixProposal,
    FixProposalReport,
    file_sha256,
    run_lint_apply,
)
from dikw_core.schemas import DocumentRecord, Layer, LinkType
from dikw_core.storage.base import Storage


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


def test_build_page_from_op_heals_scalar_list_frontmatter() -> None:
    """Fixers pass the on-disk frontmatter through ``op.new_frontmatter``
    verbatim, and apply REWRITES the page from this value — so a hand-written
    scalar (``sources: foo.md``) must neither be iterated char-by-char into
    ``['f','o','o',…]`` garbage (the old bug) nor collapsed to ``[]`` (silent
    data loss: ``sources:`` feeds ``replace_provenance_from``, and the
    missing_provenance fixer backfills FROM frontmatter, so a cleared scalar
    would be unrecoverable). It heals to a one-item list."""
    from dikw_core.domains.knowledge.lint_fix import _build_page_from_op

    op = FixOperation(
        kind="update_page",
        path="knowledge/concept/x.md",
        new_frontmatter={
            "title": "X",
            "sources": "sources/foo.md",  # scalar, not a list
            "tags": "alpha",  # scalar, not a list
        },
        new_body="A body.",
    )
    page = _build_page_from_op(op)
    assert page.sources == ["sources/foo.md"]
    assert page.tags == ["alpha"]


def test_build_page_from_op_keeps_well_formed_lists() -> None:
    from dikw_core.domains.knowledge.lint_fix import _build_page_from_op

    op = FixOperation(
        kind="update_page",
        path="knowledge/concept/x.md",
        new_frontmatter={
            "title": "X",
            "sources": ["sources/foo.md", "sources/bar.md"],
            "tags": ["alpha", "beta"],
        },
        new_body="A body.",
    )
    page = _build_page_from_op(op)
    assert page.sources == ["sources/foo.md", "sources/bar.md"]
    assert page.tags == ["alpha", "beta"]


def test_build_page_from_op_stringifies_numeric_list_entries() -> None:
    """A bare year tag (``tags: [2024, llm]``) parses as an int in YAML; the
    rewrite path must preserve it as ``"2024"`` rather than dropping it (the
    old ``list(...)`` kept it, so dropping would be a silent regression on a
    page the fixer was only supposed to backlink). Truly shapeless values
    (dict / None / bool) still drop."""
    from dikw_core.domains.knowledge.lint_fix import _build_page_from_op

    op = FixOperation(
        kind="update_page",
        path="knowledge/concept/x.md",
        new_frontmatter={
            "title": "X",
            "sources": None,
            "tags": [2024, "llm", True, {"bad": 1}],
        },
        new_body="A body.",
    )
    page = _build_page_from_op(op)
    assert page.sources == []
    assert page.tags == ["2024", "llm"]


async def _seed_page(
    *, storage: Storage, base_root: Path,
    path: str, title: str, body: str,
) -> str:
    """Write a page on disk + register a DocumentRecord. Returns the doc_id."""
    abs_path = base_root / path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    body_text = f"---\ntitle: {title}\n---\n\n{body}"
    abs_path.write_text(body_text, encoding="utf-8")
    doc_id = _wiki_doc_id(path)
    await storage.upsert_document(
        DocumentRecord(
            doc_id=doc_id, path=path, title=title,
            hash=f"hash-{path}", mtime=0.0,
            layer=Layer.KNOWLEDGE, active=True,
        )
    )
    return doc_id


@pytest.fixture
def base_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.mark.asyncio
async def test_update_page_writes_new_body_and_reconciles_links(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    storage = parametrized_storage
    target_doc_id = await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/concepts/foo-bar.md", title="Foo Bar",
        body="# Foo Bar\nbody\n",
    )
    src_doc_id = await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/concepts/source.md", title="Source",
        body="# Source\n\nSee [[foo  bar]] for context.\n",
    )
    abs_src = base_root / "knowledge/concepts/source.md"
    expected_hash = file_sha256(abs_src)

    proposal = FixProposal(
        proposal_id="p1",
        issue_kind="broken_wikilink",
        issue_path="knowledge/concepts/source.md",
        issue_detail="[[foo  bar]] has no matching knowledge page",
        issue_line=3,
        operations=[
            FixOperation(
                kind="update_page",
                path="knowledge/concepts/source.md",
                new_frontmatter={"title": "Source"},
                new_body="# Source\n\nSee [[Foo Bar]] for context.\n",
                expected_hash=expected_hash,
            )
        ],
        rationale="fuzzy match",
        source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )

    assert isinstance(report, ApplyReport)
    assert len(report.applied) == 1
    assert report.skipped == []
    assert "knowledge/concepts/source.md" in report.knowledge_paths_changed

    # File was rewritten with new body.
    rewritten = abs_src.read_text(encoding="utf-8")
    assert "[[Foo Bar]]" in rewritten
    assert "[[foo  bar]]" not in rewritten

    # Links table: outgoing edge from source → foo-bar should now exist.
    links = await storage.links_from(src_doc_id)
    wikilinks = [link_row for link_row in links if link_row.link_type is LinkType.WIKILINK]
    assert len(wikilinks) == 1
    assert wikilinks[0].dst_path == "knowledge/concepts/foo-bar.md"
    _ = target_doc_id  # held alive for the upsert side-effect


@pytest.mark.asyncio
async def test_hash_mismatch_skips_op_and_does_not_write_file(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    storage = parametrized_storage
    await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/source.md", title="Source",
        body="# Source\n\nSee [[foo]] here.\n",
    )
    abs_src = base_root / "knowledge/source.md"
    body_before = abs_src.read_text(encoding="utf-8")

    proposal = FixProposal(
        proposal_id="p1",
        issue_kind="broken_wikilink",
        issue_path="knowledge/source.md",
        issue_detail="[[foo]] has no matching knowledge page",
        issue_line=3,
        operations=[
            FixOperation(
                kind="update_page",
                path="knowledge/source.md",
                new_frontmatter={"title": "Source"},
                new_body="# Source\n\nDifferent body\n",
                # Wrong expected_hash — simulates concurrent edit.
                expected_hash="0" * 64,
            )
        ],
        rationale="r", source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )

    assert report.applied == []
    assert len(report.skipped) == 1
    assert "hash" in report.skipped[0]["reason"].lower()
    # File untouched.
    assert abs_src.read_text(encoding="utf-8") == body_before


@pytest.mark.asyncio
async def test_create_page_writes_new_file(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    storage = parametrized_storage
    new_path = "knowledge/concepts/brand-new.md"
    abs_new = base_root / new_path
    assert not abs_new.exists()

    proposal = FixProposal(
        proposal_id="p1",
        issue_kind="broken_wikilink",
        issue_path="knowledge/source.md",
        issue_detail="stub",
        operations=[
            FixOperation(
                kind="create_page",
                path=new_path,
                new_frontmatter={
                    "id": "K-newpage",
                    "type": "concept",
                    "title": "Brand New",
                    "created": "2026-05-09T00:00:00+00:00",
                    "updated": "2026-05-09T00:00:00+00:00",
                },
                new_body="# Brand New\n\nStub page.\n",
                expected_hash=None,
            )
        ],
        rationale="r", source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )

    assert len(report.applied) == 1
    assert report.skipped == []
    assert abs_new.is_file()
    content = abs_new.read_text(encoding="utf-8")
    assert "Brand New" in content


@pytest.mark.asyncio
async def test_create_page_skipped_when_file_already_exists(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    storage = parametrized_storage
    await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/concepts/exists.md", title="Exists",
        body="# Exists\n",
    )

    proposal = FixProposal(
        proposal_id="p1",
        issue_kind="broken_wikilink",
        issue_path="knowledge/source.md",
        issue_detail="stub",
        operations=[
            FixOperation(
                kind="create_page",
                path="knowledge/concepts/exists.md",
                new_frontmatter={"title": "Exists"},
                new_body="# Exists\nWill not overwrite\n",
                expected_hash=None,
            )
        ],
        rationale="r", source="heuristic",
    )
    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )
    assert report.applied == []
    assert len(report.skipped) == 1
    assert "exist" in report.skipped[0]["reason"].lower()


@pytest.mark.asyncio
async def test_delete_page_moves_to_trash_and_purges_storage(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    """delete_page is a soft-delete + hard-purge: the on-disk file moves
    to ``<base>/trash/knowledge/<original-rel-path>`` (recoverable by hand)
    while storage clears the doc + dependent rows entirely (no ghost
    records left behind by the old ``deactivate_document`` flow)."""
    storage = parametrized_storage
    doc_id = await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/concepts/dead.md", title="Dead",
        body="# Dead\n",
    )
    abs_dead = base_root / "knowledge/concepts/dead.md"
    expected_hash = file_sha256(abs_dead)

    proposal = FixProposal(
        proposal_id="p1",
        issue_kind="duplicate_title",
        issue_path="knowledge/concepts/dead.md",
        issue_detail="dup",
        operations=[
            FixOperation(
                kind="delete_page",
                path="knowledge/concepts/dead.md",
                expected_hash=expected_hash,
            )
        ],
        rationale="r", source="heuristic",
    )
    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )
    assert len(report.applied) == 1
    # Original wiki path is empty …
    assert not abs_dead.exists()
    # … the file is now under base/trash/knowledge/, preserving its rel path.
    trash_path = base_root / "trash/knowledge/concepts/dead.md"
    assert trash_path.is_file(), (
        "delete_page must move the page to <base>/trash/knowledge/<rel-path>"
    )
    # The trashed file carries a ``trashed:`` frontmatter block so users
    # auditing the trash can tell which fixer dropped it.
    import frontmatter
    post = frontmatter.loads(trash_path.read_text(encoding="utf-8"))
    trashed = post.metadata.get("trashed")
    assert isinstance(trashed, dict)
    assert trashed.get("proposal_id") == "p1"
    assert trashed.get("reason") == "duplicate_title"
    assert isinstance(trashed.get("at"), str) and trashed["at"]

    # Storage rows are fully purged — not just active=False.
    assert await storage.get_document(doc_id) is None
    all_docs = list(await storage.list_documents(layer=Layer.KNOWLEDGE, active=None))
    assert all(d.doc_id != doc_id for d in all_docs)


@pytest.mark.asyncio
async def test_move_to_trash_collision_does_not_overwrite(
    tmp_path: Path,
) -> None:
    """Two trashes of the same path within the same second must not
    overwrite each other. Regression for codex R2-2: a
    second-resolution timestamp suffix collided when called twice in a
    tight loop, silently losing the earlier soft-deleted copy."""
    from dikw_core.domains.knowledge.lint_fix import _move_to_trash

    base_root = tmp_path
    rel = "knowledge/concepts/twice.md"
    src1 = base_root / rel
    src1.parent.mkdir(parents=True, exist_ok=True)
    src1.write_text(
        "---\ntitle: First\n---\nfirst version\n", encoding="utf-8",
    )
    dest1 = _move_to_trash(
        base_root=base_root, src_abs=src1, rel_path=rel,
        reason="duplicate_title", proposal_id="p1",
    )
    # Recreate the file at the same path and re-trash immediately —
    # the timestamp will collide on the second-resolution suffix.
    src2 = base_root / rel
    src2.write_text(
        "---\ntitle: Second\n---\nsecond version\n", encoding="utf-8",
    )
    dest2 = _move_to_trash(
        base_root=base_root, src_abs=src2, rel_path=rel,
        reason="orphan_page", proposal_id="p2",
    )
    # Both files survive in trash with distinct names.
    assert dest1 != dest2
    assert dest1.is_file()
    assert dest2.is_file()
    import frontmatter
    assert frontmatter.loads(dest1.read_text(encoding="utf-8")).content.strip() \
        == "first version"
    assert frontmatter.loads(dest2.read_text(encoding="utf-8")).content.strip() \
        == "second version"


@pytest.mark.asyncio
async def test_move_to_trash_partial_write_leaves_no_dest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the trash write fails partway (disk full, short write), the
    function must not leave a partial file at the visible ``dest``
    path. Regression for codex R3-1: previously ``dest.write_text``
    wrote directly to the final path, so a mid-write failure left a
    truncated file in ``trash/`` AND the src still in ``knowledge/``.

    The two-stage write (tmp → atomic replace) means a failed write
    only leaves a ``.tmp`` to clean up; the visible ``dest`` is never
    materialised until the rename succeeds."""
    from dikw_core.domains.knowledge import lint_fix
    from dikw_core.domains.knowledge.lint_fix import _move_to_trash

    base_root = tmp_path
    rel = "knowledge/concepts/half-written.md"
    src = base_root / rel
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        "---\ntitle: HalfWritten\n---\nbody\n", encoding="utf-8",
    )

    original_write_text = Path.write_text

    def _fake_write_text(self: Path, *args: object, **kwargs: object) -> None:
        # Refuse only when writing into the trash subtree; let src
        # writes and any other tmp paths outside trash succeed.
        if str(self).startswith(str(base_root / "trash")):
            raise OSError("simulated disk full")
        return original_write_text(self, *args, **kwargs)  # type: ignore[no-any-return]

    monkeypatch.setattr(lint_fix.Path, "write_text", _fake_write_text)

    with pytest.raises(OSError, match="disk full"):
        _move_to_trash(
            base_root=base_root, src_abs=src, rel_path=rel,
            reason="orphan_page", proposal_id="p-partial",
        )

    # Post-failure: src still in knowledge/, no partial dest, no leftover tmp.
    assert src.is_file()
    trash_dir = base_root / "trash" / "knowledge" / "concepts"
    if trash_dir.exists():
        leftovers = list(trash_dir.iterdir())
        assert leftovers == [], (
            f"trash/ must be empty after a failed write, got: {leftovers}"
        )


@pytest.mark.asyncio
async def test_move_to_trash_rolls_back_when_unlink_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``src_abs.unlink()`` fails after ``dest.write_text`` succeeds,
    the function must roll back by deleting the new trash copy and
    re-raising — leaving the page in exactly one place (knowledge/) so the
    next ``dikw client ingest`` re-creates the storage row from disk.

    Regression for codex R2-1: the previous flow left the file in
    BOTH ``knowledge/`` and ``trash/`` on a partial filesystem failure."""
    from dikw_core.domains.knowledge import lint_fix
    from dikw_core.domains.knowledge.lint_fix import _move_to_trash

    base_root = tmp_path
    rel = "knowledge/concepts/doomed.md"
    src = base_root / rel
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        "---\ntitle: Doomed\n---\nbody\n", encoding="utf-8",
    )

    original_unlink = Path.unlink

    def _fake_unlink(self: Path, *args: object, **kwargs: object) -> None:
        # Refuse to remove the source; allow rollback unlink of the
        # dest to proceed (dest is in trash/, src is in knowledge/).
        if self == src:
            raise OSError("simulated permission denied on src.unlink")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(lint_fix.Path, "unlink", _fake_unlink)

    with pytest.raises(OSError, match="permission denied"):
        _move_to_trash(
            base_root=base_root, src_abs=src, rel_path=rel,
            reason="orphan_page", proposal_id="p-rollback",
        )

    # Post-rollback: src survives, trash is empty for this page.
    assert src.is_file()
    trash_target = base_root / "trash" / rel
    assert not trash_target.exists()


@pytest.mark.asyncio
async def test_delete_page_op_path_outside_knowledge_is_rejected(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    """A fixer must not try to delete files outside the ``knowledge/`` tree
    by setting ``op.path = "trash/..."`` directly. The trash redirect
    is an apply-internal behaviour; the public op contract still only
    allows wiki-relative targets."""
    storage = parametrized_storage
    # Stage a real file under base/trash/ so the path resolves to something
    # that exists on disk — the sandbox check has to reject by path shape,
    # not by file-existence.
    trash_file = base_root / "trash/knowledge/sneaky.md"
    trash_file.parent.mkdir(parents=True, exist_ok=True)
    trash_file.write_text("---\ntitle: Sneaky\n---\nstub\n", encoding="utf-8")
    expected_hash = file_sha256(trash_file)

    proposal = FixProposal(
        proposal_id="p-sandbox",
        issue_kind="duplicate_title",
        issue_path="trash/knowledge/sneaky.md",
        issue_detail="should-never-apply",
        operations=[
            FixOperation(
                kind="delete_page",
                path="trash/knowledge/sneaky.md",
                expected_hash=expected_hash,
            )
        ],
        rationale="r", source="heuristic",
    )
    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )
    assert report.applied == []
    assert len(report.skipped) == 1
    assert "outside knowledge/" in report.skipped[0]["reason"]
    # File untouched.
    assert trash_file.is_file()


@pytest.mark.asyncio
async def test_pick_filter_applies_only_picked_proposals(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    storage = parametrized_storage
    await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/foo.md", title="Foo",
        body="# Foo\n",
    )
    abs_foo = base_root / "knowledge/foo.md"
    foo_hash = file_sha256(abs_foo)

    await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/bar.md", title="Bar",
        body="# Bar\n",
    )
    abs_bar = base_root / "knowledge/bar.md"
    bar_hash = file_sha256(abs_bar)
    bar_body_before = abs_bar.read_text(encoding="utf-8")

    p_foo = FixProposal(
        proposal_id="pa",
        issue_kind="broken_wikilink",
        issue_path="knowledge/foo.md",
        issue_detail="d",
        operations=[
            FixOperation(
                kind="update_page", path="knowledge/foo.md",
                new_frontmatter={"title": "Foo"},
                new_body="# Foo\nupdated\n",
                expected_hash=foo_hash,
            )
        ],
        rationale="r", source="heuristic",
    )
    p_bar = FixProposal(
        proposal_id="pb",
        issue_kind="broken_wikilink",
        issue_path="knowledge/bar.md",
        issue_detail="d",
        operations=[
            FixOperation(
                kind="update_page", path="knowledge/bar.md",
                new_frontmatter={"title": "Bar"},
                new_body="# Bar\nupdated\n",
                expected_hash=bar_hash,
            )
        ],
        rationale="r", source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[p_foo, p_bar]),
        storage=storage, base_root=base_root,
        pick=[0],  # only the first
        reporter=_NullReporter(),
    )
    assert len(report.applied) == 1
    assert report.applied[0].path == "knowledge/foo.md"
    # Bar untouched.
    assert abs_bar.read_text(encoding="utf-8") == bar_body_before


@pytest.mark.asyncio
async def test_apply_refuses_paths_outside_base_root(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    """A malformed proposal with an absolute path or ``..`` traversal
    must never reach ``write_page`` / ``unlink`` — sandbox to knowledge/."""
    storage = parametrized_storage

    proposal = FixProposal(
        proposal_id="evil", issue_kind="broken_wikilink",
        issue_path="../../../etc/passwd",
        issue_detail="evil",
        operations=[FixOperation(
            kind="create_page",
            path="../../../etc/dikw-leak.md",
            new_frontmatter={"title": "Leak"},
            new_body="oops\n",
            expected_hash=None,
        )],
        rationale="r", source="heuristic",
    )
    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )
    assert report.applied == []
    assert len(report.skipped) == 1
    assert "outside" in report.skipped[0]["reason"].lower()
    # Confirm nothing landed on disk.
    assert not (base_root / "../../../etc/dikw-leak.md").resolve().exists()


@pytest.mark.asyncio
async def test_apply_refuses_paths_under_base_but_outside_knowledge(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    """``apply`` is a K-layer mutation contract — paths under the base
    but outside ``knowledge/`` (sources/, dikw.yml, .dikw/) must be rejected
    even though they resolve under the resolved base root."""
    storage = parametrized_storage
    # Pre-create a sources file the proposal will try to clobber.
    sources_dir = base_root / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    target = sources_dir / "victim.md"
    target.write_text("original\n", encoding="utf-8")

    proposal = FixProposal(
        proposal_id="bad", issue_kind="broken_wikilink",
        issue_path="sources/victim.md", issue_detail="d",
        operations=[FixOperation(
            kind="update_page",
            path="sources/victim.md",   # under base, NOT under knowledge/
            new_frontmatter={"title": "Victim"},
            new_body="clobbered\n",
            expected_hash=file_sha256(target),
        )],
        rationale="r", source="heuristic",
    )
    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )
    assert report.applied == []
    assert len(report.skipped) == 1
    assert "knowledge" in report.skipped[0]["reason"].lower()
    # The non-wiki file is untouched.
    assert target.read_text(encoding="utf-8") == "original\n"


@pytest.mark.asyncio
async def test_update_page_without_expected_hash_is_rejected(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    """A proposal that omits ``expected_hash`` for an update_page op
    must skip rather than silently bypass the concurrent-edit guard."""
    storage = parametrized_storage
    await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/x.md", title="X", body="# X\n",
    )
    proposal = FixProposal(
        proposal_id="p", issue_kind="broken_wikilink",
        issue_path="knowledge/x.md", issue_detail="d",
        operations=[FixOperation(
            kind="update_page", path="knowledge/x.md",
            new_frontmatter={"title": "X"},
            new_body="# X\nclobbered\n",
            expected_hash=None,  # malformed proposal — bypass attempt.
        )],
        rationale="r", source="heuristic",
    )
    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )
    assert report.applied == []
    assert len(report.skipped) == 1
    assert "expected_hash" in report.skipped[0]["reason"]


@pytest.mark.asyncio
async def test_two_ops_same_path_skip_subsequent_with_superseded_reason(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    """Real-world hazard: two ``broken_wikilink`` issues on the same
    page each generate a proposal whose ``new_body`` was computed
    against the original on-disk file. Naively applying both would
    silently revert the first fix; we want the second op to land in
    ``skipped`` with a clear ``superseded`` reason."""
    storage = parametrized_storage
    src_doc_id = await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/multi.md", title="Multi",
        body="# Multi\n\nLinks: [[alpha]] and [[beta]] here.\n",
    )
    abs_src = base_root / "knowledge/multi.md"
    snapshot_hash = file_sha256(abs_src)

    # Both ops carry the SAME pre-apply hash + each carries its own
    # full new_body that fixes one of the two broken links.
    p1 = FixProposal(
        proposal_id="p1", issue_kind="broken_wikilink",
        issue_path="knowledge/multi.md",
        issue_detail="[[alpha]] has no matching knowledge page",
        operations=[FixOperation(
            kind="update_page", path="knowledge/multi.md",
            new_frontmatter={"title": "Multi"},
            new_body="# Multi\n\nLinks: [[Alpha]] and [[beta]] here.\n",
            expected_hash=snapshot_hash,
        )],
        rationale="r", source="heuristic",
    )
    p2 = FixProposal(
        proposal_id="p2", issue_kind="broken_wikilink",
        issue_path="knowledge/multi.md",
        issue_detail="[[beta]] has no matching knowledge page",
        operations=[FixOperation(
            kind="update_page", path="knowledge/multi.md",
            new_frontmatter={"title": "Multi"},
            new_body="# Multi\n\nLinks: [[alpha]] and [[Beta]] here.\n",
            expected_hash=snapshot_hash,
        )],
        rationale="r", source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[p1, p2]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )
    # First op applies; second is skipped with a clear superseded reason
    # — not a hash-mismatch lie that misleads the user about which
    # version of the file was being targeted.
    assert len(report.applied) == 1
    assert report.applied[0].path == "knowledge/multi.md"
    assert len(report.skipped) == 1
    assert "superseded" in report.skipped[0]["reason"].lower()
    # Make sure the on-disk body was not silently reverted to op #2's
    # body (which would have lost op #1's fix).
    rewritten = abs_src.read_text(encoding="utf-8")
    assert "[[Alpha]]" in rewritten
    _ = src_doc_id


@pytest.mark.asyncio
async def test_skip_filter_drops_skipped_proposals(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    storage = parametrized_storage
    await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/a.md", title="A", body="# A\n",
    )
    await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/b.md", title="B", body="# B\n",
    )
    a_hash = file_sha256(base_root / "knowledge/a.md")
    b_hash = file_sha256(base_root / "knowledge/b.md")

    p_a = FixProposal(
        proposal_id="pa", issue_kind="broken_wikilink",
        issue_path="knowledge/a.md", issue_detail="d",
        operations=[FixOperation(
            kind="update_page", path="knowledge/a.md",
            new_frontmatter={"title": "A"},
            new_body="# A\nv2\n", expected_hash=a_hash,
        )],
        rationale="r", source="heuristic",
    )
    p_b = FixProposal(
        proposal_id="pb", issue_kind="broken_wikilink",
        issue_path="knowledge/b.md", issue_detail="d",
        operations=[FixOperation(
            kind="update_page", path="knowledge/b.md",
            new_frontmatter={"title": "B"},
            new_body="# B\nv2\n", expected_hash=b_hash,
        )],
        rationale="r", source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[p_a, p_b]),
        storage=storage, base_root=base_root,
        skip=[1],  # drop p_b
        reporter=_NullReporter(),
    )
    assert len(report.applied) == 1
    assert report.applied[0].path == "knowledge/a.md"


@pytest.mark.asyncio
async def test_proposal_is_atomic_subsequent_ops_skip_after_failure(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    """A multi-op proposal (e.g. ``non_atomic_page``: N create + 1
    delete) must be atomic at apply time. If a create_page op fails
    because its target path already exists, the delete_page that would
    drop the original page MUST also skip — otherwise the user ends
    up with a partial split AND the original deleted, silently losing
    the content the failed child would have carried.

    Sibling proposals must still apply normally (a failed proposal
    cannot abort independent fixes)."""
    storage = parametrized_storage
    # Pre-existing K-page that will collide with the first child.
    await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/concepts/topic-a.md", title="Topic A",
        body="# Topic A\n\noriginal content\n",
    )
    # The "fat" page being split.
    await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/concepts/grab-bag.md", title="Grab Bag",
        body="# Grab Bag\n\n## Topic A\n\n## Topic B\n",
    )
    grab_hash = file_sha256(base_root / "knowledge/concepts/grab-bag.md")
    # An independent broken_wikilink fix on a *different* page that
    # must NOT be aborted by the non_atomic_page proposal failure.
    await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/sibling.md", title="Sibling",
        body="# Sibling\n\nlink to [[other]]\n",
    )
    sibling_hash = file_sha256(base_root / "knowledge/sibling.md")

    split_proposal = FixProposal(
        proposal_id="split", issue_kind="non_atomic_page",
        issue_path="knowledge/concepts/grab-bag.md",
        issue_detail="page looks like multiple atomic notes glued together",
        operations=[
            # First op: create_page that collides with the seeded
            # ``knowledge/concepts/topic-a.md`` — apply will skip it with
            # "file already exists at create_page path".
            FixOperation(
                kind="create_page",
                path="knowledge/concepts/topic-a.md",
                new_frontmatter={"title": "Topic A", "type": "concept"},
                new_body="# Topic A\n\nchild a\n",
                expected_hash=None,
            ),
            FixOperation(
                kind="create_page",
                path="knowledge/concepts/topic-b.md",
                new_frontmatter={"title": "Topic B", "type": "concept"},
                new_body="# Topic B\n\nchild b\n",
                expected_hash=None,
            ),
            # Last op: would drop the original AFTER both children.
            # Atomicity rule: must NOT execute, since op #1 skipped.
            FixOperation(
                kind="delete_page",
                path="knowledge/concepts/grab-bag.md",
                expected_hash=grab_hash,
            ),
        ],
        rationale="LLM split — 2 atomic children + delete original",
        source="llm",
    )
    sibling_proposal = FixProposal(
        proposal_id="sib", issue_kind="broken_wikilink",
        issue_path="knowledge/sibling.md", issue_detail="[[other]]",
        operations=[FixOperation(
            kind="update_page", path="knowledge/sibling.md",
            new_frontmatter={"title": "Sibling"},
            new_body="# Sibling\n\nlink to [[Other]]\n",
            expected_hash=sibling_hash,
        )],
        rationale="r", source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(
            proposals=[split_proposal, sibling_proposal]
        ),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )

    # The original fat page must still exist on disk — the delete_page
    # was abandoned because the proposal was aborted after op #1's skip.
    assert (base_root / "knowledge/concepts/grab-bag.md").is_file()
    # The sibling proposal applied independently.
    sibling_after = (base_root / "knowledge/sibling.md").read_text(encoding="utf-8")
    assert "[[Other]]" in sibling_after

    applied_paths = [op.path for op in report.applied]
    skipped_paths = [s["path"] for s in report.skipped]
    assert "knowledge/concepts/grab-bag.md" not in applied_paths
    assert "knowledge/sibling.md" in applied_paths
    # All three ops of the split proposal must show as skipped — no
    # half-applied state on disk, even for the first child whose path
    # WOULD have been free in isolation. Preflight catches the
    # collision before any write happens.
    assert "knowledge/concepts/topic-a.md" in skipped_paths
    assert "knowledge/concepts/topic-b.md" in skipped_paths
    assert "knowledge/concepts/grab-bag.md" in skipped_paths
    # Reasons: every op of the split proposal carries the preflight
    # collision message; the sibling proposal is unaffected.
    split_skips = [
        s for s in report.skipped
        if s["proposal_id"] == "split"
    ]
    assert len(split_skips) == 3
    assert all("collide" in s["reason"] for s in split_skips), split_skips
    # The other-child file that would have landed on disk in isolation
    # must NOT exist (preflight refused before any write).
    assert not (base_root / "knowledge/concepts/topic-b.md").exists()


@pytest.mark.asyncio
async def test_preflight_catches_mid_proposal_hash_drift(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    """The earlier per-proposal abort guard only stopped *subsequent*
    ops once one failed — earlier writes already on disk stayed there.
    Preflight closes that hole: a multi-op proposal whose 2nd op would
    fail (hash drift on a sibling page) must NOT mutate the 1st op's
    target either."""
    storage = parametrized_storage
    await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/page-a.md", title="Page A",
        body="# Page A\n\noriginal A\n",
    )
    await _seed_page(
        storage=storage, base_root=base_root,
        path="knowledge/page-b.md", title="Page B",
        body="# Page B\n\noriginal B\n",
    )
    a_hash_correct = file_sha256(base_root / "knowledge/page-a.md")
    # Stale hash for B (as if the proposal was generated against an
    # earlier version of page-b.md and B was edited externally since).
    b_hash_stale = "0" * 64

    proposal = FixProposal(
        proposal_id="multi", issue_kind="orphan_page",
        issue_path="knowledge/page-a.md",
        issue_detail="multi-op fix",
        operations=[
            # Op #1 would succeed in isolation: hash matches A.
            FixOperation(
                kind="update_page", path="knowledge/page-a.md",
                new_frontmatter={"title": "Page A"},
                new_body="# Page A\n\nupdated A\n",
                expected_hash=a_hash_correct,
            ),
            # Op #2 has a stale hash for B → preflight rejects whole proposal.
            FixOperation(
                kind="update_page", path="knowledge/page-b.md",
                new_frontmatter={"title": "Page B"},
                new_body="# Page B\n\nupdated B\n",
                expected_hash=b_hash_stale,
            ),
        ],
        rationale="multi-op", source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )

    assert report.applied == []
    assert len(report.skipped) == 2
    # Page A on disk must still be the original, even though op #1
    # would have succeeded in isolation.
    assert "original A" in (base_root / "knowledge/page-a.md").read_text(
        encoding="utf-8"
    )
    assert "updated A" not in (base_root / "knowledge/page-a.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_create_page_apply_indexes_into_storage(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    """A successful ``create_page`` op MUST register the page in storage,
    not just write to disk. Without this, the next ``run_lint`` cannot
    see the new page (it builds its title map from
    ``storage.list_documents``)."""
    storage = parametrized_storage
    new_path = "knowledge/concepts/qin-dynasty.md"

    proposal = FixProposal(
        proposal_id="p-create",
        issue_kind="broken_wikilink",
        issue_path="knowledge/articles/china.md",
        issue_detail="[[Qin Dynasty]] has no matching knowledge page",
        issue_line=1,
        operations=[
            FixOperation(
                kind="create_page",
                path=new_path,
                new_frontmatter={
                    "id": "K-qin-dynasty",
                    "type": "concept",
                    "title": "Qin Dynasty",
                    "created": "2026-05-10T00:00:00+00:00",
                    "updated": "2026-05-10T00:00:00+00:00",
                },
                new_body="# Qin Dynasty\n\nTODO: stub.\n",
                expected_hash=None,
            )
        ],
        rationale="test fixture", source="llm",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )

    assert len(report.applied) == 1
    assert report.skipped == []

    # Document row landed.
    docs = list(await storage.list_documents(layer=Layer.KNOWLEDGE, active=True))
    qin = next((d for d in docs if d.path == new_path), None)
    assert qin is not None, "create_page must register the doc in storage"
    assert qin.title == "Qin Dynasty"
    assert qin.hash != ""

    # Chunks landed.
    chunks = await storage.list_chunks(qin.doc_id)
    assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_apply_then_lint_does_not_re_report_broken_wikilink(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    """End-to-end regression: apply ``create_page`` → immediately
    ``run_lint`` → the original ``broken_wikilink`` should NOT reappear
    (no need for a separate ``dikw client ingest`` to bridge the gap)."""
    from dikw_core.domains.knowledge.lint import run_lint

    storage = parametrized_storage
    src_path = "knowledge/articles/china-history.md"
    await _seed_page(
        storage=storage, base_root=base_root,
        path=src_path, title="China History",
        body="# China History\n\nThe [[Qin Dynasty]] unified ...\n",
    )

    before = await run_lint(storage=storage, root=base_root)
    broken_before = [
        i for i in before.issues
        if i.kind == "broken_wikilink"
        and "Qin Dynasty" in i.detail
        and i.path == src_path
    ]
    assert len(broken_before) == 1

    proposal = FixProposal(
        proposal_id="p-fix",
        issue_kind="broken_wikilink",
        issue_path=src_path,
        issue_detail=broken_before[0].detail,
        issue_line=broken_before[0].line,
        operations=[FixOperation(
            kind="create_page",
            path="knowledge/concepts/qin-dynasty.md",
            new_frontmatter={
                "id": "K-qin-dynasty",
                "type": "concept",
                "title": "Qin Dynasty",
                "created": "2026-05-10T00:00:00+00:00",
                "updated": "2026-05-10T00:00:00+00:00",
            },
            new_body="# Qin Dynasty\n\nTODO: stub.\n",
            expected_hash=None,
        )],
        rationale="test fixture", source="llm",
    )
    await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )

    after = await run_lint(storage=storage, root=base_root)
    broken_after = [
        i for i in after.issues
        if i.kind == "broken_wikilink"
        and "Qin Dynasty" in i.detail
        and i.path == src_path
    ]
    assert broken_after == []


@pytest.mark.asyncio
async def test_apply_create_page_reconciles_referrer_outgoing_links(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    """When apply creates a missing target page, the source page's
    outgoing links MUST be re-resolved against the post-apply title
    index. Otherwise:
    - storage.links_from(source) misses the new edge
    - run_lint's inbound counter never sees it
    - the freshly-created target is immediately reported as orphan_page
    despite being linked.
    """
    from dikw_core.domains.knowledge.lint import run_lint

    storage = parametrized_storage
    src_path = "knowledge/articles/china-history.md"
    src_doc_id = await _seed_page(
        storage=storage, base_root=base_root,
        path=src_path, title="China History",
        body="# China History\n\nThe [[Qin Dynasty]] unified ...\n",
    )

    proposal = FixProposal(
        proposal_id="p-fix-referrer",
        issue_kind="broken_wikilink",
        issue_path=src_path,
        issue_detail="[[Qin Dynasty]] has no matching knowledge page",
        issue_line=3,
        operations=[FixOperation(
            kind="create_page",
            path="knowledge/concepts/qin-dynasty.md",
            new_frontmatter={
                "id": "K-qin", "type": "concept", "title": "Qin Dynasty",
                "created": "2026-05-10T00:00:00+00:00",
                "updated": "2026-05-10T00:00:00+00:00",
            },
            new_body="# Qin Dynasty\n\nstub\n",
            expected_hash=None,
        )],
        rationale="test fixture", source="llm",
    )
    await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )

    # Source page's storage links must now include the edge to the new page.
    src_links = [
        link for link in await storage.links_from(src_doc_id)
        if link.link_type == LinkType.WIKILINK
    ]
    assert any(
        link.dst_path == "knowledge/concepts/qin-dynasty.md" for link in src_links
    ), "referrer page outgoing link to the newly-created target was not reconciled"

    # And the new page must NOT be reported as orphan in the next run_lint.
    after = await run_lint(storage=storage, root=base_root)
    orphan = [
        i for i in after.issues
        if i.kind == "orphan_page"
        and i.path == "knowledge/concepts/qin-dynasty.md"
    ]
    assert orphan == [], (
        "freshly-created page should not be reported as orphan when the "
        "referrer's [[Title]] now resolves to it"
    )


@pytest.mark.asyncio
async def test_apply_sibling_cross_links_resolve_in_one_pass(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    """When a single apply batch creates two pages where the
    earlier-sorted page links to the later-sorted page (common for
    ``non_atomic_page`` splits with sibling ``[[Title]]`` cross-links),
    BOTH outgoing edges must land in storage in this pass — otherwise
    the source page's link to a still-not-persisted sibling silently
    drops, and phase 2's referrer reconcile skips ``paths_changed`` so
    the gap is never recovered until the next ingest."""
    storage = parametrized_storage

    proposal = FixProposal(
        proposal_id="p-split",
        issue_kind="non_atomic_page",
        issue_path="knowledge/source.md",
        issue_detail="splitting fat page into two atomic children",
        operations=[
            FixOperation(
                kind="create_page",
                # alpha-sorts before topic-b — body links to topic-b
                # whose title only exists in this same batch.
                path="knowledge/concepts/topic-a.md",
                new_frontmatter={
                    "id": "K-a", "type": "concept", "title": "Topic A",
                    "created": "2026-05-10T00:00:00+00:00",
                    "updated": "2026-05-10T00:00:00+00:00",
                },
                new_body="# Topic A\n\nSee also [[Topic B]].\n",
                expected_hash=None,
            ),
            FixOperation(
                kind="create_page",
                path="knowledge/concepts/topic-b.md",
                new_frontmatter={
                    "id": "K-b", "type": "concept", "title": "Topic B",
                    "created": "2026-05-10T00:00:00+00:00",
                    "updated": "2026-05-10T00:00:00+00:00",
                },
                new_body="# Topic B\n\nbody.\n",
                expected_hash=None,
            ),
        ],
        rationale="split", source="llm",
    )
    await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )

    a_id = _wiki_doc_id("knowledge/concepts/topic-a.md")
    a_links = [
        link for link in await storage.links_from(a_id)
        if link.link_type == LinkType.WIKILINK
    ]
    assert any(
        link.dst_path == "knowledge/concepts/topic-b.md" for link in a_links
    ), (
        "Topic A's outgoing wikilink to Topic B was lost — phase 1 "
        "persisted A before B's title entered the resolver index"
    )


@pytest.mark.asyncio
async def test_apply_uses_configured_cjk_tokenizer(
    parametrized_storage: Storage, base_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_lint_apply must thread its cjk_tokenizer kwarg through to
    persist_knowledge so K-layer chunks land split with the same
    tokenizer ingest uses. Otherwise a base configured for ``jieba``
    silently downgrades lint-apply chunks to whitespace splitting,
    diverging from doc.hash and breaking embedding backfill."""
    storage = parametrized_storage

    captured: list[dict[str, Any]] = []
    from dikw_core.domains.knowledge import lint_fix as lint_fix_module
    real_persist = lint_fix_module.persist_knowledge

    async def _spy(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return await real_persist(**kwargs)

    monkeypatch.setattr(lint_fix_module, "persist_knowledge", _spy)

    proposal = FixProposal(
        proposal_id="p-cjk",
        issue_kind="broken_wikilink",
        issue_path="knowledge/source.md",
        issue_detail="stub",
        operations=[FixOperation(
            kind="create_page",
            path="knowledge/concepts/qin-dynasty.md",
            new_frontmatter={
                "id": "K-qin", "type": "concept", "title": "秦朝",
                "created": "2026-05-10T00:00:00+00:00",
                "updated": "2026-05-10T00:00:00+00:00",
            },
            new_body="# 秦朝\n\n秦朝是中国历史上第一个大一统的朝代。\n",
            expected_hash=None,
        )],
        rationale="r", source="llm",
    )

    await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
        cjk_tokenizer="jieba",
    )

    assert len(captured) == 1
    assert captured[0].get("cjk_tokenizer") == "jieba"


@pytest.mark.asyncio
async def test_apply_uses_path_slug_title_when_op_frontmatter_missing(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    """When an applied create_page op has no string title in
    new_frontmatter, _build_page_from_op falls back to a path-slug
    derived title (e.g. knowledge/concepts/topic-a.md → 'Topic A'). Phase
    0 must apply the SAME fallback so a sibling page that links to
    [[Topic A]] resolves in this batch — without it, the storage
    edge from the sibling silently drops until the next ingest."""
    storage = parametrized_storage

    proposal = FixProposal(
        proposal_id="p-no-title",
        issue_kind="non_atomic_page",
        issue_path="knowledge/source.md",
        issue_detail="split",
        operations=[
            FixOperation(
                kind="create_page",
                path="knowledge/concepts/topic-a.md",
                new_frontmatter={
                    "id": "K-a", "type": "concept",
                    "created": "2026-05-10T00:00:00+00:00",
                    "updated": "2026-05-10T00:00:00+00:00",
                },
                new_body="# Topic A\n\nbody.\n",
                expected_hash=None,
            ),
            FixOperation(
                kind="create_page",
                path="knowledge/concepts/topic-b.md",
                new_frontmatter={
                    "id": "K-b", "type": "concept", "title": "Topic B",
                    "created": "2026-05-10T00:00:00+00:00",
                    "updated": "2026-05-10T00:00:00+00:00",
                },
                new_body="# Topic B\n\nSee [[Topic A]] for context.\n",
                expected_hash=None,
            ),
        ],
        rationale="split", source="llm",
    )
    await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )

    b_id = _wiki_doc_id("knowledge/concepts/topic-b.md")
    b_links = [
        link for link in await storage.links_from(b_id)
        if link.link_type == LinkType.WIKILINK
    ]
    assert any(
        link.dst_path == "knowledge/concepts/topic-a.md" for link in b_links
    ), (
        "Topic B's [[Topic A]] failed to resolve — phase 0 didn't seed "
        "the path-slug fallback title for op A"
    )


@pytest.mark.asyncio
async def test_apply_strips_whitespace_in_op_title_for_resolver_index(
    parametrized_storage: Storage, base_root: Path,
) -> None:
    """A fixer that writes a title with leading/trailing whitespace
    must not break sibling cross-link resolution. Phase 0 strips the
    raw title before seeding title_to_path so a sibling [[Topic A]]
    resolves regardless of how the surrounding op spelled the title."""
    storage = parametrized_storage

    proposal = FixProposal(
        proposal_id="p-strip",
        issue_kind="non_atomic_page",
        issue_path="knowledge/source.md",
        issue_detail="split",
        operations=[
            FixOperation(
                kind="create_page",
                path="knowledge/concepts/topic-a.md",
                new_frontmatter={
                    "id": "K-a", "type": "concept",
                    "title": "  Topic A  ",  # leading + trailing whitespace
                    "created": "2026-05-10T00:00:00+00:00",
                    "updated": "2026-05-10T00:00:00+00:00",
                },
                new_body="# Topic A\n\nbody.\n",
                expected_hash=None,
            ),
            FixOperation(
                kind="create_page",
                path="knowledge/concepts/topic-b.md",
                new_frontmatter={
                    "id": "K-b", "type": "concept", "title": "Topic B",
                    "created": "2026-05-10T00:00:00+00:00",
                    "updated": "2026-05-10T00:00:00+00:00",
                },
                new_body="# Topic B\n\nSee [[Topic A]] for context.\n",
                expected_hash=None,
            ),
        ],
        rationale="split", source="llm",
    )
    await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, base_root=base_root,
        reporter=_NullReporter(),
    )

    b_id = _wiki_doc_id("knowledge/concepts/topic-b.md")
    b_links = [
        link for link in await storage.links_from(b_id)
        if link.link_type == LinkType.WIKILINK
    ]
    assert any(
        link.dst_path == "knowledge/concepts/topic-a.md" for link in b_links
    ), (
        "Topic B's [[Topic A]] did not resolve — _op_title left "
        "whitespace in the phase-0 dict key"
    )
