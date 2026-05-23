"""Tests for the ``missing_provenance`` lint kind + ``MissingProvenanceFixer``.

Covers:

* ``run_lint`` detection — emits ``missing_provenance`` when frontmatter
  ``sources:`` is non-empty but ``storage.provenance_from`` returns a
  different set (zero / partial / stale row).
* ``run_lint`` suppression via ``lint: {skip: [missing_provenance]}``
  frontmatter — same shape as the other LintKinds.
* ``MissingProvenanceFixer`` — emits a single ``reconcile_provenance``
  op carrying the frontmatter snapshot + ``expected_hash``.
* Apply path — ``replace_provenance_from`` is called with the snapshot,
  the wiki file is NOT modified (hash unchanged), and the next lint
  pass reports no issue.

Why no LLM test: ``MissingProvenanceFixer`` is purely deterministic
(no ``enable_llm`` branch); the propose path has no provider
dependency.
"""

from __future__ import annotations

from pathlib import Path

import frontmatter
import pytest

from dikw_core import api
from dikw_core.domains.data.path_norm import doc_id_for, normalize_path
from dikw_core.domains.knowledge.lint import run_lint
from dikw_core.domains.knowledge.lint_fix import FixProposalReport
from dikw_core.domains.knowledge.wiki import build_page, write_page
from dikw_core.schemas import DocumentRecord, Layer

from .fakes import init_test_wiki


async def _seed_wiki_page(
    *,
    wiki_root: Path,
    title: str,
    sources: list[str],
    extras_lint: dict | None = None,
) -> tuple[str, str]:
    """Write a K-page with the given frontmatter ``sources:`` (and
    optional ``lint:`` block), register its DocumentRecord, and return
    ``(path, doc_id)``."""
    page = build_page(
        title=title,
        body=f"# {title}\n\nBody.\n",
        type_="concept",
        sources=sources,
        extras={"lint": extras_lint} if extras_lint else {},
    )
    write_page(wiki_root, page)
    doc_id = doc_id_for(Layer.WIKI, page.path)

    _cfg, _root, storage = await api._with_storage(wiki_root)
    try:
        await storage.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                path=page.path,
                title=page.title,
                hash=f"hash-{page.path}",
                mtime=0.0,
                layer=Layer.WIKI,
                active=True,
            )
        )
    finally:
        await storage.close()
    return page.path, doc_id


async def _run_lint(wiki_root: Path):
    _cfg, root, storage = await api._with_storage(wiki_root)
    try:
        return await run_lint(storage, root=root)
    finally:
        await storage.close()


@pytest.fixture()
def empty_wiki(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    return wiki


# ---- Detection -----------------------------------------------------------


@pytest.mark.asyncio
async def test_run_lint_emits_missing_provenance_when_table_empty(
    empty_wiki: Path,
) -> None:
    """K-page frontmatter declares 2 sources but the provenance table is
    empty (typical legacy-base state) → one ``missing_provenance``
    issue surfaces."""
    path, _doc_id = await _seed_wiki_page(
        wiki_root=empty_wiki,
        title="Page",
        sources=["sources/a.md", "sources/b.md"],
    )

    report = await _run_lint(empty_wiki)
    mp = [i for i in report.issues if i.kind == "missing_provenance"]
    assert [i.path for i in mp] == [path]
    assert "declares 2" in mp[0].detail


@pytest.mark.asyncio
async def test_run_lint_emits_missing_provenance_when_rows_stale(
    empty_wiki: Path,
) -> None:
    """Provenance table has rows from a prior reconcile, but the
    frontmatter has since been edited to a different set. The
    ``existing_keys != expected_keys`` check catches it as one issue;
    the fix is the same as the "table empty" sub-case."""
    path, doc_id = await _seed_wiki_page(
        wiki_root=empty_wiki,
        title="Page",
        sources=["sources/new.md"],
    )
    # Plant a stale row that no longer matches frontmatter.
    _cfg, _root, storage = await api._with_storage(empty_wiki)
    try:
        await storage.replace_provenance_from(doc_id, ["sources/old.md"])
    finally:
        await storage.close()

    report = await _run_lint(empty_wiki)
    assert any(
        i.kind == "missing_provenance" and i.path == path
        for i in report.issues
    )


@pytest.mark.asyncio
async def test_run_lint_no_issue_when_keys_match(empty_wiki: Path) -> None:
    """Provenance rows exactly match frontmatter → no issue. Pins the
    "fixed" state so a regression that always emits the issue would
    fail."""
    _path, doc_id = await _seed_wiki_page(
        wiki_root=empty_wiki,
        title="Page",
        sources=["sources/a.md"],
    )
    _cfg, _root, storage = await api._with_storage(empty_wiki)
    try:
        await storage.replace_provenance_from(doc_id, ["sources/a.md"])
    finally:
        await storage.close()

    report = await _run_lint(empty_wiki)
    assert not any(i.kind == "missing_provenance" for i in report.issues)


@pytest.mark.asyncio
async def test_run_lint_no_issue_when_no_sources_and_table_empty(
    empty_wiki: Path,
) -> None:
    """No ``sources:`` frontmatter AND no stale rows → no issue.

    Pins the "both sides empty is clean" half of the four-case lookup
    table (frontmatter empty / non-empty, table empty / non-empty). The
    "frontmatter cleared but stale rows linger" case lives in the next
    test.
    """
    await _seed_wiki_page(
        wiki_root=empty_wiki, title="Page", sources=[]
    )
    report = await _run_lint(empty_wiki)
    assert not any(i.kind == "missing_provenance" for i in report.issues)


@pytest.mark.asyncio
async def test_run_lint_emits_missing_provenance_when_sources_cleared_but_rows_stale(
    empty_wiki: Path,
) -> None:
    """Page previously had ``sources: [foo.md]`` (and reconciled
    provenance rows), the user then hand-edited the frontmatter to drop
    ``sources:`` entirely. The page now declares zero sources but the
    table still holds the old row — lint must surface it so the apply
    pass can call ``replace_provenance_from(doc_id, [])`` and clear the
    ghosts. Without this, ``api.read_provenance`` would keep returning
    sources the frontmatter no longer claims.
    """
    path, doc_id = await _seed_wiki_page(
        wiki_root=empty_wiki, title="Page", sources=[]
    )
    _cfg, _root, storage = await api._with_storage(empty_wiki)
    try:
        await storage.replace_provenance_from(doc_id, ["sources/old.md"])
    finally:
        await storage.close()

    report = await _run_lint(empty_wiki)
    mp = [i for i in report.issues if i.kind == "missing_provenance"]
    assert [i.path for i in mp] == [path]
    assert "declares 0" in mp[0].detail


@pytest.mark.asyncio
async def test_apply_clears_stale_rows_when_frontmatter_emptied(
    empty_wiki: Path,
) -> None:
    """Full lifecycle for the "sources cleared" case: detect → propose →
    apply → re-lint clean. Asserts ``replace_provenance_from(doc_id,
    [])`` actually fires (table becomes empty) and the next lint pass
    has no ``missing_provenance`` issue."""
    _path, doc_id = await _seed_wiki_page(
        wiki_root=empty_wiki, title="Page", sources=[]
    )
    _cfg, _root, storage = await api._with_storage(empty_wiki)
    try:
        await storage.replace_provenance_from(doc_id, ["sources/old.md"])
    finally:
        await storage.close()

    proposal_report = await api.lint_propose(
        empty_wiki, rule="missing_provenance", limit=10
    )
    assert len(proposal_report.proposals) == 1
    proposal = proposal_report.proposals[0]
    op = proposal.operations[0]
    assert op.kind == "reconcile_provenance"
    assert op.source_paths == []  # the cleared snapshot

    apply_report = await api.lint_apply(
        empty_wiki, proposal_report=proposal_report
    )
    assert apply_report.applied and not apply_report.skipped

    _cfg, _root, storage = await api._with_storage(empty_wiki)
    try:
        assert await storage.provenance_from(doc_id) == []
    finally:
        await storage.close()

    post = await _run_lint(empty_wiki)
    assert not any(i.kind == "missing_provenance" for i in post.issues)


@pytest.mark.asyncio
async def test_run_lint_emits_missing_provenance_when_raw_spelling_drifts(
    empty_wiki: Path,
) -> None:
    """A user edits only the spelling/case/Unicode form of a ``sources:``
    entry (e.g., ``Sources/Foo.md`` -> ``sources/foo.md``). The
    normalized keys collapse to the same value, so a key-set comparison
    would call it clean; but the API surfaces the raw frontmatter
    spelling, so the table still holds the old casing and the
    forward-leg ``read_provenance`` answer drifts from the file. Lint
    must detect raw-value drift per normalized key so the apply pass
    rewrites the row with the new spelling.
    """
    sources_lower = ["sources/foo.md"]
    path, doc_id = await _seed_wiki_page(
        wiki_root=empty_wiki, title="Page", sources=sources_lower
    )
    # Plant the row with a different RAW spelling but the same
    # normalized key — simulates "user edited the casing in
    # frontmatter after a prior reconcile".
    _cfg, _root, storage = await api._with_storage(empty_wiki)
    try:
        await storage.replace_provenance_from(doc_id, ["Sources/Foo.md"])
        # Sanity: row exists with the old spelling.
        before = await storage.provenance_from(doc_id)
        assert [r.source_path for r in before] == ["Sources/Foo.md"]
    finally:
        await storage.close()

    report = await _run_lint(empty_wiki)
    assert any(
        i.kind == "missing_provenance" and i.path == path
        for i in report.issues
    ), (
        "raw-spelling drift must surface as missing_provenance even when "
        "normalized keys match"
    )

    # Apply the fixer and verify the table now carries the lowercase
    # spelling — round-trip pins the actual user-visible fix.
    proposal_report = await api.lint_propose(
        empty_wiki, rule="missing_provenance", limit=10
    )
    assert len(proposal_report.proposals) == 1
    await api.lint_apply(empty_wiki, proposal_report=proposal_report)

    _cfg, _root, storage = await api._with_storage(empty_wiki)
    try:
        after = await storage.provenance_from(doc_id)
        assert [r.source_path for r in after] == ["sources/foo.md"]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_run_lint_skip_frontmatter_suppresses_missing_provenance(
    empty_wiki: Path,
) -> None:
    """``lint: {skip: [missing_provenance]}`` in the page's frontmatter
    suppresses the issue. Wires through the existing per-page
    suppression mechanism — get_args(LintKind) already validates the
    new kind so no extra plumbing."""
    await _seed_wiki_page(
        wiki_root=empty_wiki,
        title="Page",
        sources=["sources/a.md"],
        extras_lint={"skip": ["missing_provenance"]},
    )
    report = await _run_lint(empty_wiki)
    assert not any(i.kind == "missing_provenance" for i in report.issues)


# ---- Fixer + apply -------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_provenance_propose_emits_reconcile_op(
    empty_wiki: Path,
) -> None:
    """``lint propose`` produces one ``reconcile_provenance`` op per
    issue, carrying the frontmatter ``sources:`` snapshot + an
    ``expected_hash`` stamp for concurrent-edit safety."""
    path, _doc_id = await _seed_wiki_page(
        wiki_root=empty_wiki,
        title="Page",
        sources=["sources/a.md", "sources/b.md"],
    )
    proposal_report = await api.lint_propose(
        empty_wiki, rule="missing_provenance", limit=10
    )
    assert isinstance(proposal_report, FixProposalReport)
    assert len(proposal_report.proposals) == 1
    proposal = proposal_report.proposals[0]
    assert proposal.source == "heuristic"
    assert proposal.issue_path == path
    op = proposal.operations[0]
    assert op.kind == "reconcile_provenance"
    assert op.path == path
    assert op.source_paths == ["sources/a.md", "sources/b.md"]
    assert op.expected_hash and len(op.expected_hash) == 64


@pytest.mark.asyncio
async def test_apply_reconcile_provenance_writes_storage_rows(
    empty_wiki: Path,
) -> None:
    """``lint apply`` calls ``replace_provenance_from`` with the
    snapshot. After apply the table matches frontmatter exactly."""
    path, doc_id = await _seed_wiki_page(
        wiki_root=empty_wiki,
        title="Page",
        sources=["sources/a.md", "sources/b.md"],
    )

    proposal_report = await api.lint_propose(
        empty_wiki, rule="missing_provenance", limit=10
    )
    apply_report = await api.lint_apply(
        empty_wiki, proposal_report=proposal_report
    )
    assert len(apply_report.applied) == 1
    assert apply_report.skipped == []

    _cfg, _root, storage = await api._with_storage(empty_wiki)
    try:
        rows = await storage.provenance_from(doc_id)
        assert {r.source_path for r in rows} == {
            "sources/a.md",
            "sources/b.md",
        }
        assert {r.source_path_key for r in rows} == {
            normalize_path("sources/a.md"),
            normalize_path("sources/b.md"),
        }
    finally:
        await storage.close()
    # The fix path doesn't touch the file → wiki_paths_changed is empty.
    assert path not in apply_report.wiki_paths_changed


@pytest.mark.asyncio
async def test_apply_reconcile_provenance_does_not_modify_wiki_file(
    empty_wiki: Path,
) -> None:
    """The reconcile op is the "narrowest possible write" — storage only.
    Wiki file bytes must be byte-identical before and after apply."""
    path, _doc_id = await _seed_wiki_page(
        wiki_root=empty_wiki,
        title="Page",
        sources=["sources/a.md"],
    )
    file_path = empty_wiki / path
    before = file_path.read_bytes()

    proposal_report = await api.lint_propose(
        empty_wiki, rule="missing_provenance", limit=10
    )
    await api.lint_apply(empty_wiki, proposal_report=proposal_report)

    after = file_path.read_bytes()
    assert before == after


@pytest.mark.asyncio
async def test_apply_then_relint_clears_missing_provenance_issue(
    empty_wiki: Path,
) -> None:
    """Full lifecycle: detect → propose → apply → re-lint clean. Pins
    the self-disabling property — once the table matches frontmatter,
    the comparison passes and no issue surfaces."""
    await _seed_wiki_page(
        wiki_root=empty_wiki,
        title="Page",
        sources=["sources/a.md"],
    )
    pre = await _run_lint(empty_wiki)
    assert any(i.kind == "missing_provenance" for i in pre.issues)

    proposal_report = await api.lint_propose(
        empty_wiki, rule="missing_provenance", limit=10
    )
    await api.lint_apply(empty_wiki, proposal_report=proposal_report)

    post = await _run_lint(empty_wiki)
    assert not any(i.kind == "missing_provenance" for i in post.issues)


@pytest.mark.asyncio
async def test_apply_skips_when_file_edited_between_propose_and_apply(
    empty_wiki: Path,
) -> None:
    """Concurrent-edit safety: if the user re-saves the page (changing
    the file hash) between propose and apply, the ``expected_hash``
    gate fires and the op is skipped — same shape as ``update_page``
    / ``delete_page``. The next lint pass will re-propose against the
    fresh state."""
    path, _doc_id = await _seed_wiki_page(
        wiki_root=empty_wiki,
        title="Page",
        sources=["sources/a.md"],
    )
    proposal_report = await api.lint_propose(
        empty_wiki, rule="missing_provenance", limit=10
    )
    # Concurrent edit: rewrite the file with a different body so the
    # hash drifts. Frontmatter sources: stays the same so the apply
    # would still be semantically correct — what we're guarding
    # against is the op silently overwriting a fresh user state.
    file_path = empty_wiki / path
    post = frontmatter.loads(file_path.read_text(encoding="utf-8"))
    post.content = "# Page\n\nEdited body.\n"
    file_path.write_text(frontmatter.dumps(post) + "\n", encoding="utf-8")

    apply_report = await api.lint_apply(
        empty_wiki, proposal_report=proposal_report
    )
    assert apply_report.applied == []
    assert len(apply_report.skipped) == 1
    assert "hash mismatch" in apply_report.skipped[0]["reason"]


@pytest.mark.asyncio
async def test_fixer_treats_scalar_sources_as_no_op(
    empty_wiki: Path,
) -> None:
    """``MissingProvenanceFixer`` reads the on-disk frontmatter directly,
    so it needs the same ``isinstance(list)`` guard as ``persist_wiki_page``
    and ``run_lint``. A scalar ``sources: foo.md`` (user hand-edit) must
    NOT iterate per-character into the op's ``source_paths``; the apply
    would then write 14 garbage rows over the stale ones we were trying
    to clear, plus the apply could raise on truthy-non-iterable values.

    Symmetric with the ``persist_wiki_page`` test in
    ``test_persist_wiki_page.py`` — the fixer is the second place that
    parses frontmatter for this field; both must agree on the
    "malformed shape -> zero sources" contract.
    """
    # Write the page directly so we can produce a scalar frontmatter
    # value (build_page always wraps in list()).
    page_path = "wiki/scalar.md"
    abs_path = empty_wiki / page_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(
        "---\n"
        "id: scalar-id\n"
        "type: concept\n"
        "title: Scalar\n"
        "sources: sources/foo.md\n"
        "---\n\n"
        "Body.\n",
        encoding="utf-8",
    )
    # Register the doc + plant stale rows so the lint pass surfaces the
    # issue (otherwise the scalar-empty + table-empty case would just be
    # clean and the fixer would never fire).
    doc_id = doc_id_for(Layer.WIKI, page_path)
    _cfg, _root, storage = await api._with_storage(empty_wiki)
    try:
        await storage.upsert_document(
            DocumentRecord(
                doc_id=doc_id, path=page_path, title="Scalar",
                hash="hash-scalar", mtime=0.0, layer=Layer.WIKI, active=True,
            )
        )
        await storage.replace_provenance_from(doc_id, ["sources/stale.md"])
    finally:
        await storage.close()

    proposal_report = await api.lint_propose(
        empty_wiki, rule="missing_provenance", limit=10
    )
    assert len(proposal_report.proposals) == 1
    op = proposal_report.proposals[0].operations[0]
    assert op.kind == "reconcile_provenance"
    # Pre-fix: source_paths = list(of chars). Fix: scalar -> empty list,
    # so apply clears the stale row instead of overwriting with garbage.
    assert op.source_paths == [], (
        f"scalar sources must produce empty source_paths; got "
        f"{op.source_paths}"
    )


@pytest.mark.asyncio
async def test_apply_reconcile_op_does_not_block_sibling_op_on_same_path(
    empty_wiki: Path,
) -> None:
    """A page that needs both ``reconcile_provenance`` and a real
    file-mutating fix (here a hand-built ``update_page``) must let both
    ops land in one apply pass. Reconcile doesn't change the file, so
    it must NOT enter ``touched_paths`` — otherwise the second op gets
    skipped as "superseded by earlier op on the same path".

    Catches the regression where adding reconcile to the unconditional
    ``touched_paths.add(op.path)`` line blocks unrelated lint fixes
    from running together.
    """
    import uuid

    from dikw_core.domains.knowledge.lint_fix import (
        FixOperation,
        FixProposal,
        FixProposalReport,
        bytes_sha256,
    )

    path, _doc_id = await _seed_wiki_page(
        wiki_root=empty_wiki,
        title="Page",
        sources=["sources/a.md"],
    )

    file_bytes = (empty_wiki / path).read_bytes()
    file_hash = bytes_sha256(file_bytes)

    # Hand-built two-proposal report: same path, reconcile first then a
    # contentless update_page (rewrite same body bytes). Both operate
    # on the same page; only the second one would be a real file
    # mutation. The update_page exercises the conflict gate.
    post = frontmatter.loads(file_bytes.decode("utf-8"))
    reconcile_proposal = FixProposal(
        proposal_id=str(uuid.uuid4()),
        issue_kind="missing_provenance",
        issue_path=path,
        issue_detail="reconcile",
        operations=[
            FixOperation(
                kind="reconcile_provenance",
                path=path,
                source_paths=["sources/a.md"],
                expected_hash=file_hash,
            )
        ],
        rationale="reconcile",
        source="heuristic",
    )
    update_proposal = FixProposal(
        proposal_id=str(uuid.uuid4()),
        issue_kind="broken_wikilink",
        issue_path=path,
        issue_detail="update",
        operations=[
            FixOperation(
                kind="update_page",
                path=path,
                new_frontmatter=dict(post.metadata),
                new_body=post.content + "\n\nNew line.\n",
                expected_hash=file_hash,
            )
        ],
        rationale="update",
        source="heuristic",
    )
    report = FixProposalReport(
        proposals=[reconcile_proposal, update_proposal]
    )

    apply_report = await api.lint_apply(
        empty_wiki, proposal_report=report
    )
    applied_kinds = [op.kind for op in apply_report.applied]
    assert applied_kinds == ["reconcile_provenance", "update_page"], (
        f"both ops on the same path must apply; got applied={applied_kinds}, "
        f"skipped={apply_report.skipped}"
    )
    # Sanity: the update_page write did land — file size grew.
    assert (empty_wiki / path).read_bytes() != file_bytes


@pytest.mark.asyncio
async def test_fixer_returns_none_when_target_file_vanished(
    empty_wiki: Path,
) -> None:
    """Direct unit test on ``MissingProvenanceFixer.propose`` — feed it
    an issue whose target file no longer exists on disk. The fixer's
    first guard (``if not abs_path.is_file(): return None``) must fire
    before ``read_bytes()``, returning ``None`` so the orchestrator
    records the skip rather than the propose call crashing.

    Hits this path directly (not via ``api.lint_propose``) because the
    upstream ``run_lint`` scan skips vanished-file docs before the
    fixer would ever be asked — the fixer's own guard exists for the
    race window between lint scan and propose iteration."""
    from dikw_core.domains.knowledge.lint import LintIssue
    from dikw_core.domains.knowledge.lint_fix import FixerContext
    from dikw_core.domains.knowledge.lint_fixers import MissingProvenanceFixer

    issue = LintIssue(
        kind="missing_provenance",
        path="wiki/ghost.md",  # never written to disk
        detail="phantom",
    )
    ctx = FixerContext(
        storage=None, llm=None, embedding=None,
        wiki_root=empty_wiki,
        all_pages=[], enable_llm=False,
    )

    class _NoopReporter:
        pass

    proposal = await MissingProvenanceFixer().propose(
        issue, ctx, _NoopReporter()
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_apply_skips_reconcile_op_with_missing_source_paths(
    empty_wiki: Path,
) -> None:
    """A hand-crafted ``reconcile_provenance`` op without ``source_paths``
    is structurally malformed — apply preflight skips with a clear
    reason rather than calling ``replace_provenance_from(None)``. The
    real fixer always stamps ``source_paths`` (even with ``[]``), so
    this path only fires for malformed external proposal JSON."""
    import uuid

    from dikw_core.domains.knowledge.lint_fix import (
        FixOperation,
        FixProposal,
        FixProposalReport,
        bytes_sha256,
    )

    path, _doc_id = await _seed_wiki_page(
        wiki_root=empty_wiki,
        title="Page",
        sources=["sources/a.md"],
    )
    file_hash = bytes_sha256((empty_wiki / path).read_bytes())

    bad = FixProposal(
        proposal_id=str(uuid.uuid4()),
        issue_kind="missing_provenance",
        issue_path=path,
        issue_detail="bad",
        operations=[
            FixOperation(
                kind="reconcile_provenance",
                path=path,
                source_paths=None,  # ← the bug we're guarding against
                expected_hash=file_hash,
            )
        ],
        rationale="malformed",
        source="heuristic",
    )

    apply_report = await api.lint_apply(
        empty_wiki, proposal_report=FixProposalReport(proposals=[bad])
    )
    assert apply_report.applied == []
    assert len(apply_report.skipped) == 1
    assert "source_paths" in apply_report.skipped[0]["reason"]


@pytest.mark.asyncio
async def test_apply_skips_reconcile_op_when_doc_id_unknown(
    empty_wiki: Path,
) -> None:
    """``_apply_one_op`` defends against a proposal whose path is no
    longer a registered K-layer document — typically because the page
    was deactivated between scan and apply. The op skips with a
    "deactivated since scan?" reason instead of writing storage rows
    for a phantom doc_id."""
    import uuid

    from dikw_core.domains.knowledge.lint_fix import (
        FixOperation,
        FixProposal,
        FixProposalReport,
        bytes_sha256,
    )

    # Stamp a phantom path that has a file on disk (so preflight
    # exists-check + hash-check pass) but is NOT registered as a
    # DocumentRecord. Manually write the file rather than seeding
    # via ``_seed_wiki_page`` (which also upserts the doc).
    phantom_path = "wiki/phantom.md"
    file_path = empty_wiki / phantom_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        "---\nid: x\ntype: concept\ntitle: P\nsources:\n  - sources/a.md\n"
        "---\n\n# P\n\nBody.\n",
        encoding="utf-8",
    )
    file_hash = bytes_sha256(file_path.read_bytes())

    op = FixOperation(
        kind="reconcile_provenance",
        path=phantom_path,
        source_paths=["sources/a.md"],
        expected_hash=file_hash,
    )
    report = FixProposalReport(
        proposals=[
            FixProposal(
                proposal_id=str(uuid.uuid4()),
                issue_kind="missing_provenance",
                issue_path=phantom_path,
                issue_detail="phantom",
                operations=[op],
                rationale="ghost",
                source="heuristic",
            )
        ]
    )
    apply_report = await api.lint_apply(empty_wiki, proposal_report=report)
    assert apply_report.applied == []
    assert len(apply_report.skipped) == 1
    assert "no doc_id" in apply_report.skipped[0]["reason"]


@pytest.mark.asyncio
async def test_preflight_skips_when_target_file_vanishes_before_apply(
    empty_wiki: Path,
) -> None:
    """Preflight's `_preflight_hash_gate` returns the
    ``{kind} target missing`` reason when the file is gone — the same
    helper now serves update_page, delete_page, AND reconcile_provenance
    after the refactor. Reconcile is the cheapest case to exercise this
    branch from."""
    path, _doc_id = await _seed_wiki_page(
        wiki_root=empty_wiki,
        title="Page",
        sources=["sources/a.md"],
    )
    proposal_report = await api.lint_propose(
        empty_wiki, rule="missing_provenance", limit=10
    )
    # File vanishes between propose (above) and apply (below) — the
    # preflight is_file() guard fires before any hash work.
    (empty_wiki / path).unlink()
    apply_report = await api.lint_apply(
        empty_wiki, proposal_report=proposal_report
    )
    assert apply_report.applied == []
    assert len(apply_report.skipped) == 1
    assert "target missing" in apply_report.skipped[0]["reason"]


@pytest.mark.asyncio
async def test_apply_rechecks_hash_after_preflight_for_reconcile_provenance(
    empty_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TOCTOU close: even if ``_preflight_proposal`` passed (because the
    file matched the expected hash at preflight time), ``_apply_one_op``
    must re-check the hash before writing storage rows. A
    monkeypatch-neutered preflight simulates the race where the user
    edits the file in the window between preflight and apply. Without
    the apply-side recheck, stale ``source_paths`` from the proposal
    would land in the table over the user's fresh state.

    Mirrors the ``update_page`` / ``delete_page`` two-check pattern —
    preflight catches the cheap mass case, apply closes the race.
    """
    path, doc_id = await _seed_wiki_page(
        wiki_root=empty_wiki,
        title="Page",
        sources=["sources/a.md"],
    )
    proposal_report = await api.lint_propose(
        empty_wiki, rule="missing_provenance", limit=10
    )
    assert len(proposal_report.proposals) == 1

    # Bypass preflight entirely — simulates "preflight saw the unedited
    # bytes, returned clean, then the user edited the file before apply
    # got to this op". The bug is real even when preflight passes; this
    # monkeypatch just removes preflight from the equation so the test
    # exercises the apply-side recheck in isolation.
    from dikw_core.domains.knowledge import lint_fix

    monkeypatch.setattr(lint_fix, "_preflight_proposal", lambda **_: None)

    file_path = empty_wiki / path
    post = frontmatter.loads(file_path.read_text(encoding="utf-8"))
    post.content = "# Page\n\nEdited body.\n"
    file_path.write_text(frontmatter.dumps(post) + "\n", encoding="utf-8")

    apply_report = await api.lint_apply(
        empty_wiki, proposal_report=proposal_report
    )
    assert apply_report.applied == []
    assert len(apply_report.skipped) == 1
    assert "hash mismatch" in apply_report.skipped[0]["reason"]

    # And confirm the stale snapshot did NOT land — without the apply
    # recheck, the table would be ``[sources/a.md]`` instead of empty
    # (apply was holding the pre-edit snapshot from propose).
    _cfg, _root, storage = await api._with_storage(empty_wiki)
    try:
        assert await storage.provenance_from(doc_id) == []
    finally:
        await storage.close()
