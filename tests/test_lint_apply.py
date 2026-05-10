"""Tests for ``run_lint_apply`` — the executor that mutates ``wiki/``
based on a :class:`FixProposalReport`.

PR1 design: apply only touches disk + outgoing-link reconciliation. It
intentionally skips re-chunking / re-embedding so apply stays
provider-free; a follow-up ``dikw ingest`` reconciles ``doc.hash``
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
    """Mirror of ``api._doc_id_for(Layer.WIKI, path)`` without the cycle."""
    from dikw_core.domains.data.path_norm import normalize_path

    return f"wiki:{normalize_path(path)}"


async def _seed_page(
    *, storage: Storage, wiki_root: Path,
    path: str, title: str, body: str,
) -> str:
    """Write a page on disk + register a DocumentRecord. Returns the doc_id."""
    abs_path = wiki_root / path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    body_text = f"---\ntitle: {title}\n---\n\n{body}"
    abs_path.write_text(body_text, encoding="utf-8")
    doc_id = _wiki_doc_id(path)
    await storage.upsert_document(
        DocumentRecord(
            doc_id=doc_id, path=path, title=title,
            hash=f"hash-{path}", mtime=0.0,
            layer=Layer.WIKI, active=True,
        )
    )
    return doc_id


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.mark.asyncio
async def test_update_page_writes_new_body_and_reconciles_links(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    storage = parametrized_storage
    target_doc_id = await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/concepts/foo-bar.md", title="Foo Bar",
        body="# Foo Bar\nbody\n",
    )
    src_doc_id = await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/concepts/source.md", title="Source",
        body="# Source\n\nSee [[foo  bar]] for context.\n",
    )
    abs_src = wiki_root / "wiki/concepts/source.md"
    expected_hash = file_sha256(abs_src)

    proposal = FixProposal(
        proposal_id="p1",
        issue_kind="broken_wikilink",
        issue_path="wiki/concepts/source.md",
        issue_detail="[[foo  bar]] has no matching wiki page",
        issue_line=3,
        operations=[
            FixOperation(
                kind="update_page",
                path="wiki/concepts/source.md",
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
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )

    assert isinstance(report, ApplyReport)
    assert len(report.applied) == 1
    assert report.skipped == []
    assert "wiki/concepts/source.md" in report.wiki_paths_changed

    # File was rewritten with new body.
    rewritten = abs_src.read_text(encoding="utf-8")
    assert "[[Foo Bar]]" in rewritten
    assert "[[foo  bar]]" not in rewritten

    # Links table: outgoing edge from source → foo-bar should now exist.
    links = await storage.links_from(src_doc_id)
    wikilinks = [link_row for link_row in links if link_row.link_type is LinkType.WIKILINK]
    assert len(wikilinks) == 1
    assert wikilinks[0].dst_path == "wiki/concepts/foo-bar.md"
    _ = target_doc_id  # held alive for the upsert side-effect


@pytest.mark.asyncio
async def test_hash_mismatch_skips_op_and_does_not_write_file(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    storage = parametrized_storage
    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/source.md", title="Source",
        body="# Source\n\nSee [[foo]] here.\n",
    )
    abs_src = wiki_root / "wiki/source.md"
    body_before = abs_src.read_text(encoding="utf-8")

    proposal = FixProposal(
        proposal_id="p1",
        issue_kind="broken_wikilink",
        issue_path="wiki/source.md",
        issue_detail="[[foo]] has no matching wiki page",
        issue_line=3,
        operations=[
            FixOperation(
                kind="update_page",
                path="wiki/source.md",
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
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )

    assert report.applied == []
    assert len(report.skipped) == 1
    assert "hash" in report.skipped[0]["reason"].lower()
    # File untouched.
    assert abs_src.read_text(encoding="utf-8") == body_before


@pytest.mark.asyncio
async def test_create_page_writes_new_file(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    storage = parametrized_storage
    new_path = "wiki/concepts/brand-new.md"
    abs_new = wiki_root / new_path
    assert not abs_new.exists()

    proposal = FixProposal(
        proposal_id="p1",
        issue_kind="broken_wikilink",
        issue_path="wiki/source.md",
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
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )

    assert len(report.applied) == 1
    assert report.skipped == []
    assert abs_new.is_file()
    content = abs_new.read_text(encoding="utf-8")
    assert "Brand New" in content


@pytest.mark.asyncio
async def test_create_page_skipped_when_file_already_exists(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    storage = parametrized_storage
    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/concepts/exists.md", title="Exists",
        body="# Exists\n",
    )

    proposal = FixProposal(
        proposal_id="p1",
        issue_kind="broken_wikilink",
        issue_path="wiki/source.md",
        issue_detail="stub",
        operations=[
            FixOperation(
                kind="create_page",
                path="wiki/concepts/exists.md",
                new_frontmatter={"title": "Exists"},
                new_body="# Exists\nWill not overwrite\n",
                expected_hash=None,
            )
        ],
        rationale="r", source="heuristic",
    )
    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )
    assert report.applied == []
    assert len(report.skipped) == 1
    assert "exist" in report.skipped[0]["reason"].lower()


@pytest.mark.asyncio
async def test_delete_page_unlinks_file_and_deactivates_doc(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    storage = parametrized_storage
    doc_id = await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/concepts/dead.md", title="Dead",
        body="# Dead\n",
    )
    abs_dead = wiki_root / "wiki/concepts/dead.md"
    expected_hash = file_sha256(abs_dead)

    proposal = FixProposal(
        proposal_id="p1",
        issue_kind="duplicate_title",
        issue_path="wiki/concepts/dead.md",
        issue_detail="dup",
        operations=[
            FixOperation(
                kind="delete_page",
                path="wiki/concepts/dead.md",
                expected_hash=expected_hash,
            )
        ],
        rationale="r", source="heuristic",
    )
    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )
    assert len(report.applied) == 1
    assert not abs_dead.exists()

    # Doc should be deactivated.
    actives = list(await storage.list_documents(layer=Layer.WIKI, active=True))
    assert all(d.doc_id != doc_id for d in actives)


@pytest.mark.asyncio
async def test_pick_filter_applies_only_picked_proposals(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    storage = parametrized_storage
    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/foo.md", title="Foo",
        body="# Foo\n",
    )
    abs_foo = wiki_root / "wiki/foo.md"
    foo_hash = file_sha256(abs_foo)

    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/bar.md", title="Bar",
        body="# Bar\n",
    )
    abs_bar = wiki_root / "wiki/bar.md"
    bar_hash = file_sha256(abs_bar)
    bar_body_before = abs_bar.read_text(encoding="utf-8")

    p_foo = FixProposal(
        proposal_id="pa",
        issue_kind="broken_wikilink",
        issue_path="wiki/foo.md",
        issue_detail="d",
        operations=[
            FixOperation(
                kind="update_page", path="wiki/foo.md",
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
        issue_path="wiki/bar.md",
        issue_detail="d",
        operations=[
            FixOperation(
                kind="update_page", path="wiki/bar.md",
                new_frontmatter={"title": "Bar"},
                new_body="# Bar\nupdated\n",
                expected_hash=bar_hash,
            )
        ],
        rationale="r", source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[p_foo, p_bar]),
        storage=storage, wiki_root=wiki_root,
        pick=[0],  # only the first
        reporter=_NullReporter(),
    )
    assert len(report.applied) == 1
    assert report.applied[0].path == "wiki/foo.md"
    # Bar untouched.
    assert abs_bar.read_text(encoding="utf-8") == bar_body_before


@pytest.mark.asyncio
async def test_apply_refuses_paths_outside_wiki_root(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    """A malformed proposal with an absolute path or ``..`` traversal
    must never reach ``write_page`` / ``unlink`` — sandbox to wiki/."""
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
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )
    assert report.applied == []
    assert len(report.skipped) == 1
    assert "outside" in report.skipped[0]["reason"].lower()
    # Confirm nothing landed on disk.
    assert not (wiki_root / "../../../etc/dikw-leak.md").resolve().exists()


@pytest.mark.asyncio
async def test_apply_refuses_paths_under_base_but_outside_wiki(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    """``apply`` is a K-layer mutation contract — paths under the base
    but outside ``wiki/`` (sources/, dikw.yml, .dikw/) must be rejected
    even though they resolve under the resolved base root."""
    storage = parametrized_storage
    # Pre-create a sources file the proposal will try to clobber.
    sources_dir = wiki_root / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    target = sources_dir / "victim.md"
    target.write_text("original\n", encoding="utf-8")

    proposal = FixProposal(
        proposal_id="bad", issue_kind="broken_wikilink",
        issue_path="sources/victim.md", issue_detail="d",
        operations=[FixOperation(
            kind="update_page",
            path="sources/victim.md",   # under base, NOT under wiki/
            new_frontmatter={"title": "Victim"},
            new_body="clobbered\n",
            expected_hash=file_sha256(target),
        )],
        rationale="r", source="heuristic",
    )
    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )
    assert report.applied == []
    assert len(report.skipped) == 1
    assert "wiki" in report.skipped[0]["reason"].lower()
    # The non-wiki file is untouched.
    assert target.read_text(encoding="utf-8") == "original\n"


@pytest.mark.asyncio
async def test_update_page_without_expected_hash_is_rejected(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    """A proposal that omits ``expected_hash`` for an update_page op
    must skip rather than silently bypass the concurrent-edit guard."""
    storage = parametrized_storage
    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/x.md", title="X", body="# X\n",
    )
    proposal = FixProposal(
        proposal_id="p", issue_kind="broken_wikilink",
        issue_path="wiki/x.md", issue_detail="d",
        operations=[FixOperation(
            kind="update_page", path="wiki/x.md",
            new_frontmatter={"title": "X"},
            new_body="# X\nclobbered\n",
            expected_hash=None,  # malformed proposal — bypass attempt.
        )],
        rationale="r", source="heuristic",
    )
    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )
    assert report.applied == []
    assert len(report.skipped) == 1
    assert "expected_hash" in report.skipped[0]["reason"]


@pytest.mark.asyncio
async def test_two_ops_same_path_skip_subsequent_with_superseded_reason(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    """Real-world hazard: two ``broken_wikilink`` issues on the same
    page each generate a proposal whose ``new_body`` was computed
    against the original on-disk file. Naively applying both would
    silently revert the first fix; we want the second op to land in
    ``skipped`` with a clear ``superseded`` reason."""
    storage = parametrized_storage
    src_doc_id = await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/multi.md", title="Multi",
        body="# Multi\n\nLinks: [[alpha]] and [[beta]] here.\n",
    )
    abs_src = wiki_root / "wiki/multi.md"
    snapshot_hash = file_sha256(abs_src)

    # Both ops carry the SAME pre-apply hash + each carries its own
    # full new_body that fixes one of the two broken links.
    p1 = FixProposal(
        proposal_id="p1", issue_kind="broken_wikilink",
        issue_path="wiki/multi.md",
        issue_detail="[[alpha]] has no matching wiki page",
        operations=[FixOperation(
            kind="update_page", path="wiki/multi.md",
            new_frontmatter={"title": "Multi"},
            new_body="# Multi\n\nLinks: [[Alpha]] and [[beta]] here.\n",
            expected_hash=snapshot_hash,
        )],
        rationale="r", source="heuristic",
    )
    p2 = FixProposal(
        proposal_id="p2", issue_kind="broken_wikilink",
        issue_path="wiki/multi.md",
        issue_detail="[[beta]] has no matching wiki page",
        operations=[FixOperation(
            kind="update_page", path="wiki/multi.md",
            new_frontmatter={"title": "Multi"},
            new_body="# Multi\n\nLinks: [[alpha]] and [[Beta]] here.\n",
            expected_hash=snapshot_hash,
        )],
        rationale="r", source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[p1, p2]),
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )
    # First op applies; second is skipped with a clear superseded reason
    # — not a hash-mismatch lie that misleads the user about which
    # version of the file was being targeted.
    assert len(report.applied) == 1
    assert report.applied[0].path == "wiki/multi.md"
    assert len(report.skipped) == 1
    assert "superseded" in report.skipped[0]["reason"].lower()
    # Make sure the on-disk body was not silently reverted to op #2's
    # body (which would have lost op #1's fix).
    rewritten = abs_src.read_text(encoding="utf-8")
    assert "[[Alpha]]" in rewritten
    _ = src_doc_id


@pytest.mark.asyncio
async def test_skip_filter_drops_skipped_proposals(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    storage = parametrized_storage
    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/a.md", title="A", body="# A\n",
    )
    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/b.md", title="B", body="# B\n",
    )
    a_hash = file_sha256(wiki_root / "wiki/a.md")
    b_hash = file_sha256(wiki_root / "wiki/b.md")

    p_a = FixProposal(
        proposal_id="pa", issue_kind="broken_wikilink",
        issue_path="wiki/a.md", issue_detail="d",
        operations=[FixOperation(
            kind="update_page", path="wiki/a.md",
            new_frontmatter={"title": "A"},
            new_body="# A\nv2\n", expected_hash=a_hash,
        )],
        rationale="r", source="heuristic",
    )
    p_b = FixProposal(
        proposal_id="pb", issue_kind="broken_wikilink",
        issue_path="wiki/b.md", issue_detail="d",
        operations=[FixOperation(
            kind="update_page", path="wiki/b.md",
            new_frontmatter={"title": "B"},
            new_body="# B\nv2\n", expected_hash=b_hash,
        )],
        rationale="r", source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[p_a, p_b]),
        storage=storage, wiki_root=wiki_root,
        skip=[1],  # drop p_b
        reporter=_NullReporter(),
    )
    assert len(report.applied) == 1
    assert report.applied[0].path == "wiki/a.md"


@pytest.mark.asyncio
async def test_proposal_is_atomic_subsequent_ops_skip_after_failure(
    parametrized_storage: Storage, wiki_root: Path,
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
        storage=storage, wiki_root=wiki_root,
        path="wiki/concepts/topic-a.md", title="Topic A",
        body="# Topic A\n\noriginal content\n",
    )
    # The "fat" page being split.
    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/concepts/grab-bag.md", title="Grab Bag",
        body="# Grab Bag\n\n## Topic A\n\n## Topic B\n",
    )
    grab_hash = file_sha256(wiki_root / "wiki/concepts/grab-bag.md")
    # An independent broken_wikilink fix on a *different* page that
    # must NOT be aborted by the non_atomic_page proposal failure.
    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/sibling.md", title="Sibling",
        body="# Sibling\n\nlink to [[other]]\n",
    )
    sibling_hash = file_sha256(wiki_root / "wiki/sibling.md")

    split_proposal = FixProposal(
        proposal_id="split", issue_kind="non_atomic_page",
        issue_path="wiki/concepts/grab-bag.md",
        issue_detail="page looks like multiple atomic notes glued together",
        operations=[
            # First op: create_page that collides with the seeded
            # ``wiki/concepts/topic-a.md`` — apply will skip it with
            # "file already exists at create_page path".
            FixOperation(
                kind="create_page",
                path="wiki/concepts/topic-a.md",
                new_frontmatter={"title": "Topic A", "type": "concept"},
                new_body="# Topic A\n\nchild a\n",
                expected_hash=None,
            ),
            FixOperation(
                kind="create_page",
                path="wiki/concepts/topic-b.md",
                new_frontmatter={"title": "Topic B", "type": "concept"},
                new_body="# Topic B\n\nchild b\n",
                expected_hash=None,
            ),
            # Last op: would drop the original AFTER both children.
            # Atomicity rule: must NOT execute, since op #1 skipped.
            FixOperation(
                kind="delete_page",
                path="wiki/concepts/grab-bag.md",
                expected_hash=grab_hash,
            ),
        ],
        rationale="LLM split — 2 atomic children + delete original",
        source="llm",
    )
    sibling_proposal = FixProposal(
        proposal_id="sib", issue_kind="broken_wikilink",
        issue_path="wiki/sibling.md", issue_detail="[[other]]",
        operations=[FixOperation(
            kind="update_page", path="wiki/sibling.md",
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
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )

    # The original fat page must still exist on disk — the delete_page
    # was abandoned because the proposal was aborted after op #1's skip.
    assert (wiki_root / "wiki/concepts/grab-bag.md").is_file()
    # The sibling proposal applied independently.
    sibling_after = (wiki_root / "wiki/sibling.md").read_text(encoding="utf-8")
    assert "[[Other]]" in sibling_after

    applied_paths = [op.path for op in report.applied]
    skipped_paths = [s["path"] for s in report.skipped]
    assert "wiki/concepts/grab-bag.md" not in applied_paths
    assert "wiki/sibling.md" in applied_paths
    # All three ops of the split proposal must show as skipped — no
    # half-applied state on disk, even for the first child whose path
    # WOULD have been free in isolation. Preflight catches the
    # collision before any write happens.
    assert "wiki/concepts/topic-a.md" in skipped_paths
    assert "wiki/concepts/topic-b.md" in skipped_paths
    assert "wiki/concepts/grab-bag.md" in skipped_paths
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
    assert not (wiki_root / "wiki/concepts/topic-b.md").exists()


@pytest.mark.asyncio
async def test_preflight_catches_mid_proposal_hash_drift(
    parametrized_storage: Storage, wiki_root: Path,
) -> None:
    """The earlier per-proposal abort guard only stopped *subsequent*
    ops once one failed — earlier writes already on disk stayed there.
    Preflight closes that hole: a multi-op proposal whose 2nd op would
    fail (hash drift on a sibling page) must NOT mutate the 1st op's
    target either."""
    storage = parametrized_storage
    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/page-a.md", title="Page A",
        body="# Page A\n\noriginal A\n",
    )
    await _seed_page(
        storage=storage, wiki_root=wiki_root,
        path="wiki/page-b.md", title="Page B",
        body="# Page B\n\noriginal B\n",
    )
    a_hash_correct = file_sha256(wiki_root / "wiki/page-a.md")
    # Stale hash for B (as if the proposal was generated against an
    # earlier version of page-b.md and B was edited externally since).
    b_hash_stale = "0" * 64

    proposal = FixProposal(
        proposal_id="multi", issue_kind="orphan_page",
        issue_path="wiki/page-a.md",
        issue_detail="multi-op fix",
        operations=[
            # Op #1 would succeed in isolation: hash matches A.
            FixOperation(
                kind="update_page", path="wiki/page-a.md",
                new_frontmatter={"title": "Page A"},
                new_body="# Page A\n\nupdated A\n",
                expected_hash=a_hash_correct,
            ),
            # Op #2 has a stale hash for B → preflight rejects whole proposal.
            FixOperation(
                kind="update_page", path="wiki/page-b.md",
                new_frontmatter={"title": "Page B"},
                new_body="# Page B\n\nupdated B\n",
                expected_hash=b_hash_stale,
            ),
        ],
        rationale="multi-op", source="heuristic",
    )

    report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage, wiki_root=wiki_root,
        reporter=_NullReporter(),
    )

    assert report.applied == []
    assert len(report.skipped) == 2
    # Page A on disk must still be the original, even though op #1
    # would have succeeded in isolation.
    assert "original A" in (wiki_root / "wiki/page-a.md").read_text(
        encoding="utf-8"
    )
    assert "updated A" not in (wiki_root / "wiki/page-a.md").read_text(
        encoding="utf-8"
    )
