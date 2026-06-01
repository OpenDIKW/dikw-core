"""OrphanPageFixer unit tests.

The fixer covers four strategies, picked from confidence-aware signals:

* ``mark_as_leaf`` — tested here (Step 3); writes ``lint: {skip:
  [orphan_page]}`` so the next lint pass treats the page as an
  acknowledged terminal note.
* ``link_from_existing_page`` / ``merge_into_existing_page`` /
  ``delete_page`` — tested in their respective TDD steps.

This file pins the registry contract (the fixer is wired up so
``run_lint_propose`` no longer skips orphan issues) and the mark_as_leaf
tail-end strategy that runs when no parent / merge / delete candidate is
strong enough.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import frontmatter
import pytest

from dikw_core.domains.data.path_norm import doc_id_for
from dikw_core.domains.knowledge.lint import LintIssue
from dikw_core.domains.knowledge.lint_fix import (
    FixerContext,
    FixProposal,
    KnowledgePageMeta,
)
from dikw_core.domains.knowledge.lint_fixers import FIXER_REGISTRY
from dikw_core.domains.knowledge.page import (
    build_page,
    category_from_path,
    write_page,
)

from .fakes import FakeLLM


def _make_page(title: str, body: str, **kw: Any) -> Any:
    return build_page(title=title, body=body, category="concept", **kw)


#: Filler paragraph long enough that ``_propose_delete_stub`` (40-byte
#: threshold on the stripped body) doesn't think the page is a stub.
#: Tests that exercise mark_as_leaf / link_from_existing strategies
#: need pages that aren't stubs, otherwise the stub gate (which runs
#: first in the strategy router) short-circuits them. The new
#: stub-detection tests intentionally pass shorter bodies.
_NON_STUB_FILLER = (
    "Substantive prose so this page does not look like a deletable "
    "stub during scoring. The exact wording does not matter; only the "
    "byte count past the stub threshold.\n"
)


def _ctx(
    *,
    pages: list[Any],
    base_root: Path,
    enable_llm: bool = False,
) -> FixerContext:
    return FixerContext(
        storage=None,
        llm=None,
        embedding=None,
        base_root=base_root,
        all_pages=[
            KnowledgePageMeta(
                path=p.path, title=p.title,
                sources=tuple(p.sources), tags=tuple(p.tags),
            )
            for p in pages
        ],
        enable_llm=enable_llm,
        cfg=None,
    )


class _NullReporter:
    def __init__(self) -> None:
        from dikw_core.progress import CancelToken
        self._token = CancelToken()

    async def progress(self, **_: Any) -> None:
        return None

    async def log(self, *_: Any, **__: Any) -> None:
        return None

    async def partial(self, *_: Any, **__: Any) -> None:
        return None

    def cancel_token(self) -> Any:
        return self._token


def _orphan_issue(path: str) -> LintIssue:
    return LintIssue(
        kind="orphan_page",
        path=path,
        detail="no inbound wikilinks from other K-layer pages",
    )


@pytest.mark.asyncio
async def test_orphan_page_fixer_is_registered() -> None:
    """The orchestrator looks up fixers by issue.kind; the registry must
    carry an ``orphan_page`` entry so the dispatch path doesn't fall
    through to ``no fixer registered``."""
    assert "orphan_page" in FIXER_REGISTRY


@pytest.mark.asyncio
async def test_mark_as_leaf_when_no_candidates_writes_lint_skip(
    tmp_path: Path,
) -> None:
    """A fully-isolated orphan (no shared sources / tags / parent
    candidates) lands on the ``mark_as_leaf`` strategy: a single
    ``update_page`` op patching the page's frontmatter with
    ``lint: {skip: [orphan_page], reason: ...}`` and the body
    untouched. The next lint pass will see the suppression and skip
    the page entirely."""
    base_root = tmp_path
    # An orphan with no shared signals — no other pages exist, so
    # nothing to link from / merge into.
    orphan = _make_page(
        "Isolated Note",
        "# Isolated Note\n\nA terminal observation, no obvious parent.\n",
    )
    write_page(base_root, orphan)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[orphan], base_root=base_root)
    proposal = await fixer.propose(_orphan_issue(orphan.path), ctx, _NullReporter())

    assert isinstance(proposal, FixProposal)
    assert proposal.source == "heuristic"
    assert len(proposal.operations) == 1
    op = proposal.operations[0]
    assert op.kind == "update_page"
    assert op.path == orphan.path
    assert op.new_body is not None  # body preserved
    fm = op.new_frontmatter or {}
    lint_block = fm.get("lint")
    assert isinstance(lint_block, dict)
    assert "orphan_page" in lint_block.get("skip", [])
    assert isinstance(lint_block.get("reason"), str) and lint_block["reason"]


@pytest.mark.asyncio
async def test_mark_as_leaf_preserves_existing_frontmatter(
    tmp_path: Path,
) -> None:
    """Patching ``lint.skip`` must not clobber existing frontmatter keys
    (title / type / tags / sources / any extras). The fixer reads the
    page's current frontmatter and emits a delta — not a replacement.
    """
    base_root = tmp_path
    orphan = _make_page(
        "Lone Topic",
        f"# Lone Topic\n\n{_NON_STUB_FILLER}",
        tags=["topic/lone"],
        sources=["sources/doc.md"],
        extras={"custom_field": "preserved"},
    )
    write_page(base_root, orphan)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[orphan], base_root=base_root)
    proposal = await fixer.propose(_orphan_issue(orphan.path), ctx, _NullReporter())

    assert proposal is not None
    op = proposal.operations[0]
    fm = op.new_frontmatter or {}
    # Original frontmatter survives.
    assert fm.get("title") == "Lone Topic"
    assert fm.get("category") == "concept"
    assert fm.get("tags") == ["topic/lone"]
    assert fm.get("sources") == ["sources/doc.md"]
    assert fm.get("custom_field") == "preserved"
    # Plus the new lint block.
    assert "orphan_page" in fm["lint"]["skip"]


@pytest.mark.asyncio
async def test_mark_as_leaf_appends_to_existing_lint_skip(
    tmp_path: Path,
) -> None:
    """If the page already declares ``lint.skip`` for some other rule
    (e.g. the user previously suppressed ``non_atomic_page``), the
    fixer must extend the list rather than overwrite. Idempotent on
    re-run for ``orphan_page`` — running propose twice doesn't double
    the entry."""
    base_root = tmp_path
    orphan = _make_page(
        "Already Marked",
        f"# Already Marked\n\n{_NON_STUB_FILLER}",
        extras={"lint": {"skip": ["non_atomic_page"], "reason": "intentional"}},
    )
    write_page(base_root, orphan)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[orphan], base_root=base_root)
    proposal = await fixer.propose(_orphan_issue(orphan.path), ctx, _NullReporter())

    assert proposal is not None
    op = proposal.operations[0]
    skip_list = (op.new_frontmatter or {}).get("lint", {}).get("skip", [])
    # Both kinds present, but only once each.
    assert sorted(skip_list) == ["non_atomic_page", "orphan_page"]


@pytest.mark.asyncio
async def test_link_from_existing_picks_shared_source_parent(
    tmp_path: Path,
) -> None:
    """When an orphan shares a ``sources`` entry with another K-page,
    that page becomes a strong parent candidate. The fixer emits an
    ``update_page`` on the parent appending a ``[[orphan-title]]``
    reference, NOT a ``mark_as_leaf`` proposal on the orphan."""
    base_root = tmp_path
    orphan = _make_page(
        "Detail of Phenomenon",
        "# Detail of Phenomenon\n\nA narrow observation.\n",
        sources=["sources/paper.md"],
    )
    parent = _make_page(
        "Phenomenon Overview",
        "# Phenomenon Overview\n\nThe broader topic.\n",
        sources=["sources/paper.md"],
    )
    write_page(base_root, orphan)
    write_page(base_root, parent)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[orphan, parent], base_root=base_root)
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    assert proposal.source == "heuristic"
    assert len(proposal.operations) == 1
    op = proposal.operations[0]
    # Op targets the PARENT, not the orphan — we're injecting a
    # backlink into the parent's body so the orphan becomes reachable.
    assert op.kind == "update_page"
    assert op.path == parent.path
    assert "[[Detail of Phenomenon]]" in (op.new_body or "")
    assert "link_from_existing" in proposal.rationale.lower() or \
           "link" in proposal.rationale.lower()


@pytest.mark.asyncio
async def test_link_skipped_when_no_shared_signal_falls_back_to_leaf(
    tmp_path: Path,
) -> None:
    """An orphan with no shared sources / tags / title overlap against
    any candidate must fall back to ``mark_as_leaf`` — the heuristic
    won't manufacture a parent just to silence lint."""
    base_root = tmp_path
    orphan = _make_page(
        "Random Note",
        "# Random Note\n\nUnrelated to anything else.\n",
        sources=["sources/lonely.md"],
        tags=["tag/lonely"],
    )
    unrelated = _make_page(
        "Different Topic",
        "# Different Topic\n\nNothing in common.\n",
        sources=["sources/other.md"],
        tags=["tag/other"],
    )
    write_page(base_root, orphan)
    write_page(base_root, unrelated)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[orphan, unrelated], base_root=base_root)
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    op = proposal.operations[0]
    # Falls back to leaf — op acts on the orphan, not on any other page.
    assert op.path == orphan.path
    assert "orphan_page" in (op.new_frontmatter or {}).get("lint", {}).get(
        "skip", []
    )


@pytest.mark.asyncio
async def test_link_with_existing_backlink_emits_reconcile_update(
    tmp_path: Path,
) -> None:
    """When the parent body already contains ``[[Orphan Title]]`` but
    the orphan is still flagged (storage ``links`` table is stale), the
    fixer must emit a no-content ``update_page`` so apply →
    ``persist_knowledge`` → ``replace_links_from`` reconciles storage.

    Falling to ``mark_as_leaf`` would mask the real bug (the link
    index is out of sync) and silence the user's only signal."""
    base_root = tmp_path
    orphan = _make_page(
        "Orphan Detail",
        f"# Orphan Detail\n\n{_NON_STUB_FILLER}",
        sources=["sources/shared.md"],
    )
    parent_body = (
        f"# Parent Topic\n\nMain prose with an inline [[Orphan Detail]] "
        f"backlink.\n\n{_NON_STUB_FILLER}"
    )
    parent = _make_page(
        "Parent Topic", parent_body,
        sources=["sources/shared.md"],
    )
    write_page(base_root, orphan)
    write_page(base_root, parent)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[orphan, parent], base_root=base_root)
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    assert proposal.source == "heuristic"
    assert len(proposal.operations) == 1
    op = proposal.operations[0]
    assert op.kind == "update_page"
    assert op.path == parent.path
    # Body is unchanged — apply runs persist_knowledge solely to
    # reconcile the stale storage.links_from snapshot.
    assert "[[Orphan Detail]]" in (op.new_body or "")
    assert "reconcile_links" in proposal.rationale.lower()


@pytest.mark.asyncio
async def test_reconcile_detects_aliased_wikilink_in_body(
    tmp_path: Path,
) -> None:
    """``[[Orphan Detail|Alias]]`` and ``[[Orphan Detail#Section]]`` both
    resolve to the orphan, so the reconcile-detector must treat the
    parent as already-linked and emit a no-content update — NOT
    append a bare ``[[Orphan Detail]]`` next to the aliased form."""
    base_root = tmp_path
    orphan = _make_page(
        "Orphan Detail",
        f"# Orphan Detail\n\n{_NON_STUB_FILLER}",
        sources=["sources/shared.md"],
    )
    # Aliased wikilink — the substring "[[Orphan Detail]]" is NOT in
    # the body, but the link does resolve to the orphan.
    parent_body = (
        f"# Parent Topic\n\nMain prose, see also [[Orphan Detail|the "
        f"detail page]] for context.\n\n{_NON_STUB_FILLER}"
    )
    parent = _make_page(
        "Parent Topic", parent_body,
        sources=["sources/shared.md"],
    )
    write_page(base_root, orphan)
    write_page(base_root, parent)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[orphan, parent], base_root=base_root)
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    op = proposal.operations[0]
    assert op.kind == "update_page"
    assert op.path == parent.path
    # Must NOT append a bare backlink — reconcile-only branch leaves
    # body content intact (modulo trailing-newline normalization the
    # frontmatter round-trip introduces).
    new_body = op.new_body or ""
    assert "## 相关" not in new_body, (
        "reconcile branch must NOT add the related-links heading"
    )
    assert new_body.count("[[Orphan Detail") == 1, (
        f"expected exactly one (aliased) backlink, got: {new_body!r}"
    )
    assert "reconcile_links" in proposal.rationale.lower()


@pytest.mark.asyncio
async def test_link_picks_highest_score_parent_when_multiple_candidates(
    tmp_path: Path,
) -> None:
    """If several pages exceed the LINK threshold, the fixer must pick
    the strongest. A page sharing source + tag + title token wins
    over a page sharing only a single tag."""
    base_root = tmp_path
    orphan = _make_page(
        "Engine Combustion",
        f"# Engine Combustion\n\n{_NON_STUB_FILLER}",
        sources=["sources/cars.md"],
        tags=["topic/engine", "topic/combustion"],
    )
    # Weak: shares 1 tag only.
    weak = _make_page(
        "Tire Pressure",
        f"# Tire Pressure\n\n{_NON_STUB_FILLER}",
        sources=["sources/tires.md"],
        tags=["topic/engine"],  # one shared tag
    )
    # Strong: shares source + tag + title token "engine".
    strong = _make_page(
        "Engine Architecture",
        f"# Engine Architecture\n\n{_NON_STUB_FILLER}",
        sources=["sources/cars.md"],
        tags=["topic/engine"],
    )
    for p in (orphan, weak, strong):
        write_page(base_root, p)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[orphan, weak, strong], base_root=base_root)
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    op = proposal.operations[0]
    assert op.path == strong.path, (
        "fixer must pick the highest-scoring parent (strong), not weak"
    )


@pytest.mark.asyncio
async def test_empty_stub_orphan_proposes_delete(tmp_path: Path) -> None:
    """An orphan whose body (post-frontmatter) is below the stub
    threshold and carries no outbound wikilinks is a throwaway page —
    propose ``delete_page`` (soft-delete to trash). The fixer must
    not mark such pages as intentional leaves: the user's intent
    was to discard them, the synth pass just left a stub."""
    base_root = tmp_path
    stub = _make_page("Quick Note", "tiny.\n")  # 5 chars in body
    write_page(base_root, stub)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[stub], base_root=base_root)
    proposal = await fixer.propose(
        _orphan_issue(stub.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    assert len(proposal.operations) == 1
    op = proposal.operations[0]
    assert op.kind == "delete_page"
    assert op.path == stub.path
    assert "stub" in proposal.rationale.lower() or \
           "delete" in proposal.rationale.lower()


@pytest.mark.asyncio
async def test_substantive_body_orphan_does_not_propose_delete(
    tmp_path: Path,
) -> None:
    """Pages with real prose (above the stub byte threshold) must NOT
    be auto-deleted; they fall through to link / leaf strategies.
    Guards against accidentally trashing real notes."""
    base_root = tmp_path
    real_page = _make_page(
        "Real Note",
        "# Real Note\n\n" + ("Substantive paragraph. " * 10) + "\n",
    )
    write_page(base_root, real_page)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[real_page], base_root=base_root)
    proposal = await fixer.propose(
        _orphan_issue(real_page.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    op = proposal.operations[0]
    assert op.kind != "delete_page"


@pytest.mark.asyncio
async def test_terse_note_with_metadata_does_not_propose_delete(
    tmp_path: Path,
) -> None:
    """Codex round 3 P2: a legitimate one-sentence definition under
    the byte threshold must NOT be auto-deleted. The stub gate must
    require additional signals (empty body, TODO marker, or absence
    of metadata) before proposing destruction."""
    base_root = tmp_path
    terse_note = _make_page(
        "Water Boiling Point",
        "Water boils at 100C.\n",
        sources=["sources/physics.md"],
        tags=["topic/physics"],
    )
    write_page(base_root, terse_note)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[terse_note], base_root=base_root)
    proposal = await fixer.propose(
        _orphan_issue(terse_note.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    op = proposal.operations[0]
    assert op.kind != "delete_page", (
        f"terse note WITH source+tag metadata must not be auto-deleted; "
        f"got delete_page (rationale: {proposal.rationale})"
    )


@pytest.mark.asyncio
async def test_empty_body_orphan_still_deletes(tmp_path: Path) -> None:
    """The stub gate must still fire when the body is empty after
    frontmatter is stripped — even if the page has metadata."""
    base_root = tmp_path
    empty = _make_page(
        "Placeholder",
        "",  # truly empty body
        sources=["sources/x.md"],
        tags=["topic/x"],
    )
    write_page(base_root, empty)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[empty], base_root=base_root)
    proposal = await fixer.propose(
        _orphan_issue(empty.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    assert proposal.operations[0].kind == "delete_page"


@pytest.mark.asyncio
async def test_todo_pattern_orphan_proposes_delete(tmp_path: Path) -> None:
    """A body whose only non-trivial content is a TODO/FIXME/WIP
    marker is a stub regardless of metadata richness — the marker
    itself signals "not done yet, intended to be filled in or
    discarded"."""
    base_root = tmp_path
    todo_stub = _make_page(
        "Future Topic",
        "TODO: write this page.\n",
        sources=["sources/plan.md"],
        tags=["status/draft"],
    )
    write_page(base_root, todo_stub)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[todo_stub], base_root=base_root)
    proposal = await fixer.propose(
        _orphan_issue(todo_stub.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    assert proposal.operations[0].kind == "delete_page"


@pytest.mark.asyncio
async def test_stub_with_outbound_wikilinks_does_not_propose_delete(
    tmp_path: Path,
) -> None:
    """Even a short page is meaningful if it links out — it's an index
    fragment, not a throwaway. Only body-< threshold AND zero outbound
    wikilinks qualifies as a deletable stub."""
    base_root = tmp_path
    stub_with_link = _make_page(
        "Tiny Index",
        "See [[Other Page]].\n",  # short body, but has a wikilink
    )
    write_page(base_root, stub_with_link)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[stub_with_link], base_root=base_root)
    proposal = await fixer.propose(
        _orphan_issue(stub_with_link.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    op = proposal.operations[0]
    assert op.kind != "delete_page"


@pytest.mark.asyncio
async def test_link_skipped_when_orphan_title_is_duplicated(
    tmp_path: Path,
) -> None:
    """If two K-pages share the same title, a generated ``[[title]]``
    backlink would resolve to the first page found by the title→path
    resolver — not necessarily to the orphan. Applying that link would
    leave the orphan with zero inbound links (regression of the very
    rule the proposal was meant to fix). The fixer must detect the
    duplicate-title and fall back to mark_as_leaf so the user
    resolves the duplicate first. Codex P2-3."""
    base_root = tmp_path
    # Two pages with the same title — orphan is the one without an
    # inbound link, but the resolver picks whichever it sees first.
    orphan = _make_page(
        "Phenomenon",
        f"# Phenomenon\n\n{_NON_STUB_FILLER}",
        sources=["sources/p.md"],
        path="knowledge/concept/phenomenon-a.md",
    )
    twin = _make_page(
        "Phenomenon",
        f"# Phenomenon\n\n{_NON_STUB_FILLER}",
        sources=["sources/p.md"],
        path="knowledge/concept/phenomenon-b.md",
    )
    # A shared-source candidate that WOULD be picked as parent if we
    # didn't short-circuit on the duplicate-title condition.
    parent = _make_page(
        "Phenomenon Overview",
        f"# Phenomenon Overview\n\n{_NON_STUB_FILLER}",
        sources=["sources/p.md"],
    )
    for p in (orphan, twin, parent):
        write_page(base_root, p)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[orphan, twin, parent], base_root=base_root)
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    op = proposal.operations[0]
    # Must NOT inject a backlink into parent — that link would resolve
    # to the twin, not the orphan.
    assert op.path == orphan.path, (
        f"duplicate-title orphan should fall through to mark_as_leaf; "
        f"got op on {op.path!r} (rationale: {proposal.rationale})"
    )
    assert "orphan_page" in (op.new_frontmatter or {}).get("lint", {}).get(
        "skip", []
    )


@pytest.mark.asyncio
async def test_link_excludes_orphan_itself_from_parent_candidates(
    tmp_path: Path,
) -> None:
    """The orphan must never be its own parent — even if its title
    fuzzy-matches itself (trivially does), the scorer must exclude
    self-references."""
    base_root = tmp_path
    orphan = _make_page(
        "Alone",
        "# Alone\n\nbody.\n",
        sources=["sources/x.md"],
    )
    write_page(base_root, orphan)
    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[orphan], base_root=base_root)
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    # Either mark_as_leaf (no other candidates) or some valid path,
    # but never link the orphan to itself.
    op = proposal.operations[0]
    # The orphan can be acted on (mark_as_leaf), but its body must NOT
    # contain a self-referential wikilink to its own title.
    if op.kind == "update_page" and op.path == orphan.path:
        # mark_as_leaf path — body unchanged, no self-link injected.
        assert "[[Alone]]" not in (op.new_body or "")


@pytest.mark.asyncio
async def test_link_uses_embedding_when_available_boosts_score(
    parametrized_storage: Any, tmp_path: Path,
) -> None:
    """If heuristic signals are weak (no shared sources/tags, no title
    overlap) but the orphan's chunks have very high cosine similarity
    to a candidate page's chunks, the embedding boost can push the
    candidate above LINK_THRESHOLD and the fixer picks it as parent.

    Without embedding-aware scoring, the orphan would fall through to
    mark_as_leaf — that's the contrast we're testing."""
    from dikw_core.domains.knowledge.lint_fix import FixerContext, KnowledgePageMeta
    from dikw_core.schemas import (
        ChunkRecord,
        DocumentRecord,
        EmbeddingRow,
        Layer,
    )
    from dikw_core.storage.base import NotSupported

    from .fakes import register_text_version_or_skip

    storage = parametrized_storage
    base_root = tmp_path

    # Orphan + candidate share no sources / tags / title tokens.
    orphan = _make_page(
        "Alpha", "# Alpha\n\nbody about photosynthesis pathways.\n",
    )
    candidate = _make_page(
        "Beta", "# Beta\n\nplant biochemistry overview.\n",
    )
    write_page(base_root, orphan)
    write_page(base_root, candidate)

    orphan_id = doc_id_for(Layer.KNOWLEDGE, orphan.path)
    cand_id = doc_id_for(Layer.KNOWLEDGE, candidate.path)
    for path, doc_id, title in (
        (orphan.path, orphan_id, orphan.title),
        (candidate.path, cand_id, candidate.title),
    ):
        await storage.upsert_document(
            DocumentRecord(
                doc_id=doc_id, path=path, title=title,
                hash=f"hash-{path}", mtime=0.0,
                layer=Layer.KNOWLEDGE, active=True,
            )
        )
    orphan_chunk_ids = await storage.replace_chunks(
        orphan_id,
        [ChunkRecord(doc_id=orphan_id, seq=0, start=0, end=10, text="alphabody")],
    )
    cand_chunk_ids = await storage.replace_chunks(
        cand_id,
        [ChunkRecord(doc_id=cand_id, seq=0, start=0, end=10, text="betabody")],
    )

    try:
        version_id = await register_text_version_or_skip(
            storage, dim=4, model="test-orphan-embed",
        )
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning")

    # Identical vectors → cosine distance 0 → similarity 1.0.
    await storage.upsert_embeddings(
        [
            EmbeddingRow(
                chunk_id=orphan_chunk_ids[0], version_id=version_id,
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            EmbeddingRow(
                chunk_id=cand_chunk_ids[0], version_id=version_id,
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
        ]
    )

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = FixerContext(
        storage=storage,
        llm=None,
        embedding=None,  # presence of storage + embeddings is enough
        base_root=base_root,
        all_pages=[
            KnowledgePageMeta(path=orphan.path, title=orphan.title),
            KnowledgePageMeta(path=candidate.path, title=candidate.title),
        ],
        enable_llm=False,
        cfg=None,
        path_to_doc_id={orphan.path: orphan_id, candidate.path: cand_id},
    )
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    op = proposal.operations[0]
    assert op.path == candidate.path, (
        f"embedding boost should have made {candidate.path!r} win; "
        f"got {op.path!r} (rationale: {proposal.rationale})"
    )
    assert "embed" in proposal.rationale.lower()


@pytest.mark.asyncio
async def test_link_embedding_vec_search_scoped_to_knowledge_layer(
    parametrized_storage: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``vec_search`` for the embedding leg must explicitly pass
    ``layer=Layer.KNOWLEDGE``. Without the filter, a base with many
    SOURCE/WISDOM chunks routinely fills the top-K with non-wiki hits
    that the fixer then silently discards, leaving the real wiki
    parent without an embedding boost. Regression for codex P2-2 —
    asserted directly on the storage call kwargs (the previous
    behavioural assertion was sensitive to chunk_id ordering)."""
    from dikw_core.domains.knowledge.lint_fix import FixerContext, KnowledgePageMeta
    from dikw_core.schemas import (
        ChunkRecord,
        DocumentRecord,
        EmbeddingRow,
        Layer,
    )
    from dikw_core.storage.base import NotSupported

    from .fakes import register_text_version_or_skip

    storage = parametrized_storage
    base_root = tmp_path

    orphan = _make_page("Alpha", f"# Alpha\n\n{_NON_STUB_FILLER}")
    parent = _make_page("Beta", f"# Beta\n\n{_NON_STUB_FILLER}")
    write_page(base_root, orphan)
    write_page(base_root, parent)
    orphan_id = doc_id_for(Layer.KNOWLEDGE, orphan.path)
    parent_id = doc_id_for(Layer.KNOWLEDGE, parent.path)
    for d in (
        DocumentRecord(
            doc_id=orphan_id, path=orphan.path, title=orphan.title,
            hash=f"hash-{orphan.path}", mtime=0.0, layer=Layer.KNOWLEDGE, active=True,
        ),
        DocumentRecord(
            doc_id=parent_id, path=parent.path, title=parent.title,
            hash=f"hash-{parent.path}", mtime=0.0, layer=Layer.KNOWLEDGE, active=True,
        ),
    ):
        await storage.upsert_document(d)

    orphan_chunks = await storage.replace_chunks(
        orphan_id,
        [ChunkRecord(doc_id=orphan_id, seq=0, start=0, end=4, text="alpha")],
    )
    parent_chunks = await storage.replace_chunks(
        parent_id,
        [ChunkRecord(doc_id=parent_id, seq=0, start=0, end=4, text="beta")],
    )
    try:
        version_id = await register_text_version_or_skip(
            storage, dim=4, model="test-layer-spy",
        )
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning")
    await storage.upsert_embeddings(
        [
            EmbeddingRow(
                chunk_id=orphan_chunks[0], version_id=version_id,
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            EmbeddingRow(
                chunk_id=parent_chunks[0], version_id=version_id,
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
        ]
    )

    # Spy on vec_search to capture every call's kwargs. Asserting on
    # the kwarg is robust to chunk_id ordering / sqlite-vec result
    # tiebreaks — those are what made the previous behavioural test
    # weak in the round 2 review.
    captured_kwargs: list[dict[str, Any]] = []
    real_vec_search = storage.vec_search

    async def _spy_vec_search(
        embedding: list[float], **kwargs: Any,
    ) -> Any:
        captured_kwargs.append(kwargs)
        return await real_vec_search(embedding, **kwargs)

    monkeypatch.setattr(storage, "vec_search", _spy_vec_search)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = FixerContext(
        storage=storage, llm=None, embedding=None, base_root=base_root,
        all_pages=[
            KnowledgePageMeta(path=orphan.path, title=orphan.title),
            KnowledgePageMeta(path=parent.path, title=parent.title),
        ],
        enable_llm=False, cfg=None,
        path_to_doc_id={orphan.path: orphan_id, parent.path: parent_id},
    )
    await fixer.propose(_orphan_issue(orphan.path), ctx, _NullReporter())

    assert captured_kwargs, "embedding leg never reached storage.vec_search"
    for kw in captured_kwargs:
        assert kw.get("layer") is Layer.KNOWLEDGE, (
            f"vec_search must be scoped to WIKI; got kwargs {kw!r}"
        )


@pytest.mark.asyncio
async def test_no_storage_falls_back_to_heuristic_only(tmp_path: Path) -> None:
    """When ``ctx.storage`` is None the embedding leg is skipped without
    error — pure heuristic determines the strategy."""
    base_root = tmp_path
    orphan = _make_page(
        "Solo", f"# Solo\n\n{_NON_STUB_FILLER}", sources=["sources/x.md"],
    )
    write_page(base_root, orphan)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[orphan], base_root=base_root)
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    # No other pages and no embeddings — must end up as mark_as_leaf.
    assert proposal is not None
    op = proposal.operations[0]
    assert op.path == orphan.path
    assert "orphan_page" in (op.new_frontmatter or {}).get("lint", {}).get(
        "skip", []
    )


@pytest.mark.asyncio
async def test_mark_as_leaf_apply_makes_next_lint_pass_suppress(
    parametrized_storage: Any, tmp_path: Path,
) -> None:
    """Full propose → apply → re-lint roundtrip: after the fixer's
    ``update_page`` op lands on disk and storage reconciles, a
    second ``run_lint`` no longer reports the page as ``orphan_page``
    and DOES include it in ``acknowledged_leaves``."""
    from dikw_core.domains.knowledge.lint import run_lint
    from dikw_core.domains.knowledge.lint_fix import (
        FixProposalReport,
        run_lint_apply,
    )
    from dikw_core.schemas import DocumentRecord, Layer

    storage = parametrized_storage
    base_root = tmp_path

    orphan = _make_page(
        "Pure Leaf",
        "# Pure Leaf\n\nA standalone observation worth keeping.\n",
    )
    write_page(base_root, orphan)
    await storage.upsert_document(
        DocumentRecord(
            doc_id=doc_id_for(Layer.KNOWLEDGE, orphan.path),
            path=orphan.path,
            title=orphan.title,
            hash=f"hash-{orphan.path}",
            mtime=0.0,
            layer=Layer.KNOWLEDGE,
            active=True,
        )
    )

    # Sanity: lint reports it as orphan before the fix.
    pre = await run_lint(storage, root=base_root)
    assert any(
        i.kind == "orphan_page" and i.path == orphan.path for i in pre.issues
    )
    assert orphan.path not in pre.acknowledged_leaves

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx(pages=[orphan], base_root=base_root)
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None

    apply_report = await run_lint_apply(
        proposal_report=FixProposalReport(proposals=[proposal]),
        storage=storage,
        base_root=base_root,
        reporter=_NullReporter(),
    )
    assert len(apply_report.applied) == 1, apply_report.skipped

    # On-disk frontmatter actually carries the lint.skip now.
    post = frontmatter.load(str(base_root / orphan.path))
    assert "orphan_page" in post.metadata.get("lint", {}).get("skip", [])

    post_report = await run_lint(storage, root=base_root)
    orphan_after = [
        i for i in post_report.issues
        if i.kind == "orphan_page" and i.path == orphan.path
    ]
    assert orphan_after == []
    assert orphan.path in post_report.acknowledged_leaves


# --- Step 6: merge_into_existing_page (LLM-only) ----------------------------


def _default_cfg() -> Any:
    """Minimal ``DikwConfig`` so LLM-gated branches can read
    ``ctx.cfg.provider.llm_model`` / ``ctx.cfg.schema_.category_paths()``.
    Heuristic-only tests above keep ``cfg=None``.
    """
    from dikw_core.config import DikwConfig
    return DikwConfig()


def _ctx_llm(
    *,
    pages: list[Any],
    base_root: Path,
    llm: Any,
    enable_llm: bool,
) -> FixerContext:
    return FixerContext(
        storage=None,
        llm=llm,
        embedding=None,
        base_root=base_root,
        all_pages=[
            KnowledgePageMeta(
                path=p.path, title=p.title,
                sources=tuple(p.sources), tags=tuple(p.tags),
            )
            for p in pages
        ],
        enable_llm=enable_llm,
        cfg=_default_cfg() if enable_llm else None,
    )


def _merge_response(
    *, parent_category: str, parent_slug: str, parent_title: str,
    merged_body: str, tags: list[str],
) -> str:
    """Build a ``<page>`` block string the synth parser will accept.

    The engine builds the path from ``category`` + ``slug`` itself —
    there is no ``path=`` / ``type=`` attribute. The fixer derives the
    expected category/slug from the parent's path, so callers echo
    ``category_from_path(parent.path)`` and ``Path(parent.path).stem``.
    """
    tags_yaml = ", ".join(tags) if tags else ""
    return (
        f'<page category="{parent_category}" slug="{parent_slug}">\n'
        "---\n"
        f"tags: [{tags_yaml}]\n"
        "---\n\n"
        f"# {parent_title}\n\n"
        f"{merged_body}\n"
        "</page>\n"
    )


@pytest.mark.asyncio
async def test_merge_strategy_skipped_when_llm_disabled(
    tmp_path: Path,
) -> None:
    """A pair scoring above ``MERGE_THRESHOLD`` (two shared sources)
    must still produce a heuristic ``link_from_existing`` proposal when
    ``ctx.enable_llm=False`` — merge can never run without the LLM."""
    base_root = tmp_path
    orphan = _make_page(
        "Auxiliary Detail",
        f"# Auxiliary Detail\n\n{_NON_STUB_FILLER}",
        sources=["sources/main.md", "sources/companion.md"],
    )
    parent = _make_page(
        "Main Topic",
        f"# Main Topic\n\n{_NON_STUB_FILLER}",
        sources=["sources/main.md", "sources/companion.md"],
    )
    write_page(base_root, orphan)
    write_page(base_root, parent)

    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx_llm(
        pages=[orphan, parent], base_root=base_root,
        llm=FakeLLM(), enable_llm=False,
    )
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    assert proposal.source == "heuristic"
    # Single-op link, not the two-op merge.
    assert len(proposal.operations) == 1
    op = proposal.operations[0]
    assert op.kind == "update_page"
    assert op.path == parent.path
    assert "link_from_existing" in proposal.rationale.lower()


@pytest.mark.asyncio
async def test_merge_strategy_generates_two_op_proposal(
    tmp_path: Path,
) -> None:
    """With ``enable_llm=True`` and a candidate scoring above
    ``MERGE_THRESHOLD``, the fixer emits a 2-op proposal:
    ``update_page(parent, merged_body)`` + ``delete_page(orphan)``,
    sources/tags unioned into the parent's frontmatter."""
    base_root = tmp_path
    orphan = _make_page(
        "Auxiliary Detail",
        f"# Auxiliary Detail\n\n{_NON_STUB_FILLER}\nUnique orphan fact.\n",
        sources=["sources/main.md", "sources/companion.md"],
        tags=["topic/aux"],
    )
    parent = _make_page(
        "Main Topic",
        f"# Main Topic\n\n{_NON_STUB_FILLER}\nUnique parent fact.\n",
        sources=["sources/main.md", "sources/companion.md"],
        tags=["topic/main"],
    )
    write_page(base_root, orphan)
    write_page(base_root, parent)

    merged_body = (
        "Combined overview.\n\n## Auxiliary Detail\n\n"
        "Unique orphan fact. Unique parent fact.\n"
    )
    fake = FakeLLM(response_text=_merge_response(
        parent_category=category_from_path(parent.path),
        parent_slug=Path(parent.path).stem,
        parent_title="Main Topic", merged_body=merged_body,
        tags=["topic/main", "topic/aux"],
    ))
    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx_llm(
        pages=[orphan, parent], base_root=base_root,
        llm=fake, enable_llm=True,
    )
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    assert proposal.source == "llm"
    assert len(proposal.operations) == 2
    update_op, delete_op = proposal.operations
    assert update_op.kind == "update_page"
    assert update_op.path == parent.path
    assert "Auxiliary Detail" in (update_op.new_body or "")
    # Union of sources/tags from both pages, parent-order-preserving.
    fm = update_op.new_frontmatter or {}
    assert list(fm.get("sources") or []) == [
        "sources/main.md", "sources/companion.md",
    ]
    assert "topic/main" in (fm.get("tags") or [])
    assert "topic/aux" in (fm.get("tags") or [])
    assert delete_op.kind == "delete_page"
    assert delete_op.path == orphan.path
    assert delete_op.expected_hash is not None  # concurrent-edit guard

    # Prompt should mention both paths so the LLM has the right inputs.
    assert orphan.path in (fake.last_user or "")
    assert parent.path in (fake.last_user or "")


@pytest.mark.asyncio
async def test_merge_unions_sources_preserving_parent_order(
    tmp_path: Path,
) -> None:
    """``_union_preserving_order`` must keep the parent's existing
    ordering and append orphan-only entries in orphan order. Regression
    guard: a naive ``set()`` union would non-deterministically reorder
    sources, churning the on-disk diff."""
    base_root = tmp_path
    # parent sources: [A, B]
    # orphan sources: [B, C, A]
    # expected union:   [A, B, C]  (parent first, then orphan-only items)
    orphan = _make_page(
        "Auxiliary Detail",
        f"# Auxiliary Detail\n\n{_NON_STUB_FILLER}",
        sources=["sources/b.md", "sources/c.md", "sources/a.md"],
        tags=["topic/main", "topic/aux"],
    )
    parent = _make_page(
        "Main Topic",
        f"# Main Topic\n\n{_NON_STUB_FILLER}",
        sources=["sources/a.md", "sources/b.md"],
        tags=["topic/main"],
    )
    write_page(base_root, orphan)
    write_page(base_root, parent)

    fake = FakeLLM(response_text=_merge_response(
        parent_category=category_from_path(parent.path),
        parent_slug=Path(parent.path).stem,
        parent_title="Main Topic", merged_body="merged.\n", tags=[],
    ))
    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx_llm(
        pages=[orphan, parent], base_root=base_root,
        llm=fake, enable_llm=True,
    )
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    fm = proposal.operations[0].new_frontmatter or {}
    assert list(fm.get("sources") or []) == [
        "sources/a.md", "sources/b.md", "sources/c.md",
    ]
    assert list(fm.get("tags") or []) == ["topic/main", "topic/aux"]


@pytest.mark.asyncio
async def test_merge_falls_through_to_link_when_llm_returns_no_pages(
    tmp_path: Path,
) -> None:
    """If the LLM returns garbage (``safe_synthesize_pages`` → ``None``),
    the fixer must NOT silently skip — it should still produce a link
    proposal so the orphan is reachable on the next lint pass."""
    base_root = tmp_path
    orphan = _make_page(
        "Auxiliary Detail",
        f"# Auxiliary Detail\n\n{_NON_STUB_FILLER}",
        sources=["sources/main.md", "sources/companion.md"],
    )
    parent = _make_page(
        "Main Topic",
        f"# Main Topic\n\n{_NON_STUB_FILLER}",
        sources=["sources/main.md", "sources/companion.md"],
    )
    write_page(base_root, orphan)
    write_page(base_root, parent)

    # No <page> block → SynthesisError swallowed → safe_synthesize → None.
    fake = FakeLLM(response_text="Sorry, I cannot merge these.")
    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx_llm(
        pages=[orphan, parent], base_root=base_root,
        llm=fake, enable_llm=True,
    )
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    assert proposal.source == "heuristic"
    assert len(proposal.operations) == 1
    assert proposal.operations[0].kind == "update_page"
    assert proposal.operations[0].path == parent.path


@pytest.mark.asyncio
async def test_merge_rejects_llm_response_with_wrong_target_path(
    tmp_path: Path,
) -> None:
    """The prompt requires the LLM to echo the parent's ``category`` +
    ``slug`` so the engine rebuilds the parent's exact path. If the LLM
    returns a block whose category/slug builds a different path, the merge
    proposal would silently rewrite the parent with foreign content
    while still deleting the orphan — a destructive misroute. Refuse."""
    base_root = tmp_path
    orphan = _make_page(
        "Auxiliary Detail",
        f"# Auxiliary Detail\n\n{_NON_STUB_FILLER}",
        sources=["sources/main.md", "sources/companion.md"],
    )
    parent = _make_page(
        "Main Topic",
        f"# Main Topic\n\n{_NON_STUB_FILLER}",
        sources=["sources/main.md", "sources/companion.md"],
    )
    write_page(base_root, orphan)
    write_page(base_root, parent)

    # LLM emits a valid block but a slug that builds a different path
    # (knowledge/concept/something-else.md != the parent's path).
    bogus = _merge_response(
        parent_category="concept", parent_slug="something-else",
        parent_title="Main Topic",
        merged_body="merged.\n", tags=[],
    )
    fake = FakeLLM(response_text=bogus)
    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx_llm(
        pages=[orphan, parent], base_root=base_root,
        llm=fake, enable_llm=True,
    )
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    # Merge must NOT run — fall through to link instead.
    assert proposal is not None
    assert proposal.source == "heuristic"
    assert len(proposal.operations) == 1
    assert proposal.operations[0].kind == "update_page"


@pytest.mark.asyncio
async def test_merge_rejects_llm_response_with_wrong_title(
    tmp_path: Path,
) -> None:
    """A wrong-title page block could rewrite the parent under a title
    that no longer matches the frontmatter — wikilink resolution would
    break for every existing inbound link. Refuse."""
    base_root = tmp_path
    orphan = _make_page(
        "Auxiliary Detail",
        f"# Auxiliary Detail\n\n{_NON_STUB_FILLER}",
        sources=["sources/main.md", "sources/companion.md"],
    )
    parent = _make_page(
        "Main Topic",
        f"# Main Topic\n\n{_NON_STUB_FILLER}",
        sources=["sources/main.md", "sources/companion.md"],
    )
    write_page(base_root, orphan)
    write_page(base_root, parent)

    # LLM emits the right path but a different title in the body.
    bogus = _merge_response(
        parent_category=category_from_path(parent.path),
        parent_slug=Path(parent.path).stem,
        parent_title="Different Title",  # wrong
        merged_body="merged.\n", tags=[],
    )
    fake = FakeLLM(response_text=bogus)
    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx_llm(
        pages=[orphan, parent], base_root=base_root,
        llm=fake, enable_llm=True,
    )
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    assert proposal.source == "heuristic"
    assert len(proposal.operations) == 1


@pytest.mark.asyncio
async def test_merge_rejects_body_with_title_not_at_start(
    tmp_path: Path,
) -> None:
    """The prompt requires the LLM to lead the body with ``# {title}``.
    A body that buries the heading after preamble prose would pass
    ``parse_synthesis_response``'s title extraction (which scans the
    whole body) but breaks the parent's wikilink resolution contract.
    Refuse."""
    base_root = tmp_path
    orphan = _make_page(
        "Auxiliary Detail",
        f"# Auxiliary Detail\n\n{_NON_STUB_FILLER}",
        sources=["sources/main.md", "sources/companion.md"],
    )
    parent = _make_page(
        "Main Topic",
        f"# Main Topic\n\n{_NON_STUB_FILLER}",
        sources=["sources/main.md", "sources/companion.md"],
    )
    write_page(base_root, orphan)
    write_page(base_root, parent)

    # Title heading appears after preamble prose — parser still finds
    # it (synth extracts the first ATX heading anywhere), but the
    # contract says it must be at the start.
    bogus = (
        f'<page category="{category_from_path(parent.path)}" '
        f'slug="{Path(parent.path).stem}">\n'
        "Merged below.\n\n# Main Topic\n\nbody.\n"
        "</page>\n"
    )
    fake = FakeLLM(response_text=bogus)
    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx_llm(
        pages=[orphan, parent], base_root=base_root,
        llm=fake, enable_llm=True,
    )
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    assert proposal.source == "heuristic"
    assert len(proposal.operations) == 1


@pytest.mark.asyncio
async def test_merge_uses_strict_mode_refuses_multi_block_partial(
    tmp_path: Path,
) -> None:
    """Merge is destructive — it deletes the orphan after rewriting the
    parent. A deterministic-partial response (one valid <page> plus a
    malformed one) must NOT be accepted: the malformed block's content
    would be silently dropped along with the original orphan."""
    base_root = tmp_path
    orphan = _make_page(
        "Auxiliary Detail",
        f"# Auxiliary Detail\n\n{_NON_STUB_FILLER}",
        sources=["sources/main.md", "sources/companion.md"],
    )
    parent = _make_page(
        "Main Topic",
        f"# Main Topic\n\n{_NON_STUB_FILLER}",
        sources=["sources/main.md", "sources/companion.md"],
    )
    write_page(base_root, orphan)
    write_page(base_root, parent)

    # One valid block + one malformed (no ATX title) — parser raises
    # SynthesisPartialError(retry=False); strict mode rejects it.
    valid = _merge_response(
        parent_category=category_from_path(parent.path),
        parent_slug=Path(parent.path).stem,
        parent_title="Main Topic", merged_body="merged.\n", tags=[],
    )
    malformed = (
        '<page category="concept" slug="bad">\n'
        "no ATX title in this block\n</page>\n"
    )
    fake = FakeLLM(response_text=valid + malformed)
    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx_llm(
        pages=[orphan, parent], base_root=base_root,
        llm=fake, enable_llm=True,
    )
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    # Strict mode + multi-block: merge refuses, falls to link.
    assert proposal is not None
    assert proposal.source == "heuristic"
    assert len(proposal.operations) == 1


@pytest.mark.asyncio
async def test_merge_skipped_when_score_below_merge_threshold(
    tmp_path: Path,
) -> None:
    """A single shared source (score 3.0 — clears LINK_T but below
    MERGE_T) must NOT go through the LLM merge path, even when the
    LLM is enabled. Confirms ``enable_llm=True`` doesn't lower the
    bar for invasive operations."""
    base_root = tmp_path
    orphan = _make_page(
        "Auxiliary Detail",
        f"# Auxiliary Detail\n\n{_NON_STUB_FILLER}",
        sources=["sources/shared.md"],  # 1 source = 3.0 score
    )
    parent = _make_page(
        "Main Topic",
        f"# Main Topic\n\n{_NON_STUB_FILLER}",
        sources=["sources/shared.md"],
    )
    write_page(base_root, orphan)
    write_page(base_root, parent)

    fake = FakeLLM()  # default response — would be malformed
    fixer = FIXER_REGISTRY["orphan_page"]
    ctx = _ctx_llm(
        pages=[orphan, parent], base_root=base_root,
        llm=fake, enable_llm=True,
    )
    proposal = await fixer.propose(
        _orphan_issue(orphan.path), ctx, _NullReporter(),
    )
    assert proposal is not None
    assert proposal.source == "heuristic"
    # FakeLLM must NOT have been called — merge gate refused first.
    assert fake.call_count == 0
