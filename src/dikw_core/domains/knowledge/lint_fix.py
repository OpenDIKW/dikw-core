"""Wiki-lint fix-proposal subsystem.

`run_lint` (in :mod:`lint`) reports four classes of K-layer hygiene
issues but never proposes how to fix them. This module adds a
``propose`` / ``apply`` pair so each lint issue can become a structured,
reviewable, applicable repair plan.

The contract is:

* :class:`Fixer` — Protocol implemented per ``LintKind``. Given a
  :class:`LintIssue` and a :class:`FixerContext`, returns a
  :class:`FixProposal` (or ``None`` if the issue isn't fixable).
* :func:`run_lint_propose` — orchestrator. Single task, serial loop:
  one ``ProgressEvent`` per issue, one ``LogEvent`` per skipped issue.
  Fixer-level failures don't fail the whole task — they accumulate in
  :attr:`FixProposalReport.skipped` and the loop moves on.
* :func:`run_lint_apply` — executor. Reads a :class:`FixProposalReport`
  produced earlier, optionally filters by ``pick`` / ``skip``, validates
  each :attr:`FixOperation.expected_hash` against the on-disk file
  bytes (concurrent-edit guard), then mutates ``wiki/`` via
  :func:`wiki.write_page` / unlink. Outgoing-link reconciliation rides
  the existing ``storage.replace_links_from`` machinery (PR #66).
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import frontmatter
from pydantic import BaseModel, Field

from ...providers.base import EmbeddingProvider, LLMProvider
from ...schemas import Layer
from ...storage.base import Storage
from ..data.hashing import hash_bytes, hash_file
from ..data.path_norm import normalize_path
from .links import parse_links, resolve_links
from .lint import LintKind
from .wiki import WikiPage, build_page, write_page


class FixOperation(BaseModel):
    """One page-level mutation in a fix proposal.

    Page-level (not line-level) is the right granularity here: the four
    rule kinds disagree on what "fix" means (rewrite a link, split into
    N pages, merge two pages, inject inbound links). A uniform
    "create / update / delete the whole page" vocabulary covers all
    cases without inventing a per-rule DSL — the cost is that even a
    one-line rewrite ships the full new body, which keeps apply trivial
    and lets an editor diff the proposal exactly like a normal commit.

    ``expected_hash`` is the sha256 of the file bytes the fixer
    observed; apply fails this op if the file has changed underneath
    (concurrent edit) and records the mismatch in ``ApplyReport.skipped``.
    Always ``None`` for ``create_page`` (the file shouldn't exist yet).
    """

    kind: Literal["create_page", "update_page", "delete_page"]
    path: str
    new_frontmatter: dict[str, Any] | None = None
    new_body: str | None = None
    expected_hash: str | None = None


class FixProposal(BaseModel):
    """One repair proposal targeting a single :class:`LintIssue`.

    The four issue fields are denormalised copies of the source
    :class:`LintIssue` (which is a frozen dataclass in :mod:`lint`,
    not a pydantic model). Embedding the values directly keeps the
    proposal record self-contained when the source lint result has
    long since been discarded — propose tasks ship to ``tasks.result``
    JSON dicts and apply tasks read them back days later.
    """

    proposal_id: str
    issue_kind: LintKind
    issue_path: str
    issue_detail: str
    issue_line: int | None = None
    operations: list[FixOperation]
    rationale: str
    source: Literal["heuristic", "llm"]


class FixProposalReport(BaseModel):
    proposals: list[FixProposal] = Field(default_factory=list)
    skipped: list[dict[str, Any]] = Field(default_factory=list)


class ApplyReport(BaseModel):
    applied: list[FixOperation] = Field(default_factory=list)
    skipped: list[dict[str, Any]] = Field(default_factory=list)
    wiki_paths_changed: list[str] = Field(default_factory=list)
    # The source ``lint.propose`` task id that produced the proposals
    # we just applied. Server runners stamp this in so the proposals
    # listing in the CLI can show which proposal tasks have been
    # applied without depending on raw task ``params`` (TaskRow only
    # exposes ``params_digest``).
    proposal_task_id: str | None = None


@dataclass(frozen=True)
class WikiPageMeta:
    """Lightweight wiki-page descriptor handed to fixers.

    Title + path is enough for PR1's broken_wikilink fuzzy matcher and
    the metadata browsing PR2 fixers (orphan / duplicate) need. Heavy
    fixers that operate on a page body re-read the body from
    ``ctx.wiki_root / path`` on demand; we don't hold every K-layer
    page's body in memory for the duration of the propose task.
    """

    path: str
    title: str | None


@dataclass(frozen=True)
class FixerContext:
    """Per-task context handed to every fixer.

    ``storage`` and ``llm`` are optional because heuristic-only fixers
    (the PR1 broken_wikilink path) never touch them. Fixers that *do*
    need them (orphan_page, duplicate_title, non_atomic_page in later
    PRs) raise their own ``ValueError`` if asked to run without one.
    ``all_pages`` is pre-built by the orchestrator from
    ``storage.list_documents`` so each fixer doesn't repeat the round-trip.
    """

    storage: Storage | None
    llm: LLMProvider | None
    embedding: EmbeddingProvider | None
    wiki_root: Path
    all_pages: list[WikiPageMeta]


class Fixer(Protocol):
    kind: LintKind

    async def propose(
        self,
        issue: Any,  # LintIssue (dataclass, no pydantic for the source type)
        ctx: FixerContext,
        reporter: Any,  # ProgressReporter
    ) -> FixProposal | None: ...


# Helpers shared across fixers ------------------------------------------------


_BROKEN_TARGET_RE = re.compile(r"\[\[([^\]]+)\]\]")


def extract_broken_target(detail: str) -> str | None:
    """Pull the ``[[<target>]]`` substring out of a lint ``detail`` string.

    The lint scanner formats every ``broken_wikilink`` issue as
    ``"[[<target>]] has no matching wiki page"``; we re-extract the
    target rather than re-parse the body so a fixer can act on the
    issue without reloading the source file.
    """
    m = _BROKEN_TARGET_RE.match(detail.strip())
    return m.group(1).strip() if m else None


# Re-export ``hash_file`` / ``hash_bytes`` under the names the fixers and
# tests already import from this module — keeps the call sites local
# while the actual implementation lives in :mod:`domains.data.hashing`.
file_sha256 = hash_file
bytes_sha256 = hash_bytes


async def run_lint_propose(
    *,
    report: Any,  # LintReport — typed as Any to avoid circular import shenanigans
    rule: LintKind | None,
    limit: int,
    ctx: FixerContext,
    reporter: Any,  # ProgressReporter
    registry: dict[LintKind, Fixer] | None = None,
) -> FixProposalReport:
    """Single-task serial orchestrator: dispatch each lint issue to its
    registered :class:`Fixer`, collect proposals, accumulate skips.

    Failures inside one fixer never fail the whole task — they land in
    :attr:`FixProposalReport.skipped` so the apply step (or a human)
    can decide what to do. Cancellation is checked at the top of every
    iteration so a user clicking "stop" mid-loop bails cooperatively.
    """
    if registry is None:
        # Local import to avoid a top-of-module import cycle: the
        # ``lint_fixers`` package imports symbols from this module.
        from .lint_fixers import FIXER_REGISTRY

        registry = FIXER_REGISTRY

    issues = list(report.issues)
    if rule is not None:
        issues = [i for i in issues if i.kind == rule]
    issues = issues[:limit]
    total = len(issues)

    proposals: list[FixProposal] = []
    skipped: list[dict[str, Any]] = []

    def _record_skip(idx: int, issue: Any, reason: str) -> None:
        skipped.append(
            {
                "issue_index": idx,
                "issue_path": issue.path,
                "issue_kind": issue.kind,
                "reason": reason,
            }
        )

    for idx, issue in enumerate(issues):
        reporter.cancel_token().raise_if_cancelled()
        await reporter.progress(
            phase="lint_propose",
            current=idx,
            total=total,
            detail={"issue_kind": issue.kind, "path": issue.path},
        )
        fixer = registry.get(issue.kind)
        if fixer is None:
            _record_skip(idx, issue, f"no fixer registered for kind {issue.kind!r}")
            continue
        try:
            proposal = await fixer.propose(issue, ctx, reporter)
        except Exception as e:
            await reporter.log(
                "WARN",
                f"fixer for {issue.path} ({issue.kind}) raised: {e}",
            )
            _record_skip(idx, issue, f"fixer raised: {e}")
            continue
        if proposal is None:
            _record_skip(idx, issue, "fixer returned None")
            continue
        proposals.append(proposal)

    if total:
        # Final progress event lets subscribers display 100% even when
        # the last issue was skipped (no per-issue 'success' event fires).
        await reporter.progress(
            phase="lint_propose",
            current=total,
            total=total,
            detail={"done": True},
        )

    return FixProposalReport(proposals=proposals, skipped=skipped)


def _wiki_doc_id(path: str) -> str:
    """Mirror of ``api._doc_id_for(Layer.WIKI, path)`` without the cycle.

    The on-disk format uses ``"<layer>:<normalized_path>"`` as the
    canonical id; we re-implement here so :mod:`lint_fix` doesn't need
    to import from :mod:`api` (the dependency arrow points the other
    way: ``api`` may import knowledge, not the reverse).
    """
    return f"{Layer.WIKI.value}:{normalize_path(path)}"


def _build_page_from_op(op: FixOperation) -> WikiPage:
    """Materialise a :class:`WikiPage` from an op's frontmatter + body.

    Defaults (stable ``id`` from :func:`wiki.make_page_id`, ISO-now
    timestamps) come from :func:`wiki.build_page` so create / update
    paths share the same construction rules synth uses. Proposal
    frontmatter overrides those defaults when present, so a fixer that
    *does* know the canonical ``id`` (e.g. an LLM stub-page proposal in
    PR2) can pin it.
    """
    if op.new_body is None:
        raise ValueError(f"op {op.kind} for {op.path} missing new_body")
    fm = dict(op.new_frontmatter or {})
    title = str(fm.pop("title", Path(op.path).stem.replace("-", " ").title()))
    type_ = str(fm.pop("type", "note"))
    tags = list(fm.pop("tags", []) or [])
    sources = list(fm.pop("sources", []) or [])
    page_id = fm.pop("id", None)
    created = fm.pop("created", None)
    updated = fm.pop("updated", None)
    page = build_page(
        title=title,
        body=op.new_body,
        type_=type_,
        tags=tags,
        sources=sources,
        path=op.path,
        extras=fm,
    )
    overrides: dict[str, Any] = {}
    if page_id is not None:
        overrides["id"] = str(page_id)
    if created is not None:
        overrides["created"] = str(created)
    if updated is not None:
        overrides["updated"] = str(updated)
    return dataclasses.replace(page, **overrides) if overrides else page


async def run_lint_apply(
    *,
    proposal_report: FixProposalReport,
    storage: Storage,
    wiki_root: Path,
    pick: list[int] | None = None,
    skip: list[int] | None = None,
    reporter: Any,  # ProgressReporter
) -> ApplyReport:
    """Mutate ``wiki/`` per a previously-produced :class:`FixProposalReport`.

    PR1 scope: write/unlink files + reconcile outgoing wikilinks for
    updated/created pages + ``deactivate_document`` for deletes. We
    intentionally do not re-chunk / re-embed — that's a follow-up
    ``dikw ingest``'s job (it'll see ``doc.hash`` mismatch and
    re-index). Keeping apply provider-free means a heuristic-only
    proposal can land without an embedder configured.

    ``pick`` / ``skip`` filter the proposal list by index. Both may be
    set; pick is applied first, then skip removes from that subset.
    """
    proposals = _filter_proposals(proposal_report.proposals, pick=pick, skip=skip)

    # Pre-load K-layer doc rows for path→doc_id and title→path resolution.
    docs = list(await storage.list_documents(layer=Layer.WIKI, active=True))
    path_to_doc_id: dict[str, str] = {d.path: d.doc_id for d in docs}
    title_to_path: dict[str, str] = {}
    for d in docs:
        if d.title and d.title not in title_to_path:
            title_to_path[d.title] = d.path

    applied: list[FixOperation] = []
    skipped: list[dict[str, Any]] = []
    paths_changed: set[str] = set()
    deleted_paths: set[str] = set()
    # Every path mutated by an in-pass op (whether write or delete) so
    # subsequent ops on the same path can't silently revert each other.
    # Each ``new_body`` was generated against the pre-apply tree;
    # applying op #2's body on top of op #1's changes would clobber
    # the first fix. We skip rather than try to compose — the user
    # re-runs ``lint propose`` against the post-apply tree.
    touched_paths: set[str] = set()

    total_ops = sum(len(p.operations) for p in proposals)
    op_counter = 0

    for proposal in proposals:
        for op in proposal.operations:
            op_counter += 1
            reporter.cancel_token().raise_if_cancelled()
            await reporter.progress(
                phase="lint_apply",
                current=op_counter,
                total=total_ops,
                detail={"op": op.kind, "path": op.path},
            )
            if op.path in touched_paths:
                skipped.append(
                    _skip(
                        proposal.proposal_id, op,
                        "superseded by earlier op on the same path in this apply pass — "
                        "re-run lint propose to refresh remaining fixes",
                    )
                )
                continue
            skip_reason = await _apply_one_op(
                op=op,
                storage=storage,
                wiki_root=wiki_root,
                proposal_id=proposal.proposal_id,
                path_to_doc_id=path_to_doc_id,
            )
            if skip_reason is None:
                applied.append(op)
                touched_paths.add(op.path)
                if op.kind == "delete_page":
                    deleted_paths.add(op.path)
                else:
                    paths_changed.add(op.path)
            else:
                skipped.append(skip_reason)

    # Reconcile outgoing wikilinks for every still-extant changed page.
    for path in sorted(paths_changed):
        abs_path = (wiki_root / path).resolve()
        if not abs_path.is_file():
            continue
        body_only = frontmatter.loads(
            abs_path.read_text(encoding="utf-8")
        ).content
        doc_id = path_to_doc_id.get(path) or _wiki_doc_id(path)
        parsed = parse_links(body_only)
        resolved, _unresolved = resolve_links(
            doc_id, parsed, title_to_path=title_to_path
        )
        # ``replace_links_from`` requires the source doc row to exist.
        # For a freshly-created page (not in path_to_doc_id) we skip —
        # the next ``dikw ingest`` will pick it up and reconcile.
        if path in path_to_doc_id:
            await storage.replace_links_from(doc_id, resolved)

    return ApplyReport(
        applied=applied,
        skipped=skipped,
        wiki_paths_changed=sorted(paths_changed | deleted_paths),
    )


def _filter_proposals(
    proposals: list[FixProposal],
    *,
    pick: list[int] | None,
    skip: list[int] | None,
) -> list[FixProposal]:
    """Return a (pick ∩ ¬skip) slice of ``proposals``, preserving order."""
    pick_set = set(pick) if pick is not None else None
    skip_set = set(skip) if skip is not None else set()
    return [
        p
        for i, p in enumerate(proposals)
        if (pick_set is None or i in pick_set) and i not in skip_set
    ]


async def _apply_one_op(
    *,
    op: FixOperation,
    storage: Storage,
    wiki_root: Path,
    proposal_id: str,
    path_to_doc_id: dict[str, str],
) -> dict[str, Any] | None:
    """Execute one op. Returns ``None`` on success, or a skip-record dict
    that the caller appends to :attr:`ApplyReport.skipped` on failure."""
    abs_path = (wiki_root / op.path).resolve()

    # Sandbox: confine ops to ``<base>/wiki/``, not the whole base.
    # ``apply``'s contract is wiki-layer mutation only — a malformed
    # proposal with a base-relative path like ``sources/foo.md`` or
    # ``wiki/../dikw.yml`` would resolve inside the base root and
    # would pass a wider check, but those targets are outside the
    # K-layer tree we're authorised to mutate.
    wiki_dir = (wiki_root / "wiki").resolve()
    try:
        abs_path.relative_to(wiki_dir)
    except ValueError:
        return _skip(
            proposal_id, op,
            f"refusing to operate outside wiki/ tree: {op.path!r}",
        )

    if op.kind in ("update_page", "delete_page"):
        if not abs_path.is_file():
            return _skip(proposal_id, op, "file not found on disk")
        # ``expected_hash`` is the contract for these ops — a proposal
        # that omits it could no-op past the concurrent-edit guard
        # (custom / persisted reports can bypass the fixer's own
        # ``hash_bytes`` stamping). Missing hash = malformed proposal.
        if not op.expected_hash:
            return _skip(
                proposal_id, op,
                f"missing expected_hash on {op.kind} — required for safety",
            )
        actual = file_sha256(abs_path)
        if actual != op.expected_hash:
            return _skip(
                proposal_id, op,
                f"hash mismatch — concurrent edit detected "
                f"(expected {op.expected_hash[:8]}…, got {actual[:8]}…)",
            )

    if op.kind in ("create_page", "update_page"):
        if op.kind == "create_page" and abs_path.exists():
            return _skip(proposal_id, op, "file already exists at create_page path")
        try:
            page = _build_page_from_op(op)
            write_page(wiki_root, page)
        except (OSError, ValueError) as e:
            return _skip(proposal_id, op, f"write_page failed: {e}")
        return None

    if op.kind == "delete_page":
        try:
            abs_path.unlink()
        except OSError as e:
            return _skip(proposal_id, op, f"unlink failed: {e}")
        doc_id = path_to_doc_id.get(op.path)
        if doc_id is not None:
            await storage.deactivate_document(doc_id)
        return None

    return _skip(proposal_id, op, f"unknown op kind {op.kind!r}")


def _skip(proposal_id: str, op: FixOperation, reason: str) -> dict[str, Any]:
    return {
        "proposal_id": proposal_id,
        "op": op.kind,
        "path": op.path,
        "reason": reason,
    }


# These provider symbols are referenced only by ``FixerContext``'s field
# annotations; the type checker reads them at module load time but they
# don't appear in any runtime expression. Keep the imports explicit
# rather than under ``TYPE_CHECKING`` so a future runtime ``isinstance``
# check (e.g. routing logic) doesn't have to gate on string forward refs.
_ = (EmbeddingProvider, LLMProvider)
