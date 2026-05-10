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
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import frontmatter
from pydantic import BaseModel, Field

from ...config import DikwConfig
from ...providers.base import EmbeddingProvider, LLMProvider
from ...schemas import Layer
from ...storage.base import Storage
from ..data.hashing import hash_bytes, hash_file
from .links import parse_links, resolve_links
from .lint import LintKind
from .page_index import persist_wiki_page
from .synthesize import (
    SynthesisError,
    SynthesisPartialError,
    synthesize_pages_from_text,
)
from .wiki import WikiPage, build_page, write_page

logger = logging.getLogger(__name__)


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

    ``enable_llm`` gates the LLM-fallback branch in fixers that have
    one (PR2 broken_wikilink stub-page generation). Default False keeps
    propose runs heuristic-only — every LLM call costs tokens, and a
    user must opt in via ``dikw client lint propose --enable-llm``.
    """

    storage: Storage | None
    llm: LLMProvider | None
    embedding: EmbeddingProvider | None
    wiki_root: Path
    all_pages: list[WikiPageMeta]
    enable_llm: bool = False
    # Passed only for fixers that need synth knobs (max_pages_per_group,
    # page_types, llm_model, llm_max_tokens_synth). Heuristic-only fixers
    # don't touch it; the orchestrator threads it through anyway so a
    # later fixer can read it without changing the FixerContext shape.
    cfg: DikwConfig | None = None


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


def page_to_op_frontmatter(page: WikiPage) -> dict[str, Any]:
    """Flatten a :class:`WikiPage` into the dict ``FixOperation`` expects.

    The inverse of :func:`_build_page_from_op` — both fixer-side
    (forward) and apply-side (read-back) live in this module so the
    field list (``id`` / ``type`` / ``title`` / ``tags`` / ``sources``
    / ``created`` / ``updated`` / ``extras``) stays defined in one
    place. ``write_page`` is the third place that knows these keys;
    a future cleanup could route through this helper too.
    """
    fm: dict[str, Any] = {
        "id": page.id,
        "type": page.type,
        "title": page.title,
        "tags": list(page.tags),
        "sources": list(page.sources),
        "created": page.created,
        "updated": page.updated,
    }
    fm.update(page.extras)
    return fm


async def safe_synthesize_pages(
    *,
    user_prompt: str,
    source_path: str,
    llm: LLMProvider,
    model: str,
    max_tokens: int,
    allowed_types: tuple[str, ...],
    system: str,
    temperature: float = 0.3,
    log_label: str,
    strict: bool = False,
) -> list[WikiPage] | None:
    """LLM call + parse, with the soft-failure contract every fixer needs.

    Returns the parsed pages on success or ``None`` to signal "fixer
    should skip this issue":

    * ``SynthesisError`` (no usable ``<page>`` block) → ``None``.
    * ``SynthesisPartialError`` with ``retry=True`` (max_tokens
      truncation — re-running with a bigger budget would yield more
      pages) → ``None``. Always — truncation is recoverable, and
      destructive splits cannot tell whether the missing content was
      important.
    * ``SynthesisPartialError`` with ``retry=False`` (deterministic
      partial — e.g. one malformed ``<page>`` block among valid ones):
      - **strict=True (destructive callers)** → ``None``. The
        non_atomic_page splitter deletes the source after writing
        children; accepting a 3-block response with 1 malformed
        block as "2 valid children, good enough" would drop the
        malformed block's content along with the original page.
      - **strict=False (additive callers)** → ``pe.pages``. The
        broken_wikilink stub fixer takes only ``pages[0]``; a
        malformed sibling block does not represent lost content,
        just a wasted LLM token budget.
    * Any other exception (provider outage, network, JSON drift) →
      log at WARNING + ``None``. Cancellation
      (:class:`asyncio.CancelledError`) is a ``BaseException`` and is
      not caught — serial-cancel semantics still hold.

    ``log_label`` identifies the calling fixer in the warning line so
    operators can grep for which rule blew up.
    """
    try:
        return await synthesize_pages_from_text(
            user_prompt=user_prompt,
            source_path=source_path,
            llm=llm,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            allowed_types=allowed_types,
            system=system,
        )
    except SynthesisPartialError as pe:
        if pe.retry:
            logger.info(
                "%s LLM response was truncated for %s — refusing partial "
                "result (retry on next propose pass with a larger budget)",
                log_label,
                source_path,
            )
            return None
        if strict:
            logger.info(
                "%s LLM response was a deterministic partial for %s — "
                "refusing in strict mode (destructive caller cannot tell "
                "whether the malformed block carried important content)",
                log_label,
                source_path,
            )
            return None
        return pe.pages
    except SynthesisError:
        return None
    except Exception as e:
        logger.warning(
            "%s LLM call failed for %s: %s", log_label, source_path, e
        )
        return None


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

    # Per-proposal preflight: simulate every op against current disk
    # state before mutating anything. Catches the "create_page #1
    # succeeds, create_page #2 collides, delete_page wipes the source
    # we already half-replaced" path. Real all-or-nothing requires
    # rollback, but preflight catches the common deterministic failures
    # — collisions, missing files, hash drift — without growing a WAL.
    # ``touched_paths`` from earlier proposals also feed in: a sibling
    # proposal that already mutated path X means a later proposal
    # acting on X will be flagged here.
    preflight_skips: dict[int, list[dict[str, Any]]] = {}
    for idx, proposal in enumerate(proposals):
        preflight_reason = _preflight_proposal(
            proposal=proposal,
            wiki_root=wiki_root,
            already_touched=touched_paths,
        )
        if preflight_reason is not None:
            preflight_skips[idx] = [
                _skip(proposal.proposal_id, op, preflight_reason)
                for op in proposal.operations
            ]

    for idx, proposal in enumerate(proposals):
        if idx in preflight_skips:
            for record in preflight_skips[idx]:
                op_counter += 1
                await reporter.progress(
                    phase="lint_apply",
                    current=op_counter,
                    total=total_ops,
                    detail={
                        "op": record["op"],
                        "path": record["path"],
                        "preflight_failed": True,
                    },
                )
                skipped.append(record)
            continue

        # Per-proposal atomicity: even after preflight, an op can still
        # fail at apply time (race between preflight and write, OS
        # error, sandbox refusal). Once any op in a proposal skips,
        # abandon the rest — half a fix is worse than no fix. The
        # remaining-ops loop below records them as skipped without
        # mutating anything else, but earlier successful writes in this
        # proposal stay on disk (no rollback). Sibling proposals are
        # unaffected; preflight already isolated them.
        proposal_aborted = False
        for op in proposal.operations:
            op_counter += 1
            reporter.cancel_token().raise_if_cancelled()
            await reporter.progress(
                phase="lint_apply",
                current=op_counter,
                total=total_ops,
                detail={"op": op.kind, "path": op.path},
            )
            if proposal_aborted:
                skipped.append(
                    _skip(
                        proposal.proposal_id, op,
                        "skipped — earlier op in the same proposal failed; "
                        "re-run lint propose to retry this fix as a whole",
                    )
                )
                continue
            if op.path in touched_paths:
                skipped.append(
                    _skip(
                        proposal.proposal_id, op,
                        "superseded by earlier op on the same path in this apply pass — "
                        "re-run lint propose to refresh remaining fixes",
                    )
                )
                proposal_aborted = True
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
                proposal_aborted = True

    # Phase 1: persist each still-extant changed page into storage:
    # upsert document + replace_chunks + reconcile outgoing links.
    # ``embedder=None`` keeps apply provider-free — the next ``dikw
    # ingest`` will detect ``doc.hash`` drift and re-embed.
    #
    # We grow ``title_to_path`` incrementally as we persist. A proposal
    # that creates pages A and B where B's body links to ``[[A]]`` would
    # otherwise see B's wikilink unresolved (A enters storage AFTER B's
    # persist call started against the pre-apply snapshot). Updating
    # the dict between calls makes intra-batch cross-links resolve.
    for path in sorted(paths_changed):
        if not (wiki_root / path).resolve().is_file():
            continue
        _, page_title = await persist_wiki_page(
            storage=storage,
            root=wiki_root,
            path=path,
            embedder=None,
            embedding_model="",
            text_version_id=None,
            title_to_path=title_to_path,
        )
        if page_title and page_title not in title_to_path:
            title_to_path[page_title] = path

    # Phase 2: re-reconcile referrers — every proposal's ``issue.path``
    # whose body references a page this batch may have just created.
    # Without this, a ``broken_wikilink`` → ``create_page`` proposal
    # leaves the source page's storage links stale: the new edge never
    # lands, ``run_lint`` reads ``links_from(source)`` and concludes
    # the freshly-created page is an orphan even though the body
    # clearly links to it. The referrer page itself wasn't mutated, so
    # we only need a link-only reconcile (no re-chunk, no document
    # row touch — those would invalidate chunk_ids and orphan
    # embeddings). Skip referrers already covered by phase 1 or
    # deleted by this pass.
    referrer_paths = {
        proposal.issue_path for proposal in proposals
    } - paths_changed - deleted_paths
    for path in sorted(referrer_paths):
        abs_path = (wiki_root / path).resolve()
        if not abs_path.is_file():
            continue
        doc_id = path_to_doc_id.get(path)
        if doc_id is None:
            continue
        body = frontmatter.loads(abs_path.read_text(encoding="utf-8")).content
        parsed_links_ = parse_links(body)
        resolved, _ = resolve_links(
            doc_id, parsed_links_, title_to_path=title_to_path,
        )
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


def _preflight_proposal(
    *,
    proposal: FixProposal,
    wiki_root: Path,
    already_touched: set[str],
) -> str | None:
    """Validate every op of a proposal against current disk state.

    Returns ``None`` when the whole proposal would apply cleanly, or a
    short reason string explaining the first op that would fail. The
    real apply pass (:func:`_apply_one_op`) re-checks each condition;
    this preflight exists so that a multi-op proposal whose 2nd op
    cannot succeed never lets its 1st op land on disk.

    Simulates op effects within the proposal so a ``create_page`` then
    ``update_page`` on the same path is recognised as valid (the
    create makes the file exist for the update). Cross-proposal state
    is captured via ``already_touched`` — any path mutated by a prior
    proposal in the same apply pass causes immediate failure here.
    """
    wiki_dir = (wiki_root / "wiki").resolve()
    sim_created: set[str] = set()
    sim_deleted: set[str] = set()

    def _exists(op_path: str) -> bool:
        if op_path in sim_deleted:
            return False
        if op_path in sim_created:
            return True
        abs_path = (wiki_root / op_path).resolve()
        return abs_path.is_file()

    for op in proposal.operations:
        abs_path = (wiki_root / op.path).resolve()
        try:
            abs_path.relative_to(wiki_dir)
        except ValueError:
            return f"op {op.kind} path is outside wiki/ tree: {op.path!r}"

        if op.path in already_touched:
            return (
                f"op {op.kind} on {op.path!r} would conflict with a "
                "sibling proposal that already mutated this path"
            )

        if op.kind == "create_page":
            if _exists(op.path):
                return f"create_page would collide: {op.path!r} already exists"
            sim_created.add(op.path)
            sim_deleted.discard(op.path)
        elif op.kind == "update_page":
            if not _exists(op.path):
                return f"update_page target missing: {op.path!r}"
            if not op.expected_hash:
                return (
                    f"update_page on {op.path!r} missing expected_hash "
                    "— required for safety"
                )
            # Hash drift only meaningful against current on-disk bytes.
            # A simulated post-create file can't have a stable hash to
            # check against, so skip the hash check on within-proposal
            # creates. Real apply will compute and verify.
            if op.path not in sim_created:
                actual = file_sha256(abs_path)
                if actual != op.expected_hash:
                    return (
                        f"update_page on {op.path!r}: hash mismatch "
                        "(concurrent edit detected)"
                    )
        elif op.kind == "delete_page":
            if not _exists(op.path):
                return f"delete_page target missing: {op.path!r}"
            if not op.expected_hash:
                return (
                    f"delete_page on {op.path!r} missing expected_hash "
                    "— required for safety"
                )
            if op.path not in sim_created:
                actual = file_sha256(abs_path)
                if actual != op.expected_hash:
                    return (
                        f"delete_page on {op.path!r}: hash mismatch "
                        "(concurrent edit detected)"
                    )
            sim_deleted.add(op.path)
            sim_created.discard(op.path)
        else:
            return f"unknown op kind {op.kind!r}"

    return None


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
