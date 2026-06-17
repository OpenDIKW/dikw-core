"""Knowledge-layer lint fix-proposal subsystem.

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
  bytes (concurrent-edit guard), then mutates ``knowledge/`` via
  :func:`page.write_page` / unlink. Outgoing-link reconciliation rides
  the existing ``storage.replace_links_from`` machinery (PR #66).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import re
from collections.abc import Callable
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
from ..data.path_norm import doc_id_for
from ..info.tokenize import CjkTokenizer
from ..trash import move_to_trash
from .links import build_fuzzy_index, parse_links, resolve_links
from .lint import LintKind
from .page import (
    KnowledgePage,
    build_page,
    category_from_path,
    path_slug_title,
    write_page,
)
from .page_index import persist_knowledge
from .synthesize import (
    SynthesisError,
    SynthesisPartialError,
    synthesize_pages_from_text,
)

logger = logging.getLogger(__name__)


class FixerSkip(Exception):
    """Structured skip signal a fixer can raise to record a specific
    ``reason`` in :attr:`FixProposalReport.skipped` instead of the
    generic ``"fixer returned None"`` orchestrator default.

    Use cases live in the per-rule fixers — broken_wikilink raises this
    when D/I evidence is insufficient, when the LLM body fails grounding
    checks, or when the proposed page path collides with an existing
    page. Agents reading the propose-task result JSON then see the
    actual product-semantic reason (``evidence_insufficient: 0 chunks``,
    ``rejected_todo_marker``, ``path_collision: ...``) rather than a
    catch-all None.

    Caught only by :func:`run_lint_propose`. Soft failures that aren't
    product-meaningful (LLM provider outages, parse errors) continue to
    use the ``return None`` path so the skipped list stays focused on
    decisions agents care about.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


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

    kind: Literal[
        "create_page",
        "update_page",
        "delete_page",
        "reconcile_provenance",
    ]
    path: str
    new_frontmatter: dict[str, Any] | None = None
    new_body: str | None = None
    expected_hash: str | None = None
    # Carrier for ``reconcile_provenance`` only: the snapshot of
    # frontmatter ``sources:`` the fixer observed. Apply passes this
    # straight into ``storage.replace_provenance_from(doc_id, …)``; the
    # op does NOT modify the wiki file (the frontmatter is already the
    # source of truth — this op only syncs the storage index to it).
    # See ``MissingProvenanceFixer`` and ADR-0001.
    source_paths: list[str] | None = None


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
    knowledge_paths_changed: list[str] = Field(default_factory=list)
    # Chunks embedded inline as part of this apply pass (0.4.0+). When
    # the caller wires an ``embedder`` + ``text_version_id``, Phase 1
    # re-chunks every changed page through ``persist_knowledge`` with
    # the embedder attached — vectors land in the per-version vec
    # table immediately so the fixed page is retrievable on return.
    # Without an embedder, ``chunks_embedded == 0`` and the count
    # surfaces under ``chunks_pending_embedding`` instead, signalling
    # the resume scan picks up the deferred work on the next ingest.
    chunks_embedded: int = 0
    chunks_pending_embedding: int = 0
    # Pages whose Phase-1 persist raised mid-pipeline; each was
    # deactivated (``active=False``) so it stays out of retrieval, and the
    # apply continued with the remaining changed pages — parity with the
    # synth path (``SynthReport.persist_errors``) and D/W. Each entry is
    # ``{"path": str, "message": str}``.
    persist_errors: list[dict[str, Any]] = Field(default_factory=list)
    # The source ``lint.propose`` task id that produced the proposals
    # we just applied. Server runners stamp this in so the proposals
    # listing in the CLI can show which proposal tasks have been
    # applied without depending on raw task ``params`` (TaskRow only
    # exposes ``params_digest``).
    proposal_task_id: str | None = None


@dataclass(frozen=True)
class KnowledgePageMeta:
    """Lightweight knowledge-page descriptor handed to fixers.

    ``path`` + ``title`` is enough for the fuzzy-link matcher. The
    scorer for ``orphan_page`` also wants ``sources`` and ``tags`` so
    candidate-parent comparison can run without a frontmatter re-read
    per orphan; both come from the frontmatter the lint pass already
    parsed (see ``LintReport.page_meta``). Heavy fixers that need the
    body re-read it from ``ctx.base_root / path`` on demand.
    """

    path: str
    title: str | None
    sources: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class FixerContext:
    """Per-task context handed to every fixer.

    ``storage`` and ``llm`` are optional because heuristic-only fixers
    never touch them; fixers that need them raise ``ValueError`` if
    asked to run without. ``all_pages`` is pre-built by the orchestrator
    from ``storage.list_documents`` so each fixer doesn't repeat the
    round-trip. ``path_to_doc_id`` is the same listing inverted to a
    O(1) lookup so per-orphan code paths can resolve ``doc_id`` without
    re-listing the entire WIKI layer.

    ``enable_llm`` gates the evidence-backed grounded repair inside
    fixers that have one (broken_wikilink grounded repair,
    non_atomic_page splitter, orphan_page merge_into_existing_page).
    Default False keeps propose runs heuristic-only — every LLM call
    costs tokens, and a user must opt in via
    ``dikw client lint propose --enable-llm``.
    """

    storage: Storage | None
    llm: LLMProvider | None
    embedding: EmbeddingProvider | None
    base_root: Path
    all_pages: list[KnowledgePageMeta]
    enable_llm: bool = False
    cfg: DikwConfig | None = None
    path_to_doc_id: dict[str, str] = dataclasses.field(default_factory=dict)


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
    ``"[[<target>]] has no matching knowledge page"``; we re-extract the
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


def page_to_op_frontmatter(page: KnowledgePage) -> dict[str, Any]:
    """Flatten a :class:`KnowledgePage` into the dict ``FixOperation`` expects.

    The inverse of :func:`_build_page_from_op` — both fixer-side
    (forward) and apply-side (read-back) live in this module so the
    field list (``id`` / ``type`` / ``title`` / ``tags`` / ``sources``
    / ``created`` / ``updated`` / ``extras``) stays defined in one
    place. ``write_page`` is the third place that knows these keys;
    a future cleanup could route through this helper too.
    """
    fm: dict[str, Any] = {
        "id": page.id,
        "category": page.category,
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
    allowed_categories: tuple[str, ...],
    fallback: str,
    system: str,
    temperature: float = 0.3,
    log_label: str,
    strict: bool = False,
) -> list[KnowledgePage] | None:
    """LLM call + parse, with the soft-failure contract every fixer needs.

    Returns the parsed pages on success or ``None`` to signal "fixer
    should skip this issue":

    * ``SynthesisError`` (no usable ``<page>`` block) → ``None``.
    * ``SynthesisPartialError`` with ``retry=True`` (token-budget
      truncation — an unclosed ``<page>`` tag or a truncation
      ``finish_reason`` from the provider; re-running with a bigger
      budget would yield more pages) → ``None``. Always — truncation is
      recoverable, and destructive splits cannot tell whether the missing
      content was important.
    * ``SynthesisPartialError`` with ``retry=False`` (deterministic
      partial — e.g. one malformed ``<page>`` block among valid ones):
      - **strict=True (destructive callers)** → ``None``. The
        non_atomic_page splitter deletes the source after writing
        children; accepting a 3-block response with 1 malformed
        block as "2 valid children, good enough" would drop the
        malformed block's content along with the original page.
      - **strict=False (additive callers)** → ``pe.pages``. The
        broken_wikilink evidence-backed grounded repair takes only
        ``pages[0]``; a malformed sibling block does not represent
        lost content, just a wasted LLM token budget.
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
            allowed_categories=allowed_categories,
            fallback=fallback,
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
        except FixerSkip as skip:
            # Structured product-semantic skip — propagate the fixer's
            # reason verbatim so agents reading the final report see why
            # this issue stayed unrepaired.
            _record_skip(idx, issue, skip.reason)
            continue
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


def _op_title(op: FixOperation) -> str:
    """Derive the canonical title for a ``FixOperation``.

    Single source of truth for the fallback chain — a fixer that
    forgets to write ``new_frontmatter["title"]`` (or writes a
    non-string YAML scalar) still gets a stable path-slug derived
    title, so phase 0's ``title_to_path`` seed and
    ``_build_page_from_op``'s KnowledgePage construction agree.
    """
    fm = op.new_frontmatter or {}
    raw = fm.get("title")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return path_slug_title(op.path)


def _pop_str_list(fm: dict[str, Any], key: str) -> list[str]:
    """Defensive list read for the REWRITE path, popped so ``extras`` doesn't
    carry the raw value out.

    Fixers pass on-disk frontmatter through ``op.new_frontmatter`` verbatim,
    and apply REWRITES the page from this value — so unlike the read-site
    helper (``frontmatter_str_list``, which safely collapses malformed shapes
    to ``[]``), collapsing here is destructive: a hand-written scalar
    ``sources: foo.md`` would be erased from disk AND from the provenance
    table (``replace_provenance_from``), unrecoverably (the
    missing_provenance fixer backfills FROM frontmatter). Heal instead:

    * scalar string → one-item list (the old ``list(...)`` char-split bug);
    * numeric list entries (a bare year tag, ``tags: [2024]``) → stringified;
    * truly shapeless values (dict / None / bool / nested list) → dropped.
    """
    raw = fm.pop(key, None)
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                out.append(str(item))
        return out
    return []


def _build_page_from_op(op: FixOperation) -> KnowledgePage:
    """Materialise a :class:`KnowledgePage` from an op's frontmatter + body.

    Defaults (stable ``id`` from :func:`wiki.make_page_id`, ISO-now
    timestamps) come from :func:`wiki.build_page` so create / update
    paths share the same construction rules synth uses. Proposal
    frontmatter overrides those defaults when present, so a fixer that
    *does* know the canonical ``id`` (e.g. an LLM-grounded proposal
    from the ``broken_wikilink`` evidence-backed grounded repair) can
    pin it.
    """
    if op.new_body is None:
        raise ValueError(f"op {op.kind} for {op.path} missing new_body")
    fm = dict(op.new_frontmatter or {})
    title = _op_title(op)
    fm.pop("title", None)
    category = str(fm.pop("category", None) or category_from_path(op.path))
    tags = _pop_str_list(fm, "tags")
    sources = _pop_str_list(fm, "sources")
    page_id = fm.pop("id", None)
    created = fm.pop("created", None)
    updated = fm.pop("updated", None)
    page = build_page(
        title=title,
        body=op.new_body,
        category=category,
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
    base_root: Path,
    pick: list[int] | None = None,
    skip: list[int] | None = None,
    reporter: Any,  # ProgressReporter
    cjk_tokenizer: CjkTokenizer = "none",
    embedder: EmbeddingProvider | None = None,
    embedding_model: str = "",
    text_version_id: int | None = None,
    embedding_error_retries: int = 0,
    embedding_error_retry_backoff_seconds: float = 0.0,
) -> ApplyReport:
    """Mutate ``knowledge/`` per a previously-produced :class:`FixProposalReport`.

    File writes + outgoing-wikilink reconciliation always run. When the
    caller wires ``embedder`` + ``text_version_id`` (0.4.0+), Phase 1
    also embeds the rebuilt chunks inline so the fixed page is
    retrievable on return. A failed embed batch (e.g. transient
    ProviderError) does NOT abort the apply — the per-batch retry-skip
    inside ``persist_knowledge`` leaves those chunks pending, surfaced
    via ``ApplyReport.chunks_pending_embedding``, and the next
    ``dikw client ingest``'s missing-embedding resume scan reconciles
    them. Without an embedder, all rebuilt chunks land in
    ``chunks_pending_embedding`` and the resume scan does the work.

    ``pick`` / ``skip`` filter the proposal list by index. Both may be
    set; pick is applied first, then skip removes from that subset.
    """
    proposals = _filter_proposals(proposal_report.proposals, pick=pick, skip=skip)

    # Pre-load K-layer doc rows for path→doc_id and title→path resolution.
    docs = list(await storage.list_documents(layer=Layer.KNOWLEDGE, active=True))
    path_to_doc_id: dict[str, str] = {d.path: d.doc_id for d in docs}
    title_to_path: dict[str, str] = {}
    for d in docs:
        if d.title and d.title not in title_to_path:
            title_to_path[d.title] = d.path

    applied: list[FixOperation] = []
    skipped: list[dict[str, Any]] = []
    persist_errors: list[dict[str, Any]] = []
    # Phase-1 persist failures: their on-disk file was written (Phase 0) but
    # the doc was deactivated, so they must be excluded from the reported
    # ``knowledge_paths_changed`` (they are surfaced via ``persist_errors``).
    persist_failed_paths: set[str] = set()
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
            base_root=base_root,
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
                base_root=base_root,
                proposal_id=proposal.proposal_id,
                issue_kind=proposal.issue_kind,
                path_to_doc_id=path_to_doc_id,
            )
            if skip_reason is None:
                applied.append(op)
                if op.kind == "delete_page":
                    touched_paths.add(op.path)
                    deleted_paths.add(op.path)
                elif op.kind == "reconcile_provenance":
                    # No file change → no Phase 1 ``persist_knowledge``
                    # re-chunk needed, no ``knowledge_paths_changed`` entry,
                    # and crucially no ``touched_paths`` entry either:
                    # the conflict gate at line 601 exists to stop a
                    # later op clobbering an earlier op's file write,
                    # but reconcile never writes the file, so blocking
                    # a sibling ``update_page`` / ``delete_page`` on the
                    # same page (e.g., a page with both
                    # ``missing_provenance`` and ``broken_wikilink``)
                    # would be a false positive. The storage write
                    # already happened inside ``_apply_one_op``.
                    pass
                else:
                    touched_paths.add(op.path)
                    paths_changed.add(op.path)
            else:
                skipped.append(skip_reason)
                proposal_aborted = True

    # Phase 0: pre-populate ``title_to_path`` with every changed page
    # BEFORE phase 1 reconciles any of their outgoing links.
    # ``paths_changed`` iterates alphabetically, not topologically, so a
    # ``non_atomic_page`` split that creates "Topic A" + "Topic B"
    # (where Topic A's body links to ``[[Topic B]]``) would otherwise
    # see A persisted before B's title entered the resolver — A's edge
    # to B would silently drop, and phase 2 explicitly skips
    # ``paths_changed`` so the gap would never recover until the next
    # ingest. Pulling titles from ``op.new_frontmatter`` avoids any
    # extra disk reads: every applied create/update op carries a title.
    for op in applied:
        if op.kind not in ("create_page", "update_page"):
            continue
        op_title = _op_title(op)
        if op_title not in title_to_path:
            title_to_path[op_title] = op.path

    # Build the companion fuzzy index alongside ``title_to_path``.
    # Without it, persist_knowledge / resolve_links degrade to
    # exact-match only — fuzzy-resolvable links like ``[[Neural
    # Networks]] → Neural Network`` silently break inside lint apply
    # and the next lint propose flags them as broken_wikilink, causing
    # churn (code-review finding, 0.4.0). ``_persist_layered_page``
    # only auto-derives a fuzzy index when ``title_to_path is None``,
    # so callers that supply ``title_to_path`` must also supply
    # ``fuzzy_index`` explicitly.
    fuzzy_index = build_fuzzy_index(title_to_path)

    # Phase 1: persist each still-extant changed page into storage:
    # upsert document + replace_chunks + reconcile outgoing links + (if
    # an embedder is configured) embed rebuilt chunks inline. The
    # caller decides whether to wire the embedder; without one,
    # ``persist_knowledge`` leaves chunks pending and the next ``dikw
    # ingest``'s missing-embedding resume scan picks them up.
    chunks_embedded_total = 0
    chunks_pending_total = 0
    for path in sorted(paths_changed):
        if not (base_root / path).resolve().is_file():
            continue
        page_doc_id = doc_id_for(Layer.KNOWLEDGE, path)
        try:
            result = await persist_knowledge(
                storage=storage,
                root=base_root,
                path=path,
                embedder=embedder,
                embedding_model=embedding_model,
                text_version_id=text_version_id,
                cjk_tokenizer=cjk_tokenizer,
                title_to_path=title_to_path,
                fuzzy_index=fuzzy_index,
                retries=embedding_error_retries,
                backoff_seconds=embedding_error_retry_backoff_seconds,
            )
        except asyncio.CancelledError:
            # CancelledError inherits from BaseException, so the
            # ``except Exception`` below misses it. Deactivate the in-flight
            # page so cancellation doesn't strand a half-written-but-active
            # doc, then re-raise to abort the apply.
            with contextlib.suppress(Exception):
                await storage.deactivate_document(page_doc_id)
            raise
        except Exception as e:
            # A hard persist failure leaves the doc row + chunks committed
            # but links/provenance unreconciled. Deactivate so the
            # half-written page is hidden from retrieval + read_page, record
            # it, and continue with the remaining changed pages — parity
            # with the synth path and D/W. A transient embed retry-skip does
            # NOT reach here (it returns chunks_pending without raising).
            # The path is also dropped from ``knowledge_paths_changed`` below
            # (it is reported via ``persist_errors`` instead) so the report
            # doesn't claim a now-inactive page as a live change — mirroring
            # synth, whose created/updated counters exclude failed pages.
            #
            # Recovery note: the deactivated page is surfaced in
            # ``ApplyReport.persist_errors``. We deliberately do NOT write a
            # ``synth_source_failed`` marker for the page's sources here (the
            # way the synth path invalidates its own done markers): a lint fix
            # is not reproducible by re-synthesis — synth regenerates the page
            # from the D-source WITHOUT the lint edit — so auto-routing
            # recovery through default ``synth`` would silently drop the
            # user's fix. The on-disk file keeps the fix; recovery is
            # ``synth --all`` (regenerates, fix lost) or the future
            # ``dikw client reindex`` (re-persists the file as-is, fix
            # preserved) — the same K "no scan-based reindex" limitation
            # documented in CLAUDE.md. Deactivating is still strictly better
            # than the prior behaviour (an active, retrievable, half-written
            # page that leaks into retrieval).
            with contextlib.suppress(Exception):
                await storage.deactivate_document(page_doc_id)
            persist_errors.append(
                {"path": path, "message": f"{type(e).__name__}: {e}"}
            )
            persist_failed_paths.add(path)
            continue
        chunks_embedded_total += result.chunks_embedded
        chunks_pending_total += result.chunks_pending_embedding

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
        abs_path = (base_root / path).resolve()
        if not abs_path.is_file():
            continue
        doc_id = path_to_doc_id.get(path)
        if doc_id is None:
            continue
        body = frontmatter.loads(abs_path.read_text(encoding="utf-8")).content
        parsed_links_ = parse_links(body)
        resolved, _ = resolve_links(
            doc_id,
            parsed_links_,
            title_to_path=title_to_path,
            fuzzy_index=fuzzy_index,
        )
        await storage.replace_links_from(doc_id, resolved)

    return ApplyReport(
        applied=applied,
        skipped=skipped,
        persist_errors=persist_errors,
        knowledge_paths_changed=sorted(
            (paths_changed - persist_failed_paths) | deleted_paths
        ),
        chunks_embedded=chunks_embedded_total,
        chunks_pending_embedding=chunks_pending_total,
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


def _preflight_hash_gate(
    op: FixOperation,
    *,
    abs_path: Path,
    exists: Callable[[str], bool],
    sim_created: set[str],
) -> str | None:
    """Shared preflight gate for ops that mutate (or sync storage from)
    a known-on-disk page: ``update_page`` / ``delete_page`` /
    ``reconcile_provenance``.

    Returns ``None`` when the gate passes, or a per-kind error string
    (``f"{op.kind} target missing: …"`` / ``f"{op.kind} on …: hash
    mismatch …"``) the caller appends to ``ApplyReport.skipped``. The
    three call sites previously open-coded this 12-line block with
    only the message prefix differing; centralising it removes a
    correctness drift surface (e.g. one branch updating its hash check
    without the others). Hash drift is skipped when the path was
    sim-created within the same proposal — a within-proposal create
    has no stable on-disk hash yet, and the real apply pass re-verifies.

    ``exists`` is the closure ``_preflight_proposal`` builds over
    ``sim_created`` / ``sim_deleted`` so simulated state propagates;
    typed as ``Callable[[str], bool]`` to match the closure's
    ``(op_path: str) -> bool`` signature without losing strict-mypy
    coverage at the call boundary.
    """
    if not exists(op.path):
        return f"{op.kind} target missing: {op.path!r}"
    if not op.expected_hash:
        return (
            f"{op.kind} on {op.path!r} missing expected_hash "
            "— required for safety"
        )
    if op.path not in sim_created:
        actual = file_sha256(abs_path)
        if actual != op.expected_hash:
            return (
                f"{op.kind} on {op.path!r}: hash mismatch "
                "(concurrent edit detected)"
            )
    return None


def _preflight_proposal(
    *,
    proposal: FixProposal,
    base_root: Path,
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
    knowledge_dir = (base_root / "knowledge").resolve()
    sim_created: set[str] = set()
    sim_deleted: set[str] = set()

    def _exists(op_path: str) -> bool:
        if op_path in sim_deleted:
            return False
        if op_path in sim_created:
            return True
        abs_path = (base_root / op_path).resolve()
        return abs_path.is_file()

    for op in proposal.operations:
        abs_path = (base_root / op.path).resolve()
        try:
            abs_path.relative_to(knowledge_dir)
        except ValueError:
            return f"op {op.kind} path is outside knowledge/ tree: {op.path!r}"

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
            gate = _preflight_hash_gate(
                op, abs_path=abs_path, exists=_exists, sim_created=sim_created
            )
            if gate is not None:
                return gate
        elif op.kind == "delete_page":
            gate = _preflight_hash_gate(
                op, abs_path=abs_path, exists=_exists, sim_created=sim_created
            )
            if gate is not None:
                return gate
            sim_deleted.add(op.path)
            sim_created.discard(op.path)
        elif op.kind == "reconcile_provenance":
            # Doesn't change the file; doesn't change ``sim_created``
            # / ``sim_deleted`` either. Same concurrent-edit safety as
            # update_page — if the user edited ``sources:`` between scan
            # and apply, the fixer's ``source_paths`` snapshot is stale
            # and we skip, letting the next lint pass re-propose.
            gate = _preflight_hash_gate(
                op, abs_path=abs_path, exists=_exists, sim_created=sim_created
            )
            if gate is not None:
                return gate
            if op.source_paths is None:
                return (
                    f"reconcile_provenance on {op.path!r} missing "
                    "source_paths"
                )
        else:
            return f"unknown op kind {op.kind!r}"

    return None


async def _apply_one_op(
    *,
    op: FixOperation,
    storage: Storage,
    base_root: Path,
    proposal_id: str,
    issue_kind: LintKind,
    path_to_doc_id: dict[str, str],
) -> dict[str, Any] | None:
    """Execute one op. Returns ``None`` on success, or a skip-record dict
    that the caller appends to :attr:`ApplyReport.skipped` on failure."""
    abs_path = (base_root / op.path).resolve()

    # Sandbox: confine ops to ``<base>/wiki/``, not the whole base.
    # ``apply``'s contract is wiki-layer mutation only — a malformed
    # proposal with a base-relative path like ``sources/foo.md`` or
    # ``wiki/../dikw.yml`` would resolve inside the base root and
    # would pass a wider check, but those targets are outside the
    # K-layer tree we're authorised to mutate.
    knowledge_dir = (base_root / "knowledge").resolve()
    try:
        abs_path.relative_to(knowledge_dir)
    except ValueError:
        return _skip(
            proposal_id, op,
            f"refusing to operate outside knowledge/ tree: {op.path!r}",
        )

    if op.kind in ("update_page", "delete_page", "reconcile_provenance"):
        if not abs_path.is_file():
            return _skip(proposal_id, op, "file not found on disk")
        # ``expected_hash`` is the contract for these ops — a proposal
        # that omits it could no-op past the concurrent-edit guard
        # (custom / persisted reports can bypass the fixer's own
        # ``hash_bytes`` stamping). Missing hash = malformed proposal.
        # ``reconcile_provenance`` rides the same gate because the op
        # carries a frontmatter ``sources:`` snapshot whose freshness
        # depends on the file hash matching at apply time — preflight
        # already does the same check, but the TOCTOU window between
        # preflight and apply is closed only by re-checking here.
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
            write_page(base_root, page)
        except (OSError, ValueError) as e:
            return _skip(proposal_id, op, f"write_page failed: {e}")
        return None

    if op.kind == "delete_page":
        # Soft-delete: storage purge first, then move the file to
        # ``<base>/trash/knowledge/<rel>``. If the trash move fails after the
        # storage row is gone, the next ``dikw client ingest`` re-creates the
        # doc row from the file still sitting at the original path
        # (ingest is idempotent on hash). The reverse order would leave
        # an orphaned doc row pointing at a missing file — irrecoverable
        # without manual SQL.
        doc_id = path_to_doc_id.get(op.path)
        if doc_id is not None:
            await storage.delete_document(doc_id)
        try:
            move_to_trash(
                base_root=base_root, src_abs=abs_path, rel_path=op.path,
                reason=issue_kind, proposal_id=proposal_id,
            )
        except OSError as e:
            return _skip(proposal_id, op, f"trash move failed: {e}")
        return None

    if op.kind == "reconcile_provenance":
        # Sync storage to frontmatter snapshot. Frontmatter is the
        # source of truth (the knowledge tree is a user-editable Obsidian
        # vault); this op only re-runs the same reconcile that
        # ``persist_knowledge`` does on every synth / lint-apply, but
        # without needing an embedder or re-chunking. Concurrent edit
        # check up front (above) catches the case where the user
        # edited ``sources:`` between scan and apply.
        if op.source_paths is None:
            return _skip(
                proposal_id, op,
                "reconcile_provenance op missing source_paths",
            )
        doc_id = path_to_doc_id.get(op.path)
        if doc_id is None:
            return _skip(
                proposal_id, op,
                f"reconcile_provenance: no doc_id for path {op.path!r} "
                "(deactivated since scan?)",
            )
        await storage.replace_provenance_from(doc_id, op.source_paths)
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
