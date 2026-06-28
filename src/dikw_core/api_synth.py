"""Synthesis cluster of the engine facade — the K-layer authoring leg.

``synthesize`` is the only place an LLM enters the engine: it turns D-layer
source docs into K-layer knowledge pages (Stage A 1:N fan-out per source),
persists each page (document + chunks + inline embed + links + provenance
via ``_persist_knowledge_page``), maintains the link graph, and records
activity in the ``knowledge_log`` table. The ``_existing_pages_*`` helpers
feed the synth prompt its duplicate-avoidance awareness.

rank3 cluster: imports ``api_core`` (``_with_storage``), providers, the
K-layer authoring primitives, and the leaf ``api_types.SynthReport`` —
never the ``api`` facade. ``api`` re-exports ``synthesize`` (public, in
``__all__``) plus the underscore helpers the synth tests reach for
(``_persist_knowledge_page`` / ``_synth_pages_from_source`` /
``_sr_replace`` / ``_LEGACY_BACKFILL_SENTINEL``).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import prompts
from .api_core import _with_storage
from .api_types import (
    PagePersistError,
    SynthReport,
    SynthVerifyLintFinding,
    SynthVerifyReport,
)
from .config import DikwConfig
from .domains.data.backends import UnsupportedFormat, parse_any
from .domains.data.backends.base import ParsedDocument
from .domains.data.path_norm import doc_id_for as _doc_id_for
from .domains.info.tokenize import CjkTokenizer
from .domains.knowledge.grouping import (
    derive_sections_from_chunks,
    group_sections,
)
from .domains.knowledge.links import (
    build_fuzzy_index,
    normalize_for_match,
    parse_links,
)
from .domains.knowledge.page import KnowledgePage, category_from_path, write_page
from .domains.knowledge.synthesize import (
    DEFAULT_SYNTH_SYSTEM,
    SynthesisError,
    SynthesisPartialError,
    dedup_pages_by_slug,
    parse_synthesis_response,
)
from .progress import CancelToken, NoopReporter, ProgressReporter
from .providers import (
    EmbeddingProvider,
    LLMProvider,
    TransientProviderError,
    build_llm,
)
from .schemas import (
    ChunkRecord,
    DocumentRecord,
    KnowledgeLogEntry,
    Layer,
    LinkType,
)
from .storage import Storage
from .storage.base import NotSupported
from .telemetry import DIKW_LAYER, DIKW_OP, record_synth_metrics, traced_op

logger = logging.getLogger(__name__)


@traced_op("dikw.synth", attributes={DIKW_LAYER: "knowledge", DIKW_OP: "synth"})
async def synthesize(
    path: str | Path | None = None,
    *,
    force_all: bool = False,
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    reporter: ProgressReporter | None = None,
    verify: bool = False,
    judge: bool = False,
) -> SynthReport:
    """Turn source docs into K-layer knowledge pages via the configured LLM.

    By default only source docs that have never been synthesised are
    processed; pass ``force_all=True`` to re-synthesise every source.
    Embedding of new knowledge pages is skipped when ``embedder`` is ``None``.

    ``reporter`` (optional) receives one ``progress`` event per source
    document processed for server-driven task wrappers.

    ``verify=True`` runs a deterministic post-synth self-check over the pages
    this run created/updated (persist + lint filtered to this run's pages +
    semantic-duplicate) and folds a :class:`SynthVerifyReport` into the
    returned report (``None`` otherwise). It only READS synth output — the
    generated pages are unchanged. See :class:`SynthVerifyReport` for the
    gated legs.

    ``judge=True`` (only meaningful with ``verify=True``) additionally runs the
    optional, **report-only** grounding/entailment leg: it samples this run's
    page claims, grounds each against its source chunks, and asks the synth LLM
    whether the evidence entails the claim. The ratio is surfaced but never
    folded into ``passed`` (see :class:`SynthVerifyReport`). It loud-skips when
    no embedder is wired.
    """
    cfg, root, storage = await _with_storage(path)
    _reporter: ProgressReporter = reporter or NoopReporter()
    try:
        _llm = llm or build_llm(cfg.provider, base_root=root)

        text_version_id: int | None = None
        text_embed_model = cfg.provider.embedding_model
        embed_deferred_no_version = False
        if embedder is not None:
            # Synthesize must NOT register a new embed version: it only
            # writes knowledge-page chunks, so flipping active here would strand
            # source-chunk vectors in the now-inactive table and gut dense
            # retrieval. Re-embedding the full corpus belongs to ingest.
            try:
                active_text = await storage.get_active_embed_version(modality="text")
            except NotSupported:
                active_text = None
            if active_text is not None and active_text.version_id is not None:
                text_version_id = active_text.version_id
                text_embed_model = active_text.model
            else:
                # An embedder was wired but there's no active text version to
                # embed against (e.g. a base that never ran a full ingest).
                # Defer: pages are authored without inline vectors and the next
                # ingest's resume scan reconciles them. Flagged here; warned
                # below only if there's actually source material to synth.
                embedder = None  # no active text version → nothing to embed against
                embed_deferred_no_version = True

        sources = list(await storage.list_documents(layer=Layer.SOURCE, active=True))
        # Heads-up only when there's real work: an embedder was wired but no
        # active text version exists, so any pages synthesized below land
        # without inline vectors (a future ingest's resume scan reconciles
        # them) — warn rather than silently shipping vector-less K pages. Gated
        # on ``sources`` so a no-op synth on an empty base stays quiet.
        if embed_deferred_no_version and sources:
            logger.warning(
                "synth: embedder wired but no active text version; %d source(s) "
                "will be synthesized without inline embeddings (a future ingest "
                "will reconcile their vectors)",
                len(sources),
            )
        already: set[str] = set()
        if not force_all:
            # ``synth_source_done`` is the post-fan-out source-completion
            # marker: per-page ``synth`` log rows can no longer be used
            # because (a) a fan-out source with one failed group + one
            # successful group writes a ``dst`` row but is NOT done, and
            # (b) a source with a legal zero-page response writes no
            # ``dst`` row at all but IS done.
            #
            # Upgrade compatibility uses a sentinel row
            # (``src=_LEGACY_BACKFILL_SENTINEL``) to record "this base has
            # already gone through the legacy-row backfill at least once".
            # Without the sentinel we can't distinguish a *legacy* dst row
            # from a *post-fan-out crash* dst row — and treating the latter
            # as legacy would silently mark crashed sources done. The
            # sentinel is written unconditionally on the first post-fan-out
            # run so any later crash leaves us in the "sentinel already
            # exists, do not backfill" state.
            has_legacy_backfill_sentinel = False
            legacy_dst_sources: set[str] = set()
            # Sources carrying a ``synth_source_failed`` marker. A failed
            # marker is always newer than any legacy per-page ``synth`` row, so
            # such a source must be excluded from the one-time legacy backfill
            # below — otherwise the backfill would re-mark it done and strand
            # the page its failure deactivated (codex review round 2).
            failed_sources: set[str] = set()
            for entry in await storage.list_knowledge_log():
                if entry.action == "synth_source_done":
                    if entry.src == _LEGACY_BACKFILL_SENTINEL:
                        has_legacy_backfill_sentinel = True
                    elif entry.src:
                        already.add(entry.src)
                elif entry.action == "synth_source_failed" and entry.src:
                    # A page persist failure invalidates a prior
                    # ``synth_source_done`` for the same source.
                    # ``list_knowledge_log`` is ordered ``ts ASC, id ASC``, so
                    # a failed marker appended after the done marker discards
                    # it (last-writer-wins); a later successful synth re-adds
                    # the done marker and re-populates this set. Without this,
                    # a ``synth --all`` re-synth whose page failed would leave
                    # the stale done marker in place and the next default
                    # synth would skip the source, leaving the page parked
                    # inactive until a separate ``untracked_file`` drift-lint
                    # reindex — the synth path should self-heal on its own.
                    already.discard(entry.src)
                    failed_sources.add(entry.src)
                elif entry.action == "synth" and entry.src and entry.dst:
                    legacy_dst_sources.add(entry.src)
            if not has_legacy_backfill_sentinel:
                ts = time.time()
                # Write the sentinel FIRST so even if the backfill loop
                # below crashes we never re-enter the backfill arm.
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=ts,
                        action="synth_source_done",
                        src=_LEGACY_BACKFILL_SENTINEL,
                        note=(
                            "fan-out pipeline initialised — subsequent runs "
                            "will not backfill legacy per-page synth rows"
                        ),
                    )
                )
                # Exclude sources with a ``synth_source_failed`` marker: that
                # marker postdates the legacy synth rows and means the source
                # has a deactivated page awaiting rebuild, so backfilling a
                # done marker (and adding it to ``already``) would strand it.
                backfill_sources = legacy_dst_sources - failed_sources
                for src_path in sorted(backfill_sources):
                    await storage.append_knowledge_log(
                        KnowledgeLogEntry(
                            ts=ts,
                            action="synth_source_done",
                            src=src_path,
                            note="backfilled from legacy per-page synth rows",
                        )
                    )
                already |= backfill_sources

        report = SynthReport()
        # ``verify`` accumulates the pages this run persisted successfully
        # (created or updated, NOT the deactivated persist-failures) so the
        # post-synth self-check scopes its lint + duplicate legs to exactly
        # this run's output. Appended to only when ``verify`` is set, so a
        # non-verify run never grows this list past the empty allocation.
        produced_pages: list[KnowledgePage] = []
        tmpl = prompts.resolve(
            "synthesize", override_path=cfg.synth.prompt_path, base_root=root
        )
        total_sources = len(sources)

        for idx, src in enumerate(sources, start=1):
            _reporter.cancel_token().raise_if_cancelled()
            report = _sr_replace(report, candidates=report.candidates + 1)
            if src.path in already:
                report = _sr_replace(report, skipped=report.skipped + 1)
                await _reporter.progress(
                    phase="synth",
                    current=idx,
                    total=total_sources,
                    detail={"path": src.path, "outcome": "skipped"},
                )
                continue

            parsed = _read_source_parsed(root, src)
            if parsed is None:
                report = _sr_replace(report, errors=report.errors + 1)
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=time.time(),
                        action="synth",
                        src=src.path,
                        note="source body missing on disk",
                    )
                )
                await _reporter.progress(
                    phase="synth",
                    current=idx,
                    total=total_sources,
                    detail={"path": src.path, "outcome": "missing_body"},
                )
                continue

            # If the source on disk drifted since ingest (user edited it
            # without re-running ``dikw client ingest``), the cached chunk offsets
            # would slice the new body at stale boundaries — silently
            # dropping appended content and marking the source done. Bail
            # out with a clear log and let the user re-ingest.
            if parsed.hash != src.hash:
                report = _sr_replace(report, errors=report.errors + 1)
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=time.time(),
                        action="synth",
                        src=src.path,
                        note=(
                            "source body changed since last ingest — "
                            "re-run `dikw client ingest` before `dikw client synth`"
                        ),
                    )
                )
                await _reporter.progress(
                    phase="synth",
                    current=idx,
                    total=total_sources,
                    detail={"path": src.path, "outcome": "stale_chunks"},
                )
                continue

            body = parsed.body
            src_chunks = await storage.list_chunks(
                _doc_id_for(Layer.SOURCE, src.path)
            )
            outcome = await _synth_pages_from_source(
                llm=_llm,
                template=tmpl,
                cfg=cfg,
                source_path=src.path,
                source_body=body,
                chunks=src_chunks,
                cancel=_reporter.cancel_token(),
                storage=storage,
                text_version_id=text_version_id,
                force_all=force_all,
                reporter=_reporter,
            )
            report = _sr_replace(
                report,
                groups_processed=report.groups_processed + outcome.groups_processed,
                errors=report.errors + outcome.parse_errors,
            )
            for note in outcome.log_notes:
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=time.time(), action="synth", src=src.path, note=note
                    )
                )

            if outcome.groups_processed == 0:
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=time.time(),
                        action="synth",
                        src=src.path,
                        note="no chunks to synthesise from",
                    )
                )
                # Source is "done" — re-running synth on a source that
                # has no chunks would just hit the same dead-end. Mark
                # it complete so default ``synth`` skips it next time.
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=time.time(),
                        action="synth_source_done",
                        src=src.path,
                    )
                )
                report = _sr_replace(
                    report, sources_processed=report.sources_processed + 1
                )
                await _reporter.progress(
                    phase="synth",
                    current=idx,
                    total=total_sources,
                    detail={"path": src.path, "outcome": "no_chunks"},
                )
                continue

            deduped = dedup_pages_by_slug(
                outcome.pages, strategy=cfg.synth.slug_dedup
            )
            # Count how many pages the dedup collapsed — the over-generation
            # signal surfaced on ``SynthReport.slug_merge_count`` (and, as a
            # normalised fraction, the synth eval's ``synth/slug_merge_ratio_max``
            # diagnostic). One source's fan-out can emit the same
            # ``<category>/<slug>`` twice; each extra copy is one merge.
            merged = len(outcome.pages) - len(deduped)
            if merged:
                report = _sr_replace(
                    report, slug_merge_count=report.slug_merge_count + merged
                )

            # Build the title→path index ONCE for this batch and seed it
            # with the deduped pages — without that seeding, page A → page B
            # wikilinks fan-out produces from the same source would only
            # resolve after B was already upserted.
            title_to_path: dict[str, str] = {}
            if deduped:
                for d in await storage.list_documents(
                    layer=Layer.KNOWLEDGE, active=True
                ):
                    if d.title and d.title not in title_to_path:
                        title_to_path[d.title] = d.path
                for page in deduped:
                    title_to_path.setdefault(page.title, page.path)
            fuzzy_index = build_fuzzy_index(title_to_path) if deduped else None

            created_for_src = 0
            updated_for_src = 0
            src_persist_failed = False
            for page in deduped:
                doc_id = _doc_id_for(Layer.KNOWLEDGE, page.path)
                pre_existing = await storage.get_document(doc_id)
                write_page(root, page)
                try:
                    page_unresolved = await _persist_knowledge_page(
                        storage=storage,
                        root=root,
                        page=page,
                        embedder=embedder,
                        embedding_model=text_embed_model,
                        text_version_id=text_version_id,
                        cjk_tokenizer=cfg.retrieval.cjk_tokenizer,
                        title_to_path=title_to_path,
                        fuzzy_index=fuzzy_index,
                        embedding_error_retries=cfg.provider.embedding_error_retries,
                        embedding_error_retry_backoff_seconds=(
                            cfg.provider.embedding_error_retry_backoff_seconds
                        ),
                    )
                except asyncio.CancelledError:
                    # CancelledError inherits from BaseException, so the
                    # ``except Exception`` below misses it. Still deactivate the
                    # in-flight page so cancellation doesn't strand a
                    # half-written-but-active doc, then re-raise to abort the
                    # run (cancellation is not "continue with the next page").
                    with contextlib.suppress(Exception):
                        await storage.deactivate_document(doc_id)
                    # Invalidate the source's prior ``synth_source_done`` too,
                    # or a cancelled ``--all`` re-synth would strand the
                    # deactivated page behind the stale done marker. Mirrors
                    # the Exception path's ``synth_source_failed`` marker
                    # (codex review round 2). A rare double-cancellation (an
                    # idempotent re-cancel or shutdown racing a user cancel)
                    # landing on this await escapes ``suppress(Exception)`` and
                    # skips the marker; we accept that window — it needs two
                    # cancels inside a sub-await gap and only strands a
                    # deactivated (non-retrievable) page, recoverable via the
                    # same reindex path.
                    with contextlib.suppress(Exception):
                        await storage.append_knowledge_log(
                            KnowledgeLogEntry(
                                ts=time.time(),
                                action="synth_source_failed",
                                src=src.path,
                            )
                        )
                    raise
                except Exception as e:
                    # A hard persist failure (replace_chunks / replace_links_from
                    # / replace_provenance_from raising, or a permanent
                    # ProviderError from inline embed) leaves the doc row +
                    # chunks committed but later steps unreconciled. Deactivate
                    # so the half-written page is hidden from every retrieval leg
                    # + read_page, then record and continue with the remaining
                    # pages — parity with D (api.ingest) and W (write_wisdom_page).
                    # A transient embed retry-skip does NOT reach here: it
                    # returns chunks_pending without raising.
                    with contextlib.suppress(Exception):
                        await storage.deactivate_document(doc_id)
                    src_persist_failed = True
                    msg = f"{type(e).__name__}: {e}"
                    report = _sr_replace(
                        report,
                        persist_errors=(
                            *report.persist_errors,
                            PagePersistError(path=page.path, message=msg),
                        ),
                    )
                    await _reporter.partial(
                        "page_error", {"path": page.path, "message": msg}
                    )
                    continue
                if page_unresolved:
                    report = _sr_replace(
                        report,
                        unresolved_wikilinks=report.unresolved_wikilinks
                        + page_unresolved,
                    )
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=time.time(),
                        action="synth",
                        src=src.path,
                        dst=page.path,
                    )
                )
                if pre_existing is None:
                    created_for_src += 1
                    report = _sr_replace(report, created=report.created + 1)
                else:
                    updated_for_src += 1
                    report = _sr_replace(report, updated=report.updated + 1)
                if verify:
                    produced_pages.append(page)

            persisted_for_src = created_for_src + updated_for_src
            report = _sr_replace(
                report, sources_processed=report.sources_processed + 1
            )
            if src_persist_failed:
                # A page was deactivated by a persist failure. Record a
                # ``synth_source_failed`` marker that invalidates any prior
                # ``synth_source_done`` for this source (applied in log order
                # by the ``already`` computation above), so the next default
                # synth re-processes the source and rebuilds the deactivated
                # page from the D-source — the ``untracked_file`` drift lint
                # would only re-project the on-disk bytes, so re-synth is the
                # synth-path recovery. Withholding the new done marker alone is
                # not enough: a ``synth --all`` re-synth of an already-done
                # source would otherwise leave the stale done marker in place.
                # Caveat: re-synth reactivates the page only if the LLM
                # re-emits it at the same slug. If the next run produces a
                # divergent page set (LLM non-determinism), the original
                # on-disk ``.md`` is left orphaned at the old slug; the
                # ``untracked_file`` drift lint re-indexes it back to
                # ``active=True`` as a standalone page.
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=time.time(),
                        action="synth_source_failed",
                        src=src.path,
                    )
                )
            elif outcome.parse_errors == 0:
                # Mark the source as fully synthesised so default ``synth``
                # skips it next run. Skip the marker when any group raised
                # a hard ``SynthesisError`` — those failures should be
                # retried (the LLM may produce parseable output next time).
                # Partial-parse outcomes don't count: the surviving pages
                # were persisted, retrying would just hit the same partial
                # response and re-emit the warning to ``knowledge_log``.
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=time.time(),
                        action="synth_source_done",
                        src=src.path,
                    )
                )
            # ``outcome`` string keeps the pre-fan-out vocabulary so
            # client-side event consumers don't need to learn new strings.
            if persisted_for_src == 0:
                outcome_str = "no_pages"
            elif created_for_src > 0:
                outcome_str = "created"
            else:
                outcome_str = "updated"
            await _reporter.progress(
                phase="synth",
                current=idx,
                total=total_sources,
                detail={
                    "path": src.path,
                    "outcome": outcome_str,
                    "pages_persisted": persisted_for_src,
                    "groups": outcome.groups_processed,
                    # A source whose page(s) failed persist still emits
                    # ``outcome="no_pages"`` (the vocabulary is kept stable) —
                    # this flag lets a stream-only consumer distinguish it from
                    # a source that legitimately produced zero pages.
                    "persist_failed": src_persist_failed,
                },
            )

        # Activity history lives in the ``knowledge_log`` storage table
        # (appended throughout this run). dikw-core no longer materialises a
        # ``knowledge/index.md`` catalogue or ``knowledge/log.md`` chronology
        # into the vault — the category folder tree is the catalogue and the
        # table is the authoritative history (see ADR-0004).
        if verify:
            # Run the post-synth self-check while storage is still open. The
            # ``embedder`` here reflects whether inline embed actually ran
            # (it is dropped to ``None`` above when there is no active text
            # version), so the duplicate leg loud-skips exactly when the
            # embeddings it would compare against do not exist.
            report = _sr_replace(
                report,
                verify=await _verify_synth_output(
                    storage=storage,
                    root=root,
                    fallback=cfg.schema_.fallback,
                    report=report,
                    produced_pages=produced_pages,
                    embedder=embedder,
                    embedding_model=text_embed_model,
                    duplicate_cosine_tau=cfg.synth.verify_duplicate_cosine_tau,
                    max_duplicate_ratio=cfg.synth.verify_max_duplicate_ratio,
                    judge=judge,
                    judge_llm=_llm,
                    judge_model=cfg.provider.llm_model,
                    judge_sample=cfg.synth.verify_judge_sample,
                ),
            )
        # Domain counters from the final report (no-op when telemetry is
        # inactive). One emission at the single success return; a cancel / hard
        # error raises before here, mirroring the GenAI-metric terminal contract.
        record_synth_metrics(
            created=report.created,
            updated=report.updated,
            unresolved_wikilinks=report.unresolved_wikilinks,
            persist_errors=len(report.persist_errors),
        )
        return report
    finally:
        await storage.close()


# Lint kinds that mark *defective new synth output* and so fail
# ``synth --verify``. ``orphan_page`` is excluded on purpose (a fresh page is
# legitimately orphan until cited — surfaced, never gated) and
# ``invalid_wisdom_status`` is wisdom-only (synth never writes W). The
# fs-drift kinds (``missing_file`` / ``stale_index`` / ``untracked_file`` /
# ``dangling_provenance``, ADR-0005) are excluded too: they describe disk↔DB
# hygiene, not a defect in the bytes synth just wrote (synth persists with a
# matching hash + row, and attributes each page to the source it just read, so
# they never fire on fresh output anyway). See :class:`SynthVerifyReport`.
_VERIFY_GATED_LINT_KINDS = frozenset(
    {
        "broken_wikilink",
        "duplicate_title",
        "non_atomic_page",
        "uncategorized",
        "missing_provenance",
        "title_slug_quality",
    }
)


async def _verify_synth_output(
    *,
    storage: Storage,
    root: Path,
    fallback: str,
    report: SynthReport,
    produced_pages: list[KnowledgePage],
    embedder: EmbeddingProvider | None,
    embedding_model: str,
    duplicate_cosine_tau: float,
    max_duplicate_ratio: float,
    judge: bool = False,
    judge_llm: LLMProvider | None = None,
    judge_model: str = "",
    judge_sample: int = 25,
) -> SynthVerifyReport:
    """Deterministic post-synth self-check over the pages THIS run produced.

    Scopes a full ``run_lint`` to ``produced_pages``, splits the issues into
    gated findings vs the informational orphan list, and (when an embedder is
    wired) measures the semantic duplicate ratio over the run's page bodies.
    See :class:`SynthVerifyReport` for the gating contract — orphans are not
    gated, and a missing embedder loud-skips the duplicate leg rather than
    silently passing it.

    ``judge=True`` adds the optional **report-only** grounding/entailment leg
    (needs both ``judge_llm`` and ``embedder``; loud-skips otherwise). It never
    affects ``passed`` — see :class:`SynthVerifyReport`.

    Local imports keep ``run_lint`` / the eval duplicate metric off the hot
    synth path when verify is not requested (and avoid any import-cycle risk
    from pulling the eval package into the engine facade at module load).
    """
    from .domains.knowledge.lint import run_lint
    from .eval.metrics import duplicate_ratio_max

    # ``produced_pages`` is appended per-page per-source, and
    # ``dedup_pages_by_slug`` only dedups WITHIN one source. Two different
    # sources can each emit the same ``<category>/<slug>.md`` (realistic under
    # ``force_all``, where the existing-pages snapshot that would nudge the LLM
    # to reference ``[[Title]]`` is skipped). On disk / in storage the later
    # write overwrites the earlier one (same doc_id), so the base holds exactly
    # ONE page per path. Dedup the verify view by path — keeping the
    # last-written body, matching what storage actually holds — so the duplicate
    # leg never fabricates a phantom near-duplicate pair out of two copies of a
    # single final page (which would flip ``passed`` to False on clean output),
    # and so the duplicate-pair denominator agrees with ``pages_checked``.
    pages = list({p.path: p for p in produced_pages}.values())
    produced_paths = {p.path for p in pages}

    lint_report = await run_lint(storage, root=root, fallback=fallback)
    findings: list[SynthVerifyLintFinding] = []
    orphans: list[str] = []
    for issue in lint_report.issues:
        if issue.path not in produced_paths:
            continue
        if issue.kind == "orphan_page":
            orphans.append(issue.path)
        elif issue.kind in _VERIFY_GATED_LINT_KINDS:
            findings.append(
                SynthVerifyLintFinding(
                    kind=issue.kind, path=issue.path, detail=issue.detail
                )
            )

    duplicate_checked = embedder is not None
    duplicate_ratio: float | None = None
    if duplicate_checked:
        assert embedder is not None  # narrow for mypy
        # ``duplicate_ratio_max`` short-circuits to 0.0 for <2 bodied pages and
        # filters empty bodies, so tiny runs cost no embed call. For ≥2 bodied
        # pages this is a deliberate SECOND embed pass over the just-written
        # page bodies (whole-body vectors — the inline-persist pass embedded
        # chunk-granularity vectors, which are not reusable for body cosine).
        # That extra cost is the price of the opt-in ``--verify`` duplicate gate.
        # Same resilience contract as the grounding leg below: every page is
        # already persisted by the time verify runs, so an embed failure on
        # this verify-only second pass (a transient 503, a permanent provider
        # misconfig surfacing only here) must degrade to a loud skip
        # (``duplicate_checked`` flips back to False) instead of discarding
        # the whole SynthReport / failing the task. ``CancelledError`` (a
        # BaseException the ``except Exception`` arm misses) re-raises so a
        # cancel mid-leg still propagates.
        try:
            duplicate_ratio = await duplicate_ratio_max(
                pages=pages,
                embedder=embedder,
                embedding_model=embedding_model,
                tau=duplicate_cosine_tau,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                "synth --verify: duplicate leg FAILED (%s) — no duplicate "
                "ratio computed; synth output is unaffected",
                e,
            )
            duplicate_checked = False

    # Optional report-only grounding/entailment leg. Needs an embedder (the
    # per-claim argmax that selects each claim's evidence chunk is embedding-
    # driven) AND an LLM (the entailment verdict). When ``judge`` was requested
    # but either is missing the leg loud-skips: ``grounding_requested`` stays
    # True so the renderer can warn, ``grounding_checked`` stays False so a
    # green verify never reads as "claims grounded" when the check never ran.
    grounding = _GroundingVerifyResult()
    if judge:
        grounding.requested = True
        if embedder is None or judge_llm is None:
            logger.warning(
                "synth --verify --judge: grounding leg SKIPPED — %s missing; "
                "no entailment ratio computed",
                "embedder" if embedder is None else "LLM",
            )
        else:
            # The leg is report-only and MUST NOT fail the synth: every page is
            # already persisted by the time we get here. ``judge_entailment``
            # already swallows per-pair LLM errors, but the grounding re-embed
            # (``compute_grounding_cosines``) and the ``list_chunks`` /
            # ``list_documents`` reads can still raise (a transient embed blip,
            # a permanent provider misconfig surfacing only on this extra embed
            # pass, a storage hiccup). Catch and degrade to a loud skip
            # (``checked`` stays False) rather than letting an exception in the
            # informational leg discard the whole SynthReport. ``CancelledError``
            # (a BaseException the ``except Exception`` arm misses) re-raises so
            # a cancel mid-leg still propagates.
            try:
                grounding = await _grounding_verify_leg(
                    storage=storage,
                    pages=pages,
                    embedder=embedder,
                    embedding_model=embedding_model,
                    llm=judge_llm,
                    judge_model=judge_model,
                    judge_sample=judge_sample,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "synth --verify --judge: grounding leg FAILED (%s) — no "
                    "entailment ratio computed; synth output is unaffected",
                    e,
                )
                grounding = _GroundingVerifyResult(requested=True)

    persist_error_count = len(report.persist_errors)
    persist_ok = persist_error_count == 0
    lint_ok = not findings
    duplicate_ok = duplicate_ratio is None or duplicate_ratio <= max_duplicate_ratio
    return SynthVerifyReport(
        pages_checked=len(produced_paths),
        persist_error_count=persist_error_count,
        unresolved_wikilinks=report.unresolved_wikilinks,
        lint_findings=tuple(findings),
        orphan_pages=tuple(sorted(set(orphans))),
        duplicate_checked=duplicate_checked,
        duplicate_ratio=duplicate_ratio,
        duplicate_cosine_tau=duplicate_cosine_tau,
        max_duplicate_ratio=max_duplicate_ratio,
        grounding_requested=grounding.requested,
        grounding_checked=grounding.checked,
        grounding_entailment_ratio=grounding.ratio,
        grounding_ci=grounding.ci,
        grounding_n_judged=grounding.n_judged,
        grounding_n_no_evidence=grounding.n_no_evidence,
        grounding_n_errors=grounding.n_errors,
        grounding_sample=grounding.sample,
        persist_ok=persist_ok,
        lint_ok=lint_ok,
        duplicate_ok=duplicate_ok,
        passed=persist_ok and lint_ok and duplicate_ok,
    )


@dataclass
class _GroundingVerifyResult:
    """Mutable accumulator for the report-only grounding leg of ``_verify``.

    A tiny internal struct (not a public DTO) so the leg can default-skip
    cleanly and the return-site stays one flat ``SynthVerifyReport(...)``.
    ``ratio`` is ``None`` when nothing was judged (no claims / all
    unverifiable) so the report omits a misleading ``0.0`` floor.
    """

    requested: bool = False
    checked: bool = False
    ratio: float | None = None
    ci: tuple[float, float] = (0.0, 0.0)
    n_judged: int = 0
    n_no_evidence: int = 0
    n_errors: int = 0
    sample: int = 0


async def _grounding_verify_leg(
    *,
    storage: Storage,
    pages: list[KnowledgePage],
    embedder: EmbeddingProvider,
    embedding_model: str,
    llm: LLMProvider,
    judge_model: str,
    judge_sample: int,
) -> _GroundingVerifyResult:
    """Ground this run's page claims against their cited source chunks and ask
    the LLM whether each is entailed. Mirrors the eval runner's construction
    (``compute_grounding_cosines`` → ``claim_evidence_from_grounding`` →
    ``judge_entailment``) but scoped to ``pages`` and their provenance sources.

    Eval imports are local: they pull the eval package's metric/judge code,
    which has no place on the hot synth path when ``--judge`` is not requested.
    """
    from .eval.judge import claim_evidence_from_grounding, judge_entailment
    from .eval.metrics import compute_grounding_cosines

    result = _GroundingVerifyResult(requested=True, sample=judge_sample)

    # Map each produced page to its first provenance source (mirrors the eval
    # runner's ``page.sources[0]`` keying) and load that source's chunks once.
    source_docs = await storage.list_documents(layer=Layer.SOURCE, active=True)
    doc_id_by_path = {doc.path: doc.doc_id for doc in source_docs}
    pages_with_sources: list[tuple[KnowledgePage, str]] = []
    needed_sources: set[str] = set()
    for page in pages:
        if not page.sources:
            continue
        key = page.sources[0]
        if key not in doc_id_by_path:
            continue
        pages_with_sources.append((page, key))
        needed_sources.add(key)

    chunks_by_source: dict[str, list[ChunkRecord]] = {}
    for src_path in needed_sources:
        chunks_by_source[src_path] = await storage.list_chunks(
            doc_id_by_path[src_path]
        )

    grounding_claims = await compute_grounding_cosines(
        pages_with_sources=pages_with_sources,
        chunks_by_source=chunks_by_source,
        embedder=embedder,
        embedding_model=embedding_model,
    )
    pairs = claim_evidence_from_grounding(grounding_claims, chunks_by_source)
    summary = await judge_entailment(
        pairs,
        llm=llm,
        model=judge_model,
        sample=judge_sample,
    )

    result.checked = True
    result.n_judged = summary.n_judged
    result.n_no_evidence = summary.n_no_evidence
    result.n_errors = summary.n_errors
    result.ci = summary.ci
    # Surface the ratio as ``None`` both when nothing was scored (n_judged 0)
    # and when the judge was majority-errored (``trustworthy`` — the shared
    # rule the eval gate fold also applies): a half-dead judge's sliver must
    # not render as "claims fully grounded". The counts above stay visible.
    result.ratio = summary.ratio if summary.trustworthy else None
    return result


def _read_source_parsed(root: Path, doc: DocumentRecord) -> ParsedDocument | None:
    """Re-parse a source from disk, returning the full ``ParsedDocument``.

    Synth needs both the body and the hash: the body to feed the LLM and
    the hash to detect drift since ingest (a user-edited file would
    otherwise be sliced at stale chunk offsets).
    """
    abs_path = (root / doc.path).resolve()
    if not abs_path.is_file():
        return None
    # Route through the backend registry so HTML (and future) sources flow
    # through synth the same way markdown does.
    try:
        return parse_any(abs_path, rel_path=doc.path)
    except (OSError, UnsupportedFormat):
        return None


async def _persist_knowledge_page(
    *,
    storage: Storage,
    root: Path,
    page: KnowledgePage,
    embedder: EmbeddingProvider | None,
    embedding_model: str,
    text_version_id: int | None,
    cjk_tokenizer: CjkTokenizer = "none",
    title_to_path: dict[str, str] | None = None,
    fuzzy_index: dict[str, list[str]] | None = None,
    embedding_error_retries: int = 0,
    embedding_error_retry_backoff_seconds: float = 0.0,
) -> int:
    """Index ``page`` into the K layer: document, chunks, embeddings, links.

    The caller writes ``page`` to disk via ``write_page`` *before*
    invoking this function — we then re-parse the file so the stored
    hash and chunk offsets match what ``read_page`` will compute on
    read. ``frontmatter.dumps`` + ``frontmatter.loads`` is not always
    byte-stable on the body portion, so hashing ``page.body`` directly
    and chunking ``page.body`` would silently diverge from the
    read-back parsed body, causing ``read_page`` to falsely flag every
    K-layer page as stale (empty anchors).

    Returns the count of unresolved outgoing wikilinks so the synth
    caller can fold it into ``SynthReport.unresolved_wikilinks``.

    Thin delegate — implementation lives in
    :mod:`dikw_core.domains.knowledge.page_index` so lint-apply can
    reuse the same indexing path without depending on api.py internals.
    """
    from .domains.knowledge.page_index import persist_knowledge
    result = await persist_knowledge(
        storage=storage,
        root=root,
        path=page.path,
        title=page.title,
        embedder=embedder,
        embedding_model=embedding_model,
        text_version_id=text_version_id,
        cjk_tokenizer=cjk_tokenizer,
        title_to_path=title_to_path,
        fuzzy_index=fuzzy_index,
        retries=embedding_error_retries,
        backoff_seconds=embedding_error_retry_backoff_seconds,
    )
    unresolved = result.unresolved_wikilinks
    return unresolved


# A knowledge_log row with ``action="synth_source_done"`` and this sentinel
# value in ``src`` records "this base has been touched by the fan-out
# synth pipeline at least once". On the very first post-fan-out run we
# always write this sentinel BEFORE the legacy backfill loop, so a later
# crash mid-fan-out can never be misread as legacy data on the next run.
# The string is intentionally not a valid file path.
_LEGACY_BACKFILL_SENTINEL = "__dikw_legacy_backfill_complete__"


# Header strings for the two prompt sections in `_synth_pages_from_source`.
# Pinned as module constants so tests, code, and any future docs stay
# aligned — drift between assertion strings and rendered prompts has
# bitten us before.
_BATCH_SECTION_HEADER = (
    "Already created in this batch (MUST reference, do NOT regenerate)"
)
_EXISTING_SECTION_HEADER = (
    "Existing knowledge pages (reference via [[Title]] when relevant)"
)
_NO_EXISTING_PAGES_SENTINEL = "(no existing pages — this is a fresh knowledge base)"


@dataclass(frozen=True)
class _ExistingPagesSnapshot:
    """Per-source snapshot of the K-layer for the synth prompt.

    Hoisted out of the per-group loop because the base K-layer is
    invariant within a single source (persist runs only after all of
    that source's groups complete). Without this hoist, a source with
    G groups against a base of W pages paid G x W storage round-trips
    per synth call.
    """

    pages: list[DocumentRecord]   # already filtered to title-bearing
    full_render_bytes: int

    @classmethod
    async def load(cls, storage: Storage) -> _ExistingPagesSnapshot:
        pages = [
            d for d in await storage.list_documents(
                layer=Layer.KNOWLEDGE, active=True
            )
            if d.title
        ]
        full_render_bytes = sum(
            len(
                f"- {d.title} [{Path(d.path).stem}] "
                f"({category_from_path(d.path)})\n".encode()
            )
            for d in pages
        )
        return cls(pages=pages, full_render_bytes=full_render_bytes)

    def full_pages(self) -> list[tuple[str, str, str]]:
        # ``(title, slug, category)`` — slug is the deterministic kebab-case
        # filename stem (``knowledge/<category>/<slug>.md``), surfaced so the
        # LLM can disambiguate two same-titled pages.
        return [
            (t, Path(d.path).stem, category_from_path(d.path))
            for d in self.pages
            if (t := d.title)
        ]


async def _existing_pages_for_prompt(
    storage: Storage,
    *,
    snapshot: _ExistingPagesSnapshot,
    group_chunks: list[ChunkRecord],
    max_bytes: int,
    top_k: int,
    version_id: int | None,
) -> list[tuple[str, str, str]]:
    """Return ``[(title, slug, category), ...]`` for the existing-pages section.

    Full render up to ``max_bytes`` of the rendered ``- Title [slug] (category)``
    bullets; above the threshold, switches to a vec_search-gated top-K
    driven by the group's chunk embeddings (per-chunk vec_search →
    union by doc_id → score sort → top-K). The retrieval branch keeps
    the prompt size bounded as the knowledge base grows; without it a base with
    thousands of pages would eventually overflow the model's context
    window.

    Returns ``[]`` (empty section) for a fresh knowledge base or a base with no
    embedded source chunks — the caller renders the falsy section as
    ``(no existing pages …)`` so the LLM sees a clear signal rather
    than a missing block.
    """
    if not snapshot.pages:
        return []
    if snapshot.full_render_bytes <= max_bytes:
        return snapshot.full_pages()

    # Over the byte threshold → retrieval-gated top-K. Per-chunk
    # vec_search against the WIKI layer is what the locked design
    # specifies; union by doc_id, keep best (smallest) distance per
    # doc, sort, take top-K. Distance is cosine (smaller = closer).
    #
    # ``_truncated_fallback`` is the safety net for "many pages but the
    # WIKI layer has no vectors" (``--no-embed`` wikis, version mismatch,
    # or chunks the source-side embedder hasn't reached). Returning ``[]``
    # would render the "(no existing pages — fresh knowledge base)" sentinel and
    # drop ALL duplicate-avoidance context exactly when the knowledge base has
    # the most to offer it. Bounded prefix is a worse signal than
    # vec-ranked top-K but a better one than "fresh knowledge base, generate
    # freely". Order matches the snapshot, which mirrors
    # ``list_documents`` order — stable across runs.
    def _truncated_fallback() -> list[tuple[str, str, str]]:
        return snapshot.full_pages()[:top_k]

    embs = await storage.get_chunk_embeddings(
        [c.chunk_id for c in group_chunks if c.chunk_id is not None],
        version_id=version_id,
    )
    if not embs:
        return _truncated_fallback()
    best_dist: dict[str, float] = {}
    for emb in embs.values():
        try:
            # Pin the lookup to the SAME version we fetched embeddings
            # under — without this, vec_search re-resolves the active
            # version and could pick a different per-version table
            # (mid-synth ingest activating a new version, or a direct
            # caller passing a non-active version_id), producing dim
            # mismatches or rankings against the wrong index.
            hits = await storage.vec_search(
                emb, layer=Layer.KNOWLEDGE, limit=top_k, version_id=version_id
            )
        except NotSupported:
            return _truncated_fallback()
        for hit in hits:
            prior = best_dist.get(hit.doc_id)
            if prior is None or hit.distance < prior:
                best_dist[hit.doc_id] = hit.distance
    if not best_dist:
        return _truncated_fallback()
    ordered_doc_ids = [
        doc_id for doc_id, _ in sorted(best_dist.items(), key=lambda kv: kv[1])
    ][:top_k]
    docs = await storage.get_documents(ordered_doc_ids)
    by_id = {d.doc_id: d for d in docs}
    out: list[tuple[str, str, str]] = []
    for doc_id in ordered_doc_ids:
        d = by_id.get(doc_id)
        if d is not None and d.title:
            out.append((d.title, Path(d.path).stem, category_from_path(d.path)))
    return out


def _render_existing_section(
    pages: list[tuple[str, str, str]], header: str
) -> str:
    """Render a list of ``(title, slug, category)`` tuples as a markdown section.

    Each page renders as ``- Title [slug] (category)``: the slug is the
    deterministic kebab-case file identifier, surfaced so the LLM can tell
    two same-titled pages apart. Empty input returns ``""`` so callers can
    concatenate two sections (batch accumulator + base snapshot) and fall
    back to a single "(no existing pages …)" sentinel only when both are
    empty. H3 heading — the section nests under the template's
    ``## Knowledge-base context`` H2 instead of competing with it.
    """
    if not pages:
        return ""
    lines = [f"### {header}", ""] + [
        f"- {title} [{slug}] ({cat})" for title, slug, cat in pages
    ]
    return "\n".join(lines) + "\n"


# Header + cap for the priority-create feedback section. Top-K caps how many
# unresolved targets a later group sees so the directive stays focused on the
# most-referenced gaps rather than a long noisy list.
_PRIORITY_SECTION_HEADER = "Priority targets (create if relevant)"
_PRIORITY_TARGETS_TOP_K = 5


def _wikilink_resolves(
    target: str,
    *,
    title_to_path: dict[str, str],
    fuzzy_index: dict[str, list[str]],
) -> bool:
    """Whether ``target`` resolves to exactly one known page.

    Mirrors the wikilink branch of ``resolve_links``: exact title hit, else a
    single fuzzy-normalize candidate. A ≥2-candidate fuzzy collision is
    deliberately NOT resolved (refuse to guess — same Karpathy rule the
    persist-time graph build follows), so an ambiguous target is treated as
    still-unresolved.
    """
    if target in title_to_path:
        return True
    key = normalize_for_match(target)
    candidates = fuzzy_index.get(key, []) if key else []
    return len(candidates) == 1


def _unresolved_wikilink_targets(
    body: str,
    *,
    title_to_path: dict[str, str],
    fuzzy_index: dict[str, list[str]],
) -> list[str]:
    """Clean ``[[target]]`` titles in ``body`` that resolve to no known page.

    Uses :func:`_wikilink_resolves` so the priority-create signal applies the
    SAME resolution rules the persist-time graph build will. ``parse_links``
    has already stripped anchors/aliases, so the returned title is exactly what
    a later group should create verbatim. Targets repeat once per occurrence;
    the caller dedups per page before counting so the rank reflects how many
    distinct pages want a target, not how often one page repeats it.

    An anchor-only / blank / punctuation-only wikilink (``[[#sec]]``, ``[[ ]]``,
    ``[[...]]``) strips to an empty or keyless target. ``resolve_links``
    surfaces those to ``lint`` via their RAW text, but as a *create* directive
    an uncreatable empty/punctuation title is junk — drop any target with no
    usable fuzzy key, so the priority section never emits a nonsensical
    ``- [[]]`` create-this line.
    """
    out: list[str] = []
    for link in parse_links(body):
        if link.kind is not LinkType.WIKILINK:
            continue
        if not normalize_for_match(link.target):
            continue
        if not _wikilink_resolves(
            link.target, title_to_path=title_to_path, fuzzy_index=fuzzy_index
        ):
            out.append(link.target)
    return out


def _render_priority_targets(targets: list[tuple[str, int]]) -> str:
    """Render the top unresolved wikilink targets as a create-this directive.

    ``targets`` is ``[(title, reference_count), ...]`` already sorted
    most-referenced-first and truncated. Empty input returns ``""`` so the
    caller omits the section entirely (group 1, or a source whose earlier
    groups left nothing unresolved).
    """
    if not targets:
        return ""
    lines = [
        f"### {_PRIORITY_SECTION_HEADER}",
        "",
        "Earlier sections of THIS source referenced these via [[wikilink]] but "
        "no page exists for them yet. If this section's content genuinely "
        "defines or covers one, prefer creating that page now using the exact "
        "title shown — do NOT invent unrelated pages just to satisfy the list:",
        "",
    ]
    lines += [
        f"- [[{title}]] ({count} prior reference{'s' if count != 1 else ''})"
        for title, count in targets
    ]
    return "\n".join(lines) + "\n"


def _build_known_title_index(
    snapshot_title_to_path: dict[str, str],
    batch_accumulator: list[tuple[str, str, str]],
) -> dict[str, str]:
    """Merge the existing-page snapshot with the in-batch pages into one
    ``title → path`` map for wikilink resolution.

    A real path per title (from the batch page's category + slug) is kept so
    the fuzzy ≥2-candidate collision check can still fire; ``setdefault`` lets
    the snapshot win on a title clash with a batch page.
    """
    known = dict(snapshot_title_to_path)
    for title, slug, cat in batch_accumulator:
        # ``cat`` is never empty — the accumulator stores ``p.category or "page"``
        # — so the path form is unconditional. The exact string is only an opaque
        # distinctness token for the fuzzy ≥2-candidate collision check anyway.
        known.setdefault(title, f"knowledge/{cat}/{slug}.md")
    return known


@dataclass(frozen=True)
class _SourceSynthOutcome:
    """Per-source aggregate of all the LLM calls Stage A made for that source."""

    pages: list[KnowledgePage]
    groups_processed: int
    parse_errors: int
    log_notes: list[str]


async def _synth_pages_from_source(
    *,
    llm: LLMProvider,
    template: str,
    cfg: DikwConfig,
    source_path: str,
    source_body: str,
    chunks: list[ChunkRecord],
    cancel: CancelToken,
    storage: Storage | None = None,
    text_version_id: int | None = None,
    force_all: bool = False,
    reporter: ProgressReporter | None = None,
) -> _SourceSynthOutcome:
    """Fan a single source out into ChunkGroups and call the LLM per group.

    The caller persists the returned pages and writes ``log_notes`` /
    counts to ``knowledge_log`` and the ``SynthReport``. ``reporter`` (optional)
    receives a ``synth_llm`` ``calling`` / ``returned`` event pair per
    group so server clients can render group-level progress instead of
    freezing on the per-source counter while a multi-group LLM call runs.

    ``storage`` + ``text_version_id`` drive the per-group existing-pages
    section: each group's prompt receives a ``### Already created in
    this batch`` accumulator (per-source state, lifecycle = this call)
    plus a ``### Existing knowledge pages`` snapshot of the base K-layer
    (full list under ``synth.existing_pages_max_bytes``, retrieval-gated
    top-K above). Without this awareness the LLM regenerates pages it
    cannot see, polluting the knowledge base with semantic duplicates that PR1's
    fuzzy resolver cannot absorb.
    """
    _reporter: ProgressReporter = reporter or NoopReporter()
    sections = derive_sections_from_chunks(
        source_body, chunks, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    groups = group_sections(
        sections, target_tokens=cfg.synth.target_tokens_per_group
    )
    if not groups:
        return _SourceSynthOutcome(
            pages=[], groups_processed=0, parse_errors=0, log_notes=[]
        )

    allowed_categories = tuple(cfg.schema_.category_paths())
    fallback = cfg.schema_.fallback
    # Rendered ``- `path` — desc`` bullets injected into the prompt's
    # ``{categories}`` slot so the LLM sees the full declared taxonomy.
    categories_block = cfg.schema_.categories_prompt_block()
    pages: list[KnowledgePage] = []
    notes: list[str] = []
    errors = 0
    total_groups = len(groups)
    # Per-source batch accumulator: each group's prompt sees the titles
    # emitted by groups 0..N-1 of the SAME source, so group N can
    # reference [[Title]] instead of regenerating. Lifecycle scoped
    # tightly to this function — a new source starts empty.
    # ``seen_titles`` mirrors the accumulator titles for O(1) dedup
    # without rebuilding a set every group.
    batch_accumulator: list[tuple[str, str, str]] = []
    seen_titles: set[str] = set()
    # Priority-create feedback (#4): wikilink targets earlier groups of THIS
    # source referenced but that no page (existing snapshot OR batch) satisfies
    # yet, counted by reference frequency. A later group covering one is nudged
    # to create it at the right title instead of leaving the graph broken.
    unresolved_counts: Counter[str] = Counter()
    priority_surfaced = 0
    # Map section-start → chunk so we can recover per-group chunks for
    # the retrieval-gated existing-pages branch. ``derive_sections_from_chunks``
    # builds sections 1:1 from chunks, so ``section.start == chunk.start``.
    start_to_chunk = {c.start: c for c in chunks}
    # The base K-layer is invariant within a single source's group loop
    # (persist runs only after this function returns), so we hoist the
    # snapshot out of the loop. Without this, a source with G groups
    # against a base of W pages paid G x W storage round-trips per call.
    #
    # ``force_all`` skips the snapshot: ``dikw client synth --all`` is the
    # documented "regenerate everything after a prompt/model change"
    # path. Showing the LLM the OLD output of the same source plus the
    # zero-block-on-duplicate instruction would cause the model to skip
    # the regeneration the user explicitly requested. The in-batch
    # accumulator still runs so groups within the same source coordinate.
    snapshot = (
        await _ExistingPagesSnapshot.load(storage)
        if storage is not None and not force_all
        else None
    )
    # Full existing-page title→path map for priority-create resolution. The
    # FULL snapshot (not the byte-truncated prompt view) is the right basis:
    # a wikilink may resolve to a page that exists but wasn't shown, and we
    # must not nudge a later group to recreate it.
    snapshot_title_to_path: dict[str, str] = {}
    if snapshot is not None:
        for d in snapshot.pages:
            if d.title:
                snapshot_title_to_path.setdefault(d.title, d.path)
    for group in groups:
        cancel.raise_if_cancelled()
        group_pos = group.index + 1
        if storage is not None and snapshot is not None:
            group_chunks = [
                start_to_chunk[s] for s in group.section_starts
                if s in start_to_chunk
            ]
            existing_pages = await _existing_pages_for_prompt(
                storage,
                snapshot=snapshot,
                group_chunks=group_chunks,
                max_bytes=cfg.synth.existing_pages_max_bytes,
                top_k=cfg.synth.existing_pages_top_k,
                version_id=text_version_id,
            )
        else:
            # Storage-less callers (narrow unit tests of LLM event shape)
            # render the no-pages sentinel — they exercise the prompt
            # plumbing, not the existing-pages contract itself.
            existing_pages = []
        # Priority-create feedback (#4): for any group after the first,
        # surface the still-unresolved targets earlier groups left behind.
        # Re-resolve each accumulated target against the FULL existing set +
        # the batch as it stands BEFORE this group (groups 0..N-1) using the
        # same fuzzy rules — so a target a prior group has since satisfied
        # (even via a plural/normalize match) is dropped, never nudging a
        # later group to create a page that already exists.
        priority_section = ""
        if group.index >= 1 and unresolved_counts:
            known_so_far = _build_known_title_index(
                snapshot_title_to_path, batch_accumulator
            )
            fuzzy_so_far = build_fuzzy_index(known_so_far)
            pending = [
                (title, count)
                for title, count in unresolved_counts.most_common()
                if not _wikilink_resolves(
                    title, title_to_path=known_so_far, fuzzy_index=fuzzy_so_far
                )
            ][:_PRIORITY_TARGETS_TOP_K]
            priority_section = _render_priority_targets(pending)
            if pending:
                priority_surfaced += 1
        existing_core = (
            _render_existing_section(batch_accumulator, _BATCH_SECTION_HEADER)
            + _render_existing_section(existing_pages, _EXISTING_SECTION_HEADER)
        ).strip() or _NO_EXISTING_PAGES_SENTINEL
        existing_pages_section = (
            f"{priority_section}\n{existing_core}"
            if priority_section
            else existing_core
        )
        user_prompt = template.format(
            source_path=source_path,
            source_body=group.text,
            group_outline=", ".join(group.headings)
            if group.headings
            else "(no headings)",
            group_index=group_pos,
            group_total=total_groups,
            max_pages=cfg.synth.max_pages_per_group,
            categories=categories_block,
            existing_pages_section=existing_pages_section,
        )
        # `current` reports groups COMPLETED — Rich's TaskProgressRenderer
        # passes it as `completed`, so a `calling` event must show one
        # less than the in-flight group_pos. Otherwise a single-group
        # source flips to 100% the moment the LLM call starts, recreating
        # the "looks finished but isn't" symptom this PR exists to fix.
        await _reporter.progress(
            phase="synth_llm",
            current=group_pos - 1,
            total=total_groups,
            detail={
                "source_path": source_path,
                "group_pos": group_pos,
                "model": cfg.provider.llm_model,
                "status": "calling",
                "section_count": len(group.section_starts),
                "approx_tokens": group.token_count,
            },
        )
        logger.debug(
            "  group %d/%d calling llm.complete (model=%s, sections=%d, ~%d tokens)",
            group_pos,
            total_groups,
            cfg.provider.llm_model,
            len(group.section_starts),
            group.token_count,
        )
        # Per-page output budget for this group — a low value (a small
        # ``llm_max_tokens_synth`` against a large ``max_pages_per_group``)
        # is the usual cause of mid-page truncation, so surface it when
        # tuning synth output quality.
        logger.debug(
            "  group %d/%d budget: max_tokens=%d / max_pages=%d = ~%d tok/page",
            group_pos,
            total_groups,
            cfg.provider.llm_max_tokens_synth,
            cfg.synth.max_pages_per_group,
            cfg.provider.llm_max_tokens_synth
            // max(1, cfg.synth.max_pages_per_group),
        )
        # Per-group ProviderError resilience (issue #134). One bad group
        # (codex empty-response, auth flap, refusal) must not abort the
        # whole task — retry up to ``provider_error_retries`` times with
        # linear backoff, then skip the group and continue with the
        # next. The skip is recorded as a parse-style error so
        # ``synth_source_done`` is NOT written (the marker is gated on
        # ``outcome.parse_errors == 0`` and re-running synth has another
        # chance to succeed on the flaky group).
        max_attempts = cfg.synth.provider_error_retries + 1
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = await llm.complete(
                    system=DEFAULT_SYNTH_SYSTEM,
                    user=user_prompt,
                    model=cfg.provider.llm_model,
                    max_tokens=cfg.provider.llm_max_tokens_synth,
                    temperature=0.3,
                )
                break
            except TransientProviderError as pe:
                # Symmetric with the embed-batch retry-skip semantics:
                # only TransientProviderError is retried-then-skipped.
                # Bare ProviderError (auth fail, invalid model id,
                # missing API key) propagates so synth fails fast
                # instead of silently retry-skipping every group and
                # reporting "succeeded with 0 pages" (code-review
                # finding, 0.4.0).
                if attempt < max_attempts:
                    backoff_s = (
                        cfg.synth.provider_error_retry_backoff_seconds
                        * attempt
                    )
                    logger.warning(
                        "  group %d/%d TransientProviderError on attempt %d/%d: "
                        "%s — retrying in %.1fs",
                        group_pos,
                        total_groups,
                        attempt,
                        max_attempts,
                        pe,
                        backoff_s,
                    )
                    await _reporter.progress(
                        phase="synth_llm",
                        current=group_pos - 1,
                        total=total_groups,
                        detail={
                            "source_path": source_path,
                            "group_pos": group_pos,
                            "status": "retrying",
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "error_kind": type(pe).__name__,
                            "error_msg": str(pe)[:200],
                        },
                    )
                    if backoff_s > 0:
                        await asyncio.sleep(backoff_s)
                    # Honor cancellation between attempts so a user-
                    # issued cancel during a flapping run terminates
                    # promptly instead of consuming the full budget.
                    cancel.raise_if_cancelled()
                else:
                    errors += 1
                    notes.append(
                        f"group {group_pos}/{total_groups} provider "
                        f"error (skipped after {attempt} attempt(s)): {pe}"
                    )
                    logger.warning(
                        "  group %d/%d TransientProviderError on attempt %d/%d "
                        "(final), skipping group: %s",
                        group_pos,
                        total_groups,
                        attempt,
                        max_attempts,
                        pe,
                    )
                    await _reporter.progress(
                        phase="synth_llm",
                        current=group_pos,
                        total=total_groups,
                        detail={
                            "source_path": source_path,
                            "group_pos": group_pos,
                            "status": "skipped",
                            "reason": "provider_error",
                            "attempts": attempt,
                            "error_kind": type(pe).__name__,
                            "error_msg": str(pe)[:200],
                        },
                    )
        if response is None:
            # All attempts exhausted; move on to the next group.
            continue
        await _reporter.progress(
            phase="synth_llm",
            current=group_pos,
            total=total_groups,
            detail={
                "source_path": source_path,
                "group_pos": group_pos,
                "status": "returned",
                "response_chars": len(response.text),
            },
        )
        logger.debug(
            "  group %d/%d ← returned (%d chars)",
            group_pos,
            total_groups,
            len(response.text),
        )
        try:
            new_pages = parse_synthesis_response(
                response.text,
                source_path=source_path,
                allowed_categories=allowed_categories,
                fallback=fallback,
                finish_reason=response.finish_reason,
            )
        except SynthesisPartialError as pe:
            notes.append(
                f"group {group_pos}/{total_groups} partial parse: "
                f"{len(pe.errors)} issue(s); first: {pe.errors[0]}"
            )
            # Truncation is recoverable next run — count it as a parse
            # error so the source-done marker is NOT written.
            if pe.retry:
                errors += 1
            new_pages = pe.pages
            logger.warning(
                "  group %d/%d PARTIAL: %d issue(s); first: %s",
                group_pos,
                total_groups,
                len(pe.errors),
                pe.errors[0],
            )
            await _reporter.progress(
                phase="synth_llm",
                current=group_pos,
                total=total_groups,
                detail={
                    "source_path": source_path,
                    "group_pos": group_pos,
                    "status": "error",
                    "error_kind": type(pe).__name__,
                    "error_msg": str(pe)[:200],
                },
            )
        except SynthesisError as e:
            errors += 1
            notes.append(
                f"group {group_pos}/{total_groups} parse error: {e}"
            )
            logger.warning(
                "  group %d/%d FAILED: %s: %s",
                group_pos,
                total_groups,
                type(e).__name__,
                e,
            )
            await _reporter.progress(
                phase="synth_llm",
                current=group_pos,
                total=total_groups,
                detail={
                    "source_path": source_path,
                    "group_pos": group_pos,
                    "status": "error",
                    "error_kind": type(e).__name__,
                    "error_msg": str(e)[:200],
                },
            )
            continue
        pages.extend(new_pages)
        # Feed group N's emitted page titles into the per-source
        # accumulator so group N+1's prompt sees them. ``seen_titles``
        # is maintained incrementally above so dedup is O(1) per page
        # without rebuilding a set every group. The slug (file stem) rides
        # along so the batch section renders ``- Title [slug] (category)``.
        for p in new_pages:
            if p.title and p.title not in seen_titles:
                batch_accumulator.append(
                    (p.title, Path(p.path).stem, p.category or "page")
                )
                seen_titles.add(p.title)

        # Accumulate this group's unresolved wikilink targets for the
        # priority-create feedback. Resolve against the FULL existing-pages
        # set + every batch page (incl. this group's own, just added above),
        # so we count only targets nobody has authored.
        known_after_group = _build_known_title_index(
            snapshot_title_to_path, batch_accumulator
        )
        fuzzy_after_group = build_fuzzy_index(known_after_group)
        for p in new_pages:
            # Dedup per page (set) so the count ranks targets by how many
            # distinct pages want them — one page repeating [[X]] ten times
            # must not outrank ten pages each wanting a different target.
            for target in set(
                _unresolved_wikilink_targets(
                    p.body,
                    title_to_path=known_after_group,
                    fuzzy_index=fuzzy_after_group,
                )
            ):
                unresolved_counts[target] += 1

    if priority_surfaced:
        notes.append(
            f"priority-create: surfaced unresolved wikilink target(s) to "
            f"{priority_surfaced} later group(s)"
        )

    return _SourceSynthOutcome(
        pages=pages,
        groups_processed=total_groups,
        parse_errors=errors,
        log_notes=notes,
    )


def _sr_replace(r: SynthReport, **kw: Any) -> SynthReport:
    return dataclasses.replace(r, **kw)


