"""High-level engine facade — server routes (``dikw_core.server``) and
the eval runner depend on this module; CLI access is via ``dikw client``
which talks HTTP to a running server instead of importing the engine.

Phase 1 surface:
  * ``ingest`` — walk configured sources, parse markdown, chunk, embed, index.
  * ``retrieve`` — hybrid search returning ranked chunks + page refs;
    no LLM call. Answer synthesis is the agent's responsibility.

Phase 2 surface:
  * ``synthesize`` — turn source docs into K-layer knowledge pages via the LLM,
    persist them to disk + storage, maintain the link graph, and refresh
    ``knowledge/index.md`` + ``knowledge/log.md``.
  * ``lint`` — report broken wikilinks, orphans, and duplicate titles.

W layer (wisdom) is being refactored to first-class documents under
``wisdom/<author>/<slug>.md`` — the prior LLM-distilled candidate/review
surface (``distill`` / ``list_candidates`` / ``approve_wisdom`` /
``reject_wisdom``) has been removed in this PR; see CHANGELOG for the
ongoing 0.3.0 refactor.

Phase 0 surface (``init_base``, ``status``) stays unchanged.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from . import prompts
from .api_core import (
    _assert_base_upgraded as _assert_base_upgraded,
)
from .api_core import (
    _with_storage as _with_storage,
)
from .api_core import (
    init_base,
    load_base,
    status,
)
from .api_graph import (
    list_graph,
    list_links,
    read_provenance,
)
from .api_health import (
    _PROBE_PNG_1X1 as _PROBE_PNG_1X1,
)
from .api_health import (
    _sanitize_base_url as _sanitize_base_url,
)
from .api_health import (
    check_providers,
    health,
)
from .api_ingest import ingest
from .api_lint import lint
from .api_lint import (
    lint_apply as lint_apply,
)
from .api_lint import (
    lint_propose as lint_propose,
)
from .api_pages import (
    list_pages,
    read_asset,
    read_page,
)
from .api_retrieve import retrieve
from .api_types import (
    AssetNotFound,
    CheckReport,
    EmbeddingInfo,
    HealthReport,
    IngestError,
    IngestErrorKind,
    IngestReport,
    LayerCounts,
    LlmInfo,
    MultimodalInfo,
    PageNotFound,
    ProbeResult,
    ProvidersInfo,
    SynthReport,
)
from .api_types import (
    BaseUpgradeRequired as BaseUpgradeRequired,
)
from .api_wisdom import write_wisdom_page
from .config import (
    DikwConfig,
    find_config,
)
from .domains.data.backends import UnsupportedFormat, parse_any
from .domains.data.backends.base import ParsedDocument
from .domains.data.path_norm import doc_id_for as _doc_id_for
from .domains.info.tokenize import CjkTokenizer
from .domains.knowledge.grouping import (
    derive_sections_from_chunks,
    group_sections,
)
from .domains.knowledge.indexgen import regenerate_index
from .domains.knowledge.links import (
    build_fuzzy_index,
)
from .domains.knowledge.log import render_log
from .domains.knowledge.page import KnowledgePage, now_iso, type_from_path, write_page
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
    DerivedPage,
    DocumentRecord,
    IncomingLink,
    KnowledgeLogEntry,
    Layer,
    OutgoingLink,
    PageAnchor,
    PageLinksResult,
    PageProvenanceResult,
    PageReadResult,
    PageRef,
    ProvenanceSource,
    RetrieveResult,
    WisdomWriteReport,
)
from .storage import Storage
from .storage.base import NotSupported

logger = logging.getLogger(__name__)

__all__ = [
    "AssetNotFound",
    "CheckReport",
    "DerivedPage",
    "EmbeddingInfo",
    "HealthReport",
    "IncomingLink",
    "IngestError",
    "IngestErrorKind",
    "IngestReport",
    "LayerCounts",
    "LlmInfo",
    "MultimodalInfo",
    "OutgoingLink",
    "PageAnchor",
    "PageLinksResult",
    "PageNotFound",
    "PageProvenanceResult",
    "PageReadResult",
    "PageRef",
    "ProbeResult",
    "ProvenanceSource",
    "ProvidersInfo",
    "RetrieveResult",
    "SynthReport",
    "WisdomWriteReport",
    "check_providers",
    "find_config",
    "health",
    "ingest",
    "init_base",
    "lint",
    "list_graph",
    "list_links",
    "list_pages",
    "load_base",
    "read_asset",
    "read_page",
    "read_provenance",
    "retrieve",
    "status",
    "synthesize",
    "write_wisdom_page",
]


# ---- Phase 2: synthesize + lint -----------------------------------------


async def synthesize(
    path: str | Path | None = None,
    *,
    force_all: bool = False,
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    reporter: ProgressReporter | None = None,
) -> SynthReport:
    """Turn source docs into K-layer knowledge pages via the configured LLM.

    By default only source docs that have never been synthesised are
    processed; pass ``force_all=True`` to re-synthesise every source.
    Embedding of new knowledge pages is skipped when ``embedder`` is ``None``.

    ``reporter`` (optional) receives one ``progress`` event per source
    document processed for server-driven task wrappers.
    """
    cfg, root, storage = await _with_storage(path)
    _reporter: ProgressReporter = reporter or NoopReporter()
    try:
        _llm = llm or build_llm(cfg.provider, base_root=root)

        text_version_id: int | None = None
        text_embed_model = cfg.provider.embedding_model
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
                embedder = None  # no active text version → nothing to embed against

        sources = list(await storage.list_documents(layer=Layer.SOURCE, active=True))
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
            for entry in await storage.list_knowledge_log():
                if entry.action == "synth_source_done":
                    if entry.src == _LEGACY_BACKFILL_SENTINEL:
                        has_legacy_backfill_sentinel = True
                    elif entry.src:
                        already.add(entry.src)
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
                for src_path in sorted(legacy_dst_sources):
                    await storage.append_knowledge_log(
                        KnowledgeLogEntry(
                            ts=ts,
                            action="synth_source_done",
                            src=src_path,
                            note="backfilled from legacy per-page synth rows",
                        )
                    )
                already |= legacy_dst_sources

        report = SynthReport()
        tmpl = prompts.load("synthesize")
        persisted_any = False
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
            for page in deduped:
                pre_existing = await storage.get_document(
                    _doc_id_for(Layer.KNOWLEDGE, page.path)
                )
                write_page(root, page)
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
                persisted_any = True
                if pre_existing is None:
                    created_for_src += 1
                    report = _sr_replace(report, created=report.created + 1)
                else:
                    updated_for_src += 1
                    report = _sr_replace(report, updated=report.updated + 1)

            persisted_for_src = created_for_src + updated_for_src
            report = _sr_replace(
                report, sources_processed=report.sources_processed + 1
            )
            # Mark the source as fully synthesised so default ``synth``
            # skips it next run. Skip the marker when any group raised
            # a hard ``SynthesisError`` — those failures should be
            # retried (the LLM may produce parseable output next time).
            # Partial-parse outcomes don't count: the surviving pages
            # were persisted, retrying would just hit the same partial
            # response and re-emit the warning to ``knowledge_log``.
            if outcome.parse_errors == 0:
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
                },
            )

        # Refresh the human-readable views after the batch so a partial run
        # still leaves the knowledge base internally consistent.
        if persisted_any or not (root / "knowledge" / "index.md").exists():
            regenerate_index(root, updated=now_iso())
        entries = await storage.list_knowledge_log()
        render_log(root, entries, updated=now_iso())

        return report
    finally:
        await storage.close()


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
            len(f"- {d.title} ({type_from_path(d.path)})\n".encode())
            for d in pages
        )
        return cls(pages=pages, full_render_bytes=full_render_bytes)

    def full_pages(self) -> list[tuple[str, str]]:
        return [(t, type_from_path(d.path)) for d in self.pages if (t := d.title)]


async def _existing_pages_for_prompt(
    storage: Storage,
    *,
    snapshot: _ExistingPagesSnapshot,
    group_chunks: list[ChunkRecord],
    max_bytes: int,
    top_k: int,
    version_id: int | None,
) -> list[tuple[str, str]]:
    """Return ``[(title, type), ...]`` for the synth prompt's existing-pages section.

    Full render up to ``max_bytes`` of the rendered ``- Title (type)``
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
    def _truncated_fallback() -> list[tuple[str, str]]:
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
    out: list[tuple[str, str]] = []
    for doc_id in ordered_doc_ids:
        d = by_id.get(doc_id)
        if d is not None and d.title:
            out.append((d.title, type_from_path(d.path)))
    return out


def _render_existing_section(
    pages: list[tuple[str, str]], header: str
) -> str:
    """Render a list of ``(title, type)`` tuples as a markdown section.

    Empty input returns ``""`` so callers can concatenate two sections
    (batch accumulator + base snapshot) and fall back to a single
    "(no existing pages …)" sentinel only when both are empty.
    """
    if not pages:
        return ""
    lines = [f"## {header}", ""] + [f"- {t} ({tp})" for t, tp in pages]
    return "\n".join(lines) + "\n"


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
    section: each group's prompt receives a ``## Already created in
    this batch`` accumulator (per-source state, lifecycle = this call)
    plus a ``## Existing knowledge pages`` snapshot of the base K-layer
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

    page_types = tuple(cfg.schema_.page_types)
    allowed_types_str = " | ".join(page_types)
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
    batch_accumulator: list[tuple[str, str]] = []
    seen_titles: set[str] = set()
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
        existing_pages_section = (
            _render_existing_section(batch_accumulator, _BATCH_SECTION_HEADER)
            + _render_existing_section(existing_pages, _EXISTING_SECTION_HEADER)
        ).strip() or _NO_EXISTING_PAGES_SENTINEL
        user_prompt = template.format(
            source_path=source_path,
            source_body=group.text,
            group_outline=", ".join(group.headings)
            if group.headings
            else "(no headings)",
            group_index=group_pos,
            group_total=total_groups,
            max_pages=cfg.synth.max_pages_per_group,
            allowed_types=allowed_types_str,
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
                allowed_types=page_types,
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
        # without rebuilding a set every group.
        for p in new_pages:
            if p.title and p.title not in seen_titles:
                batch_accumulator.append((p.title, p.type or "page"))
                seen_titles.add(p.title)

    return _SourceSynthOutcome(
        pages=pages,
        groups_processed=total_groups,
        parse_errors=errors,
        log_notes=notes,
    )


def _sr_replace(r: SynthReport, **kw: int) -> SynthReport:
    return dataclasses.replace(r, **kw)


