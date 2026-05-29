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
import contextlib
import dataclasses
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from . import prompts
from .api_core import (
    _assert_base_upgraded as _assert_base_upgraded,
)
from .api_core import (
    _preflight_embedder,
    _qualified_provider,
    _register_text_version,
    _resolve_active_text_version_for_inline_embed,
    init_base,
    load_base,
    resolve_base_root,
    status,
)
from .api_core import (
    _with_storage as _with_storage,
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
from .config import (
    DikwConfig,
    find_config,
)
from .domains.data.assets import materialize_asset
from .domains.data.backends import UnsupportedFormat, parse_any
from .domains.data.backends.base import ParsedDocument
from .domains.data.path_norm import doc_id_for as _doc_id_for
from .domains.data.persist import persist_source
from .domains.data.sources import iter_source_files
from .domains.info.embed import (
    ChunkToEmbed,
    consume_embedding_stream,
    embed_assets,
    embed_chunks,
    is_unembeddable_asset_mime,
)
from .domains.info.tokenize import CjkTokenizer
from .domains.knowledge.grouping import (
    derive_sections_from_chunks,
    group_sections,
)
from .domains.knowledge.indexgen import regenerate_index
from .domains.knowledge.links import (
    build_fuzzy_index,
)
from .domains.knowledge.lint import LintKind, LintReport, run_lint
from .domains.knowledge.lint_fix import (
    ApplyReport,
    FixerContext,
    FixProposalReport,
    KnowledgePageMeta,
    run_lint_apply,
    run_lint_propose,
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
    MultimodalEmbeddingProvider,
    TransientProviderError,
    build_embedder,
    build_llm,
    build_multimodal_embedder,
)
from .schemas import (
    AssetRecord,
    ChunkRecord,
    DerivedPage,
    DocumentRecord,
    EmbeddingVersion,
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
    WisdomStatus,
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


# One stderr Console for all embedding progress bars. Constructing one
# per ingest pass would re-probe terminal capability + color system on
# every call; rich's recommended pattern is a single shared instance.
_PROGRESS_CONSOLE = Console(stderr=True)


def _ceil_div(n: int, d: int) -> int:
    """``(N + B - 1) // B`` with the same ``batch_size > 0`` guard
    ``embed_chunks`` / ``embed_assets`` already raise on. Without this,
    a ``batch_size: 0`` config produced an opaque ``ZeroDivisionError``
    from this helper before reaching the embed function's validation.
    """
    if d <= 0:
        raise ValueError(f"batch_size must be positive, got {d}")
    return (n + d - 1) // d


@contextlib.contextmanager
def _embedding_progress(
    description: str, *, total: int
) -> Iterator[Callable[[], None]]:
    """Yield an ``advance()`` that bumps a transient stderr progress bar
    by one batch. ``rich.progress`` self-suppresses in non-TTY shells
    (CI, pipe redirects), so the bar is invisible there without a flag.
    """
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=_PROGRESS_CONSOLE,
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=total)
        yield lambda: progress.update(task, advance=1)


# ---- Phase 1: ingest -----------------------------------------------------


async def ingest(
    path: str | Path | None = None,
    *,
    embedder: EmbeddingProvider | None = None,
    multimodal_embedder: MultimodalEmbeddingProvider | None = None,
    reporter: ProgressReporter | None = None,
) -> IngestReport:
    """Ingest every markdown file listed in ``sources:`` into the D and I layers.

    Two provider knobs — strictly per-channel, since query() searches
    text and multimodal vectors via separate version_ids:

    * ``embedder`` — text-only ``EmbeddingProvider``; populates the
      ``vec_chunks_v<text_version_id>`` table.
    * ``multimodal_embedder`` — ``MultimodalEmbeddingProvider``; when
      ``cfg.assets.multimodal`` is configured this embeds image-asset
      bytes into ``vec_assets_v<mm_version_id>``. It does NOT embed
      chunk text — chunks always flow through the text channel.

    ``reporter`` (optional) receives structured progress events for
    server-driven task wrappers; in-process callers leave it ``None`` and
    rely on the ``rich`` stderr progress bar instead.

    Asset binaries referenced from markdown are materialized into
    ``<root>/<assets.dir>/`` regardless of which embedder is set —
    ``chunk_asset_refs`` always reflect the on-disk structure so
    query-time consumers can render them.
    """
    cfg, root, storage = await _with_storage(path)
    owned_mm: MultimodalEmbeddingProvider | None = None
    _reporter: ProgressReporter = reporter or NoopReporter()
    try:
        report = IngestReport()
        to_embed: list[ChunkToEmbed] = []
        # Newly-materialized assets (deduped by asset_id) collected across
        # the run for batch image embedding at the end.
        new_assets_by_id: dict[str, AssetRecord] = {}

        # Resolve text + multimodal versions once per run so every chunk
        # and asset embedded below carries a stable version_id from the
        # embed_versions registry.
        text_version_id: int | None = None
        if embedder is not None:
            text_version_id = await _register_text_version(storage, cfg.provider)

        mm_version_id: int | None = None
        mm_cfg = cfg.assets.multimodal
        if mm_cfg is not None:
            mm_version_id = await storage.upsert_embed_version(
                EmbeddingVersion(
                    # Same endpoint-aware identity as the text leg —
                    # mm_cfg.provider names the wire shape, base_url
                    # selects the actual backend.
                    provider=_qualified_provider(
                        mm_cfg.provider, mm_cfg.base_url or ""
                    ),
                    model=mm_cfg.model,
                    revision=mm_cfg.revision,
                    dim=mm_cfg.dim,
                    normalize=mm_cfg.normalize,
                    distance=mm_cfg.distance,
                    modality="multimodal",
                )
            )

        # Auto-build the multimodal embedder from config when one wasn't
        # injected — symmetric with query()'s behavior, so the typical
        # server / eval call site (which only passes `embedder`) still
        # produces chunk + asset vectors in the configured mm space
        # rather than indexing chunks in text space and querying in
        # mm space.
        if mm_cfg is not None and multimodal_embedder is None:
            multimodal_embedder = build_multimodal_embedder(
                mm_cfg.provider,
                base_url=mm_cfg.base_url,
                batch=mm_cfg.batch,
            )
            owned_mm = multimodal_embedder

        await _reporter.progress(phase="scan", current=0, total=0)
        for abs_path, logical_path in iter_source_files(cfg.sources, root=root):
            _reporter.cancel_token().raise_if_cancelled()
            try:
                parsed = parse_any(abs_path, rel_path=logical_path)
            except UnsupportedFormat:
                # Glob swept up a non-md file (asset binary, .git/*, etc.).
                # Skip silently — surfacing every one would balloon the
                # error tape on a wide ``**/*`` glob without telling the
                # user anything they didn't already know about their
                # config. ``parse_error`` / ``read_error`` / ``storage_error``
                # are the user-actionable surfaces this PR opens up.
                continue
            except (OSError, UnicodeError) as e:
                # OSError covers filesystem refusals (permission denied,
                # file disappeared mid-scan); UnicodeError catches the
                # not-UTF-8 case — ``Path.read_text`` raises that one as
                # a ``ValueError`` subclass that would otherwise fall
                # into the parse_error catch-all and mislead callers
                # branching on ``kind``. Both are read-side failures.
                report = await _record_ingest_error(
                    report,
                    _reporter,
                    path=logical_path,
                    kind="read_error",
                    message=f"{type(e).__name__}: {e}",
                )
                continue
            except Exception as e:
                # Parser-side failure: invalid YAML front-matter,
                # malformed inline syntax our backend can't tolerate, etc.
                report = await _record_ingest_error(
                    report,
                    _reporter,
                    path=logical_path,
                    kind="parse_error",
                    message=f"{type(e).__name__}: {e}",
                )
                continue
            doc_id = _doc_id_for(Layer.SOURCE, logical_path)
            existing = await storage.get_document(doc_id)

            scanned = report.scanned + 1
            # Skip the chunk/embed pipeline only when the doc body is
            # unchanged AND the doc has no asset references AND the
            # row is currently active. A row deactivated by a prior
            # ``storage_error`` arm carries the same hash but a
            # half-indexed state — falling through re-runs the whole
            # pipeline and re-upserts ``active=True``.
            if (
                existing is not None
                and existing.active
                and existing.hash == parsed.hash
                and not parsed.asset_refs
            ):
                report = _replace(
                    report,
                    scanned=scanned,
                    unchanged=report.unchanged + 1,
                )
                continue

            try:
                # Materialize image references before chunking so asset_ids
                # are available when chunk_asset_refs land. Decoupled from
                # mm_cfg so eval rigs see the chunk ↔ asset bridge even
                # without multimodal embedding configured. ``persist_source``
                # itself doesn't touch the filesystem — it consumes the
                # already-resolved ``ref_assets`` dict.
                ref_assets: dict[int, AssetRecord] = {}
                if parsed.asset_refs:
                    by_original_path: dict[str, AssetRecord] = {}
                    try:
                        for ref_idx, ref in enumerate(parsed.asset_refs):
                            cached = by_original_path.get(ref.original_path)
                            if cached is not None:
                                ref_assets[ref_idx] = cached
                                continue
                            result = await materialize_asset(
                                ref,
                                source_md_path=abs_path,
                                project_root=root,
                                get_asset=storage.get_asset,
                                upsert_asset=storage.upsert_asset,
                                dir_=cfg.assets.dir,
                            )
                            if result is not None:
                                rec, was_new = result
                                ref_assets[ref_idx] = rec
                                by_original_path[ref.original_path] = rec
                                # Only new binaries need an embedding pass;
                                # existing rows keep their cached vector.
                                if was_new and mm_cfg is not None:
                                    new_assets_by_id.setdefault(rec.asset_id, rec)
                    except NotSupported:
                        ref_assets = {}

                persist_result = await persist_source(
                    storage=storage,
                    parsed=parsed,
                    doc=_to_document(parsed, doc_id=doc_id),
                    ref_assets=ref_assets,
                    cjk_tokenizer=cfg.retrieval.cjk_tokenizer,
                )

                # Queue chunks for embedding only when a text embedder + version
                # is wired up. Chunk vectors live exclusively in the text
                # channel (vec_chunks_v<text_id>); the multimodal channel
                # owns assets, not chunks. ``persist_source`` defers embed
                # to this end-of-scan bulk pass for throughput across files.
                if (
                    embedder is not None
                    and text_version_id is not None
                    and persist_result.chunks_count
                ):
                    to_embed.extend(
                        ChunkToEmbed(chunk_id=cid, text=text)
                        for cid, text in zip(
                            persist_result.chunk_ids,
                            persist_result.chunk_texts,
                            strict=True,
                        )
                    )

                await storage.append_knowledge_log(
                    KnowledgeLogEntry(ts=time.time(), action="ingest", src=logical_path)
                )
            except Exception as e:
                # Storage / chunking / asset materialisation raised
                # mid-file. ``upsert_document`` may already have landed
                # the doc row before the failure point — without an
                # explicit deactivation, the next ingest under an
                # unchanged content hash would hit the early-skip arm
                # above and the orphaned doc would stay half-indexed
                # forever. Deactivating bypasses the skip on retry so
                # the doc gets re-processed end-to-end.
                with contextlib.suppress(Exception):
                    await storage.deactivate_document(doc_id)
                report = await _record_ingest_error(
                    report,
                    _reporter,
                    path=logical_path,
                    kind="storage_error",
                    message=f"{type(e).__name__}: {e}",
                )
                continue

            report = _replace(
                report,
                scanned=scanned,
                added=report.added if existing is not None else report.added + 1,
                updated=report.updated + 1 if existing is not None else report.updated,
                chunks=report.chunks + persist_result.chunks_count,
            )
            await _reporter.progress(
                phase="scan",
                current=scanned,
                total=0,
                detail={"path": logical_path},
            )

        # W-layer scan was removed in 0.4.0 (#BREAKING). Wisdom pages
        # are indexed exclusively by ``write_wisdom_page`` (the public
        # ``dikw client wisdom write`` / ``POST /v1/wisdom/write`` entry
        # point). A user editing a wisdom file in obsidian no longer
        # gets it auto-reindexed by ``dikw client ingest``; the
        # ``list_chunks_missing_embedding`` resume scan below still
        # covers W-layer chunks that landed without vectors (e.g.,
        # ``wisdom write --no-embed`` or a flaky embed batch that hit
        # the retry-skip path in commit 1).

        # Resume scan: pick up chunks that landed in storage during a
        # prior crashed run but never got their embedding written. The
        # doc-level shortcut above skips docs whose body_hash matched
        # storage, so without this scan a half-embedded run can NEVER
        # finish — its remaining chunks are invisible to the per-doc
        # loop. The cache lookup in slice 5 makes this nearly free for
        # chunks whose text is already cached; for true misses (the
        # tail that crashed mid-flight) we re-pay the provider.
        if embedder is not None and text_version_id is not None:
            already_queued_ids = {c.chunk_id for c in to_embed}
            missing = await storage.list_chunks_missing_embedding(
                version_id=text_version_id
            )
            for chunk in missing:
                if chunk.chunk_id is None or chunk.chunk_id in already_queued_ids:
                    continue
                to_embed.append(
                    ChunkToEmbed(chunk_id=chunk.chunk_id, text=chunk.text)
                )

        # Chunk-text embeddings — text channel only. Streaming consume:
        # each batch is upserted as soon as the provider returns it, so
        # a mid-flight crash keeps the prior batches' vectors on disk
        # instead of throwing away the entire run's API spend.
        if to_embed and embedder is not None and text_version_id is not None:
            chunk_batch_size = cfg.provider.embedding_batch_size
            chunk_total = _ceil_div(len(to_embed), chunk_batch_size)
            with _embedding_progress(
                "embedding chunks", total=chunk_total
            ) as advance_chunk:
                embed_result = await consume_embedding_stream(
                    embed_chunks(
                        embedder,
                        to_embed,
                        model=cfg.provider.embedding_model,
                        version_id=text_version_id,
                        storage=storage,
                        batch_size=chunk_batch_size,
                        retries=cfg.provider.embedding_error_retries,
                        backoff_seconds=cfg.provider.embedding_error_retry_backoff_seconds,
                        reporter=_reporter,
                    ),
                    storage,
                    on_batch=advance_chunk,
                    reporter=_reporter,
                    phase="embed_chunks",
                    total=chunk_total,
                )
            report = _replace(report, embedded=embed_result.embedded)

        # Backfill assets stored without a vector for the active mm
        # version — text-only ingest residue, prior mm version, or
        # mid-flight crash of an earlier embed pass. Kept separate from
        # ``new_assets_by_id`` so ``report.assets`` (= NEW this run)
        # stays accurate; the union below is what we feed to the
        # embed pass. Gated on the same condition as the embed block
        # to avoid a no-op SQL round-trip when no mm embedder is wired.
        backfill_by_id: dict[str, AssetRecord] = {}
        if (
            multimodal_embedder is not None
            and mm_cfg is not None
            and mm_version_id is not None
        ):
            missing_assets = await storage.list_assets_missing_embedding(
                version_id=mm_version_id
            )
            # Skip categories ``embed_assets`` deliberately discards
            # without writing a meta row — they'd reappear in every
            # subsequent ingest's "needs embedding" list forever:
            #   * unembeddable mime (SVG today; v1 doesn't rasterize)
            #   * stored binary missing on disk (asset row points at a
            #     deleted file — first time we hit it, ``embed_assets``
            #     logs the read failure; the backfill scan should not
            #     keep re-reading + re-warning on every later ingest).
            candidates = [
                rec
                for rec in missing_assets
                if not is_unembeddable_asset_mime(rec.mime)
                and rec.asset_id not in new_assets_by_id
                and (root / rec.stored_path).is_file()
            ]
            # Filter to assets still referenced by at least one live
            # chunk. An asset whose markdown ref was deleted is
            # unreachable via ``HybridSearcher`` (which promotes asset
            # hits through ``chunks_referencing_assets``), so embedding
            # those orphans burns provider calls for vectors search
            # can never surface.
            if candidates:
                refs_by_asset = await storage.chunks_referencing_assets(
                    [rec.asset_id for rec in candidates]
                )
                for rec in candidates:
                    if refs_by_asset.get(rec.asset_id):
                        backfill_by_id[rec.asset_id] = rec

        to_embed_assets: dict[str, AssetRecord] = {
            **new_assets_by_id,
            **backfill_by_id,
        }
        if (
            multimodal_embedder is not None
            and mm_cfg is not None
            and mm_version_id is not None
            and to_embed_assets
        ):
            asset_total_batches = _ceil_div(len(to_embed_assets), mm_cfg.batch)
            asset_embedded = 0
            asset_batches_done = 0
            # Per-batch upsert: a mid-flight provider failure leaves
            # prior batches' vectors on disk so the next retry's
            # backfill scan sees only the truly-missing tail. Symmetric
            # with the chunk side's ``consume_embedding_stream``.
            with _embedding_progress(
                "embedding assets", total=asset_total_batches
            ) as advance_asset:
                async for batch_rows in embed_assets(
                    multimodal_embedder,
                    list(to_embed_assets.values()),
                    project_root=root,
                    model=mm_cfg.model,
                    version_id=mm_version_id,
                    batch_size=mm_cfg.batch,
                    retries=cfg.provider.embedding_error_retries,
                    backoff_seconds=cfg.provider.embedding_error_retry_backoff_seconds,
                ):
                    if batch_rows:
                        await storage.upsert_asset_embeddings(batch_rows)
                        asset_embedded += len(batch_rows)
                    advance_asset()
                    asset_batches_done += 1
                    await _reporter.progress(
                        phase="embed_assets",
                        current=asset_batches_done,
                        total=asset_total_batches,
                    )
                    _reporter.cancel_token().raise_if_cancelled()
            report = _replace(
                report,
                assets=len(new_assets_by_id),
                asset_embedded=asset_embedded,
            )
        elif new_assets_by_id:
            # Materialized assets even without an mm embedder so the chunk
            # references resolve at query/render time.
            report = _replace(report, assets=len(new_assets_by_id))

        return report
    finally:
        if owned_mm is not None and hasattr(owned_mm, "aclose"):
            await owned_mm.aclose()
        await storage.close()


def _to_document(
    parsed: ParsedDocument, *, doc_id: str, layer: Layer = Layer.SOURCE
) -> DocumentRecord:
    # ``status`` is wisdom-only — knowledge/source rows always store NULL
    # even if the user pasted ``status:`` into the wrong frontmatter.
    # The CHECK constraint allows NULL anywhere, so this clamp is the
    # invariant guard. See ``WisdomStatus`` docstring and
    # ``test_wiki_page_status_frontmatter_forced_to_null``.
    status = parsed.status if layer is Layer.WISDOM else None
    return DocumentRecord(
        doc_id=doc_id,
        path=parsed.path,
        title=parsed.title,
        hash=parsed.hash,
        mtime=parsed.mtime,
        layer=layer,
        active=True,
        status=status,
    )


def _replace(r: IngestReport, **kwargs: Any) -> IngestReport:
    # Thin wrapper around ``dataclasses.replace`` — kept for grep-ability
    # and to keep call-sites reading "_replace(report, scanned=…)" rather
    # than reaching for a stdlib import.
    return dataclasses.replace(r, **kwargs)


async def _record_ingest_error(
    report: IngestReport,
    reporter: ProgressReporter,
    *,
    path: str,
    kind: IngestErrorKind,
    message: str,
) -> IngestReport:
    """Append a per-file failure to the report and emit a wire event.

    Counts the failed file as scanned (so the report's ``scanned``
    matches "files we tried to process"), appends an :class:`IngestError`
    to ``report.errors``, and pushes a ``partial("file_error", …)``
    event so streaming subscribers (the CLI's progress widget, the task
    NDJSON stream) can surface the failure live instead of waiting for
    the final report.
    """
    err = IngestError(path=path, kind=kind, message=message)
    await reporter.partial(
        "file_error",
        {"path": path, "kind": kind, "message": message},
    )
    return IngestReport(
        scanned=report.scanned + 1,
        added=report.added,
        updated=report.updated,
        unchanged=report.unchanged,
        chunks=report.chunks,
        embedded=report.embedded,
        assets=report.assets,
        asset_embedded=report.asset_embedded,
        errors=(*report.errors, err),
    )


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


async def lint(path: str | Path | None = None) -> LintReport:
    """Run the K-layer hygiene checker."""
    _cfg, root, storage = await _with_storage(path)
    try:
        return await run_lint(storage, root=root)
    finally:
        await storage.close()


async def lint_propose(
    path: str | Path | None = None,
    *,
    rule: LintKind | None = None,
    limit: int = 10,
    llm: Any = None,
    embedder: Any = None,
    enable_llm: bool = False,
    reporter: ProgressReporter | None = None,
) -> FixProposalReport:
    """Run lint + dispatch fixers, returning a :class:`FixProposalReport`.

    ``enable_llm`` opts into LLM-powered fixer paths: the
    broken_wikilink evidence-backed grounded repair (D-layer hybrid
    search must yield enough evidence before the LLM is asked to write
    a real page; outputs containing ``TODO`` / ``stub page`` /
    ``placeholder`` markers are rejected), the entire non_atomic_page
    splitter, and orphan_page's ``merge_into_existing_page`` strategy.
    When False, propose runs heuristic-only — no LLM call is made and
    pure-heuristic paths (``broken_wikilink`` fuzzy-match,
    ``orphan_page`` delete/link/leaf strategies, ``missing_provenance``)
    still work. The default keeps a ``propose`` invocation cheap and
    deterministic; users opt in via ``--enable-llm``.

    ``llm`` / ``embedder`` are passthrough overrides used by tests; in
    production both are built from ``cfg.provider`` the same way
    :func:`synthesize` / :func:`retrieve` do, so ``$DIKW_*_API_KEY``
    resolution flows through one path. The embedder powers the D-layer
    hybrid evidence search the ``broken_wikilink`` grounded repair
    relies on — without it the search silently degrades to BM25 and
    semantically relevant but lexically distant evidence is missed.
    """
    cfg, root, storage = await _with_storage(path)
    try:
        report = await run_lint(storage, root=root)
        # Build KnowledgePageMeta + path→doc_id index in one pass over the
        # WIKI listing. ``report.page_meta`` already carries the
        # frontmatter slice ``run_lint`` parsed, so the orphan scorer's
        # sources/tags signal lands here without a second disk read.
        knowledge_docs = list(
            await storage.list_documents(layer=Layer.KNOWLEDGE, active=True)
        )
        all_pages: list[KnowledgePageMeta] = []
        path_to_doc_id: dict[str, str] = {}
        for doc in knowledge_docs:
            pm = report.page_meta.get(doc.path)
            all_pages.append(
                KnowledgePageMeta(
                    path=doc.path, title=doc.title,
                    sources=pm.sources if pm else (),
                    tags=pm.tags if pm else (),
                )
            )
            path_to_doc_id[doc.path] = doc.doc_id
        # Skip the LLM + embedder builds entirely on ``--enable-llm
        # False`` so the provider-import + key-lookup cost stays out
        # of heuristic-only propose runs. The embedder powers the
        # D-layer hybrid evidence search the broken_wikilink grounded
        # repair (#83) relies on — without it the search silently
        # degrades to BM25 and semantically relevant but lexically
        # distant evidence is missed.
        _llm: Any = llm
        _embedder: Any = embedder
        if enable_llm:
            if _llm is None:
                _llm = build_llm(cfg.provider, base_root=root)
            if _embedder is None:
                _embedder = build_embedder(cfg.provider)
        ctx = FixerContext(
            storage=storage,
            llm=_llm,
            embedding=_embedder,
            base_root=root,
            all_pages=all_pages,
            enable_llm=enable_llm,
            cfg=cfg,
            path_to_doc_id=path_to_doc_id,
        )
        used_reporter: ProgressReporter = reporter or NoopReporter()
        return await run_lint_propose(
            report=report,
            rule=rule,
            limit=limit,
            ctx=ctx,
            reporter=used_reporter,
        )
    finally:
        await storage.close()


async def lint_apply(
    path: str | Path | None = None,
    *,
    proposal_report: FixProposalReport,
    pick: list[int] | None = None,
    skip: list[int] | None = None,
    reporter: ProgressReporter | None = None,
    embedder: EmbeddingProvider | None = None,
) -> ApplyReport:
    """Mutate ``knowledge/`` per a previously-produced proposal report.

    When ``DIKW_EMBEDDING_API_KEY`` is configured (or the caller passes
    ``embedder``), Phase 1 re-embeds every rebuilt page inline so the
    fix is retrievable on return. Without the key, embedding defers to
    the next ``dikw client ingest``'s missing-embedding resume scan —
    the same fallback the W-layer ``no_embed`` write path uses.

    ``pick`` / ``skip`` filter the proposal list by index. Both may be
    set; pick is applied first, then skip removes from that subset.
    """
    cfg, root, storage = await _with_storage(path)
    try:
        used_reporter: ProgressReporter = reporter or NoopReporter()

        # Inline-embed when the user has an embedding key on hand. The
        # OpenAICompatEmbeddings provider only reads the key at embed
        # time, so a missing key wouldn't fail ``build_embedder`` — we
        # check the env up-front to keep apply heuristic-only when the
        # user hasn't configured embeddings yet.
        #
        # Reuse the active text embed version rather than registering a
        # new one from cfg: lint apply only re-embeds the pages it
        # changes, so flipping ``embed_versions.is_active`` here would
        # strand every other vector in the now-inactive table and gut
        # dense retrieval until ``dikw client ingest`` re-embeds the
        # full corpus. Synth uses the same pattern (api.py:2484-2497).
        active_embedder: EmbeddingProvider | None = embedder
        text_version_id: int | None = None
        embedding_model = ""
        if active_embedder is None and os.environ.get("DIKW_EMBEDDING_API_KEY"):
            active_embedder = build_embedder(cfg.provider)
        if active_embedder is not None:
            resolved = await _resolve_active_text_version_for_inline_embed(
                storage, cfg.provider
            )
            if resolved is not None:
                text_version_id, embedding_model = resolved
                # Preflight the embedder BEFORE Phase 0 mutates any
                # files — a permanent ProviderError (bad API key, 401,
                # invalid model id) raised mid-persist would otherwise
                # abort with files rewritten / deleted but storage
                # partially updated and no ApplyReport returned (codex
                # review finding, 0.4.0).
                await _preflight_embedder(active_embedder, embedding_model)
            else:
                # No active text version yet (fresh base, no ingest run)
                # or cfg drifted from active identity. Defer the
                # embedding to the next ingest's resume scan instead of
                # registering a new active version here or storing
                # vectors under the wrong version table.
                active_embedder = None

        return await run_lint_apply(
            proposal_report=proposal_report,
            storage=storage,
            base_root=root,
            pick=pick,
            skip=skip,
            reporter=used_reporter,
            cjk_tokenizer=cfg.retrieval.cjk_tokenizer,
            embedder=active_embedder,
            embedding_model=embedding_model,
            text_version_id=text_version_id,
            embedding_error_retries=cfg.provider.embedding_error_retries,
            embedding_error_retry_backoff_seconds=(
                cfg.provider.embedding_error_retry_backoff_seconds
            ),
        )
    finally:
        await storage.close()


# Per-(base, wisdom-path) locks serialise concurrent writers against the
# same logical wisdom file. Two HTTP submissions for the same
# ``(author, slug)`` would otherwise race: both call ``get_document``
# before either writes, both observe ``existing is None``, both report
# ``created=True``, and the storage row + chunks + on-disk file can end
# up describing different submissions depending on interleaving. The
# task system schedules submissions concurrently, so even single-base
# single-process deployments need this guard.
_WISDOM_WRITE_LOCKS: dict[str, asyncio.Lock] = {}
_WISDOM_WRITE_LOCKS_GUARD = asyncio.Lock()


async def _acquire_wisdom_write_lock(key: str) -> asyncio.Lock:
    """Return (creating if needed) the asyncio.Lock for ``key``.

    The guard lock around the dict avoids two writers racing on the
    ``setdefault``-style check itself. Lock objects accumulate in the
    process for the lifetime of the engine; the cardinality is bounded
    by the number of distinct wisdom paths the base ever sees, which is
    fine for a per-user knowledge base.
    """
    async with _WISDOM_WRITE_LOCKS_GUARD:
        lock = _WISDOM_WRITE_LOCKS.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _WISDOM_WRITE_LOCKS[key] = lock
        return lock


async def write_wisdom_page(
    path: str | Path | None = None,
    *,
    slug: str,
    title: str,
    body: str,
    author: str | None = None,
    status: WisdomStatus | None = None,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    extras: dict[str, object] | None = None,
    no_embed: bool = False,
    embedder: EmbeddingProvider | None = None,
    reporter: ProgressReporter | None = None,
) -> WisdomWriteReport:
    """Create or update a wisdom page at ``wisdom/[<author>/]<slug>.md``.

    Upsert semantics — re-writing the same ``(author, slug)`` overwrites
    the existing file and refreshes its storage row, chunks, embeddings,
    outgoing wikilinks, and provenance edges. The caller is responsible
    for reading the current state first if a no-overwrite contract is
    needed (read ``GET /v1/base/pages/<path>`` and decide before POST).

    ``slug`` and ``author`` are validated to ASCII kebab-case; the
    on-disk path becomes part of the wisdom vault layout, so the engine
    refuses anything that would render awkwardly in Obsidian (spaces,
    uppercase, underscores).

    When ``no_embed`` is true the file is written + chunked + linked
    but vectors are not produced; the next ``dikw client ingest``
    resolves embeddings via the missing-embedding resume scan. With
    ``no_embed`` false the engine builds an embedder from the base
    config — a missing API key or unreachable provider surfaces as the
    embedder factory's error and fails the write (open
    ``no_embed=True`` if the caller wants to author wisdom without an
    embedding provider configured). With a real embedder the page is
    retrievable immediately on return.

    Concurrent writes to the same ``(author, slug)`` are serialised by
    a per-path async lock so two HTTP submissions never both claim
    ``created=True`` against an empty document row.

    Cross-file ``[[wikilink]]`` resolution uses a freshly built
    cross-layer title index (knowledge + wisdom), so a wisdom page linking
    an existing knowledge or wisdom title resolves on this single write. A
    forward reference to a wisdom page that hasn't been written yet
    surfaces on ``WisdomWriteReport.unresolved_wikilinks`` and is
    reconciled on the next ``dikw client ingest``.
    """
    # Lazy import — domains.knowledge.page_index imports schemas + storage
    # primitives, and api.py imports those eagerly already; importing
    # the persist functions eagerly here would re-introduce the same
    # circular risk the existing wisdom-branch (1320 area) avoided by
    # deferring. ``persist_wisdom`` is imported lazily at its single
    # use site below.
    from .domains.knowledge.links import build_title_indexes
    from .domains.wisdom import make_wisdom_path, write_wisdom_file

    # Validate first — never write a file or open storage on a malformed
    # input. ``make_wisdom_path`` re-runs ``validate_kebab`` so a direct
    # Python caller bypassing the Pydantic schema still gets the same
    # invariant.
    logical_path = make_wisdom_path(slug=slug, author=author)
    used_reporter: ProgressReporter = reporter or NoopReporter()

    # Key locks by (canonical base root, logical wisdom path). Resolve
    # the base via ``resolve_base_root`` *before* acquiring the lock so
    # two callers targetting the same base through different aliases
    # (the base dir, the dikw.yml file inside it, a relative path)
    # map to the same lock. ``resolve_base_root`` is a pure path walk
    # + config-file lookup with no side effects, so running it here
    # and again inside ``_with_storage`` is cheap and safe.
    canonical_root = resolve_base_root(path).resolve()
    lock_key = f"{canonical_root}::{logical_path}"
    lock = await _acquire_wisdom_write_lock(lock_key)
    async with lock:
        cfg, root, storage = await _with_storage(path)
        try:
            doc_id = _doc_id_for(Layer.WISDOM, logical_path)
            existing = await storage.get_document(doc_id)
            created = existing is None or not existing.active

            # Resolve embedder: caller-injected wins (tests pass a fake);
            # otherwise build from cfg unless ``no_embed`` defers it. The
            # build_embedder call is the one that touches the env API key —
            # keep it inside the ``not no_embed`` branch so a write that
            # explicitly opted out doesn't fail on a missing key.
            #
            # Reuse the active text embed version rather than registering
            # a new one from cfg: wisdom write only embeds the page being
            # written, so flipping ``embed_versions.is_active`` here
            # would strand every other vector in the now-inactive table
            # and gut dense retrieval until ``dikw client ingest`` re-
            # embeds the full corpus. Mirrors synth's pattern
            # (api.py:2484-2497) and lint apply's pattern above.
            active_embedder: EmbeddingProvider | None = None
            text_version_id: int | None = None
            embedding_model = ""
            if not no_embed:
                active_embedder = embedder
                if active_embedder is None:
                    active_embedder = build_embedder(cfg.provider)
                resolved = await _resolve_active_text_version_for_inline_embed(
                    storage, cfg.provider
                )
                if resolved is not None:
                    text_version_id, embedding_model = resolved
                    # Preflight before the file write so a permanent
                    # ProviderError surfaces while state is still clean.
                    # Mirrors the lint apply preflight (codex review
                    # finding, 0.4.0).
                    await _preflight_embedder(active_embedder, embedding_model)
                else:
                    # Either no active text version yet OR cfg has drifted
                    # from the active identity — defer embedding to the
                    # next ingest's resume scan instead of activating a
                    # fresh version here or storing vectors under the
                    # wrong version table.
                    active_embedder = None

            # Cooperative cancellation poll: synth/lint/ingest all check
            # the token between work units (api.py:1130 / 1353 / 1589 /
            # 2717), so a wisdom-write task is expected to honour the same
            # contract. Without poll sites, ``TaskManager.cancel`` only
            # lands at the next httpx ``await`` deep inside the embedder.
            used_reporter.cancel_token().raise_if_cancelled()

            # Write the file to disk first. ``persist_wisdom`` re-parses the
            # written file so the stored hash and chunk offsets match what
            # ``read_page`` will compute — same contract as
            # ``_persist_knowledge_page``. ``frontmatter.dumps`` is not always
            # byte-stable on the body, so hashing ``body`` directly and
            # chunking ``body`` would diverge from the read-back parsed body.
            write_wisdom_file(
                root,
                logical_path=logical_path,
                title=title,
                body=body,
                status=status,
                tags=tags,
                sources=sources,
                extras=extras,
            )

            # Build the cross-layer title index that ``persist_wisdom`` uses
            # for ``[[wikilink]]`` resolve. Mirrors the ingest wisdom branch
            # (around line 1330) - include the new page's own NEW title so
            # a self-reference resolves, but EXCLUDE the stored row for
            # ``logical_path`` itself: on an update that changes the page's
            # title, the storage row still holds the old title until
            # ``persist_wisdom`` rewrites it, and pulling that into the index
            # would let ``[[Old Title]]`` in the new body resolve to this
            # same page via the stale title.
            title_docs: list[tuple[str, str]] = []
            for layer_for_index in (Layer.KNOWLEDGE, Layer.WISDOM):
                for d in await storage.list_documents(
                    layer=layer_for_index, active=True
                ):
                    if d.path == logical_path:
                        continue
                    if d.title:
                        title_docs.append((d.title, d.path))
            title_docs.append((title, logical_path))
            title_to_path, fuzzy_index = build_title_indexes(title_docs)

            used_reporter.cancel_token().raise_if_cancelled()
            await used_reporter.progress(
                phase="wisdom_write",
                current=0,
                total=0,
                detail={"path": logical_path, "step": "indexing"},
            )

            # ``persist_wisdom`` writes documents + chunks + FTS + embeddings
            # + links + provenance in sequence. A mid-pipeline failure
            # (embed timeout, link resolve crash, vec_search error) leaves
            # those rows partially updated — the document row + chunks
            # already reflect the new content while outgoing links /
            # provenance edges still point at the old. ``deactivate_document``
            # flips the doc to inactive so the next ``dikw ingest`` resume
            # scan rebuilds the row end-to-end.
            from .domains.wisdom.persist import persist_wisdom

            try:
                persist_result = await persist_wisdom(
                    storage=storage,
                    root=root,
                    path=logical_path,
                    title=title,
                    embedder=active_embedder,
                    embedding_model=embedding_model,
                    text_version_id=text_version_id,
                    cjk_tokenizer=cfg.retrieval.cjk_tokenizer,
                    title_to_path=title_to_path,
                    fuzzy_index=fuzzy_index,
                    retries=cfg.provider.embedding_error_retries,
                    backoff_seconds=cfg.provider.embedding_error_retry_backoff_seconds,
                )
                unresolved = persist_result.unresolved_wikilinks
            except (Exception, asyncio.CancelledError):
                # ``asyncio.CancelledError`` inherits from ``BaseException``
                # so a bare ``except Exception`` misses it. Cancellation
                # mid-``persist_wisdom`` can leave the doc row + chunks
                # rewritten but links / provenance stale; deactivating
                # ensures the next ingest's resume scan rebuilds the row
                # end-to-end. (CodeRabbit finding, 0.4.0).
                with contextlib.suppress(Exception):
                    await storage.deactivate_document(doc_id)
                raise

            used_reporter.cancel_token().raise_if_cancelled()

            chunks = await storage.list_chunks(doc_id)
            doc = await storage.get_document(doc_id)
            if doc is None:
                raise RuntimeError(
                    f"wisdom write succeeded but document {doc_id!r} not in storage"
                )

            # Read back actual embedded count rather than assuming
            # ``len(chunks)``. ``consume_embedding_stream`` can complete
            # with partial coverage when the provider retries-and-gives-up
            # on a per-batch failure without raising — ``embedded =
            # len(chunks)`` would then lie to the caller about retrieval
            # readiness. ``get_chunk_embeddings`` returns hits-only, so
            # ``len(...)`` is the truthful count of vectors persisted for
            # this doc at ``text_version_id``.
            if active_embedder is not None and text_version_id is not None:
                chunk_ids = [c.chunk_id for c in chunks if c.chunk_id is not None]
                vecs = await storage.get_chunk_embeddings(
                    chunk_ids, version_id=text_version_id
                )
                embedded = len(vecs)
            else:
                embedded = 0

            await storage.append_knowledge_log(
                KnowledgeLogEntry(
                    ts=time.time(), action="wisdom_write", src=logical_path
                )
            )

            await used_reporter.progress(
                phase="wisdom_write",
                current=1,
                total=1,
                detail={"path": logical_path, "step": "done"},
            )

            # ``chunks_pending_embedding`` mirrors ApplyReport — non-zero
            # when no_embed=True, when the inline-embed path deferred
            # (no active text version yet, or cfg drifted), or when
            # transient embed failures exhausted the retry budget.
            chunks_pending_embedding = len(chunks) - embedded
            return WisdomWriteReport(
                path=logical_path,
                created=created,
                hash=doc.hash,
                chunks=len(chunks),
                embedded=embedded,
                chunks_pending_embedding=chunks_pending_embedding,
                unresolved_wikilinks=unresolved,
            )
        finally:
            # Match ``_with_storage``'s connect-failure path
            # (api.py:665): a close() error must not shadow the real
            # cause carried by the in-flight exception, otherwise the
            # task manager records "OperationalError: connection
            # closed" instead of "embedder timed out".
            with contextlib.suppress(Exception):
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


