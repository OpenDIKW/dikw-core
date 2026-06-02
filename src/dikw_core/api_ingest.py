"""Ingest cluster of the engine facade.

``ingest`` is the D-layer write entry: walk ``<base>/sources/``, parse +
chunk + index every new/changed markdown file, materialize referenced
image assets, and embed chunks (text channel) + assets (multimodal
channel) in one bulk pass at end-of-scan. Idempotent — files whose
content hash is unchanged are skipped.

rank3 cluster: imports ``api_core`` (``_with_storage`` /
``_register_text_version`` / ``_qualified_provider``), providers, the
D-layer persist/asset pipeline, and the leaf ``api_types`` DTOs — never
the ``api`` facade. ``api`` re-exports ``ingest`` (public, in ``__all__``).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from collections.abc import Callable, Iterator
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

from .api_core import _qualified_provider, _register_text_version, _with_storage
from .api_types import IngestError, IngestErrorKind, IngestReport
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
from .progress import NoopReporter, ProgressReporter
from .providers import (
    EmbeddingProvider,
    MultimodalEmbeddingProvider,
    build_multimodal_embedder,
)
from .schemas import (
    AssetRecord,
    DocumentRecord,
    EmbeddingVersion,
    KnowledgeLogEntry,
    Layer,
)
from .storage.base import NotSupported

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
            # row is currently active AND its stored mtime is usable.
            # A row deactivated by a prior ``storage_error`` arm carries
            # the same hash but a half-indexed state — falling through
            # re-runs the whole pipeline and re-upserts ``active=True``.
            # A row whose stored ``mtime`` is broken (``<= 0``, a legacy
            # byte-stable import) likewise falls through so it re-persists
            # once and self-heals (#145).
            if (
                existing is not None
                and existing.active
                and existing.hash == parsed.hash
                and not parsed.asset_refs
                and existing.mtime > 0
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
                    doc=_to_document(
                        parsed,
                        doc_id=doc_id,
                        mtime=_resolve_ingest_mtime(parsed.mtime, parsed.hash, existing),
                    ),
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
            except asyncio.CancelledError:
                # CancelledError inherits from BaseException, so the
                # ``except Exception`` below misses it. ``upsert_document``
                # may already have committed the doc row (``active=True``)
                # before the cancellation point, so deactivate the in-flight
                # doc — otherwise the next ingest under an unchanged hash hits
                # the early-skip arm above and the half-indexed doc stays
                # active forever. Then re-raise to abort the run (cancellation
                # is not "continue with the next file"). Parity with the K
                # (synth / lint apply) and W cancel handlers.
                with contextlib.suppress(Exception):
                    await storage.deactivate_document(doc_id)
                raise
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


def _resolve_ingest_mtime(
    parsed_mtime: float, parsed_hash: str, existing: DocumentRecord | None
) -> float:
    """Fall back to ingest wall-clock when the source file carries no usable
    mtime. A byte-stable import tarball (dikw-web) zeroes the tar ``mtime``
    field so identical bytes dedup, so the extracted file lands with
    ``st_mtime == 0``; storing that renders as ``1970-01-01`` and feeds the
    graph change-hash a constant. Preserve an already-stored positive mtime
    only when the body is unchanged (same hash) — e.g. an image-bearing doc
    that re-persists every ingest — so it doesn't flap. A genuine content
    change (hash differs) still gets a fresh wall-clock so rendered dates and
    ``since_ts`` sync cursors advance. See #145.
    """
    if parsed_mtime > 0:
        return parsed_mtime
    if existing is not None and existing.mtime > 0 and existing.hash == parsed_hash:
        return existing.mtime
    return time.time()


def _to_document(
    parsed: ParsedDocument, *, doc_id: str, layer: Layer = Layer.SOURCE, mtime: float | None = None
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
        mtime=parsed.mtime if mtime is None else mtime,
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
