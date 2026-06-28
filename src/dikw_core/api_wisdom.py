"""Wisdom-write cluster of the engine facade: ``write_wisdom_page``.

W layer is hand-written first-class documents under
``wisdom/[<author>/]<slug>.md``. ``write_wisdom_page`` is the sole
programmatic write entry — it owns the full upsert + chunk + FTS +
inline-embed + link + provenance pipeline (via ``persist_wisdom``), with
a per-(base, path) async lock serialising concurrent writers.

rank3 cluster: imports ``api_core`` (``_with_storage`` /
``resolve_base_root`` / ``_resolve_active_text_version_for_inline_embed``
/ ``_preflight_embedder``), providers, and — lazily — the W-layer persist
pipeline, never the ``api`` facade. ``api`` re-exports ``write_wisdom_page``
(public, in ``__all__``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path

from .api_core import (
    _preflight_embedder,
    _resolve_active_text_version_for_inline_embed,
    _with_storage,
    resolve_base_root,
)
from .domains.data.path_norm import doc_id_for as _doc_id_for
from .progress import NoopReporter, ProgressReporter
from .providers import EmbeddingProvider, build_embedder
from .schemas import KnowledgeLogEntry, Layer, WisdomStatus, WisdomWriteReport

logger = logging.getLogger(__name__)

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
                    # wrong version table. Warn: embedding was requested
                    # (``not no_embed``) but the page lands without vectors
                    # until that resume scan, so surface the deferral rather
                    # than silently shipping a vector-less wisdom page.
                    logger.warning(
                        "wisdom write '%s': embedding deferred (no active text "
                        "version or cfg drifted from the active identity); a "
                        "future ingest will reconcile its vectors",
                        logical_path,
                    )
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
