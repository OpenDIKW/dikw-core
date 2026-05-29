"""Lint cluster of the engine facade: ``lint`` / ``lint_propose`` /
``lint_apply``.

``lint`` reports K-layer hygiene issues (broken wikilinks, orphans,
duplicate titles, missing provenance); ``lint_propose`` dispatches fixers
into a :class:`FixProposalReport`; ``lint_apply`` mutates ``knowledge/``
per a previously-produced report, re-embedding rebuilt pages inline when
an embedder is reachable.

rank3 cluster: imports ``api_core`` (``_with_storage`` /
``_preflight_embedder`` / ``_resolve_active_text_version_for_inline_embed``),
providers, and the K-layer lint primitives — never the ``api`` facade.
``api`` re-exports ``lint`` (public, in ``__all__``); ``lint_propose`` /
``lint_apply`` are reached through the facade by the server routes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .api_core import (
    _preflight_embedder,
    _resolve_active_text_version_for_inline_embed,
    _with_storage,
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
from .progress import NoopReporter, ProgressReporter
from .providers import EmbeddingProvider, build_embedder, build_llm
from .schemas import Layer


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
