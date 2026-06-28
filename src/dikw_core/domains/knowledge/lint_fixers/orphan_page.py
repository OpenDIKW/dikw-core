"""Fixer for ``orphan_page`` issues.

An orphan page has no inbound K-layer wikilinks. The fixer triages
each orphan into one of four reviewable strategies, picked in order
from most conservative to most invasive:

1. ``delete_page`` — body is a near-empty stub with no outbound
   wikilinks; soft-delete to ``trash/``.
2. ``merge_into_existing_page`` — a candidate parent scores above
   ``MERGE_THRESHOLD`` and ``ctx.enable_llm`` is on. The LLM rewrites
   the parent body to absorb the orphan; apply emits two ops
   ``[update_page(parent), delete_page(orphan)]``.
3. ``link_from_existing_page`` — a candidate parent scores above
   ``LINK_THRESHOLD``; emit an ``update_page`` op that appends a
   ``[[orphan]]`` reference to the parent's body.
4. ``mark_as_leaf`` — nothing else applies. Write
   ``lint: {skip: [orphan_page], reason}`` into the orphan's
   frontmatter so the next lint pass treats it as an acknowledged
   terminal note. Always-on tail strategy — better an actionable
   proposal (the user can reject it) than a silent skip.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter

from .... import prompts
from ....schemas import Layer, LinkType
from ....storage.base import NotSupported
from ..links import normalize_base, parse_links
from ..lint import LintKind
from ..lint_fix import (
    FixerContext,
    FixOperation,
    FixProposal,
    KnowledgePageMeta,
    bytes_sha256,
    safe_synthesize_pages,
)
from ..page import category_from_path
from ..synthesize import DEFAULT_ALLOWED_CATEGORIES

logger = logging.getLogger(__name__)

#: A candidate parent must score this high to win the
#: ``link_from_existing_page`` strategy. Below this, the fixer falls
#: through to mark_as_leaf rather than manufacture a low-confidence
#: backlink (cf. Karpathy's "deterministic scoping" rule — refuse the
#: dubious match, surface it as a leaf the user can review).
#:
#: Calibrated against the scoring weights below: a single shared
#: ``sources`` entry already clears it, while one shared tag plus
#: thin title overlap does not.
LINK_THRESHOLD = 3.0

#: A candidate parent must score this high AND ``ctx.enable_llm`` must
#: be on for the fixer to attempt a merge. Higher than ``LINK_THRESHOLD``
#: because merge rewrites the parent body and soft-deletes the orphan —
#: an irreversible-ish change at the page level (trash recovery exists,
#: but the merged body is what the user gets back). Two shared sources,
#: or one shared source plus a very strong embedding match, clear it.
MERGE_THRESHOLD = 6.0

#: Token budget for the merge LLM call. The merged body can be roughly
#: ``parent + orphan`` in length plus light restructuring, so this needs
#: more headroom than the broken_wikilink evidence-backed grounded
#: repair call.
_MERGE_MAX_TOKENS = 4096

_MERGE_SYSTEM = prompts.load("lint_fix_orphan_merge_system")

# --- scoring weights ---------------------------------------------------------
_W_SHARED_SOURCE = 3.0
_W_SHARED_TAG = 1.0
_W_SHARED_TAG_DOMAIN = 0.5
_W_TITLE_JACCARD = 2.0
#: Multiplied by the cosine similarity in [0, 1]; a perfect match
#: contributes the same weight as a single shared source.
_W_EMBED_SIMILARITY = 3.0
#: Per-orphan chunk over-fetch budget for the embedding leg. Each of
#: the orphan's chunk embeddings drives one ``vec_search`` call; this
#: caps the candidate pool size that gets folded into per-doc max
#: similarity.
_EMBED_VEC_LIMIT = 20

#: Maximum body size (post-frontmatter strip, UTF-8 bytes) that can
#: still qualify as a deletable stub. The byte gate is a NECESSARY
#: condition — short prose passes it — but never sufficient on its
#: own (see ``_is_deletable_stub``). The combined gate protects
#: legitimate one-sentence facts / definitions from auto-deletion.
_STUB_BODY_BYTES = 40

#: Stub-marker patterns. A body whose stripped content matches any of
#: these (case-insensitive, allowing trailing prose like
#: "TODO: write this page") is treated as an intentional placeholder,
#: independent of metadata richness.
_STUB_PATTERN = re.compile(
    r"^\s*(?:#\s*\S.*\n+)?\s*(?:todo|fixme|wip|stub|placeholder|draft)\b",
    flags=re.IGNORECASE,
)

def _title_tokens(title: str | None) -> frozenset[str]:
    """Tokenize via the canonical K-layer title-normalize (NFKC +
    casefold + trailing-boundary strip). Shared with the wikilink
    fuzzy-resolver so a title that resolves as ``[[Foo Bar]]`` and a
    parent-candidate score on ``Foo Bar`` use the exact same tokens."""
    if not title:
        return frozenset()
    return frozenset(normalize_base(title).split())


def _tag_domains(tags: tuple[str, ...]) -> frozenset[str]:
    return frozenset(
        t.split("/", 1)[0].strip() for t in tags
        if isinstance(t, str) and "/" in t
    )


@dataclass(frozen=True)
class _ScoredCandidate:
    page: KnowledgePageMeta
    score: float
    reason: str  # human-readable breakdown of why this score


@dataclass(frozen=True)
class _OrphanSignals:
    """Orphan-side scoring inputs precomputed once per orphan.

    ``_score_candidate`` runs once per candidate (O(N_pages) per orphan).
    Computing the orphan's source-set / tag-set / tag-domains / title
    tokens fresh on every call would do N_pages * 4 redundant set
    operations; this struct hoists them.
    """

    page: KnowledgePageMeta
    sources: frozenset[str]
    tags: frozenset[str]
    tag_domains: frozenset[str]
    title_tokens: frozenset[str]


def _orphan_signals(orphan: KnowledgePageMeta) -> _OrphanSignals:
    return _OrphanSignals(
        page=orphan,
        sources=frozenset(orphan.sources),
        tags=frozenset(orphan.tags),
        tag_domains=_tag_domains(orphan.tags),
        title_tokens=_title_tokens(orphan.title),
    )


def _score_candidate(
    orphan: _OrphanSignals,
    candidate: KnowledgePageMeta,
    *,
    embed_similarity: float = 0.0,
) -> _ScoredCandidate:
    """Heuristic score for "how likely is ``candidate`` to be a meaningful
    parent of ``orphan``". Each signal is independent and additive.

    ``embed_similarity`` is the max cosine similarity in [0, 1]
    between any orphan chunk and any candidate chunk; pass 0.0 (the
    default) when storage / embeddings aren't available.
    """
    if candidate.path == orphan.page.path:
        return _ScoredCandidate(page=candidate, score=0.0, reason="self")

    shared_sources = orphan.sources & frozenset(candidate.sources)
    shared_tags = orphan.tags & frozenset(candidate.tags)
    shared_domains = (
        orphan.tag_domains & _tag_domains(candidate.tags)
    ) - {t.split("/", 1)[0] for t in shared_tags}  # don't double-count

    candidate_tokens = _title_tokens(candidate.title)
    if orphan.title_tokens and candidate_tokens:
        union = orphan.title_tokens | candidate_tokens
        jaccard = len(orphan.title_tokens & candidate_tokens) / len(union)
    else:
        jaccard = 0.0

    embed_clamped = max(0.0, min(1.0, embed_similarity))

    score = (
        _W_SHARED_SOURCE * len(shared_sources)
        + _W_SHARED_TAG * len(shared_tags)
        + _W_SHARED_TAG_DOMAIN * len(shared_domains)
        + _W_TITLE_JACCARD * jaccard
        + _W_EMBED_SIMILARITY * embed_clamped
    )
    parts: list[str] = []
    if shared_sources:
        parts.append(f"{len(shared_sources)} shared source(s)")
    if shared_tags:
        parts.append(f"{len(shared_tags)} shared tag(s)")
    if shared_domains:
        parts.append(f"{len(shared_domains)} shared tag domain(s)")
    if jaccard > 0:
        parts.append(f"title jaccard {jaccard:.2f}")
    if embed_clamped > 0:
        parts.append(f"embed similarity {embed_clamped:.2f}")
    return _ScoredCandidate(
        page=candidate, score=score, reason="; ".join(parts) or "no overlap",
    )


async def _embedding_similarity_by_path(
    *, ctx: FixerContext, orphan_meta: KnowledgePageMeta,
) -> dict[str, float]:
    """Per-candidate max cosine similarity from the orphan's chunk
    embeddings, keyed by candidate path. Returns ``{}`` if storage is
    unavailable, the wiki has no text-embedding version yet, or the
    orphan has no embeddings recorded.

    Strategy:
    1. Pull the orphan's chunks via ``list_chunks``.
    2. Fetch each chunk's embedding vector (active text version).
    3. Run ``vec_search`` per vector against WIKI; aggregate hits to
       ``doc_id → min_distance``.
    4. Resolve ``doc_id → path`` via the WIKI document list.
    5. Convert distance → similarity (``1 - distance``).

    Single ``NotSupported`` on any step makes the whole leg degrade
    silently to ``{}`` — the pure-heuristic path stays usable.
    """
    storage = ctx.storage
    if storage is None:
        return {}
    # ctx.path_to_doc_id is built once at propose-task startup
    # (api.lint_propose) from the same ``list_documents`` call that
    # built ``all_pages`` — using it here saves a per-orphan listing.
    orphan_doc_id = ctx.path_to_doc_id.get(orphan_meta.path)
    if orphan_doc_id is None:
        return {}
    path_by_doc_id = {
        doc_id: path for path, doc_id in ctx.path_to_doc_id.items()
    }

    try:
        chunks = await storage.list_chunks(orphan_doc_id)
    except NotSupported:
        return {}
    chunk_ids = [c.chunk_id for c in chunks if c.chunk_id is not None]
    if not chunk_ids:
        return {}

    try:
        embeddings = await storage.get_chunk_embeddings(chunk_ids)
    except NotSupported:
        return {}
    if not embeddings:
        return {}

    # vec_search per chunk in parallel — on Postgres each is a network
    # round-trip, so K chunks serialised would be K hops; gather lets
    # the connection pool overlap them. SQLite serialises on its
    # connection regardless, so no harm there.
    #
    # ``layer=Layer.KNOWLEDGE`` is critical: without it, a base with many
    # SOURCE/WISDOM chunks routinely fills the limit with non-wiki hits,
    # and those get discarded later (absent from path_to_doc_id) so a
    # genuinely-similar wiki parent never enters the candidate pool.
    try:
        results = await asyncio.gather(
            *(
                storage.vec_search(
                    emb, limit=_EMBED_VEC_LIMIT, layer=Layer.KNOWLEDGE,
                )
                for emb in embeddings.values()
            )
        )
    except NotSupported:
        return {}
    min_distance_by_doc: dict[str, float] = {}
    for hits in results:
        for hit in hits:
            if hit.doc_id == orphan_doc_id:
                continue
            prior = min_distance_by_doc.get(hit.doc_id)
            if prior is None or hit.distance < prior:
                min_distance_by_doc[hit.doc_id] = hit.distance

    similarity_by_path: dict[str, float] = {}
    for doc_id, dist in min_distance_by_doc.items():
        path = path_by_doc_id.get(doc_id)
        if path is None:
            continue
        # Cosine distance is in [0, 2]; similarity = 1 - distance maps
        # the typical [0, 1] cosine-distance band to [0, 1] similarity.
        # Distances above 1 (extreme dissimilarity) clamp to 0 in the
        # scorer's ``embed_clamped``.
        similarity_by_path[path] = 1.0 - dist
    return similarity_by_path

#: Reason recorded in the frontmatter ``lint.reason`` when the fixer
#: picks the leaf strategy because no better candidate existed. The
#: string is user-facing — keep it short and explanatory.
_LEAF_REASON_NO_PARENT = (
    "no high-confidence parent or merge candidate; acknowledged as leaf"
)


class OrphanPageFixer:
    kind: LintKind = "orphan_page"

    async def propose(
        self,
        issue: Any,
        ctx: FixerContext,
        reporter: Any,
    ) -> FixProposal | None:
        orphan_meta = next(
            (p for p in ctx.all_pages if p.path == issue.path), None,
        )
        if orphan_meta is None:
            # Orphan vanished between lint scan and fix proposal.
            return None

        delete_proposal = _propose_delete_stub(
            issue=issue, ctx=ctx, orphan_meta=orphan_meta,
        )
        if delete_proposal is not None:
            return delete_proposal

        # Ambiguous-title orphans gate out BOTH merge and link strategies:
        # the wikilink the apply step would write back into the parent
        # cannot disambiguate, so merge would silently drop the orphan
        # and link would keep zero-inbound. Fall through to leaf.
        if orphan_meta.title and _title_is_ambiguous(orphan_meta, ctx):
            return _propose_mark_as_leaf(
                issue=issue, ctx=ctx, reason=_LEAF_REASON_NO_PARENT,
            )

        embedding_sim = await _embedding_similarity_by_path(
            ctx=ctx, orphan_meta=orphan_meta,
        )
        scored = _rank_candidates(
            orphan_meta=orphan_meta, ctx=ctx, embedding_sim=embedding_sim,
        )
        best = scored[0] if scored else None

        if (
            best is not None
            and best.score >= MERGE_THRESHOLD
            and ctx.enable_llm
            and ctx.llm is not None
            and ctx.cfg is not None
        ):
            merge_proposal = await _propose_merge_into_existing(
                issue=issue, ctx=ctx, orphan_meta=orphan_meta, best=best,
            )
            if merge_proposal is not None:
                return merge_proposal
            # LLM failed / returned no usable page; fall through to link.

        if best is not None and best.score >= LINK_THRESHOLD:
            link_proposal = _build_link_proposal(
                issue=issue, ctx=ctx, orphan_meta=orphan_meta, best=best,
            )
            if link_proposal is not None:
                return link_proposal

        return _propose_mark_as_leaf(
            issue=issue, ctx=ctx, reason=_LEAF_REASON_NO_PARENT,
        )


def _title_is_ambiguous(orphan: KnowledgePageMeta, ctx: FixerContext) -> bool:
    """``True`` when ≥ 2 K-pages share the orphan's title.

    A backlink ``[[Title]]`` written by the link or merge strategy
    resolves to the FIRST page found with that title, so any backlink
    targeting an ambiguous title risks landing on the wrong page. The
    user must first resolve the ``duplicate_title`` issue.
    """
    return sum(1 for p in ctx.all_pages if p.title == orphan.title) > 1


def _rank_candidates(
    *,
    orphan_meta: KnowledgePageMeta,
    ctx: FixerContext,
    embedding_sim: dict[str, float],
) -> list[_ScoredCandidate]:
    """Score every K-page (other than the orphan) and return descending.

    Pages scoring 0 are dropped — they contribute nothing to either
    strategy's decision and pollute the rationale string.
    """
    orphan = _orphan_signals(orphan_meta)
    scored: list[_ScoredCandidate] = []
    for candidate in ctx.all_pages:
        if candidate.path == orphan_meta.path:
            continue
        s = _score_candidate(
            orphan, candidate,
            embed_similarity=embedding_sim.get(candidate.path, 0.0),
        )
        if s.score > 0:
            scored.append(s)
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored


def _propose_delete_stub(
    *, issue: Any, ctx: FixerContext, orphan_meta: KnowledgePageMeta,
) -> FixProposal | None:
    """Soft-delete an orphan that is clearly a throwaway stub.

    The byte size of the body, the absence of outbound wikilinks, an
    explicit TODO/FIXME marker, and the presence (or absence) of
    metadata are each weak signals on their own. Combined they need to
    keep one-sentence definitions safe ("Water boils at 100C.") while
    still catching synth artifacts ("TODO: write this page",
    completely empty bodies, untagged unsourced placeholders).

    The combined gate: body must be short AND have no outbound
    wikilinks (necessary conditions), AND at least one of:
      1. body is empty after strip — definitive throwaway
      2. body matches a stub marker (TODO/FIXME/WIP/stub/placeholder/draft)
      3. orphan carries no ``sources`` and no ``tags`` — the synth or
         user never bothered to attribute or categorise it; a real
         page would have at least one
    """
    abs_path = (ctx.base_root / issue.path).resolve()
    if not abs_path.is_file():
        return None
    file_bytes = abs_path.read_bytes()
    try:
        post = frontmatter.loads(file_bytes.decode("utf-8"))
    except Exception:
        return None
    body = post.content.strip()
    if len(body.encode("utf-8")) >= _STUB_BODY_BYTES:
        return None
    has_outbound = any(
        link.kind is LinkType.WIKILINK for link in parse_links(body)
    )
    if has_outbound:
        return None

    is_empty = not body
    is_marker = bool(_STUB_PATTERN.search(body))
    no_metadata = not orphan_meta.sources and not orphan_meta.tags
    if not (is_empty or is_marker or no_metadata):
        return None

    if is_empty:
        signal = "empty body"
    elif is_marker:
        signal = "TODO/FIXME/WIP marker"
    else:
        signal = "no sources or tags attributed"

    op = FixOperation(
        kind="delete_page",
        path=issue.path,
        expected_hash=bytes_sha256(file_bytes),
    )
    return FixProposal(
        proposal_id=str(uuid.uuid4()),
        issue_kind=issue.kind,
        issue_path=issue.path,
        issue_detail=issue.detail,
        issue_line=issue.line,
        operations=[op],
        rationale=(
            f"delete stub — body is {len(body)} bytes ({signal}); "
            f"soft-delete to trash"
        ),
        source="heuristic",
    )


def _build_link_proposal(
    *,
    issue: Any,
    ctx: FixerContext,
    orphan_meta: KnowledgePageMeta,
    best: _ScoredCandidate,
) -> FixProposal | None:
    """Emit an ``update_page`` op that appends ``[[<orphan-title>]]`` to
    the best parent's body. Returns ``None`` if the file is missing,
    the orphan has no title (the backlink wouldn't resolve), or the
    backlink is already present.
    """
    parent_abs = (ctx.base_root / best.page.path).resolve()
    if not parent_abs.is_file():
        return None
    file_bytes = parent_abs.read_bytes()
    try:
        post = frontmatter.loads(file_bytes.decode("utf-8"))
    except Exception:
        return None
    if not orphan_meta.title:
        # No title means the wikilink we'd append wouldn't resolve;
        # safer to fall back to leaf than write a malformed link.
        return None
    backlink = f"[[{orphan_meta.title}]]"
    body = post.content
    # The orphan detection above runs from ``storage.links_from``, not
    # by re-parsing the parent body. When the body already contains the
    # backlink in ANY form (``[[Title]]``, ``[[Title|Alias]]``,
    # ``[[Title#Anchor]]``), the ``links`` table is out of sync
    # (typically: user hand-edited the body without re-ingest, or a
    # prior fix didn't re-trigger ``replace_links_from``). Emit a
    # no-content update so apply → ``persist_knowledge`` →
    # ``replace_links_from`` reconciles storage. A substring
    # ``"[[Title]]" in body`` check would miss aliased / anchored
    # variants and double-append the backlink.
    orphan_title_norm = normalize_base(orphan_meta.title)
    body_already_linked = any(
        link.kind is LinkType.WIKILINK
        and normalize_base(link.target) == orphan_title_norm
        for link in parse_links(body)
    )
    # Append under a dedicated heading so successive auto-fixes
    # accumulate links in one place, not scattered through prose. The
    # reconcile branch keeps the body byte-identical so persist only
    # has to refresh the storage links table.
    new_body = (
        body if body_already_linked
        else _append_related_section(body, backlink)
    )
    op = FixOperation(
        kind="update_page",
        path=best.page.path,
        new_frontmatter=dict(post.metadata),
        new_body=new_body,
        expected_hash=bytes_sha256(file_bytes),
    )
    rationale_prefix = (
        "reconcile_links — parent already contains backlink in body but "
        "links table is stale; re-persisting to refresh storage"
        if body_already_linked
        else "link_from_existing"
    )
    return FixProposal(
        proposal_id=str(uuid.uuid4()),
        issue_kind=issue.kind,
        issue_path=issue.path,
        issue_detail=issue.detail,
        issue_line=issue.line,
        operations=[op],
        rationale=(
            f"{rationale_prefix} — parent '{best.page.title}' "
            f"(score {best.score:.2f}: {best.reason})"
        ),
        source="heuristic",
    )


async def _propose_merge_into_existing(
    *,
    issue: Any,
    ctx: FixerContext,
    orphan_meta: KnowledgePageMeta,
    best: _ScoredCandidate,
) -> FixProposal | None:
    """LLM-driven 2-op proposal: rewrite the parent body absorbing the
    orphan, then soft-delete the orphan.

    Caller has already gated on ``ctx.enable_llm`` + ``ctx.llm`` +
    ``ctx.cfg`` + ``best.score >= MERGE_THRESHOLD``. Returns ``None``
    on any IO / parse / LLM failure so the router falls through to
    the link strategy — merge failure must never silently swallow the
    orphan.
    """
    llm = ctx.llm
    cfg = ctx.cfg
    if llm is None or cfg is None:
        return None

    parent_path = best.page.path
    parent_abs = (ctx.base_root / parent_path).resolve()
    orphan_abs = (ctx.base_root / orphan_meta.path).resolve()
    if not parent_abs.is_file() or not orphan_abs.is_file():
        return None

    parent_bytes = parent_abs.read_bytes()
    orphan_bytes = orphan_abs.read_bytes()
    try:
        parent_post = frontmatter.loads(parent_bytes.decode("utf-8"))
        orphan_post = frontmatter.loads(orphan_bytes.decode("utf-8"))
    except Exception:
        return None

    parent_meta = dict(parent_post.metadata)
    orphan_meta_fm = dict(orphan_post.metadata)
    # The merged page must overwrite the parent at its EXACT path. The engine
    # builds ``knowledge/<category>/<slug>.md`` from the category + slug the
    # LLM echoes, so we derive both from the parent's path (authoritative for
    # location) and tell the model to emit them verbatim.
    target_category = category_from_path(parent_path)
    target_slug = Path(parent_path).stem
    target_title = str(
        parent_meta.get("title") or best.page.title or ""
    ).strip()
    if not target_title:
        return None

    # The parent's own category is always accepted even if the operator has
    # since changed the taxonomy — otherwise the parser would coerce it to the
    # fallback and the merge would land at the wrong path (and be refused).
    base_categories = tuple(cfg.schema_.category_paths()) or DEFAULT_ALLOWED_CATEGORIES
    allowed_categories = (*base_categories, target_category)
    user_prompt = prompts.resolve(
        "lint_fix_orphan_merge",
        override_path=cfg.lint.fixer_prompts.get("orphan_merge"),
        base_root=ctx.base_root,
    ).format(
        target_path=parent_path,
        target_category=target_category,
        target_slug=target_slug,
        target_title=target_title,
        target_body=parent_post.content,
        orphan_path=orphan_meta.path,
        orphan_body=orphan_post.content,
        score_reason=best.reason,
    )
    # ``strict=True`` matches the non_atomic_page splitter: merge is
    # destructive (the orphan gets soft-deleted), so a deterministic
    # partial response — one valid <page> plus a malformed sibling —
    # must NOT count as success. Otherwise content from the malformed
    # block is silently dropped along with the original orphan.
    pages = await safe_synthesize_pages(
        user_prompt=user_prompt,
        source_path=parent_path,
        llm=llm,
        model=cfg.provider.llm_model,
        max_tokens=_MERGE_MAX_TOKENS,
        temperature=0.2,
        allowed_categories=allowed_categories,
        fallback=cfg.schema_.fallback,
        system=_MERGE_SYSTEM,
        log_label=f"orphan_page merge {orphan_meta.path}",
        strict=True,
    )
    if not pages or len(pages) != 1:
        # Multi-page output also violates the prompt contract — refuse
        # rather than guess which block represents the merged parent.
        return None
    merged = pages[0]
    if not merged.body.strip():
        return None
    # Prompt requires the LLM to emit `path={target_path}` and a body
    # starting with `# {target_title}` so wikilink resolution stays
    # stable. Validate both before we queue a destructive delete —
    # a wrong-target / wrong-title page block would otherwise be
    # rewritten onto the parent verbatim, masking the violation.
    if merged.path != parent_path:
        logger.info(
            "orphan_page merge for %s: LLM returned path %r, expected %r — "
            "refusing destructive merge",
            orphan_meta.path, merged.path, parent_path,
        )
        return None
    if (merged.title or "").strip() != target_title:
        logger.info(
            "orphan_page merge for %s: LLM returned title %r, expected %r — "
            "refusing destructive merge",
            orphan_meta.path, merged.title, target_title,
        )
        return None
    # ``parse_synthesis_response`` finds the title via ATX heading
    # anywhere in the body, so a body like ``Merged below.\n\n# Main
    # Topic ...`` would pass the title check above. The prompt
    # explicitly requires the title heading at the start. Enforce
    # that here so the merged page's first non-whitespace content is
    # the heading the parent's wikilinks resolve against.
    first_heading_line = merged.body.lstrip().splitlines()[0] if merged.body.strip() else ""
    if first_heading_line.strip() != f"# {target_title}":
        logger.info(
            "orphan_page merge for %s: LLM body does not start with "
            "`# %s` heading — refusing destructive merge",
            orphan_meta.path, target_title,
        )
        return None

    # LLM only emits the body — frontmatter is owned by the fixer so
    # the parent's identity (id, category, title, created) survives intact
    # and sources/tags merge deterministically.
    merged_fm = dict(parent_meta)
    for key in ("sources", "tags"):
        unioned = _union_preserving_order(
            parent_meta.get(key), orphan_meta_fm.get(key),
        )
        if unioned:
            merged_fm[key] = unioned
        else:
            merged_fm.pop(key, None)

    update_op = FixOperation(
        kind="update_page",
        path=parent_path,
        new_frontmatter=merged_fm,
        new_body=merged.body,
        expected_hash=bytes_sha256(parent_bytes),
    )
    delete_op = FixOperation(
        kind="delete_page",
        path=orphan_meta.path,
        expected_hash=bytes_sha256(orphan_bytes),
    )
    return FixProposal(
        proposal_id=str(uuid.uuid4()),
        issue_kind=issue.kind,
        issue_path=issue.path,
        issue_detail=issue.detail,
        issue_line=issue.line,
        operations=[update_op, delete_op],
        rationale=(
            f"merge_into_existing — into '{best.page.title}' "
            f"(score {best.score:.2f}: {best.reason})"
        ),
        source="llm",
    )


def _union_preserving_order(*lists: Any) -> list[str]:
    """De-duplicating union that preserves first-seen order.

    Frontmatter ``sources`` / ``tags`` are user-facing — keeping the
    parent's order makes the merged page diff-friendly. Non-list /
    non-string entries are silently dropped.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in lists:
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, str):
                continue
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
    return out


#: Heading the fixer appends backlinks under. Stable so multiple
#: auto-fixes pile onto the same list rather than each adding their
#: own heading.
_RELATED_HEADING = "## 相关"


def _append_related_section(body: str, backlink: str) -> str:
    """Append ``backlink`` (a ``[[Title]]`` literal) to a ``## 相关``
    section at the end of ``body``. Creates the section if missing.

    The heading is stable so repeated runs accumulate links in one
    bulleted list rather than each fixer pass adding its own heading.
    """
    rstripped = body.rstrip()
    if _RELATED_HEADING in rstripped:
        # Heading already there — append a new bullet at the end.
        return rstripped + f"\n- {backlink}\n"
    sep = "\n\n" if rstripped else ""
    return f"{rstripped}{sep}{_RELATED_HEADING}\n\n- {backlink}\n"


def _propose_mark_as_leaf(
    *,
    issue: Any,
    ctx: FixerContext,
    reason: str,
) -> FixProposal | None:
    """Build an ``update_page`` proposal that injects ``lint.skip =
    [orphan_page]`` into the orphan page's frontmatter.

    Reads the on-disk file once: hash + parsed frontmatter come from
    the same byte payload, matching ``BrokenWikilinkFixer`` so the
    concurrent-edit guard at apply time uses the bytes the fixer
    actually observed.
    """
    abs_path = (ctx.base_root / issue.path).resolve()
    if not abs_path.is_file():
        return None
    file_bytes = abs_path.read_bytes()
    try:
        post = frontmatter.loads(file_bytes.decode("utf-8"))
    except Exception:
        # Corrupted frontmatter; skip rather than rewrite a malformed
        # YAML block. The user can hand-edit and re-run propose.
        return None

    fm: dict[str, Any] = dict(post.metadata)
    lint_block = fm.get("lint")
    if not isinstance(lint_block, dict):
        lint_block = {}
    skip_list = lint_block.get("skip")
    if not isinstance(skip_list, list):
        skip_list = []
    # Idempotent extend — don't duplicate ``orphan_page`` if a prior
    # propose run already wrote it.
    new_skip = [k for k in skip_list if isinstance(k, str)]
    if "orphan_page" not in new_skip:
        new_skip.append("orphan_page")
    # Preserve any prior reason / metadata so a user that previously
    # set their own justification doesn't lose it; only fill reason in
    # if the user hadn't already set one.
    new_lint = dict(lint_block)
    new_lint["skip"] = new_skip
    if not isinstance(new_lint.get("reason"), str) or not new_lint["reason"]:
        new_lint["reason"] = reason
    fm["lint"] = new_lint

    op = FixOperation(
        kind="update_page",
        path=issue.path,
        new_frontmatter=fm,
        new_body=post.content,
        expected_hash=bytes_sha256(file_bytes),
    )
    return FixProposal(
        proposal_id=str(uuid.uuid4()),
        issue_kind=issue.kind,
        issue_path=issue.path,
        issue_detail=issue.detail,
        issue_line=issue.line,
        operations=[op],
        rationale=f"mark as leaf — {reason}",
        source="heuristic",
    )
