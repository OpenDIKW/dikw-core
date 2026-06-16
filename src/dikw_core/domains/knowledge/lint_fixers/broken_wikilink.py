"""Fixer for ``broken_wikilink`` issues.

Two-stage strategy:

1. **Heuristic** (always on). Normalize every K-layer page title
   (lower + strip puncs + collapse whitespace) and the broken target
   the lint scanner recovered, fuzzy-match via
   :class:`difflib.SequenceMatcher`, and propose an in-place
   ``[[link]]`` rewrite when the ratio crosses
   :data:`HEURISTIC_RATIO_THRESHOLD`.

2. **Evidence-backed LLM repair** (#83, opt-in via
   :attr:`FixerContext.enable_llm`). When the heuristic misses, search
   D/I-layer chunks for the broken target; if there is enough source
   evidence to ground a real page, ask the LLM to write one. If
   evidence is insufficient, or if the LLM returns a TODO-laced /
   too-short body, raise :class:`FixerSkip` with a structured reason
   so the propose-task result tells an agent *why* the wikilink stays
   unrepaired (was previously a generic ``"fixer returned None"``).

   This replaces the PR2 TODO-stub fallback: ``--enable-llm`` now means
   "let the LLM perform an evidence-backed repair when possible", not
   "let the LLM fabricate placeholder pages that hide broken_wikilink
   from the next lint pass".

Provider outages / unparseable LLM output remain soft failures
(``return None``) — they are operational noise, not product semantics,
and would otherwise flood ``FixProposalReport.skipped`` with reasons
the user can't act on.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import uuid
from difflib import SequenceMatcher
from typing import Any, NoReturn

import frontmatter

from .... import prompts
from ....providers import build_embedder
from ....schemas import Hit, Layer
from ....storage.base import NotSupported
from ...info.search import HybridSearcher
from ..links import _normalize_base, _normalize_for_match
from ..lint import LintKind
from ..lint_fix import (
    FixerContext,
    FixerSkip,
    FixOperation,
    FixProposal,
    bytes_sha256,
    extract_broken_target,
    page_to_op_frontmatter,
    safe_synthesize_pages,
)
from ..synthesize import DEFAULT_ALLOWED_CATEGORIES

logger = logging.getLogger(__name__)

#: A page rewrites the broken link only when the closest existing title
#: clears this similarity ratio. Calibrated empirically: 0.85 catches
#: case / whitespace / punctuation drift while rejecting mid-word
#: substrings (a 3-character broken target hits 0.85 against many
#: titles by chance, so we also gate on ``len(target) >= 4``).
HEURISTIC_RATIO_THRESHOLD = 0.85
_MIN_TARGET_LEN = 4

# Unicode-aware normalize: ``\w`` (with ``re.UNICODE``, the py3 default)
# covers Latin + CJK + Cyrillic + Greek + everything else. The earlier
# ``[a-z0-9]`` filter stripped every CJK glyph, making the fuzzy match
# silently no-op for multilingual bases — the bug codex caught in
# round 2. Lowercasing is still useful for Western titles and a no-op
# for scripts without case (Han, Arabic).
_NON_WORD = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def _normalize_title(s: str) -> str:
    lowered = s.lower()
    no_puncts = _NON_WORD.sub(" ", lowered)
    return _WS.sub(" ", no_puncts).strip()


class BrokenWikilinkFixer:
    kind: LintKind = "broken_wikilink"

    async def propose(
        self,
        issue: Any,
        ctx: FixerContext,
        reporter: Any,
    ) -> FixProposal | None:
        target = extract_broken_target(issue.detail)
        if not target:
            return None
        target_norm = _normalize_title(target)

        # Heuristic fuzzy match only runs above the length gate — short
        # targets (e.g. 2-3 char CJK like ``[[秦朝]]``) yield too many
        # spurious 0.85 ratios. Below the gate we skip heuristic and
        # fall straight through to the evidence-backed grounded repair,
        # which is still gated by ``enable_llm`` so heuristic-only users
        # pay no LLM cost.
        best_title: str | None = None
        best_ratio = 0.0
        if len(target_norm) >= _MIN_TARGET_LEN:
            # Rank candidate titles by similarity, excluding the source
            # page itself so a self-referential broken link doesn't
            # collapse onto its own title.
            for page in ctx.all_pages:
                if page.path == issue.path:
                    continue
                title = page.title
                if not title:
                    continue
                ratio = SequenceMatcher(
                    None, target_norm, _normalize_title(title)
                ).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_title = title

        if best_title is None or best_ratio < HEURISTIC_RATIO_THRESHOLD:
            return await _propose_llm_grounded(issue, ctx, target, reporter)

        abs_path = (ctx.base_root / issue.path).resolve()
        if not abs_path.is_file():
            # Source file vanished between lint scan and fix proposal —
            # skip rather than synthesising an op against missing state.
            return None

        # Read once: hash from bytes, parse the frontmatter from the same
        # payload. Avoids the second disk roundtrip ``frontmatter.load
        # + file_sha256`` would do.
        file_bytes = abs_path.read_bytes()
        post = frontmatter.loads(file_bytes.decode("utf-8"))
        body = post.content
        old = f"[[{target}]]"
        new = f"[[{best_title}]]"
        if old not in body:
            # The on-disk body uses an alias / anchor / casing the
            # detail string can't reproduce — a regex-aware replacement
            # covering ``[[t|alias]]`` is a follow-up.
            return None
        new_body = body.replace(old, new)

        op = FixOperation(
            kind="update_page",
            path=issue.path,
            new_frontmatter=dict(post.metadata),
            new_body=new_body,
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
                f"fuzzy match — '{target}' → '{best_title}' "
                f"(ratio {best_ratio:.2f})"
            ),
            source="heuristic",
        )


# --- Evidence-backed LLM repair (#83) ----------------------------------------


_GROUNDED_SYSTEM = (
    "You write K-layer knowledge pages for `dikw-core`, grounded strictly in "
    "the evidence the user supplies. Emit exactly one <page> block. "
    "Every claim in the body must be traceable to the evidence chunks; "
    "if the evidence cannot support at least one well-grounded paragraph, "
    "output a single line `REFUSE: insufficient evidence` instead of a "
    "<page> block. Never invent biographical or factual claims."
)

# Bound the LLM context window for the *source page* excerpt: a few lines
# around the broken link disambiguate the target. The actual evidence
# comes from D/I-layer chunks, not from the source page body.
_CONTEXT_WINDOW_LINES = 6

# Token budget for the grounded body: 2048 leaves room for ~1500 chars
# of prose + frontmatter + the <page> wrapper.
_GROUNDED_MAX_TOKENS = 2048

# Evidence thresholds — module-level constants matching
# ``HEURISTIC_RATIO_THRESHOLD``'s style. Hardcoded for #83's first pass;
# promote to ``dikw.yml`` only if real-world tuning needs it.
_MIN_EVIDENCE_CHUNKS = 1
_MIN_EVIDENCE_CHARS = 200

# Cap on the number of evidence chunks injected into the prompt. Above
# this the LLM gains nothing but loses focus.
_EVIDENCE_TOP_K = 8

# Post-generation body checks. The TODO check defends against the
# specific regression #83 fixes; the length floor defends against
# "Topic A is a topic." filler that would clear the wikilink without
# adding knowledge.
_MIN_BODY_CHARS = 200
_FORBIDDEN_BODY_TOKENS: tuple[str, ...] = (
    "todo",
    "stub page",
    "placeholder",
)


async def _propose_llm_grounded(
    issue: Any, ctx: FixerContext, target: str, reporter: Any
) -> FixProposal | None:
    if not ctx.enable_llm or ctx.llm is None:
        return None
    if ctx.cfg is None:
        # Production always sets cfg via api.lint_propose; tests that
        # exercise this path build a real DikwConfig() (see
        # ``_default_cfg`` in tests/test_lint_fixers.py).
        logger.warning(
            "broken_wikilink grounded repair for %s skipped: ctx.cfg is None",
            issue.path,
        )
        return None

    # Obsidian wikilink syntax allows ``[[Target|alias]]`` and
    # ``[[Target#anchor]]``; the resolver matches the bare "Target"
    # title, so the grounded page needs the same canonical name. Without
    # stripping, the LLM would title the page "Target|alias" and the
    # next lint pass would still flag the wikilink as broken.
    canonical_target = _strip_alias_anchor(target)
    if not canonical_target:
        return None

    src_abs = (ctx.base_root / issue.path).resolve()
    if not src_abs.is_file():
        return None
    body = frontmatter.loads(src_abs.read_text(encoding="utf-8")).content
    excerpt = _excerpt_around_line(body, issue.line, window=_CONTEXT_WINDOW_LINES)

    hits = await _collect_evidence(ctx, canonical_target, excerpt, issue.path)
    if (reason := _evidence_is_sufficient(hits)) is not None:
        await _skip(reporter, issue.path, reason)

    allowed_categories = tuple(ctx.cfg.schema_.category_paths()) or DEFAULT_ALLOWED_CATEGORIES
    evidence_block = _render_evidence_block(hits)
    user_prompt = prompts.resolve(
        "lint_fix_broken_wikilink_grounded",
        override_path=ctx.cfg.lint.fixer_prompts.get("broken_wikilink"),
        base_root=ctx.base_root,
    ).format(
        broken_target=canonical_target,
        source_path=issue.path,
        source_context=excerpt,
        evidence_block=evidence_block,
        categories=ctx.cfg.schema_.categories_prompt_block(),
    )
    pages = await safe_synthesize_pages(
        user_prompt=user_prompt,
        source_path=issue.path,
        llm=ctx.llm,
        model=ctx.cfg.provider.llm_model,
        max_tokens=_GROUNDED_MAX_TOKENS,
        temperature=0.2,
        allowed_categories=allowed_categories,
        fallback=ctx.cfg.schema_.fallback,
        system=_GROUNDED_SYSTEM,
        log_label=f"broken_wikilink grounded [[{target}]]",
    )
    if not pages:
        # Soft failure (provider outage, REFUSE line, no parseable
        # <page>). Stays as ``return None`` — orchestrator records a
        # generic skip; the reporter already logged the specific cause.
        return None
    page = pages[0]

    if (reason := _body_passes_grounding_checks(page.body)) is not None:
        await _skip(reporter, issue.path, f"{reason}: {canonical_target!r}")

    # The LLM sometimes titles its page after a *related* concept the
    # evidence mentions instead of the canonical target. Such a page
    # would NOT resolve the original `[[canonical_target]]` reference —
    # applying it just adds an unrelated K-page while the wikilink
    # stays broken.
    #
    # Mirror the resolver's ASYMMETRIC normalize (see
    # ``resolve_links`` + ``build_fuzzy_index`` in ``links.py``): stored
    # titles index via ``_normalize_base`` (no plural stem) but lookup
    # via ``_normalize_for_match`` (trailing-plural stem on last token).
    # A page titled ``Networks`` indexes as ``networks``; a wikilink
    # ``[[Network]]`` looks up as ``network`` and would NOT find that
    # page even though a symmetric ``_normalize_for_match`` on both
    # sides would falsely accept the match. Replicate the exact rule
    # the resolver applies so we never propose a "fix" that leaves the
    # link broken.
    if (
        page.title != canonical_target
        and _normalize_base(page.title) != _normalize_for_match(canonical_target)
    ):
        await _skip(
            reporter,
            issue.path,
            (
                f"rejected_title_mismatch: page titled {page.title!r} "
                f"would not resolve {canonical_target!r}"
            ),
        )

    # The LLM may pick a slug that already names a different concept.
    # Apply would refuse the create anyway, but raising here keeps a
    # doomed proposal out of the user's review pile and tells the agent
    # exactly why this fix didn't land.
    if any(p.path == page.path for p in ctx.all_pages):
        await _skip(
            reporter,
            issue.path,
            f"path_collision: {page.path!r} already exists in wiki",
        )

    # ``parse_synthesis_response`` unconditionally stamps
    # ``sources=[source_path]`` (the K-page that referenced the broken
    # wikilink) onto every parsed page. For a grounded repair that is wrong —
    # the page was built from D-layer evidence, not the referrer — so we
    # replace ``sources`` with the distinct hit paths. The parser drops every
    # non-``tags`` front-matter key the LLM emits, so ``page.extras`` is empty
    # and can't smuggle a conflicting ``sources:`` into ``page_to_op_frontmatter``.
    evidence_sources = _evidence_source_paths(hits)
    if evidence_sources:
        page = dataclasses.replace(page, sources=evidence_sources)

    op = FixOperation(
        kind="create_page",
        path=page.path,
        new_frontmatter=page_to_op_frontmatter(page),
        new_body=page.body,
        expected_hash=None,
    )
    evidence_chars = sum(len(h.text or "") for h in hits)
    return FixProposal(
        proposal_id=str(uuid.uuid4()),
        issue_kind=issue.kind,
        issue_path=issue.path,
        issue_detail=issue.detail,
        issue_line=issue.line,
        operations=[op],
        rationale=(
            f"LLM-grounded ({len(hits)} evidence chunks, {evidence_chars} chars) "
            f"for missing target '[[{canonical_target}]]'"
        ),
        source="llm",
    )


async def _collect_evidence(
    ctx: FixerContext, canonical_target: str, source_excerpt: str, issue_path: str
) -> list[Hit]:
    """Hybrid-search the D-layer for chunks relevant to the broken target.

    Returns at most :data:`_EVIDENCE_TOP_K` hits, excluding any hit that
    points back at the source K-page itself (a wikilink in page X is
    not evidence for itself). Returns an empty list when storage /
    embedding plumbing isn't wired (defensive — production always
    threads them through, but unit tests that don't monkeypatch this
    helper would otherwise crash).
    """
    if ctx.storage is None or ctx.cfg is None:
        return []
    query = f"{canonical_target}\n{source_excerpt}".strip()
    if not query:
        return []

    # Pin the text leg to the **active** embed version (mirrors the
    # anti-drift guard in ``_retrieve_inner``). If ``dikw.yml``'s
    # embedding_model / embedding_dim was edited after ingest, the
    # vectors in ``vec_chunks_v<active>`` still belong to the old
    # version — embedding our query with the new model/dim would
    # either error or rank against an incompatible space. Resolve the
    # stored version, override model + dim before building the
    # embedder, and tell ``HybridSearcher`` exactly which version_id
    # to read.
    text_version_id: int | None = None
    text_query_model = ctx.cfg.provider.embedding_model
    text_query_dim: int | None = None
    try:
        active_text = await ctx.storage.get_active_embed_version(modality="text")
    except NotSupported:
        active_text = None
    if active_text is not None and active_text.version_id is not None:
        text_version_id = active_text.version_id
        text_query_model = active_text.model
        text_query_dim = active_text.dim

    # When a stored version exists, rebuild the embedder with its
    # dim_override (matching ``_retrieve_inner``'s anti-drift pattern)
    # — even if ``ctx.embedding`` was already built upstream, that
    # build used ``cfg.provider.embedding_dim`` which may have drifted
    # away from the stored version's dim after a config edit. Trust
    # the stored version. Without a stored version (fresh base, no
    # text vectors ingested) fall back to ``ctx.embedding`` so BM25
    # still works.
    embedder = ctx.embedding
    if text_version_id is not None:
        try:
            embedder = build_embedder(
                ctx.cfg.provider, dim_override=text_query_dim
            )
        except Exception as e:
            logger.warning(
                "broken_wikilink: failed to build dim-aligned embedder "
                "for evidence search (%s); falling back to BM25-only", e,
            )
            embedder = None

    searcher = HybridSearcher.from_config(
        ctx.storage,
        embedder,
        ctx.cfg.retrieval,
        embedding_model=text_query_model,
        text_version_id=text_version_id,
    )
    # Try hybrid first; on failure (no embedding credentials, transient
    # provider outage, dim mismatch) retry with BM25-only so a base
    # with indexed text still produces lexical evidence. Losing
    # semantic recall is acceptable; losing every BM25-recoverable
    # repair is not — and a raw raise here would surface as
    # ``"fixer raised: ..."`` on the propose report, which agents
    # can't distinguish from a real bug.
    try:
        hits = await searcher.search(
            query, limit=_EVIDENCE_TOP_K, layer=Layer.SOURCE, mode="hybrid",
        )
    except Exception as e:
        logger.warning(
            "broken_wikilink: hybrid evidence search failed (%s); "
            "falling back to BM25-only", e,
        )
        hits = await searcher.search(
            query, limit=_EVIDENCE_TOP_K, layer=Layer.SOURCE, mode="bm25",
        )
    # Filter self-references (rare with layer=SOURCE, but cheap insurance).
    return [h for h in hits if h.path != issue_path]


async def _skip(reporter: Any, issue_path: str, reason: str) -> NoReturn:
    """Log the skip on the live stream + raise :class:`FixerSkip`.

    Centralises the "tell the user why this fix didn't land" contract
    so every gating check in :func:`_propose_llm_grounded` writes the
    same shape — one INFO log line keyed by ``issue.path``, one
    structured ``FixProposalReport.skipped[].reason``.
    """
    await reporter.log("INFO", f"{issue_path}: {reason}")
    raise FixerSkip(reason)


def _evidence_is_sufficient(hits: list[Hit]) -> str | None:
    """Gate the LLM call on real source evidence.

    Returns ``None`` when evidence clears both thresholds, or a reason
    string the caller passes to :func:`_skip`.
    """
    n_chunks = len(hits)
    total_chars = sum(len(h.text or "") for h in hits)
    if n_chunks < _MIN_EVIDENCE_CHUNKS or total_chars < _MIN_EVIDENCE_CHARS:
        return (
            f"evidence_insufficient: {n_chunks} chunks, {total_chars} chars "
            f"(need ≥{_MIN_EVIDENCE_CHUNKS} chunks and ≥{_MIN_EVIDENCE_CHARS} chars)"
        )
    return None


def _body_passes_grounding_checks(body: str) -> str | None:
    """Reject LLM bodies that would defeat the point of #83.

    Returns ``None`` when the body passes, or a reason string:

    * ``rejected_todo_marker`` — body still carries ``TODO`` / ``stub
      page`` / ``placeholder``; the wikilink would resolve but the page
      adds no knowledge.
    * ``rejected_body_too_short`` — body cleared the token check but is
      shorter than :data:`_MIN_BODY_CHARS`; same outcome dressed up
      differently ("Topic A is a topic.").
    """
    stripped = body.strip()
    lowered = stripped.lower()
    for token in _FORBIDDEN_BODY_TOKENS:
        if token in lowered:
            return f"rejected_todo_marker: body contains {token!r}"
    if len(stripped) < _MIN_BODY_CHARS:
        return (
            f"rejected_body_too_short: {len(stripped)} chars "
            f"(need ≥{_MIN_BODY_CHARS})"
        )
    return None


def _evidence_source_paths(hits: list[Hit]) -> list[str]:
    """Distinct source paths citing in stable hit-rank order.

    Used to stamp ``page.sources`` for grounded repairs so generated
    pages point at the D-layer evidence that justified them, not at
    the K-layer referrer ``parse_synthesis_response`` would otherwise
    write. Hits that lack a ``path`` (rare — searcher resolves it via
    ``get_documents``) are skipped silently.
    """
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h.path and h.path not in seen:
            seen.add(h.path)
            out.append(h.path)
    return out


def _render_evidence_block(hits: list[Hit]) -> str:
    """Format hits as the ``{evidence_block}`` substring the prompt expects.

    One markdown subsection per hit, source path in the header so the
    LLM can cite, full chunk text in the body. Truncation is up to the
    caller — we render every hit passed in.
    """
    if not hits:
        return "(no evidence)"
    parts: list[str] = []
    for idx, h in enumerate(hits, start=1):
        src = h.path or h.doc_id
        body = (h.text or "").strip()
        parts.append(f"### Evidence {idx} — `{src}`\n\n{body}\n")
    return "\n".join(parts)


def _strip_alias_anchor(target: str) -> str:
    """Drop Obsidian-style ``|alias`` and ``#anchor`` suffixes from a wikilink target.

    The resolver matches the bare title — both ``[[Target|label]]`` and
    ``[[Target#section]]`` resolve against a page titled ``Target`` —
    so the page the LLM authors must use that bare name. Without this,
    the LLM would title the page ``Target|label`` and the next lint
    pass would still report the wikilink as broken.
    """
    base = target.split("|", 1)[0]
    base = base.split("#", 1)[0]
    return base.strip()


def _excerpt_around_line(
    body: str, line: int | None, *, window: int
) -> str:
    """Return ±``window`` lines of ``body`` centred on ``line``.

    ``line`` is 1-indexed (matching ``LintIssue.line``). When the line
    number is missing or out of range, fall back to the body's first
    ``2 * window + 1`` lines so the LLM still has *some* context.
    """
    lines = body.splitlines()
    if not lines:
        return ""
    if line is None or line < 1 or line > len(lines):
        return "\n".join(lines[: 2 * window + 1])
    start = max(0, line - 1 - window)
    end = min(len(lines), line - 1 + window + 1)
    return "\n".join(lines[start:end])
