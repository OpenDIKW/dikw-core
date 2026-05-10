"""Fixer for ``broken_wikilink`` issues.

Two-stage strategy:

1. **Heuristic** (always on). Normalize every K-layer page title
   (lower + strip puncs + collapse whitespace) and the broken target
   the lint scanner recovered, fuzzy-match via
   :class:`difflib.SequenceMatcher`, and propose an in-place
   ``[[link]]`` rewrite when the ratio crosses
   :data:`HEURISTIC_RATIO_THRESHOLD`.

2. **LLM stub fallback** (PR2, opt-in via
   :attr:`FixerContext.enable_llm`). When the heuristic misses, ask
   the configured LLM for a stub page so the link resolves on the
   next lint pass. The body is intentionally a TODO marker — the
   point is to unbreak the wikilink graph, not to invent content.
   Default off because each propose iteration would otherwise pay an
   LLM round trip; users opt in via
   ``dikw client lint propose --enable-llm``.

A failure in either path (heuristic miss + LLM disabled / failed /
returned no usable page) returns ``None`` so the orchestrator
records ``"skipped"`` and moves on — single-issue failures must not
fail the whole propose task.
"""

from __future__ import annotations

import logging
import re
import uuid
from difflib import SequenceMatcher
from typing import Any

import frontmatter

from .... import prompts
from ..lint import LintKind
from ..lint_fix import (
    FixerContext,
    FixOperation,
    FixProposal,
    bytes_sha256,
    extract_broken_target,
    page_to_op_frontmatter,
    safe_synthesize_pages,
)
from ..synthesize import DEFAULT_ALLOWED_TYPES

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
        if not target or len(_normalize_title(target)) < _MIN_TARGET_LEN:
            return None

        # Rank candidate titles by similarity, excluding the source page itself
        # so a self-referential broken link doesn't collapse onto its own title.
        target_norm = _normalize_title(target)
        best_title: str | None = None
        best_ratio = 0.0
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
            return await _propose_llm_stub(issue, ctx, target)

        abs_path = (ctx.wiki_root / issue.path).resolve()
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


# --- LLM stub fallback -------------------------------------------------------


_STUB_SYSTEM = (
    "You generate K-layer wiki stub pages for dikw-core. Your output "
    "must be exactly one <page> block, with a TODO marker in the body. "
    "Never invent biographical or factual claims."
)
# Bound the LLM context window: a few lines around the broken link
# are enough to disambiguate the target. The body is a TODO placeholder
# so larger excerpts add tokens for no improvement in the stub.
_CONTEXT_WINDOW_LINES = 6
_STUB_MAX_TOKENS = 1024


async def _propose_llm_stub(
    issue: Any, ctx: FixerContext, target: str
) -> FixProposal | None:
    if not ctx.enable_llm or ctx.llm is None:
        return None
    if ctx.cfg is None:
        # Production always sets cfg via api.lint_propose; tests that
        # exercise this path build a real DikwConfig() (see
        # ``_default_cfg`` in tests/test_lint_fixers.py).
        logger.warning(
            "broken_wikilink LLM stub for %s skipped: ctx.cfg is None",
            issue.path,
        )
        return None

    # Obsidian wikilink syntax allows ``[[Target|alias]]`` and
    # ``[[Target#anchor]]``; the resolver matches the bare "Target"
    # title, so the stub page needs the same canonical name. Without
    # stripping, the LLM would title the page "Target|alias" and the
    # next lint pass would still flag the wikilink as broken.
    canonical_target = _strip_alias_anchor(target)
    if not canonical_target:
        return None

    src_abs = (ctx.wiki_root / issue.path).resolve()
    if not src_abs.is_file():
        return None
    body = frontmatter.loads(src_abs.read_text(encoding="utf-8")).content
    excerpt = _excerpt_around_line(body, issue.line, window=_CONTEXT_WINDOW_LINES)

    allowed_types = tuple(ctx.cfg.schema_.page_types) or DEFAULT_ALLOWED_TYPES
    user_prompt = prompts.load("lint_fix_broken_wikilink_stub").format(
        broken_target=canonical_target,
        source_path=issue.path,
        source_context=excerpt,
        allowed_types=" | ".join(allowed_types),
    )
    pages = await safe_synthesize_pages(
        user_prompt=user_prompt,
        source_path=issue.path,
        llm=ctx.llm,
        model=ctx.cfg.provider.llm_model,
        max_tokens=_STUB_MAX_TOKENS,
        temperature=0.2,
        allowed_types=allowed_types,
        system=_STUB_SYSTEM,
        log_label=f"broken_wikilink stub [[{target}]]",
    )
    if not pages:
        return None
    page = pages[0]

    # The LLM may pick a slug that already names a different concept.
    # Apply would refuse the create anyway, but skipping here keeps a
    # doomed proposal out of the user's review pile.
    if any(p.path == page.path for p in ctx.all_pages):
        logger.info(
            "broken_wikilink LLM stub picked existing path %s — skipping",
            page.path,
        )
        return None

    op = FixOperation(
        kind="create_page",
        path=page.path,
        new_frontmatter=page_to_op_frontmatter(page),
        new_body=page.body,
        expected_hash=None,
    )
    return FixProposal(
        proposal_id=str(uuid.uuid4()),
        issue_kind=issue.kind,
        issue_path=issue.path,
        issue_detail=issue.detail,
        issue_line=issue.line,
        operations=[op],
        rationale=f"LLM-generated stub for missing target '[[{canonical_target}]]'",
        source="llm",
    )


def _strip_alias_anchor(target: str) -> str:
    """Drop Obsidian-style ``|alias`` and ``#anchor`` suffixes from a wikilink target.

    The resolver matches the bare title — both ``[[Target|label]]`` and
    ``[[Target#section]]`` resolve against a page titled ``Target`` —
    so the stub the LLM authors must use that bare name. Without this,
    the LLM would title the stub ``Target|label`` and the next lint
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
