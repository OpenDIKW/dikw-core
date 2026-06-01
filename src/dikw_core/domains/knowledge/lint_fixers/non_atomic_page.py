"""Fixer for ``non_atomic_page`` issues.

Stage A's atomicity rule says one knowledge page = one self-contained idea
or entity. The lint scanner flags pages whose body, H1/H2 count, or
distinct-wikilink count signals "this is N pages glued together"; this
fixer asks the synth LLM to split the body into atomic children.

The fixer is **LLM-only** — there is no useful pure-heuristic split.
It short-circuits to ``None`` when ``ctx.enable_llm`` is False,
``ctx.llm`` / ``ctx.cfg`` is missing, the body is too large to feed
safely (see :data:`MAX_SPLIT_BODY_BYTES`), or the call yields fewer
than two children.

External wikilinks pointing at the original page are intentionally
**not** rewritten. Once the original is deleted, sibling pages with
``[[Original Title]]`` links surface as ``broken_wikilink`` issues
under the next lint pass, where the broken_wikilink fixer attempts
to repair them — fuzzy-match first, then the evidence-backed grounded
repair when ``--enable-llm`` is on (which may ``FixerSkip`` if
evidence is insufficient or the LLM body fails the grounding checks).
Keeping this fixer's scope narrow avoids the embedding-similarity
machinery needed to pick the "right" child for each external referrer.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import frontmatter

from .... import prompts
from ..lint import LintKind
from ..lint_fix import (
    FixerContext,
    FixOperation,
    FixProposal,
    bytes_sha256,
    page_to_op_frontmatter,
    safe_synthesize_pages,
)
from ..synthesize import (
    DEFAULT_ALLOWED_CATEGORIES,
    DEFAULT_SYNTH_SYSTEM,
    dedup_pages_by_slug,
)

logger = logging.getLogger(__name__)

# A "split" only makes sense when the LLM finds at least two atomic
# children — one means "page is fine as-is", and we'd rather skip
# than churn its slug. Higher floors silently mask real splits.
_MIN_CHILDREN = 2

# Upper bound on children the LLM may emit per split. ``cfg.synth.max_pages_per_group``
# is calibrated for chunk-group fan-out during ingestion (default 4); reusing it
# here would silently cap fat pages with 5+ atomic topics, and the LLM stops at
# the cap WITHOUT raising any truncation signal — there's no way to tell whether
# topics 5/6/7 just didn't exist or got dropped on the floor. Use a much higher
# ceiling so the cap is informational rather than load-bearing for correctness;
# the safety net below ("emitted == cap → refuse") then guards the rare page
# that genuinely fans out beyond this limit.
_MAX_CHILDREN_CEILING = 16

# Hard cap on the body bytes we will hand to the LLM. ``non_atomic_page``
# specifically targets fat pages, so the upper tail of inputs IS the
# common case here, and at least one configured provider (openai_codex)
# has a known SSE hang on very large prompts. Large bodies are silently
# skipped and surface as a normal "skipped" entry — the user re-runs
# after splitting by hand. 32 KB matches a typical synth-group budget.
MAX_SPLIT_BODY_BYTES = 32 * 1024


class NonAtomicPageFixer:
    kind: LintKind = "non_atomic_page"

    async def propose(
        self,
        issue: Any,
        ctx: FixerContext,
        reporter: Any,
    ) -> FixProposal | None:
        if not ctx.enable_llm or ctx.llm is None or ctx.cfg is None:
            return None

        abs_path = (ctx.base_root / issue.path).resolve()
        if not abs_path.is_file():
            return None
        file_bytes = abs_path.read_bytes()
        post = frontmatter.loads(file_bytes.decode("utf-8"))
        body = post.content
        if not body.strip():
            return None
        if len(body.encode("utf-8")) > MAX_SPLIT_BODY_BYTES:
            logger.info(
                "non_atomic_page LLM split skipped for %s: body %d bytes "
                "> cap %d (split by hand and re-run)",
                issue.path,
                len(body.encode("utf-8")),
                MAX_SPLIT_BODY_BYTES,
            )
            return None

        cfg = ctx.cfg
        allowed_categories = tuple(cfg.schema_.category_paths()) or DEFAULT_ALLOWED_CATEGORIES

        user_prompt = prompts.resolve(
            "synthesize", override_path=cfg.synth.prompt_path, base_root=ctx.base_root
        ).format(
            source_path=issue.path,
            source_body=body,
            group_outline="(whole page — split into atomic children)",
            group_index=1,
            group_total=1,
            max_pages=_MAX_CHILDREN_CEILING,
            categories=cfg.schema_.categories_prompt_block(),
            # Mirrors api.py:_NO_EXISTING_PAGES_SENTINEL. The split fixer
            # operates on a single page in isolation; the existing-pages
            # awareness contract from PR #69 is for ingestion fan-out where
            # earlier groups in the same source need to be visible to later
            # groups. A future enhancement could surface ctx.all_pages here
            # to discourage children that duplicate existing K-pages, but
            # the immediate post-collision filter already catches that.
            existing_pages_section="(no existing pages — this is a fresh knowledge base)",
        )

        pages = await safe_synthesize_pages(
            user_prompt=user_prompt,
            source_path=issue.path,
            llm=ctx.llm,
            model=cfg.provider.llm_model,
            max_tokens=cfg.provider.llm_max_tokens_synth,
            allowed_categories=allowed_categories,
            fallback=cfg.schema_.fallback,
            system=DEFAULT_SYNTH_SYSTEM,
            log_label="non_atomic_page split",
            strict=True,
        )
        if not pages:
            return None

        # Refuse if the LLM hit the ceiling — the page may have had more
        # atomic topics that the prompt cap silently swallowed (the model
        # voluntarily stops at "emit at most N", no SynthesisPartialError
        # fires). Better to report skipped and let the user split by hand
        # than to delete the original after losing topic N+1 onwards.
        if len(pages) >= _MAX_CHILDREN_CEILING:
            logger.info(
                "non_atomic_page split for %s refused: LLM emitted %d "
                "children at the prompt cap — possible topic loss, "
                "split by hand and re-run",
                issue.path,
                len(pages),
            )
            return None

        children = dedup_pages_by_slug(pages, strategy="merge_body")
        # Refuse the whole split on ANY child collision — silently
        # filtering would let the proposal still emit delete_page for
        # the original, dropping the colliding child's content along
        # with the source. The user resolves the collision by hand
        # (rename, merge, delete) and re-runs propose.
        existing_paths = {p.path for p in ctx.all_pages}
        for child in children:
            if child.path == issue.path:
                logger.info(
                    "non_atomic_page split for %s refused: child path "
                    "collides with original — split by hand and re-run",
                    issue.path,
                )
                return None
            if child.path in existing_paths:
                logger.info(
                    "non_atomic_page split for %s refused: child %s "
                    "collides with existing K-page — resolve by hand "
                    "and re-run",
                    issue.path,
                    child.path,
                )
                return None
        if len(children) < _MIN_CHILDREN:
            return None

        ops: list[FixOperation] = [
            FixOperation(
                kind="create_page",
                path=child.path,
                new_frontmatter=page_to_op_frontmatter(child),
                new_body=child.body,
                expected_hash=None,
            )
            for child in children
        ]
        ops.append(
            FixOperation(
                kind="delete_page",
                path=issue.path,
                expected_hash=bytes_sha256(file_bytes),
            )
        )

        return FixProposal(
            proposal_id=str(uuid.uuid4()),
            issue_kind=issue.kind,
            issue_path=issue.path,
            issue_detail=issue.detail,
            issue_line=issue.line,
            operations=ops,
            rationale=(
                f"LLM split — {len(children)} atomic children + delete original"
            ),
            source="llm",
        )


