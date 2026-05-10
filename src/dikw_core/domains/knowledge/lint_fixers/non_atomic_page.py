"""Fixer for ``non_atomic_page`` issues.

Stage A's atomicity rule says one wiki page = one self-contained idea
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
under the next lint pass, where the broken_wikilink fixer (with
``--enable-llm`` for stub fallback if needed) handles them. Keeping
this fixer's scope narrow avoids the embedding-similarity machinery
needed to pick the "right" child for each external referrer.
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
    DEFAULT_ALLOWED_TYPES,
    DEFAULT_SYNTH_SYSTEM,
    dedup_pages_by_slug,
)

logger = logging.getLogger(__name__)

# A "split" only makes sense when the LLM finds at least two atomic
# children — one means "page is fine as-is", and we'd rather skip
# than churn its slug. Higher floors silently mask real splits.
_MIN_CHILDREN = 2

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

        abs_path = (ctx.wiki_root / issue.path).resolve()
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
        allowed_types = tuple(cfg.schema_.page_types) or DEFAULT_ALLOWED_TYPES
        max_pages = cfg.synth.max_pages_per_group

        user_prompt = prompts.load("synthesize").format(
            source_path=issue.path,
            source_body=body,
            group_outline="(whole page — split into atomic children)",
            group_index=1,
            group_total=1,
            max_pages=max_pages,
            allowed_types=" | ".join(allowed_types),
        )

        pages = await safe_synthesize_pages(
            user_prompt=user_prompt,
            source_path=issue.path,
            llm=ctx.llm,
            model=cfg.provider.llm_model,
            max_tokens=cfg.provider.llm_max_tokens_synth,
            allowed_types=allowed_types,
            system=DEFAULT_SYNTH_SYSTEM,
            log_label="non_atomic_page split",
        )
        if not pages:
            return None

        children = dedup_pages_by_slug(pages, strategy="merge_body")
        # Skip a child whose path is the original (a no-op delete +
        # recreate cycle) or collides with an existing K-layer page
        # (apply would refuse the create anyway). Filter the children,
        # not the whole proposal — remaining children still split the
        # original cleanly.
        existing_paths = {p.path for p in ctx.all_pages}
        children = [
            c for c in children
            if c.path != issue.path and c.path not in existing_paths
        ]
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


