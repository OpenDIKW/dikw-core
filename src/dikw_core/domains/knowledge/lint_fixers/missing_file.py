"""Fixer for ``missing_file`` issues — purge an orphaned document row.

Pure deterministic — no LLM, no fuzzy match. A ``missing_file`` issue
means an *active* ``documents`` row (D, K, or W layer) whose backing file
is gone from disk. The file is the source of truth (ADR-0005: disk is
authoritative, the DB is a rebuildable projection), so the row is the
stale side: this fixer emits a single ``purge_document`` op that calls
``Storage.delete_document`` — removing the row + its outgoing edges.

Inbound edges from *live* pages are intentionally left (``delete_document``
clears only outgoing links/provenance); they surface as ``broken_wikilink``
on the next lint (D5: expose, never silently rewrite a user's page). A
truly dangling edge — one whose *both* ends were purged — clears itself
when each purge removes that page's outgoing edges.

The fixer probes storage to resolve which layer the path lives in (the
issue carries only the path), and bails when the file has reappeared or
the row has already been purged since the lint scan — the apply pass
re-checks both to close the propose→apply race.
"""

from __future__ import annotations

import uuid
from typing import Any

from ....schemas import Layer
from ...data.path_norm import doc_id_for
from ..lint import LintKind
from ..lint_fix import FixerContext, FixOperation, FixProposal


class MissingFileFixer:
    """Purge the orphaned ``documents`` row behind a ``missing_file`` issue.

    Always ``source="heuristic"`` (the spelling for deterministic fixers)
    — no LLM call, no provider dependencies.
    """

    kind: LintKind = "missing_file"

    async def propose(
        self,
        issue: Any,
        ctx: FixerContext,
        reporter: Any,
    ) -> FixProposal | None:
        abs_path = (ctx.base_root / issue.path).resolve()
        if abs_path.is_file():
            # File reappeared between scan and propose — the row is valid
            # again, nothing to purge. Cheap check first so a restored file
            # short-circuits before any storage round-trip.
            return None
        if ctx.storage is None:
            # Defensive: production always wires storage. Without it there's
            # nothing to probe or purge — skip rather than crash.
            return None

        # Resolve the orphaned row's layer (the issue carries only the path)
        # and confirm it still exists. A layer-prefixed path can only match
        # its own layer, so the first hit is unambiguous; ``None`` for all
        # three means the row was already purged since the scan.
        matched_layer: Layer | None = None
        for layer in (Layer.SOURCE, Layer.KNOWLEDGE, Layer.WISDOM):
            if await ctx.storage.get_document(doc_id_for(layer, issue.path)) is not None:
                matched_layer = layer
                break
        if matched_layer is None:
            return None

        op = FixOperation(
            kind="purge_document",
            path=issue.path,
            layer=matched_layer,
            # No ``expected_hash`` — the file is gone; the safety invariant
            # is "still absent", re-checked at apply, not "bytes unchanged".
            expected_hash=None,
        )
        return FixProposal(
            proposal_id=str(uuid.uuid4()),
            issue_kind=issue.kind,
            issue_path=issue.path,
            issue_detail=issue.detail,
            issue_line=None,
            operations=[op],
            rationale=(
                f"purge orphaned {matched_layer.value}-layer document row "
                "(backing file gone from disk)"
            ),
            source="heuristic",
        )
