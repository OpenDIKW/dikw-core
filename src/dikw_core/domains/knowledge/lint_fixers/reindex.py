"""Fixer for ``stale_index`` + ``untracked_file`` — re-project a K/W file.

Pure deterministic — no LLM, no fuzzy match. Both issue kinds resolve to
the same action: the file on disk is the source of truth (ADR-0005: disk
is authoritative, the DB is a rebuildable projection), so a row that
disagrees with disk (``stale_index``) or is missing entirely
(``untracked_file``) is the lagging side. This fixer emits a single
``reindex_page`` op that re-projects the *current* on-disk bytes through
``persist_knowledge`` / ``persist_wisdom`` at apply time — re-chunk,
re-link, re-provenance, (inline-or-deferred) embed. It **never** rewrites
the user's file (D1: the user owns the on-disk markdown) and **never**
re-runs synth (synth regenerates from the D-source; reindex preserves the
hand-edited bytes verbatim).

One fixer, registered under both kinds: the remediation is identical, and
``run_lint_propose`` dispatches by the registry key, not by ``.kind``.

``expected_hash`` is intentionally ``None`` — re-projecting whatever bytes
are currently on disk is always "make the DB match disk", so an edit
between propose and apply is harmless. The only safety re-check at apply is
"the file is still present" (else the row is left for ``missing_file``).
"""

from __future__ import annotations

import uuid
from typing import Any

from ....schemas import Layer
from ..lint import LintKind
from ..lint_fix import FixerContext, FixOperation, FixProposal


def _layer_for_path(path: str) -> Layer | None:
    """Map a base-relative path to its DIKW layer by tree prefix.

    K/W only — D-layer source discovery is owned by ``ingest`` and the
    detector never emits a ``sources/`` reindex issue, but the fixer guards
    its own input (a malformed / persisted proposal could carry one).
    """
    norm = path.replace("\\", "/")
    if norm == "knowledge" or norm.startswith("knowledge/"):
        return Layer.KNOWLEDGE
    if norm == "wisdom" or norm.startswith("wisdom/"):
        return Layer.WISDOM
    return None


class ReindexPageFixer:
    """Re-project a hand-edited (``stale_index``) or untracked
    (``untracked_file``) K/W markdown file into storage.

    Always ``source="heuristic"`` (the spelling for deterministic fixers)
    — no LLM call, no provider dependencies. Registered for both kinds.
    """

    # Cosmetic Protocol marker only — dispatch is by registry key, and this
    # one instance is registered under both ``stale_index`` and
    # ``untracked_file`` (see ``lint_fixers/__init__.py``).
    kind: LintKind = "stale_index"

    async def propose(
        self,
        issue: Any,
        ctx: FixerContext,
        reporter: Any,
    ) -> FixProposal | None:
        abs_path = (ctx.base_root / issue.path).resolve()
        if not abs_path.is_file():
            # File vanished between scan and propose — nothing to re-project.
            # A ``stale_index`` issue degrades to ``missing_file`` on the next
            # lint; an ``untracked_file`` simply disappears.
            return None
        layer = _layer_for_path(issue.path)
        if layer is None:
            return None

        op = FixOperation(
            kind="reindex_page",
            path=issue.path,
            layer=layer,
            # No ``expected_hash`` — apply re-projects the current bytes; an
            # intervening edit is harmless (still "make the DB match disk").
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
                f"re-project {layer.value}-layer page from current on-disk "
                "bytes (re-chunk + re-link + re-provenance; file unchanged)"
            ),
            source="heuristic",
        )
