"""Per-rule fix-proposal implementations.

Coverage: ``broken_wikilink`` (heuristic + evidence-backed LLM repair),
``non_atomic_page`` (LLM split), ``orphan_page`` (heuristic strategy
router), ``missing_provenance`` (pure deterministic — sync provenance
table from frontmatter), ``missing_file`` (pure deterministic — purge an
orphaned row whose file is gone, D/K/W), ``stale_index`` + ``untracked_file``
(pure deterministic — re-project the on-disk K/W bytes into storage; one
``ReindexPageFixer`` serves both). The ``duplicate_title`` and
``dangling_provenance`` rules have no fixer — the propose pipeline still
reports them for human triage (``dangling_provenance`` is read-only by
design: the user owns the ``sources:`` frontmatter, ADR-0005/ADR-0001).
"""

from __future__ import annotations

from ..lint import LintKind
from ..lint_fix import Fixer
from .broken_wikilink import BrokenWikilinkFixer
from .missing_file import MissingFileFixer
from .missing_provenance import MissingProvenanceFixer
from .non_atomic_page import NonAtomicPageFixer
from .orphan_page import OrphanPageFixer
from .reindex import ReindexPageFixer

# One ``ReindexPageFixer`` instance serves both fs-drift kinds — the
# remediation (re-project disk bytes) is identical.
_reindex_fixer = ReindexPageFixer()

FIXER_REGISTRY: dict[LintKind, Fixer] = {
    "broken_wikilink": BrokenWikilinkFixer(),
    "non_atomic_page": NonAtomicPageFixer(),
    "orphan_page": OrphanPageFixer(),
    "missing_provenance": MissingProvenanceFixer(),
    "missing_file": MissingFileFixer(),
    "stale_index": _reindex_fixer,
    "untracked_file": _reindex_fixer,
}

__all__ = [
    "FIXER_REGISTRY",
    "BrokenWikilinkFixer",
    "MissingFileFixer",
    "MissingProvenanceFixer",
    "NonAtomicPageFixer",
    "OrphanPageFixer",
    "ReindexPageFixer",
]
