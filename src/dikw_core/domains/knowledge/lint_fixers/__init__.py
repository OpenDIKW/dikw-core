"""Per-rule fix-proposal implementations.

Coverage: ``broken_wikilink`` (heuristic + evidence-backed LLM repair),
``non_atomic_page`` (LLM split), ``orphan_page`` (heuristic strategy
router), ``missing_provenance`` (pure deterministic — sync provenance
table from frontmatter), ``missing_file`` (pure deterministic — purge an
orphaned row whose file is gone, D/K/W). The ``duplicate_title`` rule has
no fixer — the propose pipeline still reports it for human triage.
"""

from __future__ import annotations

from ..lint import LintKind
from ..lint_fix import Fixer
from .broken_wikilink import BrokenWikilinkFixer
from .missing_file import MissingFileFixer
from .missing_provenance import MissingProvenanceFixer
from .non_atomic_page import NonAtomicPageFixer
from .orphan_page import OrphanPageFixer

FIXER_REGISTRY: dict[LintKind, Fixer] = {
    "broken_wikilink": BrokenWikilinkFixer(),
    "non_atomic_page": NonAtomicPageFixer(),
    "orphan_page": OrphanPageFixer(),
    "missing_provenance": MissingProvenanceFixer(),
    "missing_file": MissingFileFixer(),
}

__all__ = [
    "FIXER_REGISTRY",
    "BrokenWikilinkFixer",
    "MissingFileFixer",
    "MissingProvenanceFixer",
    "NonAtomicPageFixer",
    "OrphanPageFixer",
]
