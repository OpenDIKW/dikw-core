"""Per-rule fix-proposal implementations.

Coverage: ``broken_wikilink`` (heuristic + LLM stub), ``non_atomic_page``
(LLM split), ``orphan_page`` (heuristic strategy router),
``missing_provenance`` (pure deterministic — sync provenance table from
frontmatter). The ``duplicate_title`` rule has no fixer — the propose
pipeline still reports it for human triage.
"""

from __future__ import annotations

from ..lint import LintKind
from ..lint_fix import Fixer
from .broken_wikilink import BrokenWikilinkFixer
from .missing_provenance import MissingProvenanceFixer
from .non_atomic_page import NonAtomicPageFixer
from .orphan_page import OrphanPageFixer

FIXER_REGISTRY: dict[LintKind, Fixer] = {
    "broken_wikilink": BrokenWikilinkFixer(),
    "non_atomic_page": NonAtomicPageFixer(),
    "orphan_page": OrphanPageFixer(),
    "missing_provenance": MissingProvenanceFixer(),
}

__all__ = [
    "FIXER_REGISTRY",
    "BrokenWikilinkFixer",
    "MissingProvenanceFixer",
    "NonAtomicPageFixer",
    "OrphanPageFixer",
]
