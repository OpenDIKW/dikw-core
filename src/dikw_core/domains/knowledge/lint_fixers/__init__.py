"""Per-rule fix-proposal implementations.

PR1 shipped ``broken_wikilink`` (heuristic-only). PR2 adds the LLM
stub fallback for ``broken_wikilink`` + the ``non_atomic_page`` fixer;
PR3 will add ``orphan_page`` + ``duplicate_title``.
"""

from __future__ import annotations

from ..lint import LintKind
from ..lint_fix import Fixer
from .broken_wikilink import BrokenWikilinkFixer
from .non_atomic_page import NonAtomicPageFixer

FIXER_REGISTRY: dict[LintKind, Fixer] = {
    "broken_wikilink": BrokenWikilinkFixer(),
    "non_atomic_page": NonAtomicPageFixer(),
}

__all__ = ["FIXER_REGISTRY", "BrokenWikilinkFixer", "NonAtomicPageFixer"]
