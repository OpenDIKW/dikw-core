"""Per-rule fix-proposal implementations.

PR1 ships only ``broken_wikilink`` (heuristic-only). PR2 adds the LLM
stub fallback for ``broken_wikilink`` + the ``non_atomic_page`` fixer;
PR3 adds ``orphan_page`` + ``duplicate_title``.
"""

from __future__ import annotations

from ..lint import LintKind
from ..lint_fix import Fixer
from .broken_wikilink import BrokenWikilinkFixer

FIXER_REGISTRY: dict[LintKind, Fixer] = {
    "broken_wikilink": BrokenWikilinkFixer(),
}

__all__ = ["FIXER_REGISTRY", "BrokenWikilinkFixer"]
