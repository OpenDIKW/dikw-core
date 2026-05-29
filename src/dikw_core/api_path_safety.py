"""Path-safety guard shared by the page / graph / asset / wisdom paths.

A leaf module (stdlib only) so the cluster modules can import it without
depending on the ``api`` facade.
"""

from __future__ import annotations

from pathlib import Path


def _assert_within(base: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and assert it stays under ``base``.

    Returns the resolved absolute path on success. Raises :class:`ValueError`
    (via :meth:`Path.relative_to`) when ``candidate`` escapes — callers
    translate to whatever domain exception fits (``PageNotFound`` /
    ``AssetNotFound``).
    """
    base_resolved = base.resolve()
    candidate_resolved = candidate.resolve()
    candidate_resolved.relative_to(base_resolved)
    return candidate_resolved
