"""Wisdom page utilities (0.3.0 PR2).

The W layer is hand-authored markdown under ``wisdom/<author>/<slug>.md``.
Authorship is encoded by directory rather than a frontmatter field, so
users writing in Obsidian don't have to mirror their path in YAML.
``author_from_path`` is the deterministic extractor reused by ingest
(for log entries / future per-author counters), and by retrieve and
lint in PR3.

A wisdom file directly under ``wisdom/<slug>.md`` (no author subdir)
is allowed and indexed with ``author = None`` — the engine never
synthesises a placeholder author from the slug.
"""

from __future__ import annotations

from pathlib import PurePosixPath


def author_from_path(path: str) -> str | None:
    """Return the author directory for a wisdom page path, or ``None``.

    ``path`` is the logical (POSIX-style, forward-slash) path the engine
    stores in ``DocumentRecord.path`` — e.g.
    ``wisdom/elon-musk/first-principles.md``. Anything not anchored at
    ``wisdom/<author>/<...>.md`` returns ``None`` (a file directly under
    ``wisdom/``, a non-wisdom path, or an empty input).
    """
    if not path:
        return None
    parts = PurePosixPath(path).parts
    # Need at least ``wisdom`` + author + a tail (the file itself, or
    # a deeper subtree the file lives under).
    if len(parts) < 3 or parts[0] != "wisdom":
        return None
    return parts[1]
