"""Path normalization for cross-platform / cross-rename uniqueness.

The ``documents.path`` column preserves the user's original spelling (the
display path); a sibling ``path_key`` column carries the normalized form
that the engine uses for uniqueness and lookup. Splitting the two lets
``MyDoc.md`` and ``mydoc.md`` resolve to the same logical document while
the UI still shows whichever spelling is currently on disk.

Normalization composes two steps:

1. **Unicode NFC.** macOS HFS+/APFS hands out filenames in NFD (``é`` =
   ``e + ́`` as two codepoints). Linux and Windows write NFC (``é`` as a
   single codepoint). A round-trip through Dropbox / iCloud / git can
   yield byte-different strings for the same logical filename — Python
   string equality (``==``) is False on those, so a re-ingest after a
   sync would otherwise insert a duplicate row.

2. **``casefold()``.** macOS HFS+ and Windows NTFS are case-insensitive
   by default; renaming ``MyDoc.md`` to ``mydoc.md`` in Finder produces
   the same on-disk inode but a different Python string. ``casefold()``
   is a more aggressive ``.lower()`` that also handles Unicode-edge
   pairs that ``.lower()`` skips (e.g. German sharp-s expands to
   ``ss``, Greek final sigma collapses to mid-position sigma).

Use the result for ``WHERE path_key = ?`` lookup and for ``doc_id``
derivation. Never display ``path_key`` to the user — it is lossy by
design.
"""

from __future__ import annotations

import unicodedata

from ...schemas import Layer


def normalize_path(path: str) -> str:
    """Return ``path`` canonicalized for uniqueness comparison.

    NFC + ``casefold`` — see module docstring for the rationale.
    """
    return unicodedata.normalize("NFC", path).casefold()


def doc_id_for(layer: Layer, logical_path: str) -> str:
    """Canonical ``doc_id`` for a ``(layer, path)`` pair.

    Deterministic over the normalized path so the same file written
    under different macOS NFD / NTFS-case spellings still resolves to
    the same row. The single source of truth — engine and tests
    construct doc_ids through here so they cannot drift.
    """
    return f"{layer.value}:{normalize_path(logical_path)}"


__all__ = ["doc_id_for", "normalize_path"]
