"""Unit tests for ``domains/wisdom/page.py::author_from_path``.

The author is encoded by the directory under ``wisdom/`` rather than a
frontmatter field — the path itself is the attribution, so users
authoring in Obsidian don't have to mirror ``wisdom/elon-musk/...`` in
each file's frontmatter. ``author_from_path`` is the deterministic
extractor reused by ingest, retrieve (PR3), and lint (PR3).
"""

from __future__ import annotations

import pytest

from dikw_core.domains.wisdom.page import author_from_path


@pytest.mark.parametrize(
    "path,expected",
    [
        ("wisdom/elon-musk/first-principles-thinking.md", "elon-musk"),
        ("wisdom/elon-musk/nested/sub/be-relentless.md", "elon-musk"),
        ("wisdom/be-relentless.md", None),
        ("knowledge/first-principles.md", None),
        ("sources/notes/musk-bio.md", None),
        ("README.md", None),
        ("", None),
    ],
)
def test_author_from_path(path: str, expected: str | None) -> None:
    assert author_from_path(path) == expected
