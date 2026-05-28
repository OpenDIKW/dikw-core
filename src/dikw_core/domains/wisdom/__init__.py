"""Wisdom (W) layer module.

0.3.0 PR2 wired wisdom into the documents pipeline. 0.4.0 promotes
``persist_wisdom`` to a first-class entry point alongside K-layer
``persist_knowledge`` and D-layer ``persist_source``. ``write_wisdom_page``
is the sole engine caller; ``dikw client ingest`` no longer scans the
wisdom tree (see CHANGELOG 0.4.0 for the rationale).
"""

from .page import author_from_path, make_wisdom_path, validate_kebab, write_wisdom_file
from .persist import persist_wisdom

__all__ = [
    "author_from_path",
    "make_wisdom_path",
    "persist_wisdom",
    "validate_kebab",
    "write_wisdom_file",
]
