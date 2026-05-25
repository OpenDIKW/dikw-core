"""Wisdom (W) layer module.

0.3.0 PR2 wires wisdom into the documents pipeline. ``author_from_path``
is the directory-driven authorship extractor; the persistence path
itself goes through ``domains.knowledge.page_index.persist_page``
(generalised in PR2 to take a layer parameter).
"""

from .page import author_from_path

__all__ = ["author_from_path"]
