from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dikw_core.domains.knowledge.page import (
    build_page,
    default_page_path,
    make_page_id,
    read_page,
    slugify,
    write_page,
)


def test_slugify_strips_punctuation_and_accents() -> None:
    assert slugify("DIKW Pyramid — overview") == "dikw-pyramid-overview"
    assert slugify("") == "untitled"


def test_make_page_id_is_stable() -> None:
    a = make_page_id("DIKW Pyramid", "concept")
    b = make_page_id("DIKW Pyramid", "concept")
    c = make_page_id("DIKW Pyramid", "entity")
    assert a == b
    assert a != c
    assert a.startswith("K-")


def test_default_page_path_buckets_by_type() -> None:
    assert default_page_path("concept", "DIKW Pyramid") == "knowledge/concepts/dikw-pyramid.md"
    assert default_page_path("entity", "Andrej Karpathy") == "knowledge/entities/andrej-karpathy.md"
    assert default_page_path("note", "Random thought") == "knowledge/notes/random-thought.md"
    # Custom types declared in SchemaConfig.page_types pluralize as <type>s
    # so a "topic" bucket lands under knowledge/topics/ without needing
    # _TYPE_FOLDERS to be expanded for every new type the user declares.
    assert default_page_path("topic", "SpaceX") == "knowledge/topics/spacex.md"


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    page = build_page(
        title="DIKW pyramid",
        body="# DIKW pyramid\n\nA layered model.",
        type_="concept",
        tags=["dikw", "model"],
        sources=["sources/notes/dikw.md"],
    )
    write_page(tmp_path, page)
    read_back = read_page(tmp_path, page.path)
    assert read_back.title == "DIKW pyramid"
    assert read_back.type == "concept"
    assert "dikw" in read_back.tags
    assert "sources/notes/dikw.md" in read_back.sources
    assert read_back.id == page.id


def test_user_aliases_frontmatter_survives_roundtrip(tmp_path: Path) -> None:
    # Obsidian users (and gbrain-style enrich workflows) write a top-level
    # `aliases:` frontmatter list. dikw-core does NOT consume aliases yet —
    # PR_alias is deferred — but the field must survive write_page →
    # read_page via `extras` so a future consumer (and Obsidian itself)
    # can still see what the user wrote.
    page = build_page(
        title="Elon Musk",
        body="# Elon Musk\n\nFounder, several companies.",
        type_="entity",
        extras={"aliases": ["Musk", "Elon R. Musk"]},
    )
    write_page(tmp_path, page)
    read_back = read_page(tmp_path, page.path)
    assert read_back.extras.get("aliases") == ["Musk", "Elon R. Musk"]


@pytest.mark.parametrize("bad_path", ["../escaped.md", "knowledge/../../escaped.md"])
def test_write_page_rejects_path_escape(tmp_path: Path, bad_path: str) -> None:
    # A page.path that escapes the base must be refused before any disk
    # write. The synth parser already rejects traversal paths (#146/#149),
    # but write_page is a shared sink reachable by lint-apply and any future
    # writer, so it guards its own input as defense in depth.
    page = build_page(title="Escape", body="# Escape\n\nx", type_="concept")
    escaping = replace(page, path=bad_path)
    with pytest.raises(ValueError, match="outside base"):
        write_page(tmp_path, escaping)
    assert not (tmp_path.parent / "escaped.md").exists()
