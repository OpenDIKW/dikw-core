from __future__ import annotations

from pathlib import Path

from dikw_core.domains.knowledge.wiki import (
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
    assert default_page_path("concept", "DIKW Pyramid") == "wiki/concepts/dikw-pyramid.md"
    assert default_page_path("entity", "Andrej Karpathy") == "wiki/entities/andrej-karpathy.md"
    assert default_page_path("note", "Random thought") == "wiki/notes/random-thought.md"
    # Custom types declared in SchemaConfig.page_types pluralize as <type>s
    # so a "topic" bucket lands under wiki/topics/ without needing
    # _TYPE_FOLDERS to be expanded for every new type the user declares.
    assert default_page_path("topic", "SpaceX") == "wiki/topics/spacex.md"


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
