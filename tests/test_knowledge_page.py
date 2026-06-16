from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dikw_core.domains.knowledge.page import (
    build_page,
    category_from_path,
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


def test_default_page_path_uses_category_path_verbatim() -> None:
    # The category path is the on-disk folder, used verbatim (no pluralization).
    assert default_page_path("concept", "DIKW Pyramid") == "knowledge/concept/dikw-pyramid.md"
    assert default_page_path("entity", "Andrej Karpathy") == "knowledge/entity/andrej-karpathy.md"
    # Arbitrary-depth, Unicode category paths land verbatim; only the filename
    # slug is ASCII-kebab.
    assert default_page_path("技术/架构", "RRF Fusion") == "knowledge/技术/架构/rrf-fusion.md"
    assert (
        default_page_path("产品/移动端", "App Onboarding")
        == "knowledge/产品/移动端/app-onboarding.md"
    )


def test_category_from_path_arbitrary_depth() -> None:
    assert category_from_path("knowledge/技术/架构/rrf-fusion.md") == "技术/架构"
    assert category_from_path("knowledge/concept/dikw.md") == "concept"
    # a file directly under knowledge/ has no category folder
    assert category_from_path("knowledge/x.md") == ""


def test_default_page_path_empty_category_collapses_to_knowledge_root() -> None:
    # A root-level page (no category folder, e.g. a hand-created knowledge/foo.md)
    # has category == "". ``default_page_path`` must collapse the empty segment
    # to ``knowledge/<slug>.md`` — NOT ``knowledge//<slug>.md`` — so it
    # round-trips with ``category_from_path`` and the orphan-merge guard, which
    # rebuilds the path from (category, slug), can still match the parent.
    assert default_page_path("", "My Title") == "knowledge/my-title.md"
    # round-trip: the rebuilt path reports the same (empty) category back
    assert category_from_path(default_page_path("", "My Title")) == ""


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    page = build_page(
        title="DIKW pyramid",
        body="# DIKW pyramid\n\nA layered model.",
        category="concept",
        tags=["dikw", "model"],
        sources=["sources/notes/dikw.md"],
    )
    write_page(tmp_path, page)
    read_back = read_page(tmp_path, page.path)
    assert read_back.title == "DIKW pyramid"
    assert read_back.category == "concept"
    assert "dikw" in read_back.tags
    assert "sources/notes/dikw.md" in read_back.sources
    assert read_back.id == page.id


def test_read_page_derives_category_from_path_when_frontmatter_missing(tmp_path: Path) -> None:
    # A hand-edited page that omits `category:` falls back to the folder path.
    target = tmp_path / "knowledge" / "技术" / "架构" / "rrf.md"
    target.parent.mkdir(parents=True)
    target.write_text("---\ntitle: RRF\n---\n\n# RRF\n\nbody\n", encoding="utf-8")
    read_back = read_page(tmp_path, "knowledge/技术/架构/rrf.md")
    assert read_back.category == "技术/架构"


def test_user_aliases_frontmatter_survives_roundtrip(tmp_path: Path) -> None:
    # Obsidian users (and gbrain-style enrich workflows) write a top-level
    # `aliases:` frontmatter list. dikw-core does NOT consume aliases yet —
    # PR_alias is deferred — but the field must survive write_page →
    # read_page via `extras` so a future consumer (and Obsidian itself)
    # can still see what the user wrote.
    page = build_page(
        title="Elon Musk",
        body="# Elon Musk\n\nFounder, several companies.",
        category="entity",
        extras={"aliases": ["Musk", "Elon R. Musk"]},
    )
    write_page(tmp_path, page)
    read_back = read_page(tmp_path, page.path)
    assert read_back.extras.get("aliases") == ["Musk", "Elon R. Musk"]


def test_write_page_extras_cannot_override_engine_keys(tmp_path: Path) -> None:
    # write_page is a shared sink (synth output, lint-apply, future writers).
    # ``extras`` must not override the engine-managed front-matter keys (id /
    # category / title / created / updated / tags / sources) — mirrors the
    # W-layer ``_RESERVED_FRONTMATTER_KEYS`` guard in write_wisdom_file. A
    # non-reserved user key (``aliases``) still passes through.
    page = build_page(
        title="Real Title",
        body="# Real Title\n\nReal body.",
        category="concept",
        tags=["real-tag"],
        sources=["sources/real.md"],
        extras={
            "id": "K-evil",
            "category": "假分类",
            "title": "EVIL Title",
            "created": "1999-01-01",
            "updated": "1999-01-01",
            "tags": ["evil-tag"],
            "sources": ["sources/EVIL.md"],
            "aliases": ["ok-passthrough"],
        },
    )
    write_page(tmp_path, page)
    read_back = read_page(tmp_path, page.path)
    assert read_back.title == "Real Title"
    assert read_back.category == "concept"
    assert read_back.id == page.id
    assert "real-tag" in read_back.tags
    assert "evil-tag" not in read_back.tags
    assert read_back.sources == ["sources/real.md"]
    # Non-reserved user key survives the round-trip.
    assert read_back.extras.get("aliases") == ["ok-passthrough"]
    # None of the EVIL overrides reached disk.
    text = (tmp_path / page.path).read_text(encoding="utf-8")
    assert "EVIL Title" not in text
    assert "假分类" not in text
    assert "sources/EVIL.md" not in text


def test_write_page_extras_cannot_corrupt_file_via_post_kwargs(tmp_path: Path) -> None:
    # A key colliding with ``frontmatter.Post(content, handler=None, **meta)`` —
    # ``handler`` / ``content`` from a disobedient LLM or a hand-edited file
    # flowing through lint — must not corrupt the file. With the naive ``**meta``
    # expansion, extras={"handler": "evil"} collapses the whole file to the
    # literal string "evil". Mirrors the W-layer guard.
    page = build_page(
        title="Real Title",
        body="# Real Title\n\nReal body content.",
        category="concept",
        extras={"handler": "evil", "content": "also evil"},
    )
    write_page(tmp_path, page)
    text = (tmp_path / page.path).read_text(encoding="utf-8")
    assert "Real Title" in text
    assert "Real body content." in text
    assert "evil" not in text
    # And the page still round-trips cleanly.
    read_back = read_page(tmp_path, page.path)
    assert read_back.title == "Real Title"


@pytest.mark.parametrize("bad_path", ["../escaped.md", "knowledge/../../escaped.md"])
def test_write_page_rejects_path_escape(tmp_path: Path, bad_path: str) -> None:
    # A page.path that escapes the base must be refused before any disk
    # write. The synth parser already rejects traversal paths (#146/#149),
    # but write_page is a shared sink reachable by lint-apply and any future
    # writer, so it guards its own input as defense in depth.
    page = build_page(title="Escape", body="# Escape\n\nx", category="concept")
    escaping = replace(page, path=bad_path)
    with pytest.raises(ValueError, match="outside base"):
        write_page(tmp_path, escaping)
    assert not (tmp_path.parent / "escaped.md").exists()
