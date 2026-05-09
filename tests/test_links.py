from __future__ import annotations

from dikw_core.domains.knowledge.links import parse_links, resolve_links
from dikw_core.schemas import LinkType


def test_parse_wikilink_with_anchor_and_alias() -> None:
    body = "See [[Karpathy Wiki#Operations|the ops section]] for details.\n"
    links = parse_links(body)
    assert len(links) == 1
    assert links[0].kind is LinkType.WIKILINK
    assert links[0].target == "Karpathy Wiki"
    assert links[0].anchor == "Operations"
    assert links[0].line == 1


def test_parse_plain_markdown_link_and_url() -> None:
    body = (
        "line one\n"
        "\n"
        "Visit [RRF paper](https://example.com/rrf) or just https://example.com/home.\n"
    )
    links = parse_links(body)
    # we expect two: one MARKDOWN (as URL because https), one URL for the bare link
    kinds = [link.kind for link in links]
    assert LinkType.URL in kinds
    assert len(links) >= 2


def test_parse_local_markdown_link_is_markdown_type() -> None:
    body = "See [related](../notes/other.md) for more.\n"
    links = parse_links(body)
    assert len(links) == 1
    assert links[0].kind is LinkType.MARKDOWN
    assert links[0].target == "../notes/other.md"


def test_resolve_links_flags_unresolved_wikilinks() -> None:
    body = "Both [[Known Page]] and [[Unknown Page]] appear here."
    links = parse_links(body)
    resolved, unresolved = resolve_links(
        "doc:test",
        links,
        title_to_path={"Known Page": "wiki/concepts/known.md"},
    )
    assert len(resolved) == 1
    assert resolved[0].dst_path == "wiki/concepts/known.md"
    assert len(unresolved) == 1
    assert "Unknown Page" in unresolved[0].target_text


# ---- fuzzy normalize (PR1) -----------------------------------------------
#
# `resolve_links` does exact + case-insensitive lookup today; that misses
# typing variations users hit constantly: "Neural Network" vs
# "Neural Networks", "Elon Musk" vs "Elon Musk.", full-width 中文 punctuation
# trailing the title, etc. PR1 adds a deterministic L1 (NFKC + casefold +
# punctuation strip) + L2 (trailing s/es/ies stem) normalize that fires
# *after* exact + case-insensitive miss. Collision-on-normalize must NOT
# resolve — falling back to broken_wikilink keeps `dikw lint` honest.


def test_resolve_fuzzy_plural_variant() -> None:
    # body says "[[Neural Networks]]" but the page title is the singular form
    body = "See [[Neural Networks]] for the architecture."
    links = parse_links(body)
    resolved, unresolved = resolve_links(
        "doc:test",
        links,
        title_to_path={"Neural Network": "wiki/concepts/neural-network.md"},
    )
    assert len(resolved) == 1
    assert resolved[0].dst_path == "wiki/concepts/neural-network.md"
    assert unresolved == []


def test_resolve_fuzzy_punctuation_strip() -> None:
    body = "Per [[Elon Musk.]] biographies."
    links = parse_links(body)
    resolved, _ = resolve_links(
        "doc:test",
        links,
        title_to_path={"Elon Musk": "wiki/entities/elon-musk.md"},
    )
    assert len(resolved) == 1
    assert resolved[0].dst_path == "wiki/entities/elon-musk.md"


def test_resolve_fuzzy_collision_returns_unresolved() -> None:
    # "Tesla" the company and "tesla" the SI magnetic flux unit normalize to
    # the same key. We refuse to guess — leave broken so `dikw lint` reports.
    body = "Note about [[TESLA]] in the Maxwell paper."
    links = parse_links(body)
    resolved, unresolved = resolve_links(
        "doc:test",
        links,
        title_to_path={
            "Tesla": "wiki/entities/tesla-company.md",
            "tesla": "wiki/concepts/tesla-unit.md",
        },
    )
    assert resolved == []
    assert len(unresolved) == 1


def test_resolve_fuzzy_unicode_full_width_punctuation() -> None:
    # NFKC normalizes 中文/全角 punctuation; trailing comma must strip
    body = "见 [[DIKW pyramid，]] 模型。"  # noqa: RUF001 - intentional CJK punctuation in test data
    links = parse_links(body)
    resolved, _ = resolve_links(
        "doc:test",
        links,
        title_to_path={"DIKW pyramid": "wiki/concepts/dikw.md"},
    )
    assert len(resolved) == 1
    assert resolved[0].dst_path == "wiki/concepts/dikw.md"


def test_resolve_fuzzy_preserves_internal_punctuation() -> None:
    # ``C++`` and bare ``C`` are distinct concepts. The earlier
    # strip-everything-non-token implementation collapsed both onto
    # bare "c" and would have falsely cross-linked them. Boundary-only
    # strip preserves the differentiating ``++`` so each link resolves
    # to its own page.
    body = "Compare [[C++]] with bare [[C]]."
    links = parse_links(body)
    resolved, _ = resolve_links(
        "doc:test",
        links,
        title_to_path={
            "C++": "wiki/concepts/cpp.md",
            "C": "wiki/concepts/c-language.md",
        },
    )
    by_target = {r.dst_path for r in resolved}
    assert by_target == {
        "wiki/concepts/cpp.md",
        "wiki/concepts/c-language.md",
    }


def test_resolve_fuzzy_does_not_invent_link_to_unrelated_technical_title() -> None:
    # With only ``C++`` in the index, plain ``[[C]]`` must NOT
    # fuzzy-resolve to it — the strip-everything implementation
    # collapsed both onto bare "c", inventing an irreversible false
    # edge from one page to a totally unrelated one.
    body = "Mention [[C]] alone."
    links = parse_links(body)
    resolved, unresolved = resolve_links(
        "doc:test",
        links,
        title_to_path={"C++": "wiki/concepts/cpp.md"},
    )
    assert resolved == []
    assert len(unresolved) == 1


def test_resolve_fuzzy_handles_e_plural_singulars() -> None:
    # The earlier ``-es`` / ``-ies`` rewrites collapsed Uses->"us",
    # Databases->"databas", Movies->"movy" — none of which match the
    # actual singular page. Drop-trailing-s gets the regular cases
    # right and is what we ship.
    body = "Refer to [[Uses]], [[Databases]] and [[Movies]] sections."
    links = parse_links(body)
    resolved, unresolved = resolve_links(
        "doc:test",
        links,
        title_to_path={
            "Use": "wiki/concepts/use.md",
            "Database": "wiki/concepts/database.md",
            "Movie": "wiki/concepts/movie.md",
        },
    )
    assert {r.dst_path for r in resolved} == {
        "wiki/concepts/use.md",
        "wiki/concepts/database.md",
        "wiki/concepts/movie.md",
    }
    assert unresolved == []


def test_resolve_fuzzy_does_not_invent_us_link_from_uses() -> None:
    # Pre-fix: ``Uses`` normalized to ``us`` and a body ``[[US]]`` would
    # resolve to the Uses page (or vice versa) — a wrong-page edge.
    # With drop-trailing-s, ``Uses`` normalizes to ``use`` and ``US``
    # to ``us`` — distinct keys, no false link.
    body = "See [[US]] for the country page."
    links = parse_links(body)
    resolved, unresolved = resolve_links(
        "doc:test",
        links,
        title_to_path={"Uses": "wiki/concepts/uses.md"},
    )
    assert resolved == []
    assert len(unresolved) == 1


def test_resolve_fuzzy_preserves_leading_punctuation() -> None:
    # ``.NET`` (Microsoft framework) — leading ``.`` is meaningful.
    # Stripping it would let bare ``[[NET]]`` collapse onto the same
    # key as ``.NET`` and falsely fuzzy-resolve. Trailing-only strip
    # preserves the distinguishing dot.
    body = "Mention [[NET]] alone."
    links = parse_links(body)
    resolved, unresolved = resolve_links(
        "doc:test",
        links,
        title_to_path={".NET": "wiki/concepts/dotnet.md"},
    )
    assert resolved == []
    assert len(unresolved) == 1


def test_resolve_fuzzy_does_not_stem_index_side() -> None:
    # A singular page title that happens to end in ``s`` (``Mars`` the
    # planet, ``OS``, ``HTTPS``) must NOT be indexed under its
    # plural-stemmed key. Otherwise bare ``[[Mar]]`` would falsely
    # fuzzy-resolve to the Mars page even though ``Mar`` carries no
    # plural marker. Index keys skip stemming; lookups still stem.
    body = "Visit [[Mar]] for context."
    links = parse_links(body)
    resolved, unresolved = resolve_links(
        "doc:test",
        links,
        title_to_path={"Mars": "wiki/entities/mars.md"},
    )
    assert resolved == []
    assert len(unresolved) == 1
