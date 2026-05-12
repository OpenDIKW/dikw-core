"""Unit tests for ``check_atomicity`` — the pure-function form of the
``non_atomic_page`` heuristic.

These tests pin down each violation independently so threshold tweaks or
refactors are easy to validate without spinning up a wiki + storage like
``test_lint.py`` does. End-to-end behaviour through ``run_lint`` stays
covered by ``test_lint.py``.
"""

from __future__ import annotations

import dataclasses

import pytest

from dikw_core.domains.knowledge.lint import (
    _ATOMIC_BODY_CHARS,
    _ATOMIC_H2_COUNT,
    _ATOMIC_TAG_DOMAIN_COUNT,
    _ATOMIC_WIKILINK_COUNT,
    AtomicityVerdict,
    check_atomicity,
)


def _body_with_h2s(n: int) -> str:
    return "# Title\n\n" + "".join(f"## Section {i}\n\nbody\n\n" for i in range(n))


def _body_with_wikilinks(n: int) -> str:
    links = " ".join(f"[[Topic {i}]]" for i in range(n))
    return f"# Title\n\nLinks: {links}\n"


# ---------- happy path ----------


def test_atomic_page_returns_atomic_verdict() -> None:
    body = (
        "# Tight Page\n\n"
        "Two short paragraphs about a single subject.\n\n"
        "Linked to [[Some Other Concept]] for context.\n"
    )
    v = check_atomicity(body=body, tags=["topic/example"])
    assert isinstance(v, AtomicityVerdict)
    assert v.atomic is True
    assert v.violations == ()


def test_empty_body_is_atomic() -> None:
    v = check_atomicity(body="", tags=[])
    assert v.atomic is True
    assert v.violations == ()


# ---------- body length ----------


def test_long_body_violates() -> None:
    body = "# Long\n\n" + ("paragraph filler " * 200)
    v = check_atomicity(body=body, tags=[])
    assert v.atomic is False
    assert any("body" in s and "chars" in s for s in v.violations)


def test_body_at_threshold_is_atomic() -> None:
    body = "# T\n\n" + ("a" * (_ATOMIC_BODY_CHARS - len("# T\n\n")))
    assert len(body) == _ATOMIC_BODY_CHARS
    v = check_atomicity(body=body, tags=[])
    assert v.atomic is True


def test_body_one_over_threshold_violates() -> None:
    body = "# T\n\n" + ("a" * (_ATOMIC_BODY_CHARS - len("# T\n\n") + 1))
    assert len(body) == _ATOMIC_BODY_CHARS + 1
    v = check_atomicity(body=body, tags=[])
    assert v.atomic is False
    assert any("body" in s and "chars" in s for s in v.violations)


# ---------- H1 count ----------


def test_multiple_h1_violates() -> None:
    body = "# First\n\nbody\n\n# Second\n\nbody\n"
    v = check_atomicity(body=body, tags=[])
    assert v.atomic is False
    assert any("H1" in s for s in v.violations)


def test_single_h1_is_atomic() -> None:
    body = "# Only One\n\nbody\n"
    v = check_atomicity(body=body, tags=[])
    assert v.atomic is True


def test_h1_inside_code_fence_does_not_count() -> None:
    body = (
        "# Real Title\n\n"
        "```\n"
        "# install deps\n"
        "# more comments\n"
        "```\n\n"
        "body\n"
    )
    v = check_atomicity(body=body, tags=[])
    assert v.atomic is True


# ---------- H2 count ----------


def test_h2_count_at_threshold_is_atomic() -> None:
    body = _body_with_h2s(_ATOMIC_H2_COUNT)
    v = check_atomicity(body=body, tags=[])
    assert v.atomic is True


def test_h2_count_one_over_threshold_violates() -> None:
    body = _body_with_h2s(_ATOMIC_H2_COUNT + 1)
    v = check_atomicity(body=body, tags=[])
    assert v.atomic is False
    assert any("H2" in s for s in v.violations)


def test_h2_inside_code_fence_does_not_count() -> None:
    body = (
        "# Title\n\n"
        "```\n"
        "## a\n## b\n## c\n## d\n## e\n"
        "```\n\n"
        "body\n"
    )
    v = check_atomicity(body=body, tags=[])
    assert v.atomic is True


# ---------- wikilinks ----------


def test_distinct_wikilink_count_at_threshold_is_atomic() -> None:
    body = _body_with_wikilinks(_ATOMIC_WIKILINK_COUNT)
    v = check_atomicity(body=body, tags=[])
    assert v.atomic is True


def test_distinct_wikilink_count_over_threshold_violates() -> None:
    body = _body_with_wikilinks(_ATOMIC_WIKILINK_COUNT + 1)
    v = check_atomicity(body=body, tags=[])
    assert v.atomic is False
    assert any("wikilink" in s for s in v.violations)


def test_repeated_wikilinks_count_as_one() -> None:
    body = "# Title\n\n" + ("[[Elon Musk]] " * 30) + "\n"
    v = check_atomicity(body=body, tags=[])
    # 30 occurrences of same target → distinct=1, under threshold
    assert v.atomic is True


# ---------- tag domains ----------


def test_flat_tags_dont_count() -> None:
    v = check_atomicity(body="# T\n\nbody\n", tags=["a", "b", "c", "d", "e"])
    assert v.atomic is True


def test_namespaced_tags_one_domain_is_atomic() -> None:
    v = check_atomicity(
        body="# T\n\nbody\n",
        tags=["topic/a", "topic/b"],
    )
    assert v.atomic is True


def test_namespaced_tags_two_domains_violate() -> None:
    v = check_atomicity(
        body="# T\n\nbody\n",
        tags=["topic/a", "area/b"],
    )
    assert v.atomic is False
    assert any("domains" in s for s in v.violations)
    assert _ATOMIC_TAG_DOMAIN_COUNT == 1  # sanity: spec assumption


# ---------- multi-violation ----------


def test_multiple_violations_all_reported() -> None:
    body = (
        _body_with_h2s(_ATOMIC_H2_COUNT + 2)
        + _body_with_wikilinks(_ATOMIC_WIKILINK_COUNT + 5)
    )
    v = check_atomicity(body=body, tags=["topic/a", "area/b"])
    assert v.atomic is False
    # should have at least three independent violations (h2, wikilink, tags)
    assert len(v.violations) >= 3


# ---------- immutability ----------


def test_verdict_is_immutable() -> None:
    v = check_atomicity(body="# T\n\nbody\n", tags=[])
    with pytest.raises(dataclasses.FrozenInstanceError):
        # AtomicityVerdict should be frozen
        v.atomic = False  # type: ignore[misc]
