from __future__ import annotations

import pytest

from dikw_core.domains.knowledge.page import build_page
from dikw_core.domains.knowledge.synthesize import (
    SynthesisError,
    SynthesisPartialError,
    dedup_pages_by_slug,
    parse_synthesis_response,
    synthesize_pages_from_text,
)

from .fakes import FakeLLM

_SINGLE_PAGE_RESPONSE = """
Here's the page:

<page path="knowledge/concepts/dikw-pyramid.md" type="concept">
---
tags: [dikw, model]
---

# DIKW pyramid

The DIKW pyramid organises data into four layers.

See also [[Karpathy LLM Wiki]].
</page>
"""


def test_parse_single_page_returns_one_element_list() -> None:
    pages = parse_synthesis_response(
        _SINGLE_PAGE_RESPONSE, source_path="sources/notes/dikw.md"
    )
    assert len(pages) == 1
    page = pages[0]
    assert page.path == "knowledge/concepts/dikw-pyramid.md"
    assert page.type == "concept"
    assert page.title == "DIKW pyramid"
    assert "Karpathy LLM Wiki" in page.body
    assert page.tags == ["dikw", "model"]
    assert page.sources == ["sources/notes/dikw.md"]


def test_parse_no_block_returns_empty_list() -> None:
    # Stage A: a section with nothing worth a knowledge page legitimately
    # responds with zero <page> blocks. This used to raise — verify the
    # new contract.
    assert parse_synthesis_response("no page here", source_path="x") == []


def test_parse_bad_type_falls_back_to_note() -> None:
    raw = (
        '<page path="knowledge/notes/x.md" type="random">\n'
        "---\ntags: []\n---\n\n"
        "# Random\n\nbody\n"
        "</page>"
    )
    pages = parse_synthesis_response(raw, source_path="sources/x.md")
    assert len(pages) == 1
    assert pages[0].type == "note"


def test_parse_accepts_custom_allowed_types() -> None:
    raw = (
        '<page path="knowledge/topics/spacex.md" type="topic">\n'
        "---\ntags: []\n---\n\n"
        "# SpaceX topic\n\nbody\n"
        "</page>"
    )
    pages = parse_synthesis_response(
        raw,
        source_path="sources/x.md",
        allowed_types=("entity", "concept", "note", "topic"),
    )
    assert len(pages) == 1
    assert pages[0].type == "topic"
    # default_page_path picks the right folder from the custom type.
    assert pages[0].path == "knowledge/topics/spacex.md"


def test_parse_unknown_type_falls_back_within_custom_allowed_types() -> None:
    raw = (
        '<page path="knowledge/notes/x.md" type="bogus">\n'
        "---\ntags: []\n---\n\n"
        "# Bogus\n\nbody\n"
        "</page>"
    )
    pages = parse_synthesis_response(
        raw,
        source_path="sources/x.md",
        allowed_types=("entity", "concept", "note", "topic"),
    )
    assert len(pages) == 1
    # "note" is in allowed_types → fall back to note (historical behaviour).
    assert pages[0].type == "note"


_TRUNCATED_LONE = """
<page path="knowledge/entities/spacex.md" type="entity">
---
tags: [aerospace]
---

# SpaceX

Founded by Elon Musk in 2002. The body keeps going but the LLM ran
out of tokens before it could write the closing tag.
"""


def test_parse_truncated_only_block_raises_synthesis_error() -> None:
    """LLM ran out of tokens before closing the page tag — must NOT be
    silently treated as "no page worth writing"."""
    with pytest.raises(SynthesisError) as excinfo:
        parse_synthesis_response(_TRUNCATED_LONE, source_path="src.md")
    assert not isinstance(excinfo.value, SynthesisPartialError)
    assert "unclosed <page>" in str(excinfo.value)
    assert "truncated" in str(excinfo.value)


_TRUNCATED_AFTER_GOOD = """
<page path="knowledge/entities/spacex.md" type="entity">
---
tags: [aerospace]
---

# SpaceX

Aerospace firm.
</page>

<page path="knowledge/entities/tesla.md" type="entity">
---
tags: [automotive]
---

# Tesla

EV maker that scaled production aggressively under Elon Musk.
The body keeps going but max_tokens cuts off here.
"""


def test_parse_truncation_after_good_block_is_partial_with_retry() -> None:
    """One complete block + one truncated opener: keep the good page,
    flag retry so the synth pipeline doesn't mark the source done."""
    with pytest.raises(SynthesisPartialError) as excinfo:
        parse_synthesis_response(_TRUNCATED_AFTER_GOOD, source_path="src.md")
    pe = excinfo.value
    assert len(pe.pages) == 1
    assert pe.pages[0].title == "SpaceX"
    assert pe.retry is True
    assert any("unclosed" in e for e in pe.errors)


def test_parse_partial_block_failure_does_not_request_retry() -> None:
    """Malformed individual block (no ATX title) is NOT recoverable — the
    same response would parse the same way next run."""
    with pytest.raises(SynthesisPartialError) as excinfo:
        parse_synthesis_response(_PARTIAL_RESPONSE, source_path="src.md")
    assert excinfo.value.retry is False


def test_parse_unknown_type_without_note_falls_back_to_first_allowed() -> None:
    raw = (
        '<page path="knowledge/x/x.md" type="bogus">\n'
        "---\ntags: []\n---\n\n"
        "# X\n\nbody\n"
        "</page>"
    )
    pages = parse_synthesis_response(
        raw,
        source_path="sources/x.md",
        # "note" NOT in allowed_types → fall back to first allowed.
        allowed_types=("entity", "concept", "topic"),
    )
    assert len(pages) == 1
    assert pages[0].type == "entity"


_MULTI_PAGE_RESPONSE = """
<page path="knowledge/entities/elon-musk.md" type="entity">
---
tags: [person]
---

# 埃隆·马斯克

Founder of [[SpaceX]] and [[Tesla]].
</page>

<page path="knowledge/entities/spacex.md" type="entity">
---
tags: [company, space]
---

# SpaceX

Aerospace manufacturer led by [[埃隆·马斯克]].
</page>

<page path="knowledge/concepts/falcon-1.md" type="concept">
---
tags: [rocket]
---

# Falcon 1

The first orbital rocket built by [[SpaceX]].
</page>
"""


def test_parse_multiple_blocks_returns_all_pages() -> None:
    pages = parse_synthesis_response(_MULTI_PAGE_RESPONSE, source_path="src/elon.md")
    assert len(pages) == 3
    titles = [p.title for p in pages]
    assert titles == ["埃隆·马斯克", "SpaceX", "Falcon 1"]
    types = {p.type for p in pages}
    assert types == {"entity", "concept"}


_PARTIAL_RESPONSE = """
<page path="knowledge/entities/ok.md" type="entity">
---
tags: []
---

# OK Entity

Body.
</page>

<page path="knowledge/concepts/no-title.md" type="concept">
---
tags: []
---

This block has no ATX title — should fail to parse on its own.
</page>
"""


def test_parse_partial_failure_keeps_good_pages() -> None:
    with pytest.raises(SynthesisPartialError) as excinfo:
        parse_synthesis_response(_PARTIAL_RESPONSE, source_path="src.md")
    err = excinfo.value
    assert len(err.pages) == 1
    assert err.pages[0].title == "OK Entity"
    assert len(err.errors) == 1
    assert "ATX" in err.errors[0]


def test_parse_all_blocks_fail_raises_synthesis_error() -> None:
    raw = (
        '<page path="knowledge/notes/a.md" type="note">\n'
        "---\ntags: []\n---\n\n"
        "no atx title here\n"
        "</page>\n"
        '<page path="knowledge/notes/b.md" type="note">\n'
        "---\ntags: []\n---\n\n"
        "still no title\n"
        "</page>"
    )
    with pytest.raises(SynthesisError) as excinfo:
        parse_synthesis_response(raw, source_path="src.md")
    # Should NOT be SynthesisPartialError — all blocks failed.
    assert not isinstance(excinfo.value, SynthesisPartialError)
    assert "all 2 <page> blocks failed" in str(excinfo.value)


# --- dedup tests --------------------------------------------------------


def _page(title: str, body: str, *, tags: list[str], sources: list[str]):
    return build_page(
        title=title,
        body=body,
        type_="entity",
        tags=tags,
        sources=sources,
        path=None,
        extras={},
    )


def test_dedup_merge_body_concatenates_and_unions_metadata() -> None:
    p1 = _page(
        "埃隆·马斯克",
        "# 埃隆·马斯克\n\nFrom group 1.\n",
        tags=["person"],
        sources=["src/elon.md"],
    )
    p2 = _page(
        "埃隆·马斯克",
        "# 埃隆·马斯克\n\nFrom group 2.\n",
        tags=["person", "tesla-ceo"],
        sources=["src/elon.md", "src/biography.md"],
    )

    out = dedup_pages_by_slug([p1, p2], strategy="merge_body")

    assert len(out) == 1
    merged = out[0]
    assert "From group 1." in merged.body
    assert "From group 2." in merged.body
    assert "---" in merged.body  # separator between contributions
    assert merged.tags == ["person", "tesla-ceo"]
    assert merged.sources == ["src/elon.md", "src/biography.md"]


def test_dedup_keep_first_drops_subsequent() -> None:
    p1 = _page(
        "埃隆·马斯克", "# 埃隆·马斯克\n\nFirst.\n", tags=["person"], sources=["src.md"]
    )
    p2 = _page(
        "埃隆·马斯克", "# 埃隆·马斯克\n\nSecond.\n", tags=["other"], sources=["src.md"]
    )

    out = dedup_pages_by_slug([p1, p2], strategy="keep_first")

    assert len(out) == 1
    assert "First." in out[0].body
    assert "Second." not in out[0].body
    assert out[0].tags == ["person"]


def test_dedup_preserves_input_order_for_distinct_paths() -> None:
    p1 = _page(
        "B Entity", "# B Entity\n\nb.\n", tags=[], sources=["src.md"]
    )
    p2 = _page(
        "A Entity", "# A Entity\n\na.\n", tags=[], sources=["src.md"]
    )
    p3 = _page(
        "C Entity", "# C Entity\n\nc.\n", tags=[], sources=["src.md"]
    )

    out = dedup_pages_by_slug([p1, p2, p3], strategy="merge_body")

    assert [p.title for p in out] == ["B Entity", "A Entity", "C Entity"]


# --- synthesize_pages_from_text shared helper -------------------------------


@pytest.mark.asyncio
async def test_synthesize_pages_from_text_returns_parsed_pages() -> None:
    """Helper drives one ``llm.complete`` call and returns the parsed
    pages — the shared "text → N pages" primitive that lint fixers
    (broken_wikilink evidence-backed grounded repair, non_atomic_page
    splitter) reuse."""
    fake = FakeLLM(response_text=_SINGLE_PAGE_RESPONSE)

    pages = await synthesize_pages_from_text(
        user_prompt="dummy prompt body",
        source_path="sources/notes/dikw.md",
        llm=fake,  # type: ignore[arg-type]
        model="fake-model",
        max_tokens=1024,
    )

    assert len(pages) == 1
    assert pages[0].title == "DIKW pyramid"
    # The helper must thread system + user_prompt + model through to
    # the provider untouched — fixers tune temperature / max_tokens
    # via the kwargs.
    assert fake.last_user == "dummy prompt body"
    assert fake.last_max_tokens == 1024


@pytest.mark.asyncio
async def test_synthesize_pages_from_text_propagates_parse_errors() -> None:
    """When the LLM returns no usable ``<page>`` block, the helper
    surfaces an empty list (legal "nothing to write" signal under the
    Stage A fan-out contract). Hard parse failures still raise."""
    fake = FakeLLM(response_text="No page block in here.")
    pages = await synthesize_pages_from_text(
        user_prompt="prompt",
        source_path="src.md",
        llm=fake,  # type: ignore[arg-type]
        model="fake-model",
        max_tokens=512,
    )
    assert pages == []
