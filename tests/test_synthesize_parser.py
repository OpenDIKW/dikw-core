from __future__ import annotations

import pytest

from dikw_core.domains.knowledge.page import build_page
from dikw_core.domains.knowledge.synthesize import (
    DEFAULT_FALLBACK,
    SynthesisError,
    SynthesisPartialError,
    dedup_pages_by_slug,
    parse_synthesis_response,
    synthesize_pages_from_text,
)

from .fakes import FakeLLM

_SINGLE_PAGE_RESPONSE = """
Here's the page:

<page category="concept" slug="dikw-pyramid">
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
    # Engine builds the path from category + slug — no LLM-supplied path.
    assert page.path == "knowledge/concept/dikw-pyramid.md"
    assert page.category == "concept"
    assert page.title == "DIKW pyramid"
    assert "Karpathy LLM Wiki" in page.body
    assert page.tags == ["dikw", "model"]
    assert page.sources == ["sources/notes/dikw.md"]


def test_parse_drops_non_tags_llm_frontmatter() -> None:
    # The LLM is told to emit ONLY ``tags`` in front-matter; everything else is
    # engine-managed (``title`` from the body H1, ``category``/``slug`` from the
    # <page> attributes, ``id``/``sources``/``created``/``updated`` by the
    # engine). A disobedient model that also emits those keys — or a
    # Post-collision key like ``handler``/``content`` — must have them dropped
    # at parse time so they never reach ``extras`` and can neither override the
    # engine fields nor corrupt the file on write. Enforcing the whitelist here
    # covers every LLM-sourced page (synth fan-out + the lint grounded/split/
    # merge fixers that share this parser) at one point.
    raw = (
        '<page category="concept" slug="dikw-pyramid">\n'
        "---\n"
        "tags: [dikw, model]\n"
        "id: K-llm-fake\n"
        "title: EVIL Title\n"
        "category: 假分类\n"
        "sources: [sources/HALLUCINATED.md]\n"
        "created: '1999-01-01'\n"
        "updated: '1999-01-01'\n"
        "lint: {skip: [orphan_page]}\n"
        "handler: evil\n"
        "content: evil\n"
        "---\n\n"
        "# DIKW pyramid\n\n"
        "Body.\n"
        "</page>"
    )
    pages = parse_synthesis_response(raw, source_path="sources/notes/dikw.md")
    assert len(pages) == 1
    page = pages[0]
    # Only ``tags`` survives from the LLM front-matter; extras is empty.
    assert page.tags == ["dikw", "model"]
    assert page.extras == {}
    # Engine-managed fields keep their engine-derived values, NOT the LLM's.
    assert page.sources == ["sources/notes/dikw.md"]
    assert page.category == "concept"
    assert page.title == "DIKW pyramid"
    assert page.path == "knowledge/concept/dikw-pyramid.md"


def test_parse_no_block_returns_empty_list() -> None:
    # Stage A: a section with nothing worth a knowledge page legitimately
    # responds with zero <page> blocks. This used to raise — verify the
    # new contract.
    assert parse_synthesis_response("no page here", source_path="x") == []


def test_parse_unknown_category_falls_back_to_fallback_bucket() -> None:
    # Closed-set taxonomy: a category the LLM invents (or that isn't declared)
    # is filed under the fallback bucket, not silently coerced to a sibling.
    raw = (
        '<page category="made-up" slug="x">\n'
        "---\ntags: []\n---\n\n# Random\n\nbody\n"
        "</page>"
    )
    pages = parse_synthesis_response(raw, source_path="sources/x.md")
    assert len(pages) == 1
    assert pages[0].category == DEFAULT_FALLBACK
    assert pages[0].path == f"knowledge/{DEFAULT_FALLBACK}/x.md"


def test_parse_omitted_category_falls_back() -> None:
    # The prompt tells the model to omit `category` when none fits → fallback.
    raw = "<page slug=\"x\">\n---\ntags: []\n---\n\n# X\n\nbody\n</page>"
    pages = parse_synthesis_response(raw, source_path="sources/x.md")
    assert pages[0].category == DEFAULT_FALLBACK


def test_parse_accepts_declared_hierarchical_category() -> None:
    raw = (
        '<page category="技术/架构" slug="rrf-fusion">\n'
        "---\ntags: []\n---\n\n# RRF Fusion\n\nbody\n"
        "</page>"
    )
    pages = parse_synthesis_response(
        raw,
        source_path="sources/x.md",
        allowed_categories=("技术/架构", "产品/移动端"),
    )
    assert len(pages) == 1
    assert pages[0].category == "技术/架构"
    assert pages[0].path == "knowledge/技术/架构/rrf-fusion.md"


def test_parse_custom_fallback_is_honored() -> None:
    raw = '<page category="nope" slug="x">\n---\n---\n\n# X\n\nb\n</page>'
    pages = parse_synthesis_response(
        raw, source_path="s.md", allowed_categories=("entity",), fallback="待归档"
    )
    assert pages[0].category == "待归档"
    assert pages[0].path == "knowledge/待归档/x.md"


def test_parse_category_nfc_normalized_before_membership() -> None:
    # ``config._validate_category_path`` NFC-normalizes every declared category,
    # so ``allowed_categories`` holds NFC forms. If the LLM emits a decomposed
    # (NFD) spelling of the same name, the parser must still accept it instead of
    # silently bucketing a validly-declared category into the fallback.
    import unicodedata

    declared = unicodedata.normalize("NFC", "café")
    nfd = unicodedata.normalize("NFD", "café")
    assert declared != nfd  # genuinely different code-point sequences
    raw = f'<page category="{nfd}" slug="x">\n---\n---\n\n# X\n\nb\n</page>'
    pages = parse_synthesis_response(
        raw, source_path="s.md", allowed_categories=(declared,), fallback="未分类"
    )
    assert pages[0].category == declared
    assert pages[0].path == f"knowledge/{declared}/x.md"


def test_parse_slug_defaults_to_slugified_title_when_omitted() -> None:
    raw = '<page category="concept">\n---\n---\n\n# DIKW Pyramid\n\nbody\n</page>'
    pages = parse_synthesis_response(raw, source_path="s.md")
    assert pages[0].path == "knowledge/concept/dikw-pyramid.md"


def test_parse_preserves_explicit_ascii_slug_for_cjk_title() -> None:
    """The model emits a pinyin/ASCII slug for a non-ASCII title (``slugify``
    would collapse the CJK title to ``untitled`` and collide every CJK page),
    while the title itself stays in the source language."""
    raw = (
        '<page category="concept" slug="shen-jing-wang-luo">\n'
        "---\ntags: []\n---\n\n# 神经网络\n\n关于神经网络的说明。\n"
        "</page>"
    )
    pages = parse_synthesis_response(raw, source_path="sources/x.md")
    assert len(pages) == 1
    assert pages[0].path == "knowledge/concept/shen-jing-wang-luo.md"
    assert pages[0].title == "神经网络"


def test_parse_llm_slug_cannot_escape_base() -> None:
    """The slug is run through ``slugify`` (ASCII-kebab), so a prompt-injected
    traversal slug can't produce a ``..`` path segment. The category is a
    closed-set value (here it falls to the fallback), so neither leg of the
    path is attacker-controlled raw text."""
    raw = (
        '<page category="concept" slug="../../etc/passwd">\n'
        "---\n---\n\n# Evil\n\nbody\n"
        "</page>"
    )
    pages = parse_synthesis_response(raw, source_path="sources/x.md")
    assert len(pages) == 1
    assert ".." not in pages[0].path.split("/")
    assert pages[0].path.startswith("knowledge/concept/")


_TRUNCATED_LONE = """
<page category="entity" slug="spacex">
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
<page category="entity" slug="spacex">
---
tags: [aerospace]
---

# SpaceX

Aerospace firm.
</page>

<page category="entity" slug="tesla">
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


_TWO_COMPLETE_PAGES = """
<page category="entity" slug="spacex">
---
tags: [aerospace]
---

# SpaceX

Aerospace firm founded by Elon Musk.
</page>

<page category="entity" slug="tesla">
---
tags: [automotive]
---

# Tesla

EV maker.
</page>
"""


def test_parse_clean_blocks_with_length_finish_reason_is_partial_retry() -> None:
    """All ``<page>`` blocks closed but the provider reports a budget cutoff
    (OpenAI-style ``finish_reason='length'``): the model complied with the
    "never open a block you cannot finish" prompt and dropped the tail
    cleanly. Without consulting ``finish_reason`` the parser sees no unclosed
    tag, the source is marked done, and the dropped pages are stranded."""
    with pytest.raises(SynthesisPartialError) as excinfo:
        parse_synthesis_response(
            _TWO_COMPLETE_PAGES, source_path="src.md", finish_reason="length"
        )
    pe = excinfo.value
    assert len(pe.pages) == 2  # survivors still recovered for persist
    assert pe.retry is True
    assert any("finish_reason" in e or "truncat" in e for e in pe.errors)


def test_parse_clean_blocks_with_max_tokens_finish_reason_is_partial_retry() -> None:
    """``anthropic_compat`` passes Anthropic's raw ``stop_reason`` through,
    which is ``"max_tokens"`` (NOT ``"length"``). MiniMax-M3 — the synth
    workhorse — runs on that provider, so a check that only knows ``"length"``
    would miss every truncation it produces. Guard the cross-provider set."""
    with pytest.raises(SynthesisPartialError) as excinfo:
        parse_synthesis_response(
            _TWO_COMPLETE_PAGES, source_path="src.md", finish_reason="max_tokens"
        )
    assert excinfo.value.retry is True
    assert len(excinfo.value.pages) == 2


def test_parse_zero_blocks_with_length_finish_reason_raises() -> None:
    """Zero ``<page>`` blocks under a truncation ``finish_reason`` is NOT the
    legal "no page worth writing" signal — it's a budget-starved cutoff (e.g.
    a reasoning model spending the whole budget on hidden thinking). Must
    raise so synth does not mark the source done."""
    with pytest.raises(SynthesisError) as excinfo:
        parse_synthesis_response(
            "the model was still thinking", source_path="x", finish_reason="length"
        )
    assert not isinstance(excinfo.value, SynthesisPartialError)
    assert "truncat" in str(excinfo.value).lower()


@pytest.mark.parametrize("reason", ["stop", "end_turn", None])
def test_parse_zero_blocks_with_clean_finish_reason_returns_empty(
    reason: str | None,
) -> None:
    # A clean stop with no blocks is still the legal zero-page signal.
    assert (
        parse_synthesis_response("no page here", source_path="x", finish_reason=reason)
        == []
    )


@pytest.mark.parametrize("reason", ["stop", "end_turn", None])
def test_parse_clean_blocks_with_clean_finish_reason_returns_pages(
    reason: str | None,
) -> None:
    # Complete blocks + a clean finish_reason → normal success, no exception.
    pages = parse_synthesis_response(
        _TWO_COMPLETE_PAGES, source_path="src.md", finish_reason=reason
    )
    assert len(pages) == 2


@pytest.mark.parametrize("reason", ["Length", " MAX_TOKENS "])
def test_parse_truncation_finish_reason_is_case_and_space_insensitive(
    reason: str,
) -> None:
    with pytest.raises(SynthesisPartialError) as excinfo:
        parse_synthesis_response(
            _TWO_COMPLETE_PAGES, source_path="src.md", finish_reason=reason
        )
    assert excinfo.value.retry is True


_MULTI_PAGE_RESPONSE = """
<page category="entity" slug="elon-musk">
---
tags: [person]
---

# 埃隆·马斯克

Founder of [[SpaceX]] and [[Tesla]].
</page>

<page category="entity" slug="spacex">
---
tags: [company, space]
---

# SpaceX

Aerospace manufacturer led by [[埃隆·马斯克]].
</page>

<page category="concept" slug="falcon-1">
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
    categories = {p.category for p in pages}
    assert categories == {"entity", "concept"}


_PARTIAL_RESPONSE = """
<page category="entity" slug="ok">
---
tags: []
---

# OK Entity

Body.
</page>

<page category="concept" slug="no-title">
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
        '<page category="note" slug="a">\n'
        "---\ntags: []\n---\n\n"
        "no atx title here\n"
        "</page>\n"
        '<page category="note" slug="b">\n'
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
        category="entity",
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
    p1 = _page("B Entity", "# B Entity\n\nb.\n", tags=[], sources=["src.md"])
    p2 = _page("A Entity", "# A Entity\n\na.\n", tags=[], sources=["src.md"])
    p3 = _page("C Entity", "# C Entity\n\nc.\n", tags=[], sources=["src.md"])

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
