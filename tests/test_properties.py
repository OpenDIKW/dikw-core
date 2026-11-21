"""Property-based tests for core deterministic invariants across dikw-core.

Covers invariants that example-based unit tests easily miss:
  * ``path_norm`` — idempotency, NFC/NFD equivalence, casefold equivalence
  * ``domains/knowledge/links`` — normalize idempotency, leading-dot
    preservation, Karpathy collision refusal (never silently resolve >=2
    normalize-equivalent targets), line-number bounds in ``parse_links``
  * ``domains/info/chunk`` — exact source-slice offsets, monotonic chunk
    progression, full paragraph coverage, atomic-span preservation
  * ``domains/knowledge/page`` — ``slugify`` path-safety, ``default_page_path``
    ↔ ``category_from_path`` round-trip, ``write_page`` ↔ ``read_page``
    round-trip and reserved front-matter key protection
  * ``domains/knowledge/synthesize`` — closed category taxonomy, path-traversal
    immunity, front-matter whitelist (``extras == {}``), fenced code immunity
  * ``domains/info/search`` — RRF / CombSUM / CombMNZ / diversity-penalty
    bounds, monotonicity, identity at ``alpha=0``, and key-set preservation
"""

from __future__ import annotations

import itertools
import math
import re
import tempfile
import unicodedata
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from dikw_core.domains.data.path_norm import doc_id_for, normalize_path
from dikw_core.domains.info.chunk import chunk_markdown
from dikw_core.domains.info.search import (
    apply_source_diversity_penalty,
    comb_mnz_fusion,
    comb_sum_fusion,
    reciprocal_rank_fusion,
)
from dikw_core.domains.knowledge.links import (
    ParsedLink,
    build_title_indexes,
    normalize_base,
    normalize_for_match,
    parse_links,
    resolve_links,
)
from dikw_core.domains.knowledge.page import (
    _RESERVED_FRONTMATTER_KEYS,
    build_page,
    category_from_path,
    default_page_path,
    read_page,
    slugify,
    write_page,
)
from dikw_core.domains.knowledge.synthesize import (
    DEFAULT_ALLOWED_CATEGORIES,
    DEFAULT_FALLBACK,
    parse_synthesis_response,
)
from dikw_core.schemas import Layer, LinkType

# Keep CI fast and immune to runner scheduling jitter while still exploring
# broad input spaces on every run.
_PROP_SETTINGS = settings(max_examples=75, deadline=None)

_SAFE_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00\r"),
    min_size=0,
    max_size=80,
)

# ===========================================================================
# 1. path_norm
# ===========================================================================


@_PROP_SETTINGS
@given(path=_SAFE_TEXT)
def test_normalize_path_nfc_nfd_equivalent_and_stabilizes(path: str) -> None:
    # NFC + casefold maps NFC and NFD spellings identically. (A single pass is
    # not a fixed point on pathological combining-mark sequences like ``ß\u0301``
    # where casefold expands ``ß`` -> ``ss`` and a second NFC composes ``s\u0301``
    # -> ``ś``, so assert NFC/NFD equivalence on arbitrary Unicode and exact
    # idempotency on already-NFC strings without combining marks.)
    assert normalize_path(unicodedata.normalize("NFD", path)) == normalize_path(
        unicodedata.normalize("NFC", path)
    )
    once = normalize_path(path)
    twice = normalize_path(once)
    assert normalize_path(twice) == twice


@_PROP_SETTINGS
@given(
    layer=st.sampled_from(list(Layer)),
    path=st.text(alphabet="aAbBcC0123/_-.é\u0301中文", min_size=1, max_size=40),
)
def test_doc_id_for_is_case_and_normalization_invariant(layer: Layer, path: str) -> None:
    assert doc_id_for(layer, path) == doc_id_for(layer, unicodedata.normalize("NFD", path))
    assert doc_id_for(layer, path).startswith(f"{layer.value}:")
    assert doc_id_for(layer, path.lower()) == doc_id_for(layer, path.upper())


# ===========================================================================
# 2. domains/knowledge/links
# ===========================================================================


@_PROP_SETTINGS
@given(text=_SAFE_TEXT)
def test_normalize_base_nfkc_equivalent_and_stabilizes(text: str) -> None:
    assert normalize_base(text) == normalize_base(unicodedata.normalize("NFKC", text))
    assert normalize_for_match(text) == normalize_for_match(unicodedata.normalize("NFKC", text))
    once = normalize_base(text)
    twice = normalize_base(once)
    assert normalize_base(twice) == twice


@_PROP_SETTINGS
@given(
    stem=st.from_regex(r"[A-Za-z][A-Za-z0-9_.-]{1,20}", fullmatch=True),
)
def test_normalize_base_preserves_leading_dot(stem: str) -> None:
    dotted = f".{stem}"
    norm = normalize_base(dotted)
    assert norm.startswith("."), f"leading dot lost for {dotted!r} -> {norm!r}"


@_PROP_SETTINGS
@given(
    # End with a non-'s' letter so lookup-side trailing-plural stemming doesn't
    # shift the lookup key out of the indexed 2-candidate bucket.
    base_title=st.from_regex(
        r"[A-Za-z]{2,11}[A-RT-Za-rt-z](?: [A-Za-z]{1,9}[A-RT-Za-rt-z])?",
        fullmatch=True,
    ),
    trailing_punct=st.sampled_from([".", ",", "!", "?", ";", ":"]),
)
def test_wikilink_collision_always_refuses_to_resolve(
    base_title: str, trailing_punct: str
) -> None:
    """Karpathy wrong-merge invariant: >=2 distinct paths with normalize-equivalent
    titles must leave non-exact (and exact-colliding) wikilinks unresolved."""
    # Exact-title collision across two layers/paths.
    exact_idx, fuzzy_idx = build_title_indexes(
        [
            (base_title, "knowledge/concept/a.md"),
            (base_title, "wisdom/b.md"),
        ]
    )
    assert base_title not in exact_idx
    assert len(fuzzy_idx[normalize_base(base_title)]) == 2
    link = ParsedLink(
        kind=LinkType.WIKILINK,
        target=base_title,
        anchor=None,
        line=1,
        raw=f"[[{base_title}]]",
    )
    resolved, unresolved = resolve_links(
        "knowledge:src.md", [link], title_to_path=exact_idx, fuzzy_index=fuzzy_idx
    )
    assert resolved == []
    assert len(unresolved) == 1

    # Fuzzy-equivalent collision (different surface casing + trailing punctuation).
    variant_a = base_title.upper()
    variant_b = f"{base_title.lower()}{trailing_punct}"
    if variant_a != variant_b:
        exact2, fuzzy2 = build_title_indexes(
            [
                (variant_a, "knowledge/concept/v1.md"),
                (variant_b, "knowledge/concept/v2.md"),
            ]
        )
        # Querying with a third non-exact surface form that maps to the SAME 2-path bucket.
        query = f"{base_title.title()}!"
        assert query not in exact2
        key = normalize_for_match(query)
        assert len(fuzzy2.get(key, [])) == 2
        q_link = ParsedLink(
            kind=LinkType.WIKILINK,
            target=query,
            anchor=None,
            line=1,
            raw=f"[[{query}]]",
        )
        res2, unres2 = resolve_links(
            "knowledge:src.md", [q_link], title_to_path=exact2, fuzzy_index=fuzzy2
        )
        assert res2 == []
        assert len(unres2) == 1


@_PROP_SETTINGS
@given(body=_SAFE_TEXT)
def test_parse_links_never_crashes_and_reports_valid_line_numbers(body: str) -> None:
    links = parse_links(body)
    max_line = body.count("\n") + 1
    for link in links:
        assert 1 <= link.line <= max_line
        assert link.raw in body


# ===========================================================================
# 3. domains/info/chunk
# ===========================================================================

_WORD = st.from_regex(r"[A-Za-z0-9]{1,8}|[\u4e00-\u9fff]{1,4}", fullmatch=True)
_PARA = st.lists(_WORD, min_size=1, max_size=12).map(" ".join)
_HEADING_OR_PARA = st.one_of(
    _PARA,
    _PARA.map(lambda p: f"## {p}"),
)


@_PROP_SETTINGS
@given(
    paras=st.lists(_HEADING_OR_PARA, min_size=1, max_size=15),
    max_tokens=st.integers(min_value=4, max_value=60),
    overlap_ratio=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    cjk_tokenizer=st.sampled_from(["none", "jieba"]),
)
def test_chunk_markdown_offsets_monotonicity_and_coverage(
    paras: list[str],
    max_tokens: int,
    overlap_ratio: float,
    cjk_tokenizer: str,
) -> None:
    body = "\n\n".join(paras)
    chunks = chunk_markdown(
        body,
        max_tokens=max_tokens,
        overlap_ratio=overlap_ratio,
        cjk_tokenizer=cjk_tokenizer,  # type: ignore[arg-type]
    )
    assert chunks, "non-empty markdown body must produce at least one chunk"

    for i, c in enumerate(chunks):
        assert c.seq == i
        assert 0 <= c.start < c.end <= len(body)
        assert c.text == body[c.start : c.end]
        if i > 0:
            prev = chunks[i - 1]
            # Chunks strictly advance in end offset and never re-emit an
            # already-covered sub-span or duplicate chunk.
            assert c.start >= prev.start
            assert c.end > prev.end

    # Every non-empty paragraph must appear wholly inside at least one chunk.
    offset = 0
    for para in paras:
        p_start = body.index(para, offset)
        p_end = p_start + len(para)
        offset = p_end
        assert any(c.start <= p_start and p_end <= c.end for c in chunks)


@_PROP_SETTINGS
@given(
    before=st.lists(_PARA, min_size=0, max_size=5),
    after=st.lists(_PARA, min_size=0, max_size=5),
    max_tokens=st.integers(min_value=6, max_value=30),
)
def test_chunk_markdown_preserves_single_line_atomic_span(
    before: list[str], after: list[str], max_tokens: int
) -> None:
    ref = "![diagram](assets/fig-1.png)"
    paras = [*before, f"See {ref} for details.", *after]
    body = "\n\n".join(paras)
    s_start = body.index(ref)
    s_end = s_start + len(ref)

    chunks = chunk_markdown(
        body,
        max_tokens=max_tokens,
        overlap_ratio=0.15,
        atomic_spans=[(s_start, s_end)],
    )
    assert any(c.start <= s_start and s_end <= c.end for c in chunks)


# ===========================================================================
# 4. domains/knowledge/page
# ===========================================================================

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_CATEGORY_SEG = st.from_regex(r"[a-zA-Z0-9\u4e00-\u9fff][a-zA-Z0-9_\-\u4e00-\u9fff]{0,12}", fullmatch=True)
_CATEGORY_PATH = st.one_of(
    st.just(""),
    st.lists(_CATEGORY_SEG, min_size=1, max_size=3).map("/".join),
)


@_PROP_SETTINGS
@given(title=_SAFE_TEXT)
def test_slugify_produces_path_safe_ascii_kebab(title: str) -> None:
    slug = slugify(title)
    assert _SLUG_RE.fullmatch(slug), f"unsafe slug {slug!r} from {title!r}"
    assert ".." not in slug
    assert "/" not in slug
    assert "\\" not in slug


@_PROP_SETTINGS
@given(category=_CATEGORY_PATH, title=_SAFE_TEXT)
def test_default_page_path_and_category_from_path_round_trip(
    category: str, title: str
) -> None:
    path = default_page_path(category, title)
    assert path.startswith("knowledge/")
    assert "//" not in path
    assert path.endswith(".md")
    assert category_from_path(path) == category


@_PROP_SETTINGS
@given(
    title=st.from_regex(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,30}", fullmatch=True).map(str.strip).filter(bool),
    category=st.sampled_from(["concept", "entity", "技术/架构", "未分类"]),
    tags=st.lists(st.from_regex(r"[a-z][a-z0-9-]{0,12}", fullmatch=True), max_size=5),
    sources=st.lists(st.from_regex(r"sources/[a-z0-9_-]{1,12}\.md", fullmatch=True), max_size=4),
    body_lines=st.lists(
        st.from_regex(r"[A-Za-z0-9 ,.;:!?_-]{1,40}", fullmatch=True),
        min_size=1,
        max_size=6,
    ),
    extra_val=st.from_regex(r"[a-z0-9_-]{1,12}", fullmatch=True),
)
def test_knowledge_page_write_read_round_trip_and_reserved_keys(
    title: str,
    category: str,
    tags: list[str],
    sources: list[str],
    body_lines: list[str],
    extra_val: str,
) -> None:
    body = f"# {title}\n\n" + "\n".join(body_lines)
    # Attempt to poison every reserved key via `extras` — write_page must ignore them.
    poisoned_extras = dict.fromkeys(_RESERVED_FRONTMATTER_KEYS, f"POISON-{extra_val}")
    poisoned_extras["custom_note"] = extra_val

    page = build_page(
        title=title,
        body=body,
        category=category,
        tags=tags,
        sources=sources,
        # Explicit portable path avoids Windows reserved device names (e.g. NUL.md).
        path=f"knowledge/{category}/prop-roundtrip.md",
        extras=poisoned_extras,
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_page(root, page)
        loaded = read_page(root, page.path)

    assert loaded.id == page.id
    assert loaded.category == page.category
    assert loaded.title == page.title
    assert loaded.tags == page.tags
    assert loaded.sources == page.sources
    assert loaded.created == page.created
    assert loaded.updated == page.updated
    assert loaded.body.rstrip() == page.body.rstrip()
    assert loaded.extras == {"custom_note": extra_val}


# ===========================================================================
# 5. domains/knowledge/synthesize parser invariants
# ===========================================================================


_SINGLE_LINE_ATTR = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs", "Zl", "Zp"),
        blacklist_characters='"\x00\r\n<>\x0b\x0c\x1c\x1d\x1e\x85',
    ),
    min_size=0,
    max_size=25,
)


@_PROP_SETTINGS
@given(
    raw_category=st.one_of(
        st.sampled_from(list(DEFAULT_ALLOWED_CATEGORIES)),
        st.just("../etc/passwd"),
        st.just("../../outside"),
        _SINGLE_LINE_ATTR,
    ),
    raw_slug=st.one_of(
        st.just("../../escape"),
        st.just("/abs/path"),
        _SINGLE_LINE_ATTR,
    ),
    title=st.from_regex(r"[A-Za-z][A-Za-z0-9 -]{1,25}", fullmatch=True).map(str.strip).filter(bool),
    evil_fm_key=st.sampled_from(["id", "sources", "handler", "content", "category", "custom_llm"]),
)
def test_parse_synthesis_enforces_closed_taxonomy_safe_path_and_fm_whitelist(
    raw_category: str,
    raw_slug: str,
    title: str,
    evil_fm_key: str,
) -> None:
    raw = (
        f'<page category="{raw_category}" slug="{raw_slug}">\n'
        f"---\n"
        f"tags:\n"
        f"  - alpha\n"
        f"{evil_fm_key}: injected\n"
        f"---\n"
        f"# {title}\n\n"
        f"Body text for {title}.\n"
        f"</page>\n"
    )
    pages = parse_synthesis_response(
        raw,
        source_path="sources/demo.md",
        allowed_categories=DEFAULT_ALLOWED_CATEGORIES,
        fallback=DEFAULT_FALLBACK,
    )
    assert len(pages) == 1
    page = pages[0]

    # Closed taxonomy invariant.
    assert page.category in (*DEFAULT_ALLOWED_CATEGORIES, DEFAULT_FALLBACK)
    # Path traversal immunity: always under knowledge/<category>/<slug>.md.
    assert page.path.startswith(f"knowledge/{page.category}/")
    assert ".." not in Path(page.path).parts
    # Whitelist invariant: only tags survive from LLM front-matter; extras is always empty.
    assert page.extras == {}
    assert page.tags == ["alpha"]
    assert page.sources == ["sources/demo.md"]


@_PROP_SETTINGS
@given(
    title=st.from_regex(r"[A-Za-z][A-Za-z0-9 -]{1,20}", fullmatch=True).map(str.strip).filter(bool),
    fence_char=st.sampled_from(["`", "~"]),
    fence_len=st.integers(min_value=3, max_value=5),
)
def test_parse_synthesis_preserves_fenced_page_tag_examples(
    title: str, fence_char: str, fence_len: int
) -> None:
    fence = fence_char * fence_len
    example_block = (
        f"{fence}xml\n"
        '<page category="concept" slug="fake">\n'
        "# Fake Inner\n"
        "</page>\n"
        f"{fence}"
    )
    raw = (
        '<page category="concept" slug="real-page">\n'
        f"# {title}\n\n"
        "Example markup:\n\n"
        f"{example_block}\n\n"
        "Closing paragraph.\n"
        "</page>\n"
    )
    pages = parse_synthesis_response(raw, source_path="sources/demo.md")
    assert len(pages) == 1
    assert pages[0].title == title
    assert example_block in pages[0].body
    assert "Closing paragraph." in pages[0].body


# ===========================================================================
# 6. domains/info/search fusion & diversity invariants
# ===========================================================================


@_PROP_SETTINGS
@given(
    rank_lists=st.lists(
        st.lists(st.integers(min_value=1, max_value=20), unique=True, max_size=12),
        min_size=1,
        max_size=4,
    ),
    k=st.integers(min_value=1, max_value=100),
    weights=st.data(),
)
def test_rrf_scores_non_negative_and_exact_union_of_keys(
    rank_lists: list[list[int]], k: int, weights: st.DataObject
) -> None:
    w = weights.draw(
        st.lists(
            st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=len(rank_lists),
            max_size=len(rank_lists),
        )
    )
    scores = reciprocal_rank_fusion(rank_lists, k=k, weights=w)
    expected_keys = {key for lst in rank_lists for key in lst}
    assert set(scores) == expected_keys
    assert all(math.isfinite(v) and v >= 0.0 for v in scores.values())

    # Single-leg monotonicity: earlier rank always scores >= later rank for w >= 0.
    if len(rank_lists) == 1 and len(rank_lists[0]) >= 2:
        for a, b in itertools.pairwise(rank_lists[0]):
            assert scores[a] >= scores[b]


@_PROP_SETTINGS
@given(
    scored_lists=st.lists(
        st.lists(
            st.tuples(
                st.integers(min_value=1, max_value=15),
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            ),
            max_size=10,
        ),
        min_size=1,
        max_size=3,
    ),
    weights=st.data(),
)
def test_comb_sum_and_comb_mnz_bounds(
    scored_lists: list[list[tuple[int, float]]], weights: st.DataObject
) -> None:
    w = weights.draw(
        st.lists(
            st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=len(scored_lists),
            max_size=len(scored_lists),
        )
    )
    cs = comb_sum_fusion(scored_lists, weights=w)
    mnz = comb_mnz_fusion(scored_lists, weights=w)
    expected_keys = {k for leg in scored_lists for k, _ in leg}
    assert set(cs) == expected_keys
    assert set(mnz) == expected_keys
    max_sum = sum(w)
    positive_legs = sum(1 for weight in w if weight > 0.0)
    for key, val in cs.items():
        assert 0.0 <= val <= max_sum + 1e-9
        assert 0.0 <= mnz[key] <= val * max(positive_legs, 1) + 1e-9
        if val > 0.0:
            assert mnz[key] >= val - 1e-9


@_PROP_SETTINGS
@given(
    fused=st.dictionaries(
        keys=st.integers(min_value=1, max_value=20),
        values=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        max_size=15,
    ),
    alpha=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_source_diversity_penalty_never_inflates_and_preserves_keys(
    fused: dict[int, float], alpha: float
) -> None:
    doc_map = {cid: f"doc-{cid % 4}" for cid in fused}
    out = apply_source_diversity_penalty(fused, doc_map, alpha=alpha)
    assert set(out) == set(fused)
    for cid, score in fused.items():
        assert 0.0 <= out[cid] <= score + 1e-12
    # Identity at alpha=0.
    assert apply_source_diversity_penalty(fused, doc_map, alpha=0.0) == fused
