"""Metrics module lives in src, not tests — test the public import path."""

from __future__ import annotations

import math

import pytest

from dikw_core.domains.knowledge.page import KnowledgePage, build_page
from dikw_core.eval.fake_embedder import FakeEmbeddings
from dikw_core.eval.metrics import (
    GroundingClaim,
    atomicity_score,
    classify_lang,
    compute_grounding_cosines,
    duplicate_ratio_max,
    expected_coverage,
    fact_grounding_ratio,
    hit_at_k,
    language_fidelity,
    mean_hit_at_k,
    mean_ndcg_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    page_density,
    recall_at_k,
    reciprocal_rank,
    reduce_grounding_ratio,
    split_claims,
    wikilink_resolved_ratio,
)
from dikw_core.schemas import ChunkRecord


def test_hit_at_k_matches_any_expected() -> None:
    assert hit_at_k(["a", "b", "c"], ["b"], 3) == 1.0
    assert hit_at_k(["a", "b", "c"], ["x", "a"], 1) == 1.0
    assert hit_at_k(["a", "b", "c"], ["x"], 10) == 0.0


def test_hit_at_k_respects_k_cutoff() -> None:
    # "c" is position 3; k=2 excludes it
    assert hit_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert hit_at_k(["a", "b", "c"], ["c"], 3) == 1.0


def test_hit_at_k_zero_or_negative_k_is_zero() -> None:
    assert hit_at_k(["a"], ["a"], 0) == 0.0
    assert hit_at_k(["a"], ["a"], -1) == 0.0


def test_reciprocal_rank_uses_first_match() -> None:
    assert reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
    assert reciprocal_rank(["a", "b", "c"], ["c", "b"]) == 0.5  # b is earlier
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
    assert reciprocal_rank(["a", "b", "c"], ["x"]) == 0.0


def test_mean_aggregates_preserve_semantics() -> None:
    # two queries, one full hit@1, one miss at k=10
    results: list[tuple[list[str], list[str]]] = [
        (["a", "b"], ["a"]),
        (["c", "d"], ["x"]),
    ]
    assert mean_hit_at_k(results, 1) == 0.5
    assert mean_reciprocal_rank(results) == 0.5  # 1.0 + 0.0 / 2


def test_empty_results_aggregate_to_zero() -> None:
    assert mean_hit_at_k([], 10) == 0.0
    assert mean_reciprocal_rank([]) == 0.0


# ---- nDCG@k -----------------------------------------------------------------


def test_ndcg_perfect_ranking_is_one() -> None:
    # All relevant docs ranked first → DCG == IDCG.
    assert ndcg_at_k(["a", "b", "c", "x"], ["a", "b", "c"], 3) == pytest.approx(1.0)


def test_ndcg_no_match_is_zero() -> None:
    assert ndcg_at_k(["x", "y", "z"], ["a"], 3) == 0.0


def test_ndcg_single_hit_at_first_position() -> None:
    # rel = [1]; DCG = 1/log2(2) = 1.0
    # IDCG (1 relevant doc, capped at k=3) = 1/log2(2) = 1.0
    assert ndcg_at_k(["a", "x", "y"], ["a"], 3) == pytest.approx(1.0)


def test_ndcg_single_hit_at_second_position() -> None:
    # rel = [0, 1]; DCG = 1/log2(3); IDCG = 1/log2(2) = 1.0
    expected = 1.0 / math.log2(3)
    assert ndcg_at_k(["x", "a", "y"], ["a"], 3) == pytest.approx(expected)


def test_ndcg_partial_match_with_one_at_top_one_lower() -> None:
    # ranked = [a, x, b], expected = {a, b, c}; k = 3.
    # rel = [1, 0, 1]; DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG: 3 relevant docs, k=3 → 1/log2(2) + 1/log2(3) + 1/log2(4)
    dcg = 1.0 + 1.0 / math.log2(4)
    idcg = 1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    assert ndcg_at_k(["a", "x", "b"], ["a", "b", "c"], 3) == pytest.approx(dcg / idcg)


def test_ndcg_caps_idcg_at_k() -> None:
    # 5 relevant docs but k=2 → IDCG only counts 2 of them.
    # ranked all hits → DCG = 1/log2(2) + 1/log2(3); IDCG identical.
    assert ndcg_at_k(["a", "b"], ["a", "b", "c", "d", "e"], 2) == pytest.approx(1.0)


def test_ndcg_zero_or_negative_k_is_zero() -> None:
    assert ndcg_at_k(["a"], ["a"], 0) == 0.0
    assert ndcg_at_k(["a"], ["a"], -1) == 0.0


def test_ndcg_empty_expected_is_zero() -> None:
    assert ndcg_at_k(["a", "b"], [], 3) == 0.0


# ---- Recall@k ---------------------------------------------------------------


def test_recall_full_coverage_is_one() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0


def test_recall_partial_coverage() -> None:
    # 1 of 2 expected found in top-2
    assert recall_at_k(["a", "x"], ["a", "b"], 2) == 0.5


def test_recall_respects_k_cutoff() -> None:
    # "b" is at position 3; k=2 excludes it → only "a" hit
    assert recall_at_k(["a", "x", "b"], ["a", "b"], 2) == 0.5
    assert recall_at_k(["a", "x", "b"], ["a", "b"], 3) == 1.0


def test_recall_no_match_is_zero() -> None:
    assert recall_at_k(["x", "y"], ["a", "b"], 5) == 0.0


def test_recall_zero_or_negative_k_is_zero() -> None:
    assert recall_at_k(["a"], ["a"], 0) == 0.0
    assert recall_at_k(["a"], ["a"], -1) == 0.0


def test_recall_empty_expected_is_zero() -> None:
    assert recall_at_k(["a", "b"], [], 3) == 0.0


# ---- mean_* aggregations ----------------------------------------------------


def test_mean_ndcg_averages_per_query() -> None:
    results: list[tuple[list[str], list[str]]] = [
        (["a", "x"], ["a"]),  # nDCG@2 = 1.0
        (["x", "y"], ["a"]),  # nDCG@2 = 0.0
    ]
    assert mean_ndcg_at_k(results, 2) == pytest.approx(0.5)


def test_mean_recall_averages_per_query() -> None:
    results: list[tuple[list[str], list[str]]] = [
        (["a", "b"], ["a", "b"]),  # recall@2 = 1.0
        (["x", "y"], ["a", "b"]),  # recall@2 = 0.0
    ]
    assert mean_recall_at_k(results, 2) == pytest.approx(0.5)


def test_mean_aggregates_empty_inputs_are_zero() -> None:
    assert mean_ndcg_at_k([], 10) == 0.0
    assert mean_recall_at_k([], 10) == 0.0


# ============================================================================
# K-layer synth quality metrics
# ============================================================================


def _page(
    title: str,
    body: str,
    tags: list[str] | None = None,
    category: str = "concept",
) -> KnowledgePage:
    return build_page(
        title=title,
        body=body,
        category=category,
        tags=tags or [],
        sources=[],
        path=None,
        extras={},
    )


def _chunk(doc_id: str, seq: int, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=None,
        doc_id=doc_id,
        seq=seq,
        start=0,
        end=len(text),
        text=text,
    )


# ---- expected_coverage ------------------------------------------------------


def test_expected_coverage_all_exact_match() -> None:
    cov = expected_coverage(
        page_titles=["Alpha", "Beta", "Gamma"],
        expected_titles=["Alpha", "Beta", "Gamma"],
    )
    assert cov == 1.0


def test_expected_coverage_fuzzy_plural_match() -> None:
    cov = expected_coverage(
        page_titles=["Neural Network"],
        expected_titles=["Neural Networks"],
    )
    assert cov == 1.0


def test_expected_coverage_fuzzy_punctuation_and_case() -> None:
    cov = expected_coverage(
        page_titles=["elon musk"],
        expected_titles=["Elon Musk."],
    )
    assert cov == 1.0


def test_expected_coverage_partial() -> None:
    cov = expected_coverage(
        page_titles=["Alpha", "Beta"],
        expected_titles=["Alpha", "Beta", "Gamma", "Delta"],
    )
    assert cov == 0.5


def test_expected_coverage_empty_expected_returns_one() -> None:
    # Vacuously satisfied — runner can skip; metric defaults to 1.0 so the
    # threshold never triggers when expected.yaml is absent.
    assert expected_coverage(page_titles=["A"], expected_titles=[]) == 1.0


def test_expected_coverage_empty_pages_zero() -> None:
    assert expected_coverage(page_titles=[], expected_titles=["A"]) == 0.0


# ---- wikilink_resolved_ratio ------------------------------------------------


def test_wikilink_resolved_ratio_no_links_returns_one() -> None:
    assert wikilink_resolved_ratio(total=0, unresolved=0) == 1.0


def test_wikilink_resolved_ratio_all_resolved() -> None:
    assert wikilink_resolved_ratio(total=10, unresolved=0) == 1.0


def test_wikilink_resolved_ratio_partial() -> None:
    assert wikilink_resolved_ratio(total=10, unresolved=2) == 0.8


def test_wikilink_resolved_ratio_all_unresolved() -> None:
    assert wikilink_resolved_ratio(total=10, unresolved=10) == 0.0


# ---- atomicity_score --------------------------------------------------------


def test_atomicity_score_all_atomic() -> None:
    pages = [
        _page("A", "# A\n\nshort\n"),
        _page("B", "# B\n\nshort\n"),
    ]
    assert atomicity_score(pages) == 1.0


def test_atomicity_score_half_non_atomic() -> None:
    pages = [
        _page("A", "# A\n\nshort\n"),
        _page("B", "# B\n\n" + ("filler " * 1000)),  # body > 2500 chars
    ]
    assert atomicity_score(pages) == 0.5


def test_atomicity_score_empty_returns_one() -> None:
    # 0/0: no failures observed, define as perfect score
    assert atomicity_score([]) == 1.0


# ---- language_fidelity ------------------------------------------------------


def test_language_fidelity_all_english_match() -> None:
    pages_with_sources = [
        (_page("A", "# A\n\nPlain English page body content.\n"),
         "Plain English source body content."),
    ]
    assert language_fidelity(pages_with_sources) == 1.0


def test_language_fidelity_all_cjk_match() -> None:
    pages_with_sources = [
        (_page("中文", "# 中文\n\n这是一段中文内容。\n"),
         "这是一段中文源文档。"),
    ]
    assert language_fidelity(pages_with_sources) == 1.0


def test_language_fidelity_mismatch_cjk_source_english_page() -> None:
    pages_with_sources = [
        (_page("A", "# A\n\nThis page is in English even though source was CJK.\n"),
         "中文源文档,内容应保留中文。"),
    ]
    assert language_fidelity(pages_with_sources) == 0.0


def test_language_fidelity_empty_returns_one() -> None:
    assert language_fidelity([]) == 1.0


# ---- page_density -----------------------------------------------------------


def test_page_density_typical() -> None:
    assert page_density(n_pages=5, n_chunks=20) == 0.25


def test_page_density_zero_chunks_returns_zero() -> None:
    # div-by-zero guard
    assert page_density(n_pages=0, n_chunks=0) == 0.0
    assert page_density(n_pages=3, n_chunks=0) == 0.0


# ---- fact_grounding_ratio (async, embedding-driven) -------------------------


@pytest.mark.asyncio
async def test_fact_grounding_ratio_verbatim_match_is_one() -> None:
    """Page claim sentence is verbatim equal to a chunk → cosine = 1.0 → grounded."""
    embedder = FakeEmbeddings()
    page = _page("A", "# A\n\nThe sky is blue today.\n")
    chunks_by_source = {
        "knowledge/sources/a.md": [
            _chunk("knowledge/sources/a.md", 0, "The sky is blue today."),
        ],
    }
    pages_with_sources = [(page, "knowledge/sources/a.md")]
    score = await fact_grounding_ratio(
        pages_with_sources=pages_with_sources,
        chunks_by_source=chunks_by_source,
        embedder=embedder,
        embedding_model="fake",
        tau=0.5,
    )
    assert score == 1.0


@pytest.mark.asyncio
async def test_fact_grounding_ratio_disjoint_vocab_is_zero() -> None:
    """Page claim has no overlapping tokens with any chunk → cosine 0 → ungrounded."""
    embedder = FakeEmbeddings()
    page = _page("A", "# A\n\nDeep ocean trenches are mysterious.\n")
    chunks_by_source = {
        "knowledge/sources/a.md": [
            _chunk("knowledge/sources/a.md", 0, "Pizza toppings vary widely."),
        ],
    }
    pages_with_sources = [(page, "knowledge/sources/a.md")]
    score = await fact_grounding_ratio(
        pages_with_sources=pages_with_sources,
        chunks_by_source=chunks_by_source,
        embedder=embedder,
        embedding_model="fake",
        tau=0.5,
    )
    assert score == 0.0


@pytest.mark.asyncio
async def test_compute_grounding_cosines_emits_one_entry_per_claim() -> None:
    """The new split — embed once, reduce at any tau — keeps page/source
    paths attached so a sweep / hand-label workflow can inspect rows."""
    embedder = FakeEmbeddings()
    page = _page(
        "A",
        "# A\n\nFirst claim here is grounded. "
        "Second claim covers another topic.\n",
    )
    chunks_by_source = {
        "knowledge/sources/a.md": [
            _chunk(
                "knowledge/sources/a.md", 0, "First claim here is grounded."
            ),
            _chunk("knowledge/sources/a.md", 1, "Unrelated content."),
        ],
    }
    pages_with_sources = [(page, "knowledge/sources/a.md")]
    claims = await compute_grounding_cosines(
        pages_with_sources=pages_with_sources,
        chunks_by_source=chunks_by_source,
        embedder=embedder,
        embedding_model="fake",
    )
    assert len(claims) == 2
    assert all(isinstance(c, GroundingClaim) for c in claims)
    assert {c.claim for c in claims} == {
        "First claim here is grounded.",
        "Second claim covers another topic.",
    }
    assert all(c.page_path == page.path for c in claims)
    # tau-sweep: each tau bucket is a deterministic reduction of the cosines.
    at_low = reduce_grounding_ratio(
        claims, pages_with_sources=pages_with_sources, tau=0.0
    )
    at_high = reduce_grounding_ratio(
        claims, pages_with_sources=pages_with_sources, tau=0.99
    )
    assert at_low >= at_high


def test_reduce_grounding_ratio_pages_with_no_claims_score_one() -> None:
    """Same vacuous-1.0 floor as the metric: a page that emitted no claim
    sentences shouldn't drag the aggregate down."""
    page = _page("A", "# A\n\n[[Other]]\n")
    pages_with_sources = [(page, "knowledge/sources/a.md")]
    # No GroundingClaim entries for page A (mirroring what
    # compute_grounding_cosines does for claim-less pages).
    ratio = reduce_grounding_ratio(
        [], pages_with_sources=pages_with_sources, tau=0.5
    )
    assert ratio == 1.0


@pytest.mark.asyncio
async def test_fact_grounding_ratio_empty_pages_returns_one() -> None:
    score = await fact_grounding_ratio(
        pages_with_sources=[],
        chunks_by_source={},
        embedder=FakeEmbeddings(),
        embedding_model="fake",
        tau=0.5,
    )
    assert score == 1.0


@pytest.mark.asyncio
async def test_fact_grounding_ratio_skips_page_with_no_claims() -> None:
    """Page body contains only heading + wikilinks → no claim sentences →
    contributes neutrally (treated as 1.0, not 0.0)."""
    embedder = FakeEmbeddings()
    page = _page("A", "# A\n\n[[Other]]\n")
    chunks_by_source = {
        "knowledge/sources/a.md": [
            _chunk("knowledge/sources/a.md", 0, "Unrelated text."),
        ],
    }
    pages_with_sources = [(page, "knowledge/sources/a.md")]
    score = await fact_grounding_ratio(
        pages_with_sources=pages_with_sources,
        chunks_by_source=chunks_by_source,
        embedder=embedder,
        embedding_model="fake",
        tau=0.5,
    )
    assert score == 1.0


# ---- duplicate_ratio_max (async, embedding-driven) --------------------------


@pytest.mark.asyncio
async def test_duplicate_ratio_max_no_duplicates() -> None:
    embedder = FakeEmbeddings()
    pages = [
        _page("Alpha", "# Alpha\n\nDistinct topic about alpha.\n"),
        _page("Beta", "# Beta\n\nDifferent vocabulary on beta.\n"),
        _page("Gamma", "# Gamma\n\nUnrelated topic gamma.\n"),
    ]
    ratio = await duplicate_ratio_max(
        pages=pages, embedder=embedder, embedding_model="fake", tau=0.99,
    )
    assert ratio == 0.0


@pytest.mark.asyncio
async def test_duplicate_ratio_max_identical_body_flagged() -> None:
    embedder = FakeEmbeddings()
    # Two pages with byte-identical body content (titles in metadata differ
    # but never reach the embedder — duplicate detection looks at body).
    body = "# Shared Heading\n\nSame body verbatim about networks.\n"
    pages = [
        _page("Page Alpha", body),
        _page("Page Beta", body),
    ]
    # 1 pair, cosine = 1.0 ≥ tau → ratio = 1/1 = 1.0
    ratio = await duplicate_ratio_max(
        pages=pages, embedder=embedder, embedding_model="fake", tau=0.85,
    )
    assert ratio == 1.0


@pytest.mark.asyncio
async def test_duplicate_ratio_max_one_page_no_pairs() -> None:
    embedder = FakeEmbeddings()
    pages = [_page("A", "# A\n\nLonely.\n")]
    ratio = await duplicate_ratio_max(
        pages=pages, embedder=embedder, embedding_model="fake", tau=0.85,
    )
    assert ratio == 0.0


@pytest.mark.asyncio
async def test_duplicate_ratio_max_empty_pages() -> None:
    embedder = FakeEmbeddings()
    ratio = await duplicate_ratio_max(
        pages=[], embedder=embedder, embedding_model="fake", tau=0.85,
    )
    assert ratio == 0.0


@pytest.mark.asyncio
async def test_duplicate_ratio_max_skips_empty_body_pages() -> None:
    """Pages with empty / whitespace-only bodies are degenerate synth
    output, not duplicates — most embedding APIs 400 on empty input so
    we filter them out before embedding. Verified against a tracking
    embedder that records every text it saw."""

    class TrackingEmbedder(FakeEmbeddings):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[str] = []

        async def embed(  # type: ignore[override]
            self, texts: list[str], *, model: str
        ) -> list[list[float]]:
            self.seen.extend(texts)
            return await super().embed(texts, model=model)

    embedder = TrackingEmbedder()
    pages = [
        _page("A", "# A\n\nNon-empty body.\n"),
        _page("B", ""),  # truly empty
        _page("C", "   \n\t"),  # whitespace only
        _page("D", "# D\n\nAnother non-empty body.\n"),
    ]
    ratio = await duplicate_ratio_max(
        pages=pages, embedder=embedder, embedding_model="fake", tau=0.85,
    )
    # Only A and D were embedded — one pair, distinct bodies → 0.0.
    assert ratio == 0.0
    assert len(embedder.seen) == 2
    assert all("body" in t.lower() for t in embedder.seen)


# ---- split_claims ----------------------------------------------------------


def test_split_claims_basic_sentences() -> None:
    """split_claims requires ≥3 en words per fragment (filters out
    sentence-splitter junk like ``"."`` and ``"embed("``)."""
    body = (
        "# Title\n\nFirst claim here. Second claim follows.\n\n"
        "Third claim wraps the body.\n"
    )
    claims = split_claims(body)
    assert len(claims) == 3
    assert "First claim here" in claims[0]
    assert "Second claim follows" in claims[1]
    assert "Third claim wraps" in claims[2]


def test_split_claims_drops_punctuation_only_fragments() -> None:
    """Fragments under the en-word / cjk-char threshold are dropped at
    the splitter so downstream embedders don't see ``"."`` or
    ``"embed("`` (real garbage seen in the 2026-05-13 tau sweep)."""
    body = (
        "# Title\n\nThe real claim runs for several words.\n"
        "embed(. )\n"
        ".\n"
    )
    claims = split_claims(body)
    # Only the real claim survives.
    assert claims == ["The real claim runs for several words."]


def test_split_claims_strips_wikilinks() -> None:
    body = "# T\n\nThis mentions [[Alice]] and also [[Bob|Robert]] briefly.\n"
    claims = split_claims(body)
    text = " ".join(claims)
    # Wikilink markup gone; target text may stay but [[ ]] should not
    assert "[[" not in text
    assert "]]" not in text


def test_split_claims_strips_headings() -> None:
    body = "# Big Title Heading\n\n## Subhead Heading\n\nA real claim sentence here.\n"
    claims = split_claims(body)
    text = " ".join(claims)
    assert "Big Title Heading" not in text
    assert "Subhead Heading" not in text
    assert "A real claim sentence here" in text


def test_split_claims_strips_fenced_code() -> None:
    body = (
        "# T\n\nA prose claim sentence.\n\n"
        "```python\nignored_code_line()\n```\n\n"
        "Another claim sentence.\n"
    )
    claims = split_claims(body)
    text = " ".join(claims)
    assert "ignored_code_line" not in text
    assert "prose claim sentence" in text


def test_split_claims_chinese_period() -> None:
    body = "# T\n\n这是第一个声明。这是第二个声明。\n"
    claims = split_claims(body)
    assert len(claims) == 2


def test_split_claims_empty_body() -> None:
    assert split_claims("") == []


def test_split_claims_only_heading() -> None:
    assert split_claims("# Just a title\n") == []


# ---- classify_lang ---------------------------------------------------------


def test_classify_lang_pure_english() -> None:
    assert classify_lang("This is plain English without any non-ASCII.") == "en"


def test_classify_lang_pure_cjk() -> None:
    assert classify_lang("这是一段全部由中文字符组成的文本内容") == "cjk"


def test_classify_lang_mostly_cjk() -> None:
    text = "这是一段以中文为主的内容,只是夹杂了 a few English words 用于测试。"
    assert classify_lang(text) == "cjk"


def test_classify_lang_mostly_english() -> None:
    text = "Mostly English content with just 中文 a couple of characters."
    assert classify_lang(text) == "en"


def test_classify_lang_empty_is_other() -> None:
    assert classify_lang("") == "other"
