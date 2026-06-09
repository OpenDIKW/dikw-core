"""Unit tests for the LLM judge — parse + aggregate behaviour."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from dikw_core.domains.knowledge.page import KnowledgePage, build_page
from dikw_core.eval.judge import (
    CategoryOption,
    CategorySummary,
    CategoryVerdict,
    ClaimEvidence,
    EntailmentSummary,
    EntailmentVerdict,
    JudgeScore,
    JudgeSummary,
    PageJudgeEntry,
    WikilinkUnit,
    bootstrap_ci,
    claim_evidence_from_grounding,
    judge_category,
    judge_entailment,
    judge_synthesis,
    judge_wikilinks,
    parse_category_verdict,
    parse_entailment_verdict,
    parse_judge_response,
    recommended_judge_sample,
    wikilink_units_from_pages,
)
from dikw_core.eval.metrics import GroundingClaim
from dikw_core.schemas import ChunkRecord, LinkRecord, LinkType

from .fakes import FakeLLM


def _valid_payload(**overrides: object) -> str:
    base: dict[str, object] = {
        "grounding": 4,
        "atomicity": 5,
        "completeness": 3,
        "clarity": 5,
        "rationale": "ok",
    }
    base.update(overrides)
    return json.dumps(base)


def _page(title: str, body: str = "# T\n\nbody.\n", source: str = "sources/a.md") -> KnowledgePage:
    return build_page(
        title=title,
        body=body,
        category="concept",
        tags=[],
        sources=[source],
        path=None,
        extras={},
    )


# ---- parse_judge_response ---------------------------------------------------


def test_parse_judge_response_valid_json() -> None:
    score = parse_judge_response(_valid_payload())
    assert isinstance(score, JudgeScore)
    assert score.grounding == 4
    assert score.atomicity == 5
    assert score.completeness == 3
    assert score.clarity == 5
    assert score.rationale == "ok"


def test_parse_judge_response_strips_fenced_json() -> None:
    text = "```json\n" + _valid_payload() + "\n```"
    score = parse_judge_response(text)
    assert score is not None
    assert score.grounding == 4


def test_parse_judge_response_strips_unfenced_code_block() -> None:
    text = "```\n" + _valid_payload() + "\n```"
    score = parse_judge_response(text)
    assert score is not None


def test_parse_judge_response_malformed_json_returns_none() -> None:
    assert parse_judge_response("not json") is None
    assert parse_judge_response("{broken") is None


def test_parse_judge_response_array_at_top_returns_none() -> None:
    """``JudgeScore`` expects a JSON object, not array."""
    assert parse_judge_response("[1, 2, 3]") is None


def test_parse_judge_response_missing_field_returns_none() -> None:
    text = json.dumps(
        {"grounding": 4, "atomicity": 5, "completeness": 3, "rationale": "x"}
    )  # missing clarity
    assert parse_judge_response(text) is None


def test_parse_judge_response_out_of_range_returns_none() -> None:
    assert parse_judge_response(_valid_payload(grounding=6)) is None
    assert parse_judge_response(_valid_payload(grounding=-1)) is None


def test_parse_judge_response_float_score_rejected() -> None:
    """0-5 scale is integer-only by contract; floats must be rejected."""
    assert parse_judge_response(_valid_payload(grounding=3.7)) is None


def test_parse_judge_response_bool_score_rejected() -> None:
    """``True`` is technically an int in Python — reject explicitly."""
    assert parse_judge_response(_valid_payload(grounding=True)) is None


def test_parse_judge_response_string_score_rejected() -> None:
    assert parse_judge_response(_valid_payload(grounding="four")) is None


# ---- judge_synthesis aggregation -------------------------------------------


@pytest.mark.asyncio
async def test_judge_synthesis_aggregates_means() -> None:
    pages = [_page("A"), _page("B")]
    sources = {"sources/a.md": "source text A"}
    llm = FakeLLM(
        responses=[
            _valid_payload(grounding=4, atomicity=5, completeness=4, clarity=5),
            _valid_payload(grounding=2, atomicity=3, completeness=2, clarity=4),
        ]
    )
    summary = await judge_synthesis(
        pages, sources=sources, llm=llm, model="x"
    )
    assert isinstance(summary, JudgeSummary)
    assert summary.n_judged == 2
    assert summary.n_errors == 0
    assert summary.mean_grounding == 3.0
    assert summary.mean_atomicity == 4.0
    assert summary.mean_completeness == 3.0
    assert summary.mean_clarity == 4.5


@pytest.mark.asyncio
async def test_judge_synthesis_counts_parse_failures() -> None:
    pages = [_page("A"), _page("B"), _page("C")]
    sources = {"sources/a.md": "src"}
    llm = FakeLLM(
        responses=[
            _valid_payload(grounding=4, atomicity=4, completeness=4, clarity=4),
            "not json",
            _valid_payload(grounding=2, atomicity=2, completeness=2, clarity=2),
        ]
    )
    summary = await judge_synthesis(pages, sources=sources, llm=llm, model="x")
    assert summary.n_judged == 2
    assert summary.n_errors == 1
    assert summary.mean_grounding == 3.0


@pytest.mark.asyncio
async def test_judge_synthesis_all_failures_returns_zero_means() -> None:
    pages = [_page("A"), _page("B")]
    llm = FakeLLM(responses=["bogus", "also bogus"])
    summary = await judge_synthesis(
        pages, sources={}, llm=llm, model="x"
    )
    assert summary.n_judged == 0
    assert summary.n_errors == 2
    assert summary.mean_grounding == 0.0
    assert summary.mean_clarity == 0.0
    assert summary.per_page == []


@pytest.mark.asyncio
async def test_judge_synthesis_sample_caps_at_n() -> None:
    """``sample`` larger than the page count is a no-op (no truncation)."""
    pages = [_page(f"P{i}") for i in range(3)]
    llm = FakeLLM(responses=[_valid_payload() for _ in range(3)])
    summary = await judge_synthesis(
        pages, sources={}, llm=llm, model="x", sample=10
    )
    assert summary.n_judged == 3


@pytest.mark.asyncio
async def test_judge_synthesis_sample_subsamples_deterministically() -> None:
    """A seeded RNG picks the same subset on repeated runs of same dataset."""
    pages = [_page(f"P{i}") for i in range(5)]
    llm1 = FakeLLM(responses=[_valid_payload() for _ in range(5)])
    llm2 = FakeLLM(responses=[_valid_payload() for _ in range(5)])
    s1 = await judge_synthesis(
        pages, sources={}, llm=llm1, model="x", sample=3, seed="abc"
    )
    s2 = await judge_synthesis(
        pages, sources={}, llm=llm2, model="x", sample=3, seed="abc"
    )
    assert s1.n_judged == 3
    assert [e.path for e in s1.per_page] == [e.path for e in s2.per_page]


@pytest.mark.asyncio
async def test_judge_synthesis_per_page_entries_carry_path_and_score() -> None:
    pages = [_page("Only")]
    sources = {"sources/a.md": "src"}
    llm = FakeLLM(responses=[_valid_payload(grounding=5)])
    summary = await judge_synthesis(pages, sources=sources, llm=llm, model="x")
    assert len(summary.per_page) == 1
    entry = summary.per_page[0]
    assert isinstance(entry, PageJudgeEntry)
    assert entry.path == pages[0].path
    assert entry.score.grounding == 5


@pytest.mark.asyncio
async def test_judge_synthesis_handles_llm_exception() -> None:
    """A per-page LLM exception is counted, not raised — other pages
    keep going."""

    class BoomLLM(FakeLLM):
        async def complete(self, **kwargs: object) -> object:  # type: ignore[override]
            raise RuntimeError("simulated provider failure")

    pages = [_page("A")]
    summary = await judge_synthesis(
        pages, sources={}, llm=BoomLLM(), model="x"
    )
    assert summary.n_judged == 0
    assert summary.n_errors == 1


@pytest.mark.asyncio
async def test_judge_synthesis_empty_pages_returns_empty_summary() -> None:
    summary = await judge_synthesis(
        [], sources={}, llm=FakeLLM(), model="x"
    )
    assert summary.n_judged == 0
    assert summary.n_errors == 0
    assert summary.per_page == []


@pytest.mark.asyncio
async def test_judge_synthesis_default_max_tokens_is_reasoning_safe() -> None:
    # A reasoning LLM (e.g. MiniMax-M3) spends a hidden thinking trace that
    # counts against max_tokens before the JSON score; a small cap truncates
    # the page-judge response to empty and every call logs a parse error.
    # The default must leave headroom for that trace — see judge.py.
    pages = [_page("A")]
    llm = FakeLLM(response_text=_valid_payload())
    await judge_synthesis(pages, sources={"sources/a.md": "src"}, llm=llm, model="m")
    assert llm.last_max_tokens == 4096


# ---- bootstrap_ci -----------------------------------------------------------


def test_bootstrap_ci_empty_is_zero() -> None:
    assert bootstrap_ci([], seed="x") == (0.0, 0.0)


def test_bootstrap_ci_single_value_has_zero_width() -> None:
    assert bootstrap_ci([3.0], seed="x") == (3.0, 3.0)


def test_bootstrap_ci_deterministic_same_seed() -> None:
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert bootstrap_ci(vals, seed="abc") == bootstrap_ci(vals, seed="abc")


def test_bootstrap_ci_low_le_high_and_within_value_range() -> None:
    vals = [0.0, 1.0, 2.0, 5.0, 5.0]
    lo, hi = bootstrap_ci(vals, seed="abc")
    assert lo <= hi
    # Resample means can never escape [min, max] of the observations.
    assert min(vals) <= lo <= max(vals)
    assert min(vals) <= hi <= max(vals)


def test_bootstrap_ci_brackets_the_mean() -> None:
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean = sum(vals) / len(vals)
    lo, hi = bootstrap_ci(vals, seed="abc")
    assert lo <= mean <= hi


def test_bootstrap_ci_width_shrinks_with_more_samples() -> None:
    """A larger sample of the same {0, 5} mix gives a tighter CI on the
    mean — the bootstrap SE shrinks ~1/sqrt(n)."""
    small = [0.0, 5.0] * 2  # n=4
    large = [0.0, 5.0] * 25  # n=50, same mean
    lo_s, hi_s = bootstrap_ci(small, seed="w")
    lo_l, hi_l = bootstrap_ci(large, seed="w")
    assert (hi_l - lo_l) < (hi_s - lo_s)


@pytest.mark.asyncio
async def test_judge_synthesis_populates_dimension_cis() -> None:
    """Each dimension's CI brackets its mean and stays on the 0-5 scale."""
    pages = [_page("A"), _page("B"), _page("C")]
    sources = {"sources/a.md": "src"}
    llm = FakeLLM(
        responses=[
            _valid_payload(grounding=5, atomicity=4, completeness=3, clarity=5),
            _valid_payload(grounding=3, atomicity=4, completeness=2, clarity=4),
            _valid_payload(grounding=1, atomicity=4, completeness=4, clarity=3),
        ]
    )
    summary = await judge_synthesis(pages, sources=sources, llm=llm, model="x")
    assert summary.n_judged == 3
    lo, hi = summary.ci_grounding
    assert lo <= summary.mean_grounding <= hi
    assert 0.0 <= lo <= hi <= 5.0
    # A dimension with identical scores across pages has a degenerate CI
    # pinned to that value (every resample mean is 4.0).
    assert summary.ci_atomicity == (4.0, 4.0)


# ---- recommended_judge_sample (power analysis) ------------------------------


def test_recommended_judge_sample_default_margin_is_25() -> None:
    # 1.96*0.5/0.2 = 4.9; 4.9**2 = 24.01 -> ceil 25.
    assert recommended_judge_sample() == 25
    assert recommended_judge_sample(0.2) == 25


def test_recommended_judge_sample_clamps_tight_margin_to_max() -> None:
    # 1.96*0.5/0.1 = 9.8; 9.8**2 ~ 96 -> clamped to the 50 ceiling.
    assert recommended_judge_sample(0.1) == 50


def test_recommended_judge_sample_clamps_wide_margin_to_min() -> None:
    # 1.96*0.5/0.5 = 1.96; 1.96**2 ~ 3.84 -> ceil 4 -> clamped up to the 5 floor.
    assert recommended_judge_sample(0.5) == 5


def test_recommended_judge_sample_monotonic_in_margin() -> None:
    # A tighter target margin never decreases the required sample (within clamps).
    samples = [recommended_judge_sample(m) for m in (0.5, 0.3, 0.2, 0.15, 0.1)]
    assert samples == sorted(samples)


def test_recommended_judge_sample_nonpositive_margin_returns_max() -> None:
    assert recommended_judge_sample(0.0) == 50
    assert recommended_judge_sample(-1.0) == 50


# ---- fact-entailment judge --------------------------------------------------


def _verdict(v: str, rationale: str = "ok") -> str:
    return json.dumps({"verdict": v, "rationale": rationale})


def _ce(
    claim: str,
    evidence: str | None,
    *,
    page: str = "knowledge/concept/a.md",
    source: str = "sources/a.md",
) -> ClaimEvidence:
    return ClaimEvidence(
        page_path=page, source_path=source, claim=claim, evidence=evidence
    )


# ---- parse_entailment_verdict ----------------------------------------------


def test_parse_entailment_verdict_maps_to_scores() -> None:
    for token, score in (("yes", 1.0), ("partial", 0.5), ("no", 0.0)):
        v = parse_entailment_verdict(_verdict(token))
        assert isinstance(v, EntailmentVerdict)
        assert v.verdict == token
        assert v.score == score


def test_parse_entailment_verdict_strips_fences() -> None:
    v1 = parse_entailment_verdict("```json\n" + _verdict("yes") + "\n```")
    v2 = parse_entailment_verdict("```\n" + _verdict("no") + "\n```")
    assert v1 is not None and v1.score == 1.0
    assert v2 is not None and v2.score == 0.0


def test_parse_entailment_verdict_case_and_whitespace_lenient() -> None:
    """LLM casing varies — ``YES`` / `` Partial `` normalise; the token set
    stays strict."""
    v1 = parse_entailment_verdict(_verdict("YES"))
    v2 = parse_entailment_verdict(_verdict(" Partial "))
    assert v1 is not None and v1.score == 1.0
    assert v2 is not None and v2.score == 0.5


def test_parse_entailment_verdict_unknown_token_returns_none() -> None:
    for bad in ("maybe", "true", "1", ""):
        assert parse_entailment_verdict(_verdict(bad)) is None


def test_parse_entailment_verdict_malformed_and_non_object_returns_none() -> None:
    assert parse_entailment_verdict("not json") is None
    assert parse_entailment_verdict("{broken") is None
    assert parse_entailment_verdict('["yes"]') is None  # top-level array
    # object without a verdict key
    assert parse_entailment_verdict(json.dumps({"rationale": "x"})) is None
    # verdict present but not a string
    assert parse_entailment_verdict(json.dumps({"verdict": 1})) is None


def test_parse_entailment_verdict_rationale_optional() -> None:
    """Score only needs the verdict — a thin ``{"verdict": "yes"}`` parses."""
    v = parse_entailment_verdict(json.dumps({"verdict": "yes"}))
    assert v is not None
    assert v.score == 1.0
    assert v.rationale == ""


def test_entailment_verdict_score_mapping() -> None:
    assert EntailmentVerdict(verdict="yes").score == 1.0
    assert EntailmentVerdict(verdict="partial").score == 0.5
    assert EntailmentVerdict(verdict="no").score == 0.0


# ---- claim_evidence_from_grounding -----------------------------------------


def test_claim_evidence_from_grounding_resolves_chunk_text() -> None:
    gcs = [
        GroundingClaim(
            page_path="knowledge/c/p.md",
            source_path="sources/a.md",
            claim="alpha is great",
            max_cosine=0.9,
            best_chunk_seq=1,
        )
    ]
    chunks = {
        "sources/a.md": [
            ChunkRecord(doc_id="d", seq=0, start=0, end=4, text="zero"),
            ChunkRecord(
                doc_id="d", seq=1, start=5, end=26, text="alpha is great indeed"
            ),
        ]
    }
    pairs = claim_evidence_from_grounding(gcs, chunks)
    assert len(pairs) == 1
    assert pairs[0].evidence == "alpha is great indeed"
    assert pairs[0].claim == "alpha is great"
    assert pairs[0].page_path == "knowledge/c/p.md"


def test_claim_evidence_from_grounding_none_when_no_chunk() -> None:
    """A claim whose source had no embeddable chunk (best_chunk_seq=None) gets
    evidence=None — counted as unverifiable, not dropped."""
    gcs = [
        GroundingClaim(
            page_path="knowledge/c/p.md",
            source_path="sources/a.md",
            claim="ungrounded",
            max_cosine=float("-inf"),
            best_chunk_seq=None,
        )
    ]
    pairs = claim_evidence_from_grounding(gcs, {"sources/a.md": []})
    assert len(pairs) == 1
    assert pairs[0].evidence is None


# ---- judge_entailment aggregation ------------------------------------------


@pytest.mark.asyncio
async def test_judge_entailment_averages_verdict_scores() -> None:
    pairs = [_ce("c1", "e1"), _ce("c2", "e2"), _ce("c3", "e3")]
    llm = FakeLLM(
        responses=[_verdict("yes"), _verdict("partial"), _verdict("no")]
    )
    summary = await judge_entailment(pairs, llm=llm, model="x")
    assert isinstance(summary, EntailmentSummary)
    assert summary.n_judged == 3
    assert summary.n_errors == 0
    assert summary.n_no_evidence == 0
    assert summary.ratio == pytest.approx((1.0 + 0.5 + 0.0) / 3)


@pytest.mark.asyncio
async def test_judge_entailment_empty_input_returns_zero_summary() -> None:
    summary = await judge_entailment([], llm=FakeLLM(), model="x")
    assert summary.n_judged == 0
    assert summary.ratio == 0.0
    assert summary.ci == (0.0, 0.0)


@pytest.mark.asyncio
async def test_judge_entailment_all_evidence_none_scores_zero() -> None:
    pairs = [_ce("c1", None), _ce("c2", None)]
    llm = FakeLLM()
    summary = await judge_entailment(pairs, llm=llm, model="x")
    assert summary.n_judged == 2
    assert summary.n_no_evidence == 2
    assert summary.ratio == 0.0
    assert llm.call_count == 0  # no evidence → no LLM call


@pytest.mark.asyncio
async def test_judge_entailment_parse_failures_counted_not_raised() -> None:
    pairs = [_ce("c1", "e1"), _ce("c2", "e2"), _ce("c3", "e3")]
    llm = FakeLLM(responses=[_verdict("yes"), "garbage", _verdict("yes")])
    summary = await judge_entailment(pairs, llm=llm, model="x")
    assert summary.n_judged == 2  # the unparseable one is excluded
    assert summary.n_errors == 1
    assert summary.ratio == 1.0


@pytest.mark.asyncio
async def test_judge_entailment_llm_exception_counted_not_raised() -> None:
    class BoomLLM(FakeLLM):
        async def complete(self, **kwargs: object) -> object:  # type: ignore[override]
            raise RuntimeError("simulated provider failure")

    summary = await judge_entailment([_ce("c", "e")], llm=BoomLLM(), model="x")
    assert summary.n_judged == 0
    assert summary.n_errors == 1
    assert summary.ratio == 0.0


@pytest.mark.asyncio
async def test_judge_entailment_sample_caps_claim_count_seeded_stable() -> None:
    pairs = [_ce(f"c{i}", f"e{i}") for i in range(6)]
    llm1 = FakeLLM(response_text=_verdict("yes"))
    llm2 = FakeLLM(response_text=_verdict("yes"))
    s1 = await judge_entailment(pairs, llm=llm1, model="x", sample=3, seed="d")
    s2 = await judge_entailment(pairs, llm=llm2, model="x", sample=3, seed="d")
    assert s1.n_judged == 3
    assert llm1.call_count == 3  # only the sampled claims hit the LLM
    # Same seed → same subset → identical aggregate.
    assert s1.ratio == s2.ratio == 1.0


@pytest.mark.asyncio
async def test_judge_entailment_sample_larger_than_n_is_noop() -> None:
    pairs = [_ce("c1", "e1"), _ce("c2", "e2")]
    llm = FakeLLM(response_text=_verdict("yes"))
    summary = await judge_entailment(pairs, llm=llm, model="x", sample=10)
    assert summary.n_judged == 2


@pytest.mark.asyncio
async def test_judge_entailment_caches_identical_pairs() -> None:
    """Two identical (claim, evidence) pairs are judged once; the verdict is
    reused so a repeated claim doesn't pay double LLM spend."""
    pairs = [_ce("same", "eq"), _ce("same", "eq")]
    llm = FakeLLM(response_text=_verdict("yes"))
    summary = await judge_entailment(pairs, llm=llm, model="x")
    assert summary.n_judged == 2
    assert summary.ratio == 1.0
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_judge_entailment_calls_llm_at_temperature_zero() -> None:
    captured: dict[str, object] = {}

    class CaptureLLM(FakeLLM):
        async def complete(self, **kwargs: object) -> object:  # type: ignore[override]
            captured.update(kwargs)
            return await super().complete(**kwargs)  # type: ignore[arg-type]

    llm = CaptureLLM(response_text=_verdict("yes"))
    await judge_entailment([_ce("c", "e")], llm=llm, model="m")
    assert captured["temperature"] == 0.0
    assert "entailment" in str(captured["system"]).lower()


@pytest.mark.asyncio
async def test_judge_entailment_default_max_tokens_is_reasoning_safe() -> None:
    # Reasoning LLMs (e.g. MiniMax-M3) spend a hidden thinking trace that
    # counts against max_tokens before emitting the JSON verdict; the old
    # 256 cap truncated dense-claim judgments to empty (~75% parse errors on
    # MiniMax-M3). The default must leave room for that trace — see judge.py.
    llm = FakeLLM(response_text=_verdict("yes"))
    await judge_entailment([_ce("c", "e")], llm=llm, model="m")
    assert llm.last_max_tokens == 4096


@pytest.mark.asyncio
async def test_judge_entailment_mixed_evidence_and_no_evidence() -> None:
    """No-evidence claims count as 0.0 in the ratio alongside judged ones."""
    pairs = [_ce("c1", "e1"), _ce("c2", None), _ce("c3", "e3")]
    llm = FakeLLM(responses=[_verdict("yes"), _verdict("yes")])
    summary = await judge_entailment(pairs, llm=llm, model="x")
    assert summary.n_judged == 3
    assert summary.n_no_evidence == 1
    assert summary.n_errors == 0
    # two yes (1.0 each) + one no-evidence (0.0) → 2/3
    assert summary.ratio == pytest.approx(2.0 / 3.0)
    assert llm.call_count == 2


# ---- category-correctness judge --------------------------------------------


_CAT_OPTS = [
    CategoryOption(path="entity", desc="A named thing."),
    CategoryOption(path="concept", desc="An idea or pattern."),
    CategoryOption(path="note", desc="An observation."),
    CategoryOption(path="未分类", desc="None of the above."),
]


def _catv(chosen: str, also_fits: str | None = None, rationale: str = "ok") -> str:
    payload: dict[str, object] = {"chosen": chosen, "rationale": rationale}
    if also_fits is not None:
        payload["also_fits"] = also_fits
    return json.dumps(payload)


def _cpage(category: str, *, title: str = "A") -> KnowledgePage:
    return build_page(
        title=title,
        body="# T\n\nbody.\n",
        category=category,
        tags=[],
        sources=["sources/a.md"],
        path=None,
        extras={},
    )


# ---- parse_category_verdict -------------------------------------------------


def test_parse_category_verdict_valid() -> None:
    v = parse_category_verdict(_catv("concept"), allowed=frozenset({"concept", "note"}))
    assert isinstance(v, CategoryVerdict)
    assert v.chosen == "concept"
    assert v.also_fits is None
    assert v.rationale == "ok"


def test_parse_category_verdict_keeps_allowed_also_fits() -> None:
    v = parse_category_verdict(
        _catv("concept", also_fits="note"), allowed=frozenset({"concept", "note"})
    )
    assert v is not None
    assert v.also_fits == "note"


def test_parse_category_verdict_strips_fences() -> None:
    text = "```json\n" + _catv("entity") + "\n```"
    v = parse_category_verdict(text, allowed=frozenset({"entity"}))
    assert v is not None
    assert v.chosen == "entity"


def test_parse_category_verdict_strips_whitespace_tokens() -> None:
    # An LLM's stray surrounding space ("concept ") is whitespace, not a
    # different category — strip before the closed-set check so a valid choice
    # doesn't inflate n_errors.
    v = parse_category_verdict(
        _catv("concept ", also_fits=" note"),
        allowed=frozenset({"concept", "note"}),
    )
    assert v is not None
    assert v.chosen == "concept"
    assert v.also_fits == "note"


def test_parse_category_verdict_chosen_not_in_allowed_returns_none() -> None:
    # Closed-set discipline: an invented category is a parse failure, never a
    # silent re-file (mirrors synth's own refusal).
    assert parse_category_verdict(_catv("invented"), allowed=frozenset({"concept"})) is None


def test_parse_category_verdict_also_fits_not_in_allowed_coerced_to_none() -> None:
    # A hallucinated secondary can never match the page's real category, so drop
    # it (score-neutral) rather than reject the whole verdict.
    v = parse_category_verdict(
        _catv("concept", also_fits="invented"), allowed=frozenset({"concept"})
    )
    assert v is not None
    assert v.chosen == "concept"
    assert v.also_fits is None


def test_parse_category_verdict_also_fits_wrong_type_coerced_to_none() -> None:
    text = json.dumps({"chosen": "concept", "also_fits": 5, "rationale": "x"})
    v = parse_category_verdict(text, allowed=frozenset({"concept"}))
    assert v is not None
    assert v.also_fits is None


def test_parse_category_verdict_malformed_and_non_object_returns_none() -> None:
    allowed = frozenset({"concept"})
    assert parse_category_verdict("not json", allowed=allowed) is None
    assert parse_category_verdict("[1, 2]", allowed=allowed) is None
    assert parse_category_verdict(json.dumps({"rationale": "x"}), allowed=allowed) is None


def test_parse_category_verdict_rationale_optional() -> None:
    text = json.dumps({"chosen": "concept"})
    v = parse_category_verdict(text, allowed=frozenset({"concept"}))
    assert v is not None
    assert v.rationale == ""


# ---- judge_category ---------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_category_exact_match_scores_one() -> None:
    llm = FakeLLM(response_text=_catv("concept"))
    summary = await judge_category([_cpage("concept")], options=_CAT_OPTS, llm=llm, model="x")
    assert isinstance(summary, CategorySummary)
    assert summary.n_judged == 1
    assert summary.n_errors == 0
    assert summary.ratio == 1.0


@pytest.mark.asyncio
async def test_judge_category_also_fits_match_scores_half() -> None:
    # synth filed it under ``note``; the judge's top pick is ``concept`` but it
    # names ``note`` as an equally-valid co-equal → 0.5.
    llm = FakeLLM(response_text=_catv("concept", also_fits="note"))
    summary = await judge_category([_cpage("note")], options=_CAT_OPTS, llm=llm, model="x")
    assert summary.ratio == 0.5


@pytest.mark.asyncio
async def test_judge_category_mismatch_scores_zero() -> None:
    llm = FakeLLM(response_text=_catv("concept"))
    summary = await judge_category([_cpage("entity")], options=_CAT_OPTS, llm=llm, model="x")
    assert summary.n_judged == 1
    assert summary.ratio == 0.0


@pytest.mark.asyncio
async def test_judge_category_averages_scores() -> None:
    pages = [_cpage("concept", title="A"), _cpage("entity", title="B"), _cpage("note", title="C")]
    # exact (1.0), mismatch (0.0), also_fits (0.5) → mean 0.5
    llm = FakeLLM(
        responses=[_catv("concept"), _catv("concept"), _catv("entity", also_fits="note")]
    )
    summary = await judge_category(pages, options=_CAT_OPTS, llm=llm, model="x")
    assert summary.n_judged == 3
    assert summary.ratio == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_judge_category_empty_input_returns_zero_summary() -> None:
    summary = await judge_category([], options=_CAT_OPTS, llm=FakeLLM(), model="x")
    assert summary.n_judged == 0
    assert summary.n_errors == 0
    assert summary.ratio == 0.0


@pytest.mark.asyncio
async def test_judge_category_invented_choice_counted_as_error() -> None:
    # A category not in the closed option set is a parse failure → n_errors.
    llm = FakeLLM(response_text=_catv("invented"))
    summary = await judge_category([_cpage("concept")], options=_CAT_OPTS, llm=llm, model="x")
    assert summary.n_judged == 0
    assert summary.n_errors == 1


@pytest.mark.asyncio
async def test_judge_category_parse_failure_counted_not_raised() -> None:
    pages = [_cpage("concept", title="A"), _cpage("concept", title="B")]
    llm = FakeLLM(responses=[_catv("concept"), "garbage"])
    summary = await judge_category(pages, options=_CAT_OPTS, llm=llm, model="x")
    assert summary.n_judged == 1
    assert summary.n_errors == 1
    assert summary.ratio == 1.0


@pytest.mark.asyncio
async def test_judge_category_llm_exception_counted_not_raised() -> None:
    class BoomLLM(FakeLLM):
        async def complete(self, **kwargs: object) -> object:  # type: ignore[override]
            raise RuntimeError("boom")

    summary = await judge_category([_cpage("concept")], options=_CAT_OPTS, llm=BoomLLM(), model="x")
    assert summary.n_judged == 0
    assert summary.n_errors == 1


@pytest.mark.asyncio
async def test_judge_category_sample_caps_pages_seeded_stable() -> None:
    pages = [_cpage("concept", title=f"P{i}") for i in range(6)]
    llm1 = FakeLLM(response_text=_catv("concept"))
    llm2 = FakeLLM(response_text=_catv("concept"))
    s1 = await judge_category(pages, options=_CAT_OPTS, llm=llm1, model="x", sample=3, seed="d")
    s2 = await judge_category(pages, options=_CAT_OPTS, llm=llm2, model="x", sample=3, seed="d")
    assert s1.n_judged == 3
    assert llm1.call_count == 3
    assert s1.ratio == s2.ratio


@pytest.mark.asyncio
async def test_judge_category_default_max_tokens_is_reasoning_safe() -> None:
    llm = FakeLLM(response_text=_catv("concept"))
    await judge_category([_cpage("concept")], options=_CAT_OPTS, llm=llm, model="m")
    assert llm.last_max_tokens == 4096


# ---- wikilink-correctness judge ---------------------------------------------


def _linked_page(title: str, body: str, *, path: str) -> KnowledgePage:
    p = build_page(
        title=title,
        body=body,
        category="concept",
        tags=[],
        sources=["sources/a.md"],
        path=None,
        extras={},
    )
    # build_page derives a path; the tests pin explicit paths so the
    # links_by_src_path keys and dst_path values line up deterministically.
    return replace(p, path=path)


def _wlink(dst_path: str, line: int, *, kind: LinkType = LinkType.WIKILINK) -> LinkRecord:
    return LinkRecord(
        src_doc_id="doc-src", dst_path=dst_path, link_type=kind, anchor=None, line=line
    )


def _two_linked_pages() -> tuple[list[KnowledgePage], dict[str, list[LinkRecord]]]:
    alpha = _linked_page(
        "Alpha", "# Alpha\n\nAlpha is a concept.\n", path="knowledge/concept/alpha.md"
    )
    beta = _linked_page(
        "Beta",
        "# Beta\n\nBeta builds on [[Alpha]] heavily.\nMore beta detail.\n",
        path="knowledge/concept/beta.md",
    )
    links = {
        "knowledge/concept/beta.md": [_wlink("knowledge/concept/alpha.md", line=3)]
    }
    return [alpha, beta], links


# ---- wikilink_units_from_pages ----------------------------------------------


def test_wikilink_units_basic_unit_built_with_context() -> None:
    pages, links = _two_linked_pages()
    units = wikilink_units_from_pages(pages, links)
    assert len(units) == 1
    u = units[0]
    assert u.src_path == "knowledge/concept/beta.md"
    assert u.src_title == "Beta"
    assert u.target_path == "knowledge/concept/alpha.md"
    assert u.target_title == "Alpha"
    assert u.target_category == "concept"
    assert "Alpha is a concept." in u.target_body
    # Context carries the link line plus its neighbours.
    assert "[[Alpha]]" in u.context
    assert "More beta detail." in u.context


def test_wikilink_units_dangling_target_skipped() -> None:
    pages, _ = _two_linked_pages()
    links = {"knowledge/concept/beta.md": [_wlink("knowledge/concept/ghost.md", line=3)]}
    assert wikilink_units_from_pages(pages, links) == []


def test_wikilink_units_non_wikilink_records_skipped() -> None:
    pages, _ = _two_linked_pages()
    links = {
        "knowledge/concept/beta.md": [
            _wlink("knowledge/concept/alpha.md", line=3, kind=LinkType.MARKDOWN),
            _wlink("https://example.com", line=3, kind=LinkType.URL),
        ]
    }
    assert wikilink_units_from_pages(pages, links) == []


def test_wikilink_units_self_link_skipped() -> None:
    pages, _ = _two_linked_pages()
    links = {"knowledge/concept/beta.md": [_wlink("knowledge/concept/beta.md", line=3)]}
    assert wikilink_units_from_pages(pages, links) == []


def test_wikilink_units_target_body_capped() -> None:
    pages, links = _two_linked_pages()
    long_alpha = replace(pages[0], body="# Alpha\n\n" + "x" * 5000)
    units = wikilink_units_from_pages([long_alpha, pages[1]], links, target_body_cap=100)
    assert len(units) == 1
    assert len(units[0].target_body) <= 100


def test_wikilink_units_out_of_range_line_clamped_not_crash() -> None:
    # A defensive guard: a stale line number (body edited between persist and
    # eval) must clamp to the body's bounds, never raise.
    pages, _ = _two_linked_pages()
    links = {"knowledge/concept/beta.md": [_wlink("knowledge/concept/alpha.md", line=999)]}
    units = wikilink_units_from_pages(pages, links)
    assert len(units) == 1
    assert units[0].context  # non-empty: clamped to the last lines


def test_wikilink_units_src_page_missing_from_pages_skipped() -> None:
    # A links key whose src page isn't in the page set (deactivated doc) is
    # skipped rather than KeyError-ing.
    pages, _ = _two_linked_pages()
    links = {"knowledge/concept/ghost.md": [_wlink("knowledge/concept/alpha.md", line=1)]}
    assert wikilink_units_from_pages(pages, links) == []


# ---- judge_wikilinks aggregation ---------------------------------------------


def _unit(i: int = 0) -> WikilinkUnit:
    return WikilinkUnit(
        src_path=f"knowledge/concept/src{i}.md",
        src_title=f"Src {i}",
        context=f"Src {i} builds on [[Alpha]].",
        target_path="knowledge/concept/alpha.md",
        target_title="Alpha",
        target_category="concept",
        target_body="# Alpha\n\nAlpha is a concept.",
    )


@pytest.mark.asyncio
async def test_judge_wikilinks_averages_verdict_scores() -> None:
    units = [_unit(0), _unit(1), _unit(2)]
    llm = FakeLLM(responses=[_verdict("yes"), _verdict("partial"), _verdict("no")])
    summary = await judge_wikilinks(units, llm=llm, model="x")
    assert summary.n_judged == 3
    assert summary.n_errors == 0
    assert summary.ratio == pytest.approx(0.5)  # (1.0 + 0.5 + 0.0) / 3


@pytest.mark.asyncio
async def test_judge_wikilinks_empty_input_returns_zero_summary() -> None:
    summary = await judge_wikilinks([], llm=FakeLLM(), model="x")
    assert summary.n_judged == 0
    assert summary.n_errors == 0
    assert summary.ratio == 0.0
    assert summary.ci == (0.0, 0.0)


@pytest.mark.asyncio
async def test_judge_wikilinks_parse_failure_counted_not_raised() -> None:
    llm = FakeLLM(responses=[_verdict("yes"), "garbage"])
    summary = await judge_wikilinks([_unit(0), _unit(1)], llm=llm, model="x")
    assert summary.n_judged == 1
    assert summary.n_errors == 1
    assert summary.ratio == 1.0


@pytest.mark.asyncio
async def test_judge_wikilinks_llm_exception_counted_not_raised() -> None:
    class BoomLLM(FakeLLM):
        async def complete(self, **kwargs: object) -> object:  # type: ignore[override]
            raise RuntimeError("boom")

    summary = await judge_wikilinks([_unit(0)], llm=BoomLLM(), model="x")
    assert summary.n_judged == 0
    assert summary.n_errors == 1


@pytest.mark.asyncio
async def test_judge_wikilinks_sample_caps_units_seeded_stable() -> None:
    units = [_unit(i) for i in range(6)]
    llm1 = FakeLLM(response_text=_verdict("yes"))
    llm2 = FakeLLM(response_text=_verdict("yes"))
    s1 = await judge_wikilinks(units, llm=llm1, model="x", sample=3, seed="d")
    s2 = await judge_wikilinks(units, llm=llm2, model="x", sample=3, seed="d")
    assert s1.n_judged == 3
    assert llm1.call_count == 3
    assert s1.ratio == s2.ratio


@pytest.mark.asyncio
async def test_judge_wikilinks_calls_llm_at_temperature_zero() -> None:
    captured: dict[str, object] = {}

    class CaptureLLM(FakeLLM):
        async def complete(self, **kwargs: object) -> object:  # type: ignore[override]
            captured.update(kwargs)
            return await super().complete(**kwargs)  # type: ignore[arg-type]

    llm = CaptureLLM(response_text=_verdict("yes"))
    await judge_wikilinks([_unit(0)], llm=llm, model="m")
    assert captured["temperature"] == 0.0
    assert "wikilink judge" in str(captured["system"]).lower()


@pytest.mark.asyncio
async def test_judge_wikilinks_default_max_tokens_is_reasoning_safe() -> None:
    llm = FakeLLM(response_text=_verdict("yes"))
    await judge_wikilinks([_unit(0)], llm=llm, model="m")
    assert llm.last_max_tokens == 4096


@pytest.mark.asyncio
async def test_judge_wikilinks_prompt_carries_context_and_target() -> None:
    captured: dict[str, object] = {}

    class CaptureLLM(FakeLLM):
        async def complete(self, **kwargs: object) -> object:  # type: ignore[override]
            captured.update(kwargs)
            return await super().complete(**kwargs)  # type: ignore[arg-type]

    await judge_wikilinks([_unit(0)], llm=CaptureLLM(response_text=_verdict("yes")), model="m")
    user = str(captured["user"])
    assert "[[Alpha]]" in user  # the context with the link as written
    assert "Alpha is a concept." in user  # the target body
