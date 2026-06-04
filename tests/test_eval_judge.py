"""Unit tests for the LLM judge — parse + aggregate behaviour."""

from __future__ import annotations

import json

import pytest

from dikw_core.domains.knowledge.page import KnowledgePage, build_page
from dikw_core.eval.judge import (
    JudgeScore,
    JudgeSummary,
    PageJudgeEntry,
    bootstrap_ci,
    judge_synthesis,
    parse_judge_response,
)

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
