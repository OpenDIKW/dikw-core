"""Unit tests for the A/B experiment harness — `evals/tools/ab_experiment.py`.

Covers the pure statistics layer (Welch t-test against exact analytic
references, Cohen's d, direction-aware ship gate), the run comparison, and
the JSON persistence round-trip. The live ``collect_synth_eval_runs`` driver
(real LLM) is exercised manually, not here.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# Import the tool by adding the repo root to sys.path — `evals/` is developer
# tooling, not an installed package (same pattern as test_sweep_rrf.py).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.tools.ab_experiment import (  # noqa: E402
    ExperimentResult,
    _json_safe,
    append_runs,
    betai,
    cohens_d,
    collect_runs,
    compare_metric,
    compare_runs,
    flatten_synth_report,
    load_runs,
    mean_std,
    metric_direction,
    student_t_two_sided_p,
    welch_t_test,
    write_result,
)

# ---- mean_std ---------------------------------------------------------------


def test_mean_std_basic() -> None:
    m, s = mean_std([2.0, 4.0, 6.0])
    assert m == 4.0
    assert s == pytest.approx(2.0)  # ddof=1 sample std


def test_mean_std_single_and_empty() -> None:
    assert mean_std([5.0]) == (5.0, 0.0)
    assert mean_std([]) == (0.0, 0.0)


# ---- incomplete beta / Student-t (exact analytic references) ----------------


def test_betai_symmetry_at_half() -> None:
    # I_{0.5}(a, a) == 0.5 for any a by symmetry.
    assert betai(0.5, 0.5, 0.5) == pytest.approx(0.5, abs=1e-9)
    assert betai(2.0, 2.0, 0.5) == pytest.approx(0.5, abs=1e-9)


def test_student_t_cauchy_reference() -> None:
    # df=1 is Cauchy: two-sided P(|T|>1) = 1 - (2/pi)*arctan(1) = 0.5.
    assert student_t_two_sided_p(1.0, 1.0) == pytest.approx(0.5, abs=1e-9)


def test_student_t_df2_reference() -> None:
    # df=2, t=sqrt(2): closed form two-sided p = 1 - 1/sqrt(2).
    expected = 1.0 - 1.0 / math.sqrt(2.0)
    assert student_t_two_sided_p(math.sqrt(2.0), 2.0) == pytest.approx(
        expected, abs=1e-9
    )


def test_student_t_zero_and_degenerate() -> None:
    assert student_t_two_sided_p(0.0, 5.0) == 1.0
    assert student_t_two_sided_p(2.0, 0.0) == 1.0  # df<=0 → no test
    assert student_t_two_sided_p(math.inf, 5.0) == 0.0


# ---- welch_t_test -----------------------------------------------------------


def test_welch_clear_separation() -> None:
    t, df, p = welch_t_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert t == pytest.approx(3.6742, abs=1e-3)
    assert df == pytest.approx(4.0, abs=1e-6)
    assert p < 0.05


def test_welch_insufficient_data() -> None:
    assert welch_t_test([1.0], [4.0, 5.0]) == (0.0, 0.0, 1.0)


def test_welch_zero_variance_equal_means() -> None:
    t, _df, p = welch_t_test([2.0, 2.0, 2.0], [2.0, 2.0, 2.0])
    assert t == 0.0
    assert p == 1.0


def test_welch_zero_variance_deterministic_separation() -> None:
    t, _df, p = welch_t_test([0.0, 0.0, 0.0], [5.0, 5.0, 5.0])
    assert t == math.inf
    assert p == 0.0


# ---- cohens_d ---------------------------------------------------------------


def test_cohens_d_pooled() -> None:
    assert cohens_d([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(3.0)


def test_cohens_d_undefined_cases() -> None:
    assert cohens_d([1.0], [2.0, 3.0]) is None  # n<2
    assert cohens_d([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == 0.0  # equal constants
    assert cohens_d([0.0, 0.0, 0.0], [5.0, 5.0, 5.0]) is None  # zero var, gap


# ---- metric_direction -------------------------------------------------------


def test_metric_direction() -> None:
    assert metric_direction("synth/fact_grounding_ratio") == "min"
    assert metric_direction("synth/duplicate_ratio_max") == "max"
    assert metric_direction("duplicate_ratio_max") == "max"


# ---- compare_metric ship gate ----------------------------------------------


def test_compare_metric_min_metric_ships_on_real_gain() -> None:
    c = compare_metric(
        "synth/fact_grounding_ratio",
        [0.60, 0.61, 0.59],
        [0.75, 0.76, 0.74],
        p_max=0.05,
        effect_min=0.10,
    )
    assert c.direction == "min"
    assert c.improvement == pytest.approx(0.15, abs=1e-6)
    assert c.significant
    assert c.ships
    assert not c.regressed


def test_compare_metric_max_metric_improvement_is_a_decrease() -> None:
    # duplicate_ratio_max: lower is better, so a drop is a positive improvement.
    c = compare_metric(
        "synth/duplicate_ratio_max",
        [0.20, 0.21, 0.19],
        [0.05, 0.06, 0.04],
        p_max=0.05,
        effect_min=0.10,
    )
    assert c.direction == "max"
    assert c.delta == pytest.approx(-0.15, abs=1e-6)
    assert c.improvement == pytest.approx(0.15, abs=1e-6)
    assert c.ships


def test_compare_metric_noise_does_not_ship() -> None:
    c = compare_metric(
        "synth/fact_grounding_ratio",
        [0.60, 0.62, 0.58],
        [0.61, 0.63, 0.59],
        p_max=0.05,
        effect_min=0.10,
    )
    # ~0.01 gain is below the 0.10 noise floor → must not ship.
    assert not c.ships
    assert not c.regressed


def test_compare_metric_regression_flagged() -> None:
    c = compare_metric(
        "synth/fact_grounding_ratio",
        [0.75, 0.76, 0.74],
        [0.60, 0.61, 0.59],
        p_max=0.05,
        effect_min=0.10,
    )
    assert c.improvement == pytest.approx(-0.15, abs=1e-6)
    assert c.regressed
    assert not c.ships


def test_compare_metric_max_metric_increase_regresses_not_ships() -> None:
    """A `_max` (lower-is-better) metric that INCREASES is a regression, never
    a ship — the bug the direction convention exists to prevent. Pins both
    the new informational over-generation metrics (slug_merge_ratio_max,
    fallback_ratio_max) via their shared `_max` classification."""
    for name in ("synth/slug_merge_ratio_max", "synth/fallback_ratio_max"):
        c = compare_metric(
            name,
            [0.05, 0.06, 0.04],  # baseline: low (good)
            [0.30, 0.31, 0.29],  # intervention: high (worse)
            p_max=0.05,
            effect_min=0.10,
        )
        assert c.direction == "max", name
        assert c.delta == pytest.approx(0.25, abs=1e-6), name
        # Higher is worse → a +0.25 rise is a NEGATIVE improvement.
        assert c.improvement == pytest.approx(-0.25, abs=1e-6), name
        assert c.regressed, name
        assert not c.ships, name


def test_compare_metric_max_metric_decrease_ships() -> None:
    """The mirror: a `_max` metric that DROPS (less over-generation) ships."""
    c = compare_metric(
        "synth/slug_merge_ratio_max",
        [0.30, 0.31, 0.29],
        [0.05, 0.06, 0.04],
        p_max=0.05,
        effect_min=0.10,
    )
    assert c.improvement == pytest.approx(0.25, abs=1e-6)
    assert c.ships
    assert not c.regressed


# ---- compare_runs -----------------------------------------------------------


def test_compare_runs_only_shared_metrics() -> None:
    baseline = [
        {"synth/a": 0.6, "synth/only_base": 1.0},
        {"synth/a": 0.61, "synth/only_base": 1.0},
    ]
    intervention = [
        {"synth/a": 0.75, "synth/only_int": 2.0},
        {"synth/a": 0.76, "synth/only_int": 2.0},
    ]
    result = compare_runs(baseline, intervention, p_max=0.05, effect_min=0.10)
    names = [c.name for c in result.metrics]
    assert names == ["synth/a"]  # only the shared metric is compared


def test_compare_runs_populates_shipped_list() -> None:
    baseline = [{"synth/g": 0.60}, {"synth/g": 0.61}, {"synth/g": 0.59}]
    intervention = [{"synth/g": 0.75}, {"synth/g": 0.76}, {"synth/g": 0.74}]
    result = compare_runs(baseline, intervention)
    assert isinstance(result, ExperimentResult)
    assert result.shipped == ["synth/g"]
    assert result.regressed == []


def test_compare_runs_ignores_non_numeric_and_bool() -> None:
    baseline = [{"synth/a": 0.6, "flag": True, "label": "x"}]
    intervention = [{"synth/a": 0.7, "flag": False, "label": "y"}]
    result = compare_runs(baseline, intervention)
    assert [c.name for c in result.metrics] == ["synth/a"]


# ---- collect_runs -----------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_runs_invokes_n_times() -> None:
    calls = {"n": 0}

    async def _run_once() -> dict[str, float]:
        calls["n"] += 1
        return {"synth/a": float(calls["n"])}

    runs = await collect_runs(_run_once, 3)
    assert calls["n"] == 3
    assert runs == [{"synth/a": 1.0}, {"synth/a": 2.0}, {"synth/a": 3.0}]


# ---- persistence ------------------------------------------------------------


def test_append_and_load_runs_round_trip(tmp_path: Path) -> None:
    exp = tmp_path / "exp1"
    append_runs(exp, "baseline", [{"synth/a": 0.6}])
    combined = append_runs(exp, "baseline", [{"synth/a": 0.61}])
    assert combined == [{"synth/a": 0.6}, {"synth/a": 0.61}]
    assert load_runs(exp, "baseline") == [{"synth/a": 0.6}, {"synth/a": 0.61}]
    # A never-written arm loads as empty, not an error.
    assert load_runs(exp, "intervention") == []


def test_write_result_is_valid_json(tmp_path: Path) -> None:
    exp = tmp_path / "exp2"
    exp.mkdir()
    baseline = [{"synth/g": 0.60}, {"synth/g": 0.61}, {"synth/g": 0.59}]
    intervention = [{"synth/g": 0.75}, {"synth/g": 0.76}, {"synth/g": 0.74}]
    result = compare_runs(baseline, intervention)
    out = write_result(exp, result)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["shipped"] == ["synth/g"]
    assert payload["metrics"][0]["name"] == "synth/g"


def test_json_safe_maps_non_finite_to_null() -> None:
    cleaned = _json_safe(
        {"x": float("inf"), "y": [float("nan"), 1.0], "z": -math.inf}
    )
    assert cleaned == {"x": None, "y": [None, 1.0], "z": None}
    # And it survives json.dumps (strict — rejects Infinity/NaN tokens).
    json.dumps(cleaned, allow_nan=False)


# ---- flatten_synth_report ---------------------------------------------------


@dataclass
class _FakeJudge:
    n_judged: int = 3
    mean_grounding: float = 4.0
    mean_atomicity: float = 4.5
    mean_completeness: float = 3.0
    mean_clarity: float = 5.0


@dataclass
class _FakeReport:
    metrics: dict[str, float] = field(default_factory=dict)
    informational: dict[str, float] = field(default_factory=dict)
    judge_summary: object | None = None


def test_flatten_merges_metrics_and_informational() -> None:
    report = _FakeReport(
        metrics={"synth/fact_grounding_ratio": 0.6},
        informational={"synth/page_density": 0.5},
    )
    flat = flatten_synth_report(report)
    assert flat == {
        "synth/fact_grounding_ratio": 0.6,
        "synth/page_density": 0.5,
    }


def test_flatten_includes_judge_means_when_judged() -> None:
    report = _FakeReport(
        metrics={"synth/a": 0.6},
        informational={},
        judge_summary=_FakeJudge(),
    )
    flat = flatten_synth_report(report)
    assert flat["judge/grounding"] == 4.0
    assert flat["judge/clarity"] == 5.0


def test_flatten_skips_judge_when_nothing_judged() -> None:
    report = _FakeReport(
        metrics={"synth/a": 0.6},
        judge_summary=_FakeJudge(n_judged=0),
    )
    flat = flatten_synth_report(report)
    assert not any(k.startswith("judge/") for k in flat)
