"""Unit tests for the horizontal model-comparison harness —
``evals/tools/compare_models.py``.

Covers the pure layer: arms-spec parsing/validation, direction-aware
best-per-metric, retrieval + synth matrix construction (incl. the Welch p vs the
baseline arm), table rendering, and the JSON round-trip. The live provider-wired
drivers (``run_retrieval_arm`` / ``run_synth_arm``) are exercised manually with
real keys, exactly like ``ab_experiment.py``'s ``collect``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

# ``evals/`` is dev tooling, not an installed package (same pattern as
# test_ab_experiment.py).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.tools import ab_experiment  # noqa: E402
from evals.tools.compare_models import (  # noqa: E402
    best_per_metric,
    build_retrieval_matrix,
    build_synth_matrix,
    format_matrix_table,
    matrix_to_json,
    metric_lower_is_better,
    parse_arms_spec,
    welch_t_test,
)


def _provider_block(**over: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "llm": "anthropic_compat",
        "llm_api_key_env": "ANTHROPIC_API_KEY",
        "embedding": "openai_compat",
        "embedding_model": "bge-m3",
        "embedding_base_url": "https://ai.gitee.com/v1",
        "embedding_api_key_env": "GITEE_API_KEY",
        "embedding_dim": 1024,
        "embedding_revision": "",
        "embedding_normalize": True,
        "embedding_distance": "cosine",
    }
    block.update(over)
    return block


# --- arms-spec parsing ---------------------------------------------------- #


def test_parse_arms_spec_valid_retrieval() -> None:
    raw = {
        "dataset": "scifact",
        "mode": "retrieval",
        "arms": [
            {"name": "bge-m3", "provider": _provider_block(embedding_model="bge-m3")},
            {"name": "qwen", "provider": _provider_block(embedding_model="Qwen3-Embedding-0.6B")},
        ],
    }
    dataset, mode, runs, judge, arms = parse_arms_spec(raw)
    assert dataset == "scifact"
    assert mode == "retrieval"
    assert runs == 5  # default
    assert judge is False
    assert [a.name for a in arms] == ["bge-m3", "qwen"]
    # the *_api_key_env field landed on the provider config, not lost
    assert arms[0].provider.embedding_api_key_env == "GITEE_API_KEY"
    assert arms[1].provider.embedding_model == "Qwen3-Embedding-0.6B"


def test_parse_arms_spec_valid_synth_overrides() -> None:
    raw = {
        "dataset": "mvp",
        "mode": "synth",
        "runs": 3,
        "judge": True,
        "arms": [
            {"name": "a", "provider": _provider_block(llm_model="x")},
            {"name": "b", "provider": _provider_block(llm_model="y")},
        ],
    }
    dataset, mode, runs, judge, arms = parse_arms_spec(raw)
    assert (dataset, mode, runs, judge) == ("mvp", "synth", 3, True)
    assert arms[0].name == "a"  # first arm == baseline


def test_parse_arms_spec_rejects_too_few_arms() -> None:
    raw = {"dataset": "d", "mode": "retrieval", "arms": [{"name": "a", "provider": _provider_block()}]}
    with pytest.raises(ValueError, match=">= 2"):
        parse_arms_spec(raw)


def test_parse_arms_spec_rejects_duplicate_names() -> None:
    raw = {
        "dataset": "d",
        "mode": "retrieval",
        "arms": [
            {"name": "a", "provider": _provider_block()},
            {"name": "a", "provider": _provider_block()},
        ],
    }
    with pytest.raises(ValueError, match="duplicate arm name"):
        parse_arms_spec(raw)


def test_parse_arms_spec_rejects_bad_mode() -> None:
    raw = {"dataset": "d", "mode": "answer", "arms": []}
    with pytest.raises(ValueError, match="mode"):
        parse_arms_spec(raw)


def test_parse_arms_spec_rejects_invalid_provider() -> None:
    # Missing the required embedding_api_key_env → pydantic ValidationError,
    # re-raised with the arm name.
    bad = _provider_block()
    del bad["embedding_api_key_env"]
    raw = {
        "dataset": "d",
        "mode": "retrieval",
        "arms": [
            {"name": "good", "provider": _provider_block()},
            {"name": "bad", "provider": bad},
        ],
    }
    with pytest.raises(ValueError, match="arm 'bad' has an invalid provider"):
        parse_arms_spec(raw)


# --- best-per-metric (direction-aware) ------------------------------------ #


def test_best_per_metric_higher_is_better() -> None:
    assert best_per_metric({"a": 0.5, "b": 0.9, "c": 0.7}, "hit_at_3") == "b"


def test_best_per_metric_lower_is_better_for_max_suffix() -> None:
    assert best_per_metric({"a": 0.5, "b": 0.2, "c": 0.7}, "synth/duplicate_ratio_max") == "b"


def test_best_per_metric_ties_pick_first_in_order() -> None:
    assert best_per_metric({"a": 0.8, "b": 0.8}, "mrr") == "a"


def test_best_per_metric_all_none() -> None:
    assert best_per_metric({"a": None, "b": None}, "mrr") is None


# --- retrieval matrix ----------------------------------------------------- #


def test_build_retrieval_matrix_cells_and_best() -> None:
    per_arm = {
        "bge": {"hit_at_3": 0.8, "mrr": 0.5},
        "qwen": {"hit_at_3": 0.6, "mrr": 0.7},
    }
    m = build_retrieval_matrix(per_arm, dataset="scifact", arms_order=["bge", "qwen"])
    assert m.mode == "retrieval"
    assert m.arms == ["bge", "qwen"]
    assert m.metrics == ["hit_at_3", "mrr"]
    assert m.cells[("bge", "hit_at_3")].value == 0.8
    assert m.cells[("bge", "hit_at_3")].is_best is True
    assert m.cells[("qwen", "hit_at_3")].is_best is False
    assert m.cells[("qwen", "mrr")].is_best is True
    # retrieval cells carry no std / p
    assert m.cells[("bge", "mrr")].std is None
    assert m.cells[("bge", "mrr")].p_vs_baseline is None


def test_build_retrieval_matrix_missing_metric_is_none_never_best() -> None:
    per_arm = {"a": {"hit_at_3": 0.5}, "b": {}}
    m = build_retrieval_matrix(per_arm, dataset="d", arms_order=["a", "b"])
    assert m.cells[("b", "hit_at_3")].value is None
    assert m.cells[("b", "hit_at_3")].is_best is False
    assert m.cells[("a", "hit_at_3")].is_best is True


# --- synth matrix --------------------------------------------------------- #


def test_build_synth_matrix_mean_std_and_welch_p() -> None:
    per_arm_runs = {
        "deepseek": [
            {"synth/fact_grounding_ratio": 0.70, "synth/duplicate_ratio_max": 0.10},
            {"synth/fact_grounding_ratio": 0.72, "synth/duplicate_ratio_max": 0.12},
            {"synth/fact_grounding_ratio": 0.71, "synth/duplicate_ratio_max": 0.11},
        ],
        "minimax": [
            {"synth/fact_grounding_ratio": 0.90, "synth/duplicate_ratio_max": 0.30},
            {"synth/fact_grounding_ratio": 0.91, "synth/duplicate_ratio_max": 0.31},
            {"synth/fact_grounding_ratio": 0.92, "synth/duplicate_ratio_max": 0.29},
        ],
    }
    m = build_synth_matrix(
        per_arm_runs, dataset="mvp", arms_order=["deepseek", "minimax"], baseline_arm="deepseek"
    )
    # mean/std match mean_std of the runs
    exp_mean, exp_std = ab_experiment.mean_std([0.70, 0.72, 0.71])
    cell = m.cells[("deepseek", "synth/fact_grounding_ratio")]
    assert cell.value == pytest.approx(exp_mean)
    assert cell.std == pytest.approx(exp_std)
    # baseline arm has no p; the other arm's p == welch vs baseline
    assert cell.p_vs_baseline is None
    other = m.cells[("minimax", "synth/fact_grounding_ratio")]
    exp_p = welch_t_test([0.70, 0.72, 0.71], [0.90, 0.91, 0.92])[2]
    assert other.p_vs_baseline == pytest.approx(exp_p)
    assert exp_p < 0.05  # clear separation
    # best per metric is direction-aware: grounding higher-better → minimax;
    # duplicate_ratio_max lower-better → deepseek
    assert m.cells[("minimax", "synth/fact_grounding_ratio")].is_best is True
    assert m.cells[("deepseek", "synth/duplicate_ratio_max")].is_best is True


# --- table render + JSON -------------------------------------------------- #


def test_format_matrix_table_retrieval() -> None:
    per_arm = {"bge": {"hit_at_3": 0.8}, "qwen": {"hit_at_3": 0.6}}
    out = format_matrix_table(
        build_retrieval_matrix(per_arm, dataset="scifact", arms_order=["bge", "qwen"])
    )
    assert "bge" in out and "qwen" in out
    assert "hit_at_3" in out
    assert "*" in out  # a best cell is marked


def test_format_matrix_table_synth_shows_std_and_p() -> None:
    per_arm_runs = {
        "a": [{"synth/fact_grounding_ratio": 0.70}, {"synth/fact_grounding_ratio": 0.72}],
        "b": [{"synth/fact_grounding_ratio": 0.90}, {"synth/fact_grounding_ratio": 0.92}],
    }
    out = format_matrix_table(
        build_synth_matrix(per_arm_runs, dataset="mvp", arms_order=["a", "b"], baseline_arm="a")
    )
    assert "+/-" in out
    assert "p=" in out
    assert "baseline: a" in out


def test_matrix_to_json_roundtrip_and_non_finite() -> None:
    # A deterministic separation yields a non-finite Welch statistic path; p
    # itself is 0.0 here, but force a None p by giving the baseline a single run
    # (welch returns p=1.0 for n<2 — still finite). Assert json strictness holds.
    per_arm_runs = {
        "a": [{"m": 0.5}, {"m": 0.6}],
        "b": [{"m": 0.9}, {"m": 0.95}],
    }
    matrix = build_synth_matrix(
        per_arm_runs, dataset="mvp", arms_order=["a", "b"], baseline_arm="a"
    )
    blob = matrix_to_json(matrix)
    text = json.dumps(blob, allow_nan=False)  # must not raise on Infinity/NaN
    recovered = json.loads(text)
    assert recovered["arms"] == ["a", "b"]
    assert recovered["metrics"] == ["m"]
    assert recovered["cells"]["a"]["m"]["value"] == pytest.approx(0.55)
    assert recovered["baseline_arm"] == "a"


def test_matrix_to_json_maps_non_finite_p_to_null() -> None:
    # Hand-craft a matrix whose cell carries a non-finite p (mimics a Welch inf
    # case) and assert _json_safe maps it to null so json stays strict-parseable.
    from evals.tools.compare_models import ComparisonMatrix, MetricCell

    cell = MetricCell(arm="b", metric="m", value=0.9, std=0.0, p_vs_baseline=math.inf, is_best=True)
    matrix = ComparisonMatrix(
        mode="synth",
        dataset="d",
        baseline_arm="a",
        arms=["a", "b"],
        metrics=["m"],
        cells={
            ("a", "m"): MetricCell("a", "m", 0.5, 0.0, None, False),
            ("b", "m"): cell,
        },
    )
    text = json.dumps(matrix_to_json(matrix), allow_nan=False)
    assert json.loads(text)["cells"]["b"]["m"]["p_vs_baseline"] is None


# --- reuse, not reinvent -------------------------------------------------- #


def test_reuses_canonical_symbols() -> None:
    """The harness must reuse the tested statistics + direction rule rather than
    redefine them, so a fix to either lands here for free."""
    assert welch_t_test is ab_experiment.welch_t_test

    from dikw_core.client.baseline import metric_lower_is_better as canonical

    assert metric_lower_is_better is canonical
