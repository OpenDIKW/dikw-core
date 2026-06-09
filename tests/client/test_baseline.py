"""Pure logic behind ``dikw client eval --against`` / ``--write-baseline``.

A single-run regression *gate*: compare a fresh eval run's metrics to a
committed baseline, direction-aware, and fail when any metric moved the wrong
way past the baseline's tolerance. Not an A/B significance test — that needs
sample distributions and lives in ``evals/tools/ab_experiment.py``. These tests
pin the direction convention (the ``_max`` suffix), the tolerance band, the
missing/extra-metric handling, and the two eval-result payload shapes.
"""

from __future__ import annotations

import json
from pathlib import Path

from dikw_core.client.baseline import (
    DEFAULT_TOLERANCE,
    baseline_document,
    compare_to_baseline,
    extract_metrics,
    load_baseline,
    metric_lower_is_better,
)


def test_direction_max_suffix_is_lower_better() -> None:
    assert metric_lower_is_better("synth/duplicate_ratio_max") is True
    assert metric_lower_is_better("fallback_ratio_max") is True
    assert metric_lower_is_better("doc/ndcg_at_10") is False
    assert metric_lower_is_better("hit_at_3") is False


def test_higher_is_better_regression_improvement_flat() -> None:
    base = {"doc/ndcg_at_10": 0.50}
    regressed = compare_to_baseline(base, {"doc/ndcg_at_10": 0.40}, tolerance=0.02)
    assert not regressed.ok
    assert [r.name for r in regressed.regressions] == ["doc/ndcg_at_10"]

    improved = compare_to_baseline(base, {"doc/ndcg_at_10": 0.60}, tolerance=0.02)
    assert improved.ok and improved.rows[0].status == "improved"

    flat = compare_to_baseline(base, {"doc/ndcg_at_10": 0.49}, tolerance=0.02)
    assert flat.ok and flat.rows[0].status == "flat"


def test_lower_is_better_direction_inverted() -> None:
    base = {"synth/fallback_ratio_max": 0.10}
    # ratio rose → worse for a _max metric → regressed
    worse = compare_to_baseline(base, {"synth/fallback_ratio_max": 0.20}, tolerance=0.02)
    assert not worse.ok and worse.rows[0].status == "regressed"
    # ratio fell → better
    better = compare_to_baseline(base, {"synth/fallback_ratio_max": 0.02}, tolerance=0.02)
    assert better.ok and better.rows[0].status == "improved"


def test_tolerance_boundary_is_flat_not_regressed() -> None:
    # a drop of exactly the tolerance is FLAT (improvement == -tol is not < -tol)
    c = compare_to_baseline({"x": 0.50}, {"x": 0.48}, tolerance=0.02)
    assert c.rows[0].status == "flat" and c.ok


def test_missing_metric_surfaced_but_not_gated() -> None:
    c = compare_to_baseline({"a": 0.5, "b": 0.5}, {"a": 0.5}, tolerance=0.02)
    assert c.missing == ("b",)
    assert [r.name for r in c.rows] == ["a"]
    assert c.ok  # a vanished baseline metric warns, it doesn't fail the gate


def test_extra_metric_is_informational() -> None:
    c = compare_to_baseline({"a": 0.5}, {"a": 0.5, "z": 0.9}, tolerance=0.02)
    assert c.extra == ("z",) and c.ok


def test_extract_metrics_flat_report() -> None:
    payload = {
        "dataset": "mvp",
        "mode": "retrieval",
        "metrics": {"doc/ndcg_at_10": 0.5, "n": 12},
        "passed": True,
    }
    assert extract_metrics(payload) == {"doc/ndcg_at_10": 0.5, "n": 12.0}


def test_extract_metrics_multi_dataset_envelope_is_none() -> None:
    assert extract_metrics({"datasets": [{"metrics": {}}], "passed": True}) is None


def test_extract_metrics_filters_non_numeric() -> None:
    # category_distribution is a dict; a JSON bool is not a metric value.
    payload = {"metrics": {"x": 0.5, "category_distribution": {"a": 1}, "flag": True}}
    assert extract_metrics(payload) == {"x": 0.5}


def test_extract_metrics_no_metrics_key_is_none() -> None:
    assert extract_metrics({"dataset": "mvp", "passed": True}) is None


def test_load_baseline_roundtrip(tmp_path: Path) -> None:
    doc = baseline_document(
        dataset="mvp",
        modes=["synth"],
        metrics={"b": 0.2, "a": 0.8},
        tolerance=0.03,
        created="2026-06-09",
    )
    p = tmp_path / "b.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    metrics, tol = load_baseline(p)
    assert metrics == {"a": 0.8, "b": 0.2}
    assert tol == 0.03


def test_load_baseline_default_tolerance(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"metrics": {"a": 0.5}}), encoding="utf-8")
    _, tol = load_baseline(p)
    assert tol == DEFAULT_TOLERANCE


def test_baseline_document_sorts_metrics_and_carries_fields() -> None:
    doc = baseline_document(
        dataset="mvp",
        modes=["synth"],
        metrics={"b": 1.0, "a": 2.0},
        tolerance=0.02,
        created="2026-06-09",
    )
    assert list(doc["metrics"]) == ["a", "b"]
    assert doc["tolerance"] == 0.02
    assert doc["dataset"] == "mvp"
    assert doc["modes"] == ["synth"]
    assert doc["created"] == "2026-06-09"
