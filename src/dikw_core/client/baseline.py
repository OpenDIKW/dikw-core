"""Machine-readable eval baselines + a direction-aware regression gate.

``dikw client eval --write-baseline <p>`` dumps a run's metrics to a committed
JSON; ``--against <p>`` compares a fresh run to that baseline and fails (exit 1)
when any metric moved the wrong way past the baseline's tolerance.

This is a single-run regression **gate**, not an A/B significance test. For a
deterministic retrieval eval the comparison is effectively exact (tighten the
tolerance); for an LLM-driven synth eval the tolerance must be generous enough
to absorb run-to-run model noise, so the gate catches large regressions rather
than jitter. The statistical A/B path (Welch t-test over sample distributions)
lives in ``evals/tools/ab_experiment.py`` — it needs per-query / multi-run
samples this gate deliberately does not collect.

Pure stdlib only: the client must not import ``dikw_core.{eval,api,storage,
server}`` (it is meant to package as a standalone wheel).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Absolute per-metric noise floor used when a baseline file omits ``tolerance``.
DEFAULT_TOLERANCE = 0.02

#: Float-dust slack on the band edges so a drop of *exactly* the tolerance stays
#: ``flat`` instead of flipping to ``regressed`` (``0.50 - 0.48`` is
#: ``-0.020000000000000018`` in IEEE-754, which is < ``-0.02``). Far below any
#: tolerance a user would set, so it never masks a real difference.
_BAND_EPSILON = 1e-9


def metric_lower_is_better(name: str) -> bool:
    """``True`` for a lower-is-better metric (bare name ends in ``_max``).

    Mirrors the engine's naming convention (``eval.runner._threshold_direction``
    / ``ab_experiment.metric_direction``): the ``_max`` suffix marks a ratio
    where a *higher* value is *worse* (e.g. ``duplicate_ratio_max``). The
    ``<view>/`` prefix is stripped so ``synth/fallback_ratio_max`` classifies
    like ``fallback_ratio_max``. Restated here rather than imported because the
    client must not depend on ``dikw_core.eval`` — the naming rule itself is the
    shared source of truth.
    """
    return name.rpartition("/")[-1].endswith("_max")


def _as_float(value: Any) -> float | None:
    # ``bool`` is an ``int`` subclass; a JSON ``true`` is a flag, not a metric.
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _float_metrics(raw: Any) -> dict[str, float]:
    """Keep only the scalar (float-coercible) entries of a metrics map.

    Eval reports mix scalars with non-scalar diagnostics (e.g. synth's
    ``category_distribution`` is a dict); those are dropped, not compared.
    """
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        coerced = _as_float(value)
        if coerced is not None:
            out[str(key)] = coerced
    return out


@dataclass(frozen=True)
class MetricDelta:
    name: str
    baseline: float
    observed: float
    delta: float  # observed - baseline (raw, signed)
    improvement: float  # direction-aware: > 0 means better than baseline
    status: str  # "improved" | "flat" | "regressed"


@dataclass(frozen=True)
class BaselineComparison:
    tolerance: float
    rows: tuple[MetricDelta, ...]
    missing: tuple[str, ...]  # pinned by the baseline, absent/non-numeric this run
    extra: tuple[str, ...]  # produced this run, not pinned by the baseline

    @property
    def regressions(self) -> tuple[MetricDelta, ...]:
        return tuple(r for r in self.rows if r.status == "regressed")

    @property
    def ok(self) -> bool:
        return not self.regressions


def extract_metrics(payload: Mapping[str, Any]) -> dict[str, float] | None:
    """Pull a single flat ``{name: value}`` metrics map from an eval task result.

    Returns ``None`` for the multi-dataset envelope (``{"datasets": [...]}``) or
    a result with no ``metrics`` map — the caller surfaces that as "needs a
    single --dataset / --eval mode" rather than silently comparing the wrong
    thing.
    """
    if "datasets" in payload:
        return None
    raw = payload.get("metrics")
    if not isinstance(raw, Mapping):
        return None
    return _float_metrics(raw)


def compare_to_baseline(
    baseline: Mapping[str, float],
    observed: Mapping[str, float],
    *,
    tolerance: float,
) -> BaselineComparison:
    """Direction-aware comparison of an observed run against a pinned baseline.

    A metric is ``regressed`` only when its direction-aware ``improvement`` drops
    below ``-tolerance``; a symmetric rise above ``+tolerance`` is ``improved``;
    anything in between is ``flat``. Metrics pinned by the baseline but absent
    this run are surfaced in ``missing`` (a warning, not a gate failure — a
    metric set can legitimately change); metrics new this run land in ``extra``.
    """
    rows: list[MetricDelta] = []
    missing: list[str] = []
    for name in sorted(baseline):
        base = baseline[name]
        if name not in observed:
            missing.append(name)
            continue
        obs = observed[name]
        delta = obs - base
        improvement = -delta if metric_lower_is_better(name) else delta
        if improvement < -(tolerance + _BAND_EPSILON):
            status = "regressed"
        elif improvement > tolerance + _BAND_EPSILON:
            status = "improved"
        else:
            status = "flat"
        rows.append(MetricDelta(name, base, obs, delta, improvement, status))
    extra = tuple(sorted(n for n in observed if n not in baseline))
    return BaselineComparison(tolerance, tuple(rows), tuple(missing), extra)


def load_baseline(path: Path) -> tuple[dict[str, float], float]:
    """Return ``(metrics, tolerance)`` from a baseline JSON.

    ``tolerance`` falls back to :data:`DEFAULT_TOLERANCE` when absent or
    non-numeric.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"baseline {path} is not a JSON object")
    metrics = _float_metrics(data.get("metrics"))
    tol = _as_float(data.get("tolerance"))
    return metrics, (DEFAULT_TOLERANCE if tol is None else tol)


def baseline_document(
    *,
    dataset: str | None,
    modes: list[str] | None,
    metrics: Mapping[str, float],
    tolerance: float,
    created: str,
) -> dict[str, Any]:
    """The committed baseline shape: identity + pinned metrics + the gate's
    tolerance. Metrics are sorted so re-writing a baseline produces a stable,
    review-friendly diff."""
    return {
        "dataset": dataset,
        "modes": list(modes or []),
        "created": created,
        "tolerance": tolerance,
        "metrics": dict(sorted(metrics.items())),
    }
