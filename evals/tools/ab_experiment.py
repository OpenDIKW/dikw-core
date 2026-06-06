"""A/B experiment harness for synth-quality optimization.

The synth pipeline's output varies run-to-run (the LLM is non-deterministic),
so a single before/after eval can't tell a real improvement from ±0.05 noise.
This tool runs the **same** synth eval N times per arm, then compares the two
arms with proper statistics: per-metric ``mean ± std``, a Welch two-sample
t-test (unequal variance — the two arms are independent run sets, not paired),
a standardized effect size (Cohen's d), and a **ship gate**: a metric only
"ships" when the change is statistically significant (``p < p_max``) *and* the
direction-aware improvement clears a noise-floor (``improvement > effect_min``).

Workflow (the human drives the git state between arms — the harness can't
rebuild the code itself)::

    # on the baseline commit:
    uv run python evals/tools/ab_experiment.py collect \\
        --base /path/to/base --dataset mvp --arm baseline \\
        --runs 3 --exp evals/experiments/enriched-prompt
    # check out the intervention commit, then:
    uv run python evals/tools/ab_experiment.py collect \\
        --base /path/to/base --dataset mvp --arm intervention \\
        --runs 3 --exp evals/experiments/enriched-prompt
    # back on any commit (pure stats, no LLM):
    uv run python evals/tools/ab_experiment.py compare \\
        --exp evals/experiments/enriched-prompt

``compare`` writes ``result.json`` and prints a table. ``collect`` appends to
``<arm>_runs.json`` so two ``--runs 3`` calls accumulate to 6 if you want more
power. The statistics layer (everything except ``collect``'s provider wiring)
is pure and unit-tested in ``tests/test_ab_experiment.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Statistics primitives (pure — no scipy/numpy in this repo).
# ---------------------------------------------------------------------------


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Sample mean and standard deviation (ddof=1). ``std`` is 0 for n<2."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    m = sum(values) / n
    if n == 1:
        return (m, 0.0)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return (m, math.sqrt(var))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta (Lentz's method,
    after Numerical Recipes ``betacf``)."""
    maxit = 300
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value ``P(|T| >= |t|)`` for Student's t with ``df`` df.

    Uses the standard identity ``p = I_{df/(df+t^2)}(df/2, 1/2)`` (Abramowitz &
    Stegun 26.5.27). Exact references: ``t=1, df=1 -> 0.5`` (Cauchy) and
    ``t=sqrt(2), df=2 -> 1 - 1/sqrt(2)``.
    """
    if df <= 0 or t == 0.0:
        return 1.0
    if not math.isfinite(t):
        return 0.0
    x = df / (df + t * t)
    return betai(df / 2.0, 0.5, x)


def welch_t_test(
    baseline: Sequence[float], intervention: Sequence[float]
) -> tuple[float, float, float]:
    """Welch's unequal-variance two-sample t-test.

    Returns ``(t, df, p)`` where ``t`` is signed so positive means the
    intervention mean is higher. ``t`` is ``+/-inf`` when both arms have zero
    variance but different means (a deterministic separation); ``(0, df, 1.0)``
    when neither arm has enough data (n<2 either side) or the arms are
    identical.
    """
    n1, n2 = len(baseline), len(intervention)
    if n1 < 2 or n2 < 2:
        return (0.0, 0.0, 1.0)
    m1, s1 = mean_std(baseline)
    m2, s2 = mean_std(intervention)
    v1 = s1 * s1 / n1
    v2 = s2 * s2 / n2
    denom = v1 + v2
    if denom == 0.0:
        if m1 == m2:
            return (0.0, float(n1 + n2 - 2), 1.0)
        return (math.inf if m2 > m1 else -math.inf, float(n1 + n2 - 2), 0.0)
    t = (m2 - m1) / math.sqrt(denom)
    df = denom * denom / (
        (v1 * v1) / (n1 - 1) + (v2 * v2) / (n2 - 1)
    )
    return (t, df, student_t_two_sided_p(t, df))


def cohens_d(
    baseline: Sequence[float], intervention: Sequence[float]
) -> float | None:
    """Pooled-SD Cohen's d (intervention - baseline). ``None`` when the
    pooled SD is undefined (n<2 either side, or zero variance with a real
    mean gap); ``0.0`` when both arms are identical constants."""
    n1, n2 = len(baseline), len(intervention)
    if n1 < 2 or n2 < 2:
        return None
    m1, s1 = mean_std(baseline)
    m2, s2 = mean_std(intervention)
    sp = math.sqrt(
        ((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2)
    )
    if sp == 0.0:
        return 0.0 if m1 == m2 else None
    return (m2 - m1) / sp


# ---------------------------------------------------------------------------
# Metric direction (mirror of runner._threshold_direction — the naming
# convention *is* the single source of truth, so we restate the one rule
# rather than import a private symbol).
# ---------------------------------------------------------------------------


def metric_direction(name: str) -> str:
    """``"max"`` (lower is better) for names ending in ``_max``, else ``"min"``
    (higher is better). Strips a leading ``<view>/`` prefix so
    ``synth/duplicate_ratio_max`` classifies like ``duplicate_ratio_max``."""
    bare = name.rpartition("/")[-1] or name
    return "max" if bare.endswith("_max") else "min"


# ---------------------------------------------------------------------------
# Comparison.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricComparison:
    name: str
    direction: str  # "min" | "max"
    n_baseline: int
    n_intervention: int
    baseline_mean: float
    baseline_std: float
    intervention_mean: float
    intervention_std: float
    delta: float  # raw intervention_mean - baseline_mean
    improvement: float  # direction-aware (>0 == better)
    cohens_d: float | None
    t: float | None  # None when non-finite (deterministic separation)
    df: float
    p_value: float
    significant: bool
    ships: bool
    regressed: bool


@dataclass(frozen=True)
class ExperimentResult:
    p_max: float
    effect_min: float
    metrics: list[MetricComparison] = field(default_factory=list)
    shipped: list[str] = field(default_factory=list)
    regressed: list[str] = field(default_factory=list)


def compare_metric(
    name: str,
    baseline: Sequence[float],
    intervention: Sequence[float],
    *,
    p_max: float,
    effect_min: float,
) -> MetricComparison:
    """Compare one metric's baseline vs intervention value sets."""
    direction = metric_direction(name)
    m1, s1 = mean_std(baseline)
    m2, s2 = mean_std(intervention)
    delta = m2 - m1
    improvement = delta if direction == "min" else -delta
    t, df, p = welch_t_test(baseline, intervention)
    significant = p < p_max
    ships = significant and improvement > effect_min
    regressed = significant and improvement < -effect_min
    return MetricComparison(
        name=name,
        direction=direction,
        n_baseline=len(baseline),
        n_intervention=len(intervention),
        baseline_mean=m1,
        baseline_std=s1,
        intervention_mean=m2,
        intervention_std=s2,
        delta=delta,
        improvement=improvement,
        cohens_d=cohens_d(baseline, intervention),
        t=t if math.isfinite(t) else None,
        df=df,
        p_value=p,
        significant=significant,
        ships=ships,
        regressed=regressed,
    )


def _collect_metric_values(
    runs: Sequence[Mapping[str, float]],
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for run in runs:
        for key, value in run.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            out.setdefault(key, []).append(float(value))
    return out


def compare_runs(
    baseline_runs: Sequence[Mapping[str, float]],
    intervention_runs: Sequence[Mapping[str, float]],
    *,
    p_max: float = 0.05,
    effect_min: float = 0.10,
) -> ExperimentResult:
    """Compare two arms of flat metric dicts; one ``MetricComparison`` per
    metric present in **both** arms, sorted by name."""
    base_vals = _collect_metric_values(baseline_runs)
    int_vals = _collect_metric_values(intervention_runs)
    shared = sorted(base_vals.keys() & int_vals.keys())
    comparisons = [
        compare_metric(
            name,
            base_vals[name],
            int_vals[name],
            p_max=p_max,
            effect_min=effect_min,
        )
        for name in shared
    ]
    return ExperimentResult(
        p_max=p_max,
        effect_min=effect_min,
        metrics=comparisons,
        shipped=[c.name for c in comparisons if c.ships],
        regressed=[c.name for c in comparisons if c.regressed],
    )


# ---------------------------------------------------------------------------
# Synth-report flattening + persistence.
# ---------------------------------------------------------------------------


def flatten_synth_report(report: Any) -> dict[str, float]:
    """Flatten a ``SynthEvalReport`` into a single metric→value dict.

    Merges gated ``metrics`` + ``informational`` (both already keyed
    ``synth/*``) and, when a judge ran, the four ``judge/<dim>`` means. Judge
    CIs are deliberately dropped — they are *per-run* uncertainty; the A/B
    harness derives its own cross-run interval from the t-test.
    """
    flat: dict[str, float] = {}
    flat.update({k: float(v) for k, v in dict(report.metrics).items()})
    flat.update({k: float(v) for k, v in dict(report.informational).items()})
    js = getattr(report, "judge_summary", None)
    if js is not None and js.n_judged > 0:
        flat["judge/grounding"] = float(js.mean_grounding)
        flat["judge/atomicity"] = float(js.mean_atomicity)
        flat["judge/completeness"] = float(js.mean_completeness)
        flat["judge/clarity"] = float(js.mean_clarity)
    return flat


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` so ``json.dumps``
    produces valid JSON (``Infinity``/``NaN`` are rejected by strict parsers)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    return obj


def runs_path(exp_dir: Path, arm: str) -> Path:
    return exp_dir / f"{arm}_runs.json"


def load_runs(exp_dir: Path, arm: str) -> list[dict[str, float]]:
    path = runs_path(exp_dir, arm)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs", []) if isinstance(payload, dict) else []
    return [dict(r) for r in runs]


def append_runs(
    exp_dir: Path, arm: str, new_runs: Sequence[Mapping[str, float]]
) -> list[dict[str, float]]:
    """Append ``new_runs`` to ``<arm>_runs.json`` (creating it) and return the
    full accumulated list."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    existing = load_runs(exp_dir, arm)
    combined = existing + [dict(r) for r in new_runs]
    runs_path(exp_dir, arm).write_text(
        # ``allow_nan=False`` makes the strict-JSON contract self-enforcing at
        # the boundary: a non-finite value ``_json_safe`` somehow missed raises
        # here instead of silently emitting invalid ``Infinity``/``NaN`` tokens.
        json.dumps(_json_safe({"runs": combined}), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return combined


def write_result(exp_dir: Path, result: ExperimentResult) -> Path:
    path = exp_dir / "result.json"
    path.write_text(
        json.dumps(_json_safe(asdict(result)), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Collection (generic + live).
# ---------------------------------------------------------------------------


async def collect_runs(
    run_once: Callable[[], Awaitable[Mapping[str, float]]], n: int
) -> list[dict[str, float]]:
    """Call ``run_once`` ``n`` times, returning each result as a flat dict.

    Generic so the live synth-eval driver and unit tests share one loop.
    """
    out: list[dict[str, float]] = []
    for _ in range(n):
        out.append(dict(await run_once()))
    return out


async def collect_synth_eval_runs(
    base: Path,
    dataset: str,
    n: int,
    *,
    judge: bool = False,
    judge_sample: int | None = None,
    target_tokens_per_group: int | None = None,
) -> list[dict[str, float]]:
    """Live driver: build providers from ``base``'s ``dikw.yml`` and run the
    synth eval ``n`` times, returning one flat metric dict per run.

    Imports the engine lazily so the pure stats layer (and its tests) never
    pull in the provider factory / dataset loader.
    """
    from dikw_core.config import CONFIG_FILENAME, load_config
    from dikw_core.eval.dataset import load_dataset
    from dikw_core.eval.runner import run_synth_eval
    from dikw_core.providers import build_embedder, build_llm

    cfg = load_config(base / CONFIG_FILENAME)
    spec = load_dataset(dataset)
    llm = build_llm(cfg.provider, base_root=base)
    embedder = build_embedder(cfg.provider)

    async def _run_once() -> dict[str, float]:
        report = await run_synth_eval(
            spec,
            llm=llm,
            embedder=embedder,
            provider_config=cfg.provider,
            retrieval_config=cfg.retrieval,
            judge=judge,
            judge_sample=judge_sample,
            target_tokens_per_group=target_tokens_per_group,
        )
        return flatten_synth_report(report)

    return await collect_runs(_run_once, n)


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------


def format_comparison_table(result: ExperimentResult) -> str:
    """Plain-text table — one row per metric, plus a ship-gate summary."""
    lines = [
        f"ship gate: p < {result.p_max}  AND  improvement > {result.effect_min}",
        "",
        f"{'metric':32s} {'base':>8s} {'interv':>8s} {'Δimprove':>9s} "
        f"{'p':>7s} {'ships':>6s}",
        "-" * 75,
    ]
    for c in result.metrics:
        p_str = f"{c.p_value:.4f}" if math.isfinite(c.p_value) else "  -  "
        verdict = "SHIP" if c.ships else ("REGR" if c.regressed else "—")
        lines.append(
            f"{c.name:32s} {c.baseline_mean:8.4f} {c.intervention_mean:8.4f} "
            f"{c.improvement:+9.4f} {p_str:>7s} {verdict:>6s}"
        )
    lines.append("")
    lines.append(
        f"shipped: {result.shipped or '(none)'}   "
        f"regressed: {result.regressed or '(none)'}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _cmd_collect(args: argparse.Namespace) -> int:
    exp_dir = Path(args.exp)
    runs = asyncio.run(
        collect_synth_eval_runs(
            Path(args.base),
            args.dataset,
            args.runs,
            judge=args.judge,
            judge_sample=args.judge_sample,
            target_tokens_per_group=args.target_tokens,
        )
    )
    combined = append_runs(exp_dir, args.arm, runs)
    print(
        f"collected {len(runs)} run(s) for arm '{args.arm}' "
        f"({len(combined)} total) → {runs_path(exp_dir, args.arm)}"
    )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    exp_dir = Path(args.exp)
    baseline = load_runs(exp_dir, "baseline")
    intervention = load_runs(exp_dir, "intervention")
    if not baseline or not intervention:
        print(
            f"error: need both baseline_runs.json and intervention_runs.json "
            f"in {exp_dir} (have {len(baseline)} / {len(intervention)} runs)",
            file=sys.stderr,
        )
        return 2
    result = compare_runs(
        baseline,
        intervention,
        p_max=args.p_max,
        effect_min=args.effect_min,
    )
    out = write_result(exp_dir, result)
    print(format_comparison_table(result))
    print(f"\nwrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A/B experiment harness for synth-quality optimization."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("collect", help="run the synth eval N times for one arm")
    pc.add_argument("--base", required=True, help="dikw base with the provider config")
    pc.add_argument("--dataset", required=True, help="packaged dataset name (e.g. mvp)")
    pc.add_argument(
        "--arm",
        required=True,
        choices=["baseline", "intervention"],
        help="which arm these runs belong to",
    )
    pc.add_argument("--runs", type=int, default=3, help="number of eval runs")
    pc.add_argument("--exp", required=True, help="experiment directory")
    pc.add_argument("--judge", action="store_true", help="also run the LLM judge")
    pc.add_argument("--judge-sample", type=int, default=None, dest="judge_sample")
    pc.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        dest="target_tokens",
        help=(
            "override synth.target_tokens_per_group for this run; a small value "
            "fans a small corpus into multiple groups so grouping-sensitive "
            "changes (priority-create / existing-pages) are exercised"
        ),
    )
    pc.set_defaults(func=_cmd_collect)

    pp = sub.add_parser("compare", help="compare collected arms (pure stats)")
    pp.add_argument("--exp", required=True, help="experiment directory")
    pp.add_argument("--p-max", type=float, default=0.05, dest="p_max")
    pp.add_argument("--effect-min", type=float, default=0.10, dest="effect_min")
    pp.set_defaults(func=_cmd_compare)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
