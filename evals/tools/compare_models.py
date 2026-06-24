"""Horizontal model comparison harness.

Runs the **same** eval dataset against N model arms and emits an arm-by-metric
comparison matrix (+ per-arm JSON). Two axes:

* ``compare`` (retrieval) — arms differ in their ``embedding_*`` config. Retrieval
  metrics are deterministic for a fixed (embedder, corpus), so one run per arm.
  Metrics: hit_at_3 / hit_at_10 / mrr / ndcg_at_10 / recall_at_100.
* ``compare-synth`` (synth) — arms differ in their ``llm_*`` config. Synth output
  is non-deterministic, so N runs per arm + a Welch t-test of each arm vs the
  first (baseline) arm. Metrics carry the ``synth/`` prefix that
  ``flatten_synth_report`` emits: synth/fact_grounding_ratio / synth/atomicity_score /
  synth/duplicate_ratio_max / synth/wikilink_resolved_ratio / synth/language_fidelity
  (+ judge dims when ``--judge``).

Because the API-key env var is config-driven (``provider.{llm,embedding}_api_key_env``)
each arm carries a full ``provider:`` block, so two arms that both speak the
``anthropic_compat`` protocol (e.g. DeepSeek + MiniMax) resolve distinct keys
without the harness touching ``os.environ``.

Arms spec (YAML)::

    dataset: scifact            # name or path
    mode: retrieval             # retrieval | synth
    runs: 5                     # synth only (default 5)
    judge: false                # synth only
    arms:
      - name: bge-m3            # first arm == baseline (Welch reference)
        provider:               # a full dikw.yml provider block
          embedding: openai_compat
          embedding_model: bge-m3
          embedding_base_url: https://ai.gitee.com/v1
          embedding_api_key_env: GITEE_API_KEY
          embedding_dim: 1024
          embedding_revision: ""
          embedding_normalize: true
          embedding_distance: cosine
          embedding_batch_size: 16
          llm_api_key_env: ANTHROPIC_API_KEY   # required field; unused in retrieval mode
      - name: qwen3-0.6b
        provider: { ... embedding_model: Qwen3-Embedding-0.6B ... }

Usage::

    uv run --env-file .env python evals/tools/compare_models.py compare \\
        --spec arms.yaml --exp evals/experiments/embed-bgem3-vs-qwen
    uv run --env-file .env python evals/tools/compare_models.py compare-synth \\
        --spec arms.yaml --exp evals/experiments/llm-deepseek-vs-minimax --runs 5

The pure layer (spec parse, matrix build, Welch p, table render, JSON shape) is
unit-tested in ``tests/test_compare_models.py``; the live provider-wired drivers
are exercised manually, exactly like ``ab_experiment.py``'s ``collect``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ``evals/`` is developer tooling, not an installed package — add the repo root
# so the cross-tool + engine imports resolve when run as a script (mirrors
# test_ab_experiment.py / test_sweep_rrf.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.tools.ab_experiment import (  # noqa: E402
    _json_safe,
    flatten_synth_report,
    mean_std,
    welch_t_test,
)

from dikw_core.client.baseline import metric_lower_is_better  # noqa: E402

# Arm names are used directly as ``<arm>.json`` output filenames, so restrict
# them to a filesystem-safe charset (no separators / traversal / spaces).
_SAFE_ARM_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

# Canonical retrieval metric order for the matrix (doc-view aliases).
_RETRIEVAL_METRICS: tuple[str, ...] = (
    "hit_at_3",
    "hit_at_10",
    "mrr",
    "ndcg_at_10",
    "recall_at_100",
)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArmSpec:
    name: str
    provider: Any  # ProviderConfig (typed Any so the pure layer needs no engine import)


@dataclass(frozen=True)
class MetricCell:
    arm: str
    metric: str
    value: float | None  # headline value: single run (retrieval) or mean (synth)
    std: float | None  # synth only (None for retrieval / missing)
    p_vs_baseline: float | None  # synth only; None for the baseline arm / retrieval
    is_best: bool


@dataclass(frozen=True)
class ComparisonMatrix:
    mode: str
    dataset: str
    baseline_arm: str | None
    arms: list[str]
    metrics: list[str]
    cells: dict[tuple[str, str], MetricCell]


# --------------------------------------------------------------------------- #
# Pure layer (unit-tested; no engine / network)
# --------------------------------------------------------------------------- #


def parse_arms_spec(raw: Mapping[str, Any]) -> tuple[str, str, int, bool, list[ArmSpec]]:
    """Validate + split an arms-spec mapping into
    ``(dataset, mode, runs, judge, arms)``.

    Builds each arm's ``ProviderConfig`` (pure — pydantic validation, no I/O).
    The first arm is the baseline (the Welch reference for synth mode).
    """
    from dikw_core.config import ProviderConfig  # local: pydantic only, no engine I/O

    dataset = raw.get("dataset")
    if not dataset or not isinstance(dataset, str):
        raise ValueError("arms spec: 'dataset' (str) is required")
    mode = raw.get("mode")
    if mode not in ("retrieval", "synth"):
        raise ValueError(f"arms spec: 'mode' must be 'retrieval' or 'synth', got {mode!r}")
    runs = int(raw.get("runs", 5))
    if runs < 1:
        raise ValueError("arms spec: 'runs' must be >= 1")
    judge = bool(raw.get("judge", False))

    raw_arms = raw.get("arms")
    if not isinstance(raw_arms, list) or len(raw_arms) < 2:
        raise ValueError("arms spec: 'arms' must be a list of >= 2 entries")

    arms: list[ArmSpec] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw_arms):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"arms spec: arm #{i} must be a mapping, got {type(entry).__name__}"
            )
        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"arms spec: arm #{i} is missing a 'name'")
        if not _SAFE_ARM_NAME.match(name):
            raise ValueError(
                f"arms spec: arm name {name!r} must match [A-Za-z0-9._-]+ "
                "(it is used directly as a per-arm output filename)"
            )
        if name in seen:
            raise ValueError(f"arms spec: duplicate arm name {name!r}")
        seen.add(name)
        provider_block = entry.get("provider")
        if not isinstance(provider_block, Mapping):
            raise ValueError(f"arms spec: arm {name!r} is missing a 'provider' block")
        try:
            provider = ProviderConfig.model_validate(dict(provider_block))
        except Exception as exc:  # pydantic.ValidationError — annotate with arm name
            raise ValueError(f"arms spec: arm {name!r} has an invalid provider: {exc}") from exc
        arms.append(ArmSpec(name=name, provider=provider))
    return (dataset, mode, runs, judge, arms)


def best_per_metric(arm_values: Mapping[str, float | None], metric: str) -> str | None:
    """The winning arm for ``metric`` (direction-aware). Ties → first in order;
    all-None → None."""
    lower_better = metric_lower_is_better(metric)
    best_arm: str | None = None
    best_val: float | None = None
    for arm, val in arm_values.items():
        if val is None:
            continue
        if best_val is None or (val < best_val if lower_better else val > best_val):
            best_val, best_arm = val, arm
    return best_arm


def _ordered_metric_union(per_arm_keys: Sequence[Sequence[str]], preferred: Sequence[str]) -> list[str]:
    """Union of metric keys, ``preferred`` order first then the rest sorted."""
    present: set[str] = set()
    for keys in per_arm_keys:
        present.update(keys)
    head = [m for m in preferred if m in present]
    tail = sorted(present - set(head))
    return head + tail


def build_retrieval_matrix(
    per_arm: Mapping[str, Mapping[str, float]], *, dataset: str, arms_order: Sequence[str]
) -> ComparisonMatrix:
    """Build a deterministic (1 run/arm) retrieval comparison matrix."""
    metrics = _ordered_metric_union([list(per_arm[a].keys()) for a in arms_order], _RETRIEVAL_METRICS)
    cells: dict[tuple[str, str], MetricCell] = {}
    for metric in metrics:
        arm_values = {a: per_arm[a].get(metric) for a in arms_order}
        winner = best_per_metric(arm_values, metric)
        for arm in arms_order:
            cells[(arm, metric)] = MetricCell(
                arm=arm,
                metric=metric,
                value=arm_values[arm],
                std=None,
                p_vs_baseline=None,
                is_best=(arm == winner),
            )
    return ComparisonMatrix(
        mode="retrieval",
        dataset=dataset,
        baseline_arm=None,
        arms=list(arms_order),
        metrics=metrics,
        cells=cells,
    )


def build_synth_matrix(
    per_arm_runs: Mapping[str, Sequence[Mapping[str, float]]],
    *,
    dataset: str,
    arms_order: Sequence[str],
    baseline_arm: str,
) -> ComparisonMatrix:
    """Build an N-run synth comparison matrix: mean+/-std per cell + a Welch
    p-value of each non-baseline arm vs ``baseline_arm`` per metric."""
    metrics = _ordered_metric_union(
        [list(r.keys()) for a in arms_order for r in per_arm_runs[a]], ()
    )

    def _values(arm: str, metric: str) -> list[float]:
        return [r[metric] for r in per_arm_runs[arm] if metric in r]

    cells: dict[tuple[str, str], MetricCell] = {}
    for metric in metrics:
        means: dict[str, float | None] = {}
        stds: dict[str, float | None] = {}
        for arm in arms_order:
            vals = _values(arm, metric)
            if vals:
                m, s = mean_std(vals)
                means[arm], stds[arm] = m, s
            else:
                means[arm], stds[arm] = None, None
        winner = best_per_metric(means, metric)
        base_vals = _values(baseline_arm, metric)
        for arm in arms_order:
            p: float | None = None
            if arm != baseline_arm:
                arm_vals = _values(arm, metric)
                if base_vals and arm_vals:
                    p = welch_t_test(base_vals, arm_vals)[2]
            cells[(arm, metric)] = MetricCell(
                arm=arm,
                metric=metric,
                value=means[arm],
                std=stds[arm],
                p_vs_baseline=p,
                is_best=(arm == winner),
            )
    return ComparisonMatrix(
        mode="synth",
        dataset=dataset,
        baseline_arm=baseline_arm,
        arms=list(arms_order),
        metrics=metrics,
        cells=cells,
    )


def _fmt(value: float | None) -> str:
    return "  -  " if value is None else f"{value:.3f}"


def format_matrix_table(matrix: ComparisonMatrix) -> str:
    """Render a markdown arm-by-metric table. Best cell per metric marked ``*``.
    Synth mode additionally shows ``+/-std`` and a ``(p=...)`` vs the baseline arm.
    The leading ``^``/``v`` marks metric direction (higher- vs lower-is-better)."""
    header = f"# model comparison - {matrix.dataset} ({matrix.mode})"
    if matrix.baseline_arm:
        header += f"  [baseline: {matrix.baseline_arm}]"
    cols = ["metric", *matrix.arms]
    rows: list[list[str]] = []
    for metric in matrix.metrics:
        arrow = "v" if metric_lower_is_better(metric) else "^"
        row = [f"{arrow} {metric}"]
        for arm in matrix.arms:
            cell = matrix.cells[(arm, metric)]
            text = _fmt(cell.value)
            if matrix.mode == "synth" and cell.std is not None:
                text += f"+/-{cell.std:.3f}"
            if cell.p_vs_baseline is not None:
                text += f" (p={cell.p_vs_baseline:.3f})"
            if cell.is_best:
                text += " *"
            row.append(text)
        rows.append(row)
    widths = [max(len(cols[i]), *(len(r[i]) for r in rows)) for i in range(len(cols))]
    def _line(parts: Sequence[str]) -> str:
        return "| " + " | ".join(p.ljust(widths[i]) for i, p in enumerate(parts)) + " |"
    out = [header, "", _line(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
    out += [_line(r) for r in rows]
    out.append("")
    footer = (
        "synth: mean+/-std, p = Welch vs baseline."
        if matrix.mode == "synth"
        else "retrieval: deterministic, 1 run/arm."
    )
    out.append("`*` = best per row (direction-aware). " + footer)
    return "\n".join(out)


def matrix_to_json(matrix: ComparisonMatrix) -> dict[str, Any]:
    """JSON-safe nested dict (non-finite floats → null) for ``comparison.json``."""
    cells: dict[str, dict[str, Any]] = {}
    for arm in matrix.arms:
        cells[arm] = {}
        for metric in matrix.metrics:
            c = matrix.cells[(arm, metric)]
            cells[arm][metric] = {
                "value": c.value,
                "std": c.std,
                "p_vs_baseline": c.p_vs_baseline,
                "is_best": c.is_best,
            }
    return _json_safe(
        {
            "mode": matrix.mode,
            "dataset": matrix.dataset,
            "baseline_arm": matrix.baseline_arm,
            "arms": matrix.arms,
            "metrics": matrix.metrics,
            "cells": cells,
        }
    )


# --------------------------------------------------------------------------- #
# Impure layer (engine-wired; exercised manually with real keys)
# --------------------------------------------------------------------------- #


def _build_arm_embedder(arm: ArmSpec) -> Any:
    from dikw_core.providers import build_embedder

    return build_embedder(arm.provider)


def _build_arm_llm(arm: ArmSpec) -> Any:
    if arm.provider.llm == "openai_codex":
        raise ValueError(
            f"arm {arm.name!r}: openai_codex is unsupported here (OAuth token store, "
            "no env key) — use anthropic_compat or openai_compat arms"
        )
    from dikw_core.providers import build_llm

    return build_llm(arm.provider)


async def run_retrieval_arm(spec: Any, arm: ArmSpec, *, cache_mode: str) -> dict[str, float]:
    from dikw_core.eval.runner import run_eval

    embedder = _build_arm_embedder(arm)
    report = await run_eval(
        spec,
        embedder=embedder,
        provider_config=arm.provider,
        mode="hybrid",
        cache_mode=cache_mode,  # type: ignore[arg-type]
    )
    metrics = dict(report.metrics)
    return {k: float(metrics[k]) for k in _RETRIEVAL_METRICS if k in metrics}


async def run_synth_arm(
    spec: Any,
    arm: ArmSpec,
    *,
    runs: int,
    judge: bool,
    judge_llm: Any = None,
    judge_model: str | None = None,
    judge_sample: int | None = None,
) -> list[dict[str, float]]:
    import asyncio

    import httpx

    from dikw_core.eval.runner import run_synth_eval

    # A synth run fans out many streaming LLM calls (synth + up to 3 judges over
    # every page); a single mid-stream drop (ReadTimeout / RemoteProtocolError —
    # the SDK does not retry an already-consumed stream) would otherwise crash
    # the whole experiment and write nothing. Retry the run on transient network
    # errors so the comparison survives provider flakiness.
    #
    # A per-run hard ceiling via ``asyncio.wait_for`` is the real backstop: some
    # provider stalls hang a socket read PAST the SDK's own timeout (a half-open
    # connection that dribbles no bytes), so the SDK timeout never fires and the
    # run wedges indefinitely. ``wait_for`` cancels + raises ``TimeoutError``,
    # which the same retry arm catches — turning an indefinite hang into a bounded
    # retry. Sized for synth + a sampled judge pass with generous headroom.
    _MAX_ATTEMPTS = 4
    # Sized from measured judged runs: synth + the codex judge over
    # ``--judge-sample 25`` items lands at ~370-540s wall-clock, so the original
    # 360s ceiling mis-flagged every healthy judged run as a hang and then
    # crashed the whole comparison. 900s clears that with headroom while still
    # bounding a genuine half-open stall.
    _PER_RUN_TIMEOUT_S = 900.0

    def _is_transient(exc: BaseException) -> bool:
        if isinstance(exc, (httpx.HTTPError, TimeoutError)):
            return True
        # The Gitee embedder intermittently returns one extra vector ("returned
        # N vectors for M texts") — a bare RuntimeError the embed retry layer
        # does not classify as transient. Pure provider glitch; a fresh-base
        # retry clears it. Other RuntimeErrors (real bugs) still propagate.
        return isinstance(exc, RuntimeError) and "vectors for" in str(exc)

    llm = _build_arm_llm(arm)
    embedder = _build_arm_embedder(arm)
    out: list[dict[str, float]] = []
    for i in range(runs):
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                report = await asyncio.wait_for(
                    run_synth_eval(
                        spec,
                        llm=llm,
                        embedder=embedder,
                        provider_config=arm.provider,
                        judge=judge,
                        judge_llm=judge_llm,
                        judge_model=judge_model,
                        judge_sample=judge_sample,
                    ),
                    timeout=_PER_RUN_TIMEOUT_S,
                )
                break
            except (httpx.HTTPError, TimeoutError, RuntimeError) as exc:
                if not _is_transient(exc) or attempt == _MAX_ATTEMPTS:
                    raise
                label = (
                    f"hung > {_PER_RUN_TIMEOUT_S:.0f}s"
                    if isinstance(exc, TimeoutError)
                    else f"{type(exc).__name__}: {exc}"
                )
                print(
                    f"  [{arm.name}] run {i + 1}/{runs} attempt {attempt} transient "
                    f"failure ({label}); retrying after backoff",
                    file=sys.stderr,
                )
                await asyncio.sleep(5.0 * attempt)
        out.append(flatten_synth_report(report))
        print(f"  [{arm.name}] run {i + 1}/{runs} done", file=sys.stderr)
    return out


def _write_outputs(exp_dir: Path, matrix: ComparisonMatrix, per_arm: Mapping[str, Any]) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    for arm, payload in per_arm.items():
        (exp_dir / f"{arm}.json").write_text(
            json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    (exp_dir / "comparison.json").write_text(
        json.dumps(matrix_to_json(matrix), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _load_spec_file(path: Path) -> Mapping[str, Any]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return raw


async def _cmd_compare(args: argparse.Namespace) -> int:
    from dikw_core.eval.dataset import load_dataset

    dataset, mode, _runs, _judge, arms = parse_arms_spec(_load_spec_file(Path(args.spec)))
    if mode != "retrieval":
        sys.stderr.write(f"error: `compare` needs mode: retrieval (spec says {mode!r})\n")
        return 2
    spec = load_dataset(dataset)
    per_arm: dict[str, dict[str, float]] = {}
    for arm in arms:
        print(f"[retrieval] arm {arm.name} ({arm.provider.embedding_model})", file=sys.stderr)
        per_arm[arm.name] = await run_retrieval_arm(spec, arm, cache_mode=args.cache_mode)
    matrix = build_retrieval_matrix(per_arm, dataset=dataset, arms_order=[a.name for a in arms])
    print(format_matrix_table(matrix))
    _write_outputs(Path(args.exp), matrix, per_arm)
    return 0


async def _cmd_compare_synth(args: argparse.Namespace) -> int:
    from dikw_core.eval.dataset import load_dataset

    dataset, mode, runs, judge, arms = parse_arms_spec(_load_spec_file(Path(args.spec)))
    if mode != "synth":
        sys.stderr.write(f"error: `compare-synth` needs mode: synth (spec says {mode!r})\n")
        return 2
    runs = args.runs if args.runs is not None else runs
    if runs < 1:
        sys.stderr.write(f"error: --runs must be >= 1 (got {runs})\n")
        return 2
    judge = judge or args.judge
    spec = load_dataset(dataset)
    if "synth" not in spec.modes:
        sys.stderr.write(f"error: dataset {dataset!r} does not declare 'synth' mode\n")
        return 2

    # Optional neutral judge: a top-level ``judge_provider:`` block routes ALL
    # judge legs (grounding / entailment / category / wikilink / atomicity) to
    # one shared model instead of each arm grading itself — removing the
    # self-evaluation bias that makes the judge dims otherwise non-comparable
    # across arms. ``openai_codex`` is allowed here (unlike the synth arms): the
    # judge needs no env key, it reads its OAuth token from ``<base_root>/.dikw``.
    judge_llm: Any = None
    judge_model: str | None = None
    if judge:
        raw = _load_spec_file(Path(args.spec))
        jp = raw.get("judge_provider")
        if isinstance(jp, Mapping):
            from dikw_core.config import ProviderConfig
            from dikw_core.providers import build_llm

            judge_cfg = ProviderConfig.model_validate(dict(jp))
            judge_llm = build_llm(judge_cfg, base_root=_REPO_ROOT)
            judge_model = judge_cfg.llm_model
            print(
                f"[synth] neutral judge: {judge_cfg.llm} ({judge_model})", file=sys.stderr
            )

    per_arm: dict[str, list[dict[str, float]]] = {}
    for arm in arms:
        print(f"[synth] arm {arm.name} ({arm.provider.llm_model}), {runs} run(s)", file=sys.stderr)
        per_arm[arm.name] = await run_synth_arm(
            spec,
            arm,
            runs=runs,
            judge=judge,
            judge_llm=judge_llm,
            judge_model=judge_model,
            judge_sample=args.judge_sample,
        )
    matrix = build_synth_matrix(
        per_arm, dataset=dataset, arms_order=[a.name for a in arms], baseline_arm=arms[0].name
    )
    print(format_matrix_table(matrix))
    _write_outputs(Path(args.exp), matrix, per_arm)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Horizontal model comparison harness.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="retrieval comparison (embed arms, 1 run each)")
    c.add_argument("--spec", required=True, help="arms-spec YAML (mode: retrieval)")
    c.add_argument("--exp", required=True, help="output dir under evals/experiments/")
    # Default ``off``: the corpus cache keys on (model, dim) only, so two arms
    # sharing a model but differing in revision/normalize/distance would silently
    # reuse one snapshot (see eval.runner._corpus_cache_key's own ablation note).
    # ``read_write`` is a safe speed-up only when arms differ by embedding_model.
    c.add_argument(
        "--cache-mode",
        default="off",
        choices=["read_write", "rebuild", "off"],
        help="corpus snapshot cache (default off — safe for arm comparison)",
    )

    s = sub.add_parser("compare-synth", help="synth comparison (LLM arms, N runs + Welch)")
    s.add_argument("--spec", required=True, help="arms-spec YAML (mode: synth)")
    s.add_argument("--exp", required=True, help="output dir under evals/experiments/")
    s.add_argument("--runs", type=int, default=None, help="runs per arm (overrides spec)")
    s.add_argument("--judge", action="store_true", help="run the LLM grounding judge")
    s.add_argument(
        "--judge-sample",
        type=int,
        default=None,
        help="cap each judge leg to N items (pages / claims) — bounds the judge "
        "call count + wall-clock for slow reasoning judges (default: judge all)",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    import asyncio

    args = build_parser().parse_args(argv)
    if args.cmd == "compare":
        return asyncio.run(_cmd_compare(args))
    if args.cmd == "compare-synth":
        return asyncio.run(_cmd_compare_synth(args))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
