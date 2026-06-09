# Eval baselines

Machine-readable baselines for the `dikw client eval --against` regression gate.
Each file pins a known-good run's metrics so a later run can be failed (exit 1)
when a metric moves the wrong way beyond a tolerance.

This is a single-run regression **gate**, not an A/B significance test. The
statistical A/B path (Welch t-test over sample distributions) lives in
[`../tools/ab_experiment.py`](../tools/ab_experiment.py) — it needs per-query /
multi-run samples this gate deliberately does not collect. Prose baselines (the
real-data outcome narrative gated by `eval-gate.yml`) live in
[`../BASELINES.md`](../BASELINES.md); the JSON files here are the machine-readable
companion.

## File format

```json
{
  "dataset": "mvp",
  "modes": ["synth"],
  "created": "2026-06-09",
  "tolerance": 0.02,
  "metrics": {
    "synth/fallback_ratio_max": 0.10,
    "synth/source_chunk_coverage": 0.83
  }
}
```

- `metrics` — a flat `{name: value}` map of scalar metrics. Non-scalar
  diagnostics (e.g. `category_distribution`) are ignored by the gate.
- `tolerance` — absolute per-metric noise floor; a metric must move *more* than
  this (in the wrong direction) to count as a regression. Omitted → `0.02`.
- The direction is read from the metric **name**: a bare name ending in `_max`
  is lower-is-better (it regresses when it rises); everything else is
  higher-is-better. This mirrors the engine's convention
  (`eval.runner._threshold_direction`).
- `dataset` / `modes` / `created` are provenance only (not read by the gate).

## Generate and use

```bash
# Write a baseline from a canonical run (single --dataset + one --eval mode).
uv run dikw client eval --dataset mvp --eval synth --write-baseline evals/baselines/mvp-synth.json

# Commit it, then gate later runs (exit 1 on regression):
uv run dikw client eval --dataset mvp --eval synth --against evals/baselines/mvp-synth.json
```

`--against` / `--write-baseline` imply `--wait` and require a **single**
`--dataset` plus one `--eval` mode, so the result carries exactly one metrics
set. Regenerate a baseline (re-run `--write-baseline`) whenever an intended,
reviewed improvement lands — the committed JSON is the reviewable record of what
"good" means.
