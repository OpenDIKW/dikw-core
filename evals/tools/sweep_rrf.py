"""Offline RRF weight sweep for retrieval evals.

Reads a JSONL dump of per-(query, mode) ranked lists — one row per line
with fields ``dataset``, ``mode`` (``bm25`` / ``vector`` / ``hybrid``),
``q_id``, ``q``, ``expect_any``, ``expect_none``, ``ranked`` — re-fuses
the bm25 and vector legs at arbitrary ``(rrf_k, bm25_weight,
vector_weight)`` combinations, and re-scores every positive query with
the existing metric helpers. This lets us find the fusion knobs that
maximise a metric (e.g. nDCG@10) against a target dataset **without
re-running embedding** — the expensive part of a real-vector eval.

This dump is no longer auto-emitted by ``dikw client eval`` (the
``--dump-raw`` flag was dropped in the client/server split); prepare it
by hand from the retrieval primitives if you want to sweep.

Usage::

    uv run python evals/tools/sweep_rrf.py \\
        --raw-dump /tmp/scifact-raw.jsonl \\
        --top-n 10

Prints the top-N (rrf_k, bm25_w, vec_w) combinations by nDCG@10, plus
a pinned row for the equal-weight (1.0, 1.0) baseline so the winner's
absolute delta is obvious. Does not mutate any source of truth — purely
a research tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

# Local import — we call into the production RRF so the sweep uses the
# same code path production does.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from dikw_core.config import RetrievalConfig  # noqa: E402
from dikw_core.domains.info.search import reciprocal_rank_fusion  # noqa: E402
from dikw_core.eval.metrics import (  # noqa: E402
    mean_hit_at_k,
    mean_ndcg_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
)

DEFAULT_K_GRID = (40, 60, 100)
DEFAULT_W_BM25_GRID = (0.3, 0.5, 0.7, 1.0)
DEFAULT_W_VEC_GRID = (0.5, 1.0, 1.5, 2.0)


@dataclass(frozen=True)
class QueryLegs:
    """One query's rankings from both retrieval legs."""

    q: str
    expect_any: list[str]
    bm25_ranked: list[str]
    vector_ranked: list[str]


@dataclass(frozen=True)
class SweepResult:
    rrf_k: int
    bm25_weight: float
    vector_weight: float
    hit_at_3: float
    hit_at_10: float
    mrr: float
    ndcg_at_10: float
    recall_at_100: float


def load_raw_dump(path: Path) -> dict[str, list[QueryLegs]]:
    """Parse a ranked-list JSONL dump into per-dataset query legs.

    Drops negatives (no positive relevance set to score against). Keys
    by ``q_id`` so duplicate query texts don't collide. No back-compat
    for ``q_id``-less dumps — re-dump.
    """

    class _Bundle(TypedDict):
        bm25: list[str]
        vector: list[str]
        expect_any: list[str]
        q: str

    def _new_bundle() -> _Bundle:
        return {"bm25": [], "vector": [], "expect_any": [], "q": ""}

    per_dataset: dict[str, dict[int, _Bundle]] = defaultdict(
        lambda: defaultdict(_new_bundle)
    )

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("expect_none"):
                continue
            ds = row["dataset"]
            qid = row["q_id"]
            mode = row["mode"]
            # Hybrid rankings can't drive sweep re-fusion (we re-fuse the legs).
            if mode == "hybrid":
                continue
            bundle = per_dataset[ds][qid]
            if mode in ("bm25", "vector"):
                bundle[mode] = list(row["ranked"])
            bundle["expect_any"] = list(row["expect_any"])
            bundle["q"] = row["q"]

    result: dict[str, list[QueryLegs]] = {}
    for ds, qmap in per_dataset.items():
        legs: list[QueryLegs] = []
        for qid in sorted(qmap):
            bundle = qmap[qid]
            if not bundle["bm25"] and not bundle["vector"]:
                continue
            legs.append(
                QueryLegs(
                    q=bundle["q"],
                    expect_any=list(bundle["expect_any"]),
                    bm25_ranked=list(bundle["bm25"]),
                    vector_ranked=list(bundle["vector"]),
                )
            )
        result[ds] = legs
    return result


def evaluate_weights(
    legs: Iterable[QueryLegs], *, rrf_k: int, bm25_weight: float, vector_weight: float
) -> SweepResult:
    """Fuse each query's legs with the given weights and re-score."""
    pairs: list[tuple[list[str], list[str]]] = []
    for ql in legs:
        fused = reciprocal_rank_fusion(
            [ql.bm25_ranked, ql.vector_ranked],
            k=rrf_k,
            weights=[bm25_weight, vector_weight],
        )
        ranked = [doc for doc, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
        pairs.append((ranked, ql.expect_any))

    return SweepResult(
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        hit_at_3=mean_hit_at_k(pairs, 3),
        hit_at_10=mean_hit_at_k(pairs, 10),
        mrr=mean_reciprocal_rank(pairs),
        ndcg_at_10=mean_ndcg_at_k(pairs, 10),
        recall_at_100=mean_recall_at_k(pairs, 100),
    )


def sweep(
    legs: Sequence[QueryLegs],
    *,
    k_grid: Sequence[int] = DEFAULT_K_GRID,
    w_bm25_grid: Sequence[float] = DEFAULT_W_BM25_GRID,
    w_vec_grid: Sequence[float] = DEFAULT_W_VEC_GRID,
) -> list[SweepResult]:
    """Grid-search every (k, w_bm25, w_vec) combination. Order-stable."""
    results: list[SweepResult] = []
    for k in k_grid:
        for wb in w_bm25_grid:
            for wv in w_vec_grid:
                results.append(
                    evaluate_weights(
                        legs, rrf_k=k, bm25_weight=wb, vector_weight=wv
                    )
                )
    return results


def format_table(
    results: Sequence[SweepResult],
    legs: Sequence[QueryLegs] | None = None,
    *,
    top_n: int = 10,
) -> str:
    """Render the top-N by nDCG@10 plus two pinned reference rows.

    Reference rows: the equal-weight (1,1,60) row — the "before tuning"
    comparison point — and the current ``RetrievalConfig()`` defaults —
    what ``dikw query`` actually ships with. Showing both tells the user
    both "how much tuning bought us vs vanilla RRF" and "how far my
    corpus's best config sits from the shipped default".

    ``legs`` lets the formatter compute the current-default row on the
    fly when it's not in the grid (e.g. if defaults shift to 0.25 and
    the grid has 0.30). Pass ``None`` to skip that — the legacy path
    that just reports whichever rows landed in the grid.

    Layout matches the other ablation tables in the eval CLI output —
    markdown-ish pipe columns, fixed 3-digit precision so a 0.002 nDCG@10
    difference is visible.
    """
    header = (
        "|   k | w_bm25 | w_vec | hit@3 | hit@10 |    mrr | nDCG@10 | recall@100 |"
    )
    sep = (
        "|----:|-------:|------:|------:|-------:|-------:|--------:|-----------:|"
    )

    def row(r: SweepResult, *, tag: str = "") -> str:
        return (
            f"| {r.rrf_k:>3d} | {r.bm25_weight:>6.2f} | {r.vector_weight:>5.2f} | "
            f"{r.hit_at_3:>5.3f} | {r.hit_at_10:>6.3f} | {r.mrr:>6.3f} | "
            f"{r.ndcg_at_10:>7.3f} | {r.recall_at_100:>10.3f} |"
            + (f"  {tag}" if tag else "")
        )

    ordered = sorted(results, key=lambda r: -r.ndcg_at_10)
    picked = ordered[:top_n]

    def find_or_compute(rrf_k: int, wb: float, wv: float) -> SweepResult | None:
        for r in results:
            if r.rrf_k == rrf_k and r.bm25_weight == wb and r.vector_weight == wv:
                return r
        if legs is None:
            return None
        return evaluate_weights(legs, rrf_k=rrf_k, bm25_weight=wb, vector_weight=wv)

    vanilla = find_or_compute(60, 1.0, 1.0)
    default_cfg = RetrievalConfig()
    shipped = find_or_compute(
        default_cfg.rrf_k, default_cfg.bm25_weight, default_cfg.vector_weight
    )

    lines: list[str] = [f"Top {len(picked)} by nDCG@10:", header, sep]
    for r in picked:
        lines.append(row(r))

    reference_rows: list[tuple[SweepResult, str]] = []
    if vanilla is not None:
        reference_rows.append((vanilla, "← equal-weight (pre-tuning)"))
    if shipped is not None and (
        vanilla is None
        or (shipped.rrf_k, shipped.bm25_weight, shipped.vector_weight)
        != (vanilla.rrf_k, vanilla.bm25_weight, vanilla.vector_weight)
    ):
        reference_rows.append((shipped, "← shipped default"))

    if reference_rows:
        lines.append("")
        lines.append("Reference points:")
        lines.append(header)
        lines.append(sep)
        for r, tag in reference_rows:
            lines.append(row(r, tag=tag))
    return "\n".join(lines)


def _cli(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--raw-dump",
        type=Path,
        required=True,
        help="Per-(query, mode) ranked-list JSONL dump (see module docstring).",
    )
    p.add_argument(
        "--top-n", type=int, default=10, help="How many rows to show (default: 10)."
    )
    args = p.parse_args(argv)

    per_ds = load_raw_dump(args.raw_dump)
    if not per_ds:
        print(f"error: no positive query rows found in {args.raw_dump}", file=sys.stderr)
        return 2

    for ds, legs in per_ds.items():
        print(f"\n=== {ds}  ({len(legs)} positive queries) ===")
        results = sweep(legs)
        print(format_table(results, legs, top_n=args.top_n))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
