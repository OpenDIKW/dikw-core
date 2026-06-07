"""Test-the-test for the eval-gate content check (``tools/check_baselines.py``).

The gate exists to kill a false-green: the old eval-gate only asked *was
BASELINES.md diffed?*, which a blank-line edit satisfied. These cases assert the
content check goes RED on the bad inputs (empty edit, no new header, a *reused*
header, K-layer entry with neither signal nor a non-destructiveness claim,
missing the elon-musk corpus, retrieval entry with no ablation numbers, prose
mentions without values) and GREEN on real entries — a synth A/B entry, the
sparse 5-metric author practice, a lint-fix entry, and an additive/non-
destructive entry (the #122 shape) — so the gate never regresses into blocking
legitimate K-layer practice.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.check_baselines import (  # noqa: E402
    _MIN_SYNTH_METRICS,
    _SYNTH_METRICS,
    check_baseline_addition,
)

# Stable substring of the K-layer quantitative-leg violation (signal-or-non-
# destructiveness), used to assert that leg fired (or didn't).
_QUANT_LEG = "neither signal nor"


def _added(text: str) -> list[str]:
    """Diff-added lines (leading ``+`` already stripped), as the workflow passes them."""
    return text.strip("\n").split("\n")


def _check(
    lines: list[str],
    *,
    kn: bool = False,
    info: bool = False,
    storage: bool = False,
    existing: tuple[str, ...] = (),
) -> list[str]:
    return check_baseline_addition(
        lines,
        existing_headers=set(existing),
        touches_knowledge=kn,
        touches_info=info,
        touches_storage=storage,
    )


# A complete, well-formed K-layer baseline entry: header + all 7 synth metrics +
# elon-musk corpus reference.
_GOOD_KLAYER = _added(
    """
## 2026-06-08 — synth tweak A/B

Provider: openai_codex (gpt-5.5) + Qwen3-Embedding-0.6B@1024.

| metric | value | threshold |
|---|---|---|
| synth/fact_grounding_ratio | 0.787 | 0.55 |
| synth/atomicity_score | 1.000 | 0.85 |
| synth/duplicate_ratio_max | 0.000 | 0.05 |
| synth/wikilink_resolved_ratio | 0.588 | 0.55 |
| synth/language_fidelity | 1.000 | 0.95 |
| synth/expected_coverage | 0.333 | (info) |
| synth/page_density | 0.571 | (info) |

elon-musk.md 1500-line subset: 76 pages, 0 errors.
"""
)

# A faithful trim of the 2026-06-07 merged entry — names exactly FIVE synth
# metrics on numeric lines, the sparsest real K-layer synth entry. Must still
# pass: the floor must not block current practice.
_GOOD_KLAYER_SPARSE = _added(
    """
## 2026-06-07 — existing-pages slug + priority-create A/B (Phase 2)

| metric | baseline | intervention |
|---|---|---|
| synth/wikilink_resolved_ratio | 0.4784 | 0.4380 |
| synth/fact_grounding_ratio | 0.5775 | 0.5683 |
| synth/page_density | 0.9048 | 0.9206 |

elon-musk 1500-line subset: atomicity_score 1.000, duplicate_ratio_max 0.00036.
"""
)

# A lint-fix K-layer entry (the merged #82/#83 shape): elon-musk corpus + lint
# issue-kind counts and TP/FP, NO synth metrics. Must pass via the lint-outcome
# signal leg (docs/eval-plan.md lists this as the primary K-layer baseline shape).
_GOOD_KLAYER_LINT = _added(
    """
## 2026-05-14 — orphan-page governance v2

Base: elon-musk-validation. Lint baseline: 135 issues.
By kind: broken_wikilink=96, orphan_page=39.
Proposals: 39/39 orphans, strategy link_from_existing=39.
TP/FP spot check: 5/5 accepted.
"""
)

# An additive / non-destructive K-layer entry (the merged #122 shape): no synth
# metrics, no numeric lint counts — the lint kinds appear only in methodology
# prose — but an explicit non-destructiveness claim. Must pass via the
# non-destructiveness leg (eval-plan.md: "either signal or non-destructiveness").
_GOOD_KLAYER_ADDITIVE = _added(
    """
## 2026-05-26 — wisdom as a retrieval layer (#122)

Status: purely additive — wisdom files now surface in retrieve; wiki + source
ranking is untouched. Against the packaged elon-musk-validation base (ships zero
wisdom files), the retrieval eval is byte-identical to the prior baseline.
Extends broken_wikilink / orphan_page lint to wisdom pages; unit-tested.
"""
)

_GOOD_RETRIEVAL = _added(
    """
## 2026-06-08 — RRF weight sweep on scifact

| mode | hit_at_3 | hit_at_10 | mrr | ndcg_at_10 | recall_at_100 |
|---|---|---|---|---|---|
| hybrid | 0.98 | 0.99 | 0.96 | 0.93 | 0.98 |
"""
)

_GOOD_STORAGE = _added(
    """
## 2026-06-08 — postgres migration 0007 backfill

Ran the contract suite against pgvector/pgvector:pg18 — all green. No retrieval
behavior change; documents the real-Postgres round-trip for the migration.
"""
)


# ---- GREEN: real entries pass for their change type ----------------------


def test_good_klayer_entry_passes() -> None:
    assert _check(_GOOD_KLAYER, kn=True) == []


def test_sparse_real_klayer_entry_still_passes() -> None:
    # Anti-brittleness guard: the 5-metric 2026-06-07 shape must not be blocked.
    assert _check(_GOOD_KLAYER_SPARSE, kn=True) == []


def test_lint_fix_klayer_entry_passes() -> None:
    # Anti-brittleness guard: a lint-fix baseline (no synth metrics) passes via
    # the lint-outcome signal — the merged #82/#83 PR class.
    assert _check(_GOOD_KLAYER_LINT, kn=True) == []


def test_additive_nondestructive_klayer_entry_passes() -> None:
    # Anti-brittleness guard: an additive change with no numbers passes via the
    # non-destructiveness claim — the merged #122 PR class.
    assert _check(_GOOD_KLAYER_ADDITIVE, kn=True) == []


def test_good_retrieval_entry_passes() -> None:
    assert _check(_GOOD_RETRIEVAL, info=True) == []


def test_good_storage_entry_passes() -> None:
    assert _check(_GOOD_STORAGE, storage=True) == []


def test_semver_header_recognized() -> None:
    # Version headers (`## 0.5.0 — …`) are valid new entries too.
    lines = _added(
        """
## 0.6.0 — taxonomy v2

| synth/fact_grounding_ratio | 0.78 |
| synth/atomicity_score | 1.0 |
| synth/wikilink_resolved_ratio | 0.6 |

elon-musk subset verified (76 pages).
"""
    )
    assert _check(lines, kn=True) == []


# ---- RED: the false-green inputs the gate exists to catch -----------------


def test_empty_edit_is_flagged() -> None:
    # The original false-green: a blank-line / no-real-content edit.
    violations = _check(_added("\n   \n"), kn=True)
    assert any("no NEW dated/versioned entry" in v for v in violations)


def test_untouched_file_is_flagged() -> None:
    violations = _check([], storage=True)
    assert any("no NEW dated/versioned entry" in v for v in violations)


def test_reused_header_is_flagged() -> None:
    # The headline guarantee: a copy-pasted / stale header (already in the base
    # file) is NOT a new entry, even with valid metrics + corpus below it.
    header = "## 2026-06-07 — existing-pages slug + priority-create A/B (Phase 2)"
    lines = _added(
        f"""
{header}

| synth/fact_grounding_ratio | 0.78 |
| synth/atomicity_score | 1.0 |
| synth/wikilink_resolved_ratio | 0.6 |
elon-musk re-run (76 pages).
"""
    )
    violations = _check(lines, kn=True, existing=(header,))
    assert any("no NEW dated/versioned entry" in v for v in violations)
    # ...and the very same body under a genuinely new header passes.
    assert _check(lines, kn=True, existing=("## 2099-01-01 — unrelated",)) == []


def test_header_without_date_or_version_is_flagged() -> None:
    # A `## Notes` header is not a dated/versioned entry.
    lines = _added(
        """
## Notes

synth/fact_grounding_ratio 0.1 synth/atomicity_score 0.2 synth/duplicate_ratio_max 0.3
elon-musk 0
"""
    )
    violations = _check(lines, kn=True)
    assert any("no NEW dated/versioned entry" in v for v in violations)


def test_header_mentioning_date_elsewhere_is_flagged() -> None:
    # The date/semver must LEAD the header, not merely appear on the line.
    lines = _added(
        """
## TODO before 2026-12-31 deadline

synth/fact_grounding_ratio 0.1, synth/atomicity_score 0.2, synth/page_density 0.3
elon-musk (1)
"""
    )
    violations = _check(lines, kn=True)
    assert any("no NEW dated/versioned entry" in v for v in violations)


def test_edit_to_existing_entry_is_flagged() -> None:
    # Adding metric rows to an OLD entry adds no `## ` header — must fail even
    # though synth metrics + elon-musk are present.
    lines = _added(
        """
| synth/fact_grounding_ratio | 0.79 |
| synth/atomicity_score | 1.0 |
| synth/wikilink_resolved_ratio | 0.6 |
elon-musk subset re-run (76 pages).
"""
    )
    violations = _check(lines, kn=True)
    assert any("no NEW dated/versioned entry" in v for v in violations)


def test_klayer_missing_elon_musk_is_flagged() -> None:
    lines = _added(
        """
## 2026-06-08 — synth tweak

synth/fact_grounding_ratio 0.78, synth/atomicity_score 1.0,
synth/wikilink_resolved_ratio 0.6 on the mvp corpus.
"""
    )
    violations = _check(lines, kn=True)
    assert any("elon-musk" in v for v in violations)
    # ...the quantitative leg is satisfied here (3 metrics), so it's the only miss.
    assert not any(_QUANT_LEG in v for v in violations)


def test_klayer_no_signal_no_nondestructive_is_flagged() -> None:
    lines = _added(
        """
## 2026-06-08 — prose-only synth note

We changed the synth prompt. It feels better. Ran it on elon-musk and the pages
look more atomic. synth/atomicity_score improved to 0.9.
"""
    )
    violations = _check(lines, kn=True)
    assert any(_QUANT_LEG in v for v in violations)


def test_klayer_metrics_only_in_prose_without_values_is_flagged() -> None:
    # Metric NAMES with no numbers on their lines do not count, and the entry
    # makes no non-destructiveness claim — a baseline reports values or asserts
    # non-destructiveness, not adjectives.
    lines = _added(
        """
## 2026-06-08 — vibes-based synth note

On elon-musk, fact_grounding_ratio and atomicity_score and wikilink_resolved_ratio
all looked fine to me. Shipping. Numbers omitted for brevity.
"""
    )
    violations = _check(lines, kn=True)
    assert any(_QUANT_LEG in v for v in violations)


def test_retrieval_without_metrics_is_flagged() -> None:
    lines = _added(
        """
## 2026-06-08 — retrieval refactor

Refactored the fusion code. It runs. No numbers recorded.
"""
    )
    violations = _check(lines, info=True)
    assert any("retrieval metric" in v for v in violations)


def test_retrieval_metric_word_in_prose_is_flagged() -> None:
    # Bare "recall"/"ndcg" as prose words must not satisfy the retrieval leg —
    # only value-bearing tokens (recall@k / ndcg_at_10 / hit_at_3 / mrr) count.
    lines = _added(
        """
## 2026-06-08 — fusion refactor

Refactored RRF. Recall that the prior weights were arbitrary; behavior unchanged.
"""
    )
    violations = _check(lines, info=True)
    assert any("retrieval metric" in v for v in violations)


def test_combined_knowledge_and_info_requires_both() -> None:
    # A diff touching both layers: a retrieval-only entry misses the K-layer legs.
    violations = _check(_GOOD_RETRIEVAL, kn=True, info=True)
    assert any(_QUANT_LEG in v for v in violations)
    assert any("elon-musk" in v for v in violations)
    # the retrieval leg, however, is satisfied by _GOOD_RETRIEVAL
    assert not any("retrieval metric" in v for v in violations)


def test_floor_constant_is_sane() -> None:
    # Guard the floor stays a strict-but-reachable subset of the seven metrics.
    assert 1 <= _MIN_SYNTH_METRICS <= len(_SYNTH_METRICS)
    assert len(_SYNTH_METRICS) == 7
