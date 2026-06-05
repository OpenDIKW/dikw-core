"""Hermetic K-layer eval gate.

Smoke-tests ``run_synth_eval`` end-to-end with ``FakeLLM`` +
``FakeEmbeddings`` — no API keys, no network. The dataset is built
in-process so the test stays independent of changes to packaged
datasets under ``evals/datasets/``.

Per spec: hermetic mode does NOT enforce realistic thresholds. The
test only verifies the runner wires up correctly, every K-layer
metric is present and in ``[0, 1]`` range, and the report's
structural invariants hold (``n_pages > 0``, ``threshold_results``
matches declared thresholds, etc.). Realistic thresholds get
calibrated separately by ``dikw client eval mvp --eval synth`` against the
real LLM in Step 8 dogfood.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dikw_core.eval.dataset import load_dataset
from dikw_core.eval.fake_embedder import FakeEmbeddings
from dikw_core.eval.runner import (
    SynthEvalError,
    SynthEvalReport,
    run_synth_eval,
)
from dikw_core.providers.base import LLMResponse

from .fakes import FakeLLM

_SYNTH_PAGE_RESPONSE = """<page path="knowledge/concepts/alpha.md" type="concept">
---
tags: [topic/sample]
---

# Alpha

Alpha is a concept described in the corpus.
</page>
"""

_SYNTH_BETA_RESPONSE = """<page path="knowledge/concepts/beta.md" type="concept">
---
tags: [topic/sample]
---

# Beta

Beta is another concept the test corpus mentions.
</page>
"""


def _write_synth_dataset(root: Path, *, name: str = "synth-toy") -> Path:
    """Build a minimal synth-eval dataset; permissive thresholds + a
    single concept page_type so the FakeLLM responses parse cleanly."""
    ds = root / name
    (ds / "corpus").mkdir(parents=True, exist_ok=True)
    (ds / "corpus" / "alpha.md").write_text(
        "# Alpha\n\nAlpha is a concept described in the corpus.\n",
        encoding="utf-8",
    )
    (ds / "corpus" / "beta.md").write_text(
        "# Beta\n\nBeta is another concept the test corpus mentions.\n",
        encoding="utf-8",
    )
    (ds / "dataset.yaml").write_text(
        "name: " + name + "\n"
        "modes: [synth]\n"
        "thresholds:\n"
        "  synth/fact_grounding_ratio: 0.0\n"
        "  synth/atomicity_score: 0.0\n"
        "  synth/duplicate_ratio_max: 1.0\n"
        "  synth/wikilink_resolved_ratio: 0.0\n"
        "  synth/language_fidelity: 0.0\n"
        "synth:\n"
        "  page_types: [concept]\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n  - q: about alpha\n    expect_any: [alpha]\n",
        encoding="utf-8",
    )
    return ds


@pytest.mark.asyncio
async def test_run_synth_eval_smokes_full_pipeline(tmp_path: Path) -> None:
    ds = _write_synth_dataset(tmp_path)
    spec = load_dataset(ds)
    llm = FakeLLM(responses=[_SYNTH_PAGE_RESPONSE, _SYNTH_BETA_RESPONSE])
    embedder = FakeEmbeddings()

    report = await run_synth_eval(spec, llm=llm, embedder=embedder)

    assert isinstance(report, SynthEvalReport)
    assert report.mode == "synth"
    assert report.dataset_name == "synth-toy"
    assert report.n_sources == 2
    assert report.n_pages > 0

    expected_metrics = {
        "synth/fact_grounding_ratio",
        "synth/atomicity_score",
        "synth/duplicate_ratio_max",
        "synth/wikilink_resolved_ratio",
        "synth/language_fidelity",
    }
    for m in expected_metrics:
        assert m in report.metrics, f"missing metric: {m}"
        assert 0.0 <= report.metrics[m] <= 1.0, (
            f"{m} out of [0, 1]: {report.metrics[m]}"
        )

    # Threshold result keys mirror what dataset.yaml declared.
    assert {r.name for r in report.threshold_results} == expected_metrics
    # Permissive thresholds → every gate passes.
    assert report.passed is True

    # page_density lives in informational, never gated.
    assert "synth/page_density" in report.informational


@pytest.mark.asyncio
async def test_run_synth_eval_zero_pages_raises(tmp_path: Path) -> None:
    """LLM returns garbage → parser yields zero pages → SynthEvalError
    with the allowed page_types hint."""
    ds = _write_synth_dataset(tmp_path)
    spec = load_dataset(ds)
    llm = FakeLLM(response_text="totally not a synth response")
    embedder = FakeEmbeddings()

    with pytest.raises(SynthEvalError, match="zero pages"):
        await run_synth_eval(spec, llm=llm, embedder=embedder)


@pytest.mark.asyncio
async def test_run_synth_eval_judge_off_by_default(tmp_path: Path) -> None:
    ds = _write_synth_dataset(tmp_path)
    spec = load_dataset(ds)
    llm = FakeLLM(responses=[_SYNTH_PAGE_RESPONSE, _SYNTH_BETA_RESPONSE])

    report = await run_synth_eval(spec, llm=llm, embedder=FakeEmbeddings())
    assert report.judge_summary is None


@pytest.mark.asyncio
async def test_run_synth_eval_rejects_non_synth_dataset(tmp_path: Path) -> None:
    """A dataset declaring only retrieval mode rejects synth eval up
    front rather than silently doing nothing."""
    ds = tmp_path / "no-synth"
    (ds / "corpus").mkdir(parents=True)
    (ds / "corpus" / "a.md").write_text("# A\n\nbody\n", encoding="utf-8")
    (ds / "dataset.yaml").write_text(
        "name: no-synth\nthresholds:\n  hit_at_3: 0.5\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n  - q: a\n    expect_any: [a]\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)

    with pytest.raises(SynthEvalError, match="does not declare 'synth' mode"):
        await run_synth_eval(
            spec, llm=FakeLLM(), embedder=FakeEmbeddings()
        )


@pytest.mark.asyncio
async def test_run_synth_eval_without_synth_thresholds_is_ungated(
    tmp_path: Path,
) -> None:
    """A dataset that declares synth mode but no synth thresholds runs
    as informational — ``gated=False`` so aggregate callers can tell
    "no checks declared" apart from "all checks passed". ``passed``
    stays ``True`` (vacuously) for back-compat with renderers that
    treat it as a bool, but ``gated`` is the load-bearing signal."""
    ds = tmp_path / "ungated"
    (ds / "corpus").mkdir(parents=True)
    (ds / "corpus" / "alpha.md").write_text(
        "# Alpha\n\nAlpha content.\n", encoding="utf-8"
    )
    (ds / "dataset.yaml").write_text(
        "name: ungated\n"
        "modes: [synth]\n"
        "synth:\n"
        "  page_types: [concept]\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n  - q: a\n    expect_any: [alpha]\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    llm = FakeLLM(response_text=_SYNTH_PAGE_RESPONSE)
    report = await run_synth_eval(spec, llm=llm, embedder=FakeEmbeddings())
    assert report.threshold_results == []
    assert report.gated is False
    assert report.passed is True  # vacuous all([]) — guarded by ``gated``


@pytest.mark.asyncio
async def test_run_synth_eval_missing_metric_serialises_to_null(
    tmp_path: Path,
) -> None:
    """A threshold whose metric was not computed (e.g. ``expected_coverage``
    declared without an ``expected.yaml``) lands as ``observed=None`` so
    ``model_dump(mode="json")`` emits ``null`` instead of the invalid
    JSON token ``NaN`` (which Postgres JSONB and strict parsers reject)."""
    ds = tmp_path / "missing"
    (ds / "corpus").mkdir(parents=True)
    (ds / "corpus" / "alpha.md").write_text(
        "# Alpha\n\nAlpha content.\n", encoding="utf-8"
    )
    (ds / "dataset.yaml").write_text(
        "name: missing\n"
        "modes: [synth]\n"
        "thresholds:\n"
        # expected_coverage requires expected.yaml; we don't provide
        # one, so the metric is never computed → check_thresholds
        # records a miss with observed=None.
        "  synth/expected_coverage: 0.5\n"
        "synth:\n"
        "  page_types: [concept]\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n  - q: a\n    expect_any: [alpha]\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    llm = FakeLLM(response_text=_SYNTH_PAGE_RESPONSE)
    report = await run_synth_eval(spec, llm=llm, embedder=FakeEmbeddings())
    miss = next(
        r for r in report.threshold_results
        if r.name == "synth/expected_coverage"
    )
    assert miss.observed is None
    assert miss.passed is False
    # Round-trip through JSON-mode dump must not crash on NaN.
    import json
    payload = json.dumps(report.model_dump(mode="json"))
    assert '"observed":null' in payload or '"observed": null' in payload


@pytest.mark.asyncio
async def test_run_synth_eval_ignores_retrieval_thresholds(
    tmp_path: Path,
) -> None:
    """Mixed thresholds — retrieval keys must NOT make synth ``passed``
    flip to False. ``run_synth_eval`` only sees synth metrics, so the
    retrieval thresholds would otherwise show up as ``missing →
    passed=False`` and the gate would always fail on mvp-style datasets.
    """
    ds = tmp_path / "mixed"
    (ds / "corpus").mkdir(parents=True)
    (ds / "corpus" / "alpha.md").write_text(
        "# Alpha\n\nAlpha is a concept described in the corpus.\n",
        encoding="utf-8",
    )
    (ds / "dataset.yaml").write_text(
        "name: mixed\n"
        "modes: [retrieval, synth]\n"
        "thresholds:\n"
        "  hit_at_3: 0.5\n"
        "  hit_at_10: 0.5\n"
        "  mrr: 0.3\n"
        "  synth/fact_grounding_ratio: 0.0\n"
        "synth:\n"
        "  page_types: [concept]\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n  - q: alpha\n    expect_any: [alpha]\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    llm = FakeLLM(response_text=_SYNTH_PAGE_RESPONSE)
    report = await run_synth_eval(spec, llm=llm, embedder=FakeEmbeddings())
    assert {r.name for r in report.threshold_results} == {
        "synth/fact_grounding_ratio"
    }
    assert report.passed is True


@pytest.mark.asyncio
async def test_run_synth_eval_threshold_failure_blocks_passed(
    tmp_path: Path,
) -> None:
    """A pessimistic threshold makes ``passed`` False — direction-aware
    check_thresholds wires through to the report's pass flag."""
    ds = tmp_path / "tight"
    (ds / "corpus").mkdir(parents=True)
    (ds / "corpus" / "alpha.md").write_text(
        "# Alpha\n\nAlpha content.\n", encoding="utf-8"
    )
    (ds / "dataset.yaml").write_text(
        "name: tight\n"
        "modes: [synth]\n"
        "thresholds:\n"
        # Impossible bar: every page must have a perfect grounding score.
        "  synth/fact_grounding_ratio: 1.0\n"
        "synth:\n"
        "  page_types: [concept]\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n  - q: a\n    expect_any: [a]\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    # Body that doesn't echo source text → grounding ratio < 1.0 with
    # FakeEmbeddings' bag-of-words even on a single sentence.
    response = """<page path="knowledge/concepts/zeta.md" type="concept">
---
tags: [topic/sample]
---

# Zeta

Zeta references pizza pineapple unicorn xylophone.
</page>
"""
    llm = FakeLLM(response_text=response)
    report = await run_synth_eval(
        spec, llm=llm, embedder=FakeEmbeddings()
    )
    assert report.passed is False
    # Threshold result for that one metric should carry direction="min"
    # and passed=False with the observed value preserved.
    grounding_rows = [
        r
        for r in report.threshold_results
        if r.name == "synth/fact_grounding_ratio"
    ]
    assert len(grounding_rows) == 1
    assert grounding_rows[0].direction == "min"
    assert grounding_rows[0].passed is False


# ---- fact-entailment judge wiring ------------------------------------------


class _DispatchLLM:
    """Routes ``complete`` by system prompt — synth pages, page-judge scores,
    entailment verdicts, category verdicts — so a single fake drives the full
    judge run without brittle call-order scripting."""

    def __init__(
        self,
        synth_pages: list[str],
        *,
        verdict: str,
        page_score: str,
        category: str,
    ) -> None:
        self._synth_pages = synth_pages
        self._synth_idx = 0
        self._verdict = verdict
        self._page_score = page_score
        self._category = category

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: object = None,
    ) -> LLMResponse:
        s = system.lower()
        if "entailment judge" in s:
            return LLMResponse(text=self._verdict, finish_reason="end_turn")
        if "taxonomy judge" in s:
            return LLMResponse(text=self._category, finish_reason="end_turn")
        if "evaluation judge" in s:
            return LLMResponse(text=self._page_score, finish_reason="end_turn")
        page = self._synth_pages[
            min(self._synth_idx, len(self._synth_pages) - 1)
        ]
        self._synth_idx += 1
        return LLMResponse(text=page, finish_reason="end_turn")


def _dispatch_llm() -> _DispatchLLM:
    return _DispatchLLM(
        [_SYNTH_PAGE_RESPONSE, _SYNTH_BETA_RESPONSE],
        verdict=json.dumps({"verdict": "yes", "rationale": "ok"}),
        page_score=json.dumps(
            {
                "grounding": 4,
                "atomicity": 4,
                "completeness": 4,
                "clarity": 4,
                "rationale": "ok",
            }
        ),
        category=json.dumps(
            {"chosen": "concept", "also_fits": None, "rationale": "ok"}
        ),
    )


def _enable_judge(
    ds: Path, *, entailment: bool = False, category: bool = False
) -> None:
    """Append a single ``judge:`` block enabling the requested judge legs.

    One block even when both legs are requested, so the YAML never carries
    duplicate ``judge:`` keys (which a per-leg append helper would produce).
    """
    flags: list[str] = []
    if entailment:
        flags.append("  entailment_grounding_enabled: true")
    if category:
        flags.append("  category_correctness_enabled: true")
    if not flags:
        return
    p = ds / "dataset.yaml"
    p.write_text(
        p.read_text(encoding="utf-8") + "judge:\n" + "\n".join(flags) + "\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_run_synth_eval_entailment_runs_when_enabled(tmp_path: Path) -> None:
    ds = _write_synth_dataset(tmp_path)
    _enable_judge(ds, entailment=True)
    spec = load_dataset(ds)

    report = await run_synth_eval(
        spec, llm=_dispatch_llm(), embedder=FakeEmbeddings(), judge=True
    )
    assert report.entailment_summary is not None
    assert report.entailment_summary.n_judged >= 1
    # The ratio is mirrored into informational for the A/B harness + display.
    assert "synth/fact_entailment_ratio" in report.informational
    assert (
        report.informational["synth/fact_entailment_ratio"]
        == report.entailment_summary.ratio
    )
    # All verdicts scripted "yes" → ratio 1.0; never gated.
    assert report.entailment_summary.ratio == 1.0
    assert "synth/fact_entailment_ratio" not in report.metrics


@pytest.mark.asyncio
async def test_run_synth_eval_entailment_off_when_judge_off(tmp_path: Path) -> None:
    """Flag on but ``judge=False`` → no entailment leg (it requires --judge)."""
    ds = _write_synth_dataset(tmp_path)
    _enable_judge(ds, entailment=True)
    spec = load_dataset(ds)

    report = await run_synth_eval(
        spec, llm=_dispatch_llm(), embedder=FakeEmbeddings(), judge=False
    )
    assert report.entailment_summary is None
    assert "synth/fact_entailment_ratio" not in report.informational


@pytest.mark.asyncio
async def test_run_synth_eval_entailment_off_when_flag_unset(tmp_path: Path) -> None:
    """``judge=True`` but the dataset didn't opt in → only the page judge runs."""
    ds = _write_synth_dataset(tmp_path)  # no entailment flag
    spec = load_dataset(ds)

    report = await run_synth_eval(
        spec, llm=_dispatch_llm(), embedder=FakeEmbeddings(), judge=True
    )
    assert report.judge_summary is not None  # page judge still ran
    assert report.entailment_summary is None
    assert "synth/fact_entailment_ratio" not in report.informational


# ---- category-correctness judge wiring -------------------------------------


@pytest.mark.asyncio
async def test_run_synth_eval_category_runs_when_enabled(tmp_path: Path) -> None:
    ds = _write_synth_dataset(tmp_path)
    _enable_judge(ds, category=True)
    spec = load_dataset(ds)

    report = await run_synth_eval(
        spec, llm=_dispatch_llm(), embedder=FakeEmbeddings(), judge=True
    )
    assert report.category_summary is not None
    assert report.category_summary.n_judged >= 1
    assert report.category_summary.n_errors == 0
    # Mirrored into informational (for the A/B harness + display), never gated.
    assert "synth/category_correctness_ratio" in report.informational
    assert (
        report.informational["synth/category_correctness_ratio"]
        == report.category_summary.ratio
    )
    assert "synth/category_correctness_ratio" not in report.metrics


@pytest.mark.asyncio
async def test_run_synth_eval_category_off_when_judge_off(tmp_path: Path) -> None:
    """Flag on but ``judge=False`` → no category leg (it requires --judge)."""
    ds = _write_synth_dataset(tmp_path)
    _enable_judge(ds, category=True)
    spec = load_dataset(ds)

    report = await run_synth_eval(
        spec, llm=_dispatch_llm(), embedder=FakeEmbeddings(), judge=False
    )
    assert report.category_summary is None
    assert "synth/category_correctness_ratio" not in report.informational


@pytest.mark.asyncio
async def test_run_synth_eval_category_off_when_flag_unset(tmp_path: Path) -> None:
    """``judge=True`` but the dataset didn't opt in → only the page judge runs."""
    ds = _write_synth_dataset(tmp_path)  # no category flag
    spec = load_dataset(ds)

    report = await run_synth_eval(
        spec, llm=_dispatch_llm(), embedder=FakeEmbeddings(), judge=True
    )
    assert report.judge_summary is not None  # page judge still ran
    assert report.category_summary is None
    assert "synth/category_correctness_ratio" not in report.informational
