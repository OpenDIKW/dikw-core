"""Hermetic K-layer eval gate.

Smoke-tests ``run_synth_eval`` end-to-end with ``FakeLLM`` +
``FakeEmbeddings`` — no API keys, no network. Most tests build the
dataset in-process so they stay independent of changes to packaged
datasets under ``evals/datasets/``; ``test_mvp_synth_half_runs_hermetically``
is the deliberate exception — it runs the runner against the REAL packaged
``mvp`` dataset's synth half so a break in the committed dataset wiring or
the synth pipeline on the real corpus is caught in CI (the synth twin of
``test_retrieval_quality``'s ``run_eval('mvp')``).

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


def _write_synth_dataset(
    root: Path,
    *,
    name: str = "synth-toy",
    entailment_threshold: float | None = None,
) -> Path:
    """Build a minimal synth-eval dataset; permissive thresholds + a
    single concept page_type so the FakeLLM responses parse cleanly.

    ``entailment_threshold`` (optional) declares a ``synth/fact_entailment_ratio``
    floor AND opts the dataset into the entailment judge — the pair needed to
    exercise the judge-only conditional gate. Left unset, the dataset gates
    only the five deterministic synth metrics (the common case).
    """
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
    thresholds = [
        "  synth/fact_grounding_ratio: 0.0",
        "  synth/atomicity_score: 0.0",
        "  synth/duplicate_ratio_max: 1.0",
        "  synth/wikilink_resolved_ratio: 0.0",
        "  synth/language_fidelity: 0.0",
    ]
    judge_block = ""
    if entailment_threshold is not None:
        thresholds.append(f"  synth/fact_entailment_ratio: {entailment_threshold}")
        judge_block = "judge:\n  entailment_grounding_enabled: true\n"
    (ds / "dataset.yaml").write_text(
        "name: " + name + "\n"
        "modes: [synth]\n"
        "thresholds:\n" + "\n".join(thresholds) + "\n"
        "synth:\n"
        "  page_types: [concept]\n" + judge_block,
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


def _mvp_page(i: int) -> str:
    """A distinct, parseable concept page (new ``category=``/``slug=`` format,
    matching the ``mvp`` dataset's ``schema.categories``)."""
    return (
        f'<page category="concept" slug="mvp-concept-{i}">\n'
        "---\n"
        "tags: [topic/sample]\n"
        "---\n\n"
        f"# MVP Concept {i}\n\n"
        f"Concept {i} is one of the ideas the source corpus discusses.\n"
        "</page>\n"
    )


class _UniqueSynthLLM:
    """Emits a distinct, parseable ``<page>`` per synth-group call so a real
    multi-source dataset runs the full ingest→synth→parse→persist→metrics
    pipeline without slug collisions. It ignores the prompt — the point is
    pipeline integrity against the real dataset, not prompt-faithful
    generation (which only a live LLM can give)."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list | None = None,
    ) -> LLMResponse:
        _ = (system, user, model, max_tokens, temperature, tools)
        self.calls += 1
        return LLMResponse(text=_mvp_page(self.calls), finish_reason="end_turn")


@pytest.mark.asyncio
async def test_mvp_synth_half_runs_hermetically() -> None:
    """The real packaged ``mvp`` dataset's *synth* half must run end-to-end in
    CI — not just its retrieval half (which ``test_retrieval_quality`` already
    gates via ``run_eval('mvp')``).

    Before this, ``run_synth_eval`` was only exercised in CI against a
    synthetic in-test toy dataset; the real ``mvp`` corpus + the real
    ``mvp`` ``dataset.yaml`` synth section (categories, thresholds) never ran
    hermetically, so a change that broke synth on the packaged dataset stayed
    green until someone ran the manual real-LLM eval.

    Hermetic: a deterministic ``FakeLLM`` + ``FakeEmbeddings``, no keys /
    network. We assert STRUCTURAL completion — the real dataset loads, the
    real corpus flows through ingest→synth→parse→persist→metrics, every
    declared synth metric is computed and in ``[0, 1]``, and
    ``threshold_results`` mirrors the dataset's declared synth thresholds.

    We deliberately do NOT assert ``report.passed``. ``FakeEmbeddings`` is
    lexical and the canned pages don't echo the Karpathy corpus, so the real
    thresholds (e.g. ``fact_grounding_ratio >= 0.55``) cannot be met by
    Fake-generated output — and asserting they were would be a meaningless
    gate. Real synth-quality numbers come from the manual real-LLM
    ``evals/BASELINES.md`` discipline, which hermetic CI cannot replace; this
    test guards the *machinery*, not the *quality*.
    """
    spec = load_dataset("mvp")
    report = await run_synth_eval(
        spec, llm=_UniqueSynthLLM(), embedder=FakeEmbeddings()
    )

    assert report.mode == "synth"
    assert report.dataset_name == "mvp"
    assert report.n_sources == 3  # three Karpathy essays
    assert report.n_pages > 0
    assert report.gated is True

    gated = {
        "synth/fact_grounding_ratio",
        "synth/atomicity_score",
        "synth/duplicate_ratio_max",
        "synth/wikilink_resolved_ratio",
        "synth/language_fidelity",
    }
    for m in gated:
        assert m in report.metrics, f"missing synth metric: {m}"
        assert 0.0 <= report.metrics[m] <= 1.0, (
            f"{m} out of [0, 1]: {report.metrics[m]}"
        )
    # Threshold gate keys mirror the FIVE deterministic synth thresholds mvp
    # declares. mvp also declares ``synth/fact_entailment_ratio`` — but that is
    # a judge-only metric, and this hermetic run passes no ``--judge``, so the
    # conditional gate drops it (rather than recording a spurious ``None`` miss).
    assert {r.name for r in report.threshold_results} == gated
    assert "synth/fact_entailment_ratio" not in {
        r.name for r in report.threshold_results
    }
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


# ---- entailment as a conditional (judge-only) gate -------------------------


@pytest.mark.asyncio
async def test_entailment_gated_when_judge_runs(tmp_path: Path) -> None:
    """A dataset declaring a ``synth/fact_entailment_ratio`` floor AND ``--judge``
    folds the judge-only ratio into the gate — it becomes a hard pass/fail row
    in ``threshold_results``. The ratio stays mirrored in ``informational`` (for
    display / the A/B harness) and is NOT promoted into the deterministic
    ``metrics`` dict."""
    ds = _write_synth_dataset(tmp_path, entailment_threshold=0.55)
    spec = load_dataset(ds)

    report = await run_synth_eval(
        spec, llm=_dispatch_llm(), embedder=FakeEmbeddings(), judge=True
    )

    rows = [
        r for r in report.threshold_results
        if r.name == "synth/fact_entailment_ratio"
    ]
    assert len(rows) == 1
    assert rows[0].observed == 1.0  # all verdicts scripted "yes"
    assert rows[0].direction == "min"  # higher entailment is better
    assert rows[0].passed is True
    assert "synth/fact_entailment_ratio" in report.informational
    assert "synth/fact_entailment_ratio" not in report.metrics


@pytest.mark.asyncio
async def test_entailment_gate_fails_below_threshold(tmp_path: Path) -> None:
    """All-``no`` verdicts → ratio 0.0 < the 0.55 floor → that gate FAILS
    (min-direction) and drags ``report.passed`` to False."""
    ds = _write_synth_dataset(tmp_path, entailment_threshold=0.55)
    spec = load_dataset(ds)
    llm = _DispatchLLM(
        [_SYNTH_PAGE_RESPONSE, _SYNTH_BETA_RESPONSE],
        verdict=json.dumps({"verdict": "no", "rationale": "unsupported"}),
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

    report = await run_synth_eval(
        spec, llm=llm, embedder=FakeEmbeddings(), judge=True
    )

    row = next(
        r for r in report.threshold_results
        if r.name == "synth/fact_entailment_ratio"
    )
    assert row.observed == 0.0
    assert row.passed is False
    assert report.passed is False


@pytest.mark.asyncio
async def test_entailment_threshold_dropped_when_judge_off(tmp_path: Path) -> None:
    """The conditional-gating contract: a dataset may declare a
    ``synth/fact_entailment_ratio`` floor, but a NON-judge run (hermetic CI,
    plain ``--eval synth`` without ``--judge``) must NOT be failed by it — the
    metric was never computed. The threshold is dropped, not recorded as a
    ``observed=None`` miss; the five deterministic gates still apply."""
    ds = _write_synth_dataset(tmp_path, entailment_threshold=0.55)
    spec = load_dataset(ds)

    report = await run_synth_eval(
        spec,
        llm=FakeLLM(responses=[_SYNTH_PAGE_RESPONSE, _SYNTH_BETA_RESPONSE]),
        embedder=FakeEmbeddings(),
        judge=False,
    )

    names = {r.name for r in report.threshold_results}
    assert "synth/fact_entailment_ratio" not in names
    assert names == {
        "synth/fact_grounding_ratio",
        "synth/atomicity_score",
        "synth/duplicate_ratio_max",
        "synth/wikilink_resolved_ratio",
        "synth/language_fidelity",
    }
    assert report.entailment_summary is None


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
