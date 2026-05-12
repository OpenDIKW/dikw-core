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
calibrated separately by ``dikw eval mvp --eval synth`` against the
real LLM in Step 8 dogfood.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core.eval.dataset import load_dataset
from dikw_core.eval.fake_embedder import FakeEmbeddings
from dikw_core.eval.runner import (
    SynthEvalError,
    SynthEvalReport,
    run_synth_eval,
)

from .fakes import FakeLLM

_SYNTH_PAGE_RESPONSE = """<page path="wiki/concepts/alpha.md" type="concept">
---
tags: [topic/sample]
---

# Alpha

Alpha is a concept described in the corpus.
</page>
"""

_SYNTH_BETA_RESPONSE = """<page path="wiki/concepts/beta.md" type="concept">
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
    response = """<page path="wiki/concepts/zeta.md" type="concept">
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
