"""Loud-skip for the eval embedder fallback (verification roadmap 0.6).

``run_synth_eval`` / ``run_eval`` substitute ``FakeEmbeddings`` (lexical
bag-of-words) when no embedder is passed. That is intentional for the hermetic
pytest gate (which passes ``FakeEmbeddings()`` *explicitly*), but a caller who
silently reaches the fallback — a programmatic caller, a misconfigured A/B run,
or a refactor that drops the embedder — then gets ``fact_grounding_ratio`` /
``duplicate_ratio_max`` (and the retrieval vector leg) computed on lexical
vectors that *look* like a real semantic measurement. These tests assert the
fallback is announced LOUDLY (report warning + WARN log) when, and only when,
the embedder is actually missing — not when ``FakeEmbeddings`` is the caller's
explicit choice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core.eval.fake_embedder import FakeEmbeddings
from dikw_core.eval.runner import run_eval, run_synth_eval

from .fakes import FakeLLM
from .test_progress_reporter import ListReporter
from .test_synth_quality import (
    _SYNTH_BETA_RESPONSE,
    _SYNTH_PAGE_RESPONSE,
    _write_synth_dataset,
)


def _warn_logs(reporter: ListReporter) -> list[str]:
    return [
        e.payload["message"]
        for e in reporter.events
        if e.kind == "log" and e.payload.get("level") == "WARN"
    ]


def _mentions_fallback(text: str) -> bool:
    low = text.lower()
    return "embedder" in low and "fakeembeddings" in low


# ---- synth path (the named 0.6 target: grounding / duplicate) -------------


@pytest.mark.asyncio
async def test_synth_eval_warns_when_embedder_falls_back_to_fake(
    tmp_path: Path,
) -> None:
    spec = _load(tmp_path)
    reporter = ListReporter()

    report = await run_synth_eval(
        spec,
        llm=FakeLLM(responses=[_SYNTH_PAGE_RESPONSE, _SYNTH_BETA_RESPONSE]),
        embedder=None,
        reporter=reporter,
    )

    assert any(_mentions_fallback(w) for w in report.warnings), report.warnings
    assert any(_mentions_fallback(m) for m in _warn_logs(reporter)), reporter.events


@pytest.mark.asyncio
async def test_synth_eval_no_fallback_warning_with_explicit_embedder(
    tmp_path: Path,
) -> None:
    # The hermetic gate's intentional FakeEmbeddings() must NOT trip the warning.
    spec = _load(tmp_path)
    reporter = ListReporter()

    report = await run_synth_eval(
        spec,
        llm=FakeLLM(responses=[_SYNTH_PAGE_RESPONSE, _SYNTH_BETA_RESPONSE]),
        embedder=FakeEmbeddings(),
        reporter=reporter,
    )

    assert not any(_mentions_fallback(w) for w in report.warnings), report.warnings
    assert not any(_mentions_fallback(m) for m in _warn_logs(reporter)), reporter.events


# ---- retrieval path (same `embedder or FakeEmbeddings()` line) ------------


@pytest.mark.asyncio
async def test_retrieval_eval_warns_when_embedder_falls_back_to_fake(
    tmp_path: Path,
) -> None:
    spec = _load(tmp_path)
    reporter = ListReporter()

    await run_eval(spec, embedder=None, reporter=reporter)

    assert any(_mentions_fallback(m) for m in _warn_logs(reporter)), reporter.events


@pytest.mark.asyncio
async def test_retrieval_eval_no_fallback_warning_with_explicit_embedder(
    tmp_path: Path,
) -> None:
    spec = _load(tmp_path)
    reporter = ListReporter()

    await run_eval(spec, embedder=FakeEmbeddings(), reporter=reporter)

    assert not any(_mentions_fallback(m) for m in _warn_logs(reporter)), reporter.events


def _load(tmp_path: Path):
    from dikw_core.eval.dataset import load_dataset

    return load_dataset(_write_synth_dataset(tmp_path))
