"""``dikw client synth --verify --judge`` — the optional grounding/entailment leg.

Verification roadmap 1.3 ("live-base grounding sample"). The deterministic
``--verify`` legs (persist / scoped-lint / semantic-duplicate) answer "is this
output structurally sound?"; ``--judge`` adds the one probabilistic question they
can't: "are the claims on these fresh pages actually supported by the sources
they cite?". It reuses the eval grounding pipeline (``compute_grounding_cosines``
→ ``claim_evidence_from_grounding`` → ``judge_entailment``) but points it at THIS
run's pages instead of an eval dataset.

The leg is **report-only**: it surfaces the entailment ratio + CI + counts but
does NOT fold into ``passed``. An LLM entailment judge is noisy (the roadmap
defers gating it to a calibrated Phase 2.2 threshold), and a false-red on the
flagship verify verdict erodes trust as much as a false-green. The probabilistic
threshold belongs in the orchestrating skill, not a hard CLI gate (Karpathy's
rule: deterministic gate in the engine, probabilistic call in the agent layer).

These tests pin: the leg runs and reports when both an LLM and an embedder are
wired; a low entailment ratio does NOT flip ``passed`` (report-only); the leg
loud-skips (requested-but-not-checked) when no embedder is available; and the
sample cap is honoured.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.providers import LLMResponse

from .fakes import FakeEmbeddings, init_test_base

FIXTURES = Path(__file__).parent / "fixtures" / "notes"


class _SynthAndJudgeLLM:
    """One fake LLM that answers both legs of a ``--verify --judge`` run.

    Synth prompts (matched by source path in the user message) return a canned
    ``<page>`` block; entailment-judge prompts (matched by the judge system
    string) return a canned ``{"verdict": ...}``. ``calls`` records how many
    entailment verdicts were asked for, so a test can assert the sample cap.
    """

    def __init__(self, script: dict[str, str], *, verdict: str = "yes") -> None:
        self._script = script
        self._verdict = verdict
        self.entailment_calls = 0

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
        _ = (model, max_tokens, temperature, tools)
        if "entailment judge" in system.lower():
            self.entailment_calls += 1
            return LLMResponse(
                text=f'{{"verdict": "{self._verdict}"}}', finish_reason="end_turn"
            )
        for src_path, resp in self._script.items():
            if src_path in user:
                return LLMResponse(text=resp, finish_reason="end_turn")
        raise AssertionError(f"no script entry matched prompt: {user[:200]}")


def _page(slug: str, title: str, body: str, *, source: str, tags: str = "sample") -> str:
    # ``sources:`` frontmatter is the provenance edge the grounding leg follows
    # to find the D-source chunks to ground each claim against.
    return (
        f'<page category="concept" slug="{slug}">\n'
        f"---\ntags: [{tags}]\nsources: [{source}]\n---\n\n"
        f"# {title}\n\n{body}\n"
        "</page>"
    )


# Two cross-linked pages, each attributed to its source. Bodies are full
# sentences so ``split_claims`` yields claims to ground.
def _script() -> dict[str, str]:
    return {
        "sources/notes/dikw.md": _page(
            "dikw-pyramid",
            "DIKW pyramid",
            "The DIKW pyramid organises raw data into four layers. "
            "It complements the [[Karpathy LLM Wiki]] approach.",
            source="sources/notes/dikw.md",
        ),
        "sources/notes/karpathy-wiki.md": _page(
            "karpathy-llm-wiki",
            "Karpathy LLM Wiki",
            "Karpathy's pattern builds a wiki from sources. "
            "It complements the [[DIKW pyramid]] model.",
            source="sources/notes/karpathy-wiki.md",
        ),
    }


def _seed(tmp_path: Path) -> Path:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    dest = wiki / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("dikw.md", "karpathy-wiki.md"):
        shutil.copy2(FIXTURES / name, dest / name)
    return wiki


async def _synth_twice(wiki: Path, llm: object, embedder: object, **kw: object) -> object:
    """First pass settles forward references; the second (force_all) verifies."""
    await api.synthesize(wiki, llm=llm, embedder=embedder)  # type: ignore[arg-type]
    return await api.synthesize(
        wiki, llm=llm, embedder=embedder, force_all=True, verify=True, **kw  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_judge_leg_runs_and_reports(tmp_path: Path) -> None:
    """With an LLM + embedder wired, ``judge=True`` grounds this run's claims and
    reports an entailment ratio. A canned ``yes`` judge → ratio 1.0."""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    llm = _SynthAndJudgeLLM(_script(), verdict="yes")
    report = await _synth_twice(wiki, llm, embedder, judge=True)

    v = report.verify
    assert v is not None
    assert v.grounding_requested is True
    assert v.grounding_checked is True
    assert v.grounding_n_judged > 0
    assert v.grounding_n_errors == 0
    assert v.grounding_entailment_ratio == pytest.approx(1.0)
    # Report-only: the grounding leg is NOT one of the gated legs.
    assert v.passed is True


@pytest.mark.asyncio
async def test_low_grounding_does_not_fail_verify(tmp_path: Path) -> None:
    """A canned ``no`` judge drives the entailment ratio to 0.0, but ``passed``
    stays True — the grounding leg is report-only, never a gate (the whole point:
    a noisy LLM judge must not false-red the flagship verdict)."""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    llm = _SynthAndJudgeLLM(_script(), verdict="no")
    report = await _synth_twice(wiki, llm, embedder, judge=True)

    v = report.verify
    assert v is not None
    assert v.grounding_checked is True
    assert v.grounding_entailment_ratio == pytest.approx(0.0)
    # Deterministic legs are clean → passed, despite the ground-truth ratio 0.
    assert v.persist_ok and v.lint_ok and v.duplicate_ok
    assert v.passed is True


class _FirstVerdictThenGarbageLLM(_SynthAndJudgeLLM):
    """First entailment call parses (``yes``), every later one errors — the
    half-dead-judge shape (e.g. provider times out after one verdict)."""

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
        if "entailment judge" in system.lower():
            self.entailment_calls += 1
            if self.entailment_calls > 1:
                return LLMResponse(
                    text="not a parseable verdict", finish_reason="end_turn"
                )
        return await super().complete(
            system=system,
            user=user,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
        )


@pytest.mark.asyncio
async def test_judge_majority_errors_withhold_ratio(tmp_path: Path) -> None:
    """A half-dead judge (1 verdict + N errors) must not report its sliver as
    the entailment ratio — ``EntailmentSummary.trustworthy``, the same rule the
    eval gate fold applies, withholds it. The leg still counts as checked (it
    ran), the error counts stay visible, and the ratio is ``None`` instead of a
    misleading 1.0 over a denominator of 1."""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    llm = _FirstVerdictThenGarbageLLM(_script(), verdict="yes")
    report = await _synth_twice(wiki, llm, embedder, judge=True)

    v = report.verify
    assert v is not None
    assert v.grounding_checked is True
    assert v.grounding_n_judged == 1
    assert v.grounding_n_errors >= 1
    assert v.grounding_n_errors >= v.grounding_n_judged
    assert v.grounding_entailment_ratio is None
    # Report-only leg: a withheld ratio is never a verify failure.
    assert v.passed is True


@pytest.mark.asyncio
async def test_judge_off_by_default(tmp_path: Path) -> None:
    """``verify=True`` without ``judge`` leaves the grounding leg untouched — it
    is strictly opt-in (an extra grounding-embed pass + LLM judge calls)."""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    llm = _SynthAndJudgeLLM(_script())
    report = await _synth_twice(wiki, llm, embedder)  # no judge=

    v = report.verify
    assert v is not None
    assert v.grounding_requested is False
    assert v.grounding_checked is False
    assert v.grounding_entailment_ratio is None
    assert llm.entailment_calls == 0


@pytest.mark.asyncio
async def test_judge_loud_skips_without_embedder(tmp_path: Path) -> None:
    """``judge=True`` but no embedder → the leg is requested but cannot run (the
    grounding argmax needs embeddings). It loud-skips: ``grounding_requested`` is
    True, ``grounding_checked`` is False — never a silent pass."""
    wiki = _seed(tmp_path)
    # Ingest WITH an embedder so there's an active text version, then synth
    # WITHOUT one — the judge leg should still skip for lack of an embedder.
    await api.ingest(wiki, embedder=FakeEmbeddings())

    llm = _SynthAndJudgeLLM(_script())
    await api.synthesize(wiki, llm=llm)
    report = await api.synthesize(
        wiki, llm=llm, force_all=True, verify=True, judge=True
    )

    v = report.verify
    assert v is not None
    assert v.grounding_requested is True
    assert v.grounding_checked is False
    assert v.grounding_entailment_ratio is None
    assert llm.entailment_calls == 0


@pytest.mark.asyncio
async def test_judge_leg_failure_does_not_crash_synth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The grounding leg is report-only: a raise inside it (e.g. the grounding
    re-embed hitting a provider error) must degrade to a loud skip, never
    discard the SynthReport — every page is already persisted by then."""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    async def _boom(**_kw: object) -> list:
        raise RuntimeError("grounding embed exploded")

    # Local import in ``_grounding_verify_leg`` resolves the name at call time,
    # so patching the source module is enough.
    monkeypatch.setattr(
        "dikw_core.eval.metrics.compute_grounding_cosines", _boom
    )

    llm = _SynthAndJudgeLLM(_script(), verdict="yes")
    # Must NOT raise.
    report = await _synth_twice(wiki, llm, embedder, judge=True)

    v = report.verify
    assert v is not None
    assert v.grounding_requested is True
    assert v.grounding_checked is False  # degraded to a loud skip
    assert v.grounding_entailment_ratio is None
    # Deterministic legs still decided the verdict.
    assert v.passed is True


@pytest.mark.asyncio
async def test_judge_ratio_is_none_when_nothing_judged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The leg can run (checked) yet judge nothing (no claims / all
    unverifiable). The engine maps ``n_judged == 0`` to ``ratio = None`` so the
    report omits a misleading 0.0 floor — pinned here by forcing the judge to
    return an empty summary (synth auto-attributes provenance, so an
    all-unsourced run isn't reachable through the normal path)."""
    from dikw_core.eval.judge import EntailmentSummary

    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    async def _empty_summary(*_a: object, **_kw: object) -> EntailmentSummary:
        return EntailmentSummary(
            ratio=0.0, n_judged=0, n_errors=0, n_calls_ok=0, n_no_evidence=0
        )

    # Local import in ``_grounding_verify_leg`` resolves the name at call time.
    monkeypatch.setattr("dikw_core.eval.judge.judge_entailment", _empty_summary)

    llm = _SynthAndJudgeLLM(_script(), verdict="yes")
    report = await _synth_twice(wiki, llm, embedder, judge=True)

    v = report.verify
    assert v is not None
    assert v.grounding_checked is True
    assert v.grounding_n_judged == 0
    assert v.grounding_entailment_ratio is None


@pytest.mark.asyncio
async def test_judge_sample_cap_is_honoured(tmp_path: Path) -> None:
    """``synth.verify_judge_sample`` caps how many claims are judged — the fake's
    entailment-call count must not exceed the configured cap."""
    wiki = _seed(tmp_path)
    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    # Pin a tiny cap via dikw.yml so the test is independent of the default.
    cfg_path = wiki / "dikw.yml"
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg_path.write_text(
        cfg_text + "\nsynth:\n  verify_judge_sample: 1\n", encoding="utf-8"
    )

    llm = _SynthAndJudgeLLM(_script(), verdict="yes")
    report = await _synth_twice(wiki, llm, embedder, judge=True)

    v = report.verify
    assert v is not None
    assert v.grounding_checked is True
    assert v.grounding_sample == 1
    assert llm.entailment_calls <= 1
