"""LLM judge for K-layer synth output.

The soft layer of the spec's "hard gate + soft judge" pair. Each page
gets one ``llm.complete`` call with the ``eval_judge_synth`` prompt;
the model returns four 0-5 integer scores plus a one-line rationale.
Per-page parse failures are counted (``n_errors``) rather than raised
so one bad response doesn't kill the whole eval.

Results never block a PR — they go into ``BASELINES.md`` as the
quality trend the author watches when tuning ``synthesize.md``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..domains.knowledge.page import KnowledgePage
from ..progress import NoopReporter, ProgressReporter
from ..prompts import load as load_prompt
from ..providers.base import LLMProvider
from ..schemas import ChunkRecord, LinkRecord, LinkType
from .metrics import GroundingClaim

logger = logging.getLogger(__name__)

# Reasoning LLMs (e.g. MiniMax-M3) emit a hidden chain-of-thought that counts
# against ``max_tokens`` before the final JSON. Measured against MiniMax-M3, a
# dense entailment judgment spends ~1350 output tokens (mostly that trace), so
# the old 256/512 caps truncated such responses to empty text — ~75% / ~58% of
# judge calls then failed to parse. 4096 leaves ~3x headroom; non-reasoning
# models stop at end_turn far below it, so the higher ceiling is effectively
# free (it bounds, it doesn't pad). Callers may still override per-call.
_JUDGE_MAX_TOKENS = 4096


class JudgeScore(BaseModel):
    """Four 0-5 integer scores + a one-sentence rationale."""

    model_config = ConfigDict(frozen=True)

    grounding: int = Field(ge=0, le=5)
    atomicity: int = Field(ge=0, le=5)
    completeness: int = Field(ge=0, le=5)
    clarity: int = Field(ge=0, le=5)
    rationale: str

    @field_validator(
        "grounding", "atomicity", "completeness", "clarity", mode="before"
    )
    @classmethod
    def _reject_non_integer(cls, v: object) -> object:
        # Reject bools (``True`` is technically an int in Python) and
        # floats — the 0-5 scale is integer-only by contract; pydantic
        # would otherwise coerce ``3.7 → 3`` silently.
        if isinstance(v, bool):
            raise ValueError("score must be int, not bool")
        if isinstance(v, float):
            raise ValueError("score must be int, got float")
        return v


class PageJudgeEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    score: JudgeScore


class JudgeSummary(BaseModel):
    """Aggregate of judge scores across all (or sampled) pages.

    Each ``ci_<dim>`` is a deterministic bootstrap 95% confidence interval
    ``(low, high)`` on that dimension's mean. With a small judge sample the
    raw mean is noisy (±0.05 swings between identical re-runs are common);
    the CI width is the honest uncertainty band an author reads before
    declaring a ``synthesize.md`` change an improvement. ``(0.0, 0.0)`` when
    nothing was judged.
    """

    model_config = ConfigDict(frozen=True)

    n_judged: int
    n_errors: int
    mean_grounding: float
    mean_atomicity: float
    mean_completeness: float
    mean_clarity: float
    ci_grounding: tuple[float, float] = (0.0, 0.0)
    ci_atomicity: tuple[float, float] = (0.0, 0.0)
    ci_completeness: tuple[float, float] = (0.0, 0.0)
    ci_clarity: tuple[float, float] = (0.0, 0.0)
    per_page: list[PageJudgeEntry] = Field(default_factory=list)


def bootstrap_ci(
    values: Sequence[float],
    *,
    seed: str,
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of ``values``.

    Resamples ``values`` with replacement ``n_resamples`` times, takes the
    mean of each resample, and returns the ``(low, high)`` percentile bounds
    for the requested ``confidence``. The RNG is seeded from
    ``hashlib.sha1(seed)`` so the interval is byte-stable across runs (same
    contract as the page-sampling RNG in :func:`judge_synthesis`).

    Edge cases: empty → ``(0.0, 0.0)``; a single observation → ``(v, v)``
    (no spread to estimate). Every resample mean lies within
    ``[min(values), max(values)]``, so the CI never escapes the 0-5 scale.
    """
    if not values:
        return (0.0, 0.0)
    vals = list(values)
    n = len(vals)
    if n == 1:
        return (vals[0], vals[0])
    rng = random.Random(hashlib.sha1(seed.encode("utf-8")).digest()[:8])
    means: list[float] = []
    for _ in range(n_resamples):
        resample_sum = 0.0
        for _ in range(n):
            resample_sum += vals[rng.randrange(n)]
        means.append(resample_sum / n)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    lo_idx = min(n_resamples - 1, max(0, int(tail * n_resamples)))
    hi_idx = min(n_resamples - 1, max(0, int((1.0 - tail) * n_resamples)))
    return (means[lo_idx], means[hi_idx])


# Bounds for ``--judge-sample auto``: never trust a judge ratio on fewer than 5
# items, never spend on more than 50 (past ~50 the LLM cost outweighs the
# marginal CI tightening -- diminishing 1/sqrt(n) returns).
_AUTO_SAMPLE_MIN = 5
_AUTO_SAMPLE_MAX = 50


def recommended_judge_sample(target_margin: float = 0.2) -> int:
    """Smallest judge sample whose bootstrap 95% CI half-width clears
    ``target_margin`` for any [0, 1] judge ratio, clamped to ``[5, 50]``.

    A [0, 1] ratio metric (entailment / category / ...) has a bootstrap 95% CI
    half-width of about ``1.96 * sd / sqrt(n)``, maximised at the worst-case
    standard deviation ``sd = 0.5`` (variance 0.25, a 50/50 split). Solving
    ``1.96 * 0.5 / sqrt(n) <= target_margin`` gives
    ``n >= (1.96 * 0.5 / target_margin) ** 2`` -> ``25`` at the default ``0.2``.
    The bound is the worst case over *all* score distributions, so it holds for
    any dataset or metric -- no per-corpus calibration can push it higher. The
    two real MiniMax-M3 calibrations confirm the ``1/sqrt(n)`` model (entailment
    n=20 -> +/-0.13, category n=8 -> +/-0.19). Datasets with fewer items than
    the result are judged in full (the sample cap is a no-op there).
    """
    if target_margin <= 0.0:
        return _AUTO_SAMPLE_MAX
    raw = math.ceil((1.96 * 0.5 / target_margin) ** 2)
    return max(_AUTO_SAMPLE_MIN, min(_AUTO_SAMPLE_MAX, raw))


def parse_judge_response(text: str) -> JudgeScore | None:
    """Parse a judge LLM response into ``JudgeScore``, or ``None`` on any
    failure (malformed JSON, missing fields, out-of-range or non-integer
    scores, etc.).

    Strips an optional `````json ... `````
    fence before parsing — some LLMs still wrap JSON in markdown fences
    despite the prompt asking for raw output.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline > 0 and stripped.endswith("```"):
            stripped = stripped[first_newline + 1 : -3].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return JudgeScore.model_validate(payload)
    except ValidationError:
        return None


def _format_prompt(*, page: KnowledgePage, source_text: str) -> str:
    return (
        load_prompt("eval_judge_synth")
        .replace("{page_path}", page.path)
        .replace("{page_title}", page.title)
        .replace("{page_body}", page.body)
        .replace("{source_text}", source_text)
    )


_DEFAULT_JUDGE_SYSTEM = (
    "You are an evaluation judge. Score knowledge pages on four 0-5 "
    "dimensions. Return raw JSON only — no prose, no fences."
)


async def judge_synthesis(
    pages: Sequence[KnowledgePage],
    *,
    sources: Mapping[str, str],
    llm: LLMProvider,
    model: str,
    sample: int | None = None,
    reporter: ProgressReporter | None = None,
    seed: str = "dikw",
    max_tokens: int = _JUDGE_MAX_TOKENS,
    temperature: float = 0.0,
) -> JudgeSummary:
    """Run the judge across (sampled) pages, aggregate to ``JudgeSummary``.

    ``sources`` maps each page's primary source path (``page.sources[0]``)
    to the raw source text. Pages whose source isn't in the map are
    judged against an empty source — they typically score low on
    ``grounding`` and ``completeness``, which is the right signal.

    ``sample`` (optional) caps pages judged; selection is seeded via
    ``hashlib.sha1(seed)`` so repeated runs on the same dataset draw
    the same pages. ``sample`` ≥ ``len(pages)`` is a no-op cap.

    A per-page LLM exception or parse failure increments ``n_errors``
    and skips the page; ``n_judged`` is the number that produced a
    valid score. Means are computed over valid entries (zero when all
    fail).
    """
    _reporter: ProgressReporter = reporter or NoopReporter()
    selected = list(pages)
    if sample is not None and sample < len(selected):
        rng = random.Random(
            hashlib.sha1(seed.encode("utf-8")).digest()[:8]
        )
        selected = rng.sample(selected, sample)

    per_page: list[PageJudgeEntry] = []
    n_errors = 0

    for idx, page in enumerate(selected):
        primary = page.sources[0] if page.sources else ""
        source_text = sources.get(primary, "")
        await _reporter.progress(
            phase="judge",
            current=idx,
            total=len(selected),
            detail={"path": page.path},
        )
        try:
            response = await llm.complete(
                system=_DEFAULT_JUDGE_SYSTEM,
                user=_format_prompt(page=page, source_text=source_text),
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning(
                "judge: LLM call failed for %s — %s", page.path, e
            )
            n_errors += 1
            continue
        score = parse_judge_response(response.text)
        if score is None:
            n_errors += 1
            continue
        per_page.append(PageJudgeEntry(path=page.path, score=score))

    n_judged = len(per_page)
    if n_judged == 0:
        return JudgeSummary(
            n_judged=0,
            n_errors=n_errors,
            mean_grounding=0.0,
            mean_atomicity=0.0,
            mean_completeness=0.0,
            mean_clarity=0.0,
            per_page=[],
        )
    grounding_vals = [float(e.score.grounding) for e in per_page]
    atomicity_vals = [float(e.score.atomicity) for e in per_page]
    completeness_vals = [float(e.score.completeness) for e in per_page]
    clarity_vals = [float(e.score.clarity) for e in per_page]
    # Per-dimension seed suffix so the four bootstraps draw independent
    # resample indices rather than sharing one RNG stream.
    return JudgeSummary(
        n_judged=n_judged,
        n_errors=n_errors,
        mean_grounding=sum(grounding_vals) / n_judged,
        mean_atomicity=sum(atomicity_vals) / n_judged,
        mean_completeness=sum(completeness_vals) / n_judged,
        mean_clarity=sum(clarity_vals) / n_judged,
        ci_grounding=bootstrap_ci(grounding_vals, seed=f"{seed}:grounding"),
        ci_atomicity=bootstrap_ci(atomicity_vals, seed=f"{seed}:atomicity"),
        ci_completeness=bootstrap_ci(completeness_vals, seed=f"{seed}:completeness"),
        ci_clarity=bootstrap_ci(clarity_vals, seed=f"{seed}:clarity"),
        per_page=per_page,
    )


# ===========================================================================
# Fact-entailment judge — the LLM grounding leg the cosine metric is blind to
# ===========================================================================
#
# ``fact_grounding_ratio`` reduces to a cosine: "GPT-4 is 4x faster than GPT-3"
# (a fabricated ratio) and "GPT-4 is faster than GPT-3" (supported) land in the
# same cosine band, so the embedding metric cannot tell them apart. The
# entailment judge asks an LLM whether the nearest source chunk actually
# *supports* the claim (yes/partial/no), catching invented specifics the cosine
# metric is blind to. Reuses the per-claim argmax (``best_chunk_seq``) that
# ``compute_grounding_cosines`` already produced — no re-embedding.

_ENTAILMENT_SCORE: dict[str, float] = {"yes": 1.0, "partial": 0.5, "no": 0.0}


@dataclass(frozen=True)
class ClaimEvidence:
    """One page claim paired with the source-chunk text that best matches it.

    Built by :func:`claim_evidence_from_grounding` from the cosines
    :func:`..metrics.compute_grounding_cosines` already produced. ``evidence``
    is ``None`` when the claim's source had no embeddable chunk (the grounding
    claim's ``best_chunk_seq`` is ``None`` / ``max_cosine == -inf``); the judge
    scores such a claim as unverifiable (``0.0``) rather than dropping it from
    the denominator.
    """

    page_path: str
    source_path: str
    claim: str
    evidence: str | None


def claim_evidence_from_grounding(
    grounding_claims: Sequence[GroundingClaim],
    chunks_by_source: Mapping[str, Sequence[ChunkRecord]],
) -> list[ClaimEvidence]:
    """Resolve each grounding claim's nearest source chunk to its text.

    Reuses the per-claim ``best_chunk_seq`` argmax from
    :func:`..metrics.compute_grounding_cosines` (no re-embedding): looks up the
    chunk text for ``(source_path, best_chunk_seq)`` and pairs it with the
    claim. A claim whose ``best_chunk_seq`` is ``None`` gets ``evidence=None``
    so the entailment judge counts it as unverifiable (scored ``0.0``) rather
    than silently dropping it.
    """
    text_by_key: dict[tuple[str, int], str] = {}
    for src, chunks in chunks_by_source.items():
        for c in chunks:
            text_by_key[(src, c.seq)] = c.text
    out: list[ClaimEvidence] = []
    for gc in grounding_claims:
        evidence = (
            None
            if gc.best_chunk_seq is None
            else text_by_key.get((gc.source_path, gc.best_chunk_seq))
        )
        out.append(
            ClaimEvidence(
                page_path=gc.page_path,
                source_path=gc.source_path,
                claim=gc.claim,
                evidence=evidence,
            )
        )
    return out


class EntailmentVerdict(BaseModel):
    """One claim's entailment verdict + its 0.0/0.5/1.0 score.

    ``verdict`` is the LLM's ``yes`` / ``partial`` / ``no`` call (see
    ``prompts/eval_judge_entailment.md``); ``score`` maps it to ``1.0`` /
    ``0.5`` / ``0.0`` for the ratio. ``rationale`` is optional — the score only
    needs the verdict, so a thin ``{"verdict": "yes"}`` response still parses.
    """

    model_config = ConfigDict(frozen=True)

    verdict: Literal["yes", "partial", "no"]
    rationale: str = ""

    @property
    def score(self) -> float:
        return _ENTAILMENT_SCORE[self.verdict]


class EntailmentSummary(BaseModel):
    """Aggregate of fact-entailment verdicts across (sampled) page claims.

    ``ratio`` is the mean verdict score over ``n_judged`` claims. Claims whose
    source had no evidence chunk count as ``0.0`` (tallied in ``n_no_evidence``)
    rather than being dropped; parse / LLM failures are excluded and counted in
    ``n_errors`` instead. ``ci`` is a deterministic bootstrap 95% interval on
    the ratio — the same honesty band as :class:`JudgeSummary`. ``ratio == 0.0``
    with ``n_judged == 0`` means nothing was judged: the caller omits the metric
    rather than reporting a misleading floor.
    """

    model_config = ConfigDict(frozen=True)

    ratio: float
    n_judged: int
    n_errors: int
    n_no_evidence: int
    ci: tuple[float, float] = (0.0, 0.0)


def parse_entailment_verdict(text: str) -> EntailmentVerdict | None:
    """Parse an entailment-judge response into ``EntailmentVerdict``, or
    ``None`` on any failure (malformed JSON, non-object, missing / unknown
    verdict).

    The ``verdict`` token is matched case-insensitively after stripping
    surrounding whitespace (``"YES "`` → ``yes``) — LLMs vary on casing — but
    only ``yes`` / ``partial`` / ``no`` are accepted; anything else (``maybe``,
    ``true``, ``1``) is a parse failure, never a silent default. Strips an
    optional ```` ```json … ``` ```` fence first, mirroring
    :func:`parse_judge_response`.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline > 0 and stripped.endswith("```"):
            stripped = stripped[first_newline + 1 : -3].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    raw_verdict = payload.get("verdict")
    if not isinstance(raw_verdict, str):
        return None
    verdict = raw_verdict.strip().lower()
    if verdict not in _ENTAILMENT_SCORE:
        return None
    raw_rationale = payload.get("rationale")
    rationale = raw_rationale if isinstance(raw_rationale, str) else ""
    try:
        return EntailmentVerdict.model_validate(
            {"verdict": verdict, "rationale": rationale}
        )
    except ValidationError:
        return None


def _format_entailment_prompt(*, claim: str, evidence: str) -> str:
    # ``str.replace`` (not ``str.format``) so a claim or evidence chunk
    # containing literal ``{`` / ``}`` (a JSON snippet, code) can't raise
    # ``KeyError`` / ``IndexError`` at format time.
    return (
        load_prompt("eval_judge_entailment")
        .replace("{claim}", claim)
        .replace("{evidence}", evidence)
    )


_ENTAILMENT_SYSTEM = (
    "You are an entailment judge. Decide whether the evidence supports the "
    "claim. Return raw JSON only — no prose, no fences."
)


async def judge_entailment(
    pairs: Sequence[ClaimEvidence],
    *,
    llm: LLMProvider,
    model: str,
    sample: int | None = None,
    reporter: ProgressReporter | None = None,
    seed: str = "dikw",
    max_tokens: int = _JUDGE_MAX_TOKENS,
    temperature: float = 0.0,
) -> EntailmentSummary:
    """Score each (claim, evidence) pair for entailment, aggregate to a ratio.

    For every pair the judge LLM is asked whether the evidence entails the
    claim (``yes`` / ``partial`` / ``no`` → ``1.0`` / ``0.5`` / ``0.0``). A pair
    with no evidence (the claim's source had no embeddable chunk) scores ``0.0``
    without an LLM call and is counted in ``n_no_evidence``. Identical
    ``(claim, evidence)`` pairs are judged once and the verdict reused, so a
    corpus with repeated claims doesn't pay double spend.

    ``sample`` caps the number of claims judged; selection is seeded via
    ``hashlib.sha1(seed)`` so repeated runs draw the same subset (``sample`` ≥
    ``len(pairs)`` is a no-op cap). A per-pair LLM exception or parse failure
    increments ``n_errors`` and skips the pair (the ratio is over
    successfully-scored claims only) — one bad response never kills the run.
    """
    _reporter: ProgressReporter = reporter or NoopReporter()
    selected = list(pairs)
    if sample is not None and sample < len(selected):
        rng = random.Random(hashlib.sha1(seed.encode("utf-8")).digest()[:8])
        selected = rng.sample(selected, sample)

    scores: list[float] = []
    n_errors = 0
    n_no_evidence = 0
    cache: dict[str, float] = {}

    for idx, pair in enumerate(selected):
        await _reporter.progress(
            phase="entailment",
            current=idx,
            total=len(selected),
            detail={"path": pair.page_path},
        )
        if not (pair.evidence and pair.evidence.strip()):
            # No embeddable source chunk (or a blank one) → unverifiable: score
            # 0.0 without an LLM call rather than dropping the claim from the
            # denominator. Grounding already filters blank chunks, so this is
            # belt-and-suspenders, but it keeps "no evidence" airtight.
            scores.append(0.0)
            n_no_evidence += 1
            continue
        key = hashlib.sha1(
            (pair.claim + "\0" + pair.evidence).encode("utf-8")
        ).hexdigest()
        cached = cache.get(key)
        if cached is not None:
            scores.append(cached)
            continue
        try:
            response = await llm.complete(
                system=_ENTAILMENT_SYSTEM,
                user=_format_entailment_prompt(
                    claim=pair.claim, evidence=pair.evidence
                ),
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning(
                "entailment: LLM call failed for %s — %s", pair.page_path, e
            )
            n_errors += 1
            continue
        verdict = parse_entailment_verdict(response.text)
        if verdict is None:
            n_errors += 1
            continue
        cache[key] = verdict.score
        scores.append(verdict.score)

    n_judged = len(scores)
    ratio = sum(scores) / n_judged if n_judged else 0.0
    return EntailmentSummary(
        ratio=ratio,
        n_judged=n_judged,
        n_errors=n_errors,
        n_no_evidence=n_no_evidence,
        ci=bootstrap_ci(scores, seed=f"{seed}:entailment"),
    )


# --- Category-correctness judge ---------------------------------------------
#
# The embedding metrics see *where* each page was filed (``category_distribution``
# / ``fallback_ratio_max``) but never whether that filing is *right*: a page
# about a named tool mis-filed under ``concept`` looks identical to a correctly
# filed one. This judge re-derives the best category independently — body +
# the closed declared set (fallback included) — and compares it to where synth
# actually put the page. Karpathy's rule: the closed set is deterministic
# scoping; whether a page belongs in it is the probabilistic call.


@dataclass(frozen=True)
class CategoryOption:
    """One choosable category: its ``path`` (the on-disk folder / taxonomy node)
    and ``desc`` (the guidance the synth LLM was given for it). The runner passes
    the declared categories plus the fallback bucket as the closed option set."""

    path: str
    desc: str


class CategoryVerdict(BaseModel):
    """The judge's independent placement: the best-fit ``chosen`` path, an
    optional co-equal ``also_fits`` (a genuine borderline, else ``None``), and a
    one-line ``rationale``. Both paths are validated against the closed option
    set at parse time, so a verdict can only name a declared category."""

    model_config = ConfigDict(frozen=True)

    chosen: str
    also_fits: str | None = None
    rationale: str = ""


class CategorySummary(BaseModel):
    """Aggregate of the category judge: ``ratio`` is the mean correctness score
    over successfully-judged pages (exact ``1.0`` / co-equal ``0.5`` / wrong
    ``0.0``), with a deterministic bootstrap CI."""

    model_config = ConfigDict(frozen=True)

    ratio: float
    n_judged: int
    n_errors: int
    ci: tuple[float, float] = (0.0, 0.0)


def parse_category_verdict(
    text: str, *, allowed: frozenset[str]
) -> CategoryVerdict | None:
    """Parse a category-judge response, or ``None`` on any failure.

    ``chosen`` must be one of ``allowed`` (the closed declared set) — an invented
    or re-spelled category is a parse failure, never a silent re-file, mirroring
    synth's own refusal to honour an out-of-set category. ``also_fits`` is a
    *secondary* signal: ``null``, missing, wrong-typed, or not-in-``allowed`` all
    collapse to ``None`` (no co-equal) rather than rejecting the whole verdict —
    a hallucinated second choice can never match the page's real category anyway,
    so dropping it is score-neutral and preserves the primary ``chosen`` signal.
    Strips an optional ```` ```json … ``` ```` fence first.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline > 0 and stripped.endswith("```"):
            stripped = stripped[first_newline + 1 : -3].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    # ``.strip()`` the returned paths before the closed-set check — a stray
    # ``"concept "`` is the LLM's whitespace, not a different category, and
    # shouldn't inflate ``n_errors``. (The allowed set is already NFC-normalized
    # by ``CategoryNode``; the prompt hands the LLM those exact paths to copy.)
    raw_chosen = payload.get("chosen")
    if not isinstance(raw_chosen, str):
        return None
    chosen = raw_chosen.strip()
    if chosen not in allowed:
        return None
    raw_also = payload.get("also_fits")
    also_fits = raw_also.strip() if isinstance(raw_also, str) else None
    if also_fits not in allowed:
        also_fits = None
    raw_rationale = payload.get("rationale")
    rationale = raw_rationale if isinstance(raw_rationale, str) else ""
    try:
        return CategoryVerdict.model_validate(
            {"chosen": chosen, "also_fits": also_fits, "rationale": rationale}
        )
    except ValidationError:
        return None


def _score_category(verdict: CategoryVerdict, actual: str) -> float:
    if verdict.chosen == actual:
        return 1.0
    if verdict.also_fits is not None and verdict.also_fits == actual:
        return 0.5
    return 0.0


def _format_category_prompt(*, body: str, options: Sequence[CategoryOption]) -> str:
    # ``str.replace`` (not ``str.format``) so a page body containing literal
    # ``{`` / ``}`` (a JSON snippet, code) can't raise at format time. Render the
    # closed set first, then inject the body last so body content is never itself
    # treated as a placeholder.
    rendered = "\n".join(f"- `{o.path}`: {o.desc}" for o in options)
    return (
        load_prompt("eval_judge_category")
        .replace("{categories}", rendered)
        .replace("{page_body}", body)
    )


_CATEGORY_SYSTEM = (
    "You are a taxonomy judge. Choose the single best-fit category for the page "
    "from the closed set provided. Return raw JSON only — no prose, no fences."
)


async def judge_category(
    pages: Sequence[KnowledgePage],
    *,
    options: Sequence[CategoryOption],
    llm: LLMProvider,
    model: str,
    sample: int | None = None,
    reporter: ProgressReporter | None = None,
    seed: str = "dikw",
    max_tokens: int = _JUDGE_MAX_TOKENS,
    temperature: float = 0.0,
) -> CategorySummary:
    """Score each page's category assignment, aggregate to a correctness ratio.

    For every page the judge independently picks the best category from
    ``options`` (the declared set + fallback) given the page body, then the
    score compares that to the page's actual ``category``: exact match ``1.0``,
    the judge's co-equal ``also_fits`` matching ``0.5``, otherwise ``0.0``.

    ``sample`` caps the pages judged; selection is seeded via ``hashlib.sha1``
    so repeated runs draw the same subset (``sample`` ≥ ``len(pages)`` is a
    no-op). A per-page LLM exception or parse failure increments ``n_errors``
    and skips the page — one bad response never kills the run.
    """
    _reporter: ProgressReporter = reporter or NoopReporter()
    selected = list(pages)
    if sample is not None and sample < len(selected):
        rng = random.Random(hashlib.sha1(seed.encode("utf-8")).digest()[:8])
        selected = rng.sample(selected, sample)

    allowed = frozenset(o.path for o in options)
    scores: list[float] = []
    n_errors = 0

    for idx, page in enumerate(selected):
        await _reporter.progress(
            phase="category",
            current=idx,
            total=len(selected),
            detail={"path": page.path},
        )
        try:
            response = await llm.complete(
                system=_CATEGORY_SYSTEM,
                user=_format_category_prompt(body=page.body, options=options),
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning("category: LLM call failed for %s — %s", page.path, e)
            n_errors += 1
            continue
        verdict = parse_category_verdict(response.text, allowed=allowed)
        if verdict is None:
            n_errors += 1
            continue
        scores.append(_score_category(verdict, page.category))

    n_judged = len(scores)
    ratio = sum(scores) / n_judged if n_judged else 0.0
    return CategorySummary(
        ratio=ratio,
        n_judged=n_judged,
        n_errors=n_errors,
        ci=bootstrap_ci(scores, seed=f"{seed}:category"),
    )


# --- Wikilink-correctness judge -----------------------------------------------
#
# ``wikilink_resolved_ratio`` counts how many ``[[wikilinks]]`` RESOLVED — it is
# blind to whether each resolved link points at the RIGHT page. The fuzzy
# resolver (NFKC + casefold + punctuation strip + plural stem) deliberately
# absorbs surface variation, so a wrong-referent link (``[[Mercury]]`` in a
# planetary context resolving to the chemical-element page) makes the resolved
# ratio look *better* while silently corrupting the graph. This judge reads each
# resolved link in its body context next to the target page it landed on and
# asks the one question the deterministic resolver cannot: is that the thing the
# context refers to? Karpathy's rule: which links resolved to which page is
# deterministic scoping (the ``links`` table is the engine's truth); whether the
# referent is right is the probabilistic call.


@dataclass(frozen=True)
class WikilinkUnit:
    """One resolved wikilink to judge: the referencing page's identity, the
    body lines around the link (the ``[[...]]`` as written stays visible in
    ``context``), and the target page the engine resolved it to."""

    src_path: str
    src_title: str
    context: str
    target_path: str
    target_title: str
    target_category: str
    target_body: str


class WikilinkSummary(BaseModel):
    """Aggregate of wikilink-correctness verdicts across (sampled) resolved
    links: ``ratio`` is the mean verdict score (right referent ``1.0`` /
    related-but-imprecise ``0.5`` / wrong ``0.0``), with a deterministic
    bootstrap CI. ``ratio == 0.0`` with ``n_judged == 0`` means nothing was
    judged (e.g. the run produced no resolved page-to-page wikilink): the
    caller omits the metric rather than reporting a misleading floor."""

    model_config = ConfigDict(frozen=True)

    ratio: float
    n_judged: int
    n_errors: int
    ci: tuple[float, float] = (0.0, 0.0)


def wikilink_units_from_pages(
    pages: Sequence[KnowledgePage],
    links_by_src_path: Mapping[str, Sequence[LinkRecord]],
    *,
    context_lines: int = 1,
    target_body_cap: int = 1200,
) -> list[WikilinkUnit]:
    """Pair each resolved page→page wikilink with its body context and target.

    ``links_by_src_path`` maps a referencing page's path to its stored
    ``LinkRecord`` rows (``storage.links_from`` — the engine's actual
    resolution, fuzzy results included). Deterministic, no LLM. Skipped rows:

    * non-``WIKILINK`` records (markdown / URL links aren't page references);
    * targets outside ``pages`` (a dangling or non-knowledge ``dst_path`` —
      broken links are ``wikilink_resolved_ratio``'s concern, not this judge's);
    * self-links (a page referencing its own title judges nothing useful);
    * a ``src`` path missing from ``pages`` (a deactivated doc's stale rows).

    ``context`` is the link's body line ± ``context_lines`` (the record's
    1-based body-relative line, clamped to the body's bounds so a stale line
    number degrades to nearby context instead of raising). ``target_body`` is
    capped at ``target_body_cap`` chars — atomic K-pages fit comfortably; the
    cap only guards the judge prompt against a pathological page.
    """
    pages_by_path: dict[str, KnowledgePage] = {p.path: p for p in pages}
    units: list[WikilinkUnit] = []
    for src_path, records in links_by_src_path.items():
        src = pages_by_path.get(src_path)
        if src is None or not src.body:
            continue
        # Split on "\n" ONLY — the link parser's line counter
        # (``links._line_starts``) counts just newline characters, so this is
        # the basis ``rec.line`` was computed in. ``str.splitlines()`` would
        # additionally split on U+2028/U+2029/\x0b/\x0c/\x85 (U+2028 is a known
        # LLM output artifact), shifting every later index and silently handing
        # the judge a window that no longer contains the ``[[wikilink]]``.
        body_lines = src.body.split("\n")
        for rec in records:
            if rec.link_type is not LinkType.WIKILINK:
                continue
            if rec.dst_path == src_path:
                continue
            target = pages_by_path.get(rec.dst_path)
            if target is None:
                continue
            # 1-based → 0-based, clamped into the body's line range.
            center = min(max(rec.line - 1, 0), len(body_lines) - 1)
            lo = max(center - context_lines, 0)
            hi = min(center + context_lines + 1, len(body_lines))
            context = "\n".join(body_lines[lo:hi])
            units.append(
                WikilinkUnit(
                    src_path=src_path,
                    src_title=src.title,
                    context=context,
                    target_path=rec.dst_path,
                    target_title=target.title,
                    target_category=target.category,
                    target_body=target.body[:target_body_cap],
                )
            )
    return units


def _format_wikilink_prompt(*, unit: WikilinkUnit) -> str:
    # Single-pass substitution: all five placeholders are replaced in ONE
    # regex scan over the template, so a page-authored value (title, body,
    # context) that itself contains a literal placeholder token — a templating
    # page showing ``{context}`` in a code example — is never re-expanded.
    # Chained ``str.replace`` would rescan previously injected page text; with
    # four page-authored fields that is a real splice vector, not a theory.
    # (Not ``str.format`` either, which would raise on any literal brace.)
    mapping = {
        "{src_title}": unit.src_title,
        "{target_title}": unit.target_title,
        "{target_category}": unit.target_category,
        "{target_body}": unit.target_body,
        "{context}": unit.context,
    }
    pattern = re.compile("|".join(re.escape(token) for token in mapping))
    return pattern.sub(
        lambda m: mapping[m.group(0)], load_prompt("eval_judge_wikilink")
    )


_WIKILINK_SYSTEM = (
    "You are a wikilink judge. Decide whether the link in the context refers "
    "to the target page shown. Return raw JSON only — no prose, no fences."
)


async def judge_wikilinks(
    units: Sequence[WikilinkUnit],
    *,
    llm: LLMProvider,
    model: str,
    sample: int | None = None,
    reporter: ProgressReporter | None = None,
    seed: str = "dikw",
    max_tokens: int = _JUDGE_MAX_TOKENS,
    temperature: float = 0.0,
) -> WikilinkSummary:
    """Score each resolved wikilink for referent correctness, aggregate to a ratio.

    For every unit the judge sees the link in its body context plus the target
    page the engine resolved it to, and answers ``yes`` / ``partial`` / ``no``
    (``1.0`` / ``0.5`` / ``0.0``) — the same verdict JSON contract as the
    entailment judge, parsed by :func:`parse_entailment_verdict`.

    ``sample`` caps the units judged; selection is seeded via ``hashlib.sha1``
    so repeated runs draw the same subset (``sample`` ≥ ``len(units)`` is a
    no-op). A per-unit LLM exception or parse failure increments ``n_errors``
    and skips the unit — one bad response never kills the run.
    """
    _reporter: ProgressReporter = reporter or NoopReporter()
    selected = list(units)
    if sample is not None and sample < len(selected):
        rng = random.Random(hashlib.sha1(seed.encode("utf-8")).digest()[:8])
        selected = rng.sample(selected, sample)

    scores: list[float] = []
    n_errors = 0

    for idx, unit in enumerate(selected):
        await _reporter.progress(
            phase="wikilink",
            current=idx,
            total=len(selected),
            detail={"path": unit.src_path, "target": unit.target_path},
        )
        try:
            response = await llm.complete(
                system=_WIKILINK_SYSTEM,
                user=_format_wikilink_prompt(unit=unit),
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning(
                "wikilink: LLM call failed for %s — %s", unit.src_path, e
            )
            n_errors += 1
            continue
        # Same ``{"verdict": yes/partial/no, "rationale"}`` contract as the
        # entailment judge — one parser, two judges.
        verdict = parse_entailment_verdict(response.text)
        if verdict is None:
            n_errors += 1
            continue
        scores.append(verdict.score)

    n_judged = len(scores)
    ratio = sum(scores) / n_judged if n_judged else 0.0
    return WikilinkSummary(
        ratio=ratio,
        n_judged=n_judged,
        n_errors=n_errors,
        ci=bootstrap_ci(scores, seed=f"{seed}:wikilink"),
    )


# --- Semantic-atomicity judge ---------------------------------------------------
#
# ``atomicity_score`` is a *form* heuristic — body chars, H1/H2 counts, distinct
# wikilink targets, tag domains (``check_atomicity``, shared with ``dikw client
# lint``). It is blind in both directions: a short single paragraph stuffed with
# three unrelated concepts passes every count, while a thorough single-concept
# page can trip the length counters. This judge asks the semantic question the
# counts cannot: does the page develop exactly ONE atomic concept? (The
# four-dimension page judge does score an ``atomicity`` dim, but bundled with
# three other dimensions in one call — halo-prone — and on a 0-5 scale that
# can't ride the yes/partial/no ratio plumbing the other judge legs share, nor
# their per-leg opt-in flag.) Karpathy's rule: which pages exist is
# deterministic scoping; whether a page is one idea is the probabilistic call.


class SemanticAtomicitySummary(BaseModel):
    """Aggregate of semantic-atomicity verdicts across (sampled) pages:
    ``ratio`` is the mean verdict score (one concept ``1.0`` / dominant concept
    plus a developed tangent ``0.5`` / multiple concepts bolted together
    ``0.0``), with a deterministic bootstrap CI. ``ratio == 0.0`` with
    ``n_judged == 0`` means nothing was judged: the caller omits the metric
    rather than reporting a misleading floor."""

    model_config = ConfigDict(frozen=True)

    ratio: float
    n_judged: int
    n_errors: int
    ci: tuple[float, float] = (0.0, 0.0)


def _format_atomicity_prompt(*, page: KnowledgePage) -> str:
    # Single-pass substitution, mirroring the wikilink prompt fill: both
    # placeholders are page-authored (title AND body), so a title containing a
    # literal ``{page_body}`` token must never have the body expanded into it.
    mapping = {
        "{page_title}": page.title,
        "{page_body}": page.body,
    }
    pattern = re.compile("|".join(re.escape(token) for token in mapping))
    return pattern.sub(
        lambda m: mapping[m.group(0)], load_prompt("eval_judge_atomicity")
    )


_ATOMICITY_SYSTEM = (
    "You are an atomicity judge. Decide whether the page develops exactly one "
    "atomic concept. Return raw JSON only — no prose, no fences."
)


async def judge_semantic_atomicity(
    pages: Sequence[KnowledgePage],
    *,
    llm: LLMProvider,
    model: str,
    sample: int | None = None,
    reporter: ProgressReporter | None = None,
    seed: str = "dikw",
    max_tokens: int = _JUDGE_MAX_TOKENS,
    temperature: float = 0.0,
) -> SemanticAtomicitySummary:
    """Score each page for semantic atomicity, aggregate to a ratio.

    For every page the judge reads title + body — atomicity is intrinsic to
    the page, no source text needed — and answers ``yes`` / ``partial`` /
    ``no`` (``1.0`` / ``0.5`` / ``0.0``), the same verdict JSON contract as
    the entailment judge, parsed by :func:`parse_entailment_verdict`.

    ``sample`` caps the pages judged; selection is seeded via ``hashlib.sha1``
    so repeated runs draw the same subset (``sample`` ≥ ``len(pages)`` is a
    no-op). A per-page LLM exception or parse failure increments ``n_errors``
    and skips the page — one bad response never kills the run.
    """
    _reporter: ProgressReporter = reporter or NoopReporter()
    selected = list(pages)
    if sample is not None and sample < len(selected):
        rng = random.Random(hashlib.sha1(seed.encode("utf-8")).digest()[:8])
        selected = rng.sample(selected, sample)

    scores: list[float] = []
    n_errors = 0

    for idx, page in enumerate(selected):
        await _reporter.progress(
            phase="semantic_atomicity",
            current=idx,
            total=len(selected),
            detail={"path": page.path},
        )
        try:
            response = await llm.complete(
                system=_ATOMICITY_SYSTEM,
                user=_format_atomicity_prompt(page=page),
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning(
                "semantic_atomicity: LLM call failed for %s — %s", page.path, e
            )
            n_errors += 1
            continue
        # Same ``{"verdict": yes/partial/no, "rationale"}`` contract as the
        # entailment + wikilink judges — one parser, three judges.
        verdict = parse_entailment_verdict(response.text)
        if verdict is None:
            n_errors += 1
            continue
        scores.append(verdict.score)

    n_judged = len(scores)
    ratio = sum(scores) / n_judged if n_judged else 0.0
    # Seed suffix distinct from the four-dimension judge's ``:atomicity``
    # stream so the two bootstraps stay self-documentingly independent.
    return SemanticAtomicitySummary(
        ratio=ratio,
        n_judged=n_judged,
        n_errors=n_errors,
        ci=bootstrap_ci(scores, seed=f"{seed}:semantic_atomicity"),
    )
