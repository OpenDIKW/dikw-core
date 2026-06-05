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
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..domains.knowledge.page import KnowledgePage
from ..progress import NoopReporter, ProgressReporter
from ..prompts import load as load_prompt
from ..providers.base import LLMProvider
from ..schemas import ChunkRecord
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
