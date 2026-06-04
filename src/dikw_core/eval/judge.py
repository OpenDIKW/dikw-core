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

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..domains.knowledge.page import KnowledgePage
from ..progress import NoopReporter, ProgressReporter
from ..prompts import load as load_prompt
from ..providers.base import LLMProvider

logger = logging.getLogger(__name__)


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
    max_tokens: int = 512,
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
