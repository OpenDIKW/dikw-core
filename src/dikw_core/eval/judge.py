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
    """Aggregate of judge scores across all (or sampled) pages."""

    model_config = ConfigDict(frozen=True)

    n_judged: int
    n_errors: int
    mean_grounding: float
    mean_atomicity: float
    mean_completeness: float
    mean_clarity: float
    per_page: list[PageJudgeEntry] = Field(default_factory=list)


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
    return JudgeSummary(
        n_judged=n_judged,
        n_errors=n_errors,
        mean_grounding=sum(e.score.grounding for e in per_page) / n_judged,
        mean_atomicity=sum(e.score.atomicity for e in per_page) / n_judged,
        mean_completeness=sum(e.score.completeness for e in per_page) / n_judged,
        mean_clarity=sum(e.score.clarity for e in per_page) / n_judged,
        per_page=per_page,
    )
