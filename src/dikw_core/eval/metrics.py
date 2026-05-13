"""Retrieval and K-layer (synth) quality metrics.

Retrieval metrics use binary ``expect_any`` ground truth — a query is a
hit at k if **any** listed identity appears in the top-k ranked results.
Paraphrased dogfood Q/A often lives in multiple docs; requiring all
would be artificially punitive. ``ndcg_at_k`` / ``recall_at_k`` exposed
for BEIR/CMTEB calibration.

K-layer metrics quantify synth output without an LLM judge. The
``atomicity_score`` shares ``check_atomicity`` with ``dikw lint`` so
one heuristic governs both interactive surfacing and the hard gate.

The two embedding-driven K-layer metrics (``fact_grounding_ratio`` /
``duplicate_ratio_max``) are ``async`` so they can drive an
``EmbeddingProvider``; everything else is pure-sync.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ..domains.knowledge.links import WIKILINK_RE, normalize_for_match
from ..domains.knowledge.lint import _FENCED_CODE, check_atomicity
from ..domains.knowledge.wiki import WikiPage
from ..providers.base import EmbeddingProvider
from ..schemas import ChunkRecord


def hit_at_k(ranked: Sequence[str], expected_any: Iterable[str], k: int) -> float:
    """1.0 if any ``expected_any`` is in ``ranked[:k]``, else 0.0."""
    if k <= 0:
        return 0.0
    top = set(ranked[:k])
    return 1.0 if any(e in top for e in expected_any) else 0.0


def reciprocal_rank(ranked: Sequence[str], expected_any: Iterable[str]) -> float:
    """1 / rank of the first ``expected_any`` match (1-indexed); 0.0 if none."""
    expected = set(expected_any)
    for idx, key in enumerate(ranked, start=1):
        if key in expected:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(ranked: Sequence[str], expected_any: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k.

    DCG = Σ rel_i / log2(i+1) for i in 1..k (rel_i ∈ {0, 1}).
    IDCG = same with all relevant docs ranked first (capped at k).
    """
    if k <= 0:
        return 0.0
    expected = set(expected_any)
    if not expected:
        return 0.0
    dcg = 0.0
    for idx, key in enumerate(ranked[:k], start=1):
        if key in expected:
            dcg += 1.0 / math.log2(idx + 1)
    n_rel = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], expected_any: Iterable[str], k: int) -> float:
    """|hits ∩ expected| / |expected|, capped at k."""
    if k <= 0:
        return 0.0
    expected = set(expected_any)
    if not expected:
        return 0.0
    top = set(ranked[:k])
    return len(top & expected) / len(expected)


def mean_hit_at_k(
    results: Sequence[tuple[Sequence[str], Iterable[str]]], k: int
) -> float:
    """Average ``hit_at_k`` across queries. Empty input returns 0.0."""
    if not results:
        return 0.0
    return sum(hit_at_k(r, e, k) for r, e in results) / len(results)


def mean_reciprocal_rank(
    results: Sequence[tuple[Sequence[str], Iterable[str]]],
) -> float:
    """Average ``reciprocal_rank`` across queries. Empty input returns 0.0."""
    if not results:
        return 0.0
    return sum(reciprocal_rank(r, e) for r, e in results) / len(results)


def mean_ndcg_at_k(
    results: Sequence[tuple[Sequence[str], Iterable[str]]], k: int
) -> float:
    """Average ``ndcg_at_k`` across queries. Empty input returns 0.0."""
    if not results:
        return 0.0
    return sum(ndcg_at_k(r, e, k) for r, e in results) / len(results)


def mean_recall_at_k(
    results: Sequence[tuple[Sequence[str], Iterable[str]]], k: int
) -> float:
    """Average ``recall_at_k`` across queries. Empty input returns 0.0."""
    if not results:
        return 0.0
    return sum(recall_at_k(r, e, k) for r, e in results) / len(results)


# ===========================================================================
# K-layer (synth) quality metrics
# ===========================================================================

# ATX-style headings; setext is rare in LLM-generated wiki pages.
_HEADING_LINE = re.compile(r"^\s{0,3}#+\s+.*$", flags=re.MULTILINE)
# Zero-width sentence boundary after ``.`` / ``。`` (CJK rarely has a space
# after the full stop) plus blank-line splits for paragraph-style claims.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.。])|\n{2,}")

# Han (U+4E00-9FFF) + Hiragana (U+3040-309F) + Katakana (U+30A0-30FF) +
# Hangul Syllables (U+AC00-D7AF). Escape sequences (not literal glyphs)
# silence ruff RUF001 and keep the source file editor-portable.
# Avoid ``langdetect`` (LGPL + heavy) for a single 0.95-target metric on
# the en/zh/ja/ko range.
_CJK_CHAR = re.compile(
    "[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]"
)
_ASCII_LETTER = re.compile(r"[A-Za-z]")
_LANG_SAMPLE_PREFIX = 200
_LANG_CJK_RATIO_THRESHOLD = 0.3


# Filter junk fragments that ``_SENTENCE_BOUNDARY.split`` lets through.
# The 2026-05-13 tau-sweep dump exposed ~30 garbage "claims" (``.``,
# ``embed(.``, ``)`` from code residue, single ``"``) that survived
# the empty-string check and matched arbitrary chunks at low cosine,
# dragging ``fact_grounding_ratio`` down at every tau without carrying
# any factual signal. A claim must clear EITHER the en threshold
# (≥ 3 word tokens of ≥ 2 letters each) OR the cjk threshold (≥ 4
# CJK characters), so the filter works on bilingual corpora.
_CLAIM_EN_WORD = re.compile(r"[A-Za-z]{2,}")
# Same CJK ranges as ``_CJK_CHAR`` above — written via unicode escapes
# so the regex source doesn't trip ruff's RUF001 ambiguous-character
# lint on the literal-CJK-in-source line.
_CLAIM_CJK = re.compile(
    "[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]"
)
_CLAIM_MIN_EN_WORDS = 3
_CLAIM_MIN_CJK_CHARS = 4


def _has_claim_substance(text: str) -> bool:
    return (
        len(_CLAIM_EN_WORD.findall(text)) >= _CLAIM_MIN_EN_WORDS
        or len(_CLAIM_CJK.findall(text)) >= _CLAIM_MIN_CJK_CHARS
    )


def split_claims(body: str) -> list[str]:
    """Tokenise a wiki-page body into claim-bearing sentences.

    Strips fenced code, headings, and wikilink markup (replaced by space
    so a body that's just ``[[Other]]`` yields zero claims and doesn't
    embed-match against a misleading "Other" word), then splits on
    sentence terminators plus paragraph breaks.

    Fragments with fewer than ``_CLAIM_MIN_LETTERS`` letters / CJK
    characters are dropped — punctuation-only splits (``"."``, ``")``,
    code-snippet residue like ``embed(.``) are not claims and would
    embed-match against arbitrary chunks at low cosine, lowering the
    grounding ratio without carrying any factual signal.
    """
    text = _FENCED_CODE.sub("", body)
    text = _HEADING_LINE.sub("", text)
    text = WIKILINK_RE.sub(" ", text)
    out: list[str] = []
    for p in _SENTENCE_BOUNDARY.split(text):
        stripped = p.strip()
        if not stripped:
            continue
        if not _has_claim_substance(stripped):
            continue
        out.append(stripped)
    return out


def classify_lang(text: str) -> Literal["en", "cjk", "other"]:
    """CJK-char-ratio language classifier over the first 200 chars.

    Returns ``"other"`` when the sample has no countable CJK or ASCII
    characters (empty, pure punctuation, etc.).
    """
    sample = text[:_LANG_SAMPLE_PREFIX]
    cjk_count = len(_CJK_CHAR.findall(sample))
    ascii_count = len(_ASCII_LETTER.findall(sample))
    total = cjk_count + ascii_count
    if total == 0:
        return "other"
    if cjk_count / total > _LANG_CJK_RATIO_THRESHOLD:
        return "cjk"
    return "en"


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product — equals cosine when both vectors are L2-normalised,
    which is the contract for ``EmbeddingProvider.embed``."""
    return sum(x * y for x, y in zip(a, b, strict=False))


def expected_coverage(
    *,
    page_titles: Iterable[str],
    expected_titles: Iterable[str],
) -> float:
    """Fraction of ``expected_titles`` that a generated page covers.

    Both sides go through ``normalize_for_match`` so the comparison shares
    the same fuzzy semantics as wikilink resolution: ``"Neural Networks"``
    matches ``"Neural Network"``, ``"Elon Musk."`` matches ``"elon musk"``,
    etc.

    Empty ``expected_titles`` → 1.0 (vacuous), so a dataset without an
    ``expected.yaml`` doesn't tank this metric.
    """
    expected_list = list(expected_titles)
    if not expected_list:
        return 1.0
    page_norms = {normalize_for_match(t) for t in page_titles}
    page_norms.discard("")
    matched = sum(
        1 for et in expected_list if normalize_for_match(et) in page_norms
    )
    return matched / len(expected_list)


def wikilink_resolved_ratio(*, total: int, unresolved: int) -> float:
    """``(total - unresolved) / total``; zero-denominator returns 1.0.

    Reads directly from ``SynthReport`` counters — no re-parsing of bodies.
    """
    if total <= 0:
        return 1.0
    return (total - unresolved) / total


def atomicity_score(pages: Sequence[WikiPage]) -> float:
    """``1 - non_atomic / total``. Empty input returns 1.0 (no failures)."""
    if not pages:
        return 1.0
    atomic_count = sum(
        1 for p in pages if check_atomicity(body=p.body, tags=p.tags).atomic
    )
    return atomic_count / len(pages)


def language_fidelity(
    pages_with_sources: Sequence[tuple[WikiPage, str]],
) -> float:
    """Fraction of pages whose dominant language matches their source's.

    Source is supplied as raw text (runner provides this from disk or
    storage). Both sides go through ``classify_lang`` on the first 200
    chars; ``"other"`` matching ``"other"`` is still a match so empty-
    bodied pages don't false-mismatch.
    """
    if not pages_with_sources:
        return 1.0
    matched = sum(
        1
        for page, source_text in pages_with_sources
        if classify_lang(page.body) == classify_lang(source_text)
    )
    return matched / len(pages_with_sources)


def page_density(*, n_pages: int, n_chunks: int) -> float:
    """``pages / chunks``. Informational — no threshold direction.

    Zero chunks returns 0.0 (no input to generate from, so the ratio is
    meaningless but ``NaN`` would break threshold comparison)."""
    if n_chunks <= 0:
        return 0.0
    return n_pages / n_chunks


@dataclass(frozen=True)
class GroundingClaim:
    """One claim sentence + its peak cosine against the page's source chunks.

    Emitted by :func:`compute_grounding_cosines` so callers (the metric,
    tau-sweep scripts, debug dumpers) can apply any threshold without
    re-running the embedder. ``page_path`` and ``source_path`` are kept
    for hand-labelling and per-source breakdowns.
    """

    page_path: str
    source_path: str
    claim: str
    max_cosine: float


async def compute_grounding_cosines(
    *,
    pages_with_sources: Sequence[tuple[WikiPage, str]],
    chunks_by_source: Mapping[str, Sequence[ChunkRecord]],
    embedder: EmbeddingProvider,
    embedding_model: str,
) -> list[GroundingClaim]:
    """Compute the peak cosine for every claim sentence in every page.

    The expensive half of :func:`fact_grounding_ratio` — embed each
    source's chunks once, embed each page's claims, take per-claim max
    cosine against the source's chunks. Returned as a flat list so the
    caller can reduce at any tau (the metric does ``ratio = (count ≥ tau)
    / total``; the tau-sweep script does the same for several taus).

    Pages with no claim sentences emit nothing — same semantics as the
    metric's ``1.0`` floor (vacuous truth). Pages whose source has zero
    chunks emit one ``max_cosine=-inf`` entry per claim — so the metric
    sees them as ungrounded at every tau.
    """
    if not pages_with_sources:
        return []
    chunk_embeds_cache: dict[str, list[list[float]]] = {}

    async def _chunks_for(source_path: str) -> list[list[float]]:
        cached = chunk_embeds_cache.get(source_path)
        if cached is not None:
            return cached
        chunks = chunks_by_source.get(source_path, [])
        # Defensive: skip empty / whitespace-only chunks. Gitee /
        # OpenAI embedding APIs 400 on empty input rather than emitting
        # a zero vector, which would tank the whole run.
        chunk_texts = [c.text for c in chunks if c.text.strip()]
        if not chunk_texts:
            chunk_embeds_cache[source_path] = []
            return []
        embeds = await _embed_batched(
            embedder, chunk_texts, model=embedding_model
        )
        chunk_embeds_cache[source_path] = embeds
        return embeds

    out: list[GroundingClaim] = []
    for page, source_path in pages_with_sources:
        claims = [c for c in split_claims(page.body) if c.strip()]
        if not claims:
            continue
        chunk_embeds = await _chunks_for(source_path)
        if not chunk_embeds:
            for claim in claims:
                out.append(
                    GroundingClaim(
                        page_path=page.path,
                        source_path=source_path,
                        claim=claim,
                        max_cosine=float("-inf"),
                    )
                )
            continue
        claim_embeds = await _embed_batched(
            embedder, claims, model=embedding_model
        )
        for claim, ce in zip(claims, claim_embeds, strict=True):
            best = max((_cosine(ce, ch) for ch in chunk_embeds), default=0.0)
            out.append(
                GroundingClaim(
                    page_path=page.path,
                    source_path=source_path,
                    claim=claim,
                    max_cosine=best,
                )
            )
    return out


def reduce_grounding_ratio(
    claims: Sequence[GroundingClaim],
    *,
    pages_with_sources: Sequence[tuple[WikiPage, str]],
    tau: float,
) -> float:
    """Apply tau to per-claim cosines → per-page ratios → mean across pages.

    Pure function. Same semantics as :func:`fact_grounding_ratio`:
    pages with no claims score 1.0 (vacuous), pages whose source had
    zero chunks score 0.0 (emitted as ``-inf`` claims by
    :func:`compute_grounding_cosines`).
    """
    if not pages_with_sources:
        return 1.0
    by_page: dict[str, list[float]] = {}
    for c in claims:
        by_page.setdefault(c.page_path, []).append(c.max_cosine)
    per_page: list[float] = []
    for page, _ in pages_with_sources:
        cosines = by_page.get(page.path)
        if not cosines:
            per_page.append(1.0)
            continue
        grounded = sum(1 for cos in cosines if cos >= tau)
        per_page.append(grounded / len(cosines))
    return sum(per_page) / len(per_page)


async def fact_grounding_ratio(
    *,
    pages_with_sources: Sequence[tuple[WikiPage, str]],
    chunks_by_source: Mapping[str, Sequence[ChunkRecord]],
    embedder: EmbeddingProvider,
    embedding_model: str,
    tau: float,
) -> float:
    """Fraction of page claims whose nearest source chunk has cosine ≥ tau.

    Per page: split body into claim sentences, take per-claim max cosine
    against the source's chunk embeddings; claim is "grounded" if max ≥
    tau. Page score = grounded / total_claims. Final ratio = mean across
    pages.

    Pages with no claim sentences (only headings or wikilinks) score 1.0
    — nothing to ground, so they don't unfairly tank the aggregate.
    Pages whose source has zero chunks score 0.0 (we can't verify them).

    Each source's chunks are embedded once (not once per referencing page)
    — for a 100-page wiki with 10 sources, that's 10 chunk-embed calls
    instead of 100, the dominant cost in real-LLM runs.
    """
    claims = await compute_grounding_cosines(
        pages_with_sources=pages_with_sources,
        chunks_by_source=chunks_by_source,
        embedder=embedder,
        embedding_model=embedding_model,
    )
    return reduce_grounding_ratio(
        claims, pages_with_sources=pages_with_sources, tau=tau
    )


# Most embedding providers (Gitee, MiniMax, OpenAI batch tier-2) cap
# ``input`` length per request well below the synth-eval payload size —
# Gitee rejects ~30+ texts in one shot with a cryptic
# "Validation error for body application/json: No schema matches"
# 400. Batch every direct ``embedder.embed`` call from this module at
# 16 items, matching ``DikwConfig.provider.embedding_batch_size``'s
# default. ``EmbeddingProvider`` doesn't batch internally because the
# synth + ingest paths route through ``consume_embedding_stream``
# which already batches; eval-only callers don't go through that path.
_EMBED_BATCH = 16


async def _embed_batched(
    embedder: EmbeddingProvider,
    texts: list[str],
    *,
    model: str,
) -> list[list[float]]:
    if not texts:
        return []
    out: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        chunk = texts[start : start + _EMBED_BATCH]
        out.extend(await embedder.embed(chunk, model=model))
    return out


async def duplicate_ratio_max(
    *,
    pages: Sequence[WikiPage],
    embedder: EmbeddingProvider,
    embedding_model: str,
    tau: float,
) -> float:
    """Fraction of distinct page pairs whose body cosine ≥ tau.

    Total pairs = ``n*(n-1)/2``. Reverse-direction metric: lower is
    better. Fewer than two pages → 0.0 (no pair to compare).

    Pages with empty / whitespace-only body are excluded from the pair
    set — they're degenerate synth output, not duplicates, and most
    embedding APIs (Gitee, OpenAI) 400 on empty input strings rather
    than emitting a zero vector.
    """
    bodied = [p for p in pages if p.body.strip()]
    if len(bodied) < 2:
        return 0.0
    embeds = await _embed_batched(
        embedder, [p.body for p in bodied], model=embedding_model
    )
    n = len(bodied)
    total_pairs = n * (n - 1) // 2
    above = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _cosine(embeds[i], embeds[j]) >= tau:
                above += 1
    return above / total_pairs
