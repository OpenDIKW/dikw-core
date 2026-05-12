"""Retrieval and K-layer (synth) quality metrics.

Retrieval section (the historical content of this module):
    Each query's ground truth is an ``expect_any`` set — retrieval
    succeeds if **any** member appears in the top-k ranked results. This
    matches how dogfood Q/A is authored (paraphrased answers often live
    in multiple docs; requiring all of them would be artificially
    punitive). ``ndcg_at_k`` and ``recall_at_k`` exposed for public
    benchmark calibration (BEIR/CMTEB).

K-layer (synth) section:
    Deterministic per-page metrics that quantify synth output quality
    without an LLM judge. The same heuristics drive ``run_synth_eval``'s
    hard gate, so a single source of truth governs both ``dikw lint``
    surfacing and PR-blocking thresholds.

All retrieval functions are pure and synchronous. K-layer functions are
mostly pure-sync; the two embedding-driven ones (``fact_grounding_ratio``,
``duplicate_ratio_max``) are async so they can call ``EmbeddingProvider``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from ..domains.knowledge.links import normalize_for_match
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
#
# Two families. The pure-sync ones (atomicity, expected_coverage,
# wikilink_resolved_ratio, language_fidelity, page_density) just inspect
# WikiPage state. The two async ones (fact_grounding_ratio,
# duplicate_ratio_max) drive an EmbeddingProvider — semantic similarity
# is the only deterministic way to spot "this claim isn't in the source"
# / "these two pages are near-duplicates" without an LLM judge.

# Regex for stripping markdown structure before sentence-splitting.
# Code fences come from the lint module so atomicity and grounding share
# one strip rule. Heading lines (ATX style only — setext is rare in
# generated wiki pages) are removed because they're titles, not claims.
# Wikilinks are stripped entirely (not replaced by alias text) so a body
# that's just ``[[Other]]`` produces zero claim sentences instead of a
# misleading "Other" claim; prose like ``mentions [[Alice]] briefly``
# still yields ``mentions briefly`` as a legitimate (though stub) claim.
_HEADING_LINE = re.compile(r"^\s{0,3}#+\s+.*$", flags=re.MULTILINE)
_WIKILINK_INLINE = re.compile(r"\[\[([^\]\|\n]+?)(?:\|([^\]\n]+?))?\]\]")
# Sentence boundary: zero-width after ``.`` / ``。`` so we don't require
# whitespace after the period (CJK rarely has it), plus blank-line splits
# for paragraph-style claims.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.。])|\n{2,}")

# Language classification — ASCII-vs-CJK character-ratio heuristic on the
# first 200 chars. No external dependency: see spec Open question (c) —
# langdetect is LGPL + heavy for a single 0.95-target metric. The
# character class covers Han (U+4E00-9FFF), Hiragana (U+3040-309F),
# Katakana (U+30A0-30FF), and Hangul Syllables (U+AC00-D7AF).
# Using escape sequences (not literal CJK glyphs) keeps the source file
# editor-friendly across font setups and silences ruff's RUF001 warning.
_CJK_CHAR = re.compile(
    "[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]"
)
_ASCII_LETTER = re.compile(r"[A-Za-z]")
_LANG_SAMPLE_PREFIX = 200
_LANG_CJK_RATIO_THRESHOLD = 0.3


def _split_claims(body: str) -> list[str]:
    """Tokenise a wiki-page body into claim-bearing sentences.

    Strips fenced code, headings, and wikilink markup (keeping the alias
    text so embedding sees real words), then splits on sentence terminators
    plus paragraph breaks. Returns the trimmed non-empty fragments.
    """
    text = _FENCED_CODE.sub("", body)
    text = _HEADING_LINE.sub("", text)
    text = _WIKILINK_INLINE.sub(" ", text)
    parts = _SENTENCE_BOUNDARY.split(text)
    out: list[str] = []
    for p in parts:
        cleaned = p.strip()
        if cleaned:
            out.append(cleaned)
    return out


def _classify_lang(text: str) -> Literal["en", "cjk", "other"]:
    """Cheap language classifier — CJK char ratio over the first 200 chars."""
    sample = text[:_LANG_SAMPLE_PREFIX]
    if not sample.strip():
        return "other"
    cjk_count = len(_CJK_CHAR.findall(sample))
    ascii_count = len(_ASCII_LETTER.findall(sample))
    total = cjk_count + ascii_count
    if total == 0:
        return "other"
    if cjk_count / total > _LANG_CJK_RATIO_THRESHOLD:
        return "cjk"
    if ascii_count > 0:
        return "en"
    return "other"


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Assumes both vectors are L2-normalised (the
    contract for ``EmbeddingProvider.embed`` and ``FakeEmbeddings``);
    dot product equals cosine in that case."""
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
    storage). Both sides go through ``_classify_lang`` on the first 200
    chars; ``"other"`` matching ``"other"`` is still a match so empty-
    bodied pages don't false-mismatch.
    """
    if not pages_with_sources:
        return 1.0
    matched = sum(
        1
        for page, source_text in pages_with_sources
        if _classify_lang(page.body) == _classify_lang(source_text)
    )
    return matched / len(pages_with_sources)


def page_density(*, n_pages: int, n_chunks: int) -> float:
    """``pages / chunks``. Informational — no threshold direction.

    Zero chunks returns 0.0 (no input to generate from, so the ratio is
    meaningless but ``NaN`` would break threshold comparison)."""
    if n_chunks <= 0:
        return 0.0
    return n_pages / n_chunks


async def fact_grounding_ratio(
    *,
    pages_with_sources: Sequence[tuple[WikiPage, str]],
    chunks_by_source: Mapping[str, Sequence[ChunkRecord]],
    embedder: EmbeddingProvider,
    embedding_model: str,
    tau: float,
) -> float:
    """Fraction of page claims whose nearest source chunk has cosine ≥ tau.

    Per page: split body into claim sentences, embed claims + chunks of
    the originating source in two batches, take per-claim max cosine
    against the chunk set; claim is "grounded" if max ≥ tau. Page score
    = grounded / total_claims. Final ratio = mean across pages.

    Pages with no claim sentences (only headings or wikilinks) score 1.0
    — nothing to ground, so they don't unfairly tank the aggregate.
    Pages whose source has zero chunks score 0.0 (we can't verify them).
    """
    if not pages_with_sources:
        return 1.0
    per_page: list[float] = []
    for page, source_path in pages_with_sources:
        claims = _split_claims(page.body)
        if not claims:
            per_page.append(1.0)
            continue
        chunks = chunks_by_source.get(source_path, [])
        if not chunks:
            per_page.append(0.0)
            continue
        claim_embeds = await embedder.embed(claims, model=embedding_model)
        chunk_embeds = await embedder.embed(
            [c.text for c in chunks], model=embedding_model
        )
        grounded = 0
        for ce in claim_embeds:
            best = max((_cosine(ce, ch) for ch in chunk_embeds), default=0.0)
            if best >= tau:
                grounded += 1
        per_page.append(grounded / len(claims))
    return sum(per_page) / len(per_page)


async def duplicate_ratio_max(
    *,
    pages: Sequence[WikiPage],
    embedder: EmbeddingProvider,
    embedding_model: str,
    tau: float,
) -> float:
    """Fraction of distinct page pairs whose body cosine ≥ tau.

    Total pairs = ``n*(n-1)/2``. Reverse-direction metric: lower is
    better. Fewer than two pages → 0.0 (no pair to compare)."""
    if len(pages) < 2:
        return 0.0
    embeds = await embedder.embed(
        [p.body for p in pages], model=embedding_model
    )
    n = len(pages)
    total_pairs = n * (n - 1) // 2
    above = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _cosine(embeds[i], embeds[j]) >= tau:
                above += 1
    return above / total_pairs
