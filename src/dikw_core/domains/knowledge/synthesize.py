"""Synthesize K-layer knowledge pages from D-layer source documents.

The LLM emits one or more ``<page>`` XML blocks per call; the parser
turns each into a ``KnowledgePage``. XML output (rather than JSON) avoids
escaping pain and is easy to unit-test with a ``FakeLLM``. Long sources
fan out into multiple LLM calls upstream (see ``grouping.py``);
``dedup_pages_by_slug`` then merges duplicates so the same entity
surfaced from multiple calls collapses into one page.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import yaml

from ...providers.base import LLMProvider
from .page import KnowledgePage, build_page, default_page_path, now_iso

_PAGE_BLOCK = re.compile(
    r"<page\s+([^>]+?)>\s*(.*?)\s*</page>",
    flags=re.DOTALL | re.IGNORECASE,
)
# Used to detect truncated responses: an open ``<page ...>`` tag without
# a matching ``</page>`` close indicates the LLM ran out of tokens
# mid-block. Treating it as a legal "zero pages" response would silently
# drop the truncated page AND mark the source done so it never retries.
_PAGE_OPEN_TAG = re.compile(r"<page\b[^>]*>", flags=re.IGNORECASE)
# finish_reason values that mean the model was cut off mid-generation rather
# than stopping on its own. ``openai_compat`` / ``openai_codex`` normalize to
# ``"length"``; ``anthropic_compat`` passes Anthropic's raw ``stop_reason``
# through, which is ``"max_tokens"``. Both mean the tail was dropped — so a
# clean-but-truncated response (the model obeyed "never open a <page> block
# you cannot finish" and ended without an unclosed tag) is still recoverable
# next run with a bigger budget. MiniMax-M3, the synth workhorse, runs on
# ``anthropic_compat``, so a ``== "length"`` check alone would miss it.
_TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens"})
_ATTR = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", flags=re.DOTALL)
_ATX_TITLE = re.compile(r"^\s{0,3}#\s+(.+?)\s*#*\s*$", flags=re.MULTILINE)


SlugDedupStrategy = Literal["merge_body", "keep_first"]


class SynthesisError(RuntimeError):
    """The LLM response didn't contain a usable ``<page>`` block."""


class SynthesisPartialError(SynthesisError):
    """Some ``<page>`` blocks parsed, others failed.

    Carries the ``pages`` that did parse so the caller can persist what
    succeeded; ``errors`` describes what was lost. ``retry=True`` means
    the missing content can be recovered next run (the response was
    truncated by the token budget — either an unclosed ``<page>`` tag or a
    truncation ``finish_reason`` reported by the provider) — callers should
    bump their parse-error counter so the source is NOT marked done.
    ``retry=False`` means the failure was deterministic (e.g. malformed
    block) and rerunning would just hit the same warning.
    """

    def __init__(
        self,
        message: str,
        *,
        pages: list[KnowledgePage],
        errors: list[str],
        retry: bool = False,
    ) -> None:
        super().__init__(message)
        self.pages = pages
        self.errors = errors
        self.retry = retry


@dataclass(frozen=True)
class SynthesisOutcome:
    page: KnowledgePage
    source_path: str


DEFAULT_ALLOWED_CATEGORIES: tuple[str, ...] = ("entity", "concept", "note")
DEFAULT_FALLBACK: str = "未分类"


def _parse_one_page_block(
    attrs_str: str,
    inner: str,
    *,
    source_path: str,
    allowed_categories: tuple[str, ...],
    fallback: str,
) -> KnowledgePage:
    attrs = dict(_ATTR.findall(attrs_str))

    # The LLM picks a category from the declared (closed) taxonomy; anything
    # it can't confidently place — including an unrecognised / hallucinated
    # value — lands in the ``fallback`` bucket. Karpathy's rule: a wrong
    # category is a cheap re-file, an invented folder is irreversible drift.
    # NFC-normalize first so the membership test shares the config's comparison
    # space — ``config._validate_category_path`` stores ``allowed_categories``
    # in NFC, so a decomposed (NFD) spelling the LLM might emit would otherwise
    # miss a validly-declared accented / Hangul category and fall to fallback.
    category = unicodedata.normalize("NFC", attrs.get("category", "")).strip()
    if category not in allowed_categories:
        category = fallback

    fm_match = _FRONTMATTER.match(inner)
    if fm_match is None:
        frontmatter_yaml: dict[str, Any] = {}
        body = inner.strip()
    else:
        try:
            parsed_fm = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError as e:
            raise SynthesisError(f"invalid YAML front-matter from LLM: {e}") from e
        if not isinstance(parsed_fm, dict):
            raise SynthesisError("front-matter must be a YAML mapping")
        frontmatter_yaml = parsed_fm
        body = fm_match.group(2).lstrip("\n")

    title_match = _ATX_TITLE.search(body)
    if title_match is None:
        raise SynthesisError("no ATX `# Title` found in page body")
    title = title_match.group(1).strip()

    tags = frontmatter_yaml.pop("tags", [])
    if not isinstance(tags, list):
        tags = []

    # The engine owns path construction: ``knowledge/<category>/<slug>.md``.
    # ``category`` is a config-validated closed-set value (or the validated
    # ``fallback``) so it is filesystem-safe by construction; ``slug`` is run
    # through ``slugify`` (inside ``default_page_path``) which strips it to
    # ASCII kebab-case, so no ``..``/backslash traversal can survive. The LLM
    # is told to emit an ASCII slug for non-ASCII titles (``神经网络`` →
    # ``shen-jing-wang-luo``); we slugify whichever of ``slug`` / title it gave.
    slug = attrs.get("slug", "").strip()
    path = default_page_path(category, slug or title)
    return build_page(
        title=title,
        body=body.rstrip() + "\n",
        category=category,
        tags=[str(t) for t in tags],
        sources=[source_path],
        path=path,
        extras={k: v for k, v in frontmatter_yaml.items() if k not in {"tags"}},
    )


def _is_truncation_finish_reason(reason: str | None) -> bool:
    return reason is not None and reason.strip().lower() in _TRUNCATION_FINISH_REASONS


def parse_synthesis_response(
    raw: str,
    *,
    source_path: str,
    allowed_categories: tuple[str, ...] | None = None,
    fallback: str = DEFAULT_FALLBACK,
    finish_reason: str | None = None,
) -> list[KnowledgePage]:
    """Extract one or more ``KnowledgePage`` objects from the LLM's output.

    Returns an empty list when the response contains no ``<page>`` block —
    that's a legal "this section is not worth a knowledge page" signal under
    Stage A's fan-out prompt. Raises ``SynthesisError`` only when there
    are blocks but every one of them failed to parse;
    ``SynthesisPartialError`` carries the surviving pages plus the error
    list when *some* blocks failed.

    ``allowed_categories`` mirrors ``SchemaConfig.category_paths()`` and gates
    which ``category=`` values the parser accepts; an unrecognised value is
    filed under ``fallback`` (``SchemaConfig.fallback``). ``None`` falls back
    to the default ``(entity, concept, note)`` so direct callers (tests, quick
    scripts) don't have to thread config through.

    ``finish_reason`` is the provider's stop signal (``LLMResponse.finish_reason``).
    When it indicates a budget cutoff (``"length"`` / ``"max_tokens"`` — see
    :data:`_TRUNCATION_FINISH_REASONS`) the response was truncated even if every
    ``<page>`` block is closed: the model obeyed "never open a block you cannot
    finish" and dropped the tail cleanly, leaving no unclosed-tag signal. Treat
    it like unclosed-tag truncation — zero parsed blocks becomes a hard
    ``SynthesisError`` (not the legal zero-page signal), and surviving blocks
    raise ``SynthesisPartialError(retry=True)`` so the source is not marked done.
    ``None`` (the default) preserves the tag-only behaviour for callers that
    don't thread it through.
    """
    blocks = list(_PAGE_BLOCK.finditer(raw))
    open_tags = len(_PAGE_OPEN_TAG.findall(raw))
    truncated = max(open_tags - len(blocks), 0)
    length_truncated = _is_truncation_finish_reason(finish_reason)

    if not blocks:
        if truncated > 0 or length_truncated:
            # Zero complete blocks under a truncation signal is a
            # budget-starved cutoff (e.g. a reasoning model spending the
            # whole budget on hidden thinking), NOT the legal "no page worth
            # writing" response — raise so the source is not marked done.
            raise SynthesisError(
                f"LLM response for {source_path} was truncated "
                f"(finish_reason={finish_reason!r}, {truncated} unclosed "
                f"<page> tag(s)) with no complete blocks — likely cut off by "
                f"the token budget"
            )
        return []

    categories = allowed_categories or DEFAULT_ALLOWED_CATEGORIES
    pages: list[KnowledgePage] = []
    errors: list[str] = []
    for m in blocks:
        try:
            pages.append(
                _parse_one_page_block(
                    m.group(1),
                    m.group(2),
                    source_path=source_path,
                    allowed_categories=categories,
                    fallback=fallback,
                )
            )
        except SynthesisError as e:
            errors.append(str(e))

    if truncated > 0:
        # A response with N complete blocks and M unclosed openers means
        # the LLM emitted M+N pages but ran out of tokens mid-write on
        # the last M. Tag retry=True so the caller marks the source as
        # NOT done (the missing content can be recovered next run with a
        # bigger budget).
        errors.append(
            f"detected {truncated} unclosed <page> tag(s) — likely truncated"
        )
    elif length_truncated:
        # All blocks closed but the provider reports a budget cutoff: the
        # model dropped the tail cleanly. No unclosed tag to count, but the
        # missing pages are just as real — flag retry so the source is not
        # marked done.
        errors.append(
            f"finish_reason={finish_reason!r} — response cut off by the "
            f"token budget; tail pages may be missing"
        )

    if errors and not pages:
        raise SynthesisError(
            f"all {len(blocks)} <page> blocks failed for {source_path}: "
            f"{errors[0]}"
        )
    if errors:
        raise SynthesisPartialError(
            f"{len(errors)} issue(s) parsing <page> blocks for {source_path}",
            pages=pages,
            errors=errors,
            retry=truncated > 0 or length_truncated,
        )
    return pages


def dedup_pages_by_slug(
    pages: Sequence[KnowledgePage],
    *,
    strategy: SlugDedupStrategy = "merge_body",
) -> list[KnowledgePage]:
    """Collapse pages that resolve to the same knowledge-page path.

    Stage A fan-out lets the same entity surface in multiple
    ``ChunkGroup`` LLM calls (e.g. "Elon Musk" mentioned across ten
    chapters). Without dedup each group's page would overwrite the
    last on disk, losing earlier contributions.

    * ``merge_body`` (default): keep the first page's metadata, append
      subsequent bodies separated by ``---``, take the union of
      ``tags`` and ``sources``.
    * ``keep_first``: drop subsequent pages with the same path.
    """
    seen: dict[str, KnowledgePage] = {}
    order: list[str] = []

    for p in pages:
        existing = seen.get(p.path)
        if existing is None:
            seen[p.path] = p
            order.append(p.path)
            continue
        if strategy == "keep_first":
            continue
        merged_body = existing.body.rstrip() + "\n\n---\n\n" + p.body.lstrip()
        merged_tags = list(existing.tags) + [t for t in p.tags if t not in existing.tags]
        merged_sources = list(existing.sources) + [
            s for s in p.sources if s not in existing.sources
        ]
        seen[p.path] = replace(
            existing,
            body=merged_body,
            tags=merged_tags,
            sources=merged_sources,
        )

    return [seen[k] for k in order]


def touch(page: KnowledgePage) -> KnowledgePage:
    """Return a copy of ``page`` with ``updated`` bumped to now."""
    return replace(page, updated=now_iso())


# System prompt for the synth fan-out leg; the non-atomic-page lint splitter
# reuses it verbatim, while the orphan-merge / broken-wikilink fixers carry
# their own (`_MERGE_SYSTEM` / `_GROUNDED_SYSTEM`) — guidance added here does
# NOT reach those two. This is the standing-policy spine: it states the page
# invariants (atomicity, faithfulness, reuse, closed taxonomy, honest linking,
# source language) and defers every per-call input AND the operational detail
# those rules lean on — the quantitative length/density norms, the exact output
# format, the category list, the worked examples — to the user-prompt template
# (`synthesize.md`). Each rule has one home so the cached SP and the per-call UP
# do not drift — bar two restated in both tiers on purpose (source language, a
# second-line defence if the UP is truncated; and category-omission-is-a-last-
# resort, reminded at the point of emission), each pinned in both by a guard
# test. anthropic_compat applies prompt caching (`cache_control`) to the system
# prompt — keep it
# byte-stable and free of per-base/per-call content (no `str.format`
# placeholders). Link density is framed as a CEILING here and in the UP
# (honest linking, never "dense"); a system prompt pushing dense linking would
# fight the rules the template states.
DEFAULT_SYNTH_SYSTEM = """You are the **synthesis** component of `dikw-core`, an AI-native knowledge engine that refines raw sources up the Data → Information → Knowledge → Wisdom (DIKW) pyramid. You write its **knowledge (K) layer**: a Zettelkasten of small, atomic, precisely-linked markdown pages, each filed under one path of a closed category taxonomy and cross-referenced with [[wikilinks]].

## Invariants (standing policy — never trade these away)

1. **Atomicity.** Each <page> block captures exactly one self-contained idea, entity, or note — a body answering a single "what / who / why / how about <subject>" question. Split rather than let one page answer two unrelated questions. (Length norms come from the task message.)
2. **Faithfulness.** Preserve facts; never state a claim absent from the source you are given, and never add precision the source does not state — if it says "recent growth", do not write "grew 40% in 2023".
3. **Reuse over regeneration.** When the task message lists an existing page that already covers a candidate at the same granularity, emit no page for it — reference it inline via [[Title]], spelled exactly as listed. Never translate or paraphrase an existing page's title.
4. **Closed taxonomy.** File each page under exactly one category path copied verbatim from the list in the task message. Nearly every page fits a declared path — treat omitting the category attribute as a last resort, never a routine choice. Never invent a category path.
5. **Honest linking.** Write [[Wikilink Title]] inline only where the prose genuinely leans on another page; manufactured links dilute the knowledge graph that retrieval depends on. (Density norms come from the task message.)
6. **Source language.** Emit page titles, the body H1, body paragraphs, tags, and new wikilink titles in the dominant language of the source section — never translate a concept into another language ([[神经网络]], not [[Neural Network]]). The slug is always lowercase ASCII kebab-case.

The exact output format and every per-call input — the category list, this call's section numbers, the knowledge-base context, and the source text — follow in the task message."""
# Underscore alias for legacy callers; new code should use the public name.
_DEFAULT_SYNTH_SYSTEM = DEFAULT_SYNTH_SYSTEM


async def synthesize_pages_from_text(
    *,
    user_prompt: str,
    source_path: str,
    llm: LLMProvider,
    model: str,
    max_tokens: int,
    temperature: float = 0.3,
    allowed_categories: tuple[str, ...] | None = None,
    fallback: str = DEFAULT_FALLBACK,
    system: str = DEFAULT_SYNTH_SYSTEM,
) -> list[KnowledgePage]:
    """One LLM call + parse — the shared "text to N pages" primitive.

    Used by:

    * the ingestion pipeline (``_synth_pages_from_source`` in
      :mod:`api`) for per-chunk-group fan-out, and
    * lint fixers (``broken_wikilink`` evidence-backed grounded repair,
      ``non_atomic_page`` splitter, ``orphan_page``
      merge_into_existing_page) that need to drive the same
      ``llm.complete → parse_synthesis_response`` pair from a different
      prompt template.

    The caller builds the ``user_prompt`` string so each call site can
    pick its own template (the synth prompt for fan-out, the
    ``lint_fix_*`` prompts for fixers). Returns the parsed pages —
    possibly empty if the model emitted no ``<page>`` blocks. Raises
    :class:`SynthesisError` / :class:`SynthesisPartialError` exactly
    like :func:`parse_synthesis_response`; callers decide whether to
    treat partial output as success.
    """
    response = await llm.complete(
        system=system,
        user=user_prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return parse_synthesis_response(
        response.text,
        source_path=source_path,
        allowed_categories=allowed_categories,
        fallback=fallback,
        finish_reason=response.finish_reason,
    )
