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
    the missing content can be recovered next run (e.g. the response was
    truncated by max_tokens) — callers should bump their parse-error
    counter so the source is NOT marked done. ``retry=False`` means the
    failure was deterministic (e.g. malformed block) and rerunning would
    just hit the same warning.
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


def parse_synthesis_response(
    raw: str,
    *,
    source_path: str,
    allowed_categories: tuple[str, ...] | None = None,
    fallback: str = DEFAULT_FALLBACK,
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
    """
    blocks = list(_PAGE_BLOCK.finditer(raw))
    open_tags = len(_PAGE_OPEN_TAG.findall(raw))
    truncated = max(open_tags - len(blocks), 0)

    if not blocks:
        if truncated > 0:
            raise SynthesisError(
                f"LLM response for {source_path} contains {truncated} "
                f"unclosed <page> tag(s) and no complete blocks — likely "
                f"truncated by max_tokens"
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
            retry=truncated > 0,
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


# Default system prompt for the synth fan-out leg; the non-atomic-page lint
# splitter reuses it verbatim, while the orphan-merge / broken-wikilink fixers
# carry their own (`_MERGE_SYSTEM` / `_GROUNDED_SYSTEM`) — guidance added here
# does NOT reach those two. anthropic_compat applies prompt caching
# (`cache_control`) to the system prompt — keep it byte-stable, free of
# per-base/per-call content, and aligned with the user prompt's
# link-density-as-ceiling framing (a system prompt pushing "dense" linking
# would fight the rules the template states).
DEFAULT_SYNTH_SYSTEM = (
    "You synthesise the knowledge (K) layer of `dikw-core`: a Zettelkasten of "
    "small, atomic notes. Each page captures one self-contained idea, entity, "
    "or note that stands on its own, connected to related pages through "
    "precise [[wikilinks]]: link where a reference genuinely clarifies, and "
    "reuse an existing page rather than regenerating a near-duplicate. A page "
    "should be complete on its single subject; split a page that conflates "
    "subjects instead of letting it sprawl. Preserve the dominant language of "
    "the source section in page titles, body text, tags, and new wikilink "
    "titles."
)
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
    )
