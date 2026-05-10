"""Synthesize K-layer wiki pages from D-layer source documents.

The LLM emits one or more ``<page>`` XML blocks per call; the parser
turns each into a ``WikiPage``. XML output (rather than JSON) avoids
escaping pain and is easy to unit-test with a ``FakeLLM``. Long sources
fan out into multiple LLM calls upstream (see ``grouping.py``);
``dedup_pages_by_slug`` then merges duplicates so the same entity
surfaced from multiple calls collapses into one page.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import yaml

from ...providers.base import LLMProvider
from .wiki import WikiPage, build_page, now_iso

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
        pages: list[WikiPage],
        errors: list[str],
        retry: bool = False,
    ) -> None:
        super().__init__(message)
        self.pages = pages
        self.errors = errors
        self.retry = retry


@dataclass(frozen=True)
class SynthesisOutcome:
    page: WikiPage
    source_path: str


DEFAULT_ALLOWED_TYPES: tuple[str, ...] = ("entity", "concept", "note")
# Backwards-compat alias for the few internal call sites still using
# the underscore-prefixed name; new code should import the public one.
_DEFAULT_ALLOWED_TYPES = DEFAULT_ALLOWED_TYPES


def _parse_one_page_block(
    attrs_str: str,
    inner: str,
    *,
    source_path: str,
    allowed_types: tuple[str, ...],
) -> WikiPage:
    attrs = dict(_ATTR.findall(attrs_str))

    type_ = attrs.get("type", "note").strip().lower()
    if type_ not in allowed_types:
        # Fall back to ``note`` if it's allowed (preserves the historical
        # "anything weird becomes a note" behaviour); otherwise pick the
        # first declared type so we don't synthesise an unallowed value.
        type_ = "note" if "note" in allowed_types else allowed_types[0]

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

    path = attrs.get("path") or None
    return build_page(
        title=title,
        body=body.rstrip() + "\n",
        type_=type_,
        tags=[str(t) for t in tags],
        sources=[source_path],
        path=path,
        extras={k: v for k, v in frontmatter_yaml.items() if k not in {"tags"}},
    )


def parse_synthesis_response(
    raw: str,
    *,
    source_path: str,
    allowed_types: tuple[str, ...] | None = None,
) -> list[WikiPage]:
    """Extract one or more ``WikiPage`` objects from the LLM's output.

    Returns an empty list when the response contains no ``<page>`` block —
    that's a legal "this section is not worth a wiki page" signal under
    Stage A's fan-out prompt. Raises ``SynthesisError`` only when there
    are blocks but every one of them failed to parse;
    ``SynthesisPartialError`` carries the surviving pages plus the error
    list when *some* blocks failed.

    ``allowed_types`` mirrors ``SchemaConfig.page_types`` and gates which
    ``type=`` values the parser accepts. ``None`` falls back to the
    default ``(entity, concept, note)`` so direct callers (tests, quick
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

    types = allowed_types or _DEFAULT_ALLOWED_TYPES
    pages: list[WikiPage] = []
    errors: list[str] = []
    for m in blocks:
        try:
            pages.append(
                _parse_one_page_block(
                    m.group(1),
                    m.group(2),
                    source_path=source_path,
                    allowed_types=types,
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
    pages: Sequence[WikiPage],
    *,
    strategy: SlugDedupStrategy = "merge_body",
) -> list[WikiPage]:
    """Collapse pages that resolve to the same wiki path.

    Stage A fan-out lets the same entity surface in multiple
    ``ChunkGroup`` LLM calls (e.g. "Elon Musk" mentioned across ten
    chapters). Without dedup each group's page would overwrite the
    last on disk, losing earlier contributions.

    * ``merge_body`` (default): keep the first page's metadata, append
      subsequent bodies separated by ``---``, take the union of
      ``tags`` and ``sources``.
    * ``keep_first``: drop subsequent pages with the same path.
    """
    seen: dict[str, WikiPage] = {}
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


def touch(page: WikiPage) -> WikiPage:
    """Return a copy of ``page`` with ``updated`` bumped to now."""
    return replace(page, updated=now_iso())


DEFAULT_SYNTH_SYSTEM = "You synthesise K-layer wiki pages for dikw-core."
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
    allowed_types: tuple[str, ...] | None = None,
    system: str = DEFAULT_SYNTH_SYSTEM,
) -> list[WikiPage]:
    """One LLM call + parse — the shared "text to N pages" primitive.

    Used by:

    * the ingestion pipeline (``_synth_pages_from_source`` in
      :mod:`api`) for per-chunk-group fan-out, and
    * lint fixers (``broken_wikilink`` LLM stub fallback,
      ``non_atomic_page`` splitter) that need to drive the same
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
        allowed_types=allowed_types,
    )
