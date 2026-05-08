"""Section-derive + group sections for the K-layer fan-out synth pipeline.

The D-layer ``chunk_markdown`` chunker emits chunks with a ~15% paragraph
overlap (``info/chunk.py``). Concatenating those chunks naively for a
synth LLM call would feed the model the same paragraph twice across the
seam. This module exposes two helpers the synth pipeline composes:

* ``derive_sections_from_chunks(body, chunks)`` — cuts the original
  source body at each chunk's ``start`` offset, yielding contiguous,
  non-overlapping sections that together cover the whole body. The trick
  is to use ``chunks[i+1].start`` (not ``chunks[i].end``) as the right
  edge: chunk_markdown is paragraph-aligned, so consecutive ``start``
  values are always at paragraph boundaries even when chunks overlap.

* ``group_sections(sections, target_tokens=...)`` — greedily packs
  sections into ``ChunkGroup``s under a token budget. A section that
  begins with an H1 forces a group break (so a multi-chapter source
  never bleeds chapters into one LLM call), and oversize single
  sections are emitted as their own group.

Both helpers are pure functions over offsets + text — no I/O, no
storage. The synth pipeline owns reading the source body and calling
``Storage.list_chunks(doc_id)``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ...schemas import ChunkRecord
from ..info.tokenize import CjkTokenizer, count_tokens

_HEADING_LINE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", flags=re.MULTILINE)
_H1_LINE = re.compile(r"^\s{0,3}#\s+\S", flags=re.MULTILINE)


@dataclass(frozen=True)
class Section:
    """One contiguous, non-overlapping slice of the source body."""

    start: int
    end: int
    text: str
    headings: tuple[str, ...]
    tokens: int

    @property
    def starts_with_h1(self) -> bool:
        # Only consider the first non-blank line — a body that opens with
        # a paragraph but has an H1 deeper inside should not be treated
        # as an H1-section.
        head = self.text.lstrip("\r\n\t ")
        return bool(_H1_LINE.match(head))


@dataclass(frozen=True)
class ChunkGroup:
    """One LLM-call worth of source body, assembled from contiguous sections."""

    index: int
    text: str
    headings: tuple[str, ...]
    token_count: int
    section_starts: tuple[int, ...]


def _extract_headings(text: str) -> tuple[str, ...]:
    """Return heading text (level marker stripped) in document order."""
    seen: list[str] = []
    for m in _HEADING_LINE.finditer(text):
        title = m.group(2).strip()
        if title:
            seen.append(title)
    return tuple(seen)


def derive_sections_from_chunks(
    body: str,
    chunks: Sequence[ChunkRecord],
    *,
    cjk_tokenizer: CjkTokenizer = "jieba",
) -> list[Section]:
    """Cut ``body`` at chunk boundaries to recover non-overlapping sections.

    Returns ``[]`` when ``chunks`` is empty. Otherwise the union of section
    spans is exactly ``[chunks[0].start, chunks[-1].end)`` — any leading
    or trailing whitespace outside that range (e.g. a YAML front-matter
    block the chunker stripped) is *not* part of any section.
    """
    if not chunks:
        return []
    ordered = sorted(chunks, key=lambda c: c.seq)
    edges = [c.start for c in ordered] + [ordered[-1].end]
    sections: list[Section] = []
    for i in range(len(ordered)):
        start, end = edges[i], edges[i + 1]
        if end <= start:
            # Defensive: a degenerate chunk pair (shouldn't happen with
            # the heading-aware chunker) — skip rather than emit empty.
            continue
        text = body[start:end]
        sections.append(
            Section(
                start=start,
                end=end,
                text=text,
                headings=_extract_headings(text),
                tokens=count_tokens(text, tokenizer=cjk_tokenizer),
            )
        )
    return sections


def group_sections(
    sections: Sequence[Section],
    *,
    target_tokens: int = 3600,
) -> list[ChunkGroup]:
    """Greedy-pack sections into groups under ``target_tokens``.

    Group break rules, in order:

    1. A section that *starts with* an H1 forces a break before it (so
       chapter / book-section boundaries never bleed across LLM calls).
    2. Adding the next section would overshoot ``target_tokens`` *and*
       the current group is non-empty → break before this section.
    3. A single section that is itself larger than ``target_tokens`` is
       emitted as its own group (the LLM gets a longer-than-target call,
       which is preferable to splitting mid-section and stranding
       headings from their bodies).
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    groups: list[ChunkGroup] = []
    current: list[Section] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        text = "".join(s.text for s in current)
        headings: list[str] = []
        seen: set[str] = set()
        for s in current:
            for h in s.headings:
                if h not in seen:
                    seen.add(h)
                    headings.append(h)
        groups.append(
            ChunkGroup(
                index=len(groups),
                text=text,
                headings=tuple(headings),
                token_count=current_tokens,
                section_starts=tuple(s.start for s in current),
            )
        )
        current = []
        current_tokens = 0

    for sec in sections:
        if current and sec.starts_with_h1:
            flush()
        if current and current_tokens + sec.tokens > target_tokens:
            flush()
        current.append(sec)
        current_tokens += sec.tokens

    flush()
    return groups
