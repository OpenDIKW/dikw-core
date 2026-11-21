"""Heading-aware markdown chunker.

Strategy (inspired by ``qmd/src/store.ts:257-310`` and
``mineru-doc-explorer/src/store.ts``):

* Split the body into paragraphs separated by blank lines.
* Greedily pack paragraphs into chunks of up to ``max_tokens`` tokens.
* Force a chunk boundary immediately before any heading if the current chunk
  already has substantive content — this keeps headings at chunk starts,
  which matches retrieval intuition.
* Carry the tail of the previous chunk into the next as a paragraph-level
  overlap (~``overlap_ratio`` of ``max_tokens``). Overlap is by paragraph,
  not by word, so character offsets stay exact and no paragraph is split
  mid-sentence.

Token counting goes through ``info/tokenize.count_tokens`` so CJK
paragraphs (which contain no whitespace) get jieba-segmented before the
budget comparison. ASCII bodies fall through to ``len(text.split())``.
``cjk_tokenizer`` plumbs through ``RetrievalConfig.cjk_tokenizer`` so
existing wikis configured with ``"none"`` keep their original chunk
boundaries (and cached embeddings) instead of silently re-chunking.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import NamedTuple

from pydantic import BaseModel

from .tokenize import CjkTokenizer, count_tokens

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_PARA_SEP = re.compile(r"\n\s*\n")

# Bump whenever ``chunk_markdown`` changes emitted chunk boundaries so eval
# snapshot caches (``eval/runner._corpus_cache_key``) rebuild instead of
# silently reusing pre-fix indexes.
CHUNKER_VERSION = 2


class MarkdownChunk(BaseModel):
    """A paragraph-aligned slice of a document body."""

    seq: int
    start: int  # inclusive char offset
    end: int    # exclusive char offset
    text: str


class _Para(NamedTuple):
    start: int
    end: int
    tokens: int


def _paragraph_spans(text: str, *, cjk_tokenizer: CjkTokenizer) -> list[_Para]:
    """Return (start, end, tokens) for each paragraph, preserving char offsets."""
    spans: list[_Para] = []
    if not text.strip():
        return spans
    pos = 0
    n = len(text)
    while pos < n:
        while pos < n and text[pos] in "\r\n":
            pos += 1
        if pos >= n:
            break
        start = pos
        m = _PARA_SEP.search(text, pos)
        if m is None:
            end = n
            pos = n
        else:
            end = m.start()
            pos = m.end()
        # strip trailing whitespace within the paragraph span so tokens is accurate,
        # but keep char offsets pointing into the original body
        para_text = text[start:end]
        stripped_end = end - (len(para_text) - len(para_text.rstrip()))
        if stripped_end <= start:
            continue
        tokens = count_tokens(para_text, tokenizer=cjk_tokenizer)
        if tokens == 0:
            continue
        spans.append(_Para(start, stripped_end, tokens))
    return spans


def chunk_markdown(
    body: str,
    *,
    max_tokens: int = 900,
    overlap_ratio: float = 0.15,
    atomic_spans: Sequence[tuple[int, int]] = (),
    cjk_tokenizer: CjkTokenizer = "jieba",
) -> list[MarkdownChunk]:
    """Chunk ``body`` into (mostly) ``max_tokens``-sized paragraph-aligned windows.

    ``atomic_spans`` declares ``(start, end)`` byte ranges that must NOT be
    split across chunk boundaries — typically the spans of image references
    (``![…](…)`` / ``![[…]]``) returned by ``extract_image_refs``. The v1
    paragraph-aligned chunker honors this naturally for any single-line
    reference; the explicit parameter (a) documents the guarantee on the
    API surface and (b) hard-fails (rather than silently corrupting) if a
    pathological input would split a span. Each span must lie fully within
    at least one chunk; an unsatisfiable span raises ``ValueError``.
    """
    if not body:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1)")

    paras = _paragraph_spans(body, cjk_tokenizer=cjk_tokenizer)
    if not paras:
        return []

    max_overlap = int(max_tokens * overlap_ratio)
    chunks: list[MarkdownChunk] = []
    current: list[_Para] = []
    current_tokens = 0

    def _is_heading(p: _Para) -> bool:
        return bool(_HEADING.match(body[p.start : p.end]))

    def _flush_and_overlap() -> list[_Para]:
        """Emit current chunk and return the tail paragraphs to carry forward.

        Always carries the final paragraph (unless it's oversize on its own —
        that would risk an infinite re-emission loop). Then adds earlier
        paragraphs while staying under the overlap budget.
        """
        if not current:
            return []
        start = current[0].start
        end = current[-1].end
        chunks.append(
            MarkdownChunk(
                seq=len(chunks),
                start=start,
                end=end,
                text=body[start:end],
            )
        )
        overlap: list[_Para] = []
        total = 0
        last = current[-1]
        if last.tokens < max_tokens:
            overlap.append(last)
            total = last.tokens
        for p in reversed(current[:-1]):
            if total + p.tokens > max_overlap:
                break
            overlap.insert(0, p)
            total += p.tokens
        return overlap

    for p in paras:
        heading = _is_heading(p)
        flushed_for_heading = False

        # 1) Start a new chunk before a heading when the current chunk has real content.
        if heading and current_tokens > max_overlap and current:
            carry = _flush_and_overlap()
            current = list(carry)
            current_tokens = sum(c.tokens for c in current)
            flushed_for_heading = True

        # 2) Close the current chunk if adding this paragraph would blow the budget.
        if current and current_tokens + p.tokens > max_tokens:
            if flushed_for_heading:
                # ``current`` holds only tail overlap carried from the chunk just
                # emitted at step 1; flushing again would re-emit an already-
                # covered sub-span. Drop the carry so the heading starts cleanly.
                current = []
                current_tokens = 0
            else:
                carry = _flush_and_overlap()
                current = list(carry)
                current_tokens = sum(c.tokens for c in current)
                # Guard against an endless loop: if the carry itself already overflows
                # (shouldn't happen because of the overlap cap) discard it.
                if current_tokens + p.tokens > max_tokens and current:
                    current = []
                    current_tokens = 0

        current.append(p)
        current_tokens += p.tokens

    if current:
        start = current[0].start
        end = current[-1].end
        chunks.append(
            MarkdownChunk(
                seq=len(chunks), start=start, end=end, text=body[start:end]
            )
        )

    _verify_atomic_spans(chunks, atomic_spans, body)
    return chunks


def _verify_atomic_spans(
    chunks: list[MarkdownChunk],
    atomic_spans: Sequence[tuple[int, int]],
    body: str,
) -> None:
    """Hard-fail if any atomic span crosses a chunk boundary.

    This is a post-condition check, not a corrective pass — the
    paragraph-aligned chunker should never produce a violation for normal
    single-line image refs, so a violation indicates either a pathological
    input (e.g. a multi-line image reference) or a future chunker variant
    that broke the invariant. Either way, silent corruption is worse than
    a loud abort.
    """
    if not atomic_spans:
        return
    for s_start, s_end in atomic_spans:
        if not any(c.start <= s_start and s_end <= c.end for c in chunks):
            preview = body[s_start:s_end][:80]
            raise ValueError(
                f"atomic span [{s_start}:{s_end}] = {preview!r} would be "
                f"split across chunk boundaries; chunks here are "
                f"{[(c.start, c.end) for c in chunks]}"
            )
