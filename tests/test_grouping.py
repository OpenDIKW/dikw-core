"""Unit tests for section derivation + group packing.

The non-overlap invariant is the load-bearing piece — if it ever
regressed, the synth LLM would see ~15% of every section twice across
seams and start hallucinating duplicates. The first test exercises it
against a real ``chunk_markdown`` output (which by default produces
overlapping chunks).
"""

from __future__ import annotations

from dikw_core.domains.info.chunk import chunk_markdown
from dikw_core.domains.knowledge.grouping import (
    Section,
    derive_sections_from_chunks,
    group_sections,
)
from dikw_core.schemas import ChunkRecord


def _record(*, seq: int, start: int, end: int, text: str) -> ChunkRecord:
    return ChunkRecord(doc_id="test", seq=seq, start=start, end=end, text=text)


# --- derive_sections_from_chunks ---------------------------------------


def test_derive_returns_empty_when_no_chunks() -> None:
    assert derive_sections_from_chunks("body", []) == []


def test_derive_single_chunk_covers_chunk_range() -> None:
    body = "# Heading\n\nParagraph one.\n"
    chunks = [_record(seq=0, start=0, end=len(body), text=body)]
    sections = derive_sections_from_chunks(body, chunks, cjk_tokenizer="none")
    assert len(sections) == 1
    s = sections[0]
    assert s.start == 0
    assert s.end == len(body)
    assert s.text == body
    assert s.headings == ("Heading",)
    assert s.starts_with_h1 is True


def test_derive_uses_next_chunk_start_as_right_edge() -> None:
    """Even with synthetic overlap, sections must use ``chunks[i+1].start``
    rather than ``chunks[i].end`` as the right edge — that's the whole
    point of this module."""
    body = "AAAA BBBB CCCC DDDD"
    # Construct chunks that overlap on purpose: chunk[0] ends at 14
    # ("...DDDD"), chunk[1] starts at 10 ("CCCC DDDD"). A naive concat
    # would feed "DDDD" twice.
    c0 = _record(seq=0, start=0, end=14, text=body[0:14])
    c1 = _record(seq=1, start=10, end=19, text=body[10:19])
    sections = derive_sections_from_chunks(body, [c0, c1], cjk_tokenizer="none")
    assert len(sections) == 2
    # First section's end equals the SECOND chunk's start, not the first
    # chunk's end — that's the overlap-elimination invariant.
    assert sections[0].end == 10
    assert sections[0].text == body[0:10]
    # Second section spans from chunk[1].start to chunk[1].end.
    assert sections[1].start == 10
    assert sections[1].end == 19
    assert sections[1].text == body[10:19]


def test_derive_against_real_chunker_has_no_overlap() -> None:
    # Many small paragraphs + a tiny budget force the chunker into
    # the carry-overlap arm (each chunk emits, then the trailing
    # paragraph re-appears at the start of the next chunk).
    paragraphs = [f"alpha beta gamma delta epsilon-{i}" for i in range(40)]
    body = "\n\n".join(paragraphs) + "\n"
    real_chunks = chunk_markdown(body, max_tokens=12, cjk_tokenizer="none")
    assert len(real_chunks) >= 2  # sanity: chunker produced multiple chunks
    # Confirm at least one pair overlaps (otherwise the test would be
    # tautological).
    overlaps = any(
        real_chunks[i].end > real_chunks[i + 1].start
        for i in range(len(real_chunks) - 1)
    )
    assert overlaps, "fixture should produce overlapping chunks for this test to mean anything"

    records = [
        ChunkRecord(doc_id="t", seq=c.seq, start=c.start, end=c.end, text=c.text)
        for c in real_chunks
    ]
    sections = derive_sections_from_chunks(body, records, cjk_tokenizer="none")

    # Adjacent sections never overlap and meet exactly.
    for i in range(len(sections) - 1):
        assert sections[i].end == sections[i + 1].start

    # Union of section spans covers exactly chunks[0].start..chunks[-1].end
    assert sections[0].start == real_chunks[0].start
    assert sections[-1].end == real_chunks[-1].end


# --- group_sections ----------------------------------------------------


def _section(start: int, text: str, tokens: int) -> Section:
    return Section(
        start=start,
        end=start + len(text),
        text=text,
        headings=(),
        tokens=tokens,
    )


def _h1_section(start: int, text: str, tokens: int) -> Section:
    return Section(
        start=start,
        end=start + len(text),
        text=text,
        headings=("Some H1",),
        tokens=tokens,
    )


def test_group_packs_within_budget_into_single_group() -> None:
    sections = [_section(0, "a", 5), _section(1, "b", 5), _section(2, "c", 5)]
    groups = group_sections(sections, target_tokens=20)
    assert len(groups) == 1
    assert groups[0].token_count == 15
    assert groups[0].text == "abc"
    assert groups[0].section_starts == (0, 1, 2)


def test_group_breaks_when_exceeding_budget() -> None:
    sections = [_section(0, "a", 8), _section(1, "b", 8), _section(2, "c", 8)]
    groups = group_sections(sections, target_tokens=10)
    # 8 + 8 = 16 > 10 → break before second; 8 + 8 = 16 > 10 → break before third
    assert len(groups) == 3
    assert [g.token_count for g in groups] == [8, 8, 8]


def test_group_h1_forces_break_even_under_budget() -> None:
    sections = [
        _section(0, "intro paragraph", 5),
        _h1_section(15, "# Chapter 2\n\nbody", 8),
        _section(32, "more body", 4),
    ]
    groups = group_sections(sections, target_tokens=100)
    # Even though all three fit under 100, the H1 forces a break.
    assert len(groups) == 2
    assert groups[0].text == "intro paragraph"
    assert groups[1].text == "# Chapter 2\n\nbodymore body"


def test_group_oversize_section_emits_alone() -> None:
    # A section larger than the budget should just be its own group —
    # better than splitting mid-section and stranding headings.
    sections = [_section(0, "small", 5), _section(5, "huge" * 20, 50)]
    groups = group_sections(sections, target_tokens=20)
    assert len(groups) == 2
    assert groups[0].token_count == 5
    assert groups[1].token_count == 50


def test_group_collects_unique_headings_in_order() -> None:
    s1 = Section(
        start=0,
        end=10,
        text="...",
        headings=("Intro", "Background"),
        tokens=5,
    )
    s2 = Section(
        start=10,
        end=20,
        text="...",
        headings=("Background", "Method"),  # "Background" repeats
        tokens=5,
    )
    groups = group_sections([s1, s2], target_tokens=100)
    assert len(groups) == 1
    assert groups[0].headings == ("Intro", "Background", "Method")


def test_group_empty_input() -> None:
    assert group_sections([], target_tokens=10) == []
