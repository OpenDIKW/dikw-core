from __future__ import annotations

from dikw_core.domains.info.chunk import chunk_markdown


def test_empty_body_returns_no_chunks() -> None:
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  \n") == []


def test_short_body_is_single_chunk() -> None:
    body = "# Title\n\nHello world paragraph."
    chunks = chunk_markdown(body, max_tokens=900)
    assert len(chunks) == 1
    assert chunks[0].seq == 0
    assert body[chunks[0].start : chunks[0].end] == chunks[0].text


def test_heading_forces_chunk_boundary() -> None:
    para_a = "alpha " * 200
    para_b = "beta " * 200
    body = f"# A\n\n{para_a}\n\n## B\n\n{para_b}"
    chunks = chunk_markdown(body, max_tokens=300, overlap_ratio=0.1)
    assert len(chunks) >= 2
    # at least one chunk should begin with an ATX heading
    assert any(chunks[i].text.lstrip().startswith("#") for i in range(len(chunks)))


def test_overflow_splits_with_overlap() -> None:
    # 10 paragraphs of ~40 tokens each → should overflow a 100-token budget.
    paragraphs = ["lorem " * 40 for _ in range(10)]
    body = "\n\n".join(p.strip() for p in paragraphs)
    chunks = chunk_markdown(body, max_tokens=100, overlap_ratio=0.2)
    assert len(chunks) >= 3
    # consecutive chunks overlap in character range (end_N > start_{N+1})
    overlaps = [chunks[i].end > chunks[i + 1].start for i in range(len(chunks) - 1)]
    assert any(overlaps)


def test_char_offsets_round_trip() -> None:
    body = "# Intro\n\nAlpha beta.\n\n## Deep\n\nGamma delta epsilon.\n"
    for c in chunk_markdown(body, max_tokens=5):
        assert body[c.start : c.end] == c.text


def test_paragraph_larger_than_budget_does_not_loop() -> None:
    huge = "word " * 500
    body = f"{huge}\n\nsmall paragraph"
    chunks = chunk_markdown(body, max_tokens=100, overlap_ratio=0.1)
    # Just terminating is the contract; we also expect both paragraphs represented.
    joined = " ".join(c.text for c in chunks)
    assert "small paragraph" in joined


def test_cjk_long_doc_splits_on_token_budget() -> None:
    """Chinese docs at default ``max_tokens=900`` must split. Whitespace
    counting reports ~1 token per CJK paragraph, which would let the
    whole doc past the embedding context — the budget must trip on
    jieba-segmented counts instead.
    """
    para = "机器学习是一种实现人工智能的方法,深度学习是机器学习的一个分支." * 80
    body = "\n\n".join([para] * 3)
    chunks = chunk_markdown(body, max_tokens=900)
    assert len(chunks) >= 2


def test_cjk_tokenizer_none_preserves_legacy_chunking() -> None:
    """Backward-compat: an existing wiki configured with
    ``cjk_tokenizer="none"`` must keep its original whitespace-split
    chunking even on CJK content. Otherwise upgrading the engine would
    silently re-chunk Chinese corpora and re-bill embeddings.
    """
    para = "机器学习是一种实现人工智能的方法,深度学习是机器学习的一个分支." * 80
    body = "\n\n".join([para] * 3)
    chunks = chunk_markdown(body, max_tokens=900, cjk_tokenizer="none")
    # whitespace-split sees ~3 tokens (one per paragraph); never trips 900
    assert len(chunks) == 1


def test_ascii_token_count_equals_whitespace_split() -> None:
    """Invariant: an all-ASCII body's chunk-budget count equals
    ``len(body.split())``. Locks budget tuning against drift if anyone
    swaps the underlying tokenizer.
    """
    from dikw_core.domains.info.tokenize import count_tokens

    for body in [
        "lorem " * 40,
        "alpha " * 200,
        "Hello world. This is plain English.",
        "retrieval.rrf_k tuning_param weight_a",
    ]:
        assert count_tokens(body) == len(body.split())


def test_heading_flush_does_not_emit_duplicate_sub_chunk() -> None:
    """When step 1 flushes before a heading and carries tail overlap that still
    overflows ``max_tokens`` with the heading, step 2 must drop the carry
    instead of re-emitting the just-flushed span as a duplicate chunk."""
    # Exact counterexample at overlap_ratio=0.0:
    chunks_zero = chunk_markdown("a\n\n## b c d", max_tokens=4, overlap_ratio=0.0)
    assert [c.text for c in chunks_zero] == ["a", "## b c d"]

    # Multi-paragraph strict sub-span suppression (carry "c" was already inside
    # the just-emitted "a b\n\nc" chunk and must not be re-emitted standalone):
    chunks_subspan = chunk_markdown("a b\n\nc\n\n## d e f g", max_tokens=4, overlap_ratio=0.0)
    assert [c.text for c in chunks_subspan] == ["a b\n\nc", "## d e f g"]

    # Same hazard at the default overlap_ratio=0.15 when carry + heading > max_tokens:
    body = "alpha beta\n\n## " + " ".join(f"w{i}" for i in range(9))
    chunks_default = chunk_markdown(body, max_tokens=10, overlap_ratio=0.15)
    assert len(chunks_default) == 2
    assert chunks_default[0].text == "alpha beta"
    assert chunks_default[1].text.startswith("## ")
    assert chunks_default[1].end > chunks_default[0].end
