from __future__ import annotations

import asyncio

import pytest

from dikw_core import telemetry
from dikw_core.domains.info.search import (
    HybridSearcher,
    MultimodalSearch,
    _sanitize_fts,
    _traced_leg,
    apply_source_diversity_penalty,
    comb_mnz_fusion,
    comb_sum_fusion,
    reciprocal_rank_fusion,
)
from dikw_core.providers.base import ProviderError, TransientProviderError
from dikw_core.storage.base import NotSupported
from dikw_core.storage.sqlite import SQLiteStorage

from .fakes import (
    FakeEmbeddings,
    FakeMultimodalEmbedding,
    FakeReranker,
    register_text_version,
)


def test_rrf_favors_consensus() -> None:
    # Doc "a" is #1 in both lists → top score
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
    assert max(fused, key=lambda k: fused[k]) == "a"
    # "b" and "c" tied on consensus weight; ordering should be stable-equal-ish
    assert fused["a"] > fused["b"]
    assert fused["a"] > fused["c"]


def test_rrf_handles_disjoint_lists() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "d"]])
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_weights_default_preserves_legacy() -> None:
    """weights=None must reproduce the pre-weighting numbers bit-for-bit.

    Regression guard: a silent change to the default would shift every
    existing hybrid ranking. Compute both paths with the same k and assert
    they match at full float precision.
    """
    lists = [["a", "b", "c"], ["a", "c", "b"]]
    legacy = reciprocal_rank_fusion(lists, k=60)
    explicit = reciprocal_rank_fusion(lists, k=60, weights=[1.0, 1.0])
    assert legacy == explicit


def test_rrf_weights_shift_ranking_toward_weighted_leg() -> None:
    """When one leg is weighted higher, its solo-discovered docs beat the
    other leg's solo-discovered docs — what the SciFact fix needs.
    """
    # "bm25_only" appears only in list[0]; "vec_only" only in list[1].
    # Both at rank 0, so equal weights → tie. Asymmetric weights → winner.
    lists = [["bm25_only"], ["vec_only"]]

    equal = reciprocal_rank_fusion(lists, k=60, weights=[1.0, 1.0])
    assert equal["bm25_only"] == equal["vec_only"]

    vec_heavy = reciprocal_rank_fusion(lists, k=60, weights=[0.5, 1.0])
    assert vec_heavy["vec_only"] > vec_heavy["bm25_only"]

    bm25_heavy = reciprocal_rank_fusion(lists, k=60, weights=[1.0, 0.5])
    assert bm25_heavy["bm25_only"] > bm25_heavy["vec_only"]


def test_rrf_weights_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_rrf_k_controls_rank_decay() -> None:
    """Smaller k = steeper decay; rank-1 doc wins by a larger margin."""
    lists = [["first", "second"]]
    tight = reciprocal_rank_fusion(lists, k=10)
    loose = reciprocal_rank_fusion(lists, k=100)
    # first/second ratio grows as k shrinks
    assert tight["first"] / tight["second"] > loose["first"] / loose["second"]


def test_rrf_accepts_int_keys() -> None:
    """Generic key type — chunk_id (int) flows through fusion the same as
    doc_id (str). Required for chunk-level fusion (Phase 1 plan).
    """
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 3, 4]])
    assert max(fused, key=lambda k: fused[k]) == 2
    assert isinstance(next(iter(fused.keys())), int)
    # Same numeric semantics as the str-keyed path.
    str_fused = reciprocal_rank_fusion([["1", "2", "3"], ["2", "3", "4"]])
    assert {str(k): v for k, v in fused.items()} == str_fused


# ---- comb_sum_fusion / comb_mnz_fusion --------------------------------------


def test_comb_sum_normalises_per_leg() -> None:
    """Per-leg min-max maps each leg's range to [0, 1] independently — a
    raw 100 in a small-magnitude leg shouldn't dwarf a raw 2 in a
    different-scale leg post-fusion.
    """
    leg_big = [("a", 100.0), ("b", 50.0), ("c", 0.0)]
    leg_small = [("d", 2.0), ("a", 1.0), ("c", 0.0)]
    fused = comb_sum_fusion([leg_big, leg_small], weights=[1.0, 1.0])
    # Per-leg [1.0, 0.5, 0.0] in each. "a" sums 1.0 + 0.5 = 1.5; "d" tops
    # leg_small alone so is 1.0; "c" is 0.0 in both.
    assert fused["a"] == pytest.approx(1.5)
    assert fused["d"] == pytest.approx(1.0)
    assert fused["c"] == pytest.approx(0.0)


def test_comb_sum_weights_shift_ranking() -> None:
    """Solo-discovered keys flip with asymmetric weights — same property
    that test_rrf_weights_shift_ranking_toward_weighted_leg pins for RRF.
    """
    lists = [[("bm25_only", 1.0)], [("vec_only", 1.0)]]

    equal = comb_sum_fusion(lists, weights=[1.0, 1.0])
    assert equal["bm25_only"] == equal["vec_only"]

    vec_heavy = comb_sum_fusion(lists, weights=[0.5, 1.0])
    assert vec_heavy["vec_only"] > vec_heavy["bm25_only"]

    bm25_heavy = comb_sum_fusion(lists, weights=[1.0, 0.5])
    assert bm25_heavy["bm25_only"] > bm25_heavy["vec_only"]


def test_comb_sum_handles_empty_leg() -> None:
    """An empty leg contributes nothing instead of crashing. Real-world
    case: vec_search may return [] when no embedder is wired.
    """
    fused = comb_sum_fusion([[("a", 1.0), ("b", 0.0)], []], weights=[1.0, 1.0])
    assert fused == {"a": pytest.approx(1.0), "b": pytest.approx(0.0)}


def test_comb_sum_single_score_leg_collapses_to_one() -> None:
    """Degenerate leg where every entry has identical raw score (max ==
    min) — must produce all 1.0 instead of dividing by zero.
    """
    leg = [("a", 5.0), ("b", 5.0)]
    fused = comb_sum_fusion([leg], weights=[1.0])
    assert fused == {"a": pytest.approx(1.0), "b": pytest.approx(1.0)}


def test_comb_sum_weights_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        comb_sum_fusion([[("a", 1.0)], [("b", 1.0)]], weights=[1.0])


def test_comb_mnz_multiplies_by_leg_count() -> None:
    """A key found by 2 legs scores 2 * CombSUM; a key found by 1 leg
    scores 1 * CombSUM (i.e., plain CombSUM).
    """
    leg_a = [("shared", 1.0), ("solo_a", 0.0)]
    leg_b = [("shared", 1.0), ("solo_b", 0.0)]
    sum_ = comb_sum_fusion([leg_a, leg_b], weights=[1.0, 1.0])
    mnz = comb_mnz_fusion([leg_a, leg_b], weights=[1.0, 1.0])
    assert mnz["shared"] == pytest.approx(2 * sum_["shared"])
    assert mnz["solo_a"] == pytest.approx(1 * sum_["solo_a"])
    assert mnz["solo_b"] == pytest.approx(1 * sum_["solo_b"])


def test_comb_mnz_single_leg_equals_comb_sum() -> None:
    """One-leg case: leg_count == 1 for every key, so MNZ collapses to
    CombSUM exactly.
    """
    leg = [("a", 3.0), ("b", 1.0), ("c", 0.0)]
    sum_ = comb_sum_fusion([leg], weights=[1.0])
    mnz = comb_mnz_fusion([leg], weights=[1.0])
    assert mnz == sum_


def test_comb_mnz_zero_weight_leg_excluded_from_count() -> None:
    """Ablation guard: a leg with ``weights[i] == 0`` contributes zero
    to ``CombSUM``; CombMNZ must not bump the consensus multiplier for
    that leg either, or "turn off BM25" would still double-count
    chunks BM25 retrieved.
    """
    leg_disabled = [("shared", 1.0)]
    leg_active = [("shared", 1.0)]
    mnz = comb_mnz_fusion([leg_disabled, leg_active], weights=[0.0, 1.0])
    sum_ = comb_sum_fusion([leg_disabled, leg_active], weights=[0.0, 1.0])
    # Both legs hit ``shared`` but BM25 leg is disabled → MNZ must
    # collapse to CombSUM (multiplier == 1, not 2).
    assert mnz == sum_


# ---- apply_source_diversity_penalty (1.3.1) --------------------------------


def test_diversity_penalty_alpha_zero_is_identity() -> None:
    """alpha=0 returns the input dict unchanged — no-op opt-out."""
    fused = {1: 1.0, 2: 0.9, 3: 0.8}
    doc_by_chunk = {1: "A", 2: "A", 3: "B"}
    out = apply_source_diversity_penalty(fused, doc_by_chunk, alpha=0.0)
    assert out == fused


def test_diversity_penalty_factor_semantics() -> None:
    """1st chunk per doc unpenalized; N-th chunk scaled by 1/(1+alpha*(N-1)).

    Pinning numerics so future contributors can't silently retune the
    factor. Inputs sorted by score desc; c1/c2 share doc A, c3 is doc B.
    """
    fused = {1: 1.0, 2: 0.9, 3: 0.8}
    doc_by_chunk = {1: "A", 2: "A", 3: "B"}
    out = apply_source_diversity_penalty(fused, doc_by_chunk, alpha=0.3)
    assert out[1] == pytest.approx(1.0)            # 1st in doc A
    assert out[2] == pytest.approx(0.9 / 1.3)      # 2nd in doc A
    assert out[3] == pytest.approx(0.8)            # 1st in doc B


def test_diversity_penalty_demotes_third_same_doc_chunk() -> None:
    """3rd chunk from one doc gets factor 1/(1+0.3*2) = 1/1.6 = 0.625."""
    fused = {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.5}
    doc_by_chunk = {1: "A", 2: "A", 3: "A", 4: "B"}
    out = apply_source_diversity_penalty(fused, doc_by_chunk, alpha=0.3)
    assert out[3] == pytest.approx(0.8 / 1.6)
    # And the 4th chunk (1st in doc B) remains unpenalized.
    assert out[4] == pytest.approx(0.5)


def test_diversity_penalty_reorders_when_demotion_overtakes_neighbor() -> None:
    """Penalty pushes 2nd same-doc chunk below a fresh-doc chunk that was
    behind it pre-penalty. Cross-doc chunk floats up — exactly the source
    diversification the knob exists to deliver.
    """
    fused = {1: 1.0, 2: 0.9, 3: 0.85}
    doc_by_chunk = {1: "A", 2: "A", 3: "B"}
    out = apply_source_diversity_penalty(fused, doc_by_chunk, alpha=0.3)
    after = sorted(out.items(), key=lambda kv: kv[1], reverse=True)
    assert [k for k, _ in after] == [1, 3, 2]
    assert out[2] == pytest.approx(0.9 / 1.3)


def test_diversity_penalty_unmapped_chunk_unpenalized() -> None:
    """Defensive: chunks missing from doc_by_chunk are passed through
    unmodified — never raise, never penalize what we can't identify.
    """
    fused = {1: 1.0, 2: 0.5}
    doc_by_chunk = {1: "A"}  # 2 is missing
    out = apply_source_diversity_penalty(fused, doc_by_chunk, alpha=0.3)
    assert out[2] == 0.5


# ---- _sanitize_fts ----------------------------------------------------------


def test_sanitize_fts_or_joins_word_tokens() -> None:
    # Natural-language query → bag-of-words OR. The Phase 1 phrase-quote
    # implementation returned `"foo bar baz"` and matched ~nothing in real
    # corpora; this is the regression guard.
    assert _sanitize_fts("foo bar baz") == '"foo" OR "bar" OR "baz"'


def test_sanitize_fts_strips_punctuation() -> None:
    # Hyphens, question marks, parentheses → whitespace; tokens preserved.
    assert _sanitize_fts("DIKW-core (what is it?)") == (
        '"DIKW" OR "core" OR "what" OR "is" OR "it"'
    )


def test_sanitize_fts_drops_fts5_reserved_tokens() -> None:
    # AND / OR / NOT / NEAR are FTS5 operators — silently dropping them
    # avoids syntax errors when a user query happens to contain them.
    assert _sanitize_fts("cats AND dogs OR birds") == '"cats" OR "dogs" OR "birds"'
    assert _sanitize_fts("just NOT a phrase") == '"just" OR "a" OR "phrase"'


def test_sanitize_fts_preserves_underscores() -> None:
    # Identifier-like tokens (e.g., "expect_any") survive intact because
    # \w matches underscore.
    assert _sanitize_fts("test expect_any field") == (
        '"test" OR "expect_any" OR "field"'
    )


def test_sanitize_fts_preserves_cjk() -> None:
    # CJK Unified Ideographs survive the punctuation strip. Whitespace-
    # separated CJK tokens are still tokenized at whitespace; running text
    # without spaces becomes a single token (a known FTS5 limitation, not
    # something this sanitizer can fix without a real CJK tokenizer).
    assert _sanitize_fts("机器 学习 入门") == '"机器" OR "学习" OR "入门"'


def test_sanitize_fts_with_jieba_segments_cjk_runs() -> None:
    """Under ``cjk_tokenizer='jieba'`` a whitespace-free Chinese phrase
    should produce multiple word-level tokens — exactly what the indexed
    FTS5 body will have stored, so MATCH actually fires.
    """
    out = _sanitize_fts("机器学习入门", cjk_tokenizer="jieba")
    # Result shape: `"tok1" OR "tok2" OR ...`; at least one multi-char
    # Chinese word present. Pre-feature path would have produced the
    # degenerate `"机器学习入门"` single-token expression.
    assert " OR " in out, f"no tokenization happened: {out!r}"
    assert '"机器学习"' in out or '"机器"' in out
    assert '"入门"' in out


def test_sanitize_fts_with_jieba_keeps_ascii_intact() -> None:
    """Dev-doc query like ``retrieval.rrf_k 参数`` must not shred the
    ASCII identifier. Mixed-language users are the biggest at-risk group
    for a naive ``jieba.cut_for_search(full_text)``.
    """
    out = _sanitize_fts("retrieval.rrf_k 参数", cjk_tokenizer="jieba")
    assert '"rrf_k"' in out
    # Either of these is fine depending on jieba's dictionary
    assert '"retrieval"' in out or '"retrieval_rrf_k"' in out
    assert '"参数"' in out


def test_sanitize_fts_empty_or_punctuation_only_returns_empty() -> None:
    assert _sanitize_fts("") == ""
    assert _sanitize_fts("   ") == ""
    assert _sanitize_fts("???") == ""


@pytest.mark.asyncio
async def test_hybrid_search_returns_hits(tmp_path) -> None:
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()

    # Load the fixture corpus into the storage engine.
    import time
    from pathlib import Path

    from dikw_core.domains.data.backends.markdown import parse_file
    from dikw_core.domains.info.chunk import chunk_markdown
    from dikw_core.domains.info.embed import ChunkToEmbed, embed_chunks
    from dikw_core.schemas import ChunkRecord, DocumentRecord, Layer

    fixtures = Path(__file__).parent / "fixtures" / "notes"
    to_embed: list[ChunkToEmbed] = []
    for md_path in sorted(fixtures.glob("*.md")):
        parsed = parse_file(md_path, rel_path=f"sources/notes/{md_path.name}")
        doc_id = f"source:sources/notes/{md_path.name}"
        await storage.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                path=parsed.path,
                title=parsed.title,
                hash=parsed.hash,
                mtime=parsed.mtime or time.time(),
                layer=Layer.SOURCE,
                active=True,
            )
        )
        chunks = chunk_markdown(parsed.body)
        records = [
            ChunkRecord(doc_id=doc_id, seq=c.seq, start=c.start, end=c.end, text=c.text)
            for c in chunks
        ]
        ids = await storage.replace_chunks(doc_id, records)
        to_embed.extend(ChunkToEmbed(chunk_id=cid, text=r.text) for cid, r in zip(ids, records, strict=True))

    embedder = FakeEmbeddings()
    version_id = await register_text_version(storage)
    async for batch in embed_chunks(
        embedder, to_embed, model="fake", version_id=version_id
    ):
        await storage.upsert_embeddings(batch.rows)

    searcher = HybridSearcher(
        storage, embedder, embedding_model="fake", text_version_id=version_id
    )
    hits = await searcher.search("reciprocal rank fusion", limit=3)
    assert hits, "no hits returned"
    assert any("retrieval.md" in (h.path or "") for h in hits)

    # the question about DIKW pyramid should surface dikw.md
    hits = await searcher.search("DIKW pyramid", limit=3)
    assert any("dikw.md" in (h.path or "") for h in hits)

    await storage.close()


# ---- retrieval mode (bm25 / vector / hybrid) --------------------------------


async def _populate_fixture_corpus(storage):
    """Helper: load tests/fixtures/notes/ into ``storage`` (FakeEmbeddings).

    Returns ``(embedder, text_version_id)`` so callers can pass the
    version_id to ``HybridSearcher``.
    """
    import time
    from pathlib import Path

    from dikw_core.domains.data.backends.markdown import parse_file
    from dikw_core.domains.info.chunk import chunk_markdown
    from dikw_core.domains.info.embed import ChunkToEmbed, embed_chunks
    from dikw_core.schemas import ChunkRecord, DocumentRecord, Layer

    fixtures = Path(__file__).parent / "fixtures" / "notes"
    to_embed: list[ChunkToEmbed] = []
    for md_path in sorted(fixtures.glob("*.md")):
        parsed = parse_file(md_path, rel_path=f"sources/notes/{md_path.name}")
        doc_id = f"source:sources/notes/{md_path.name}"
        await storage.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                path=parsed.path,
                title=parsed.title,
                hash=parsed.hash,
                mtime=parsed.mtime or time.time(),
                layer=Layer.SOURCE,
                active=True,
            )
        )
        chunks = chunk_markdown(parsed.body)
        records = [
            ChunkRecord(doc_id=doc_id, seq=c.seq, start=c.start, end=c.end, text=c.text)
            for c in chunks
        ]
        ids = await storage.replace_chunks(doc_id, records)
        to_embed.extend(
            ChunkToEmbed(chunk_id=cid, text=r.text)
            for cid, r in zip(ids, records, strict=True)
        )

    embedder = FakeEmbeddings()
    version_id = await register_text_version(storage)
    async for batch in embed_chunks(
        embedder, to_embed, model="fake", version_id=version_id
    ):
        await storage.upsert_embeddings(batch.rows)
    return embedder, version_id


@pytest.mark.asyncio
async def test_mode_bm25_skips_vector_leg(tmp_path) -> None:
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_fixture_corpus(storage)

    # Spy on vec_search to verify it never fires in bm25 mode.
    real_vec = storage.vec_search
    calls = {"vec": 0}

    async def spy_vec(*args, **kwargs):
        calls["vec"] += 1
        return await real_vec(*args, **kwargs)

    storage.vec_search = spy_vec  # type: ignore[method-assign]

    searcher = HybridSearcher(
        storage, embedder, embedding_model="fake", text_version_id=version_id
    )
    hits = await searcher.search("DIKW pyramid", limit=3, mode="bm25")
    assert hits, "bm25 mode returned no hits"
    assert calls["vec"] == 0, "vec_search must not run in bm25 mode"

    await storage.close()


@pytest.mark.asyncio
async def test_mode_vector_skips_fts_leg(tmp_path) -> None:
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_fixture_corpus(storage)

    real_fts = storage.fts_search
    calls = {"fts": 0}

    async def spy_fts(*args, **kwargs):
        calls["fts"] += 1
        return await real_fts(*args, **kwargs)

    storage.fts_search = spy_fts  # type: ignore[method-assign]

    searcher = HybridSearcher(
        storage, embedder, embedding_model="fake", text_version_id=version_id
    )
    hits = await searcher.search("DIKW pyramid", limit=3, mode="vector")
    assert hits, "vector mode returned no hits"
    assert calls["fts"] == 0, "fts_search must not run in vector mode"

    await storage.close()


@pytest.mark.asyncio
async def test_mode_hybrid_runs_both_legs(tmp_path) -> None:
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_fixture_corpus(storage)

    real_fts = storage.fts_search
    real_vec = storage.vec_search
    calls = {"fts": 0, "vec": 0}

    async def spy_fts(*args, **kwargs):
        calls["fts"] += 1
        return await real_fts(*args, **kwargs)

    async def spy_vec(*args, **kwargs):
        calls["vec"] += 1
        return await real_vec(*args, **kwargs)

    storage.fts_search = spy_fts  # type: ignore[method-assign]
    storage.vec_search = spy_vec  # type: ignore[method-assign]

    searcher = HybridSearcher(
        storage, embedder, embedding_model="fake", text_version_id=version_id
    )
    hits = await searcher.search("DIKW pyramid", limit=3, mode="hybrid")
    assert hits
    assert calls["fts"] == 1 and calls["vec"] == 1

    await storage.close()


@pytest.mark.asyncio
async def test_mode_bm25_works_without_embedder(tmp_path) -> None:
    """bm25 mode should not require an embedder at all."""
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    await _populate_fixture_corpus(storage)

    searcher = HybridSearcher(storage, embedder=None, embedding_model=None)
    hits = await searcher.search("DIKW pyramid", limit=3, mode="bm25")
    assert hits

    await storage.close()


@pytest.mark.asyncio
async def test_mode_vector_requires_embedder(tmp_path) -> None:
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    await _populate_fixture_corpus(storage)

    searcher = HybridSearcher(storage, embedder=None, embedding_model=None)
    with pytest.raises(ValueError, match="vector"):
        await searcher.search("anything", limit=3, mode="vector")

    await storage.close()


# ---- chunk-level fusion (Phase 1) ------------------------------------------


async def _populate_multi_chunk_corpus(storage: SQLiteStorage) -> FakeEmbeddings:
    """Build a 2-doc corpus with multiple distinct chunks per doc.

    Doc A has 3 chunks, all matching the test query "alpha foo".
    Doc B has 2 chunks, only one matching "alpha foo".

    Designed so chunk-level fusion can demonstrate "multiple chunks from
    same doc" while staying small enough for FakeEmbeddings to rank
    deterministically by bag-of-words.
    """
    import time

    from dikw_core.domains.info.embed import ChunkToEmbed, embed_chunks
    from dikw_core.schemas import ChunkRecord, DocumentRecord, Layer

    doc_specs: list[tuple[str, list[str]]] = [
        (
            "alpha",
            [
                "alpha foo bar topic one. alpha foo bar.",
                "alpha foo bar topic two with details.",
                "alpha foo bar topic three closing.",
            ],
        ),
        (
            "beta",
            [
                "alpha foo unrelated baseline content.",
                "completely separate beta material with no overlap.",
            ],
        ),
    ]

    to_embed: list[ChunkToEmbed] = []
    for stem, chunk_texts in doc_specs:
        doc_id = f"source:sources/{stem}.md"
        await storage.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                path=f"sources/{stem}.md",
                title=stem,
                hash=f"h-{stem}",
                mtime=time.time(),
                layer=Layer.SOURCE,
                active=True,
            )
        )
        records: list[ChunkRecord] = []
        offset = 0
        for i, text in enumerate(chunk_texts):
            records.append(
                ChunkRecord(
                    doc_id=doc_id, seq=i, start=offset, end=offset + len(text), text=text
                )
            )
            offset += len(text) + 2  # account for separator
        ids = await storage.replace_chunks(doc_id, records)
        to_embed.extend(
            ChunkToEmbed(chunk_id=cid, text=r.text)
            for cid, r in zip(ids, records, strict=True)
        )

    embedder = FakeEmbeddings()
    version_id = await register_text_version(storage)
    async for batch in embed_chunks(
        embedder, to_embed, model="fake", version_id=version_id
    ):
        await storage.upsert_embeddings(batch.rows)
    return embedder, version_id


@pytest.mark.asyncio
async def test_chunk_level_fusion_returns_multiple_chunks_per_doc(tmp_path) -> None:
    """alpha=0 (no diversity penalty) → top-K contains multiple chunks
    from the same doc when every leg ranks them highly.
    """
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    searcher = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        same_doc_penalty_alpha=0.0,
    )
    hits = await searcher.search("alpha foo bar", limit=5)
    await storage.close()

    paths = [h.path for h in hits]
    alpha_count = sum(1 for p in paths if p == "sources/alpha.md")
    # All three alpha chunks match strongly; with no penalty they should
    # all surface.
    assert alpha_count >= 2, f"expected >=2 alpha chunks, got paths={paths}"
    # Each Hit must carry a chunk_id (non-optional after Phase 1).
    assert all(h.chunk_id is not None for h in hits)
    # And a seq populated from the chunk record.
    assert all(h.seq is not None for h in hits)


@pytest.mark.asyncio
async def test_chunk_level_fusion_distinct_chunks_have_distinct_ids(tmp_path) -> None:
    """No two hits share a chunk_id — chunk-level fusion deduplicates
    keys across the legs by construction.
    """
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    searcher = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        same_doc_penalty_alpha=0.0,
    )
    hits = await searcher.search("alpha foo bar", limit=5)
    await storage.close()

    chunk_ids = [h.chunk_id for h in hits]
    assert len(chunk_ids) == len(set(chunk_ids))


@pytest.mark.asyncio
async def test_same_doc_penalty_zero_vs_default_changes_ranking(tmp_path) -> None:
    """alpha=0 yields more same-doc concentration than alpha=0.3 on a
    corpus with one obviously-dominant doc.
    """
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    pure = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        same_doc_penalty_alpha=0.0,
    )
    diversified = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        same_doc_penalty_alpha=0.3,
    )

    pure_hits = await pure.search("alpha foo bar", limit=5)
    div_hits = await diversified.search("alpha foo bar", limit=5)
    await storage.close()

    pure_alpha = sum(1 for h in pure_hits if h.path == "sources/alpha.md")
    div_alpha = sum(1 for h in div_hits if h.path == "sources/alpha.md")
    # Diversification should not increase same-doc concentration; a
    # measurable demotion lands at <= pure on a corpus this small.
    assert div_alpha <= pure_alpha, (
        f"diversified alpha-count {div_alpha} should be <= pure {pure_alpha}; "
        f"pure_paths={[h.path for h in pure_hits]}, "
        f"div_paths={[h.path for h in div_hits]}"
    )


@pytest.mark.asyncio
async def test_hybrid_searcher_from_config_threads_alpha(tmp_path) -> None:
    """RetrievalConfig.same_doc_penalty_alpha → HybridSearcher.from_config
    → search() — the Protocol triad wiring (Phase 1.3.0).
    """
    from dikw_core.config import RetrievalConfig

    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    cfg = RetrievalConfig(same_doc_penalty_alpha=0.0)
    searcher = HybridSearcher.from_config(
        storage, embedder, cfg, embedding_model="fake", text_version_id=version_id
    )
    assert searcher._same_doc_penalty_alpha == 0.0
    hits = await searcher.search("alpha foo bar", limit=5)
    await storage.close()

    # Sanity: with alpha=0 we get >=2 same-doc chunks (matches the pure
    # path of the parametric test above).
    alpha_paths = [h.path for h in hits if h.path == "sources/alpha.md"]
    assert len(alpha_paths) >= 2


@pytest.mark.asyncio
async def test_hybrid_searcher_combsum_dispatch(tmp_path) -> None:
    """RetrievalConfig.fusion='combsum' → HybridSearcher.from_config →
    search() exercises the score-bearing dispatch path. Asserts the attr
    plumbs and the run produces hits without crashing on either fuser.
    """
    from dikw_core.config import RetrievalConfig

    for mode in ("combsum", "combmnz"):
        storage = SQLiteStorage(tmp_path / f"idx-{mode}.sqlite")
        await storage.connect()
        await storage.migrate()
        embedder, version_id = await _populate_fixture_corpus(storage)

        cfg = RetrievalConfig(fusion=mode)  # type: ignore[arg-type]
        searcher = HybridSearcher.from_config(
            storage, embedder, cfg, embedding_model="fake", text_version_id=version_id
        )
        assert searcher._fusion == mode
        hits = await searcher.search("DIKW pyramid", limit=3)
        await storage.close()
        assert hits, f"fusion={mode} should still return hits on the corpus"


@pytest.mark.asyncio
async def test_hybrid_searcher_rejects_unknown_fusion_mode(tmp_path) -> None:
    """Direct ``HybridSearcher`` callers bypass pydantic's ``Literal``
    validation. The dispatcher must reject typos at search() time
    instead of silently running CombMNZ.
    """
    storage = SQLiteStorage(tmp_path / "idx-bogus.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_fixture_corpus(storage)

    searcher = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        fusion="combsmm",  # typo — not a member of FusionMode  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="unknown fusion mode"):
        await searcher.search("DIKW pyramid", limit=3)
    await storage.close()


# ---- rerank stage -----------------------------------------------------------


_RERANK_QUERY = "alpha foo bar"


async def _fused_baseline_ids(storage, embedder, version_id, *, limit: int) -> list[int]:
    """Hits a no-reranker searcher returns — the post-fusion order a
    reranker re-sorts. Captured so the rerank tests assert *against* the
    deterministic fused order rather than hard-coding chunk_ids."""
    baseline = HybridSearcher(
        storage, embedder, embedding_model="fake", text_version_id=version_id
    )
    hits = await baseline.search(_RERANK_QUERY, limit=limit)
    return [h.chunk_id for h in hits if h.chunk_id is not None]


@pytest.mark.asyncio
async def test_rerank_reorders_window_by_rerank_score(tmp_path) -> None:
    """A reranker re-sorts the fused window by its own scores, not RRF.

    ``FakeReranker`` (default) scores by input position ascending, so the
    LAST fused-window chunk gets the highest score → the reranked output is
    the fused order reversed, and each ``Hit.score`` carries the rerank
    score (not the fused score).
    """
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    fused = await _fused_baseline_ids(storage, embedder, version_id, limit=5)

    reranker = FakeReranker()
    searcher = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        reranker=reranker,
        rerank_model="fake-rerank",
        rerank_candidate_k=40,
    )
    hits = await searcher.search(_RERANK_QUERY, limit=5)
    await storage.close()

    assert reranker.call_count == 1
    reranked = [h.chunk_id for h in hits]
    # Window covered the whole fused pool (candidate_k >> pool), so reranked
    # order is the fused order reversed (FakeReranker scores last input top).
    assert reranked == fused[::-1]
    # Hit.score is the rerank relevance score, strictly descending.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]


@pytest.mark.asyncio
async def test_rerank_none_is_byte_identical_to_baseline(tmp_path) -> None:
    """``reranker=None`` runs the current path unchanged — same hits, order,
    and scores as a searcher built without rerank support at all."""
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    baseline = HybridSearcher(
        storage, embedder, embedding_model="fake", text_version_id=version_id
    )
    with_none = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        reranker=None,
        rerank_candidate_k=40,
    )
    base_hits = await baseline.search(_RERANK_QUERY, limit=5)
    none_hits = await with_none.search(_RERANK_QUERY, limit=5)
    await storage.close()

    assert [(h.chunk_id, h.score) for h in none_hits] == [
        (h.chunk_id, h.score) for h in base_hits
    ]


@pytest.mark.asyncio
async def test_rerank_disabled_does_not_call_reranker(tmp_path) -> None:
    """``rerank_enabled=False`` is a kill switch: even with a configured
    reranker the searcher never calls it and returns the fused order."""
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    fused = await _fused_baseline_ids(storage, embedder, version_id, limit=5)

    reranker = FakeReranker()
    searcher = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        reranker=reranker,
        rerank_model="fake-rerank",
        rerank_enabled=False,
    )
    hits = await searcher.search(_RERANK_QUERY, limit=5)
    await storage.close()

    assert reranker.call_count == 0
    assert [h.chunk_id for h in hits] == fused


@pytest.mark.asyncio
async def test_rerank_widens_candidate_window_beyond_limit(tmp_path) -> None:
    """The reranker scores a window larger than ``limit`` — that wider pool
    is the whole point (recover precision@k from recall@candidate_k)."""
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    reranker = FakeReranker()
    searcher = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        reranker=reranker,
        rerank_model="fake-rerank",
        rerank_candidate_k=40,
    )
    hits = await searcher.search(_RERANK_QUERY, limit=2)
    await storage.close()

    # Only 2 hits returned, but the reranker saw the full fused pool (5
    # chunks) — the window was widened from limit=2 up to candidate_k.
    assert len(hits) == 2
    assert len(reranker.last_documents) > 2
    assert reranker.last_query == _RERANK_QUERY
    assert reranker.last_model == "fake-rerank"


@pytest.mark.asyncio
async def test_rerank_candidate_k_clamped_to_at_least_limit(tmp_path) -> None:
    """A candidate window smaller than ``limit`` would starve the result —
    the window is clamped to at least ``limit`` so ``limit`` hits survive."""
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    reranker = FakeReranker()
    searcher = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        reranker=reranker,
        rerank_model="fake-rerank",
        rerank_candidate_k=1,
    )
    hits = await searcher.search(_RERANK_QUERY, limit=3)
    await storage.close()

    assert len(reranker.last_documents) >= 3
    assert len(hits) == 3


@pytest.mark.asyncio
async def test_rerank_introduces_no_chunks_outside_fused_pool(tmp_path) -> None:
    """Rerank only reorders the deterministically-retrieved pool — it never
    surfaces a chunk the fused legs didn't retrieve (recall is unchanged)."""
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    fused_pool = set(await _fused_baseline_ids(storage, embedder, version_id, limit=100))

    reranker = FakeReranker()
    searcher = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        reranker=reranker,
        rerank_model="fake-rerank",
        rerank_candidate_k=40,
    )
    hits = await searcher.search(_RERANK_QUERY, limit=5)
    await storage.close()

    assert {h.chunk_id for h in hits} <= fused_pool


@pytest.mark.asyncio
async def test_rerank_all_window_chunks_vanished_degrades(tmp_path) -> None:
    """If every window chunk vanished between fusion and fetch (race), there is
    nothing to score — degrade to the fused order rather than rerank an empty
    set. The reranker is never called."""
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    fused = await _fused_baseline_ids(storage, embedder, version_id, limit=5)

    reranker = FakeReranker()
    searcher = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        reranker=reranker,
        rerank_model="fake-rerank",
        rerank_candidate_k=40,
    )
    real_get_chunks = storage.get_chunks
    calls = {"n": 0}

    async def empty_first_call(chunk_ids):
        # Starve ONLY the first get_chunks (the rerank-window fetch in
        # _apply_rerank) so scored_ids is empty → degrade; the materialization
        # fetch that follows (degraded path re-fetches top-K) gets real rows.
        calls["n"] += 1
        if calls["n"] == 1:
            return []
        return await real_get_chunks(chunk_ids)

    storage.get_chunks = empty_first_call  # type: ignore[method-assign]
    try:
        hits = await searcher.search(_RERANK_QUERY, limit=5)
    finally:
        storage.get_chunks = real_get_chunks  # type: ignore[method-assign]
    await storage.close()

    assert reranker.call_count == 0
    assert [h.chunk_id for h in hits] == fused


@pytest.mark.asyncio
async def test_rerank_transient_error_degrades_to_fused_order(tmp_path) -> None:
    """A transient rerank failure must not 500 the read path — the searcher
    logs and falls back to the fused order."""
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    fused = await _fused_baseline_ids(storage, embedder, version_id, limit=5)

    reranker = FakeReranker(
        raise_error=TransientProviderError("rerank vendor 503")
    )
    searcher = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        reranker=reranker,
        rerank_model="fake-rerank",
        rerank_candidate_k=40,
    )
    hits = await searcher.search(_RERANK_QUERY, limit=5)
    await storage.close()

    assert [h.chunk_id for h in hits] == fused


@pytest.mark.asyncio
async def test_rerank_permanent_error_propagates(tmp_path) -> None:
    """A permanent rerank misconfig (bad key/model) fails fast rather than
    silently degrading — same fail-loud contract as the embedding path."""
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    reranker = FakeReranker(raise_error=ProviderError("rerank 401 bad key"))
    searcher = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        reranker=reranker,
        rerank_model="fake-rerank",
        rerank_candidate_k=40,
    )
    with pytest.raises(ProviderError):
        await searcher.search(_RERANK_QUERY, limit=5)
    await storage.close()


@pytest.mark.asyncio
async def test_rerank_only_fires_in_hybrid_mode(tmp_path) -> None:
    """Rerank is a hybrid-pipeline stage — it must NOT touch the bm25 / vector
    single-leg ablations (mirrors the graph leg's `mode == "hybrid"` gate), so
    `dikw client eval --retrieval all` keeps those rows pure vs published
    baselines. Production `retrieve` is always hybrid, so it still reranks.
    """
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    reranker = FakeReranker()
    searcher = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        reranker=reranker,
        rerank_model="fake-rerank",
    )
    await searcher.search(_RERANK_QUERY, limit=3, mode="bm25")
    await searcher.search(_RERANK_QUERY, limit=3, mode="vector")
    assert reranker.call_count == 0, "rerank must not fire on single-leg ablations"

    await searcher.search(_RERANK_QUERY, limit=3, mode="hybrid")
    await storage.close()
    assert reranker.call_count == 1, "rerank must fire on hybrid (the production mode)"


@pytest.mark.asyncio
async def test_rerank_leg_span_only_in_hybrid(tmp_path, span_exporter) -> None:
    """The rerank leg emits a `dikw.retrieve.leg` span (with a hit count) in
    hybrid mode, and none in a bm25 ablation — same leg-purity contract the
    other legs honor.
    """
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    searcher = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        text_version_id=version_id,
        reranker=FakeReranker(),
        rerank_model="fake-rerank",
    )
    await searcher.search(_RERANK_QUERY, limit=3, mode="hybrid")
    hybrid_legs = {
        s.attributes[telemetry.DIKW_RETRIEVAL_LEG]
        for s in span_exporter.get_finished_spans()
        if s.name == "dikw.retrieve.leg"
    }
    assert "rerank" in hybrid_legs
    rerank_spans = [
        s
        for s in span_exporter.get_finished_spans()
        if s.name == "dikw.retrieve.leg"
        and s.attributes[telemetry.DIKW_RETRIEVAL_LEG] == "rerank"
    ]
    assert all(telemetry.DIKW_LEG_HIT_COUNT in s.attributes for s in rerank_spans)

    await searcher.search(_RERANK_QUERY, limit=3, mode="bm25")
    await storage.close()
    # The bm25 run must add no new rerank leg span.
    bm25_rerank_spans = [
        s
        for s in span_exporter.get_finished_spans()
        if s.name == "dikw.retrieve.leg"
        and s.attributes[telemetry.DIKW_RETRIEVAL_LEG] == "rerank"
    ]
    assert len(bm25_rerank_spans) == len(rerank_spans), (
        "bm25 ablation must not open a rerank leg span"
    )


@pytest.mark.asyncio
async def test_hybrid_searcher_from_config_threads_rerank_knobs(tmp_path) -> None:
    """RetrievalConfig rerank knobs → from_config → search() wiring."""
    from dikw_core.config import RetrievalConfig

    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_multi_chunk_corpus(storage)

    cfg = RetrievalConfig(rerank_candidate_k=12, rerank_enabled=True)
    reranker = FakeReranker()
    searcher = HybridSearcher.from_config(
        storage,
        embedder,
        cfg,
        embedding_model="fake",
        text_version_id=version_id,
        reranker=reranker,
        rerank_model="fake-rerank",
    )
    assert searcher._rerank_candidate_k == 12
    assert searcher._rerank_enabled is True
    assert searcher._reranker is reranker
    hits = await searcher.search(_RERANK_QUERY, limit=3)
    await storage.close()
    assert reranker.call_count == 1
    assert len(hits) == 3


# ---- cjk tokenizer integration ----------------------------------------------


@pytest.mark.asyncio
async def test_cjk_tokenizer_jieba_ingest_stores_segmented_body(tmp_path) -> None:
    """End-to-end: ingest CJK text with cjk_tokenizer='jieba' → the
    documents_fts.body column holds whitespace-separated Chinese words.

    Without the preprocessor, body would contain ``机器学习入门`` as one
    run and FTS5's unicode61 would split per-character. With the
    preprocessor, body contains ``机器 学习 机器学习 入门`` (or similar,
    depending on jieba's dictionary) and unicode61 respects the
    whitespace boundaries.
    """
    import time

    from dikw_core.schemas import ChunkRecord, DocumentRecord, Layer

    storage = SQLiteStorage(tmp_path / "idx.sqlite", cjk_tokenizer="jieba")
    await storage.connect()
    await storage.migrate()

    doc_id = "source:zh.md"
    await storage.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            path="sources/zh.md",
            title="机器学习",
            hash="h1",
            mtime=time.time(),
            layer=Layer.SOURCE,
            active=True,
        )
    )
    await storage.replace_chunks(
        doc_id,
        [ChunkRecord(doc_id=doc_id, seq=0, start=0, end=6, text="机器学习入门")],
    )

    # ``documents_fts`` is body-only — JOIN onto ``chunks`` to scope
    # by doc_id (rowid alignment is the contract).
    def _read_body() -> str:
        conn = storage._require_conn()  # type: ignore[attr-defined]
        row = conn.execute(
            "SELECT body FROM documents_fts "
            "JOIN chunks c ON c.chunk_id = documents_fts.rowid "
            "WHERE c.doc_id = ?",
            (doc_id,),
        ).fetchone()
        return str(row["body"])

    body = _read_body()
    # At minimum jieba has surfaced at least one 2+ char Chinese word
    # as a whitespace-bounded token. The pre-feature path would store
    # the raw string "机器学习入门" with no whitespace.
    assert " " in body, f"body not segmented: {body!r}"
    await storage.close()


@pytest.mark.asyncio
async def test_cjk_tokenizer_jieba_end_to_end_bm25_recovers(tmp_path) -> None:
    """End-to-end proof the fix actually fires BM25 on Chinese text.

    Baseline (``cjk_tokenizer='none'``): indexing "机器学习入门" under
    unicode61 per-character tokenization means a MATCH on "机器学习"
    produces 0 hits — the indexed form is four unrelated single chars
    and the query's sanitized form asks for those same chars in
    isolation. Turn cjk_tokenizer on and the same storage + query
    sequence starts returning a hit.
    """
    import time

    from dikw_core.schemas import ChunkRecord, DocumentRecord, Layer

    async def _populate(cjk: str) -> SQLiteStorage:
        s = SQLiteStorage(tmp_path / f"idx-{cjk}.sqlite", cjk_tokenizer=cjk)  # type: ignore[arg-type]
        await s.connect()
        await s.migrate()
        doc_id = "source:zh.md"
        body = "机器学习入门是一本经典的机器学习教材"
        await s.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                path="sources/zh.md",
                title="机器学习入门",
                hash="h",
                mtime=time.time(),
                layer=Layer.SOURCE,
                active=True,
            )
        )
        await s.replace_chunks(
            doc_id,
            [ChunkRecord(doc_id=doc_id, seq=0, start=0, end=len(body), text=body)],
        )
        return s

    # default mode: searcher sees raw 机器学习 → per-char sanitize, FTS5
    # indexed per-char, match exists but is weak/noisy. Not asserting
    # strict 0 hits because per-char union *can* coincidentally match;
    # what matters is the jieba path produces a cleaner hit.
    storage_jieba = await _populate("jieba")
    searcher = HybridSearcher(
        storage_jieba, embedder=None, embedding_model=None, cjk_tokenizer="jieba"
    )
    hits = await searcher.search("机器学习", limit=5, mode="bm25")
    await storage_jieba.close()
    assert hits, "jieba path returned no bm25 hits on Chinese query"
    assert any("zh.md" in (h.path or "") for h in hits)


@pytest.mark.asyncio
async def test_cjk_tokenizer_none_leaves_body_verbatim(tmp_path) -> None:
    """Default 'none' mode must not touch body — regression guard for
    every existing ASCII-only wiki.
    """
    import time

    from dikw_core.schemas import ChunkRecord, DocumentRecord, Layer

    storage = SQLiteStorage(tmp_path / "idx.sqlite")  # default cjk_tokenizer="none"
    await storage.connect()
    await storage.migrate()

    doc_id = "source:en.md"
    body_text = "reciprocal rank fusion survives preprocessing"
    await storage.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            path="sources/en.md",
            title="fusion",
            hash="h2",
            mtime=time.time(),
            layer=Layer.SOURCE,
            active=True,
        )
    )
    await storage.replace_chunks(
        doc_id,
        [ChunkRecord(doc_id=doc_id, seq=0, start=0, end=len(body_text), text=body_text)],
    )

    def _read_body() -> str:
        conn = storage._require_conn()  # type: ignore[attr-defined]
        row = conn.execute(
            "SELECT body FROM documents_fts "
            "JOIN chunks c ON c.chunk_id = documents_fts.rowid "
            "WHERE c.doc_id = ?",
            (doc_id,),
        ).fetchone()
        return str(row["body"])

    assert _read_body() == body_text
    await storage.close()


@pytest.mark.asyncio
async def test_multimodal_q_vec_does_not_leak_into_text_vec_search(tmp_path) -> None:
    """T5 regression: text and multimodal legs use independent q_vecs.

    When both a text embedder (dim=64) and a multimodal embedder (dim=4)
    are configured, the multimodal q_vec must NOT be passed to
    ``vec_search`` (the text leg) — pre-fix that caused either dim
    mismatch (StorageError caught and silently dropped the leg) or
    semantic drift (when dims happened to match).
    """
    import time

    from dikw_core.schemas import (
        ChunkRecord,
        DocumentRecord,
        EmbeddingRow,
        EmbeddingVersion,
        Layer,
    )

    storage = SQLiteStorage(tmp_path / "t5.sqlite")
    await storage.connect()
    await storage.migrate()
    try:
        # Register both modalities; intentionally different dims so a
        # cross-leg q_vec leak would dim-mismatch loudly.
        text_v = await register_text_version(storage, dim=64, model="text-fake")
        mm_v = await storage.upsert_embed_version(
            EmbeddingVersion(
                provider="fake_mm",
                model="mm-fake",
                dim=4,
                normalize=True,
                distance="cosine",
                modality="multimodal",
            )
        )

        # Seed one doc + one chunk; populate text-vec table only (asset
        # leg has no asset registered, so vec_search_assets returns []).
        doc = DocumentRecord(
            doc_id="d1",
            path="sources/d.md",
            title="d",
            hash="h",
            mtime=time.time(),
            layer=Layer.SOURCE,
        )
        await storage.upsert_document(doc)
        cids = await storage.replace_chunks(
            "d1",
            [ChunkRecord(doc_id="d1", seq=0, start=0, end=5, text="alpha")],
        )
        text_emb = [1.0] + [0.0] * 63  # 64-dim
        await storage.upsert_embeddings(
            [EmbeddingRow(chunk_id=cids[0], version_id=text_v, embedding=text_emb)]
        )

        # Stub the vec legs so the spies just record the q_vec routed
        # to each leg — the test's invariant is "each modality gets its
        # own dim", not "the legs return real hits". Avoiding the real
        # SQL also dodges a known concurrent-access race on sqlite3
        # connections when both legs hit `to_thread` simultaneously.
        captured: dict[str, list[float]] = {}

        async def stub_vec_search(emb, **kwargs):
            captured["text"] = list(emb)
            return []

        async def stub_vec_search_assets(emb, **kwargs):
            captured["mm"] = list(emb)
            return []

        storage.vec_search = stub_vec_search  # type: ignore[method-assign]
        storage.vec_search_assets = stub_vec_search_assets  # type: ignore[method-assign]

        text_embedder = FakeEmbeddings()
        mm_embedder = FakeMultimodalEmbedding(dim=4)
        searcher = HybridSearcher(
            storage,
            text_embedder,
            embedding_model="text-fake",
            text_version_id=text_v,
            multimodal=MultimodalSearch(
                embedder=mm_embedder,
                model="mm-fake",
                asset_version_id=mm_v,
            ),
        )

        # Run a hybrid search — both vec legs should fire.
        await searcher.search("hello world", limit=3)

        # Independent q_vecs: each leg got a vector in its own dim.
        assert "text" in captured, "vec_search must run on text leg"
        assert "mm" in captured, "vec_search_assets must run on mm leg"
        assert len(captured["text"]) == 64, (
            f"text vec_search got dim {len(captured['text'])}, expected 64 — "
            f"multimodal q_vec leaked into the text leg (T5 regression)"
        )
        assert len(captured["mm"]) == 4, (
            f"mm vec_search_assets got dim {len(captured['mm'])}, expected 4"
        )
        assert captured["text"] != captured["mm"], (
            "text and mm q_vecs must be distinct objects from distinct embedders"
        )
    finally:
        await storage.close()


# ---- retrieval-leg tracing spans (PR2b) -------------------------------------
#
# The fusion legs (bm25 / vector / graph) are pure in-process work with no
# provider call, so they are dark to the PR2a gen_ai.* spans — these pin that
# each leg now emits a ``dikw.retrieve.leg`` span carrying its hit count, that
# ablation modes only open the legs they run, and (critically, since search.py
# is eval-gated) that the instrumentation does NOT perturb retrieval results.


@pytest.mark.asyncio
async def test_search_emits_a_leg_span_per_active_leg_hybrid(tmp_path, span_exporter):
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_fixture_corpus(storage)

    searcher = HybridSearcher(
        storage, embedder, embedding_model="fake", text_version_id=version_id
    )
    hits = await searcher.search("DIKW pyramid", limit=3, mode="hybrid")
    assert hits
    await storage.close()

    leg_spans = [
        s for s in span_exporter.get_finished_spans() if s.name == "dikw.retrieve.leg"
    ]
    legs = {s.attributes[telemetry.DIKW_RETRIEVAL_LEG] for s in leg_spans}
    # hybrid runs bm25 + vector + graph (no asset leg without a multimodal index).
    assert legs == {"bm25", "vector", "graph"}
    # every leg span carries a hit count.
    for s in leg_spans:
        assert telemetry.DIKW_LEG_HIT_COUNT in s.attributes


@pytest.mark.asyncio
async def test_search_bm25_mode_opens_only_the_bm25_leg(tmp_path, span_exporter):
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_fixture_corpus(storage)

    searcher = HybridSearcher(
        storage, embedder, embedding_model="fake", text_version_id=version_id
    )
    await searcher.search("DIKW pyramid", limit=3, mode="bm25")
    await storage.close()

    legs = {
        s.attributes[telemetry.DIKW_RETRIEVAL_LEG]
        for s in span_exporter.get_finished_spans()
        if s.name == "dikw.retrieve.leg"
    }
    # bm25 ablation: vector + graph legs must not open.
    assert legs == {"bm25"}


@pytest.mark.asyncio
async def test_leg_tracing_does_not_perturb_results(tmp_path, span_exporter):
    """The OTel recording machinery does not perturb retrieval results: identical
    (chunk_id, score) ordering whether telemetry is active (``span_exporter``
    installs a recording provider) or torn down to a no-op
    (``reset_telemetry_for_testing``). The ``_traced_leg`` wrapper code runs in
    BOTH halves, so the stronger behavior-preserving guarantee for this
    eval-gated file comes from the *pre-existing* fusion/RRF/ranking tests above
    (untouched, purely additive diff) + the eval-gate ``no-baseline-needed``
    claim; this pins that span recording itself is side-effect-free."""
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_fixture_corpus(storage)
    searcher = HybridSearcher(
        storage, embedder, embedding_model="fake", text_version_id=version_id
    )

    traced = await searcher.search("DIKW pyramid", limit=5, mode="hybrid")
    # Tear down the recording provider → get_tracer() goes back to no-op.
    telemetry.reset_telemetry_for_testing()
    untraced = await searcher.search("DIKW pyramid", limit=5, mode="hybrid")
    await storage.close()

    assert [(h.chunk_id, h.score) for h in traced] == [
        (h.chunk_id, h.score) for h in untraced
    ]


def _leg_spans(exporter):
    return [s for s in exporter.get_finished_spans() if s.name == "dikw.retrieve.leg"]


@pytest.mark.asyncio
async def test_search_asset_leg_emits_span(tmp_path, span_exporter):
    """The multimodal/asset leg — the one whose NotSupported handling moved into
    _traced_leg — emits its own ``dikw.retrieve.leg`` span ("asset")."""
    import time

    from dikw_core.schemas import (
        ChunkRecord,
        DocumentRecord,
        EmbeddingRow,
        EmbeddingVersion,
        Layer,
    )

    storage = SQLiteStorage(tmp_path / "asset.sqlite")
    await storage.connect()
    await storage.migrate()
    try:
        text_v = await register_text_version(storage, dim=64, model="text-fake")
        mm_v = await storage.upsert_embed_version(
            EmbeddingVersion(
                provider="fake_mm", model="mm-fake", dim=4,
                normalize=True, distance="cosine", modality="multimodal",
            )
        )
        await storage.upsert_document(
            DocumentRecord(
                doc_id="d1", path="sources/d.md", title="d", hash="h",
                mtime=time.time(), layer=Layer.SOURCE,
            )
        )
        cids = await storage.replace_chunks(
            "d1", [ChunkRecord(doc_id="d1", seq=0, start=0, end=5, text="alpha")]
        )
        await storage.upsert_embeddings(
            [EmbeddingRow(chunk_id=cids[0], version_id=text_v, embedding=[1.0] + [0.0] * 63)]
        )

        # Stub both vec legs (return []) — the asset leg span emits whenever the
        # asset task is created (multimodal configured), regardless of hits, and
        # stubbing dodges the known concurrent-sqlite race when two real vec
        # legs hit to_thread at once (see the T5 test above).
        async def stub_vec(emb, **kwargs):
            return []

        async def stub_assets(emb, **kwargs):
            return []

        storage.vec_search = stub_vec  # type: ignore[method-assign]
        storage.vec_search_assets = stub_assets  # type: ignore[method-assign]
        searcher = HybridSearcher(
            storage, FakeEmbeddings(), embedding_model="text-fake",
            text_version_id=text_v,
            multimodal=MultimodalSearch(
                embedder=FakeMultimodalEmbedding(dim=4), model="mm-fake",
                asset_version_id=mm_v,
            ),
        )
        await searcher.search("alpha", limit=3, mode="hybrid")

        legs = {s.attributes[telemetry.DIKW_RETRIEVAL_LEG] for s in _leg_spans(span_exporter)}
        assert "asset" in legs, legs
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_graceful_leg_swallows_notsupported_to_empty_leg(tmp_path, span_exporter):
    """A graceful leg (vector/asset, graceful_notsupported=True) degrades to an
    empty leg on a backend that raises NotSupported — span ends OK with
    hit_count 0, search returns the surviving (fts/graph) results, no raise."""
    from opentelemetry.trace import StatusCode

    storage = SQLiteStorage(tmp_path / "ns.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_fixture_corpus(storage)

    async def raise_ns(*args, **kwargs):
        raise NotSupported("vec_search unavailable")

    storage.vec_search = raise_ns  # type: ignore[method-assign]
    searcher = HybridSearcher(
        storage, embedder, embedding_model="fake", text_version_id=version_id
    )
    hits = await searcher.search("DIKW pyramid", limit=3, mode="hybrid")
    assert hits, "fts/graph legs should still return hits when vector degrades"
    await storage.close()

    vec_legs = [
        s for s in _leg_spans(span_exporter)
        if s.attributes[telemetry.DIKW_RETRIEVAL_LEG] == "vector"
    ]
    assert len(vec_legs) == 1
    assert vec_legs[0].attributes[telemetry.DIKW_LEG_HIT_COUNT] == 0
    # graceful degradation is NOT an error — the backend simply lacks the leg.
    assert vec_legs[0].status.status_code != StatusCode.ERROR


@pytest.mark.asyncio
async def test_fts_leg_does_not_swallow_notsupported(tmp_path, span_exporter):
    """The always-available FTS leg keeps graceful_notsupported=False: a
    NotSupported propagates (matches main, where the fts await had no except)
    and the leg span is marked ERROR."""
    from opentelemetry.trace import StatusCode

    storage = SQLiteStorage(tmp_path / "fts-ns.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder, version_id = await _populate_fixture_corpus(storage)

    async def raise_ns(*args, **kwargs):
        raise NotSupported("fts_search unavailable")

    storage.fts_search = raise_ns  # type: ignore[method-assign]
    searcher = HybridSearcher(
        storage, embedder, embedding_model="fake", text_version_id=version_id
    )
    with pytest.raises(NotSupported):
        await searcher.search("DIKW pyramid", limit=3, mode="bm25")
    await storage.close()

    bm25_legs = [
        s for s in _leg_spans(span_exporter)
        if s.attributes[telemetry.DIKW_RETRIEVAL_LEG] == "bm25"
    ]
    assert len(bm25_legs) == 1
    assert bm25_legs[0].status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_traced_leg_cancel_marks_span_cancelled(span_exporter):
    """Cancelling a leg task mid-await (the _traced_leg create_task seam): the
    leg span opened INSIDE that task ends flagged dikw.cancelled, status UNSET,
    not ERROR — and the contextvars Token closes in the same task it opened in."""
    from opentelemetry.trace import StatusCode

    started = asyncio.Event()

    async def _hang() -> list[int]:
        started.set()
        await asyncio.Event().wait()  # never completes
        return []

    task = asyncio.create_task(_traced_leg("bm25", _hang()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    leg = _leg_spans(span_exporter)
    assert len(leg) == 1
    assert leg[0].attributes[telemetry.DIKW_CANCELLED] is True
    assert leg[0].status.status_code == StatusCode.UNSET
