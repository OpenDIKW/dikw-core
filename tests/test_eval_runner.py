"""Runner — drive ingest + hybrid search + metrics for a single dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core.eval.dataset import (
    DatasetSpec,
    Query,
    load_dataset,
)
from dikw_core.eval.runner import (
    EvalError,
    EvalReport,
    SynthEvalError,
    ThresholdResult,
    check_thresholds,
    run_eval,
)


def _write_dataset(
    root: Path,
    *,
    queries: list[tuple[str, list[str]]],
    thresholds: dict[str, float] | None = None,
    docs: dict[str, str] | None = None,
) -> Path:
    ds = root / "toy"
    (ds / "corpus").mkdir(parents=True, exist_ok=True)
    payload = docs or {
        "alpha": "# Alpha\n\nAlpha describes foo and bar topics.\n",
        "beta": "# Beta\n\nBeta discusses baz and qux.\n",
    }
    for stem, body in payload.items():
        (ds / "corpus" / f"{stem}.md").write_text(body, encoding="utf-8")

    thr = thresholds if thresholds is not None else {
        "hit_at_3": 0.5,
        "hit_at_10": 0.5,
        "mrr": 0.3,
    }
    thr_yaml = "\n".join(f"  {k}: {v}" for k, v in thr.items())
    (ds / "dataset.yaml").write_text(
        f"name: toy\ndescription: runner smoke test\nthresholds:\n{thr_yaml}\n",
        encoding="utf-8",
    )

    q_lines: list[str] = ["queries:"]
    for q, exp in queries:
        q_lines.append(f"  - q: {q}")
        q_lines.append(f"    expect_any: [{', '.join(exp)}]")
    (ds / "queries.yaml").write_text("\n".join(q_lines) + "\n", encoding="utf-8")
    return ds


def test_materialise_base_honors_target_tokens_override(tmp_path: Path) -> None:
    """``run_synth_eval``'s ``target_tokens_per_group`` override is threaded
    into the throwaway base's ``dikw.yml`` so grouping-sensitive changes (the
    priority-create / slug features) can be A/B'd by fanning a small corpus
    into multiple groups. ``None`` leaves the production default (3600)."""
    from dikw_core.config import (
        CONFIG_FILENAME,
        AssetsConfig,
        ProviderConfig,
        RetrievalConfig,
        load_config,
    )
    from dikw_core.eval.runner import _materialise_base

    spec = load_dataset(_write_dataset(tmp_path, queries=[("foo", ["alpha"])]))
    provider_cfg = ProviderConfig(
        llm_api_key_env="ANTHROPIC_API_KEY",
        embedding_model="fake",
        embedding_dim=64,
        embedding_revision="",
        embedding_normalize=True,
        embedding_distance="cosine",
        embedding_api_key_env="OPENAI_API_KEY",
    )
    kwargs = {
        "provider_cfg": provider_cfg,
        "retrieval_cfg": RetrievalConfig(),
        "assets_cfg": AssetsConfig(),
    }

    overridden = _materialise_base(
        tmp_path / "ov", spec, target_tokens_per_group=512, **kwargs
    )
    assert load_config(overridden / CONFIG_FILENAME).synth.target_tokens_per_group == 512

    default = _materialise_base(tmp_path / "def", spec, **kwargs)
    assert load_config(default / CONFIG_FILENAME).synth.target_tokens_per_group == 3600


@pytest.mark.asyncio
async def test_run_eval_returns_report_with_metrics_and_passed_flag(
    tmp_path: Path,
) -> None:
    ds = _write_dataset(
        tmp_path,
        queries=[
            ("foo and bar topics", ["alpha"]),
            ("baz and qux", ["beta"]),
        ],
    )
    spec = load_dataset(ds)
    report = await run_eval(spec)

    assert isinstance(report, EvalReport)
    # Doc-only dataset (no targets.yaml) emits the 5 unprefixed legacy
    # keys plus their ``doc/`` namespaced aliases — same numbers under
    # both shapes so back-compat thresholds keep working.
    assert set(report.metrics.keys()) == {
        "hit_at_3",
        "hit_at_10",
        "mrr",
        "ndcg_at_10",
        "recall_at_100",
        "doc/hit_at_3",
        "doc/hit_at_10",
        "doc/mrr",
        "doc/ndcg_at_10",
        "doc/recall_at_100",
    }
    for metric in ("hit_at_3", "hit_at_10", "mrr", "ndcg_at_10", "recall_at_100"):
        assert report.metrics[metric] == report.metrics[f"doc/{metric}"]
    assert all(0.0 <= v <= 1.0 for v in report.metrics.values())
    # FakeEmbeddings' bag-of-words on a 2-doc corpus with keyword-rich
    # queries should land cleanly in top-3.
    assert report.metrics["hit_at_3"] == 1.0
    assert report.metrics["mrr"] == 1.0
    assert report.metrics["ndcg_at_10"] == 1.0
    assert report.metrics["recall_at_100"] == 1.0
    # Single-mode default → modes carries the one mode that ran.
    assert report.modes == ["hybrid"]
    assert report.thresholds == spec.thresholds
    assert report.passed is True
    # Per-query diagnostic data preserved for the CLI's failure table.
    assert len(report.per_query) == 2
    q0 = report.per_query[0]
    assert q0["q"] == "foo and bar topics"
    assert q0["expect_any"] == ["alpha"]
    assert "alpha" in q0["ranked"][:3]


@pytest.mark.asyncio
async def test_run_eval_passed_false_when_any_metric_below_threshold(
    tmp_path: Path,
) -> None:
    # Expected stem "ghost" is not in the corpus — guaranteed miss
    # regardless of what HybridSearcher decides to rank first.
    ds = _write_dataset(
        tmp_path,
        queries=[
            ("foo and bar", ["ghost"]),
        ],
        thresholds={"hit_at_3": 1.0, "hit_at_10": 1.0, "mrr": 1.0},
    )
    spec = load_dataset(ds)
    report = await run_eval(spec)
    assert report.passed is False
    assert report.metrics["hit_at_3"] == 0.0


@pytest.mark.asyncio
async def test_run_eval_empty_queries_guarded_at_dataset_load(
    tmp_path: Path,
) -> None:
    """Empty queries.yaml is a DatasetError — runner needn't re-validate."""
    from dikw_core.eval.dataset import DatasetError

    ds = _write_dataset(tmp_path, queries=[("x", ["alpha"])])
    (ds / "queries.yaml").write_text("queries: []\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(ds)


@pytest.mark.asyncio
async def test_run_eval_with_missing_thresholds_defaults_passed_true(
    tmp_path: Path,
) -> None:
    """A dataset without thresholds still runs — passed defaults to True.

    Useful for exploratory datasets where the user hasn't calibrated yet.
    """
    ds = _write_dataset(
        tmp_path,
        queries=[("foo", ["alpha"])],
        thresholds={},
    )
    spec = load_dataset(ds)
    report = await run_eval(spec)
    assert report.thresholds == {}
    assert report.passed is True  # nothing to fail against


@pytest.mark.asyncio
async def test_run_eval_synthetic_spec_direct(tmp_path: Path) -> None:
    """Runner accepts a ``DatasetSpec`` constructed directly, not just via disk.

    Guards against accidentally tying runner to the yaml loader.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nalpha content\n", encoding="utf-8")
    (corpus / "b.md").write_text("# B\n\nbeta content\n", encoding="utf-8")
    spec = DatasetSpec(
        name="synthetic",
        description="",
        thresholds={"hit_at_3": 0.5},
        corpus_dir=corpus,
        queries=[Query(q="alpha content", expect_any=["a"])],
    )
    report = await run_eval(spec)
    assert report.metrics["hit_at_3"] == 1.0
    assert report.passed


@pytest.mark.asyncio
async def test_run_eval_mode_all_emits_per_mode_and_canonical_metrics(
    tmp_path: Path,
) -> None:
    """``mode='all'`` runs each retrieval leg, emits prefixed metrics for
    every (mode, key) pair, and mirrors the hybrid mode unprefixed so
    existing dataset thresholds keep gating.
    """
    ds = _write_dataset(
        tmp_path,
        queries=[
            ("foo and bar topics", ["alpha"]),
            ("baz and qux", ["beta"]),
        ],
    )
    spec = load_dataset(ds)
    report = await run_eval(spec, mode="all")

    assert sorted(report.modes) == ["bm25", "hybrid", "vector"]
    base_keys = {"hit_at_3", "hit_at_10", "mrr", "ndcg_at_10", "recall_at_100"}
    for m in ("bm25", "vector", "hybrid"):
        for k in base_keys:
            assert f"{m}/{k}" in report.metrics, f"missing {m}/{k}"
    # Unprefixed mirror equals the hybrid mode (canonical) — gating
    # against existing thresholds keeps working under --retrieval all.
    for k in base_keys:
        assert report.metrics[k] == report.metrics[f"hybrid/{k}"]
    # Sanity: this hermetic 2-doc corpus passes its 0.5 thresholds on hybrid.
    assert report.passed


@pytest.mark.asyncio
async def test_run_eval_negative_query_does_not_drag_positive_metrics(
    tmp_path: Path,
) -> None:
    """Negative queries (expect_none=True) must not contribute to hit@k/MRR.

    Two queries: one hit, one ``expect_none``. If the runner averaged
    hit@3 over both, the negative would count as a miss and halve the
    score. Correct behaviour is to compute metrics over positives only.
    """
    # Positive query with a guaranteed hit + negative query (no expected match).
    ds = _write_dataset(
        tmp_path,
        queries=[
            ("foo and bar topics", ["alpha"]),  # positive — alpha is the answer
        ],
    )
    # Append a negative query manually (helper only supports positives).
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: foo and bar topics\n"
        "    expect_any: [alpha]\n"
        "  - q: totally unrelated out-of-domain question\n"
        "    expect_none: true\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    report = await run_eval(spec)

    # hit@3 = 1.0 over 1 positive, not 0.5 over positive + negative
    assert report.metrics["hit_at_3"] == 1.0
    assert report.metrics["hit_at_10"] == 1.0
    assert report.metrics["mrr"] == 1.0
    # Only the positive query lands in per_query (it's what metrics table uses).
    assert len(report.per_query) == 1
    assert report.per_query[0]["q"] == "foo and bar topics"


@pytest.mark.asyncio
async def test_run_eval_exposes_negative_diagnostics(tmp_path: Path) -> None:
    """Negative queries still get executed — their top-k is surfaced as
    observational diagnostics so humans can eyeball "what DID get retrieved
    for this out-of-domain query?".
    """
    ds = _write_dataset(tmp_path, queries=[("foo and bar", ["alpha"])])
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: foo and bar\n"
        "    expect_any: [alpha]\n"
        "  - q: totally unrelated\n"
        "    expect_none: true\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    report = await run_eval(spec)

    assert len(report.negative_diagnostics) == 1
    neg = report.negative_diagnostics[0]
    assert neg["q"] == "totally unrelated"
    # Retrieval always returns SOMETHING from a non-empty corpus; the point
    # is to observe what, not to pass/fail on it.
    assert "ranked" in neg
    assert isinstance(neg["ranked"], list)


@pytest.mark.asyncio
async def test_run_eval_all_negative_dataset_metrics_empty(tmp_path: Path) -> None:
    """A dataset of only negatives produces no hit@k/MRR values — nothing
    to average. The report's ``passed`` flag still works (trivially True
    if no thresholds, or would fail if the user set one for a metric the
    runner now skips).
    """
    ds = _write_dataset(tmp_path, queries=[("placeholder", ["alpha"])], thresholds={})
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: out of domain one\n"
        "    expect_none: true\n"
        "  - q: out of domain two\n"
        "    expect_none: true\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    report = await run_eval(spec)

    # Metrics over an empty positive set are not meaningful — runner
    # omits them so a spurious 0.0 doesn't fail a threshold that was
    # never intended for a negatives-only corpus.
    assert "hit_at_3" not in report.metrics
    assert len(report.negative_diagnostics) == 2
    assert report.passed is True


def test_eval_error_is_raised_for_nonexistent_corpus_dir(tmp_path: Path) -> None:
    """Programmatic DatasetSpec with bad corpus path → EvalError at run time."""
    spec = DatasetSpec(
        name="bad",
        description="",
        thresholds={},
        corpus_dir=tmp_path / "nonexistent",
        queries=[Query(q="x", expect_any=["y"])],
    )
    import asyncio

    with pytest.raises(EvalError, match="corpus"):
        asyncio.run(run_eval(spec))


@pytest.mark.asyncio
async def test_run_eval_ranked_docs_deduped_after_chunk_level_fusion(
    tmp_path: Path,
) -> None:
    """Chunk-level fusion (Phase 1) repeats the same doc_id across hits
    when multiple chunks of one doc rank highly. Doc-level metrics
    (hit@k, MRR, nDCG@k, recall@k) assume unique doc identities in rank
    order, so the runner must dedup ranked stems while preserving the
    first-occurrence order. Without the dedup, BEIR/CMTEB benchmarks
    silently regress.

    Setup: a doc with two chunks both highly relevant + a hit doc that
    must still appear within top-k after dedup.
    """
    ds = tmp_path / "dedup"
    (ds / "corpus").mkdir(parents=True)
    # alpha.md has two strongly-matching chunks (separated by a header
    # so the chunker emits two distinct chunks).
    (ds / "corpus" / "alpha.md").write_text(
        "# Alpha\n\n"
        + ("alpha rank fusion topic body. " * 30)
        + "\n\n## Subsection\n\n"
        + ("alpha rank fusion topic continues. " * 30),
        encoding="utf-8",
    )
    (ds / "corpus" / "beta.md").write_text(
        "# Beta\n\n" + ("alpha rank fusion topic " * 30),
        encoding="utf-8",
    )
    (ds / "dataset.yaml").write_text(
        "name: dedup\n"
        "description: chunk-level dedup regression guard\n"
        "thresholds: {}\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: alpha rank fusion\n"
        "    expect_any: [alpha, beta]\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    report = await run_eval(spec)

    ranked = report.per_query[0]["ranked"]
    # The exact ranking depends on chunker + RRF, but the dedup invariant
    # is unconditional: no consecutive (or any) duplicate stems.
    assert len(ranked) == len(set(ranked)), (
        f"ranked stems must be unique after dedup; got {ranked}"
    )


# ---- chunk + asset views (4-file dataset contract) -----------------------


def _write_mm_dataset_with_targets(root: Path) -> Path:
    """3-doc / 6-image / 12-query multimodal dataset with targets.yaml.

    Each doc has two H2 sections, each section has its own image. Queries
    cover all three views (doc / chunk / asset) plus a negative.
    """
    ds = root / "mm-mini"
    corpus = ds / "corpus"
    (corpus / "images").mkdir(parents=True, exist_ok=True)

    # Three category docs, each with two H2 sections padded long enough
    # that the heading-aware chunker actually splits them — otherwise
    # multiple anchors collapse into one chunk and the runner loud-fails
    # on collision (the "loud failure" path is exercised by
    # ``test_run_eval_chunk_target_collision_raises``).
    # ~300+ token padding per section so the heading-aware chunker
    # crosses its overlap budget (135 tokens at default max_tokens=900,
    # overlap_ratio=0.15) and forces a split at each H2. Built as ~30
    # short paragraphs of distinct filler words.
    pad = "\n\n".join(
        f"Filler paragraph number {i} contains lorem ipsum dolor sit amet "
        f"consectetur adipiscing elit alpha bravo charlie delta echo "
        f"foxtrot golf hotel india juliet kilo lima mike november."
        for i in range(15)
    ) + "\n\n"
    docs = {
        "fruits": (
            "# Fruits\n\n"
            "## 苹果 / Apple\n\n"
            "Target: fruits.apple\n\n"
            "Apple section: red round fruit, green leaves, sweet flavor.\n\n"
            f"{pad}"
            "![apple](images/fruits-apple.png)\n\n"
            "## 香蕉 / Banana\n\n"
            "Target: fruits.banana\n\n"
            "Banana section: long curved yellow tropical fruit.\n\n"
            f"{pad}"
            "![banana](images/fruits-banana.png)\n"
        ),
        "animals": (
            "# Animals\n\n"
            "## 猫 / Cat\n\n"
            "Target: animals.cat\n\n"
            "Cat section: feline whiskers triangular ears purring pet.\n\n"
            f"{pad}"
            "![cat](images/animals-cat.png)\n\n"
            "## 狗 / Dog\n\n"
            "Target: animals.dog\n\n"
            "Dog section: canine loyal companion barking wagging tail.\n\n"
            f"{pad}"
            "![dog](images/animals-dog.png)\n"
        ),
        "vehicles": (
            "# Vehicles\n\n"
            "## 汽车 / Car\n\n"
            "Target: vehicles.car\n\n"
            "Car section: four-wheel road automobile passenger driving.\n\n"
            f"{pad}"
            "![car](images/vehicles-car.png)\n\n"
            "## 公交 / Bus\n\n"
            "Target: vehicles.bus\n\n"
            "Bus section: large public transport multiple seats route.\n\n"
            f"{pad}"
            "![bus](images/vehicles-bus.png)\n"
        ),
    }
    for stem, body in docs.items():
        (corpus / f"{stem}.md").write_text(body, encoding="utf-8")

    # Distinct fake-PNG bytes per image (not real PNGs — markdown ingest
    # detects MIME from magic bytes; we need a valid header so
    # ``materialize_asset`` doesn't reject as "unrecognized format").
    png_header = bytes(
        [
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
            0x89,
        ]
    )
    image_names = [
        "fruits-apple", "fruits-banana",
        "animals-cat", "animals-dog",
        "vehicles-car", "vehicles-bus",
    ]
    for i, name in enumerate(image_names):
        # Append a unique tail so each image has a distinct sha256.
        (corpus / "images" / f"{name}.png").write_bytes(
            png_header + bytes([i, i, i, i])
        )

    (ds / "dataset.yaml").write_text(
        "name: mm-mini\n"
        "description: hermetic multimodal fixture\n"
        "thresholds:\n"
        "  hit_at_3: 0.5\n"
        "  chunk/hit_at_3: 0.4\n",
        encoding="utf-8",
    )
    (ds / "targets.yaml").write_text(
        "assets:\n"
        "  - id: fruits.apple.image\n    doc: fruits\n    path: images/fruits-apple.png\n    anchor: fruits.apple\n"
        "  - id: fruits.banana.image\n    doc: fruits\n    path: images/fruits-banana.png\n    anchor: fruits.banana\n"
        "  - id: animals.cat.image\n    doc: animals\n    path: images/animals-cat.png\n    anchor: animals.cat\n"
        "  - id: animals.dog.image\n    doc: animals\n    path: images/animals-dog.png\n    anchor: animals.dog\n"
        "  - id: vehicles.car.image\n    doc: vehicles\n    path: images/vehicles-car.png\n    anchor: vehicles.car\n"
        "  - id: vehicles.bus.image\n    doc: vehicles\n    path: images/vehicles-bus.png\n    anchor: vehicles.bus\n"
        "chunks:\n"
        "  - id: fruits.apple.text\n    doc: fruits\n    anchor: fruits.apple\n    asset_id: fruits.apple.image\n"
        "  - id: fruits.banana.text\n    doc: fruits\n    anchor: fruits.banana\n    asset_id: fruits.banana.image\n"
        "  - id: animals.cat.text\n    doc: animals\n    anchor: animals.cat\n    asset_id: animals.cat.image\n"
        "  - id: animals.dog.text\n    doc: animals\n    anchor: animals.dog\n    asset_id: animals.dog.image\n"
        "  - id: vehicles.car.text\n    doc: vehicles\n    anchor: vehicles.car\n    asset_id: vehicles.car.image\n"
        "  - id: vehicles.bus.text\n    doc: vehicles\n    anchor: vehicles.bus\n    asset_id: vehicles.bus.image\n",
        encoding="utf-8",
    )
    # Mix of doc-only / chunk / asset queries.
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - id: q-doc-fruits\n    q: red round fruit sweet\n    expect_any: [fruits]\n"
        "  - id: q-chunk-apple\n    query_type: text_chunk\n    q: red round fruit\n"
        "    expect_any: [fruits]\n    expect_chunk_any: [fruits.apple.text]\n"
        "  - id: q-chunk-banana\n    query_type: text_chunk\n    q: long curved yellow\n"
        "    expect_any: [fruits]\n    expect_chunk_any: [fruits.banana.text]\n"
        "  - id: q-asset-apple\n    query_type: asset\n    q: apple image\n"
        "    expect_any: [fruits]\n    expect_asset_any: [fruits.apple.image]\n",
        encoding="utf-8",
    )
    return ds


@pytest.mark.asyncio
async def test_run_eval_emits_chunk_and_asset_view_metrics(
    tmp_path: Path,
) -> None:
    """A 4-file dataset (with targets.yaml) produces ``chunk/<metric>``
    and ``asset/<metric>`` keys alongside the legacy unprefixed doc
    metrics. The view list reflects what was actually scored.
    """
    ds = _write_mm_dataset_with_targets(tmp_path)
    spec = load_dataset(ds)
    report = await run_eval(spec)

    # Doc view back-compat.
    assert "hit_at_3" in report.metrics
    assert "doc/hit_at_3" in report.metrics
    assert report.metrics["hit_at_3"] == report.metrics["doc/hit_at_3"]

    # Chunk view emitted (queries opted in via expect_chunk_any).
    assert "chunk/hit_at_3" in report.metrics
    assert "chunk/mrr" in report.metrics
    assert 0.0 <= report.metrics["chunk/hit_at_3"] <= 1.0

    # Asset view emitted (queries opted in via expect_asset_any).
    assert "asset/hit_at_3" in report.metrics
    assert 0.0 <= report.metrics["asset/hit_at_3"] <= 1.0

    # Views list reflects what was scored.
    assert sorted(report.views) == ["asset", "chunk", "doc"]


@pytest.mark.asyncio
async def test_run_eval_targets_yaml_anchor_resolution_loud_failure(
    tmp_path: Path,
) -> None:
    """Anchor missing from doc body → EvalError at runtime (not a
    silent zero score). Eval datasets are authored, mismatches must surface.
    """
    ds = _write_mm_dataset_with_targets(tmp_path)
    # Use a fixture variant where the queries reference a single
    # chunk-target whose anchor is corrupted. Loader-time named-id
    # validation passes (the id exists in targets.yaml), but the
    # runner's anchor-resolver loud-fails when it scans the doc body
    # — which is the path under test here.
    (ds / "targets.yaml").write_text(
        "assets: []\n"
        "chunks:\n"
        "  - id: fruits.apple.text\n    doc: fruits\n    anchor: completely.missing.anchor\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - id: q-chunk\n"
        "    q: section about apples\n"
        "    expect_any: [fruits]\n"
        "    expect_chunk_any: [fruits.apple.text]\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    with pytest.raises(EvalError, match="anchor"):
        await run_eval(spec)


@pytest.mark.asyncio
async def test_run_eval_threads_multimodal_config_into_temp_wiki(
    tmp_path: Path,
) -> None:
    """When ``assets_config`` is passed, the temp wiki's ``dikw.yml``
    must carry the same ``assets.multimodal`` block — otherwise
    ``HybridSearcher`` builds without an asset leg and the asset
    metrics would silently fall back to chunk-promoted refs only.
    """
    from dikw_core import api as api_mod
    from dikw_core.config import (
        AssetsConfig,
        MultimodalEmbedConfig,
        load_config,
    )

    ds = _write_mm_dataset_with_targets(tmp_path)
    spec = load_dataset(ds)

    assets_cfg = AssetsConfig(
        multimodal=MultimodalEmbedConfig(
            provider="gitee_multimodal",
            model="qwen3-vl-fake",
            dim=4,
        )
    )

    captured_cfgs: list[dict[str, object]] = []

    # Stop after _materialise_base + _copy_corpus run by intercepting
    # api.ingest — we only need to verify the wiki's dikw.yml shape.
    real_ingest = api_mod.ingest

    async def _capture_ingest(wiki, **kwargs):  # type: ignore[no-untyped-def]
        cfg = load_config(wiki / "dikw.yml")
        captured_cfgs.append(
            {
                "mm_provider": cfg.assets.multimodal.provider if cfg.assets.multimodal else None,
                "mm_model": cfg.assets.multimodal.model if cfg.assets.multimodal else None,
                "mm_dim": cfg.assets.multimodal.dim if cfg.assets.multimodal else None,
            }
        )
        # Skip real ingest; we only care that the cfg landed correctly.
        from dikw_core.api import IngestReport
        return IngestReport(scanned=0, added=0, updated=0, unchanged=0, chunks=0, embedded=0)

    api_mod.ingest = _capture_ingest  # type: ignore[assignment]
    try:
        # Will fail downstream (no chunks ingested) — that's fine,
        # we only need the cfg-write step to land.
        with pytest.raises(EvalError):
            await run_eval(spec, assets_config=assets_cfg, cache_mode="off")
    finally:
        api_mod.ingest = real_ingest  # type: ignore[assignment]

    assert len(captured_cfgs) == 1
    assert captured_cfgs[0]["mm_provider"] == "gitee_multimodal"
    assert captured_cfgs[0]["mm_model"] == "qwen3-vl-fake"
    assert captured_cfgs[0]["mm_dim"] == 4


@pytest.mark.asyncio
async def test_run_eval_chunk_target_collision_raises(tmp_path: Path) -> None:
    """Two chunk targets resolving to the same runtime ``(doc, seq)`` —
    the chunker fit both anchors into one chunk — must loud-fail with
    EvalError. Silent collapse would make chunk-level metrics quietly
    wrong (an earlier-named target would become unreachable even when
    its chunk ranks first).
    """
    # Tiny doc: two H2 sections short enough that the heading-aware
    # chunker keeps both in chunk 0 (single accumulator under
    # max_overlap). Both anchors then resolve to ``(toy, 0)``.
    ds = tmp_path / "collide"
    (ds / "corpus").mkdir(parents=True)
    (ds / "corpus" / "toy.md").write_text(
        "# Toy\n\n"
        "## Section A\n\nTarget: toy.a\n\nApples are red.\n\n"
        "## Section B\n\nTarget: toy.b\n\nBananas are yellow.\n",
        encoding="utf-8",
    )
    (ds / "dataset.yaml").write_text("name: collide\nthresholds: {}\n", encoding="utf-8")
    (ds / "targets.yaml").write_text(
        "assets: []\nchunks:\n"
        "  - id: toy.a.text\n    doc: toy\n    anchor: toy.a\n"
        "  - id: toy.b.text\n    doc: toy\n    anchor: toy.b\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n  - q: anything\n    expect_any: [toy]\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    with pytest.raises(EvalError, match="collision"):
        await run_eval(spec)


@pytest.mark.asyncio
async def test_run_eval_thresholds_namespaced_chunk_view_gates_pass_fail(
    tmp_path: Path,
) -> None:
    """Setting ``chunk/hit_at_3`` in dataset.yaml gates ``passed`` against
    the chunk view's metric, independent of the doc-view threshold.
    """
    ds = _write_mm_dataset_with_targets(tmp_path)
    # ``passed`` is computed against ``self.metrics``; setting a threshold
    # above any achievable score (max metric value is 1.0) forces the
    # gate to fail. ``_parse_thresholds`` doesn't bound > 1.0 because
    # the runner deliberately stays metric-agnostic.
    (ds / "dataset.yaml").write_text(
        "name: mm-mini\nthresholds:\n  chunk/hit_at_3: 1.5\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    report = await run_eval(spec)
    # Chunk threshold is unattainable → passed must be False even if
    # the doc view is perfect.
    assert report.passed is False


# ---- Slice 6: eval-snapshot cache (D1-D5) -------------------------------


async def test_corpus_cache_key_is_deterministic_and_path_stable(
    tmp_path: Path,
) -> None:
    """D1: same corpus → same key; touching content → key changes; key
    uses POSIX paths so Windows + Linux yield the same digest.
    """
    from dikw_core.eval.runner import _corpus_cache_key

    ds = _write_dataset(
        tmp_path,
        queries=[("foo", ["alpha"])],
    )
    spec = load_dataset(ds)
    k1 = _corpus_cache_key(spec, "fake", 64, cjk_tokenizer="jieba")
    k2 = _corpus_cache_key(spec, "fake", 64, cjk_tokenizer="jieba")
    assert k1 == k2

    # Mutate one file's bytes — key must change.
    (ds / "corpus" / "alpha.md").write_text(
        "# Alpha\n\nDifferent body now.\n", encoding="utf-8"
    )
    k3 = _corpus_cache_key(spec, "fake", 64, cjk_tokenizer="jieba")
    assert k3 != k1

    # Different model → different key.
    k4 = _corpus_cache_key(spec, "other-model", 64, cjk_tokenizer="jieba")
    assert k4 != k1
    # Format sanity.
    assert k1.startswith("toy/fake__64__")
    assert "__sf" in k1

    # Different multimodal fingerprint → different key (text-only and
    # multimodal eval runs against the same corpus must NOT share a
    # snapshot — the asset index lives in different vec tables).
    k5 = _corpus_cache_key(
        spec, "fake", 64, cjk_tokenizer="jieba", mm_fingerprint="qwen3-vl-8b@4096"
    )
    assert k5 != k1
    assert "__mmqwen3-vl-8b@4096__" in k5

    # #250: cjk_tokenizer is the one INGEST-TIME RetrievalConfig field — it's
    # baked into the FTS index, so it must be in the key. Changing it must
    # produce a distinct key (a fresh snapshot), never a stale reuse.
    k6 = _corpus_cache_key(spec, "fake", 64, cjk_tokenizer="none")
    assert k6 != k1
    assert "__cjkjieba__" in k1
    assert "__cjknone__" in k6


async def test_eval_snapshot_cache_hit_skips_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D2: second run_eval against the same spec skips api.ingest.

    Spy on ``api.ingest`` via monkeypatch; expect call_count == 1
    across two run_eval calls (cold then warm).
    """
    from dikw_core import api as api_mod
    from dikw_core.eval import runner as runner_mod

    ds = _write_dataset(
        tmp_path / "ds",
        queries=[("alpha", ["alpha"])],
    )
    spec = load_dataset(ds)
    cache_root = tmp_path / "cache"

    ingest_calls = 0
    real_ingest = api_mod.ingest

    async def spy_ingest(*args: object, **kwargs: object) -> object:
        nonlocal ingest_calls
        ingest_calls += 1
        return await real_ingest(*args, **kwargs)

    monkeypatch.setattr(runner_mod.api, "ingest", spy_ingest)

    await run_eval(spec, cache_root=cache_root)
    assert ingest_calls == 1
    await run_eval(spec, cache_root=cache_root)
    assert ingest_calls == 1, "warm run must not re-ingest"
    # Snapshot dir landed at the expected location.
    snapshot_dirs = list((cache_root / "toy").glob("fake__*"))
    assert len(snapshot_dirs) == 1
    assert (snapshot_dirs[0] / "base" / ".dikw" / "index.sqlite").exists()


async def test_eval_cache_mode_rebuild_forces_cold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3: cache_mode='rebuild' deletes any existing snapshot and rebuilds."""
    from dikw_core import api as api_mod
    from dikw_core.eval import runner as runner_mod

    ds = _write_dataset(
        tmp_path / "ds",
        queries=[("alpha", ["alpha"])],
    )
    spec = load_dataset(ds)
    cache_root = tmp_path / "cache"

    ingest_calls = 0
    real_ingest = api_mod.ingest

    async def spy_ingest(*args: object, **kwargs: object) -> object:
        nonlocal ingest_calls
        ingest_calls += 1
        return await real_ingest(*args, **kwargs)

    monkeypatch.setattr(runner_mod.api, "ingest", spy_ingest)

    await run_eval(spec, cache_root=cache_root)  # cold
    assert ingest_calls == 1
    await run_eval(spec, cache_root=cache_root, cache_mode="rebuild")  # rebuild
    assert ingest_calls == 2, "rebuild must re-ingest"
    await run_eval(spec, cache_root=cache_root)  # warm again
    assert ingest_calls == 2, "post-rebuild warm hits the new snapshot"


async def test_eval_cache_mode_off_neither_reads_nor_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D4: cache_mode='off' uses tempdir (legacy path); never touches cache_root."""
    from dikw_core import api as api_mod
    from dikw_core.eval import runner as runner_mod

    ds = _write_dataset(
        tmp_path / "ds",
        queries=[("alpha", ["alpha"])],
    )
    spec = load_dataset(ds)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()  # exists, but empty
    assert list(cache_root.iterdir()) == []

    ingest_calls = 0
    real_ingest = api_mod.ingest

    async def spy_ingest(*args: object, **kwargs: object) -> object:
        nonlocal ingest_calls
        ingest_calls += 1
        return await real_ingest(*args, **kwargs)

    monkeypatch.setattr(runner_mod.api, "ingest", spy_ingest)

    await run_eval(spec, cache_root=cache_root, cache_mode="off")
    await run_eval(spec, cache_root=cache_root, cache_mode="off")
    assert ingest_calls == 2, "off mode must ingest every run"
    # cache_root never touched.
    assert list(cache_root.iterdir()) == []


async def test_eval_cache_mid_build_crash_leaves_partial_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D5: when api.ingest raises mid-build, .partial/ survives, <key>/ does not."""
    from dikw_core.eval import runner as runner_mod

    ds = _write_dataset(
        tmp_path / "ds",
        queries=[("alpha", ["alpha"])],
    )
    spec = load_dataset(ds)
    cache_root = tmp_path / "cache"

    async def boom_ingest(*args: object, **kwargs: object) -> object:
        raise RuntimeError("simulated mid-build crash")

    monkeypatch.setattr(runner_mod.api, "ingest", boom_ingest)

    with pytest.raises(RuntimeError, match="simulated mid-build crash"):
        await run_eval(spec, cache_root=cache_root)

    # Final snapshot dir must NOT exist (atomic rename never happened).
    final_dirs = list((cache_root / "toy").glob("fake__*"))
    final_dirs = [d for d in final_dirs if not d.name.endswith(".partial")]
    assert final_dirs == [], f"partial state leaked into final cache: {final_dirs}"
    # .partial/ should still be there for diagnostics.
    partial_dirs = list((cache_root / "toy").glob("fake__*.partial"))
    assert len(partial_dirs) == 1


async def test_eval_cache_search_time_config_read_live_on_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#250: a SEARCH-TIME retrieval knob (rrf_k) changed under read_write takes
    effect on a cache HIT — the searcher is built from the caller's LIVE config,
    not the baked snapshot — and the change does NOT re-ingest (fast + correct).
    """
    from dikw_core.config import RetrievalConfig
    from dikw_core.eval import runner as runner_mod

    ds = _write_dataset(tmp_path / "ds", queries=[("alpha", ["alpha"])])
    spec = load_dataset(ds)
    cache_root = tmp_path / "cache"

    ingest_calls = 0
    real_ingest = runner_mod.api.ingest

    async def spy_ingest(*args: object, **kwargs: object) -> object:
        nonlocal ingest_calls
        ingest_calls += 1
        return await real_ingest(*args, **kwargs)

    monkeypatch.setattr(runner_mod.api, "ingest", spy_ingest)

    # Capture the rrf_k the searcher is actually built with each run. The
    # ``from_config`` classmethod, set as a plain attr on the class, is called
    # unbound (``HybridSearcher.from_config(storage, embedder, cfg, ...)``), so
    # the spy sees the same positional args.
    seen_rrf_k: list[int] = []
    real_from_config = runner_mod.HybridSearcher.from_config

    def spy_from_config(storage, embedder, retrieval_cfg, **kwargs):  # type: ignore[no-untyped-def]
        seen_rrf_k.append(retrieval_cfg.rrf_k)
        return real_from_config(storage, embedder, retrieval_cfg, **kwargs)

    monkeypatch.setattr(runner_mod.HybridSearcher, "from_config", spy_from_config)

    await run_eval(
        spec, cache_root=cache_root, retrieval_config=RetrievalConfig(rrf_k=60)
    )
    await run_eval(
        spec, cache_root=cache_root, retrieval_config=RetrievalConfig(rrf_k=11)
    )

    assert ingest_calls == 1, "search-time change must reuse the snapshot (no re-ingest)"
    assert seen_rrf_k == [60, 11], (
        "warm run must build the searcher from the LIVE rrf_k (11), not the "
        f"baked 60; saw {seen_rrf_k}"
    )


async def test_eval_cache_cjk_tokenizer_change_forces_reingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#250: cjk_tokenizer is INGEST-TIME — it's pinned into the cache key, so
    changing it produces a fresh snapshot (re-ingest), never a stale reuse of an
    index built with the old tokenizer.
    """
    from dikw_core.config import RetrievalConfig
    from dikw_core.eval import runner as runner_mod

    ds = _write_dataset(tmp_path / "ds", queries=[("alpha", ["alpha"])])
    spec = load_dataset(ds)
    cache_root = tmp_path / "cache"

    ingest_calls = 0
    real_ingest = runner_mod.api.ingest

    async def spy_ingest(*args: object, **kwargs: object) -> object:
        nonlocal ingest_calls
        ingest_calls += 1
        return await real_ingest(*args, **kwargs)

    monkeypatch.setattr(runner_mod.api, "ingest", spy_ingest)

    await run_eval(
        spec, cache_root=cache_root, retrieval_config=RetrievalConfig(cjk_tokenizer="jieba")
    )
    assert ingest_calls == 1
    await run_eval(
        spec, cache_root=cache_root, retrieval_config=RetrievalConfig(cjk_tokenizer="none")
    )
    assert ingest_calls == 2, "ingest-time tokenizer change must re-ingest"
    # Two distinct snapshots coexist under the dataset's cache subdir.
    snapshot_dirs = [
        d for d in (cache_root / "toy").glob("fake__*") if not d.name.endswith(".partial")
    ]
    assert len(snapshot_dirs) == 2


def test_retrieval_config_fields_are_classified_for_eval_cache() -> None:
    """#250 tripwire: the eval snapshot cache hand-splits ``RetrievalConfig`` into
    ONE ingest-time field (``cjk_tokenizer`` — baked into the FTS index, so keyed
    in ``_corpus_cache_key``) and the rest search-time (read live in
    ``_run_queries``, never keyed). That split is encoded by hand, so adding a new
    ``RetrievalConfig`` field would silently default it to "search-time, read
    live" — which is WRONG and reintroduces the #250 stale-snapshot bug if the new
    field actually affects ingest. This test fails on any field-set change, forcing
    the contributor to classify the new field:
      * ingest-time (changes chunking / the FTS index / stored vectors) → add it
        to ``_corpus_cache_key`` (and the ``cjk_tokenizer`` guard in
        ``_run_queries``);
      * search-time (query-time ranking only) → it is already read live; no key
        change needed.
    """
    from dikw_core.config import RetrievalConfig

    ingest_time = {"cjk_tokenizer"}
    search_time = {
        "rrf_k",
        "bm25_weight",
        "vector_weight",
        "fusion",
        "same_doc_penalty_alpha",
        "graph_enabled",
        "graph_seed_top_k",
        "graph_weight",
        "rerank_enabled",
        "rerank_candidate_k",
    }
    assert set(RetrievalConfig.model_fields) == ingest_time | search_time, (
        "RetrievalConfig fields changed — classify the new/removed field for the "
        "eval snapshot cache (src/dikw_core/eval/runner.py): ingest-time → add to "
        "_corpus_cache_key (+ the cjk_tokenizer assert in _run_queries); "
        "search-time → it is read live, no key change needed. See #250."
    )


async def test_run_queries_guards_against_tokenizer_mismatch(
    tmp_path: Path,
) -> None:
    """#250 defence-in-depth: if a future ``_corpus_cache_key`` regression ever let
    a snapshot built with one ``cjk_tokenizer`` be reused under a different live
    tokenizer, ``_run_queries`` fails loud rather than querying an FTS index built
    with the wrong tokenizer. The cache key makes this impossible in normal
    operation, so it's exercised here by calling ``_run_queries`` directly on a
    jieba-built snapshot with a mismatched live ``cjk_tokenizer``.
    """
    from dikw_core.config import ProviderConfig, RetrievalConfig
    from dikw_core.eval.fake_embedder import FakeEmbeddings
    from dikw_core.eval.runner import _run_queries

    ds = _write_dataset(tmp_path / "ds", queries=[("alpha", ["alpha"])])
    spec = load_dataset(ds)
    cache_root = tmp_path / "cache"
    await run_eval(
        spec, cache_root=cache_root, retrieval_config=RetrievalConfig(cjk_tokenizer="jieba")
    )
    base = next((cache_root / "toy").glob("fake__*")) / "base"
    assert base.is_dir()

    provider = ProviderConfig(
        llm_api_key_env="ANTHROPIC_API_KEY",
        embedding_model="fake",
        embedding_dim=64,
        embedding_revision="",
        embedding_normalize=True,
        embedding_distance="cosine",
        embedding_api_key_env="OPENAI_API_KEY",
    )
    with pytest.raises(EvalError, match="snapshot cjk_tokenizer"):
        await _run_queries(
            base,
            spec,
            embedder=FakeEmbeddings(),
            embedding_model="fake",
            modes=["hybrid"],
            retrieval_config=RetrievalConfig(cjk_tokenizer="none"),
            provider_config=provider,
        )

# ---- check_thresholds direction-aware ---------------------------------------


def test_check_thresholds_min_direction_pass() -> None:
    results = check_thresholds(
        metrics={'synth/fact_grounding_ratio': 0.85},
        thresholds={'synth/fact_grounding_ratio': 0.80},
    )
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, ThresholdResult)
    assert r.direction == 'min'
    assert r.passed is True


def test_check_thresholds_min_direction_fail() -> None:
    results = check_thresholds(
        metrics={'synth/fact_grounding_ratio': 0.75},
        thresholds={'synth/fact_grounding_ratio': 0.80},
    )
    assert results[0].direction == 'min'
    assert results[0].passed is False


def test_check_thresholds_max_direction_pass() -> None:
    # _max suffix → lower is better
    results = check_thresholds(
        metrics={'synth/duplicate_ratio_max': 0.03},
        thresholds={'synth/duplicate_ratio_max': 0.05},
    )
    assert results[0].direction == 'max'
    assert results[0].passed is True


def test_check_thresholds_max_direction_fail() -> None:
    results = check_thresholds(
        metrics={'synth/duplicate_ratio_max': 0.10},
        thresholds={'synth/duplicate_ratio_max': 0.05},
    )
    assert results[0].direction == 'max'
    assert results[0].passed is False


def test_check_thresholds_missing_metric_marks_observed_none() -> None:
    """Missing observation must use ``None``, not ``NaN`` — NaN serialises
    as the literal token ``NaN`` in JSON, which strict parsers reject
    and Postgres JSONB refuses to store."""
    results = check_thresholds(
        metrics={},
        thresholds={'synth/atomicity_score': 0.90},
    )
    r = results[0]
    assert r.observed is None
    assert r.passed is False


def test_check_thresholds_namespaced_max_suffix_still_classified() -> None:
    # The <view>/<metric> prefix doesn't change direction parsing.
    results = check_thresholds(
        metrics={'doc/something_max': 1.5},
        thresholds={'doc/something_max': 1.0},
    )
    assert results[0].direction == 'max'
    assert results[0].passed is False


def test_synth_eval_error_is_eval_error_subclass() -> None:
    # CLI/server can catch the broader EvalError; K-layer-specific
    # handlers catch SynthEvalError.
    assert issubclass(SynthEvalError, EvalError)

