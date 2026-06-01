"""Dataset loader — the convention-over-configuration contract.

Each dataset is a directory with three things: ``dataset.yaml``, ``corpus/``,
``queries.yaml``. ``load_dataset`` accepts either a registered name (for
packaged datasets) or a directory path (user-provided); both resolve to the
same ``DatasetSpec`` shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core.eval.dataset import (
    DatasetError,
    DatasetSpec,
    Query,
    iter_packaged_datasets,
    load_dataset,
)


def _write_valid_dataset(root: Path, *, name: str = "toy") -> Path:
    """Build a minimal valid dataset directory; return its root."""
    ds = root / name
    (ds / "corpus").mkdir(parents=True, exist_ok=True)
    (ds / "corpus" / "alpha.md").write_text("# Alpha\n\nabout alpha.\n", encoding="utf-8")
    (ds / "corpus" / "beta.md").write_text("# Beta\n\nabout beta.\n", encoding="utf-8")
    (ds / "dataset.yaml").write_text(
        "name: toy\n"
        "description: two-doc toy dataset\n"
        "thresholds:\n"
        "  hit_at_3: 0.5\n"
        "  hit_at_10: 0.5\n"
        "  mrr: 0.3\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: what is alpha\n"
        "    expect_any: [alpha]\n"
        "  - q: what is beta\n"
        "    expect_any: [beta]\n",
        encoding="utf-8",
    )
    return ds


def test_iter_packaged_datasets_returns_known_names() -> None:
    """The packaged datasets directory ships at least one dataset; the
    enumerator powers ``dikw client eval`` (no-arg) on the server."""
    names = iter_packaged_datasets()
    # Known packaged datasets at the time of writing (kept loose so a
    # future addition doesn't break the assertion).
    assert "scifact" in names or "mvp" in names or "cmteb-t2-subset" in names


def test_load_dataset_from_path_returns_populated_spec(tmp_path: Path) -> None:
    ds_dir = _write_valid_dataset(tmp_path)
    spec = load_dataset(ds_dir)
    assert isinstance(spec, DatasetSpec)
    assert spec.name == "toy"
    assert spec.description == "two-doc toy dataset"
    assert spec.thresholds == {"hit_at_3": 0.5, "hit_at_10": 0.5, "mrr": 0.3}
    assert spec.corpus_dir == ds_dir / "corpus"
    assert len(spec.queries) == 2
    assert isinstance(spec.queries[0], Query)
    assert spec.queries[0].q == "what is alpha"
    assert spec.queries[0].expect_any == ["alpha"]


def test_load_dataset_missing_dataset_yaml_raises(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    (ds / "dataset.yaml").unlink()
    with pytest.raises(DatasetError, match=r"dataset\.yaml"):
        load_dataset(ds)


def test_load_dataset_missing_corpus_dir_raises(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    # remove corpus contents then the dir
    for f in (ds / "corpus").iterdir():
        f.unlink()
    (ds / "corpus").rmdir()
    with pytest.raises(DatasetError, match="corpus"):
        load_dataset(ds)


def test_load_dataset_empty_corpus_raises(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    for f in (ds / "corpus").iterdir():
        f.unlink()
    with pytest.raises(DatasetError, match="empty"):
        load_dataset(ds)


def test_load_dataset_missing_queries_yaml_raises(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    (ds / "queries.yaml").unlink()
    with pytest.raises(DatasetError, match=r"queries\.yaml"):
        load_dataset(ds)


def test_load_dataset_empty_queries_raises(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    (ds / "queries.yaml").write_text("queries: []\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="at least one query"):
        load_dataset(ds)


def test_load_dataset_rejects_unknown_threshold_key(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    (ds / "dataset.yaml").write_text(
        "name: toy\n"
        "description: bad thresholds\n"
        "thresholds:\n"
        "  hit_at_99: 0.5\n",  # not a supported metric
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="hit_at_99"):
        load_dataset(ds)


def test_query_accepts_expect_none_with_empty_expect_any() -> None:
    """``expect_none: true`` marks an out-of-domain query — no stems expected."""
    q = Query(q="weather in Tokyo?", expect_none=True)
    assert q.expect_none is True
    assert q.expect_any == []


def test_query_rejects_both_expect_any_and_expect_none() -> None:
    """Polarity must be unambiguous — can't claim both 'some doc' and 'no doc'."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        Query(q="ambiguous", expect_any=["foo"], expect_none=True)


def test_query_rejects_both_fields_empty() -> None:
    """A query with neither polarity given is almost always a YAML typo."""
    with pytest.raises(ValueError, match=r"expect_any.*expect_none"):
        Query(q="nothing specified")


def test_load_dataset_parses_negative_query_yaml(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: what is alpha\n"
        "    expect_any: [alpha]\n"
        "  - q: weather in Tokyo?\n"
        "    expect_none: true\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    assert len(spec.queries) == 2
    assert spec.queries[0].expect_none is False
    assert spec.queries[1].expect_none is True
    assert spec.queries[1].expect_any == []


def test_load_dataset_rejects_mixed_polarity_query(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: muddled\n"
        "    expect_any: [alpha]\n"
        "    expect_none: true\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="mutually exclusive"):
        load_dataset(ds)


def test_load_dataset_by_name_finds_packaged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When given a plain name, load_dataset searches datasets_root().

    We redirect datasets_root() via monkeypatch so the test doesn't depend on
    whether evals/datasets/mvp/ exists yet (E5 will move real data there).
    """
    _write_valid_dataset(tmp_path, name="toy")
    monkeypatch.setattr("dikw_core.eval.dataset.datasets_root", lambda: tmp_path)
    spec = load_dataset("toy")
    assert spec.name == "toy"


def test_load_dataset_name_not_found_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dikw_core.eval.dataset.datasets_root", lambda: tmp_path)
    with pytest.raises(DatasetError, match="no such dataset"):
        load_dataset("does-not-exist")


# ---- 4-file contract: targets.yaml + chunk/asset views --------------------


def _write_mm_dataset(root: Path) -> Path:
    """Minimal 4-file (dataset+corpus+targets+queries) dataset for tests."""
    ds = root / "mm-toy"
    corpus = ds / "corpus"
    (corpus / "images").mkdir(parents=True, exist_ok=True)
    (corpus / "alpha.md").write_text(
        "# Alpha doc\n\n"
        "## 第一节 / Section One\n\nTarget: alpha.s1\n\n"
        "Section one talks about apples.\n\n"
        "## 第二节 / Section Two\n\nTarget: alpha.s2\n\n"
        "Section two covers oranges.\n",
        encoding="utf-8",
    )
    (corpus / "images" / "img-a.png").write_bytes(b"fake-png-bytes-A")
    (corpus / "images" / "img-b.png").write_bytes(b"fake-png-bytes-B")
    (ds / "dataset.yaml").write_text(
        "name: mm-toy\nthresholds:\n  hit_at_3: 0.5\n  chunk/hit_at_3: 0.4\n",
        encoding="utf-8",
    )
    (ds / "targets.yaml").write_text(
        "assets:\n"
        "  - id: alpha.s1.image\n"
        "    doc: alpha\n"
        "    path: images/img-a.png\n"
        "    heading: 'Section One'\n"
        "    anchor: alpha.s1\n"
        "  - id: alpha.s2.image\n"
        "    doc: alpha\n"
        "    path: images/img-b.png\n"
        "    heading: 'Section Two'\n"
        "    anchor: alpha.s2\n"
        "chunks:\n"
        "  - id: alpha.s1.text\n"
        "    doc: alpha\n"
        "    heading: 'Section One'\n"
        "    anchor: alpha.s1\n"
        "    asset_id: alpha.s1.image\n"
        "  - id: alpha.s2.text\n"
        "    doc: alpha\n"
        "    heading: 'Section Two'\n"
        "    anchor: alpha.s2\n"
        "    asset_id: alpha.s2.image\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - id: q-doc\n"
        "    q: what does alpha doc cover\n"
        "    expect_any: [alpha]\n"
        "  - id: q-chunk\n"
        "    query_type: text_chunk\n"
        "    q: section about apples\n"
        "    expect_any: [alpha]\n"
        "    expect_chunk_any: [alpha.s1.text]\n"
        "  - id: q-asset\n"
        "    query_type: asset\n"
        "    q: image of section two\n"
        "    expect_any: [alpha]\n"
        "    expect_asset_any: [alpha.s2.image]\n",
        encoding="utf-8",
    )
    return ds


def test_load_dataset_targets_yaml_optional_missing_ok(tmp_path: Path) -> None:
    """Datasets without targets.yaml load fine; targets is empty."""
    ds_dir = _write_valid_dataset(tmp_path)
    spec = load_dataset(ds_dir)
    assert spec.targets.assets == []
    assert spec.targets.chunks == []


def test_load_dataset_with_targets_parses_assets_and_chunks(
    tmp_path: Path,
) -> None:
    ds_dir = _write_mm_dataset(tmp_path)
    spec = load_dataset(ds_dir)
    assert [a.id for a in spec.targets.assets] == [
        "alpha.s1.image",
        "alpha.s2.image",
    ]
    assert [c.id for c in spec.targets.chunks] == [
        "alpha.s1.text",
        "alpha.s2.text",
    ]
    assert spec.targets.chunks[0].asset_id == "alpha.s1.image"
    # threshold key with view prefix accepted
    assert spec.thresholds["chunk/hit_at_3"] == 0.4


def test_query_expect_chunk_any_field() -> None:
    q = Query(
        q="x",
        expect_any=["doc1"],
        expect_chunk_any=["doc1.s1.text"],
    )
    assert q.expect_chunk_any == ["doc1.s1.text"]
    assert q.doc_positives == ["doc1"]


def test_query_expect_asset_any_field() -> None:
    q = Query(q="x", expect_asset_any=["a.png.image"])
    assert q.expect_asset_any == ["a.png.image"]
    # asset-only positive is valid
    assert q.doc_positives == []


def test_query_rejects_both_expect_any_and_expect_doc_any() -> None:
    """Legacy alias and explicit field set together is ambiguous."""
    with pytest.raises(ValueError, match="legacy alias"):
        Query(q="x", expect_any=["a"], expect_doc_any=["b"])


def test_query_negative_with_any_positive_field_rejected() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        Query(q="x", expect_none=True, expect_chunk_any=["c"])


def test_query_with_only_chunk_positives_is_valid() -> None:
    q = Query(q="x", expect_chunk_any=["c.x.text"])
    # Pure chunk-level positive — legacy doc field empty is fine.
    assert q.doc_positives == []
    assert q.expect_chunk_any == ["c.x.text"]


def test_thresholds_namespaced_view_key_unknown_view_raises(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    (ds / "dataset.yaml").write_text(
        "name: toy\nthresholds:\n  bogus/hit_at_3: 0.5\n", encoding="utf-8"
    )
    with pytest.raises(DatasetError, match="threshold view"):
        load_dataset(ds)


def test_thresholds_namespaced_view_unknown_metric_raises(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    (ds / "dataset.yaml").write_text(
        "name: toy\nthresholds:\n  chunk/bogus_metric: 0.5\n", encoding="utf-8"
    )
    with pytest.raises(DatasetError, match="not a retrieval metric"):
        load_dataset(ds)


def test_thresholds_synth_view_with_retrieval_metric_raises(tmp_path: Path) -> None:
    """``synth/hit_at_3`` is rejected — retrieval metrics under the synth
    view would silently become a never-computable miss."""
    ds = _write_valid_dataset(tmp_path)
    (ds / "dataset.yaml").write_text(
        "name: toy\nthresholds:\n  synth/hit_at_3: 0.5\n", encoding="utf-8"
    )
    with pytest.raises(DatasetError, match="not a synth metric"):
        load_dataset(ds)


def test_thresholds_bare_synth_metric_raises(tmp_path: Path) -> None:
    """``atomicity_score`` (bare) is rejected — without the ``synth/``
    prefix the synth runner's threshold filter would silently drop it."""
    ds = _write_valid_dataset(tmp_path)
    (ds / "dataset.yaml").write_text(
        "name: toy\nthresholds:\n  atomicity_score: 0.9\n", encoding="utf-8"
    )
    with pytest.raises(DatasetError, match="must use the 'synth/' prefix"):
        load_dataset(ds)


def test_thresholds_retrieval_view_with_synth_metric_raises(tmp_path: Path) -> None:
    """``chunk/atomicity_score`` is rejected — synth metrics belong only
    to the ``synth`` view."""
    ds = _write_valid_dataset(tmp_path)
    (ds / "dataset.yaml").write_text(
        "name: toy\nthresholds:\n  chunk/atomicity_score: 0.9\n", encoding="utf-8"
    )
    with pytest.raises(DatasetError, match="not a retrieval metric"):
        load_dataset(ds)


def test_thresholds_synth_page_density_rejected_as_informational(
    tmp_path: Path,
) -> None:
    """``synth/page_density`` is informational only — the runner never
    fills it into ``SynthEvalReport.metrics`` so any threshold for it
    would always fail with ``observed=None``. Reject at load with a
    clear "informational only" message rather than accepting a gate
    the runner can't satisfy."""
    ds = _write_valid_dataset(tmp_path)
    (ds / "dataset.yaml").write_text(
        "name: toy\nthresholds:\n  synth/page_density: 0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="informational only"):
        load_dataset(ds)


def test_targets_yaml_duplicate_asset_ids_rejected(tmp_path: Path) -> None:
    ds = _write_mm_dataset(tmp_path)
    (ds / "targets.yaml").write_text(
        "assets:\n"
        "  - id: dup.id\n    doc: alpha\n    path: images/img-a.png\n"
        "  - id: dup.id\n    doc: alpha\n    path: images/img-b.png\n"
        "chunks: []\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="duplicate asset"):
        load_dataset(ds)


def test_query_expect_chunk_any_unknown_target_rejected(tmp_path: Path) -> None:
    """A typo in queries.yaml's expect_chunk_any must surface at load
    time, not silently lower metrics. Mirrors the anchor-resolution
    loud-fail in the runner.
    """
    ds = _write_mm_dataset(tmp_path)
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - id: q-typo\n"
        "    q: bogus chunk reference\n"
        "    expect_any: [alpha]\n"
        "    expect_chunk_any: [alpha.s9.text]\n",  # not in targets.yaml
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="unknown chunk targets"):
        load_dataset(ds)


def test_query_expect_asset_any_unknown_target_rejected(tmp_path: Path) -> None:
    ds = _write_mm_dataset(tmp_path)
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - id: q-typo\n"
        "    q: bogus asset reference\n"
        "    expect_any: [alpha]\n"
        "    expect_asset_any: [alpha.s9.image]\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="unknown asset targets"):
        load_dataset(ds)


# ---- K-layer (synth) mode contract -----------------------------------------


def _write_synth_dataset(root: Path, *, name: str = "synth-toy") -> Path:
    """Dataset declaring synth mode + expected.yaml + judge section."""
    ds = root / name
    (ds / "corpus").mkdir(parents=True, exist_ok=True)
    (ds / "corpus" / "src1.md").write_text(
        "# Src One\n\nbody one.\n", encoding="utf-8"
    )
    (ds / "corpus" / "src2.md").write_text(
        "# Src Two\n\nbody two.\n", encoding="utf-8"
    )
    (ds / "dataset.yaml").write_text(
        "name: synth-toy\n"
        "modes: [retrieval, synth]\n"
        "thresholds:\n"
        "  hit_at_3: 0.5\n"
        "  synth/fact_grounding_ratio: 0.8\n"
        "  synth/atomicity_score: 0.9\n"
        "  synth/duplicate_ratio_max: 0.05\n"
        "  synth/wikilink_resolved_ratio: 0.85\n"
        "  synth/expected_coverage: 0.8\n"
        "  synth/language_fidelity: 0.95\n"
        "synth:\n"
        "  grounding_threshold: 0.7\n"
        "  duplicate_threshold: 0.9\n"
        "  categories: [entity, concept]\n"
        "judge:\n"
        "  model: claude-sonnet\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: about src one\n"
        "    expect_any: [src1]\n",
        encoding="utf-8",
    )
    (ds / "expected.yaml").write_text(
        "sources:\n"
        "  - path: src1.md\n"
        "    expected_titles: [Src One, Topic Alpha]\n"
        "    expected_keywords: [keyword1]\n"
        "  - path: src2.md\n"
        "    expected_titles: [Src Two]\n",
        encoding="utf-8",
    )
    return ds


def test_load_dataset_default_modes_is_retrieval(tmp_path: Path) -> None:
    """Existing retrieval-only datasets parse with modes = ['retrieval']."""
    ds = _write_valid_dataset(tmp_path)
    spec = load_dataset(ds)
    assert spec.modes == ["retrieval"]


def test_load_dataset_explicit_modes_list(tmp_path: Path) -> None:
    ds = _write_synth_dataset(tmp_path)
    spec = load_dataset(ds)
    assert spec.modes == ["retrieval", "synth"]


def test_load_dataset_synth_section_parsed(tmp_path: Path) -> None:
    ds = _write_synth_dataset(tmp_path)
    spec = load_dataset(ds)
    assert spec.synth.grounding_threshold == 0.7
    assert spec.synth.duplicate_threshold == 0.9
    # Bare-string categories stay back-compatible (coerced to path-only nodes).
    assert [c.path for c in spec.synth.categories] == ["entity", "concept"]
    assert all(c.desc == "" for c in spec.synth.categories)


def test_load_dataset_synth_categories_mapping_carries_desc(tmp_path: Path) -> None:
    """A dataset may declare ``categories`` as ``{path, desc}`` mappings so the
    synth eval feeds the LLM the same per-category guidance a production base
    gets (keeps the K-layer gate representative — review finding)."""
    ds = _write_synth_dataset(tmp_path)
    (ds / "dataset.yaml").write_text(
        (ds / "dataset.yaml").read_text(encoding="utf-8").replace(
            "  categories: [entity, concept]\n",
            "  categories:\n"
            "    - {path: entity, desc: A named thing.}\n"
            "    - {path: concept, desc: An idea or framework.}\n",
        ),
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    assert [c.path for c in spec.synth.categories] == ["entity", "concept"]
    assert spec.synth.categories[0].desc == "A named thing."
    assert spec.synth.categories[1].desc == "An idea or framework."


def test_load_dataset_judge_section_parsed(tmp_path: Path) -> None:
    ds = _write_synth_dataset(tmp_path)
    spec = load_dataset(ds)
    assert spec.judge.model == "claude-sonnet"


def test_load_dataset_judge_rejects_unknown_field(tmp_path: Path) -> None:
    """``judge.provider`` was rejected post-review — a typo or stale
    field shouldn't silently disappear with no judge override applied."""
    ds = _write_synth_dataset(tmp_path)
    (ds / "dataset.yaml").write_text(
        (ds / "dataset.yaml").read_text(encoding="utf-8").replace(
            "judge:\n  model: claude-sonnet\n",
            "judge:\n  provider: anthropic\n  model: claude-sonnet\n",
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError):
        load_dataset(ds)


def test_load_dataset_synth_section_defaults(tmp_path: Path) -> None:
    """When dataset.yaml omits synth:, defaults apply (back-compat with
    retrieval-only datasets)."""
    ds = _write_valid_dataset(tmp_path)
    spec = load_dataset(ds)
    # Default lowered 0.65 → 0.50 after the 2026-05-13 tau sweep
    # (BASELINES.md); 0.65 was tighter than the embedder's similarity
    # scale supports for natural-language claims.
    assert spec.synth.grounding_threshold == 0.50
    assert spec.synth.duplicate_threshold == 0.85
    # The default taxonomy mirrors config._default_categories — entity/concept/
    # note WITH their descriptions — so an omitted ``categories:`` still feeds
    # the synth LLM the production-default guidance (review finding).
    assert [c.path for c in spec.synth.categories] == ["entity", "concept", "note"]
    assert all(c.desc for c in spec.synth.categories)


def test_load_dataset_judge_section_defaults(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    spec = load_dataset(ds)
    assert spec.judge.model is None


def test_load_dataset_accepts_synth_namespaced_thresholds(tmp_path: Path) -> None:
    ds = _write_synth_dataset(tmp_path)
    spec = load_dataset(ds)
    assert spec.thresholds["synth/fact_grounding_ratio"] == 0.8
    assert spec.thresholds["synth/duplicate_ratio_max"] == 0.05
    assert spec.thresholds["synth/wikilink_resolved_ratio"] == 0.85


def test_load_dataset_rejects_unknown_synth_metric(tmp_path: Path) -> None:
    ds = _write_synth_dataset(tmp_path)
    (ds / "dataset.yaml").write_text(
        "name: synth-toy\n"
        "modes: [retrieval, synth]\n"
        "thresholds:\n"
        "  hit_at_3: 0.5\n"
        "  synth/bogus_metric: 0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="not a synth metric"):
        load_dataset(ds)


def test_load_dataset_rejects_unknown_mode(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    (ds / "dataset.yaml").write_text(
        "name: toy\n"
        "modes: [retrieval, bogus]\n"
        "thresholds:\n"
        "  hit_at_3: 0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="mode"):
        load_dataset(ds)


def test_load_dataset_rejects_empty_synth_categories(tmp_path: Path) -> None:
    ds = _write_synth_dataset(tmp_path)
    (ds / "dataset.yaml").write_text(
        "name: synth-toy\n"
        "modes: [synth]\n"
        "thresholds:\n"
        "  synth/atomicity_score: 0.9\n"
        "synth:\n"
        "  categories: []\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="categories"):
        load_dataset(ds)


def test_load_dataset_rejects_synth_category_blank(tmp_path: Path) -> None:
    ds = _write_synth_dataset(tmp_path)
    (ds / "dataset.yaml").write_text(
        "name: synth-toy\n"
        "modes: [synth]\n"
        "thresholds:\n"
        "  synth/atomicity_score: 0.9\n"
        "synth:\n"
        "  categories: ['  ', 'concept']\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="categories"):
        load_dataset(ds)


def test_load_dataset_expected_yaml_missing_ok(tmp_path: Path) -> None:
    """expected.yaml is optional; spec.expected is None when absent."""
    ds = _write_valid_dataset(tmp_path)
    spec = load_dataset(ds)
    assert spec.expected is None


def test_load_dataset_expected_yaml_parsed(tmp_path: Path) -> None:
    ds = _write_synth_dataset(tmp_path)
    spec = load_dataset(ds)
    assert spec.expected is not None
    assert len(spec.expected.sources) == 2
    src1 = spec.expected.sources[0]
    assert src1.path == "src1.md"
    assert src1.expected_titles == ["Src One", "Topic Alpha"]
    assert src1.expected_keywords == ["keyword1"]
    assert spec.expected.sources[1].expected_keywords == []  # default empty


def test_load_dataset_expected_yaml_unknown_source_path_raises(
    tmp_path: Path,
) -> None:
    """A typo in expected.yaml's source path must loud-fail at load time,
    same as a typo in queries.yaml's expect_chunk_any."""
    ds = _write_synth_dataset(tmp_path)
    (ds / "expected.yaml").write_text(
        "sources:\n"
        "  - path: nonexistent.md\n"
        "    expected_titles: [X]\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="unknown source"):
        load_dataset(ds)


def test_load_dataset_synth_thresholds_namespaced_view_synth_accepted(
    tmp_path: Path,
) -> None:
    """The new 'synth' view name is in SUPPORTED_VIEWS so thresholds parse."""
    from dikw_core.eval.dataset import SUPPORTED_VIEWS

    assert "synth" in SUPPORTED_VIEWS
