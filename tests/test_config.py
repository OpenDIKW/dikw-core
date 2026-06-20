from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core.config import (
    CONFIG_FILENAME,
    CategoryNode,
    DikwConfig,
    LintConfig,
    PostgresStorageConfig,
    RetrievalConfig,
    SchemaConfig,
    SQLiteStorageConfig,
    TelemetryConfig,
    default_config,
    dump_config_yaml,
    find_config,
    load_config,
)

from .fakes import make_provider_cfg as ProviderConfig


def test_default_config_roundtrip(tmp_path: Path) -> None:
    cfg = default_config(description="unit-test wiki")
    yaml_text = dump_config_yaml(cfg)
    path = tmp_path / CONFIG_FILENAME
    path.write_text(yaml_text, encoding="utf-8")

    loaded = load_config(path)
    assert isinstance(loaded, DikwConfig)
    assert loaded.schema_.description == "unit-test wiki"
    assert isinstance(loaded.storage, SQLiteStorageConfig)
    assert loaded.storage.backend == "sqlite"


def test_load_config_discriminated_storage(tmp_path: Path) -> None:
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  llm: anthropic_compat
  llm_model: claude-sonnet-4-6
  embedding: openai_compat
  embedding_model: text-embedding-3-small
  embedding_base_url: https://example.invalid/v1
  embedding_dim: 1536
  embedding_revision: ''
  embedding_normalize: true
  embedding_distance: cosine
  llm_api_key_env: ANTHROPIC_API_KEY
  embedding_api_key_env: OPENAI_API_KEY
storage:
  backend: postgres
  dsn: postgresql://u:p@h:5432/db
  schema: dikw
  pool_size: 4
schema:
  description: pg wiki
sources:
  - path: ./sources
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert isinstance(cfg.storage, PostgresStorageConfig)
    assert cfg.storage.dsn.startswith("postgresql://")
    assert cfg.storage.schema_ == "dikw"


def test_find_config_walks_up(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    nested = root / "a" / "b" / "c"
    nested.mkdir(parents=True)
    (root / CONFIG_FILENAME).write_text(dump_config_yaml(default_config()), encoding="utf-8")

    found = find_config(nested)
    assert found is not None
    assert found.parent == root


def test_find_config_returns_none_when_missing(tmp_path: Path) -> None:
    assert find_config(tmp_path) is None


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / CONFIG_FILENAME
    path.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_provider_config_llm_max_tokens_defaults() -> None:
    """Per-op max_tokens default leaves headroom for a full fan-out group."""
    cfg = ProviderConfig()
    assert cfg.llm_max_tokens_synth == 3072


def test_synth_token_budget_covers_max_pages_per_group() -> None:
    """The default synth budget must leave at least ~512 tokens per page for a
    full fan-out group — otherwise a dense ``max_pages_per_group`` group clips
    mid-page. Guards against bumping ``max_pages_per_group`` without the budget.
    """
    from dikw_core.config import SynthConfig

    budget = ProviderConfig().llm_max_tokens_synth
    pages = SynthConfig().max_pages_per_group
    assert budget >= 512 * pages


def test_provider_config_llm_max_tokens_override_via_yaml(tmp_path: Path) -> None:
    """Users can shrink (or grow) per-op budgets via dikw.yml to fit their vendor."""
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  embedding_dim: 1536
  embedding_revision: ''
  embedding_normalize: true
  embedding_distance: cosine
  llm_api_key_env: ANTHROPIC_API_KEY
  embedding_api_key_env: OPENAI_API_KEY
  llm_max_tokens_synth: 4096
sources: []
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.provider.llm_max_tokens_synth == 4096


def test_provider_config_max_retries_defaults() -> None:
    """Both legs default to 5 retries — above the SDK default of 2 to give
    MiniMax 529 / Gemini 429 class errors a bit more breathing room.
    """
    cfg = ProviderConfig()
    assert cfg.llm_max_retries == 5
    assert cfg.embedding_max_retries == 5


def test_provider_config_max_retries_round_trip(tmp_path: Path) -> None:
    """Retry budgets are independently tunable per leg via dikw.yml."""
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  embedding_dim: 1536
  embedding_revision: ''
  embedding_normalize: true
  embedding_distance: cosine
  llm_api_key_env: ANTHROPIC_API_KEY
  embedding_api_key_env: OPENAI_API_KEY
  llm_max_retries: 3
  embedding_max_retries: 7
sources: []
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.provider.llm_max_retries == 3
    assert cfg.provider.embedding_max_retries == 7


def test_retrieval_config_defaults_are_scifact_tuned() -> None:
    """Defaults are the 2026-04-23 SciFact sweep winner, not equal weights.

    Equal (1.0, 1.0) starting point left hybrid 0.037 nDCG@10 behind the
    vector-only leg on BEIR/SciFact. Tuning to vector-heavy (0.3 / 1.5)
    at k=60 closes that gap — see ``evals/BASELINES.md`` for the sweep.
    This test pins those numbers so a silent drift doesn't reintroduce
    the regression.
    """
    cfg = RetrievalConfig()
    assert cfg.rrf_k == 60
    assert cfg.bm25_weight == 0.3
    assert cfg.vector_weight == 1.5


def test_dikw_config_retrieval_block_omitted_fills_defaults(tmp_path: Path) -> None:
    """A wiki whose dikw.yml predates this feature loads cleanly — the
    runtime supplies the SciFact-tuned defaults, not an error.
    """
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  llm: anthropic_compat
  embedding_dim: 1536
  embedding_revision: ''
  embedding_normalize: true
  embedding_distance: cosine
  llm_api_key_env: ANTHROPIC_API_KEY
  embedding_api_key_env: OPENAI_API_KEY
sources: []
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.retrieval.rrf_k == 60
    assert cfg.retrieval.bm25_weight == 0.3
    assert cfg.retrieval.vector_weight == 1.5


def test_dikw_config_retrieval_block_round_trip(tmp_path: Path) -> None:
    """Fusion knobs parse from YAML + survive dump → load."""
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  embedding_dim: 1536
  embedding_revision: ''
  embedding_normalize: true
  embedding_distance: cosine
  llm_api_key_env: ANTHROPIC_API_KEY
  embedding_api_key_env: OPENAI_API_KEY
retrieval:
  rrf_k: 40
  bm25_weight: 0.5
  vector_weight: 1.5
sources: []
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.retrieval.rrf_k == 40
    assert cfg.retrieval.bm25_weight == 0.5
    assert cfg.retrieval.vector_weight == 1.5

    # round-trip: dump → re-load yields identical values
    yaml_text = dump_config_yaml(cfg)
    path.write_text(yaml_text, encoding="utf-8")
    cfg2 = load_config(path)
    assert cfg2.retrieval.rrf_k == 40
    assert cfg2.retrieval.bm25_weight == 0.5
    assert cfg2.retrieval.vector_weight == 1.5


def test_retrieval_config_cjk_tokenizer_defaults_to_jieba() -> None:
    """Defaulting to ``jieba`` makes ``dikw client ingest`` correctly chunk and
    index Chinese content out of the box; ``has_cjk`` short-circuits
    ASCII inputs so all-ASCII corpora pay no segmentation cost. Users
    who want the legacy whitespace behaviour set ``cjk_tokenizer: none``
    explicitly.
    """
    cfg = RetrievalConfig()
    assert cfg.cjk_tokenizer == "jieba"


def test_retrieval_config_cjk_tokenizer_round_trips(tmp_path: Path) -> None:
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  embedding_dim: 1536
  embedding_revision: ''
  embedding_normalize: true
  embedding_distance: cosine
  llm_api_key_env: ANTHROPIC_API_KEY
  embedding_api_key_env: OPENAI_API_KEY
retrieval:
  cjk_tokenizer: jieba
sources: []
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.retrieval.cjk_tokenizer == "jieba"

    yaml_text = dump_config_yaml(cfg)
    path.write_text(yaml_text, encoding="utf-8")
    cfg2 = load_config(path)
    assert cfg2.retrieval.cjk_tokenizer == "jieba"


def test_retrieval_config_rejects_unknown_cjk_tokenizer(tmp_path: Path) -> None:
    """Guard against typos — ``trigram`` is tempting but not shipped."""
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  embedding_dim: 1536
  embedding_revision: ''
  embedding_normalize: true
  embedding_distance: cosine
  llm_api_key_env: ANTHROPIC_API_KEY
  embedding_api_key_env: OPENAI_API_KEY
retrieval:
  cjk_tokenizer: trigram
sources: []
""",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="cjk_tokenizer"):
        load_config(path)


# ---- knowledge taxonomy (categories) -----------------------------------


def test_schema_config_default_categories_are_entity_concept_note() -> None:
    """The default taxonomy preserves the historic page-type set so a fresh
    ``dikw init`` behaves as before — entity/concept/note as depth-1 categories.
    """
    cfg = SchemaConfig()
    assert [c.path for c in cfg.categories] == ["entity", "concept", "note"]
    # descriptions carry the synth-prompt semantics so default synth quality holds
    assert all(c.desc for c in cfg.categories)
    assert cfg.fallback == "未分类"
    # page_types / log_style are gone (clean break, no alias)
    assert not hasattr(cfg, "page_types")
    assert not hasattr(cfg, "log_style")


def test_schema_config_hierarchical_categories_round_trip(tmp_path: Path) -> None:
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  embedding_dim: 1536
  embedding_revision: ''
  embedding_normalize: true
  embedding_distance: cosine
  llm_api_key_env: ANTHROPIC_API_KEY
  embedding_api_key_env: OPENAI_API_KEY
schema:
  categories:
    - path: 产品/移动端
      desc: 移动端 App 产品
    - path: 技术/架构
  fallback: 待归档
sources: []
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert [c.path for c in cfg.schema_.categories] == ["产品/移动端", "技术/架构"]
    assert cfg.schema_.categories[0].desc == "移动端 App 产品"
    assert cfg.schema_.fallback == "待归档"
    # dump → reload is stable
    path.write_text(dump_config_yaml(cfg), encoding="utf-8")
    cfg2 = load_config(path)
    assert [c.path for c in cfg2.schema_.categories] == ["产品/移动端", "技术/架构"]
    assert cfg2.schema_.fallback == "待归档"


def test_schema_config_category_path_segments_validated(tmp_path: Path) -> None:
    """Each path segment must be filesystem-safe — traversal / absolute /
    backslash / reserved chars are rejected at config load (closed-set ⇒ this
    is the only place untrusted-ish path strings enter)."""
    for bad in ("产品/../etc", "/abs/path", "a\\b", "tech/da:ta", "a/ /b"):
        with pytest.raises(Exception, match="category path"):
            SchemaConfig(categories=[CategoryNode(path=bad)])


def test_schema_config_fallback_validated() -> None:
    with pytest.raises(Exception, match="category path"):
        SchemaConfig(fallback="../escape")


def test_schema_config_fallback_must_differ_from_declared_category() -> None:
    """The fallback bucket must be its own folder. If it coincides with a
    declared category path, synth files real and unplaceable pages into the
    same folder and the ``uncategorized`` lint flags every legitimately-filed
    page there — so reject the collision at config load (closed-set contract)."""
    with pytest.raises(Exception, match="must differ from every declared category"):
        SchemaConfig(categories=[CategoryNode(path="note")], fallback="note")


def test_synth_config_prompt_path_defaults_none_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  embedding_dim: 1536
  embedding_revision: ''
  embedding_normalize: true
  embedding_distance: cosine
  llm_api_key_env: ANTHROPIC_API_KEY
  embedding_api_key_env: OPENAI_API_KEY
synth:
  prompt_path: ./prompts/my_synth.md
sources: []
""",
        encoding="utf-8",
    )
    assert DikwConfig().synth.prompt_path is None
    cfg = load_config(path)
    assert cfg.synth.prompt_path == "./prompts/my_synth.md"


def test_lint_config_fixer_prompts_round_trip(tmp_path: Path) -> None:
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  embedding_dim: 1536
  embedding_revision: ''
  embedding_normalize: true
  embedding_distance: cosine
  llm_api_key_env: ANTHROPIC_API_KEY
  embedding_api_key_env: OPENAI_API_KEY
lint:
  fixer_prompts:
    orphan_merge: ./prompts/orphan.md
    broken_wikilink: ./prompts/bw.md
sources: []
""",
        encoding="utf-8",
    )
    assert DikwConfig().lint.fixer_prompts == {}
    cfg = load_config(path)
    assert cfg.lint.fixer_prompts["orphan_merge"] == "./prompts/orphan.md"
    assert cfg.lint.fixer_prompts["broken_wikilink"] == "./prompts/bw.md"


def test_lint_config_rejects_unknown_fixer_prompt_key() -> None:
    with pytest.raises(Exception, match="fixer_prompts"):
        LintConfig(fixer_prompts={"not_a_fixer": "./x.md"})


# ---- telemetry (OTel export config) ------------------------------------


def test_telemetry_config_defaults_are_off() -> None:
    """Default install is telemetry-off so a fresh ``dikw serve`` never tries
    to export to a non-existent collector."""
    cfg = TelemetryConfig()
    assert cfg.enabled is False
    assert cfg.endpoint is None
    assert cfg.service_name == "dikw-core"
    assert cfg.sample_ratio == 1.0


def test_dikw_config_telemetry_block_omitted_fills_defaults(tmp_path: Path) -> None:
    """A dikw.yml predating this feature loads cleanly with telemetry off."""
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  embedding_dim: 1536
  embedding_revision: ''
  embedding_normalize: true
  embedding_distance: cosine
  llm_api_key_env: ANTHROPIC_API_KEY
  embedding_api_key_env: OPENAI_API_KEY
sources: []
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.telemetry.enabled is False
    assert cfg.telemetry.service_name == "dikw-core"


def test_telemetry_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  embedding_dim: 1536
  embedding_revision: ''
  embedding_normalize: true
  embedding_distance: cosine
  llm_api_key_env: ANTHROPIC_API_KEY
  embedding_api_key_env: OPENAI_API_KEY
telemetry:
  enabled: true
  endpoint: http://collector:4318
  service_name: my-dikw
  sample_ratio: 0.25
sources: []
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.telemetry.enabled is True
    assert cfg.telemetry.endpoint == "http://collector:4318"
    assert cfg.telemetry.service_name == "my-dikw"
    assert cfg.telemetry.sample_ratio == 0.25

    # dump → reload is stable
    path.write_text(dump_config_yaml(cfg), encoding="utf-8")
    cfg2 = load_config(path)
    assert cfg2.telemetry.enabled is True
    assert cfg2.telemetry.endpoint == "http://collector:4318"
    assert cfg2.telemetry.sample_ratio == 0.25


def test_telemetry_config_rejects_out_of_range_sample_ratio() -> None:
    with pytest.raises(Exception, match="sample_ratio"):
        TelemetryConfig(sample_ratio=1.5)
