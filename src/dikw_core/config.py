"""Configuration loader for `dikw.yml`.

The config mirrors the top-level sections in the design doc: `provider`, `storage`,
`schema`, `sources`. Storage-specific fields live under a single `storage` block
and are validated per backend via a discriminated union.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .domains.info.tokenize import CjkTokenizer


class ProviderConfig(BaseModel):
    # ``anthropic_compat`` / ``openai_compat`` / ``openai_codex`` are protocol
    # names, not vendor names — pick the wire protocol the SDK speaks, then
    # pin the vendor via ``llm_base_url`` (e.g., ``anthropic_compat`` +
    # MiniMax's https://api.minimaxi.com/anthropic). ``openai_codex`` is
    # hard-bound to the ChatGPT backend (the validator below requires a
    # matching ``llm_base_url``); it speaks the OpenAI Responses API, not
    # Chat Completions, and rotates an OAuth access_token from
    # ``<wiki>/.dikw/auth.json`` (dikw self-managed; bootstrap with
    # ``dikw auth login openai-codex`` or ``dikw auth import openai-codex``)
    # instead of an ``OPENAI_API_KEY``. Defaults to ``anthropic_compat`` so a
    # fresh ``dikw init`` against api.anthropic.com is one key away.
    llm: Literal["anthropic_compat", "openai_compat", "openai_codex"] = "anthropic_compat"
    llm_model: str = "claude-sonnet-4-6"
    embedding: Literal["openai_compat"] = "openai_compat"
    embedding_model: str = "text-embedding-3-small"
    # The OpenAI-compat base URL is used for BOTH `openai_compat` LLM calls and
    # for embeddings when the LLM provider is anthropic_compat (which has no
    # embeddings API on the Anthropic protocol).
    embedding_base_url: str = "https://api.openai.com/v1"
    # The four fields below form the version identity registered into
    # ``embed_versions``. All required so dim/normalize/distance drift
    # is impossible to introduce silently (the version row is the
    # invariant the storage layer's per-version vec table relies on);
    # bump ``embedding_revision`` to force a new version when a vendor
    # silently refreshes weights behind a stable model name.
    embedding_dim: int
    embedding_revision: str
    embedding_normalize: bool
    embedding_distance: Literal["cosine", "l2", "dot"]
    # Max texts per ``/v1/embeddings`` request. OpenAI accepts ~2048;
    # Gitee AI caps at ~25. Keep the default safe for OpenAI and drop it
    # via config when hitting a stricter backend.
    embedding_batch_size: int = 64
    # Optional free-form display label surfaced in ``dikw check`` output
    # (e.g., "gitee-ai", "openai", "azure-east"). Describes which vendor
    # the embedding endpoint points at; purely for human diagnostics.
    embedding_provider_label: str | None = None
    # Used by both LLM protocols. For ``anthropic_compat``, retargets the
    # Anthropic SDK at any Anthropic-protocol-compatible endpoint (e.g.,
    # MiniMax's https://api.minimaxi.com/anthropic). Leave null to use the
    # SDK's default endpoint (api.anthropic.com / api.openai.com).
    llm_base_url: str | None = None
    # Per-operation response budget handed to ``LLMProvider.complete`` via
    # ``max_tokens``. Defaults match the values previously hardcoded in
    # ``api.py``; shrink for cost-optimised models (some GLM-Flash / Gemini
    # Nano variants cap below 2048), grow if synth/distill responses get
    # truncated.
    llm_max_tokens_query: int = 1024
    llm_max_tokens_synth: int = 2048
    llm_max_tokens_distill: int = 2048
    # Per-leg SDK retry budget. Anthropic and OpenAI SDKs retry 408/409/429/5xx
    # (incl. MiniMax 529) with exponential backoff + jitter; their default is
    # 2. We bump to 5 by default to absorb intermittent overload without
    # pulling in a third-party retry layer. Split per-leg because LLM and
    # embedding frequently target different vendors with different failure
    # profiles (e.g., MiniMax LLM + Gitee AI embeddings).
    llm_max_retries: int = 5
    embedding_max_retries: int = 5
    # Per-request timeout in seconds. The OpenAI/Anthropic SDKs default to
    # 600s, which lets a stale keepalive connection hang the whole pipeline
    # for 10 minutes before the SDK gives up and reconnects (observed
    # against Gitee AI mid-batch). Bound it tightly per-leg so a dead TCP
    # connection raises a timeout error fast and the SDK's retry path
    # establishes a fresh connection on the next attempt.
    llm_timeout_seconds: float = 120.0
    embedding_timeout_seconds: float = 60.0

    @model_validator(mode="after")
    def _require_codex_base_url(self) -> ProviderConfig:
        # ``openai_codex`` is the only protocol that has no SDK-default
        # endpoint — Codex models live on chatgpt.com/backend-api/codex
        # exclusively. Surface a missing or blank ``llm_base_url`` at
        # config-load time rather than at first ``complete()`` call so
        # ``dikw check`` fails fast and the error message tells the user
        # what to paste. ``None`` and empty/whitespace strings are both
        # rejected — yaml ``llm_base_url: ""`` would otherwise reach the
        # SDK as a malformed URL and surface as a low-level connection
        # error instead.
        if self.llm == "openai_codex" and (
            self.llm_base_url is None or not self.llm_base_url.strip()
        ):
            raise ValueError(
                "openai_codex requires llm_base_url to be set explicitly. "
                "Use https://chatgpt.com/backend-api/codex unless your "
                "deployment fronts a custom Codex-protocol gateway."
            )
        return self


class RetrievalConfig(BaseModel):
    """Fusion knobs for ``HybridSearcher``.

    Defaults are calibrated against BEIR/SciFact (2026-04-23 sweep,
    300 queries, Qwen3-Embedding-8B): the equal-weight ``(1.0, 1.0)``
    starting point left hybrid 0.037 nDCG@10 behind vector-only because
    RRF gave equal vote to a ~0.10-nDCG-weaker BM25 leg. Shifting to a
    vector-heavy ratio (BM25 0.3 / vector 1.5) closes that gap: hybrid
    lands at nDCG@10 ≈ 0.771 (≈ vector 0.773, well inside noise) while
    keeping hybrid's recall@100 advantage (0.970 vs 0.947 dense-only).

    Users whose corpus is **keyword-heavy** (code, identifiers, rare
    terminology) should raise ``bm25_weight`` back toward 1.0 — the
    SciFact tuning over-favours dense semantics and will under-rank
    exact-term matches. Tune per-corpus with
    ``dikw eval --retrieval all --dump-raw`` +
    ``evals/tools/sweep_rrf.py``. See ``evals/BASELINES.md`` for the
    full sweep table.
    """

    # Reciprocal Rank Fusion's rank-offset constant. Smaller = steeper
    # decay (rank-1 wins by more). 60 is the value used in the original
    # RRF paper and by both reference projects; the SciFact sweep finds
    # it near-optimal (k=40 scores 0.002 higher but the curve is flat
    # across 40/60/100, so keep the historical constant).
    rrf_k: int = 60
    # Per-leg contribution factor. Asymmetric because — see the class
    # docstring — the BM25 leg on BEIR-style corpora is measurably
    # behind the dense leg; equal weights drag the fused ranking toward
    # the weaker signal. A leg with weight 0.3 still has every doc it
    # alone found enter the pool (recall preserved); the weight only
    # scales how much that leg's rank order influences the top-k.
    bm25_weight: float = 0.3
    vector_weight: float = 1.5
    # Fusion algorithm. ``rrf`` (default) is rank-only and byte-identical
    # to pre-CombSUM baselines; ``combsum`` / ``combmnz`` consume raw
    # per-leg scores and preserve magnitude. See ``docs/providers.md`` →
    # "Score-normalised fusion alternatives" for when to reach for each.
    fusion: Literal["rrf", "combsum", "combmnz"] = "rrf"
    # Preprocesses CJK text with ``jieba`` before FTS5 indexing/querying
    # AND drives the chunker's token budget so long Chinese paragraphs
    # split. Required for Chinese corpora; ``unicode61`` otherwise splits
    # per-character and collapses BM25 to single-char IDF. ``jieba`` is
    # the default so ``dikw ingest`` does the right thing on Chinese
    # input without configuration; install via ``uv sync --extra cjk``
    # (or rely on the char-based fallback in ``count_tokens`` when the
    # extra is absent). **Locked at first ingest** — same shape as
    # ``embedding_dim``; flip requires wiping the index. Set to
    # ``"none"`` to opt back into the legacy whitespace behaviour. See
    # ``docs/providers.md`` gotcha #7 and ``evals/BASELINES.md``.
    cjk_tokenizer: CjkTokenizer = "jieba"
    # Diminishing-returns demotion for repeat same-doc chunks after
    # chunk-level RRF fusion. The 1st chunk per doc is unpenalized; the
    # N-th chunk is scaled by ``1 / (1 + alpha * (N - 1))``. Lightweight
    # source diversification (Stage 3 of the RAG retrieval stack); set
    # to ``0`` to disable, leave at ``0.3`` to soften same-book
    # dominance without hard-collapsing it. Tuned empirically per
    # corpus via Phase 3 dogfood (see plan A/B/baseline matrix).
    same_doc_penalty_alpha: float = Field(default=0.3, ge=0.0)


class SQLiteStorageConfig(BaseModel):
    backend: Literal["sqlite"] = "sqlite"
    path: str = ".dikw/index.sqlite"


class PostgresStorageConfig(BaseModel):
    backend: Literal["postgres"] = "postgres"
    dsn: str
    schema_: str = Field(default="dikw", alias="schema")
    pool_size: int = 10

    model_config = {"populate_by_name": True}


StorageConfig = Annotated[
    SQLiteStorageConfig | PostgresStorageConfig,
    Field(discriminator="backend"),
]


class SchemaConfig(BaseModel):
    description: str = ""
    page_types: list[str] = Field(default_factory=lambda: ["entity", "concept", "note"])
    wisdom_kinds: list[str] = Field(
        default_factory=lambda: ["principle", "lesson", "pattern"]
    )
    log_style: Literal["append", "daily"] = "append"


class SourceConfig(BaseModel):
    path: str
    pattern: str = "**/*.md"
    ignore: list[str] = Field(default_factory=list)


class SynthConfig(BaseModel):
    """Knobs for the K-layer ``dikw synth`` pipeline.

    Synth consumes the D-layer chunks already produced by ``ingest`` —
    ``target_tokens_per_group`` controls how many adjacent chunks the engine
    bundles into one LLM call (heading-aware), and ``max_pages_per_group``
    caps how many ``<page>`` blocks the LLM may emit per call. The page-type
    whitelist lives on ``SchemaConfig.page_types`` to avoid a second source
    of truth.
    """

    target_tokens_per_group: int = 3600
    max_pages_per_group: int = 4
    slug_dedup: Literal["merge_body", "keep_first"] = "merge_body"


class MultimodalEmbedConfig(BaseModel):
    """Native multimodal embedding configuration.

    When this section is present in ``dikw.yml`` the engine routes both
    chunk text and image bytes through the same multimodal model so they
    share one vector space. When absent, the engine stays in legacy
    text-only mode (text-embed for chunks, no asset retrieval).
    """

    provider: Literal["gitee_multimodal"] = "gitee_multimodal"
    model: str
    revision: str = ""  # bump to force a new version when weights change
    dim: int  # must match the model's actual output dim; vec table dim-locks on it
    normalize: bool = True
    distance: Literal["cosine", "l2", "dot"] = "cosine"
    batch: int = 16
    base_url: str | None = None  # override the provider's default endpoint


class AssetsConfig(BaseModel):
    """Multimedia asset materialization config."""

    dir: str = "assets"  # relative to project root
    multimodal: MultimodalEmbedConfig | None = None


def _default_provider_config() -> ProviderConfig:
    """``DikwConfig.provider`` factory — defaults to a text-embedding-3-small
    profile. ``ProviderConfig`` itself still requires the 4 embedding-identity
    fields explicitly so user-provided yml stays unambiguous; this factory
    exists so test fixtures and ``api.init_wiki`` can build a default
    ``DikwConfig`` without restating those values."""
    return ProviderConfig(
        embedding_dim=1536,
        embedding_revision="",
        embedding_normalize=True,
        embedding_distance="cosine",
    )


class DikwConfig(BaseModel):
    provider: ProviderConfig = Field(default_factory=_default_provider_config)
    storage: StorageConfig = Field(default_factory=SQLiteStorageConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    schema_: SchemaConfig = Field(default_factory=SchemaConfig, alias="schema")
    sources: list[SourceConfig] = Field(default_factory=list)
    assets: AssetsConfig = Field(default_factory=AssetsConfig)
    synth: SynthConfig = Field(default_factory=SynthConfig)

    model_config = {"populate_by_name": True}

    @field_validator("sources")
    @classmethod
    def _require_at_least_one_source_path(cls, v: list[SourceConfig]) -> list[SourceConfig]:
        # allow an empty list at init time (newly scaffolded wiki); engine-level
        # operations that need sources can validate at call time.
        return v


CONFIG_FILENAME = "dikw.yml"


def load_config(path: str | Path) -> DikwConfig:
    """Load and validate a `dikw.yml` file."""
    p = Path(path)
    if p.is_dir():
        p = p / CONFIG_FILENAME
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: top-level YAML must be a mapping, got {type(raw).__name__}")
    return DikwConfig.model_validate(raw)


def find_config(start: str | Path) -> Path | None:
    """Walk up from `start` looking for `dikw.yml`. Returns None if not found."""
    p = Path(start).resolve()
    for candidate in (p, *p.parents):
        cfg = candidate / CONFIG_FILENAME
        if cfg.is_file():
            return cfg
    return None


def default_config(description: str = "A dikw-core knowledge base") -> DikwConfig:
    """Return a DikwConfig populated with sensible defaults for `dikw init`.

    Ships one source entry covering markdown — the only built-in backend — so
    a fresh wiki picks up `sources/**/*.md` without extra config.
    """
    return DikwConfig(
        provider=ProviderConfig(
            embedding_dim=1536,  # text-embedding-3-small native
            embedding_revision="",
            embedding_normalize=True,
            embedding_distance="cosine",
        ),
        storage=SQLiteStorageConfig(),
        schema=SchemaConfig(description=description),
        sources=[
            SourceConfig(path="./sources", pattern="**/*.md"),
        ],
    )


def dump_config_yaml(cfg: DikwConfig) -> str:
    """Render a DikwConfig as a YAML string suitable for `dikw.yml`."""
    data = cfg.model_dump(mode="json", by_alias=True, exclude_defaults=False)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
