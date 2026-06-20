"""Configuration loader for `dikw.yml`.

The config mirrors the top-level sections in the design doc: `provider`, `storage`,
`schema`, `sources`. Storage-specific fields live under a single `storage` block
and are validated per backend via a discriminated union.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .domains.info.tokenize import CjkTokenizer

# Characters that are illegal in a path segment on at least one supported
# filesystem (Windows is the strictest). ``/`` is the taxonomy separator and
# is handled by splitting, so it is intentionally absent here.
_CATEGORY_RESERVED_CHARS = set('<>:"\\|?*')


def _validate_category_path(raw: str) -> str:
    """Validate + NFC-normalize a category folder path.

    A category path is a ``/``-separated, arbitrary-depth, base-relative
    folder path used verbatim as the on-disk directory under ``knowledge/``
    (e.g. ``产品/移动端``). Because the taxonomy is a *closed set* declared by
    the operator, this is the one place a folder-name string enters the
    engine — so it carries the full guard here rather than trusting LLM
    output downstream. Unicode is allowed; traversal / absolute / backslash /
    filesystem-reserved characters are not.
    """
    norm = unicodedata.normalize("NFC", raw).strip()
    if not norm:
        raise ValueError("category path must not be empty")
    if norm.startswith(("/", "\\")):
        raise ValueError(f"category path {raw!r} must be base-relative (no leading separator)")
    if "\\" in norm:
        raise ValueError(f"category path {raw!r} must use '/' separators, not backslash")
    segments = norm.split("/")
    for seg in segments:
        if not seg.strip():
            raise ValueError(f"category path {raw!r} has an empty segment")
        if seg.strip() != seg:
            raise ValueError(
                f"category path {raw!r} segment {seg!r} has leading/trailing whitespace"
            )
        if seg in (".", ".."):
            raise ValueError(f"category path {raw!r} must not contain '.' or '..' segments")
        if seg.endswith((".", " ")):
            raise ValueError(f"category path {raw!r} segment {seg!r} must not end with '.' or space")
        if any(ch in _CATEGORY_RESERVED_CHARS for ch in seg):
            raise ValueError(f"category path {raw!r} segment {seg!r} contains a reserved character")
        if any(ord(ch) < 32 for ch in seg):
            raise ValueError(f"category path {raw!r} segment {seg!r} contains a control character")
    return "/".join(segments)


class CategoryNode(BaseModel):
    """One node in the knowledge ``category`` taxonomy.

    ``path`` is the base-relative folder path under ``knowledge/`` (slash-
    separated, arbitrary depth); ``desc`` is optional guidance shown to the
    synth LLM so it can pick the right category. See
    ``docs/adr/0003-configurable-knowledge-taxonomy.md``.
    """

    path: str
    desc: str = ""

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        return _validate_category_path(v)


class ProviderConfig(BaseModel):
    # ``anthropic_compat`` / ``openai_compat`` / ``openai_codex`` are protocol
    # names, not vendor names — pick the wire protocol the SDK speaks, then
    # pin the vendor via ``llm_base_url`` (e.g., ``anthropic_compat`` +
    # MiniMax's https://api.minimaxi.com/anthropic). ``openai_codex`` is
    # hard-bound to the ChatGPT backend (the validator below requires a
    # matching ``llm_base_url``); it speaks the OpenAI Responses API, not
    # Chat Completions, and rotates an OAuth access_token from
    # ``<base>/.dikw/auth.json`` (dikw self-managed; bootstrap with
    # ``dikw auth login openai-codex`` or ``dikw auth import openai-codex``)
    # instead of an ``OPENAI_API_KEY``. Defaults to ``anthropic_compat`` so a
    # fresh ``dikw init`` against api.anthropic.com is one key away.
    llm: Literal["anthropic_compat", "openai_compat", "openai_codex"] = "anthropic_compat"
    llm_model: str = "claude-sonnet-4-6"
    # Name of the env var holding the LLM key. Vendor-canonical — e.g.
    # ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` / ``DEEPSEEK_API_KEY`` /
    # ``MINIMAX_API_KEY`` — so multiple same-protocol vendors (DeepSeek +
    # MiniMax both speak ``anthropic_compat``) coexist in one ``.env`` under
    # distinct names. Required: the engine never hardcodes a key var name.
    # Ignored by ``openai_codex`` (OAuth tokens live in
    # ``<base>/.dikw/auth.json``; there is no env API key).
    llm_api_key_env: str
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
    # Name of the env var holding the embedding key (vendor-canonical, e.g.
    # ``OPENAI_API_KEY`` / ``GITEE_API_KEY``). Required. LLM/embedding key
    # separation is achieved by naming distinct vars here — point both legs at
    # the same var to share one key, or at different vars to split vendors.
    embedding_api_key_env: str
    # Max texts per ``/v1/embeddings`` request. OpenAI accepts ~2048;
    # Gitee AI caps at ~25. Keep the default safe for OpenAI and drop it
    # via config when hitting a stricter backend.
    embedding_batch_size: int = 64
    # Optional free-form display label surfaced in ``dikw client check`` output
    # (e.g., "gitee-ai", "openai", "azure-east"). Describes which vendor
    # the embedding endpoint points at; purely for human diagnostics.
    embedding_provider_label: str | None = None
    # Used by both LLM protocols. For ``anthropic_compat``, retargets the
    # Anthropic SDK at any Anthropic-protocol-compatible endpoint (e.g.,
    # MiniMax's https://api.minimaxi.com/anthropic). Leave null to use the
    # SDK's default endpoint (api.anthropic.com / api.openai.com).
    llm_base_url: str | None = None
    # Per-operation response budget handed to ``LLMProvider.complete`` via
    # ``max_tokens``. 3072 leaves ~768 tokens per page at the default
    # ``max_pages_per_group=4``, so a full fan-out group rarely truncates
    # mid-page (the old 2048 default left only ~512/page and clipped dense
    # groups). Shrink for cost-optimised models (some GLM-Flash / Gemini Nano
    # variants cap below 2048); grow a lot for reasoning models, whose hidden
    # chain-of-thought also draws on this budget (MiniMax-M3 needs >= 8192,
    # 16384 recommended — see docs/providers.md).
    llm_max_tokens_synth: int = 3072
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
    # Per-batch ``ProviderError`` resilience. ``embedding_max_retries``
    # above is the SDK's own retry budget for transient HTTP failures;
    # the two fields below are a higher-level guard for what happens
    # when the SDK gives up. A batch that fails after exhausted SDK
    # retries is retried ``embedding_error_retries`` more times with
    # linear backoff (``embedding_error_retry_backoff_seconds`` * attempt)
    # and then SKIPPED — its chunks remain in storage without vectors
    # and get reconciled by the next ingest's missing-embedding resume
    # scan. A single bad batch can no longer abort a whole persist or
    # ingest run. ``embedding_error_retries=0`` means one attempt then
    # skip (no retries before the skip).
    embedding_error_retries: int = Field(default=2, ge=0)
    embedding_error_retry_backoff_seconds: float = Field(default=2.0, ge=0.0)

    @model_validator(mode="after")
    def _require_codex_base_url(self) -> ProviderConfig:
        # ``openai_codex`` is the only protocol that has no SDK-default
        # endpoint — Codex models live on chatgpt.com/backend-api/codex
        # exclusively. Surface a missing or blank ``llm_base_url`` at
        # config-load time rather than at first ``complete()`` call so
        # ``dikw client check`` fails fast and the error message tells the user
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
    exact-term matches. Tune per-corpus by editing the ``retrieval:``
    weights in ``dikw.yml`` and re-running ``dikw client eval
    --retrieval all`` to compare. See ``evals/BASELINES.md`` for the
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
    # the default so ``dikw client ingest`` does the right thing on Chinese
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
    # Wikilink-graph retrieval leg. When ``graph_enabled`` is True, the
    # searcher takes the top ``graph_seed_top_k`` chunks from the
    # BM25+vector fused result, asks storage for chunks reachable via
    # K-layer wikilinks, and folds them in as a fourth RRF leg with
    # ``graph_weight``. Default-off until eval evidence shows the leg
    # actually moves nDCG — wikilink graphs need to be dense enough for
    # one-hop neighbor expansion to be informative.
    #
    # ``graph_weight`` defaults to ``bm25_weight`` so a graph-only
    # neighbor never outranks an exact BM25 match by itself — the leg
    # augments rather than overpowers the lexical signal. Override to
    # raise/lower per corpus.
    graph_enabled: bool = False
    graph_seed_top_k: int = Field(default=20, ge=1)
    graph_weight: float = Field(default=0.3, ge=0.0)


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


def _default_categories() -> list[CategoryNode]:
    """The out-of-the-box taxonomy — the historic ``entity``/``concept``/``note``
    page types as depth-1 categories, carrying the synth-prompt semantics as
    ``desc`` so default synth quality is preserved."""
    return [
        CategoryNode(path="entity", desc="A named thing: person, tool, product, organization."),
        CategoryNode(path="concept", desc='An idea, framework, or pattern (e.g. "DIKW pyramid").'),
        CategoryNode(
            path="note",
            desc=(
                "An observation, lesson, or material card focused on a single subject; "
                "must reference at least one entity or concept via a [[Wikilink]]."
            ),
        ),
    ]


class SchemaConfig(BaseModel):
    description: str = ""
    # The knowledge classification taxonomy (replaces the pre-0.5.0
    # ``page_types`` flat list). A closed set of arbitrary-depth category
    # paths; synth files each page under one of them and falls back to
    # ``fallback`` when it can't confidently classify. See
    # ``docs/adr/0003-configurable-knowledge-taxonomy.md``.
    categories: list[CategoryNode] = Field(default_factory=_default_categories)
    # Bucket folder for pages synth cannot place in a declared category.
    fallback: str = "未分类"

    @field_validator("categories")
    @classmethod
    def _validate_categories(cls, v: list[CategoryNode]) -> list[CategoryNode]:
        if not v:
            raise ValueError("schema.categories must declare at least one category")
        seen: set[str] = set()
        for c in v:
            if c.path in seen:
                raise ValueError(f"schema.categories has a duplicate path: {c.path!r}")
            seen.add(c.path)
        return v

    @field_validator("fallback")
    @classmethod
    def _validate_fallback(cls, v: str) -> str:
        return _validate_category_path(v)

    @model_validator(mode="after")
    def _fallback_distinct_from_categories(self) -> SchemaConfig:
        # Closed-set invariant: the fallback bucket must be its own folder. If
        # it coincided with a declared category path, synth would file
        # correctly-classified and unplaceable pages into the same folder, and
        # ``run_lint``'s uncategorized detector (``category_from_path(path) ==
        # fallback``) would flag every legitimately-filed page there. Both
        # ``fallback`` and ``categories[].path`` are NFC-normalized by the field
        # validators above, so a direct comparison is sufficient.
        if self.fallback in self.category_paths():
            raise ValueError(
                f"schema.fallback {self.fallback!r} must differ from every "
                "declared category path — the fallback is a distinct bucket for "
                "pages synth cannot place; sharing a folder with a declared "
                "category breaks the closed-set / uncategorized-lint contract "
                "(ADR-0003)."
            )
        return self

    def category_paths(self) -> list[str]:
        """Declared category paths, in config order (excludes ``fallback``)."""
        return [c.path for c in self.categories]

    def categories_prompt_block(self) -> str:
        """Render the taxonomy as ``- `path` — desc`` bullets for the synth /
        lint-fixer prompts' ``{categories}`` slot (single source of truth so
        the LLM sees identical category guidance across all authoring paths)."""
        return "\n".join(
            f"- `{c.path}`" + (f" — {c.desc}" if c.desc else "") for c in self.categories
        )


class SourceConfig(BaseModel):
    """One ``sources`` entry in ``dikw.yml``.

    ``path`` is resolved against the base root (a relative path is anchored
    there; an absolute one is taken as-is) and **must stay under the base** —
    ``sources`` is a managed tree, so an escaping ``../`` prefix or an external
    absolute path is rejected at ingest time (``iter_source_files``).
    """

    path: str
    pattern: str = "**/*.md"
    ignore: list[str] = Field(default_factory=list)


class SynthConfig(BaseModel):
    """Knobs for the K-layer ``dikw client synth`` pipeline.

    Synth consumes the D-layer chunks already produced by ``ingest`` —
    ``target_tokens_per_group`` controls how many adjacent chunks the engine
    bundles into one LLM call (heading-aware), and ``max_pages_per_group``
    caps how many ``<page>`` blocks the LLM may emit per call. The category
    taxonomy lives on ``SchemaConfig.categories`` to avoid a second source
    of truth.
    """

    target_tokens_per_group: int = 3600
    max_pages_per_group: int = 4
    slug_dedup: Literal["merge_body", "keep_first"] = "merge_body"
    # Optional base-relative path to a markdown file that overrides the
    # packaged ``synthesize`` prompt template. Resolved against the base root
    # and validated (required ``{placeholders}`` + output markers) at load /
    # ``dikw client check``; ``None`` uses the packaged default. Also overrides
    # the prompt used by the ``non_atomic_page`` lint fixer, which shares the
    # synth template. See ``docs/adr/0003-configurable-knowledge-taxonomy.md``.
    prompt_path: str | None = None
    # Per-group prompt awareness of existing K-layer pages. Below the
    # byte threshold the prompt enumerates every page; above it,
    # truncation switches to a vec_search-gated top-K driven by the
    # group's own chunk embeddings. 16384 B ≈ 500 ``Title (type)``
    # bullets at ~25 B/line — a base much larger than that needs
    # retrieval to stay within the model's context window.
    existing_pages_max_bytes: int = 16384
    existing_pages_top_k: int = 50
    # Per-group ``ProviderError`` resilience. A single bad source group
    # (e.g. the openai_codex empty-response edge case in issue #134)
    # used to abort the whole synth task; now we retry the call up to
    # ``provider_error_retries`` times with linear backoff before
    # skipping the group and continuing. ``retries=0`` means "no
    # retries — one attempt, then skip on failure"; the synth task
    # never re-raises a per-group ProviderError, by design.
    provider_error_retries: int = Field(default=2, ge=0)
    provider_error_retry_backoff_seconds: float = Field(default=2.0, ge=0.0)
    # ``dikw client synth --verify`` semantic-duplicate gate. The post-synth
    # self-check embeds this run's page bodies and flags any pair whose cosine
    # is ``>= verify_duplicate_cosine_tau`` as a near-duplicate; the run fails
    # the duplicate leg when the flagged fraction exceeds
    # ``verify_max_duplicate_ratio``. Defaults mirror the synth-eval dataset
    # defaults (``duplicate_threshold=0.85``, ``synth/duplicate_ratio_max=0.05``)
    # so the interactive gate and the eval gate agree.
    verify_duplicate_cosine_tau: float = Field(default=0.85, ge=0.0, le=1.0)
    verify_max_duplicate_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    # ``dikw client synth --verify --judge`` grounding leg: how many of this
    # run's page claims the LLM entailment judge scores (seeded subset; a cap
    # ``>=`` the claim count judges them all). Default 25 mirrors
    # ``eval.judge.recommended_judge_sample()`` — the smallest sample whose
    # bootstrap 95% CI half-width clears ±0.2 for any [0, 1] ratio.
    verify_judge_sample: int = Field(default=25, ge=1)


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
    profile. ``ProviderConfig`` itself still requires the embedding-identity
    fields and the two ``*_api_key_env`` fields explicitly so user-provided yml
    stays unambiguous; this factory exists so test fixtures and ``api.init_base``
    can build a default ``DikwConfig`` without restating those values."""
    return ProviderConfig(
        llm_api_key_env="ANTHROPIC_API_KEY",
        embedding_dim=1536,
        embedding_revision="",
        embedding_normalize=True,
        embedding_distance="cosine",
        embedding_api_key_env="OPENAI_API_KEY",
    )


# Lint fixers whose prompt template a base may override via
# ``lint.fixer_prompts``. ``non_atomic_page`` is intentionally absent — it
# reuses the ``synthesize`` template, so it is overridden via
# ``synth.prompt_path`` instead.
_KNOWN_FIXER_PROMPTS = frozenset({"orphan_merge", "broken_wikilink"})


class LintConfig(BaseModel):
    """Knobs for the K-layer ``dikw client lint`` pipeline.

    ``fixer_prompts`` maps a lint-fixer key to a base-relative markdown path
    that overrides that fixer's packaged prompt template (validated like
    ``synth.prompt_path``). Unset keys use the packaged default.
    """

    fixer_prompts: dict[str, str] = Field(default_factory=dict)

    @field_validator("fixer_prompts")
    @classmethod
    def _validate_fixer_prompt_keys(cls, v: dict[str, str]) -> dict[str, str]:
        unknown = set(v) - _KNOWN_FIXER_PROMPTS
        if unknown:
            raise ValueError(
                f"lint.fixer_prompts has unknown key(s) {sorted(unknown)}; "
                f"overridable fixers are {sorted(_KNOWN_FIXER_PROMPTS)}"
            )
        return v


class TelemetryConfig(BaseModel):
    """OpenTelemetry export config for ``dikw serve`` (server-side).

    Read by the server lifespan from ``dikw.yml`` and handed to
    ``telemetry.configure_telemetry``. Requires the ``[otel]`` extra to be
    installed; without it every field is inert and telemetry stays no-op.

    Standard ``OTEL_*`` env vars are still honoured: ``OTEL_SDK_DISABLED``
    force-disables regardless of ``enabled``, and ``endpoint`` left null falls
    back to ``OTEL_EXPORTER_OTLP_ENDPOINT``. The remote client CLI
    (``dikw client …``) has no base config, so its telemetry — if wanted —
    is driven purely by ``OTEL_*`` env vars, not this section; see
    ``telemetry.configure_client_telemetry_from_env`` (``OTEL_SERVICE_NAME``
    default ``dikw-client``, gated on the ``[otel]`` extra) and the client
    section of ``docs/server.md``.
    """

    enabled: bool = False
    # OTLP/HTTP base URL (e.g. ``http://collector:4318``); when set, the
    # per-signal ``/v1/traces`` + ``/v1/metrics`` paths are appended for you.
    # Null → the SDK reads the standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var
    # (and appends the paths itself).
    endpoint: str | None = None
    service_name: str = "dikw-core"
    # ParentBased(TraceIdRatio) head sampling — traces only; metrics are not
    # sampled. 1.0 = sample everything.
    sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)


class DikwConfig(BaseModel):
    provider: ProviderConfig = Field(default_factory=_default_provider_config)
    storage: StorageConfig = Field(default_factory=SQLiteStorageConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    schema_: SchemaConfig = Field(default_factory=SchemaConfig, alias="schema")
    sources: list[SourceConfig] = Field(default_factory=list)
    assets: AssetsConfig = Field(default_factory=AssetsConfig)
    synth: SynthConfig = Field(default_factory=SynthConfig)
    lint: LintConfig = Field(default_factory=LintConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    model_config = {"populate_by_name": True}

    @field_validator("sources")
    @classmethod
    def _require_at_least_one_source_path(cls, v: list[SourceConfig]) -> list[SourceConfig]:
        # allow an empty list at init time (newly scaffolded base); engine-level
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
    a fresh knowledge base picks up `sources/**/*.md` without extra config.
    """
    return DikwConfig(
        provider=ProviderConfig(
            llm_api_key_env="ANTHROPIC_API_KEY",  # default LLM: Anthropic native
            embedding_dim=1536,  # text-embedding-3-small native
            embedding_revision="",
            embedding_normalize=True,
            embedding_distance="cosine",
            embedding_api_key_env="OPENAI_API_KEY",  # default embed: OpenAI
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
