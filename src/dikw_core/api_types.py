"""Cross-cutting DTOs and exceptions for the engine facade.

Kept dependency-light (stdlib + pydantic + ``config``) so every
``api_*`` cluster module can import the result/exception types it needs
without importing the ``api`` facade itself — that would be an import
cycle, since the facade imports the clusters. ``api`` re-exports every
public name here so the ``api.X`` surface is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from .config import MultimodalEmbedConfig


class PageNotFound(LookupError):
    """Raised by :func:`read_page` when the given path is not a registered
    document in the base. Path-escape attempts (``..``, files outside the
    base root) and unindexed files (``dikw.yml``) all surface here so the
    server route can map a single exception type to a uniform 404.
    """


class AssetNotFound(LookupError):
    """Raised by :func:`read_asset` for unknown id, vanished file, or
    ``stored_path`` escape (DB tampering / migration drift). Mirrors
    :class:`PageNotFound`: a single exception keeps the route's 404
    uniform so existing ids can't be probed.
    """


class BaseUpgradeRequired(RuntimeError):
    """Raised when a base predates a breaking on-disk / config change.

    Two shapes are flagged, both under the alpha rebuild-over-migrate
    policy (there is no in-place migration; the message carries the exact
    recipe). See ADR-0004 / ADR-0003.

    1. A base from dikw-core ≤0.3.6 whose K layer still lives under
       ``wiki/`` — the ``wiki/`` → ``knowledge/`` directory rename and the
       ``wiki_log`` → ``knowledge_log`` SQL-table rename.
    2. A pre-0.5.0 base whose ``dikw.yml`` carries ``schema.page_types``
       (replaced by the configurable ``schema.categories`` taxonomy; the
       K-layer frontmatter changed ``type:`` → ``category:``), so the base
       must be rebuilt under the new taxonomy.
    """


# ---- public result models ------------------------------------------------


IngestErrorKind = Literal["parse_error", "read_error", "storage_error"]


@dataclass(frozen=True)
class IngestError:
    """One per-file ingest failure surfaced on the report.

    Non-fatal by default — ingest continues with the next file so a
    single bad markdown doesn't kill a 1000-file run; the CLI's
    ``--strict`` flag opts into exit-on-error semantics.

    ``kind`` is a discriminator chosen so callers can branch without
    regex-matching ``message``: ``parse_error`` (parser rejected the
    content), ``read_error`` (filesystem refused the read), or
    ``storage_error`` (post-parse pipeline raised; the engine
    deactivates the doc so the next run re-processes it).
    Unsupported-extension files are silently skipped — the prior
    behaviour — to keep wide-glob configs from drowning the error
    channel.
    """

    path: str
    kind: IngestErrorKind
    message: str


@dataclass(frozen=True)
class IngestReport:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    chunks: int = 0
    embedded: int = 0
    assets: int = 0  # NEW assets materialized this run
    asset_embedded: int = 0  # asset-level vectors written this run
    errors: tuple[IngestError, ...] = ()


@dataclass(frozen=True)
class SynthReport:
    # ``candidates`` and ``skipped`` count *sources* (the unit synth iterates
    # at the outer level); ``created`` / ``updated`` / ``errors`` count
    # *pages* (the unit synth produces after fan-out + dedup). Mixing the
    # two units in one report would have been clearer if Stage A were
    # post-1.0 — pre-alpha just adds the per-unit ones explicitly.
    candidates: int = 0
    sources_processed: int = 0
    groups_processed: int = 0
    created: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0
    # Outgoing ``[[wikilinks]]`` from this run's pages that did not resolve
    # to any K-layer page (after exact + fuzzy normalize + collision
    # refusal). High counts signal LLM-generated references that nobody
    # has authored yet — actionable signal even before ``dikw client lint`` runs.
    unresolved_wikilinks: int = 0


class ProbeResult(BaseModel):
    """One leg of a ``check`` — either the LLM or the embedding endpoint."""

    ok: bool
    target: str  # the configured endpoint (or "(provider default)")
    detail: str  # on success: timing + basic stats; on failure: error message


class CheckReport(BaseModel):
    """Result of ``check_providers`` — per-leg connectivity probes plus
    per-base prompt-override validation.

    Either provider leg may be ``None`` when skipped via ``llm_only`` /
    ``embed_only``. ``prompts`` holds one entry per *configured* prompt
    override (``synth.prompt_path`` / ``lint.fixer_prompts``); empty when none
    are configured. ``ok`` is True when every *present* provider leg is ok (and
    at least one is present) and every configured prompt override is valid.
    """

    llm: ProbeResult | None = None
    embed: ProbeResult | None = None
    prompts: list[ProbeResult] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        legs = [p for p in (self.llm, self.embed) if p is not None]
        return (
            bool(legs)
            and all(p.ok for p in legs)
            and all(p.ok for p in self.prompts)
        )


# ---- /v1/health DTOs ----------------------------------------------------
#
# Surface what an agent needs to drive dikw without leaking what it
# doesn't: the health report exposes the *resolved* provider config
# (provider type, model, base_url, dim/normalize/distance, batch, retry
# budgets) so an agent inspects what server it just attached to without
# re-reading dikw.yml; ``api_key_present`` is a bool — never the value.
# Storage DSN / SQLite path / API keys are deliberately omitted.


class LlmInfo(BaseModel):
    """Resolved LLM config in /v1/health response. ``api_key_present``
    is a bool — never the key value; the env var (``ANTHROPIC_API_KEY``
    or ``OPENAI_API_KEY``) is selected by ``provider``.
    """

    provider: Literal["anthropic_compat", "openai_compat", "openai_codex"]
    model: str
    base_url: str | None
    max_retries: int = Field(ge=0)
    max_tokens_synth: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    api_key_present: bool


class MultimodalInfo(MultimodalEmbedConfig):
    """Resolved multimodal embedding config in /v1/health response.

    Inherits all fields from ``MultimodalEmbedConfig`` (provider, model,
    revision, dim, normalize, distance, batch, base_url) so the two
    schemas can never drift. No ``api_key_present`` here — the
    multimodal embedder shares ``DIKW_EMBEDDING_API_KEY`` with the text
    embedder, surfaced once on ``EmbeddingInfo``.
    """


class EmbeddingInfo(BaseModel):
    """Resolved embedding config in /v1/health response.

    ``api_key_present`` reflects ``DIKW_EMBEDDING_API_KEY`` — dikw
    never falls back to ``OPENAI_API_KEY`` here so LLM and embedding
    keys can differ. ``multimodal`` nests under embedding because
    multimodal is a sub-mode of the embedding leg, not a sibling.
    """

    provider: Literal["openai_compat"]
    model: str
    base_url: str | None
    dim: int = Field(gt=0)
    revision: str
    normalize: bool
    distance: Literal["cosine", "l2", "dot"]
    batch_size: int = Field(gt=0)
    max_retries: int = Field(ge=0)
    timeout_seconds: float = Field(gt=0)
    provider_label: str | None
    api_key_present: bool
    multimodal: MultimodalInfo | None = None


class ProvidersInfo(BaseModel):
    llm: LlmInfo
    embedding: EmbeddingInfo


class LayerCounts(BaseModel):
    """Flat agent-facing counts derived from ``StorageCounts``.

    Keep the shape stable across releases: agents probing health rely on
    these names. The richer ``StorageCounts`` (embeddings, links, …)
    stays available via ``GET /v1/status``.
    """

    sources: int
    knowledge_pages: int
    wisdom_items: int
    chunks: int


class HealthReport(BaseModel):
    """``GET /v1/health`` payload — server self-description.

    Intentionally narrow vs ``StorageCounts`` + ``CheckReport``: a probing
    agent should be able to learn (a) is a server running here, (b) which
    base it points at, (c) what providers are wired up, in one round-trip
    that never blocks on outbound provider calls.
    """

    status: Literal["ok"] = "ok"
    version: str
    base_root: str
    storage_engine: Literal["sqlite", "postgres"]
    layer_counts: LayerCounts
    providers: ProvidersInfo
