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
class PagePersistError:
    """One per-page K-layer persist failure surfaced on ``SynthReport``.

    The synth-path analogue of :class:`IngestError`: a hard storage failure
    mid-``persist_knowledge`` (``replace_chunks`` / ``replace_links_from`` /
    ``replace_provenance_from`` raising, or a permanent ``ProviderError``
    from inline embed) deactivates the page and records it here so the run
    continues with the remaining pages instead of aborting — parity with D
    (``api.ingest``) and W (``write_wisdom_page``). A transient embed
    retry-skip is NOT a failure: it surfaces as pending chunks, not here.
    """

    path: str
    message: str


@dataclass(frozen=True)
class SynthVerifyLintFinding:
    """One scoped K-layer lint issue on a page THIS synth run produced.

    A flattened, JSON-serialisable slice of ``domains.knowledge.lint.LintIssue``
    (the client renders from ``dataclasses.asdict``, so it never imports the
    engine type). ``kind`` is the ``LintKind`` string value.
    """

    kind: str
    path: str
    detail: str


@dataclass(frozen=True)
class SynthVerifyReport:
    """Post-synth self-check over the pages THIS run created/updated.

    The "open the vault and click around" pass made automatic: after synth
    writes K pages, ``synthesize(verify=True)`` runs a deterministic, no-extra-
    LLM check scoped to just this run's output and folds the verdict in here so
    the user doesn't have to remember to run ``dikw client lint`` afterwards.
    Purely additive — it READS synth output, never alters it.

    Three independently-gated legs feed ``passed``:

    * **persist** — ``persist_error_count == 0``. A page deactivated mid-
      pipeline (``SynthReport.persist_errors``) is never "clean output".
    * **lint** — a full-base ``run_lint`` whose issues are then filtered to
      this run's pages (the scan itself must see the whole base so
      ``broken_wikilink`` / ``duplicate_title`` resolve against every page),
      gated on the kinds that mark *defective new output*: ``broken_wikilink``
      / ``duplicate_title`` / ``non_atomic_page`` / ``uncategorized`` /
      ``missing_provenance``. ``orphan_page`` is deliberately excluded —
      surfaced on ``orphan_pages`` but NOT gated, because a freshly
      synthesised page is legitimately orphan until something cites it;
      gating it would make ``--verify`` perpetually red on healthy runs
      (Karpathy's rule: a missed backlink is a fixable warning, not
      defective output). Known limitation: ``run_lint`` reports
      ``duplicate_title`` only on the *extra* path of a colliding pair (the
      one that sorts after the first in ``list_documents`` order), so a NEW
      page that collides on title with a PRE-EXISTING page is gated only when
      the new page is the extra. On SQLite the pre-existing page keeps its
      lower rowid (re-synth upserts in place) so the new page is the extra and
      is caught; on Postgres after heap churn the order is unspecified and the
      collision can land on the pre-existing (out-of-scope) path and be
      dropped. Within-run collisions (both paths produced this run) are always
      caught, and a standalone ``dikw client lint`` surfaces either case.
    * **duplicate** — semantic ``duplicate_ratio_max`` over this run's page
      bodies, gated on ``<= max_duplicate_ratio``. Requires an embedder;
      when none was wired the leg is SKIPPED LOUDLY (``duplicate_checked``
      is False, ``duplicate_ratio`` is None) rather than silently passing —
      a green verify must never imply "no duplicates" when the check never
      ran (mirrors the 0.6 loud-skip contract). A skip is NOT a failure.

    ``unresolved_wikilinks`` is surfaced from ``SynthReport`` for context; its
    gated form is the lint leg's ``broken_wikilink`` (re-resolved against the
    final base), so it is not a separate gate.

    The boolean legs (``persist_ok`` / ``lint_ok`` / ``duplicate_ok`` /
    ``passed``) are stored fields, not properties, so they survive
    ``dataclasses.asdict`` and reach the (engine-free) client unchanged.
    """

    pages_checked: int = 0
    persist_error_count: int = 0
    unresolved_wikilinks: int = 0
    lint_findings: tuple[SynthVerifyLintFinding, ...] = ()
    orphan_pages: tuple[str, ...] = ()
    duplicate_checked: bool = False
    duplicate_ratio: float | None = None
    duplicate_cosine_tau: float = 0.0
    max_duplicate_ratio: float = 0.0
    persist_ok: bool = True
    lint_ok: bool = True
    duplicate_ok: bool = True
    passed: bool = True


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
    # Pages that ``dedup_pages_by_slug`` collapsed into an earlier same-path
    # page (summed across every source this run). A non-zero count is the
    # over-generation signal the five gated metrics never surface: the LLM
    # emitted two ``<page>`` blocks that resolve to the same
    # ``knowledge/<category>/<slug>.md`` — usually because a too-small
    # ``target_tokens_per_group`` split one entity across groups, or the
    # model re-described a page it had already written in the batch.
    slug_merge_count: int = 0
    # Pages whose persist raised mid-pipeline; each was deactivated
    # (``active=False``) so it stays out of retrieval. ``errors`` (above)
    # counts LLM-parse failures per group; this is the storage-side analogue.
    persist_errors: tuple[PagePersistError, ...] = ()
    # Post-synth self-check over this run's pages, populated only when
    # ``synthesize(verify=True)`` (``dikw client synth --verify``); ``None``
    # otherwise. See :class:`SynthVerifyReport`.
    verify: SynthVerifyReport | None = None


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
