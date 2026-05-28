"""High-level engine facade — server routes (``dikw_core.server``) and
the eval runner depend on this module; CLI access is via ``dikw client``
which talks HTTP to a running server instead of importing the engine.

Phase 1 surface:
  * ``ingest`` — walk configured sources, parse markdown, chunk, embed, index.
  * ``retrieve`` — hybrid search returning ranked chunks + page refs;
    no LLM call. Answer synthesis is the agent's responsibility.

Phase 2 surface:
  * ``synthesize`` — turn source docs into K-layer knowledge pages via the LLM,
    persist them to disk + storage, maintain the link graph, and refresh
    ``knowledge/index.md`` + ``knowledge/log.md``.
  * ``lint`` — report broken wikilinks, orphans, and duplicate titles.

W layer (wisdom) is being refactored to first-class documents under
``wisdom/<author>/<slug>.md`` — the prior LLM-distilled candidate/review
surface (``distill`` / ``list_candidates`` / ``approve_wisdom`` /
``reject_wisdom``) has been removed in this PR; see CHANGELOG for the
ongoing 0.3.0 refactor.

Phase 0 surface (``init_base``, ``status``) stays unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import logging
import os
import struct
import time
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse, urlsplit, urlunsplit

from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from . import __version__ as _pkg_version
from . import prompts
from .config import (
    CONFIG_FILENAME,
    DikwConfig,
    MultimodalEmbedConfig,
    ProviderConfig,
    default_config,
    dump_config_yaml,
    find_config,
    load_config,
)
from .domains.data.assets import materialize_asset
from .domains.data.backends import UnsupportedFormat, parse_any
from .domains.data.backends.base import ParsedDocument
from .domains.data.path_norm import doc_id_for as _doc_id_for
from .domains.data.path_norm import normalize_path
from .domains.data.persist import persist_source
from .domains.data.sources import iter_source_files
from .domains.info.embed import (
    ChunkToEmbed,
    consume_embedding_stream,
    embed_assets,
    embed_chunks,
    is_unembeddable_asset_mime,
)
from .domains.info.search import HybridSearcher, MultimodalSearch
from .domains.info.tokenize import CjkTokenizer
from .domains.knowledge.grouping import (
    derive_sections_from_chunks,
    group_sections,
)
from .domains.knowledge.indexgen import regenerate_index
from .domains.knowledge.links import (
    build_fuzzy_index,
    normalize_base,
    normalize_for_match,
    parse_links,
)
from .domains.knowledge.lint import LintKind, LintReport, run_lint
from .domains.knowledge.lint_fix import (
    ApplyReport,
    FixerContext,
    FixProposalReport,
    KnowledgePageMeta,
    run_lint_apply,
    run_lint_propose,
)
from .domains.knowledge.log import render_log
from .domains.knowledge.page import KnowledgePage, now_iso, type_from_path, write_page
from .domains.knowledge.synthesize import (
    DEFAULT_SYNTH_SYSTEM,
    SynthesisError,
    SynthesisPartialError,
    dedup_pages_by_slug,
    parse_synthesis_response,
)
from .progress import CancelToken, NoopReporter, ProgressReporter
from .providers import (
    EmbeddingProvider,
    LLMProvider,
    MultimodalEmbeddingProvider,
    ProviderError,
    build_embedder,
    build_llm,
    build_multimodal_embedder,
)
from .schemas import (
    ASSET_URL_TEMPLATE,
    AssetRecord,
    ChunkRecord,
    DerivedPage,
    DocumentRecord,
    EmbeddingVersion,
    GraphEdge,
    GraphNode,
    GraphResult,
    GraphStats,
    GraphUnresolvedLink,
    Hit,
    ImageContent,
    IncomingLink,
    KnowledgeLogEntry,
    Layer,
    LinkDirection,
    LinkType,
    MultimodalInput,
    OutgoingLink,
    PageAnchor,
    PageAsset,
    PageLinksResult,
    PageProvenanceResult,
    PageReadResult,
    PageRef,
    ProvenanceDirection,
    ProvenanceSource,
    RetrieveResult,
    StorageCounts,
    WisdomStatus,
    WisdomWriteReport,
)
from .storage import Storage, build_storage
from .storage.base import NotSupported

logger = logging.getLogger(__name__)

__all__ = [
    "AssetNotFound",
    "CheckReport",
    "DerivedPage",
    "EmbeddingInfo",
    "HealthReport",
    "IncomingLink",
    "IngestError",
    "IngestErrorKind",
    "IngestReport",
    "LayerCounts",
    "LlmInfo",
    "MultimodalInfo",
    "OutgoingLink",
    "PageAnchor",
    "PageLinksResult",
    "PageNotFound",
    "PageProvenanceResult",
    "PageReadResult",
    "PageRef",
    "ProbeResult",
    "ProvenanceSource",
    "ProvidersInfo",
    "RetrieveResult",
    "SynthReport",
    "WisdomWriteReport",
    "check_providers",
    "find_config",
    "health",
    "ingest",
    "init_base",
    "lint",
    "list_graph",
    "list_links",
    "list_pages",
    "load_base",
    "read_asset",
    "read_page",
    "read_provenance",
    "retrieve",
    "status",
    "synthesize",
    "write_wisdom_page",
]


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

KNOWLEDGE_INIT_FILES: dict[str, str] = {
    "sources/.gitkeep": "",
    "knowledge/index.md": (
        "---\n"
        "type: index\n"
        "updated: auto\n"
        "---\n\n"
        "# Knowledge Index\n\n"
        "This file is regenerated by `dikw client synth` and `dikw client lint`.\n"
        "It lists every knowledge page with a one-line summary, grouped by category.\n"
    ),
    "knowledge/log.md": (
        "# Knowledge Log\n\n"
        "Append-only chronological record of engine activity.\n"
    ),
    "knowledge/entities/.gitkeep": "",
    "knowledge/concepts/.gitkeep": "",
    "knowledge/notes/.gitkeep": "",
    "wisdom/.gitkeep": "",
    ".dikw/.gitkeep": "",
    ".gitignore": ".dikw/\n",
}


# One stderr Console for all embedding progress bars. Constructing one
# per ingest pass would re-probe terminal capability + color system on
# every call; rich's recommended pattern is a single shared instance.
_PROGRESS_CONSOLE = Console(stderr=True)


def _ceil_div(n: int, d: int) -> int:
    """``(N + B - 1) // B`` with the same ``batch_size > 0`` guard
    ``embed_chunks`` / ``embed_assets`` already raise on. Without this,
    a ``batch_size: 0`` config produced an opaque ``ZeroDivisionError``
    from this helper before reaching the embed function's validation.
    """
    if d <= 0:
        raise ValueError(f"batch_size must be positive, got {d}")
    return (n + d - 1) // d


@contextlib.contextmanager
def _embedding_progress(
    description: str, *, total: int
) -> Iterator[Callable[[], None]]:
    """Yield an ``advance()`` that bumps a transient stderr progress bar
    by one batch. ``rich.progress`` self-suppresses in non-TTY shells
    (CI, pipe redirects), so the bar is invisible there without a flag.
    """
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=_PROGRESS_CONSOLE,
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=total)
        yield lambda: progress.update(task, advance=1)


def _qualified_provider(protocol: str, base_url: str) -> str:
    """Return ``"<protocol>@<host>"`` for embed_versions.provider.

    The ``embedding`` config field only names the wire protocol
    (``"openai_compat"``); the actual backend is whatever ``base_url``
    points at. We fold the host into the version-identity provider so
    e.g. OpenAI text-embedding-3-small and Gitee's namesake serve under
    distinct version_ids — their vectors live in different spaces and
    must not share a vec table.

    Empty ``base_url`` falls through as bare protocol — the multimodal
    config makes ``base_url`` optional ("use provider default") so we
    avoid synthesising a bogus host placeholder there.
    """
    if not base_url:
        return protocol
    host = urlparse(base_url).hostname or base_url
    return f"{protocol}@{host}"


async def _register_text_version(
    storage: Storage, cfg_provider: ProviderConfig
) -> int:
    """Register the text ``embed_versions`` row from the provider config."""
    return await storage.upsert_embed_version(
        EmbeddingVersion(
            # Encode the endpoint host into ``provider`` so two
            # OpenAI-compatible vendors serving the same model name
            # don't collide on a single version_id (their vectors
            # live in different spaces and must not share a table).
            provider=_qualified_provider(
                cfg_provider.embedding, cfg_provider.embedding_base_url
            ),
            model=cfg_provider.embedding_model,
            revision=cfg_provider.embedding_revision,
            dim=cfg_provider.embedding_dim,
            normalize=cfg_provider.embedding_normalize,
            distance=cfg_provider.embedding_distance,
            modality="text",
        )
    )


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
    """Result of ``check_providers`` — per-leg connectivity probes.

    Either leg may be ``None`` when skipped via ``llm_only`` / ``embed_only``.
    ``ok`` is True when every *present* leg is ok (and at least one is present).
    """

    llm: ProbeResult | None = None
    embed: ProbeResult | None = None

    @property
    def ok(self) -> bool:
        legs = [p for p in (self.llm, self.embed) if p is not None]
        return bool(legs) and all(p.ok for p in legs)


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


# ---- base scaffolding (Phase 0) -----------------------------------------


class BaseUpgradeRequired(RuntimeError):
    """Raised when a base from ≤0.3.6 is opened by 0.4.0+.

    0.4.0 renamed the K-layer on-disk directory from ``wiki/`` to
    ``knowledge/`` and the ``wiki_log`` SQL table to ``knowledge_log``.
    There is no in-place migration — the breaking-change policy is
    rebuild on incompatibility. The exception's message carries the
    exact command the user needs to run.
    """


def _assert_base_upgraded(root: Path) -> None:
    """Refuse to operate against a legacy ``wiki/`` tree.

    Fires whenever ``<root>/wiki/`` exists and contains any markdown
    content — covering both the unmodified 0.3.x layout AND the partial-
    migration case where a user (or tool) created ``knowledge/`` ahead
    of ``mv wiki knowledge`` and would otherwise silently abandon the
    legacy ``wiki/*.md`` content. We do not auto-migrate because the
    SQLite/Postgres schema also changed (``Layer`` enum value +
    ``wiki_log`` table name) and the alpha policy is to ask the user to
    rebuild rather than carry compatibility code forever.
    """
    legacy = root / "wiki"
    if not legacy.is_dir():
        return
    # A bare empty ``wiki/`` left behind by an earlier rename attempt
    # is harmless; only flag if it still carries markdown content the
    # user would lose. ``rglob`` is bounded by the on-disk knowledge
    # tree size — same order of magnitude as ingest's existing scan.
    if not any(legacy.rglob("*.md")):
        return
    raise BaseUpgradeRequired(
        f"base at {root} carries a legacy `wiki/` directory from "
        "dikw-core ≤0.3.6; the K layer directory must be renamed to "
        "`knowledge/` and the database rebuilt. Run:\n"
        f"    cd {root} && mv wiki knowledge && rm -rf .dikw\n"
        "then start the server (`dikw serve --base .`) and run "
        "`dikw client ingest` to reindex. (If `knowledge/` already "
        "exists and you intended to merge content, do the merge by "
        "hand before retrying — the engine refuses to silently abandon "
        "wiki/*.md files.)"
    )


def init_base(root: str | Path, *, description: str | None = None) -> Path:
    base_root = Path(root).resolve()
    base_root.mkdir(parents=True, exist_ok=True)

    existing = base_root / CONFIG_FILENAME
    if existing.exists():
        raise FileExistsError(f"{existing} already exists; refusing to overwrite")

    cfg = default_config(description=description or f"dikw base at {base_root.name}")
    existing.write_text(dump_config_yaml(cfg), encoding="utf-8")

    for rel_path, body in KNOWLEDGE_INIT_FILES.items():
        target = base_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(body, encoding="utf-8")

    return base_root


def resolve_base_root(path: str | Path | None) -> Path:
    start = Path(path) if path is not None else Path.cwd()
    found = find_config(start)
    if found is None:
        raise FileNotFoundError(
            f"no {CONFIG_FILENAME} found at or above {start.resolve()}"
        )
    return found.parent


def load_base(path: str | Path | None = None) -> tuple[DikwConfig, Path]:
    root = resolve_base_root(path)
    _assert_base_upgraded(root)
    return load_config(root / CONFIG_FILENAME), root


def _assert_within(base: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and assert it stays under ``base``.

    Returns the resolved absolute path on success. Raises :class:`ValueError`
    (via :meth:`Path.relative_to`) when ``candidate`` escapes — callers
    translate to whatever domain exception fits (``PageNotFound`` /
    ``AssetNotFound``).
    """
    base_resolved = base.resolve()
    candidate_resolved = candidate.resolve()
    candidate_resolved.relative_to(base_resolved)
    return candidate_resolved


async def _collect_page_assets(
    storage: Storage, chunks: list[ChunkRecord]
) -> list[PageAsset]:
    """Walk ``chunks → chunk_asset_refs → assets`` for one page.

    Deduplicates by ``asset_id`` in first-appearance order (chunk seq,
    then ref ord) so a page that references the same image twice
    surfaces one entry. Asset rows missing between the ref scan and the
    batch fetch (e.g. concurrent delete) are dropped silently.
    """
    chunk_ids = [c.chunk_id for c in chunks if c.chunk_id is not None]
    if not chunk_ids:
        return []
    refs_by_chunk = await storage.chunk_asset_refs_for_chunks(chunk_ids)
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id is None:
            continue
        for ref in refs_by_chunk.get(chunk.chunk_id, ()):
            if ref.asset_id in seen:
                continue
            seen.add(ref.asset_id)
            ordered_ids.append(ref.asset_id)
    if not ordered_ids:
        return []
    records_by_id = {r.asset_id: r for r in await storage.get_assets(ordered_ids)}
    return [
        PageAsset(
            asset_id=r.asset_id,
            kind=r.kind,
            mime=r.mime,
            bytes=r.bytes,
            original_paths=list(r.original_paths),
            media_meta=r.media_meta,
            url=ASSET_URL_TEMPLATE.format(asset_id=r.asset_id),
        )
        for r in (records_by_id[aid] for aid in ordered_ids if aid in records_by_id)
    ]


async def _with_storage(path: str | Path | None) -> tuple[DikwConfig, Path, Storage]:
    cfg, root = load_base(path)
    storage = build_storage(
        cfg.storage, root=root, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    # If connect or migrate raises, the partially opened pool / fd leaks
    # unless we close it on the failure path. Agents probe ``/v1/health``
    # frequently — a repeated migrate failure would otherwise accumulate
    # SQLite fds or Postgres pool slots.
    try:
        await storage.connect()
        await storage.migrate()
    except BaseException:
        with contextlib.suppress(Exception):
            await storage.close()
        raise
    return cfg, root, storage


async def status(path: str | Path | None = None) -> StorageCounts:
    _cfg, _root, storage = await _with_storage(path)
    try:
        return await storage.counts()
    finally:
        await storage.close()


def _sanitize_base_url(url: str | None) -> str | None:
    """Strip userinfo / query / fragment from a ``base_url`` before
    exposing it on /v1/health.

    Defends against credential leakage when a user puts a token directly
    in the URL — ``https://user:token@api.example/`` or
    ``…?api_key=…`` — by keeping only ``scheme://host[:port]/path``.
    Returns ``None`` when the input is empty, unparseable, or has no
    scheme/host: leaving a malformed URL on a probe response is worse
    than dropping it.
    """
    if not url:
        return None
    try:
        parts = urlsplit(url)
        # ``hostname`` and ``port`` are properties that can raise on
        # malformed input (e.g. ``port`` raises ``ValueError`` for an
        # out-of-range or non-numeric port); pull them inside the try.
        host = parts.hostname
        port = parts.port
    except (ValueError, TypeError):
        return None
    if not parts.scheme or not host:
        return None
    # ``urlsplit.hostname`` strips IPv6 brackets — re-bracket so
    # ``http://[::1]:8080/v1`` doesn't round-trip as ``http://::1:8080/v1``.
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _llm_credentials_present(
    provider: Literal["anthropic_compat", "openai_compat", "openai_codex"],
    *,
    base_root: Path,
) -> bool:
    """Whether credentials for the given LLM provider are resolvable.

    Env-keyed providers (anthropic_compat, openai_compat) check the
    matching ``API_KEY_ENV`` constant; the codex protocol checks the
    dikw-managed store at ``<base_root>/.dikw/auth.json``, falling back
    to the codex CLI store iff lazy migration would succeed there
    (fresh, non-expired tokens). That predicts what
    ``resolve_access_token`` will do, so /v1/health agrees with the
    runtime even right before the first LLM call triggers migration —
    and a logged-out dikw store with a stale codex CLI file still
    reports false rather than silently relying on the leftover.

    Explicit per-provider branch (rather than ``else: openai_compat``)
    so adding a new LLM provider surfaces as a typed mypy error here +
    a runtime ``ValueError`` instead of silently reporting the wrong
    credentials shape.
    """
    if provider == "anthropic_compat":
        from .providers.anthropic_compat import API_KEY_ENV

        return bool(os.environ.get(API_KEY_ENV))
    if provider == "openai_compat":
        from .providers.openai_compat import API_KEY_ENV

        return bool(os.environ.get(API_KEY_ENV))
    if provider == "openai_codex":
        from .providers.codex_auth import (
            _read_codex_cli_tokens_if_valid,
            list_providers,
        )

        if "openai-codex" in list_providers(base_root):
            return True
        # Lazy migration: if the dikw store is empty but the codex CLI
        # store has fresh tokens, the next ``resolve_access_token`` call
        # will populate the dikw store automatically. Surface that as
        # "credentials present" so /v1/health doesn't false-negative on
        # upgraded users who haven't issued an LLM call yet.
        return _read_codex_cli_tokens_if_valid() is not None
    raise ValueError(f"unknown llm provider: {provider!r}")


async def health(path: str | Path | None = None) -> HealthReport:
    """Server self-description for agent bootstrap probes.

    Opens storage briefly to read ``counts()``; never invokes the LLM /
    embedding providers (so a misconfigured key does not 5xx the health
    probe). Returned config is the *resolved* shape — what the server
    actually uses — minus secrets (no API keys, no DSN, no SQLite path).
    """
    from .providers.openai_compat import EMBEDDING_API_KEY_ENV

    cfg, root, storage = await _with_storage(path)
    try:
        counts = await storage.counts()
    finally:
        await storage.close()

    by_layer = counts.documents_by_layer
    # PR1 of the 0.3.0 wisdom refactor: ``wisdom_items`` count is 0 here
    # — the legacy ``wisdom_items`` table is removed and wisdom pages
    # don't yet flow into ``documents`` (PR2 lands that). Keep the
    # ``LayerCounts.wisdom_items`` field for wire compatibility so
    # agents reading health responses can transition over two releases.
    layer_counts = LayerCounts(
        sources=int(by_layer.get(Layer.SOURCE.value, 0)),
        knowledge_pages=int(by_layer.get(Layer.KNOWLEDGE.value, 0)),
        wisdom_items=int(by_layer.get(Layer.WISDOM.value, 0)),
        chunks=counts.chunks,
    )

    p = cfg.provider
    llm_info = LlmInfo(
        provider=p.llm,
        model=p.llm_model,
        base_url=_sanitize_base_url(p.llm_base_url),
        max_retries=p.llm_max_retries,
        max_tokens_synth=p.llm_max_tokens_synth,
        timeout_seconds=p.llm_timeout_seconds,
        api_key_present=_llm_credentials_present(p.llm, base_root=root),
    )

    mm_info: MultimodalInfo | None = None
    mm_cfg = cfg.assets.multimodal
    if mm_cfg is not None:
        mm_dump = mm_cfg.model_dump()
        mm_dump["base_url"] = _sanitize_base_url(mm_dump.get("base_url"))
        mm_info = MultimodalInfo.model_validate(mm_dump)

    embedding_info = EmbeddingInfo(
        provider=p.embedding,
        model=p.embedding_model,
        base_url=_sanitize_base_url(p.embedding_base_url),
        dim=p.embedding_dim,
        revision=p.embedding_revision,
        normalize=p.embedding_normalize,
        distance=p.embedding_distance,
        batch_size=p.embedding_batch_size,
        max_retries=p.embedding_max_retries,
        timeout_seconds=p.embedding_timeout_seconds,
        provider_label=p.embedding_provider_label,
        api_key_present=bool(os.environ.get(EMBEDDING_API_KEY_ENV)),
        multimodal=mm_info,
    )

    return HealthReport(
        version=_pkg_version,
        base_root=str(Path(root).resolve()),
        storage_engine=cfg.storage.backend,
        layer_counts=layer_counts,
        providers=ProvidersInfo(llm=llm_info, embedding=embedding_info),
    )


# ---- verifiable config tool ---------------------------------------------


async def _probe_llm(
    llm: LLMProvider, model: str, target: str
) -> ProbeResult:
    started = time.perf_counter()
    try:
        resp = await llm.complete(
            system="You are a connectivity check. Reply with just: OK",
            user="ping",
            model=model,
            max_tokens=4,
            temperature=0.0,
        )
    except Exception as e:  # provider exceptions are intentionally heterogeneous
        return ProbeResult(ok=False, target=target, detail=f"{type(e).__name__}: {e}")
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    input_tok = int((resp.usage or {}).get("input_tokens", 0))
    return ProbeResult(
        ok=True,
        target=target,
        detail=f"{elapsed_ms}ms, {input_tok} input tokens",
    )


async def _probe_embed(
    embedder: EmbeddingProvider,
    model: str,
    target: str,
    *,
    provider_label: str | None = None,
) -> ProbeResult:
    started = time.perf_counter()
    try:
        vectors = await embedder.embed(["ping"], model=model)
    except Exception as e:
        return ProbeResult(ok=False, target=target, detail=f"{type(e).__name__}: {e}")
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    dim = len(vectors[0]) if vectors else 0
    detail = f"{elapsed_ms}ms, dim={dim}"
    if provider_label:
        detail = f"{detail}, provider={provider_label}"
    return ProbeResult(ok=True, target=target, detail=detail)


def _build_probe_png_1x1() -> bytes:
    """Smallest valid PNG (1x1 black RGB pixel, 69 bytes).

    Built at module load via stdlib ``struct`` + ``zlib`` so the chunk
    CRCs are guaranteed correct — a hand-written byte literal here was
    truncated by one CRC byte once and Gitee's image decoder rejected
    the whole multimodal probe with a misleading "Supported image
    type:" error that hid the real cause.
    """

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: width=1, height=1, bit_depth=8, color_type=2 (RGB), the rest 0.
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    # IDAT: one scanline of [filter=0, R=0, G=0, B=0], deflate-compressed.
    idat = _chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00", 9))
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


_PROBE_PNG_1X1 = _build_probe_png_1x1()

async def _probe_multimodal(
    embedder: MultimodalEmbeddingProvider,
    model: str,
    target: str,
    *,
    provider_label: str | None = None,
) -> ProbeResult:
    """Probe a multimodal embedder with a single batched text+image request.

    Sends two per-modality inputs (one text, one tiny PNG) in **one** HTTP
    call so latency stays bounded by a single RTT — no sequential probes
    that would double end-to-end time. Validation hinges on both vectors
    coming back with a consistent dim; an empty or shape-mismatched
    response surfaces as an error rather than a silent pass.
    """
    inputs = [
        MultimodalInput(text="ping"),
        MultimodalInput(images=[ImageContent(bytes=_PROBE_PNG_1X1, mime="image/png")]),
    ]
    started = time.perf_counter()
    try:
        vectors = await embedder.embed(inputs, model=model)
    except Exception as e:
        return ProbeResult(ok=False, target=target, detail=f"{type(e).__name__}: {e}")
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if len(vectors) != 2:
        return ProbeResult(
            ok=False,
            target=target,
            detail=(
                f"{elapsed_ms}ms, expected 2 vectors (text+image), got "
                f"{len(vectors)} — provider returned a shape-mismatched batch"
            ),
        )
    dim_text = len(vectors[0])
    dim_image = len(vectors[1])
    if dim_text != dim_image:
        return ProbeResult(
            ok=False,
            target=target,
            detail=(
                f"{elapsed_ms}ms, dim mismatch text={dim_text} image={dim_image} — "
                f"per-modality vectors must share one space"
            ),
        )
    detail = f"{elapsed_ms}ms, dim={dim_text}, modalities=text+image"
    if provider_label:
        detail = f"{detail}, provider={provider_label}"
    return ProbeResult(ok=True, target=target, detail=detail)


async def check_providers(
    path: str | Path | None = None,
    *,
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    multimodal_embedder: MultimodalEmbeddingProvider | None = None,
    llm_only: bool = False,
    embed_only: bool = False,
) -> CheckReport:
    """Ping the configured LLM and embedding providers.

    ``llm``, ``embedder``, and ``multimodal_embedder`` are injectable for
    tests; production callers leave them ``None`` and the factory builds
    them from ``provider:`` / ``assets.multimodal:`` in ``dikw.yml``. Two
    legs run in parallel and each reports its own result — an LLM failure
    does not short-circuit the embedding probe.

    When ``cfg.assets.multimodal`` is configured, the embed leg routes
    through ``_probe_multimodal`` (one batched text+image request) instead
    of the text-only ``_probe_embed`` — the multimodal embedder is what
    ingest actually uses for both chunks and assets, so the check must
    follow that route to remain truthful.

    ``llm_only`` / ``embed_only`` (mutually exclusive) verify a single leg.
    The skipped leg is never built or called, so a misconfigured embedding
    side cannot fail an ``--llm-only`` run.
    """
    if llm_only and embed_only:
        raise ValueError("llm_only and embed_only are mutually exclusive")

    cfg, _root = load_base(path)

    llm_probe: ProbeResult | None = None
    embed_probe: ProbeResult | None = None
    llm_target = cfg.provider.llm_base_url or "(provider default)"
    embed_label = cfg.provider.embedding_provider_label
    mm_cfg = cfg.assets.multimodal
    # Per-leg target. Multimodal probe hits ``assets.multimodal.base_url``
    # (or the provider's default), which is independent of the text leg's
    # ``embedding_base_url``; reporting the wrong one makes a green check
    # misleading in split-vendor setups.
    if mm_cfg is not None:
        embed_target = mm_cfg.base_url or "(provider default)"
    else:
        embed_target = cfg.provider.embedding_base_url
    # Track an internally-built multimodal embedder so we close its httpx
    # client in ``finally`` — mirrors the ``owned_mm`` pattern in ingest/
    # query. An *injected* embedder is the caller's lifetime to manage.
    owned_mm: MultimodalEmbeddingProvider | None = None

    if not embed_only:
        llm_inst = llm if llm is not None else build_llm(cfg.provider, base_root=_root)
    if not llm_only:
        if mm_cfg is not None:
            if multimodal_embedder is not None:
                mm_inst = multimodal_embedder
            else:
                mm_inst = build_multimodal_embedder(
                    mm_cfg.provider, base_url=mm_cfg.base_url, batch=mm_cfg.batch
                )
                owned_mm = mm_inst
        else:
            embed_inst = (
                embedder if embedder is not None else build_embedder(cfg.provider)
            )

    async def _embed_leg() -> ProbeResult:
        if mm_cfg is not None:
            return await _probe_multimodal(
                mm_inst, mm_cfg.model, embed_target, provider_label=embed_label
            )
        return await _probe_embed(
            embed_inst,
            cfg.provider.embedding_model,
            embed_target,
            provider_label=embed_label,
        )

    try:
        if llm_only:
            llm_probe = await _probe_llm(llm_inst, cfg.provider.llm_model, llm_target)
        elif embed_only:
            embed_probe = await _embed_leg()
        else:
            llm_probe, embed_probe = await asyncio.gather(
                _probe_llm(llm_inst, cfg.provider.llm_model, llm_target),
                _embed_leg(),
            )
    finally:
        if owned_mm is not None and hasattr(owned_mm, "aclose"):
            await owned_mm.aclose()
    return CheckReport(llm=llm_probe, embed=embed_probe)


# ---- Phase 1: ingest -----------------------------------------------------


async def ingest(
    path: str | Path | None = None,
    *,
    embedder: EmbeddingProvider | None = None,
    multimodal_embedder: MultimodalEmbeddingProvider | None = None,
    reporter: ProgressReporter | None = None,
) -> IngestReport:
    """Ingest every markdown file listed in ``sources:`` into the D and I layers.

    Two provider knobs — strictly per-channel, since query() searches
    text and multimodal vectors via separate version_ids:

    * ``embedder`` — text-only ``EmbeddingProvider``; populates the
      ``vec_chunks_v<text_version_id>`` table.
    * ``multimodal_embedder`` — ``MultimodalEmbeddingProvider``; when
      ``cfg.assets.multimodal`` is configured this embeds image-asset
      bytes into ``vec_assets_v<mm_version_id>``. It does NOT embed
      chunk text — chunks always flow through the text channel.

    ``reporter`` (optional) receives structured progress events for
    server-driven task wrappers; in-process callers leave it ``None`` and
    rely on the ``rich`` stderr progress bar instead.

    Asset binaries referenced from markdown are materialized into
    ``<root>/<assets.dir>/`` regardless of which embedder is set —
    ``chunk_asset_refs`` always reflect the on-disk structure so
    query-time consumers can render them.
    """
    cfg, root, storage = await _with_storage(path)
    owned_mm: MultimodalEmbeddingProvider | None = None
    _reporter: ProgressReporter = reporter or NoopReporter()
    try:
        report = IngestReport()
        to_embed: list[ChunkToEmbed] = []
        # Newly-materialized assets (deduped by asset_id) collected across
        # the run for batch image embedding at the end.
        new_assets_by_id: dict[str, AssetRecord] = {}

        # Resolve text + multimodal versions once per run so every chunk
        # and asset embedded below carries a stable version_id from the
        # embed_versions registry.
        text_version_id: int | None = None
        if embedder is not None:
            text_version_id = await _register_text_version(storage, cfg.provider)

        mm_version_id: int | None = None
        mm_cfg = cfg.assets.multimodal
        if mm_cfg is not None:
            mm_version_id = await storage.upsert_embed_version(
                EmbeddingVersion(
                    # Same endpoint-aware identity as the text leg —
                    # mm_cfg.provider names the wire shape, base_url
                    # selects the actual backend.
                    provider=_qualified_provider(
                        mm_cfg.provider, mm_cfg.base_url or ""
                    ),
                    model=mm_cfg.model,
                    revision=mm_cfg.revision,
                    dim=mm_cfg.dim,
                    normalize=mm_cfg.normalize,
                    distance=mm_cfg.distance,
                    modality="multimodal",
                )
            )

        # Auto-build the multimodal embedder from config when one wasn't
        # injected — symmetric with query()'s behavior, so the typical
        # server / eval call site (which only passes `embedder`) still
        # produces chunk + asset vectors in the configured mm space
        # rather than indexing chunks in text space and querying in
        # mm space.
        if mm_cfg is not None and multimodal_embedder is None:
            multimodal_embedder = build_multimodal_embedder(
                mm_cfg.provider,
                base_url=mm_cfg.base_url,
                batch=mm_cfg.batch,
            )
            owned_mm = multimodal_embedder

        await _reporter.progress(phase="scan", current=0, total=0)
        for abs_path, logical_path in iter_source_files(cfg.sources, root=root):
            _reporter.cancel_token().raise_if_cancelled()
            try:
                parsed = parse_any(abs_path, rel_path=logical_path)
            except UnsupportedFormat:
                # Glob swept up a non-md file (asset binary, .git/*, etc.).
                # Skip silently — surfacing every one would balloon the
                # error tape on a wide ``**/*`` glob without telling the
                # user anything they didn't already know about their
                # config. ``parse_error`` / ``read_error`` / ``storage_error``
                # are the user-actionable surfaces this PR opens up.
                continue
            except (OSError, UnicodeError) as e:
                # OSError covers filesystem refusals (permission denied,
                # file disappeared mid-scan); UnicodeError catches the
                # not-UTF-8 case — ``Path.read_text`` raises that one as
                # a ``ValueError`` subclass that would otherwise fall
                # into the parse_error catch-all and mislead callers
                # branching on ``kind``. Both are read-side failures.
                report = await _record_ingest_error(
                    report,
                    _reporter,
                    path=logical_path,
                    kind="read_error",
                    message=f"{type(e).__name__}: {e}",
                )
                continue
            except Exception as e:
                # Parser-side failure: invalid YAML front-matter,
                # malformed inline syntax our backend can't tolerate, etc.
                report = await _record_ingest_error(
                    report,
                    _reporter,
                    path=logical_path,
                    kind="parse_error",
                    message=f"{type(e).__name__}: {e}",
                )
                continue
            doc_id = _doc_id_for(Layer.SOURCE, logical_path)
            existing = await storage.get_document(doc_id)

            scanned = report.scanned + 1
            # Skip the chunk/embed pipeline only when the doc body is
            # unchanged AND the doc has no asset references AND the
            # row is currently active. A row deactivated by a prior
            # ``storage_error`` arm carries the same hash but a
            # half-indexed state — falling through re-runs the whole
            # pipeline and re-upserts ``active=True``.
            if (
                existing is not None
                and existing.active
                and existing.hash == parsed.hash
                and not parsed.asset_refs
            ):
                report = _replace(
                    report,
                    scanned=scanned,
                    unchanged=report.unchanged + 1,
                )
                continue

            try:
                # Materialize image references before chunking so asset_ids
                # are available when chunk_asset_refs land. Decoupled from
                # mm_cfg so eval rigs see the chunk ↔ asset bridge even
                # without multimodal embedding configured. ``persist_source``
                # itself doesn't touch the filesystem — it consumes the
                # already-resolved ``ref_assets`` dict.
                ref_assets: dict[int, AssetRecord] = {}
                if parsed.asset_refs:
                    by_original_path: dict[str, AssetRecord] = {}
                    try:
                        for ref_idx, ref in enumerate(parsed.asset_refs):
                            cached = by_original_path.get(ref.original_path)
                            if cached is not None:
                                ref_assets[ref_idx] = cached
                                continue
                            result = await materialize_asset(
                                ref,
                                source_md_path=abs_path,
                                project_root=root,
                                get_asset=storage.get_asset,
                                upsert_asset=storage.upsert_asset,
                                dir_=cfg.assets.dir,
                            )
                            if result is not None:
                                rec, was_new = result
                                ref_assets[ref_idx] = rec
                                by_original_path[ref.original_path] = rec
                                # Only new binaries need an embedding pass;
                                # existing rows keep their cached vector.
                                if was_new and mm_cfg is not None:
                                    new_assets_by_id.setdefault(rec.asset_id, rec)
                    except NotSupported:
                        ref_assets = {}

                persist_result = await persist_source(
                    storage=storage,
                    parsed=parsed,
                    doc=_to_document(parsed, doc_id=doc_id),
                    ref_assets=ref_assets,
                    cjk_tokenizer=cfg.retrieval.cjk_tokenizer,
                )

                # Queue chunks for embedding only when a text embedder + version
                # is wired up. Chunk vectors live exclusively in the text
                # channel (vec_chunks_v<text_id>); the multimodal channel
                # owns assets, not chunks. ``persist_source`` defers embed
                # to this end-of-scan bulk pass for throughput across files.
                if (
                    embedder is not None
                    and text_version_id is not None
                    and persist_result.chunks_count
                ):
                    to_embed.extend(
                        ChunkToEmbed(chunk_id=cid, text=text)
                        for cid, text in zip(
                            persist_result.chunk_ids,
                            persist_result.chunk_texts,
                            strict=True,
                        )
                    )

                await storage.append_knowledge_log(
                    KnowledgeLogEntry(ts=time.time(), action="ingest", src=logical_path)
                )
            except Exception as e:
                # Storage / chunking / asset materialisation raised
                # mid-file. ``upsert_document`` may already have landed
                # the doc row before the failure point — without an
                # explicit deactivation, the next ingest under an
                # unchanged content hash would hit the early-skip arm
                # above and the orphaned doc would stay half-indexed
                # forever. Deactivating bypasses the skip on retry so
                # the doc gets re-processed end-to-end.
                with contextlib.suppress(Exception):
                    await storage.deactivate_document(doc_id)
                report = await _record_ingest_error(
                    report,
                    _reporter,
                    path=logical_path,
                    kind="storage_error",
                    message=f"{type(e).__name__}: {e}",
                )
                continue

            report = _replace(
                report,
                scanned=scanned,
                added=report.added if existing is not None else report.added + 1,
                updated=report.updated + 1 if existing is not None else report.updated,
                chunks=report.chunks + persist_result.chunks_count,
            )
            await _reporter.progress(
                phase="scan",
                current=scanned,
                total=0,
                detail={"path": logical_path},
            )

        # W-layer scan was removed in 0.4.0 (#BREAKING). Wisdom pages
        # are indexed exclusively by ``write_wisdom_page`` (the public
        # ``dikw client wisdom write`` / ``POST /v1/wisdom/write`` entry
        # point). A user editing a wisdom file in obsidian no longer
        # gets it auto-reindexed by ``dikw client ingest``; the
        # ``list_chunks_missing_embedding`` resume scan below still
        # covers W-layer chunks that landed without vectors (e.g.,
        # ``wisdom write --no-embed`` or a flaky embed batch that hit
        # the retry-skip path in commit 1).

        # Resume scan: pick up chunks that landed in storage during a
        # prior crashed run but never got their embedding written. The
        # doc-level shortcut above skips docs whose body_hash matched
        # storage, so without this scan a half-embedded run can NEVER
        # finish — its remaining chunks are invisible to the per-doc
        # loop. The cache lookup in slice 5 makes this nearly free for
        # chunks whose text is already cached; for true misses (the
        # tail that crashed mid-flight) we re-pay the provider.
        if embedder is not None and text_version_id is not None:
            already_queued_ids = {c.chunk_id for c in to_embed}
            missing = await storage.list_chunks_missing_embedding(
                version_id=text_version_id
            )
            for chunk in missing:
                if chunk.chunk_id is None or chunk.chunk_id in already_queued_ids:
                    continue
                to_embed.append(
                    ChunkToEmbed(chunk_id=chunk.chunk_id, text=chunk.text)
                )

        # Chunk-text embeddings — text channel only. Streaming consume:
        # each batch is upserted as soon as the provider returns it, so
        # a mid-flight crash keeps the prior batches' vectors on disk
        # instead of throwing away the entire run's API spend.
        if to_embed and embedder is not None and text_version_id is not None:
            chunk_batch_size = cfg.provider.embedding_batch_size
            chunk_total = _ceil_div(len(to_embed), chunk_batch_size)
            with _embedding_progress(
                "embedding chunks", total=chunk_total
            ) as advance_chunk:
                embed_result = await consume_embedding_stream(
                    embed_chunks(
                        embedder,
                        to_embed,
                        model=cfg.provider.embedding_model,
                        version_id=text_version_id,
                        storage=storage,
                        batch_size=chunk_batch_size,
                        retries=cfg.provider.embedding_error_retries,
                        backoff_seconds=cfg.provider.embedding_error_retry_backoff_seconds,
                        reporter=_reporter,
                    ),
                    storage,
                    on_batch=advance_chunk,
                    reporter=_reporter,
                    phase="embed_chunks",
                    total=chunk_total,
                )
            report = _replace(report, embedded=embed_result.embedded)

        # Backfill assets stored without a vector for the active mm
        # version — text-only ingest residue, prior mm version, or
        # mid-flight crash of an earlier embed pass. Kept separate from
        # ``new_assets_by_id`` so ``report.assets`` (= NEW this run)
        # stays accurate; the union below is what we feed to the
        # embed pass. Gated on the same condition as the embed block
        # to avoid a no-op SQL round-trip when no mm embedder is wired.
        backfill_by_id: dict[str, AssetRecord] = {}
        if (
            multimodal_embedder is not None
            and mm_cfg is not None
            and mm_version_id is not None
        ):
            missing_assets = await storage.list_assets_missing_embedding(
                version_id=mm_version_id
            )
            # Skip categories ``embed_assets`` deliberately discards
            # without writing a meta row — they'd reappear in every
            # subsequent ingest's "needs embedding" list forever:
            #   * unembeddable mime (SVG today; v1 doesn't rasterize)
            #   * stored binary missing on disk (asset row points at a
            #     deleted file — first time we hit it, ``embed_assets``
            #     logs the read failure; the backfill scan should not
            #     keep re-reading + re-warning on every later ingest).
            candidates = [
                rec
                for rec in missing_assets
                if not is_unembeddable_asset_mime(rec.mime)
                and rec.asset_id not in new_assets_by_id
                and (root / rec.stored_path).is_file()
            ]
            # Filter to assets still referenced by at least one live
            # chunk. An asset whose markdown ref was deleted is
            # unreachable via ``HybridSearcher`` (which promotes asset
            # hits through ``chunks_referencing_assets``), so embedding
            # those orphans burns provider calls for vectors search
            # can never surface.
            if candidates:
                refs_by_asset = await storage.chunks_referencing_assets(
                    [rec.asset_id for rec in candidates]
                )
                for rec in candidates:
                    if refs_by_asset.get(rec.asset_id):
                        backfill_by_id[rec.asset_id] = rec

        to_embed_assets: dict[str, AssetRecord] = {
            **new_assets_by_id,
            **backfill_by_id,
        }
        if (
            multimodal_embedder is not None
            and mm_cfg is not None
            and mm_version_id is not None
            and to_embed_assets
        ):
            asset_total_batches = _ceil_div(len(to_embed_assets), mm_cfg.batch)
            asset_embedded = 0
            asset_batches_done = 0
            # Per-batch upsert: a mid-flight provider failure leaves
            # prior batches' vectors on disk so the next retry's
            # backfill scan sees only the truly-missing tail. Symmetric
            # with the chunk side's ``consume_embedding_stream``.
            with _embedding_progress(
                "embedding assets", total=asset_total_batches
            ) as advance_asset:
                async for batch_rows in embed_assets(
                    multimodal_embedder,
                    list(to_embed_assets.values()),
                    project_root=root,
                    model=mm_cfg.model,
                    version_id=mm_version_id,
                    batch_size=mm_cfg.batch,
                ):
                    if batch_rows:
                        await storage.upsert_asset_embeddings(batch_rows)
                        asset_embedded += len(batch_rows)
                    advance_asset()
                    asset_batches_done += 1
                    await _reporter.progress(
                        phase="embed_assets",
                        current=asset_batches_done,
                        total=asset_total_batches,
                    )
                    _reporter.cancel_token().raise_if_cancelled()
            report = _replace(
                report,
                assets=len(new_assets_by_id),
                asset_embedded=asset_embedded,
            )
        elif new_assets_by_id:
            # Materialized assets even without an mm embedder so the chunk
            # references resolve at query/render time.
            report = _replace(report, assets=len(new_assets_by_id))

        return report
    finally:
        if owned_mm is not None and hasattr(owned_mm, "aclose"):
            await owned_mm.aclose()
        await storage.close()


def _to_document(
    parsed: ParsedDocument, *, doc_id: str, layer: Layer = Layer.SOURCE
) -> DocumentRecord:
    # ``status`` is wisdom-only — knowledge/source rows always store NULL
    # even if the user pasted ``status:`` into the wrong frontmatter.
    # The CHECK constraint allows NULL anywhere, so this clamp is the
    # invariant guard. See ``WisdomStatus`` docstring and
    # ``test_wiki_page_status_frontmatter_forced_to_null``.
    status = parsed.status if layer is Layer.WISDOM else None
    return DocumentRecord(
        doc_id=doc_id,
        path=parsed.path,
        title=parsed.title,
        hash=parsed.hash,
        mtime=parsed.mtime,
        layer=layer,
        active=True,
        status=status,
    )


def _replace(r: IngestReport, **kwargs: Any) -> IngestReport:
    # Thin wrapper around ``dataclasses.replace`` — kept for grep-ability
    # and to keep call-sites reading "_replace(report, scanned=…)" rather
    # than reaching for a stdlib import.
    return dataclasses.replace(r, **kwargs)


async def _record_ingest_error(
    report: IngestReport,
    reporter: ProgressReporter,
    *,
    path: str,
    kind: IngestErrorKind,
    message: str,
) -> IngestReport:
    """Append a per-file failure to the report and emit a wire event.

    Counts the failed file as scanned (so the report's ``scanned``
    matches "files we tried to process"), appends an :class:`IngestError`
    to ``report.errors``, and pushes a ``partial("file_error", …)``
    event so streaming subscribers (the CLI's progress widget, the task
    NDJSON stream) can surface the failure live instead of waiting for
    the final report.
    """
    err = IngestError(path=path, kind=kind, message=message)
    await reporter.partial(
        "file_error",
        {"path": path, "kind": kind, "message": message},
    )
    return IngestReport(
        scanned=report.scanned + 1,
        added=report.added,
        updated=report.updated,
        unchanged=report.unchanged,
        chunks=report.chunks,
        embedded=report.embedded,
        assets=report.assets,
        asset_embedded=report.asset_embedded,
        errors=(*report.errors, err),
    )


# ---- Phase 1: query ------------------------------------------------------


async def _retrieve_inner(
    storage: Storage,
    cfg: DikwConfig,
    q: str,
    *,
    limit: int,
    embedder: EmbeddingProvider | None = None,
    multimodal_embedder: MultimodalEmbeddingProvider | None = None,
    reporter: ProgressReporter | None = None,
) -> tuple[list[Hit], MultimodalEmbeddingProvider | None]:
    """Run hybrid search and return (hits, owned_mm_embedder).

    Shared helper between ``query`` (LLM-driven RAG) and ``retrieve``
    (retrieval-only). The caller is responsible for closing
    ``owned_mm_embedder`` (returned non-None only when this helper had
    to build the multimodal embedder itself; a caller-supplied embedder
    is never owned here).

    Emits the ``retrieval_done`` partial via ``reporter`` so the
    ``/v1/retrieve`` wire surface reports a stable shape; pass ``None``
    for ``reporter`` to silence the partial.
    """
    _reporter: ProgressReporter = reporter or NoopReporter()
    owned_mm: MultimodalEmbeddingProvider | None = None

    try:
        # Pin the text leg to the active text version's stored model AND
        # dim so a mid-flight cfg edit (new embedding_model /
        # embedding_dim in dikw.yml, no re-ingest) doesn't corrupt query
        # rankings — same anti-drift guard the multimodal path applies
        # below. We resolve the active version BEFORE building the
        # embedder so the override can flow into ``default_dimensions``.
        text_version_id: int | None = None
        text_query_model = cfg.provider.embedding_model
        text_query_dim: int | None = None
        try:
            active_text = await storage.get_active_embed_version(modality="text")
        except NotSupported as e:
            logger.warning(
                "storage backend doesn't support text versioning (%s); "
                "querying with the cfg embedding_model unchecked",
                e,
            )
            active_text = None
        if active_text is not None and active_text.version_id is not None:
            text_version_id = active_text.version_id
            text_query_model = active_text.model
            text_query_dim = active_text.dim

        _embedder = embedder
        if _embedder is None:
            _embedder = build_embedder(cfg.provider, dim_override=text_query_dim)

        mm_search: MultimodalSearch | None = None
        mm_cfg = cfg.assets.multimodal
        if mm_cfg is not None:
            try:
                active = await storage.get_active_embed_version(
                    modality="multimodal"
                )
            except NotSupported as e:
                logger.warning(
                    "storage backend doesn't support multimodal versioning "
                    "(%s); querying with text-only retrieval",
                    e,
                )
                active = None
            if active is not None and active.version_id is not None:
                mm_embedder = multimodal_embedder
                if mm_embedder is None:
                    mm_embedder = build_multimodal_embedder(
                        mm_cfg.provider,
                        base_url=mm_cfg.base_url,
                        batch=mm_cfg.batch,
                    )
                    # Assign immediately so an exception between here and
                    # the `return` below still goes through this scope's
                    # cleanup — caller's finally only sees ``owned_mm``
                    # after a successful return.
                    owned_mm = mm_embedder
                # Use the model recorded on the active version, not the
                # current cfg model — if the user just edited dikw.yml to
                # point at a new model but hasn't re-ingested yet, the
                # asset vectors in vec_assets_v<active> were produced by
                # the OLD model; querying with the new model would either
                # mismatch dim or rank against an incompatible space.
                mm_search = MultimodalSearch(
                    embedder=mm_embedder,
                    model=active.model,
                    asset_version_id=active.version_id,
                )

        searcher = HybridSearcher.from_config(
            storage,
            _embedder,
            cfg.retrieval,
            embedding_model=text_query_model,
            text_version_id=text_version_id,
            multimodal=mm_search,
        )
        hits = await searcher.search(q, limit=limit)
        # Include full ``text`` so a streaming agent can prompt off the
        # partial without waiting for ``final``. Cost: chunk bodies
        # duplicate on ``final.result.chunks`` — clients that don't
        # need the partial can stop reading after ``final``.
        await _reporter.partial(
            "retrieval_done",
            {"hits": [h.model_dump(mode="json") for h in hits]},
        )
        return hits, owned_mm
    except BaseException:
        # Catch ``BaseException`` (not just ``Exception``) so the cleanup
        # runs on ``asyncio.CancelledError`` too — a cancelled retrieve
        # mid-flight must not leak the multimodal embedder we just built.
        #
        # Inner ``except Exception`` (not ``BaseException``) is
        # intentional: if ``aclose`` itself raises ``CancelledError`` /
        # ``SystemExit`` / ``KeyboardInterrupt`` we let it propagate and
        # replace the original exception. asyncio convention treats
        # cancellation as a higher-priority signal that callers must see
        # — masking it under ``raise`` of the original would break
        # cooperative shutdown. Regular cleanup failures (network,
        # provider crash) are logged and the original exception wins.
        if owned_mm is not None and hasattr(owned_mm, "aclose"):
            try:
                await owned_mm.aclose()
            except Exception:
                logger.exception(
                    "multimodal embedder aclose failed during _retrieve_inner cleanup"
                )
        raise


def _build_page_refs(hits: list[Hit]) -> list[PageRef]:
    """Aggregate fusion-ranked chunks into page-level refs.

    ``score`` is the max chunk score for each path so an agent can
    rank pages without re-aggregating. ``hit_chunk_ids`` is captured in
    fusion-rank order (insertion order of hits) so the caller can
    cross-reference back to ``chunks[]`` deterministically. Hits with
    ``path=None`` are dropped — they cannot be cited as a page.
    """
    accum: dict[str, dict[str, Any]] = {}
    for h in hits:
        if h.path is None:
            continue
        bucket = accum.get(h.path)
        if bucket is None:
            accum[h.path] = {
                "path": h.path,
                "layer": h.layer,
                "title": h.title,
                "score": h.score,
                "hit_chunk_ids": [h.chunk_id],
            }
        else:
            bucket["hit_chunk_ids"].append(h.chunk_id)
            if h.score > bucket["score"]:
                bucket["score"] = h.score
    refs = [PageRef(**bucket) for bucket in accum.values()]
    refs.sort(key=lambda r: r.score, reverse=True)
    return refs


async def list_pages(
    root: str | Path | None,
    *,
    layer: Layer | None = None,
    active: bool | None = True,
    since_ts: float | None = None,
) -> list[DocumentRecord]:
    """Return registered documents (D / K layer pages) under ``root``.

    Thin facade around :meth:`Storage.list_documents` so server routes
    don't reach into :func:`_with_storage` directly — keeps the
    engine/server boundary symmetric with :func:`read_page`. Default
    ``active=True`` matches the list endpoint's wire contract (deactivated
    docs are not surfaced).

    The W (wisdom) layer is reachable here via ``layer=Layer.WISDOM``
    only as a forward-compat hook — PR1 of the 0.3.0 wisdom refactor
    removes the legacy ``wisdom_items`` table but doesn't yet wire
    wisdom files into ``documents``; PR2 lands that pipeline.
    """
    cfg, _root, storage = await _with_storage(root)
    del cfg
    try:
        docs = await storage.list_documents(
            layer=layer, active=active, since_ts=since_ts
        )
        return list(docs)
    finally:
        await storage.close()


async def read_page(
    root: str | Path | None, path: str
) -> PageReadResult:
    """Read a registered page (D or K layer) + its chunk anchors.

    Path safety is index-driven: only paths present in the ``documents``
    table are reachable, so unindexed files (``dikw.yml``, files outside
    the base root, ``..`` traversal attempts) all get a uniform
    :class:`PageNotFound`.

    ``body`` is the **parsed** body — front-matter stripped — because
    chunk anchors live in that coordinate space (see
    ``markdown.parse_text`` which strips ``---`` front-matter before
    chunking). Returning the raw on-disk text would put anchors at
    wrong offsets when a file has YAML front-matter.

    ``anchors`` is empty if the file has been edited since ingest
    (current parsed-body hash differs from ``match.hash``) — stale
    anchors would silently misalign, so we drop them and let the caller
    re-ingest. Empty is also returned for docs that produced zero
    chunks at ingest time.

    Used by ``GET /v1/base/pages/{path}`` to let an agent that hit a
    chunk via ``/v1/retrieve`` fetch the full page body and align hit
    chunks back onto it via ``Hit.chunk_id`` / ``Hit.seq``.

    The W (wisdom) layer is reachable here via the standard
    ``documents``-table path once PR2 of the 0.3.0 wisdom refactor wires
    wisdom files into ``documents``. PR1 leaves wisdom unreachable via
    ``read_page`` because no wisdom files are indexed yet.
    """
    # Reject obviously-malformed paths up front so a bare ``Path()``
    # call later doesn't surface as a 500 (``\x00`` raises ValueError on
    # Linux). Empty / whitespace-only also can't legitimately match a
    # document.
    if not path or not path.strip() or "\x00" in path:
        raise PageNotFound(path)

    cfg, base_root, storage = await _with_storage(root)
    del cfg
    try:
        # ``_doc_id_for`` is deterministic over ``(layer, normalize_path(path))``,
        # and ``doc_id`` is the PK on ``documents``. Probing each
        # registered layer turns the lookup into N indexed point
        # queries — versus a full-table scan if we went via
        # ``list_documents`` + Python filter. Inactive docs are excluded
        # so the read-by-path policy matches the list endpoint's
        # ``active=True`` default.
        match: DocumentRecord | None = None
        for layer in (Layer.SOURCE, Layer.KNOWLEDGE, Layer.WISDOM):
            candidate = await storage.get_document(_doc_id_for(layer, path))
            if candidate is not None and candidate.active:
                match = candidate
                break
        if match is None:
            raise PageNotFound(path)
        chunks = await storage.list_chunks(match.doc_id)
        page_assets = await _collect_page_assets(storage, chunks)
    finally:
        await storage.close()

    try:
        abs_path = _assert_within(base_root, base_root / match.path)
    except ValueError as e:
        # Defence in depth: a doc registered with a path that escapes
        # the base root is corruption — refuse to read.
        raise PageNotFound(path) from e

    # File I/O + parsing is sync; offload so a slow disk / large file
    # doesn't stall the event loop alongside other in-flight requests
    # (retrieve, query stream). ``body_hash=None`` signals a parse
    # failure (e.g. user broke the YAML front-matter externally) — the
    # natural hash-mismatch path then drops anchors instead of 500-ing
    # the route.
    def _read_and_parse() -> tuple[str, str | None, dict[str, Any]]:
        if not abs_path.is_file():
            # Document row exists but the file is gone (mid-flight
            # delete, or an inactive doc whose file was removed).
            raise PageNotFound(path)
        try:
            parsed = parse_any(abs_path, rel_path=match.path)
            return parsed.body, parsed.hash, parsed.frontmatter
        except Exception:
            # Parse failure (e.g. user broke the YAML front-matter
            # externally) — degrade gracefully: serve the raw text, drop
            # anchors via ``body_hash=None``, and surface ``{}`` for
            # frontmatter so callers don't have to ``None``-check.
            return abs_path.read_text(encoding="utf-8"), None, {}

    body, body_hash, page_frontmatter = await asyncio.to_thread(_read_and_parse)

    # If the file was edited (or its front-matter broken) since ingest,
    # the indexed chunk offsets no longer line up with the current
    # parsed body — silently serving stale anchors would produce
    # off-by-N slicing in agent callers. Drop them and let the caller
    # re-ingest.
    anchors_valid = body_hash is not None and body_hash == match.hash
    anchors = (
        [
            PageAnchor(chunk_id=c.chunk_id, seq=c.seq, start=c.start, end=c.end)
            for c in chunks
            if c.chunk_id is not None
        ]
        if anchors_valid
        else []
    )
    return PageReadResult(
        doc_id=match.doc_id,
        path=match.path,
        layer=match.layer,
        title=match.title,
        body=body,
        anchors=anchors,
        assets=page_assets,
        frontmatter=page_frontmatter,
    )


async def read_asset(
    root: str | Path | None, asset_id: str
) -> tuple[Path, AssetRecord]:
    """Resolve an ``asset_id`` to its on-disk path + :class:`AssetRecord`.

    Raises :class:`AssetNotFound` on unknown id, ``stored_path`` that
    escapes the configured assets dir (DB tampering / migration drift),
    or a vanished file. The single exception keeps the route's 404
    uniform so existing ids can't be probed.
    """
    cfg, base_root, storage = await _with_storage(root)
    try:
        record = await storage.get_asset(asset_id)
    finally:
        await storage.close()
    if record is None:
        raise AssetNotFound(asset_id)

    try:
        abs_path = _assert_within(
            base_root / cfg.assets.dir, base_root / record.stored_path
        )
    except ValueError as e:
        raise AssetNotFound(asset_id) from e

    if not abs_path.is_file():
        raise AssetNotFound(asset_id)

    return abs_path, record


async def list_links(
    root: str | Path | None,
    path: str,
    *,
    direction: LinkDirection = "both",
    limit: int | None = None,
) -> PageLinksResult:
    """Return the page's K-layer link neighbourhood without reading body.

    Companion to :func:`read_page` for graph traversal: lets an agent ask
    "which pages does this page link to" / "which pages link to this
    page" without scanning the body for ``[[wikilink]]`` syntax. The
    answers come from the ``links`` table that ``_persist_knowledge_page``
    keeps in sync on every synth — so an outgoing edge here is exactly
    the edge that survived resolve_links (fuzzy match, anchor
    preservation, collision-refuse). ``direction`` filters which lists
    are populated; ``limit`` caps each list independently so a hub page
    with many edges on both sides doesn't see one side starved.

    Index-driven path safety, same as ``read_page``: only paths present
    in the ``documents`` table are reachable, and inbound src_doc_ids
    that don't resolve to a current document row are dropped (defence
    against orphan edges left by a deactivated doc).
    """
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    if not path or not path.strip() or "\x00" in path:
        raise PageNotFound(path)

    cfg, _root, storage = await _with_storage(root)
    del cfg
    try:
        match: DocumentRecord | None = None
        for layer in (Layer.SOURCE, Layer.KNOWLEDGE, Layer.WISDOM):
            candidate = await storage.get_document(_doc_id_for(layer, path))
            if candidate is not None and candidate.active:
                match = candidate
                break
        if match is None:
            raise PageNotFound(path)

        outgoing: list[OutgoingLink] = []
        incoming: list[IncomingLink] = []

        if direction in ("out", "both"):
            edges_out = await storage.links_from(match.doc_id)
            # Filter to dst paths that resolve to an active document.
            # Without this, bare URLs, markdown links to non-indexed
            # files, and pages-deactivated-since-synth leak into
            # ``outgoing[]`` and the graph-hop contract breaks — the
            # caller would 404 trying to follow ``dst_path`` back into
            # ``GET /v1/base/pages/{path}``. dst_path doesn't carry its
            # layer so we probe both candidate doc_ids in one batch.
            candidate_ids: list[str] = []
            for e in edges_out:
                candidate_ids.append(_doc_id_for(Layer.SOURCE, e.dst_path))
                candidate_ids.append(_doc_id_for(Layer.KNOWLEDGE, e.dst_path))
                candidate_ids.append(_doc_id_for(Layer.WISDOM, e.dst_path))
            dst_docs = {
                d.doc_id: d
                for d in await storage.get_documents(candidate_ids)
            }
            resolved_out: list[OutgoingLink] = []
            for e in edges_out:
                dst = (
                    dst_docs.get(_doc_id_for(Layer.SOURCE, e.dst_path))
                    or dst_docs.get(_doc_id_for(Layer.KNOWLEDGE, e.dst_path))
                    or dst_docs.get(_doc_id_for(Layer.WISDOM, e.dst_path))
                )
                if dst is None or not dst.active:
                    continue
                resolved_out.append(
                    OutgoingLink(
                        dst_path=e.dst_path,
                        link_type=e.link_type,
                        anchor=e.anchor,
                        line=e.line,
                    )
                )
            if limit is not None:
                resolved_out = resolved_out[:limit]
            outgoing = resolved_out

        if direction in ("in", "both"):
            edges_in = await storage.links_to(match.path)
            # Batch-fetch src docs in one round trip — without this a
            # hub page with N inbound edges costs N sequential
            # ``get_document`` calls. Orphan inbound edges (src doc
            # deactivated / deleted) are dropped: agents can't follow
            # them anyway.
            src_docs = {
                d.doc_id: d
                for d in await storage.get_documents(
                    e.src_doc_id for e in edges_in
                )
            }
            resolved_inc: list[IncomingLink] = []
            for edge in edges_in:
                src_doc = src_docs.get(edge.src_doc_id)
                if src_doc is None or not src_doc.active:
                    continue
                resolved_inc.append(
                    IncomingLink(
                        src_doc_id=edge.src_doc_id,
                        src_path=src_doc.path,
                        link_type=edge.link_type,
                        anchor=edge.anchor,
                        line=edge.line,
                    )
                )
            if limit is not None:
                resolved_inc = resolved_inc[:limit]
            incoming = resolved_inc
    finally:
        await storage.close()

    return PageLinksResult(path=match.path, outgoing=outgoing, incoming=incoming)


async def read_provenance(
    root: str | Path | None,
    path: str,
    *,
    direction: ProvenanceDirection = "both",
    limit: int | None = None,
) -> PageProvenanceResult:
    """Return the page's K↔D provenance neighbourhood.

    Provenance is the K-page → D-source attribution recorded in a
    K-page's ``sources:`` frontmatter, reconciled into the dedicated
    ``provenance`` storage table on every ``_persist_knowledge_page``. It is a
    different edge from ``[[wikilink]]`` (which lives in the body and the
    ``links`` table — see :func:`list_links`); the two are deliberately
    not unified so graph-leg retrieval and ``orphan_page`` /
    ``broken_wikilink`` lint stay clean. See
    ``docs/adr/0001-provenance-as-separate-edge.md``.

    ``direction``:
      * ``"out"`` — populates ``derived_from`` only (the K-page's
        forward attribution; meaningful for ``Layer.KNOWLEDGE`` paths,
        always empty for ``Layer.SOURCE``).
      * ``"in"`` — populates ``derived_pages`` only (every K-page
        whose ``sources:`` claims this path; meaningful for
        ``Layer.SOURCE`` paths, always empty for ``Layer.KNOWLEDGE``).
      * ``"both"`` — populates both lists; the empty side is the answer
        for the path's layer.

    Forward entries are returned **faithfully**, including ones whose
    ``source_path`` does not currently resolve to an active
    ``Layer.SOURCE`` document — those carry ``resolved=False`` so agents
    can detect provenance drift (a user-deleted source, a frontmatter
    typo) instead of having storage silently swallow them.

    Index-driven path safety, same as ``list_links``: only paths present
    in the ``documents`` table are reachable; reverse-side ``src_doc_id``
    rows that no longer resolve to an active document are dropped (the
    K-page was hard-deleted or deactivated).

    ``limit`` caps each list independently so a hub source with many
    derived pages doesn't starve the forward side and vice versa.
    """
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    if not path or not path.strip() or "\x00" in path:
        raise PageNotFound(path)

    cfg, _root, storage = await _with_storage(root)
    del cfg
    try:
        match: DocumentRecord | None = None
        for layer in (Layer.SOURCE, Layer.KNOWLEDGE, Layer.WISDOM):
            candidate = await storage.get_document(_doc_id_for(layer, path))
            if candidate is not None and candidate.active:
                match = candidate
                break
        if match is None:
            raise PageNotFound(path)

        derived_from: list[ProvenanceSource] = []
        derived_pages: list[DerivedPage] = []

        if direction in ("out", "both"):
            edges_out = await storage.provenance_from(match.doc_id)
            # Batch-resolve the source side. Forward edges from a
            # K-page point at Layer.SOURCE only (we set the layer prefix
            # explicitly when building the candidate doc_id). For a
            # SOURCE-layer path, ``provenance_from`` returns [] by
            # construction (nothing inserts SOURCE → SOURCE rows), so
            # forward leg is naturally empty there.
            candidate_ids = [
                _doc_id_for(Layer.SOURCE, e.source_path) for e in edges_out
            ]
            src_docs = {
                d.doc_id: d
                for d in await storage.get_documents(candidate_ids)
            }
            for e in edges_out:
                src = src_docs.get(_doc_id_for(Layer.SOURCE, e.source_path))
                if src is not None and src.active:
                    derived_from.append(
                        ProvenanceSource(
                            source_path=e.source_path,
                            doc_id=src.doc_id,
                            title=src.title,
                            resolved=True,
                        )
                    )
                else:
                    # Dangling — source deleted, renamed, or never
                    # ingested. Surfaced faithfully so agents can
                    # detect provenance drift; storage already filters
                    # nothing here.
                    derived_from.append(
                        ProvenanceSource(
                            source_path=e.source_path,
                            doc_id=None,
                            title=None,
                            resolved=False,
                        )
                    )
            if limit is not None:
                derived_from = derived_from[:limit]

        # Reverse leg is meaningful only for ``Layer.SOURCE`` paths.
        # ``storage.provenance_to`` is layer-agnostic and keyed by
        # ``source_path_key``, so a WIKI path whose key happens to match
        # a malformed K-page's frontmatter entry (e.g., a K-page that
        # accidentally lists ``knowledge/...`` in its ``sources:``) would
        # otherwise come back as a ``derived_pages`` entry — violating
        # the documented "WIKI paths have empty reverse provenance"
        # contract and letting agents treat a K-page as a D-source.
        # The forward leg already marks such malformed entries
        # ``resolved=False``; this gate keeps the reverse leg honest.
        if direction in ("in", "both") and match.layer == Layer.SOURCE:
            edges_in = await storage.provenance_to(match.path_key)
            src_docs_in = {
                d.doc_id: d
                for d in await storage.get_documents(
                    e.src_doc_id for e in edges_in
                )
            }
            for edge in edges_in:
                src_doc = src_docs_in.get(edge.src_doc_id)
                # Orphan rows (K-page deactivated / hard-deleted) are
                # dropped — agents can't follow them anyway.
                # ``delete_document`` cascades provenance, so the only
                # way to hit this is a deactivate without delete.
                if src_doc is None or not src_doc.active:
                    continue
                derived_pages.append(
                    DerivedPage(
                        doc_id=src_doc.doc_id,
                        path=src_doc.path,
                        title=src_doc.title,
                    )
                )
            if limit is not None:
                derived_pages = derived_pages[:limit]
    finally:
        await storage.close()

    return PageProvenanceResult(
        path=match.path,
        derived_from=derived_from,
        derived_pages=derived_pages,
    )


def _read_doc_body_for_graph(base_root: Path, doc: DocumentRecord) -> str | None:
    """Read + parse a document body for graph link extraction.

    Returns ``None`` (caller skips the doc) when the file is missing,
    unreadable, or the doc's stored ``path`` resolves outside
    ``base_root`` (defence-in-depth against a corrupted documents
    table — same guard ``read_page`` uses, see line ~1782). ``parse_any``
    strips YAML front-matter so the link parser doesn't see ``[[X]]``
    references inside front-matter blocks. On parse failure we fall
    back to the raw text — losing front-matter stripping is acceptable
    here, while losing the page entirely would silently degrade the
    graph.
    """
    try:
        abs_path = _assert_within(base_root, base_root / doc.path)
    except ValueError:
        return None
    if not abs_path.is_file():
        return None
    try:
        parsed = parse_any(abs_path, rel_path=doc.path)
        return parsed.body
    except Exception:
        try:
            return abs_path.read_text(encoding="utf-8")
        except OSError:
            return None


_BODY_HASH_MISSING = "missing"


def _compute_base_revision(
    docs_sorted_by_path: list[DocumentRecord],
    body_hashes: dict[str, str],
) -> str:
    """Content-addressed digest over everything that influences the
    graph response, so a client cache keyed on this can't serve a
    stale graph.

    Includes per doc: ``path``, ``title``, ``layer``, ``mtime``,
    ``active`` — every field that surfaces on ``GraphNode`` or affects
    wikilink resolution (``title_to_paths``) — plus the *current*
    on-disk body sha256 (or ``_BODY_HASH_MISSING`` for vanished
    files). Hashing only ``DocumentRecord.mtime`` would miss user
    edits between ingests; hashing only the body would miss
    title/layer/active metadata changes that re-ingest persists
    without touching the bytes.

    Caller passes ``docs`` already sorted by ``path`` so the digest is
    deterministic without a second sort. Not a cryptographic
    commitment to response content; clients that need that should
    hash the response themselves.
    """
    h = hashlib.sha256()
    for d in docs_sorted_by_path:
        h.update(d.path.encode("utf-8"))
        h.update(b"|")
        h.update((d.title or "").encode("utf-8"))
        h.update(b"|")
        h.update(d.layer.value.encode("ascii"))
        h.update(b"|")
        h.update(f"{d.mtime}".encode("ascii"))
        h.update(b"|")
        h.update(body_hashes.get(d.path, _BODY_HASH_MISSING).encode("ascii"))
        h.update(b"|")
        h.update(b"1" if d.active else b"0")
        h.update(b"\n")
    return h.hexdigest()


async def list_graph(
    root: str | Path | None,
    *,
    active: bool | None = True,
) -> GraphResult:
    """Return the full base graph in one read-only pass — every
    document is a node, every resolvable wikilink / cross-page markdown
    link is an edge, every broken wikilink lands in ``unresolved``.

    Engine for ``GET /v1/base/graph`` (issue #89): replaces the
    web-side workaround of looping ``GET /v1/base/pages/{path}`` and
    re-doing wikilink resolution in the browser. Re-parses bodies on
    every call (no edge cache), keyed off ``base_revision`` so a client
    can short-circuit unchanged graphs.

    Resolution semantics match the K-layer link store: exact
    title → fuzzy normalize → collision-refuse (see
    ``domains/knowledge/links.resolve_links``). URLs and bare URLs
    parse out of the body but are intentionally dropped from both
    ``edges`` and ``unresolved`` — they're out-of-graph by definition.
    Markdown links count as edges only when their href matches a
    document path in the base, so out-of-base relative paths fall
    through silently.

    ``active`` matches the ``list_pages`` convention: ``True`` (default)
    keeps only active docs in the node set so a wikilink to a
    deactivated page is reported as unresolved (matches the
    user-visible "this page is hidden, treat the link as broken"
    expectation); ``None`` includes both active and deactivated docs;
    ``False`` returns only deactivated. The resolution index is
    always over the response's own node set, so cross-mode the rule
    holds: an edge endpoint is always reachable via
    ``GET /v1/base/pages/{path}``.

    No writes: ``status()`` counts before / after a call are equal.
    """
    cfg, base_root, storage = await _with_storage(root)
    del cfg
    try:
        unsorted_docs = await storage.list_documents(layer=None, active=active)
    finally:
        await storage.close()

    # Defence-in-depth: a corrupted documents row whose path escapes
    # the base (absolute path, ``..`` traversal) would otherwise let
    # graph mode read + hash + emit nodes for arbitrary off-base files.
    # ``read_page`` rejects the same case via ``_assert_within``; do
    # the same here, before any node/edge work, by silently dropping
    # the offending docs.
    safe_docs: list[DocumentRecord] = []
    for d in unsorted_docs:
        try:
            _assert_within(base_root, base_root / d.path)
        except ValueError:
            continue
        safe_docs.append(d)

    # Sort once, share with both ``_compute_base_revision`` and the
    # nodes-builder loop below — saves an O(n log n) pass on large bases.
    docs = sorted(safe_docs, key=lambda x: x.path)

    # Resolution universe == node set: a wikilink can only resolve to a
    # node that's actually in the response. Build a path-list per title
    # (NOT a path-per-title dict) so two docs sharing an exact title —
    # extremely common when graph mode mixes a source page and its
    # synthesized knowledge page (both titled e.g. "Elon Musk") — can be
    # detected and collision-refused. Karpathy's rule: wrong-merge is
    # irreversible damage, missed-resolve is a fixable lint warning.
    title_to_paths: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        if d.title and d.path not in title_to_paths[d.title]:
            title_to_paths[d.title].append(d.path)
    # Inline the fuzzy index instead of using ``build_fuzzy_index``:
    # we need *every* path of an ambiguous title to land in its
    # normalize bucket, so a third doc whose title normalizes to the
    # same key triggers a 3+-way collision-refuse instead of becoming
    # the sole fuzzy candidate. Feeding ``build_fuzzy_index`` only
    # the unambiguous titles would silently bind ``[[Foos]]`` to a
    # ``Foo!`` page when two ``Foo`` pages also exist.
    fuzzy_index: dict[str, list[str]] = defaultdict(list)
    for title, paths in title_to_paths.items():
        key = normalize_base(title)
        if not key:
            continue
        for p in paths:
            if p not in fuzzy_index[key]:
                fuzzy_index[key].append(p)
    # Path lookup index: markdown links are spelled by the user
    # (``[B](Wiki/Foo.md)``) and may not byte-match the canonical
    # ``DocumentRecord.path`` on case-insensitive / NFC-normalizing
    # filesystems. Route every membership check through ``path_key``
    # (NFC + casefold) so a valid edge isn't dropped just because the
    # author typed mixed case.
    path_key_to_path: dict[str, str] = {d.path_key: d.path for d in docs}
    node_path_set = set(path_key_to_path.values())

    edge_acc: dict[tuple[str, str, str, str | None], dict[str, Any]] = {}
    unresolved_acc: dict[tuple[str, str, str | None], int] = {}
    inbound: defaultdict[str, set[str]] = defaultdict(set)
    outbound: defaultdict[str, set[str]] = defaultdict(set)
    body_hashes: dict[str, str] = {}

    # Bound parallelism: each ``to_thread`` hop costs ~0.1-1ms of
    # event-loop overhead even when the read is fast, so on a 1k-doc
    # base a serial loop wastes ~1s before any real disk work. 8 keeps
    # the thread pool warm without flooding it; tune later if needed.
    sem = asyncio.Semaphore(8)

    async def _read(d: DocumentRecord) -> tuple[DocumentRecord, str | None]:
        async with sem:
            body = await asyncio.to_thread(
                _read_doc_body_for_graph, base_root, d
            )
        return d, body

    for doc, body in await asyncio.gather(*(_read(d) for d in docs)):
        if body is None:
            body_hashes[doc.path] = _BODY_HASH_MISSING
            continue
        body_hashes[doc.path] = hashlib.sha256(body.encode("utf-8")).hexdigest()
        for parsed in parse_links(body):
            if parsed.kind is LinkType.URL:
                # URLs are out-of-graph by design — neither an edge nor
                # a missing wikilink. Issue #89 v1 does not surface
                # external links at all.
                continue
            if parsed.kind is LinkType.WIKILINK:
                exact = title_to_paths.get(parsed.target, [])
                if len(exact) == 1:
                    target_path: str | None = exact[0]
                elif len(exact) >= 2:
                    # Ambiguous exact match — refuse and surface as
                    # unresolved instead of guessing one of the
                    # collisions.
                    target_path = None
                else:
                    key = normalize_for_match(parsed.target)
                    cands = fuzzy_index.get(key, []) if key else []
                    target_path = cands[0] if len(cands) == 1 else None
                if target_path is None:
                    ukey = (doc.path, parsed.target, parsed.anchor)
                    unresolved_acc[ukey] = unresolved_acc.get(ukey, 0) + 1
                    continue
                target_text = parsed.target
            else:  # MARKDOWN
                # Normalize the user-typed href through ``path_key`` so
                # mixed-case / NFD spellings match the canonical doc
                # path on case-insensitive or NFC-normalizing
                # filesystems.
                canonical = path_key_to_path.get(normalize_path(parsed.target))
                if canonical is None:
                    continue
                target_path = canonical
                target_text = parsed.target
            if target_path not in node_path_set:
                # Out-of-base href, or wikilink that resolved to a
                # deactivated doc when active=True. Drop quietly so the
                # graph-hop contract holds — every edge endpoint is a
                # node the client can fetch via /v1/base/pages/{path}.
                continue
            ekey = (doc.path, target_path, target_text, parsed.anchor)
            entry = edge_acc.get(ekey)
            if entry is None:
                edge_acc[ekey] = {"weight": 1, "type": parsed.kind}
            else:
                entry["weight"] += 1
            outbound[doc.path].add(target_path)
            inbound[target_path].add(doc.path)

    nodes = [
        GraphNode(
            id=d.path,
            path=d.path,
            title=d.title,
            layer=d.layer,
            active=d.active,
            mtime=d.mtime,
            inbound=len(inbound[d.path]),
            outbound=len(outbound[d.path]),
        )
        for d in docs  # already sorted by path above
    ]
    edges = [
        GraphEdge(
            id=f"{src}->{tgt}",
            source=src,
            target=tgt,
            type=meta["type"],
            target_text=ttext,
            anchor=anchor,
            weight=meta["weight"],
        )
        # Coerce ``anchor=None`` to ``""`` for the sort key so a mixed
        # set of anchored / unanchored edges between the same pair
        # doesn't blow up with ``str < NoneType`` TypeError.
        for (src, tgt, ttext, anchor), meta in sorted(
            edge_acc.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3] or "")
        )
    ]
    unresolved = [
        GraphUnresolvedLink(
            source=src, target_text=ttext, anchor=anchor, count=cnt
        )
        for (src, ttext, anchor), cnt in sorted(
            unresolved_acc.items(),
            key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or ""),
        )
    ]
    return GraphResult(
        base_revision=_compute_base_revision(docs, body_hashes),
        generated_at=datetime.now(UTC),
        nodes=nodes,
        edges=edges,
        unresolved=unresolved,
        stats=GraphStats(
            node_count=len(nodes),
            edge_count=len(edges),
            unresolved_count=sum(unresolved_acc.values()),
        ),
    )


async def retrieve(
    q: str,
    path: str | Path | None = None,
    *,
    limit: int = 5,
    embedder: EmbeddingProvider | None = None,
    multimodal_embedder: MultimodalEmbeddingProvider | None = None,
    reporter: ProgressReporter | None = None,
) -> RetrieveResult:
    """Hybrid-search the knowledge base and return chunks + page-level refs only.

    Companion to ``query`` for retrieval-only consumers (typically AI
    agents that intend to assemble their own answer using their own
    LLM): runs the fusion + multimodal pipeline via ``_retrieve_inner``
    and stops there. dikw-core no longer ships an in-engine query verb
    (PR-1 removed it); ``retrieve`` is the sole knowledge-access entry
    point from the engine side.

    ``reporter`` (optional) emits a ``retrieval_done`` partial; the
    route layer wraps this in a ``final{result}`` event for the
    ``POST /v1/retrieve`` NDJSON wire.
    """
    cfg, _root, storage = await _with_storage(path)
    owned_mm: MultimodalEmbeddingProvider | None = None
    try:
        hits, owned_mm = await _retrieve_inner(
            storage,
            cfg,
            q,
            limit=limit,
            embedder=embedder,
            multimodal_embedder=multimodal_embedder,
            reporter=reporter,
        )
        return RetrieveResult(chunks=hits, page_refs=_build_page_refs(hits))
    finally:
        if owned_mm is not None and hasattr(owned_mm, "aclose"):
            await owned_mm.aclose()
        await storage.close()


# ---- Phase 2: synthesize + lint -----------------------------------------


async def synthesize(
    path: str | Path | None = None,
    *,
    force_all: bool = False,
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    reporter: ProgressReporter | None = None,
) -> SynthReport:
    """Turn source docs into K-layer knowledge pages via the configured LLM.

    By default only source docs that have never been synthesised are
    processed; pass ``force_all=True`` to re-synthesise every source.
    Embedding of new knowledge pages is skipped when ``embedder`` is ``None``.

    ``reporter`` (optional) receives one ``progress`` event per source
    document processed for server-driven task wrappers.
    """
    cfg, root, storage = await _with_storage(path)
    _reporter: ProgressReporter = reporter or NoopReporter()
    try:
        _llm = llm or build_llm(cfg.provider, base_root=root)

        text_version_id: int | None = None
        text_embed_model = cfg.provider.embedding_model
        if embedder is not None:
            # Synthesize must NOT register a new embed version: it only
            # writes knowledge-page chunks, so flipping active here would strand
            # source-chunk vectors in the now-inactive table and gut dense
            # retrieval. Re-embedding the full corpus belongs to ingest.
            try:
                active_text = await storage.get_active_embed_version(modality="text")
            except NotSupported:
                active_text = None
            if active_text is not None and active_text.version_id is not None:
                text_version_id = active_text.version_id
                text_embed_model = active_text.model
            else:
                embedder = None  # no active text version → nothing to embed against

        sources = list(await storage.list_documents(layer=Layer.SOURCE, active=True))
        already: set[str] = set()
        if not force_all:
            # ``synth_source_done`` is the post-fan-out source-completion
            # marker: per-page ``synth`` log rows can no longer be used
            # because (a) a fan-out source with one failed group + one
            # successful group writes a ``dst`` row but is NOT done, and
            # (b) a source with a legal zero-page response writes no
            # ``dst`` row at all but IS done.
            #
            # Upgrade compatibility uses a sentinel row
            # (``src=_LEGACY_BACKFILL_SENTINEL``) to record "this base has
            # already gone through the legacy-row backfill at least once".
            # Without the sentinel we can't distinguish a *legacy* dst row
            # from a *post-fan-out crash* dst row — and treating the latter
            # as legacy would silently mark crashed sources done. The
            # sentinel is written unconditionally on the first post-fan-out
            # run so any later crash leaves us in the "sentinel already
            # exists, do not backfill" state.
            has_legacy_backfill_sentinel = False
            legacy_dst_sources: set[str] = set()
            for entry in await storage.list_knowledge_log():
                if entry.action == "synth_source_done":
                    if entry.src == _LEGACY_BACKFILL_SENTINEL:
                        has_legacy_backfill_sentinel = True
                    elif entry.src:
                        already.add(entry.src)
                elif entry.action == "synth" and entry.src and entry.dst:
                    legacy_dst_sources.add(entry.src)
            if not has_legacy_backfill_sentinel:
                ts = time.time()
                # Write the sentinel FIRST so even if the backfill loop
                # below crashes we never re-enter the backfill arm.
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=ts,
                        action="synth_source_done",
                        src=_LEGACY_BACKFILL_SENTINEL,
                        note=(
                            "fan-out pipeline initialised — subsequent runs "
                            "will not backfill legacy per-page synth rows"
                        ),
                    )
                )
                for src_path in sorted(legacy_dst_sources):
                    await storage.append_knowledge_log(
                        KnowledgeLogEntry(
                            ts=ts,
                            action="synth_source_done",
                            src=src_path,
                            note="backfilled from legacy per-page synth rows",
                        )
                    )
                already |= legacy_dst_sources

        report = SynthReport()
        tmpl = prompts.load("synthesize")
        persisted_any = False
        total_sources = len(sources)

        for idx, src in enumerate(sources, start=1):
            _reporter.cancel_token().raise_if_cancelled()
            report = _sr_replace(report, candidates=report.candidates + 1)
            if src.path in already:
                report = _sr_replace(report, skipped=report.skipped + 1)
                await _reporter.progress(
                    phase="synth",
                    current=idx,
                    total=total_sources,
                    detail={"path": src.path, "outcome": "skipped"},
                )
                continue

            parsed = _read_source_parsed(root, src)
            if parsed is None:
                report = _sr_replace(report, errors=report.errors + 1)
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=time.time(),
                        action="synth",
                        src=src.path,
                        note="source body missing on disk",
                    )
                )
                await _reporter.progress(
                    phase="synth",
                    current=idx,
                    total=total_sources,
                    detail={"path": src.path, "outcome": "missing_body"},
                )
                continue

            # If the source on disk drifted since ingest (user edited it
            # without re-running ``dikw client ingest``), the cached chunk offsets
            # would slice the new body at stale boundaries — silently
            # dropping appended content and marking the source done. Bail
            # out with a clear log and let the user re-ingest.
            if parsed.hash != src.hash:
                report = _sr_replace(report, errors=report.errors + 1)
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=time.time(),
                        action="synth",
                        src=src.path,
                        note=(
                            "source body changed since last ingest — "
                            "re-run `dikw client ingest` before `dikw client synth`"
                        ),
                    )
                )
                await _reporter.progress(
                    phase="synth",
                    current=idx,
                    total=total_sources,
                    detail={"path": src.path, "outcome": "stale_chunks"},
                )
                continue

            body = parsed.body
            src_chunks = await storage.list_chunks(
                _doc_id_for(Layer.SOURCE, src.path)
            )
            outcome = await _synth_pages_from_source(
                llm=_llm,
                template=tmpl,
                cfg=cfg,
                source_path=src.path,
                source_body=body,
                chunks=src_chunks,
                cancel=_reporter.cancel_token(),
                storage=storage,
                text_version_id=text_version_id,
                force_all=force_all,
                reporter=_reporter,
            )
            report = _sr_replace(
                report,
                groups_processed=report.groups_processed + outcome.groups_processed,
                errors=report.errors + outcome.parse_errors,
            )
            for note in outcome.log_notes:
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=time.time(), action="synth", src=src.path, note=note
                    )
                )

            if outcome.groups_processed == 0:
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=time.time(),
                        action="synth",
                        src=src.path,
                        note="no chunks to synthesise from",
                    )
                )
                # Source is "done" — re-running synth on a source that
                # has no chunks would just hit the same dead-end. Mark
                # it complete so default ``synth`` skips it next time.
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=time.time(),
                        action="synth_source_done",
                        src=src.path,
                    )
                )
                report = _sr_replace(
                    report, sources_processed=report.sources_processed + 1
                )
                await _reporter.progress(
                    phase="synth",
                    current=idx,
                    total=total_sources,
                    detail={"path": src.path, "outcome": "no_chunks"},
                )
                continue

            deduped = dedup_pages_by_slug(
                outcome.pages, strategy=cfg.synth.slug_dedup
            )

            # Build the title→path index ONCE for this batch and seed it
            # with the deduped pages — without that seeding, page A → page B
            # wikilinks fan-out produces from the same source would only
            # resolve after B was already upserted.
            title_to_path: dict[str, str] = {}
            if deduped:
                for d in await storage.list_documents(
                    layer=Layer.KNOWLEDGE, active=True
                ):
                    if d.title and d.title not in title_to_path:
                        title_to_path[d.title] = d.path
                for page in deduped:
                    title_to_path.setdefault(page.title, page.path)
            fuzzy_index = build_fuzzy_index(title_to_path) if deduped else None

            created_for_src = 0
            updated_for_src = 0
            for page in deduped:
                pre_existing = await storage.get_document(
                    _doc_id_for(Layer.KNOWLEDGE, page.path)
                )
                write_page(root, page)
                page_unresolved = await _persist_knowledge_page(
                    storage=storage,
                    root=root,
                    page=page,
                    embedder=embedder,
                    embedding_model=text_embed_model,
                    text_version_id=text_version_id,
                    cjk_tokenizer=cfg.retrieval.cjk_tokenizer,
                    title_to_path=title_to_path,
                    fuzzy_index=fuzzy_index,
                )
                if page_unresolved:
                    report = _sr_replace(
                        report,
                        unresolved_wikilinks=report.unresolved_wikilinks
                        + page_unresolved,
                    )
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=time.time(),
                        action="synth",
                        src=src.path,
                        dst=page.path,
                    )
                )
                persisted_any = True
                if pre_existing is None:
                    created_for_src += 1
                    report = _sr_replace(report, created=report.created + 1)
                else:
                    updated_for_src += 1
                    report = _sr_replace(report, updated=report.updated + 1)

            persisted_for_src = created_for_src + updated_for_src
            report = _sr_replace(
                report, sources_processed=report.sources_processed + 1
            )
            # Mark the source as fully synthesised so default ``synth``
            # skips it next run. Skip the marker when any group raised
            # a hard ``SynthesisError`` — those failures should be
            # retried (the LLM may produce parseable output next time).
            # Partial-parse outcomes don't count: the surviving pages
            # were persisted, retrying would just hit the same partial
            # response and re-emit the warning to ``knowledge_log``.
            if outcome.parse_errors == 0:
                await storage.append_knowledge_log(
                    KnowledgeLogEntry(
                        ts=time.time(),
                        action="synth_source_done",
                        src=src.path,
                    )
                )
            # ``outcome`` string keeps the pre-fan-out vocabulary so
            # client-side event consumers don't need to learn new strings.
            if persisted_for_src == 0:
                outcome_str = "no_pages"
            elif created_for_src > 0:
                outcome_str = "created"
            else:
                outcome_str = "updated"
            await _reporter.progress(
                phase="synth",
                current=idx,
                total=total_sources,
                detail={
                    "path": src.path,
                    "outcome": outcome_str,
                    "pages_persisted": persisted_for_src,
                    "groups": outcome.groups_processed,
                },
            )

        # Refresh the human-readable views after the batch so a partial run
        # still leaves the knowledge base internally consistent.
        if persisted_any or not (root / "knowledge" / "index.md").exists():
            regenerate_index(root, updated=now_iso())
        entries = await storage.list_knowledge_log()
        render_log(root, entries, updated=now_iso())

        return report
    finally:
        await storage.close()


async def lint(path: str | Path | None = None) -> LintReport:
    """Run the K-layer hygiene checker."""
    _cfg, root, storage = await _with_storage(path)
    try:
        return await run_lint(storage, root=root)
    finally:
        await storage.close()


async def lint_propose(
    path: str | Path | None = None,
    *,
    rule: LintKind | None = None,
    limit: int = 10,
    llm: Any = None,
    embedder: Any = None,
    enable_llm: bool = False,
    reporter: ProgressReporter | None = None,
) -> FixProposalReport:
    """Run lint + dispatch fixers, returning a :class:`FixProposalReport`.

    ``enable_llm`` opts into LLM-powered fixer paths: the
    broken_wikilink evidence-backed grounded repair (D-layer hybrid
    search must yield enough evidence before the LLM is asked to write
    a real page; outputs containing ``TODO`` / ``stub page`` /
    ``placeholder`` markers are rejected), the entire non_atomic_page
    splitter, and orphan_page's ``merge_into_existing_page`` strategy.
    When False, propose runs heuristic-only — no LLM call is made and
    pure-heuristic paths (``broken_wikilink`` fuzzy-match,
    ``orphan_page`` delete/link/leaf strategies, ``missing_provenance``)
    still work. The default keeps a ``propose`` invocation cheap and
    deterministic; users opt in via ``--enable-llm``.

    ``llm`` / ``embedder`` are passthrough overrides used by tests; in
    production both are built from ``cfg.provider`` the same way
    :func:`synthesize` / :func:`retrieve` do, so ``$DIKW_*_API_KEY``
    resolution flows through one path. The embedder powers the D-layer
    hybrid evidence search the ``broken_wikilink`` grounded repair
    relies on — without it the search silently degrades to BM25 and
    semantically relevant but lexically distant evidence is missed.
    """
    cfg, root, storage = await _with_storage(path)
    try:
        report = await run_lint(storage, root=root)
        # Build KnowledgePageMeta + path→doc_id index in one pass over the
        # WIKI listing. ``report.page_meta`` already carries the
        # frontmatter slice ``run_lint`` parsed, so the orphan scorer's
        # sources/tags signal lands here without a second disk read.
        knowledge_docs = list(
            await storage.list_documents(layer=Layer.KNOWLEDGE, active=True)
        )
        all_pages: list[KnowledgePageMeta] = []
        path_to_doc_id: dict[str, str] = {}
        for doc in knowledge_docs:
            pm = report.page_meta.get(doc.path)
            all_pages.append(
                KnowledgePageMeta(
                    path=doc.path, title=doc.title,
                    sources=pm.sources if pm else (),
                    tags=pm.tags if pm else (),
                )
            )
            path_to_doc_id[doc.path] = doc.doc_id
        # Skip the LLM + embedder builds entirely on ``--enable-llm
        # False`` so the provider-import + key-lookup cost stays out
        # of heuristic-only propose runs. The embedder powers the
        # D-layer hybrid evidence search the broken_wikilink grounded
        # repair (#83) relies on — without it the search silently
        # degrades to BM25 and semantically relevant but lexically
        # distant evidence is missed.
        _llm: Any = llm
        _embedder: Any = embedder
        if enable_llm:
            if _llm is None:
                _llm = build_llm(cfg.provider, base_root=root)
            if _embedder is None:
                _embedder = build_embedder(cfg.provider)
        ctx = FixerContext(
            storage=storage,
            llm=_llm,
            embedding=_embedder,
            base_root=root,
            all_pages=all_pages,
            enable_llm=enable_llm,
            cfg=cfg,
            path_to_doc_id=path_to_doc_id,
        )
        used_reporter: ProgressReporter = reporter or NoopReporter()
        return await run_lint_propose(
            report=report,
            rule=rule,
            limit=limit,
            ctx=ctx,
            reporter=used_reporter,
        )
    finally:
        await storage.close()


async def lint_apply(
    path: str | Path | None = None,
    *,
    proposal_report: FixProposalReport,
    pick: list[int] | None = None,
    skip: list[int] | None = None,
    reporter: ProgressReporter | None = None,
    embedder: EmbeddingProvider | None = None,
) -> ApplyReport:
    """Mutate ``knowledge/`` per a previously-produced proposal report.

    When ``DIKW_EMBEDDING_API_KEY`` is configured (or the caller passes
    ``embedder``), Phase 1 re-embeds every rebuilt page inline so the
    fix is retrievable on return. Without the key, embedding defers to
    the next ``dikw client ingest``'s missing-embedding resume scan —
    the same fallback the W-layer ``no_embed`` write path uses.

    ``pick`` / ``skip`` filter the proposal list by index. Both may be
    set; pick is applied first, then skip removes from that subset.
    """
    cfg, root, storage = await _with_storage(path)
    try:
        used_reporter: ProgressReporter = reporter or NoopReporter()

        # Inline-embed when the user has an embedding key on hand. The
        # OpenAICompatEmbeddings provider only reads the key at embed
        # time, so a missing key wouldn't fail ``build_embedder`` — we
        # check the env up-front to keep apply heuristic-only when the
        # user hasn't configured embeddings yet.
        active_embedder: EmbeddingProvider | None = embedder
        text_version_id: int | None = None
        embedding_model = ""
        if active_embedder is None and os.environ.get("DIKW_EMBEDDING_API_KEY"):
            active_embedder = build_embedder(cfg.provider)
        if active_embedder is not None:
            text_version_id = await _register_text_version(storage, cfg.provider)
            embedding_model = cfg.provider.embedding_model

        return await run_lint_apply(
            proposal_report=proposal_report,
            storage=storage,
            base_root=root,
            pick=pick,
            skip=skip,
            reporter=used_reporter,
            cjk_tokenizer=cfg.retrieval.cjk_tokenizer,
            embedder=active_embedder,
            embedding_model=embedding_model,
            text_version_id=text_version_id,
            embedding_error_retries=cfg.provider.embedding_error_retries,
            embedding_error_retry_backoff_seconds=(
                cfg.provider.embedding_error_retry_backoff_seconds
            ),
        )
    finally:
        await storage.close()


# Per-(base, wisdom-path) locks serialise concurrent writers against the
# same logical wisdom file. Two HTTP submissions for the same
# ``(author, slug)`` would otherwise race: both call ``get_document``
# before either writes, both observe ``existing is None``, both report
# ``created=True``, and the storage row + chunks + on-disk file can end
# up describing different submissions depending on interleaving. The
# task system schedules submissions concurrently, so even single-base
# single-process deployments need this guard.
_WISDOM_WRITE_LOCKS: dict[str, asyncio.Lock] = {}
_WISDOM_WRITE_LOCKS_GUARD = asyncio.Lock()


async def _acquire_wisdom_write_lock(key: str) -> asyncio.Lock:
    """Return (creating if needed) the asyncio.Lock for ``key``.

    The guard lock around the dict avoids two writers racing on the
    ``setdefault``-style check itself. Lock objects accumulate in the
    process for the lifetime of the engine; the cardinality is bounded
    by the number of distinct wisdom paths the base ever sees, which is
    fine for a per-user knowledge base.
    """
    async with _WISDOM_WRITE_LOCKS_GUARD:
        lock = _WISDOM_WRITE_LOCKS.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _WISDOM_WRITE_LOCKS[key] = lock
        return lock


async def write_wisdom_page(
    path: str | Path | None = None,
    *,
    slug: str,
    title: str,
    body: str,
    author: str | None = None,
    status: WisdomStatus | None = None,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    extras: dict[str, object] | None = None,
    no_embed: bool = False,
    embedder: EmbeddingProvider | None = None,
    reporter: ProgressReporter | None = None,
) -> WisdomWriteReport:
    """Create or update a wisdom page at ``wisdom/[<author>/]<slug>.md``.

    Upsert semantics — re-writing the same ``(author, slug)`` overwrites
    the existing file and refreshes its storage row, chunks, embeddings,
    outgoing wikilinks, and provenance edges. The caller is responsible
    for reading the current state first if a no-overwrite contract is
    needed (read ``GET /v1/base/pages/<path>`` and decide before POST).

    ``slug`` and ``author`` are validated to ASCII kebab-case; the
    on-disk path becomes part of the wisdom vault layout, so the engine
    refuses anything that would render awkwardly in Obsidian (spaces,
    uppercase, underscores).

    When ``no_embed`` is true the file is written + chunked + linked
    but vectors are not produced; the next ``dikw client ingest``
    resolves embeddings via the missing-embedding resume scan. With
    ``no_embed`` false the engine builds an embedder from the base
    config — a missing API key or unreachable provider surfaces as the
    embedder factory's error and fails the write (open
    ``no_embed=True`` if the caller wants to author wisdom without an
    embedding provider configured). With a real embedder the page is
    retrievable immediately on return.

    Concurrent writes to the same ``(author, slug)`` are serialised by
    a per-path async lock so two HTTP submissions never both claim
    ``created=True`` against an empty document row.

    Cross-file ``[[wikilink]]`` resolution uses a freshly built
    cross-layer title index (knowledge + wisdom), so a wisdom page linking
    an existing knowledge or wisdom title resolves on this single write. A
    forward reference to a wisdom page that hasn't been written yet
    surfaces on ``WisdomWriteReport.unresolved_wikilinks`` and is
    reconciled on the next ``dikw client ingest``.
    """
    # Lazy import — domains.knowledge.page_index imports schemas + storage
    # primitives, and api.py imports those eagerly already; importing
    # the persist functions eagerly here would re-introduce the same
    # circular risk the existing wisdom-branch (1320 area) avoided by
    # deferring. ``persist_wisdom`` is imported lazily at its single
    # use site below.
    from .domains.knowledge.links import build_title_indexes
    from .domains.wisdom import make_wisdom_path, write_wisdom_file

    # Validate first — never write a file or open storage on a malformed
    # input. ``make_wisdom_path`` re-runs ``validate_kebab`` so a direct
    # Python caller bypassing the Pydantic schema still gets the same
    # invariant.
    logical_path = make_wisdom_path(slug=slug, author=author)
    used_reporter: ProgressReporter = reporter or NoopReporter()

    # Key locks by (canonical base root, logical wisdom path). Resolve
    # the base via ``resolve_base_root`` *before* acquiring the lock so
    # two callers targetting the same base through different aliases
    # (the base dir, the dikw.yml file inside it, a relative path)
    # map to the same lock. ``resolve_base_root`` is a pure path walk
    # + config-file lookup with no side effects, so running it here
    # and again inside ``_with_storage`` is cheap and safe.
    canonical_root = resolve_base_root(path).resolve()
    lock_key = f"{canonical_root}::{logical_path}"
    lock = await _acquire_wisdom_write_lock(lock_key)
    async with lock:
        cfg, root, storage = await _with_storage(path)
        try:
            doc_id = _doc_id_for(Layer.WISDOM, logical_path)
            existing = await storage.get_document(doc_id)
            created = existing is None or not existing.active

            # Resolve embedder: caller-injected wins (tests pass a fake);
            # otherwise build from cfg unless ``no_embed`` defers it. The
            # build_embedder call is the one that touches the env API key —
            # keep it inside the ``not no_embed`` branch so a write that
            # explicitly opted out doesn't fail on a missing key.
            active_embedder: EmbeddingProvider | None = None
            text_version_id: int | None = None
            embedding_model = ""
            if not no_embed:
                active_embedder = embedder
                if active_embedder is None:
                    active_embedder = build_embedder(cfg.provider)
                text_version_id = await _register_text_version(storage, cfg.provider)
                embedding_model = cfg.provider.embedding_model

            # Cooperative cancellation poll: synth/lint/ingest all check
            # the token between work units (api.py:1130 / 1353 / 1589 /
            # 2717), so a wisdom-write task is expected to honour the same
            # contract. Without poll sites, ``TaskManager.cancel`` only
            # lands at the next httpx ``await`` deep inside the embedder.
            used_reporter.cancel_token().raise_if_cancelled()

            # Write the file to disk first. ``persist_page`` re-parses the
            # written file so the stored hash and chunk offsets match what
            # ``read_page`` will compute — same contract as
            # ``_persist_knowledge_page``. ``frontmatter.dumps`` is not always
            # byte-stable on the body, so hashing ``body`` directly and
            # chunking ``body`` would diverge from the read-back parsed body.
            write_wisdom_file(
                root,
                logical_path=logical_path,
                title=title,
                body=body,
                status=status,
                tags=tags,
                sources=sources,
                extras=extras,
            )

            # Build the cross-layer title index that ``persist_wisdom`` uses
            # for ``[[wikilink]]`` resolve. Mirrors the ingest wisdom branch
            # (around line 1330) - include the new page's own NEW title so
            # a self-reference resolves, but EXCLUDE the stored row for
            # ``logical_path`` itself: on an update that changes the page's
            # title, the storage row still holds the old title until
            # ``persist_wisdom`` rewrites it, and pulling that into the index
            # would let ``[[Old Title]]`` in the new body resolve to this
            # same page via the stale title.
            title_docs: list[tuple[str, str]] = []
            for layer_for_index in (Layer.KNOWLEDGE, Layer.WISDOM):
                for d in await storage.list_documents(
                    layer=layer_for_index, active=True
                ):
                    if d.path == logical_path:
                        continue
                    if d.title:
                        title_docs.append((d.title, d.path))
            title_docs.append((title, logical_path))
            title_to_path, fuzzy_index = build_title_indexes(title_docs)

            used_reporter.cancel_token().raise_if_cancelled()
            await used_reporter.progress(
                phase="wisdom_write",
                current=0,
                total=0,
                detail={"path": logical_path, "step": "indexing"},
            )

            # ``persist_wisdom`` writes documents + chunks + FTS + embeddings
            # + links + provenance in sequence. A mid-pipeline failure
            # (embed timeout, link resolve crash, vec_search error) leaves
            # those rows partially updated — the document row + chunks
            # already reflect the new content while outgoing links /
            # provenance edges still point at the old. ``deactivate_document``
            # flips the doc to inactive so the next ``dikw ingest`` resume
            # scan rebuilds the row end-to-end.
            from .domains.wisdom.persist import persist_wisdom

            try:
                persist_result = await persist_wisdom(
                    storage=storage,
                    root=root,
                    path=logical_path,
                    title=title,
                    embedder=active_embedder,
                    embedding_model=embedding_model,
                    text_version_id=text_version_id,
                    cjk_tokenizer=cfg.retrieval.cjk_tokenizer,
                    title_to_path=title_to_path,
                    fuzzy_index=fuzzy_index,
                    retries=cfg.provider.embedding_error_retries,
                    backoff_seconds=cfg.provider.embedding_error_retry_backoff_seconds,
                )
                unresolved = persist_result.unresolved_wikilinks
            except Exception:
                with contextlib.suppress(Exception):
                    await storage.deactivate_document(doc_id)
                raise

            used_reporter.cancel_token().raise_if_cancelled()

            chunks = await storage.list_chunks(doc_id)
            doc = await storage.get_document(doc_id)
            if doc is None:
                raise RuntimeError(
                    f"wisdom write succeeded but document {doc_id!r} not in storage"
                )

            # Read back actual embedded count rather than assuming
            # ``len(chunks)``. ``consume_embedding_stream`` can complete
            # with partial coverage when the provider retries-and-gives-up
            # on a per-batch failure without raising — ``embedded =
            # len(chunks)`` would then lie to the caller about retrieval
            # readiness. ``get_chunk_embeddings`` returns hits-only, so
            # ``len(...)`` is the truthful count of vectors persisted for
            # this doc at ``text_version_id``.
            if active_embedder is not None and text_version_id is not None:
                chunk_ids = [c.chunk_id for c in chunks if c.chunk_id is not None]
                vecs = await storage.get_chunk_embeddings(
                    chunk_ids, version_id=text_version_id
                )
                embedded = len(vecs)
            else:
                embedded = 0

            await storage.append_knowledge_log(
                KnowledgeLogEntry(
                    ts=time.time(), action="wisdom_write", src=logical_path
                )
            )

            await used_reporter.progress(
                phase="wisdom_write",
                current=1,
                total=1,
                detail={"path": logical_path, "step": "done"},
            )

            return WisdomWriteReport(
                path=logical_path,
                created=created,
                hash=doc.hash,
                chunks=len(chunks),
                embedded=embedded,
                unresolved_wikilinks=unresolved,
            )
        finally:
            # Match ``_with_storage``'s connect-failure path
            # (api.py:665): a close() error must not shadow the real
            # cause carried by the in-flight exception, otherwise the
            # task manager records "OperationalError: connection
            # closed" instead of "embedder timed out".
            with contextlib.suppress(Exception):
                await storage.close()


def _read_source_parsed(root: Path, doc: DocumentRecord) -> ParsedDocument | None:
    """Re-parse a source from disk, returning the full ``ParsedDocument``.

    Synth needs both the body and the hash: the body to feed the LLM and
    the hash to detect drift since ingest (a user-edited file would
    otherwise be sliced at stale chunk offsets).
    """
    abs_path = (root / doc.path).resolve()
    if not abs_path.is_file():
        return None
    # Route through the backend registry so HTML (and future) sources flow
    # through synth the same way markdown does.
    try:
        return parse_any(abs_path, rel_path=doc.path)
    except (OSError, UnsupportedFormat):
        return None


async def _persist_knowledge_page(
    *,
    storage: Storage,
    root: Path,
    page: KnowledgePage,
    embedder: EmbeddingProvider | None,
    embedding_model: str,
    text_version_id: int | None,
    cjk_tokenizer: CjkTokenizer = "none",
    title_to_path: dict[str, str] | None = None,
    fuzzy_index: dict[str, list[str]] | None = None,
) -> int:
    """Index ``page`` into the K layer: document, chunks, embeddings, links.

    The caller writes ``page`` to disk via ``write_page`` *before*
    invoking this function — we then re-parse the file so the stored
    hash and chunk offsets match what ``read_page`` will compute on
    read. ``frontmatter.dumps`` + ``frontmatter.loads`` is not always
    byte-stable on the body portion, so hashing ``page.body`` directly
    and chunking ``page.body`` would silently diverge from the
    read-back parsed body, causing ``read_page`` to falsely flag every
    K-layer page as stale (empty anchors).

    Returns the count of unresolved outgoing wikilinks so the synth
    caller can fold it into ``SynthReport.unresolved_wikilinks``.

    Thin delegate — implementation lives in
    :mod:`dikw_core.domains.knowledge.page_index` so lint-apply can
    reuse the same indexing path without depending on api.py internals.
    """
    from .domains.knowledge.page_index import persist_knowledge
    result = await persist_knowledge(
        storage=storage,
        root=root,
        path=page.path,
        title=page.title,
        embedder=embedder,
        embedding_model=embedding_model,
        text_version_id=text_version_id,
        cjk_tokenizer=cjk_tokenizer,
        title_to_path=title_to_path,
        fuzzy_index=fuzzy_index,
    )
    unresolved = result.unresolved_wikilinks
    return unresolved


# A knowledge_log row with ``action="synth_source_done"`` and this sentinel
# value in ``src`` records "this base has been touched by the fan-out
# synth pipeline at least once". On the very first post-fan-out run we
# always write this sentinel BEFORE the legacy backfill loop, so a later
# crash mid-fan-out can never be misread as legacy data on the next run.
# The string is intentionally not a valid file path.
_LEGACY_BACKFILL_SENTINEL = "__dikw_legacy_backfill_complete__"


# Header strings for the two prompt sections in `_synth_pages_from_source`.
# Pinned as module constants so tests, code, and any future docs stay
# aligned — drift between assertion strings and rendered prompts has
# bitten us before.
_BATCH_SECTION_HEADER = (
    "Already created in this batch (MUST reference, do NOT regenerate)"
)
_EXISTING_SECTION_HEADER = (
    "Existing knowledge pages (reference via [[Title]] when relevant)"
)
_NO_EXISTING_PAGES_SENTINEL = "(no existing pages — this is a fresh knowledge base)"


@dataclass(frozen=True)
class _ExistingPagesSnapshot:
    """Per-source snapshot of the K-layer for the synth prompt.

    Hoisted out of the per-group loop because the base K-layer is
    invariant within a single source (persist runs only after all of
    that source's groups complete). Without this hoist, a source with
    G groups against a base of W pages paid G x W storage round-trips
    per synth call.
    """

    pages: list[DocumentRecord]   # already filtered to title-bearing
    full_render_bytes: int

    @classmethod
    async def load(cls, storage: Storage) -> _ExistingPagesSnapshot:
        pages = [
            d for d in await storage.list_documents(
                layer=Layer.KNOWLEDGE, active=True
            )
            if d.title
        ]
        full_render_bytes = sum(
            len(f"- {d.title} ({type_from_path(d.path)})\n".encode())
            for d in pages
        )
        return cls(pages=pages, full_render_bytes=full_render_bytes)

    def full_pages(self) -> list[tuple[str, str]]:
        return [(t, type_from_path(d.path)) for d in self.pages if (t := d.title)]


async def _existing_pages_for_prompt(
    storage: Storage,
    *,
    snapshot: _ExistingPagesSnapshot,
    group_chunks: list[ChunkRecord],
    max_bytes: int,
    top_k: int,
    version_id: int | None,
) -> list[tuple[str, str]]:
    """Return ``[(title, type), ...]`` for the synth prompt's existing-pages section.

    Full render up to ``max_bytes`` of the rendered ``- Title (type)``
    bullets; above the threshold, switches to a vec_search-gated top-K
    driven by the group's chunk embeddings (per-chunk vec_search →
    union by doc_id → score sort → top-K). The retrieval branch keeps
    the prompt size bounded as the knowledge base grows; without it a base with
    thousands of pages would eventually overflow the model's context
    window.

    Returns ``[]`` (empty section) for a fresh knowledge base or a base with no
    embedded source chunks — the caller renders the falsy section as
    ``(no existing pages …)`` so the LLM sees a clear signal rather
    than a missing block.
    """
    if not snapshot.pages:
        return []
    if snapshot.full_render_bytes <= max_bytes:
        return snapshot.full_pages()

    # Over the byte threshold → retrieval-gated top-K. Per-chunk
    # vec_search against the WIKI layer is what the locked design
    # specifies; union by doc_id, keep best (smallest) distance per
    # doc, sort, take top-K. Distance is cosine (smaller = closer).
    #
    # ``_truncated_fallback`` is the safety net for "many pages but the
    # WIKI layer has no vectors" (``--no-embed`` wikis, version mismatch,
    # or chunks the source-side embedder hasn't reached). Returning ``[]``
    # would render the "(no existing pages — fresh knowledge base)" sentinel and
    # drop ALL duplicate-avoidance context exactly when the knowledge base has
    # the most to offer it. Bounded prefix is a worse signal than
    # vec-ranked top-K but a better one than "fresh knowledge base, generate
    # freely". Order matches the snapshot, which mirrors
    # ``list_documents`` order — stable across runs.
    def _truncated_fallback() -> list[tuple[str, str]]:
        return snapshot.full_pages()[:top_k]

    embs = await storage.get_chunk_embeddings(
        [c.chunk_id for c in group_chunks if c.chunk_id is not None],
        version_id=version_id,
    )
    if not embs:
        return _truncated_fallback()
    best_dist: dict[str, float] = {}
    for emb in embs.values():
        try:
            # Pin the lookup to the SAME version we fetched embeddings
            # under — without this, vec_search re-resolves the active
            # version and could pick a different per-version table
            # (mid-synth ingest activating a new version, or a direct
            # caller passing a non-active version_id), producing dim
            # mismatches or rankings against the wrong index.
            hits = await storage.vec_search(
                emb, layer=Layer.KNOWLEDGE, limit=top_k, version_id=version_id
            )
        except NotSupported:
            return _truncated_fallback()
        for hit in hits:
            prior = best_dist.get(hit.doc_id)
            if prior is None or hit.distance < prior:
                best_dist[hit.doc_id] = hit.distance
    if not best_dist:
        return _truncated_fallback()
    ordered_doc_ids = [
        doc_id for doc_id, _ in sorted(best_dist.items(), key=lambda kv: kv[1])
    ][:top_k]
    docs = await storage.get_documents(ordered_doc_ids)
    by_id = {d.doc_id: d for d in docs}
    out: list[tuple[str, str]] = []
    for doc_id in ordered_doc_ids:
        d = by_id.get(doc_id)
        if d is not None and d.title:
            out.append((d.title, type_from_path(d.path)))
    return out


def _render_existing_section(
    pages: list[tuple[str, str]], header: str
) -> str:
    """Render a list of ``(title, type)`` tuples as a markdown section.

    Empty input returns ``""`` so callers can concatenate two sections
    (batch accumulator + base snapshot) and fall back to a single
    "(no existing pages …)" sentinel only when both are empty.
    """
    if not pages:
        return ""
    lines = [f"## {header}", ""] + [f"- {t} ({tp})" for t, tp in pages]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class _SourceSynthOutcome:
    """Per-source aggregate of all the LLM calls Stage A made for that source."""

    pages: list[KnowledgePage]
    groups_processed: int
    parse_errors: int
    log_notes: list[str]


async def _synth_pages_from_source(
    *,
    llm: LLMProvider,
    template: str,
    cfg: DikwConfig,
    source_path: str,
    source_body: str,
    chunks: list[ChunkRecord],
    cancel: CancelToken,
    storage: Storage | None = None,
    text_version_id: int | None = None,
    force_all: bool = False,
    reporter: ProgressReporter | None = None,
) -> _SourceSynthOutcome:
    """Fan a single source out into ChunkGroups and call the LLM per group.

    The caller persists the returned pages and writes ``log_notes`` /
    counts to ``knowledge_log`` and the ``SynthReport``. ``reporter`` (optional)
    receives a ``synth_llm`` ``calling`` / ``returned`` event pair per
    group so server clients can render group-level progress instead of
    freezing on the per-source counter while a multi-group LLM call runs.

    ``storage`` + ``text_version_id`` drive the per-group existing-pages
    section: each group's prompt receives a ``## Already created in
    this batch`` accumulator (per-source state, lifecycle = this call)
    plus a ``## Existing knowledge pages`` snapshot of the base K-layer
    (full list under ``synth.existing_pages_max_bytes``, retrieval-gated
    top-K above). Without this awareness the LLM regenerates pages it
    cannot see, polluting the knowledge base with semantic duplicates that PR1's
    fuzzy resolver cannot absorb.
    """
    _reporter: ProgressReporter = reporter or NoopReporter()
    sections = derive_sections_from_chunks(
        source_body, chunks, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    groups = group_sections(
        sections, target_tokens=cfg.synth.target_tokens_per_group
    )
    if not groups:
        return _SourceSynthOutcome(
            pages=[], groups_processed=0, parse_errors=0, log_notes=[]
        )

    page_types = tuple(cfg.schema_.page_types)
    allowed_types_str = " | ".join(page_types)
    pages: list[KnowledgePage] = []
    notes: list[str] = []
    errors = 0
    total_groups = len(groups)
    # Per-source batch accumulator: each group's prompt sees the titles
    # emitted by groups 0..N-1 of the SAME source, so group N can
    # reference [[Title]] instead of regenerating. Lifecycle scoped
    # tightly to this function — a new source starts empty.
    # ``seen_titles`` mirrors the accumulator titles for O(1) dedup
    # without rebuilding a set every group.
    batch_accumulator: list[tuple[str, str]] = []
    seen_titles: set[str] = set()
    # Map section-start → chunk so we can recover per-group chunks for
    # the retrieval-gated existing-pages branch. ``derive_sections_from_chunks``
    # builds sections 1:1 from chunks, so ``section.start == chunk.start``.
    start_to_chunk = {c.start: c for c in chunks}
    # The base K-layer is invariant within a single source's group loop
    # (persist runs only after this function returns), so we hoist the
    # snapshot out of the loop. Without this, a source with G groups
    # against a base of W pages paid G x W storage round-trips per call.
    #
    # ``force_all`` skips the snapshot: ``dikw client synth --all`` is the
    # documented "regenerate everything after a prompt/model change"
    # path. Showing the LLM the OLD output of the same source plus the
    # zero-block-on-duplicate instruction would cause the model to skip
    # the regeneration the user explicitly requested. The in-batch
    # accumulator still runs so groups within the same source coordinate.
    snapshot = (
        await _ExistingPagesSnapshot.load(storage)
        if storage is not None and not force_all
        else None
    )
    for group in groups:
        cancel.raise_if_cancelled()
        group_pos = group.index + 1
        if storage is not None and snapshot is not None:
            group_chunks = [
                start_to_chunk[s] for s in group.section_starts
                if s in start_to_chunk
            ]
            existing_pages = await _existing_pages_for_prompt(
                storage,
                snapshot=snapshot,
                group_chunks=group_chunks,
                max_bytes=cfg.synth.existing_pages_max_bytes,
                top_k=cfg.synth.existing_pages_top_k,
                version_id=text_version_id,
            )
        else:
            # Storage-less callers (narrow unit tests of LLM event shape)
            # render the no-pages sentinel — they exercise the prompt
            # plumbing, not the existing-pages contract itself.
            existing_pages = []
        existing_pages_section = (
            _render_existing_section(batch_accumulator, _BATCH_SECTION_HEADER)
            + _render_existing_section(existing_pages, _EXISTING_SECTION_HEADER)
        ).strip() or _NO_EXISTING_PAGES_SENTINEL
        user_prompt = template.format(
            source_path=source_path,
            source_body=group.text,
            group_outline=", ".join(group.headings)
            if group.headings
            else "(no headings)",
            group_index=group_pos,
            group_total=total_groups,
            max_pages=cfg.synth.max_pages_per_group,
            allowed_types=allowed_types_str,
            existing_pages_section=existing_pages_section,
        )
        # `current` reports groups COMPLETED — Rich's TaskProgressRenderer
        # passes it as `completed`, so a `calling` event must show one
        # less than the in-flight group_pos. Otherwise a single-group
        # source flips to 100% the moment the LLM call starts, recreating
        # the "looks finished but isn't" symptom this PR exists to fix.
        await _reporter.progress(
            phase="synth_llm",
            current=group_pos - 1,
            total=total_groups,
            detail={
                "source_path": source_path,
                "group_pos": group_pos,
                "model": cfg.provider.llm_model,
                "status": "calling",
                "section_count": len(group.section_starts),
                "approx_tokens": group.token_count,
            },
        )
        logger.debug(
            "  group %d/%d calling llm.complete (model=%s, sections=%d, ~%d tokens)",
            group_pos,
            total_groups,
            cfg.provider.llm_model,
            len(group.section_starts),
            group.token_count,
        )
        # Per-group ProviderError resilience (issue #134). One bad group
        # (codex empty-response, auth flap, refusal) must not abort the
        # whole task — retry up to ``provider_error_retries`` times with
        # linear backoff, then skip the group and continue with the
        # next. The skip is recorded as a parse-style error so
        # ``synth_source_done`` is NOT written (the marker is gated on
        # ``outcome.parse_errors == 0`` and re-running synth has another
        # chance to succeed on the flaky group).
        max_attempts = cfg.synth.provider_error_retries + 1
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = await llm.complete(
                    system=DEFAULT_SYNTH_SYSTEM,
                    user=user_prompt,
                    model=cfg.provider.llm_model,
                    max_tokens=cfg.provider.llm_max_tokens_synth,
                    temperature=0.3,
                )
                break
            except ProviderError as pe:
                if attempt < max_attempts:
                    backoff_s = (
                        cfg.synth.provider_error_retry_backoff_seconds
                        * attempt
                    )
                    logger.warning(
                        "  group %d/%d ProviderError on attempt %d/%d: "
                        "%s — retrying in %.1fs",
                        group_pos,
                        total_groups,
                        attempt,
                        max_attempts,
                        pe,
                        backoff_s,
                    )
                    await _reporter.progress(
                        phase="synth_llm",
                        current=group_pos - 1,
                        total=total_groups,
                        detail={
                            "source_path": source_path,
                            "group_pos": group_pos,
                            "status": "retrying",
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "error_kind": type(pe).__name__,
                            "error_msg": str(pe)[:200],
                        },
                    )
                    if backoff_s > 0:
                        await asyncio.sleep(backoff_s)
                    # Honor cancellation between attempts so a user-
                    # issued cancel during a flapping run terminates
                    # promptly instead of consuming the full budget.
                    cancel.raise_if_cancelled()
                else:
                    errors += 1
                    notes.append(
                        f"group {group_pos}/{total_groups} provider "
                        f"error (skipped after {attempt} attempt(s)): {pe}"
                    )
                    logger.warning(
                        "  group %d/%d ProviderError on attempt %d/%d "
                        "(final), skipping group: %s",
                        group_pos,
                        total_groups,
                        attempt,
                        max_attempts,
                        pe,
                    )
                    await _reporter.progress(
                        phase="synth_llm",
                        current=group_pos,
                        total=total_groups,
                        detail={
                            "source_path": source_path,
                            "group_pos": group_pos,
                            "status": "skipped",
                            "reason": "provider_error",
                            "attempts": attempt,
                            "error_kind": type(pe).__name__,
                            "error_msg": str(pe)[:200],
                        },
                    )
        if response is None:
            # All attempts exhausted; move on to the next group.
            continue
        await _reporter.progress(
            phase="synth_llm",
            current=group_pos,
            total=total_groups,
            detail={
                "source_path": source_path,
                "group_pos": group_pos,
                "status": "returned",
                "response_chars": len(response.text),
            },
        )
        logger.debug(
            "  group %d/%d ← returned (%d chars)",
            group_pos,
            total_groups,
            len(response.text),
        )
        try:
            new_pages = parse_synthesis_response(
                response.text,
                source_path=source_path,
                allowed_types=page_types,
            )
        except SynthesisPartialError as pe:
            notes.append(
                f"group {group_pos}/{total_groups} partial parse: "
                f"{len(pe.errors)} issue(s); first: {pe.errors[0]}"
            )
            # Truncation is recoverable next run — count it as a parse
            # error so the source-done marker is NOT written.
            if pe.retry:
                errors += 1
            new_pages = pe.pages
            logger.warning(
                "  group %d/%d PARTIAL: %d issue(s); first: %s",
                group_pos,
                total_groups,
                len(pe.errors),
                pe.errors[0],
            )
            await _reporter.progress(
                phase="synth_llm",
                current=group_pos,
                total=total_groups,
                detail={
                    "source_path": source_path,
                    "group_pos": group_pos,
                    "status": "error",
                    "error_kind": type(pe).__name__,
                    "error_msg": str(pe)[:200],
                },
            )
        except SynthesisError as e:
            errors += 1
            notes.append(
                f"group {group_pos}/{total_groups} parse error: {e}"
            )
            logger.warning(
                "  group %d/%d FAILED: %s: %s",
                group_pos,
                total_groups,
                type(e).__name__,
                e,
            )
            await _reporter.progress(
                phase="synth_llm",
                current=group_pos,
                total=total_groups,
                detail={
                    "source_path": source_path,
                    "group_pos": group_pos,
                    "status": "error",
                    "error_kind": type(e).__name__,
                    "error_msg": str(e)[:200],
                },
            )
            continue
        pages.extend(new_pages)
        # Feed group N's emitted page titles into the per-source
        # accumulator so group N+1's prompt sees them. ``seen_titles``
        # is maintained incrementally above so dedup is O(1) per page
        # without rebuilding a set every group.
        for p in new_pages:
            if p.title and p.title not in seen_titles:
                batch_accumulator.append((p.title, p.type or "page"))
                seen_titles.add(p.title)

    return _SourceSynthOutcome(
        pages=pages,
        groups_processed=total_groups,
        parse_errors=errors,
        log_notes=notes,
    )


def _sr_replace(r: SynthReport, **kw: int) -> SynthReport:
    return dataclasses.replace(r, **kw)


