"""Cross-layer record types shared by the engine and storage adapters.

These DTOs cross the Storage Protocol boundary, so they must stay backend-agnostic:
no SQL types, no ORM handles, no cursors.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Layer(StrEnum):
    """Which DIKW layer a record belongs to."""

    SOURCE = "source"
    WIKI = "wiki"
    WISDOM = "wisdom"


class LinkType(StrEnum):
    WIKILINK = "wikilink"
    MARKDOWN = "markdown"
    URL = "url"


class AssetKind(StrEnum):
    """Multimedia asset modality. v1 ships ``IMAGE`` only; ``AUDIO`` /
    ``VIDEO`` are reserved for v2 transcription support."""

    IMAGE = "image"


class WisdomStatus(StrEnum):
    """Frontmatter status enum for wisdom-layer documents (0.3.0 PR2).

    The user sets this in the YAML frontmatter of a
    ``wisdom/<author>/<slug>.md`` file. Values mean exactly what they
    read — ``draft`` is a working note the user is still shaping;
    ``published`` is the canonical state and the implicit default when
    the frontmatter key is omitted; ``favorite`` is an Obsidian-style
    pin (no retrieval boost yet — PR2 only validates the value via
    lint, retrieval consumption is deferred); ``archived`` marks
    superseded thinking the user keeps for history.

    The enum is wisdom-only: ``DocumentRecord.status`` is forced to
    ``None`` when ``layer != Layer.WISDOM`` so the field never carries
    a stray value on wiki/source rows even if a user pastes ``status:``
    into the wrong frontmatter. The schema's CHECK constraint allows
    NULL anywhere; the application-side clamp is what keeps the
    semantics tight.

    The name reuses the PR1-deleted ``WisdomStatus`` slot deliberately:
    no value collides (old set: ``candidate / approved / archived`` was
    a review-state machine; new set is a flat author-set page state),
    and the import path stays stable for the schemas-layer DTO that
    storage adapters round-trip through.
    """

    DRAFT = "draft"
    PUBLISHED = "published"
    FAVORITE = "favorite"
    ARCHIVED = "archived"


class DocumentRecord(BaseModel):
    doc_id: str
    # ``path`` is the user's spelling (display path); ``path_key`` is
    # the engine's lookup key — NFC-normalised + casefolded so the same
    # logical file under different macOS NFD / NTFS-case spellings
    # collapses to one row. See ``data/path_norm.py``.
    #
    # ``path_key`` is auto-derived from ``path`` at construction when not
    # supplied (model_validator below) so engine call sites don't have
    # to thread the normalization through every construction. Adapters
    # that round-trip a row out of storage pass the stored value
    # explicitly, which then short-circuits the derivation.
    path: str
    path_key: str = ""
    title: str | None = None
    hash: str
    mtime: float
    layer: Layer
    active: bool = True
    # Wisdom-only frontmatter status (0.3.0 PR2). Always ``None`` for
    # non-wisdom rows — the engine forces this clamp in ``api._to_document``
    # so a stray ``status:`` in a wiki/source frontmatter never persists.
    # See ``WisdomStatus`` for the value semantics.
    status: WisdomStatus | None = None

    @model_validator(mode="before")
    @classmethod
    def _derive_path_key(cls, values: object) -> object:
        # Local import to avoid the domains.data → schemas import cycle
        # at module load time.
        from .domains.data.path_norm import normalize_path

        if isinstance(values, dict) and not values.get("path_key"):
            path = values.get("path")
            if isinstance(path, str):
                values = {**values, "path_key": normalize_path(path)}
        return values


class ChunkRecord(BaseModel):
    chunk_id: int | None = None  # assigned by storage on insert
    doc_id: str
    seq: int
    start: int
    end: int
    text: str


class EmbeddingRow(BaseModel):
    chunk_id: int
    version_id: int
    embedding: list[float]


class CachedEmbeddingRow(BaseModel):
    """Content-addressed embedding cache row.

    The chunk-level embed cache (``embed_cache`` table) stores vectors
    keyed by ``sha256(chunk.text)`` + ``version_id``, decoupled from
    ``chunks.chunk_id`` so re-ingest under ``replace_chunks``'s
    delete-and-reinsert semantics doesn't lose the API spend on
    byte-identical text. Keying on ``version_id`` (rather than ``model``)
    isolates the cache against silent corruption when normalize/distance
    flip without a model name change.
    """

    content_hash: str  # sha256 hex of chunk.text
    version_id: int
    dim: int  # implied by len(embedding) but stored for fast filtering
    embedding: list[float]


class FTSHit(BaseModel):
    doc_id: str
    chunk_id: int | None = None
    score: float
    snippet: str | None = None


class VecHit(BaseModel):
    doc_id: str
    chunk_id: int
    distance: float  # smaller = more similar (cosine distance)


class Hit(BaseModel):
    """One fused search result.

    Anchored to a concrete chunk (``chunk_id`` is required); ``seq``
    is the chunk's ordinal within its document — used for
    disambiguating multiple hits from the same document and for
    eval-side ``(path, seq) → named-id`` resolution. Image references
    appear via ``asset_refs``; per-asset retrieval as a first-class
    Hit kind is reserved for a follow-up PR.

    ``layer`` / ``start`` / ``end`` / ``text`` are the agent-facing
    fields populated by ``HybridSearcher.search`` for retrieval-only
    consumers (``POST /v1/retrieve``): they let an agent assemble a
    self-answer without re-fetching chunk text or guessing the doc's
    layer from path conventions, and they form the anchor key joined
    with ``GET /v1/base/pages/{path}`` 's ``anchors[]``. Pre-existing
    ``snippet`` is the FTS-highlighted preview kept for human display;
    ``text`` is the full chunk body for prompt assembly.
    """

    doc_id: str
    chunk_id: int
    seq: int | None = None
    score: float
    snippet: str | None = None
    path: str | None = None
    title: str | None = None
    asset_refs: list[AssetRecord] = Field(default_factory=list)
    layer: Layer | None = None
    start: int | None = None
    end: int | None = None
    text: str | None = None


class PageRef(BaseModel):
    """One page-level aggregation of a retrieve response.

    ``hit_chunk_ids`` lists the chunk_ids that landed under this page in
    fusion-rank order; ``score`` is the max chunk score so the agent can
    rank pages without re-aggregating. Exists only on the retrieve
    surface — query keeps its citation list — so agents can decide
    "read whole page" vs "use chunk excerpts" at one glance.
    """

    path: str
    layer: Layer | None = None
    title: str | None = None
    score: float
    hit_chunk_ids: list[int] = Field(default_factory=list)


class RetrieveResult(BaseModel):
    """Final payload for ``POST /v1/retrieve``.

    The caller (typically an AI agent) assembles its own answer from
    ``chunks`` + ``page_refs`` using its own LLM — dikw-core does not
    synthesize. ``chunks`` repeats the hits emitted on the
    ``retrieval_done`` partial (full ``Hit`` dump on both events), so a
    streaming agent can prompt off the partial and treat ``final`` as a
    checkpoint, while a non-streaming caller can ignore the partial and
    read ``final.result.chunks`` directly.
    """

    chunks: list[Hit] = Field(default_factory=list)
    page_refs: list[PageRef] = Field(default_factory=list)


class PageAnchor(BaseModel):
    """One chunk's location inside a page body.

    Mirrors the ``Hit.chunk_id`` / ``seq`` / ``start`` / ``end`` quadruple
    from ``/v1/retrieve`` so an agent that hit a chunk can resolve which
    region of the full page that hit covers without re-running search.
    """

    chunk_id: int
    seq: int
    start: int
    end: int


# Wire contract for the asset bytes route. Lives here so the engine
# (which formats ``PageAsset.url``) and the server (which serves the
# route) share one source of truth without crossing the layering
# boundary in either direction.
ASSET_URL_TEMPLATE = "/v1/assets/{asset_id}"


class PageAsset(BaseModel):
    """One image (or other media asset) referenced by a page body.

    ``url`` is the server-relative endpoint that streams the bytes —
    always ``/v1/assets/{asset_id}`` so clients zero-parse the route.
    ``original_paths`` is what the user typed in markdown, so a client
    can map a literal body reference back to this entry.
    """

    asset_id: str
    kind: AssetKind
    mime: str
    bytes: int
    original_paths: list[str] = Field(default_factory=list)
    media_meta: MediaMeta | None = None
    url: str


class PageReadResult(BaseModel):
    """Final payload for ``GET /v1/base/pages/{path}``.

    ``body`` is the on-disk file content (unchunked); ``anchors`` are the
    chunk boundaries the engine produced at last ingest, in ``seq`` order.
    A page that has never been chunked (e.g. a just-imported source not
    yet ingested) returns an empty ``anchors`` list.

    ``assets`` is the deduped union of every asset referenced by any
    chunk of the page (by ``asset_id``). Empty for text-only pages.
    """

    doc_id: str
    path: str
    layer: Layer
    title: str | None = None
    body: str
    anchors: list[PageAnchor] = Field(default_factory=list)
    assets: list[PageAsset] = Field(default_factory=list)


class LinkRecord(BaseModel):
    src_doc_id: str
    dst_path: str
    link_type: LinkType
    anchor: str | None = None
    line: int


class ProvenanceEdge(BaseModel):
    """One row of the ``provenance`` storage table.

    Provenance is the K-page → D-source attribution recorded in a K-page's
    ``sources:`` frontmatter list — a separate edge from ``LinkRecord``
    (which is body-derived ``[[wikilink]]``s parsed into the ``links``
    table). The two are deliberately kept in different tables because
    provenance has no body-coordinate fields (no ``anchor`` / ``line``),
    must NOT feed graph-leg retrieval, and is reconciled from frontmatter
    instead of the body. See ADR-0001.

    Mirrors the ``DocumentRecord.path`` / ``path_key`` pattern: the raw
    spelling is preserved for faithful forward display, the NFC + casefold
    normalized key powers reverse lookup (``provenance_to``) so that
    case-drift / Unicode-form-drift on Windows or macOS still matches.
    """

    src_doc_id: str  # K-page doc_id (always ``layer:wiki``)
    source_path: str  # frontmatter ``sources:`` entry, raw spelling
    source_path_key: str  # ``normalize_path(source_path)``, reverse-lookup key


class OutgoingLink(BaseModel):
    """One outgoing edge from a page (page → ``dst_path``).

    Mirrors ``LinkRecord`` minus ``src_doc_id`` — the source is the page
    being queried, so repeating its id on every entry is pure bloat over
    the wire. ``anchor`` carries the ``#section`` fragment for wikilinks
    that target a specific heading.
    """

    dst_path: str
    link_type: LinkType
    anchor: str | None = None
    line: int


class IncomingLink(BaseModel):
    """One incoming edge to a page (``src_path`` → page).

    Includes both ``src_doc_id`` (storage primary key, useful when the
    agent already has a chunk hit and wants to cross-reference) and
    ``src_path`` (the on-disk path the agent will read next), resolved
    via the documents table so callers don't need a second round trip.
    """

    src_doc_id: str
    src_path: str
    link_type: LinkType
    anchor: str | None = None
    line: int


class PageLinksResult(BaseModel):
    """Final payload for ``GET /v1/base/pages/{path}/links``.

    Splits the K-layer link graph at a page boundary: ``outgoing`` is
    every edge whose source is this page, ``incoming`` every edge whose
    destination is this page. ``direction=in|out|both`` on the request
    filters which lists are populated; ``limit`` caps each list
    independently (a hub page with many inbound and outbound edges sees
    both halves trimmed, not a per-total split).
    """

    path: str
    outgoing: list[OutgoingLink] = Field(default_factory=list)
    incoming: list[IncomingLink] = Field(default_factory=list)


# Shared closed-set type for the page-links direction parameter. Engine,
# server route, and CLI all import this so the allowed values live in
# one place — bumping it (e.g. adding a hypothetical "diagonal") only
# touches one site.
LinkDirection = Literal["in", "out", "both"]


class ProvenanceSource(BaseModel):
    """One forward provenance edge: K-page → a D-source it claims.

    Mirrors ``OutgoingLink`` in spirit but carries resolution status
    instead of ``link_type`` / ``anchor`` / ``line``. ``resolved=False``
    means ``source_path`` does not currently index to an active
    ``Layer.SOURCE`` document — the page references a source that was
    deleted, renamed, or never ingested. Surfaced faithfully (not
    silently dropped) so agents can detect provenance drift.
    """

    source_path: str  # as written in frontmatter (raw)
    doc_id: str | None = None  # resolved Layer.SOURCE doc_id, None if dangling
    title: str | None = None  # resolved source title, None if dangling
    resolved: bool


class DerivedPage(BaseModel):
    """One reverse provenance edge: D-source → a K-page derived from it.

    Mirrors ``IncomingLink``. Always carries ``doc_id`` / ``path`` /
    ``title`` because ``src_doc_id`` always resolves to a real K-page
    (``delete_document`` removes provenance rows when a K-page is
    deleted, so dangling on the reverse side cannot occur).
    """

    doc_id: str
    path: str
    title: str | None = None


class PageProvenanceResult(BaseModel):
    """Final payload for ``GET /v1/base/pages/{path}/provenance``.

    Splits the page-source attribution graph at a page boundary:
    ``derived_from`` is meaningful when the path is a K-page
    (``Layer.WIKI``); ``derived_pages`` is meaningful when the path is a
    D-source (``Layer.SOURCE``). For a path that resolves to
    ``Layer.WIKI``, ``derived_pages`` is always empty; vice versa for
    ``Layer.SOURCE``. We do not filter — agents can ask
    ``direction=both`` against any path; the empty list IS the answer.
    ``direction=in|out|both`` on the request filters which lists are
    populated.
    """

    path: str
    derived_from: list[ProvenanceSource] = Field(default_factory=list)  # out
    derived_pages: list[DerivedPage] = Field(default_factory=list)  # in


# Shared closed-set type for the provenance direction parameter. Mirrors
# ``LinkDirection`` so engine / server / CLI all reference the same
# literal.
ProvenanceDirection = Literal["in", "out", "both"]


class GraphNode(BaseModel):
    """One vertex in the base graph returned by ``GET /v1/base/graph``.

    ``inbound`` / ``outbound`` count *distinct connected pages*, not
    raw link occurrences — a page that links to ``B`` three times
    contributes 1 to ``B.inbound`` (issue #89).
    """

    id: str
    path: str
    title: str | None = None
    layer: Layer
    active: bool
    mtime: float
    inbound: int
    outbound: int


class GraphEdge(BaseModel):
    """One directed edge in the base graph.

    ``weight`` aggregates byte-identical
    ``(source, target, target_text, anchor)`` occurrences; distinct
    ``target_text`` / ``anchor`` variants between the same pair stay
    as separate edges (so a renderer can label each), and ``id``
    repeats when that happens — clients treat the 4-tuple as the
    unique key, not ``id``.
    """

    id: str
    source: str
    target: str
    type: LinkType
    target_text: str
    anchor: str | None = None
    weight: int


class GraphUnresolvedLink(BaseModel):
    """A wikilink that could not be resolved to a node in the graph.

    Issue #89 v1: no ghost nodes — broken wikilinks live only in this
    list, never as fake nodes. ``suggestions`` is deliberately omitted
    in v1; the lint subsystem owns that flow.
    """

    source: str
    target_text: str
    anchor: str | None = None
    count: int


class GraphStats(BaseModel):
    """Top-level totals; ``unresolved_count`` sums per-pair counts (two
    pages each linking ``[[Missing]]`` once → 2, not 1)."""

    node_count: int
    edge_count: int
    unresolved_count: int


class GraphResult(BaseModel):
    """Wire payload for ``GET /v1/base/graph``.

    ``base_revision`` is content-addressed (sha256 over sorted per-doc
    ``(path, title, layer, mtime, body_sha256, active)`` tuples) so a
    client can cheaply skip re-render when the base hasn't changed —
    observes on-disk body edits between ingests AND title/metadata
    changes that re-ingest persists without touching bytes.
    ``generated_at`` is informational; the determinism contract is on
    ``base_revision``.
    """

    base_revision: str
    generated_at: datetime
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    unresolved: list[GraphUnresolvedLink] = Field(default_factory=list)
    stats: GraphStats


class ChunkNeighborRecord(BaseModel):
    """A chunk reachable from a seed chunk via the K-layer link graph.

    ``edge_count`` counts how many seed-side links land on this neighbor —
    a popular page reached from many seeds ranks higher than one reached
    from a single seed. ``doc_id`` is included so the search-side
    same-doc diversity penalty applies to graph-only chunks too.
    Returned by ``Storage.neighbor_chunks_via_links`` in
    ``edge_count``-descending order so callers can RRF-fuse with one
    more leg without resorting.
    """

    chunk_id: int
    doc_id: str
    edge_count: int


class WikiLogEntry(BaseModel):
    # ``id`` is None on construction; the storage layer assigns it on
    # insert via SQLite AUTOINCREMENT / Postgres BIGSERIAL. Acts as a
    # monotonic tiebreaker when two events land in the same float-second
    # ``ts`` — ``list_wiki_log`` orders by ``(ts, id)`` so retrieval
    # order matches insert order even within a sub-second burst.
    id: int | None = None
    ts: float
    # ``synth_source_done`` is the per-source completion marker the
    # fan-out synth pipeline writes after every group for a source has
    # been processed without a hard parse error. Lets default ``synth``
    # skip done sources without misfiring on partial-failure or legal
    # zero-page responses (which never write a per-page ``synth`` row).
    action: Literal[
        "ingest", "synth", "synth_source_done", "lint", "delete", "wisdom_write"
    ]
    src: str | None = None
    dst: str | None = None
    note: str | None = None


class StorageCounts(BaseModel):
    """Diagnostic counts surfaced by `Storage.counts()`."""

    documents_by_layer: dict[str, int] = Field(default_factory=dict)
    chunks: int = 0
    embeddings: int = 0
    links: int = 0
    last_wiki_log_ts: float | None = None
    assets: int = 0
    asset_embeddings: int = 0


# ---- Media / Assets ------------------------------------------------------


class AssetRef(BaseModel):
    """One image reference encountered while parsing a markdown body.

    ``start`` / ``end`` are character offsets into the *original* document
    body (the same coordinate space ``ChunkRecord`` uses), so the chunker
    can keep the reference intact and ``chunk_asset_refs`` can be populated
    after chunking with chunk-relative offsets.

    ``original_path`` is canonicalised to forward slashes at construction
    so Windows-pasted backslash paths resolve on POSIX CI.
    """

    original_path: str
    alt: str = ""
    start: int
    end: int
    syntax: Literal["markdown", "wikilink"]

    @field_validator("original_path")
    @classmethod
    def _forward_slashes(cls, v: str) -> str:
        return v.replace("\\", "/")


class ImageContent(BaseModel):
    """Raw image payload destined for a multimodal embedding provider."""

    bytes: bytes
    mime: str  # "image/png" | "image/jpeg" | "image/webp" | "image/gif"


class MultimodalInput(BaseModel):
    """One input unit for a ``MultimodalEmbeddingProvider``.

    v1 uses either ``text`` or ``images`` (not both); the schema permits
    arbitrary combinations so v1.5 chunk-with-images joint encoding can
    land without breaking the wire format.
    """

    text: str | None = None
    images: list[ImageContent] = Field(default_factory=list)


class ImageMediaMeta(BaseModel):
    """Per-kind media metadata for an image asset.

    The ``kind`` literal is the pydantic discriminator that lets ``MediaMeta``
    grow ``AudioMediaMeta`` / ``VideoMediaMeta`` siblings later without
    touching the ``assets`` table schema.
    """

    kind: Literal["image"] = "image"
    width: int | None = None
    height: int | None = None


# Discriminated union over per-kind metadata. Add ``AudioMediaMeta`` /
# ``VideoMediaMeta`` here when ``AssetKind`` grows.
MediaMeta = Annotated[ImageMediaMeta, Field(discriminator="kind")]


def dump_media_meta(meta: MediaMeta | None) -> str | None:
    """Serialize ``MediaMeta`` to the JSON text stored in ``assets.media_meta``.

    Mirror of ``load_media_meta`` — both adapters call this pair so the
    storage codec stays consistent across SQLite ``TEXT`` and Postgres
    ``TEXT`` (and any future ``JSONB`` upgrade)."""
    return meta.model_dump_json() if meta is not None else None


def load_media_meta(payload: str | None) -> MediaMeta | None:
    """Inverse of ``dump_media_meta``."""
    # While ``AssetKind`` is image-only the dispatch is trivial; switch to
    # ``TypeAdapter[MediaMeta]`` here when a second member earns its keep.
    return ImageMediaMeta.model_validate_json(payload) if payload is not None else None


class AssetRecord(BaseModel):
    """A multimedia asset materialized into the engine-managed vault path.

    ``asset_id`` is the sha256 hex of the bytes and is the sole identity
    column — content-addressed, like ``documents.hash``. ``media_meta`` is
    a per-kind discriminated union — for v1 (image-only) it carries width
    and height; future modalities slot in their own fields without an
    ``ALTER TABLE``.
    """

    asset_id: str
    kind: AssetKind
    mime: str
    stored_path: str  # relative to project_root, e.g. "assets/ab/ab3f12ef-foo.png"
    original_paths: list[str] = Field(default_factory=list)
    bytes: int
    media_meta: MediaMeta | None = None
    created_ts: float


class ChunkAssetRef(BaseModel):
    """Many-to-many bridge: which assets a text chunk references, and where
    inside the chunk text each reference lives."""

    chunk_id: int
    asset_id: str
    ord: int  # ordinal within the chunk (0, 1, 2, …)
    alt: str = ""
    start_in_chunk: int
    end_in_chunk: int


class AssetEmbeddingRow(BaseModel):
    """One asset-level embedding, written to ``vec_assets_v<version_id>``.

    The vector dimension is fixed by ``EmbeddingVersion.dim``; mismatched
    rows are rejected at the storage boundary."""

    asset_id: str
    version_id: int
    embedding: list[float]


class AssetVecHit(BaseModel):
    """One result row from ``Storage.vec_search_assets``."""

    asset_id: str
    distance: float  # smaller = more similar (cosine distance, by default)


# ---- Embedding versioning ------------------------------------------------


class EmbeddingVersion(BaseModel):
    """Composite identity for an embedding generation.

    Any field different = different version = different vector table. The
    storage layer enforces ``UNIQUE (provider, model, revision, dim,
    normalize, distance)`` and assigns ``version_id``.

    ``revision`` is the user-facing escape hatch when a provider silently
    refreshes weights behind a stable model name (e.g. OpenAI's monthly
    "text-embedding-3-small" reroll); bumping it triggers a new version
    row and a fresh ``vec_*_v<id>`` table without touching any other knob.
    """

    version_id: int | None = None  # assigned by storage on insert
    provider: str
    model: str
    revision: str = ""
    dim: int
    normalize: bool
    distance: Literal["cosine", "l2", "dot"]
    modality: Literal["text", "multimodal"]
    created_ts: float | None = None
    is_active: bool = True


# ---- Wisdom write API (0.3.1) -------------------------------------------


class WisdomWriteSubmit(BaseModel):
    """Body for ``POST /v1/base/wisdom`` and the engine ``write_wisdom_page``.

    ``slug`` and ``author`` are validated to ASCII kebab-case at the
    Pydantic boundary (so HTTP callers see a 422 instead of a 500); the
    engine API repeats the check via :func:`domains.wisdom.validate_kebab`
    so direct Python callers get the same guarantee. ``no_embed`` lets a
    bulk-writing caller defer embedding to the next ``dikw ingest`` pass
    — symmetric with ``IngestSubmit.no_embed``.
    """

    slug: str
    title: str
    body: str
    author: str | None = None
    status: WisdomStatus | None = None
    tags: list[str] | None = None
    sources: list[str] | None = None
    extras: dict[str, object] | None = None
    no_embed: bool = False

    @field_validator("slug")
    @classmethod
    def _check_slug(cls, v: str) -> str:
        # Lazy import keeps schemas.py free of a domains.wisdom import
        # at module load — same pattern as ``DocumentRecord._derive_path_key``.
        from .domains.wisdom import validate_kebab

        validate_kebab(v, label="slug")
        return v

    @field_validator("author")
    @classmethod
    def _check_author(cls, v: str | None) -> str | None:
        if v is None:
            return None
        from .domains.wisdom import validate_kebab

        validate_kebab(v, label="author")
        return v


class WisdomWriteReport(BaseModel):
    """Result of a single wisdom write.

    ``created`` is ``True`` on first write to ``path`` (no prior
    ``DocumentRecord``), ``False`` on subsequent upserts to the same
    path. ``embedded`` is the count of chunks whose vectors landed in
    the per-version vec table this call; ``0`` when ``no_embed=True``
    or no embedder was resolvable.
    """

    path: str
    created: bool
    hash: str
    chunks: int
    embedded: int
    unresolved_wikilinks: int
