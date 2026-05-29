"""Page-read cluster of the engine facade: ``list_pages`` / ``read_page`` /
``read_asset`` plus the ``_collect_page_assets`` helper.

These are the deterministic navigation primitives behind
``GET /v1/base/pages/...`` and ``GET /v1/base/assets/...`` — no LLM, just
indexed document/asset lookups with path-safety guards.

rank3 cluster: imports ``api_core`` (``_with_storage``), ``api_path_safety``
(``_assert_within``) and the leaf ``api_types`` exceptions, never the ``api``
facade. ``api`` re-exports ``list_pages`` / ``read_page`` / ``read_asset``
(all public, in ``__all__``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .api_core import _with_storage
from .api_path_safety import _assert_within
from .api_types import AssetNotFound, PageNotFound
from .domains.data.backends import parse_any
from .domains.data.path_norm import doc_id_for as _doc_id_for
from .schemas import (
    ASSET_URL_TEMPLATE,
    AssetRecord,
    ChunkRecord,
    DocumentRecord,
    Layer,
    PageAnchor,
    PageAsset,
    PageReadResult,
)
from .storage import Storage


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
