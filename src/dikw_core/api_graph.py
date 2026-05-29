"""Graph / link / provenance cluster of the engine facade.

``list_links`` and ``read_provenance`` return a single page's link and
K↔D provenance neighbourhood; ``list_graph`` returns the whole base graph
(every document a node, every resolvable wikilink/markdown link an edge)
in one read-only pass. All deterministic — no LLM.

rank3 cluster: imports ``api_core`` (``_with_storage``), ``api_path_safety``
(``_assert_within``), the leaf ``api_types`` exceptions, and the
``domains.knowledge.links`` resolution primitives, never the ``api``
facade. ``api`` re-exports ``list_links`` / ``read_provenance`` /
``list_graph`` (all public, in ``__all__``).
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .api_core import _with_storage
from .api_path_safety import _assert_within
from .api_types import PageNotFound
from .domains.data.backends import parse_any
from .domains.data.path_norm import doc_id_for as _doc_id_for
from .domains.data.path_norm import normalize_path
from .domains.knowledge.links import (
    normalize_base,
    normalize_for_match,
    parse_links,
)
from .schemas import (
    DerivedPage,
    DocumentRecord,
    GraphEdge,
    GraphNode,
    GraphResult,
    GraphStats,
    GraphUnresolvedLink,
    IncomingLink,
    Layer,
    LinkDirection,
    LinkType,
    OutgoingLink,
    PageLinksResult,
    PageProvenanceResult,
    ProvenanceDirection,
    ProvenanceSource,
)


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
