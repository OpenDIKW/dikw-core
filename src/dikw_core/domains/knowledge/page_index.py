"""Persist a K- or W-layer page into storage.

Three public functions own the per-layer write paths:

* :func:`persist_knowledge` — K-layer page (synth output or lint-apply
  fix). Always ``layer=Layer.KNOWLEDGE``, ``status`` clamped to None.
* :func:`persist_wisdom` — W-layer page (hand-written via
  ``write_wisdom_page``). ``status`` flows through from frontmatter.
  Lives in ``domains/wisdom/persist.py`` so the W layer owns its own
  entry point; it delegates to the shared layered impl below.
* (D-layer ``persist_source`` lives in ``domains/data/persist.py`` —
  it owns asset materialise + chunk_asset_refs which K/W don't have.)

The private :func:`_persist_layered_page` implements the K/W shared
pipeline: upsert_document + chunk + FTS + (inline-or-deferred embed)
+ replace_links_from + replace_provenance_from. ``persist_knowledge``
and ``persist_wisdom`` are thin typed wrappers around it, differing
only in (a) the ``Layer`` they pass through and (b) whether they
preserve the frontmatter ``status`` (W only).

Embed timing: when ``embedder`` and ``text_version_id`` are both
supplied, chunks are inline-embedded and counted in
``chunks_embedded``. When either is None (lint apply without an
embedder configured, ``wisdom write --no-embed``, …), chunks are
written + FTS-indexed but NOT embedded; ``chunks_pending_embedding``
captures the count. Those chunks are reconciled by the next ingest's
cross-layer ``list_chunks_missing_embedding`` resume scan — this is
independent of ``doc.hash`` drift.

The caller MUST have written ``page`` to disk before calling — we
re-parse the file via the backend registry so the stored hash and
chunk offsets match what ``read_page`` will compute on read.
``frontmatter.dumps`` + ``frontmatter.loads`` is not byte-stable on
the body portion, so hashing ``page.body`` directly diverges from the
read-back parsed body.

Title index for wikilink resolve is cross-layer: ``[[wikilinks]]`` in
a wisdom page resolve against the union of KNOWLEDGE + WISDOM titles,
and vice versa. Title collisions across layers fall through the
existing refuse-to-resolve mechanism in ``links.resolve_links``.
"""

from __future__ import annotations

from pathlib import Path

from ...providers.base import EmbeddingProvider
from ...schemas import (
    ChunkRecord,
    DocumentRecord,
    KnowledgePersistResult,
    Layer,
    WisdomStatus,
)
from ...storage.base import Storage
from ..data.backends import parse_any
from ..data.path_norm import doc_id_for
from ..info.chunk import chunk_markdown
from ..info.embed import ChunkToEmbed, consume_embedding_stream, embed_chunks
from ..info.tokenize import CjkTokenizer
from .links import build_title_indexes, parse_links, resolve_links
from .page import frontmatter_str_list


async def _persist_layered_page(
    *,
    storage: Storage,
    root: Path,
    path: str,
    layer: Layer,
    title: str | None,
    embedder: EmbeddingProvider | None,
    embedding_model: str,
    text_version_id: int | None,
    cjk_tokenizer: CjkTokenizer,
    title_to_path: dict[str, str] | None,
    fuzzy_index: dict[str, list[str]] | None,
    retries: int,
    backoff_seconds: float,
) -> KnowledgePersistResult:
    """Shared K/W persist implementation.

    Private — call :func:`persist_knowledge` or :func:`persist_wisdom`
    instead. Returns ``KnowledgePersistResult``; the W-layer caller
    re-wraps it as ``WisdomPersistResult`` (identical fields today).
    """
    if layer not in (Layer.KNOWLEDGE, Layer.WISDOM):
        raise ValueError(
            f"_persist_layered_page is for K/W layers only; got {layer!r}. "
            "Use persist_source for D layer."
        )

    doc_id = doc_id_for(layer, path)
    abs_path = (root / path).resolve()
    parsed = parse_any(abs_path, rel_path=path)
    resolved_title = title if title is not None else parsed.title

    # status is wisdom-only; the engine clamps elsewhere too (api._to_document),
    # but persist_*_page is its own write path so the same invariant applies.
    status: WisdomStatus | None = parsed.status if layer is Layer.WISDOM else None

    await storage.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            path=path,
            title=resolved_title,
            hash=parsed.hash,
            mtime=parsed.mtime,
            layer=layer,
            active=True,
            status=status,
        )
    )

    chunks = chunk_markdown(parsed.body, cjk_tokenizer=cjk_tokenizer)
    records = [
        ChunkRecord(doc_id=doc_id, seq=c.seq, start=c.start, end=c.end, text=c.text)
        for c in chunks
    ]
    chunk_ids = await storage.replace_chunks(doc_id, records)

    chunks_embedded = 0
    chunks_pending_embedding = 0
    if embedder is not None and records and text_version_id is not None:
        to_embed = [
            ChunkToEmbed(chunk_id=cid, text=r.text)
            for cid, r in zip(chunk_ids, records, strict=True)
        ]
        embed_result = await consume_embedding_stream(
            embed_chunks(
                embedder,
                to_embed,
                model=embedding_model,
                version_id=text_version_id,
                storage=storage,
                retries=retries,
                backoff_seconds=backoff_seconds,
            ),
            storage,
        )
        chunks_embedded = embed_result.embedded
        chunks_pending_embedding = embed_result.chunks_skipped
    elif records:
        # No embedder configured — every chunk is "pending" until the
        # next ingest's resume scan picks them up. ``chunks_embedded``
        # stays zero. This keeps the caller's
        # ApplyReport/WisdomWriteReport accounting symmetric: the sum
        # ``embedded + pending`` always equals chunk count when the
        # caller wanted vectors.
        chunks_pending_embedding = len(records)

    if title_to_path is None:
        # Cross-layer title index: a wisdom page may link to a knowledge
        # page and vice versa via title. ``build_title_indexes`` drops
        # exact-title collisions from the exact-match dict and pushes
        # both colliding paths into the fuzzy bucket — that way the
        # second-stage ≥2-candidate refusal in ``resolve_links``
        # actually fires when a knowledge and wisdom page share a title,
        # rather than the first-seen layer silently winning.
        #
        # Pairing contract: ``title_to_path`` and ``fuzzy_index`` must be
        # built from the SAME title set. When we rebuild the exact index
        # from storage here we ALSO rebuild the fuzzy index from it and
        # discard any caller-supplied ``fuzzy_index`` — a fuzzy index keyed
        # off a different title set would resolve wikilinks against a
        # mismatched key space (fresh-exact + stale-fuzzy).
        docs_iter: list[tuple[str, str]] = []
        for layer_for_index in (Layer.KNOWLEDGE, Layer.WISDOM):
            for d in await storage.list_documents(
                layer=layer_for_index, active=True
            ):
                if d.title:
                    docs_iter.append((d.title, d.path))
        title_to_path, fuzzy_index = build_title_indexes(docs_iter)

    # Reconcile outgoing links atomically — removing a [[wikilink]]
    # from the body must drop the edge from storage, not leave a ghost
    # that pollutes graph-leg retrieval and orphan/broken-link lint.
    # ``replace_links_from`` no-ops the leading delete on a fresh page
    # (no prior edges to wipe).
    parsed_links = parse_links(parsed.body)
    resolved, unresolved = resolve_links(
        doc_id,
        parsed_links,
        title_to_path=title_to_path,
        fuzzy_index=fuzzy_index,
    )
    await storage.replace_links_from(doc_id, resolved)

    # Reconcile provenance edges (K/W-page → D-source attribution) from
    # the page's ``sources:`` frontmatter — frontmatter is the source of
    # truth (the knowledge/wisdom tree is a user-editable Obsidian vault),
    # so re-running this on every persist self-heals when the user
    # hand-edits the list. Mirrors the wikilink reconcile above;
    # deliberately kept off the wikilink graph (separate ``provenance``
    # table — see docs/adr/0001-provenance-as-separate-edge.md) so
    # graph-leg retrieval and orphan/broken-link lint stay clean.
    await storage.replace_provenance_from(
        doc_id, frontmatter_str_list(parsed.frontmatter, "sources")
    )

    return KnowledgePersistResult(
        chunk_ids=chunk_ids,
        chunks_embedded=chunks_embedded,
        chunks_pending_embedding=chunks_pending_embedding,
        unresolved_wikilinks=len(unresolved),
        resolved_title=resolved_title,
    )


async def persist_knowledge(
    *,
    storage: Storage,
    root: Path,
    path: str,
    title: str | None = None,
    embedder: EmbeddingProvider | None = None,
    embedding_model: str = "",
    text_version_id: int | None = None,
    cjk_tokenizer: CjkTokenizer = "none",
    title_to_path: dict[str, str] | None = None,
    fuzzy_index: dict[str, list[str]] | None = None,
    retries: int = 0,
    backoff_seconds: float = 0.0,
) -> KnowledgePersistResult:
    """Index a K-layer page (synth output or lint-apply fix) into storage.

    Owns: ``upsert_document`` (``status=None``, hard-clamped) + chunk
    + FTS + (inline embed when ``embedder`` and ``text_version_id`` are
    both supplied; otherwise defers to the next ingest's resume scan)
    + ``replace_links_from`` (wikilinks) + ``replace_provenance_from``
    (``sources:`` frontmatter).

    Two callers today:

    * ``api._persist_knowledge_page`` (synth path) — passes
      ``embedder`` and ``text_version_id`` so chunk embeddings land in
      the per-version ``vec_chunks_v<id>`` table inline.
    * ``lint_fix.run_lint_apply`` (lint-fix path) — historically
      passed ``embedder=None`` to keep apply provider-free; from 0.4.0
      passes the configured embedder when available so K pages are
      retrieval-ready on apply.

    See module docstring for the cross-layer wikilink resolve contract.
    """
    return await _persist_layered_page(
        storage=storage,
        root=root,
        path=path,
        layer=Layer.KNOWLEDGE,
        title=title,
        embedder=embedder,
        embedding_model=embedding_model,
        text_version_id=text_version_id,
        cjk_tokenizer=cjk_tokenizer,
        title_to_path=title_to_path,
        fuzzy_index=fuzzy_index,
        retries=retries,
        backoff_seconds=backoff_seconds,
    )
