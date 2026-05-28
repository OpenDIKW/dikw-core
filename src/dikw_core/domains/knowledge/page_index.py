"""Persist a K or W layer page into storage.

Single source of truth for page indexing. Three callers today:

* ``api._persist_knowledge_page`` (synth path) — passes ``embedder`` and
  ``text_version_id`` so chunk embeddings land in the per-version
  ``vec_chunks_v<id>`` table. Always ``layer=Layer.KNOWLEDGE``.
* ``run_lint_apply`` (lint-fix path) — passes ``embedder=None`` to keep
  apply provider-free; the next ``dikw client ingest`` reconciles
  embeddings via ``doc.hash`` drift. Always ``layer=Layer.KNOWLEDGE``.
* ``api.ingest`` wisdom branch (0.3.0 PR2) — passes
  ``layer=Layer.WISDOM`` and queues embeddings onto the shared
  ``to_embed`` list rather than embedding inline, so wisdom and
  source/wiki chunks all flow through one ``embed_chunks`` batch.

The caller MUST have written ``page`` to disk before calling — we
re-parse the file via the backend registry so the stored hash and
chunk offsets match what ``read_page`` will compute on read.
``frontmatter.dumps`` + ``frontmatter.loads`` is not byte-stable on the
body portion, so hashing ``page.body`` directly diverges from the
read-back parsed body.

Title index for wikilink resolve is cross-layer: ``[[wikilinks]]`` in a
wisdom page resolve against the union of WIKI + WISDOM titles, and
vice versa. Title collisions across layers fall through the existing
refuse-to-resolve mechanism in ``links.resolve_links``.
"""

from __future__ import annotations

from pathlib import Path

from ...providers.base import EmbeddingProvider
from ...schemas import ChunkRecord, DocumentRecord, Layer, WisdomStatus
from ...storage.base import Storage
from ..data.backends import parse_any
from ..data.path_norm import doc_id_for
from ..info.chunk import chunk_markdown
from ..info.embed import ChunkToEmbed, consume_embedding_stream, embed_chunks
from ..info.tokenize import CjkTokenizer
from .links import build_title_indexes, parse_links, resolve_links
from .page import frontmatter_str_list


def wiki_doc_id(path: str) -> str:
    """Backwards-compatible alias for ``doc_id_for(Layer.KNOWLEDGE, path)``."""
    return doc_id_for(Layer.KNOWLEDGE, path)


async def persist_page(
    *,
    storage: Storage,
    root: Path,
    path: str,
    layer: Layer = Layer.KNOWLEDGE,
    title: str | None = None,
    embedder: EmbeddingProvider | None = None,
    embedding_model: str = "",
    text_version_id: int | None = None,
    cjk_tokenizer: CjkTokenizer = "none",
    title_to_path: dict[str, str] | None = None,
    fuzzy_index: dict[str, list[str]] | None = None,
) -> tuple[int, str]:
    """Index a page already on disk into the K or W layer.

    See module docstring for the cross-caller contract. ``layer``
    controls (a) the doc-id prefix and (b) the cross-layer wikilink
    index built when ``title_to_path`` is not supplied.

    Returns ``(unresolved_count, resolved_title)`` so callers can fold
    the unresolved count into reports and update an incremental
    ``title_to_path`` without re-reading the file's frontmatter.
    """
    doc_id = doc_id_for(layer, path)
    abs_path = (root / path).resolve()
    parsed = parse_any(abs_path, rel_path=path)
    resolved_title = title if title is not None else parsed.title

    # status is wisdom-only; the engine clamps elsewhere too (api._to_document),
    # but persist_page is its own write-path so the same invariant applies.
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

    if embedder is not None and records and text_version_id is not None:
        to_embed = [
            ChunkToEmbed(chunk_id=cid, text=r.text)
            for cid, r in zip(chunk_ids, records, strict=True)
        ]
        await consume_embedding_stream(
            embed_chunks(
                embedder,
                to_embed,
                model=embedding_model,
                version_id=text_version_id,
                storage=storage,
            ),
            storage,
        )

    if title_to_path is None:
        # Cross-layer title index: a wisdom page may link to a wiki
        # page and vice versa via title. ``build_title_indexes`` drops
        # exact-title collisions from the exact-match dict and pushes
        # both colliding paths into the fuzzy bucket — that way the
        # second-stage ≥2-candidate refusal in ``resolve_links``
        # actually fires when a wiki and wisdom page share a title,
        # rather than the first-seen layer silently winning.
        docs_iter: list[tuple[str, str]] = []
        for layer_for_index in (Layer.KNOWLEDGE, Layer.WISDOM):
            for d in await storage.list_documents(
                layer=layer_for_index, active=True
            ):
                if d.title:
                    docs_iter.append((d.title, d.path))
        title_to_path, derived_fuzzy = build_title_indexes(docs_iter)
        if fuzzy_index is None:
            fuzzy_index = derived_fuzzy

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
    # truth (the wiki/wisdom tree is a user-editable Obsidian vault),
    # so re-running this on every persist self-heals when the user
    # hand-edits the list. Mirrors the wikilink reconcile above;
    # deliberately kept off the wikilink graph (separate ``provenance``
    # table — see docs/adr/0001-provenance-as-separate-edge.md) so
    # graph-leg retrieval and orphan/broken-link lint stay clean.
    #
    # ``frontmatter_str_list`` enforces the same malformed-shape guard
    # as ``run_lint`` and ``MissingProvenanceFixer`` — a YAML scalar
    # (``sources: foo.md``) collapses to ``[]`` rather than iterating
    # character-by-character into the provenance table.
    await storage.replace_provenance_from(
        doc_id, frontmatter_str_list(parsed.frontmatter, "sources")
    )

    return len(unresolved), resolved_title


async def persist_knowledge_page(
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
) -> tuple[int, str]:
    """Backwards-compatible K-layer wrapper around ``persist_page``.

    Existing synth + lint-apply call sites land here unchanged; PR2
    generalised the underlying implementation to take a ``layer``
    parameter without churning every caller.
    """
    return await persist_page(
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
    )
