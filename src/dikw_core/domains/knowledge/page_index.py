"""Persist a K-layer wiki page into storage.

Single source of truth for wiki-page indexing. Two callers today:

* ``api._persist_wiki_page`` (synth path) — passes ``embedder`` and
  ``text_version_id`` so chunk embeddings land in the per-version
  ``vec_chunks_v<id>`` table.
* ``run_lint_apply`` (lint-fix path) — passes ``embedder=None`` to keep
  apply provider-free; the next ``dikw client ingest`` reconciles embeddings
  via ``doc.hash`` drift.

The caller MUST have written ``page`` to disk before calling — we
re-parse the file via the backend registry so the stored hash and
chunk offsets match what ``read_page`` will compute on read.
``frontmatter.dumps`` + ``frontmatter.loads`` is not byte-stable on the
body portion, so hashing ``page.body`` directly diverges from the
read-back parsed body.
"""

from __future__ import annotations

from pathlib import Path

from ...providers.base import EmbeddingProvider
from ...schemas import ChunkRecord, DocumentRecord, Layer
from ...storage.base import Storage
from ..data.backends import parse_any
from ..data.path_norm import normalize_path
from ..info.chunk import chunk_markdown
from ..info.embed import ChunkToEmbed, consume_embedding_stream, embed_chunks
from ..info.tokenize import CjkTokenizer
from .links import parse_links, resolve_links
from .wiki import frontmatter_str_list


def wiki_doc_id(path: str) -> str:
    """The canonical ``"wiki:<normalized_path>"`` doc-id for a K-layer page."""
    return f"{Layer.WIKI.value}:{normalize_path(path)}"


async def persist_wiki_page(
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
    """Index a wiki page already on disk into the K layer.

    Re-parses the file via the backend registry so the stored hash and
    chunk offsets match what ``read_page`` will compute on read —
    ``frontmatter.dumps`` is not byte-stable on the body, so hashing
    ``page.body`` directly diverges from the read-back parsed body.

    ``title`` overrides the title inferred from the file's frontmatter
    when the caller already has a canonical value (synth path, where
    the LLM's ``<page>`` block names the title). Lint-apply leaves it
    ``None`` and trusts ``parse_any``.

    ``title_to_path`` lets fan-out callers (Stage A synth) avoid a
    per-page ``list_documents`` round-trip; when ``None`` we read it
    once here.

    When ``embedder`` is ``None`` (lint-apply) we skip the embedding
    stream entirely — apply stays provider-free and embeddings
    reconcile on the next ``dikw client ingest`` via ``doc.hash`` drift.

    Returns ``(unresolved_count, resolved_title)`` so callers can fold
    the unresolved count into reports and update an incremental
    ``title_to_path`` without re-reading the file's frontmatter.
    """
    doc_id = wiki_doc_id(path)
    abs_path = (root / path).resolve()
    parsed = parse_any(abs_path, rel_path=path)
    resolved_title = title if title is not None else parsed.title

    await storage.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            path=path,
            title=resolved_title,
            hash=parsed.hash,
            mtime=parsed.mtime,
            layer=Layer.WIKI,
            active=True,
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
        k_docs = await storage.list_documents(layer=Layer.WIKI, active=True)
        title_to_path = {}
        for d in k_docs:
            if d.title and d.title not in title_to_path:
                title_to_path[d.title] = d.path

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

    # Reconcile provenance edges (K-page → D-source attribution) from
    # the page's ``sources:`` frontmatter — frontmatter is the source of
    # truth (the wiki tree is a user-editable Obsidian vault), so
    # re-running this on every persist self-heals when the user hand-
    # edits the list. Mirrors the wikilink reconcile above; deliberately
    # kept off the wikilink graph (separate ``provenance`` table — see
    # docs/adr/0001-provenance-as-separate-edge.md) so graph-leg
    # retrieval and orphan/broken-link lint stay clean.
    #
    # ``frontmatter_str_list`` enforces the same malformed-shape guard
    # as ``run_lint`` and ``MissingProvenanceFixer`` — a YAML scalar
    # (``sources: foo.md``) collapses to ``[]`` rather than iterating
    # character-by-character into the provenance table.
    await storage.replace_provenance_from(
        doc_id, frontmatter_str_list(parsed.frontmatter, "sources")
    )

    return len(unresolved), resolved_title
