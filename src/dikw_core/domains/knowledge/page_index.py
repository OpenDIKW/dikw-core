"""Persist a K-layer wiki page into storage.

Single source of truth for wiki-page indexing. Two callers today:

* ``api._persist_wiki_page`` (synth path) — passes ``embedder`` and
  ``text_version_id`` so chunk embeddings land in the per-version
  ``vec_chunks_v<id>`` table.
* ``run_lint_apply`` (lint-fix path) — passes ``embedder=None`` to keep
  apply provider-free; the next ``dikw ingest`` reconciles embeddings
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
from .wiki import WikiPage


async def persist_wiki_page(
    *,
    storage: Storage,
    root: Path,
    page: WikiPage,
    embedder: EmbeddingProvider | None = None,
    embedding_model: str = "",
    text_version_id: int | None = None,
    cjk_tokenizer: CjkTokenizer = "none",
    title_to_path: dict[str, str] | None = None,
    fuzzy_index: dict[str, list[str]] | None = None,
) -> int:
    """Index ``page`` into the K layer (document, chunks, embeddings, links).

    Returns the count of unresolved outgoing wikilinks so callers can
    fold it into reports (synth: ``SynthReport.unresolved_wikilinks``;
    lint-apply: surfaced via ``ApplyReport`` skip records).

    ``title_to_path`` may be supplied to skip the per-page
    ``list_documents`` round-trip when persisting many pages in a row
    (Stage A fan-out persists N deduped pages per source — without this,
    each call would re-pull the whole K-layer doc list).

    When ``embedder`` is ``None`` (lint-apply) we skip the embedding
    stream entirely. The document, chunks, and links still land so the
    next ``run_lint`` sees the page; embeddings reconcile on the next
    ``dikw ingest`` via ``doc.hash`` drift.
    """
    doc_id = f"{Layer.WIKI.value}:{normalize_path(page.path)}"
    abs_path = (root / page.path).resolve()
    parsed = parse_any(abs_path, rel_path=page.path)

    await storage.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            path=page.path,
            title=page.title,
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
    return len(unresolved)
