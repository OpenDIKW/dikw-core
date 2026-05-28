"""D-layer (source) persist entry point.

Owns the per-source-file write pipeline: ``upsert_document`` +
``chunk_markdown`` (with ``atomic_spans`` from already-materialized
asset refs) + ``replace_chunks`` (which side-effect-writes FTS in the
same storage transaction) + ``replace_chunk_asset_refs`` for the
chunk ↔ asset bridge.

Asymmetry with persist_knowledge / persist_wisdom: D-layer **defers**
embedding to ``api.ingest``'s end-of-scan bulk pass so the embedder
batches across all source files for throughput. ``persist_source``
returns the chunk_ids the caller needs to feed into its shared
``to_embed`` queue; the bulk pass at ingest end runs the per-batch
retry-skip + the cross-layer missing-embedding resume scan.

Asset materialize (the read-file → upsert_asset step) is also the
caller's responsibility: it crosses storage AND filesystem and needs
``mm_cfg`` / project_root context that the persist path shouldn't
re-derive. The caller passes the pre-materialized
``ref_assets: dict[int, AssetRecord]`` keyed by asset_ref index, and
``persist_source`` projects body-relative ref offsets into
chunk-relative offsets for the ``chunk_asset_refs`` rows.

Single caller: ``api.ingest`` — D-layer is index-by-batch by design;
there is no per-file public write entry for sources (users place
markdown into ``<base>/sources/`` and run ingest).
"""

from __future__ import annotations

from ...schemas import (
    AssetRecord,
    ChunkAssetRef,
    ChunkRecord,
    DocumentRecord,
    SourcePersistResult,
)
from ...storage.base import Storage
from ..data.backends.base import ParsedDocument
from ..info.chunk import chunk_markdown
from ..info.tokenize import CjkTokenizer


async def persist_source(
    *,
    storage: Storage,
    parsed: ParsedDocument,
    doc: DocumentRecord,
    ref_assets: dict[int, AssetRecord],
    cjk_tokenizer: CjkTokenizer,
) -> SourcePersistResult:
    """Index a single source markdown into D-layer storage.

    The caller has already (a) parsed the file, (b) decided this row
    needs re-indexing (hash drift or active=False), and (c)
    materialized any asset refs into ``ref_assets``. This function
    handles the storage-side writes.

    ``ref_assets`` maps each asset_ref's index (0-based, in the order
    they appear in ``parsed.asset_refs``) to its resolved
    ``AssetRecord``. Refs that failed materialization (remote URL,
    missing file) are absent from the dict — their chunks still land,
    just without a ``chunk_asset_refs`` bridge row.
    """
    await storage.upsert_document(doc)

    atomic_spans = [(r.start, r.end) for r in parsed.asset_refs]
    chunks = chunk_markdown(
        parsed.body,
        atomic_spans=atomic_spans,
        cjk_tokenizer=cjk_tokenizer,
    )
    chunk_records = [
        ChunkRecord(
            doc_id=doc.doc_id, seq=c.seq, start=c.start, end=c.end, text=c.text
        )
        for c in chunks
    ]
    chunk_ids = await storage.replace_chunks(doc.doc_id, chunk_records)

    # Project body-relative ref offsets into chunk-relative offsets and
    # persist the chunk ↔ asset bridge rows.
    for chunk_record, chunk_id in zip(chunk_records, chunk_ids, strict=True):
        chunk_refs: list[ChunkAssetRef] = []
        ord_counter = 0
        for ref_idx, ref in enumerate(parsed.asset_refs):
            if not (
                chunk_record.start <= ref.start and ref.end <= chunk_record.end
            ):
                continue
            asset = ref_assets.get(ref_idx)
            if asset is None:
                continue  # unresolved (remote URL, missing file) — already logged
            chunk_refs.append(
                ChunkAssetRef(
                    chunk_id=chunk_id,
                    asset_id=asset.asset_id,
                    ord=ord_counter,
                    alt=ref.alt,
                    start_in_chunk=ref.start - chunk_record.start,
                    end_in_chunk=ref.end - chunk_record.start,
                )
            )
            ord_counter += 1
        if chunk_refs:
            await storage.replace_chunk_asset_refs(chunk_id, chunk_refs)

    return SourcePersistResult(
        chunk_ids=chunk_ids,
        chunk_texts=[r.text for r in chunk_records],
        chunks_count=len(chunk_records),
    )
