"""W-layer persist entry point.

Owns the wisdom write pipeline: ``upsert_document`` (with the WISDOM-
only ``status`` field flowing through from frontmatter) + chunk + FTS
+ (inline embed when ``embedder`` + ``text_version_id`` given,
otherwise defer to next ingest's resume scan) + ``replace_links_from``
(cross-layer wikilinks) + ``replace_provenance_from`` (``sources:``
frontmatter).

Sole caller: ``api.write_wisdom_page`` (the public W-layer write
entry, surfaced via ``dikw client wisdom write`` and
``POST /v1/wisdom/write``). From 0.4.0 the engine's ``dikw client
ingest`` no longer scans ``<base>/wisdom/`` — W pages are indexed
exclusively when written through this path.

Implementation note: the K/W layered pipeline is shared with
``persist_knowledge`` via the private ``_persist_layered_page`` in
``domains.knowledge.page_index``. The two public functions differ in
(a) the ``Layer`` they pass through and (b) whether they preserve the
frontmatter ``status`` (W only). Keeping ``persist_wisdom`` in this
W-owned module makes the layer ownership obvious in import statements.
"""

from __future__ import annotations

from pathlib import Path

from ...providers.base import EmbeddingProvider
from ...schemas import Layer, WisdomPersistResult
from ...storage.base import Storage
from ..info.tokenize import CjkTokenizer
from ..knowledge.page_index import _persist_layered_page


async def persist_wisdom(
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
) -> WisdomPersistResult:
    """Index a W-layer page (hand-written wisdom) into storage.

    Owns: ``upsert_document`` (``status`` flows from parsed
    frontmatter — WISDOM-only field) + chunk + FTS + (inline embed
    when ``embedder`` and ``text_version_id`` are both supplied;
    otherwise defers to the next ingest's resume scan) +
    ``replace_links_from`` (wikilinks) + ``replace_provenance_from``
    (``sources:`` frontmatter).

    Single caller: ``api.write_wisdom_page``. The ``no_embed`` user
    flag is translated upstream by the caller into "do not pass
    embedder/text_version_id"; this function sees a uniform
    embedder-or-None signal.

    See ``domains/knowledge/page_index.py`` module docstring for the
    cross-layer wikilink resolve contract that applies to both K and W
    pages.
    """
    result = await _persist_layered_page(
        storage=storage,
        root=root,
        path=path,
        layer=Layer.WISDOM,
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
    return WisdomPersistResult(
        chunk_ids=result.chunk_ids,
        chunks_embedded=result.chunks_embedded,
        chunks_pending_embedding=result.chunks_pending_embedding,
        unresolved_wikilinks=result.unresolved_wikilinks,
        resolved_title=result.resolved_title,
    )
