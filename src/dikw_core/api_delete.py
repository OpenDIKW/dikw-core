"""Delete cluster of the engine facade: ``delete_page``.

``delete_page`` is the immediate, single-document delete verb. It spans
all three DIKW layers (``sources/`` / ``knowledge/`` / ``wisdom/``):
resolve which layer the path lives in (storage probe), purge the document
row + its outgoing edges via ``Storage.delete_document``, then soft-delete
the on-disk file to ``<base>/trash/<rel>`` with an audit stamp. ``trash/``
is the recovery safety net, so the verb runs immediately — no propose/apply
gate (contrast ``lint``'s scan-discovered batch hygiene). It is the write
-class inverse of ``write_wisdom_page``.

rank3 cluster: imports ``api_core`` (``_with_storage``), ``api_path_safety``
(``_assert_within``), the leaf ``api_types`` exceptions, the shared
``domains.trash`` soft-delete primitive, and ``schemas`` — never the ``api``
facade. ``api`` re-exports ``delete_page`` (public, in ``__all__``).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

from .api_core import _with_storage
from .api_path_safety import _assert_within
from .api_types import PageNotFound
from .domains.data.path_norm import doc_id_for as _doc_id_for
from .domains.trash import move_to_trash
from .progress import NoopReporter, ProgressReporter
from .schemas import DeleteReport, DocumentRecord, KnowledgeLogEntry, Layer


async def delete_page(
    root: str | Path | None,
    path: str,
    *,
    reason: str | None = None,
    reporter: ProgressReporter | None = None,
) -> DeleteReport:
    """Delete a registered document at ``path`` (D, K, or W layer).

    Resolves which layer ``path`` belongs to by probing each layer's
    deterministic ``doc_id`` against storage — path safety is index-driven
    (only registered paths are reachable; ``..`` traversal, unindexed
    files, and out-of-base paths all surface as :class:`PageNotFound`),
    matching :func:`read_page`. Unlike ``read_page`` the probe matches
    **regardless of ``active``**, so a half-written (``active=False``) row
    is still deletable.

    Storage rows are purged FIRST (``delete_document``), then the on-disk
    file is moved to ``<base>/trash/<rel>``. The ordering mirrors lint's
    ``delete_page`` fixer: if the trash move fails after the row is gone,
    the file still sits at its original path and the next
    ``dikw client ingest`` re-creates the row (idempotent on hash); the
    reverse order would strand an orphaned row pointing at a missing file.

    ``delete_document`` clears the doc's **outgoing** links + provenance.
    Inbound edges from *live* pages are deliberately left to surface as
    ``broken_wikilink`` (and dangling provenance) on the next lint — the
    verb never silently rewrites another page's body or frontmatter.

    A row whose backing file is already gone (the ``missing_file`` drift
    case) still purges cleanly: there is nothing to trash, so
    :attr:`DeleteReport.trashed_to` is ``None``. ``reason`` is stamped into
    the trashed file's audit block (default ``"delete"``).
    """
    used_reporter: ProgressReporter = reporter or NoopReporter()

    # Reject obviously-malformed paths up front so a bare ``Path()`` call
    # later can't surface as a 500 (``\x00`` raises ValueError on Linux).
    # Empty / whitespace-only can't legitimately match a document. Mirrors
    # ``read_page``'s guard.
    if not path or not path.strip() or "\x00" in path:
        raise PageNotFound(path)

    cfg, base_root, storage = await _with_storage(root)
    del cfg
    try:
        # Probe each registered layer for the path's deterministic doc_id.
        # First hit wins — a layer-prefixed path (``sources/…`` /
        # ``knowledge/…`` / ``wisdom/…``) can only match its own layer, so
        # the loop is unambiguous. Match regardless of ``active``.
        match: DocumentRecord | None = None
        for layer in (Layer.SOURCE, Layer.KNOWLEDGE, Layer.WISDOM):
            candidate = await storage.get_document(_doc_id_for(layer, path))
            if candidate is not None:
                match = candidate
                break
        if match is None:
            raise PageNotFound(path)

        # Defence in depth: a row registered with a path that escapes the
        # base root is corruption — refuse rather than trash outside the
        # base. ``_assert_within`` resolves both sides so ``..`` and
        # in-tree symlink escapes are caught.
        try:
            abs_path = _assert_within(base_root, base_root / match.path)
        except ValueError as e:
            raise PageNotFound(path) from e

        used_reporter.cancel_token().raise_if_cancelled()
        await used_reporter.progress(
            phase="delete",
            current=0,
            total=1,
            detail={"path": match.path, "layer": match.layer.value, "step": "purging"},
        )

        # Purge storage rows first (see ordering rationale in the docstring).
        await storage.delete_document(match.doc_id)

        # Move the on-disk file to trash. A file already gone is not an
        # error — the row is what we deleted. ``move_to_trash`` does sync
        # filesystem I/O (read / write / replace / unlink); offload it so a
        # slow disk doesn't stall the event loop alongside other in-flight
        # requests.
        trashed_to: str | None = None
        if abs_path.is_file():
            def _trash() -> Path:
                return move_to_trash(
                    base_root=base_root,
                    src_abs=abs_path,
                    rel_path=match.path,
                    reason=reason or "delete",
                )

            dest = await asyncio.to_thread(_trash)
            trashed_to = dest.relative_to(base_root).as_posix()

        await storage.append_knowledge_log(
            KnowledgeLogEntry(ts=time.time(), action="delete", src=match.path)
        )
        await used_reporter.progress(
            phase="delete",
            current=1,
            total=1,
            detail={"path": match.path, "step": "done"},
        )
        return DeleteReport(
            path=match.path,
            layer=match.layer,
            trashed_to=trashed_to,
        )
    finally:
        # Match ``write_wisdom_page``: a close() error must not shadow the
        # real cause carried by an in-flight exception.
        with contextlib.suppress(Exception):
            await storage.close()


__all__ = ["delete_page"]
