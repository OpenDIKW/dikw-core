"""Storage factory: resolves a backend instance from a ``StorageConfig``."""

from __future__ import annotations

from pathlib import Path

from ..config import (
    PostgresStorageConfig,
    SQLiteStorageConfig,
    StorageConfig,
)
from ..domains.info.tokenize import CjkTokenizer
from .base import NotSupported, Storage, StorageError
from .sqlite import SQLiteStorage


def build_storage(
    config: StorageConfig,
    *,
    root: str | Path | None = None,
    cjk_tokenizer: CjkTokenizer = "none",
) -> Storage:
    """Instantiate a Storage adapter. Paths are resolved relative to ``root`` if given.

    ``cjk_tokenizer`` controls how CJK text is segmented before being
    written to the backend's full-text index. Both backends honour it
    symmetrically via ``preprocess_for_fts`` on ingest + query: SQLite
    inserts the segmented body into ``documents_fts``; Postgres feeds
    the same segmented string through ``to_tsvector('simple', …)`` into
    a plain ``chunks.fts`` tsvector column. Same Python helper, same
    byte-level result on both adapters.
    See ``RetrievalConfig.cjk_tokenizer`` for the base-level surface.
    """
    if isinstance(config, SQLiteStorageConfig):
        path = Path(config.path)
        if not path.is_absolute() and root is not None:
            path = Path(root) / path
        return SQLiteStorage(path, cjk_tokenizer=cjk_tokenizer)
    if isinstance(config, PostgresStorageConfig):
        try:
            from .postgres import PostgresStorage
        except ImportError as e:  # pragma: no cover - only without the extra
            raise StorageError(
                "Postgres adapter requires the `postgres` extra — "
                "install via `uv pip install dikw-core[postgres]`"
            ) from e
        return PostgresStorage(
            config.dsn,
            schema=config.schema_,
            pool_size=config.pool_size,
            cjk_tokenizer=cjk_tokenizer,
        )
    raise StorageError(f"unknown storage backend: {config!r}")


__all__ = [
    "NotSupported",
    "SQLiteStorage",
    "Storage",
    "StorageError",
    "build_storage",
]
