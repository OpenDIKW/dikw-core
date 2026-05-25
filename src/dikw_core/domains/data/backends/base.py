"""Source-backend Protocol + registry.

Each backend handles a family of file extensions and knows how to turn a
file on disk into a ``ParsedDocument`` the engine can ingest. The engine
never imports concrete backend modules — it asks ``get_backend(path)`` or
``parse_any(path, rel_path=...)`` and the registry dispatches.

Adding a backend in a new phase means (a) implementing this Protocol and
(b) calling ``register`` in ``data/backends/__init__.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ....schemas import AssetRef, WisdomStatus


class ParsedDocument(BaseModel):
    """Backend-agnostic representation of a parsed source file."""

    path: str
    title: str
    body: str
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    hash: str
    mtime: float
    asset_refs: list[AssetRef] = Field(default_factory=list)
    # Frontmatter ``status:`` enum, parsed only when the raw value is in
    # the four-value set. An out-of-set value collapses to ``None`` so
    # ingest is never blocked — ``invalid_wisdom_status`` lint surfaces
    # the divergence using ``frontmatter.get("status")`` as the raw
    # spelling. Wiki/source layers may carry this value through the
    # parse but the engine clamps it to ``None`` at write time
    # (``api._to_document``) — status is wisdom-only.
    status: WisdomStatus | None = None


@runtime_checkable
class SourceBackend(Protocol):
    """Extension point for file-format-specific parsing."""

    extensions: tuple[str, ...]

    def parse(self, path: Path, *, rel_path: str) -> ParsedDocument: ...


_BY_EXT: dict[str, SourceBackend] = {}


def register(backend: SourceBackend) -> None:
    """Register a backend for each extension it declares."""
    for ext in backend.extensions:
        _BY_EXT[ext.lower()] = backend


def get_backend(path: Path | str) -> SourceBackend | None:
    ext = Path(path).suffix.lower()
    return _BY_EXT.get(ext)


def parse_any(path: Path, *, rel_path: str | None = None) -> ParsedDocument:
    """Dispatch to the backend registered for ``path``'s extension."""
    backend = get_backend(path)
    if backend is None:
        raise UnsupportedFormat(f"no backend registered for {path.suffix!r} ({path})")
    return backend.parse(path, rel_path=rel_path or str(path))


def supported_extensions() -> tuple[str, ...]:
    return tuple(sorted(_BY_EXT.keys()))


class UnsupportedFormat(RuntimeError):
    """Raised when a file's extension has no registered backend."""
