"""Source discovery.

Walks the ``sources`` entries from ``dikw.yml`` and yields files that match
each source's glob pattern while honoring its ignore list. Paths returned are
(absolute_path, logical_path) pairs where ``logical_path`` is relative to the
wiki root — this is what ends up in the ``documents.path`` column so the
engine stays portable across checkouts.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ...config import SourceConfig


def iter_source_files(
    sources: list[SourceConfig], *, root: Path
) -> Iterator[tuple[Path, str]]:
    """Yield (absolute, logical) path pairs for every file matching a source entry.

    ``wisdom/`` is a reserved first-class layer with its own ingest branch
    in ``api.ingest``; we hard-skip it here so a broad user config like
    ``sources: [{path: '.', pattern: '**/*.md'}]`` cannot double-yield
    wisdom files as ``Layer.SOURCE`` rows. Without this guard the same
    file ingests twice (once at ``source:wisdom/...``, once at
    ``wisdom:wisdom/...``), producing duplicate chunks + double embed
    spend for one on-disk page.
    """
    for src in sources:
        base = Path(src.path)
        if not base.is_absolute():
            base = (root / base).resolve()
        if not base.exists():
            continue
        ignore_spec = src.ignore

        for path in sorted(base.rglob(src.pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(root) if path.is_relative_to(root) else path
            rel_str = str(rel).replace("\\", "/")
            if rel_str == "wisdom" or rel_str.startswith("wisdom/"):
                continue
            if _matches_any(rel_str, ignore_spec) or _matches_any(
                str(path.relative_to(base)).replace("\\", "/"), ignore_spec
            ):
                continue
            yield path, rel_str


def _matches_any(path_str: str, patterns: list[str]) -> bool:
    from fnmatch import fnmatchcase

    return any(fnmatchcase(path_str, pat) for pat in patterns)
