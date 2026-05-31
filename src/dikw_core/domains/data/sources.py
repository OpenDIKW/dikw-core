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


def _resolve_source_root(src: SourceConfig, base_root: Path) -> Path:
    """Absolute, resolved root directory for one ``sources`` entry.

    A relative ``path`` is anchored at the base root; an absolute one is
    taken as-is. Both are ``resolve()``d so the containment check in
    :func:`iter_source_files` sees a normalized path.
    """
    p = Path(src.path)
    if not p.is_absolute():
        p = base_root / p
    return p.resolve()


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

    ``sources`` is a managed tree under the base. A configured ``path``
    that resolves outside the base root (a ``../`` prefix or an absolute
    path elsewhere) is a config error, not a license to read + index
    arbitrary files into the ``Layer.SOURCE`` index (their doc-ids would
    also degrade to absolute paths). Validate every entry UP FRONT and
    raise: the generator is lazy, so a per-iteration check would let
    earlier sources index before a later bad entry aborts — we want
    all-or-nothing.
    """
    root_resolved = root.resolve()
    for src in sources:
        base = _resolve_source_root(src, root)
        if not base.is_relative_to(root_resolved):
            raise ValueError(
                f"source path {src.path!r} resolves to {base}, outside the "
                f"base root {root_resolved}; sources must live under the base."
            )

    for src in sources:
        base = _resolve_source_root(src, root)
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
