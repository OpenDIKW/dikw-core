"""Wiki page I/O for the K (Knowledge) layer.

Pages are plain markdown files under ``wiki/`` with YAML front-matter. They
follow Obsidian-friendly conventions so the same folder can be opened in
Obsidian alongside the engine:

* ``id`` — stable K-page identifier (``K-<hash12>``).
* ``type`` — one of ``entity`` / ``concept`` / ``note`` (configurable).
* ``created`` / ``updated`` — ISO-8601 timestamps.
* ``tags`` — list of freeform tags.
* ``sources`` — list of D-layer paths this page summarises.

Page slugs are derived from the title (kebab-case, ASCII-safe). Folders
match the ``type`` by default: ``wiki/<type>s/<slug>.md``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import frontmatter

_DEFAULT_TYPES: tuple[str, ...] = ("entity", "concept", "note")
# Explicit folder map for the irregular built-in plurals (``entity`` →
# ``entities``, not ``entitys``). Custom page types added via
# ``SchemaConfig.page_types`` (e.g. ``topic`` → ``topics``) get a simple
# ``<type>s`` fallback so the Obsidian vault layout stays predictable.
_TYPE_FOLDERS: dict[str, str] = {
    "entity": "entities",
    "concept": "concepts",
    "note": "notes",
}
_SLUG_ILLEGAL = re.compile(r"[^a-z0-9]+")


def type_to_folder(type_: str) -> str:
    """Resolve a page ``type`` to its directory under ``wiki/``."""
    return _TYPE_FOLDERS.get(type_, f"{type_}s")


@dataclass(frozen=True)
class WikiPage:
    """In-memory representation of a K-layer wiki page."""

    path: str                 # wiki-relative, e.g. ``wiki/concepts/dikw.md``
    id: str
    type: str
    title: str
    body: str
    tags: list[str]
    sources: list[str]
    created: str
    updated: str
    extras: dict[str, Any]    # any front-matter keys we didn't explicitly model


def now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def make_page_id(title: str, type_: str) -> str:
    digest = hashlib.blake2b(f"{type_}:{title}".encode(), digest_size=6).hexdigest()
    return f"K-{digest}"


def slugify(title: str) -> str:
    ascii_ = title.lower().encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_ILLEGAL.sub("-", ascii_).strip("-")
    return slug or "untitled"


def default_page_path(type_: str, title: str) -> str:
    """Return the wiki-relative path the engine writes a new page to."""
    return f"wiki/{type_to_folder(type_)}/{slugify(title)}.md"


def read_page(root: Path, path: str) -> WikiPage:
    abs_path = (root / path).resolve()
    if not abs_path.is_file():
        raise FileNotFoundError(path)
    post = frontmatter.load(str(abs_path))
    meta = dict(post.metadata)
    return WikiPage(
        path=path,
        id=str(meta.pop("id", make_page_id(str(meta.get("title", path)), str(meta.get("type", "note"))))),
        type=str(meta.pop("type", "note")),
        title=str(meta.pop("title", _fallback_title(post.content, path))),
        body=post.content,
        tags=list(meta.pop("tags", []) or []),
        sources=list(meta.pop("sources", []) or []),
        created=str(meta.pop("created", now_iso())),
        updated=str(meta.pop("updated", now_iso())),
        extras=meta,
    )


def write_page(root: Path, page: WikiPage) -> Path:
    """Serialize ``page`` to disk under ``root / page.path``. Returns the absolute path."""
    abs_path = (root / page.path).resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "id": page.id,
        "type": page.type,
        "title": page.title,
        "created": page.created,
        "updated": page.updated,
    }
    if page.tags:
        meta["tags"] = page.tags
    if page.sources:
        meta["sources"] = page.sources
    meta.update(page.extras)
    post = frontmatter.Post(page.body.rstrip() + "\n", **meta)
    serialized = frontmatter.dumps(post)
    abs_path.write_text(serialized + "\n", encoding="utf-8")
    return abs_path


def build_page(
    *,
    title: str,
    body: str,
    type_: str = "note",
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    path: str | None = None,
    extras: dict[str, Any] | None = None,
) -> WikiPage:
    """Construct a fresh ``WikiPage`` with engine defaults filled in."""
    now = now_iso()
    return WikiPage(
        path=path or default_page_path(type_, title),
        id=make_page_id(title, type_),
        type=type_,
        title=title,
        body=body,
        tags=list(tags or []),
        sources=list(sources or []),
        created=now,
        updated=now,
        extras=dict(extras or {}),
    )


def _fallback_title(body: str, path: str) -> str:
    for line in body.splitlines():
        stripped = line.lstrip(" #\t").strip()
        if line.lstrip().startswith("#") and stripped:
            return stripped
    return Path(path).stem.replace("-", " ").title()
