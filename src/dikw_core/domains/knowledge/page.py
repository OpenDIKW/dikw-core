"""Knowledge page I/O for the K (Knowledge) layer.

Pages are plain markdown files under ``knowledge/`` with YAML front-matter.
They follow Obsidian-friendly conventions so the same folder can be opened in
Obsidian alongside the engine:

* ``id`` — stable K-page identifier (``K-<hash12>``).
* ``category`` — the page's node in the configured ``schema.categories``
  taxonomy (e.g. ``concept`` or ``技术/架构``); also the on-disk folder path.
* ``created`` / ``updated`` — ISO-8601 timestamps.
* ``tags`` — list of freeform tags.
* ``sources`` — list of D-layer paths this page summarises.

A page is filed at ``knowledge/<category>/<slug>.md``: the ``category`` is a
slash-separated, arbitrary-depth folder path used verbatim (validated as a
closed set at config load, see ``config.CategoryNode``), and ``<slug>`` is an
ASCII-kebab slug derived from the title.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import frontmatter

_SLUG_ILLEGAL = re.compile(r"[^a-z0-9]+")

# The stem ``slugify`` collapses to when a title carries no ASCII/CJK-romanised
# characters (e.g. a pure-CJK title with no LLM-provided pinyin slug). Exposed as
# a constant so the ``title_slug_quality`` lint can detect the degenerate slug
# without copy-pasting the literal.
SLUG_FALLBACK = "untitled"


def category_from_path(path: str) -> str:
    """Reverse-derive a page's ``category`` (folder path) from its base-relative path.

    The category is everything between the ``knowledge/`` root and the
    filename, joined by ``/`` — arbitrary depth (``knowledge/技术/架构/rrf.md``
    → ``技术/架构``). Used by callers (e.g. the synth existing-pages section)
    that have a ``DocumentRecord`` — which doesn't carry ``category`` — and need
    the label without paying frontmatter I/O. Returns ``""`` for a file
    directly under ``knowledge/`` (no category folder) or a non-knowledge path.
    """
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "knowledge":
        return "/".join(parts[1:-1])
    return ""


@dataclass(frozen=True)
class KnowledgePage:
    """In-memory representation of a K-layer knowledge page."""

    path: str                 # base-relative, e.g. ``knowledge/技术/架构/rrf.md``
    id: str
    category: str             # taxonomy node / folder path, e.g. ``concept`` or ``技术/架构``
    title: str
    body: str
    tags: list[str]
    sources: list[str]
    created: str
    updated: str
    extras: dict[str, Any]    # any front-matter keys we didn't explicitly model


def now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def frontmatter_str_list(metadata: dict[str, Any], key: str) -> list[str]:
    """Read a list-of-strings frontmatter field defensively.

    ``sources:`` and ``tags:`` are both user-editable list fields on
    every K-page. A hand-written scalar (``sources: foo.md``) parses as
    a string, not a single-item list — iterating it would yield one
    character per row. A dict / int / null value would raise. This
    helper collapses all three malformed shapes to ``[]`` and drops
    non-string entries from a well-formed list, so every caller can
    write ``for item in frontmatter_str_list(meta, "sources")`` without
    a per-site ``isinstance`` guard. See ADR-0001 for why ``sources``
    in particular must never propagate garbage into the provenance
    table.
    """
    raw = metadata.get(key)
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str)]


def make_page_id(title: str, category: str) -> str:
    digest = hashlib.blake2b(f"{category}:{title}".encode(), digest_size=6).hexdigest()
    return f"K-{digest}"


def slugify(title: str) -> str:
    ascii_ = title.lower().encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_ILLEGAL.sub("-", ascii_).strip("-")
    return slug or SLUG_FALLBACK


def default_page_path(category: str, title: str) -> str:
    """Return the base-relative path the engine writes a new page to.

    The ``category`` path is the on-disk folder, used verbatim (it is a
    config-validated closed-set value); only the filename is slugified. An
    empty ``category`` (a page directly under ``knowledge/`` — e.g. a
    hand-created root-level page an orphan-merge targets) collapses to
    ``knowledge/<slug>.md`` rather than emitting a ``knowledge//<slug>.md``
    double slash, so it round-trips with :func:`category_from_path`.
    """
    prefix = f"knowledge/{category}/" if category else "knowledge/"
    return f"{prefix}{slugify(title)}.md"


def read_page(root: Path, path: str) -> KnowledgePage:
    abs_path = (root / path).resolve()
    if not abs_path.is_file():
        raise FileNotFoundError(path)
    post = frontmatter.load(str(abs_path))
    meta = dict(post.metadata)
    # ``tags`` and ``sources`` go through the shared malformed-shape
    # guard for the same reason ``persist_knowledge`` / ``run_lint`` /
    # ``MissingProvenanceFixer`` do — a hand-written YAML scalar
    # (``sources: foo.md``) would otherwise become a character-per-row
    # list. The ``pop`` happens explicitly first so ``extras`` doesn't
    # carry the raw value back out.
    tags = frontmatter_str_list(meta, "tags")
    sources = frontmatter_str_list(meta, "sources")
    meta.pop("tags", None)
    meta.pop("sources", None)
    # ``category`` defaults to the page's folder path when frontmatter omits it
    # (a hand-edited page) — the folder is the category by construction.
    category = str(meta.pop("category", None) or category_from_path(path))
    return KnowledgePage(
        path=path,
        id=str(meta.pop("id", make_page_id(str(meta.get("title", path)), category))),
        category=category,
        title=str(meta.pop("title", _fallback_title(post.content, path))),
        body=post.content,
        tags=tags,
        sources=sources,
        created=str(meta.pop("created", now_iso())),
        updated=str(meta.pop("updated", now_iso())),
        extras=meta,
    )


def write_page(root: Path, page: KnowledgePage) -> Path:
    """Serialize ``page`` to disk under ``root / page.path``. Returns the absolute path."""
    abs_path = (root / page.path).resolve()
    # Defense in depth: refuse a page.path that escapes the base before any
    # mkdir/write. The synth parser rejects traversal paths upstream
    # (#146/#149), but write_page is a shared sink (lint-apply + future
    # writers) so it guards its own input. Mirrors write_wisdom_file.
    try:
        abs_path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"knowledge page path {page.path!r} resolves outside base {root!s}"
        ) from exc
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "id": page.id,
        "category": page.category,
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
    category: str = "note",
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    path: str | None = None,
    extras: dict[str, Any] | None = None,
) -> KnowledgePage:
    """Construct a fresh ``KnowledgePage`` with engine defaults filled in."""
    now = now_iso()
    return KnowledgePage(
        path=path or default_page_path(category, title),
        id=make_page_id(title, category),
        category=category,
        title=title,
        body=body,
        tags=list(tags or []),
        sources=list(sources or []),
        created=now,
        updated=now,
        extras=dict(extras or {}),
    )


def path_slug_title(path: str) -> str:
    """Derive a human-readable title from a knowledge page's path stem.

    The convention across the K layer: ``knowledge/concept/topic-a.md`` →
    ``"Topic A"``. Centralised here so ``_fallback_title`` and
    ``lint_fix._op_title`` agree on the same rule — if we ever change it
    (NFKC, CJK handling, etc.), one edit covers every caller.
    """
    return Path(path).stem.replace("-", " ").title()


def _fallback_title(body: str, path: str) -> str:
    for line in body.splitlines():
        stripped = line.lstrip(" #\t").strip()
        if line.lstrip().startswith("#") and stripped:
            return stripped
    return path_slug_title(path)
