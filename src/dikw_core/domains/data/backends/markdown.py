"""Markdown source backend.

Parses a markdown file into a ``ParsedDocument``: front-matter dict, body
text (with front-matter stripped), title, and a stable content hash.

Title resolution order:
1. ``title:`` in front-matter
2. First ATX heading (``# Title``) in the body
3. File stem (``my-note`` -> ``my-note``)

The asset-reference regex pair lives in ``dikw_core.md_inspect`` so the
client import command (which can't import from ``domains/``) and the
D-layer parser both share one source of truth. ``extract_image_refs``
is re-exported here for existing callers.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import frontmatter

from ....md_inspect import extract_image_refs
from ....schemas import WisdomStatus
from .base import ParsedDocument

# Backwards-compatible alias for existing callers.
ParsedMarkdown = ParsedDocument

_ATX_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def content_hash(body: str) -> str:
    """SHA-256 of the raw body; stable across runs so D-layer rows are idempotent."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _first_heading(body: str) -> str | None:
    m = _ATX_HEADING.search(body)
    return m.group(1).strip() if m else None


_VALID_STATUS = {s.value for s in WisdomStatus}


def _parse_status(raw: Any) -> WisdomStatus | None:
    """Map a frontmatter ``status`` value to the enum, or ``None``.

    The raw frontmatter is preserved in ``ParsedDocument.frontmatter``
    so ``invalid_wisdom_status`` lint can quote the bad spelling — we
    only collapse to ``None`` here to keep ingest non-blocking.
    """
    if not isinstance(raw, str):
        return None
    return WisdomStatus(raw) if raw in _VALID_STATUS else None


def parse_text(*, path: str, text: str, mtime: float) -> ParsedDocument:
    """Parse raw markdown text. Exposed so callers can test without filesystem I/O."""
    post = frontmatter.loads(text)
    body = post.content
    fm: dict[str, Any] = dict(post.metadata)

    title = fm.get("title") or _first_heading(body) or Path(path).stem

    return ParsedDocument(
        path=path,
        title=str(title),
        body=body,
        frontmatter=fm,
        hash=content_hash(body),
        mtime=mtime,
        asset_refs=extract_image_refs(body),
        status=_parse_status(fm.get("status")),
    )


def parse_file(path: Path, *, rel_path: str | None = None) -> ParsedDocument:
    """Read ``path`` and return a ``ParsedDocument``. ``rel_path`` becomes the D-layer path."""
    text = path.read_text(encoding="utf-8")
    mtime = path.stat().st_mtime
    return parse_text(path=rel_path or str(path), text=text, mtime=mtime)


class MarkdownBackend:
    """``SourceBackend`` impl for .md / .markdown files."""

    extensions: tuple[str, ...] = (".md", ".markdown")

    def parse(self, path: Path, *, rel_path: str) -> ParsedDocument:
        return parse_file(path, rel_path=rel_path)
