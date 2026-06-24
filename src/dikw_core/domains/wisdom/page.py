"""Wisdom page utilities (0.3.0 PR2).

The W layer is hand-authored markdown under ``wisdom/<author>/<slug>.md``.
Authorship is encoded by directory rather than a frontmatter field, so
users writing in Obsidian don't have to mirror their path in YAML.
``author_from_path`` is the deterministic extractor reused by ingest
(for log entries / future per-author counters), and by retrieve and
lint in PR3.

A wisdom file directly under ``wisdom/<slug>.md`` (no author subdir)
is allowed and indexed with ``author = None`` — the engine never
synthesises a placeholder author from the slug.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any

import frontmatter

from ...schemas import WisdomStatus
from .._atomic import atomic_write_text

# Kebab-case ASCII: lowercase letters / digits separated by single
# hyphens, no leading/trailing/double hyphens. Centralised here so the
# write API (engine + Pydantic schema + HTTP layer) all reject the same
# shapes — the on-disk path becomes part of the wisdom vault layout, so
# the engine doesn't accept anything Obsidian would render awkwardly
# (spaces, uppercase, underscores).
_KEBAB_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def validate_kebab(value: str, *, label: str) -> None:
    """Raise ``ValueError`` if ``value`` is not ASCII kebab-case.

    ``label`` names the offending field in the error message (``"slug"``
    or ``"author"``) so a caller layering this behind a higher-level API
    surfaces the precise input that needs fixing without inspecting the
    regex.
    """
    if not isinstance(value, str) or not _KEBAB_RE.match(value):
        raise ValueError(
            f"{label} must be ASCII kebab-case "
            f"(lowercase letters/digits, single hyphens, no leading/trailing/"
            f"double hyphens), got {value!r}"
        )


def make_wisdom_path(*, slug: str, author: str | None) -> str:
    """Return the logical wisdom path for ``(author, slug)``.

    With ``author`` set, the path is ``wisdom/<author>/<slug>.md``;
    without it, ``wisdom/<slug>.md``. Both inputs go through
    ``validate_kebab`` first so a malformed component fails fast at the
    write boundary, before any file I/O or storage write.
    """
    validate_kebab(slug, label="slug")
    if author is not None:
        validate_kebab(author, label="author")
        return f"wisdom/{author}/{slug}.md"
    return f"wisdom/{slug}.md"


# Frontmatter keys that the typed parameters own; ``extras`` is denied
# write access to them so a caller can't silently desynchronise the
# on-disk frontmatter from the validated request (and from the storage
# row, which always sees the typed values).
#
# ``author`` is reserved even though :func:`write_wisdom_file` itself
# doesn't take an ``author`` parameter — author is encoded by the
# on-disk directory (``wisdom/<author>/<slug>.md``), and allowing
# ``extras={"author": ...}`` to inject a contradicting value into the
# frontmatter would break the invariant that path author and
# frontmatter author agree.
#
# ``content`` and ``handler`` are reserved because they collide with
# ``frontmatter.Post.__init__(content, handler=None, **metadata)``:
# storing them in YAML frontmatter is fine in isolation, but every
# read-back through ``frontmatter.loads`` re-constructs
# ``Post(content, handler, **metadata)`` and raises ``TypeError`` (for
# ``content``) or silently replaces the dump handler with a string
# (for ``handler``, producing a file whose entire content is that
# literal string). Refusing them at write time keeps the on-disk vault
# round-trippable through the frontmatter library.
_RESERVED_FRONTMATTER_KEYS = frozenset(
    {"title", "status", "tags", "sources", "author", "content", "handler"}
)


def write_wisdom_file(
    root: Path,
    *,
    logical_path: str,
    title: str,
    body: str,
    status: WisdomStatus | None = None,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    """Serialize a wisdom page to disk under ``root / logical_path``.

    Returns the absolute path. Frontmatter stays intentionally minimal:
    wisdom is user-authored content, and the engine writes only the
    fields the caller explicitly supplied so a file round-tripped
    through this API isn't visibly different from one hand-edited in
    Obsidian. Distinct from :func:`domains.knowledge.page.write_page`,
    which always serialises ``id``/``type``/``created``/``updated`` —
    those are wiki-only frontmatter conventions and would pollute a
    user-authored wisdom file with engine metadata.

    ``logical_path`` must resolve under ``root`` — defense in depth for
    any caller that bypasses :func:`make_wisdom_path` (e.g. a unit test
    or a future write path that constructs its own relative path). The
    higher-level API enforces ASCII kebab-case on ``slug`` / ``author``
    via ``make_wisdom_path`` so this guard only fires on a direct
    misuse, not on validated user input.

    ``extras`` is a passthrough for caller-supplied frontmatter, but it
    is denied write access to the reserved keys
    (``title``/``status``/``tags``/``sources``) — those are owned by
    the typed parameters and silently overwriting them would
    desynchronise the on-disk frontmatter from the storage row.
    """
    abs_path = (root / logical_path).resolve()
    try:
        abs_path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"wisdom logical_path {logical_path!r} resolves outside root "
            f"{root!s}"
        ) from exc
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"title": title}
    if status is not None:
        meta["status"] = status.value
    if tags:
        meta["tags"] = list(tags)
    if sources:
        meta["sources"] = list(sources)
    if extras:
        for key, value in extras.items():
            if key in _RESERVED_FRONTMATTER_KEYS:
                continue
            meta[key] = value
    # Construct the Post with body only, then assign metadata via the
    # dict. ``frontmatter.Post`` is ``(content, handler=None, **metadata)``
    # — using ``**meta`` to pass user-supplied keys would let
    # ``extras={"handler": "evil"}`` silently overwrite the dump handler
    # (verified: frontmatter.dumps then returns the raw string "evil"
    # and the on-disk file loses title + body entirely). Assigning to
    # ``post.metadata`` bypasses the kwarg path and treats every key as
    # frontmatter metadata, immune to collisions with ``Post.__init__``
    # parameter names regardless of which keys are in extras.
    post = frontmatter.Post(body.rstrip() + "\n")
    post.metadata.update(meta)
    atomic_write_text(abs_path, frontmatter.dumps(post) + "\n")
    return abs_path


def author_from_path(path: str) -> str | None:
    """Return the author directory for a wisdom page path, or ``None``.

    ``path`` is the logical (POSIX-style, forward-slash) path the engine
    stores in ``DocumentRecord.path`` — e.g.
    ``wisdom/elon-musk/first-principles.md``. Anything not anchored at
    ``wisdom/<author>/<...>.md`` returns ``None`` (a file directly under
    ``wisdom/``, a non-wisdom path, or an empty input).
    """
    if not path:
        return None
    parts = PurePosixPath(path).parts
    # Need at least ``wisdom`` + author + a tail (the file itself, or
    # a deeper subtree the file lives under).
    if len(parts) < 3 or parts[0] != "wisdom":
        return None
    return parts[1]
