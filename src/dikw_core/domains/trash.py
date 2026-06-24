"""Cross-layer soft-delete: move a file into ``<base>/trash/`` with an
audit stamp.

``move_to_trash`` is the layer-agnostic primitive behind every
soft-delete in the engine — the lint ``delete_page`` fixer (knowledge
pages) and the user-facing ``delete`` verb (any of ``sources/`` /
``knowledge/`` / ``wisdom/``). It lives here, at the ``domains`` root,
rather than inside ``domains/knowledge/lint_fix.py`` because the
destination it builds (``<base>/trash/<rel_path>``) mirrors whatever
layer prefix ``rel_path`` already carries — there is no knowledge-only
logic. A cross-layer caller importing it from the knowledge domain would
be an altitude smell; keeping it shared keeps the import graph honest.

Pure filesystem utility: depends only on stdlib + ``python-frontmatter``,
never on storage / config / the api facade.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import shutil
from pathlib import Path

import frontmatter

from ._atomic import reserve_path


def move_to_trash(
    *,
    base_root: Path,
    src_abs: Path,
    rel_path: str,
    reason: str,
    proposal_id: str = "",
) -> Path:
    """Move ``src_abs`` to ``<base_root>/trash/<rel_path>`` and stamp a
    ``trashed:`` frontmatter block on the file before the move.

    ``rel_path`` is the base-relative path of the file, layer prefix
    included (e.g. ``"knowledge/concepts/dead.md"``, ``"sources/notes/x.md"``,
    or ``"wisdom/elon-musk/never-sell.md"``). The file ends up at
    ``<base_root>/trash/<rel_path>`` so the original directory layout is
    preserved verbatim inside ``trash/`` — "rescue this file" is a plain
    ``mv`` back into place for the user. Collisions get a timestamp suffix
    so a re-trash of the same path doesn't clobber the earlier copy.

    ``reason`` records *why* the file was trashed (a ``LintKind`` from the
    lint fixers, or ``"delete"`` / a user-supplied note from the ``delete``
    verb). ``proposal_id`` ties a lint-originated trash back to its
    proposal; it is omitted from the audit block when empty (a direct
    ``delete`` has no proposal), so a manual delete reads as a clean
    ``trashed: {at, reason}``.

    Why frontmatter and not a separate manifest: the audit metadata
    lives WITH the file. A user grep-ing ``trash/`` for "what dropped
    this and when" doesn't have to cross-reference a sibling JSON.
    ``frontmatter.dumps`` round-trips other keys, so the ``trashed:``
    block is added in-place without rewriting body or losing existing
    metadata.

    Returns the destination path; raises ``OSError`` on filesystem failure.
    """
    # Read + parse BEFORE reserving a destination name, so a read failure
    # (file vanished mid-delete) can't strand a reserved empty file in trash/.
    raw = src_abs.read_text(encoding="utf-8")
    try:
        post = frontmatter.loads(raw)
    except Exception:
        # Malformed frontmatter: keep the body intact, don't try to
        # parse-and-rewrite — drop the file as-is into trash. Better to
        # preserve byte-identical contents than to risk content loss
        # while trying to inject an audit marker.
        post = None

    dest = base_root / "trash" / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        # Same path trashed twice in the same wall-clock second would
        # otherwise collide on a second-resolution suffix and overwrite
        # the earlier copy. Spin a millisecond counter and claim each
        # candidate atomically via ``reserve_path`` (O_CREAT | O_EXCL):
        # an ``exists()`` probe leaves a window where two concurrent
        # trashers both see the name free and the second clobbers the
        # first. Bounded at 1000 — the next wall-clock second rolls the
        # timestamp anyway. The reserved empty file is overwritten in
        # place by the move/replace below.
        base_dest = dest
        for ms in range(1000):
            ts = _dt.datetime.now(tz=_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
            suffix = f".{ts}.{ms:03d}" if ms else f".{ts}"
            candidate = base_dest.with_name(
                f"{base_dest.stem}{suffix}{base_dest.suffix}"
            )
            if reserve_path(candidate):
                dest = candidate
                break
        else:
            raise OSError(
                f"trash collision: {base_dest} exists and 1000 timestamp "
                "fallbacks were all taken"
            )

    if post is None:
        shutil.move(str(src_abs), str(dest))
        return dest
    trashed: dict[str, str] = {
        "at": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
        "reason": reason,
    }
    if proposal_id:
        trashed["proposal_id"] = proposal_id
    post.metadata["trashed"] = trashed
    # Two-stage write so a mid-write failure (disk full, short write)
    # cannot leave a partial file at the visible ``dest`` path: write
    # to a sibling ``.tmp`` first, then atomic-replace into place. On
    # failure clean up both the ``.tmp`` AND ``dest`` — on the collision
    # path ``dest`` is the empty file ``reserve_path`` created, which would
    # otherwise be stranded (on the non-collision path ``dest`` doesn't
    # exist yet, so the unlink is a suppressed no-op).
    tmp_dest = dest.with_name(dest.name + ".tmp")
    try:
        tmp_dest.write_text(frontmatter.dumps(post), encoding="utf-8")
    except OSError:
        with contextlib.suppress(OSError):
            tmp_dest.unlink()
        with contextlib.suppress(OSError):
            dest.unlink()
        raise
    try:
        tmp_dest.replace(dest)
    except OSError:
        with contextlib.suppress(OSError):
            tmp_dest.unlink()
        with contextlib.suppress(OSError):
            dest.unlink()
        raise
    try:
        src_abs.unlink()
    except OSError:
        # Roll back the trash copy so we never leave the same file in
        # BOTH its original tree and trash/. After rollback the file
        # stays where it was — if a caller already purged the doc row,
        # recovery is by re-indexing that file (a D-layer source
        # self-heals on the next ``dikw client ingest``, idempotent on
        # hash; a restored K/W page is re-projected by the
        # ``untracked_file`` drift lint, or rebuilt via ``synth --all`` /
        # ``wisdom write``), no manual SQL needed.
        with contextlib.suppress(OSError):
            dest.unlink()
        raise
    return dest


__all__ = ["move_to_trash"]
