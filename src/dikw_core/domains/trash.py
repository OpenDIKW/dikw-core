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
    dest = base_root / "trash" / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        # Same path trashed twice in the same wall-clock second would
        # otherwise collide on a second-resolution suffix and overwrite
        # the earlier copy. Spin a millisecond counter until the
        # candidate name is free — bounded at 1000 because the next
        # second will roll the timestamp anyway.
        base_dest = dest
        for ms in range(1000):
            ts = _dt.datetime.now(tz=_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
            suffix = f".{ts}.{ms:03d}" if ms else f".{ts}"
            candidate = base_dest.with_name(
                f"{base_dest.stem}{suffix}{base_dest.suffix}"
            )
            if not candidate.exists():
                dest = candidate
                break
        else:
            raise OSError(
                f"trash collision: {base_dest} exists and 1000 timestamp "
                "fallbacks were all taken"
            )

    raw = src_abs.read_text(encoding="utf-8")
    try:
        post = frontmatter.loads(raw)
    except Exception:
        # Malformed frontmatter: keep the body intact, don't try to
        # parse-and-rewrite — drop the file as-is into trash. Better to
        # preserve byte-identical contents than to risk content loss
        # while trying to inject an audit marker.
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
    # to a sibling ``.tmp`` first, then atomic-replace into place. A
    # failed ``write_text`` only leaves the ``.tmp`` to clean up; only
    # after the rename does ``dest`` materialise visibly under
    # ``trash/``.
    tmp_dest = dest.with_name(dest.name + ".tmp")
    try:
        tmp_dest.write_text(frontmatter.dumps(post), encoding="utf-8")
    except OSError:
        with contextlib.suppress(OSError):
            tmp_dest.unlink()
        raise
    try:
        tmp_dest.replace(dest)
    except OSError:
        with contextlib.suppress(OSError):
            tmp_dest.unlink()
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
