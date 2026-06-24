"""Filesystem write primitives shared across the DIKW layers.

Two stdlib-only helpers that make on-disk writes safe against a crash or a
concurrent writer:

* :func:`atomic_write_text` — write to a sibling temp file, then
  ``os.replace`` it into place, so a reader never sees a half-written file
  and a failed write leaves the existing file exactly as it was.
* :func:`reserve_path` — atomically claim a not-yet-existing path via
  ``O_CREAT | O_EXCL``, closing the check-then-write TOCTOU that a
  ``Path.exists()`` probe leaves open.

Neither fsyncs: the goal is *atomic visibility* (a reader sees old-or-new,
never partial), not durability across power loss — matching the existing
two-stage writes in ``data/assets.py`` and ``trash.py``.
"""

from __future__ import annotations

import contextlib
import os
import stat
from pathlib import Path


def atomic_write_text(path: Path, data: str, *, encoding: str = "utf-8") -> None:
    """Atomically write ``data`` to ``path``.

    Writes a sibling ``<name>.<rand>.tmp`` first, then ``os.replace``s it
    onto ``path`` (atomic for a same-directory rename on POSIX and Windows).
    On any failure the temp file is removed and ``path`` is left untouched,
    so a reader never observes a truncated or half-written file.

    When ``path`` already exists its permission bits are copied onto the temp
    file before the replace: ``os.replace`` swaps inodes, so without this a
    page the user had ``chmod``-ed (e.g. ``0o600``) would silently inherit the
    process umask. The plain ``Path.write_text`` this replaced kept the
    existing inode + mode, so this preserves that behaviour.
    """
    tmp = path.with_name(f"{path.name}.{os.urandom(6).hex()}.tmp")
    try:
        tmp.write_text(data, encoding=encoding)
        with contextlib.suppress(FileNotFoundError):
            os.chmod(tmp, stat.S_IMODE(os.stat(path).st_mode))
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def reserve_path(path: Path) -> bool:
    """Atomically create ``path`` as an empty file, claiming the name.

    Returns ``True`` when this call created it and ``False`` when it already
    existed. Uses ``O_CREAT | O_EXCL`` so two racing callers can never both
    win the same name — unlike a ``path.exists()`` probe, which leaves a
    window between the check and the create.
    """
    try:
        # 0o600 placeholder: the reserved file is a momentary name claim,
        # overwritten in place by the caller's replace/move (which carries the
        # real content's mode). Keep it owner-only rather than world-readable.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    os.close(fd)
    return True


__all__ = ["atomic_write_text", "reserve_path"]
