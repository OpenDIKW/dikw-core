"""Tests for the cross-layer soft-delete primitive ``move_to_trash``.

``move_to_trash`` was promoted out of ``domains/knowledge/lint_fix.py`` into
the shared ``domains/trash.py`` so the user-facing ``delete`` verb can reuse
it for D / K / W files (not just knowledge pages). These tests pin the
filesystem invariants directly against the helper: collision-safety,
two-stage-write atomicity, unlink rollback, layer-prefix preservation, and
the audit ``trashed:`` frontmatter block.

The lint-apply path that drives the helper through ``delete_page`` is
covered separately in ``tests/test_lint_apply.py``.
"""

from __future__ import annotations

from pathlib import Path

import frontmatter
import pytest

from dikw_core.domains import trash
from dikw_core.domains.trash import move_to_trash


def test_move_to_trash_collision_does_not_overwrite(tmp_path: Path) -> None:
    """Two trashes of the same path within the same second must not
    overwrite each other — a second-resolution timestamp suffix would
    collide when called twice in a tight loop, silently losing the
    earlier soft-deleted copy."""
    base_root = tmp_path
    rel = "knowledge/concepts/twice.md"
    src1 = base_root / rel
    src1.parent.mkdir(parents=True, exist_ok=True)
    src1.write_text(
        "---\ntitle: First\n---\nfirst version\n", encoding="utf-8",
    )
    dest1 = move_to_trash(
        base_root=base_root, src_abs=src1, rel_path=rel,
        reason="duplicate_title", proposal_id="p1",
    )
    # Recreate the file at the same path and re-trash immediately —
    # the timestamp will collide on the second-resolution suffix.
    src2 = base_root / rel
    src2.write_text(
        "---\ntitle: Second\n---\nsecond version\n", encoding="utf-8",
    )
    dest2 = move_to_trash(
        base_root=base_root, src_abs=src2, rel_path=rel,
        reason="orphan_page", proposal_id="p2",
    )
    # Both files survive in trash with distinct names.
    assert dest1 != dest2
    assert dest1.is_file()
    assert dest2.is_file()
    assert frontmatter.loads(dest1.read_text(encoding="utf-8")).content.strip() \
        == "first version"
    assert frontmatter.loads(dest2.read_text(encoding="utf-8")).content.strip() \
        == "second version"


def test_move_to_trash_partial_write_leaves_no_dest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the trash write fails partway (disk full, short write), the
    function must not leave a partial file at the visible ``dest`` path.

    The two-stage write (tmp → atomic replace) means a failed write
    only leaves a ``.tmp`` to clean up; the visible ``dest`` is never
    materialised until the rename succeeds."""
    base_root = tmp_path
    rel = "knowledge/concepts/half-written.md"
    src = base_root / rel
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        "---\ntitle: HalfWritten\n---\nbody\n", encoding="utf-8",
    )

    original_write_text = Path.write_text

    def _fake_write_text(self: Path, *args: object, **kwargs: object) -> None:
        # Refuse only when writing into the trash subtree; let src
        # writes and any other tmp paths outside trash succeed.
        if str(self).startswith(str(base_root / "trash")):
            raise OSError("simulated disk full")
        return original_write_text(self, *args, **kwargs)  # type: ignore[no-any-return]

    monkeypatch.setattr(trash.Path, "write_text", _fake_write_text)

    with pytest.raises(OSError, match="disk full"):
        move_to_trash(
            base_root=base_root, src_abs=src, rel_path=rel,
            reason="orphan_page", proposal_id="p-partial",
        )

    # Post-failure: src still in place, no partial dest, no leftover tmp.
    assert src.is_file()
    trash_dir = base_root / "trash" / "knowledge" / "concepts"
    if trash_dir.exists():
        leftovers = list(trash_dir.iterdir())
        assert leftovers == [], (
            f"trash/ must be empty after a failed write, got: {leftovers}"
        )


def test_move_to_trash_uses_atomic_tmp_then_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the two-stage write: bytes land in a sibling ``.tmp`` and only an
    atomic ``replace`` materialises ``dest``. Failing the ``replace`` step
    must clean up the ``.tmp`` and leave src untouched. A naive
    direct-to-``dest`` implementation would never call ``replace``, so this
    test (which requires the OSError to come from ``replace``) would fail
    for it — distinguishing the atomic path from a direct write."""
    base_root = tmp_path
    rel = "knowledge/concepts/atomic.md"
    src = base_root / rel
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("---\ntitle: Atomic\n---\nbody\n", encoding="utf-8")

    original_replace = Path.replace

    def _fake_replace(self: Path, *args: object, **kwargs: object) -> Path:
        if str(self).startswith(str(base_root / "trash")):
            raise OSError("simulated replace failure")
        return original_replace(self, *args, **kwargs)  # type: ignore[no-any-return]

    monkeypatch.setattr(trash.Path, "replace", _fake_replace)

    with pytest.raises(OSError, match="replace failure"):
        move_to_trash(
            base_root=base_root, src_abs=src, rel_path=rel,
            reason="orphan_page", proposal_id="p-atomic",
        )

    # src untouched (unlink happens only after a successful replace), dest
    # never materialised, and the .tmp was cleaned up.
    assert src.is_file()
    assert not (base_root / "trash" / rel).exists()
    trash_dir = base_root / "trash" / "knowledge" / "concepts"
    if trash_dir.exists():
        assert list(trash_dir.iterdir()) == [], (
            "the .tmp must be cleaned up after a failed atomic replace"
        )


def test_move_to_trash_rolls_back_when_unlink_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``src_abs.unlink()`` fails after the trash copy is written, the
    function must roll back by deleting the new trash copy and re-raising
    — leaving the file in exactly one place (its original tree) so the
    next ``dikw client ingest`` re-creates the storage row from disk."""
    base_root = tmp_path
    rel = "knowledge/concepts/doomed.md"
    src = base_root / rel
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        "---\ntitle: Doomed\n---\nbody\n", encoding="utf-8",
    )

    original_unlink = Path.unlink

    def _fake_unlink(self: Path, *args: object, **kwargs: object) -> None:
        # Refuse to remove the source; allow rollback unlink of the
        # dest to proceed (dest is in trash/, src is in knowledge/).
        if self == src:
            raise OSError("simulated permission denied on src.unlink")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(trash.Path, "unlink", _fake_unlink)

    with pytest.raises(OSError, match="permission denied"):
        move_to_trash(
            base_root=base_root, src_abs=src, rel_path=rel,
            reason="orphan_page", proposal_id="p-rollback",
        )

    # Post-rollback: src survives, trash is empty for this file.
    assert src.is_file()
    trash_target = base_root / "trash" / rel
    assert not trash_target.exists()


@pytest.mark.parametrize(
    "rel",
    [
        "sources/notes/raw.md",
        "knowledge/concepts/page.md",
        "wisdom/elon-musk/never-sell.md",
    ],
)
def test_move_to_trash_preserves_layer_prefix(tmp_path: Path, rel: str) -> None:
    """The destination mirrors the input ``rel_path`` verbatim under
    ``trash/`` — so a D source lands at ``trash/sources/...``, a K page at
    ``trash/knowledge/...``, and a W page at ``trash/wisdom/...``. This is
    the cross-layer contract the ``delete`` verb relies on (no per-layer
    branching in the helper)."""
    base_root = tmp_path
    src = base_root / rel
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("---\ntitle: T\n---\nbody\n", encoding="utf-8")

    dest = move_to_trash(
        base_root=base_root, src_abs=src, rel_path=rel,
        reason="delete", proposal_id="",
    )

    assert dest == base_root / "trash" / rel
    assert dest.is_file()
    assert not src.exists()


def test_move_to_trash_omits_empty_proposal_id(tmp_path: Path) -> None:
    """A direct ``delete`` passes no proposal id; the audit block must
    carry ``at`` + ``reason`` but omit the empty ``proposal_id`` key so a
    manual delete reads as a clean ``trashed: {at, reason}``."""
    base_root = tmp_path
    rel = "wisdom/scratch.md"
    src = base_root / rel
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("---\ntitle: Scratch\n---\nbody\n", encoding="utf-8")

    dest = move_to_trash(
        base_root=base_root, src_abs=src, rel_path=rel,
        reason="delete",
    )

    trashed = frontmatter.loads(dest.read_text(encoding="utf-8")).metadata.get(
        "trashed"
    )
    assert isinstance(trashed, dict)
    assert trashed.get("reason") == "delete"
    assert isinstance(trashed.get("at"), str) and trashed["at"]
    assert "proposal_id" not in trashed
