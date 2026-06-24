"""F4 — atomic on-disk writes.

These tests pin the crash-safety contract added in F4: a failed write must
leave the *existing* file untouched (no half-written / truncated bytes at the
visible path) and must not strand a temp file, and the trash collision loop
must claim a unique name atomically rather than via an exists()-probe.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dikw_core.domains._atomic import atomic_write_text, reserve_path
from dikw_core.domains.knowledge.page import build_page, write_page
from dikw_core.domains.wisdom.page import write_wisdom_file


def test_atomic_write_text_writes(tmp_path: Path) -> None:
    p = tmp_path / "a.md"
    atomic_write_text(p, "hello\n")
    assert p.read_text(encoding="utf-8") == "hello\n"


def test_atomic_write_text_failure_preserves_old_and_cleans_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "a.md"
    atomic_write_text(p, "OLD\n")

    def boom(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(p, "NEW\n")

    # The visible path still holds the old bytes — never a partial "NEW".
    assert p.read_text(encoding="utf-8") == "OLD\n"
    # No stranded temp file beside it.
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_text_preserves_existing_mode(tmp_path: Path) -> None:
    import stat

    p = tmp_path / "p.md"
    atomic_write_text(p, "first\n")
    os.chmod(p, 0o600)
    # Overwrite: os.replace swaps inodes, so without copying the mode the
    # rewritten page would silently inherit the umask (commonly 0o644).
    atomic_write_text(p, "second\n")
    assert p.read_text(encoding="utf-8") == "second\n"
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600


def test_reserve_path_claims_name_exactly_once(tmp_path: Path) -> None:
    p = tmp_path / "claim"
    assert reserve_path(p) is True
    assert p.exists()
    # A second caller racing for the same name loses — this is the property
    # an exists()-probe cannot guarantee under concurrency.
    assert reserve_path(p) is False


def test_write_page_failure_preserves_old(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page_a = build_page(title="T", body="OLD body", category="concept")
    write_page(tmp_path, page_a)
    abs_path = tmp_path / page_a.path
    assert "OLD body" in abs_path.read_text(encoding="utf-8")

    def boom(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    page_b = build_page(title="T", body="NEW body", category="concept", path=page_a.path)
    with pytest.raises(OSError):
        write_page(tmp_path, page_b)

    text = abs_path.read_text(encoding="utf-8")
    assert "OLD body" in text
    assert "NEW body" not in text
    # No stranded *.tmp beside the page.
    assert not list(abs_path.parent.glob("*.tmp"))


def test_write_wisdom_file_failure_preserves_old(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rel = "wisdom/alice/lesson.md"
    write_wisdom_file(tmp_path, logical_path=rel, title="L", body="OLD body")
    abs_path = tmp_path / rel
    assert "OLD body" in abs_path.read_text(encoding="utf-8")

    def boom(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_wisdom_file(tmp_path, logical_path=rel, title="L", body="NEW body")

    text = abs_path.read_text(encoding="utf-8")
    assert "OLD body" in text
    assert "NEW body" not in text
    assert not list(abs_path.parent.glob("*.tmp"))
