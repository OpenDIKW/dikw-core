"""Runtime helpers — focused unit tests.

The runtime is mostly exercised end-to-end by the route-level tests,
but a couple of small helpers carry enough logic that a focused
test makes regressions much easier to spot than waiting for an
HTTP-layer flake.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core.server import runtime as rt


def test_base_scope_id_is_persisted_under_dikw_dir(tmp_path: Path) -> None:
    """First call generates + writes ``<root>/.dikw/base_id``; second
    call (and any other process mounting the same volume) reads the
    same value back. Without persistence, replicas mounting the base
    at different paths would compute different scope IDs and the
    cross-replica task APIs would silently break."""
    a = rt._base_scope_id(tmp_path)
    assert a, "base id must not be empty"
    assert (tmp_path / ".dikw" / "base_id").read_text(encoding="utf-8").strip() == a
    # Second call returns the same value.
    assert rt._base_scope_id(tmp_path) == a


def test_base_scope_id_env_override_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operators can pin the scope ID via env (e.g. when intentionally
    pooling tasks across bases or when the base has no writable
    ``.dikw/`` for some reason)."""
    monkeypatch.setenv("DIKW_BASE_INSTANCE_ID", "pinned-id")
    assert rt._base_scope_id(tmp_path) == "pinned-id"
    # Env override does NOT touch the on-disk file.
    assert not (tmp_path / ".dikw" / "base_id").exists()


def test_base_scope_id_stable_across_path_aliasing(tmp_path: Path) -> None:
    """The base id is stored on the volume — two ``Path`` objects
    pointing at the same physical base must produce the same id, even
    if the input paths differ syntactically."""
    a = rt._base_scope_id(tmp_path)
    # Re-entry with a path that resolves to the same dir.
    aliased = tmp_path / "."
    assert rt._base_scope_id(aliased) == a
