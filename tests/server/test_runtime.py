"""Runtime helpers — focused unit tests.

The runtime is mostly exercised end-to-end by the route-level tests,
but a couple of small helpers carry enough logic that a focused
test makes regressions much easier to spot than waiting for an
HTTP-layer flake.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dikw_core.server import runtime as rt
from dikw_core.server.auth import AuthConfig


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


def test_base_scope_id_concurrent_first_create_converges(tmp_path: Path) -> None:
    """Two processes cold-starting the same base must agree on one id.

    The old read-then-write raced: concurrent first-runs both saw no file,
    both generated a UUID, and ended up holding *different* ids in memory
    (only one reached disk) — silently splitting the task-store scope. The
    exclusive ``open(path, "x")`` create makes every racer but one adopt
    the winner's persisted id.
    """
    n = 24
    for round_no in range(6):
        root = tmp_path / f"r{round_no}"
        root.mkdir()
        barrier = threading.Barrier(n)

        def run(_: int, root: Path = root, barrier: threading.Barrier = barrier) -> str:
            barrier.wait()
            return rt._base_scope_id(root)

        with ThreadPoolExecutor(max_workers=n) as ex:
            ids = list(ex.map(run, range(n)))

        assert len(set(ids)) == 1, f"round {round_no}: divergent ids {set(ids)}"
        on_disk = (root / ".dikw" / "base_id").read_text(encoding="utf-8").strip()
        assert ids[0] == on_disk


async def test_orphan_staging_cleaned_when_owning_sqlite_store(base_root: Path) -> None:
    """A per-base SQLite task store means this process owns the base
    exclusively, so startup wipes orphaned import staging (a crash leftover)."""
    staging = base_root / ".dikw" / "staging" / "orphan123"
    staging.mkdir(parents=True)
    (staging / "leftover.bin").write_text("x", encoding="utf-8")

    runtime = await rt.build_runtime(root=base_root, auth=AuthConfig(host="127.0.0.1", token=None))
    try:
        assert not staging.exists()
    finally:
        await rt.teardown_runtime(runtime)


async def test_orphan_staging_preserved_when_not_owning_store(
    base_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a shared task store and reap-on-start unset (the multi-replica
    case), startup must NOT wipe staging — another live replica of the same
    base may have an import in flight. Gated on the same predicate as the
    task restart-cleanup."""
    staging = base_root / ".dikw" / "staging" / "orphan123"
    staging.mkdir(parents=True)
    (staging / "leftover.bin").write_text("x", encoding="utf-8")

    # Make the SqliteTaskStore isinstance check fail so build_runtime takes
    # the shared-store branch, and ensure the opt-in reap flag is unset.
    monkeypatch.setattr(rt, "SqliteTaskStore", type("NotSqlite", (), {}))
    monkeypatch.delenv("DIKW_TASK_REAP_ON_START", raising=False)

    runtime = await rt.build_runtime(root=base_root, auth=AuthConfig(host="127.0.0.1", token=None))
    try:
        assert staging.exists()
        assert (staging / "leftover.bin").exists()
    finally:
        await rt.teardown_runtime(runtime)
