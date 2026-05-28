"""Tests for the 0.3.x → 0.4.0 base-upgrade gate.

``BaseUpgradeRequired`` is the hard-break safety net for the
``wiki/`` → ``knowledge/`` rename: any base whose K-layer still lives
under ``wiki/`` must be migrated by hand before the engine will open
it. These tests pin the gate's behavior — including the partial-
migration bypass guard — so a future refactor cannot quietly defang
the protection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api


def test_fresh_base_loads_without_upgrade_error(tmp_path: Path) -> None:
    """A base created by ``init_base`` (0.4.0 layout) must load cleanly."""
    api.init_base(tmp_path, description="fresh")
    _cfg, root = api.load_base(tmp_path)
    assert root == tmp_path.resolve()
    assert (tmp_path / "knowledge").is_dir()


def test_legacy_wiki_with_content_raises(tmp_path: Path) -> None:
    """A 0.3.x layout (``wiki/`` with markdown, no ``knowledge/``)
    must be refused with the exact migration command."""
    api.init_base(tmp_path)
    # Simulate the pre-0.4.0 on-disk shape: rename knowledge/ back to
    # wiki/ so the gate sees the legacy directory carrying markdown.
    (tmp_path / "knowledge").rename(tmp_path / "wiki")
    with pytest.raises(api.BaseUpgradeRequired) as exc_info:
        api.load_base(tmp_path)
    msg = str(exc_info.value)
    # Pin the migration command — this is the user-facing upgrade
    # contract; silent regressions here strand every existing user.
    assert "mv wiki knowledge" in msg
    assert "rm -rf .dikw" in msg
    assert "dikw serve --base" in msg
    assert "dikw client ingest" in msg


def test_partial_migration_bypass_is_caught(tmp_path: Path) -> None:
    """The gate must fire even when ``knowledge/`` already exists.

    If the user creates ``knowledge/`` (manually, via tooling, or by
    re-running ``dikw init`` against a wiped ``dikw.yml``) BEFORE
    moving the legacy ``wiki/*.md`` content over, a naive
    ``wiki and not knowledge`` check would silently pass and the
    user's K-layer pages would be orphaned. Tightening the gate to
    detect any ``wiki/`` carrying markdown closes that bypass.
    """
    api.init_base(tmp_path)
    # Re-create a populated wiki/ alongside the fresh knowledge/.
    legacy = tmp_path / "wiki"
    (legacy / "concepts").mkdir(parents=True)
    (legacy / "concepts" / "stranded.md").write_text(
        "# Stranded\n\nleftover from 0.3.x\n", encoding="utf-8"
    )
    with pytest.raises(api.BaseUpgradeRequired) as exc_info:
        api.load_base(tmp_path)
    assert "wiki" in str(exc_info.value).lower()


def test_empty_legacy_wiki_dir_is_tolerated(tmp_path: Path) -> None:
    """A bare empty ``wiki/`` directory left behind from a half-done
    rename is harmless and must not block startup. The gate fires
    only when there is markdown content the user would lose."""
    api.init_base(tmp_path)
    (tmp_path / "wiki").mkdir()
    _cfg, root = api.load_base(tmp_path)
    assert root == tmp_path.resolve()


def test_migration_path_completes_end_to_end(tmp_path: Path) -> None:
    """The recipe in the ``BaseUpgradeRequired`` message must actually
    work end-to-end on a legacy base — moving ``wiki/`` to
    ``knowledge/`` and wiping ``.dikw/`` lets ``load_base`` succeed."""
    import shutil

    api.init_base(tmp_path)
    (tmp_path / "knowledge").rename(tmp_path / "wiki")
    # Initial guard raises on the legacy layout.
    with pytest.raises(api.BaseUpgradeRequired):
        api.load_base(tmp_path)
    # Apply the documented migration command.
    (tmp_path / "wiki").rename(tmp_path / "knowledge")
    shutil.rmtree(tmp_path / ".dikw", ignore_errors=True)
    # load_base should now succeed (dikw.yml is preserved).
    _cfg, root = api.load_base(tmp_path)
    assert root == tmp_path.resolve()
