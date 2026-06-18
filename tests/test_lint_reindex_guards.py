"""Unit coverage for the ``reindex_page`` safety guards (ADR-0005 PR3).

These guards are defense-in-depth: ``_preflight_proposal`` normally rejects a
malformed op before ``_apply_one_op`` runs, so the apply-side checks are only
reachable by calling ``_apply_one_op`` directly (a hand-crafted / persisted
proposal that bypassed propose). Pinning them here documents the contract and
keeps the safety net honest.
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import pytest
from rich.console import Console

from dikw_core import api
from dikw_core.client.progress import render_lint_apply_report
from dikw_core.domains.knowledge.lint_fix import (
    FixOperation,
    FixProposal,
    _apply_one_op,
    _preflight_proposal,
)
from dikw_core.schemas import Layer

from .fakes import init_test_base


@pytest.fixture()
def base_root(tmp_path: Path) -> Path:
    root = tmp_path / "base"
    init_test_base(root)
    return root


def _write(base: Path, rel: str, text: str) -> None:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


async def _apply(base_root: Path, op: FixOperation) -> dict | None:
    _cfg, _root, storage = await api._with_storage(base_root)
    try:
        return await _apply_one_op(
            op=op,
            storage=storage,
            base_root=base_root,
            proposal_id="p",
            issue_kind="untracked_file",
            path_to_doc_id={},
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_apply_reindex_rejects_non_kw_layer(base_root: Path) -> None:
    """A reindex_page carrying a non-K/W layer (e.g. a malformed SOURCE) is
    skipped before it can hit the wrong persist pipeline."""
    _write(base_root, "sources/x.md", "# X\n\nbody\n")
    op = FixOperation(kind="reindex_page", path="sources/x.md", layer=Layer.SOURCE)
    skip = await _apply(base_root, op)
    assert skip is not None and "knowledge/wisdom" in skip["reason"]


@pytest.mark.asyncio
async def test_apply_reindex_rejects_path_outside_layer_tree(base_root: Path) -> None:
    """A reindex_page whose path lives outside its declared layer's tree
    (here: layer=KNOWLEDGE but path under wisdom/) is skipped."""
    _write(base_root, "wisdom/holo/x.md", "# X\n\nbody\n")
    op = FixOperation(kind="reindex_page", path="wisdom/holo/x.md", layer=Layer.KNOWLEDGE)
    skip = await _apply(base_root, op)
    assert skip is not None and "tree" in skip["reason"]


@pytest.mark.asyncio
async def test_apply_reindex_skips_absent_file(base_root: Path) -> None:
    """A reindex_page whose file is gone is skipped (nothing to re-project)."""
    op = FixOperation(
        kind="reindex_page", path="knowledge/concepts/gone.md", layer=Layer.KNOWLEDGE
    )
    skip = await _apply(base_root, op)
    assert skip is not None and "absent" in skip["reason"].lower()


def test_preflight_reindex_rejects_non_kw_layer(base_root: Path) -> None:
    """``_preflight_proposal`` rejects a non-K/W reindex op up front."""
    _write(base_root, "sources/x.md", "# X\n\nbody\n")
    proposal = FixProposal(
        proposal_id=str(uuid.uuid4()),
        issue_kind="untracked_file",
        issue_path="sources/x.md",
        issue_detail="x",
        operations=[
            FixOperation(kind="reindex_page", path="sources/x.md", layer=Layer.SOURCE)
        ],
        rationale="x",
        source="heuristic",
    )
    reason = _preflight_proposal(
        proposal=proposal, base_root=base_root, already_touched=set()
    )
    assert reason is not None and "knowledge/wisdom" in reason


def test_render_lint_apply_report_shows_reindexed() -> None:
    """The CLI renderer surfaces ``reindexed_documents`` (the
    stale_index/untracked_file re-projection channel)."""
    buf = io.StringIO()
    console = Console(file=buf, width=200, no_color=True)
    render_lint_apply_report(
        console,
        {
            "applied": [{"kind": "reindex_page", "path": "knowledge/concepts/a.md"}],
            "skipped": [],
            "knowledge_paths_changed": [],
            "reindexed_documents": ["knowledge/concepts/a.md", "wisdom/holo/b.md"],
        },
    )
    out = buf.getvalue()
    assert "reindexed" in out
    assert "knowledge/concepts/a.md" in out
