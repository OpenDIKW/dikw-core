"""Coverage for ``evals/tools/*`` recovery scripts.

The Phase 1.5 replay tool's active-version pinning silently drifts if
``api.query()`` ever changes its embedding-space contract; this test
is the canary.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import pytest

from dikw_core.schemas import EmbeddingVersion
from dikw_core.storage.base import NotSupported, Storage

# evals/tools is a CLI dir, not a package — add to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO_ROOT / "evals" / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from run_phase15_from_snapshot import resolve_active_text_version  # noqa: E402

from .fakes import init_test_base, register_text_version  # noqa: E402


@pytest.mark.asyncio
async def test_resolve_active_version_uses_active_row_dim(tmp_path: Path) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki, dim=128)

    from dikw_core.config import load_config
    from dikw_core.storage import build_storage

    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        version_id = await register_text_version(
            storage, dim=128, model="qwen3-test", provider="gitee"
        )
        result = await resolve_active_text_version(
            storage, default_model="cfg-default-model"
        )
    finally:
        await storage.close()

    assert result == (version_id, "qwen3-test", 128)


@pytest.mark.asyncio
async def test_resolve_active_version_falls_back_to_cfg_when_none(
    tmp_path: Path,
) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    from dikw_core.config import load_config
    from dikw_core.storage import build_storage

    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        result = await resolve_active_text_version(
            storage, default_model="fallback-model"
        )
    finally:
        await storage.close()

    assert result == (None, "fallback-model", None)


class _NotSupportedStorage:
    """Stand-in for an adapter that raises NotSupported from versioning."""

    async def get_active_embed_version(
        self, *, modality: Literal["text", "multimodal"]
    ) -> EmbeddingVersion | None:
        raise NotSupported(f"embed versions not supported (modality={modality})")


@pytest.mark.asyncio
async def test_resolve_active_version_handles_notsupported() -> None:
    storage: Storage = _NotSupportedStorage()  # type: ignore[assignment]

    result = await resolve_active_text_version(
        storage, default_model="fallback-model"
    )

    assert result == (None, "fallback-model", None)
