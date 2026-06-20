"""Trip-wire: the filesystem storage backend was removed.

Mirrors tests/test_no_html_backend.py. If these fail, someone is
trying to revive the filesystem backend — read docs/design.md first.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from dikw_core.config import CONFIG_FILENAME, load_config


def test_filesystem_module_is_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("dikw_core.storage.filesystem")


def test_filesystem_config_class_is_gone() -> None:
    import dikw_core.config as cfg_mod

    assert not hasattr(cfg_mod, "FilesystemStorageConfig")


def test_filesystem_storage_class_is_gone() -> None:
    import dikw_core.storage as storage_mod

    assert not hasattr(storage_mod, "FilesystemStorage")


def test_load_config_rejects_filesystem_backend(tmp_path: Path) -> None:
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  llm_api_key_env: ANTHROPIC_API_KEY
  embedding_dim: 1536
  embedding_revision: ''
  embedding_normalize: true
  embedding_distance: cosine
  embedding_api_key_env: OPENAI_API_KEY
storage:
  backend: filesystem
  root: .dikw/fs
sources: []
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(path)
