"""Acceptance tests for the D/I/K/W -> domains/ migration.

These are invariant tests: they protect the new layout from regressing
back to the old top-level layout, and from accidental backwards-compat
shims being added later.
"""
from __future__ import annotations

import importlib

import pytest

_LAYERS = ("data", "info", "knowledge", "wisdom")


def test_new_domains_packages_importable() -> None:
    for sub in _LAYERS:
        importlib.import_module(f"dikw_core.domains.{sub}")


def test_old_top_level_layers_gone() -> None:
    for sub in _LAYERS:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"dikw_core.{sub}")


def test_api_facade_intact() -> None:
    from dikw_core.api import ingest, retrieve, synthesize  # noqa: F401


def test_removed_wisdom_api_symbols_stay_removed() -> None:
    api_mod = importlib.import_module("dikw_core.api")
    for removed in ("distill", "list_candidates", "approve_wisdom", "reject_wisdom"):
        assert not hasattr(api_mod, removed), (
            f"dikw_core.api.{removed} should be removed in 0.3.0 PR1 but is still exported"
        )
