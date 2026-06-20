"""Slow end-to-end wrappers for the local-mode harness (``tools/e2e_verify.py``).

These subprocess the orchestrator rather than re-importing its lifecycle —
keeping the "pure subprocess orchestration" boundary and avoiding nested
event loops. Two shapes:

* hermetic — force keys absent; the structural floor must stand (rc 0) with
  every tier-2 leg loudly SKIPPED. Proves the no-key contract.
* keyed — run the full real-provider sweep; skipped unless live keys are set.
  Marked ``requires_embedding_key`` so conftest's autouse fixture doesn't strip
  ``DIKW_EMBEDDING_API_KEY`` before we read it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = _REPO_ROOT / "tools" / "e2e_verify.py"
_SECRET_ENV = ("ANTHROPIC_API_KEY", "DIKW_EMBEDDING_API_KEY")


def _run(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_HARNESS), *args],
        env=env, capture_output=True, text=True, timeout=1800, check=False)


@pytest.mark.slow
def test_e2e_local_hermetic_floor_stands_without_keys() -> None:
    """No keys -> structural floor green, tier-2 legs loudly SKIPPED, rc 0."""
    # Strip the gating secrets AND any ambient DIKW_E2E_* provider overrides, so
    # the floor under test is the committed template, not a dev's shell config.
    env = {k: v for k, v in os.environ.items()
           if k not in _SECRET_ENV and not k.startswith("DIKW_E2E_")}
    proc = _run(["--mode", "local", "--corpus", "all", "--env-file", os.devnull], env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RESULT: PASS" in proc.stdout
    assert "SKIPPED" in proc.stdout  # tier-2 legs skipped loudly
    assert "cli-coverage" in proc.stdout
    # coverage must be satisfied even hermetically (skips count as covered)
    cov = [ln for ln in proc.stdout.splitlines() if ln.startswith("cli-coverage")]
    assert cov and "PASS" in cov[0], proc.stdout


@pytest.mark.slow
@pytest.mark.requires_embedding_key
def test_e2e_local_full_with_real_providers() -> None:
    """Full real-provider sweep against M3 + Qwen; skipped without live keys."""
    missing = [k for k in _SECRET_ENV if not os.environ.get(k)]
    if missing:
        pytest.skip(f"tier-2: live keys absent ({'+'.join(missing)})")
    proc = _run(["--mode", "local", "--corpus", "all"], dict(os.environ))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RESULT: PASS" in proc.stdout
