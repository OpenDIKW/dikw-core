"""Slow end-to-end wrapper for the docker-mode harness (``tools/e2e_verify.py``).

Builds the dikw-core image from the local working tree, brings up server +
pgvector Postgres, runs the full ``dikw client`` verb sequence against the
containerized server, then tears it all down. Skipped unless both a live
Docker daemon and live provider keys are present (so CI without either stays
green via a loud skip, not a false pass).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = _REPO_ROOT / "tools" / "e2e_verify.py"
_SECRET_ENV = ("ANTHROPIC_API_KEY", "DIKW_EMBEDDING_API_KEY")


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    r = subprocess.run(["docker", "info"], capture_output=True, check=False, timeout=30)
    return r.returncode == 0


@pytest.mark.slow
@pytest.mark.requires_embedding_key
def test_e2e_docker_full_with_real_providers() -> None:
    if not _docker_ready():
        pytest.skip("docker daemon not reachable (loud skip)")
    missing = [k for k in _SECRET_ENV if not os.environ.get(k)]
    if missing:
        pytest.skip(f"tier-2: live keys absent ({'+'.join(missing)})")
    proc = subprocess.run(
        [sys.executable, str(_HARNESS), "--mode", "docker", "--corpus", "all"],
        env=dict(os.environ), capture_output=True, text=True, timeout=2400, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RESULT: PASS" in proc.stdout
