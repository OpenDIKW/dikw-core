"""Hash helpers shared across D layer + server commit + client import.

``hash_file`` lives in :mod:`dikw_core.md_inspect` so the client (which
can't import from ``domains/``) and the engine share one implementation.
This module re-exports it under the historical name so existing callers
stay intact.
"""

from __future__ import annotations

import hashlib

from ...md_inspect import sha256_file as hash_file

__all__ = ["hash_bytes", "hash_file"]


def hash_bytes(data: bytes) -> str:
    """SHA-256 hex digest of ``data`` already in memory."""
    return hashlib.sha256(data).hexdigest()
