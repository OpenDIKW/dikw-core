"""Schema version fingerprint shared by SQL adapters.

Pre-alpha policy is "rebuild on incompatibility": each adapter ships a
single ``schema.sql`` representing the desired shape, and ``migrate()``
either applies it to a fresh DB or refuses to touch a DB whose stored
fingerprint doesn't match the code's expected ``SCHEMA_VERSION``. There
is no in-place upgrade path; the user is told to rebuild.

Bump ``SCHEMA_VERSION`` whenever a breaking schema change lands. The
number is opaque — only equality matters.

The key the fingerprint is stored under (``SCHEMA_VERSION_KEY``) is
**deliberately distinct from the legacy ``schema_version`` key written
by the deleted per-migration-file counter**. A pre-fingerprint install
still has a ``meta_kv['schema_version']`` row carrying its old counter
value (1, 2, or 3), but the new code never reads that key — it sees
``None`` under ``schema_fingerprint`` instead and routes to the
fresh-DB branch in ``migrate()``, which then loud-fails on the
documents-table check (``_REBUILD_HINT`` per adapter). No risk of a
collision where an old counter value compares equal to the new
fingerprint.
"""

from __future__ import annotations

SCHEMA_VERSION = 3
SCHEMA_VERSION_KEY = "schema_fingerprint"


def mismatch_message(stored: int, hint: str) -> str:
    """``StorageError`` body for a schema_version mismatch.

    ``stored`` is shown so a downstream log reader can tell what the DB
    is at; the expected version is taken from ``SCHEMA_VERSION``. The
    ``hint`` is per-adapter — ``rm -rf .dikw/`` is wrong advice for a
    Postgres install where the stale state lives in a DB schema, not a
    file. See each adapter's ``_REBUILD_HINT`` constant.
    """
    return (
        f"schema fingerprint mismatch: DB at v{stored}, code expects "
        f"v{SCHEMA_VERSION}. {hint}"
    )


__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_KEY",
    "mismatch_message",
]
