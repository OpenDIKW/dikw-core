"""HTTP server for dikw-core (FastAPI + NDJSON).

Wraps the in-process engine (``dikw_core.api``) behind a FastAPI app, exposing
sync RPC endpoints, async task endpoints (NDJSON event streams), and a
multipart sources import endpoint. ``server/*`` may import the engine; the
reverse direction is forbidden so the engine remains transport-agnostic.

Phase 2 surface (this commit):
  * ``app.build_app`` — FastAPI app factory.
  * ``runtime`` — engine handle (cfg + storage + task subsystem) + lifespan.
  * ``auth`` — bearer token + localhost-default policy.
  * ``routes_sync`` — status / check / lint / wiki / doc / pages.
  * ``routes_tasks`` — submit (echo), list, get, result, events, cancel.
  * ``routes_import`` — multipart packages import → ``<base>/sources/``.
  * ``ndjson`` — replay + live tail + heartbeat helper.
  * ``errors`` — ApiError → JSON.
  * ``tasks/`` — TaskStore, TaskManager, ProgressBus, TaskBusReporter.
"""

from .app import build_app, build_app_from_disk

__all__ = ["build_app", "build_app_from_disk"]
