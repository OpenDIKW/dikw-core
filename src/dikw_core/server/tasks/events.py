"""NDJSON event schema for async task streams.

Every event the server pushes through ``GET /v1/tasks/{id}/events`` is one
of these models, serialised as a single JSON line. The ``type`` field is
the discriminator so a Pydantic-typed client can deserialise without
guesswork. ``seq`` is the monotonic anchor used for resume-by-seq when a
client reconnects mid-stream.

Sequence ordering invariants (enforced by ``TaskStore.append_event``):
  * ``task_started`` is always seq=1.
  * ``progress`` / ``log`` / ``partial`` events fire in real time and
    increment seq.
  * ``final`` is always the last event for a task (status =
    succeeded / failed / cancelled). ``append_event`` upholds this by
    dropping any NON-``final`` event that arrives once the row is terminal
    (an obsolete late write from a cancelled runner whose ``to_thread``
    emit outlived its await). In the rare case a cancel lands *during* the
    ``task_started`` emit, that detached write is dropped too, so the tape
    can collapse to ``[final]`` only — ``final``-last is preserved at the
    cost of the seq=1 ``task_started`` (a consumer keys on ``final``, so
    this is benign).
  * ``heartbeat`` is injected by the streamer, NOT persisted, so it has
    seq=0 and is recognisable as ephemeral.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class _BaseEvent(BaseModel):
    """Common fields for every event in the NDJSON tape.

    ``model_config = {"extra": "forbid"}`` keeps wire stability tight —
    a typo in a producer surfaces immediately rather than silently
    appearing on the network.
    """

    seq: int = Field(ge=0)
    ts: str  # ISO8601 UTC

    model_config = {"extra": "forbid"}


class TaskStartedEvent(_BaseEvent):
    type: Literal["task_started"] = "task_started"
    task_id: str
    op: str


class ProgressEvent(_BaseEvent):
    type: Literal["progress"] = "progress"
    phase: str
    current: int = 0
    total: int = 0
    detail: dict[str, Any] | None = None


class LogEvent(_BaseEvent):
    type: Literal["log"] = "log"
    level: str  # "INFO" | "WARN" | "ERROR" — kept as str so producers can
    # emit lowercase / numeric levels without coupling to a closed enum.
    message: str


class PartialEvent(_BaseEvent):
    type: Literal["partial"] = "partial"
    kind: str
    payload: dict[str, Any]


class HeartbeatEvent(_BaseEvent):
    """Injected by the NDJSON streamer to keep proxies and middleware from
    closing the long-poll. Not persisted in the TaskStore — ``seq=0`` is
    the marker for ephemeral events."""

    type: Literal["heartbeat"] = "heartbeat"


class FinalEvent(_BaseEvent):
    type: Literal["final"] = "final"
    status: Literal["succeeded", "failed", "cancelled"]
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class ErrorEvent(_BaseEvent):
    """Out-of-band error inside an open stream — *not* a final state. Used
    when the server cannot continue producing the requested events but
    the underlying task may still be alive (e.g. store I/O failure on
    replay)."""

    type: Literal["error"] = "error"
    code: str
    message: str


__all__ = [
    "ErrorEvent",
    "FinalEvent",
    "HeartbeatEvent",
    "LogEvent",
    "PartialEvent",
    "ProgressEvent",
    "TaskStartedEvent",
]
