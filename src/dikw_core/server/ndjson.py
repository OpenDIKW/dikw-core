"""NDJSON streaming helper for task event endpoints.

The server emits one JSON object per line (``application/x-ndjson``) so
clients can incrementally parse without HTTP/2 server-sent-events
ceremony. This module provides:

  * ``ndjson_lines`` — joins a historical replay (from ``TaskStore``)
    with a live tail (from ``ProgressBus``) and injects a heartbeat
    every ``HEARTBEAT_INTERVAL`` seconds so reverse proxies don't close
    the long-poll.
  * ``stream_response`` — wraps the iterator in a ``StreamingResponse``
    with the right Content-Type and a no-buffering hint header.

Heartbeats are NOT persisted in the store; they carry ``seq=0`` and
clients drop them on parse.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse

from ..progress import CancelToken
from ._time import isoformat_utc_ms
from .tasks import TERMINAL_STATUSES, ProgressBus, TaskStore

# Tuned to undercut typical proxy idle timeouts (60s on most CDNs / nginx
# defaults) by 4x; bumping it doesn't hurt the engine but does increase
# the time a client takes to notice a fully stalled task.
HEARTBEAT_INTERVAL = 15.0
# How often to re-read the store while tailing a task whose live events
# never landed on this worker's bus (multi-worker / multi-replica
# deployments). Sub-second feels live enough; longer would feel laggy.
_STORE_POLL_INTERVAL = 0.5


def isoformat_now() -> str:
    """Backwards-compatible alias for :func:`isoformat_utc_ms` — kept so
    existing imports of ``isoformat_now`` keep working."""
    return isoformat_utc_ms()


async def ndjson_lines(
    *,
    task_id: str,
    store: TaskStore,
    bus: ProgressBus,
    from_seq: int = 0,
    heartbeat_interval: float = HEARTBEAT_INTERVAL,
) -> AsyncIterator[bytes]:
    """Yield bytes for one full NDJSON stream.

    Strategy:
      1. Replay every event with ``seq >= from_seq`` from the store.
      2. If the bus is already closed for this task (terminal state
         persisted), stop — the replay above contains the full tape.
      3. Otherwise subscribe to the bus AND poll the store on a short
         interval. The bus is the fast path for in-process tasks; the
         store-poll is the cross-worker safety net — in a multi-worker
         or multi-replica deployment the follower may land on a worker
         that never saw the task in memory, so its bus has nothing to
         hand back. Polling lets that follower still tail the task to
         terminal even though every event arrives via the persistent
         tape, not the in-process bus.
      4. Inject a heartbeat every ``heartbeat_interval`` seconds while
         the stream is otherwise quiet.
    """
    seen_seq = max(0, from_seq - 1)

    # Step 1 — historical replay.
    historical = await store.list_events(task_id, from_seq=from_seq)
    for event in historical:
        yield serialise_event(event)
        seen_seq = max(seen_seq, int(event.get("seq", 0)))

    # Step 2 — short-circuit if the task is already terminal.
    if bus.is_closed(task_id):
        return

    # Step 3 — subscribe + catch-up gap-fill.
    sub = await bus.subscribe(task_id)
    catchup = await store.list_events(task_id, from_seq=seen_seq + 1)
    for event in catchup:
        yield serialise_event(event)
        seen_seq = max(seen_seq, int(event.get("seq", 0)))

    # Step 4 — bus tail + store-poll fallback + heartbeats.
    sub_iter = sub.__aiter__()
    next_event_task: asyncio.Task[dict[str, Any]] | None = (
        asyncio.ensure_future(sub_iter.__anext__())
    )
    poll_interval = min(_STORE_POLL_INTERVAL, heartbeat_interval)
    last_heartbeat = time.monotonic()

    async def _drain_store() -> bool:
        """Yield any new events from the store. Returns True if the row
        is now terminal (caller should exit after one more drain)."""
        nonlocal seen_seq
        polled = await store.list_events(task_id, from_seq=seen_seq + 1)
        for e in polled:
            yield_buf.append(serialise_event(e))
            seen_seq = max(seen_seq, int(e.get("seq", 0)))
        row = await store.get(task_id)
        return row is not None and row.status in TERMINAL_STATUSES

    yield_buf: list[bytes] = []
    try:
        while True:
            if next_event_task is not None:
                try:
                    done, _pending = await asyncio.wait(
                        {next_event_task}, timeout=poll_interval
                    )
                except asyncio.CancelledError:
                    next_event_task.cancel()
                    raise
                if done:
                    try:
                        event = next_event_task.result()
                    except StopAsyncIteration:
                        # Bus closed for this worker — switch to pure
                        # polling. The task may still be running on
                        # another worker, so we can't return yet.
                        next_event_task = None
                        continue
                    seq = int(event.get("seq", 0))
                    if seq > seen_seq:
                        yield serialise_event(event)
                        seen_seq = seq
                    next_event_task = asyncio.ensure_future(
                        sub_iter.__anext__()
                    )
                    continue
            else:
                # Pure-polling mode (cross-worker follower) — pace the
                # loop so we don't burn CPU on store reads.
                await asyncio.sleep(poll_interval)

            # Timeout / poll path — drain new events from the store
            # (covers the cross-worker case) plus heartbeat.
            terminal = await _drain_store()
            for buf in yield_buf:
                yield buf
            yield_buf.clear()

            now = time.monotonic()
            if not terminal and now - last_heartbeat >= heartbeat_interval:
                yield serialise_event(
                    {"type": "heartbeat", "seq": 0, "ts": isoformat_now()}
                )
                last_heartbeat = now

            if terminal:
                # One more drain to be safe (a final event can land
                # between drain + status check).
                await _drain_store()
                for buf in yield_buf:
                    yield buf
                yield_buf.clear()
                return
    finally:
        if next_event_task is not None and not next_event_task.done():
            next_event_task.cancel()


def stream_response(stream: AsyncIterator[bytes]) -> StreamingResponse:
    """Wrap a byte stream as an NDJSON ``StreamingResponse``."""
    return StreamingResponse(
        stream,
        media_type="application/x-ndjson",
        # Hint to nginx + similar proxies that this stream must not be
        # buffered. Document this in docs/server.md so deployers know
        # to pair with `proxy_buffering off`.
        headers={"X-Accel-Buffering": "no"},
    )


def serialise_event(event: dict[str, Any]) -> bytes:
    """Encode a single event as one NDJSON line (trailing ``\\n``)."""
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


class StreamReporterBase:
    """Common ``ProgressReporter`` plumbing for short-lived streaming routes.

    Subclasses override ``partial`` to translate engine-specific
    ``partial(kind, payload)`` calls into the route's wire-event vocabulary;
    progress / log / cancel handling is shared because every streaming
    route handles them identically.
    """

    def __init__(self, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        self._queue = queue
        self._token = CancelToken()

    async def progress(
        self,
        *,
        phase: str,
        current: int = 0,
        total: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "type": "progress",
            "ts": isoformat_now(),
            "phase": phase,
            "current": current,
            "total": total,
        }
        if detail is not None:
            event["detail"] = detail
        await self._queue.put(event)

    async def log(self, level: str, message: str) -> None:
        await self._queue.put(
            {
                "type": "log",
                "ts": isoformat_now(),
                "level": level,
                "message": message,
            }
        )

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        # Default fallthrough — generic envelope so future engine events
        # surface without route plumbing. Subclasses override for
        # op-specific wire shapes.
        await self._queue.put(
            {
                "type": "partial",
                "ts": isoformat_now(),
                "kind": kind,
                "payload": payload,
            }
        )

    def cancel_token(self) -> CancelToken:
        return self._token


__all__ = [
    "HEARTBEAT_INTERVAL",
    "StreamReporterBase",
    "isoformat_now",
    "ndjson_lines",
    "serialise_event",
    "stream_response",
]
