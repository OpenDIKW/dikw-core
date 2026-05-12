"""Retrieve NDJSON streaming.

``POST /v1/retrieve`` is the sole knowledge-access verb on dikw-core's
HTTP surface. PR-1 removed ``/v1/query``: in-engine LLM synthesis is no
longer dikw-core's job. Agents call retrieve to get ranked chunks +
page-level refs and assemble their own answer with their own LLM. Wire
shape:

  {"type":"retrieve_started","q":"...","limit":5}
  {"type":"retrieval_done","hits":[...]}      # hits carry full chunk ``text``
  {"type":"final","status":"succeeded","result":{...RetrieveResult...}}

Validation errors (empty ``q``, out-of-range ``limit``) return HTTP 4xx
*before* the stream starts — clients branching on status code stay
simple. Worker / runtime errors produce ``final{status:"failed"}`` with
an ``error`` body so a partial stream is never surfaced as success. The
hit list is intentionally repeated on ``final.result.chunks`` so a
non-streaming caller can pick the single ``final`` event and drop the
intermediate ``retrieval_done``. Cancellation: when the client
disconnects, the StreamingResponse generator's ``finally`` cancels the
worker task — the engine's ``CancelToken`` then short-circuits the next
retrieval step.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from pydantic import BaseModel

from .. import api
from .errors import ApiError
from .ndjson import (
    HEARTBEAT_INTERVAL,
    StreamReporterBase,
    isoformat_now,
    serialise_event,
    stream_response,
)
from .runtime import ServerRuntime, get_runtime

logger = logging.getLogger(__name__)


class RetrieveRequest(BaseModel):
    q: str
    limit: int = 5


class _RetrieveStreamReporter(StreamReporterBase):
    """Only ``retrieval_done`` is meaningful for retrieve (no LLM tokens
    to stream). Anything else falls through to the generic ``partial``
    envelope inherited from the base."""

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "retrieval_done":
            await self._queue.put(
                {
                    "type": "retrieval_done",
                    "ts": isoformat_now(),
                    "hits": payload.get("hits", []),
                }
            )
        else:
            await super().partial(kind, payload)


def make_router(*, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/v1", dependencies=[Depends(auth_dep)])

    @router.post("/retrieve")
    async def post_retrieve(
        request: Request,
        body: RetrieveRequest = Body(...),
    ) -> Any:
        rt: ServerRuntime = get_runtime(request.app)
        # Validate up front so a bad input fails the HTTP request rather
        # than landing as a ``final{failed}`` event mid-stream — clients
        # branching on status code stay simple.
        if not body.q.strip():
            raise ApiError(
                "query 'q' must not be empty",
                code="bad_request",
                status_code=400,
            )
        if body.limit < 1 or body.limit > 100:
            raise ApiError(
                f"limit must be in [1, 100], got {body.limit}",
                code="bad_request",
                status_code=400,
            )

        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        reporter = _RetrieveStreamReporter(queue)

        async def _run() -> None:
            try:
                result = await api.retrieve(
                    body.q, rt.root, limit=body.limit, reporter=reporter
                )
                await queue.put(
                    {
                        "type": "final",
                        "ts": isoformat_now(),
                        "status": "succeeded",
                        "result": result.model_dump(mode="json"),
                    }
                )
            except asyncio.CancelledError:
                await queue.put(
                    {
                        "type": "final",
                        "ts": isoformat_now(),
                        "status": "cancelled",
                    }
                )
                raise
            except ApiError as e:
                await queue.put(
                    {
                        "type": "final",
                        "ts": isoformat_now(),
                        "status": "failed",
                        "error": {
                            "code": e.code,
                            "message": e.message,
                            **({"detail": e.detail} if e.detail else {}),
                        },
                    }
                )
            except Exception as e:
                logger.exception("retrieve stream worker failed")
                await queue.put(
                    {
                        "type": "final",
                        "ts": isoformat_now(),
                        "status": "failed",
                        "error": {
                            "code": "engine_error",
                            "message": f"{type(e).__name__}: {e}",
                        },
                    }
                )
            finally:
                # Sentinel — tells the response generator to stop pulling.
                await queue.put(None)

        worker = asyncio.create_task(_run())

        async def _gen() -> AsyncIterator[bytes]:
            yield serialise_event(
                {
                    "type": "retrieve_started",
                    "ts": isoformat_now(),
                    "q": body.q,
                    "limit": body.limit,
                }
            )
            try:
                # Heartbeat-or-event loop: retrieve is typically fast
                # (sub-second) but a slow embedding endpoint or a large
                # multimodal asset table can stretch the wire silent
                # past the client's read timeout. Mirrors query's 15s
                # cadence so reverse proxies don't close the long-poll
                # either.
                while True:
                    try:
                        event = await asyncio.wait_for(
                            queue.get(), timeout=HEARTBEAT_INTERVAL
                        )
                    except TimeoutError:
                        yield serialise_event(
                            {"type": "heartbeat", "ts": isoformat_now()}
                        )
                        continue
                    if event is None:
                        break
                    yield serialise_event(event)
            finally:
                # Client disconnect: ask the engine to bail and wait.
                if not worker.done():
                    reporter.cancel_token().cancel()
                    worker.cancel()
                # Surface worker failures by awaiting; suppress both
                # CancelledError (disconnect path) and any worker
                # exception — worker errors already landed as
                # ``final{failed}`` in the stream, so re-raising here
                # would replace the streamed body with a 500.
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await worker

        return stream_response(_gen())

    return router


__all__ = ["make_router"]
