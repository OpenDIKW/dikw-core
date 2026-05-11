"""Async task endpoints: submit, query, cancel, follow.

Phase 2 ships the *plumbing* — the only registered op is ``echo``, a
five-event mock that lets tests exercise the full submit → events →
final → result loop end-to-end. Phase 3 wires real ops (ingest, synth,
distill, eval) on top of these same routes.

URL shape:
  POST   /v1/{op}                 → submit, returns {task_id, op, links}
  GET    /v1/tasks                → list with status / op filters
  GET    /v1/tasks/{task_id}      → row snapshot
  GET    /v1/tasks/{task_id}/result    → terminal result (404 until terminal)
  GET    /v1/tasks/{task_id}/events    → NDJSON stream (?from_seq=N to resume)
  POST   /v1/tasks/{task_id}/cancel    → request cancellation
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..domains.knowledge.lint import LintKind
from .errors import BadRequest, NotFoundError
from .ingest_op import make_ingest_runner
from .lint_op import make_lint_apply_runner, make_lint_propose_runner
from .ndjson import ndjson_lines, stream_response
from .runtime import ServerRuntime, get_runtime
from .synth_op import make_distill_runner, make_eval_runner, make_synth_runner
from .tasks import (
    TaskRow,
    TaskRunner,
    TaskStatus,
)

# ---- op runners ---------------------------------------------------------


async def _echo_runner(reporter: Any, *, count: int = 5) -> dict[str, Any]:
    """Mock op for Phase 2 end-to-end testing.

    Emits ``count`` progress events with a simulated phase plus a single
    partial, then returns a result dict the manager wraps in the final
    event. Honours cancellation between iterations.
    """
    for i in range(1, count + 1):
        reporter.cancel_token().raise_if_cancelled()
        await reporter.progress(
            phase="echo",
            current=i,
            total=count,
            detail={"step": i},
        )
        # Sleep long enough that a test can race a cancel against an
        # in-flight task without flakiness, but short enough that the
        # happy-path test finishes quickly. The scale comes from the
        # caller via params.
        await asyncio.sleep(0.0)
    await reporter.partial("echo_done", {"count": count})
    return {"echoed": count}


# ---- request bodies -----------------------------------------------------


class EchoSubmit(BaseModel):
    """Phase 2 mock submit body. Real ops will define their own."""

    count: int = 5


class IngestSubmit(BaseModel):
    """Body for ``POST /v1/ingest``.

    ``no_embed`` skips the dense embedding pass; useful for FTS-only
    wikis or when the embedding provider isn't reachable.

    Sources are imported separately via ``POST /v1/import``
    (which commits straight into ``<base>/sources/``); ingest then
    walks that tree, hashing for idempotency and chunking + embedding.
    """

    model_config = {"extra": "forbid"}

    no_embed: bool = False


class SynthSubmit(BaseModel):
    """Body for ``POST /v1/synth``.

    ``force_all`` re-synthesises every source doc even if the wiki log
    already records a ``synth`` entry — needed when the prompt or model
    has changed and prior pages should be regenerated.

    ``no_embed`` skips the K-layer embed pass; pages still land on disk
    so dense retrieval recovers on the next ingest run.
    """

    force_all: bool = False
    no_embed: bool = False


class DistillSubmit(BaseModel):
    """Body for ``POST /v1/distill``.

    ``pages_per_call`` caps how many K-layer pages are fed to a single
    LLM call; the default mirrors ``api.distill`` and keeps prompt
    tokens bounded for vendors with tight context limits.
    """

    pages_per_call: int = 8


class LintProposeSubmit(BaseModel):
    """Body for ``POST /v1/lint/propose``.

    ``rule`` filters the lint scan to one ``LintKind`` (default: every
    kind dispatched). ``limit`` caps how many issues the orchestrator
    consumes — the propose task runs serially over those, so ``limit``
    is also effectively a wall-clock cap on the LLM-heavy fixers.

    ``enable_llm`` opts into LLM-fallback paths (broken_wikilink stub
    pages, the non_atomic_page splitter). Default False keeps propose
    runs heuristic-only and free of token spend; users opt in
    explicitly via ``--enable-llm`` on the CLI.
    """

    rule: LintKind | None = None
    limit: int = Field(default=10, ge=1, le=200)
    enable_llm: bool = False


class LintApplySubmit(BaseModel):
    """Body for ``POST /v1/lint/apply``.

    ``proposal_task_id`` references a ``lint.propose`` task that
    terminated SUCCEEDED. ``pick`` / ``skip`` index into
    ``proposal_report.proposals`` so callers can choose to apply a
    subset (review then partial-accept). Both may be set; pick is
    applied first, then skip removes from that subset.
    """

    proposal_task_id: str
    pick: list[int] | None = None
    skip: list[int] | None = None


class EvalSubmit(BaseModel):
    """Body for ``POST /v1/eval``.

    ``dataset`` accepts a registered dataset name (resolved under the
    packaged datasets root) or a directory path — same surface as
    ``dikw eval``. Omit it to run every packaged dataset back-to-back
    (preserves the in-process ``dikw eval`` no-arg workflow).
    ``mode`` and ``cache_mode`` mirror ``run_eval`` knobs.
    """

    dataset: str | None = None
    mode: str = "hybrid"
    cache_mode: str = "read_write"


class TaskHandle(BaseModel):
    task_id: str
    op: str
    status: TaskStatus
    created_at: str
    links: dict[str, str]


def _handle(row: TaskRow) -> TaskHandle:
    base = f"/v1/tasks/{row.task_id}"
    return TaskHandle(
        task_id=row.task_id,
        op=row.op,
        status=row.status,
        created_at=row.created_at,
        links={
            "self": base,
            "events": f"{base}/events",
            "result": f"{base}/result",
            "cancel": f"{base}/cancel",
        },
    )


# ---- router -------------------------------------------------------------


class TaskResultBody(BaseModel):
    task_id: str
    status: TaskStatus
    started_at: str | None
    finished_at: str | None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class CancelResponse(BaseModel):
    task_id: str
    cancelled: bool
    already_terminal: bool


def make_router(*, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/v1", dependencies=[Depends(auth_dep)])

    # ---- submit echo (Phase 2 only) -----------------------------------

    @router.post("/echo", response_model=TaskHandle)
    async def submit_echo(
        request: Request,
        body: EchoSubmit = Body(default_factory=EchoSubmit),
    ) -> TaskHandle:
        rt: ServerRuntime = get_runtime(request.app)
        count = max(1, int(body.count))

        async def _runner(reporter: Any) -> dict[str, Any]:
            return await _echo_runner(reporter, count=count)

        runner: TaskRunner = _runner
        row = await rt.manager.submit(
            op="echo", runner=runner, params={"count": count}
        )
        return _handle(row)

    @router.post("/ingest", response_model=TaskHandle)
    async def submit_ingest(
        request: Request,
        body: IngestSubmit = Body(default_factory=IngestSubmit),
    ) -> TaskHandle:
        rt: ServerRuntime = get_runtime(request.app)
        runner: TaskRunner = make_ingest_runner(
            wiki_root=rt.root,
            no_embed=body.no_embed,
            lock=rt.ingest_lock,
        )
        row = await rt.manager.submit(
            op="ingest",
            runner=runner,
            params={"no_embed": body.no_embed},
        )
        return _handle(row)

    @router.post("/synth", response_model=TaskHandle)
    async def submit_synth(
        request: Request,
        body: SynthSubmit = Body(default_factory=SynthSubmit),
    ) -> TaskHandle:
        rt: ServerRuntime = get_runtime(request.app)
        runner: TaskRunner = make_synth_runner(
            wiki_root=rt.root,
            force_all=body.force_all,
            no_embed=body.no_embed,
        )
        row = await rt.manager.submit(
            op="synth",
            runner=runner,
            params={
                "force_all": body.force_all,
                "no_embed": body.no_embed,
            },
        )
        return _handle(row)

    @router.post("/distill", response_model=TaskHandle)
    async def submit_distill(
        request: Request,
        body: DistillSubmit = Body(default_factory=DistillSubmit),
    ) -> TaskHandle:
        rt: ServerRuntime = get_runtime(request.app)
        if body.pages_per_call < 1:
            raise BadRequest(
                f"pages_per_call must be >= 1, got {body.pages_per_call}"
            )
        runner: TaskRunner = make_distill_runner(
            wiki_root=rt.root,
            pages_per_call=body.pages_per_call,
        )
        row = await rt.manager.submit(
            op="distill",
            runner=runner,
            params={"pages_per_call": body.pages_per_call},
        )
        return _handle(row)

    @router.post("/lint/propose", response_model=TaskHandle)
    async def submit_lint_propose(
        request: Request,
        body: LintProposeSubmit = Body(default_factory=LintProposeSubmit),
    ) -> TaskHandle:
        rt: ServerRuntime = get_runtime(request.app)
        runner: TaskRunner = make_lint_propose_runner(
            wiki_root=rt.root,
            rule=body.rule,
            limit=body.limit,
            enable_llm=body.enable_llm,
        )
        row = await rt.manager.submit(
            op="lint.propose",
            runner=runner,
            params={
                "rule": body.rule,
                "limit": body.limit,
                "enable_llm": body.enable_llm,
            },
        )
        return _handle(row)

    @router.post("/lint/apply", response_model=TaskHandle)
    async def submit_lint_apply(
        request: Request,
        body: LintApplySubmit = Body(...),
    ) -> TaskHandle:
        rt: ServerRuntime = get_runtime(request.app)
        runner: TaskRunner = make_lint_apply_runner(
            wiki_root=rt.root,
            proposal_task_id=body.proposal_task_id,
            task_store=rt.task_store,
            pick=body.pick,
            skip=body.skip,
        )
        row = await rt.manager.submit(
            op="lint.apply",
            runner=runner,
            params={
                "proposal_task_id": body.proposal_task_id,
                "pick": body.pick,
                "skip": body.skip,
            },
        )
        return _handle(row)

    @router.post("/eval", response_model=TaskHandle)
    async def submit_eval(
        request: Request,
        body: EvalSubmit = Body(...),
    ) -> TaskHandle:
        rt: ServerRuntime = get_runtime(request.app)
        if body.mode not in ("hybrid", "bm25", "vector", "all"):
            raise BadRequest(
                f"mode must be one of hybrid|bm25|vector|all, got {body.mode!r}"
            )
        if body.cache_mode not in ("read_write", "rebuild", "off"):
            raise BadRequest(
                f"cache_mode must be one of read_write|rebuild|off, got {body.cache_mode!r}"
            )
        runner: TaskRunner = make_eval_runner(
            wiki_root=rt.root,
            dataset=body.dataset,
            mode=body.mode,
            cache_mode=body.cache_mode,
        )
        row = await rt.manager.submit(
            op="eval",
            runner=runner,
            params={
                "dataset": body.dataset,
                "mode": body.mode,
                "cache_mode": body.cache_mode,
            },
        )
        return _handle(row)

    # ---- task lifecycle endpoints -------------------------------------

    @router.get("/tasks", response_model=list[TaskRow])
    async def list_tasks(
        request: Request,
        status: TaskStatus | None = Query(default=None),
        op: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[TaskRow]:
        rt: ServerRuntime = get_runtime(request.app)
        return await rt.task_store.list_tasks(
            status=status, op=op, limit=limit
        )

    @router.get("/tasks/{task_id}", response_model=TaskRow)
    async def get_task(request: Request, task_id: str) -> TaskRow:
        rt: ServerRuntime = get_runtime(request.app)
        row = await rt.task_store.get(task_id)
        if row is None:
            raise NotFoundError(f"task_id {task_id!r} not found")
        return row

    @router.get(
        "/tasks/{task_id}/result", response_model=TaskResultBody
    )
    async def get_task_result(
        request: Request, task_id: str
    ) -> TaskResultBody:
        rt: ServerRuntime = get_runtime(request.app)
        row = await rt.task_store.get(task_id)
        if row is None:
            raise NotFoundError(f"task_id {task_id!r} not found")
        if row.status not in (
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ):
            raise BadRequest(
                f"task {task_id!r} is still {row.status.value}; "
                "subscribe to /events to wait for the final state",
                code="task_not_terminal",
                detail={"current_status": row.status.value},
            )
        return TaskResultBody(
            task_id=row.task_id,
            status=row.status,
            started_at=row.started_at,
            finished_at=row.finished_at,
            result=row.result,
            error=row.error,
        )

    @router.get("/tasks/{task_id}/events")
    async def follow_task(
        request: Request,
        task_id: str,
        from_seq: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        rt: ServerRuntime = get_runtime(request.app)
        row = await rt.task_store.get(task_id)
        if row is None:
            raise NotFoundError(f"task_id {task_id!r} not found")

        stream = ndjson_lines(
            task_id=task_id,
            store=rt.task_store,
            bus=rt.bus,
            from_seq=from_seq,
        )
        return stream_response(stream)

    @router.post(
        "/tasks/{task_id}/cancel", response_model=CancelResponse
    )
    async def cancel_task(
        request: Request, task_id: str
    ) -> CancelResponse:
        rt: ServerRuntime = get_runtime(request.app)
        row = await rt.task_store.get(task_id)
        if row is None:
            raise NotFoundError(f"task_id {task_id!r} not found")
        was_running = await rt.manager.cancel(task_id)
        terminal = row.status in (
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )
        return CancelResponse(
            task_id=task_id,
            cancelled=was_running,
            already_terminal=terminal,
        )

    return router


__all__ = ["make_router"]
