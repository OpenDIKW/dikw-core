"""Async task endpoints: submit, query, cancel, page events.

The cursor JSON ``/events`` endpoint serves both UI history paging
(``wait=0``) and agent follow loops (``wait>0`` long-poll). It replaces
the prior NDJSON streaming tail. ``GET /v1/tasks`` itself is a
cursor-paged summary view since 0.2.0 — full ``result``/``error``
payloads live on ``GET /v1/tasks/{id}`` and ``/result``.
See ``docs/server.md``.

URL shape:
  POST   /v1/{op}                 → submit, returns {task_id, op, links}
  GET    /v1/tasks                → TaskListPage {tasks, next_cursor, has_more}
  GET    /v1/tasks/{task_id}      → row snapshot (full TaskRow incl. result)
  GET    /v1/tasks/{task_id}/result    → terminal result (404 until terminal)
  GET    /v1/tasks/{task_id}/events    → cursor JSON {events, next_from_seq, has_more, last_seq, task_status}
  POST   /v1/tasks/{task_id}/cancel    → request cancellation
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
from typing import Any, Literal

from fastapi import APIRouter, Body, Depends, Query, Request
from pydantic import BaseModel, Field

from ..domains.knowledge.lint import LintKind
from ..schemas import WisdomWriteSubmit
from .errors import BadRequest, NotFoundError
from .ingest_op import make_ingest_runner
from .lint_op import make_lint_apply_runner, make_lint_propose_runner
from .runtime import ServerRuntime, get_runtime
from .synth_op import _resolve_judge_sample, make_eval_runner, make_synth_runner
from .tasks import (
    TERMINAL_STATUSES,
    TaskRow,
    TaskRunner,
    TaskStatus,
)
from .wisdom_op import make_wisdom_write_runner

# ---- op runners ---------------------------------------------------------


async def _echo_runner(
    reporter: Any, *, count: int = 5, delay_ms: int = 0
) -> dict[str, Any]:
    """Mock op for end-to-end testing.

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
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)
        else:
            # Yield so a test can race a cancel against an in-flight
            # task without flakiness, but short enough that the
            # happy-path test finishes quickly.
            await asyncio.sleep(0.0)
    await reporter.partial("echo_done", {"count": count})
    return {"echoed": count}


# ---- request bodies -----------------------------------------------------


class EchoSubmit(BaseModel):
    """Phase 2 mock submit body. Real ops will define their own.

    ``delay_ms`` is a test affordance — paces the runner's progress
    emits so long-poll tests can race the handler's wake-up path
    against actual event production. Default 0 leaves existing
    behaviour unchanged.
    """

    count: int = 5
    delay_ms: int = Field(default=0, ge=0, le=10000)


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

    ``verify`` runs the post-synth self-check (scoped lint + persist +
    semantic duplicate) over this run's pages and returns a
    ``SynthVerifyReport`` under the result's ``verify`` key.

    ``judge`` (only meaningful with ``verify``) adds the optional report-only
    grounding/entailment leg — the LLM scores whether this run's claims are
    supported by their cited sources. It never changes ``passed``.
    """

    force_all: bool = False
    no_embed: bool = False
    verify: bool = False
    judge: bool = False


class LintProposeSubmit(BaseModel):
    """Body for ``POST /v1/lint/propose``.

    ``rule`` filters the lint scan to one ``LintKind`` (default: every
    kind dispatched). ``limit`` caps how many issues the orchestrator
    consumes — the propose task runs serially over those, so ``limit``
    is also effectively a wall-clock cap on the LLM-heavy fixers.

    ``enable_llm`` opts into LLM-powered fixer paths: broken_wikilink's
    evidence-backed grounded repair (writes a real page when D-layer
    hybrid-search hits provide enough evidence), the non_atomic_page
    splitter, and orphan_page's merge_into_existing_page strategy.
    Default False keeps propose runs heuristic-only and free of token
    spend; users opt in explicitly via ``--enable-llm`` on the CLI.
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
    ``dikw client eval``. Omit it to run every packaged dataset back-to-back
    (preserves the in-process ``dikw client eval`` no-arg workflow).
    ``mode`` and ``cache_mode`` mirror ``run_eval`` knobs.

    ``eval_modes`` selects which eval families to run for each dataset:
    ``"retrieval"`` (current behaviour, default), ``"synth"`` (K-layer
    quality gate), or both. ``None`` falls back to the dataset's own
    ``modes:`` declaration. ``judge`` + ``judge_sample`` only apply when
    synth mode is selected — they add the LLM-judge soft layer.
    """

    dataset: str | None = None
    mode: str = "hybrid"
    cache_mode: str = "read_write"
    eval_modes: list[str] | None = None
    judge: bool = False
    judge_sample: int | Literal["auto"] | None = None


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


class TaskRowSummary(BaseModel):
    """Summary projection of ``TaskRow`` served by ``GET /v1/tasks``.

    Deliberately omits ``result`` and ``error`` — a synth/eval row's
    full payload can be tens of KB, and the list view exists to *find*
    tasks, not to read their bodies. Detail goes through
    ``GET /v1/tasks/{id}/result`` or ``GET /v1/tasks/{id}``.
    """

    task_id: str
    op: str
    status: TaskStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    params_digest: str = ""


class TaskListPage(BaseModel):
    """Cursor-paged response from ``GET /v1/tasks``.

    The ``next_cursor`` value is an opaque base64url string — the
    server's encoding is an internal detail and may evolve; clients
    must only treat it as opaque and replay it verbatim on the next
    request. When ``has_more`` is ``False``, ``next_cursor`` is
    ``None``.
    """

    tasks: list[TaskRowSummary]
    next_cursor: str | None = None
    has_more: bool = False


def _encode_cursor(*, created_at: str, task_id: str) -> str:
    """Encode (created_at, task_id) into a base64url cursor token.

    The payload is a tiny JSON object — base64url avoids `+/=` characters
    that need URL escaping when the cursor flows through `?cursor=` query
    parameters. Padding is stripped on encode and re-added on decode."""
    raw = json.dumps(
        {"created_at": created_at, "task_id": task_id},
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _decode_cursor(token: str) -> tuple[str, str]:
    """Decode a ``_encode_cursor`` token back into (created_at, task_id).

    Raises ``BadRequest("invalid_cursor")`` for any decode/shape error so
    agents see a stable failure mode (their bug) instead of a 500.
    Validation here is strict because the only legitimate way to obtain
    a cursor is to read it from a prior response — any tampering or
    forgery is by definition a client-side defect.
    """
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (binascii.Error, ValueError) as e:
        # ``ValueError`` is the base class of both ``UnicodeEncodeError``
        # (non-ASCII token hits ``encode("ascii")``) and
        # ``UnicodeDecodeError`` (non-UTF-8 payload hits ``decode``), so it
        # alone catches every malformed-cursor shape.
        raise BadRequest(
            "invalid cursor",
            code="invalid_cursor",
            detail={"reason": "decode_failed"},
        ) from e
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("created_at"), str)
        or not isinstance(payload.get("task_id"), str)
    ):
        raise BadRequest(
            "invalid cursor",
            code="invalid_cursor",
            detail={"reason": "payload_shape"},
        )
    return payload["created_at"], payload["task_id"]


class EventsPage(BaseModel):
    """Cursor-paged response from ``GET /v1/tasks/{id}/events``.

    Serves both UI history browsing (``wait=0``, snapshot) and agent
    follow loops (``wait>0``, long-poll). See ``docs/server.md`` for
    the full contract."""

    task_id: str
    task_status: TaskStatus
    events: list[dict[str, Any]]
    next_from_seq: int
    has_more: bool
    last_seq: int


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
        delay_ms = int(body.delay_ms)

        async def _runner(reporter: Any) -> dict[str, Any]:
            return await _echo_runner(reporter, count=count, delay_ms=delay_ms)

        runner: TaskRunner = _runner
        row = await rt.manager.submit(
            op="echo",
            runner=runner,
            params={"count": count, "delay_ms": delay_ms},
        )
        return _handle(row)

    @router.post("/ingest", response_model=TaskHandle)
    async def submit_ingest(
        request: Request,
        body: IngestSubmit = Body(default_factory=IngestSubmit),
    ) -> TaskHandle:
        rt: ServerRuntime = get_runtime(request.app)
        runner: TaskRunner = make_ingest_runner(
            base_root=rt.root,
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
            base_root=rt.root,
            force_all=body.force_all,
            no_embed=body.no_embed,
            verify=body.verify,
            judge=body.judge,
        )
        row = await rt.manager.submit(
            op="synth",
            runner=runner,
            params={
                "force_all": body.force_all,
                "no_embed": body.no_embed,
                "verify": body.verify,
                "judge": body.judge,
            },
        )
        return _handle(row)

    @router.post("/lint/propose", response_model=TaskHandle)
    async def submit_lint_propose(
        request: Request,
        body: LintProposeSubmit = Body(default_factory=LintProposeSubmit),
    ) -> TaskHandle:
        rt: ServerRuntime = get_runtime(request.app)
        runner: TaskRunner = make_lint_propose_runner(
            base_root=rt.root,
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
            base_root=rt.root,
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

    @router.post("/base/wisdom", response_model=TaskHandle)
    async def submit_wisdom_write(
        request: Request,
        body: WisdomWriteSubmit = Body(...),
    ) -> TaskHandle:
        rt: ServerRuntime = get_runtime(request.app)
        # Share ``ingest_lock`` with the ingest path: a concurrent
        # ``dikw ingest`` and a single-page wisdom write on the same
        # base would otherwise race on storage rows and on the
        # cross-layer title index snapshot.
        runner: TaskRunner = make_wisdom_write_runner(
            base_root=rt.root, submit=body, lock=rt.ingest_lock
        )
        # Keep ``params`` lightweight — the full body (title + body +
        # tags + sources + extras) can be tens of KB and bloats the task
        # row + params_digest. Mirror lint.apply's choice to pin only
        # identity-bearing knobs.
        row = await rt.manager.submit(
            op="wisdom.write",
            runner=runner,
            params={
                "slug": body.slug,
                "author": body.author,
                "no_embed": body.no_embed,
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
        if body.eval_modes is not None:
            unknown = [m for m in body.eval_modes if m not in ("retrieval", "synth")]
            if unknown:
                raise BadRequest(
                    f"eval_modes entries must be retrieval|synth, got {unknown!r}"
                )
            if not body.eval_modes:
                raise BadRequest("eval_modes must be non-empty when provided")
        # Resolve ``"auto"`` to the calibrated sample size engine-side (the
        # client forwards the sentinel; the recommended n is engine knowledge).
        judge_sample = _resolve_judge_sample(body.judge_sample)
        runner: TaskRunner = make_eval_runner(
            base_root=rt.root,
            dataset=body.dataset,
            mode=body.mode,
            cache_mode=body.cache_mode,
            eval_modes=body.eval_modes,
            judge=body.judge,
            judge_sample=judge_sample,
        )
        row = await rt.manager.submit(
            op="eval",
            runner=runner,
            params={
                "dataset": body.dataset,
                "mode": body.mode,
                "cache_mode": body.cache_mode,
                "eval_modes": body.eval_modes,
                "judge": body.judge,
                "judge_sample": judge_sample,
            },
        )
        return _handle(row)

    # ---- task lifecycle endpoints -------------------------------------

    @router.get("/tasks", response_model=TaskListPage)
    async def list_tasks(
        request: Request,
        status: TaskStatus | None = Query(default=None),
        op: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
        cursor: str | None = Query(default=None),
    ) -> TaskListPage:
        """Cursor-paged summary listing.

        Page boundary detection uses the "fetch limit+1" trick: ask the
        store for one extra row, and if it comes back we know there's at
        least one more page after this one — strip the sentinel before
        returning and stamp its ``(created_at, task_id)`` into
        ``next_cursor`` so the caller can resume from exactly where this
        page ended.
        """
        rt: ServerRuntime = get_runtime(request.app)
        after_created_at: str | None = None
        after_task_id: str | None = None
        if cursor is not None:
            after_created_at, after_task_id = _decode_cursor(cursor)
        # Fetch one extra row to detect ``has_more`` without a second
        # round-trip / count query.
        rows = await rt.task_store.list_tasks(
            status=status,
            op=op,
            limit=limit + 1,
            after_created_at=after_created_at,
            after_task_id=after_task_id,
        )
        has_more = len(rows) > limit
        page_rows = rows[:limit]
        next_cursor: str | None = None
        if has_more and page_rows:
            tail = page_rows[-1]
            next_cursor = _encode_cursor(
                created_at=tail.created_at, task_id=tail.task_id
            )
        summaries = [
            TaskRowSummary.model_validate(r, from_attributes=True)
            for r in page_rows
        ]
        return TaskListPage(
            tasks=summaries, next_cursor=next_cursor, has_more=has_more
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
        if row.status not in TERMINAL_STATUSES:
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

    @router.get("/tasks/{task_id}/events", response_model=EventsPage)
    async def get_events(
        request: Request,
        task_id: str,
        from_seq: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1000),
        wait: int = Query(default=0, ge=0, le=60),
    ) -> EventsPage:
        """Cursor-paged event read.

        ``wait=0`` is a pure snapshot for UI paging or stateless agent
        polling. ``wait>0`` is a short long-poll — the handler parks on
        the task's ``asyncio.Condition`` until a new event lands or the
        timeout fires. Multi-worker safety net: an inner 0.5s ceiling
        on each Condition wait means handlers on a worker that never
        sees the in-process notify still pick up changes by re-reading
        the store ≤ 0.5s later.
        """
        rt: ServerRuntime = get_runtime(request.app)
        row = await rt.task_store.get(task_id)
        if row is None:
            raise NotFoundError(f"task_id {task_id!r} not found")

        events = await rt.task_store.list_events(
            task_id, from_seq=from_seq, limit=limit
        )

        # Long-poll only when the task is still live AND no backlog —
        # skip the wait loop entirely on already-terminal tasks so a
        # client polling a finished task doesn't burn 500ms before
        # checking row status.
        if not events and wait > 0 and row.status not in TERMINAL_STATUSES:
            cond = rt.manager.condition_for(task_id)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + wait
            while loop.time() < deadline:
                async with cond:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(
                            cond.wait(),
                            timeout=min(0.5, deadline - loop.time()),
                        )
                events = await rt.task_store.list_events(
                    task_id, from_seq=from_seq, limit=limit
                )
                if events:
                    break
                # Re-read row to catch a quiet terminal transition that
                # may have happened without an emit firing on this worker.
                row = await rt.task_store.get(task_id)
                if row is None:
                    raise NotFoundError(f"task_id {task_id!r} not found")
                if row.status in TERMINAL_STATUSES:
                    break

        # Re-read once more for an up-to-date status — it may have
        # transitioned during the wait above.
        latest = row if row.status in TERMINAL_STATUSES else await rt.task_store.get(task_id)
        if latest is None:
            raise NotFoundError(f"task_id {task_id!r} not found")

        last_seq = await rt.task_store.max_seq(task_id)
        next_from_seq = (
            int(events[-1]["seq"]) + 1 if events else from_seq
        )
        # ``has_more`` is "there's at least one persisted event past the
        # cursor". An empty tape (``last_seq == 0``) trivially has none —
        # the naive ``next_from_seq <= last_seq`` check would return
        # ``True`` for the common ``from_seq=0, last_seq=0`` polling
        # cold-start and force clients into an infinite empty-page loop.
        has_more = last_seq > 0 and next_from_seq <= last_seq

        return EventsPage(
            task_id=task_id,
            task_status=latest.status,
            events=events,
            next_from_seq=next_from_seq,
            has_more=has_more,
            last_seq=last_seq,
        )

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
        terminal = row.status in TERMINAL_STATUSES
        return CancelResponse(
            task_id=task_id,
            cancelled=was_running,
            already_terminal=terminal,
        )

    return router


__all__ = ["make_router"]
