"""``dikw client *`` Typer app.

Every command resolves a :class:`ClientConfig`, opens a single
:class:`Transport`, calls the matching HTTP endpoint, and renders the
response. Long ops (ingest / synth / distill / eval) submit a task,
follow its NDJSON event stream, and dispatch to the op-specific final
renderer; sync ops just decode the JSON body and render directly.

Each command body sits inside an ``async def`` and is driven by
``asyncio.run`` from the Typer wrapper — this keeps the transport pool
lifecycle bounded to a single command invocation. We deliberately don't
share a transport across commands: a CLI run is short, the
``AsyncClient`` constructor is cheap, and per-command lifetime makes
cancel-on-Ctrl-C work without extra plumbing.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

import typer
from rich.console import Console
from rich.table import Table

from ..schemas import Layer
from . import serve_and_run as _sar
from .config import ClientConfig, resolve
from .converters import Converter, Registry, discover, pick
from .importer import SourceImportError, build_import
from .progress import (
    RetrieveStreamRenderer,
    TaskProgressRenderer,
    render_check_report,
    render_distill_report,
    render_eval_report,
    render_health_report,
    render_import_report,
    render_ingest_errors,
    render_ingest_report,
    render_lint_report,
    render_retrieve_table,
    render_status,
    render_synth_eval_report,
    render_synth_report,
)
from .serve_and_run import ENV_SERVE_AND_RUN_AUTO_WAIT
from .task_follow import follow_to_terminal
from .transport import ClientError, Transport

# Exit-code contract for ``--wait`` on op commands and ``dikw client tasks
# wait``. Agents script against these — keep them stable.
_EXIT_FAILED = 1
_EXIT_CANCELLED = 130  # POSIX SIGINT convention
_EXIT_TIMEOUT = 124  # POSIX timeout(1) convention


_DRAIN_PAGE_GUARD = 200


class DrainPageGuardError(Exception):
    """Raised when ``_drain_task_list`` hits ``_DRAIN_PAGE_GUARD`` with the
    server still reporting ``has_more=true``.

    Silently returning a partial list would make ``--all`` /
    ``lint proposals`` look successful while losing rows past the
    ceiling — agents (and cross-reference logic like apply→propose)
    would then act on incomplete data. Fail loud instead; surface the
    last cursor so the user can resume manually if their dataset is
    genuinely that large.
    """

    def __init__(self, *, pages: int, rows_collected: int, last_cursor: str | None) -> None:
        super().__init__(
            f"drained {pages} pages ({rows_collected} rows) but server still "
            f"reports has_more=true; refusing to silently truncate. Last "
            f"cursor: {last_cursor!r}"
        )
        self.pages = pages
        self.rows_collected = rows_collected
        self.last_cursor = last_cursor


async def _drain_task_list(
    t: Transport,
    *,
    op: str | None = None,
    status: str | None = None,
    page_size: int = 200,
) -> list[dict[str, Any]]:
    """Walk ``GET /v1/tasks`` to completion, returning every matching row.

    The 0.2.0 list endpoint returns a ``TaskListPage`` envelope with a
    cursor — callers that want the *full* matching set (rather than
    just the first page) need to follow ``next_cursor`` until
    ``has_more`` flips to ``False``. This helper centralises that walk
    so individual commands don't reinvent it.

    Used by ``lint proposals`` (where the propose/apply listings must
    be complete for the cross-reference to be correct) and any future
    aggregator-style commands. The hard ceiling of ``_DRAIN_PAGE_GUARD``
    pages guards against a server-side cursor bug looping forever; if
    we exhaust it while the server still flags ``has_more=true``, we
    raise ``DrainPageGuardError`` rather than handing back a partial
    list that looks complete.
    """
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(_DRAIN_PAGE_GUARD):
        params: dict[str, Any] = {"limit": page_size}
        if op is not None:
            params["op"] = op
        if status is not None:
            params["status"] = status
        if cursor is not None:
            params["cursor"] = cursor
        body = await t.get_json("/v1/tasks", params=params)
        if not isinstance(body, dict):
            return rows
        tasks = body.get("tasks")
        if isinstance(tasks, list):
            rows.extend(r for r in tasks if isinstance(r, dict))
        if not body.get("has_more"):
            return rows
        next_cursor = body.get("next_cursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            return rows
        cursor = next_cursor
    raise DrainPageGuardError(
        pages=_DRAIN_PAGE_GUARD, rows_collected=len(rows), last_cursor=cursor
    )


async def _gather_task_results(
    t: Transport, rows: list[dict[str, Any]], *, concurrency: int = 8
) -> list[Any]:
    """Fan out ``GET /v1/tasks/{id}/result`` for every row carrying a
    ``task_id``, bounded to ``concurrency`` in-flight requests.

    The bound matters: ``Transport`` shares one ``httpx.AsyncClient``
    whose pool keeps 20 keepalive connections behind a 5s acquisition
    timeout. An unbounded gather over a base with hundreds of terminal
    tasks would queue past the pool cap and raise ``PoolTimeout``
    instead of merely running slowly.
    """
    ids = [
        str(r["task_id"]) for r in rows
        if isinstance(r, dict) and r.get("task_id")
    ]
    sem = asyncio.Semaphore(concurrency)

    async def _one(tid: str) -> Any:
        async with sem:
            return await t.get_json(f"/v1/tasks/{tid}/result")

    return list(await asyncio.gather(*(_one(tid) for tid in ids)))


def _serve_and_run_forces_wait() -> bool:
    """True when this CLI invocation is the inner command of a
    ``dikw client serve-and-run`` lifecycle without ``--keep-alive``.

    The outer ``serve-and-run`` tears down the temporary server the
    moment the inner exits, so an async-default op command would
    submit a task, print a handle, exit 0, and have the task killed
    mid-flight. Auto-flipping to ``--wait`` keeps the lifecycle
    coherent — the inner blocks until the task finishes, the server
    only shuts down after."""
    return os.environ.get(ENV_SERVE_AND_RUN_AUTO_WAIT) == "1"

app = typer.Typer(
    name="client",
    help="Talk to a running ``dikw serve`` instance.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

# ---- shared options ----------------------------------------------------


def _server_option() -> Any:
    # Inside ``Annotated[…, typer.Option(...)]`` Typer expects the
    # *param decls* only — the default value is supplied by the
    # parameter's ``= None`` assignment. Passing ``None`` as the first
    # argument here is the legacy non-Annotated form and trips a
    # confusing ``isidentifier`` AttributeError deep in click.
    return typer.Option(
        "--server",
        help="Server URL. Default: env $DIKW_SERVER_URL or http://127.0.0.1:8765.",
    )


def _token_option() -> Any:
    return typer.Option(
        "--token",
        help="Bearer token. Default: env $DIKW_SERVER_TOKEN or client.toml.",
    )


def _resolve(server: str | None, token: str | None) -> ClientConfig:
    return resolve(server_url=server, token=token)


def _converter_resolver(
    cli_choice: str | None, cfg: ClientConfig
) -> Callable[[str], Converter]:
    """Build a lazy converter resolver — ``discover()`` runs only on
    first invocation and is memoised for subsequent calls (so a future
    directory-import flow that dispatches one file at a time doesn't
    re-instantiate every installed plugin per file). Md-only imports
    never trigger discover() at all because the resolver itself is
    never called."""

    registry: Registry | None = None

    def _resolve_one(ext: str) -> Converter:
        nonlocal registry
        if registry is None:
            registry = discover()
        return pick(
            ext, registry, converter=cli_choice, config=cfg.converters
        )

    return _resolve_one


def _validate_format(fmt: str) -> None:
    """Reject a ``--format`` value outside the json/table contract.

    Every command that ships ``--format`` shares the same json-or-table
    contract — keep the validation in one place so the error message
    and exit code stay consistent."""
    if fmt not in ("json", "table"):
        console.print(
            f"[red]error[/red]: --format must be 'json' or 'table', got {fmt!r}"
        )
        raise typer.Exit(code=2)


def _on_error(err: ClientError) -> None:
    """Translate a transport-layer error into a terse stderr line + exit.

    ``cancelled`` is the one expected non-zero status — surface it as a
    yellow notice rather than red so users don't think their cancel
    command failed.
    """
    if err.status == 0:
        console.print(f"[red]network error:[/red] {err.message}")
    else:
        console.print(
            f"[red]error[/red] [{err.status} {err.code}]: {err.message}"
        )
    if err.detail:
        console.print(f"[dim]detail: {err.detail}[/dim]")


def _run(coro: Any) -> Any:
    """Run an async command with a uniform error → exit-code mapping.

    ``SourceImportError`` exits 2 (Unix convention for user-supplied
    bad input — the pre-flight inspection caught a problem the user
    needs to fix locally). Server-side failures and transport errors
    exit 1 (operation failed at the remote end).
    """
    try:
        return asyncio.run(coro)
    except ClientError as e:
        _on_error(e)
        raise typer.Exit(code=1) from e
    except SourceImportError as e:
        console.print(f"[red]import error:[/red] {e}")
        raise typer.Exit(code=2) from e
    except DrainPageGuardError as e:
        # The drain helper refused to silently truncate a paginated
        # listing. Render the resume hint so the user can either retry
        # with a more selective filter (``--op`` / ``--status``) or
        # walk the cursor manually starting from ``last_cursor``.
        console.print(f"[red]listing exhausted page guard:[/red] {e}")
        raise typer.Exit(code=1) from e


# ---- meta + sync commands ---------------------------------------------


@app.command("info")
def info_cmd(
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Print the server's ``GET /v1/info`` response as JSON.

    Uses ``console.print_json`` so long values (paths, URLs) don't get
    rich's soft-wrap injected mid-string — agent parsers need clean
    JSON regardless of terminal width.
    """

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            payload = await t.get_json("/v1/info")
        console.print_json(json.dumps(payload, ensure_ascii=False))

    _run(_go())


@app.command("status")
def status_cmd(
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: 'json' (default, agent-friendly) or 'table' (human).",
        ),
    ] = "json",
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Show storage-backend counts."""
    _validate_format(fmt)

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            counts = await t.get_json("/v1/status")
        if fmt == "json":
            console.print_json(json.dumps(counts, ensure_ascii=False))
        else:
            render_status(console, counts)

    _run(_go())


@app.command(
    "health",
    epilog=(
        "Examples:\n\n"
        "  dikw client health\n\n"
        "  dikw client health --format table"
    ),
)
def health_cmd(
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: 'json' (default, agent-friendly) or 'table' (human).",
        ),
    ] = "json",
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Probe the server's self-description (base_root, layer counts, providers).

    Designed as the first call an AI agent makes after attaching to a
    running ``dikw serve`` — confirms the server is up, shows which base
    it points at, and exposes the resolved provider config (model /
    base_url / dim / ``api_key_present``) without leaking secrets.
    """
    _validate_format(fmt)

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            report = await t.get_json("/v1/health")
        if fmt == "json":
            console.print_json(json.dumps(report, ensure_ascii=False))
        else:
            render_health_report(console, report)

    _run(_go())


@app.command(
    "check",
    epilog=(
        "Examples:\n\n"
        "  dikw client check\n\n"
        "  dikw client check --format table\n\n"
        "  dikw client check --llm-only"
    ),
)
def check_cmd(
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: 'json' (default, agent-friendly) or 'table' (human).",
        ),
    ] = "json",
    llm_only: Annotated[
        bool, typer.Option("--llm-only", help="Probe only the LLM leg.")
    ] = False,
    embed_only: Annotated[
        bool,
        typer.Option("--embed-only", help="Probe only the embedding leg."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Verify configured providers via the server."""
    _validate_format(fmt)
    if llm_only and embed_only:
        console.print(
            "[red]error:[/red] --llm-only and --embed-only are mutually exclusive"
        )
        raise typer.Exit(code=2)

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            report = await t.post_json(
                "/v1/check",
                json_body={"llm_only": llm_only, "embed_only": embed_only},
            )
        if fmt == "json":
            console.print_json(json.dumps(report, ensure_ascii=False))
        else:
            render_check_report(console, report)
        # ``CheckReport.ok`` is a ``@property`` that pydantic drops on
        # serialization; recompute here from the per-leg probe results
        # so the exit code matches the engine's intent.
        legs = [
            report.get("llm"),
            report.get("embed"),
        ]
        present = [leg for leg in legs if isinstance(leg, dict)]
        if not present or not all(bool(leg.get("ok")) for leg in present):
            raise typer.Exit(code=1)

    _run(_go())


lint_app = typer.Typer(
    name="lint",
    help=(
        "K-layer hygiene checker. Default action runs the scan; "
        "`propose` / `apply` produce + execute structured fix proposals."
    ),
    no_args_is_help=False,
    invoke_without_command=True,
)
app.add_typer(lint_app, name="lint")


@lint_app.callback(invoke_without_command=True)
def lint_root(
    ctx: typer.Context,
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: 'table' (default, human) or 'json' (agent-friendly).",
        ),
    ] = "table",
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Run lint against the server's base (default subaction)."""
    if ctx.invoked_subcommand is not None:
        return
    _validate_format(fmt)

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            report = await t.post_json("/v1/lint")
        if fmt == "json":
            console.print_json(json.dumps(report, ensure_ascii=False))
        else:
            render_lint_report(console, report)
        # ``LintReport`` is a dataclass with ``ok`` defined as a
        # ``@property``; pydantic's response serializer drops properties
        # so the wire shape is just ``{"issues": [...]}``. Compute
        # ``ok`` from issue presence here so CI can still gate on the
        # exit code.
        issues = report.get("issues") or []
        if isinstance(issues, list) and issues:
            raise typer.Exit(code=1)

    _run(_go())


@lint_app.command("propose")
def lint_propose_cmd(
    rule: Annotated[
        str | None,
        typer.Option(
            "--rule",
            help=(
                "Filter to one lint kind: broken_wikilink | orphan_page | "
                "duplicate_title | non_atomic_page. "
                "PR1 only ships a fixer for broken_wikilink; other kinds "
                "are accepted but every issue lands in `skipped`."
            ),
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1, max=200,
            help="Cap the number of lint issues consumed (default 10).",
        ),
    ] = 10,
    enable_llm: Annotated[
        bool,
        typer.Option(
            "--enable-llm",
            help=(
                "Allow fixers to call the configured LLM for fallback "
                "paths (broken_wikilink stub-page generation, the "
                "non_atomic_page splitter). Off by default — opt in "
                "explicitly because each issue may incur a token cost."
            ),
        ),
    ] = False,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            help=(
                "Block until the task finishes; render the proposal "
                "summary and map the final status to the standard exit "
                "code. Without ``--wait`` the command exits immediately "
                "with the task handle JSON."
            ),
        ),
    ] = False,
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable progress widget."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Propose fixes for the current lint findings.

    Default is async — submit + print JSON task handle. Use ``--wait``
    to block + render the proposal summary + emit the
    ``dikw client lint apply <id>`` hint line."""

    async def _go() -> None:
        body: dict[str, Any] = {"limit": limit, "enable_llm": enable_llm}
        if rule is not None:
            body["rule"] = rule
        async with Transport.from_config(_resolve(server, token)) as t:
            handle = await t.post_json("/v1/lint/propose", json_body=body)
            task_id = str(handle["task_id"])
            if not wait and not _serve_and_run_forces_wait():
                _print_task_handle(task_id, str(handle.get("status") or "pending"))
                return
            status, payload = await _wait_and_render(t, task_id, plain=plain)
        if status == "succeeded" and payload is not None:
            console.print(
                f"[green]propose task succeeded[/green] — id=[bold]{task_id}[/bold]"
            )
            from .progress import render_lint_proposals_summary
            render_lint_proposals_summary(console, payload)
            console.print(
                f"\n[dim]apply with:[/dim] dikw client lint apply {task_id}"
            )
        _exit_for_status(status, payload)

    _run(_go())


@lint_app.command("proposals")
def lint_proposals_cmd(
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: 'table' (default) or 'json'.",
        ),
    ] = "table",
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """List succeeded ``lint.propose`` tasks (= pending fix proposals)."""
    _validate_format(fmt)

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            # propose + apply listings are independent reads — walk both
            # cursors concurrently.
            propose_rows, apply_rows = await asyncio.gather(
                _drain_task_list(t, op="lint.propose", status="succeeded"),
                _drain_task_list(t, op="lint.apply", status="succeeded"),
            )
            # 0.2.0: ``GET /v1/tasks`` is summary-only. Re-fetch each
            # row's result so the propose payload (``result.proposals`` /
            # ``result.skipped``) and the apply cross-reference
            # (``result.proposal_task_id``) both survive the projection.
            propose_results, apply_results = await asyncio.gather(
                _gather_task_results(t, propose_rows),
                _gather_task_results(t, apply_rows),
            )
        # Stitch propose result payloads back into the summary rows so
        # both JSON output and the table renderer see ``row["result"]``
        # the way they did pre-0.2.0.
        result_by_id: dict[str, Any] = {}
        for body in propose_results:
            if not isinstance(body, dict):
                continue
            tid = body.get("task_id")
            if isinstance(tid, str):
                result_by_id[tid] = body.get("result")
        hydrated_propose_rows: list[dict[str, Any]] = []
        for row in propose_rows:
            tid = row.get("task_id")
            merged = dict(row)
            if isinstance(tid, str) and tid in result_by_id:
                merged["result"] = result_by_id[tid]
            hydrated_propose_rows.append(merged)
        applied_ids: set[str] = set()
        for body in apply_results:
            result = (body or {}).get("result") or {}
            ref = result.get("proposal_task_id")
            if isinstance(ref, str):
                applied_ids.add(ref)
        from .progress import render_lint_proposals_listing
        if fmt == "json":
            console.print_json(
                json.dumps(
                    {
                        "proposals": hydrated_propose_rows,
                        "applied_ids": sorted(applied_ids),
                    },
                    ensure_ascii=False,
                )
            )
        else:
            render_lint_proposals_listing(console, hydrated_propose_rows, applied_ids)

    _run(_go())


@lint_app.command("apply")
def lint_apply_cmd(
    proposal_task_id: Annotated[
        str,
        typer.Argument(
            help="The task_id of a successful `lint propose` invocation.",
        ),
    ],
    pick: Annotated[
        str | None,
        typer.Option(
            "--pick",
            help="Comma-separated proposal indices to apply (e.g. '0,2').",
        ),
    ] = None,
    skip: Annotated[
        str | None,
        typer.Option(
            "--skip",
            help="Comma-separated proposal indices to drop.",
        ),
    ] = None,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            help=(
                "Block until the task finishes; render the apply summary "
                "and map the final status to the standard exit code."
            ),
        ),
    ] = False,
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable progress widget."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Apply a previously-proposed fix to the wiki.

    Default is async — submit + print JSON task handle. Use ``--wait``
    to block + render the apply report."""

    def _parse_index_list(v: str | None) -> list[int] | None:
        if v is None:
            return None
        try:
            return [int(p.strip()) for p in v.split(",") if p.strip()]
        except ValueError as e:
            raise typer.BadParameter(f"index list must be comma-separated ints: {e}") from e

    pick_list = _parse_index_list(pick)
    skip_list = _parse_index_list(skip)

    async def _go() -> None:
        body: dict[str, Any] = {"proposal_task_id": proposal_task_id}
        if pick_list is not None:
            body["pick"] = pick_list
        if skip_list is not None:
            body["skip"] = skip_list
        async with Transport.from_config(_resolve(server, token)) as t:
            handle = await t.post_json("/v1/lint/apply", json_body=body)
            task_id = str(handle["task_id"])
            if not wait and not _serve_and_run_forces_wait():
                _print_task_handle(task_id, str(handle.get("status") or "pending"))
                return
            status, payload = await _wait_and_render(t, task_id, plain=plain)
        if status == "succeeded" and payload is not None:
            from .progress import render_lint_apply_report
            render_lint_apply_report(console, payload)
        _exit_for_status(status, payload)

    _run(_go())


# ---- retrieve (NDJSON stream, retrieval-only) -------------------------


@app.command(
    "retrieve",
    epilog=(
        "Examples:\n\n"
        "  dikw client retrieve \"deterministic scoping\"\n\n"
        "  dikw client retrieve \"...\" --limit 10 --format table\n\n"
        "  dikw client retrieve \"...\" --plain | jq '.chunks[].text'"
    ),
)
def retrieve_cmd(
    question: Annotated[str, typer.Argument(help="Natural-language query.")],
    limit: Annotated[
        int, typer.Option("--limit", "-k", help="Chunks to retrieve.")
    ] = 5,
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: 'json' (default, agent-friendly) or 'table' (human).",
        ),
    ] = "json",
    plain: Annotated[
        bool,
        typer.Option(
            "--plain",
            help="Disable rich rendering of the retrieving status banner.",
        ),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Retrieve chunks + page-level refs without invoking an LLM.

    The agent-facing knowledge-access verb: streams an NDJSON sequence
    ``retrieve_started → retrieval_done → final`` and stops there.
    Answer synthesis happens in the agent layer, with the agent's own
    LLM. ``--format json`` prints ``final.result`` (chunks + page_refs)
    so the caller can pipe it into ``jq`` or another tool.
    """
    _validate_format(fmt)

    async def _go() -> None:
        renderer = RetrieveStreamRenderer(console, plain=plain)
        async with (
            Transport.from_config(_resolve(server, token)) as t,
            t.stream_ndjson(
                "POST",
                "/v1/retrieve",
                json_body={"q": question, "limit": limit},
            ) as events,
        ):
            final = await renderer.run(events)
        if final.status != "succeeded":
            console.print(f"[red]retrieve {final.status}[/red]")
            if final.error:
                console.print(f"[dim]{final.error}[/dim]")
            raise typer.Exit(code=1)
        if final.result is None:
            console.print("[red]retrieve returned empty result[/red]")
            raise typer.Exit(code=1)
        if fmt == "json":
            # ``console.print_json`` re-pretty-prints; pass already-encoded
            # JSON so non-ASCII (e.g. Chinese chunk text) survives intact.
            console.print_json(json.dumps(final.result, ensure_ascii=False))
        else:
            render_retrieve_table(console, final.result)

    _run(_go())


# ---- async task commands ----------------------------------------------


def _print_task_handle(task_id: str, status: str) -> None:
    """Default async-submit output: agent-friendly JSON envelope.

    Per ``feedback_cli_agent_first_default`` the no-``--wait`` path is
    a single JSON object on stdout — agents pipe it to ``jq``; humans
    follow up with ``dikw client tasks wait <task_id>``."""
    print(
        json.dumps(
            {
                "task_id": task_id,
                "status": status,
                "events_url": f"/v1/tasks/{task_id}/events",
                "wait_command": f"dikw client tasks wait {task_id}",
            },
            ensure_ascii=False,
        )
    )


async def _wait_and_render(
    t: Transport,
    task_id: str,
    *,
    plain: bool,
    poll_wait: int = 30,
    total_timeout: float | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Drive the long-poll cursor loop with the rich progress widget.

    Shared by every op command's ``--wait`` path and by
    ``dikw client tasks wait``. Returns the final ``(status, payload)``
    where ``payload`` is the result dict on success or the error dict
    on failure. Raises :class:`TimeoutError` on local budget expiry —
    the caller maps it to exit ``124``.
    """
    renderer = TaskProgressRenderer(console, plain=plain)
    with renderer.live():
        return await follow_to_terminal(
            t,
            task_id,
            renderer=renderer,
            poll_wait=poll_wait,
            total_timeout=total_timeout,
        )


def _exit_for_status(status: str, payload: dict[str, Any] | None) -> None:
    """Map a terminal task status to the CLI's exit code contract.

    ``succeeded`` → no-op (caller already rendered the report);
    ``failed`` → exit 1; ``cancelled`` → exit 130 (POSIX SIGINT).
    """
    if status == "succeeded":
        return
    if status == "cancelled":
        console.print("[yellow]task cancelled[/yellow]")
        raise typer.Exit(code=_EXIT_CANCELLED)
    console.print(f"[red]task {status}[/red]")
    if payload:
        console.print(f"[dim]{payload}[/dim]")
    raise typer.Exit(code=_EXIT_FAILED)


@app.command(
    "import",
    epilog=(
        "Examples:\n\n"
        "  dikw client import ./inbox\n\n"
        "  dikw client import ./note.md\n\n"
        "  dikw client import ./paper.pdf --converter=marker"
    ),
)
def import_cmd(
    path: Annotated[
        Path,
        typer.Argument(
            help=(
                "Local markdown file or directory to import into the base. "
                "Non-md single files (``paper.pdf``, ``book.epub``, …) are "
                "dispatched to an installed converter plugin first."
            ),
        ),
    ],
    converter: Annotated[
        str | None,
        typer.Option(
            "--converter",
            help=(
                "Engine name to use for non-md inputs (e.g. "
                "``--converter=marker`` for ``paper.pdf``). Overrides the "
                "``client.toml`` ``[default.converters]`` entry for one "
                "call. Plugins ship in the dikw-plugins repo."
            ),
        ),
    ] = None,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Pre-flight + import markdown packages into the server's ``sources/``.

    Each markdown file becomes a single package alongside any asset
    (image, pdf) it embeds. Non-md single-file inputs go through a
    client-side converter plugin first; install the plugin you want
    (e.g. ``dikw-converter-pdf``) and the importer dispatches by
    extension automatically. Pre-flight inspection (frontmatter parse,
    asset existence, non-empty body, no orphan asset) runs locally
    first; failures exit 2 before any bytes leave the machine.

    On success the server commits well-formed packages straight into
    ``<base>/sources/`` (per-package via ``os.replace``) and returns
    a ``committed`` / ``rejected`` summary. Run ``dikw client ingest``
    afterwards to chunk + embed the new sources.

    Distinct from ``dikw auth import``, which loads OAuth credentials
    into the per-base auth store — different target, different command.
    """

    cfg = _resolve(server, token)
    converter_for = _converter_resolver(converter, cfg)

    async def _go() -> None:
        with build_import(path, converter_for=converter_for) as bundle:
            async with Transport.from_config(cfg) as t:
                response = await t.post_multipart(
                    "/v1/import",
                    files={
                        "payload": (
                            "payload.tar.gz",
                            bundle.payload,
                            "application/gzip",
                        )
                    },
                    data={"manifest": bundle.manifest_json},
                )
        render_import_report(console, response)
        if response.get("rejected"):
            raise typer.Exit(code=1)

    _run(_go())


@app.command("ingest")
def ingest_cmd(
    no_embed: Annotated[
        bool,
        typer.Option("--no-embed", help="Skip the dense embedding pass."),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help=(
                "Exit non-zero if any file errored during ingest. "
                "Default behaviour treats per-file errors as warnings "
                "so a single bad file doesn't fail a CI run. Implies "
                "``--wait`` since the per-file error list only exists "
                "after the task finishes."
            ),
        ),
    ] = False,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            help=(
                "Block until the task finishes; render the IngestReport "
                "and map the final status to the standard exit code "
                "(succeeded=0, failed=1, cancelled=130)."
            ),
        ),
    ] = False,
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable progress widget."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Run ingest against the server's ``<base>/sources/`` tree.

    Sources are imported separately via ``dikw client import``. Default
    is async — the command submits the task and prints a JSON handle
    so agents can move on; pass ``--wait`` to block until terminal.
    """

    async def _go() -> None:
        # ``--strict`` only makes sense alongside the final IngestReport,
        # which the async-default path doesn't fetch. Inside
        # ``serve-and-run`` (no ``--keep-alive``) the temporary server
        # dies on inner exit, so an async-default submit would orphan
        # the task — auto-wait there too.
        should_wait = wait or strict or _serve_and_run_forces_wait()
        async with Transport.from_config(_resolve(server, token)) as t:
            handle = await t.post_json(
                "/v1/ingest", json_body={"no_embed": no_embed}
            )
            task_id = str(handle["task_id"])
            if not should_wait:
                _print_task_handle(task_id, str(handle.get("status") or "pending"))
                return
            status, payload = await _wait_and_render(t, task_id, plain=plain)
        if status == "succeeded" and payload is not None:
            render_ingest_report(console, payload)
            errors = payload.get("errors") or []
            if errors:
                render_ingest_errors(console, errors)
                if strict:
                    raise typer.Exit(code=_EXIT_FAILED)
        _exit_for_status(status, payload)

    _run(_go())


@app.command("synth")
def synth_cmd(
    force_all: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Re-synthesise every source, even ones already wiki-ified.",
        ),
    ] = False,
    no_embed: Annotated[
        bool,
        typer.Option(
            "--no-embed",
            help="Skip embedding the generated K-layer pages.",
        ),
    ] = False,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            help=(
                "Block until the task finishes; render the SynthReport "
                "and map the final status to the standard exit code."
            ),
        ),
    ] = False,
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable progress widget."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Synthesise K-layer wiki pages from D-layer sources.

    Default is async — submit + print JSON task handle. Use ``--wait``
    to block + render + exit with task status."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            handle = await t.post_json(
                "/v1/synth",
                json_body={"force_all": force_all, "no_embed": no_embed},
            )
            task_id = str(handle["task_id"])
            if not wait and not _serve_and_run_forces_wait():
                _print_task_handle(task_id, str(handle.get("status") or "pending"))
                return
            status, payload = await _wait_and_render(t, task_id, plain=plain)
        if status == "succeeded" and payload is not None:
            render_synth_report(console, payload)
        _exit_for_status(status, payload)

    _run(_go())


@app.command("distill")
def distill_cmd(
    pages_per_call: Annotated[
        int,
        typer.Option(
            "--batch",
            help="K-layer pages packed into one distill LLM call.",
        ),
    ] = 8,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            help=(
                "Block until the task finishes; render the DistillReport "
                "and map the final status to the standard exit code."
            ),
        ),
    ] = False,
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable progress widget."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Propose W-layer candidates from current K-layer pages.

    Default is async — submit + print JSON task handle. Use ``--wait``
    to block + render + exit with task status."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            handle = await t.post_json(
                "/v1/distill", json_body={"pages_per_call": pages_per_call}
            )
            task_id = str(handle["task_id"])
            if not wait and not _serve_and_run_forces_wait():
                _print_task_handle(task_id, str(handle.get("status") or "pending"))
                return
            status, payload = await _wait_and_render(t, task_id, plain=plain)
        if status == "succeeded" and payload is not None:
            render_distill_report(console, payload)
            if int(payload.get("candidates_added") or 0):
                console.print(
                    "Review with [cyan]dikw client review list[/cyan] / "
                    "[cyan]dikw client review approve <id>[/cyan]."
                )
        _exit_for_status(status, payload)

    _run(_go())


@app.command("eval")
def eval_cmd(
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            "-d",
            help=(
                "Dataset name (resolved on the server) or path on the "
                "server. The client doesn't ship dataset bytes — the "
                "server reads them from its packaged datasets root. "
                "Omit to run every packaged dataset."
            ),
        ),
    ] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--retrieval",
            help="Retrieval mode: hybrid|bm25|vector|all.",
        ),
    ] = "hybrid",
    cache_mode: Annotated[
        str,
        typer.Option(
            "--cache",
            help="Eval-snapshot cache: read_write|rebuild|off.",
        ),
    ] = "read_write",
    eval_modes: Annotated[
        list[str] | None,
        typer.Option(
            "--eval",
            help=(
                "Eval family to run (repeatable): retrieval|synth. "
                "Omit to run whatever the dataset declares in modes:."
            ),
        ),
    ] = None,
    judge: Annotated[
        bool,
        typer.Option(
            "--judge",
            help=(
                "Run the LLM judge soft score on synth-eval pages. "
                "Only meaningful with --eval synth."
            ),
        ),
    ] = False,
    judge_sample: Annotated[
        int | None,
        typer.Option(
            "--judge-sample",
            help=(
                "Sample N pages for the LLM judge instead of judging "
                "all. Ignored when --judge is not set."
            ),
        ),
    ] = None,
    pretty: Annotated[
        bool,
        typer.Option(
            "--pretty",
            help="Render rich tables instead of NDJSON (default).",
        ),
    ] = False,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            help=(
                "Block until the task finishes; render the EvalReport "
                "and map the final status to the standard exit code "
                "(succeeded=0 with gate=pass, failed=1 / gate=fail, "
                "cancelled=130, ``--eval synth`` ungated=2)."
            ),
        ),
    ] = False,
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable progress widget."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Run an eval dataset on the server (retrieval and/or synth).

    Default is async — submit + print JSON task handle. Use ``--wait``
    to block + render + exit with task / gate status."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            handle = await t.post_json(
                "/v1/eval",
                json_body={
                    "dataset": dataset,
                    "mode": mode,
                    "cache_mode": cache_mode,
                    "eval_modes": eval_modes,
                    "judge": judge,
                    "judge_sample": judge_sample,
                },
            )
            task_id = str(handle["task_id"])
            if not wait and not _serve_and_run_forces_wait():
                _print_task_handle(task_id, str(handle.get("status") or "pending"))
                return
            status, payload = await _wait_and_render(t, task_id, plain=plain)
        if status == "succeeded" and payload is not None:
            _render_eval_result(payload, pretty=pretty)
            if not bool(payload.get("passed", True)):
                raise typer.Exit(code=_EXIT_FAILED)
            # An explicit ``--eval synth`` request must have at least
            # one gated synth report — otherwise the user asked for a
            # K-layer gate run and the dataset declared no synth
            # thresholds (informational only). Exit 2 to distinguish
            # "no gate ran" from "gate ran and passed" (exit 0) and
            # "gate failed" (exit 1).
            if eval_modes and "synth" in eval_modes:
                synth_reports = _synth_reports(payload)
                if synth_reports and not any(
                    r.get("gated") for r in synth_reports
                ):
                    console.print(
                        "[yellow]warning:[/yellow] --eval synth ran but "
                        "no synth thresholds were declared; result is "
                        "informational, not a gate pass.",
                        markup=True,
                    )
                    raise typer.Exit(code=2)
        _exit_for_status(status, payload)

    _run(_go())


def _synth_reports(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Pluck the synth-mode reports out of a single- or multi-report
    eval result. Used by the ``--eval synth`` exit-code logic to tell
    "no gate ran" from "all gates passed"."""
    if "datasets" in result and isinstance(result["datasets"], list):
        rows = list(result["datasets"])
    else:
        rows = [result]
    return [
        row for row in rows
        if isinstance(row, dict) and row.get("mode") == "synth"
    ]


def _render_eval_result(result: Mapping[str, Any], *, pretty: bool) -> None:
    """Dispatch one eval result to the right renderer.

    Single-report runs come back at top level; multi-report envelopes
    carry ``datasets: [...]``. Each report's ``mode`` field tells us
    which renderer to use (``retrieval`` → ``render_eval_report``,
    ``synth`` → ``render_synth_eval_report``). When ``--pretty`` is off
    we emit one NDJSON line per report so an agent reading stdout can
    parse without rich's ANSI cruft.
    """
    if "datasets" in result and isinstance(result["datasets"], list):
        rows = list(result["datasets"])
    else:
        rows = [result]
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not pretty:
            print(json.dumps(row, ensure_ascii=False, default=str))
            continue
        if row.get("mode") == "synth":
            render_synth_eval_report(console, row)
        else:
            render_eval_report(console, row)


# ---- review subcommands -----------------------------------------------

review_app = typer.Typer(
    help="Review wisdom candidates.", no_args_is_help=True
)
app.add_typer(review_app, name="review")


@review_app.command("list")
def review_list_cmd(
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: 'table' (default, human) or 'json' (agent-friendly).",
        ),
    ] = "table",
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """List candidate W-layer items awaiting review."""
    _validate_format(fmt)

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            items = await t.get_json(
                "/v1/wisdom", params={"status": "candidate"}
            )
        if fmt == "json":
            console.print_json(json.dumps(items, ensure_ascii=False))
            return
        if not items:
            console.print("[green]no candidates[/green]")
            return
        table = Table(
            title="wisdom candidates", show_header=True, header_style="bold"
        )
        table.add_column("id")
        table.add_column("kind")
        table.add_column("conf", justify="right")
        table.add_column("title")
        for item in items:
            if not isinstance(item, dict):
                continue
            conf_val = item.get("confidence")
            conf_str = (
                f"{float(conf_val):.2f}" if isinstance(conf_val, int | float) else "-"
            )
            table.add_row(
                str(item.get("item_id") or ""),
                str(item.get("kind") or ""),
                conf_str,
                str(item.get("title") or ""),
            )
        console.print(table)

    _run(_go())


@review_app.command("approve")
def review_approve_cmd(
    item_id: Annotated[str, typer.Argument(help="Wisdom item id (W-xxxxxx).")],
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Approve a candidate."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            result = await t.post_json(f"/v1/wisdom/{item_id}/approve")
        console.print(
            f"[green]{result.get('item_id')} -> "
            f"{result.get('new_status')}[/green]"
        )

    _run(_go())


@review_app.command("reject")
def review_reject_cmd(
    item_id: Annotated[str, typer.Argument(help="Wisdom item id (W-xxxxxx).")],
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Reject a candidate."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            result = await t.post_json(f"/v1/wisdom/{item_id}/reject")
        console.print(
            f"[yellow]{result.get('item_id')} -> "
            f"{result.get('new_status')}[/yellow]"
        )

    _run(_go())


# ---- pages subcommands ------------------------------------------------

pages_app = typer.Typer(
    help="Read pages (D / K / W) directly from the server's base.",
    no_args_is_help=True,
)
app.add_typer(pages_app, name="pages")


@pages_app.command(
    "list",
    epilog=(
        "Examples:\n\n"
        "  dikw client pages list\n\n"
        "  dikw client pages list --layer source\n\n"
        "  dikw client pages list --format table"
    ),
)
def pages_list_cmd(
    layer: Annotated[
        Layer | None,
        typer.Option(
            "--layer",
            help="Filter by layer. Default: all layers.",
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: 'json' (default, agent-friendly) or 'table' (human).",
        ),
    ] = "json",
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """List pages registered under the server's base."""
    _validate_format(fmt)

    async def _go() -> None:
        params: dict[str, str] | None = (
            {"layer": layer.value} if layer is not None else None
        )
        async with Transport.from_config(_resolve(server, token)) as t:
            rows = await t.get_json("/v1/base/pages", params=params)
        if fmt == "json":
            console.print_json(json.dumps(rows, ensure_ascii=False))
            return
        table = Table(title="pages", show_header=True, header_style="bold")
        table.add_column("layer")
        table.add_column("path")
        table.add_column("title")
        for row in rows:
            if not isinstance(row, dict):
                continue
            table.add_row(
                str(row.get("layer") or ""),
                str(row.get("path") or ""),
                str(row.get("title") or ""),
            )
        console.print(table)

    _run(_go())


@pages_app.command(
    "get",
    epilog=(
        "Examples:\n\n"
        "  dikw client pages get sources/notes/alpha.md\n\n"
        "  dikw client pages get wiki/Some-Page.md\n\n"
        "  dikw client pages get \"sources/has space.md\""
    ),
)
def pages_get_cmd(
    path: Annotated[
        str,
        typer.Argument(help="Page path under the base (e.g. sources/foo.md)."),
    ],
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Read a page (body + chunk anchors) by its registered path.

    The path must already exist as a ``DocumentRecord`` in the server's
    base — paths that aren't indexed return 404 (use ``dikw client pages
    list`` to discover registered paths)."""

    # Percent-encode each segment so paths with ``?`` ``#`` ``%`` ``&``
    # or whitespace (all legal in markdown filenames) don't get parsed
    # as URL query / fragment / spec-violation by httpx. ``safe="/"``
    # preserves the path-segment separator so FastAPI's ``{path:path}``
    # still sees the correct hierarchy.
    encoded = quote(path, safe="/")

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            payload = await t.get_json(f"/v1/base/pages/{encoded}")
        console.print_json(json.dumps(payload, ensure_ascii=False))

    _run(_go())


@pages_app.command(
    "links",
    epilog=(
        "Examples:\n\n"
        "  dikw client pages links wiki/Some-Page.md\n\n"
        "  dikw client pages links wiki/Some-Page.md --direction out\n\n"
        "  dikw client pages links wiki/Hub.md --limit 20 --format table"
    ),
)
def pages_links_cmd(
    path: Annotated[
        str,
        typer.Argument(help="Page path under the base (e.g. wiki/foo.md)."),
    ],
    direction: Annotated[
        str,
        typer.Option(
            "--direction",
            help="Edge direction: 'in', 'out', or 'both' (default).",
        ),
    ] = "both",
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            help="Cap each list (outgoing AND incoming) at N entries.",
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: 'json' (default, agent-friendly) or 'table' (human).",
        ),
    ] = "json",
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """List the K-layer link graph neighbours of a page.

    Returns ``outgoing`` (edges from this page) and ``incoming`` (edges
    to this page). The path must already exist as a ``DocumentRecord``
    in the server's base — paths that aren't indexed return 404. Use
    ``dikw client pages list`` first to discover registered paths."""
    _validate_format(fmt)
    if direction not in ("in", "out", "both"):
        console.print(
            f"[red]error[/red]: --direction must be 'in', 'out', or 'both', "
            f"got {direction!r}"
        )
        raise typer.Exit(code=2)

    encoded = quote(path, safe="/")

    async def _go() -> None:
        params: dict[str, Any] = {"direction": direction}
        if limit is not None:
            params["limit"] = limit
        async with Transport.from_config(_resolve(server, token)) as t:
            payload = await t.get_json(
                f"/v1/base/pages/{encoded}/links", params=params
            )
        if fmt == "json":
            console.print_json(json.dumps(payload, ensure_ascii=False))
            return
        # ``table`` mode: two stacked tables — outgoing on top, incoming
        # below — labelled so the section boundary is obvious in a tty.
        out_table = Table(
            title=f"outgoing ({len(payload.get('outgoing', []))})",
            show_header=True,
            header_style="bold",
        )
        out_table.add_column("dst_path")
        out_table.add_column("link_type")
        out_table.add_column("line")
        out_table.add_column("anchor")
        for edge in payload.get("outgoing", []):
            if not isinstance(edge, dict):
                continue
            out_table.add_row(
                str(edge.get("dst_path") or ""),
                str(edge.get("link_type") or ""),
                str(edge.get("line") or ""),
                str(edge.get("anchor") or ""),
            )
        console.print(out_table)

        in_table = Table(
            title=f"incoming ({len(payload.get('incoming', []))})",
            show_header=True,
            header_style="bold",
        )
        in_table.add_column("src_path")
        in_table.add_column("link_type")
        in_table.add_column("line")
        in_table.add_column("anchor")
        for edge in payload.get("incoming", []):
            if not isinstance(edge, dict):
                continue
            in_table.add_row(
                str(edge.get("src_path") or ""),
                str(edge.get("link_type") or ""),
                str(edge.get("line") or ""),
                str(edge.get("anchor") or ""),
            )
        console.print(in_table)

    _run(_go())


# ---- assets subcommands -----------------------------------------------

assets_app = typer.Typer(
    help="Download media assets (images) materialized into the server's base.",
    no_args_is_help=True,
)
app.add_typer(assets_app, name="assets")


@assets_app.command(
    "get",
    epilog=(
        "Examples:\n\n"
        "  dikw client assets get a649f5dd...409a --output diagram.jpg\n\n"
        "  dikw client assets get $ID --output ./figures/$ID.png"
    ),
)
def assets_get_cmd(
    asset_id: Annotated[
        str,
        typer.Argument(help="sha256-hex asset id (64 lower-case hex chars)."),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Local file to write the asset bytes into. Required.",
        ),
    ],
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Download an asset's bytes by id and write them to a local file.

    Binary content always lands in ``--output`` (never stdout) so the
    command stays agent-friendly: stdout carries a JSON envelope with
    ``asset_id`` / ``path`` / ``bytes`` that downstream scripts parse.
    """

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            payload = await t.get_bytes(f"/v1/assets/{asset_id}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        console.print_json(
            json.dumps(
                {
                    "asset_id": asset_id,
                    "path": str(output),
                    "bytes": len(payload),
                },
                ensure_ascii=False,
            )
        )

    _run(_go())


# ---- graph subcommand -------------------------------------------------

graph_app = typer.Typer(
    help="Read the full base graph (nodes + edges + unresolved wikilinks).",
    no_args_is_help=True,
)
app.add_typer(graph_app, name="graph")


@graph_app.command(
    "get",
    epilog=(
        "Examples:\n\n"
        "  dikw client graph get | jq '.stats'\n\n"
        "  dikw client graph get --no-active | jq '.nodes | length'"
    ),
)
def graph_get_cmd(
    active: Annotated[
        bool,
        typer.Option(
            "--active/--no-active",
            help=(
                "Whether to include only active docs (default true) or only "
                "deactivated ones (--no-active). Mirrors GET /v1/base/pages."
            ),
        ),
    ] = True,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Fetch the full base graph in one read-only request.

    Replaces the old web-side workaround of looping
    ``dikw client pages get`` for every page and re-parsing wikilinks.
    The server returns nodes (every doc), edges (every resolvable
    wikilink / cross-page markdown link), and ``unresolved[]`` (broken
    wikilinks). Output is single-payload JSON — pipe into ``jq`` or
    your agent's JSON parser.
    """

    async def _go() -> None:
        # str(bool).lower() so the wire receives ``true`` / ``false``
        # (FastAPI's bool coercion accepts ``true``, ``false``, ``1``,
        # ``0``, ``yes``, ``no``).
        params = {"active": str(active).lower()}
        async with Transport.from_config(_resolve(server, token)) as t:
            payload = await t.get_json("/v1/base/graph", params=params)
        console.print_json(json.dumps(payload, ensure_ascii=False))

    _run(_go())


# ---- tasks subcommands ------------------------------------------------

tasks_app = typer.Typer(
    help="Inspect server-side async tasks.", no_args_is_help=True
)
app.add_typer(tasks_app, name="tasks")


@tasks_app.command("list")
def tasks_list_cmd(
    op: Annotated[
        str | None, typer.Option("--op", help="Filter by op name.")
    ] = None,
    status_filter: Annotated[
        str | None,
        typer.Option(
            "--status",
            help="Filter by status (pending|running|succeeded|failed|cancelled).",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            help="Page size. Default 100, max 1000.",
        ),
    ] = 100,
    all_pages: Annotated[
        bool,
        typer.Option(
            "--all",
            help=(
                "Walk the cursor until exhausted and emit a flat array "
                "(--format json) or a combined table. Default is a "
                "single page with the server envelope passed through."
            ),
        ),
    ] = False,
    cursor: Annotated[
        str | None,
        typer.Option(
            "--cursor",
            help=(
                "Opaque cursor from a prior response's ``next_cursor``. "
                "Ignored when --all is set."
            ),
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: 'table' (default, human) or 'json' (agent-friendly).",
        ),
    ] = "table",
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """List server-side tasks.

    Default (single page): the server envelope ``{tasks, next_cursor,
    has_more}`` flows through unchanged on ``--format json`` so agents
    can advance the cursor themselves. ``--all`` drains the cursor and
    emits a flat array — convenient for humans piping into ``jq`` /
    ``less``.
    """
    _validate_format(fmt)

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            if all_pages:
                rows = await _drain_task_list(
                    t,
                    op=op,
                    status=status_filter,
                    page_size=limit,
                )
                if fmt == "json":
                    console.print_json(
                        json.dumps(rows, ensure_ascii=False)
                    )
                    return
                _render_tasks_table(rows)
                return

            params: dict[str, Any] = {"limit": limit}
            if op is not None:
                params["op"] = op
            if status_filter is not None:
                params["status"] = status_filter
            if cursor is not None:
                params["cursor"] = cursor
            envelope = await t.get_json("/v1/tasks", params=params)
            if fmt == "json":
                console.print_json(json.dumps(envelope, ensure_ascii=False))
                return
            tasks = (envelope or {}).get("tasks") or []
            _render_tasks_table(tasks)

    _run(_go())


def _render_tasks_table(rows: list[dict[str, Any]]) -> None:
    """Shared table renderer for ``tasks list`` (single-page and --all).

    Empty input prints the dim "no tasks" hint so the human-default
    table mode still has an empty-state signal after the 0.2.0
    envelope refactor."""
    if not rows:
        console.print("[dim]no tasks[/dim]")
        return
    table = Table(title="tasks", show_header=True, header_style="bold")
    table.add_column("task_id")
    table.add_column("op")
    table.add_column("status")
    table.add_column("created_at")
    for row in rows:
        if not isinstance(row, dict):
            continue
        table.add_row(
            str(row.get("task_id") or ""),
            str(row.get("op") or ""),
            str(row.get("status") or ""),
            str(row.get("created_at") or ""),
        )
    console.print(table)


@tasks_app.command("status")
def tasks_status_cmd(
    task_id: Annotated[str, typer.Argument(help="Task id (12-char hex).")],
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Print the JSON snapshot of a task row.

    Uses ``print(json.dumps(...))`` instead of rich's ``print_json`` so
    long values (timestamps, queue identifiers, error messages) don't
    get rich's soft-wrap injected mid-string — agent parsers need clean
    JSON regardless of terminal width.
    """

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            row = await t.get_json(f"/v1/tasks/{task_id}")
        print(json.dumps(row, ensure_ascii=False))

    _run(_go())


@tasks_app.command("events")
def tasks_events_cmd(
    task_id: Annotated[str, typer.Argument(help="Task id (12-char hex).")],
    from_seq: Annotated[
        int,
        typer.Option(
            "--from-seq",
            help="First seq to return (cursor).",
            min=0,
        ),
    ] = 0,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            help="Cap on events in this page (1..1000).",
            min=1,
            max=1000,
        ),
    ] = 100,
    wait: Annotated[
        int,
        typer.Option(
            "--wait",
            help=(
                "Server hold time in seconds (0..60). 0 is a snapshot; "
                ">0 long-polls until a new event lands or the timeout "
                "fires server-side."
            ),
            min=0,
            max=60,
        ),
    ] = 0,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Fetch one ``EventsPage`` from a task's cursor endpoint.

    The agent paging primitive: one HTTP call, raw JSON to stdout,
    exit 0 regardless of task status. Agents script their own
    cursor-advance loop on top; humans usually want ``tasks wait``
    instead."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            page = await t.get_task_events_page(
                task_id, from_seq=from_seq, limit=limit, wait=wait
            )
        print(json.dumps(page, ensure_ascii=False))

    _run(_go())


@tasks_app.command("wait")
def tasks_wait_cmd(
    task_id: Annotated[str, typer.Argument(help="Task id (12-char hex).")],
    poll_wait: Annotated[
        int,
        typer.Option(
            "--poll-wait",
            help=(
                "Per-HTTP-call hold time in seconds (clamped to server "
                "cap 60). Default 30."
            ),
            min=1,
            max=60,
        ),
    ] = 30,
    total_timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout",
            help=(
                "Client-side total budget in seconds. On expiry the "
                "command exits 124; the task is NOT auto-cancelled "
                "(chain ``dikw client tasks cancel`` if needed)."
            ),
        ),
    ] = None,
    plain: Annotated[
        bool, typer.Option("--plain", help="Disable progress widget.")
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Block until the task reaches a terminal state.

    Renders a rich progress widget to a TTY (or one tidy plain-text
    line per progress tick under ``--plain``), then maps the final
    status to the standard exit code: succeeded=0, failed=1,
    cancelled=130, client-side timeout=124."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            try:
                status, payload = await _wait_and_render(
                    t,
                    task_id,
                    plain=plain,
                    poll_wait=poll_wait,
                    total_timeout=total_timeout,
                )
            except TimeoutError:
                console.print(
                    f"[yellow]timeout[/yellow] — task {task_id} still running"
                )
                raise typer.Exit(code=_EXIT_TIMEOUT) from None
        _exit_for_status(status, payload)

    _run(_go())


@tasks_app.command("cancel")
def tasks_cancel_cmd(
    task_id: Annotated[str, typer.Argument(help="Task id (12-char hex).")],
    pretty: Annotated[
        bool,
        typer.Option(
            "--pretty",
            help=(
                "Render a colored human-readable line instead of the "
                "default raw CancelResponse JSON."
            ),
        ),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Request cancellation of a running task.

    Default output is the raw ``CancelResponse`` JSON
    (``{task_id, cancelled, already_terminal}``) on stdout so agents
    can pipe to ``jq`` without stripping rich's ANSI. Pass
    ``--pretty`` for the colored human line."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            payload = await t.post_json(f"/v1/tasks/{task_id}/cancel")
        if not pretty:
            print(json.dumps(payload, ensure_ascii=False))
            return
        if payload.get("already_terminal"):
            console.print(
                f"[dim]task {task_id} already terminal — no-op[/dim]"
            )
        else:
            console.print(
                f"[yellow]cancel requested[/yellow] for {task_id}"
            )

    _run(_go())


# ---- serve-and-run ----------------------------------------------------


@app.command(
    "serve-and-run",
    help=(
        "Start a local server, run an inner CLI command against it, "
        "and tear it down. Pass the inner command after ``--``."
    ),
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def serve_and_run_cmd(
    ctx: typer.Context,
    base: Annotated[
        Path,
        typer.Option(
            "--base",
            "-b",
            help="Path to the dikw base (must contain dikw.yml). Defaults to cwd.",
        ),
    ] = Path("."),
    host: Annotated[
        str,
        typer.Option(
            "--host",
            "-H",
            help="Interface to bind. 0.0.0.0 requires --token.",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="TCP port to bind. ``0`` (default) picks a free one.",
        ),
    ] = 0,
    token: Annotated[
        str | None,
        typer.Option(
            "--token",
            help=(
                "Bearer token. Forwarded to both the server (--token) "
                "and the inner client (DIKW_SERVER_TOKEN). Required when "
                "--host is non-loopback."
            ),
        ),
    ] = None,
    ready_timeout: Annotated[
        float,
        typer.Option(
            "--ready-timeout",
            help="Seconds to wait for /v1/healthz before giving up.",
        ),
    ] = 30.0,
    keep_alive: Annotated[
        bool,
        typer.Option(
            "--keep-alive",
            help=(
                "After the inner command exits, leave the server "
                "running and print its connection details."
            ),
        ),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="uvicorn log level for the spawned server.",
        ),
    ] = "warning",
) -> None:
    """One-shot server + inner-command lifecycle.

    Examples:

        dikw client serve-and-run -- status
        dikw client serve-and-run --base ./my-base -- ingest --no-embed
        dikw client serve-and-run --keep-alive -- retrieve "..."
    """
    inner_cmd = list(ctx.args)
    opts = _sar.ServeAndRunOptions(
        base=base,
        host=host,
        port=port,
        token=token,
        ready_timeout=ready_timeout,
        keep_alive=keep_alive,
        log_level=log_level,
        inner_cmd=inner_cmd,
    )
    rc = _sar.run(opts)
    if rc != 0:
        raise typer.Exit(code=rc)


__all__ = ["app"]
