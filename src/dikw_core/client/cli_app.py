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
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

import typer
from rich.console import Console
from rich.table import Table

from ..schemas import Layer
from . import serve_and_run as _sar
from .config import ClientConfig, resolve
from .importer import SourceImportError, build_import
from .progress import (
    QueryStreamRenderer,
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
    render_synth_report,
)
from .transport import ClientError, Transport

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


# ---- meta + sync commands ---------------------------------------------


@app.command("info")
def info_cmd(
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Print the server's ``GET /v1/info`` response."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            payload = await t.get_json("/v1/info")
        console.print(json.dumps(payload, indent=2))

    _run(_go())


@app.command("status")
def status_cmd(
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
        "  dikw client check --llm-only\n\n"
        "  dikw client check --embed-only"
    ),
)
def check_cmd(
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


@app.command("init")
def init_cmd(
    description: Annotated[
        str | None,
        typer.Option(
            "--description",
            "-d",
            help="One-line description for dikw.yml on the server.",
        ),
    ] = None,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Ask the server to scaffold its bound base (no-op when already scaffolded).

    For starting a fresh base *locally*, use the top-level ``dikw init``
    command — it works without a running server.
    """

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            try:
                payload = await t.post_json(
                    "/v1/init",
                    json_body={"description": description}
                    if description is not None
                    else {},
                )
            except ClientError as e:
                # The server's own runtime refuses to start without a
                # ``dikw.yml`` already present, so the only way this 409
                # fires in normal use is "wiki is already scaffolded" —
                # treat it as a no-op success per the command's own
                # docstring instead of exiting non-zero.
                if e.code == "wiki_already_initialised":
                    console.print(
                        "[yellow]base already initialized[/yellow] — no-op"
                    )
                    return
                raise
        console.print(f"[green]initialized[/green] base at {payload.get('root')}")

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
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable progress widget."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Propose fixes for the current lint findings.

    Submits a ``lint.propose`` task, follows its event stream, and on
    success prints the task_id + per-proposal summary. The task_id is
    what `dikw client lint apply <id>` consumes."""

    async def _go() -> None:
        body: dict[str, Any] = {"limit": limit, "enable_llm": enable_llm}
        if rule is not None:
            body["rule"] = rule
        async with Transport.from_config(_resolve(server, token)) as t:
            handle = await t.post_json("/v1/lint/propose", json_body=body)
            task_id = str(handle["task_id"])
            renderer = TaskProgressRenderer(console, plain=plain)
            with renderer.live():
                async with t.stream_ndjson(
                    "GET", f"/v1/tasks/{task_id}/events"
                ) as events:
                    final = await renderer.run(events)
        if final.status == "succeeded" and final.result is not None:
            console.print(
                f"[green]propose task succeeded[/green] — id=[bold]{task_id}[/bold]"
            )
            from .progress import render_lint_proposals_summary
            render_lint_proposals_summary(console, final.result)
            console.print(
                f"\n[dim]apply with:[/dim] dikw client lint apply {task_id}"
            )
        _exit_on_failure(final.status, final.error)

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
            # Fetch propose + apply listings in parallel — they're
            # independent reads and the second saves a round trip.
            tasks_resp, applies_resp = await asyncio.gather(
                t.get_json("/v1/tasks?op=lint.propose&status=succeeded"),
                t.get_json("/v1/tasks?op=lint.apply&status=succeeded"),
            )
        propose_rows = tasks_resp if isinstance(tasks_resp, list) else []
        applies = applies_resp if isinstance(applies_resp, list) else []
        # Cross-reference applies via apply.result.proposal_task_id.
        # TaskRow exposes only ``params_digest`` (not raw params), so
        # the cross-reference key has to live in the apply runner's
        # result payload — which lint_op stamps in before completion.
        applied_ids: set[str] = set()
        for r in applies:
            result = (r or {}).get("result") or {}
            ref = result.get("proposal_task_id")
            if isinstance(ref, str):
                applied_ids.add(ref)
        from .progress import render_lint_proposals_listing
        if fmt == "json":
            console.print_json(
                json.dumps(
                    {"proposals": propose_rows, "applied_ids": sorted(applied_ids)},
                    ensure_ascii=False,
                )
            )
        else:
            render_lint_proposals_listing(console, propose_rows, applied_ids)

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
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable progress widget."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Apply a previously-proposed fix to the wiki."""

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
            renderer = TaskProgressRenderer(console, plain=plain)
            with renderer.live():
                async with t.stream_ndjson(
                    "GET", f"/v1/tasks/{task_id}/events"
                ) as events:
                    final = await renderer.run(events)
        if final.status == "succeeded" and final.result is not None:
            from .progress import render_lint_apply_report
            render_lint_apply_report(console, final.result)
        _exit_on_failure(final.status, final.error)

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
    """Retrieve chunks + page-level refs without invoking the LLM.

    Companion to ``query`` for AI agents that want to assemble their
    own answer: streams the same NDJSON event sequence as ``query`` but
    skips ``llm_token`` events. ``--format json`` prints
    ``final.result`` (chunks + page_refs) so the caller can pipe it into
    ``jq`` or another tool.
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


# ---- query (NDJSON stream) --------------------------------------------


@app.command(
    "query",
    epilog=(
        "Examples:\n\n"
        "  dikw client query \"What is X?\"\n\n"
        "  dikw client query \"...\" --limit 10 --show-hits\n\n"
        "  dikw client query \"...\" --plain"
    ),
)
def query_cmd(
    question: Annotated[str, typer.Argument(help="Natural-language question.")],
    limit: Annotated[
        int, typer.Option("--limit", "-k", help="Excerpts to retrieve.")
    ] = 5,
    show_hits: Annotated[
        bool,
        typer.Option(
            "--show-hits",
            help="Also dump raw retrieval hits after the citations table.",
        ),
    ] = False,
    plain: Annotated[
        bool,
        typer.Option(
            "--plain",
            help="Disable rich rendering (useful for piping into other tools).",
        ),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Answer ``question`` via streaming NDJSON."""

    async def _go() -> None:
        renderer = QueryStreamRenderer(
            console, plain=plain, show_hits=show_hits
        )
        async with (
            Transport.from_config(_resolve(server, token)) as t,
            t.stream_ndjson(
                "POST",
                "/v1/query",
                json_body={"q": question, "limit": limit},
            ) as events,
        ):
            final = await renderer.run(events)
        if final.status != "succeeded":
            console.print(f"[red]query {final.status}[/red]")
            if final.error:
                console.print(f"[dim]{final.error}[/dim]")
            raise typer.Exit(code=1)

    _run(_go())


# ---- async task commands ----------------------------------------------


async def _follow_task(
    t: Transport,
    *,
    submit_path: str,
    body: dict[str, Any],
    plain: bool,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    handle = await t.post_json(submit_path, json_body=body)
    task_id = str(handle["task_id"])
    renderer = TaskProgressRenderer(console, plain=plain)
    with renderer.live():
        async with t.stream_ndjson(
            "GET", f"/v1/tasks/{task_id}/events"
        ) as events:
            final = await renderer.run(events)
    return final.status, final.result, final.error


def _exit_on_failure(
    status: str, error: dict[str, Any] | None
) -> None:
    if status == "succeeded":
        return
    console.print(f"[red]task {status}[/red]")
    if error:
        console.print(f"[dim]{error}[/dim]")
    raise typer.Exit(code=1)


@app.command(
    "import",
    epilog=(
        "Examples:\n\n"
        "  dikw client import ./inbox\n\n"
        "  dikw client import ./note.md"
    ),
)
def import_cmd(
    path: Annotated[
        Path,
        typer.Argument(
            help=(
                "Local markdown file or directory to import into the base. "
                "Each ``*.md`` becomes one package together with its "
                "referenced assets."
            ),
        ),
    ],
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Pre-flight + import markdown packages into the server's ``sources/``.

    Each markdown file becomes a single package alongside any
    asset (image, pdf) it embeds. Pre-flight inspection (frontmatter
    parse, asset existence, non-empty body, no orphan asset) runs
    locally first; failures exit 2 before any bytes leave the
    machine.

    On success the server commits well-formed packages straight into
    ``<base>/sources/`` (per-package via ``os.replace``) and returns
    a ``committed`` / ``rejected`` summary. Run ``dikw client ingest``
    afterwards to chunk + embed the new sources.

    Distinct from ``dikw auth import``, which loads OAuth credentials
    into the per-base auth store — different target, different command.
    """

    async def _go() -> None:
        with build_import(path) as bundle:
            async with Transport.from_config(_resolve(server, token)) as t:
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
                "so a single bad file doesn't fail a CI run."
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

    Sources are imported separately via ``dikw client import``.
    """

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            status, result, error = await _follow_task(
                t,
                submit_path="/v1/ingest",
                body={"no_embed": no_embed},
                plain=plain,
            )
        if status == "succeeded" and result is not None:
            render_ingest_report(console, result)
            errors = result.get("errors") or []
            if errors:
                render_ingest_errors(console, errors)
                if strict:
                    raise typer.Exit(code=1)
        _exit_on_failure(status, error)

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
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable progress widget."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Synthesise K-layer wiki pages from D-layer sources."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            status, result, error = await _follow_task(
                t,
                submit_path="/v1/synth",
                body={"force_all": force_all, "no_embed": no_embed},
                plain=plain,
            )
        if status == "succeeded" and result is not None:
            render_synth_report(console, result)
        _exit_on_failure(status, error)

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
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable progress widget."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Propose W-layer candidates from current K-layer pages."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            status, result, error = await _follow_task(
                t,
                submit_path="/v1/distill",
                body={"pages_per_call": pages_per_call},
                plain=plain,
            )
        if status == "succeeded" and result is not None:
            render_distill_report(console, result)
            if int(result.get("candidates_added") or 0):
                console.print(
                    "Review with [cyan]dikw client review list[/cyan] / "
                    "[cyan]dikw client review approve <id>[/cyan]."
                )
        _exit_on_failure(status, error)

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
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable progress widget."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Run a packaged retrieval-eval dataset on the server."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            status, result, error = await _follow_task(
                t,
                submit_path="/v1/eval",
                body={
                    "dataset": dataset,
                    "mode": mode,
                    "cache_mode": cache_mode,
                },
                plain=plain,
            )
        if status == "succeeded" and result is not None:
            # No-arg server path returns ``{datasets: [...], passed: bool}``;
            # render each report and gate exit on the aggregate pass.
            if "datasets" in result and isinstance(result["datasets"], list):
                for ds in result["datasets"]:
                    render_eval_report(console, ds)
                if not bool(result.get("passed", True)):
                    raise typer.Exit(code=1)
            else:
                render_eval_report(console, result)
                if not bool(result.get("passed", True)):
                    raise typer.Exit(code=1)
        _exit_on_failure(status, error)

    _run(_go())


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
        int, typer.Option("--limit", help="Max rows to return.")
    ] = 100,
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
    """List server-side tasks."""
    _validate_format(fmt)

    async def _go() -> None:
        params: dict[str, Any] = {"limit": limit}
        if op is not None:
            params["op"] = op
        if status_filter is not None:
            params["status"] = status_filter
        async with Transport.from_config(_resolve(server, token)) as t:
            rows = await t.get_json("/v1/tasks", params=params)
        if fmt == "json":
            console.print_json(json.dumps(rows, ensure_ascii=False))
            return
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

    _run(_go())


@tasks_app.command("show")
def tasks_show_cmd(
    task_id: Annotated[str, typer.Argument(help="Task id (12-char hex).")],
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Print the JSON snapshot of a task row."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            row = await t.get_json(f"/v1/tasks/{task_id}")
        console.print(json.dumps(row, indent=2))

    _run(_go())


@tasks_app.command("follow")
def tasks_follow_cmd(
    task_id: Annotated[str, typer.Argument(help="Task id (12-char hex).")],
    from_seq: Annotated[
        int, typer.Option("--from-seq", help="Resume from this seq number.")
    ] = 0,
    plain: Annotated[
        bool, typer.Option("--plain", help="Disable progress widget.")
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Subscribe to a task's NDJSON event stream."""

    async def _go() -> None:
        renderer = TaskProgressRenderer(console, plain=plain)
        async with Transport.from_config(_resolve(server, token)) as t:
            with renderer.live():
                async with t.stream_ndjson(
                    "GET",
                    f"/v1/tasks/{task_id}/events",
                    params={"from_seq": from_seq} if from_seq else None,
                ) as events:
                    final = await renderer.run(events)
        if final.status != "succeeded":
            console.print(f"[yellow]task {final.status}[/yellow]")
            if final.error:
                console.print(f"[dim]{final.error}[/dim]")
            raise typer.Exit(code=1)

    _run(_go())


@tasks_app.command("cancel")
def tasks_cancel_cmd(
    task_id: Annotated[str, typer.Argument(help="Task id (12-char hex).")],
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Request cancellation of a running task."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            payload = await t.post_json(f"/v1/tasks/{task_id}/cancel")
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
        dikw client serve-and-run --keep-alive -- query "..."
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
