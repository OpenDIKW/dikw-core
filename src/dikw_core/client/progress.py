"""NDJSON event → rich rendering for ``dikw client`` commands.

Two renderers, one per wire shape:

* :class:`TaskProgressRenderer` — driven by ``GET /v1/tasks/{id}/events``.
  Updates a rich ``Progress`` widget while the server emits ``progress``
  events; logs WARN/ERROR lines above it; yields the terminal ``final``
  event so the caller can render an op-specific result table.
* :class:`RetrieveStreamRenderer` — driven by ``POST /v1/retrieve``.
  Collects the NDJSON event stream until ``final`` and returns it; the
  CLI picks JSON vs table output. dikw-core no longer ships an
  in-engine query renderer — LLM synthesis is the agent's job.

Both renderers degrade gracefully when ``--no-progress`` is set: the
caller can pass ``plain=True`` to get unstyled, line-oriented output
suitable for piping into other tools.

Final-result rendering helpers (``render_ingest_report`` etc.) are
imported by ``cli_app`` and mirror the in-process tables that used to
live in ``cli.py``. Keeping them as plain functions instead of methods
on the renderer means callers that already have a result dict (e.g. a
sync RPC, no progress stream) can render it without owning a renderer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table


@dataclass
class FinalEvent:
    """The terminal NDJSON event from a task / stream.

    Decouples the wire-level ``final`` envelope from the caller's
    rendering: the caller picks an op-specific renderer based on
    ``status`` (succeeded → render table, failed → format error,
    cancelled → polite notice) without re-parsing JSON.
    """

    status: str  # "succeeded" | "failed" | "cancelled"
    result: dict[str, Any] | None
    error: dict[str, Any] | None


class TaskProgressRenderer:
    """Drives a rich Progress while consuming a task event stream.

    Use as ``async with renderer.live(): final = await renderer.run(stream)``.
    The ``Progress`` widget creates one row per distinct ``phase`` and
    updates ``current/total`` in place; no flicker, no scrollback churn.
    Rendering is best-effort — any unknown event types are logged at
    DEBUG and ignored so a server bumping the wire schema doesn't break
    older clients.
    """

    def __init__(self, console: Console, *, plain: bool = False) -> None:
        self._console = console
        self._plain = plain
        self._progress: Progress | None = None
        self._tasks: dict[str, TaskID] = {}

    @contextmanager
    def live(self) -> Any:
        """Open the Progress widget; yields a context where ``run`` works.

        In ``plain`` mode we don't open a live widget — we just write
        plain text lines so the caller can pipe to a logfile.
        """
        if self._plain:
            yield self
            return
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=self._console,
            transient=False,
        )
        with self._progress:
            yield self
        self._progress = None
        self._tasks.clear()

    async def run(
        self, events: AsyncIterator[dict[str, Any]]
    ) -> FinalEvent:
        async for event in events:
            ev_type = event.get("type")
            if ev_type == "progress":
                self._on_progress(event)
            elif ev_type == "log":
                self._on_log(event)
            elif ev_type == "partial":
                self._on_partial(event)
            elif ev_type == "task_started":
                pass  # handled implicitly by the first progress row
            elif ev_type == "final":
                return FinalEvent(
                    status=str(event.get("status") or "failed"),
                    result=_as_dict(event.get("result")),
                    error=_as_dict(event.get("error")),
                )
        # Stream closed without a final event — surface as a synthetic
        # failure so the caller doesn't have to special-case it.
        return FinalEvent(status="failed", result=None, error=None)

    def _on_progress(self, event: Mapping[str, Any]) -> None:
        phase = str(event.get("phase") or "work")
        current = int(event.get("current") or 0)
        total = int(event.get("total") or 0)
        if self._progress is None:
            # Plain mode: print one tidy line per progress tick. Skip
            # zero-total events (engine emits them at phase start) so
            # the log stays readable.
            if total:
                self._console.print(
                    f"  {phase}: {current}/{total}", style="dim"
                )
            return
        if phase not in self._tasks:
            self._tasks[phase] = self._progress.add_task(
                description=phase, total=total or None
            )
        task_id = self._tasks[phase]
        self._progress.update(
            task_id, completed=current, total=total or None
        )

    def _on_log(self, event: Mapping[str, Any]) -> None:
        level = str(event.get("level") or "INFO").upper()
        message = str(event.get("message") or "")
        style = {
            "WARN": "yellow",
            "WARNING": "yellow",
            "ERROR": "red",
        }.get(level, "dim")
        self._console.print(f"[{style}]{level}[/{style}] {message}")

    def _on_partial(self, event: Mapping[str, Any]) -> None:
        # Most partial events are op-specific structured payloads
        # consumed by the final renderer (e.g. query's retrieval_done);
        # the renderer treats them as opaque so unknown ones don't
        # surface as visual noise mid-progress.
        return


class RetrieveStreamRenderer:
    """Drain a ``POST /v1/retrieve`` NDJSON stream into a ``FinalEvent``.

    Retrieve has no LLM stage to stream, so this renderer just collects
    events until ``final`` and returns it — the CLI command picks the
    output format (json / table) based on its own flag. We deliberately
    don't print citations here: the retrieve consumer is typically an
    agent that wants raw chunks + page_refs JSON, and humans driving
    ``--format table`` get a purpose-built table renderer at the CLI
    layer.
    """

    def __init__(self, console: Console, *, plain: bool = False) -> None:
        self._console = console
        self._plain = plain

    async def run(
        self, events: AsyncIterator[dict[str, Any]]
    ) -> FinalEvent:
        async for event in events:
            ev_type = event.get("type")
            if ev_type == "retrieve_started":
                if not self._plain:
                    self._console.print("[dim]retrieving…[/dim]", end="\r")
            elif ev_type == "final":
                return FinalEvent(
                    status=str(event.get("status") or "succeeded"),
                    result=_as_dict(event.get("result")),
                    error=_as_dict(event.get("error")),
                )
            # retrieval_done / progress / log / partial / heartbeat all
            # ignored — heartbeat is dropped at the transport layer; the
            # rest are not user-facing for retrieve.
        # Stream closed without final — synthesise a failed FinalEvent so
        # callers see a structured outcome instead of a bare None.
        return FinalEvent(status="failed", result=None, error=None)


# Excerpt length cap for the retrieve table preview. Keeps each row to
# roughly one terminal line; full text remains in ``--format json``
# output for agents that need it.
_RETRIEVE_TABLE_EXCERPT_CHARS = 120


def render_retrieve_table(
    console: Console, result: Mapping[str, Any]
) -> None:
    """Render a retrieve final.result as two stacked rich tables.

    Optimised for human debugging of a retrieve call (``--format table``);
    agents should use ``--format json`` instead.
    """
    chunks = result.get("chunks") or []
    chunks_table = Table(
        title="dikw retrieve · chunks",
        show_header=True,
        header_style="bold",
    )
    chunks_table.add_column("#", justify="right")
    chunks_table.add_column("layer")
    chunks_table.add_column("path")
    chunks_table.add_column("seq", justify="right")
    chunks_table.add_column("score", justify="right")
    chunks_table.add_column("excerpt")
    if isinstance(chunks, list):
        for i, c in enumerate(chunks, start=1):
            if not isinstance(c, dict):
                continue
            seq = c.get("seq")
            score = c.get("score")
            excerpt = str(c.get("snippet") or c.get("text") or "")
            preview = excerpt[:_RETRIEVE_TABLE_EXCERPT_CHARS] + (
                "…" if len(excerpt) > _RETRIEVE_TABLE_EXCERPT_CHARS else ""
            )
            chunks_table.add_row(
                str(i),
                str(c.get("layer") or ""),
                str(c.get("path") or ""),
                "" if seq is None else str(seq),
                "" if score is None else f"{float(score):.3f}",
                preview,
            )
    console.print(chunks_table)

    page_refs = result.get("page_refs") or []
    if isinstance(page_refs, list) and page_refs:
        refs_table = Table(
            title="dikw retrieve · page_refs",
            show_header=True,
            header_style="bold",
        )
        refs_table.add_column("layer")
        refs_table.add_column("path")
        refs_table.add_column("score", justify="right")
        refs_table.add_column("hit chunks")
        for r in page_refs:
            if not isinstance(r, dict):
                continue
            score = r.get("score")
            hit_ids = r.get("hit_chunk_ids") or []
            refs_table.add_row(
                str(r.get("layer") or ""),
                str(r.get("path") or ""),
                "" if score is None else f"{float(score):.3f}",
                ", ".join(str(c) for c in hit_ids) if isinstance(hit_ids, list) else "",
            )
        console.print(refs_table)


# ---- final-result rendering ---------------------------------------------


def render_ingest_report(console: Console, report: Mapping[str, Any]) -> None:
    table = Table(title="dikw ingest", show_header=True, header_style="bold")
    table.add_column("metric", justify="left")
    table.add_column("count", justify="right")
    for key in ("scanned", "added", "updated", "unchanged"):
        table.add_row(key, str(int(report.get(key) or 0)))
    table.add_row("chunks", str(int(report.get("chunks") or 0)))
    table.add_row("embeddings", str(int(report.get("embedded") or 0)))
    console.print(table)


def render_ingest_errors(
    console: Console, errors: list[Mapping[str, Any]]
) -> None:
    """Render the per-file failure list emitted by ``api.ingest`` as a
    ``kind | path | message`` table — same vocabulary as the other
    report renderers in this module."""
    if not errors:
        return
    table = Table(
        title=f"file errors ({len(errors)})",
        show_header=True,
        header_style="bold",
        title_style="yellow",
    )
    table.add_column("kind", style="red")
    table.add_column("path")
    table.add_column("message", overflow="fold")
    for err in errors:
        if not isinstance(err, Mapping):
            continue
        table.add_row(
            str(err.get("kind") or "?"),
            str(err.get("path") or "?"),
            str(err.get("message") or ""),
        )
    console.print(table)


def render_import_report(
    console: Console, report: Mapping[str, Any]
) -> None:
    """Render the ``ImportResponse`` returned by ``POST /v1/import``.

    Shows the import summary (files, bytes), the list of committed
    package ids, and a table of rejected packages with reason codes."""
    files_count = int(report.get("files_count") or 0)
    total_bytes = int(report.get("bytes") or 0)
    committed = list(report.get("committed") or [])
    rejected = list(report.get("rejected") or [])

    summary = Table(
        title="dikw import", show_header=True, header_style="bold"
    )
    summary.add_column("metric", justify="left")
    summary.add_column("count", justify="right")
    summary.add_row("files", str(files_count))
    summary.add_row("bytes", str(total_bytes))
    summary.add_row("packages committed", str(len(committed)))
    summary.add_row("packages rejected", str(len(rejected)))
    console.print(summary)

    if committed:
        ids = ", ".join(str(i) for i in committed)
        console.print(f"[green]committed[/green]: {ids}")

    if rejected:
        table = Table(
            title=f"rejected packages ({len(rejected)})",
            show_header=True,
            header_style="bold",
            title_style="yellow",
        )
        table.add_column("id", justify="right")
        table.add_column("code", style="red")
        table.add_column("detail", overflow="fold")
        for r in rejected:
            if not isinstance(r, Mapping):
                continue
            # ``id`` can legitimately be 0 — ``or "?"`` would mask it,
            # so explicit None / missing check.
            id_value = r.get("id")
            id_str = "?" if id_value is None else str(id_value)
            table.add_row(
                id_str,
                str(r.get("code") or "?"),
                str(r.get("detail") or ""),
            )
        console.print(table)


def render_synth_report(console: Console, report: Mapping[str, Any]) -> None:
    table = Table(title="dikw synth", show_header=True, header_style="bold")
    table.add_column("metric", justify="left")
    table.add_column("count", justify="right")
    for key in (
        "candidates",
        "created",
        "updated",
        "skipped",
        "errors",
        "unresolved_wikilinks",
    ):
        table.add_row(key, str(int(report.get(key) or 0)))
    console.print(table)


def render_distill_report(console: Console, report: Mapping[str, Any]) -> None:
    table = Table(title="dikw distill", show_header=True, header_style="bold")
    table.add_column("metric", justify="left")
    table.add_column("count", justify="right")
    table.add_row("K pages read", str(int(report.get("pages_read") or 0)))
    table.add_row(
        "candidates added", str(int(report.get("candidates_added") or 0))
    )
    table.add_row(
        "rejected (invariant)", str(int(report.get("rejected") or 0))
    )
    table.add_row("errors", str(int(report.get("errors") or 0)))
    console.print(table)


def render_lint_report(console: Console, report: Mapping[str, Any]) -> None:
    issues = report.get("issues") or []
    if not isinstance(issues, list) or not issues:
        console.print("[green]lint clean[/green] — 0 issues")
        return
    by_kind: dict[str, int] = {}
    for issue in issues:
        if isinstance(issue, dict):
            kind = str(issue.get("kind") or "?")
            by_kind[kind] = by_kind.get(kind, 0) + 1
    summary = " · ".join(f"{k}: {v}" for k, v in sorted(by_kind.items()))
    console.print(f"[yellow]lint issues[/yellow] — {summary}")
    table = Table(show_header=True, header_style="bold")
    table.add_column("kind")
    table.add_column("path")
    table.add_column("line", justify="right")
    table.add_column("detail")
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        line = issue.get("line")
        table.add_row(
            str(issue.get("kind") or ""),
            str(issue.get("path") or ""),
            "" if line is None else str(line),
            str(issue.get("detail") or ""),
        )
    console.print(table)


def render_lint_proposals_summary(
    console: Console, report: Mapping[str, Any]
) -> None:
    """Print the per-proposal summary returned by a ``lint.propose`` task.

    Skips when the result has no proposals so the caller's ``[green]
    succeeded[/green]`` line still leads the output."""
    proposals = report.get("proposals") or []
    skipped = report.get("skipped") or []
    if not proposals and not skipped:
        console.print("[yellow]no fix proposals produced[/yellow]")
        return
    if proposals:
        table = Table(title="proposals", show_header=True, header_style="bold")
        table.add_column("idx", justify="right")
        table.add_column("kind")
        table.add_column("path")
        table.add_column("source")
        table.add_column("ops", justify="right")
        for idx, p in enumerate(proposals):
            if not isinstance(p, dict):
                continue
            ops = p.get("operations") or []
            table.add_row(
                str(idx),
                str(p.get("issue_kind") or ""),
                str(p.get("issue_path") or ""),
                str(p.get("source") or ""),
                str(len(ops)),
            )
        console.print(table)
    if skipped:
        console.print(f"[dim]{len(skipped)} issue(s) skipped[/dim]")


def render_lint_proposals_listing(
    console: Console,
    rows: list[Mapping[str, Any]],
    applied_ids: set[str],
) -> None:
    """Render the ``dikw client lint proposals`` table.

    Each row is a successful ``lint.propose`` task. ``applied_ids``
    is the set of propose-task ids referenced by some ``lint.apply``
    task's ``result.proposal_task_id``."""
    if not rows:
        console.print("[green]no pending lint proposals[/green]")
        return
    table = Table(title="lint proposals", show_header=True, header_style="bold")
    table.add_column("task_id")
    table.add_column("created_at")
    table.add_column("proposals", justify="right")
    table.add_column("skipped", justify="right")
    table.add_column("applied?")
    for row in rows:
        if not isinstance(row, dict):
            continue
        result = row.get("result") or {}
        task_id = str(row.get("task_id") or "")
        applied = "yes" if task_id in applied_ids else "no"
        proposals_count = len(result.get("proposals") or [])
        skipped_count = len(result.get("skipped") or [])
        table.add_row(
            task_id,
            str(row.get("created_at") or ""),
            str(proposals_count),
            str(skipped_count),
            applied,
        )
    console.print(table)


def render_lint_apply_report(
    console: Console, report: Mapping[str, Any]
) -> None:
    """Print the per-op summary of a ``lint.apply`` task result."""
    applied = report.get("applied") or []
    skipped = report.get("skipped") or []
    paths = report.get("wiki_paths_changed") or []
    console.print(
        f"[green]applied[/green] {len(applied)} op(s); "
        f"[yellow]skipped[/yellow] {len(skipped)}"
    )
    if paths:
        console.print(f"[dim]paths changed:[/dim] {', '.join(paths)}")
    if skipped:
        table = Table(title="skipped ops", show_header=True, header_style="bold")
        table.add_column("op")
        table.add_column("path")
        table.add_column("reason")
        for s in skipped:
            if not isinstance(s, dict):
                continue
            table.add_row(
                str(s.get("op") or ""),
                str(s.get("path") or ""),
                str(s.get("reason") or ""),
            )
        console.print(table)


def render_health_report(console: Console, report: Mapping[str, Any]) -> None:
    """Render a ``GET /v1/health`` response as stacked rich tables.

    Used by ``dikw client health --format table``; the JSON-default path
    is the agent contract and skips this renderer.
    """
    overview = Table(title="dikw health", show_header=True, header_style="bold")
    overview.add_column("field", justify="left")
    overview.add_column("value", justify="left")
    overview.add_row("status", str(report.get("status") or ""))
    overview.add_row("version", str(report.get("version") or ""))
    overview.add_row("base_root", str(report.get("base_root") or ""))
    overview.add_row("storage_engine", str(report.get("storage_engine") or ""))
    console.print(overview)

    counts = report.get("layer_counts")
    counts_table = Table(
        title="layer counts", show_header=True, header_style="bold"
    )
    counts_table.add_column("layer")
    counts_table.add_column("count", justify="right")
    if isinstance(counts, dict):
        for key in ("sources", "wiki_pages", "wisdom_items", "chunks"):
            raw = counts.get(key)
            # ``isinstance(_, int | float)`` admits ``bool`` (subtype of
            # int); exclude it so a future schema slip-up that puts True
            # into a count field renders as ``?`` instead of ``1``.
            if isinstance(raw, int | float) and not isinstance(raw, bool):
                counts_table.add_row(key, str(int(raw)))
            else:
                counts_table.add_row(key, "?")
    else:
        # Surface schema drift instead of silently skipping the section.
        counts_table.add_row("(unavailable)", "?")
    console.print(counts_table)

    providers = report.get("providers") or {}
    if isinstance(providers, dict):
        providers_table = Table(
            title="providers", show_header=True, header_style="bold"
        )
        providers_table.add_column("leg")
        providers_table.add_column("provider")
        providers_table.add_column("model")
        providers_table.add_column("base_url")
        providers_table.add_column("api_key", justify="center")
        for leg in ("llm", "embedding"):
            info = providers.get(leg)
            if not isinstance(info, dict):
                continue
            providers_table.add_row(
                leg,
                str(info.get("provider") or ""),
                str(info.get("model") or ""),
                str(info.get("base_url") or "(default)"),
                "✓" if info.get("api_key_present") else "✗",
            )
        console.print(providers_table)
        embedding = providers.get("embedding")
        if isinstance(embedding, dict):
            mm = embedding.get("multimodal")
            if isinstance(mm, dict):
                mm_table = Table(
                    title="multimodal embedding",
                    show_header=True,
                    header_style="bold",
                )
                mm_table.add_column("field")
                mm_table.add_column("value")
                for key in ("provider", "model", "dim", "distance", "base_url"):
                    mm_table.add_row(
                        key,
                        str(mm.get(key) if mm.get(key) is not None else "(default)"),
                    )
                console.print(mm_table)


def render_status(console: Console, counts: Mapping[str, Any]) -> None:
    table = Table(title="dikw status", show_header=True, header_style="bold")
    table.add_column("layer", justify="left")
    table.add_column("count", justify="right")
    layers = counts.get("documents_by_layer") or {}
    if isinstance(layers, dict):
        for layer in ("source", "wiki", "wisdom"):
            table.add_row(layer, str(int(layers.get(layer) or 0)))
    table.add_row("chunks (I)", str(int(counts.get("chunks") or 0)))
    table.add_row("embeddings (I)", str(int(counts.get("embeddings") or 0)))
    table.add_row("links (K)", str(int(counts.get("links") or 0)))
    wisdom = counts.get("wisdom_by_status") or {}
    if isinstance(wisdom, dict):
        for status_name in ("candidate", "approved", "archived"):
            table.add_row(
                f"wisdom {status_name}",
                str(int(wisdom.get(status_name) or 0)),
            )
    console.print(table)
    last = counts.get("last_wiki_log_ts")
    if last is not None:
        console.print(f"last wiki_log ts: [dim]{last}[/dim]")


def render_check_report(console: Console, report: Mapping[str, Any]) -> None:
    table = Table(title="dikw check", show_header=True, header_style="bold")
    table.add_column("provider", justify="left")
    table.add_column("target", justify="left")
    table.add_column("status", justify="left")
    table.add_column("detail", justify="left")
    for label, key in (("LLM", "llm"), ("Embedding", "embed")):
        probe = report.get(key)
        if not isinstance(probe, dict):
            continue
        ok = bool(probe.get("ok"))
        status = "[green]✓ OK[/green]" if ok else "[red]✗ FAIL[/red]"
        table.add_row(
            label,
            str(probe.get("target") or ""),
            status,
            str(probe.get("detail") or ""),
        )
    console.print(table)


def render_synth_eval_report(
    console: Console, report: Mapping[str, Any]
) -> None:
    """Render a ``SynthEvalReport`` (K-layer eval).

    Mirrors ``render_eval_report`` shape but iterates the report's own
    ``threshold_results`` list — that carries the direction-aware
    ``passed`` flag (``synth/duplicate_ratio_max`` is ≤, others are ≥),
    so we don't replay the direction logic in the renderer."""
    name = str(report.get("dataset_name") or report.get("dataset") or "synth eval")
    title = f"dikw eval — {name} (mode: synth)"
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_column("threshold", justify="right")
    table.add_column("direction", justify="center")
    table.add_column("result")

    threshold_results = report.get("threshold_results") or []
    if isinstance(threshold_results, list):
        for row in threshold_results:
            if not isinstance(row, dict):
                continue
            name_ = str(row.get("name") or "?")
            observed = row.get("observed")
            threshold = row.get("threshold")
            direction = str(row.get("direction") or "min")
            passed = bool(row.get("passed"))
            val_str = (
                f"{observed:.3f}" if isinstance(observed, int | float) else "-"
            )
            thr_str = (
                f"{threshold:.3f}"
                if isinstance(threshold, int | float)
                else "-"
            )
            verdict = (
                "[green]✓ pass[/green]"
                if passed
                else "[red]✗ FAIL[/red]"
            )
            table.add_row(name_, val_str, thr_str, direction, verdict)

    metrics = report.get("metrics") or {}
    informational = report.get("informational") or {}
    if isinstance(metrics, dict):
        scored = {
            r.get("name") for r in threshold_results if isinstance(r, dict)
        }
        for m_name, value in sorted(metrics.items()):
            if m_name in scored:
                continue
            val_str = (
                f"{value:.3f}" if isinstance(value, int | float) else "-"
            )
            table.add_row(m_name, val_str, "-", "-", "[dim](info)[/dim]")
    if isinstance(informational, dict):
        for m_name, value in sorted(informational.items()):
            val_str = (
                f"{value:.3f}" if isinstance(value, int | float) else "-"
            )
            table.add_row(m_name, val_str, "-", "-", "[dim](info)[/dim]")

    console.print(table)

    n_sources = report.get("n_sources")
    n_pages = report.get("n_pages")
    passed = bool(report.get("passed", True))
    footer = (
        f"n_sources={n_sources}  n_pages={n_pages}  "
        f"passed={'[green]True[/green]' if passed else '[red]False[/red]'}"
    )
    console.print(footer)

    judge_summary = report.get("judge_summary")
    if isinstance(judge_summary, dict):
        judge_table = Table(
            title="LLM judge (soft score, 0-5)",
            show_header=True,
            header_style="bold",
        )
        judge_table.add_column("dimension")
        judge_table.add_column("mean", justify="right")
        for dim in ("grounding", "atomicity", "completeness", "clarity"):
            value = judge_summary.get(f"mean_{dim}")
            judge_table.add_row(
                dim,
                f"{value:.2f}" if isinstance(value, int | float) else "-",
            )
        console.print(judge_table)
        console.print(
            f"judged {judge_summary.get('n_judged', 0)} pages, "
            f"errors: {judge_summary.get('n_errors', 0)}"
        )

    warnings = report.get("warnings") or []
    if isinstance(warnings, list):
        for w in warnings:
            console.print(f"[yellow]warning[/yellow] {w}")


def render_eval_report(console: Console, report: Mapping[str, Any]) -> None:
    """Mirror of cli.py's ``_print_eval_report`` minus the diagnostic block.

    Diagnostics (``per_query``, ``negative_diagnostics``) are omitted in
    the client-side rendering because they're only useful when the
    operator can correlate hits to local files; for a client/server
    setup the diagnostic block belongs in a dedicated ``eval --verbose``
    flow we'll add when there's user demand.
    """
    name = str(report.get("dataset_name") or "eval")
    title = f"dikw eval — {name}"
    metrics = report.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    thresholds = report.get("thresholds") or {}
    if not isinstance(thresholds, dict):
        thresholds = {}

    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_column("threshold", justify="right")
    table.add_column("result")
    for metric, value in sorted(metrics.items()):
        thr = thresholds.get(metric)
        if not isinstance(value, int | float):
            val_str = "-"
            verdict = "[dim]—[/dim]"
        else:
            val_str = f"{value:.3f}"
            if not isinstance(thr, int | float):
                verdict = "[dim]—[/dim]"
            elif value >= thr:
                verdict = "[green]✓ pass[/green]"
            else:
                verdict = "[red]✗ FAIL[/red]"
        thr_str = (
            f"{thr:.3f}" if isinstance(thr, int | float) else "-"
        )
        table.add_row(metric, val_str, thr_str, verdict)
    console.print(table)


# ---- helpers ------------------------------------------------------------


def _as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    return None


__all__ = [
    "FinalEvent",
    "RetrieveStreamRenderer",
    "TaskProgressRenderer",
    "render_check_report",
    "render_distill_report",
    "render_eval_report",
    "render_health_report",
    "render_ingest_report",
    "render_lint_report",
    "render_retrieve_table",
    "render_status",
    "render_synth_eval_report",
    "render_synth_report",
]
