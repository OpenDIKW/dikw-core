"""Append-only wiki activity log.

The authoritative store is ``storage.knowledge_log`` (rows). ``wiki/log.md`` is a
materialised view of that table rendered in Karpathy-friendly markdown with
one bullet per event:

    ## [YYYY-MM-DD HH:MM] action | src -> dst
    optional note text

The renderer fully rewrites ``log.md`` every time so the on-disk file
always matches the database. We deliberately do not treat the file as
canonical — it's a convenience for human readers and for Obsidian.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from ...schemas import KnowledgeLogEntry

LOG_PATH = "knowledge/log.md"

_HEADER = (
    "---\n"
    "type: log\n"
    "updated: {updated}\n"
    "---\n\n"
    "# Knowledge Log\n\n"
    "> Auto-generated from the engine's activity log.\n"
)


def format_entry(entry: KnowledgeLogEntry) -> str:
    ts = datetime.fromtimestamp(entry.ts, tz=UTC).strftime("%Y-%m-%d %H:%M")
    head = f"## [{ts}] {entry.action}"
    if entry.src and entry.dst:
        head += f" | {entry.src} -> {entry.dst}"
    elif entry.src:
        head += f" | {entry.src}"
    elif entry.dst:
        head += f" | -> {entry.dst}"
    lines = [head]
    if entry.note:
        lines.append("")
        lines.append(entry.note.strip())
    lines.append("")
    return "\n".join(lines)


def render_log(
    root: Path, entries: Sequence[KnowledgeLogEntry], *, updated: str
) -> Path:
    abs_log = root / LOG_PATH
    abs_log.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = [_HEADER.format(updated=updated)]
    if not entries:
        parts.append("\n_No activity recorded yet._\n")
    else:
        # newest first — matches how humans scan an activity log
        for entry in sorted(entries, key=lambda e: e.ts, reverse=True):
            parts.append("\n")
            parts.append(format_entry(entry))

    abs_log.write_text("".join(parts), encoding="utf-8")
    return abs_log
