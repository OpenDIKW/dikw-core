"""Renderer behaviour tests.

Drives the renderers with a hand-rolled NDJSON event sequence and
inspects the rendered text via rich's ``Console(record=True)``. We check
shape (presence of expected substrings + final-event return value),
not exact whitespace — rich's table glyphs vary per terminal width and
are not part of the contract.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from rich.console import Console

from dikw_core.client.progress import (
    TaskProgressRenderer,
    render_distill_report,
    render_eval_report,
    render_health_report,
    render_ingest_report,
    render_status,
)


async def _scripted(events: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for ev in events:
        yield ev


@pytest.mark.asyncio
async def test_task_progress_renderer_returns_final_event() -> None:
    console = Console(record=True, width=80, force_terminal=False)
    renderer = TaskProgressRenderer(console, plain=True)
    events = [
        {"type": "task_started", "task_id": "abc", "op": "ingest"},
        {"type": "progress", "phase": "scan", "current": 1, "total": 3},
        {"type": "progress", "phase": "scan", "current": 3, "total": 3},
        {"type": "progress", "phase": "embed_chunks", "current": 1, "total": 1},
        {"type": "log", "level": "WARN", "message": "low disk"},
        {
            "type": "final",
            "status": "succeeded",
            "result": {"scanned": 3, "added": 3, "embedded": 7},
        },
    ]
    with renderer.live():
        final = await renderer.run(_scripted(events))
    assert final.status == "succeeded"
    assert final.result == {"scanned": 3, "added": 3, "embedded": 7}
    assert final.error is None
    out = console.export_text()
    assert "low disk" in out


@pytest.mark.asyncio
async def test_task_progress_renderer_renders_multi_phase_streams_distinctly() -> None:
    """Outer (``synth`` source counter) and inner (``synth_llm`` group
    counter) phases must each get their own line — without phase-keyed
    rows the inner counter would overwrite the outer one and the user
    would lose the ``2/43`` source progress as soon as group events fire."""
    console = Console(record=True, width=80, force_terminal=False)
    renderer = TaskProgressRenderer(console, plain=True)
    events = [
        {"type": "progress", "phase": "synth", "current": 1, "total": 3},
        {
            "type": "progress",
            "phase": "synth_llm",
            "current": 1,
            "total": 4,
            "detail": {"status": "calling"},
        },
        {
            "type": "progress",
            "phase": "synth_llm",
            "current": 1,
            "total": 4,
            "detail": {"status": "returned"},
        },
        {
            "type": "final",
            "status": "succeeded",
            "result": {"candidates": 1, "created": 1},
        },
    ]
    with renderer.live():
        final = await renderer.run(_scripted(events))
    assert final.status == "succeeded"
    out = console.export_text()
    assert "synth: 1/3" in out
    assert "synth_llm: 1/4" in out


@pytest.mark.asyncio
async def test_task_progress_renderer_falls_back_when_no_final() -> None:
    """A stream that closes without ``final`` surfaces as a synthetic
    ``failed`` so the caller doesn't have to special-case it."""
    console = Console(record=True, width=80, force_terminal=False)
    renderer = TaskProgressRenderer(console, plain=True)
    events: list[dict[str, Any]] = [
        {"type": "progress", "phase": "scan", "current": 0, "total": 1},
    ]
    with renderer.live():
        final = await renderer.run(_scripted(events))
    assert final.status == "failed"
    assert final.result is None


def test_render_ingest_report_table_has_metrics() -> None:
    console = Console(record=True, width=80, force_terminal=False)
    render_ingest_report(
        console,
        {
            "scanned": 4,
            "added": 4,
            "updated": 0,
            "unchanged": 0,
            "chunks": 8,
            "embedded": 8,
        },
    )
    out = console.export_text()
    # Renderer labels the embedding row "embeddings" (mirroring
    # IngestReport.embedded → display name); accept either spelling.
    assert "scanned" in out
    assert "embeddings" in out or "embedded" in out


def test_render_status_handles_missing_keys() -> None:
    """A pre-init wiki returns mostly zeros; renderer must not blow up
    on missing optional fields like ``last_wiki_log_ts``."""
    console = Console(record=True, width=80, force_terminal=False)
    render_status(
        console,
        {
            "documents_by_layer": {},
            "chunks": 0,
            "embeddings": 0,
            "links": 0,
            "wisdom_by_status": {},
        },
    )
    out = console.export_text()
    assert "source" in out and "wisdom" in out


def test_render_distill_report_renders_zeroes() -> None:
    console = Console(record=True, width=80, force_terminal=False)
    render_distill_report(
        console,
        {"pages_read": 3, "candidates_added": 0, "rejected": 0, "errors": 0},
    )
    out = console.export_text()
    assert "K pages read" in out


def test_render_health_report_renders_all_blocks() -> None:
    """Drive ``render_health_report`` with a fully-populated payload so
    the table-mode CLI path stays covered (the JSON-default path is the
    agent contract; this is the human-debug surface).
    """
    console = Console(record=True, width=120, force_terminal=False)
    render_health_report(
        console,
        {
            "status": "ok",
            "version": "0.0.0+test",
            "base_root": "/tmp/test-base",
            "storage_engine": "sqlite",
            "layer_counts": {
                "sources": 3,
                "wiki_pages": 2,
                "wisdom_items": 0,
                "chunks": 11,
            },
            "providers": {
                "llm": {
                    "provider": "openai_compat",
                    "model": "gpt-5-mini",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_present": True,
                },
                "embedding": {
                    "provider": "openai_compat",
                    "model": "text-embed-3-large",
                    "base_url": None,
                    "api_key_present": False,
                    "multimodal": {
                        "provider": "openai_compat",
                        "model": "mm-embed-1",
                        "dim": 1024,
                        "distance": "cosine",
                        "base_url": "https://mm.example.com/v1",
                    },
                },
            },
        },
    )
    out = console.export_text()
    # Every block should land at least its title in the captured output.
    assert "dikw health" in out
    assert "layer counts" in out
    assert "providers" in out
    assert "multimodal embedding" in out
    # api_key flag rendering: present → ✓, absent → ✗.
    assert "✓" in out and "✗" in out


def test_render_eval_report_marks_failures() -> None:
    """A failing metric must show up as ``FAIL`` so CI logs are
    grep-able for regressions."""
    console = Console(record=True, width=80, force_terminal=False)
    render_eval_report(
        console,
        {
            "dataset_name": "toy",
            "metrics": {"hit_at_3": 0.10, "mrr": 0.40},
            "thresholds": {"hit_at_3": 0.50, "mrr": 0.30},
        },
    )
    out = console.export_text()
    assert "FAIL" in out
    assert "pass" in out
