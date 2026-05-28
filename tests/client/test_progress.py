"""Renderer behaviour tests.

Drives the renderers with a hand-rolled NDJSON event sequence and
inspects the rendered text via rich's ``Console(record=True)``. We check
shape (presence of expected substrings + final-event return value),
not exact whitespace — rich's table glyphs vary per terminal width and
are not part of the contract.
"""

from __future__ import annotations

from rich.console import Console

from dikw_core.client.progress import (
    TaskProgressRenderer,
    render_eval_report,
    render_health_report,
    render_import_report,
    render_ingest_report,
    render_retrieve_table,
    render_status,
    render_synth_eval_report,
    render_synth_report,
)


def test_task_progress_renderer_logs_warning() -> None:
    """``render(event)`` for a ``log`` event prints it to the console.

    Since PR4 the renderer is a per-event sink driven by
    :func:`follow_to_terminal`; this test asserts the dispatch shape
    rather than NDJSON stream iteration."""
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
        for ev in events:
            renderer.render(ev)
    out = console.export_text()
    assert "low disk" in out


def test_task_progress_renderer_renders_multi_phase_streams_distinctly() -> None:
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
        for ev in events:
            renderer.render(ev)
    out = console.export_text()
    assert "synth: 1/3" in out
    assert "synth_llm: 1/4" in out


def test_task_progress_renderer_unknown_event_is_noop() -> None:
    """Unknown event types must not raise — schema growth on the server
    side should never crash the renderer."""
    console = Console(record=True, width=80, force_terminal=False)
    renderer = TaskProgressRenderer(console, plain=True)
    with renderer.live():
        renderer.render({"type": "future_event_kind", "weird": 42})
        renderer.render({"type": "progress", "phase": "scan", "current": 0, "total": 1})
    # No assertion needed — the test passes if no exception was raised.


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
    on missing optional fields like ``last_knowledge_log_ts``."""
    console = Console(record=True, width=80, force_terminal=False)
    render_status(
        console,
        {
            "documents_by_layer": {},
            "chunks": 0,
            "embeddings": 0,
            "links": 0,
        },
    )
    out = console.export_text()
    assert "source" in out and "wisdom" in out


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
                "knowledge_pages": 2,
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
    assert "dikw client health" in out
    assert "layer counts" in out
    assert "providers" in out
    assert "multimodal embedding" in out
    # api_key flag rendering: present → ✓, absent → ✗.
    assert "✓" in out and "✗" in out


def test_render_retrieve_table_smoke() -> None:
    """Smoke: ``render_retrieve_table`` produces both chunks and
    page_refs tables when both are present. Guards the title strings
    (which were rewritten 0.1.0 to ``dikw client retrieve …``) and
    catches obvious shape regressions."""
    console = Console(record=True, width=120, force_terminal=False)
    render_retrieve_table(
        console,
        {
            "chunks": [
                {
                    "layer": "source",
                    "path": "sources/a.md",
                    "seq": 0,
                    "score": 0.91,
                    "snippet": "alpha snippet",
                }
            ],
            "page_refs": [
                {
                    "layer": "source",
                    "path": "sources/a.md",
                    "score": 0.88,
                    "hit_chunk_ids": [0, 1],
                }
            ],
        },
    )
    out = console.export_text()
    assert "dikw client retrieve" in out
    assert "alpha snippet" in out


def test_render_import_report_smoke() -> None:
    """Smoke: ``render_import_report`` lists the summary block + the
    rejected-packages table when any rejects are present."""
    console = Console(record=True, width=120, force_terminal=False)
    render_import_report(
        console,
        {
            "files_count": 3,
            "bytes": 4096,
            "committed": ["pkg-0", "pkg-1"],
            "rejected": [
                {"id": 2, "code": "frontmatter_invalid", "detail": "bad yaml"}
            ],
        },
    )
    out = console.export_text()
    assert "dikw client import" in out
    assert "frontmatter_invalid" in out


def test_render_synth_report_smoke() -> None:
    """Smoke: ``render_synth_report`` lists every K-layer counter so
    the table title rewrite (``dikw synth`` → ``dikw client synth``)
    can't regress silently."""
    console = Console(record=True, width=80, force_terminal=False)
    render_synth_report(
        console,
        {
            "candidates": 4,
            "created": 2,
            "updated": 1,
            "skipped": 1,
            "errors": 0,
            "unresolved_wikilinks": 0,
        },
    )
    out = console.export_text()
    assert "dikw client synth" in out
    assert "candidates" in out


def test_render_synth_eval_report_smoke() -> None:
    """Smoke: ``render_synth_eval_report`` renders the direction-aware
    threshold rows + the informational metrics tail."""
    console = Console(record=True, width=120, force_terminal=False)
    render_synth_eval_report(
        console,
        {
            "dataset_name": "toy-synth",
            "threshold_results": [
                {
                    "name": "synth/atomicity",
                    "observed": 0.7,
                    "threshold": 0.5,
                    "direction": "min",
                    "passed": True,
                }
            ],
            "metrics": {"synth/atomicity": 0.7, "synth/coverage": 0.55},
            "informational": {},
        },
    )
    out = console.export_text()
    assert "dikw client eval" in out
    assert "synth/atomicity" in out


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
