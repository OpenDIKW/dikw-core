"""Tests for the engine ↔ ProgressReporter wiring (Phase 1 of the
client/server migration).

The engine must emit progress / partial events through the reporter when
one is supplied, AND keep the legacy reporter=None path 100% intact for
in-process callers (CLI today, tests). Cancellation must surface as
``asyncio.CancelledError`` so server task wrappers see the standard
async cancellation signal.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from dikw_core import api
from dikw_core.progress import CancelToken, NoopReporter

from .fakes import FakeEmbeddings, FakeLLM, init_test_wiki

FIXTURES = Path(__file__).parent / "fixtures" / "notes"


@dataclass
class _Event:
    kind: str  # "progress" | "log" | "partial"
    payload: dict[str, Any]


@dataclass
class ListReporter:
    """In-memory reporter that captures every event for assertions.

    Mirrors the ``ProgressReporter`` Protocol; tests that need the full
    event tape just check ``events`` after the engine call returns.
    Optionally pre-seeds a ``CancelToken`` so a test can request
    cooperative cancellation before the engine starts.
    """

    events: list[_Event] = field(default_factory=list)
    _token: CancelToken = field(default_factory=CancelToken)

    async def progress(
        self,
        *,
        phase: str,
        current: int = 0,
        total: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            _Event(
                "progress",
                {
                    "phase": phase,
                    "current": current,
                    "total": total,
                    "detail": detail,
                },
            )
        )

    async def log(self, level: str, message: str) -> None:
        self.events.append(_Event("log", {"level": level, "message": message}))

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        self.events.append(_Event("partial", {"kind": kind, "payload": payload}))

    def cancel_token(self) -> CancelToken:
        return self._token

    def phases(self) -> list[str]:
        return [
            e.payload["phase"]
            for e in self.events
            if e.kind == "progress"
        ]

    def partial_kinds(self) -> list[str]:
        return [e.payload["kind"] for e in self.events if e.kind == "partial"]


# ---- primitives ---------------------------------------------------------


class TestPrimitives:
    def test_cancel_token_is_initially_clean(self) -> None:
        tok = CancelToken()
        assert tok.raised is False
        tok.raise_if_cancelled()  # should not raise

    def test_cancel_token_raises_after_cancel(self) -> None:
        tok = CancelToken()
        tok.cancel()
        assert tok.raised is True
        with pytest.raises(asyncio.CancelledError):
            tok.raise_if_cancelled()

    @pytest.mark.asyncio
    async def test_noop_reporter_swallows_everything(self) -> None:
        r = NoopReporter()
        await r.progress(phase="x", current=1, total=2, detail={"a": 1})
        await r.log("INFO", "hello")
        await r.partial("foo", {"k": "v"})
        assert r.cancel_token().raised is False


# ---- engine integration -------------------------------------------------


@pytest.fixture()
def wiki_with_fixtures(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki, description="reporter wiki")
    dest = wiki / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    return wiki


@pytest.mark.asyncio
async def test_ingest_emits_scan_and_embed_phases(wiki_with_fixtures: Path) -> None:
    reporter = ListReporter()
    embedder = FakeEmbeddings()

    report = await api.ingest(
        wiki_with_fixtures, embedder=embedder, reporter=reporter
    )

    # Engine still returns the same dataclass — reporter is a side-channel.
    assert report.scanned == 3
    assert report.added == 3

    phases = reporter.phases()
    # Scan phase fires once before any file + once per scanned file.
    scan_count = sum(1 for p in phases if p == "scan")
    assert scan_count == 1 + report.scanned

    # Each embed_chunks batch fires one progress event.
    embed_phases = [p for p in phases if p == "embed_chunks"]
    assert len(embed_phases) >= 1, "expected at least one embed_chunks event"

    # Per-event invariants on the embed_chunks phase.
    embed_events = [
        e for e in reporter.events
        if e.kind == "progress" and e.payload["phase"] == "embed_chunks"
    ]
    totals = {e.payload["total"] for e in embed_events}
    assert totals == {len(embed_events)}, (
        "embed_chunks events should agree on the total batch count"
    )
    currents = [e.payload["current"] for e in embed_events]
    assert currents == sorted(currents) and currents[-1] == len(embed_events)


@pytest.mark.asyncio
async def test_ingest_default_path_unchanged_without_reporter(
    wiki_with_fixtures: Path,
) -> None:
    """reporter=None must keep the legacy CLI behaviour: no errors, same
    report fields. The rich progress bar is silenced in non-TTY pytest."""
    embedder = FakeEmbeddings()
    report = await api.ingest(wiki_with_fixtures, embedder=embedder)
    assert report.added == 3
    assert report.embedded >= 3


@pytest.mark.asyncio
async def test_synthesize_emits_one_event_per_source(
    wiki_with_fixtures: Path,
) -> None:
    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)

    reporter = ListReporter()
    # FakeLLM returns "STUB: wired up." — synthesize will record a parse
    # error per source, but a progress event still fires for every
    # source so the reporter wiring is exercised end-to-end.
    llm = FakeLLM()
    report = await api.synthesize(
        wiki_with_fixtures, llm=llm, embedder=embedder, reporter=reporter
    )

    synth_events = [
        e for e in reporter.events
        if e.kind == "progress" and e.payload["phase"] == "synth"
    ]
    assert len(synth_events) == report.candidates
    # current monotonically increases and matches enumerate(start=1).
    currents = [e.payload["current"] for e in synth_events]
    assert currents == list(range(1, len(synth_events) + 1))
    # total is the number of source documents on every event.
    totals = {e.payload["total"] for e in synth_events}
    assert totals == {report.candidates}


class _ScriptedSynthLLM:
    """Synthesize stub: returns a canned <page> block per source."""

    def __init__(self, by_source: dict[str, str]) -> None:
        self._by_source = by_source

    async def complete(self, *, system: str, user: str, model: str, **_: Any) -> Any:
        from dikw_core.providers import LLMResponse

        for src_path, body in self._by_source.items():
            if src_path in user:
                return LLMResponse(text=body, finish_reason="end_turn")
        raise AssertionError(f"no scripted page for prompt: {user[:200]}")

    def complete_stream(self, **_: Any) -> Any:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_distill_emits_one_event_per_batch(
    wiki_with_fixtures: Path,
) -> None:
    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)

    # Seed K-layer pages via synthesize so distill has docs to batch over.
    synth_script = {
        "sources/notes/karpathy-wiki.md": (
            '<page path="wiki/concepts/karpathy.md" type="concept">\n'
            "---\ntags: [karpathy]\n---\n\n"
            "# Karpathy\n\nDeterministic scoping matters.\n"
            "</page>"
        ),
        "sources/notes/dikw.md": (
            '<page path="wiki/concepts/dikw.md" type="concept">\n'
            "---\ntags: [dikw]\n---\n\n"
            "# DIKW\n\nFour layers stacked.\n"
            "</page>"
        ),
        "sources/notes/retrieval.md": (
            '<page path="wiki/concepts/retrieval.md" type="concept">\n'
            "---\ntags: [retrieval]\n---\n\n"
            "# Retrieval\n\nRRF fuses BM25 with dense.\n"
            "</page>"
        ),
    }
    await api.synthesize(
        wiki_with_fixtures,
        llm=_ScriptedSynthLLM(synth_script),
        embedder=embedder,
    )

    reporter = ListReporter()
    # FakeLLM returns "STUB: wired up." — parse_distill_response will
    # produce zero candidates per batch, but the per-batch progress event
    # still fires regardless of LLM output quality.
    llm = FakeLLM()
    report = await api.distill(
        wiki_with_fixtures, llm=llm, pages_per_call=1, reporter=reporter
    )

    distill_events = [
        e for e in reporter.events
        if e.kind == "progress" and e.payload["phase"] == "distill"
    ]
    # pages_per_call=1 → one event per K-layer page synthesised above.
    assert report.pages_read >= 3
    assert len(distill_events) == report.pages_read
    # detail dict carries the per-batch counters server clients render.
    for ev in distill_events:
        detail = ev.payload["detail"]
        assert detail is not None
        assert {"pages", "candidates_added", "rejected"} <= set(detail.keys())


@pytest.mark.asyncio
async def test_cancellation_aborts_ingest(wiki_with_fixtures: Path) -> None:
    reporter = ListReporter()
    reporter.cancel_token().cancel()  # pre-seed: bail on first checkpoint
    with pytest.raises(asyncio.CancelledError):
        await api.ingest(
            wiki_with_fixtures, embedder=FakeEmbeddings(), reporter=reporter
        )


