"""Per-group / per-LLM-call progress events emitted by Stage A synth fan-out.

Without these events the client UI freezes on the source counter (e.g.
``synth 2/43``) for the entire duration of a multi-group source — large
markdown books take minutes per LLM call and the user can't tell the
process apart from a deadlock. The fan-out helper ``_synth_pages_from_source``
must emit a ``synth_llm`` ``calling`` / ``returned`` event pair per group
(plus an ``error`` event when the LLM call or parser fails) so a server
task wrapper can fan them out to NDJSON subscribers.

The event tape and detail-dict contract are pinned here so a future
refactor of the group loop can't silently regress observability.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from dikw_core import api
from dikw_core.config import DikwConfig
from dikw_core.progress import CancelToken
from dikw_core.providers import LLMResponse, LLMStreamEvent, ToolSpec
from dikw_core.schemas import ChunkRecord

from .fakes import make_provider_cfg
from .test_progress_reporter import ListReporter


def _build_cfg(target_tokens: int = 40) -> DikwConfig:
    """Tiny ``target_tokens_per_group`` so even compact bodies split."""
    cfg = DikwConfig(provider=make_provider_cfg())
    cfg.synth.target_tokens_per_group = target_tokens
    return cfg


def _three_chunk_body() -> tuple[str, list[ChunkRecord]]:
    """Three H1-led sections — each forces a group break, yielding 3 groups."""
    sections = [
        "# Section one\n\nAlpha alpha alpha alpha alpha.\n",
        "# Section two\n\nBravo bravo bravo bravo bravo.\n",
        "# Section three\n\nCharlie charlie charlie charlie charlie.\n",
    ]
    body = "".join(sections)
    chunks: list[ChunkRecord] = []
    cursor = 0
    for seq, text in enumerate(sections):
        end = cursor + len(text)
        chunks.append(
            ChunkRecord(
                doc_id="D::sources/multi.md",
                seq=seq,
                start=cursor,
                end=end,
                text=text,
            )
        )
        cursor = end
    return body, chunks


@dataclass
class _ScriptedLLM:
    """Returns a fixed text on every ``complete`` call; tracks call count."""

    response_text: str = "<page path=\"wiki/x.md\" type=\"concept\">\n# X\n\nbody\n</page>"
    calls: int = 0

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        _ = (system, user, model, max_tokens, temperature, tools)
        self.calls += 1
        return LLMResponse(text=self.response_text, finish_reason="end_turn")

    def complete_stream(
        self, **_: Any
    ) -> AsyncIterator[LLMStreamEvent]:
        raise NotImplementedError


@dataclass
class _FailingLLM:
    """Raises ``SynthesisError`` from inside ``complete`` to drive the
    error-event branch. The synth loop catches and continues; the test
    asserts the error event was reported before the next group started."""

    calls: int = 0

    async def complete(self, **_: Any) -> LLMResponse:
        self.calls += 1
        # Simulate a parser failure by returning text that fails to parse;
        # SynthesisError surfaces from parse_synthesis_response, not from
        # the LLM call itself, so we return junk text and let the parser
        # raise. This matches the production failure mode.
        return LLMResponse(text="not a <page> block", finish_reason="end_turn")


# ---- per-group event tape ------------------------------------------------


@pytest.mark.asyncio
async def test_emits_calling_and_returned_event_per_group() -> None:
    """≥2 groups → reporter gets matching ``calling`` / ``returned`` pairs."""
    body, chunks = _three_chunk_body()
    cfg = _build_cfg()
    llm = _ScriptedLLM()
    reporter = ListReporter()

    await api._synth_pages_from_source(
        llm=llm,
        template="src={source_path} body={source_body} idx={group_index}/"
        "{group_total} headings={group_outline} max={max_pages} "
        "types={allowed_types}",
        cfg=cfg,
        source_path="sources/multi.md",
        source_body=body,
        chunks=chunks,
        cancel=CancelToken(),
        reporter=reporter,
    )

    llm_events = [
        e for e in reporter.events
        if e.kind == "progress" and e.payload["phase"] == "synth_llm"
    ]
    statuses = [e.payload["detail"]["status"] for e in llm_events]
    calling = statuses.count("calling")
    returned = statuses.count("returned")
    assert calling == llm.calls, "one `calling` event per LLM call"
    assert returned == llm.calls, "one `returned` event per successful LLM call"
    assert calling >= 2, (
        f"expected ≥2 groups (forced via tiny target_tokens), got {calling}"
    )
    # current/total monotonically advances and matches LLM call count.
    currents = [
        e.payload["current"] for e in llm_events
        if e.payload["detail"]["status"] == "calling"
    ]
    assert currents == sorted(currents)
    totals = {e.payload["total"] for e in llm_events}
    assert totals == {llm.calls}, "total field carries group count on every event"


@pytest.mark.asyncio
async def test_synth_llm_event_payload_contract() -> None:
    """Lock the detail-dict shape so server NDJSON consumers can rely on
    a stable field set. Adding fields is fine; renaming or dropping is not."""
    body, chunks = _three_chunk_body()
    cfg = _build_cfg()
    llm = _ScriptedLLM()
    reporter = ListReporter()

    await api._synth_pages_from_source(
        llm=llm,
        template="x={source_body} {group_index}/{group_total} "
        "{group_outline} {max_pages} {allowed_types} {source_path}",
        cfg=cfg,
        source_path="sources/multi.md",
        source_body=body,
        chunks=chunks,
        cancel=CancelToken(),
        reporter=reporter,
    )

    llm_events = [
        e for e in reporter.events
        if e.kind == "progress" and e.payload["phase"] == "synth_llm"
    ]
    calling = next(
        e for e in llm_events if e.payload["detail"]["status"] == "calling"
    )
    returned = next(
        e for e in llm_events if e.payload["detail"]["status"] == "returned"
    )

    calling_keys = set(calling.payload["detail"].keys())
    assert {
        "source_path", "model", "status", "section_count", "approx_tokens",
    } <= calling_keys, (
        f"calling event missing required fields; got {calling_keys}"
    )
    assert calling.payload["detail"]["source_path"] == "sources/multi.md"
    assert calling.payload["detail"]["status"] == "calling"
    assert calling.payload["detail"]["model"] == cfg.provider.llm_model
    assert calling.payload["detail"]["section_count"] >= 1
    assert calling.payload["detail"]["approx_tokens"] >= 1

    returned_keys = set(returned.payload["detail"].keys())
    assert {
        "source_path", "status", "response_chars",
    } <= returned_keys, (
        f"returned event missing required fields; got {returned_keys}"
    )
    assert returned.payload["detail"]["status"] == "returned"
    assert returned.payload["detail"]["response_chars"] == len(llm.response_text)


@pytest.mark.asyncio
async def test_emits_error_event_when_parse_fails() -> None:
    """A parser failure must surface a ``status="error"`` event before
    the loop moves on, so operators see *which* group blew up rather
    than only an aggregate ``parse_errors`` count post-hoc."""
    body, chunks = _three_chunk_body()
    cfg = _build_cfg()
    llm = _FailingLLM()
    reporter = ListReporter()

    outcome = await api._synth_pages_from_source(
        llm=llm,
        template="x={source_body} {group_index}/{group_total} "
        "{group_outline} {max_pages} {allowed_types} {source_path}",
        cfg=cfg,
        source_path="sources/multi.md",
        source_body=body,
        chunks=chunks,
        cancel=CancelToken(),
        reporter=reporter,
    )

    # Parser raised — synth loop logged it and continued.
    assert outcome.parse_errors >= 1

    error_events = [
        e for e in reporter.events
        if e.kind == "progress"
        and e.payload["phase"] == "synth_llm"
        and e.payload["detail"].get("status") == "error"
    ]
    assert error_events, "expected at least one synth_llm error event"
    detail = error_events[0].payload["detail"]
    assert detail["error_kind"] in {"SynthesisError", "SynthesisPartialError"}
    assert isinstance(detail["error_msg"], str) and detail["error_msg"]
    assert detail["source_path"] == "sources/multi.md"


@pytest.mark.asyncio
async def test_no_reporter_keeps_legacy_silent_path() -> None:
    """Backwards-compat: ``reporter=None`` (or omitted) must not raise.
    Callers that don't care about events should keep working unchanged."""
    body, chunks = _three_chunk_body()
    cfg = _build_cfg()
    llm = _ScriptedLLM()

    outcome = await api._synth_pages_from_source(
        llm=llm,
        template="x={source_body} {group_index}/{group_total} "
        "{group_outline} {max_pages} {allowed_types} {source_path}",
        cfg=cfg,
        source_path="sources/multi.md",
        source_body=body,
        chunks=chunks,
        cancel=CancelToken(),
    )
    assert outcome.groups_processed >= 2
