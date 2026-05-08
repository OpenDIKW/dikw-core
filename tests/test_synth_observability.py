"""``_synth_pages_from_source`` per-group ``synth_llm`` event contract.

Pins the event tape (calling/returned/error per group) and the detail
dict shape — server NDJSON consumers branch on both.
"""

from __future__ import annotations

import pytest

from dikw_core import api
from dikw_core.config import DikwConfig
from dikw_core.progress import CancelToken
from dikw_core.schemas import ChunkRecord

from .fakes import FakeLLM, make_provider_cfg
from .test_progress_reporter import ListReporter

_VALID_PAGE = '<page path="wiki/x.md" type="concept">\n# X\n\nbody\n</page>'
_UNPARSEABLE = "not a <page> block"
# One closed page + one truncated opener → SynthesisPartialError.
_PARTIAL_PAGE = (
    '<page path="wiki/x.md" type="concept">\n# X\n\nbody\n</page>\n'
    '<page path="wiki/y.md" type="concept">\n# Y'
)


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


# ---- per-group event tape ------------------------------------------------


_TEMPLATE = (
    "src={source_path} body={source_body} idx={group_index}/"
    "{group_total} headings={group_outline} max={max_pages} "
    "types={allowed_types}"
)


@pytest.mark.asyncio
async def test_emits_calling_and_returned_event_per_group() -> None:
    body, chunks = _three_chunk_body()
    cfg = _build_cfg()
    reporter = ListReporter()

    outcome = await api._synth_pages_from_source(
        llm=FakeLLM(response_text=_VALID_PAGE),
        template=_TEMPLATE,
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
    assert calling == outcome.groups_processed
    assert returned == outcome.groups_processed
    assert calling >= 2, (
        f"expected ≥2 groups (forced via tiny target_tokens), got {calling}"
    )
    # `calling` events report COMPLETED groups (group_pos - 1) so a
    # progress widget treating current as "completed" doesn't flip to
    # 100% the moment a long LLM call starts. `returned` events bump
    # current to group_pos. The semantic group identifier stays in
    # detail.group_pos.
    calling_currents = [
        e.payload["current"] for e in llm_events
        if e.payload["detail"]["status"] == "calling"
    ]
    returned_currents = [
        e.payload["current"] for e in llm_events
        if e.payload["detail"]["status"] == "returned"
    ]
    assert calling_currents == list(range(0, outcome.groups_processed))
    assert returned_currents == list(range(1, outcome.groups_processed + 1))
    calling_group_pos = [
        e.payload["detail"]["group_pos"] for e in llm_events
        if e.payload["detail"]["status"] == "calling"
    ]
    assert calling_group_pos == list(range(1, outcome.groups_processed + 1))
    totals = {e.payload["total"] for e in llm_events}
    assert totals == {outcome.groups_processed}


@pytest.mark.asyncio
async def test_synth_llm_event_payload_contract() -> None:
    body, chunks = _three_chunk_body()
    cfg = _build_cfg()
    reporter = ListReporter()

    await api._synth_pages_from_source(
        llm=FakeLLM(response_text=_VALID_PAGE),
        template=_TEMPLATE,
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

    assert {
        "source_path", "group_pos", "model", "status",
        "section_count", "approx_tokens",
    } <= set(calling.payload["detail"]), calling.payload["detail"]
    assert calling.payload["detail"]["source_path"] == "sources/multi.md"
    assert calling.payload["detail"]["group_pos"] == 1
    assert calling.payload["detail"]["model"] == cfg.provider.llm_model
    assert calling.payload["detail"]["section_count"] >= 1
    assert calling.payload["detail"]["approx_tokens"] >= 1

    assert {
        "source_path", "group_pos", "status", "response_chars",
    } <= set(returned.payload["detail"]), returned.payload["detail"]
    assert returned.payload["detail"]["response_chars"] == len(_VALID_PAGE)


@pytest.mark.asyncio
async def test_emits_error_event_when_parse_fails() -> None:
    body, chunks = _three_chunk_body()
    cfg = _build_cfg()
    reporter = ListReporter()

    outcome = await api._synth_pages_from_source(
        llm=FakeLLM(response_text=_UNPARSEABLE),
        template=_TEMPLATE,
        cfg=cfg,
        source_path="sources/multi.md",
        source_body=body,
        chunks=chunks,
        cancel=CancelToken(),
        reporter=reporter,
    )

    assert outcome.parse_errors >= 1

    error_events = [
        e for e in reporter.events
        if e.kind == "progress"
        and e.payload["phase"] == "synth_llm"
        and e.payload["detail"].get("status") == "error"
    ]
    assert error_events
    detail = error_events[0].payload["detail"]
    assert detail["error_kind"] in {"SynthesisError", "SynthesisPartialError"}
    assert detail["error_msg"]
    assert detail["source_path"] == "sources/multi.md"


@pytest.mark.asyncio
async def test_no_reporter_keeps_legacy_silent_path() -> None:
    """``reporter=None`` (or omitted) must not raise — pre-reporter callers
    keep working unchanged."""
    body, chunks = _three_chunk_body()
    cfg = _build_cfg()

    outcome = await api._synth_pages_from_source(
        llm=FakeLLM(response_text=_VALID_PAGE),
        template=_TEMPLATE,
        cfg=cfg,
        source_path="sources/multi.md",
        source_body=body,
        chunks=chunks,
        cancel=CancelToken(),
    )
    assert outcome.groups_processed >= 2


# ---- per-group logger calls ---------------------------------------------


@pytest.mark.asyncio
async def test_synth_logs_per_group_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    caplog.set_level(logging.DEBUG)
    body, chunks = _three_chunk_body()
    cfg = _build_cfg()

    outcome = await api._synth_pages_from_source(
        llm=FakeLLM(response_text=_VALID_PAGE),
        template=_TEMPLATE,
        cfg=cfg,
        source_path="sources/multi.md",
        source_body=body,
        chunks=chunks,
        cancel=CancelToken(),
    )

    debug_msgs = [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.DEBUG and r.name == "dikw_core.api"
    ]
    calling = [m for m in debug_msgs if "calling" in m and "group" in m]
    returned = [m for m in debug_msgs if "returned" in m and "group" in m]
    assert len(calling) == outcome.groups_processed
    assert len(returned) == outcome.groups_processed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "marker"),
    [
        (_UNPARSEABLE, "FAILED"),  # SynthesisError branch
        (_PARTIAL_PAGE, "PARTIAL"),  # SynthesisPartialError branch
    ],
    ids=["full-failure", "partial-failure"],
)
async def test_synth_logs_group_failure_at_warning(
    caplog: pytest.LogCaptureFixture, response: str, marker: str
) -> None:
    """Both parser failure variants surface at WARNING — visible at
    default INFO. SynthesisPartialError is a subclass of SynthesisError
    but carries surviving pages; without its own WARNING log, operators
    miss truncated/malformed pages while the loop silently keeps the
    survivors."""
    import logging

    caplog.set_level(logging.WARNING, logger="dikw_core.api")
    body, chunks = _three_chunk_body()
    cfg = _build_cfg()

    await api._synth_pages_from_source(
        llm=FakeLLM(response_text=response),
        template=_TEMPLATE,
        cfg=cfg,
        source_path="sources/multi.md",
        source_body=body,
        chunks=chunks,
        cancel=CancelToken(),
    )

    warning_msgs = [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "dikw_core.api"
    ]
    assert any("group" in m and marker in m for m in warning_msgs), warning_msgs
