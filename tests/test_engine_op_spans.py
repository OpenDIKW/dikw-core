"""PR2b: engine op-level spans (``dikw.ingest`` / ``dikw.synth`` / ``dikw.retrieve``
/ ``dikw.lint.*``).

These pin that the facade verbs emit their op-level span (root for non-server
callers; the ``dikw.layer`` / ``dikw.op`` dimensions for server callers) and that
the retrieval-leg spans nest UNDER ``dikw.retrieve`` via the active OTel context.
The span helpers themselves (``op_span`` / ``traced_op``) are unit-tested in
``test_telemetry_tracing.py``; this is the end-to-end wiring check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dikw_core import api, telemetry

from .fakes import FakeEmbeddings, FakeLLM, init_test_base


def _spans_named(exporter: Any, name: str) -> list[Any]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _one(exporter: Any, name: str) -> Any:
    spans = _spans_named(exporter, name)
    assert len(spans) == 1, f"expected one {name!r}, got {[s.name for s in spans]}"
    return spans[0]


@pytest.mark.asyncio
async def test_ingest_emits_dikw_ingest_span(tmp_path: Path, span_exporter: Any) -> None:
    wiki = tmp_path / "base"
    init_test_base(wiki)
    src = wiki / "sources" / "note.md"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        "---\ntitle: Note\n---\n# Note\n\nThe DIKW pyramid layers data to wisdom.\n",
        encoding="utf-8",
    )

    await api.ingest(wiki, embedder=FakeEmbeddings())

    span = _one(span_exporter, "dikw.ingest")
    assert span.attributes[telemetry.DIKW_LAYER] == "data"
    assert span.attributes[telemetry.DIKW_OP] == "ingest"


@pytest.mark.asyncio
async def test_retrieve_emits_facade_span_with_legs_nested(
    tmp_path: Path, span_exporter: Any
) -> None:
    wiki = tmp_path / "base"
    init_test_base(wiki)
    src = wiki / "sources" / "note.md"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        "---\ntitle: DIKW\n---\n# DIKW\n\nData, information, knowledge, wisdom.\n",
        encoding="utf-8",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    result = await api.retrieve(
        "DIKW pyramid", wiki, limit=4, embedder=FakeEmbeddings()
    )

    span = _one(span_exporter, "dikw.retrieve")
    assert span.attributes[telemetry.DIKW_LAYER] == "info"
    assert span.attributes[telemetry.DIKW_OP] == "retrieve"
    assert span.attributes[telemetry.DIKW_RETRIEVE_LIMIT] == 4
    assert span.attributes[telemetry.DIKW_RETRIEVE_HIT_COUNT] == len(result.chunks)

    leg_spans = _spans_named(span_exporter, "dikw.retrieve.leg")
    assert leg_spans, "expected at least one retrieval-leg span"
    # Every leg span nests under THIS retrieve span (context copied across the
    # leg's create_task boundary).
    for leg in leg_spans:
        assert leg.parent is not None
        assert leg.parent.span_id == span.context.span_id
        assert telemetry.DIKW_LEG_HIT_COUNT in leg.attributes
    # Pin the leg set THROUGH the facade (not just one layer down) so a leg-span
    # regression that only manifests via api.retrieve is caught.
    legs = {leg.attributes[telemetry.DIKW_RETRIEVAL_LEG] for leg in leg_spans}
    assert legs == {"bm25", "vector", "graph"}


@pytest.mark.asyncio
async def test_synthesize_emits_dikw_synth_span(
    tmp_path: Path, span_exporter: Any
) -> None:
    wiki = tmp_path / "base"
    init_test_base(wiki)
    # No sources → no LLM call (zero active SOURCE docs); the facade span still
    # opens, verifying the @traced_op decoration + its attribute strings.
    await api.synthesize(wiki, llm=FakeLLM())

    span = _one(span_exporter, "dikw.synth")
    assert span.attributes[telemetry.DIKW_LAYER] == "knowledge"
    assert span.attributes[telemetry.DIKW_OP] == "synth"


@pytest.mark.asyncio
async def test_lint_apply_emits_dikw_lint_apply_span(
    tmp_path: Path, span_exporter: Any
) -> None:
    wiki = tmp_path / "base"
    init_test_base(wiki)
    # Empty base → lint_propose yields an empty report → lint_apply does no
    # mutation, but the facade span still opens.
    report = await api.lint_propose(wiki)
    await api.lint_apply(wiki, proposal_report=report)

    span = _one(span_exporter, "dikw.lint.apply")
    assert span.attributes[telemetry.DIKW_LAYER] == "knowledge"
    assert span.attributes[telemetry.DIKW_OP] == "lint.apply"


@pytest.mark.asyncio
async def test_lint_propose_emits_dikw_lint_propose_span(
    tmp_path: Path, span_exporter: Any
) -> None:
    wiki = tmp_path / "base"
    init_test_base(wiki)

    await api.lint_propose(wiki)

    span = _one(span_exporter, "dikw.lint.propose")
    assert span.attributes[telemetry.DIKW_LAYER] == "knowledge"
    assert span.attributes[telemetry.DIKW_OP] == "lint.propose"
