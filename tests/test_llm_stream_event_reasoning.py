"""Protocol-level contract for ``LLMStreamEvent``.

The ``type`` Literal carries three event kinds today: ``token`` (text delta),
``reasoning`` (thinking-process delta — only emitted by reasoning-capable
providers like the OpenAI Codex Responses API), and ``done`` (terminal). New
providers may emit ``reasoning`` interleaved with ``token`` events; consumers
that only handle ``token`` / ``done`` must tolerate ``reasoning`` as an
unrecognized type and fall through (matching ``api.py``'s if/elif dispatch).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dikw_core.providers.base import LLMStreamEvent


def test_llm_stream_event_accepts_reasoning_type() -> None:
    ev = LLMStreamEvent(type="reasoning", delta="thinking…")
    assert ev.type == "reasoning"
    assert ev.delta == "thinking…"


def test_llm_stream_event_token_type_still_valid() -> None:
    ev = LLMStreamEvent(type="token", delta="hello")
    assert ev.type == "token"


def test_llm_stream_event_done_type_still_valid() -> None:
    ev = LLMStreamEvent(
        type="done",
        text="hello world",
        finish_reason="stop",
        usage={"input_tokens": 1, "output_tokens": 2},
    )
    assert ev.type == "done"
    assert ev.text == "hello world"


def test_llm_stream_event_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        LLMStreamEvent(type="bogus", delta="x")  # type: ignore[arg-type]
