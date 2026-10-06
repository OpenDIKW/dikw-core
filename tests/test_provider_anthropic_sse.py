"""Replay MiniMax's empty SSE turn through the real installed Anthropic SDK."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from anthropic import AsyncAnthropic

from dikw_core import api
from dikw_core.config import dump_config_yaml, load_config
from dikw_core.domains.knowledge.synthesize import SynthesisError, parse_synthesis_response
from dikw_core.progress import CancelToken
from dikw_core.providers.anthropic_compat import AnthropicCompatLLM
from dikw_core.providers.base import TransientProviderError

from .fakes import FakeEmbeddings, init_test_base
from .test_synth_provider_error_repro import _cfg, _three_groups


def _start() -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_fixture", "type": "message", "role": "assistant",
            "model": "MiniMax-M3", "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {
                "input_tokens": 31, "output_tokens": 0,
                "cache_creation_input_tokens": 7, "cache_read_input_tokens": 11,
            },
        },
    }


def _end(reason: str = "end_turn") -> list[dict[str, Any]]:
    return [
        {"type": "message_delta", "delta": {
            "stop_reason": reason, "stop_sequence": None,
        }, "usage": {"output_tokens": 0}},
        {"type": "message_stop"},
    ]


@asynccontextmanager
async def _provider(
    events: list[dict[str, Any]], *, responses: list[list[dict[str, Any]]] | None = None,
) -> AsyncIterator[AnthropicCompatLLM]:
    scripts = iter(responses) if responses is not None else None

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
        script = next(scripts) if scripts is not None else events
        body = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in script
        ).encode()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    client = AsyncAnthropic(
        api_key="sk-test", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    llm = AnthropicCompatLLM(api_key_env="UNUSED", api_key="sk-test")
    llm._client_cache = client
    try:
        yield llm
    finally:
        await client.close()


async def test_minimax_empty_turn_with_orphan_stop_is_a_legal_empty_answer() -> None:
    # Captured HTTP 200 shape from released Core 0.6.9; no start/delta for index 0.
    events = [_start(), {"type": "ping"}, {"type": "content_block_stop", "index": 0}, *_end()]
    async with _provider(events) as llm:
        response = await llm.complete(system="s", user="u", model="MiniMax-M3")
    assert response.text == ""
    assert response.finish_reason == "end_turn"
    assert response.usage == {
        "input_tokens": 31, "output_tokens": 0,
        "cache_creation_input_tokens": 7, "cache_read_input_tokens": 11,
    }


def _block(index: int = 0, text: str = "hello") -> list[dict[str, Any]]:
    return [
        {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": index},
    ]


@pytest.mark.parametrize("blocks", [[], _block(text="")])
async def test_standard_empty_answers_still_succeed(blocks: list[dict[str, Any]]) -> None:
    async with _provider([_start(), *blocks, *_end()]) as llm:
        response = await llm.complete(system="s", user="u", model="m")
    assert response.text == ""
    assert response.finish_reason == "end_turn"


async def test_text_blocks_and_thinking_preserve_text_tokens_and_cumulative_usage() -> None:
    thinking = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hidden"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 0},
    ]
    first = _block(1, "hello")
    first[0]["content_block"]["text"] = "prefix: "
    end = _end("stop_sequence")
    end[0]["usage"] = {"output_tokens": 9, "input_tokens": 32, "cache_read_input_tokens": 12}
    events = [_start(), *thinking, *first, *_block(2, " world"), *end]
    async with _provider(events) as llm:
        result = [e async for e in llm.complete_stream(system="s", user="u", model="m")]
    assert [e.delta for e in result if e.type == "token"] == ["hello", " world"]
    assert result[-1].text == "prefix: hello world"
    assert result[-1].finish_reason == "stop_sequence"
    assert result[-1].usage == {
        "input_tokens": 32, "output_tokens": 9,
        "cache_creation_input_tokens": 7, "cache_read_input_tokens": 12,
    }
    assert [e.type for e in result].count("done") == 1


@pytest.mark.parametrize("text", ["", '<page category="concept" slug="x"># X</page>'])
async def test_budget_cutoffs_are_not_successful_zero_page_synth(text: str) -> None:
    async with _provider([_start(), *_block(text=text), *_end("max_tokens")]) as llm:
        response = await llm.complete(system="s", user="u", model="m")
    assert response.finish_reason == "max_tokens"
    with pytest.raises(SynthesisError):
        parse_synthesis_response(response.text, source_path="sources/x.md", finish_reason=response.finish_reason)


@pytest.mark.parametrize("events", [
    [],
    _end(),
    [_start(), _start(), *_end()],
    [_start(), {"type": "content_block_stop", "index": 1}, *_end()],
    [_start(), {"type": "content_block_stop", "index": -1}, *_end()],
    [_start(), *_block(index=1), *_end()],
    [_start(), *_block()[1:], *_end()],
    [_start(), *_block(), {"type": "content_block_stop", "index": 0}, *_end()],
    [_start(), {"type": "content_block_stop", "index": 0}, *_block(), *_end()],
    [_start(), {"type": "content_block_stop", "index": 0}, {"type": "content_block_stop", "index": 0}, *_end()],
    [_start(), {"type": "content_block_stop", "index": 0}, *_end("max_tokens")],
    [_start(), {"type": "content_block_stop", "index": 0}, *_end("tool_use")],
    [_start(), *_block()[:-1], *_end()],
    [_start(), *_block(), *_end()[:-1]],
    [_start(), *_block(), {"type": "message_stop"}],
])
async def test_other_malformed_streams_raise_controlled_retryable_errors(events: list[dict[str, Any]]) -> None:
    async with _provider(events) as llm:
        seen = []
        with pytest.raises(TransientProviderError, match="malformed stream"):
            async for event in llm.complete_stream(system="s", user="u", model="m"):
                seen.append(event)
    assert all(event.type != "done" for event in seen)


async def test_synth_retries_malformed_group_and_accepts_real_empty_turn() -> None:
    body, chunks = _three_groups()
    valid = [_start(), *_block(text='<page category="concept" slug="x"># X</page>'), *_end()]
    malformed = [_start(), *_block()[1:], *_end()]
    empty = [_start(), {"type": "content_block_stop", "index": 0}, *_end()]
    async with _provider([], responses=[valid, malformed, empty, valid]) as llm:
        result = await api._synth_pages_from_source(
            llm=llm, template="{source_body}", cfg=_cfg(), source_path="sources/multi.md",
            source_body=body, chunks=chunks, cancel=CancelToken(),
        )
    assert result.groups_processed == 3
    assert result.parse_errors == 0
    assert len(result.pages) == 2


@pytest.mark.parametrize("events,success", [
    ([_start(), {"type": "content_block_stop", "index": 0}, *_end()], True),
    ([_start(), *_block()[1:], *_end()], False),
    ([_start(), *_block(text=""), *_end("max_tokens")], False),
])
async def test_source_done_marker_requires_a_complete_response(
    tmp_path: Path, events: list[dict[str, Any]], success: bool,
) -> None:
    init_test_base(tmp_path)
    (tmp_path / "sources" / "sample.md").write_text("# Sample\n\nA source to synthesise.\n", encoding="utf-8")
    cfg_path = tmp_path / "dikw.yml"
    cfg = load_config(cfg_path)
    cfg.synth.provider_error_retry_backoff_seconds = 0
    cfg_path.write_text(dump_config_yaml(cfg), encoding="utf-8")
    await api.ingest(tmp_path, embedder=FakeEmbeddings())
    async with _provider(events) as llm:
        first = await api.synthesize(tmp_path, llm=llm)
        second = await api.synthesize(tmp_path, llm=llm)
    assert first.sources_processed == 1
    assert first.errors == (0 if success else 1)
    assert first.created == 0
    # Normal-stop empty answers complete the source; malformed/budget-starved
    # streams remain candidates on the next default run.
    assert second.sources_processed == (0 if success else 1)


async def test_initial_text_content_and_usage_only_final_delta_are_preserved() -> None:
    start = _start()
    start["message"]["content"] = [{"type": "text", "text": "initial"}]
    events = [start, {"type": "content_block_stop", "index": 0}, *_end()[:-1],
              {"type": "message_delta", "delta": {"stop_reason": None, "stop_sequence": None},
               "usage": {"output_tokens": 2}}, {"type": "message_stop"}]
    async with _provider(events) as llm:
        response = await llm.complete(system="s", user="u", model="m")
    assert response.text == "initial"
    assert response.finish_reason == "end_turn"
    assert response.usage["output_tokens"] == 2
