"""Exercise Codex's total-loss safeguard through the real Responses SDK reducer.

OpenAI 2.x raises on terminal ``output=null``; 3.x reconstructs it from
``output_item.done`` events. MockTransport keeps this SDK compatibility seam
offline without replacing AsyncOpenAI or its high-level stream helper.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

from dikw_core.providers.base import TransientProviderError
from dikw_core.providers.openai_codex import OpenAICodexLLM


def _message(text: str) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
    }


def _reasoning() -> dict[str, Any]:
    return {"id": "rs_test", "type": "reasoning", "summary": []}


def _response(output: list[dict[str, Any]] | None, *, status: str) -> dict[str, Any]:
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "model": "gpt-5.5",
        "status": status,
        "output": output,
        "error": None,
        "incomplete_details": None,
        "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
    }


def _sse(
    output: list[dict[str, Any]] | None,
    *,
    deltas: tuple[str, ...] = (),
    message_text: str | None = None,
    reasoning_done: bool = False,
) -> bytes:
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": _response([], status="in_progress")}
    ]
    if reasoning_done:
        events.extend(
            [
                {"type": "response.output_item.added", "output_index": 0, "item": _reasoning()},
                {"type": "response.output_item.done", "output_index": 0, "item": _reasoning()},
            ]
        )
    message_index = 1 if reasoning_done else 0
    if deltas or message_text is not None:
        events.extend(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": message_index,
                    "item": {**_message(""), "status": "in_progress", "content": []},
                },
                {
                    "type": "response.content_part.added",
                    "item_id": "msg_test",
                    "output_index": message_index,
                    "content_index": 0,
                    "part": _message("")["content"][0],
                },
            ]
        )
    for delta in deltas:
        events.append(
            {
                "type": "response.output_text.delta",
                "item_id": "msg_test",
                "output_index": message_index,
                "content_index": 0,
                "delta": delta,
                "logprobs": [],
            }
        )
    if message_text is not None:
        events.append(
            {
                "type": "response.output_item.done",
                "output_index": message_index,
                "item": _message(message_text),
            }
        )
    events.append({"type": "response.completed", "response": _response(output, status="completed")})
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': seq})}\n\n"
        for seq, event in enumerate(events)
    ).encode()


@pytest.fixture
def sdk_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[bytes], OpenAICodexLLM]:
    async def resolve_token(_base: Path) -> str:
        return "test-token"

    monkeypatch.setattr("dikw_core.providers.openai_codex.resolve_access_token", resolve_token)

    def build(sse: bytes) -> OpenAICodexLLM:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/backend-api/codex/responses"
            assert json.loads(request.content)["stream"] is True
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

        def mock_client(
            timeout_seconds: float | None = None,
        ) -> tuple[httpx.Timeout, httpx.AsyncClient]:
            timeout = httpx.Timeout(10.0)
            return timeout, httpx.AsyncClient(
                timeout=timeout, transport=httpx.MockTransport(handler)
            )

        monkeypatch.setattr(
            "dikw_core.providers.openai_codex.build_no_keepalive_async_client", mock_client
        )
        return OpenAICodexLLM(
            base_url="http://stub.test/backend-api/codex", base_root=tmp_path, max_retries=0
        )

    return build


@pytest.mark.parametrize("output", [None, []], ids=["null", "empty-list"])
@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
async def test_no_output_items_or_deltas_raises_transient_error(
    sdk_provider: Callable[[bytes], OpenAICodexLLM],
    output: list[dict[str, Any]] | None,
    streaming: bool,
) -> None:
    provider = sdk_provider(_sse(output))
    emitted: list[str] = []
    with pytest.raises(TransientProviderError, match="zero text deltas"):
        if streaming:
            async for event in provider.complete_stream(system="s", user="u", model="gpt-5.5"):
                emitted.append(event.type)
        else:
            await provider.complete(system="s", user="u", model="gpt-5.5")
    assert emitted == []


@pytest.mark.parametrize("output", [None, []], ids=["null", "empty-list"])
@pytest.mark.parametrize("message_done", [False, True], ids=["deltas-only", "message-done"])
async def test_missing_terminal_output_recovers_deltas(
    sdk_provider: Callable[[bytes], OpenAICodexLLM],
    output: list[dict[str, Any]] | None,
    message_done: bool,
) -> None:
    provider = sdk_provider(
        _sse(output, deltas=("h", "i"), message_text="hi" if message_done else None)
    )
    events = [
        event async for event in provider.complete_stream(system="s", user="u", model="gpt-5.5")
    ]
    assert [event.delta for event in events if event.type == "token"] == ["h", "i"]
    assert [event.type for event in events].count("done") == 1
    assert events[-1].text == "hi"
    legacy_reducer_bug = output is None and int(openai.__version__.split(".")[0]) < 3
    assert events[-1].finish_reason == ("error" if legacy_reducer_bug else "stop")
    assert events[-1].usage == (
        {} if legacy_reducer_bug else {"input_tokens": 5, "output_tokens": 7}
    )


@pytest.mark.parametrize("deltas", [(), ("retracted",)], ids=["no-deltas", "retraction"])
@pytest.mark.parametrize("reasoning_done", [False, True], ids=["message-only", "reasoning-message"])
async def test_explicit_empty_message_remains_authoritative(
    sdk_provider: Callable[[bytes], OpenAICodexLLM],
    deltas: tuple[str, ...],
    reasoning_done: bool,
) -> None:
    provider = sdk_provider(
        _sse([_message("")], deltas=deltas, message_text="", reasoning_done=reasoning_done)
    )
    response = await provider.complete(system="s", user="u", model="gpt-5.5")
    assert response.text == ""
    assert response.finish_reason == "stop"
    assert response.usage == {"input_tokens": 5, "output_tokens": 7}


@pytest.mark.parametrize("output", [None, [_reasoning()]], ids=["null", "reasoning-only"])
@pytest.mark.parametrize("deltas", [(), ("<page>content</page>",)], ids=["no-deltas", "page"])
async def test_reasoning_items_do_not_hide_missing_message(
    sdk_provider: Callable[[bytes], OpenAICodexLLM],
    output: list[dict[str, Any]] | None,
    deltas: tuple[str, ...],
) -> None:
    """3.x can rebuild a nonempty reasoning-only output from a null final.

    That provides no authoritative message: recover streamed page text or
    raise for total loss, rather than letting synth mark the source done.
    """
    provider = sdk_provider(_sse(output, deltas=deltas, reasoning_done=True))
    if not deltas:
        with pytest.raises(TransientProviderError, match="zero text deltas"):
            await provider.complete(system="s", user="u", model="gpt-5.5")
    else:
        response = await provider.complete(system="s", user="u", model="gpt-5.5")
        assert response.text == "<page>content</page>"
        legacy_reducer_bug = output is None and int(openai.__version__.split(".")[0]) < 3
        assert response.finish_reason == ("error" if legacy_reducer_bug else "stop")
        assert response.usage == (
            {} if legacy_reducer_bug else {"input_tokens": 5, "output_tokens": 7}
        )
