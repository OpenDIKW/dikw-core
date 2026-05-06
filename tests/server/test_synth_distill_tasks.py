"""HTTP-level tests for ``POST /v1/synth`` + ``POST /v1/distill``.

Both ops go through the ``TaskManager`` plumbing exercised in
``test_ingest_task.py``; this file focuses on the synth/distill specific
event vocabulary + final shape rather than re-testing event tape replay
or cancellation (already covered for ingest).
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest

from dikw_core import api as api_module
from dikw_core.providers import LLMResponse
from dikw_core.server import synth_op as synth_op_module

from ..fakes import FakeEmbeddings, FakeLLM

FIXTURES = Path(__file__).parent.parent / "fixtures" / "notes"


async def _wait_terminal(
    client: httpx.AsyncClient, task_id: str, *, timeout: float = 15.0
) -> dict[str, Any]:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        r = await client.get(f"/v1/tasks/{task_id}")
        if r.status_code == 200 and r.json()["status"] in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            return r.json()
        await asyncio.sleep(0.05)
    raise AssertionError(f"task {task_id} never reached a terminal state")


def _patch_synth_factories(
    monkeypatch: pytest.MonkeyPatch, *, llm: FakeLLM
) -> None:
    monkeypatch.setattr(synth_op_module, "build_llm", lambda _cfg, **_kw: llm)
    monkeypatch.setattr(
        synth_op_module, "build_embedder", lambda _cfg: FakeEmbeddings()
    )


class _ScriptedSynthLLM:
    """Returns one canned ``<page>`` block per source path, matched by
    substring against the user prompt body."""

    def __init__(self, by_source: dict[str, str]) -> None:
        self._by_source = by_source

    async def complete(
        self, *, system: str, user: str, model: str, **_: Any
    ) -> LLMResponse:
        for src_path, body in self._by_source.items():
            if src_path in user:
                return LLMResponse(text=body, finish_reason="end_turn")
        raise AssertionError(f"no scripted page for prompt: {user[:200]}")

    def complete_stream(self, **_: Any) -> Any:
        raise NotImplementedError


# ---- synth -------------------------------------------------------------


@pytest.mark.asyncio
async def test_synth_task_emits_per_source_progress_and_final_report(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Seed three source markdown files + ingest them so synth has
    # documents to process.
    dest = wiki_root / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    await api_module.ingest(wiki_root, embedder=FakeEmbeddings())

    script = {
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
    _patch_synth_factories(
        monkeypatch, llm=FakeLLM()  # placeholder, overridden below
    )
    # Override build_llm to return the scripted stub for synth.
    monkeypatch.setattr(
        synth_op_module, "build_llm", lambda _cfg, **_kw: _ScriptedSynthLLM(script)
    )

    submit = await server_client.post(
        "/v1/synth", json={"force_all": True, "no_embed": False}
    )
    assert submit.status_code == 200, submit.text
    handle = submit.json()
    assert handle["op"] == "synth"
    task_id = handle["task_id"]

    row = await _wait_terminal(server_client, task_id)
    assert row["status"] == "succeeded", row

    result = (await server_client.get(f"/v1/tasks/{task_id}/result")).json()[
        "result"
    ]
    # SynthReport fields land verbatim in the final result.
    assert result["candidates"] == 3
    assert result["created"] == 3
    assert result["errors"] == 0

    # Event tape carries one progress event per source, all phase=synth.
    async with server_client.stream(
        "GET", f"/v1/tasks/{task_id}/events"
    ) as resp:
        events = [
            json.loads(line)
            for line in [ln async for ln in resp.aiter_lines()]
            if line.strip()
        ]
    synth_progress = [
        e for e in events if e["type"] == "progress" and e["phase"] == "synth"
    ]
    assert len(synth_progress) == 3
    assert {e["detail"]["outcome"] for e in synth_progress} == {"created"}
    assert events[-1]["type"] == "final" and events[-1]["status"] == "succeeded"


# ---- distill -----------------------------------------------------------


@pytest.mark.asyncio
async def test_distill_task_emits_per_batch_progress(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Seed sources + ingest + synthesise K-layer pages so distill has
    # something to batch over.
    dest = wiki_root / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    embedder = FakeEmbeddings()
    await api_module.ingest(wiki_root, embedder=embedder)

    synth_script = {
        "sources/notes/karpathy-wiki.md": (
            '<page path="wiki/concepts/karpathy.md" type="concept">\n'
            "---\ntags: [karpathy]\n---\n\n"
            "# Karpathy\n\nScoping is deterministic.\n"
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
    await api_module.synthesize(
        wiki_root,
        llm=_ScriptedSynthLLM(synth_script),
        embedder=embedder,
    )

    # Distill LLM: FakeLLM's "STUB: wired up." won't parse into
    # candidates, but the per-batch progress events still fire so we can
    # verify the task wrapper.
    monkeypatch.setattr(
        synth_op_module, "build_llm", lambda _cfg, **_kw: FakeLLM()
    )

    submit = await server_client.post(
        "/v1/distill", json={"pages_per_call": 1}
    )
    assert submit.status_code == 200
    task_id = submit.json()["task_id"]
    row = await _wait_terminal(server_client, task_id)
    assert row["status"] == "succeeded", row

    result = (await server_client.get(f"/v1/tasks/{task_id}/result")).json()[
        "result"
    ]
    assert result["pages_read"] >= 3
    assert result["candidates_added"] == 0  # FakeLLM body doesn't parse
    assert result["rejected"] == 0

    async with server_client.stream(
        "GET", f"/v1/tasks/{task_id}/events"
    ) as resp:
        events = [
            json.loads(line)
            for line in [ln async for ln in resp.aiter_lines()]
            if line.strip()
        ]
    distill_progress = [
        e for e in events if e["type"] == "progress" and e["phase"] == "distill"
    ]
    assert len(distill_progress) == result["pages_read"]
    assert events[-1]["type"] == "final"


@pytest.mark.asyncio
async def test_distill_rejects_invalid_pages_per_call(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post(
        "/v1/distill", json={"pages_per_call": 0}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_request"
