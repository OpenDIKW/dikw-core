"""HTTP-level tests for ``POST /v1/retrieve``.

Mirror of ``test_query_stream`` minus the LLM stage. Asserts the wire
shape (``retrieve_started → retrieval_done → final``), validates that
no provider keys are needed (the route never invokes the LLM), and
checks the four new agent-facing fields (``layer``/``start``/``end``/
``text``) plus the ``page_refs`` aggregation land in ``final.result``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import httpx
import pytest

from dikw_core import api as api_module

from ..fakes import FakeEmbeddings

FIXTURES = Path(__file__).parent.parent / "fixtures" / "notes"


@pytest.fixture()
async def ingested_wiki(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> Path:
    """Three fixture markdown files ingested + dense vectors.

    Reuses the test_query_stream fixture pattern so retrieve and query
    exercise an identical corpus: any divergence between the two routes
    must come from route logic, not test setup.
    """
    dest = wiki_root / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    await api_module.ingest(wiki_root, embedder=FakeEmbeddings())
    _ = server_client  # ensure runtime lifespan is up before we ingest
    return wiki_root


def _patch_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``build_embedder`` only — ``build_llm`` deliberately stays
    unpatched so any accidental LLM call would surface as a missing-key
    failure (proving retrieve really skips the LLM stage).
    """

    def _embed_factory(cfg: object, *, dim_override: int | None = None) -> object:
        _ = (cfg, dim_override)
        return FakeEmbeddings()

    monkeypatch.setattr(api_module, "build_embedder", _embed_factory)


@pytest.mark.asyncio
async def test_retrieve_stream_emits_started_retrieval_done_final(
    server_client: httpx.AsyncClient,
    ingested_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_embedder(monkeypatch)

    async with server_client.stream(
        "POST",
        "/v1/retrieve",
        json={"q": "what about scoping?", "limit": 3},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        events = [
            json.loads(line)
            for line in [ln async for ln in resp.aiter_lines()]
            if line.strip()
        ]

    types = [e["type"] for e in events]
    assert types[0] == "retrieve_started"
    assert "retrieval_done" in types
    # Crucially: NO LLM events on the wire.
    assert "llm_token" not in types
    assert "llm_done" not in types
    assert types[-1] == "final"
    final = events[-1]
    assert final["status"] == "succeeded"

    # final.result is a RetrieveResult with chunks + page_refs.
    result = final["result"]
    assert isinstance(result["chunks"], list)
    assert isinstance(result["page_refs"], list)
    assert result["chunks"], "expected at least one chunk for a non-empty corpus"

    # Each chunk carries the four new agent-facing fields.
    for chunk in result["chunks"]:
        for field in ("chunk_id", "doc_id", "path", "layer", "seq", "start", "end", "text", "score"):
            assert field in chunk, f"chunk missing field {field!r}: {chunk}"
        assert chunk["start"] is None or isinstance(chunk["start"], int)
        assert chunk["end"] is None or isinstance(chunk["end"], int)

    # The retrieval_done partial echoes the same hits set that lands in
    # final.result.chunks; clients can pick either event.
    [retrieval_done] = [e for e in events if e["type"] == "retrieval_done"]
    partial_ids = sorted(h["chunk_id"] for h in retrieval_done["hits"])
    final_ids = sorted(c["chunk_id"] for c in result["chunks"])
    assert partial_ids == final_ids


@pytest.mark.asyncio
async def test_retrieve_stream_works_without_llm_provider(
    server_client: httpx.AsyncClient,
    ingested_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrieve must succeed even when ``build_llm`` would raise.

    Proves the route never reaches the LLM init path — an agent that
    hasn't configured an LLM key (or wants to assemble its own answer)
    can still call retrieve productively.
    """

    def _broken_llm_factory(_cfg: object) -> object:
        raise RuntimeError(
            "build_llm should not be called on the retrieve path"
        )

    monkeypatch.setattr(api_module, "build_llm", _broken_llm_factory)
    _patch_embedder(monkeypatch)

    async with server_client.stream(
        "POST",
        "/v1/retrieve",
        json={"q": "scoping?", "limit": 3},
    ) as resp:
        events = [
            json.loads(line)
            for line in [ln async for ln in resp.aiter_lines()]
            if line.strip()
        ]

    final = events[-1]
    assert final["type"] == "final"
    assert final["status"] == "succeeded"
    assert final["result"]["chunks"]


@pytest.mark.asyncio
async def test_retrieve_page_refs_aggregate_chunks_per_path(
    server_client: httpx.AsyncClient,
    ingested_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``page_refs`` must group chunks by path with score = max.

    Spec: each unique chunk path maps to one page_ref; that ref's
    ``score`` is the max chunk score under it; ``hit_chunk_ids`` lists
    every chunk_id from ``chunks`` that landed under the path.
    """
    _patch_embedder(monkeypatch)

    async with server_client.stream(
        "POST",
        "/v1/retrieve",
        json={"q": "wiki", "limit": 10},
    ) as resp:
        events = [
            json.loads(line)
            for line in [ln async for ln in resp.aiter_lines()]
            if line.strip()
        ]

    final = events[-1]
    result = final["result"]

    # Reconstruct expected aggregation client-side and compare.
    by_path: dict[str, dict[str, object]] = {}
    for chunk in result["chunks"]:
        path = chunk["path"]
        if path is None:
            continue
        bucket = by_path.setdefault(
            path, {"score": chunk["score"], "hit_chunk_ids": []}
        )
        if chunk["score"] > bucket["score"]:  # type: ignore[operator]
            bucket["score"] = chunk["score"]
        bucket["hit_chunk_ids"].append(chunk["chunk_id"])  # type: ignore[union-attr]

    expected_paths = set(by_path)
    actual_paths = {ref["path"] for ref in result["page_refs"]}
    assert actual_paths == expected_paths

    for ref in result["page_refs"]:
        bucket = by_path[ref["path"]]
        assert ref["score"] == bucket["score"]
        assert sorted(ref["hit_chunk_ids"]) == sorted(bucket["hit_chunk_ids"])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_retrieve_rejects_empty_q(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post(
        "/v1/retrieve", json={"q": "   ", "limit": 5}
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "bad_request"


@pytest.mark.asyncio
async def test_retrieve_rejects_out_of_range_limit(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post(
        "/v1/retrieve", json={"q": "x", "limit": 0}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_request"

    resp = await server_client.post(
        "/v1/retrieve", json={"q": "x", "limit": 101}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_request"


@pytest.mark.asyncio
async def test_retrieve_stream_keeps_alive_during_slow_engine(
    server_client: httpx.AsyncClient,
    ingested_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow ``api.retrieve`` (e.g. cold embedder + large multimodal
    table) would otherwise leave the wire silent past the client's read
    timeout. The route must inject heartbeat events in the gap."""
    import asyncio as _asyncio

    from dikw_core.server import routes_retrieve

    real_retrieve = api_module.retrieve

    async def _slow_retrieve(*args: object, **kwargs: object) -> object:
        await _asyncio.sleep(0.2)
        return await real_retrieve(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(api_module, "retrieve", _slow_retrieve)
    monkeypatch.setattr(routes_retrieve, "HEARTBEAT_INTERVAL", 0.05)
    _patch_embedder(monkeypatch)

    async with server_client.stream(
        "POST",
        "/v1/retrieve",
        json={"q": "scoping?", "limit": 3},
    ) as resp:
        events = [
            json.loads(line)
            for line in [ln async for ln in resp.aiter_lines()]
            if line.strip()
        ]

    types = [e["type"] for e in events]
    assert "heartbeat" in types, (
        f"slow engine must trigger at least one heartbeat; got types={types}"
    )
    assert types[-1] == "final"
    assert events[-1]["status"] == "succeeded"


@pytest.mark.asyncio
async def test_retrieve_stream_emits_final_failed_on_engine_error(
    server_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker exception must surface as ``final{failed}`` with an error
    envelope rather than tearing the streaming response down with a 500."""

    async def _broken_retrieve(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated engine crash")

    monkeypatch.setattr(api_module, "retrieve", _broken_retrieve)

    async with server_client.stream(
        "POST",
        "/v1/retrieve",
        json={"q": "anything", "limit": 3},
    ) as resp:
        assert resp.status_code == 200
        events = [
            json.loads(line)
            for line in [ln async for ln in resp.aiter_lines()]
            if line.strip()
        ]

    final = events[-1]
    assert final["type"] == "final"
    assert final["status"] == "failed"
    assert final["error"]["code"] == "engine_error"
    assert "simulated engine crash" in final["error"]["message"]


@pytest.mark.asyncio
async def test_retrieve_accepts_limit_boundary_values(
    server_client: httpx.AsyncClient,
    ingested_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``limit=1`` and ``limit=100`` are inclusive bounds — must succeed
    where 0 / 101 fail. Guards against an off-by-one regression in the
    route's range check."""
    _patch_embedder(monkeypatch)

    for limit in (1, 100):
        async with server_client.stream(
            "POST",
            "/v1/retrieve",
            json={"q": "wiki", "limit": limit},
        ) as resp:
            assert resp.status_code == 200, f"limit={limit} unexpectedly rejected"
            events = [
                json.loads(line)
                for line in [ln async for ln in resp.aiter_lines()]
                if line.strip()
            ]

        assert events[-1]["status"] == "succeeded", f"limit={limit} engine failure"
        assert isinstance(events[-1]["result"]["chunks"], list)
