"""``OpenAICompatReranker`` contract.

The reranker speaks the de-facto-standard Jina/Cohere ``/rerank`` wire shape
(``{model, query, documents, top_n}`` → ``{"results": [{"index", "relevance_score"}]}``)
that Gitee AI, SiliconFlow, Jina, and Cohere all converge on. Two things must
hold for the search layer to trust it:

1. Scores come back aligned to **input order** even when the endpoint returns
   ``results`` sorted by relevance (it always does) — the searcher pairs scores
   to candidate chunks positionally, exactly like the embedder remaps ``index``.
2. Transient HTTP failures (5xx / 408 / 429 / timeout / connection drop) wrap as
   ``TransientProviderError`` so the searcher degrades to the fused order;
   permanent ones (401 / 403 / 404 / bad model) wrap as ``ProviderError`` so a
   misconfig fails fast instead of silently degrading on every query.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from dikw_core.providers import build_reranker
from dikw_core.providers.base import ProviderError, TransientProviderError
from dikw_core.providers.rerank import OpenAICompatReranker

from .fakes import make_provider_cfg


def _ok_post(results: list[dict[str, Any]]) -> AsyncMock:
    """An ``AsyncMock`` for ``client.post`` returning a 200 with ``results``."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"results": results})
    return AsyncMock(return_value=resp)


def _status_post(status: int) -> AsyncMock:
    def _raise() -> None:
        raise httpx.HTTPStatusError(
            f"{status}",
            request=httpx.Request("POST", "u"),
            response=httpx.Response(status, request=httpx.Request("POST", "u")),
        )

    resp = MagicMock()
    resp.raise_for_status = MagicMock(side_effect=_raise)
    return AsyncMock(return_value=resp)


def _make_reranker(batch_size: int = 64) -> OpenAICompatReranker:
    return OpenAICompatReranker(
        api_key_env="GITEE_API_KEY",
        base_url="https://example.test/v1",
        api_key="k",
        batch_size=batch_size,
    )


def _echo_index_post() -> AsyncMock:
    """A ``client.post`` mock that scores each document in the batch by the
    trailing integer in its text (``"doc 7"`` → 7.0), echoing per-batch indices.
    Lets a test assert the provider remaps batch-local indices to the correct
    global position when batching."""

    async def _post(url: str, *, json: dict[str, Any]) -> Any:
        docs = json["documents"]
        results = [
            {"index": i, "relevance_score": float(d.rsplit(" ", 1)[-1])}
            for i, d in enumerate(docs)
        ]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"results": results})
        return resp

    return AsyncMock(side_effect=_post)


@pytest.mark.asyncio
async def test_rerank_realigns_results_to_input_order() -> None:
    """The endpoint returns results sorted by relevance; ``rerank`` must
    return scores in INPUT order (index 0 first), not response order."""
    reranker = _make_reranker()
    # Documents fed in order [d0, d1, d2]; endpoint returns them reordered.
    reranker._client = MagicMock()
    reranker._client.post = _ok_post(
        [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
            {"index": 1, "relevance_score": 0.5},
        ]
    )

    scores = await reranker.rerank("q", ["d0", "d1", "d2"], model="bge-reranker-v2-m3")

    assert scores == [0.1, 0.5, 0.9]


@pytest.mark.asyncio
async def test_rerank_sends_top_n_equal_to_doc_count() -> None:
    """``top_n`` must equal the document count so every candidate is scored
    (the searcher needs a score for each window chunk to reorder them)."""
    reranker = _make_reranker()
    captured: dict[str, Any] = {}

    async def _capture(url: str, *, json: dict[str, Any]) -> Any:
        captured["url"] = url
        captured["json"] = json
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(
            return_value={
                "results": [
                    {"index": i, "relevance_score": 1.0} for i in range(len(json["documents"]))
                ]
            }
        )
        return resp

    reranker._client = MagicMock()
    reranker._client.post = _capture

    await reranker.rerank("q", ["a", "b", "c"], model="m")

    assert captured["url"].endswith("/rerank")
    assert captured["json"]["top_n"] == 3
    assert captured["json"]["query"] == "q"
    assert captured["json"]["documents"] == ["a", "b", "c"]
    assert captured["json"]["model"] == "m"


@pytest.mark.asyncio
async def test_rerank_batches_documents_over_batch_size() -> None:
    """Vendors cap the documents array (Gitee /rerank: 1-25). The provider must
    split a larger window into batches, score each, and remap each batch's
    local indices back to the global input position — so a 40-doc window works
    against a 25-cap endpoint and scores stay aligned to input order."""
    reranker = _make_reranker(batch_size=2)
    reranker._client = MagicMock()
    reranker._client.post = _echo_index_post()

    docs = [f"doc {i}" for i in range(5)]  # scores should come back [0,1,2,3,4]
    scores = await reranker.rerank("q", docs, model="m")

    assert scores == [0.0, 1.0, 2.0, 3.0, 4.0]
    # 5 docs at batch_size=2 → 3 POST calls (2 + 2 + 1), each within the cap.
    assert reranker._client.post.await_count == 3
    for call in reranker._client.post.await_args_list:
        assert len(call.kwargs["json"]["documents"]) <= 2


@pytest.mark.asyncio
async def test_rerank_empty_documents_short_circuits() -> None:
    """No documents → no network call, empty scores."""
    reranker = _make_reranker()
    reranker._client = MagicMock()
    reranker._client.post = AsyncMock(side_effect=AssertionError("must not POST"))

    assert await reranker.rerank("q", [], model="m") == []


@pytest.mark.asyncio
async def test_rerank_result_count_mismatch_raises() -> None:
    """A response missing scores for some documents is a contract breach —
    fail loud rather than fabricate a default score for the gaps."""
    reranker = _make_reranker()
    reranker._client = MagicMock()
    reranker._client.post = _ok_post([{"index": 0, "relevance_score": 0.5}])

    with pytest.raises(ProviderError):
        await reranker.rerank("q", ["a", "b"], model="m")


@pytest.mark.asyncio
async def test_rerank_duplicate_index_raises() -> None:
    """A response that passes the count check but repeats an `index` (so
    another document is never scored) must fail loud, not silently leave a
    document at 0.0 — same 'fail loud rather than fabricate gaps' intent as the
    length-mismatch guard."""
    reranker = _make_reranker()
    reranker._client = MagicMock()
    # Two results, both index 0 → count matches len(documents)=2, range ok, but
    # document 1 is never covered.
    reranker._client.post = _ok_post(
        [
            {"index": 0, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
        ]
    )

    with pytest.raises(ProviderError):
        await reranker.rerank("q", ["a", "b"], model="m")


@pytest.mark.asyncio
async def test_rerank_classifies_503_as_transient() -> None:
    reranker = _make_reranker()
    reranker._client = MagicMock()
    reranker._client.post = _status_post(503)

    with pytest.raises(TransientProviderError) as ei:
        await reranker.rerank("q", ["a"], model="m")
    assert "503" in str(ei.value)


@pytest.mark.asyncio
async def test_rerank_classifies_429_as_transient() -> None:
    reranker = _make_reranker()
    reranker._client = MagicMock()
    reranker._client.post = _status_post(429)

    with pytest.raises(TransientProviderError):
        await reranker.rerank("q", ["a"], model="m")


@pytest.mark.asyncio
async def test_rerank_classifies_401_as_permanent() -> None:
    reranker = _make_reranker()
    reranker._client = MagicMock()
    reranker._client.post = _status_post(401)

    with pytest.raises(ProviderError) as ei:
        await reranker.rerank("q", ["a"], model="m")
    assert not isinstance(ei.value, TransientProviderError)


@pytest.mark.asyncio
async def test_rerank_classifies_404_model_as_permanent() -> None:
    reranker = _make_reranker()
    reranker._client = MagicMock()
    reranker._client.post = _status_post(404)

    with pytest.raises(ProviderError) as ei:
        await reranker.rerank("q", ["a"], model="bogus")
    assert not isinstance(ei.value, TransientProviderError)


@pytest.mark.asyncio
async def test_rerank_classifies_connection_error_as_transient() -> None:
    reranker = _make_reranker()
    reranker._client = MagicMock()
    reranker._client.post = AsyncMock(side_effect=httpx.ConnectError("reset"))

    with pytest.raises(TransientProviderError) as ei:
        await reranker.rerank("q", ["a"], model="m")
    assert "ConnectError" in str(ei.value)


@pytest.mark.asyncio
async def test_rerank_generic_http_error_is_permanent() -> None:
    """A bare httpx.HTTPError (not status/timeout/network) wraps as a permanent
    ProviderError — the catch-all arm below the specific ones."""
    reranker = _make_reranker()
    reranker._client = MagicMock()
    reranker._client.post = AsyncMock(side_effect=httpx.HTTPError("malformed"))

    with pytest.raises(ProviderError) as ei:
        await reranker.rerank("q", ["a"], model="m")
    assert not isinstance(ei.value, TransientProviderError)


@pytest.mark.asyncio
async def test_rerank_out_of_range_index_raises() -> None:
    """A result index outside [0, len(documents)) is a contract breach."""
    reranker = _make_reranker()
    reranker._client = MagicMock()
    reranker._client.post = _ok_post(
        [
            {"index": 0, "relevance_score": 0.5},
            {"index": 9, "relevance_score": 0.5},  # out of range for 2 docs
        ]
    )

    with pytest.raises(ProviderError):
        await reranker.rerank("q", ["a", "b"], model="m")


@pytest.mark.asyncio
async def test_rerank_parse_failure_is_transient() -> None:
    """A 200 whose body lacks `results` (truncated CDN/proxy response) is a
    parse failure → transient, so the search layer degrades rather than 500s."""
    reranker = _make_reranker()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"unexpected": "shape"})  # KeyError on results
    reranker._client = MagicMock()
    reranker._client.post = AsyncMock(return_value=resp)

    with pytest.raises(TransientProviderError):
        await reranker.rerank("q", ["a"], model="m")


@pytest.mark.asyncio
async def test_get_client_sets_auth_header_and_is_reused() -> None:
    """`_get_client` builds a no-keepalive client with the resolved Bearer key
    and caches it (one client per reranker instance)."""
    reranker = _make_reranker()
    c1 = reranker._get_client()
    c2 = reranker._get_client()
    assert c1 is c2
    assert c1.headers["Authorization"] == "Bearer k"
    assert c1.headers["Content-Type"] == "application/json"
    await reranker.aclose()


@pytest.mark.asyncio
async def test_aclose_closes_and_resets_client() -> None:
    reranker = _make_reranker()
    reranker._get_client()
    assert reranker._client is not None
    await reranker.aclose()
    assert reranker._client is None
    # Idempotent: a second close on an already-closed reranker is a no-op.
    await reranker.aclose()


def test_build_reranker_none_when_unconfigured() -> None:
    """No `provider.rerank` → no reranker built (rerank off because unconfigured)."""
    assert build_reranker(make_provider_cfg()) is None


def test_build_reranker_builds_configured() -> None:
    """A configured `provider.rerank` builds an `OpenAICompatReranker` threaded
    with the model/url/key/batch from config."""
    cfg = make_provider_cfg(
        rerank="openai_compat_rerank",
        rerank_model="bge-reranker-v2-m3",
        rerank_base_url="https://ai.gitee.com/v1",
        rerank_api_key_env="GITEE_API_KEY",
        rerank_batch_size=8,
    )
    rr = build_reranker(cfg)
    assert isinstance(rr, OpenAICompatReranker)
    assert rr._base_url == "https://ai.gitee.com/v1"
    assert rr._batch_size == 8
