"""OpenAI/Jina/Cohere-compatible reranker provider.

Wraps the de-facto-standard ``/rerank`` endpoint behind the
``RerankProvider`` Protocol. Gitee AI (``BAAI/bge-reranker-v2-m3``,
``Qwen3-Reranker-*``), SiliconFlow, Jina, and Cohere all converge on the
same wire shape, so one adapter covers every vendor — the base URL, model,
and key env var pick which one runs.

**Wire format.**

    POST {base_url}/rerank          # base_url already ends in /v1
    {
      "model": "<reranker model name>",
      "query": "<user query>",
      "documents": ["chunk text 0", "chunk text 1", ...],
      "top_n": <len(documents)>     # score every candidate, never truncate here
    }

    -> {"results": [{"index": <input position>, "relevance_score": <float>}, ...]}

The endpoint returns ``results`` sorted by ``relevance_score`` descending, so
``rerank`` remaps each result's ``index`` back to input order before returning
— callers (``HybridSearcher``) pair scores to candidate chunks positionally,
exactly like ``OpenAICompatEmbeddings`` sorts on the response ``index``.

A reranker is a deterministic scoring model, not a generative one: it reorders
the deterministically retrieved candidate set and is part of *scoping*, not
*reasoning*. See ``docs/adr/0006-reranker-deterministic-scoping.md``.
"""

from __future__ import annotations

from typing import Any

import httpx

from ._http import build_no_keepalive_async_client
from .base import ProviderError, TransientProviderError, _resolve_key

_DEFAULT_TIMEOUT = 30.0


class OpenAICompatReranker:
    """``RerankProvider`` impl over a Jina/Cohere-compatible ``/rerank`` HTTP
    surface.

    Vendor-agnostic: the same wire shape works for Gitee AI's
    ``BAAI/bge-reranker-v2-m3``, SiliconFlow, Jina, and Cohere. The model name
    discriminates which reranker runs server-side; ``base_url`` + the key env
    var pick the vendor.
    """

    def __init__(
        self,
        *,
        api_key_env: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        batch_size: int = 16,
    ) -> None:
        self._api_key_env = api_key_env
        self._base_url = base_url.rstrip("/")
        self._api_key_explicit = api_key
        self._timeout = timeout
        # Vendors cap the ``documents`` array per request (Gitee /rerank: 25),
        # exactly like the embedding batch cap. Split a larger candidate window
        # into batches; each batch is scored independently (a cross-encoder
        # scores each (query, document) pair on its own, so scores are
        # comparable across batches) and remapped back to the global position.
        self._batch_size = max(1, batch_size)
        self._client: httpx.AsyncClient | None = None

    async def rerank(
        self, query: str, documents: list[str], *, model: str
    ) -> list[float]:
        if not documents:
            return []
        scores: list[float] = []
        for start in range(0, len(documents), self._batch_size):
            batch = documents[start : start + self._batch_size]
            scores.extend(await self._rerank_batch(query, batch, model=model))
        return scores

    async def _rerank_batch(
        self, query: str, documents: list[str], *, model: str
    ) -> list[float]:
        """Score one within-cap batch; returns scores aligned to ``documents``
        (input) order via the response ``index`` remap."""
        client = self._get_client()
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            # Score EVERY candidate so the searcher can reorder the whole
            # window — never let the endpoint pre-truncate to a smaller top_n.
            "top_n": len(documents),
        }
        # Wrap httpx + JSON parsing exceptions and classify into transient vs
        # permanent. The search layer catches ``TransientProviderError`` and
        # degrades to the fused order (a single vendor blip must not 500 the
        # read path); permanent errors (401/403/404, bad model) propagate so a
        # rerank misconfig fails fast instead of silently degrading every query.
        try:
            resp = await client.post(f"{self._base_url}/rerank", json=payload)
            resp.raise_for_status()
            data = resp.json()
            results: list[dict[str, Any]] = data["results"]
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            err = (
                TransientProviderError
                if status >= 500 or status in (408, 429)
                else ProviderError
            )
            raise err(
                f"rerank call failed with status {status}: {exc}"
            ) from exc
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise TransientProviderError(
                f"rerank call timed out / connection failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"rerank call failed: {type(exc).__name__}: {exc}"
            ) from exc
        except (KeyError, ValueError, TypeError) as exc:
            # Parse failures often indicate a transient CDN / proxy serving a
            # partial response — retryable from the search layer's view.
            raise TransientProviderError(
                f"rerank response parse failed: {type(exc).__name__}: {exc}"
            ) from exc

        if len(results) != len(documents):
            # top_n == len(documents), so a short result set is a contract
            # breach — fail loud rather than fabricate scores for the gaps.
            raise ProviderError(
                f"rerank returned {len(results)} scores for {len(documents)} "
                f"documents"
            )
        scores = [0.0] * len(documents)
        seen: set[int] = set()
        for r in results:
            idx = int(r["index"])
            if not 0 <= idx < len(documents):
                raise ProviderError(
                    f"rerank result index {idx} out of range for "
                    f"{len(documents)} documents"
                )
            if idx in seen:
                # A duplicate index passes the count check yet leaves another
                # document unscored — fail loud rather than silently bottom-rank
                # the gap at 0.0. With count == len(documents) + range-checked +
                # no duplicates, ``seen`` covers every index, so no slot keeps
                # its 0.0 placeholder.
                raise ProviderError(
                    f"rerank returned a duplicate index {idx} for "
                    f"{len(documents)} documents"
                )
            seen.add(idx)
            scores[idx] = float(r["relevance_score"])
        return scores

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # Reuse the shared no-keepalive client builder so the rerank leg
            # gets the same bounded connect/pool deadlines (connect=10s,
            # pool=5s) the SDK providers do — important on the interactive read
            # path, where a scalar timeout would let a slow-connect endpoint
            # burn the full read timeout just establishing a connection.
            _timeout, client = build_no_keepalive_async_client(self._timeout)
            client.headers["Authorization"] = (
                f"Bearer {_resolve_key(self._api_key_explicit, self._api_key_env)}"
            )
            client.headers["Content-Type"] = "application/json"
            self._client = client
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["OpenAICompatReranker"]
