"""Gitee AI multimodal embedding provider.

Wraps Gitee AI's hosted multimodal embedding endpoints
(``Qwen3-VL-Embedding-8B``, ``jina-clip-v2``, …) behind the
``MultimodalEmbeddingProvider`` Protocol so chunk text and image bytes
share one vector space.

**Wire format.** Gitee AI's multimodal endpoint accepts one shape across
every multimodal model it serves: a list of per-modality input dicts.

    POST {base_url}/embeddings        # base_url already ends in /v1
    {
      "model": "<multimodal model name>",
      "input": [
        {"text": "a blue cat"},
        {"image": "data:image/png;base64,...."}
      ]
    }

Each ``MultimodalInput`` becomes one wire input. The schema permits
text+image combinations on a single ``MultimodalInput`` so v1.5 joint
encoding can land without breaking the wire contract, but Gitee's
endpoint embeds per-modality and has no joint-encode mode — combined
inputs are rejected loudly here so config mistakes surface fast instead
of silently dropping a modality. Same for >1 image per input.

If a different multimodal vendor (Voyage, Cohere v4, native Jina) lands,
drop a new file under ``providers/`` and add a branch to
``build_multimodal_embedder``. Their wire shapes diverge enough that
sharing this serializer would be misleading.
"""

from __future__ import annotations

import base64
import os
from typing import Any

import httpx

from ..schemas import MultimodalInput
from ..telemetry import gen_ai_span
from .base import ProviderError, TransientProviderError
from .openai_compat import _resolve_embedding_api_key

_DEFAULT_BASE_URL = "https://ai.gitee.com/v1"
_DEFAULT_TIMEOUT = 60.0


def _serialize_input(inp: MultimodalInput) -> Any:
    """Convert one ``MultimodalInput`` into Gitee's per-modality wire shape."""
    has_text = inp.text is not None and inp.text != ""
    has_images = bool(inp.images)
    if has_text and has_images:
        raise ProviderError(
            "Gitee multimodal endpoint embeds per-modality; got a "
            "MultimodalInput with both text and images. Split into separate "
            "inputs."
        )
    if has_text:
        return {"text": inp.text}
    if has_images:
        if len(inp.images) > 1:
            raise ProviderError(
                "Gitee multimodal endpoint accepts one image per input; got "
                f"{len(inp.images)}. Split across multiple inputs."
            )
        img = inp.images[0]
        b64 = base64.b64encode(img.bytes).decode("ascii")
        return {"image": f"data:{img.mime};base64,{b64}"}
    # Empty input: send empty text to keep the 1:1 input/output shape.
    # The provider returns a usable (if uninformative) zero-ish vector;
    # Storage's vec_search filters NULL distances so degenerate hits drop.
    return {"text": ""}


class GiteeMultimodalEmbedding:
    """``MultimodalEmbeddingProvider`` impl over Gitee AI's HTTP surface.

    Model-agnostic: the same wire shape works for Qwen3-VL-Embedding-8B
    (4096-dim, end-to-end verified) and any other multimodal model Gitee
    serves on the same endpoint. The model name in
    ``assets.multimodal.model`` discriminates which model runs server-side.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        batch: int = 16,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = (
            base_url
            or os.environ.get("DIKW_EMBEDDING_BASE_URL")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        self._api_key_explicit = api_key
        self._batch = batch
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def embed(
        self, inputs: list[MultimodalInput], *, model: str
    ) -> list[list[float]]:
        if not inputs:
            return []
        client = self._get_client()
        out: list[list[float]] = []
        # One gen_ai.embeddings span per call: the per-batch httpx POSTs nest
        # under it as child wire spans. Gitee's response carries no usage block,
        # so no token attributes are set. Opened after the empty-input return.
        with gen_ai_span(operation="embeddings", system="gitee", model=model):
            for start in range(0, len(inputs), self._batch):
                batch = inputs[start : start + self._batch]
                payload = {
                    "model": model,
                    "input": [_serialize_input(i) for i in batch],
                }
                # Wrap httpx + JSON parsing exceptions and classify into
                # transient vs permanent so the embed-batch retry-skip in
                # ``info.embed._run_batch_with_retry`` retries transient
                # vendor failures (5xx, 408/429, timeouts, connection
                # resets) but propagates permanent ones (401, 403, 404,
                # invalid model id) instead of silently retrying-then-
                # skipping. Codex review finding, 0.4.0.
                try:
                    resp = await client.post(
                        f"{self._base_url}/embeddings", json=payload
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    # OpenAI-compat: {"data": [{"index": int, "embedding": [...]}]}
                    rows = sorted(data["data"], key=lambda r: r.get("index", 0))
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    err = (
                        TransientProviderError
                        if status >= 500 or status in (408, 429)
                        else ProviderError
                    )
                    raise err(
                        f"Gitee multimodal embedding call failed with status "
                        f"{status}: {exc}"
                    ) from exc
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    raise TransientProviderError(
                        f"Gitee multimodal embedding call timed out / connection "
                        f"failed: {type(exc).__name__}: {exc}"
                    ) from exc
                except httpx.HTTPError as exc:
                    raise ProviderError(
                        f"Gitee multimodal embedding call failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                except (KeyError, ValueError, TypeError) as exc:
                    # Parse failures often indicate a transient CDN / proxy
                    # serving a partial response — retry once before giving up.
                    raise TransientProviderError(
                        f"Gitee multimodal response parse failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                if len(rows) != len(batch):
                    raise ProviderError(
                        f"Gitee AI returned {len(rows)} vectors for "
                        f"{len(batch)} inputs"
                    )
                out.extend([list(r["embedding"]) for r in rows])
        return out

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={
                    "Authorization": f"Bearer {_resolve_embedding_api_key(self._api_key_explicit)}",
                    "Content-Type": "application/json",
                },
                # Gitee's batch embedding endpoints (Qwen3-VL family in
                # particular) silently drop idle TCP keepalives mid-batch;
                # forcing a fresh connection per request adds ~50ms TLS
                # handshake but eliminates the multi-minute retry storm
                # observed in fdd2cae on the OpenAI-compat leg.
                limits=httpx.Limits(max_keepalive_connections=0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["GiteeMultimodalEmbedding"]
