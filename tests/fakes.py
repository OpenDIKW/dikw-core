"""Test doubles for LLM + embedding providers.

``FakeEmbeddings`` lives in ``src/dikw_core/eval/fake_embedder.py`` so it
ships with the wheel for ``dikw eval``. This module re-exports it for
backward compatibility with existing ``from tests.fakes import FakeEmbeddings``
imports; new code should reach into ``dikw_core.eval.fake_embedder``.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from dikw_core.config import ProviderConfig
from dikw_core.eval.fake_embedder import EMBED_DIM, FakeEmbeddings
from dikw_core.providers import LLMResponse, LLMStreamEvent, ToolSpec
from dikw_core.schemas import EmbeddingVersion, MultimodalInput

__all__ = [
    "EMBED_DIM",
    "CountingEmbedder",
    "FakeEmbeddings",
    "FakeLLM",
    "FakeMultimodalEmbedding",
    "init_test_wiki",
    "make_jwt",
    "make_provider_cfg",
    "register_text_version",
    "register_text_version_or_skip",
]


def make_jwt(claims: dict[str, Any]) -> str:
    """Build a 3-segment JWT with the given payload claims.

    Header is fixed (``{"alg":"none","typ":"JWT"}``) and the signature
    segment is a placeholder string — the codex_auth helpers under test
    never verify the signature, only base64url-decode the payload. Used
    by tests that want to construct tokens with specific ``exp`` /
    ``chatgpt_account_id`` claims without depending on PyJWT.
    """

    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode("utf-8"))
    payload = _b64url(json.dumps(claims).encode("utf-8"))
    return f"{header}.{payload}.sig"


def init_test_wiki(path: Any, *, description: str = "test wiki", dim: int = EMBED_DIM) -> None:
    """``api.init_wiki`` + patch the dikw.yml so ``embedding_dim`` matches
    the test embedder. ``FakeEmbeddings`` produces ``EMBED_DIM`` (64) by
    default; tests using a different embedder pass ``dim``.
    """
    from dikw_core import api
    from dikw_core.config import dump_config_yaml, load_config

    api.init_wiki(path, description=description)
    cfg_path = path / "dikw.yml"
    cfg = load_config(cfg_path)
    cfg.provider.embedding_dim = dim
    cfg_path.write_text(dump_config_yaml(cfg), encoding="utf-8")


async def register_text_version(
    storage: Any,
    *,
    dim: int = EMBED_DIM,
    provider: str = "test",
    model: str = "fake",
    revision: str = "",
) -> int:
    """Register a text ``embed_versions`` row in ``storage`` and return its id.

    Cross-test helper; production code resolves the version from
    ``ProviderConfig`` via ``api.ingest`` / ``api.query``. Tests that
    exercise ``upsert_embeddings`` / ``embed_chunks`` directly call
    this first.
    """
    return await storage.upsert_embed_version(
        EmbeddingVersion(
            provider=provider,
            model=model,
            revision=revision,
            dim=dim,
            normalize=True,
            distance="cosine",
            modality="text",
        )
    )


async def register_text_version_or_skip(
    storage: Any,
    *,
    dim: int = EMBED_DIM,
    provider: str = "test",
    model: str = "fake",
    revision: str = "",
) -> int:
    """``register_text_version`` that ``pytest.skip``s on backends that
    don't implement embed versioning. Saves a 4-line try/except in every
    contract test that just needs a version_id to exercise."""
    import pytest

    from dikw_core.storage.base import NotSupported

    try:
        return await register_text_version(
            storage, dim=dim, provider=provider, model=model, revision=revision
        )
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")


def make_provider_cfg(**overrides: Any) -> ProviderConfig:
    """Build a ``ProviderConfig`` with sensible test defaults filled in.

    The 4 embedding identity fields (dim/revision/normalize/distance) are
    required in production so config files have to be explicit; tests
    don't care, so this helper supplies test-friendly defaults that
    callers can override per-case.
    """
    base: dict[str, Any] = {
        "embedding_dim": EMBED_DIM,
        "embedding_revision": "",
        "embedding_normalize": True,
        "embedding_distance": "cosine",
    }
    base.update(overrides)
    return ProviderConfig(**base)


@dataclass
class CountingEmbedder:
    """Counts ``embed`` calls + total texts; delegates to ``FakeEmbeddings``.

    Used by streaming + perf tests to assert "cache hit means zero
    provider calls" (``embed_calls == 0``) or to enforce per-batch
    streaming visibility (call count > 1).

    ``fail_after`` lets a test simulate a mid-flight crash: the
    ``(fail_after+1)``-th call raises ``RuntimeError``. Default
    ``None`` = never fail.
    """

    inner: FakeEmbeddings = field(default_factory=FakeEmbeddings)
    embed_calls: int = field(default=0, init=False)
    total_texts: int = field(default=0, init=False)
    fail_after: int | None = None

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        self.embed_calls += 1
        self.total_texts += len(texts)
        if self.fail_after is not None and self.embed_calls > self.fail_after:
            raise RuntimeError(
                f"CountingEmbedder simulated failure on call {self.embed_calls}"
            )
        return await self.inner.embed(texts, model=model)

    def reset(self) -> None:
        self.embed_calls = 0
        self.total_texts = 0


@dataclass
class FakeLLM:
    """Captures the last call and returns a scripted or default response.

    ``stream_chunks`` (optional) lets a test exercise the streaming path:
    when set, ``complete_stream`` yields one ``token`` event per chunk
    followed by a ``done`` event whose ``text`` is the joined chunks.
    With ``stream_chunks=None`` the default Phase-1 behaviour holds and
    ``complete_stream`` raises ``NotImplementedError``, matching the
    real provider stubs.

    ``reasoning_chunks`` (optional) emits ``LLMStreamEvent(type="reasoning")``
    events ahead of the token stream — used by tests that want to verify
    downstream consumers tolerate (or surface) reasoning fragments emitted
    by reasoning-capable providers like ``OpenAICodexLLM``. Requires
    ``stream_chunks`` to also be set.
    """

    response_text: str = "STUB: wired up."
    stream_chunks: list[str] | None = None
    reasoning_chunks: list[str] | None = None
    last_system: str | None = field(default=None, init=False)
    last_user: str | None = field(default=None, init=False)
    last_max_tokens: int | None = field(default=None, init=False)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        _ = (model, temperature, tools)
        self.last_system = system
        self.last_user = user
        self.last_max_tokens = max_tokens
        return LLMResponse(text=self.response_text, finish_reason="end_turn")

    def complete_stream(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        _ = (model, temperature, tools)
        self.last_system = system
        self.last_user = user
        self.last_max_tokens = max_tokens
        chunks = self.stream_chunks
        if chunks is None:
            raise NotImplementedError(
                "FakeLLM.complete_stream requires stream_chunks to be set"
            )

        reasoning = self.reasoning_chunks

        async def _gen() -> AsyncIterator[LLMStreamEvent]:
            if reasoning is not None:
                for r_chunk in reasoning:
                    yield LLMStreamEvent(type="reasoning", delta=r_chunk)
            for chunk in chunks:
                yield LLMStreamEvent(type="token", delta=chunk)
            yield LLMStreamEvent(
                type="done",
                text="".join(chunks),
                finish_reason="end_turn",
            )

        return _gen()


@dataclass
class FakeMultimodalEmbedding:
    """Deterministic ``MultimodalEmbeddingProvider`` for tests.

    Each input is hashed into a vector of the configured ``dim``. Same
    input → same vector; text and image inputs are tagged with distinct
    namespaces so a text payload never collides with an image of the
    same bytes.
    """

    dim: int = 4
    last_inputs: list[MultimodalInput] = field(default_factory=list, init=False)
    last_model: str | None = field(default=None, init=False)
    embed_calls: int = field(default=0, init=False)
    total_inputs: int = field(default=0, init=False)

    async def embed(
        self, inputs: list[MultimodalInput], *, model: str
    ) -> list[list[float]]:
        self.last_inputs = list(inputs)
        self.last_model = model
        self.embed_calls += 1
        self.total_inputs += len(inputs)
        return [self._vector_for(inp) for inp in inputs]

    def _vector_for(self, inp: MultimodalInput) -> list[float]:
        """Deterministic projection of an input into a ``dim``-vector.

        Hash text + image bytes (each in its own namespace prefix) and
        sample ``dim`` floats out of the resulting digest. Stable across
        runs, distinct across inputs of different modality even when
        their content is byte-identical."""
        h = hashlib.sha256()
        if inp.text is not None:
            h.update(b"text:")
            h.update(inp.text.encode("utf-8"))
        for img in inp.images:
            h.update(b"image:")
            h.update(img.mime.encode("utf-8"))
            h.update(b"|")
            h.update(img.bytes)
        digest = h.digest()
        # Tile the 32-byte digest to fill ``dim`` floats; map each byte to
        # a value in [-1, 1] so vectors are dense and non-zero.
        out: list[float] = []
        for i in range(self.dim):
            b = digest[i % len(digest)]
            out.append((b / 127.5) - 1.0)
        return out
