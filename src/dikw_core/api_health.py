"""Health + provider-check cluster of the engine facade.

``health`` is a non-blocking server self-description (storage counts only,
never calls a provider) for agent bootstrap probes; ``check_providers``
actively pings the configured LLM / embedding endpoints. The ``_probe_*``
helpers and the 1x1-PNG fixture back ``check_providers``; ``_sanitize_base_url``
strips credentials from URLs before they reach a /v1/health response.

rank2 module: imports ``api_core`` (``_with_storage`` / ``load_base``) and
the leaf ``api_types`` DTOs, never the ``api`` facade. ``api`` re-exports
``health`` / ``check_providers`` (public) plus the underscore helpers the
tests reach for (``_sanitize_base_url`` / ``_PROBE_PNG_1X1``);
``_llm_credentials_present`` stays module-internal (not re-exported).
"""

from __future__ import annotations

import asyncio
import os
import struct
import time
import zlib
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from . import __version__ as _pkg_version
from . import prompts as _prompts
from .api_core import _with_storage, load_base
from .api_types import (
    CheckReport,
    EmbeddingInfo,
    HealthReport,
    LayerCounts,
    LlmInfo,
    MultimodalInfo,
    ProbeResult,
    ProvidersInfo,
)
from .config import DikwConfig
from .providers import (
    EmbeddingProvider,
    LLMProvider,
    MultimodalEmbeddingProvider,
    build_embedder,
    build_llm,
    build_multimodal_embedder,
)
from .schemas import ImageContent, Layer, MultimodalInput


def _sanitize_base_url(url: str | None) -> str | None:
    """Strip userinfo / query / fragment from a ``base_url`` before
    exposing it on /v1/health.

    Defends against credential leakage when a user puts a token directly
    in the URL — ``https://user:token@api.example/`` or
    ``…?api_key=…`` — by keeping only ``scheme://host[:port]/path``.
    Returns ``None`` when the input is empty, unparseable, or has no
    scheme/host: leaving a malformed URL on a probe response is worse
    than dropping it.
    """
    if not url:
        return None
    try:
        parts = urlsplit(url)
        # ``hostname`` and ``port`` are properties that can raise on
        # malformed input (e.g. ``port`` raises ``ValueError`` for an
        # out-of-range or non-numeric port); pull them inside the try.
        host = parts.hostname
        port = parts.port
    except (ValueError, TypeError):
        return None
    if not parts.scheme or not host:
        return None
    # ``urlsplit.hostname`` strips IPv6 brackets — re-bracket so
    # ``http://[::1]:8080/v1`` doesn't round-trip as ``http://::1:8080/v1``.
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _llm_credentials_present(
    provider: Literal["anthropic_compat", "openai_compat", "openai_codex"],
    *,
    base_root: Path,
) -> bool:
    """Whether credentials for the given LLM provider are resolvable.

    Env-keyed providers (anthropic_compat, openai_compat) check the
    matching ``API_KEY_ENV`` constant; the codex protocol checks the
    dikw-managed store at ``<base_root>/.dikw/auth.json``, falling back
    to the codex CLI store iff lazy migration would succeed there
    (fresh, non-expired tokens). That predicts what
    ``resolve_access_token`` will do, so /v1/health agrees with the
    runtime even right before the first LLM call triggers migration —
    and a logged-out dikw store with a stale codex CLI file still
    reports false rather than silently relying on the leftover.

    Explicit per-provider branch (rather than ``else: openai_compat``)
    so adding a new LLM provider surfaces as a typed mypy error here +
    a runtime ``ValueError`` instead of silently reporting the wrong
    credentials shape.
    """
    if provider == "anthropic_compat":
        from .providers.anthropic_compat import API_KEY_ENV

        return bool(os.environ.get(API_KEY_ENV))
    if provider == "openai_compat":
        from .providers.openai_compat import API_KEY_ENV

        return bool(os.environ.get(API_KEY_ENV))
    if provider == "openai_codex":
        from .providers.codex_auth import (
            _read_codex_cli_tokens_if_valid,
            list_providers,
        )

        if "openai-codex" in list_providers(base_root):
            return True
        # Lazy migration: if the dikw store is empty but the codex CLI
        # store has fresh tokens, the next ``resolve_access_token`` call
        # will populate the dikw store automatically. Surface that as
        # "credentials present" so /v1/health doesn't false-negative on
        # upgraded users who haven't issued an LLM call yet.
        return _read_codex_cli_tokens_if_valid() is not None
    raise ValueError(f"unknown llm provider: {provider!r}")


async def health(path: str | Path | None = None) -> HealthReport:
    """Server self-description for agent bootstrap probes.

    Opens storage briefly to read ``counts()``; never invokes the LLM /
    embedding providers (so a misconfigured key does not 5xx the health
    probe). Returned config is the *resolved* shape — what the server
    actually uses — minus secrets (no API keys, no DSN, no SQLite path).
    """
    from .providers.openai_compat import EMBEDDING_API_KEY_ENV

    cfg, root, storage = await _with_storage(path)
    try:
        counts = await storage.counts()
    finally:
        await storage.close()

    by_layer = counts.documents_by_layer
    # ``wisdom_items`` counts W-layer documents: wisdom pages are
    # first-class documents indexed at ``Layer.WISDOM``, so this is a
    # real count read off the ``documents`` table — not a legacy zero.
    layer_counts = LayerCounts(
        sources=int(by_layer.get(Layer.SOURCE.value, 0)),
        knowledge_pages=int(by_layer.get(Layer.KNOWLEDGE.value, 0)),
        wisdom_items=int(by_layer.get(Layer.WISDOM.value, 0)),
        chunks=counts.chunks,
    )

    p = cfg.provider
    llm_info = LlmInfo(
        provider=p.llm,
        model=p.llm_model,
        base_url=_sanitize_base_url(p.llm_base_url),
        max_retries=p.llm_max_retries,
        max_tokens_synth=p.llm_max_tokens_synth,
        timeout_seconds=p.llm_timeout_seconds,
        api_key_present=_llm_credentials_present(p.llm, base_root=root),
    )

    mm_info: MultimodalInfo | None = None
    mm_cfg = cfg.assets.multimodal
    if mm_cfg is not None:
        mm_dump = mm_cfg.model_dump()
        mm_dump["base_url"] = _sanitize_base_url(mm_dump.get("base_url"))
        mm_info = MultimodalInfo.model_validate(mm_dump)

    embedding_info = EmbeddingInfo(
        provider=p.embedding,
        model=p.embedding_model,
        base_url=_sanitize_base_url(p.embedding_base_url),
        dim=p.embedding_dim,
        revision=p.embedding_revision,
        normalize=p.embedding_normalize,
        distance=p.embedding_distance,
        batch_size=p.embedding_batch_size,
        max_retries=p.embedding_max_retries,
        timeout_seconds=p.embedding_timeout_seconds,
        provider_label=p.embedding_provider_label,
        api_key_present=bool(os.environ.get(EMBEDDING_API_KEY_ENV)),
        multimodal=mm_info,
    )

    return HealthReport(
        version=_pkg_version,
        base_root=str(Path(root).resolve()),
        storage_engine=cfg.storage.backend,
        layer_counts=layer_counts,
        providers=ProvidersInfo(llm=llm_info, embedding=embedding_info),
    )


# ---- verifiable config tool ---------------------------------------------


async def _probe_llm(
    llm: LLMProvider, model: str, target: str, *, max_tokens: int
) -> ProbeResult:
    started = time.perf_counter()
    try:
        resp = await llm.complete(
            system="You are a connectivity check. Reply with exactly: OK",
            user="ping",
            model=model,
            # Reuse the configured synth budget rather than a tiny fixed cap. A
            # reasoning model's hidden chain-of-thought draws on ``max_tokens``
            # before it emits any visible token (MiniMax-M3 needs >= 8192 — see
            # docs/providers.md); a fixed 32 starved it into an EMPTY completion
            # and the probe false-reported a healthy provider as down. Threading
            # the synth budget makes a green check predict the synth path's
            # success (providers that drop max_tokens on the wire, e.g. codex,
            # are unaffected).
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as e:  # provider exceptions are intentionally heterogeneous
        return ProbeResult(ok=False, target=target, detail=f"{type(e).__name__}: {e}")
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    input_tok = int((resp.usage or {}).get("input_tokens", 0))
    # A call that returns without raising but yields NO text is not a healthy
    # provider — it is the issue #160 failure mode (the provider produced an
    # empty completion, e.g. a codex final-response that dropped its streamed
    # output, or a model that emitted only reasoning). Reporting ``ok`` here
    # let a 227-source synth run against a silently-empty provider. Verify the
    # returned text, not just that the call did not throw.
    if not resp.text.strip():
        out_tok = int((resp.usage or {}).get("output_tokens", 0))
        return ProbeResult(
            ok=False,
            target=target,
            detail=(
                f"{elapsed_ms}ms, provider returned an EMPTY completion "
                f"(finish_reason={resp.finish_reason!r}, output_tokens={out_tok}) "
                f"— the model produced no visible text; check the model/request shape"
            ),
        )
    return ProbeResult(
        ok=True,
        target=target,
        detail=f"{elapsed_ms}ms, {input_tok} input tokens",
    )


async def _probe_embed(
    embedder: EmbeddingProvider,
    model: str,
    target: str,
    *,
    provider_label: str | None = None,
) -> ProbeResult:
    started = time.perf_counter()
    try:
        vectors = await embedder.embed(["ping"], model=model)
    except Exception as e:
        return ProbeResult(ok=False, target=target, detail=f"{type(e).__name__}: {e}")
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    # A call that returns without raising but yields no vector (or a zero-dim
    # one) is not a healthy provider — the embed twin of ``_probe_llm``'s
    # empty-completion check and ``_probe_multimodal``'s shape check.
    # Reporting ``ok`` here would green-light a ``dikw client check`` while
    # every ingest embed call produces nothing.
    if not vectors or not vectors[0]:
        return ProbeResult(
            ok=False,
            target=target,
            detail=(
                f"{elapsed_ms}ms, provider returned an EMPTY embedding "
                f"(vectors={len(vectors)}, dim=0) — the probe text "
                f"produced no usable vector; check the model/request shape"
            ),
        )
    dim = len(vectors[0])
    detail = f"{elapsed_ms}ms, dim={dim}"
    if provider_label:
        detail = f"{detail}, provider={provider_label}"
    return ProbeResult(ok=True, target=target, detail=detail)


def _build_probe_png_1x1() -> bytes:
    """Smallest valid PNG (1x1 black RGB pixel, 69 bytes).

    Built at module load via stdlib ``struct`` + ``zlib`` so the chunk
    CRCs are guaranteed correct — a hand-written byte literal here was
    truncated by one CRC byte once and Gitee's image decoder rejected
    the whole multimodal probe with a misleading "Supported image
    type:" error that hid the real cause.
    """

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: width=1, height=1, bit_depth=8, color_type=2 (RGB), the rest 0.
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    # IDAT: one scanline of [filter=0, R=0, G=0, B=0], deflate-compressed.
    idat = _chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00", 9))
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


_PROBE_PNG_1X1 = _build_probe_png_1x1()

async def _probe_multimodal(
    embedder: MultimodalEmbeddingProvider,
    model: str,
    target: str,
    *,
    provider_label: str | None = None,
) -> ProbeResult:
    """Probe a multimodal embedder with a single batched text+image request.

    Sends two per-modality inputs (one text, one tiny PNG) in **one** HTTP
    call so latency stays bounded by a single RTT — no sequential probes
    that would double end-to-end time. Validation hinges on both vectors
    coming back with a consistent dim; an empty or shape-mismatched
    response surfaces as an error rather than a silent pass.
    """
    inputs = [
        MultimodalInput(text="ping"),
        MultimodalInput(images=[ImageContent(bytes=_PROBE_PNG_1X1, mime="image/png")]),
    ]
    started = time.perf_counter()
    try:
        vectors = await embedder.embed(inputs, model=model)
    except Exception as e:
        return ProbeResult(ok=False, target=target, detail=f"{type(e).__name__}: {e}")
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if len(vectors) != 2:
        return ProbeResult(
            ok=False,
            target=target,
            detail=(
                f"{elapsed_ms}ms, expected 2 vectors (text+image), got "
                f"{len(vectors)} — provider returned a shape-mismatched batch"
            ),
        )
    dim_text = len(vectors[0])
    dim_image = len(vectors[1])
    if dim_text != dim_image:
        return ProbeResult(
            ok=False,
            target=target,
            detail=(
                f"{elapsed_ms}ms, dim mismatch text={dim_text} image={dim_image} — "
                f"per-modality vectors must share one space"
            ),
        )
    # Two zero-dim vectors pass both the count check and the dim-equality
    # check (0 == 0) — the same silently-dead-provider shape the text probe
    # above rejects must fail here too.
    if dim_text == 0:
        return ProbeResult(
            ok=False,
            target=target,
            detail=(
                f"{elapsed_ms}ms, provider returned EMPTY embeddings "
                f"(dim=0 for both modalities) — the probe produced no usable "
                f"vectors; check the model/request shape"
            ),
        )
    detail = f"{elapsed_ms}ms, dim={dim_text}, modalities=text+image"
    if provider_label:
        detail = f"{detail}, provider={provider_label}"
    return ProbeResult(ok=True, target=target, detail=detail)


# Maps a ``lint.fixer_prompts`` config key to the packaged prompt name it
# overrides. ``non_atomic_page`` is absent — it reuses the ``synthesize``
# template, overridden via ``synth.prompt_path`` (see ADR-0003).
_FIXER_PROMPT_NAMES = {
    "orphan_merge": "lint_fix_orphan_merge",
    "broken_wikilink": "lint_fix_broken_wikilink_grounded",
}


def _check_prompt_overrides(cfg: DikwConfig, base_root: Path) -> list[ProbeResult]:
    """Validate every *configured* per-base prompt override against its contract.

    One ``ProbeResult`` per configured override (containment + placeholder /
    output-marker contract); unset overrides produce no entry. Surfaces the
    same ``PromptOverrideError`` synth / lint would raise, but at
    ``dikw client check`` time so misconfig is caught before a run.
    """
    configured: list[tuple[str, str]] = []
    if cfg.synth.prompt_path:
        configured.append(("synthesize", cfg.synth.prompt_path))
    for key, override in cfg.lint.fixer_prompts.items():
        name = _FIXER_PROMPT_NAMES.get(key)
        if name and override:
            configured.append((name, override))

    results: list[ProbeResult] = []
    for name, override in configured:
        try:
            _prompts.resolve(name, override_path=override, base_root=base_root)
        except _prompts.PromptOverrideError as e:
            results.append(ProbeResult(ok=False, target=override, detail=str(e)))
        else:
            results.append(
                ProbeResult(ok=True, target=override, detail=f"{name} override valid")
            )
    return results


async def check_providers(
    path: str | Path | None = None,
    *,
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    multimodal_embedder: MultimodalEmbeddingProvider | None = None,
    llm_only: bool = False,
    embed_only: bool = False,
) -> CheckReport:
    """Ping the configured LLM and embedding providers.

    ``llm``, ``embedder``, and ``multimodal_embedder`` are injectable for
    tests; production callers leave them ``None`` and the factory builds
    them from ``provider:`` / ``assets.multimodal:`` in ``dikw.yml``. Two
    legs run in parallel and each reports its own result — an LLM failure
    does not short-circuit the embedding probe.

    When ``cfg.assets.multimodal`` is configured, the embed leg routes
    through ``_probe_multimodal`` (one batched text+image request) instead
    of the text-only ``_probe_embed`` — the multimodal embedder is what
    ingest actually uses for both chunks and assets, so the check must
    follow that route to remain truthful.

    ``llm_only`` / ``embed_only`` (mutually exclusive) verify a single leg.
    The skipped leg is never built or called, so a misconfigured embedding
    side cannot fail an ``--llm-only`` run.
    """
    if llm_only and embed_only:
        raise ValueError("llm_only and embed_only are mutually exclusive")

    cfg, _root = load_base(path)

    llm_probe: ProbeResult | None = None
    embed_probe: ProbeResult | None = None
    llm_target = cfg.provider.llm_base_url or "(provider default)"
    embed_label = cfg.provider.embedding_provider_label
    mm_cfg = cfg.assets.multimodal
    # Per-leg target. Multimodal probe hits ``assets.multimodal.base_url``
    # (or the provider's default), which is independent of the text leg's
    # ``embedding_base_url``; reporting the wrong one makes a green check
    # misleading in split-vendor setups.
    if mm_cfg is not None:
        embed_target = mm_cfg.base_url or "(provider default)"
    else:
        embed_target = cfg.provider.embedding_base_url
    # Track an internally-built multimodal embedder so we close its httpx
    # client in ``finally`` — mirrors the ``owned_mm`` pattern in ingest/
    # query. An *injected* embedder is the caller's lifetime to manage.
    owned_mm: MultimodalEmbeddingProvider | None = None

    if not embed_only:
        llm_inst = llm if llm is not None else build_llm(cfg.provider, base_root=_root)
    if not llm_only:
        if mm_cfg is not None:
            if multimodal_embedder is not None:
                mm_inst = multimodal_embedder
            else:
                mm_inst = build_multimodal_embedder(
                    mm_cfg.provider, base_url=mm_cfg.base_url, batch=mm_cfg.batch
                )
                owned_mm = mm_inst
        else:
            embed_inst = (
                embedder if embedder is not None else build_embedder(cfg.provider)
            )

    async def _embed_leg() -> ProbeResult:
        if mm_cfg is not None:
            return await _probe_multimodal(
                mm_inst, mm_cfg.model, embed_target, provider_label=embed_label
            )
        return await _probe_embed(
            embed_inst,
            cfg.provider.embedding_model,
            embed_target,
            provider_label=embed_label,
        )

    llm_budget = cfg.provider.llm_max_tokens_synth
    try:
        if llm_only:
            llm_probe = await _probe_llm(
                llm_inst, cfg.provider.llm_model, llm_target, max_tokens=llm_budget
            )
        elif embed_only:
            embed_probe = await _embed_leg()
        else:
            llm_probe, embed_probe = await asyncio.gather(
                _probe_llm(
                    llm_inst, cfg.provider.llm_model, llm_target, max_tokens=llm_budget
                ),
                _embed_leg(),
            )
    finally:
        if owned_mm is not None and hasattr(owned_mm, "aclose"):
            await owned_mm.aclose()
    # Prompt-override validation is local + cheap, so it runs on every check
    # regardless of llm_only / embed_only — it never hits a provider.
    prompt_checks = _check_prompt_overrides(cfg, Path(_root))
    return CheckReport(llm=llm_probe, embed=embed_probe, prompts=prompt_checks)
