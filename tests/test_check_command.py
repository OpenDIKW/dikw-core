"""``dikw check`` — verifiable config tool.

Connects to the configured LLM + embedding providers with one tiny call
each and reports whether each leg roundtripped. Used by operators after
editing ``dikw.yml`` to confirm credentials + endpoints before running
ingest/query against a real budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dikw_core import api
from dikw_core.cli import app
from dikw_core.providers import LLMResponse, ToolSpec
from tests.fakes import FakeEmbeddings, FakeLLM


class BrokenLLM:
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
        _ = (system, user, model, max_tokens, temperature, tools)
        raise RuntimeError("llm backend refused connection")


class BrokenEmbedder:
    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        _ = (texts, model)
        raise RuntimeError("embedding backend refused connection")


@pytest.fixture()
def wiki(tmp_path: Path) -> Path:
    w = tmp_path / "wiki"
    api.init_wiki(w, description="check-command test wiki")
    return w


@pytest.mark.asyncio
async def test_check_passes_when_both_providers_roundtrip(wiki: Path) -> None:
    report = await api.check_providers(wiki, llm=FakeLLM(), embedder=FakeEmbeddings())
    assert report.ok
    assert report.llm is not None
    assert report.embed is not None
    assert report.llm.ok
    assert report.embed.ok


@pytest.mark.asyncio
async def test_check_reports_llm_failure_distinctly_from_embed_failure(
    wiki: Path,
) -> None:
    report = await api.check_providers(wiki, llm=BrokenLLM(), embedder=FakeEmbeddings())
    assert not report.ok
    assert report.llm is not None
    assert report.embed is not None
    assert not report.llm.ok
    assert "refused" in report.llm.detail.lower()
    # Embed leg still reports its own result — LLM failure doesn't short-circuit it.
    assert report.embed.ok


@pytest.mark.asyncio
async def test_check_reports_embed_failure_distinctly_from_llm_failure(
    wiki: Path,
) -> None:
    report = await api.check_providers(wiki, llm=FakeLLM(), embedder=BrokenEmbedder())
    assert not report.ok
    assert report.llm is not None
    assert report.embed is not None
    assert report.llm.ok
    assert not report.embed.ok
    assert "refused" in report.embed.detail.lower()


def test_cli_check_exits_nonzero_on_failure(
    asgi_client: Any,
    patch_transport_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server-routed: the engine inside the in-memory server picks up the
    monkeypatched ``api.build_llm`` and fails — the CLI's exit code
    mirrors the report's ``ok`` field."""
    monkeypatch.setattr("dikw_core.api.build_llm", lambda cfg, **_kw: BrokenLLM())
    monkeypatch.setattr("dikw_core.api.build_embedder", lambda cfg: FakeEmbeddings())
    patch_transport_factory()
    result = CliRunner().invoke(app, ["check"])
    assert result.exit_code == 1, result.stdout


def test_cli_check_exits_zero_on_success(
    asgi_client: Any,
    patch_transport_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dikw_core.api.build_llm", lambda _cfg, **_kw: FakeLLM())
    monkeypatch.setattr(
        "dikw_core.api.build_embedder", lambda _cfg: FakeEmbeddings()
    )
    patch_transport_factory()
    result = CliRunner().invoke(app, ["check"])
    assert result.exit_code == 0, result.stdout


# ---- Per-leg verification (--llm-only / --embed-only) -----------------


@pytest.mark.asyncio
async def test_check_llm_only_skips_embedding_probe(wiki: Path) -> None:
    """``llm_only=True`` must not build or call the embedding provider.

    Even a broken embedder should not fail the report — it's simply never
    consulted. This is the whole point of single-leg verification: a user
    can validate the LLM before the embedding side is configured at all.
    """
    report = await api.check_providers(
        wiki, llm=FakeLLM(), embedder=BrokenEmbedder(), llm_only=True
    )
    assert report.ok
    assert report.llm is not None and report.llm.ok
    assert report.embed is None


@pytest.mark.asyncio
async def test_check_embed_only_skips_llm_probe(wiki: Path) -> None:
    report = await api.check_providers(
        wiki, llm=BrokenLLM(), embedder=FakeEmbeddings(), embed_only=True
    )
    assert report.ok
    assert report.embed is not None and report.embed.ok
    assert report.llm is None


@pytest.mark.asyncio
async def test_check_rejects_mutually_exclusive_flags(wiki: Path) -> None:
    with pytest.raises(ValueError) as excinfo:
        await api.check_providers(
            wiki, llm=FakeLLM(), embedder=FakeEmbeddings(),
            llm_only=True, embed_only=True,
        )
    assert "mutually exclusive" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_check_llm_only_does_not_build_embedder(
    wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--llm-only`` must skip build_embedder entirely.

    A config with a broken embedder factory (e.g., missing API key) should
    still let the user verify the LLM leg. If we call build_embedder first
    and it raises, the whole probe dies — which defeats the feature.
    """
    def _boom(_cfg: Any) -> Any:
        raise RuntimeError("embedder factory should not have been called")

    monkeypatch.setattr("dikw_core.api.build_embedder", _boom)
    report = await api.check_providers(wiki, llm=FakeLLM(), llm_only=True)
    assert report.ok
    assert report.llm is not None and report.llm.ok
    assert report.embed is None


def test_cli_check_llm_only_exits_zero_when_embed_would_fail(
    asgi_client: Any,
    patch_transport_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--llm-only`` skips the embedder factory inside the server, so a
    broken embedder factory must not fail the probe."""
    monkeypatch.setattr("dikw_core.api.build_llm", lambda _cfg, **_kw: FakeLLM())

    def _boom(_cfg: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr("dikw_core.api.build_embedder", _boom)
    patch_transport_factory()
    result = CliRunner().invoke(app, ["check", "--llm-only"])
    assert result.exit_code == 0, result.stdout


def test_cli_check_rejects_both_only_flags(
    asgi_client: Any,
    patch_transport_factory: Any,
) -> None:
    patch_transport_factory()
    result = CliRunner().invoke(
        app, ["check", "--llm-only", "--embed-only"]
    )
    assert result.exit_code == 2, result.stdout


def test_cli_check_embed_only_shows_provider_label_when_yaml_set(
    asgi_client: tuple[Any, Any],
    patch_transport_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``provider.embedding_provider_label`` flows verbatim from dikw.yml
    through the server's check report into the client's rendered table.
    Edit the runtime's bound wiki on disk before running the CLI so the
    server picks up the label on its next read."""
    import yaml

    monkeypatch.setattr(
        "dikw_core.api.build_embedder", lambda _cfg: FakeEmbeddings()
    )

    _, rt = asgi_client
    cfg_path = rt.root / "dikw.yml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw.setdefault("provider", {})["embedding_provider_label"] = "gitee-ai"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    # The runtime caches its DikwConfig at lifespan startup; reload so
    # the route reads the updated label.
    from dikw_core.api import load_wiki

    rt.cfg, _ = load_wiki(rt.root)

    patch_transport_factory()
    result = CliRunner().invoke(app, ["check", "--embed-only"])
    assert result.exit_code == 0, result.stdout
    assert "gitee-ai" in result.stdout


# ---- Multimodal probe ----------------------------------------------------
#
# When ``assets.multimodal`` is configured, the engine ingests chunks AND
# assets through the multimodal embedder, never the text-only one. The
# check must follow that route, otherwise it greenlights a wiki whose
# real ingest path will fail. The probe sends one text + one image input
# in a single batched request — exactly the shape the production pipeline
# uses, no RTT stacking.


@pytest.fixture()
def multimodal_wiki(tmp_path: Path) -> Path:
    """A wiki whose dikw.yml carries an ``assets.multimodal`` block."""
    import yaml

    w = tmp_path / "wiki-mm"
    api.init_wiki(w, description="multimodal probe test wiki")
    cfg_path = w / "dikw.yml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw.setdefault("assets", {})["multimodal"] = {
        "provider": "gitee_multimodal",
        "model": "Qwen3-VL-Embedding-8B",
        "dim": 4096,
        "batch": 16,
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return w


@pytest.mark.asyncio
async def test_check_embed_probe_routes_through_multimodal_when_configured(
    multimodal_wiki: Path,
) -> None:
    """When ``assets.multimodal`` is configured, --embed-only must probe the
    multimodal embedder, not the text-only one. The probe sends one text +
    one image input in a single batched request (no RTT stacking)."""
    from tests.fakes import FakeMultimodalEmbedding

    fake_mm = FakeMultimodalEmbedding(dim=4096)
    report = await api.check_providers(
        multimodal_wiki,
        llm=FakeLLM(),
        multimodal_embedder=fake_mm,
        embed_only=True,
    )
    assert report.ok
    assert report.embed is not None and report.embed.ok
    # Detail must surface the dim probed AND that both modalities were sent.
    assert "dim=4096" in report.embed.detail
    assert "text+image" in report.embed.detail
    # Single batch call: 2 inputs (1 text + 1 image), not 2 separate calls.
    assert len(fake_mm.last_inputs) == 2
    text_inputs = [i for i in fake_mm.last_inputs if i.text]
    image_inputs = [i for i in fake_mm.last_inputs if i.images]
    assert len(text_inputs) == 1
    assert len(image_inputs) == 1
    # Probe targets the multimodal model name, not provider.embedding_model.
    assert fake_mm.last_model == "Qwen3-VL-Embedding-8B"


@pytest.mark.asyncio
async def test_check_embed_probe_uses_text_path_when_no_multimodal(
    wiki: Path,
) -> None:
    """The legacy text-embed probe must keep working when ``assets.multimodal``
    is absent — regression guard against routing every wiki through the
    multimodal path."""
    fake_text = FakeEmbeddings()
    report = await api.check_providers(
        wiki, llm=FakeLLM(), embedder=fake_text, embed_only=True
    )
    assert report.ok
    assert report.embed is not None and report.embed.ok
    # The text-only path doesn't probe images, so its detail line must
    # not advertise a "modalities=" tag.
    assert "modalities" not in report.embed.detail


@pytest.mark.asyncio
async def test_check_multimodal_probe_target_reflects_mm_base_url(
    tmp_path: Path,
) -> None:
    """The probe detail's target must point at the multimodal endpoint the
    request is actually sent to — ``assets.multimodal.base_url`` — not at
    ``provider.embedding_base_url`` (which is the text leg's URL).

    In split-vendor setups (text leg one vendor, multimodal another) the
    two URLs differ; reporting the wrong one makes a green check appear to
    validate one endpoint while it actually exercised another."""
    import yaml

    from tests.fakes import FakeMultimodalEmbedding

    w = tmp_path / "wiki-mm-split"
    api.init_wiki(w, description="split-base-url test wiki")
    cfg_path = w / "dikw.yml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw.setdefault("provider", {})["embedding_base_url"] = (
        "https://text-embed.example.com/v1"
    )
    raw.setdefault("assets", {})["multimodal"] = {
        "provider": "gitee_multimodal",
        "model": "Qwen3-VL-Embedding-8B",
        "dim": 4096,
        "batch": 16,
        "base_url": "https://multimodal.example.com/v1",
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    fake_mm = FakeMultimodalEmbedding(dim=4096)
    report = await api.check_providers(
        w, llm=FakeLLM(), multimodal_embedder=fake_mm, embed_only=True
    )
    assert report.ok
    assert report.embed is not None
    assert report.embed.target == "https://multimodal.example.com/v1"
    # The text-leg URL must NOT appear on the multimodal probe row — that
    # was the bug the codex review surfaced.
    assert "text-embed.example.com" not in report.embed.target


def test_probe_png_chunk_crcs_validate() -> None:
    """The 1x1 PNG used by ``_probe_multimodal`` must decode cleanly.

    Gitee's image decoder rejects the whole multimodal probe when the
    PNG bytes are even slightly malformed — the error message points at
    "Supported image type:" rather than the real cause (CRC mismatch),
    which makes the bug nearly impossible to diagnose from the wire.
    Walk every chunk and verify its stored CRC matches what
    ``zlib.crc32`` computes; a future hand-edit that drops a byte gets
    caught here instead of leaking to a real probe."""
    import zlib

    from dikw_core.api import _PROBE_PNG_1X1

    assert _PROBE_PNG_1X1[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    saw_iend = False
    while pos < len(_PROBE_PNG_1X1):
        length = int.from_bytes(_PROBE_PNG_1X1[pos : pos + 4], "big")
        tag = _PROBE_PNG_1X1[pos + 4 : pos + 8]
        data = _PROBE_PNG_1X1[pos + 8 : pos + 8 + length]
        crc_stored = int.from_bytes(
            _PROBE_PNG_1X1[pos + 8 + length : pos + 12 + length], "big"
        )
        crc_computed = zlib.crc32(tag + data) & 0xFFFFFFFF
        assert crc_stored == crc_computed, (
            f"chunk {tag!r} CRC mismatch: stored {crc_stored:08x}, "
            f"computed {crc_computed:08x}"
        )
        if tag == b"IEND":
            saw_iend = True
        pos += 12 + length
    assert saw_iend, "PNG missing IEND chunk"
    assert pos == len(_PROBE_PNG_1X1), "trailing bytes after IEND"


