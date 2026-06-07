"""Wiring smoke test for ``api.check_providers`` with injected fake providers.

The pure helpers (``_sanitize_base_url``, prompt-override validation) are
unit-tested in ``test_api_health.py``; this locks the end-to-end ``check``
shape — factory bypass via injection, the parallel two-leg gather, the
empty-completion detection (issue #160), the ``llm_only`` / ``embed_only``
single-leg routing, and CheckReport assembly — with no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from tests.fakes import FakeEmbeddings, FakeLLM, init_test_base


@pytest.fixture
def base(tmp_path: Path) -> Path:
    wiki = tmp_path / "base"
    init_test_base(wiki)
    return wiki


async def test_check_both_legs_ok(base: Path) -> None:
    report = await api.check_providers(
        base, llm=FakeLLM(response_text="OK"), embedder=FakeEmbeddings()
    )
    assert report.llm is not None and report.llm.ok, report.llm
    assert report.embed is not None and report.embed.ok, report.embed
    # A fresh base configures no per-base prompt overrides; the leg still runs.
    assert all(p.ok for p in report.prompts)


async def test_check_flags_empty_llm_completion(base: Path) -> None:
    # issue #160: a call that returns without raising but yields no visible
    # text is NOT a healthy provider — the probe must report not-ok.
    report = await api.check_providers(
        base, llm=FakeLLM(response_text="   "), embedder=FakeEmbeddings()
    )
    assert report.llm is not None and not report.llm.ok
    assert "EMPTY completion" in (report.llm.detail or "")


async def test_check_llm_only_skips_embed(base: Path) -> None:
    report = await api.check_providers(
        base, llm=FakeLLM(response_text="OK"), llm_only=True
    )
    assert report.llm is not None and report.llm.ok
    assert report.embed is None


async def test_check_embed_only_skips_llm(base: Path) -> None:
    report = await api.check_providers(
        base, embedder=FakeEmbeddings(), embed_only=True
    )
    assert report.embed is not None and report.embed.ok
    assert report.llm is None


async def test_check_rejects_both_only_flags(base: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        await api.check_providers(base, llm_only=True, embed_only=True)
