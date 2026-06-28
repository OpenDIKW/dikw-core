"""Write-path warnings when an embedder is unconfigured / embedding deferred.

Pins the 0.6.x degrade-logging contract: an ``ingest`` that produces chunks but
has no embedder wired leaves those chunks without vectors (deferred to a future
ingest's resume scan) and must WARN so the operator notices a vector-less
corpus, rather than the deferral being silent. Permanent embed failures still
fail fast (covered in ``test_embed_resilient.py``); this file covers only the
"not configured → warning, degrade, don't block" leg.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from dikw_core import api

from .fakes import init_test_base


@pytest.mark.asyncio
async def test_ingest_warns_when_no_embedder_but_chunks_produced(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    wiki = tmp_path / "kb"
    init_test_base(wiki)
    src = wiki / "sources"
    src.mkdir(parents=True, exist_ok=True)
    (src / "a.md").write_text(
        "# Alpha\n\nSome body text that chunks into the index.\n", encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="dikw_core.api_ingest"):
        report = await api.ingest(wiki, embedder=None)

    assert report.chunks > 0, "the file must have produced chunks"
    assert any(
        r.levelno == logging.WARNING and "embed" in r.getMessage().lower()
        for r in caplog.records
    ), "ingest with no embedder but chunks produced must warn about deferred embedding"


@pytest.mark.asyncio
async def test_ingest_with_embedder_does_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The warning is specific to the unconfigured case — a normal ingest with
    an embedder wired embeds inline and stays quiet."""
    from .fakes import FakeEmbeddings

    wiki = tmp_path / "kb"
    init_test_base(wiki)
    src = wiki / "sources"
    src.mkdir(parents=True, exist_ok=True)
    (src / "a.md").write_text(
        "# Alpha\n\nSome body text that chunks into the index.\n", encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="dikw_core.api_ingest"):
        await api.ingest(wiki, embedder=FakeEmbeddings())

    assert not [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "embed" in r.getMessage().lower()
    ], "a normal ingest with an embedder must not warn about embedding"
