from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dikw_core import api

from .fakes import FakeEmbeddings, init_test_base

FIXTURES = Path(__file__).parent / "fixtures" / "notes"


@pytest.fixture()
def wiki_with_fixtures(tmp_path: Path) -> Path:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki, description="ingest-query test wiki")
    dest = wiki / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    return wiki


@pytest.mark.asyncio
async def test_ingest_is_idempotent_and_fills_storage(wiki_with_fixtures: Path) -> None:
    embedder = FakeEmbeddings()
    report = await api.ingest(wiki_with_fixtures, embedder=embedder)
    assert report.scanned == 3
    assert report.added == 3
    assert report.updated == 0
    assert report.unchanged == 0
    assert report.chunks >= 3
    assert report.embedded >= 3

    # re-run: all files should now be unchanged
    report2 = await api.ingest(wiki_with_fixtures, embedder=embedder)
    assert report2.scanned == 3
    assert report2.unchanged == 3
    assert report2.added == 0
    assert report2.chunks == 0


