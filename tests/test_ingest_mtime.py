"""Source-layer ingest must never persist a meaningless ``mtime`` (#145).

A byte-stable import tarball (dikw-web) zeroes the tar ``mtime`` field so
identical bytes dedup, so extracted source files land with ``st_mtime == 0``
and used to be stored verbatim — rendering as ``1970-01-01`` and contributing
nothing to the graph change-hash. Ingest now falls back to wall-clock, prefers
an already-stored positive mtime (no flap for re-persisted image docs), and
re-persists rows whose stored mtime is broken so legacy ``0`` rows self-heal.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.api import _doc_id_for, _with_storage
from dikw_core.api_ingest import _resolve_ingest_mtime
from dikw_core.domains.data.backends.markdown import parse_text
from dikw_core.schemas import DocumentRecord, Layer

from .fakes import FakeEmbeddings, init_test_base, seed_doc


def _existing(mtime: float, doc_hash: str = "h") -> DocumentRecord:
    return DocumentRecord(
        doc_id="source:sources/x.md",
        path="sources/x.md",
        hash=doc_hash,
        mtime=mtime,
        layer=Layer.SOURCE,
    )


# --------------------------------------------------------------------------- #
# _resolve_ingest_mtime — pure fallback policy
# --------------------------------------------------------------------------- #


def test_resolve_ingest_mtime_prefers_valid_parsed() -> None:
    assert _resolve_ingest_mtime(123.0, "h", None) == 123.0
    assert _resolve_ingest_mtime(123.0, "h", _existing(0.0)) == 123.0


def test_resolve_ingest_mtime_falls_back_to_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("time.time", lambda: 999.0)
    assert _resolve_ingest_mtime(0.0, "h", None) == 999.0
    # existing row is also broken → still wall-clock
    assert _resolve_ingest_mtime(0.0, "h", _existing(0.0)) == 999.0


def test_resolve_ingest_mtime_preserves_existing_when_body_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Same hash (body unchanged) + a stored positive mtime → preserve it so an
    # image-bearing doc (re-persists every ingest, bypassing the hash-skip)
    # doesn't flap its mtime / the graph change-hash each run.
    monkeypatch.setattr("time.time", lambda: 999.0)
    assert _resolve_ingest_mtime(0.0, "h", _existing(500.0, "h")) == 500.0


def test_resolve_ingest_mtime_changed_body_advances_to_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Body changed (hash differs) but the re-imported file still has no usable
    # mtime — must advance to wall-clock, NOT keep the stale stored value, so
    # rendered dates and `since_ts` sync cursors move (codex P2 regression).
    monkeypatch.setattr("time.time", lambda: 999.0)
    assert _resolve_ingest_mtime(0.0, "hNEW", _existing(500.0, "hOLD")) == 999.0


# --------------------------------------------------------------------------- #
# Full ingest — forward fix + skip-clause self-heal
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_ingest_zero_mtime_source_gets_wall_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    init_test_base(base)
    src = base / "sources"
    src.mkdir(parents=True, exist_ok=True)
    f = src / "x.md"
    f.write_text("# X\n\nbody", encoding="utf-8")
    os.utime(f, (0, 0))  # byte-stable tarball leaves st_mtime == 0

    monkeypatch.setattr("time.time", lambda: 1_700_000_000.0)
    report = await api.ingest(base, embedder=FakeEmbeddings())
    assert report.added == 1

    _cfg, _root, storage = await _with_storage(base)
    try:
        doc = await storage.get_document(_doc_id_for(Layer.SOURCE, "sources/x.md"))
    finally:
        await storage.close()
    assert doc is not None
    assert doc.mtime == 1_700_000_000.0


@pytest.mark.asyncio
async def test_ingest_reheals_stored_zero_mtime_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    init_test_base(base)
    body = "# X\n\nbody"
    # Seed a source row whose stored mtime is broken (0) but whose hash
    # matches the on-disk file. Pre-fix this is skip-eligible (active +
    # hash match + no asset refs) and stays 0 forever; the broken-mtime
    # skip clause must force one re-persist that heals it.
    expected_hash = parse_text(path="sources/x.md", text=body, mtime=0.0).hash
    await seed_doc(
        base,
        layer=Layer.SOURCE,
        path="sources/x.md",
        body=body,
        mtime=0.0,
        doc_hash=expected_hash,
    )
    # The on-disk file also carries st_mtime==0 (the real byte-stable case),
    # so the heal must come from the wall-clock fallback, not the file mtime.
    os.utime(base / "sources" / "x.md", (0, 0))

    monkeypatch.setattr("time.time", lambda: 1_700_000_001.0)
    report = await api.ingest(base, embedder=FakeEmbeddings())
    assert report.updated == 1  # re-persisted via the broken-mtime skip clause

    _cfg, _root, storage = await _with_storage(base)
    try:
        doc = await storage.get_document(_doc_id_for(Layer.SOURCE, "sources/x.md"))
    finally:
        await storage.close()
    assert doc is not None
    assert doc.mtime == 1_700_000_001.0  # healed to wall-clock, was 0


@pytest.mark.asyncio
async def test_ingest_changed_zero_mtime_source_advances_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    init_test_base(base)
    # A prior ingest stored this source with a healthy positive mtime. Now the
    # same path is re-imported with DIFFERENT content but still st_mtime==0
    # (byte-stable tarball). The change must advance mtime to wall-clock, not
    # freeze at the stale stored value (codex P2 regression).
    old_body = "# X\n\nold"
    await seed_doc(
        base,
        layer=Layer.SOURCE,
        path="sources/x.md",
        body=old_body,
        mtime=500.0,
        doc_hash=parse_text(path="sources/x.md", text=old_body, mtime=0.0).hash,
    )
    f = base / "sources" / "x.md"
    f.write_text("# X\n\nNEW body", encoding="utf-8")  # hash now differs
    os.utime(f, (0, 0))

    monkeypatch.setattr("time.time", lambda: 1_700_000_000.0)
    report = await api.ingest(base, embedder=FakeEmbeddings())
    assert report.updated == 1

    _cfg, _root, storage = await _with_storage(base)
    try:
        doc = await storage.get_document(_doc_id_for(Layer.SOURCE, "sources/x.md"))
    finally:
        await storage.close()
    assert doc is not None
    assert doc.mtime == 1_700_000_000.0  # fresh wall-clock, not the stale 500.0
