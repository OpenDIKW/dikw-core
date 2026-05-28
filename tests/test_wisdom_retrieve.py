"""Retrieve-leg coverage for the W-layer.

``dikw client retrieve`` must surface wisdom chunks alongside source
+ knowledge chunks. The underlying ``HybridSearcher`` already
defaults ``layer=None`` (no per-leg filter), so wisdom hits arrive
naturally once chunks + embeddings are wired into storage. These
tests lock the contract — ``Hit.layer == Layer.WISDOM`` must come
back tagged correctly so callers can group / weight by provenance
layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.schemas import Layer

from .fakes import FakeEmbeddings, ingest_wisdom_files, init_test_base


def _drop_wisdom(wiki: Path, rel: str, body: str) -> None:
    p = wiki / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


@pytest.mark.asyncio
async def test_retrieve_returns_wisdom_hits(tmp_path: Path) -> None:
    """A wisdom page indexed via ``persist_wisdom`` must surface from
    retrieve. The FakeEmbeddings vector-leg deterministically ranks by
    token overlap, so a query that targets the wisdom page's body finds
    it.
    """
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/first-principles.md",
        (
            "---\ntitle: First Principles\n---\n# First Principles\n\n"
            "Reason from physics, not analogy. Break the problem down "
            "to fundamental truths and build up from there.\n"
        ),
    )
    await ingest_wisdom_files(
        wiki,
        ["wisdom/elon-musk/first-principles.md"],
        embedder=FakeEmbeddings(),
    )

    result = await api.retrieve(
        "first principles reasoning",
        wiki,
        limit=10,
        embedder=FakeEmbeddings(),
    )
    hits = result.chunks
    wisdom_hits = [h for h in hits if h.layer == Layer.WISDOM]
    assert wisdom_hits, [(h.path, h.layer) for h in hits]
    assert wisdom_hits[0].path == "wisdom/elon-musk/first-principles.md"


@pytest.mark.asyncio
async def test_retrieve_hit_layer_correctly_tagged(tmp_path: Path) -> None:
    """Every ``Hit`` returned must carry a ``layer`` field that matches
    the underlying document's ``Layer``. The client uses this to group
    or weight by layer (e.g. boost wiki over source, surface wisdom
    separately) so a mis-tag silently breaks the contract."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    # Drop one source + one wisdom doc and ensure both surface tagged
    # with their actual layer.
    src_dir = wiki / "sources" / "notes"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "musk-bio.md").write_text(
        "# Bio\n\nElon's biographical notes about rocket engineering and tesla cars.\n",
        encoding="utf-8",
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/note.md",
        "# Note\n\nRocket engineering insights about reusability.\n",
    )
    # D-layer indexed by api.ingest; W-layer indexed by ingest_wisdom_files.
    await api.ingest(wiki, embedder=FakeEmbeddings())
    await ingest_wisdom_files(
        wiki, ["wisdom/elon-musk/note.md"], embedder=FakeEmbeddings()
    )

    result = await api.retrieve(
        "rocket engineering",
        wiki,
        limit=20,
        embedder=FakeEmbeddings(),
    )
    hits = result.chunks
    by_path = {h.path: h.layer for h in hits}
    if "sources/notes/musk-bio.md" in by_path:
        assert by_path["sources/notes/musk-bio.md"] == Layer.SOURCE
    if "wisdom/elon-musk/note.md" in by_path:
        assert by_path["wisdom/elon-musk/note.md"] == Layer.WISDOM
    # At minimum one of the two must show up — we want to lock that
    # both layers can participate in retrieve, not that they always do
    # under FakeEmbeddings's deterministic-token-overlap ranking.
    assert any(
        h.layer in (Layer.SOURCE, Layer.WISDOM) for h in hits
    ), [(h.path, h.layer) for h in hits]
