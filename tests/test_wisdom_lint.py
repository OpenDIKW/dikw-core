"""Lint coverage for the 0.3.0 PR2 wisdom-as-documents pipeline.

PR2 adds the ``invalid_wisdom_status`` lint kind: it scans wisdom-layer
pages (``Layer.WISDOM``), and any frontmatter ``status:`` value not in
the four-enum set ``{draft, published, favorite, archived}`` surfaces
as a warning. Ingest is not blocked — the parser already collapses an
unknown value to ``DocumentRecord.status = None`` so the row lands;
lint is the user-facing nudge.

The broader wisdom-layer coverage of existing lint kinds
(``broken_wikilink``, ``orphan_page``, ``missing_provenance``) lands
in PR3 alongside the retrieve wiring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.config import load_config
from dikw_core.domains.knowledge.lint import run_lint
from dikw_core.storage import build_storage

from .fakes import FakeEmbeddings, init_test_base, seed_doc


def _drop_wisdom(wiki: Path, rel: str, body: str) -> None:
    p = wiki / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


@pytest.mark.asyncio
async def test_invalid_wisdom_status_lint_warns_but_ingest_succeeds(
    tmp_path: Path,
) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/bad.md",
        "---\nstatus: weird_value\n---\n# Weird\n\nbody.\n",
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/good.md",
        "---\nstatus: favorite\n---\n# Good\n\nbody.\n",
    )

    # Ingest must NOT raise — invalid status is a lint concern, not a
    # parse error.
    report = await api.ingest(wiki, embedder=FakeEmbeddings())
    assert report.errors == ()

    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        lint_report = await run_lint(storage, root=wiki)
    finally:
        await storage.close()

    invalid = [i for i in lint_report.issues if i.kind == "invalid_wisdom_status"]
    assert len(invalid) == 1
    assert invalid[0].path == "wisdom/elon-musk/bad.md"
    assert "weird_value" in invalid[0].detail


@pytest.mark.asyncio
async def test_valid_wisdom_status_does_not_warn(tmp_path: Path) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    for status in ("draft", "published", "favorite", "archived"):
        _drop_wisdom(
            wiki,
            f"wisdom/elon-musk/{status}.md",
            f"---\nstatus: {status}\n---\n# {status}\n\nbody.\n",
        )
    # one page with no status frontmatter — also valid
    _drop_wisdom(
        wiki, "wisdom/elon-musk/plain.md", "# Plain\n\nbody.\n"
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        lint_report = await run_lint(storage, root=wiki)
    finally:
        await storage.close()

    invalid = [i for i in lint_report.issues if i.kind == "invalid_wisdom_status"]
    assert invalid == []


@pytest.mark.asyncio
async def test_wisdom_broken_wikilink_surfaces(tmp_path: Path) -> None:
    """A wisdom page with ``[[Nonexistent Page]]`` must surface as a
    ``broken_wikilink`` lint issue. PR2 ingested wisdom into storage
    but PR3 is what makes the lint pass examine wisdom-layer docs;
    without the layer extension the broken link is silently lost.
    """
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    from dikw_core.schemas import Layer

    await seed_doc(
        wiki,
        layer=Layer.KNOWLEDGE,
        path="knowledge/existing.md",
        body="---\ntitle: Existing\n---\n# Existing\n",
        title="Existing",
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/refs.md",
        "# Refs\n\nGood [[Existing]] but bad [[Definitely Missing Page]].\n",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        lint_report = await run_lint(storage, root=wiki)
    finally:
        await storage.close()

    broken = [
        i for i in lint_report.issues
        if i.kind == "broken_wikilink"
        and i.path == "wisdom/elon-musk/refs.md"
    ]
    assert len(broken) == 1, [
        (i.kind, i.path, i.detail) for i in lint_report.issues
    ]
    assert "Definitely Missing Page" in broken[0].detail


@pytest.mark.asyncio
async def test_wisdom_orphan_not_flagged_when_wisdom_links_in(
    tmp_path: Path,
) -> None:
    """An orphan-page lint pass must scan W-layer pages alongside K
    and credit incoming wikilinks: a wisdom page cited by another
    wisdom page (the most common author pattern) must not surface as
    orphan. Without lint expansion the cited wisdom page wasn't even
    iterated; with the expansion the inbound counter must see the
    intra-wisdom edge."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/cited.md",
        "---\ntitle: Cited Wisdom\n---\n# Cited Wisdom\n\nbody.\n",
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/refs.md",
        "# Refs\n\nSee [[Cited Wisdom]].\n",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        lint_report = await run_lint(storage, root=wiki)
    finally:
        await storage.close()

    orphans = [
        i.path for i in lint_report.issues if i.kind == "orphan_page"
    ]
    assert "wisdom/elon-musk/cited.md" not in orphans


@pytest.mark.asyncio
async def test_knowledge_not_orphan_when_only_wisdom_links_in(
    tmp_path: Path,
) -> None:
    """Reverse case: a knowledge page cited ONLY from wisdom must not be
    flagged as orphan_page. PR2 lets users author wisdom that backlinks
    to wiki concepts; PR3 must make the lint pass credit those edges,
    otherwise OrphanPageFixer could delete legitimately-referenced
    knowledge pages on the next ``lint apply``."""
    from dikw_core.domains.knowledge.page_index import persist_page
    from dikw_core.schemas import Layer

    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    # Use persist_page directly to land both the knowledge page document AND
    # its outgoing links / no-links so the lint pass sees a real K-layer
    # row (seed_doc bypasses links table population).
    (wiki / "knowledge" / "concepts").mkdir(parents=True, exist_ok=True)
    (wiki / "knowledge" / "concepts" / "tesla.md").write_text(
        "---\ntitle: Tesla\n---\n# Tesla\n\nthe company.\n", encoding="utf-8"
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/musings.md",
        "# Musings\n\nSee [[Tesla]].\n",
    )

    # Persist the knowledge page first (so the title index sees it), then
    # ingest the wisdom page (which writes its outgoing wikilink to the
    # wiki dst path).
    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        await persist_page(
            storage=storage,
            root=wiki,
            path="knowledge/concepts/tesla.md",
            layer=Layer.KNOWLEDGE,
        )
    finally:
        await storage.close()
    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        lint_report = await run_lint(storage, root=wiki)
    finally:
        await storage.close()

    orphans = [
        i.path for i in lint_report.issues if i.kind == "orphan_page"
    ]
    assert "knowledge/concepts/tesla.md" not in orphans


@pytest.mark.asyncio
async def test_cross_layer_title_collision_surfaces_as_duplicate_title(
    tmp_path: Path,
) -> None:
    """A knowledge page and a wisdom page sharing the same title must trigger
    ``duplicate_title`` lint — verifying that PR3's switch to the
    ``build_title_indexes`` helper drops collisions into the per-title
    bucket the duplicate scan reads, instead of silently shadowing one
    layer behind the other (the bug PR2 introduced and PR3 inherited
    via the local ``{t: dup_paths[0]}`` shortcut)."""
    from dikw_core.domains.knowledge.page_index import persist_page
    from dikw_core.schemas import Layer

    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    (wiki / "knowledge" / "concepts").mkdir(parents=True, exist_ok=True)
    (wiki / "knowledge" / "concepts" / "tesla.md").write_text(
        "---\ntitle: Tesla\n---\n# Tesla\n\nthe company.\n", encoding="utf-8"
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/tesla.md",
        "---\ntitle: Tesla\n---\n# Tesla\n\npersonal note.\n",
    )

    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        await persist_page(
            storage=storage,
            root=wiki,
            path="knowledge/concepts/tesla.md",
            layer=Layer.KNOWLEDGE,
        )
    finally:
        await storage.close()
    await api.ingest(wiki, embedder=FakeEmbeddings())

    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        lint_report = await run_lint(storage, root=wiki)
    finally:
        await storage.close()

    duplicate = [
        i for i in lint_report.issues if i.kind == "duplicate_title"
    ]
    assert len(duplicate) >= 1, [
        (i.kind, i.path, i.detail) for i in lint_report.issues
    ]
    paths = {i.path for i in duplicate}
    assert (
        "knowledge/concepts/tesla.md" in paths
        or "wisdom/elon-musk/tesla.md" in paths
    ), paths


@pytest.mark.asyncio
async def test_wisdom_missing_provenance_surfaces(tmp_path: Path) -> None:
    """A wisdom page with ``sources:`` frontmatter pointing at a path
    that has no provenance row triggers ``missing_provenance``. PR3's
    lint expansion to wisdom must scan the K- AND W-layer
    ``sources:`` declarations symmetrically; without this a future
    regression to WIKI-only lint would silently skip every wisdom
    page's provenance scan."""
    from dikw_core.schemas import Layer

    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    src_dir = wiki / "sources" / "notes"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "musk-bio.md").write_text(
        "# Bio\n\nfacts.\n", encoding="utf-8"
    )
    _drop_wisdom(
        wiki,
        "wisdom/elon-musk/from-bio.md",
        "---\nsources:\n  - sources/notes/musk-bio.md\n---\n# From Bio\n\nbody.\n",
    )
    await api.ingest(wiki, embedder=FakeEmbeddings())

    # Manually clear the provenance row so the lint pass sees
    # frontmatter-without-table mismatch.
    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        await storage.replace_provenance_from(
            f"{Layer.WISDOM.value}:wisdom/elon-musk/from-bio.md",
            [],
        )
        lint_report = await run_lint(storage, root=wiki)
    finally:
        await storage.close()

    missing = [
        i for i in lint_report.issues
        if i.kind == "missing_provenance"
        and i.path == "wisdom/elon-musk/from-bio.md"
    ]
    assert len(missing) == 1, [
        (i.kind, i.path) for i in lint_report.issues
    ]
