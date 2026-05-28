"""Lint coverage for the W-layer document pipeline.

Verifies that the lint kinds (``invalid_wisdom_status``,
``broken_wikilink``, ``orphan_page``, ``missing_provenance``,
``duplicate_title``) scan W-layer pages symmetrically with K-layer
pages. Since 0.4.0 the W-layer pipeline is exclusively driven by
``persist_wisdom`` (no ``api.ingest`` scan), so these tests seed
wisdom rows via the ``ingest_wisdom_files`` helper before invoking
``run_lint``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core.config import load_config
from dikw_core.domains.knowledge.lint import run_lint
from dikw_core.storage import build_storage

from .fakes import FakeEmbeddings, ingest_wisdom_files, init_test_base, seed_doc


def _drop_wisdom(wiki: Path, rel: str, body: str) -> None:
    p = wiki / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


@pytest.mark.asyncio
async def test_invalid_wisdom_status_lint_warns_but_persist_succeeds(
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

    # persist must NOT raise — invalid status is a lint concern, not a
    # parse error.
    await ingest_wisdom_files(
        wiki,
        ["wisdom/elon-musk/bad.md", "wisdom/elon-musk/good.md"],
        embedder=FakeEmbeddings(),
    )

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
    paths: list[str] = []
    for status in ("draft", "published", "favorite", "archived"):
        rel = f"wisdom/elon-musk/{status}.md"
        _drop_wisdom(
            wiki, rel, f"---\nstatus: {status}\n---\n# {status}\n\nbody.\n"
        )
        paths.append(rel)
    # one page with no status frontmatter — also valid
    rel_plain = "wisdom/elon-musk/plain.md"
    _drop_wisdom(wiki, rel_plain, "# Plain\n\nbody.\n")
    paths.append(rel_plain)
    await ingest_wisdom_files(wiki, paths, embedder=FakeEmbeddings())

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
    ``broken_wikilink`` lint issue.
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
    await ingest_wisdom_files(
        wiki, ["wisdom/elon-musk/refs.md"], embedder=FakeEmbeddings()
    )

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
    """A wisdom page cited by another wisdom page (the most common
    author pattern) must not surface as orphan_page."""
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
    await ingest_wisdom_files(
        wiki,
        [
            "wisdom/elon-musk/cited.md",
            "wisdom/elon-musk/refs.md",
        ],
        embedder=FakeEmbeddings(),
    )

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
    flagged as orphan_page."""
    from dikw_core.domains.knowledge.page_index import persist_knowledge

    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    # Use persist_knowledge directly to land both the knowledge page document
    # AND its outgoing links / no-links so the lint pass sees a real K-layer
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

    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        await persist_knowledge(
            storage=storage,
            root=wiki,
            path="knowledge/concepts/tesla.md",
        )
    finally:
        await storage.close()
    await ingest_wisdom_files(
        wiki, ["wisdom/elon-musk/musings.md"], embedder=FakeEmbeddings()
    )

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
    ``duplicate_title`` lint."""
    from dikw_core.domains.knowledge.page_index import persist_knowledge

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
        await persist_knowledge(
            storage=storage,
            root=wiki,
            path="knowledge/concepts/tesla.md",
        )
    finally:
        await storage.close()
    await ingest_wisdom_files(
        wiki, ["wisdom/elon-musk/tesla.md"], embedder=FakeEmbeddings()
    )

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
    that has no provenance row triggers ``missing_provenance``."""
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
    await ingest_wisdom_files(
        wiki, ["wisdom/elon-musk/from-bio.md"], embedder=FakeEmbeddings()
    )

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
