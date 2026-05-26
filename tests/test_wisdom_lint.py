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

from .fakes import FakeEmbeddings, init_test_wiki


def _drop_wisdom(wiki: Path, rel: str, body: str) -> None:
    p = wiki / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


@pytest.mark.asyncio
async def test_invalid_wisdom_status_lint_warns_but_ingest_succeeds(
    tmp_path: Path,
) -> None:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
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
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
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
