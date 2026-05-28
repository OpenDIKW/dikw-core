"""Tests for ``run_lint`` — focused on the ``non_atomic_page`` heuristic.

The other three issue kinds (``broken_wikilink`` / ``orphan_page`` /
``duplicate_title``) are exercised end-to-end by
``test_synthesize_pipeline.py``; this file pins down the atomicity
backstop in isolation so threshold tweaks are easy to validate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.domains.knowledge.lint import run_lint
from dikw_core.domains.knowledge.page import build_page, write_page
from dikw_core.schemas import DocumentRecord, Layer

from .fakes import init_test_base


async def _seed_page(
    *,
    base_root: Path,
    title: str,
    body: str,
    type_: str = "concept",
    tags: list[str] | None = None,
    path: str | None = None,
) -> str:
    """Write ``page`` to disk + register it in storage so lint can see it."""
    page = build_page(
        title=title,
        body=body,
        type_=type_,
        tags=list(tags or []),
        sources=[],
        path=path,
        extras={},
    )
    write_page(base_root, page)

    _cfg, _root, storage = await api._with_storage(base_root)
    try:
        await storage.upsert_document(
            DocumentRecord(
                doc_id=f"K:{page.path}",
                path=page.path,
                title=page.title,
                hash=f"hash-{page.path}",
                mtime=0.0,
                layer=Layer.KNOWLEDGE,
                active=True,
            )
        )
    finally:
        await storage.close()
    return page.path


async def _run_lint(base_root: Path):
    _cfg, root, storage = await api._with_storage(base_root)
    try:
        return await run_lint(storage, root=root)
    finally:
        await storage.close()


@pytest.fixture()
def empty_wiki(tmp_path: Path) -> Path:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    return wiki


@pytest.mark.asyncio
async def test_atomic_page_emits_no_non_atomic_issue(empty_wiki: Path) -> None:
    body = (
        "# Tight Page\n\n"
        "Two short paragraphs about a single subject.\n\n"
        "Linked to [[Some Other Concept]] for context.\n"
    )
    await _seed_page(base_root=empty_wiki, title="Tight Page", body=body)
    report = await _run_lint(empty_wiki)
    kinds = report.by_kind()
    assert kinds.get("non_atomic_page", 0) == 0


@pytest.mark.asyncio
async def test_long_body_triggers_non_atomic(empty_wiki: Path) -> None:
    body = "# Long Page\n\n" + ("paragraph filler " * 200) + "\n"
    await _seed_page(base_root=empty_wiki, title="Long Page", body=body)
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert len(issues) == 1
    assert "body" in issues[0].detail and "chars" in issues[0].detail


@pytest.mark.asyncio
async def test_many_h2_sections_trigger_non_atomic(empty_wiki: Path) -> None:
    body = (
        "# Multi-section Page\n\n"
        "intro\n\n"
        "## Section One\n\nbody\n\n"
        "## Section Two\n\nbody\n\n"
        "## Section Three\n\nbody\n\n"
        "## Section Four\n\nbody\n"
    )
    await _seed_page(
        base_root=empty_wiki, title="Multi-section Page", body=body
    )
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert len(issues) == 1
    assert "H2 sections" in issues[0].detail


@pytest.mark.asyncio
async def test_many_wikilinks_trigger_non_atomic(empty_wiki: Path) -> None:
    """16 distinct wikilink targets in one short body — true MOC-style
    aggregation, not a single-event entity-rich page (those routinely
    cite 8-12 distinct entities)."""
    body = (
        "# Hub Page\n\n"
        "References [[A]], [[B]], [[C]], [[D]], [[E]], [[F]], "
        "[[G]], [[H]], [[I]], [[J]], [[K]], [[L]], [[M]], "
        "[[N]], [[O]], [[P]] all in one breath.\n"
    )
    await _seed_page(base_root=empty_wiki, title="Hub Page", body=body)
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert len(issues) == 1
    assert "wikilinks" in issues[0].detail


@pytest.mark.asyncio
async def test_repeated_wikilink_target_does_not_trigger(empty_wiki: Path) -> None:
    """A biography that mentions the same entity 20 times is atomic
    about that entity, not a 20-topic page. The heuristic counts
    *distinct* targets so this case stays clean."""
    body = (
        "# Biography\n\n"
        + " ".join(["[[Elon Musk]]"] * 20)
        + " was discussed throughout the source.\n"
    )
    await _seed_page(base_root=empty_wiki, title="Biography", body=body)
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert issues == []


@pytest.mark.asyncio
async def test_scalar_tags_value_does_not_crash(empty_wiki: Path) -> None:
    """Knowledge pages are user-editable; a hand-written ``tags: 2024``
    parses as a scalar int, not a list. Lint must tolerate the drift
    instead of raising TypeError on iteration."""
    page_path = empty_wiki / "knowledge" / "concepts" / "scalar-tag.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(
        "---\n"
        "id: K-scalar-tag\n"
        "type: concept\n"
        "title: Scalar Tag\n"
        "tags: 2024\n"
        "---\n\n"
        "# Scalar Tag\n\nA page with a scalar tags field.\n",
        encoding="utf-8",
    )
    rel_path = "knowledge/concepts/scalar-tag.md"
    _cfg, _root, storage = await api._with_storage(empty_wiki)
    try:
        await storage.upsert_document(
            DocumentRecord(
                doc_id=f"K:{rel_path}",
                path=rel_path,
                title="Scalar Tag",
                hash=f"hash-{rel_path}",
                mtime=0.0,
                layer=Layer.KNOWLEDGE,
                active=True,
            )
        )
    finally:
        await storage.close()
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert issues == []


@pytest.mark.asyncio
async def test_fenced_code_headings_do_not_trigger(empty_wiki: Path) -> None:
    """Comments inside ```` ``` ```` fences (``# install`` /
    ``## setup``) must not be counted toward H1/H2 totals — otherwise
    a single-topic technical note with a shell snippet false-flags."""
    body = (
        "# Setup Instructions\n\n"
        "```bash\n"
        "# install deps\n"
        "uv sync\n"
        "## setup db\n"
        "createdb foo\n"
        "## seed data\n"
        "psql -f seed.sql\n"
        "## run tests\n"
        "uv run pytest\n"
        "## bonus\n"
        "echo done\n"
        "```\n\n"
        "Single coherent topic.\n"
    )
    await _seed_page(base_root=empty_wiki, title="Setup Instructions", body=body)
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert issues == []


@pytest.mark.asyncio
async def test_multiple_h1_triggers_non_atomic(empty_wiki: Path) -> None:
    """Multiple H1 headers on one page is the canonical "two atomic
    notes glued together" pattern — bilingual duplicates (CN + EN
    versions of the same biography), or two unrelated subjects sharing
    a frontmatter. Body-length alone misses short bilingual pages;
    H1-count catches them deterministically."""
    body = (
        "# 乔舒亚\n\nFirst version of the biography.\n\n"
        "---\n\n"
        "# Joshua\n\nSecond version of the same biography.\n"
    )
    await _seed_page(base_root=empty_wiki, title="Joshua", body=body)
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert len(issues) == 1
    assert "H1 sections" in issues[0].detail


@pytest.mark.asyncio
async def test_event_page_with_many_entities_does_not_trigger(empty_wiki: Path) -> None:
    """Event pages routinely cite 8-12 entities (e.g. ``Tesla 2006
    funding round`` listing all participants) — these are atomic by
    topic but entity-rich. The threshold of 15 keeps them clean."""
    body = (
        "# Tesla 2006 Funding Round\n\n"
        "[[Elon Musk]], [[Kimbal Musk]], [[Marc Tarpenning]], "
        "[[Antonio Gracias]], [[Sergey Brin]], [[Larry Page]], "
        "[[Jeff Skoll]], [[Nick Pritzker]], [[Steve Jurvetson]], "
        "[[Sequoia Capital]], [[VantagePoint Capital Partners]] all "
        "took part in the round around [[Tesla Roadster]] development.\n"
    )
    await _seed_page(base_root=empty_wiki, title="Tesla 2006 Funding Round", body=body)
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert issues == []


@pytest.mark.asyncio
async def test_tags_across_domains_trigger_non_atomic(empty_wiki: Path) -> None:
    """A page tagged across two top-level namespaces (ml/* and biz/*) is
    almost certainly N atomic notes glued together. The Stage A prompt
    can't easily catch this — lint as a backstop must."""
    body = (
        "# Cross-Domain Page\n\n"
        "Discusses both ML research and startup operations in one breath.\n"
    )
    await _seed_page(
        base_root=empty_wiki,
        title="Cross-Domain Page",
        body=body,
        tags=["ml/research", "biz/startup"],
    )
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert len(issues) == 1
    assert "tags span" in issues[0].detail


@pytest.mark.asyncio
async def test_tags_single_domain_does_not_trigger(empty_wiki: Path) -> None:
    """Multiple tags within ONE top-level namespace is normal taxonomy
    practice (ml/research + ml/eval), not a cross-domain violation."""
    body = "# Single-Domain Page\n\nA focused page on ML research.\n"
    await _seed_page(
        base_root=empty_wiki,
        title="Single-Domain Page",
        body=body,
        tags=["ml/research", "ml/eval"],
    )
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert issues == []


@pytest.mark.asyncio
async def test_no_tags_does_not_trigger_tag_violation(empty_wiki: Path) -> None:
    """Empty tag list must never trigger the cross-domain heuristic."""
    body = "# Untagged Page\n\nNo frontmatter tags.\n"
    await _seed_page(base_root=empty_wiki, title="Untagged Page", body=body, tags=[])
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert issues == []


@pytest.mark.asyncio
async def test_flat_tags_only_do_not_trigger(empty_wiki: Path) -> None:
    """Flat tags (no ``/``) are *ignored* by the cross-domain heuristic.
    LLM-generated atomic pages routinely carry 3-5 flat tags
    (``entrepreneur, biography, spacex, tesla``); counting each as its
    own domain would produce false positives, so the heuristic only
    fires when the wiki has actually adopted namespaced tags."""
    body = "# Flat-Tagged Page\n\nA real-world atomic page with flat tags.\n"
    await _seed_page(
        base_root=empty_wiki,
        title="Flat-Tagged Page",
        body=body,
        tags=["entrepreneur", "biography", "spacex", "tesla"],
    )
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert issues == []


@pytest.mark.asyncio
async def test_namespaced_plus_flat_only_counts_namespaced(empty_wiki: Path) -> None:
    """A mix of namespaced (``ml/research``) and flat (``biography``)
    tags only counts namespaced ones toward the domain set. Single
    namespaced domain + arbitrary flat tags is still atomic."""
    body = "# Mixed Tags Page\n\nNamespaced + flat together.\n"
    await _seed_page(
        base_root=empty_wiki,
        title="Mixed Tags Page",
        body=body,
        tags=["ml/research", "biography", "spacex"],
    )
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert issues == []


@pytest.mark.asyncio
async def test_multiple_violations_collapse_to_one_issue(empty_wiki: Path) -> None:
    """A page that trips multiple heuristics should yield ONE issue with
    all violations in detail — avoid drowning the report in duplicates."""
    body = (
        "# Mega Page\n\n"
        + ("paragraph filler " * 200)
        + "\n\n## A\n\nbody\n\n## B\n\nbody\n\n## C\n\nbody\n\n## D\n\nbody\n\n"
        + "Links: [[a]] [[b]] [[c]] [[d]] [[e]] [[f]] [[g]] [[h]] "
        + "[[i]] [[j]] [[k]] [[l]] [[m]] [[n]] [[o]] [[p]].\n"
    )
    await _seed_page(base_root=empty_wiki, title="Mega Page", body=body)
    report = await _run_lint(empty_wiki)
    issues = [i for i in report.issues if i.kind == "non_atomic_page"]
    assert len(issues) == 1
    assert "body" in issues[0].detail
    assert "H2 sections" in issues[0].detail
    assert "wikilinks" in issues[0].detail


# ---- broken_wikilink shares fuzzy/collision semantics with resolve_links ----


@pytest.mark.asyncio
async def test_lint_broken_wikilink_uses_fuzzy_resolve(empty_wiki: Path) -> None:
    """``run_lint`` must reuse the same fuzzy normalize that engine
    persistence uses; otherwise it cries ``broken_wikilink`` over a
    plural variant that storage already resolved correctly. Pre-fix
    lint did exact + case-insensitive only, generating false positives
    for every ``[[Networks]]``-style link."""
    await _seed_page(
        base_root=empty_wiki,
        title="Neural Network",
        body="# Neural Network\n\nA layered model.\n",
    )
    await _seed_page(
        base_root=empty_wiki,
        title="Citation Page",
        body="# Citation Page\n\nSee [[Neural Networks]] for context.\n",
    )
    report = await _run_lint(empty_wiki)
    broken = [i for i in report.issues if i.kind == "broken_wikilink"]
    assert broken == []


@pytest.mark.asyncio
async def test_lint_broken_wikilink_reports_fuzzy_collision(
    empty_wiki: Path,
) -> None:
    """Mirror invariant: when fuzzy normalize collides on two distinct
    page paths, lint must report the broken link instead of silently
    accepting whichever case-insensitive lookup happens to match first.
    Pre-fix the case-insensitive shortcut hid this ambiguity entirely."""
    await _seed_page(
        base_root=empty_wiki,
        title="Tesla",
        body="# Tesla\n\nCar company.\n",
        path="knowledge/entities/tesla-company.md",
    )
    await _seed_page(
        base_root=empty_wiki,
        title="tesla",
        body="# tesla\n\nSI unit of magnetic flux density.\n",
        path="knowledge/concepts/tesla-unit.md",
    )
    await _seed_page(
        base_root=empty_wiki,
        title="Citation Page",
        body="# Citation Page\n\nNote about [[TESLA]] in the Maxwell paper.\n",
    )
    report = await _run_lint(empty_wiki)
    broken = [i for i in report.issues if i.kind == "broken_wikilink"]
    assert any("TESLA" in i.detail for i in broken)

