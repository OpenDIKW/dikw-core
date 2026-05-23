from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.providers import LLMResponse

from .fakes import FakeEmbeddings, FakeLLM, init_test_wiki

FIXTURES = Path(__file__).parent / "fixtures" / "notes"

_SCRIPT = {
    "sources/notes/dikw.md": (
        "<page path=\"wiki/concepts/dikw-pyramid.md\" type=\"concept\">\n"
        "---\ntags: [dikw, pyramid]\n---\n\n"
        "# DIKW pyramid\n\n"
        "The DIKW pyramid organises raw data into four layers. "
        "See [[Karpathy LLM Wiki]] for a related pattern.\n"
        "</page>"
    ),
    "sources/notes/karpathy-wiki.md": (
        "<page path=\"wiki/concepts/karpathy-llm-wiki.md\" type=\"concept\">\n"
        "---\ntags: [pattern, llm]\n---\n\n"
        "# Karpathy LLM Wiki\n\n"
        "Karpathy's pattern defines a wiki built from source documents. "
        "It complements the [[DIKW pyramid]] model.\n"
        "</page>"
    ),
    "sources/notes/retrieval.md": (
        "<page path=\"wiki/concepts/hybrid-retrieval.md\" type=\"concept\">\n"
        "---\ntags: [search]\n---\n\n"
        "# Hybrid retrieval\n\n"
        "BM25 + dense vectors fused with RRF. Useful background for the "
        "[[DIKW pyramid]] engine.\n"
        "</page>"
    ),
}


class ScriptedLLM:
    """Returns a canned <page> response keyed by which source appears in the prompt."""

    def __init__(self, script: dict[str, str]) -> None:
        self._script = script
        self.last_user: str | None = None

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list | None = None,
    ):
        self.last_user = user
        for src_path, resp in self._script.items():
            if src_path in user:
                return LLMResponse(text=resp, finish_reason="end_turn")
        raise AssertionError(f"no script entry matched prompt: {user[:200]}")


@pytest.fixture()
def wiki_with_fixtures(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    dest = wiki / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    return wiki


@pytest.mark.asyncio
async def test_synth_creates_linked_wiki_pages_and_clean_lint(
    wiki_with_fixtures: Path,
) -> None:
    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)

    llm = ScriptedLLM(_SCRIPT)
    report = await api.synthesize(wiki_with_fixtures, llm=llm, embedder=embedder)
    assert report.candidates == 3
    assert report.sources_processed == 3
    # Small fixtures fit in one chunk each → exactly one group per source.
    assert report.groups_processed == 3
    assert report.created == 3
    assert report.skipped == 0
    assert report.errors == 0

    # on-disk artefacts
    assert (wiki_with_fixtures / "wiki" / "concepts" / "dikw-pyramid.md").is_file()
    assert (wiki_with_fixtures / "wiki" / "concepts" / "karpathy-llm-wiki.md").is_file()
    assert (wiki_with_fixtures / "wiki" / "concepts" / "hybrid-retrieval.md").is_file()
    index_text = (wiki_with_fixtures / "wiki" / "index.md").read_text(encoding="utf-8")
    assert "DIKW pyramid" in index_text
    assert "Karpathy LLM Wiki" in index_text
    log_text = (wiki_with_fixtures / "wiki" / "log.md").read_text(encoding="utf-8")
    assert "synth" in log_text

    # Lint expectations:
    # - Each page references [[DIKW pyramid]] or [[Karpathy LLM Wiki]] which exist
    #   → no broken_wikilink.
    # - Hybrid retrieval page has no inbound wikilinks → should be reported as orphan.
    lint_report = await api.lint(wiki_with_fixtures)
    kinds = lint_report.by_kind()
    assert kinds.get("broken_wikilink", 0) == 0
    assert kinds.get("duplicate_title", 0) == 0
    assert kinds.get("orphan_page", 0) >= 1


@pytest.mark.asyncio
async def test_synth_is_idempotent_without_force_all(wiki_with_fixtures: Path) -> None:
    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)
    llm = ScriptedLLM(_SCRIPT)

    first = await api.synthesize(wiki_with_fixtures, llm=llm, embedder=embedder)
    assert first.created == 3

    second = await api.synthesize(wiki_with_fixtures, llm=llm, embedder=embedder)
    assert second.created == 0
    assert second.skipped == 3
    # Second pass shouldn't even invoke the LLM since every source was
    # marked synth-done in wiki_log.
    assert second.groups_processed == 0


@pytest.mark.asyncio
async def test_synth_empty_response_is_legal_zero_pages(
    wiki_with_fixtures: Path,
) -> None:
    """Stage A's prompt explicitly allows an empty response — "this section
    has nothing worth a wiki page". The pipeline must accept it cleanly,
    without counting an error."""
    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)

    llm = FakeLLM(response_text="this is not a page block")
    report = await api.synthesize(wiki_with_fixtures, llm=llm, embedder=embedder)
    assert report.errors == 0
    assert report.created == 0
    assert report.sources_processed == 3


@pytest.mark.asyncio
async def test_synth_counts_parse_failure_as_error(
    wiki_with_fixtures: Path,
) -> None:
    """When the LLM emits *page-shaped* output that fails to parse on
    every block, the group is an unrecoverable error → count it."""
    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)

    bad_block = (
        '<page path="wiki/notes/x.md" type="note">\n'
        "---\ntags: []\n---\n\n"
        "no atx title here, parser will reject\n"
        "</page>"
    )
    llm = FakeLLM(response_text=bad_block)
    report = await api.synthesize(wiki_with_fixtures, llm=llm, embedder=embedder)
    # One group per source, all three groups failed to parse → 3 errors.
    assert report.errors == 3
    assert report.created == 0
    assert report.sources_processed == 3


@pytest.mark.asyncio
async def test_synth_prompt_preserves_source_language(
    wiki_with_fixtures: Path,
) -> None:
    """Both the user prompt template (`prompts/synthesize.md`) and the
    hardcoded system prompt must instruct the LLM to preserve the source's
    dominant language. Without the rule the LLM defaults to English even
    on Chinese sources; the system prompt is a second-line defence in case
    the user prompt is later truncated under context-window pressure.
    """
    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)

    llm = FakeLLM(response_text="(no page worth writing)")
    await api.synthesize(wiki_with_fixtures, llm=llm, embedder=embedder)

    assert llm.last_user is not None, "synth must invoke the LLM at least once"
    assert "## Output language" in llm.last_user
    assert "dominant language" in llm.last_user
    assert "ASCII" in llm.last_user

    assert llm.last_system is not None
    assert "Preserve the dominant language" in llm.last_system


def _long_fixture_body() -> str:
    """A markdown body big enough to land across multiple ChunkGroups.

    Default ``target_tokens_per_group=3600``; each H2 block here packs
    ~600 tokens so 8 of them → ~4800 tokens → ≥ 2 groups.
    """
    chapters = []
    for i in range(8):
        chapters.append(f"## Chapter {i}\n\n")
        # ~30 paragraph repeats x ~20 token-equiv = ~600 tokens per chapter
        for j in range(30):
            chapters.append(
                f"This is paragraph {j} of chapter {i}, dense with content "
                f"about subject-{i} and its connection to subject-{(i + 1) % 8}. "
                "Each paragraph contributes roughly twenty tokens to the budget.\n\n"
            )
    return "# Long source\n\n" + "".join(chapters)


class GroupAwareLLM:
    """Returns one synthetic <page> per LLM call, keyed by call index.

    Lets the long-source test verify that the synth pipeline really makes
    multiple LLM calls for one source — a single-call ScriptedLLM can't
    distinguish "1 group" from "many".
    """

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list | None = None,
    ):
        idx = self.calls
        self.calls += 1
        text = (
            f'<page path="wiki/concepts/group-{idx}.md" type="concept">\n'
            f"---\ntags: [synthetic]\n---\n\n"
            f"# Group {idx} concept\n\n"
            f"Synthetic page emitted by the GroupAwareLLM on call {idx}.\n"
            f"</page>"
        )
        return LLMResponse(text=text, finish_reason="end_turn")


@pytest.mark.asyncio
async def test_synth_retries_source_with_failed_groups(
    wiki_with_fixtures: Path,
) -> None:
    """A source that hit a hard parse error must NOT be marked complete —
    next default ``synth`` should re-call the LLM (it may produce parseable
    output this time).
    """
    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)

    # First pass: every source returns a malformed <page> block → all
    # three groups raise SynthesisError → no source_done marker written.
    bad_block = (
        '<page path="wiki/notes/x.md" type="note">\n'
        "---\ntags: []\n---\n\n"
        "no atx title here\n"
        "</page>"
    )
    bad_llm = FakeLLM(response_text=bad_block)
    first = await api.synthesize(wiki_with_fixtures, llm=bad_llm, embedder=embedder)
    assert first.errors == 3
    assert first.created == 0

    # Second pass with a working LLM: all three sources should be
    # retried (not skipped), because no synth_source_done marker exists.
    good_llm = ScriptedLLM(_SCRIPT)
    second = await api.synthesize(wiki_with_fixtures, llm=good_llm, embedder=embedder)
    assert second.skipped == 0, "failed-group sources must not be skipped on retry"
    assert second.created == 3


@pytest.mark.asyncio
async def test_synth_skips_zero_page_source_on_second_run(
    wiki_with_fixtures: Path,
) -> None:
    """A source whose LLM legitimately emits zero <page> blocks must
    still be marked complete — re-running synth would just waste LLM
    calls hitting the same dead-end.
    """
    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)

    empty_llm = FakeLLM(response_text="this section has nothing worth a wiki page")
    first = await api.synthesize(wiki_with_fixtures, llm=empty_llm, embedder=embedder)
    assert first.errors == 0
    assert first.created == 0
    assert first.sources_processed == 3

    # Second pass: every source should be skipped — first pass marked
    # them done despite producing no pages.
    second = await api.synthesize(wiki_with_fixtures, llm=empty_llm, embedder=embedder)
    assert second.skipped == 3
    assert second.sources_processed == 0


@pytest.mark.asyncio
async def test_synth_uses_custom_page_types_end_to_end(tmp_path: Path) -> None:
    """``SchemaConfig.page_types`` propagates through prompt + parser +
    folder selection; an LLM emitting ``type="topic"`` lands under
    ``wiki/topics/`` when the schema declares ``topic`` as allowed.
    """
    from dikw_core.config import dump_config_yaml, load_config

    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    cfg_path = wiki / "dikw.yml"
    cfg = load_config(cfg_path)
    cfg.schema_.page_types = ["entity", "concept", "note", "topic"]
    cfg_path.write_text(dump_config_yaml(cfg), encoding="utf-8")

    sources_dir = wiki / "sources" / "notes"
    sources_dir.mkdir(parents=True, exist_ok=True)
    (sources_dir / "spacex.md").write_text(
        "# SpaceX\n\nAerospace firm founded by Elon Musk.\n",
        encoding="utf-8",
    )

    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    topic_response = (
        '<page path="wiki/topics/spacex.md" type="topic">\n'
        "---\ntags: [aerospace]\n---\n\n"
        "# SpaceX topic\n\n"
        "Aggregator page for [[Elon Musk]] and related rocket projects.\n"
        "</page>"
    )
    llm = FakeLLM(response_text=topic_response)
    report = await api.synthesize(wiki, llm=llm, embedder=embedder)

    assert report.created == 1
    topic_path = wiki / "wiki" / "topics" / "spacex.md"
    assert topic_path.is_file(), "custom page type 'topic' should land in wiki/topics/"
    body = topic_path.read_text(encoding="utf-8")
    assert "type: topic" in body


@pytest.mark.asyncio
async def test_synth_backfills_legacy_per_page_rows_on_first_post_upgrade_run(
    wiki_with_fixtures: Path,
) -> None:
    """Pre-fan-out synth only logged per-page rows. The first post-upgrade
    ``synth`` must promote those rows into ``synth_source_done`` markers
    so users don't pay to re-LLM their whole base on upgrade. Subsequent
    runs must NOT backfill again, so post-fan-out partial-failure rows
    are still respected as needing retry.
    """
    from dikw_core.schemas import WikiLogEntry

    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)

    # Seed legacy per-page rows directly via storage — simulates a base
    # synthesised by the pre-fan-out pipeline that just got upgraded.
    _cfg, _root, storage = await api._with_storage(wiki_with_fixtures)
    try:
        for src_path, dst_path in [
            ("sources/notes/dikw.md", "wiki/concepts/dikw-pyramid.md"),
            ("sources/notes/karpathy-wiki.md", "wiki/concepts/karpathy-llm-wiki.md"),
            ("sources/notes/retrieval.md", "wiki/concepts/hybrid-retrieval.md"),
        ]:
            await storage.append_wiki_log(
                WikiLogEntry(ts=1.0, action="synth", src=src_path, dst=dst_path)
            )
    finally:
        await storage.close()

    # Counting LLM proves backfill succeeded — if it ran, complete()
    # would be called for every source.
    class CountingLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(
            self,
            *,
            system: str,
            user: str,
            model: str,
            max_tokens: int = 4096,
            temperature: float = 0.2,
            tools: list | None = None,
        ):
            self.calls += 1
            return LLMResponse(text="", finish_reason="end_turn")

    counting_llm = CountingLLM()
    report = await api.synthesize(
        wiki_with_fixtures, llm=counting_llm, embedder=embedder
    )
    assert counting_llm.calls == 0, (
        f"backfill should skip all sources; LLM was called {counting_llm.calls}"
    )
    assert report.skipped == 3
    assert report.sources_processed == 0


@pytest.mark.asyncio
async def test_synth_does_not_backfill_when_sentinel_already_exists(
    wiki_with_fixtures: Path,
) -> None:
    """The sentinel records "fan-out pipeline has touched this base at
    least once". Once it exists, dst rows are *not* legacy data — they
    are post-fan-out partial failures that must be retried.
    """
    from dikw_core.api import _LEGACY_BACKFILL_SENTINEL
    from dikw_core.schemas import WikiLogEntry

    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)

    _cfg, _root, storage = await api._with_storage(wiki_with_fixtures)
    try:
        # Sentinel: this base has been through the fan-out pipeline
        # already. Any per-page rows present must be treated as
        # partial-failure leftovers, not legacy data.
        await storage.append_wiki_log(
            WikiLogEntry(
                ts=1.0,
                action="synth_source_done",
                src=_LEGACY_BACKFILL_SENTINEL,
            )
        )
        # Three sources have per-page rows but no per-source done marker.
        # Without backfill (sentinel blocks it), they should be retried.
        for src_path, dst_path in [
            ("sources/notes/dikw.md", "wiki/concepts/dikw.md"),
            ("sources/notes/karpathy-wiki.md", "wiki/concepts/karpathy.md"),
            ("sources/notes/retrieval.md", "wiki/concepts/retrieval.md"),
        ]:
            await storage.append_wiki_log(
                WikiLogEntry(ts=2.0, action="synth", src=src_path, dst=dst_path)
            )
    finally:
        await storage.close()

    llm = ScriptedLLM(_SCRIPT)
    report = await api.synthesize(wiki_with_fixtures, llm=llm, embedder=embedder)
    # Backfill MUST NOT fire — sentinel blocks it. Partial-failure dst
    # rows (which look identical to legacy rows on disk) get a real
    # retry.
    assert report.skipped == 0
    assert report.sources_processed == 3


@pytest.mark.asyncio
async def test_synth_writes_sentinel_even_when_no_legacy_rows(tmp_path: Path) -> None:
    """A brand-new base (no legacy data) must still write the sentinel
    on its first synth — otherwise a crash that leaves partial dst rows
    behind would be misread as legacy data on the next run.
    """
    from dikw_core.api import _LEGACY_BACKFILL_SENTINEL

    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    sources_dir = wiki / "sources" / "notes"
    sources_dir.mkdir(parents=True, exist_ok=True)
    (sources_dir / "x.md").write_text("# X\n\nbody\n", encoding="utf-8")

    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    llm = FakeLLM(response_text="(no page worth writing)")
    await api.synthesize(wiki, llm=llm, embedder=embedder)

    _cfg, _root, storage = await api._with_storage(wiki)
    try:
        entries = await storage.list_wiki_log()
    finally:
        await storage.close()
    sentinel_rows = [
        e for e in entries
        if e.action == "synth_source_done" and e.src == _LEGACY_BACKFILL_SENTINEL
    ]
    assert len(sentinel_rows) == 1, (
        "first synth on a fresh base must write exactly one sentinel row"
    )


@pytest.mark.asyncio
async def test_synth_rejects_source_with_changed_body_after_ingest(
    wiki_with_fixtures: Path,
) -> None:
    """If a source file is edited after ingest, its cached chunk offsets
    no longer match the on-disk body. Slicing the new body at stale
    offsets would silently drop appended content. Synth must detect the
    drift, refuse to process the source, and NOT mark it done.
    """
    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)

    # Mutate one of the source files on disk without re-ingesting.
    target = wiki_with_fixtures / "sources" / "notes" / "dikw.md"
    target.write_text(
        target.read_text(encoding="utf-8")
        + "\n\n## Newly appended section\n\nAdded after ingest.\n",
        encoding="utf-8",
    )

    llm = ScriptedLLM(_SCRIPT)
    report = await api.synthesize(wiki_with_fixtures, llm=llm, embedder=embedder)

    # The mutated source contributes 1 to errors, the other two run normally.
    assert report.errors == 1
    assert report.created == 2

    # Verify the stale source did NOT get a source_done marker → next
    # synth will retry it (after the user re-runs `dikw client ingest`).
    _cfg, _root, storage = await api._with_storage(wiki_with_fixtures)
    try:
        entries = await storage.list_wiki_log()
    finally:
        await storage.close()
    done_for_dikw = [
        e for e in entries
        if e.action == "synth_source_done" and e.src == "sources/notes/dikw.md"
    ]
    assert done_for_dikw == [], (
        "stale source must not be marked done; the user should re-ingest first"
    )


@pytest.mark.asyncio
async def test_long_source_fans_out_across_groups(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)
    sources_dir = wiki / "sources" / "notes"
    sources_dir.mkdir(parents=True, exist_ok=True)
    long_path = sources_dir / "long.md"
    long_path.write_text(_long_fixture_body(), encoding="utf-8")

    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    llm = GroupAwareLLM()
    report = await api.synthesize(wiki, llm=llm, embedder=embedder)

    # One source, but the long body must split into ≥ 2 groups → ≥ 2
    # LLM calls → ≥ 2 distinct synthetic pages persisted.
    assert report.sources_processed == 1
    assert report.groups_processed >= 2, (
        f"expected long fixture to fan out into multiple groups, "
        f"got groups_processed={report.groups_processed}"
    )
    assert report.created >= 2
    assert llm.calls == report.groups_processed

    # All produced wiki pages must point back to the same source — the
    # fan-out invariant: 1 source → N pages, all sourced from it.
    concepts_dir = wiki / "wiki" / "concepts"
    produced = list(concepts_dir.glob("group-*.md"))
    assert len(produced) >= 2
    for p in produced:
        text = p.read_text(encoding="utf-8")
        assert "sources:" in text
        assert "sources/notes/long.md" in text
