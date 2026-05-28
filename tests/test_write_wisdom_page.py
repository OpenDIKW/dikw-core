"""Engine-layer tests for :func:`api.write_wisdom_page`.

The wisdom write API persists a single hand-authored wisdom page from
structured input (slug + title + body + optional author/status/tags/
sources/extras), writing the file to disk and indexing it through
``persist_page(layer=Layer.WISDOM)``. Sister to ``api.ingest``'s
wisdom branch but for a single page, so an agent caller can create or
update one wisdom note and have it immediately retrievable.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.config import load_config
from dikw_core.domains.data.path_norm import doc_id_for
from dikw_core.domains.wisdom import write_wisdom_file
from dikw_core.schemas import Layer, WisdomStatus
from dikw_core.storage import Storage, build_storage

from .fakes import FakeEmbeddings, init_test_base, register_text_version, seed_doc


async def _open_storage(wiki: Path) -> Storage:
    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    return storage


@pytest.mark.asyncio
async def test_write_basic_no_author(tmp_path: Path) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    report = await api.write_wisdom_page(
        wiki,
        slug="first-principles",
        title="First Principles",
        body="Reason from physics, not analogy.\n",
        embedder=FakeEmbeddings(),
    )

    assert report.path == "wisdom/first-principles.md"
    assert report.created is True
    assert report.chunks >= 1
    assert (wiki / "wisdom" / "first-principles.md").is_file()

    storage = await _open_storage(wiki)
    try:
        doc = await storage.get_document(
            doc_id_for(Layer.WISDOM, "wisdom/first-principles.md")
        )
    finally:
        await storage.close()
    assert doc is not None
    assert doc.layer == Layer.WISDOM
    assert doc.title == "First Principles"
    assert doc.status is None


@pytest.mark.asyncio
async def test_write_with_author(tmp_path: Path) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    report = await api.write_wisdom_page(
        wiki,
        author="elon-musk",
        slug="first-principles",
        title="First Principles",
        body="Reason from physics.\n",
        embedder=FakeEmbeddings(),
    )

    assert report.path == "wisdom/elon-musk/first-principles.md"
    assert report.created is True
    abs_path = wiki / "wisdom" / "elon-musk" / "first-principles.md"
    assert abs_path.is_file(), "author subdir must be auto-created"


@pytest.mark.asyncio
async def test_write_upsert_marks_updated(tmp_path: Path) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    r1 = await api.write_wisdom_page(
        wiki,
        slug="x",
        title="X",
        body="first body.\n",
        embedder=FakeEmbeddings(),
    )
    assert r1.created is True
    h1 = r1.hash

    r2 = await api.write_wisdom_page(
        wiki,
        slug="x",
        title="X",
        body="second body — different.\n",
        embedder=FakeEmbeddings(),
    )
    assert r2.created is False
    assert r2.hash != h1


@pytest.mark.asyncio
async def test_write_status_only_change_reindexes(tmp_path: Path) -> None:
    """Same body bytes + flipped status must still update the status
    column. ``write_wisdom_page`` always calls ``persist_page`` (no
    body-hash short-circuit) so an explicit write is the user's
    contract to refresh the row, even if only frontmatter changed.
    """
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    await api.write_wisdom_page(
        wiki,
        slug="edit-me",
        title="Edit Me",
        body="stable body.\n",
        status=WisdomStatus.DRAFT,
        embedder=FakeEmbeddings(),
    )
    await api.write_wisdom_page(
        wiki,
        slug="edit-me",
        title="Edit Me",
        body="stable body.\n",
        status=WisdomStatus.PUBLISHED,
        embedder=FakeEmbeddings(),
    )

    storage = await _open_storage(wiki)
    try:
        doc = await storage.get_document(
            doc_id_for(Layer.WISDOM, "wisdom/edit-me.md")
        )
    finally:
        await storage.close()
    assert doc is not None
    assert doc.status == WisdomStatus.PUBLISHED


@pytest.mark.asyncio
async def test_write_wikilink_resolves_to_existing_knowledge_page(
    tmp_path: Path,
) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    await seed_doc(
        wiki,
        layer=Layer.KNOWLEDGE,
        path="knowledge/concepts/tesla.md",
        body="---\ntitle: Tesla\n---\n# Tesla\n\nthe company.\n",
        title="Tesla",
    )

    report = await api.write_wisdom_page(
        wiki,
        author="elon-musk",
        slug="never-sell",
        title="Never Sell",
        body="See [[Tesla]] for context.\n",
        embedder=FakeEmbeddings(),
    )
    assert report.unresolved_wikilinks == 0

    storage = await _open_storage(wiki)
    try:
        wisdom_id = doc_id_for(Layer.WISDOM, "wisdom/elon-musk/never-sell.md")
        edges = await storage.links_from(wisdom_id)
    finally:
        await storage.close()
    assert any(e.dst_path.endswith("knowledge/concepts/tesla.md") for e in edges), edges


@pytest.mark.asyncio
async def test_write_unresolved_wikilink_counted(tmp_path: Path) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    report = await api.write_wisdom_page(
        wiki,
        slug="floating",
        title="Floating",
        body="Refers [[Nonexistent Page]] that never existed.\n",
        embedder=FakeEmbeddings(),
    )
    assert report.unresolved_wikilinks == 1


@pytest.mark.asyncio
async def test_write_sources_populate_provenance(tmp_path: Path) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    src_dir = wiki / "sources" / "notes"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "musk-bio.md").write_text("# Musk Bio\n\nfacts.\n", encoding="utf-8")

    await api.write_wisdom_page(
        wiki,
        author="elon-musk",
        slug="from-bio",
        title="From Bio",
        body="body referring back to the bio.\n",
        sources=["sources/notes/musk-bio.md"],
        embedder=FakeEmbeddings(),
    )

    storage = await _open_storage(wiki)
    try:
        wisdom_id = doc_id_for(Layer.WISDOM, "wisdom/elon-musk/from-bio.md")
        edges = await storage.provenance_from(wisdom_id)
    finally:
        await storage.close()
    assert [e.source_path for e in edges] == ["sources/notes/musk-bio.md"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "slug,author",
    [
        ("Foo Bar", None),       # space
        ("foo_bar", None),       # underscore
        ("FooBar", None),        # uppercase
        ("foo--bar", None),      # double hyphen
        ("-foo", None),          # leading hyphen
        ("foo-", None),          # trailing hyphen
        ("good", "ElonMusk"),    # author uppercase
        ("good", "elon musk"),   # author space
        ("good", "elon_musk"),   # author underscore
    ],
)
async def test_write_rejects_non_kebab_slug_or_author(
    tmp_path: Path, slug: str, author: str | None
) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    with pytest.raises(ValueError):
        await api.write_wisdom_page(
            wiki,
            slug=slug,
            author=author,
            title="T",
            body="b.\n",
            embedder=FakeEmbeddings(),
        )


@pytest.mark.asyncio
async def test_write_no_embed_skips_embedder(tmp_path: Path) -> None:
    """``no_embed=True`` writes the file + indexes chunks/links but
    never calls the embedder. The next ``dikw ingest`` will pick up
    embeddings via the missing-embedding resume scan."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    fake = FakeEmbeddings()
    embed_calls_before = getattr(fake, "embed_calls", 0)

    report = await api.write_wisdom_page(
        wiki,
        slug="lazy",
        title="Lazy",
        body="body.\n",
        no_embed=True,
        embedder=fake,
    )
    assert report.embedded == 0
    # FakeEmbeddings has no counter; we verify embedded == 0 above. The
    # contract is: with no_embed=True, the report's embedded count is 0
    # regardless of whether an embedder was passed in.
    del embed_calls_before


@pytest.mark.asyncio
async def test_write_with_embedder_counts_embedded_chunks(tmp_path: Path) -> None:
    # Pre-register an active text version. From 0.4.0 ``write_wisdom_page``
    # reuses the active version rather than registering-and-activating
    # a new one (avoids flipping is_active and stranding existing
    # source / knowledge vectors); without an active version the
    # embedder is dropped and the chunks defer to the next ingest's
    # resume scan.
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    from dikw_core.storage.sqlite import SQLiteStorage

    storage = SQLiteStorage(wiki / ".dikw" / "index.sqlite")
    await storage.connect()
    await storage.migrate()
    try:
        await register_text_version(storage)
    finally:
        await storage.close()

    report = await api.write_wisdom_page(
        wiki,
        slug="embedded",
        title="Embedded",
        body=(
            "First paragraph with enough content.\n\n"
            "Second paragraph also with enough body.\n"
        ),
        embedder=FakeEmbeddings(),
    )
    # Every chunk in the page gets embedded; embedded == chunks.
    assert report.chunks >= 1
    assert report.embedded == report.chunks


@pytest.mark.asyncio
async def test_write_emits_wisdom_log_entry(tmp_path: Path) -> None:
    """Like ingest, a wisdom write appends a ``knowledge_log`` row so the
    base's activity log records the write event."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    await api.write_wisdom_page(
        wiki,
        author="elon-musk",
        slug="logged",
        title="Logged",
        body="body.\n",
        embedder=FakeEmbeddings(),
    )

    storage = await _open_storage(wiki)
    try:
        entries = await storage.list_knowledge_log()
    finally:
        await storage.close()
    assert any(
        e.action == "wisdom_write" and e.src == "wisdom/elon-musk/logged.md"
        for e in entries
    ), [(e.action, e.src) for e in entries]


@pytest.mark.asyncio
async def test_write_tags_in_frontmatter(tmp_path: Path) -> None:
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    await api.write_wisdom_page(
        wiki,
        slug="tagged",
        title="Tagged",
        body="body.\n",
        tags=["mental-model", "physics"],
        embedder=FakeEmbeddings(),
    )

    text = (wiki / "wisdom" / "tagged.md").read_text(encoding="utf-8")
    assert "tags:" in text
    assert "mental-model" in text
    assert "physics" in text


@pytest.mark.asyncio
async def test_write_status_frontmatter_persisted_to_disk(tmp_path: Path) -> None:
    """The status enum value must serialize into the on-disk frontmatter
    so opening the file in Obsidian round-trips the field."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    await api.write_wisdom_page(
        wiki,
        slug="draft-page",
        title="Draft Page",
        body="body.\n",
        status=WisdomStatus.DRAFT,
        embedder=FakeEmbeddings(),
    )

    text = (wiki / "wisdom" / "draft-page.md").read_text(encoding="utf-8")
    assert "status: draft" in text


@pytest.mark.asyncio
async def test_extras_cannot_override_reserved_keys(tmp_path: Path) -> None:
    """``extras`` is a passthrough for caller-supplied frontmatter, but
    must not silently overwrite the validated ``title`` / ``status`` /
    ``tags`` / ``sources`` fields. A request that names ``title`` in
    both ``title=`` and ``extras={"title": ...}`` is otherwise free to
    desync the on-disk frontmatter from the storage row (which always
    stores the validated ``title``)."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    await api.write_wisdom_page(
        wiki,
        slug="reserved",
        title="Real Title",
        body="body.\n",
        status=WisdomStatus.DRAFT,
        tags=["real-tag"],
        sources=["real/source.md"],
        extras={
            "title": "EVIL Title",
            "status": "not-a-status",
            "tags": ["evil-tag"],
            "sources": ["evil/source.md"],
            "custom_key": "ok-to-pass-through",
        },
        embedder=FakeEmbeddings(),
    )

    text = (wiki / "wisdom" / "reserved.md").read_text(encoding="utf-8")
    assert "Real Title" in text
    assert "EVIL Title" not in text
    assert "status: draft" in text
    assert "not-a-status" not in text
    assert "real-tag" in text
    assert "evil-tag" not in text
    assert "real/source.md" in text
    assert "evil/source.md" not in text
    # Non-reserved extras keys still pass through:
    assert "custom_key: ok-to-pass-through" in text


def test_write_wisdom_file_rejects_path_escape(tmp_path: Path) -> None:
    """The low-level file writer must refuse a ``logical_path`` that
    resolves outside ``root`` — defense in depth for any direct caller
    that bypasses :func:`make_wisdom_path`."""
    root = tmp_path / "knowledge"
    root.mkdir()
    with pytest.raises(ValueError, match="outside"):
        write_wisdom_file(
            root,
            logical_path="../escape.md",
            title="x",
            body="x\n",
        )
    with pytest.raises(ValueError, match="outside"):
        write_wisdom_file(
            root,
            logical_path="wisdom/../../escape.md",
            title="x",
            body="x\n",
        )


@pytest.mark.asyncio
async def test_concurrent_writes_same_path_serialize(tmp_path: Path) -> None:
    """Two concurrent ``write_wisdom_page`` calls for the same
    ``(author, slug)`` must not both observe an empty document row and
    both report ``created=True``; the writes must be serialised so
    exactly one creates and any subsequent calls see the existing row."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    async def write(body: str) -> object:
        return await api.write_wisdom_page(
            wiki,
            slug="hot-path",
            title="Hot Path",
            body=body,
            embedder=FakeEmbeddings(),
        )

    r1, r2 = await asyncio.gather(write("first\n"), write("second\n"))
    # At least one of the two must observe the other had already
    # created the row. With no lock, both can race past
    # ``get_document`` before either writes, and both claim created.
    assert (r1.created, r2.created).count(True) == 1
    assert (r1.created, r2.created).count(False) == 1


@pytest.mark.asyncio
async def test_concurrent_writes_aliased_paths_share_lock(tmp_path: Path) -> None:
    """Two concurrent writes targetting the same canonical base via
    different path expressions (e.g. base dir vs ``dikw.yml`` file)
    must still be serialised — the lock key has to canonicalise the
    base before stringifying, otherwise the per-path lock degrades to a
    per-input-string lock and two API callers using different aliases
    can both observe ``created=True``."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    # Alias 1: directory; alias 2: the dikw.yml file inside it.
    # ``resolve_base_root`` accepts either and walks back to the same
    # base directory.
    config_path = wiki / "dikw.yml"
    assert config_path.is_file()

    async def write(path_arg: Path, body: str) -> object:
        return await api.write_wisdom_page(
            path_arg,
            slug="alias-race",
            title="Alias Race",
            body=body,
            embedder=FakeEmbeddings(),
        )

    r1, r2 = await asyncio.gather(
        write(wiki, "from-dir\n"), write(config_path, "from-yml\n")
    )
    assert (r1.created, r2.created).count(True) == 1
    assert (r1.created, r2.created).count(False) == 1


@pytest.mark.asyncio
async def test_update_excludes_own_old_title_from_resolve(
    tmp_path: Path,
) -> None:
    """When an update changes a wisdom page's title, the body's
    ``[[Old Title]]`` references must NOT resolve back to the page
    itself via the stale storage row — the cross-layer title index
    must exclude the document we're about to overwrite."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    # First write with the old title.
    await api.write_wisdom_page(
        wiki,
        slug="renamed",
        title="Old Title",
        body="initial body.\n",
        embedder=FakeEmbeddings(),
    )

    # Now rewrite the same page with a new title, and link to "Old
    # Title" in the body. After the write, that wikilink should be
    # UNresolved — "Old Title" is no longer the title of any page.
    report = await api.write_wisdom_page(
        wiki,
        slug="renamed",
        title="New Title",
        body="references [[Old Title]] which no longer exists.\n",
        embedder=FakeEmbeddings(),
    )
    assert report.created is False
    assert report.unresolved_wikilinks >= 1


@pytest.mark.asyncio
async def test_extras_cannot_corrupt_file_via_post_kwargs(tmp_path: Path) -> None:
    """``extras`` is **not** unpacked into ``frontmatter.Post(**kwargs)``
    so a key matching a Post constructor parameter (``handler``,
    ``content``) cannot corrupt the on-disk file. With the naive
    ``**meta`` expansion, ``extras={"handler": "evil"}`` collapses the
    entire file to the literal string ``evil`` (frontmatter.dumps
    treats the handler kwarg as a custom dump handler) — silently
    destroying title, body, status, tags."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    await api.write_wisdom_page(
        wiki,
        slug="kwarg-attack",
        title="Real Title",
        body="real body content\n",
        extras={"handler": "evil", "content": "also evil"},
        embedder=FakeEmbeddings(),
    )

    text = (wiki / "wisdom" / "kwarg-attack.md").read_text(encoding="utf-8")
    # File must contain the validated frontmatter + body, not the
    # extras["handler"] / extras["content"] payload as raw content.
    assert "Real Title" in text
    assert "real body content" in text
    # Neither of the two reserved Post kwargs may have replaced the body.
    assert text.strip() != "evil"
    assert text.strip() != "also evil"


@pytest.mark.asyncio
async def test_extras_cannot_override_author_frontmatter(
    tmp_path: Path,
) -> None:
    """``author`` is encoded into the on-disk path
    (``wisdom/<author>/<slug>.md``); ``extras={"author": "other"}``
    must not silently put a contradicting ``author`` into the
    frontmatter."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)

    await api.write_wisdom_page(
        wiki,
        author="elon-musk",
        slug="authored",
        title="Authored",
        body="body.\n",
        extras={"author": "imposter"},
        embedder=FakeEmbeddings(),
    )

    text = (wiki / "wisdom" / "elon-musk" / "authored.md").read_text(
        encoding="utf-8"
    )
    assert "imposter" not in text


@pytest.mark.asyncio
async def test_empty_body_rejected_by_schema(tmp_path: Path) -> None:
    """An empty (or whitespace-only) body would index zero chunks and
    silently wipe an existing page's content on upsert. Pydantic
    rejects the submission at the validation boundary so HTTP/CLI
    callers see 422 instead of an apparently-succeeded but
    retrieval-invisible page."""
    from pydantic import ValidationError

    from dikw_core.schemas import WisdomWriteSubmit

    for bad_body in ("", " ", "\n\n\t  "):
        with pytest.raises(ValidationError) as exc_info:
            WisdomWriteSubmit(slug="x", title="X", body=bad_body)
        assert "body" in str(exc_info.value)


@pytest.mark.asyncio
async def test_persist_failure_deactivates_document(tmp_path: Path) -> None:
    """A mid-pipeline failure inside ``_persist_page`` must
    deactivate the document row so the next ingest rebuilds it
    end-to-end, instead of leaving a partial doc with stale
    links/provenance. Mirrors the ingest wisdom branch recovery
    behaviour at api.py:1433."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki)
    # Pre-register an active text version so write_wisdom_page actually
    # exercises the inline-embed path (the 0.4.0 reuse-active-version
    # contract drops the embedder when no active version exists).
    from dikw_core.storage.sqlite import SQLiteStorage

    storage = SQLiteStorage(wiki / ".dikw" / "index.sqlite")
    await storage.connect()
    await storage.migrate()
    try:
        await register_text_version(storage)
    finally:
        await storage.close()

    # First write a normal page so a doc row exists.
    await api.write_wisdom_page(
        wiki,
        slug="will-fail",
        title="Will Fail",
        body="initial body.\n",
        embedder=FakeEmbeddings(),
    )

    # Inject an embedder that raises mid-stream — simulates a provider
    # error after upsert_document + replace_chunks have already
    # committed the new content.
    class BrokenEmbedder:
        dim = 8
        normalize = True
        distance = "cosine"
        model = "broken"
        revision = ""
        modality = "text"

        async def embed(self, texts, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated embedder failure")

    with pytest.raises(RuntimeError, match="simulated embedder failure"):
        await api.write_wisdom_page(
            wiki,
            slug="will-fail",
            title="Will Fail v2",
            body="new body that will fail to embed.\n",
            embedder=BrokenEmbedder(),  # type: ignore[arg-type]
        )

    # The document row must be deactivated so retry rebuilds it.
    storage = await _open_storage(wiki)
    try:
        doc = await storage.get_document(
            doc_id_for(Layer.WISDOM, "wisdom/will-fail.md")
        )
    finally:
        await storage.close()
    assert doc is not None
    assert doc.active is False, (
        "persist_page failure must deactivate the doc so the next "
        "ingest resume scan can rebuild it"
    )
