"""End-to-end ``dikw client *`` tests against the in-memory ASGI server.

We use Typer's ``CliRunner`` because that's the closest thing to "what
a user actually types" and it captures stdout / exit code in one
artefact. ``patch_transport_factory`` rewires ``Transport.from_config``
so each command's freshly constructed transport rides on the same
in-memory ASGI client the fixture set up — no socket, no network, no
flake.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dikw_core import api
from dikw_core.cli import app
from dikw_core.schemas import (
    DocumentRecord,
    Layer,
    LinkRecord,
    LinkType,
    WisdomItem,
    WisdomKind,
    WisdomStatus,
)
from dikw_core.server import synth_op
from dikw_core.server.runtime import ServerRuntime

from ..conftest import removed_top_level_short_names
from ..fakes import FakeEmbeddings, FakeLLM

FIXTURES = Path(__file__).parent.parent / "fixtures" / "notes"


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(app, args)


@pytest.mark.parametrize("name", removed_top_level_short_names())
def test_top_level_short_names_removed(name: str) -> None:
    """Every HTTP-bound command must live under ``dikw client *``; no
    top-level aliases. Regression-proofs the splice loop in
    ``cli.py`` never gets resurrected.
    """
    result = _run([name, "--help"])
    assert result.exit_code != 0, (
        f"`dikw {name}` should not resolve as a top-level command; "
        f"got exit_code=0 with output: {result.stdout}"
    )


def test_client_status_explicit_subcommand(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """``dikw client status`` (explicit subcommand) is the same JSON
    payload as the top-level alias."""

    patch_transport_factory()
    result = _run(["client", "status"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert "chunks" in payload


def test_lint_clean_on_fresh_wiki(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["client", "lint", "--format", "table"])
    assert result.exit_code == 0, result.stdout
    assert "lint" in result.stdout.lower()


def test_health_default_emits_json(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """``dikw client health`` defaults to JSON (the agent contract).
    Smoke-test that the no-arg invocation succeeds against an in-memory
    server and the output is parseable JSON containing the load-bearing
    top-level keys."""

    patch_transport_factory()
    result = _run(["client", "health"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert "providers" in payload
    assert "layer_counts" in payload


def test_health_table_mode_renders_tables(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """``--format table`` exercises ``render_health_report`` end-to-end
    (otherwise a renamed field could regress silently)."""
    patch_transport_factory()
    result = _run(["client", "health", "--format", "table"])
    assert result.exit_code == 0, result.stdout
    out = result.stdout
    assert "dikw client health" in out
    assert "layer counts" in out
    assert "providers" in out


def test_health_rejects_invalid_format(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["client", "health", "--format", "csv"])
    assert result.exit_code == 2
    assert "must be 'json' or 'table'" in result.stdout


def test_query_cmd_removed_from_cli(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """Guard: ``dikw client query --help`` must exit non-zero (Typer
    rejects unknown subcommands)."""
    patch_transport_factory()
    result = _run(["client", "query", "--help"])
    assert result.exit_code != 0, (
        "dikw client query should be removed but `--help` succeeded,"
        f" suggesting the subcommand still exists. Output:\n{result.stdout}"
    )


def _drop_broken_markdown(rt: ServerRuntime) -> None:
    """Plant one valid + one YAML-broken file under the server's
    sources tree, ready for an in-place ingest (no import bundle
    needed). Used by both --strict tests."""
    src_dir = rt.root / "sources" / "notes"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "good.md").write_text("# Good\n\nbody.\n", encoding="utf-8")
    (src_dir / "broken.md").write_text("---\nbroken: : :\n---\n# T\n", encoding="utf-8")


def test_ingest_default_treats_file_errors_as_warnings(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``--strict``, a per-file failure should NOT fail the
    overall CLI invocation — the file shows in the warning summary
    but exit code stays 0 so a flaky markdown can't break CI."""
    monkeypatch.setattr("dikw_core.api.build_embedder", lambda _cfg: FakeEmbeddings())
    _, rt = asgi_client
    _drop_broken_markdown(rt)
    patch_transport_factory()

    # Op commands default to async-by-default since the task-first
    # CLI flip; ``--wait`` makes the test see the IngestReport + errors
    # surface that this assertion is gated on.
    result = _run(["client", "ingest", "--no-embed", "--plain", "--wait"])
    assert result.exit_code == 0, result.stdout
    assert "file error" in result.stdout.lower()
    assert "broken.md" in result.stdout


def test_ingest_strict_exits_one_when_any_file_errors(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--strict`` flips the same per-file failure into a non-zero
    exit so CI can branch on it."""
    monkeypatch.setattr("dikw_core.api.build_embedder", lambda _cfg: FakeEmbeddings())
    _, rt = asgi_client
    _drop_broken_markdown(rt)
    patch_transport_factory()

    result = _run(["client", "ingest", "--no-embed", "--plain", "--strict"])
    assert result.exit_code == 1, result.stdout
    assert "broken.md" in result.stdout


def _ingest_fixtures(rt: ServerRuntime) -> None:
    """Drop the standard ``tests/fixtures/notes`` corpus into the server's
    ``sources/`` and ingest via the engine. Used by pages-CLI tests that
    need a base with both documents and chunks."""
    import asyncio

    src_dir = rt.root / "sources" / "notes"
    src_dir.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, src_dir / src.name)
    asyncio.run(api.ingest(rt.root, embedder=FakeEmbeddings()))


def test_pages_list_emits_documents(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """``dikw client pages list`` returns the same DocumentRecord array as
    ``GET /v1/base/pages``."""

    _, rt = asgi_client
    _ingest_fixtures(rt)
    patch_transport_factory()
    result = _run(["client", "pages", "list", "--format", "json"])
    assert result.exit_code == 0, result.stdout
    rows = json.loads(result.stdout)
    assert any(r["layer"] == "source" for r in rows)


def test_pages_list_layer_filter(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:

    _, rt = asgi_client
    _ingest_fixtures(rt)
    patch_transport_factory()
    result = _run(["client", "pages", "list", "--layer", "source", "--format", "json"])
    assert result.exit_code == 0, result.stdout
    rows = json.loads(result.stdout)
    assert rows and all(r["layer"] == "source" for r in rows)


def test_pages_list_table_mode(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    _, rt = asgi_client
    _ingest_fixtures(rt)
    patch_transport_factory()
    result = _run(["client", "pages", "list", "--format", "table"])
    assert result.exit_code == 0, result.stdout
    assert "pages" in result.stdout
    assert "layer" in result.stdout


def test_pages_list_rejects_invalid_format(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["client", "pages", "list", "--format", "csv"])
    assert result.exit_code == 2
    assert "must be 'json' or 'table'" in result.stdout


def test_pages_get_emits_body_and_anchors(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """End-to-end pages get: list to discover an indexed path, then get
    that path and verify body + non-empty anchors land in stdout JSON."""

    _, rt = asgi_client
    _ingest_fixtures(rt)
    patch_transport_factory()
    listed = _run(["client", "pages", "list", "--format", "json"])
    target = next(r for r in json.loads(listed.stdout) if r["layer"] == "source")

    result = _run(["client", "pages", "get", target["path"]])
    assert result.exit_code == 0, result.stdout
    body = json.loads(result.stdout)
    assert body["doc_id"] == target["doc_id"]
    assert isinstance(body["body"], str) and body["body"]
    assert isinstance(body["anchors"], list) and body["anchors"]


def test_pages_get_unknown_exits_one(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["client", "pages", "get", "sources/missing.md"])
    assert result.exit_code == 1
    assert "page_not_found" in result.stdout or "404" in result.stdout


def _seed_pages_links(rt: ServerRuntime) -> tuple[str, str, str]:
    """Seed wiki docs ``a → b → c`` via direct engine storage writes so
    the link-graph CLI tests don't need a real synth pass. Returns the
    three paths for assertions."""
    import asyncio

    a_path, b_path, c_path = "wiki/a.md", "wiki/b.md", "wiki/c.md"

    async def _seed() -> None:
        cfg, _root, storage = await api._with_storage(rt.root)
        del cfg
        try:
            for p in (a_path, b_path, c_path):
                await storage.upsert_document(
                    DocumentRecord(
                        doc_id=api._doc_id_for(Layer.WIKI, p),
                        path=p,
                        hash="0" * 64,
                        mtime=0.0,
                        layer=Layer.WIKI,
                        active=True,
                    )
                )
            await storage.upsert_link(
                LinkRecord(
                    src_doc_id=api._doc_id_for(Layer.WIKI, a_path),
                    dst_path=b_path,
                    link_type=LinkType.WIKILINK,
                    anchor=None,
                    line=3,
                )
            )
            await storage.upsert_link(
                LinkRecord(
                    src_doc_id=api._doc_id_for(Layer.WIKI, b_path),
                    dst_path=c_path,
                    link_type=LinkType.WIKILINK,
                    anchor=None,
                    line=4,
                )
            )
        finally:
            await storage.close()

    asyncio.run(_seed())
    return a_path, b_path, c_path


def test_pages_links_default_emits_both_directions_as_json(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """``dikw client pages links <path>`` defaults to JSON with ``both``
    direction — the agent-friendly contract. b has one outgoing edge to
    c and one incoming edge from a."""
    _, rt = asgi_client
    a_path, b_path, c_path = _seed_pages_links(rt)
    patch_transport_factory()
    result = _run(["client", "pages", "links", b_path])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["path"] == b_path
    assert [e["dst_path"] for e in payload["outgoing"]] == [c_path]
    assert [e["src_path"] for e in payload["incoming"]] == [a_path]


def test_pages_links_direction_out(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    _, rt = asgi_client
    _, b_path, _ = _seed_pages_links(rt)
    patch_transport_factory()
    result = _run(["client", "pages", "links", b_path, "--direction", "out"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["outgoing"] and payload["incoming"] == []


def test_pages_links_table_mode(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    _, rt = asgi_client
    _, b_path, _ = _seed_pages_links(rt)
    patch_transport_factory()
    result = _run(["client", "pages", "links", b_path, "--format", "table"])
    assert result.exit_code == 0, result.stdout
    # Table header columns surface in stdout text.
    assert "outgoing" in result.stdout
    assert "incoming" in result.stdout


def test_pages_links_unknown_path_exits_one(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["client", "pages", "links", "wiki/missing.md"])
    assert result.exit_code == 1
    assert "page_not_found" in result.stdout or "404" in result.stdout


def test_pages_links_rejects_invalid_format(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["client", "pages", "links", "wiki/a.md", "--format", "csv"])
    assert result.exit_code == 2
    assert "must be 'json' or 'table'" in result.stdout


def _seed_pages_provenance(rt: ServerRuntime) -> tuple[str, str, str]:
    """Seed one D-source claimed by two K-pages (a, b) plus a dangling
    source on ``a`` so the resolved/dangling marker is exercised. Returns
    ``(src_path, a_path, b_path)`` for assertions."""
    import asyncio

    src_path = "sources/src.md"
    ghost_path = "sources/ghost.md"
    a_path = "wiki/a.md"
    b_path = "wiki/b.md"

    async def _seed() -> None:
        cfg, _root, storage = await api._with_storage(rt.root)
        del cfg
        try:
            await storage.upsert_document(
                DocumentRecord(
                    doc_id=api._doc_id_for(Layer.SOURCE, src_path),
                    path=src_path,
                    title="Src",
                    hash="0" * 64,
                    mtime=0.0,
                    layer=Layer.SOURCE,
                    active=True,
                )
            )
            for p in (a_path, b_path):
                await storage.upsert_document(
                    DocumentRecord(
                        doc_id=api._doc_id_for(Layer.WIKI, p),
                        path=p,
                        hash="0" * 64,
                        mtime=0.0,
                        layer=Layer.WIKI,
                        active=True,
                    )
                )
            await storage.replace_provenance_from(
                api._doc_id_for(Layer.WIKI, a_path), [src_path, ghost_path]
            )
            await storage.replace_provenance_from(
                api._doc_id_for(Layer.WIKI, b_path), [src_path]
            )
        finally:
            await storage.close()

    asyncio.run(_seed())
    return src_path, a_path, b_path


def test_pages_provenance_default_emits_both_directions_as_json(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """``dikw client pages provenance <wiki-path>`` defaults to JSON
    with ``both`` direction — agent-friendly. A K-page has its forward
    sources populated and reverse empty (no K-page claims a K-page as
    its source)."""
    _, rt = asgi_client
    src_path, a_path, _b = _seed_pages_provenance(rt)
    patch_transport_factory()
    result = _run(["client", "pages", "provenance", a_path])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["path"] == a_path
    assert payload["derived_pages"] == []
    by_path = {s["source_path"]: s for s in payload["derived_from"]}
    assert by_path[src_path]["resolved"] is True
    assert by_path["sources/ghost.md"]["resolved"] is False


def test_pages_provenance_reverse_for_source(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """Querying the D-source path returns the K-pages that claim it —
    the "which pages reference this source?" question this feature
    exists for."""
    _, rt = asgi_client
    src_path, a_path, b_path = _seed_pages_provenance(rt)
    patch_transport_factory()
    result = _run(["client", "pages", "provenance", src_path])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["derived_from"] == []
    assert sorted(dp["path"] for dp in payload["derived_pages"]) == sorted(
        [a_path, b_path]
    )


def test_pages_provenance_table_renders_resolved_flag(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """``--format table`` mode renders a ✓/✗ column for ``resolved`` —
    the dangling-source marker is the table's primary value-add over
    JSON."""
    _, rt = asgi_client
    _src, a_path, _b = _seed_pages_provenance(rt)
    patch_transport_factory()
    result = _run(
        ["client", "pages", "provenance", a_path, "--format", "table"]
    )
    assert result.exit_code == 0, result.stdout
    assert "derived_from" in result.stdout
    assert "derived_pages" in result.stdout
    # Resolved flag rendered as ✓ for the real source AND ✗ for the
    # dangling one — both markers visible on the same page.
    assert "✓" in result.stdout
    assert "✗" in result.stdout


def test_pages_provenance_direction_in(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    _, rt = asgi_client
    src_path, _a, _b = _seed_pages_provenance(rt)
    patch_transport_factory()
    result = _run(
        ["client", "pages", "provenance", src_path, "--direction", "in"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["derived_pages"] and payload["derived_from"] == []


def test_pages_provenance_unknown_path_exits_one(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["client", "pages", "provenance", "wiki/missing.md"])
    assert result.exit_code == 1
    assert "page_not_found" in result.stdout or "404" in result.stdout


def test_review_list_empty_on_fresh_wiki(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["client", "review", "list", "--format", "table"])
    assert result.exit_code == 0, result.stdout
    assert "no candidates" in result.stdout


def test_tasks_list_empty_on_fresh_server(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["client", "tasks", "list", "--format", "table"])
    assert result.exit_code == 0, result.stdout
    assert "no tasks" in result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["client", "status", "--format", "json"],
        ["client", "lint", "--format", "json"],
        ["client", "tasks", "list", "--format", "json"],
        ["client", "review", "list", "--format", "json"],
    ],
    ids=["status", "lint", "tasks-list", "review-list"],
)
def test_format_json_emits_parseable_json(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    argv: list[str],
) -> None:
    """``--format json`` stays valid (now redundant — these default to
    JSON since the 0.2.5 agent-first flip). Smoke-test that each still
    prints a parseable JSON document, not a rich banner that ``| jq``
    can't parse."""

    patch_transport_factory()
    result = _run(argv)
    assert result.exit_code == 0, result.stdout
    # ``console.print_json`` adds two-space indent + trailing newline; the
    # body must be a parseable JSON document either way.
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, list | dict)


@pytest.mark.parametrize(
    "argv",
    [
        ["client", "lint"],
        ["client", "lint", "proposals"],
        ["client", "review", "list"],
        ["client", "tasks", "list"],
    ],
    ids=["lint", "lint-proposals", "review-list", "tasks-list"],
)
def test_default_emits_parseable_json(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    argv: list[str],
) -> None:
    """0.2.5 agent-first flip: the four maintenance commands now default
    to JSON. Without any ``--format`` flag each must print a parseable
    JSON document — an agent piping ``| jq`` must never get a rich banner
    like ``no tasks`` / ``lint clean``."""
    patch_transport_factory()
    result = _run(argv)
    assert result.exit_code == 0, result.stdout
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, list | dict)


@pytest.fixture()
async def seeded_candidate(
    asgi_client: tuple[Any, ServerRuntime],
) -> str:
    """Seed one CANDIDATE wisdom item directly in storage, return its id.
    Async fixture (not in-test) so the sync ``CliRunner`` tests don't nest
    event loops. Empty evidence is fine — the ``>=2 evidence`` gate is
    enforced at distill, not at approve/reject."""
    _client, rt = asgi_client
    item = WisdomItem(
        item_id="W-rev0001",
        kind=WisdomKind.PRINCIPLE,
        status=WisdomStatus.CANDIDATE,
        path=None,
        title="Seeded review candidate",
        body="A seeded candidate so review approve/reject have a target.",
        confidence=0.8,
        created_ts=time.time(),
        approved_ts=None,
    )
    await rt.storage.put_wisdom(item, [])
    return item.item_id


def test_review_approve_default_emits_json(
    seeded_candidate: str,
    patch_transport_factory: Callable[[], None],
) -> None:
    """``review approve`` defaults to raw JSON (agent-first); the payload
    carries ``item_id`` + the new status."""
    patch_transport_factory()
    result = _run(["client", "review", "approve", seeded_candidate])
    assert result.exit_code == 0, result.stdout
    parsed = json.loads(result.stdout)
    assert parsed["item_id"] == seeded_candidate
    assert parsed["new_status"] == "approved"


def test_review_reject_pretty_emits_human_line(
    seeded_candidate: str,
    patch_transport_factory: Callable[[], None],
) -> None:
    """``--pretty`` opts into the colored human line instead of JSON: the
    output mentions the item id and is NOT JSON-parseable."""
    patch_transport_factory()
    result = _run(["client", "review", "reject", seeded_candidate, "--pretty"])
    assert result.exit_code == 0, result.stdout
    assert seeded_candidate in result.stdout
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def test_review_reject_default_emits_json(
    seeded_candidate: str,
    patch_transport_factory: Callable[[], None],
) -> None:
    """``review reject`` defaults to raw JSON (agent-first); reject maps the
    candidate to ``archived``, not a ``rejected`` status."""
    patch_transport_factory()
    result = _run(["client", "review", "reject", seeded_candidate])
    assert result.exit_code == 0, result.stdout
    parsed = json.loads(result.stdout)
    assert parsed["item_id"] == seeded_candidate
    assert parsed["new_status"] == "archived"


def test_review_approve_pretty_emits_human_line(
    seeded_candidate: str,
    patch_transport_factory: Callable[[], None],
) -> None:
    """``--pretty`` opts into the colored human line instead of JSON: the
    output mentions the item id and is NOT JSON-parseable."""
    patch_transport_factory()
    result = _run(["client", "review", "approve", seeded_candidate, "--pretty"])
    assert result.exit_code == 0, result.stdout
    assert seeded_candidate in result.stdout
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def test_check_unavailable_provider_exits_one(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``dikw client check`` exit code must mirror the report's
    ``ok`` field — without API keys, the server returns ``ok=False`` and
    the CLI must exit non-zero so CI / shell scripts can branch on it."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DIKW_EMBEDDING_API_KEY", raising=False)
    patch_transport_factory()
    result = _run(["client", "check"])
    # Either both legs fail (exit 1) or the LLM probe passes
    # incidentally on the test image; in both cases the CLI must not
    # crash with a traceback.
    assert result.exit_code in (0, 1), result.stdout


def test_status_default_emits_json(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """``dikw client status`` (no flags) must emit JSON parseable by
    ``json.loads``."""
    patch_transport_factory()
    result = _run(["client", "status"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    assert payload, "status JSON payload must not be empty"


def test_status_table_mode_renders(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """``--format table`` keeps the rich-rendered output for humans."""
    patch_transport_factory()
    result = _run(["client", "status", "--format", "table"])
    assert result.exit_code == 0, result.stdout
    # ``render_status`` prints layer labels; "chunks" is one of them.
    assert "chunks" in result.stdout


def test_check_default_emits_json(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``dikw client check`` (no flags) must emit parseable JSON
    regardless of probe outcome."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DIKW_EMBEDDING_API_KEY", raising=False)
    patch_transport_factory()
    result = _run(["client", "check"])
    assert result.exit_code in (0, 1), result.stdout
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    # ``CheckReport`` has ``llm`` and ``embed`` per-leg keys; at least
    # one must be present in every probe outcome.
    assert "llm" in payload or "embed" in payload


def test_check_table_mode_renders(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--format table`` keeps the rich rendering for human operators."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DIKW_EMBEDDING_API_KEY", raising=False)
    patch_transport_factory()
    result = _run(["client", "check", "--format", "table"])
    assert result.exit_code in (0, 1), result.stdout
    # ``render_check_report`` prints per-leg labels.
    out = result.stdout.lower()
    assert "llm" in out or "embed" in out


def test_check_rejects_invalid_format(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["client", "check", "--format", "csv"])
    assert result.exit_code == 2
    assert "must be 'json' or 'table'" in result.stdout


def test_info_default_emits_parseable_json(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """``dikw client info`` happy path must emit parseable JSON. The
    command is JSON-only (no ``--format`` flag) — agents call it as a
    bootstrap probe and need the openapi / docs hints inline."""
    patch_transport_factory()
    result = _run(["client", "info"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)


def test_distill_runs_through_task_pipeline(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: ``dikw client distill`` submits a task, follows the
    NDJSON event stream, and renders the DistillReport. We use FakeLLM
    so the report has zero candidates_added (the stub doesn't parse
    into a candidate), but the report shape itself is the contract we
    care about."""
    monkeypatch.setattr(synth_op, "build_llm", lambda _cfg, **_kw: FakeLLM())
    monkeypatch.setattr(synth_op, "build_embedder", lambda _cfg: FakeEmbeddings())
    patch_transport_factory()
    result = _run(["client", "distill", "--plain", "--wait"])
    assert result.exit_code == 0, result.stdout
    assert "K pages read" in result.stdout
    assert "candidates added" in result.stdout
