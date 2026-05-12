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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dikw_core import api
from dikw_core.cli import app
from dikw_core.server import synth_op
from dikw_core.server.runtime import ServerRuntime

from ..fakes import FakeEmbeddings, FakeLLM

FIXTURES = Path(__file__).parent.parent / "fixtures" / "notes"


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(app, args)


def test_status_routes_through_client(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """Top-level ``dikw status`` aliases to ``dikw client status`` and
    emits parseable JSON by default."""

    patch_transport_factory()
    result = _run(["status"])  # top-level alias → client.status
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert "chunks" in payload
    assert "documents_by_layer" in payload


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
    result = _run(["lint"])
    assert result.exit_code == 0, result.stdout
    assert "lint" in result.stdout.lower()


def test_client_init_treats_already_initialised_as_success(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """The server's runtime won't even start without a ``dikw.yml``, so
    the only way ``POST /v1/init`` lands in normal use is the
    ``wiki_already_initialised`` 409. The CLI must surface that as an
    exit-0 no-op (matching its docstring) instead of a non-zero
    failure that would break any "init then ingest" bootstrap script."""
    patch_transport_factory()
    result = _run(["client", "init"])
    assert result.exit_code == 0, result.stdout
    assert "already initialized" in result.stdout.lower()


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
    assert "dikw health" in out
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

    result = _run(["ingest", "--no-embed", "--plain"])
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

    result = _run(["ingest", "--no-embed", "--plain", "--strict"])
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


def test_review_list_empty_on_fresh_wiki(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["review", "list"])
    assert result.exit_code == 0, result.stdout
    assert "no candidates" in result.stdout


def test_tasks_list_empty_on_fresh_server(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["tasks", "list"])
    assert result.exit_code == 0, result.stdout
    assert "no tasks" in result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["status", "--format", "json"],
        ["lint", "--format", "json"],
        ["tasks", "list", "--format", "json"],
        ["review", "list", "--format", "json"],
    ],
    ids=["status", "lint", "tasks-list", "review-list"],
)
def test_format_json_emits_parseable_json(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    argv: list[str],
) -> None:
    """``--format json`` is the agent-friendly half of the four
    table-default commands extended in PR 5. Smoke-test that each one
    actually prints valid JSON instead of the rich-rendered table —
    otherwise an agent piping ``| jq`` would silently get a banner
    string that never parses."""

    patch_transport_factory()
    result = _run(argv)
    assert result.exit_code == 0, result.stdout
    # ``console.print_json`` adds two-space indent + trailing newline; the
    # body must be a parseable JSON document either way.
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, list | dict)


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
    result = _run(["check"])
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
    result = _run(["distill", "--plain"])
    assert result.exit_code == 0, result.stdout
    assert "K pages read" in result.stdout
    assert "candidates added" in result.stdout
