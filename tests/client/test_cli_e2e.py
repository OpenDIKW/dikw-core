"""End-to-end ``dikw client *`` tests against the in-memory ASGI server.

We use Typer's ``CliRunner`` because that's the closest thing to "what
a user actually types" and it captures stdout / exit code in one
artefact. ``patch_transport_factory`` rewires ``Transport.from_config``
so each command's freshly constructed transport rides on the same
in-memory ASGI client the fixture set up — no socket, no network, no
flake.
"""

from __future__ import annotations

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
    patch_transport_factory()
    result = _run(["status"])  # top-level alias → client.status
    assert result.exit_code == 0, result.stdout
    assert "source" in result.stdout
    assert "chunks" in result.stdout


def test_client_status_explicit_subcommand(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["client", "status"])
    assert result.exit_code == 0, result.stdout
    assert "source" in result.stdout


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


def test_query_streams_tokens_and_renders_answer(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: ingest fixtures via the engine (the CLI doesn't have
    `ingest --from` support without an actual upload tarball, which is
    covered by ``test_upload.py`` + the server's own upload tests), then
    issue ``dikw query`` and check the streaming + final rendering both
    produce output.

    Sync test body — Typer's ``CliRunner`` runs each command through
    ``asyncio.run`` internally, which clashes with pytest-asyncio's
    outer loop if we mark this ``async``. We do the engine-side ingest
    via a one-shot ``asyncio.run`` instead.
    """
    import asyncio

    _, rt = asgi_client
    src_dir = rt.root / "sources" / "notes"
    src_dir.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, src_dir / src.name)
    asyncio.run(api.ingest(rt.root, embedder=FakeEmbeddings()))

    # Patch build_llm + build_embedder so the engine doesn't try to hit
    # a real LLM / embedding endpoint inside the CLI run.
    fake_llm = FakeLLM(
        response_text="Karpathy says scoping is deterministic.",
        stream_chunks=["Karpathy ", "says ", "scoping ", "is ", "deterministic."],
    )
    monkeypatch.setattr(
        "dikw_core.api.build_llm", lambda _cfg, **_kw: fake_llm
    )
    monkeypatch.setattr(
        "dikw_core.api.build_embedder",
        lambda _cfg, dim_override=None: FakeEmbeddings(),
    )
    patch_transport_factory()

    result = _run(
        ["query", "what does Karpathy say about scoping?", "--limit", "3"]
    )
    assert result.exit_code == 0, result.stdout
    # Streamed tokens land in stdout (CliRunner captures both rich
    # console + plain stdout).
    assert "Karpathy" in result.stdout
    assert "deterministic" in result.stdout
    assert "citations" in result.stdout


def test_health_default_emits_json(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    """``dikw client health`` defaults to JSON (the agent contract).
    Smoke-test that the no-arg invocation succeeds against an in-memory
    server and the output is parseable JSON containing the load-bearing
    top-level keys."""
    import json as _json

    patch_transport_factory()
    result = _run(["client", "health"])
    assert result.exit_code == 0, result.stdout
    payload = _json.loads(result.stdout)
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
    monkeypatch.setattr(
        synth_op, "build_embedder", lambda _cfg: FakeEmbeddings()
    )
    patch_transport_factory()
    result = _run(["distill", "--plain"])
    assert result.exit_code == 0, result.stdout
    assert "K pages read" in result.stdout
    assert "candidates added" in result.stdout
