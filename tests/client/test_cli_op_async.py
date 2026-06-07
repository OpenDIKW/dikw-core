"""CLI op-command contract: async-by-default + ``--wait`` opt-in.

Covers the task-submitting op commands — ``ingest``, ``synth``,
``eval``, ``lint propose``, ``lint apply`` — flipped from blocking-by-
default (stream until terminal) to async-by-default (submit + print
task handle + exit 0). The blocking shape is opt-in via ``--wait`` and
the exit-code mapping under that flag is the agent contract.

Exit code mapping under ``--wait``:

* 0 — task ``succeeded``
* 1 — task ``failed``
* 130 — task ``cancelled`` (POSIX SIGINT convention)
* 124 — client-side timeout fired (POSIX timeout convention)

Tests drive the FastAPI runtime in-memory via the shared
``patch_transport_factory`` fixture (see ``tests/conftest.py``) — no
sockets, no flakes.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dikw_core import api
from dikw_core.cli import app
from dikw_core.providers import LLMResponse
from dikw_core.server import synth_op
from dikw_core.server.runtime import ServerRuntime

from ..fakes import FakeEmbeddings, FakeLLM

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "notes"


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(app, args)


class _ScriptedSynthLLM:
    """Returns one canned ``<page>`` block per source path (substring match)."""

    def __init__(self, by_source: dict[str, str]) -> None:
        self._by_source = by_source

    async def complete(
        self, *, system: str, user: str, model: str, **_: Any
    ) -> LLMResponse:
        for src_path, body in self._by_source.items():
            if src_path in user:
                return LLMResponse(text=body, finish_reason="end_turn")
        raise AssertionError(f"no scripted page for prompt: {user[:200]}")

    def complete_stream(self, **_: Any) -> Any:
        raise NotImplementedError


def _seed_and_ingest(rt: ServerRuntime) -> None:
    src_dir = rt.root / "sources" / "notes"
    src_dir.mkdir(parents=True, exist_ok=True)
    for src in _FIXTURES.glob("*.md"):
        shutil.copy2(src, src_dir / src.name)
    asyncio.run(api.ingest(rt.root, embedder=FakeEmbeddings()))


def _concept_page(slug: str, title: str, body: str) -> str:
    return (
        f'<page category="concept" slug="{slug}">\n---\n---\n\n'
        f"# {title}\n\n{body}\n</page>"
    )


# ---- async-by-default --------------------------------------------------


def test_ingest_default_async_prints_task_handle_exit_zero(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``--wait``: submit, print the handle JSON, exit 0
    immediately. The task itself may still be running — we don't follow
    it."""
    monkeypatch.setattr(
        "dikw_core.server.ingest_op.build_embedder", lambda _cfg: FakeEmbeddings()
    )
    patch_transport_factory()
    result = _run(["client", "ingest", "--no-embed"])
    assert result.exit_code == 0, result.stdout
    handle = json.loads(result.stdout)
    assert isinstance(handle.get("task_id"), str) and handle["task_id"]
    assert handle.get("status") in {"pending", "running", "succeeded"}
    assert handle.get("events_url") == f"/v1/tasks/{handle['task_id']}/events"
    assert (
        handle.get("wait_command")
        == f"dikw client tasks wait {handle['task_id']}"
    )


def test_synth_default_async_prints_task_handle(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(synth_op, "build_llm", lambda _cfg, **_kw: FakeLLM())
    monkeypatch.setattr(synth_op, "build_embedder", lambda _cfg: FakeEmbeddings())
    patch_transport_factory()
    result = _run(["client", "synth"])
    assert result.exit_code == 0, result.stdout
    handle = json.loads(result.stdout)
    assert handle.get("task_id")
    assert "events_url" in handle


# ---- synth --verify exit-code contract (the CI-gating purpose) ----------


def test_synth_verify_pass_exits_zero_and_implies_wait(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--verify`` on a clean run renders the verify verdict and exits 0.
    It also IMPLIES --wait: the output is the rendered report, not a bare
    async task-handle JSON, proving the command blocked to terminal."""
    _, rt = asgi_client
    _seed_and_ingest(rt)
    # Cross-linked pages: no broken links, distinct bodies (no dup). Orphans
    # may exist (single pass) but are informational, so the verdict PASSES.
    script = {
        "sources/notes/dikw.md": _concept_page(
            "dikw-pyramid",
            "DIKW pyramid",
            "Four layers. See [[Karpathy LLM Wiki]] and [[Hybrid retrieval]].",
        ),
        "sources/notes/karpathy-wiki.md": _concept_page(
            "karpathy-llm-wiki",
            "Karpathy LLM Wiki",
            "A wiki built from sources. Complements the [[DIKW pyramid]].",
        ),
        "sources/notes/retrieval.md": _concept_page(
            "hybrid-retrieval",
            "Hybrid retrieval",
            "BM25 fused with dense via RRF, background for the [[DIKW pyramid]].",
        ),
    }
    monkeypatch.setattr(
        synth_op, "build_llm", lambda _cfg, **_kw: _ScriptedSynthLLM(script)
    )
    monkeypatch.setattr(synth_op, "build_embedder", lambda _cfg: FakeEmbeddings())
    patch_transport_factory()

    result = _run(["client", "synth", "--verify", "--plain"])
    assert result.exit_code == 0, result.stdout
    # Rendered verdict (not a bare task handle) → --verify blocked to terminal.
    assert "synth verify" in result.stdout
    assert "PASS" in result.stdout


def test_synth_verify_fail_exits_nonzero(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing verify (dangling wikilink → broken_wikilink leg) maps to a
    non-zero exit — the entire CI-gating purpose of the flag."""
    _, rt = asgi_client
    _seed_and_ingest(rt)
    script = {
        "sources/notes/dikw.md": _concept_page(
            "dikw-pyramid",
            "DIKW pyramid",
            "Four layers. See [[Ghost Reference]] which nobody authors.",
        ),
        "sources/notes/karpathy-wiki.md": _concept_page(
            "karpathy-llm-wiki", "Karpathy LLM Wiki", "A wiki built from sources."
        ),
        "sources/notes/retrieval.md": _concept_page(
            "hybrid-retrieval", "Hybrid retrieval", "BM25 fused with dense via RRF."
        ),
    }
    monkeypatch.setattr(
        synth_op, "build_llm", lambda _cfg, **_kw: _ScriptedSynthLLM(script)
    )
    monkeypatch.setattr(synth_op, "build_embedder", lambda _cfg: FakeEmbeddings())
    patch_transport_factory()

    result = _run(["client", "synth", "--verify", "--plain"])
    assert result.exit_code == 1, result.stdout
    assert "FAIL" in result.stdout


def test_lint_propose_default_async_prints_task_handle(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
) -> None:
    patch_transport_factory()
    result = _run(["client", "lint", "propose", "--rule", "broken_wikilink"])
    assert result.exit_code == 0, result.stdout
    handle = json.loads(result.stdout)
    assert handle.get("task_id")


# ---- --wait opt-in, exit-code mapping ----------------------------------


def test_ingest_wait_renders_report_exits_zero(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--wait`` follows the task to terminal, renders the
    ``IngestReport`` table, and maps ``succeeded`` to exit 0."""
    monkeypatch.setattr(
        "dikw_core.server.ingest_op.build_embedder", lambda _cfg: FakeEmbeddings()
    )
    patch_transport_factory()
    result = _run(["client", "ingest", "--no-embed", "--wait", "--plain"])
    assert result.exit_code == 0, result.stdout
    # Report table renders the standard metric labels.
    assert "scanned" in result.stdout.lower()


