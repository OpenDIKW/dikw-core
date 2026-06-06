"""CLI integration for ``dikw client eval`` against the in-memory server.

Phase 5 collapses the in-process ``dikw client eval`` command into a thin
client wrapper that submits ``POST /v1/eval`` and follows the task
event stream. The previous CLI surface (``--embedder
fake|provider``, ``--retrieval all``, no-arg auto-discovery) lived
above the engine and is now exercised at the engine layer (see
``tests/test_eval_runner.py``); the CLI's job here is just argument
threading + exit-code semantics.

Server-side dataset loading covered in ``tests/server/test_synth_distill_tasks.py``;
we focus on the wire from CLI → server → renderer.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dikw_core.cli import app
from dikw_core.server import synth_op
from dikw_core.server.runtime import ServerRuntime

from .fakes import FakeEmbeddings


def _write_toy_dataset(root: Path, *, name: str = "toy") -> Path:
    ds = root / name
    (ds / "corpus").mkdir(parents=True, exist_ok=True)
    (ds / "corpus" / "alpha.md").write_text(
        "# Alpha\n\nAlpha describes foo and bar topics.\n", encoding="utf-8"
    )
    (ds / "corpus" / "beta.md").write_text(
        "# Beta\n\nBeta discusses baz and qux.\n", encoding="utf-8"
    )
    (ds / "dataset.yaml").write_text(
        "name: " + name + "\n"
        "description: cli test\n"
        "thresholds:\n"
        "  hit_at_3: 0.5\n"
        "  hit_at_10: 0.5\n"
        "  mrr: 0.3\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: foo and bar topics\n"
        "    expect_any: [alpha]\n"
        "  - q: baz and qux\n"
        "    expect_any: [beta]\n",
        encoding="utf-8",
    )
    return ds


@pytest.fixture()
def patch_eval_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the embedder factory inside the server's eval task runner.

    The default ``build_embedder`` would try to honour the wiki's
    ``embedding`` provider config, which the test wiki sets to a stub
    URL — calling out to it would either hang or fail. ``FakeEmbeddings``
    keeps the test hermetic + deterministic.
    """
    monkeypatch.setattr(
        synth_op, "build_embedder", lambda _cfg: FakeEmbeddings()
    )


def test_dataset_path_passes_thresholds(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    patch_eval_factories: None,
    tmp_path: Path,
) -> None:
    ds = _write_toy_dataset(tmp_path)
    patch_transport_factory()
    result = CliRunner().invoke(
        app, ["client", "eval", "--dataset", str(ds), "--plain", "--wait"]
    )
    assert result.exit_code == 0, result.stdout
    assert "toy" in result.stdout
    assert "hit_at_3" in result.stdout


def test_dataset_path_fails_when_thresholds_unmet(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    patch_eval_factories: None,
    tmp_path: Path,
) -> None:
    ds = tmp_path / "fail"
    (ds / "corpus").mkdir(parents=True)
    (ds / "corpus" / "alpha.md").write_text("# A\n", encoding="utf-8")
    (ds / "dataset.yaml").write_text(
        "name: fail\n"
        "description: cli test fail\n"
        "thresholds:\n"
        "  hit_at_3: 1.0\n"
        "  mrr: 1.0\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n  - q: foo bar\n    expect_any: [ghost]\n",
        encoding="utf-8",
    )
    patch_transport_factory()
    result = CliRunner().invoke(
        app, ["client", "eval", "--dataset", str(ds), "--plain", "--wait"]
    )
    assert result.exit_code == 1, result.stdout


def test_eval_judge_sample_rejects_invalid_value() -> None:
    """``--judge-sample`` takes a positive int or 'auto'; anything else is a
    client-side parse error (no server round-trip needed to reject it)."""
    result = CliRunner().invoke(
        app, ["client", "eval", "--judge-sample", "bogus"]
    )
    # Exit 2 is click's usage-error code (the BadParameter raised at parse),
    # distinct from a runtime/transport failure (exit 1) — so this proves the
    # value was rejected up front, not that some unrelated step blew up. (The
    # message itself renders in a Rich panel that truncates under test capture,
    # so assert the code, not the wrapped text.)
    assert result.exit_code == 2


def test_synth_reports_helper_filters_by_mode() -> None:
    """``_synth_reports`` is the load-bearing helper for the
    ``--eval synth`` exit-code logic: it must pluck only synth-mode
    reports out of a multi-dataset envelope so the gated/ungated
    decision is made over the right rows."""
    from dikw_core.client.cli_app import _synth_reports

    single_synth = {"mode": "synth", "gated": False, "passed": True}
    single_retrieval = {"mode": "retrieval", "passed": True}
    multi = {
        "datasets": [
            {"mode": "retrieval", "passed": True},
            {"mode": "synth", "gated": True, "passed": True},
            {"mode": "synth", "gated": False, "passed": True},
        ],
        "passed": True,
    }
    assert _synth_reports(single_synth) == [single_synth]
    assert _synth_reports(single_retrieval) == []
    synths = _synth_reports(multi)
    assert len(synths) == 2
    # The ungated-flag arithmetic the CLI relies on:
    assert any(r.get("gated") for r in synths) is True
    # If all synth reports were ungated, the CLI exits 2 (informational).
    assert not any(
        r.get("gated") for r in [{"mode": "synth", "gated": False}]
    )


def test_missing_dataset_returns_non_zero(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    patch_eval_factories: None,
    tmp_path: Path,
) -> None:
    """The server returns 400 ``dataset_not_found``; the CLI must
    propagate to a non-zero exit code so CI gates work."""
    patch_transport_factory()
    result = CliRunner().invoke(
        app,
        ["client", "eval", "--dataset", str(tmp_path / "missing"), "--plain", "--wait"],
    )
    assert result.exit_code != 0, result.stdout
