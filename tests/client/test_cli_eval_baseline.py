"""``dikw client eval --against`` / ``--write-baseline`` exit-code contract.

The real eval (dataset + embedder) is replaced by a canned task runner so the
test exercises the CLI wiring + the regression gate, not the eval engine: the
runner returns a fixed ``metrics`` payload, the client writes/compares a
baseline, and the exit code reflects the verdict (0 = ship, 1 = regression).
Drives the FastAPI runtime in-memory via the shared ``patch_transport_factory``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dikw_core.cli import app
from dikw_core.client.cli_app import _EXIT_FAILED
from dikw_core.server.runtime import ServerRuntime


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(app, args)


def _canned_eval_runner_factory(
    metrics: dict[str, float],
    *,
    passed: bool = True,
    mode: str = "retrieval",
    gated: bool | None = None,
) -> Callable[..., Any]:
    """A drop-in for ``make_eval_runner`` returning a fixed single-dataset
    result. ``gated`` is included only for synth-mode payloads."""

    def _factory(**_kwargs: Any) -> Callable[[Any], Any]:
        async def _runner(_reporter: Any) -> dict[str, Any]:
            result: dict[str, Any] = {
                "dataset": "mvp",
                "mode": mode,
                "metrics": dict(metrics),
                "thresholds": {},
                "passed": passed,
            }
            if gated is not None:
                result["gated"] = gated
            return result

        return _runner

    return _factory


@pytest.fixture()
def patch_eval(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    def _install(
        metrics: dict[str, float],
        *,
        passed: bool = True,
        mode: str = "retrieval",
        gated: bool | None = None,
    ) -> None:
        # routes_tasks imports make_eval_runner by name, so patch it there.
        monkeypatch.setattr(
            "dikw_core.server.routes_tasks.make_eval_runner",
            _canned_eval_runner_factory(
                metrics, passed=passed, mode=mode, gated=gated
            ),
        )

    return _install


def test_write_baseline_dumps_run_metrics(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    patch_eval: Callable[..., None],
    tmp_path: Path,
) -> None:
    patch_eval({"doc/ndcg_at_10": 0.5, "doc/hit_at_3": 0.6})
    patch_transport_factory()
    out = tmp_path / "mvp.json"
    result = _run(
        ["client", "eval", "--dataset", "mvp", "--write-baseline", str(out), "--plain"]
    )
    assert result.exit_code == 0, result.stdout
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["dataset"] == "mvp"
    assert doc["metrics"]["doc/ndcg_at_10"] == 0.5
    assert doc["metrics"]["doc/hit_at_3"] == 0.6
    assert "tolerance" in doc


def test_against_passes_when_run_matches_baseline(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    patch_eval: Callable[..., None],
    tmp_path: Path,
) -> None:
    patch_eval({"doc/ndcg_at_10": 0.50})
    patch_transport_factory()
    base = tmp_path / "b.json"
    base.write_text(
        json.dumps(
            {"dataset": "mvp", "metrics": {"doc/ndcg_at_10": 0.50}, "tolerance": 0.02}
        ),
        encoding="utf-8",
    )
    result = _run(
        ["client", "eval", "--dataset", "mvp", "--against", str(base), "--plain"]
    )
    assert result.exit_code == 0, result.stdout
    assert "SHIP" in result.stdout.upper()


def test_against_fails_on_regression(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    patch_eval: Callable[..., None],
    tmp_path: Path,
) -> None:
    # baseline pinned 0.55; the run scores 0.40 → a 0.15 drop > 0.02 tolerance.
    patch_eval({"doc/ndcg_at_10": 0.40})
    patch_transport_factory()
    base = tmp_path / "b.json"
    base.write_text(
        json.dumps(
            {"dataset": "mvp", "metrics": {"doc/ndcg_at_10": 0.55}, "tolerance": 0.02}
        ),
        encoding="utf-8",
    )
    result = _run(
        ["client", "eval", "--dataset", "mvp", "--against", str(base), "--plain"]
    )
    assert result.exit_code == 1, result.stdout
    assert "REGRESSION" in result.stdout.upper()


def test_against_lower_is_better_metric_regresses_when_it_rises(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    patch_eval: Callable[..., None],
    tmp_path: Path,
) -> None:
    # a `_max` metric (lower is better) that rose past tolerance → regression.
    patch_eval({"synth/fallback_ratio_max": 0.30})
    patch_transport_factory()
    base = tmp_path / "b.json"
    base.write_text(
        json.dumps(
            {"metrics": {"synth/fallback_ratio_max": 0.10}, "tolerance": 0.02}
        ),
        encoding="utf-8",
    )
    result = _run(
        ["client", "eval", "--dataset", "mvp", "--against", str(base), "--plain"]
    )
    assert result.exit_code == 1, result.stdout


def test_write_baseline_honours_tolerance_flag(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    patch_eval: Callable[..., None],
    tmp_path: Path,
) -> None:
    patch_eval({"doc/ndcg_at_10": 0.5})
    patch_transport_factory()
    out = tmp_path / "mvp.json"
    result = _run(
        [
            "client", "eval", "--dataset", "mvp",
            "--write-baseline", str(out), "--tolerance", "0.10", "--plain",
        ]
    )
    assert result.exit_code == 0, result.stdout
    assert json.loads(out.read_text(encoding="utf-8"))["tolerance"] == 0.10


def test_write_baseline_records_run_mode_when_eval_omitted(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    patch_eval: Callable[..., None],
    tmp_path: Path,
) -> None:
    # No --eval → modes must reflect what actually ran (the payload's mode),
    # not an empty list.
    patch_eval({"doc/ndcg_at_10": 0.5}, mode="retrieval")
    patch_transport_factory()
    out = tmp_path / "mvp.json"
    result = _run(
        ["client", "eval", "--dataset", "mvp", "--write-baseline", str(out), "--plain"]
    )
    assert result.exit_code == 0, result.stdout
    assert json.loads(out.read_text(encoding="utf-8"))["modes"] == ["retrieval"]


def test_write_baseline_creates_missing_parent_dir(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    patch_eval: Callable[..., None],
    tmp_path: Path,
) -> None:
    patch_eval({"doc/ndcg_at_10": 0.5})
    patch_transport_factory()
    out = tmp_path / "baselines" / "nested" / "mvp.json"
    result = _run(
        ["client", "eval", "--dataset", "mvp", "--write-baseline", str(out), "--plain"]
    )
    assert result.exit_code == 0, result.stdout
    assert out.is_file()


def test_against_missing_file_errors_cleanly(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    patch_eval: Callable[..., None],
    tmp_path: Path,
) -> None:
    patch_eval({"doc/ndcg_at_10": 0.5})
    patch_transport_factory()
    result = _run(
        [
            "client", "eval", "--dataset", "mvp",
            "--against", str(tmp_path / "nope.json"), "--plain",
        ]
    )
    assert result.exit_code == _EXIT_FAILED, result.stdout
    assert "not found" in result.stdout.lower()


def test_against_nothing_to_gate_errors(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    patch_eval: Callable[..., None],
    tmp_path: Path,
) -> None:
    # baseline pins a metric the run never produces → zero overlap → false-green
    # unless guarded. Must error, not SHIP.
    patch_eval({"doc/ndcg_at_10": 0.5})
    patch_transport_factory()
    base = tmp_path / "b.json"
    base.write_text(
        json.dumps({"metrics": {"some/other_metric": 0.9}}), encoding="utf-8"
    )
    result = _run(
        ["client", "eval", "--dataset", "mvp", "--against", str(base), "--plain"]
    )
    assert result.exit_code == _EXIT_FAILED, result.stdout
    assert "nothing to gate" in result.stdout.lower()


def test_against_synth_ungated_does_not_override_ship(
    asgi_client: tuple[Any, ServerRuntime],
    patch_transport_factory: Callable[[], None],
    patch_eval: Callable[..., None],
    tmp_path: Path,
) -> None:
    # --eval synth on an ungated dataset normally exits 2; under --against the
    # baseline is the chosen gate, and it SHIPed, so exit must be 0 not 2.
    patch_eval({"synth/source_chunk_coverage": 0.83}, mode="synth", gated=False)
    patch_transport_factory()
    base = tmp_path / "b.json"
    base.write_text(
        json.dumps({"metrics": {"synth/source_chunk_coverage": 0.83}}),
        encoding="utf-8",
    )
    result = _run(
        [
            "client", "eval", "--dataset", "mvp", "--eval", "synth",
            "--against", str(base), "--plain",
        ]
    )
    assert result.exit_code == 0, result.stdout


def test_against_requires_single_dataset(tmp_path: Path) -> None:
    # No --dataset → multi-dataset envelope; rejected up front (before the run).
    result = _run(
        ["client", "eval", "--against", str(tmp_path / "b.json"), "--plain"]
    )
    assert result.exit_code != 0


def test_against_rejects_multiple_eval_modes(tmp_path: Path) -> None:
    result = _run(
        [
            "client", "eval", "--dataset", "mvp",
            "--eval", "retrieval", "--eval", "synth",
            "--against", str(tmp_path / "b.json"), "--plain",
        ]
    )
    assert result.exit_code != 0


def test_against_and_write_baseline_are_mutually_exclusive(tmp_path: Path) -> None:
    # Rejected before any network call → no fixtures needed.
    result = _run(
        [
            "client",
            "eval",
            "--dataset",
            "mvp",
            "--against",
            str(tmp_path / "b.json"),
            "--write-baseline",
            str(tmp_path / "c.json"),
            "--plain",
        ]
    )
    assert result.exit_code != 0
