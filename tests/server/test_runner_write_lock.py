"""F2 / RC-2 regression: the synth and lint-apply task runners must serialize
their storage-mutating call under the shared base write lock (the runtime's
``ingest_lock``).

Before F2 neither runner took a lock, so a synth could interleave with ingest,
delete, or a second synth — racing the same K-page rows + embed version with no
enclosing transaction. These tests pin the serialization at the runner level
(fast, no HTTP stack) with a negative control proving the probe actually
detects interleaving.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dikw_core import api as api_module
from dikw_core.server import lint_op as lint_op_module
from dikw_core.server import synth_op as synth_op_module
from dikw_core.server.tasks.store import TaskRow, TaskStatus


class _NullReporter:
    async def progress(self, **_: Any) -> None:
        return None

    async def log(self, level: str, message: str) -> None:
        return None

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        return None

    def cancel_token(self) -> Any:
        from dikw_core.progress import CancelToken

        return CancelToken()


class _ConcurrencyProbe:
    """Tracks the max number of guarded sections running at once."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def section(self) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            # Yield repeatedly so an unserialized peer would interleave here.
            for _ in range(8):
                await asyncio.sleep(0)
        finally:
            self.active -= 1


def _patch_synth(monkeypatch: Any, probe: _ConcurrencyProbe) -> None:
    # Provider construction + cfg load are stubbed — the test exercises the
    # runner's locking, not synth itself. ``no_embed=True`` means
    # ``build_embedder`` is never called, but stub it for safety.
    monkeypatch.setattr(
        synth_op_module, "load_config", lambda _p: SimpleNamespace(provider=None)
    )
    monkeypatch.setattr(synth_op_module, "build_llm", lambda *_a, **_k: object())
    monkeypatch.setattr(synth_op_module, "build_embedder", lambda *_a, **_k: object())

    async def fake_synth(*_a: Any, **_k: Any) -> api_module.SynthReport:
        await probe.section()
        return api_module.SynthReport()

    monkeypatch.setattr(synth_op_module.api, "synthesize", fake_synth)


async def test_two_synth_runners_serialize_under_shared_lock(
    monkeypatch: Any,
) -> None:
    probe = _ConcurrencyProbe()
    _patch_synth(monkeypatch, probe)
    lock = asyncio.Lock()
    r1 = synth_op_module.make_synth_runner(
        base_root=Path("/x"), force_all=False, no_embed=True, lock=lock
    )
    r2 = synth_op_module.make_synth_runner(
        base_root=Path("/x"), force_all=True, no_embed=True, lock=lock
    )
    await asyncio.gather(r1(_NullReporter()), r2(_NullReporter()))
    assert probe.max_active == 1, (
        "synth runners sharing ingest_lock must not run synthesize concurrently"
    )


async def test_synth_runners_without_shared_lock_overlap(monkeypatch: Any) -> None:
    # Negative control: ``lock=None`` makes each runner create its own lock,
    # so the api calls overlap — proving the probe detects interleaving and
    # that the shared lock above is what enforces serialization.
    probe = _ConcurrencyProbe()
    _patch_synth(monkeypatch, probe)
    r1 = synth_op_module.make_synth_runner(
        base_root=Path("/x"), force_all=False, no_embed=True
    )
    r2 = synth_op_module.make_synth_runner(
        base_root=Path("/x"), force_all=True, no_embed=True
    )
    await asyncio.gather(r1(_NullReporter()), r2(_NullReporter()))
    assert probe.max_active == 2, (
        "independent-lock runners should overlap (probe sanity check)"
    )


class _FakeProposeStore:
    """Minimal TaskStore stand-in exposing only ``get`` for one propose row."""

    def __init__(self, row: TaskRow) -> None:
        self._row = row

    async def get(self, task_id: str) -> TaskRow | None:
        return self._row if task_id == self._row.task_id else None


async def test_lint_apply_runner_holds_lock_during_apply(monkeypatch: Any) -> None:
    from dikw_core.domains.knowledge.lint_fix import ApplyReport, FixProposalReport

    lock = asyncio.Lock()
    observed: dict[str, bool] = {}

    async def fake_apply(*_a: Any, **_k: Any) -> ApplyReport:
        observed["locked"] = lock.locked()
        return ApplyReport()

    monkeypatch.setattr(lint_op_module.api, "lint_apply", fake_apply)

    propose_row = TaskRow(
        task_id="prop-1",
        op="lint.propose",
        status=TaskStatus.SUCCEEDED,
        created_at="2026-06-22T00:00:00.000Z",
        result=FixProposalReport().model_dump(mode="json"),
    )
    runner = lint_op_module.make_lint_apply_runner(
        base_root=Path("/x"),
        proposal_task_id="prop-1",
        task_store=_FakeProposeStore(propose_row),  # type: ignore[arg-type]
        pick=None,
        skip=None,
        lock=lock,
    )
    await runner(_NullReporter())
    assert observed.get("locked") is True, (
        "lint apply must hold ingest_lock while api.lint_apply runs"
    )


def _patch_lint_apply(monkeypatch: Any, probe: _ConcurrencyProbe) -> None:
    """Patch ``api.lint_apply`` to run the shared probe section."""
    from dikw_core.domains.knowledge.lint_fix import ApplyReport

    async def fake_apply(*_a: Any, **_k: Any) -> ApplyReport:
        await probe.section()
        return ApplyReport()

    monkeypatch.setattr(lint_op_module.api, "lint_apply", fake_apply)


def _make_lint_runner(
    lock: asyncio.Lock | None,
) -> Any:
    """A lint-apply runner over one SUCCEEDED propose row (read-only store)."""
    from dikw_core.domains.knowledge.lint_fix import FixProposalReport

    propose_row = TaskRow(
        task_id="prop-1",
        op="lint.propose",
        status=TaskStatus.SUCCEEDED,
        created_at="2026-06-22T00:00:00.000Z",
        result=FixProposalReport().model_dump(mode="json"),
    )
    return lint_op_module.make_lint_apply_runner(
        base_root=Path("/x"),
        proposal_task_id="prop-1",
        task_store=_FakeProposeStore(propose_row),  # type: ignore[arg-type]
        pick=None,
        skip=None,
        lock=lock,
    )


async def test_lint_apply_runners_serialize_under_shared_lock(
    monkeypatch: Any,
) -> None:
    # Two concurrent lint applies sharing ingest_lock must not overlap their
    # api.lint_apply (each re-projects K/W rows with no enclosing transaction).
    probe = _ConcurrencyProbe()
    _patch_lint_apply(monkeypatch, probe)
    lock = asyncio.Lock()
    r1 = _make_lint_runner(lock)
    r2 = _make_lint_runner(lock)
    await asyncio.gather(r1(_NullReporter()), r2(_NullReporter()))
    assert probe.max_active == 1, (
        "lint apply runners sharing ingest_lock must not run apply concurrently"
    )


async def test_synth_and_lint_apply_serialize_under_shared_lock(
    monkeypatch: Any,
) -> None:
    # The cross-operation guarantee: a synth and a lint apply sharing one
    # ingest_lock serialize against EACH OTHER (both write the same
    # deterministic doc_id rows + on-disk K pages).
    probe = _ConcurrencyProbe()
    _patch_synth(monkeypatch, probe)
    _patch_lint_apply(monkeypatch, probe)
    lock = asyncio.Lock()
    synth_runner = synth_op_module.make_synth_runner(
        base_root=Path("/x"), force_all=False, no_embed=True, lock=lock
    )
    lint_runner = _make_lint_runner(lock)
    await asyncio.gather(synth_runner(_NullReporter()), lint_runner(_NullReporter()))
    assert probe.max_active == 1, (
        "synth and lint apply sharing ingest_lock must not overlap"
    )


async def test_synth_runner_builds_providers_under_lock(monkeypatch: Any) -> None:
    # F2 corner case (Codex P2): the runner must take the write lock BEFORE
    # reloading cfg + building the LLM/embedder. Otherwise a concurrent ingest
    # holding the lock can flip the active embed version while synth waits, and
    # synth would then run api.synthesize (which re-resolves the new cfg) with
    # the STALE embedder built from the old cfg — storing vectors from the wrong
    # embedder under the freshly-active version. Pin that cfg load happens while
    # the lock is held (it builds providers right after).
    lock = asyncio.Lock()
    observed: dict[str, bool] = {}

    def _fake_load_config(_p: Any) -> Any:
        observed["config_under_lock"] = lock.locked()
        return SimpleNamespace(provider=None)

    monkeypatch.setattr(synth_op_module, "load_config", _fake_load_config)
    monkeypatch.setattr(synth_op_module, "build_llm", lambda *_a, **_k: object())
    monkeypatch.setattr(synth_op_module, "build_embedder", lambda *_a, **_k: object())

    async def _fake_synth(*_a: Any, **_k: Any) -> api_module.SynthReport:
        return api_module.SynthReport()

    monkeypatch.setattr(synth_op_module.api, "synthesize", _fake_synth)

    runner = synth_op_module.make_synth_runner(
        base_root=Path("/x"), force_all=False, no_embed=False, lock=lock
    )
    await runner(_NullReporter())
    assert observed.get("config_under_lock") is True, (
        "make_synth_runner must hold ingest_lock before reloading cfg / building "
        "providers, so they stay aligned with api.synthesize's cfg reload"
    )


async def test_synth_and_lint_apply_overlap_without_shared_lock(
    monkeypatch: Any,
) -> None:
    # Negative control for the cross-operation case: independent locks let the
    # synth and lint apply interleave, proving the shared lock above is what
    # serializes two DIFFERENT op kinds (not merely synth-vs-synth).
    probe = _ConcurrencyProbe()
    _patch_synth(monkeypatch, probe)
    _patch_lint_apply(monkeypatch, probe)
    synth_runner = synth_op_module.make_synth_runner(
        base_root=Path("/x"), force_all=False, no_embed=True
    )
    lint_runner = _make_lint_runner(None)
    await asyncio.gather(synth_runner(_NullReporter()), lint_runner(_NullReporter()))
    assert probe.max_active == 2, (
        "independent-lock synth + lint apply should overlap (probe sanity check)"
    )
