"""Unit tests for the eval-runner helpers in ``server/synth_op.py``.

Two regressions guarded here, both surfaced by the 2026-05-12 codex
adversarial review of the synth eval framework:

  * **Legacy default**: A ``POST /v1/eval`` body that pre-dates
    ``eval_modes`` must still run retrieval-only, even if the
    dataset declares ``synth``. Otherwise existing clients would
    silently start invoking an LLM.

  * **False green on empty intersection**: ``--eval synth`` against
    a retrieval-only dataset must surface as ``eval_mode_unavailable``,
    not as a vacuous ``passed=True``.

The eval-runner end-to-end path is exercised in
``tests/test_synth_quality.py`` + ``tests/test_eval_cli.py``;
this file keeps the dispatch decision unit-testable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dikw_core.server.errors import BadRequest
from dikw_core.server.synth_op import _resolve_eval_modes, _resolve_judge_sample


def test_resolve_eval_modes_defaults_to_retrieval_only_on_synth_dataset() -> None:
    """Legacy body (eval_modes omitted) on a synth-capable dataset must
    still pick retrieval — synth opt-in is explicit."""
    spec = SimpleNamespace(modes=["retrieval", "synth"])
    assert _resolve_eval_modes(spec, None) == ["retrieval"]


def test_resolve_eval_modes_defaults_to_empty_when_retrieval_not_declared() -> None:
    """A synth-only dataset with no eval_modes specified returns empty —
    the caller raises ``eval_mode_unavailable`` rather than silently
    running synth."""
    spec = SimpleNamespace(modes=["synth"])
    assert _resolve_eval_modes(spec, None) == []


def test_resolve_eval_modes_explicit_synth_on_retrieval_only_returns_empty() -> None:
    """Explicit ``--eval synth`` on a dataset that doesn't declare synth
    returns empty — the loop layer turns that into the
    ``eval_mode_unavailable`` BadRequest."""
    spec = SimpleNamespace(modes=["retrieval"])
    assert _resolve_eval_modes(spec, ["synth"]) == []


def test_resolve_eval_modes_explicit_intersection_preserves_order() -> None:
    """Explicit list is honored as-given (modulo dataset support)."""
    spec = SimpleNamespace(modes=["retrieval", "synth"])
    assert _resolve_eval_modes(spec, ["synth", "retrieval"]) == [
        "synth",
        "retrieval",
    ]
    assert _resolve_eval_modes(spec, ["synth"]) == ["synth"]


# ---- _resolve_judge_sample (``--judge-sample auto``) ------------------------


def test_resolve_judge_sample_auto_uses_calibrated_size() -> None:
    from dikw_core.eval.judge import recommended_judge_sample

    assert _resolve_judge_sample("auto") == recommended_judge_sample()


def test_resolve_judge_sample_passes_through_positive_int_and_none() -> None:
    assert _resolve_judge_sample(7) == 7
    assert _resolve_judge_sample(None) is None


def test_resolve_judge_sample_rejects_nonpositive() -> None:
    with pytest.raises(BadRequest):
        _resolve_judge_sample(0)
