"""Wallclock perf budgets for the embed-overhaul (slice 8).

Budget rationale: the user-visible promises of this PR are
  (a) cold ingest is fast on hermetic FakeEmbeddings (no network);
  (b) re-ingest is essentially free (zero provider calls + sub-2s);
  (c) eval re-runs hit the snapshot cache and never re-ingest.

Each test asserts an absolute wallclock budget (3-5x the dev-machine
median, sized to absorb CI jitter). The ``@pytest.mark.perf`` marker
lets a noisy CI fall back to ``pytest -m 'not perf'`` without losing
correctness coverage.

Hermetic — uses ``FakeEmbeddings`` + the in-repo synthetic corpus; no
network, no real provider.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.eval import runner as eval_runner
from dikw_core.eval.dataset import load_dataset
from dikw_core.eval.runner import run_eval

from .fakes import CountingEmbedder, init_test_base

pytestmark = pytest.mark.perf


# Synthetic corpus sized so the streaming + cache paths are exercised
# end-to-end without taking long on hermetic FakeEmbeddings.
NUM_DOCS = 200
COLD_BUDGET_S = 4.0
WARM_REINGEST_BUDGET_S = 2.0
SNAPSHOT_HIT_BUDGET_S = 1.0


def _build_synthetic_wiki(tmp_path: Path) -> Path:
    """Create a wiki + ``NUM_DOCS`` deterministic markdown docs."""
    wiki = tmp_path / "knowledge"
    init_test_base(wiki, description="perf-budget wiki")
    sources = wiki / "sources" / "perf"
    sources.mkdir(parents=True, exist_ok=True)
    for i in range(NUM_DOCS):
        (sources / f"doc{i:04d}.md").write_text(
            f"# Doc {i}\n\nDeterministic body content for perf doc number {i}.\n",
            encoding="utf-8",
        )
    # Bump batch size down so streaming visibility is real (multiple
    # provider calls happen on the cold path).
    cfg = wiki / "dikw.yml"
    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace(
            "embedding_batch_size: 64",
            "embedding_batch_size: 16",
        ),
        encoding="utf-8",
    )
    return wiki


async def test_cold_ingest_under_budget(tmp_path: Path) -> None:
    """P1: cold ingest of NUM_DOCS docs finishes within COLD_BUDGET_S."""
    wiki = _build_synthetic_wiki(tmp_path)
    embedder = CountingEmbedder()
    t0 = time.perf_counter()
    report = await api.ingest(wiki, embedder=embedder)
    elapsed = time.perf_counter() - t0
    assert report.embedded == NUM_DOCS
    assert elapsed < COLD_BUDGET_S, (
        f"cold ingest of {NUM_DOCS} docs took {elapsed:.2f}s "
        f"(budget {COLD_BUDGET_S}s); embedder calls = {embedder.embed_calls}"
    )


async def test_warm_reingest_zero_provider_calls_under_budget(
    tmp_path: Path,
) -> None:
    """P2: re-ingest of unchanged corpus is fast + zero provider calls.

    Doc-level shortcut (existing.hash == parsed.hash) skips every doc
    on the second run; the resume scan finds chunk_embed_meta already filled,
    so to_embed is empty. No provider invocations, no SQL writes
    beyond the per-doc scan.
    """
    wiki = _build_synthetic_wiki(tmp_path)
    cold = CountingEmbedder()
    await api.ingest(wiki, embedder=cold)
    assert cold.embed_calls > 0  # cold path actually ran

    warm = CountingEmbedder()
    t0 = time.perf_counter()
    await api.ingest(wiki, embedder=warm)
    elapsed = time.perf_counter() - t0
    assert warm.embed_calls == 0, (
        f"warm re-ingest hit the provider {warm.embed_calls}x "
        "(should be zero — doc-level shortcut + resume scan find nothing)"
    )
    assert elapsed < WARM_REINGEST_BUDGET_S, (
        f"warm re-ingest took {elapsed:.2f}s (budget {WARM_REINGEST_BUDGET_S}s)"
    )


async def test_eval_snapshot_cache_hit_under_budget(tmp_path: Path) -> None:
    """P3: a warmed eval-snapshot replay finishes within SNAPSHOT_HIT_BUDGET_S.

    Cold run builds + caches the wiki; second run hits the cache and
    jumps straight to ``_run_queries``. Spy on api.ingest to confirm
    the second pass never re-ingests.
    """
    # Tiny dataset: 2 docs + 1 query is enough to exercise the cache
    # decision; we're measuring the no-ingest fast path.
    ds = tmp_path / "ds"
    (ds / "corpus").mkdir(parents=True)
    (ds / "corpus" / "alpha.md").write_text(
        "# Alpha\n\nAlpha content.\n", encoding="utf-8"
    )
    (ds / "corpus" / "beta.md").write_text(
        "# Beta\n\nBeta content.\n", encoding="utf-8"
    )
    (ds / "dataset.yaml").write_text(
        "name: perf_snapshot\ndescription: snapshot perf test\n"
        "thresholds:\n  hit_at_3: 0.0\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n  - q: alpha content\n    expect_any: [alpha]\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    cache_root = tmp_path / "cache"

    # Cold: builds + writes snapshot.
    await run_eval(spec, cache_root=cache_root)

    # Spy on api.ingest to prove the warm pass skips ingestion entirely.
    ingest_calls = 0
    real_ingest = eval_runner.api.ingest

    async def spy_ingest(*args: object, **kwargs: object) -> object:
        nonlocal ingest_calls
        ingest_calls += 1
        return await real_ingest(*args, **kwargs)

    eval_runner.api.ingest = spy_ingest  # type: ignore[assignment]
    try:
        t0 = time.perf_counter()
        await run_eval(spec, cache_root=cache_root)
        elapsed = time.perf_counter() - t0
    finally:
        eval_runner.api.ingest = real_ingest  # type: ignore[assignment]

    assert ingest_calls == 0, "snapshot cache hit must skip ingest"
    assert elapsed < SNAPSHOT_HIT_BUDGET_S, (
        f"snapshot-cache hit took {elapsed:.2f}s "
        f"(budget {SNAPSHOT_HIT_BUDGET_S}s)"
    )

    # Sanity: the materialised snapshot directory exists.
    snapshot_dirs = list((cache_root / spec.name).iterdir())
    assert any(d.is_dir() for d in snapshot_dirs)
    # Cleanup: remove generated cache so subsequent test runs are cold.
    shutil.rmtree(cache_root, ignore_errors=True)
