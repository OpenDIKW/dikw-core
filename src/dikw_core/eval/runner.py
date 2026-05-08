"""Runner — execute a ``DatasetSpec`` end-to-end and return an ``EvalReport``.

The contract is deliberately minimal:

- Input: a ``DatasetSpec`` (from ``load_dataset`` or constructed in code).
- Side effect: a temporary wiki is created, the corpus ingested, hybrid
  search run for each query, then everything is cleaned up.
- Output: ``EvalReport`` with per-metric values, the dataset's thresholds
  echoed for comparison, per-query diagnostic rows for the CLI's failure
  table, and a computed ``passed`` flag.

The runner is hermetic by default — ``FakeEmbeddings`` from
``dikw_core.eval.fake_embedder`` gives deterministic bag-of-words vectors
with no network or API-key dependency. Callers who want real-vector eval
pass their own embedder (e.g., via ``build_embedder(cfg.provider)``).

A single eval can also fan out to all three retrieval modes
(``mode="all"``) — the corpus is still ingested only once, and each
query is then run once per mode against the same storage connection.
This is the workflow used to compare bm25 vs vector vs hybrid against
public-benchmark baselines (BEIR, CMTEB).

When the dataset's ``targets.yaml`` declares chunks and/or assets, the
runner additionally emits ``chunk/<metric>`` and ``asset/<metric>``
keys in the report — same fused ranking, different identity projection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from .. import api
from ..config import (
    CONFIG_FILENAME,
    AssetsConfig,
    MultimodalEmbedConfig,
    ProviderConfig,
    RetrievalConfig,
    dump_config_yaml,
)
from ..domains.data.backends import parse_any
from ..domains.data.hashing import hash_file
from ..domains.info.search import HybridSearcher, MultimodalSearch, RetrievalMode
from ..progress import NoopReporter, ProgressReporter
from ..providers import (
    EmbeddingProvider,
    MultimodalEmbeddingProvider,
    build_multimodal_embedder,
)
from ..schemas import ChunkRecord, DocumentRecord, Hit
from ..storage import Storage, build_storage
from ..storage._schema import SCHEMA_VERSION
from ..storage.base import NotSupported
from .dataset import DatasetSpec
from .fake_embedder import FakeEmbeddings
from .metrics import (
    mean_hit_at_k,
    mean_ndcg_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
)

# Limit per query — large enough to compute recall@100. The same ranked
# list feeds hit@k / nDCG@10 (which slice from the top).
SEARCH_LIMIT = 100

# Metric column order for tables and per-view rendering. Single source of
# truth; cli.py imports this so terminal table and report.diagnostics_table
# stay aligned.
METRIC_KEYS: tuple[str, ...] = (
    "hit_at_3",
    "hit_at_10",
    "mrr",
    "ndcg_at_10",
    "recall_at_100",
)

EvalMode = RetrievalMode | Literal["all"]

CacheMode = Literal["read_write", "rebuild", "off"]

Granularity = Literal["doc", "chunk", "asset"]

logger = logging.getLogger(__name__)


def _default_snapshot_root() -> Path:
    """Resolve ``<repo>/evals/.cache/snapshots/`` by walking up from this file.

    Used as the default cache root when ``run_eval(cache_root=None)``.
    Raises ``EvalError`` if ``evals/`` can't be located (e.g., the wheel
    install layout has no sibling ``evals/``); the caller should pass an
    explicit ``cache_root`` in that case or use ``cache_mode="off"``.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "evals").is_dir():
            return parent / "evals" / ".cache" / "snapshots"
    raise EvalError(
        "could not locate evals/ root for snapshot cache; pass cache_root "
        "explicitly or use cache_mode='off'"
    )


def _corpus_cache_key(
    spec: DatasetSpec,
    model: str,
    dim: int | None,
    *,
    mm_fingerprint: str | None = None,
) -> str:
    """Stable cache key combining dataset name, model, dim, corpus hash, schema.

    Algorithm:
      1. sha256 over sorted (rel_posix_path, file_bytes) pairs in corpus_dir
      2. take first 8 hex chars (collision ~1/4B per dataset+model — acceptable;
         ``cache_mode="rebuild"`` is the escape hatch)
      3. format: ``{dataset}/{model}__{dim}__{digest}__mm{mm}__sf{N}``

    ``as_posix()`` keeps the key cross-platform (Windows / Linux yield
    the same digest). Embedding ``dim=None`` is rendered as ``0`` so the
    key is still usable when the provider doesn't pin a dim.

    ``mm_fingerprint`` carves out cache space per multimodal identity
    (provider/model/dim) so a hermetic snapshot built without an asset
    index can't be silently reused by a real-vector multimodal eval.
    ``None`` is rendered as ``0`` for back-compat with pre-mm caches.

    NOTE: ``RetrievalConfig`` (rrf_k, weights, fusion, graph_*, …) is
    NOT in the key. Re-running with a different retrieval config under
    ``cache_mode="read_write"`` silently reuses the first run's
    ``dikw.yml`` — the snapshot's wiki carries the old block. Retrieval
    ablations must use ``cache_mode="off"`` until this is fixed.
    """
    h = hashlib.sha256()
    for path in sorted(spec.corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        h.update(path.relative_to(spec.corpus_dir).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    digest = h.hexdigest()[:8]
    dim_str = str(dim if dim is not None else 0)
    mm_str = mm_fingerprint or "0"
    # ``sf`` = schema fingerprint. Any change to ``SCHEMA_VERSION`` (or
    # the key name behind it) must invalidate every snapshot — opening
    # an old snapshot under a new code version would risk a fingerprint
    # mismatch error or silent schema drift. The ``sf`` prefix is also
    # distinct from the legacy ``mig`` prefix so caches stamped before
    # the per-migration-counter framework was deleted never match.
    return (
        f"{spec.name}/{model}__{dim_str}__{digest}__mm{mm_str}__sf{SCHEMA_VERSION}"
    )


class EvalError(RuntimeError):
    """Raised by the runner when a dataset can't be executed end-to-end."""


@dataclass(frozen=True)
class PerQueryRow:
    """Per-query diagnostic capture across all three views.

    ``ranked_docs`` is always populated; ``ranked_chunks`` /
    ``ranked_assets`` are populated only when the dataset declares
    chunk / asset targets. Empty ``expect_*_any`` excludes the query
    from the corresponding view's mean (rather than counting as a miss).
    """

    q: str
    q_id: str | None
    expect_doc_any: tuple[str, ...]
    expect_chunk_any: tuple[str, ...]
    expect_asset_any: tuple[str, ...]
    ranked_docs: tuple[str, ...]
    ranked_chunks: tuple[str, ...]
    ranked_assets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "q": self.q,
            "expect_any": list(self.expect_doc_any),
            "ranked": list(self.ranked_docs),
        }
        if self.q_id is not None:
            d["id"] = self.q_id
        if self.expect_chunk_any:
            d["expect_chunk_any"] = list(self.expect_chunk_any)
            d["ranked_chunks"] = list(self.ranked_chunks)
        if self.expect_asset_any:
            d["expect_asset_any"] = list(self.expect_asset_any)
            d["ranked_assets"] = list(self.ranked_assets)
        return d


@dataclass(frozen=True)
class NegativeRow:
    """One ``expect_none=True`` query's observed retrieval. Diagnostic only —
    retrieval always returns something from a non-empty corpus, so there's no
    pass/fail here yet; scoring negatives requires a score threshold or an
    answer-level judge that we don't have in Phase A.
    """

    q: str
    ranked: tuple[str, ...]  # doc stems, top SEARCH_LIMIT

    def to_dict(self) -> dict[str, Any]:
        return {"q": self.q, "ranked": list(self.ranked)}


class EvalReport(BaseModel):
    """Result of a single ``run_eval`` invocation.

    ``metrics`` keys take three shapes:

    * Unprefixed (``hit_at_3``, …) — back-compat alias of the doc view
      under the canonical retrieval mode. Existing dataset thresholds
      keep gating on these.
    * ``<view>/<metric>`` (``doc/hit_at_3``, ``chunk/hit_at_3``,
      ``asset/hit_at_3``) — per-granularity metrics; ``chunk/`` and
      ``asset/`` only present when ``targets.yaml`` declares them and at
      least one query opts in.
    * ``<retrieval_mode>/<metric>`` (``bm25/hit_at_3``, …) — only
      emitted when ``mode="all"``. Doc-view only (sweep_rrf consumer
      doesn't need chunk/asset for its purpose).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_name: str
    metrics: dict[str, float]
    thresholds: dict[str, float]
    per_query: list[dict[str, Any]] = Field(default_factory=list)
    negative_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    # Modes actually executed for this run. Single-element list for the
    # default; three elements for ``mode="all"``. Used by the CLI to pick
    # between the single-table and ablation-table renderings.
    modes: list[str] = Field(default_factory=lambda: ["hybrid"])
    # Views (granularities) the report scored. Always includes "doc";
    # adds "chunk" / "asset" when the dataset has targets and queries
    # opting into those views.
    views: list[str] = Field(default_factory=lambda: ["doc"])

    @property
    def passed(self) -> bool:
        """True iff every threshold is met (or no thresholds are configured)."""
        for key, floor in self.thresholds.items():
            value = self.metrics.get(key)
            if value is None:
                # Threshold for a metric the runner didn't compute — fail loudly
                # rather than silently pass.
                return False
            if value < floor:
                return False
        return True

    def iter_metric_rows(
        self,
    ) -> Iterator[tuple[str, float | None, float | None]]:
        """Yield ``(key, value, threshold)`` once per ``view * metric``
        the report scored. The key uses the unprefixed form for the doc
        view (``hit_at_3``) so legacy thresholds keep gating, and
        ``<view>/<metric>`` for chunk / asset views.

        Single source of truth shared by ``diagnostics_table`` and the
        CLI's rich-table renderer — keeps row ordering and dedup logic
        consistent across both surfaces.
        """
        seen: set[str] = set()
        for view in self.views:
            for metric in METRIC_KEYS:
                key = metric if view == "doc" else f"{view}/{metric}"
                if key in seen:
                    continue
                seen.add(key)
                if key not in self.metrics and key not in self.thresholds:
                    continue
                yield key, self.metrics.get(key), self.thresholds.get(key)

    def diagnostics_table(self) -> str:
        """Plain-text per-query breakdown for test-assertion messages."""
        lines = [
            f"Dataset: {self.dataset_name}",
            "metric              value    threshold   result",
            "------------------  -------  ----------  ------",
        ]
        for key, val, thr in self.iter_metric_rows():
            v = val if val is not None else float("nan")
            verdict = "—" if thr is None else ("pass" if v >= thr else "FAIL")
            thr_str = f"{thr:.3f}" if thr is not None else "    -"
            lines.append(f"{key:18s}  {v:6.3f}  {thr_str:9s}  {verdict}")
        lines.append("")
        lines.append("per-query top-5 (doc view):")
        for row in self.per_query:
            q_short = row["q"] if len(row["q"]) <= 60 else row["q"][:57] + "..."
            top5 = row["ranked"][:5]
            mark = "✓" if any(e in top5 for e in row["expect_any"]) else "✗"
            lines.append(f"  {mark} {q_short}")
            lines.append(f"       expected: {row['expect_any']}")
            lines.append(f"       top-5:    {top5}")
        return "\n".join(lines)


def _resolve_modes(mode: EvalMode) -> list[RetrievalMode]:
    if mode == "all":
        return list(get_args(RetrievalMode))
    return [mode]


def _compute_view_metrics(
    positives: list[PerQueryRow], *, view: Granularity
) -> dict[str, float]:
    """Compute hit@k / mrr / nDCG@10 / recall@100 for one granularity view.

    Only positives whose corresponding ``expect_*_any`` is non-empty
    contribute to the mean — queries opting into a different view stay
    out of this view's denominator (rather than dragging the score to
    zero).
    """
    if view == "doc":
        pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
            (r.ranked_docs, r.expect_doc_any)
            for r in positives
            if r.expect_doc_any
        ]
    elif view == "chunk":
        pairs = [
            (r.ranked_chunks, r.expect_chunk_any)
            for r in positives
            if r.expect_chunk_any
        ]
    else:  # asset
        pairs = [
            (r.ranked_assets, r.expect_asset_any)
            for r in positives
            if r.expect_asset_any
        ]
    if not pairs:
        return {}
    return {
        "hit_at_3": mean_hit_at_k(pairs, 3),
        "hit_at_10": mean_hit_at_k(pairs, 10),
        "mrr": mean_reciprocal_rank(pairs),
        "ndcg_at_10": mean_ndcg_at_k(pairs, 10),
        "recall_at_100": mean_recall_at_k(pairs, 100),
    }


async def run_eval(
    spec: DatasetSpec,
    *,
    embedder: EmbeddingProvider | None = None,
    provider_config: ProviderConfig | None = None,
    retrieval_config: RetrievalConfig | None = None,
    assets_config: AssetsConfig | None = None,
    multimodal_embedder: MultimodalEmbeddingProvider | None = None,
    mode: EvalMode = "hybrid",
    raw_dump_path: Path | None = None,
    cache_mode: CacheMode = "read_write",
    cache_root: Path | None = None,
    reporter: ProgressReporter | None = None,
) -> EvalReport:
    """Run a single dataset end-to-end; return metrics + diagnostics.

    ``embedder`` + ``provider_config`` are the two text-side knobs:

    * Both ``None`` (the default) → hermetic: ``FakeEmbeddings`` + a
      ``ProviderConfig`` with ``embedding_model="fake"`` (everything else
      default). No network, no keys, <1s.
    * Both set → real-vector eval: the caller built an embedder from a
      wiki's ``ProviderConfig`` and hands both in. The runner serialises
      that config into the temp wiki's ``dikw.yml`` so ``api.ingest`` picks
      up vendor-specific ``embedding_batch_size`` / ``embedding_dim``
      exactly as the source wiki has them. Without this, a Gitee-configured
      provider would ingest at ``batch_size=64`` and get HTTP 400.

    ``assets_config`` + ``multimodal_embedder`` activate the asset
    retrieval leg end-to-end:

    * Pass an ``AssetsConfig`` whose ``.multimodal`` is populated to
      have the temp wiki's ``dikw.yml`` carry the same block — without
      this, ``cfg.assets.multimodal`` is ``None`` inside the eval and
      ``vec_search_assets`` never runs even for image-targeted queries.
    * Pass a ``MultimodalEmbeddingProvider`` so ``api.ingest`` can
      build the asset vectors. The cache key includes a multimodal
      identity fingerprint so a hermetic snapshot can't be silently
      reused by a multimodal run.

    ``mode`` selects which retrieval leg(s) to score: ``"hybrid"``
    (default, BM25+vec via RRF), ``"bm25"``, ``"vector"``, or ``"all"``
    (run all three sequentially against the same ingested corpus).

    ``raw_dump_path`` — when set (and ``mode="all"``), appends one JSONL
    row per (query, mode) capturing the top-``SEARCH_LIMIT`` ranked doc
    stems plus ``expect_any`` / ``expect_none``. Callers supply a path
    they've already truncated (or never existed); the runner only
    appends. Downstream ``evals/tools/sweep_rrf.py`` reads this to
    re-fuse offline at arbitrary ``(rrf_k, weights)`` — avoiding a
    second expensive embedding pass. For single-mode runs the flag is
    silently a no-op since sweep needs both legs' rankings.
    """
    if not spec.corpus_dir.is_dir():
        raise EvalError(f"corpus directory not found: {spec.corpus_dir}")

    _reporter: ProgressReporter = reporter or NoopReporter()
    effective_embedder: EmbeddingProvider = embedder or FakeEmbeddings()
    effective_provider_cfg = provider_config or ProviderConfig(
        embedding_model="fake",
        embedding_dim=64,  # matches dikw_core.eval.fake_embedder.EMBED_DIM
        embedding_revision="",
        embedding_normalize=True,
        embedding_distance="cosine",
    )
    effective_retrieval_cfg = retrieval_config or RetrievalConfig()
    effective_assets_cfg = assets_config or AssetsConfig()
    modes = _resolve_modes(mode)

    async def _build(target: Path) -> Path:
        wiki = _materialise_wiki(
            target,
            spec,
            provider_cfg=effective_provider_cfg,
            retrieval_cfg=effective_retrieval_cfg,
            assets_cfg=effective_assets_cfg,
        )
        _copy_corpus(spec.corpus_dir, wiki / "sources")
        await api.ingest(
            wiki,
            embedder=effective_embedder,
            multimodal_embedder=multimodal_embedder,
        )
        return wiki

    mm_fingerprint = _multimodal_fingerprint(effective_assets_cfg.multimodal)

    if cache_mode == "off":
        # Original behaviour: throwaway temp dir, no cache touched.
        with tempfile.TemporaryDirectory(prefix="dikw-eval-") as tmp:
            await _reporter.progress(phase="ingest", current=0, total=0)
            wiki = await _build(Path(tmp))
            await _reporter.progress(phase="ingest", current=1, total=1)
            per_mode = await _run_queries(
                wiki,
                spec,
                embedder=effective_embedder,
                embedding_model=effective_provider_cfg.embedding_model,
                modes=modes,
                reporter=_reporter,
            )
    else:
        # read_write or rebuild: persistent snapshot under cache_root.
        root = cache_root if cache_root is not None else _default_snapshot_root()
        key = _corpus_cache_key(
            spec,
            effective_provider_cfg.embedding_model,
            effective_provider_cfg.embedding_dim,
            mm_fingerprint=mm_fingerprint,
        )
        cache_dir = root / key
        partial_dir = cache_dir.parent / (cache_dir.name + ".partial")

        if cache_mode == "rebuild":
            # Force cold rebuild: drop both the final and any half-built
            # leftover from a prior crashed run.
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            if partial_dir.exists():
                shutil.rmtree(partial_dir)

        if not cache_dir.exists():
            # Cache miss → build under .partial, atomic-rename on success.
            if partial_dir.exists():
                shutil.rmtree(partial_dir)
            partial_dir.mkdir(parents=True)
            await _reporter.progress(phase="ingest", current=0, total=0)
            await _build(partial_dir)
            await _reporter.progress(phase="ingest", current=1, total=1)
            cache_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(partial_dir, cache_dir)
        else:
            await _reporter.progress(
                phase="ingest",
                current=1,
                total=1,
                detail={"cache_hit": True},
            )
        wiki = cache_dir / "wiki"

        per_mode = await _run_queries(
            wiki,
            spec,
            embedder=effective_embedder,
            embedding_model=effective_provider_cfg.embedding_model,
            modes=modes,
            reporter=_reporter,
        )

    # Metrics scored only over positive queries. Including negatives would
    # silently drag hit@k toward 0.0 (retrieval always returns something
    # from a non-empty corpus) and tie pass/fail to negative count.
    metrics: dict[str, float] = {}
    canonical_mode: RetrievalMode = (
        "hybrid" if "hybrid" in modes else modes[0]
    )
    canonical_positives = per_mode[canonical_mode][0]
    canonical_negatives = per_mode[canonical_mode][1]

    # Per-retrieval-mode metrics (doc-view only — sweep_rrf consumer
    # doesn't need chunk/asset).
    if len(modes) > 1:
        for m in modes:
            positives_m, _ = per_mode[m]
            for k, v in _compute_view_metrics(positives_m, view="doc").items():
                metrics[f"{m}/{k}"] = v

    # Doc view always emits both the unprefixed legacy alias and the
    # ``doc/`` namespaced form. Chunk/asset views only emit when their
    # targets exist AND at least one query opts in.
    views_emitted: list[str] = ["doc"]
    for k, v in _compute_view_metrics(canonical_positives, view="doc").items():
        metrics[f"doc/{k}"] = v
        metrics[k] = v
    optional_views: list[tuple[Granularity, bool]] = [
        ("chunk", bool(spec.targets.chunks)),
        ("asset", bool(spec.targets.assets)),
    ]
    for view, has_targets in optional_views:
        if not has_targets:
            continue
        view_metrics = _compute_view_metrics(canonical_positives, view=view)
        if not view_metrics:
            continue
        for k, v in view_metrics.items():
            metrics[f"{view}/{k}"] = v
        views_emitted.append(view)

    if raw_dump_path is not None and len(modes) > 1:
        _dump_raw_ranked(raw_dump_path, spec.name, per_mode)

    return EvalReport(
        dataset_name=spec.name,
        metrics=metrics,
        thresholds=dict(spec.thresholds),
        per_query=[r.to_dict() for r in canonical_positives],
        negative_diagnostics=[r.to_dict() for r in canonical_negatives],
        modes=list(modes),
        views=views_emitted,
    )


def _materialise_wiki(
    tmp_root: Path,
    spec: DatasetSpec,
    *,
    provider_cfg: ProviderConfig,
    retrieval_cfg: RetrievalConfig,
    assets_cfg: AssetsConfig,
) -> Path:
    """Scaffold a throwaway wiki + dikw.yml that matches ``spec``.

    ``provider_cfg`` / ``retrieval_cfg`` / ``assets_cfg`` are copied
    verbatim into the written ``dikw.yml``. Downstream ``api.ingest``
    reads the provider + assets blocks; ``_run_queries`` re-loads the
    whole file to build ``HybridSearcher``, which picks up the retrieval
    + multimodal blocks. This means eval reproducibly measures whatever
    fusion knobs the caller passed, including the asset-vector leg.
    """
    wiki = tmp_root / "wiki"
    api.init_wiki(wiki, description=f"eval/{spec.name}")

    from ..config import default_config  # local: keep module import light

    cfg = default_config(description=f"eval/{spec.name}")
    cfg.provider = provider_cfg
    cfg.retrieval = retrieval_cfg
    cfg.assets = assets_cfg
    (wiki / CONFIG_FILENAME).write_text(dump_config_yaml(cfg), encoding="utf-8")
    return wiki


def _multimodal_fingerprint(mm_cfg: MultimodalEmbedConfig | None) -> str | None:
    """Stable identity string for ``assets.multimodal`` to fingerprint
    into the eval snapshot cache key. Returns ``None`` when the leg is
    disabled — ``_corpus_cache_key`` then renders ``mm0`` and the
    pre-multimodal cache lineage stays untouched.
    """
    if mm_cfg is None:
        return None
    return (
        f"{mm_cfg.provider}@{mm_cfg.model}@{mm_cfg.revision or '0'}"
        f"@{mm_cfg.dim}@{mm_cfg.distance}"
    )


def _dump_raw_ranked(
    path: Path,
    dataset_name: str,
    per_mode: dict[RetrievalMode, tuple[list[PerQueryRow], list[NegativeRow]]],
) -> None:
    """Append per-mode ranked lists as JSONL.

    One row per (query, mode). Positives carry ``expect_any`` and (when
    present) ``expect_chunk_any`` / ``expect_asset_any`` plus their
    matching ``ranked_chunks`` / ``ranked_assets`` lists; negatives carry
    ``expect_none: true`` with empty positive fields. Consumers join on
    ``q_id`` to get each query's bm25/vector/hybrid rankings side-by-side.
    Positive and negative ``q_id``s are independent namespaces — any
    joiner must filter by ``expect_none`` before keying.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for mode, (positives, negatives) in per_mode.items():
            for q_id, row in enumerate(positives):
                payload: dict[str, Any] = {
                    "dataset": dataset_name,
                    "mode": mode,
                    "q_id": q_id,
                    "q": row.q,
                    "expect_any": list(row.expect_doc_any),
                    "expect_none": False,
                    "ranked": list(row.ranked_docs),
                }
                if row.expect_chunk_any:
                    payload["expect_chunk_any"] = list(row.expect_chunk_any)
                    payload["ranked_chunks"] = list(row.ranked_chunks)
                if row.expect_asset_any:
                    payload["expect_asset_any"] = list(row.expect_asset_any)
                    payload["ranked_assets"] = list(row.ranked_assets)
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            for q_id, neg in enumerate(negatives):
                f.write(
                    json.dumps(
                        {
                            "dataset": dataset_name,
                            "mode": mode,
                            "q_id": q_id,
                            "q": neg.q,
                            "expect_any": [],
                            "expect_none": True,
                            "ranked": list(neg.ranked),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def _copy_corpus(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)
        elif item.is_dir():
            shutil.copytree(item, dest / item.name, dirs_exist_ok=True)


@dataclass
class _DriverResult:
    per_query: list[PerQueryRow] = field(default_factory=list)


def _resolve_asset_targets(spec: DatasetSpec) -> dict[str, str]:
    """Hash each ``targets.assets[].path`` into the runtime ``asset_id``
    (sha256 hex) that ``materialize_asset`` produces during ingest.
    Pure file-system; no storage round-trip.
    """
    out: dict[str, str] = {}
    for at in spec.targets.assets:
        abs_path = spec.corpus_dir / at.path
        if not abs_path.is_file():
            raise EvalError(
                f"asset target {at.id!r}: file not found at {abs_path}"
            )
        try:
            out[at.id] = hash_file(abs_path)
        except OSError as e:
            raise EvalError(
                f"asset target {at.id!r}: cannot hash {abs_path}: {e}"
            ) from e
    return out


async def _resolve_chunk_targets(
    spec: DatasetSpec,
    storage: Storage,
) -> dict[str, tuple[str, int]]:
    """Resolve each ``targets.chunks[].anchor`` to ``(doc_stem, seq)`` of
    the chunk covering the anchor's char position in the doc body.

    Loud failure on any inconsistency — anchor missing / ambiguous /
    no covering chunk — since eval datasets are authored, not inferred.
    """
    if not spec.targets.chunks:
        return {}

    docs: list[DocumentRecord] = list(await storage.list_documents())
    docs_by_stem: dict[str, DocumentRecord] = {}
    for d in docs:
        stem = Path(d.path).stem
        if stem in docs_by_stem:
            raise EvalError(
                f"corpus has two ingested docs with stem {stem!r} "
                f"({docs_by_stem[stem].path!r} and {d.path!r}); "
                f"chunk-target resolution requires unique stems."
            )
        docs_by_stem[stem] = d

    needed_stems = {ct.doc for ct in spec.targets.chunks}
    missing = needed_stems - docs_by_stem.keys()
    if missing:
        raise EvalError(
            f"chunk targets reference unknown doc stems {sorted(missing)}; "
            f"ingested wiki has {sorted(docs_by_stem)}"
        )

    # Read each unique doc body + load its chunks once. Storage calls
    # are serialized: ``SQLiteStorage`` uses ``check_same_thread=False``
    # but a single ``sqlite3.Connection`` can't safely service two
    # ``conn.execute(...)`` calls at once. ``asyncio.gather`` over
    # ``list_chunks`` (which dispatches via ``asyncio.to_thread``) trips
    # ``sqlite3.InterfaceError`` on Python 3.13's faster asyncio
    # scheduler. ``parse_any`` matches the ingest-time backend dispatch
    # (markdown + html) so the body coordinate space matches the
    # chunker's — no .md hardcode that would break HTML corpora.
    bodies_by_stem: dict[str, str] = {}
    chunks_by_stem: dict[str, list[ChunkRecord]] = {}
    for stem in needed_stems:
        candidates = sorted(spec.corpus_dir.glob(f"{stem}.*"))
        source: Path | None = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise EvalError(
                f"corpus file for stem {stem!r} not found in {spec.corpus_dir}"
            )
        bodies_by_stem[stem] = parse_any(source, rel_path=source.name).body
        chunks_by_stem[stem] = await storage.list_chunks(
            docs_by_stem[stem].doc_id
        )

    out: dict[str, tuple[str, int]] = {}
    for ct in spec.targets.chunks:
        if not ct.anchor:
            raise EvalError(
                f"chunk target {ct.id!r}: anchor is empty; cannot resolve"
            )
        body = bodies_by_stem[ct.doc]
        first = body.find(ct.anchor)
        if first < 0:
            raise EvalError(
                f"chunk target {ct.id!r}: anchor {ct.anchor!r} not found "
                f"in body of doc {ct.doc!r}"
            )
        if body.find(ct.anchor, first + 1) >= 0:
            raise EvalError(
                f"chunk target {ct.id!r}: anchor {ct.anchor!r} occurs "
                f"more than once in body of doc {ct.doc!r}; ambiguous"
            )
        covering = next(
            (c for c in chunks_by_stem[ct.doc] if c.start <= first < c.end),
            None,
        )
        if covering is None:
            raise EvalError(
                f"chunk target {ct.id!r}: no chunk covers anchor "
                f"position {first} in doc {ct.doc!r}"
            )
        out[ct.id] = (ct.doc, covering.seq)
    return out


async def _build_multimodal_search(
    cfg: Any, storage: Storage
) -> MultimodalSearch | None:
    """Mirror ``api.query``'s multimodal wiring for the eval runner.

    Activates the asset retrieval leg only when the wiki has
    ``assets.multimodal`` configured AND the storage backend has an
    active multimodal embed version. Returns ``None`` (text-only) when
    either prerequisite is missing — including hermetic eval against
    ``FakeEmbeddings``, where chunk-promoted ``asset_refs`` already
    carry the asset signal through to the asset-view projection.
    """
    mm_cfg = cfg.assets.multimodal
    if mm_cfg is None:
        return None
    try:
        active = await storage.get_active_embed_version(modality="multimodal")
    except NotSupported:
        return None
    if active is None or active.version_id is None:
        return None
    embedder = build_multimodal_embedder(
        mm_cfg.provider, base_url=mm_cfg.base_url, batch=mm_cfg.batch
    )
    return MultimodalSearch(
        embedder=embedder,
        model=active.model,
        asset_version_id=active.version_id,
    )


def _dedup(items: Iterable[str]) -> list[str]:
    """Order-preserving dedup. ``dict.fromkeys`` is the canonical idiom."""
    return list(dict.fromkeys(items))


def _project_doc_view(hits: list[Hit]) -> list[str]:
    return _dedup(
        Path(h.path).stem if h.path else h.doc_id for h in hits
    )


def _project_chunk_view(
    hits: list[Hit], chunk_runtime_to_named: dict[tuple[str, int], str]
) -> list[str]:
    """Project Hits to chunk-named-ids; non-target chunks are skipped
    (treated as "miss" by chunk-target queries)."""
    named_ids = (
        chunk_runtime_to_named.get((Path(h.path).stem, h.seq))
        for h in hits
        if h.path is not None and h.seq is not None
    )
    return _dedup(n for n in named_ids if n is not None)


def _project_asset_view(
    hits: list[Hit], asset_runtime_to_named: dict[str, str]
) -> list[str]:
    """Project Hits to asset-named-ids via ``Hit.asset_refs``."""
    named_ids = (
        asset_runtime_to_named.get(ar.asset_id)
        for h in hits
        for ar in h.asset_refs
    )
    return _dedup(n for n in named_ids if n is not None)


async def _run_queries(
    wiki: Path,
    spec: DatasetSpec,
    *,
    embedder: EmbeddingProvider,
    embedding_model: str,
    modes: list[RetrievalMode],
    reporter: ProgressReporter | None = None,
) -> dict[RetrievalMode, tuple[list[PerQueryRow], list[NegativeRow]]]:
    """Run every query in ``spec`` once per mode against a single storage
    connection. Returns a dict keyed by mode.
    """
    cfg, _root = api.load_wiki(wiki)
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        # Resolve named ids → runtime keys once per run; reused across modes.
        asset_named_to_runtime = _resolve_asset_targets(spec)
        chunk_named_to_runtime = await _resolve_chunk_targets(spec, storage)

        # Reverse maps for projection. Multiple named chunks sharing a
        # runtime ``(stem, seq)`` key means the chunker fit multiple
        # anchors into one chunk (small sections + default max_tokens):
        # silent collapse would make chunk-level metrics quietly wrong,
        # so loud-fail and force the user to tune the chunker or split
        # the targets. Eval datasets are authored, not inferred.
        runtime_to_named: dict[tuple[str, int], list[str]] = defaultdict(list)
        for named, runtime in chunk_named_to_runtime.items():
            runtime_to_named[runtime].append(named)
        collisions = {
            rt: names for rt, names in runtime_to_named.items() if len(names) > 1
        }
        if collisions:
            details = "; ".join(
                f"{names!r} -> {rt!r}" for rt, names in collisions.items()
            )
            raise EvalError(
                f"eval: {len(collisions)} chunk-target collision(s) — "
                f"{details}. Lower chunker max_tokens (or split sections) "
                f"so each anchor lands in its own chunk."
            )
        chunk_runtime_to_named: dict[tuple[str, int], str] = {
            rt: names[0] for rt, names in runtime_to_named.items()
        }
        asset_runtime_to_named: dict[str, str] = {
            v: k for k, v in asset_named_to_runtime.items()
        }

        # Wire the multimodal leg the same way ``api.query`` does so eval
        # actually exercises ``vec_search_assets`` when the wiki has
        # ``assets.multimodal`` configured. Without this, asset metrics
        # would silently collapse to whatever the text-ranked chunks
        # happened to carry in ``Hit.asset_refs``.
        mm_search = await _build_multimodal_search(cfg, storage)
        searcher = HybridSearcher.from_config(
            storage,
            embedder,
            cfg.retrieval,
            embedding_model=embedding_model,
            multimodal=mm_search,
        )
        results: dict[
            RetrievalMode, tuple[list[PerQueryRow], list[NegativeRow]]
        ] = {}
        total_q = len(spec.queries)
        for m in modes:
            positives: list[PerQueryRow] = []
            negatives: list[NegativeRow] = []
            for q_idx, q in enumerate(spec.queries, start=1):
                if reporter is not None:
                    reporter.cancel_token().raise_if_cancelled()
                hits = await searcher.search(q.q, limit=SEARCH_LIMIT, mode=m)
                ranked_docs = _project_doc_view(hits)
                ranked_chunks = _project_chunk_view(
                    hits, chunk_runtime_to_named
                )
                ranked_assets = _project_asset_view(
                    hits, asset_runtime_to_named
                )
                if q.expect_none:
                    negatives.append(
                        NegativeRow(q=q.q, ranked=tuple(ranked_docs))
                    )
                else:
                    positives.append(
                        PerQueryRow(
                            q=q.q,
                            q_id=q.id,
                            expect_doc_any=tuple(q.doc_positives),
                            expect_chunk_any=tuple(q.expect_chunk_any),
                            expect_asset_any=tuple(q.expect_asset_any),
                            ranked_docs=tuple(ranked_docs),
                            ranked_chunks=tuple(ranked_chunks),
                            ranked_assets=tuple(ranked_assets),
                        )
                    )
                if reporter is not None:
                    await reporter.progress(
                        phase="query",
                        current=q_idx,
                        total=total_q,
                        detail={"mode": m, "q_id": q.id},
                    )
            results[m] = (positives, negatives)
        return results
    finally:
        await storage.close()
