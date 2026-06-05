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
import logging
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from .. import api
from ..config import (
    CONFIG_FILENAME,
    AssetsConfig,
    CategoryNode,
    MultimodalEmbedConfig,
    ProviderConfig,
    RetrievalConfig,
    SchemaConfig,
    dump_config_yaml,
)
from ..domains.data.backends import parse_any
from ..domains.data.hashing import hash_file
from ..domains.info.search import HybridSearcher, MultimodalSearch, RetrievalMode
from ..domains.knowledge.links import parse_links
from ..domains.knowledge.page import KnowledgePage, read_page
from ..progress import NoopReporter, ProgressReporter
from ..providers import (
    EmbeddingProvider,
    MultimodalEmbeddingProvider,
    build_multimodal_embedder,
)
from ..providers.base import LLMProvider
from ..schemas import ChunkRecord, DocumentRecord, Hit, Layer, LinkType
from ..storage import Storage, build_storage
from ..storage._schema import SCHEMA_VERSION
from ..storage.base import NotSupported
from .dataset import DatasetSpec
from .fake_embedder import FakeEmbeddings
from .judge import (
    CategoryOption,
    CategorySummary,
    ClaimEvidence,
    EntailmentSummary,
    JudgeSummary,
    claim_evidence_from_grounding,
    judge_category,
    judge_entailment,
    judge_synthesis,
)
from .metrics import (
    atomicity_score,
    category_distribution,
    compute_grounding_cosines,
    duplicate_ratio_max,
    expected_coverage,
    language_fidelity,
    mean_hit_at_k,
    mean_ndcg_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    page_density,
    reduce_grounding_ratio,
    source_chunk_coverage,
    wikilink_resolved_ratio,
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
    ``dikw.yml`` — the snapshot's base carries the old block. Retrieval
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
        """True iff every retrieval-family threshold met its floor.

        Synth thresholds (``synth/*``) are gated by ``run_synth_eval``,
        not here — drop them before checking. Any *remaining* missing
        observation is a real failure (e.g. ``chunk/hit_at_3`` declared
        but no chunk-positive queries opted in via ``expect_chunk_any``);
        previously this branch ``continue``-d which silently passed
        misconfigured retrieval gates.
        """
        for key, floor in self.thresholds.items():
            if key.startswith("synth/"):
                continue
            value = self.metrics.get(key)
            if value is None or value < floor:
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
      base's ``ProviderConfig`` and hands both in. The runner serialises
      that config into the temp base's ``dikw.yml`` so ``api.ingest`` picks
      up vendor-specific ``embedding_batch_size`` / ``embedding_dim``
      exactly as the source base has them. Without this, a Gitee-configured
      provider would ingest at ``batch_size=64`` and get HTTP 400.

    ``assets_config`` + ``multimodal_embedder`` activate the asset
    retrieval leg end-to-end:

    * Pass an ``AssetsConfig`` whose ``.multimodal`` is populated to
      have the temp base's ``dikw.yml`` carry the same block — without
      this, ``cfg.assets.multimodal`` is ``None`` inside the eval and
      ``vec_search_assets`` never runs even for image-targeted queries.
    * Pass a ``MultimodalEmbeddingProvider`` so ``api.ingest`` can
      build the asset vectors. The cache key includes a multimodal
      identity fingerprint so a hermetic snapshot can't be silently
      reused by a multimodal run.

    ``mode`` selects which retrieval leg(s) to score: ``"hybrid"``
    (default, BM25+vec via RRF), ``"bm25"``, ``"vector"``, or ``"all"``
    (run all three sequentially against the same ingested corpus).
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
        base = _materialise_base(
            target,
            spec,
            provider_cfg=effective_provider_cfg,
            retrieval_cfg=effective_retrieval_cfg,
            assets_cfg=effective_assets_cfg,
        )
        _copy_corpus(spec.corpus_dir, base / "sources")
        await api.ingest(
            base,
            embedder=effective_embedder,
            multimodal_embedder=multimodal_embedder,
        )
        return base

    mm_fingerprint = _multimodal_fingerprint(effective_assets_cfg.multimodal)

    if cache_mode == "off":
        # Original behaviour: throwaway temp dir, no cache touched.
        with tempfile.TemporaryDirectory(prefix="dikw-eval-") as tmp:
            await _reporter.progress(phase="ingest", current=0, total=0)
            base = await _build(Path(tmp))
            await _reporter.progress(phase="ingest", current=1, total=1)
            per_mode = await _run_queries(
                base,
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
        base = cache_dir / "base"

        per_mode = await _run_queries(
            base,
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

    return EvalReport(
        dataset_name=spec.name,
        metrics=metrics,
        thresholds=dict(spec.thresholds),
        per_query=[r.to_dict() for r in canonical_positives],
        negative_diagnostics=[r.to_dict() for r in canonical_negatives],
        modes=list(modes),
        views=views_emitted,
    )


def _materialise_base(
    tmp_root: Path,
    spec: DatasetSpec,
    *,
    provider_cfg: ProviderConfig,
    retrieval_cfg: RetrievalConfig,
    assets_cfg: AssetsConfig,
    schema_cfg: SchemaConfig | None = None,
) -> Path:
    """Scaffold a throwaway base + dikw.yml that matches ``spec``.

    ``provider_cfg`` / ``retrieval_cfg`` / ``assets_cfg`` are copied
    verbatim into the written ``dikw.yml``. Downstream ``api.ingest``
    reads the provider + assets blocks; ``_run_queries`` re-loads the
    whole file to build ``HybridSearcher``, which picks up the retrieval
    + multimodal blocks. This means eval reproducibly measures whatever
    fusion knobs the caller passed, including the asset-vector leg.

    ``schema_cfg`` is optional and only used by ``run_synth_eval`` so a
    synth-mode dataset can pin its own ``categories`` taxonomy into the
    throwaway base (separate from the user's production categories).
    """
    base = tmp_root / "base"
    api.init_base(base, description=f"eval/{spec.name}")

    from ..config import default_config  # local: keep module import light

    cfg = default_config(description=f"eval/{spec.name}")
    cfg.provider = provider_cfg
    cfg.retrieval = retrieval_cfg
    cfg.assets = assets_cfg
    if schema_cfg is not None:
        cfg.schema_ = schema_cfg
    (base / CONFIG_FILENAME).write_text(dump_config_yaml(cfg), encoding="utf-8")
    return base


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
            f"ingested base has {sorted(docs_by_stem)}"
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

    Activates the asset retrieval leg only when the knowledge base has
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
    base: Path,
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
    cfg, _root = api.load_base(base)
    storage = build_storage(
        cfg.storage, root=base, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
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
        # actually exercises ``vec_search_assets`` when the knowledge base has
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


# ===========================================================================
# K-layer (synth) eval — run_synth_eval + SynthEvalReport
# ===========================================================================


class SynthEvalError(EvalError):
    """``run_synth_eval`` couldn't produce a valid report.

    Distinct subclass of ``EvalError`` so the CLI / server can present a
    K-layer-specific diagnostic (e.g. ``declared categories: X, observed
    categories in raw responses: Y`` when zero pages came back)."""


def _threshold_direction(metric_name: str) -> Literal["min", "max"]:
    """Metric names ending in ``_max`` are reverse-direction (lower is
    better); everything else is min-direction (higher is better).

    Strips a leading ``<view>/`` prefix so ``synth/duplicate_ratio_max``
    classifies the same as ``duplicate_ratio_max``.
    """
    bare = metric_name.rpartition("/")[-1] or metric_name
    return "max" if bare.endswith("_max") else "min"


@dataclass(frozen=True)
class ThresholdResult:
    name: str
    observed: float | None
    threshold: float
    direction: Literal["min", "max"]
    passed: bool


def check_thresholds(
    metrics: Mapping[str, float], thresholds: Mapping[str, float]
) -> list[ThresholdResult]:
    """Direction-aware threshold comparison.

    Missing observation → ``passed=False`` with ``observed=None``.
    Previously we used ``math.nan``, which serialises as the JSON
    token ``NaN`` (invalid JSON; rejected by strict parsers and by
    Postgres JSONB) — ``None`` rides through ``json.dumps`` as
    ``null`` and renderers handle the missing case explicitly.
    """
    out: list[ThresholdResult] = []
    for name, thr in thresholds.items():
        direction = _threshold_direction(name)
        value = metrics.get(name)
        if value is None:
            out.append(ThresholdResult(name, None, thr, direction, False))
            continue
        passed = value <= thr if direction == "max" else value >= thr
        out.append(ThresholdResult(name, value, thr, direction, passed))
    return out


class SynthEvalReport(BaseModel):
    """Result of a single ``run_synth_eval`` invocation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_name: str
    mode: Literal["synth"] = "synth"
    n_sources: int
    n_pages: int
    metrics: dict[str, float] = Field(default_factory=dict)
    threshold_results: list[ThresholdResult] = Field(default_factory=list)
    pages_per_source: dict[str, int] = Field(default_factory=dict)
    informational: dict[str, float] = Field(default_factory=dict)
    judge_summary: JudgeSummary | None = None
    entailment_summary: EntailmentSummary | None = None
    category_summary: CategorySummary | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def gated(self) -> bool:
        """True when the dataset declared at least one synth threshold.

        An ungated report (``threshold_results == []``) is informational
        — ``passed`` is vacuously ``True`` because ``all([])`` is ``True``,
        not because anything was checked. Aggregate callers use
        ``gated`` to decide whether to fold this report into an
        overall pass/fail.
        """
        return bool(self.threshold_results)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.threshold_results)


@dataclass
class _SynthMetricsBundle:
    """Everything ``run_synth_eval`` needs to assemble a ``SynthEvalReport``
    and (optionally) drive the LLM judge. Built once by
    ``_collect_metrics_bundle``; the pages and source texts are reused
    for judging instead of being re-read from disk."""

    pages: list[KnowledgePage]
    source_text_by_path: dict[str, str]
    n_sources: int
    metrics: dict[str, float]
    informational: dict[str, float]
    pages_per_source: dict[str, int]
    # (claim, best-source-chunk) pairs for the optional fact-entailment judge.
    # Built deterministically here (no LLM); ``run_synth_eval`` runs the judge
    # over them only when ``judge`` AND the dataset's
    # ``judge.entailment_grounding_enabled`` are both on.
    entailment_pairs: list[ClaimEvidence]


async def run_synth_eval(
    spec: DatasetSpec,
    *,
    llm: LLMProvider,
    embedder: EmbeddingProvider | None = None,
    provider_config: ProviderConfig | None = None,
    retrieval_config: RetrievalConfig | None = None,
    judge: bool = False,
    judge_sample: int | None = None,
    reporter: ProgressReporter | None = None,
) -> SynthEvalReport:
    """End-to-end K-layer eval: ingest → synth → metrics + optional judge.

    Materialises a throwaway base under ``tempfile.TemporaryDirectory``,
    ingests the dataset's corpus, runs ``api.synthesize`` with the
    provided LLM, then computes the seven K-layer metrics off the
    resulting pages and source chunks. ``spec.synth.categories`` is
    written through to the throwaway base's ``dikw.yml`` so the
    dataset's whitelist (rather than the production base's) gates which
    page types ``parse_synthesis_response`` accepts.

    Empty page output raises ``SynthEvalError`` so a ``0 / 0`` metric
    can't silently mask a hard failure. Pass ``judge=True`` to layer
    the LLM-judge soft score on top.
    """
    if "synth" not in spec.modes:
        raise SynthEvalError(
            f"dataset {spec.name!r} does not declare 'synth' mode "
            f"(modes: {spec.modes}); add 'synth' to modes: in dataset.yaml"
        )
    if not spec.corpus_dir.is_dir():
        raise SynthEvalError(f"corpus directory not found: {spec.corpus_dir}")

    _reporter: ProgressReporter = reporter or NoopReporter()
    effective_embedder: EmbeddingProvider = embedder or FakeEmbeddings()
    effective_provider_cfg = provider_config or ProviderConfig(
        embedding_model="fake",
        embedding_dim=64,
        embedding_revision="",
        embedding_normalize=True,
        embedding_distance="cosine",
    )
    effective_retrieval_cfg = retrieval_config or RetrievalConfig()
    schema_cfg = SchemaConfig(
        categories=[
            CategoryNode(path=c.path, desc=c.desc) for c in spec.synth.categories
        ]
    )

    warnings: list[str] = []
    if spec.expected is None:
        warnings.append(
            "expected.yaml absent — expected_coverage metric skipped"
        )

    with tempfile.TemporaryDirectory(prefix="dikw-synth-eval-") as tmpdir:
        base = _materialise_base(
            Path(tmpdir),
            spec,
            provider_cfg=effective_provider_cfg,
            retrieval_cfg=effective_retrieval_cfg,
            assets_cfg=AssetsConfig(),
            schema_cfg=schema_cfg,
        )
        _copy_corpus(spec.corpus_dir, base / "sources")

        await _reporter.progress(
            phase="synth_eval/ingest", current=0, total=1
        )
        await api.ingest(base, embedder=effective_embedder, reporter=_reporter)
        await _reporter.progress(
            phase="synth_eval/ingest", current=1, total=1
        )

        await _reporter.progress(
            phase="synth_eval/synth", current=0, total=1
        )
        synth_report = await api.synthesize(
            base,
            force_all=True,
            llm=llm,
            embedder=effective_embedder,
            reporter=_reporter,
        )
        await _reporter.progress(
            phase="synth_eval/synth",
            current=1,
            total=1,
            detail={
                "pages_created": synth_report.created,
                "unresolved_wikilinks": synth_report.unresolved_wikilinks,
            },
        )

        bundle = await _collect_metrics_bundle(
            spec=spec,
            base=base,
            synth_report=synth_report,
            embedder=effective_embedder,
            embedding_model=effective_provider_cfg.embedding_model,
            fallback=schema_cfg.fallback,
        )

        judge_summary: JudgeSummary | None = None
        if judge:
            await _reporter.progress(
                phase="synth_eval/judge",
                current=0,
                total=len(bundle.pages),
            )
            judge_summary = await judge_synthesis(
                bundle.pages,
                sources=bundle.source_text_by_path,
                llm=llm,
                model=spec.judge.model or effective_provider_cfg.llm_model,
                sample=judge_sample,
                reporter=_reporter,
                seed=spec.name,
            )
            await _reporter.progress(
                phase="synth_eval/judge",
                current=judge_summary.n_judged + judge_summary.n_errors,
                total=len(bundle.pages),
            )

        # Optional fact-entailment judge — the LLM grounding leg the cosine
        # ``fact_grounding_ratio`` is blind to. Gated on ``judge`` AND the
        # dataset's opt-in flag so it's $0 unless explicitly enabled. Surfaced
        # as an informational (ungated) metric; the honest bootstrap CI rides
        # on ``entailment_summary``.
        entailment_summary: EntailmentSummary | None = None
        if judge and spec.judge.entailment_grounding_enabled:
            await _reporter.progress(
                phase="synth_eval/entailment",
                current=0,
                total=len(bundle.entailment_pairs),
            )
            entailment_summary = await judge_entailment(
                bundle.entailment_pairs,
                llm=llm,
                model=spec.judge.model or effective_provider_cfg.llm_model,
                sample=judge_sample,
                reporter=_reporter,
                seed=spec.name,
            )
            if entailment_summary.n_judged > 0:
                bundle.informational["synth/fact_entailment_ratio"] = (
                    entailment_summary.ratio
                )
            await _reporter.progress(
                phase="synth_eval/entailment",
                current=entailment_summary.n_judged + entailment_summary.n_errors,
                total=len(bundle.entailment_pairs),
            )

        # Optional category-correctness judge — the embedding metrics see *where*
        # pages were filed (``category_distribution`` / ``fallback_ratio_max``)
        # but never whether the filing is *right*. The judge re-picks the best
        # category from the closed declared set + fallback and compares it to
        # synth's choice. Same opt-in economics as the entailment leg ($0 unless
        # ``judge`` AND the dataset flag are both on); informational, CI on
        # ``category_summary``.
        category_summary: CategorySummary | None = None
        if judge and spec.judge.category_correctness_enabled:
            # Build the closed set from ``schema_cfg`` — the SAME normalized
            # categories synth was materialised with — not raw
            # ``spec.synth.categories``. ``CategoryNode`` NFC-normalizes + strips
            # each path, so persisted ``page.category`` is normalized; sourcing
            # the judge's options from the raw spec would make the judge copy an
            # un-normalized path and score a correctly-filed page as wrong.
            category_options = [
                CategoryOption(path=c.path, desc=c.desc)
                for c in schema_cfg.categories
            ]
            category_options.append(
                CategoryOption(
                    path=schema_cfg.fallback,
                    desc="None of the categories above fit this page.",
                )
            )
            await _reporter.progress(
                phase="synth_eval/category", current=0, total=len(bundle.pages)
            )
            category_summary = await judge_category(
                bundle.pages,
                options=category_options,
                llm=llm,
                model=spec.judge.model or effective_provider_cfg.llm_model,
                sample=judge_sample,
                reporter=_reporter,
                seed=spec.name,
            )
            if category_summary.n_judged > 0:
                bundle.informational["synth/category_correctness_ratio"] = (
                    category_summary.ratio
                )
            await _reporter.progress(
                phase="synth_eval/category",
                current=category_summary.n_judged + category_summary.n_errors,
                total=len(bundle.pages),
            )

        synth_thresholds = {
            name: thr
            for name, thr in spec.thresholds.items()
            if name.startswith("synth/")
        }
        return SynthEvalReport(
            dataset_name=spec.name,
            n_sources=bundle.n_sources,
            n_pages=len(bundle.pages),
            metrics=bundle.metrics,
            threshold_results=check_thresholds(bundle.metrics, synth_thresholds),
            pages_per_source=bundle.pages_per_source,
            informational=bundle.informational,
            judge_summary=judge_summary,
            entailment_summary=entailment_summary,
            category_summary=category_summary,
            warnings=warnings,
        )


async def _collect_metrics_bundle(
    *,
    spec: DatasetSpec,
    base: Path,
    synth_report: api.SynthReport,
    embedder: EmbeddingProvider,
    embedding_model: str,
    fallback: str,
) -> _SynthMetricsBundle:
    """Pull pages + source chunks + source bodies out of the throwaway
    wiki and compute every K-layer metric.

    All keying uses the storage ``doc.path`` (``sources/<rel>``), which
    matches ``KnowledgePage.sources[0]`` populated by ``synthesize`` from
    ``src.path``. Keeping the two sides on the same key (rather than
    stripping a ``sources/`` prefix on one side) is what makes
    ``fact_grounding_ratio`` and ``language_fidelity`` actually look up
    their chunks and texts.
    """
    _cfg, _base_root, storage = await api._with_storage(base)
    try:
        knowledge_docs = list(
            await storage.list_documents(layer=Layer.KNOWLEDGE, active=True)
        )
        source_docs = list(
            await storage.list_documents(layer=Layer.SOURCE, active=True)
        )

        pages: list[KnowledgePage] = []
        for doc in knowledge_docs:
            try:
                pages.append(read_page(base, doc.path))
            except FileNotFoundError:
                continue

        if not pages:
            raise SynthEvalError(
                f"synth produced zero pages from {len(source_docs)} "
                f"sources (declared categories: "
                f"{[c.path for c in spec.synth.categories]})"
                f"; check LLM responses and dataset.yaml synth.categories"
            )

        chunks_by_source: dict[str, list[ChunkRecord]] = {}
        source_text_by_path: dict[str, str] = {}
        n_chunks = 0
        for doc in source_docs:
            chunks = await storage.list_chunks(doc.doc_id)
            chunks_by_source[doc.path] = chunks
            n_chunks += len(chunks)
            source_file = base / doc.path
            if source_file.is_file():
                source_text_by_path[doc.path] = source_file.read_text(
                    encoding="utf-8"
                )
    finally:
        await storage.close()

    pages_with_source_keys: list[tuple[KnowledgePage, str]] = []
    pages_with_source_texts: list[tuple[KnowledgePage, str]] = []
    pages_per_source: dict[str, int] = {}
    for page in pages:
        if not page.sources:
            continue
        key = page.sources[0]
        pages_with_source_keys.append((page, key))
        pages_per_source[key] = pages_per_source.get(key, 0) + 1
        text = source_text_by_path.get(key)
        if text is not None:
            pages_with_source_texts.append((page, text))

    total_wikilinks = sum(
        1
        for page in pages
        for link in parse_links(page.body)
        if link.kind is LinkType.WIKILINK
    )

    # Embed claims↔chunks once: ``fact_grounding_ratio`` and the
    # ``source_chunk_coverage`` diagnostic both reduce from the same
    # per-claim cosines (with the matched chunk seq), so a 100-page base
    # pays one grounding-embed pass, not two.
    grounding_claims = await compute_grounding_cosines(
        pages_with_sources=pages_with_source_keys,
        chunks_by_source=chunks_by_source,
        embedder=embedder,
        embedding_model=embedding_model,
    )
    metrics: dict[str, float] = {
        "synth/atomicity_score": atomicity_score(pages),
        "synth/wikilink_resolved_ratio": wikilink_resolved_ratio(
            total=total_wikilinks,
            unresolved=synth_report.unresolved_wikilinks,
        ),
        "synth/language_fidelity": language_fidelity(pages_with_source_texts),
        "synth/fact_grounding_ratio": reduce_grounding_ratio(
            grounding_claims,
            pages_with_sources=pages_with_source_keys,
            tau=spec.synth.grounding_threshold,
        ),
        "synth/duplicate_ratio_max": await duplicate_ratio_max(
            pages=pages,
            embedder=embedder,
            embedding_model=embedding_model,
            tau=spec.synth.duplicate_threshold,
        ),
    }
    if spec.expected is not None:
        expected_titles: list[str] = []
        for entry in spec.expected.sources:
            expected_titles.extend(entry.expected_titles)
        metrics["synth/expected_coverage"] = expected_coverage(
            page_titles=[p.title for p in pages],
            expected_titles=expected_titles,
        )

    # ``fallback_ratio_max`` = share of pages the LLM could not place under any
    # declared category (filed under the configured fallback). High values
    # flag taxonomy miscalibration the five gated metrics never surface.
    dist = category_distribution(pages)
    # Normalise the raw run-total ``slug_merge_count`` to a [0, 1)
    # over-generation *fraction* (merges / pre-dedup pages). The raw count
    # would (a) be mis-scaled against the A/B harness's ratio-calibrated noise
    # floor and (b) lack a direction. ``len(pages)`` is post-dedup; pre-dedup
    # ≈ that plus the merges (exact under eval, where no page fails persist).
    merges = synth_report.slug_merge_count
    pre_dedup_pages = len(pages) + merges
    slug_merge_ratio = merges / pre_dedup_pages if pre_dedup_pages else 0.0
    # Both ``*_ratio_max`` metrics are lower-is-better: a higher fallback share
    # or merge fraction is worse. The ``_max`` suffix is the project's
    # direction convention (``_threshold_direction`` / the A/B harness's
    # ``metric_direction``) — without it both would be scored backwards as
    # higher-is-better. ``source_chunk_coverage`` is genuinely higher-is-better
    # (more covered chunks), so it stays min-direction (no suffix).
    informational: dict[str, float] = {
        "synth/page_density": page_density(
            n_pages=len(pages), n_chunks=n_chunks
        ),
        "synth/source_chunk_coverage": source_chunk_coverage(
            grounding_claims,
            chunks_by_source=chunks_by_source,
            tau=spec.synth.grounding_threshold,
        ),
        "synth/fallback_ratio_max": dist.get(fallback, 0.0),
        "synth/slug_merge_ratio_max": slug_merge_ratio,
    }

    # Pair each claim with the source chunk that best matched it (reusing the
    # grounding argmax — no re-embedding) for the optional entailment judge.
    entailment_pairs = claim_evidence_from_grounding(
        grounding_claims, chunks_by_source
    )

    return _SynthMetricsBundle(
        pages=pages,
        source_text_by_path=source_text_by_path,
        n_sources=len(source_docs),
        metrics=metrics,
        informational=informational,
        pages_per_source=pages_per_source,
        entailment_pairs=entailment_pairs,
    )
