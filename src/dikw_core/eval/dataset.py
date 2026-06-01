"""Dataset loading — convention-over-configuration.

A dataset is a directory with up to **five** entries:

  <dataset>/
    dataset.yaml     — name, description, thresholds (optionally namespaced
                       as ``<view>/<metric>``), modes, synth + judge config.
    corpus/*.md      — documents to ingest; image files referenced via
                       ``![](path)`` live in ``corpus/`` too (any
                       sub-directory layout is fine).
    targets.yaml     — *(optional)* assets + chunks named-id catalog. When
                       present, queries can express chunk-level / asset-level
                       ground truth via ``expect_chunk_any`` /
                       ``expect_asset_any``. Absent → only doc-level eval.
    queries.yaml     — list of {q, expect_*_any | expect_none} entries.
    expected.yaml    — *(optional, synth mode only)* per-source expected
                       page titles + informational keywords for
                       ``expected_coverage`` / report enrichment. Absent →
                       ``expected_coverage`` metric not computed.

``load_dataset(name_or_path)`` accepts either a registered name (looked up
under ``datasets_root()``) or a directory path (user-provided). No registry
file, no plugin system: the filesystem layout *is* the contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Retrieval metrics — gated by ``run_eval``. ``ndcg_at_10`` / ``recall_at_100``
# are added for public-benchmark calibration (BEIR / CMTEB report these as
# defaults); the dogfood mvp dataset does not have to set them.
_RETRIEVAL_METRICS = frozenset(
    {"hit_at_3", "hit_at_10", "mrr", "ndcg_at_10", "recall_at_100"}
)
# K-layer (synth) metrics — gated by ``run_synth_eval``. Threshold keys
# MUST be namespaced (``synth/atomicity_score``); the bare form is rejected
# at load time so a typo can't bypass the gate. ``duplicate_ratio_max``
# carries the ``_max`` suffix to signal lower-is-better (reverse-direction).
# ``page_density`` is informational only — runner emits it into
# ``SynthEvalReport.informational`` rather than ``metrics``, so it would
# never satisfy a threshold even if one were declared. Kept in the
# "known synth metric" set for the informational column, but
# ``_GATEABLE_SYNTH_METRICS`` is what threshold validation accepts.
_SYNTH_METRICS = frozenset(
    {
        "fact_grounding_ratio",
        "atomicity_score",
        "duplicate_ratio_max",
        "wikilink_resolved_ratio",
        "expected_coverage",
        "language_fidelity",
        "page_density",
    }
)
_GATEABLE_SYNTH_METRICS = _SYNTH_METRICS - {"page_density"}
SUPPORTED_METRICS = _RETRIEVAL_METRICS | _SYNTH_METRICS

# Granularity / scoring views the runner can score. ``doc`` is always
# available; ``chunk`` and ``asset`` require ``targets.yaml`` + queries
# opting in via ``expect_chunk_any`` / ``expect_asset_any``. ``synth`` is
# the K-layer view — orthogonal to retrieval granularity.
SUPPORTED_VIEWS = frozenset({"doc", "chunk", "asset", "synth"})

# Which eval mode(s) a dataset advertises. ``run_synth_eval`` only fires
# when ``synth`` is declared; ``run_eval`` only when ``retrieval`` is
# declared. Datasets that omit the ``modes:`` field default to
# ``[retrieval]`` for back-compat with the pre-synth contract.
EvalMode = Literal["retrieval", "synth"]
SUPPORTED_MODES = frozenset({"retrieval", "synth"})

QueryType = Literal["doc", "text_chunk", "asset", "mixed"]


class DatasetError(RuntimeError):
    """Raised when a dataset directory is malformed or missing pieces."""


class AssetTarget(BaseModel):
    """One row in ``targets.yaml`` ``assets:`` — a named image/asset.

    ``id`` is the stable identity used in ``queries.yaml`` ``expect_asset_any``.
    ``path`` is relative to ``corpus/`` and points at the image file the loader
    will hash to derive the runtime ``asset_id`` (sha256 over file bytes).
    ``heading`` / ``anchor`` are advisory metadata for human readability and
    report rendering — they don't participate in matching.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    doc: str
    path: str
    heading: str = ""
    anchor: str = ""


class ChunkTarget(BaseModel):
    """One row in ``targets.yaml`` ``chunks:`` — a named text chunk.

    ``anchor`` is resolved against the markdown body to find the heading's
    char position; the runner then matches that position to the chunk
    produced by the chunker covering it. ``heading`` is advisory.

    ``asset_id`` (optional) declares the chunk ↔ asset binding the dataset
    author intended (one image per H2). The eval runner doesn't enforce
    this — ``Hit.asset_refs`` provides the runtime-observed binding — but
    it's preserved for diagnostic output.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    doc: str
    heading: str = ""
    anchor: str = ""
    asset_id: str | None = None


class TargetsSpec(BaseModel):
    """Parsed ``targets.yaml``.

    Two flat lists, both keyed by user-authored stable ``id``. Either may
    be empty; an empty file (or missing file) is equivalent to
    ``TargetsSpec(assets=[], chunks=[])`` — eval falls back to doc-level
    only.
    """

    model_config = ConfigDict(frozen=True)

    assets: list[AssetTarget] = Field(default_factory=list)
    chunks: list[ChunkTarget] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_dup_ids(self) -> Self:
        seen_a: set[str] = set()
        for a in self.assets:
            if a.id in seen_a:
                raise ValueError(f"duplicate asset target id: {a.id!r}")
            seen_a.add(a.id)
        seen_c: set[str] = set()
        for c in self.chunks:
            if c.id in seen_c:
                raise ValueError(f"duplicate chunk target id: {c.id!r}")
            seen_c.add(c.id)
        return self


class Query(BaseModel):
    """One entry in ``queries.yaml``.

    Polarity is exactly one of:

    * Positive — at least one of ``expect_any`` / ``expect_doc_any`` /
      ``expect_chunk_any`` / ``expect_asset_any`` is non-empty. The query
      is a hit at k (in the corresponding view) if any listed identity
      appears in top-k of that view's ranked list.
    * Negative — ``expect_none: true`` and all expect_* lists empty. The
      observed top-k surfaces as diagnostic only and does NOT contribute
      to hit@k/MRR.

    ``expect_any`` is the legacy doc-level field (kept for SciFact / CMTEB
    back-compat). It's an alias of ``expect_doc_any``; a query that sets
    ``expect_any`` cannot also set ``expect_doc_any`` (ambiguous).

    ``id`` and ``query_type`` are optional metadata: ``id`` lets the
    runner key per-query rows to a stable identity (otherwise an array
    index is used); ``query_type`` is informational.
    """

    model_config = ConfigDict(frozen=True)

    q: str
    id: str | None = None
    query_type: QueryType | None = None
    expect_any: list[str] = Field(default_factory=list)
    expect_doc_any: list[str] = Field(default_factory=list)
    expect_chunk_any: list[str] = Field(default_factory=list)
    expect_asset_any: list[str] = Field(default_factory=list)
    expect_none: bool = False

    @model_validator(mode="after")
    def _polarity(self) -> Self:
        # ``expect_any`` is the legacy alias of ``expect_doc_any``; both
        # set at once is ambiguous.
        if self.expect_any and self.expect_doc_any:
            raise ValueError(
                "cannot set both expect_any and expect_doc_any (expect_any "
                "is the legacy alias)"
            )
        positives = (
            list(self.expect_any)
            + list(self.expect_doc_any)
            + list(self.expect_chunk_any)
            + list(self.expect_asset_any)
        )
        if self.expect_none and positives:
            raise ValueError(
                "expect_none and any expect_*_any are mutually exclusive"
            )
        if not self.expect_none and not positives:
            raise ValueError(
                "query must provide at least one of expect_any / expect_doc_any /"
                " expect_chunk_any / expect_asset_any (positive), or "
                "expect_none=true (negative)"
            )
        return self

    @property
    def doc_positives(self) -> list[str]:
        """Doc-level positives: union of legacy ``expect_any`` and
        ``expect_doc_any``."""
        return list(self.expect_any) + list(self.expect_doc_any)


class SynthCategory(BaseModel):
    """One category in a synth-eval dataset's taxonomy: a folder ``path`` plus
    optional ``desc`` guidance fed to the synth LLM (mirrors
    ``config.CategoryNode``). A bare string in ``dataset.yaml`` is coerced to
    ``{path: <string>}`` for back-compat — see ``SynthSection``."""

    model_config = ConfigDict(frozen=True)

    path: str
    desc: str = ""


def _default_synth_categories() -> list[SynthCategory]:
    """Mirror the production-default taxonomy (``config._default_categories``)
    so a dataset that omits ``categories:`` still feeds the synth LLM the same
    per-category guidance a default-config base gets — keeping the K-layer eval
    gate representative of default synth rather than feeding a desc-stripped
    prompt."""
    from ..config import _default_categories

    return [SynthCategory(path=c.path, desc=c.desc) for c in _default_categories()]


class SynthSection(BaseModel):
    """K-layer eval knobs declared in ``dataset.yaml``'s ``synth:`` block.

    Defaults reflect the 2026-05-13 tau sweep against
    Qwen3-Embedding-0.6B (BASELINES.md): ``grounding_threshold=0.50``
    (sits between the cluster of real-grounded claims at 0.55+ and the
    cluster of paraphrased / fragment claims below 0.40),
    ``duplicate_threshold=0.85``. Retrieval-only datasets that don't
    declare synth still parse and ``run_synth_eval`` (if invoked) gets
    sensible numbers. ``categories`` is dataset-local and written into the
    throwaway base's ``dikw.yml`` at eval time, so a synth-eval dataset can pin
    a tighter taxonomy than the user's production base. Each entry is a
    ``{path, desc}`` mapping (or a bare ``path`` string, coerced for
    back-compat); the optional ``desc`` mirrors ``config.CategoryNode.desc`` so
    the eval prompt's ``{categories}`` block carries the same guidance a
    production base would.
    """

    model_config = ConfigDict(frozen=True)

    grounding_threshold: float = 0.50
    duplicate_threshold: float = 0.85
    categories: list[SynthCategory] = Field(default_factory=_default_synth_categories)

    @model_validator(mode="before")
    @classmethod
    def _coerce_bare_string_categories(cls, data: Any) -> Any:
        # Back-compat: ``categories: [entity, concept]`` (bare strings) is the
        # historic shape. Promote each bare string to ``{path: <string>}`` so
        # both the legacy list and the richer ``{path, desc}`` mapping parse.
        if isinstance(data, dict) and isinstance(data.get("categories"), list):
            data = {
                **data,
                "categories": [
                    {"path": c} if isinstance(c, str) else c
                    for c in data["categories"]
                ],
            }
        return data

    @model_validator(mode="after")
    def _validate_categories(self) -> Self:
        if not self.categories:
            raise ValueError("synth.categories must not be empty")
        for c in self.categories:
            if not c.path.strip():
                raise ValueError(
                    f"synth.categories entries must have a non-empty path, got {c!r}"
                )
        return self


class JudgeSection(BaseModel):
    """LLM-judge override declared in ``dataset.yaml``'s ``judge:`` block.

    ``model`` lets a dataset pin a specific model id (handy when the
    base's configured LLM is a small / fast model unsuitable for
    judging); ``None`` falls back to ``cfg.provider.llm_model``. Judge
    runs use the knowledge base's configured provider — a separate provider per
    dataset would mean a separate auth / billing path and isn't needed
    yet. ``extra="forbid"`` ensures a ``provider:`` typo isn't silently
    ignored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str | None = None


class ExpectedSource(BaseModel):
    """One source's expected synth output, used by ``expected_coverage``.

    ``path`` is corpus-relative (matches the file under ``<dataset>/corpus/``).
    ``expected_titles`` feed the coverage metric via fuzzy match (same
    normalize rules as wikilink resolution). ``expected_keywords`` are
    informational only — they enrich reports but don't gate metrics.
    """

    model_config = ConfigDict(frozen=True)

    path: str
    expected_titles: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)


class ExpectedSpec(BaseModel):
    """Parsed ``expected.yaml`` — list of per-source expectations."""

    model_config = ConfigDict(frozen=True)

    sources: list[ExpectedSource] = Field(default_factory=list)


class DatasetSpec(BaseModel):
    """Validated view of a dataset directory on disk."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    thresholds: dict[str, float]
    corpus_dir: Path
    queries: list[Query]
    targets: TargetsSpec = Field(default_factory=TargetsSpec)
    modes: list[EvalMode] = Field(
        default_factory=lambda: ["retrieval"]  # type: ignore[arg-type]
    )
    synth: SynthSection = Field(default_factory=SynthSection)
    judge: JudgeSection = Field(default_factory=JudgeSection)
    expected: ExpectedSpec | None = None


def datasets_root() -> Path:
    """Where packaged datasets live.

    Two resolutions, tried in order:

    1. **Installed wheel** — ``<package>/dikw_core/eval/datasets/``, populated
       by a hatch ``force-include`` mapping at build time.
    2. **Editable / source checkout** — walk up from this file to find
       ``<repo>/evals/datasets/``.

    Returning the first hit means test suites running from a source checkout
    see the committed data, while installed users see the bundled copy.
    """
    here = Path(__file__).resolve().parent  # src/dikw_core/eval/
    installed = here / "datasets"
    if installed.is_dir():
        return installed
    for parent in here.parents:
        candidate = parent / "evals" / "datasets"
        if candidate.is_dir():
            return candidate
    # No evals/ anywhere — still return the installed-path target so error
    # messages in load_dataset remain legible ("no such dataset under
    # /path/to/site-packages/dikw_core/eval/datasets").
    return installed


def iter_packaged_datasets() -> list[str]:
    """List every packaged dataset (registered name only).

    Walks ``datasets_root()`` and returns dataset names — i.e. the
    immediate-subdir names that ``load_dataset`` accepts. Used by the
    ``dikw client eval`` no-arg path that runs every packaged dataset
    back-to-back. Skips dot-files and any subdir without a
    ``dataset.yaml`` so the discovery never trips on README dirs etc.
    """
    root = datasets_root()
    if not root.is_dir():
        return []
    out = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if (entry / "dataset.yaml").is_file():
            out.append(entry.name)
    return out


def load_dataset(name_or_path: str | Path) -> DatasetSpec:
    """Load and validate a dataset by registered name or filesystem path."""
    path = _resolve_path(name_or_path)
    if not path.is_dir():
        raise DatasetError(f"dataset directory not found: {path}")

    meta = _load_dataset_yaml(path)
    queries = _load_queries(path)
    corpus_dir = _require_corpus(path)
    targets = _load_targets(path)
    _validate_query_targets(queries, targets)

    thresholds = _parse_thresholds(meta.get("thresholds") or {})
    modes = _parse_modes(meta.get("modes"))
    synth = _parse_synth_section(meta.get("synth"))
    judge = _parse_judge_section(meta.get("judge"))
    expected = _load_expected(path)
    if expected is not None:
        _validate_expected_against_corpus(expected, corpus_dir)

    try:
        return DatasetSpec(
            name=str(meta.get("name") or path.name),
            description=str(meta.get("description") or ""),
            thresholds=thresholds,
            corpus_dir=corpus_dir,
            queries=queries,
            targets=targets,
            modes=modes,
            synth=synth,
            judge=judge,
            expected=expected,
        )
    except ValidationError as e:  # pragma: no cover — caught by earlier guards
        raise DatasetError(f"invalid dataset spec at {path}: {e}") from e


def _parse_modes(raw: Any) -> list[EvalMode]:
    """Parse the optional ``modes:`` list. Missing/None → ``["retrieval"]``."""
    if raw is None:
        return ["retrieval"]
    if not isinstance(raw, list) or not raw:
        raise DatasetError(
            f"modes must be a non-empty list, got {raw!r}; "
            f"supported: {sorted(SUPPORTED_MODES)}"
        )
    out: list[EvalMode] = []
    for entry in raw:
        if entry not in SUPPORTED_MODES:
            raise DatasetError(
                f"unknown mode {entry!r}; supported: {sorted(SUPPORTED_MODES)}"
            )
        out.append(entry)
    return out


def _parse_optional_section[T: BaseModel](
    raw: Any, *, name: str, model_cls: type[T]
) -> T:
    """Parse an optional ``dataset.yaml`` block; missing → model defaults.

    Surfaces pydantic's first violation message in the error (the rest are
    usually downstream noise from the same root cause)."""
    if raw is None:
        return model_cls()
    if not isinstance(raw, dict):
        raise DatasetError(
            f"{name}: must be a mapping, got {type(raw).__name__}"
        )
    try:
        return model_cls.model_validate(raw)
    except ValidationError as e:
        first_msg = e.errors()[0].get("msg", str(e))
        raise DatasetError(f"invalid {name} section: {first_msg}") from e


def _parse_synth_section(raw: Any) -> SynthSection:
    return _parse_optional_section(raw, name="synth", model_cls=SynthSection)


def _parse_judge_section(raw: Any) -> JudgeSection:
    return _parse_optional_section(raw, name="judge", model_cls=JudgeSection)


def _load_expected(path: Path) -> ExpectedSpec | None:
    """Load ``expected.yaml`` if present; absent → None (metric skipped)."""
    e_path = path / "expected.yaml"
    if not e_path.is_file():
        return None
    raw = yaml.safe_load(e_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise DatasetError(f"{e_path}: top-level YAML must be a mapping")
    try:
        return ExpectedSpec.model_validate(raw)
    except ValidationError as e:
        raise DatasetError(f"{e_path}: invalid expected spec: {e}") from e


def _validate_expected_against_corpus(
    expected: ExpectedSpec, corpus_dir: Path
) -> None:
    """Cross-check ``expected.yaml`` source paths against ``corpus/``.

    A typo here would silently drop ``expected_coverage`` for that source
    (or, worse, count it as zero coverage). Surface at load time, same
    pattern as ``_validate_query_targets``.
    """
    for entry in expected.sources:
        candidate = corpus_dir / entry.path
        if not candidate.is_file():
            raise DatasetError(
                f"expected.yaml references unknown source {entry.path!r}; "
                f"no such file under {corpus_dir}"
            )


def _validate_query_targets(
    queries: list[Query], targets: TargetsSpec
) -> None:
    """Cross-check query positives against ``targets.yaml``.

    A typo in ``expect_chunk_any`` / ``expect_asset_any`` would silently
    drop a query into "permanent miss" — same authoring failure as a
    bad anchor, which we already loud-fail on. Surface it at load time
    so the metric never quietly absorbs the bug.
    """
    chunk_ids = {c.id for c in targets.chunks}
    asset_ids = {a.id for a in targets.assets}
    for i, q in enumerate(queries, start=1):
        unknown_chunks = [x for x in q.expect_chunk_any if x not in chunk_ids]
        unknown_assets = [x for x in q.expect_asset_any if x not in asset_ids]
        if unknown_chunks:
            raise DatasetError(
                f"query #{i} ({q.id or q.q[:40]!r}): expect_chunk_any "
                f"references unknown chunk targets {unknown_chunks}; "
                f"available: {sorted(chunk_ids)}"
            )
        if unknown_assets:
            raise DatasetError(
                f"query #{i} ({q.id or q.q[:40]!r}): expect_asset_any "
                f"references unknown asset targets {unknown_assets}; "
                f"available: {sorted(asset_ids)}"
            )


def _resolve_path(name_or_path: str | Path) -> Path:
    """Accept a path (existing dir) or a name (resolved under datasets_root)."""
    if isinstance(name_or_path, Path):
        return name_or_path
    # Heuristic: anything that looks like a path (contains os.sep, starts
    # with `.`/`/`, or is an existing directory) is treated as a path.
    as_path = Path(name_or_path)
    if as_path.is_dir() or "/" in name_or_path or name_or_path.startswith("."):
        return as_path
    # Bare name → under packaged/dev datasets root
    candidate = datasets_root() / name_or_path
    if not candidate.is_dir():
        raise DatasetError(
            f"no such dataset {name_or_path!r} under {datasets_root()}"
        )
    return candidate


def _load_dataset_yaml(path: Path) -> Mapping[str, Any]:
    meta_path = path / "dataset.yaml"
    if not meta_path.is_file():
        raise DatasetError(f"missing dataset.yaml at {meta_path}")
    raw = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise DatasetError(f"{meta_path}: top-level YAML must be a mapping")
    return raw


def _require_corpus(path: Path) -> Path:
    corpus_dir = path / "corpus"
    if not corpus_dir.is_dir():
        raise DatasetError(f"missing corpus/ directory at {corpus_dir}")
    # Accept any SourceBackend-readable extension; a deeper check would
    # duplicate data/backends/__init__.py logic — for now require at least
    # one file to catch the "forgot to populate" case.
    if not any(corpus_dir.iterdir()):
        raise DatasetError(f"corpus/ at {corpus_dir} is empty")
    return corpus_dir


def _load_queries(path: Path) -> list[Query]:
    q_path = path / "queries.yaml"
    if not q_path.is_file():
        raise DatasetError(f"missing queries.yaml at {q_path}")
    raw = yaml.safe_load(q_path.read_text(encoding="utf-8")) or {}
    items = raw.get("queries") if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items:
        raise DatasetError(f"{q_path}: expected at least one query under 'queries:'")
    queries: list[Query] = []
    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            raise DatasetError(f"{q_path}: query #{i + 1} must be a mapping")
        try:
            queries.append(Query.model_validate(entry))
        except ValidationError as e:
            raise DatasetError(f"{q_path}: query #{i + 1} invalid: {e}") from e
    return queries


def _load_targets(path: Path) -> TargetsSpec:
    """Load ``targets.yaml`` if present; missing file → empty TargetsSpec."""
    t_path = path / "targets.yaml"
    if not t_path.is_file():
        return TargetsSpec()
    raw = yaml.safe_load(t_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise DatasetError(f"{t_path}: top-level YAML must be a mapping")
    try:
        return TargetsSpec.model_validate(raw)
    except ValidationError as e:
        raise DatasetError(f"{t_path}: invalid targets spec: {e}") from e


def _parse_thresholds(raw: Mapping[str, Any]) -> dict[str, float]:
    """Parse threshold dict; supports plain and ``<view>/<metric>`` keys.

    Plain retrieval keys (``hit_at_3``) gate on the canonical doc view
    (back-compat). Namespaced keys (``chunk/hit_at_3``, ``asset/hit_at_10``)
    gate on the corresponding granularity view; the runner mirrors the
    canonical doc view's plain key for both ``hit_at_3`` and
    ``doc/hit_at_3`` so existing dataset.yaml files keep gating without
    modification.

    K-layer metrics MUST use the ``synth/<metric>`` namespace; bare
    synth metric names and retrieval metrics under ``synth/`` are both
    rejected here so the load is the single source of "which family
    this gate belongs to". Otherwise ``run_synth_eval`` would silently
    drop bare synth thresholds (its filter only keeps ``synth/*``) and
    ``check_thresholds`` would record a never-computable miss for keys
    like ``synth/hit_at_3``.
    """
    out: dict[str, float] = {}
    for raw_key, value in raw.items():
        key = str(raw_key)
        if "/" in key:
            view, _, metric = key.partition("/")
            if view not in SUPPORTED_VIEWS:
                raise DatasetError(
                    f"unknown threshold view {view!r} in key {key!r}; "
                    f"supported: {sorted(SUPPORTED_VIEWS)}"
                )
            if view == "synth" and metric not in _GATEABLE_SYNTH_METRICS:
                if metric in _SYNTH_METRICS:
                    raise DatasetError(
                        f"threshold {key!r}: {metric!r} is informational "
                        f"only and cannot be gated (runner emits it into "
                        f"``informational``, not ``metrics``)"
                    )
                raise DatasetError(
                    f"threshold {key!r}: {metric!r} is not a synth metric; "
                    f"available synth metrics: "
                    f"{sorted(_GATEABLE_SYNTH_METRICS)}"
                )
            if view != "synth" and metric not in _RETRIEVAL_METRICS:
                raise DatasetError(
                    f"threshold {key!r}: {metric!r} is not a retrieval "
                    f"metric (the {view!r} view only gates retrieval); "
                    f"K-layer thresholds must use the 'synth/' prefix"
                )
        elif key in _SYNTH_METRICS:
            raise DatasetError(
                f"threshold {key!r}: K-layer metrics must use the "
                f"'synth/' prefix (got bare {key!r}, expected 'synth/{key}')"
            )
        elif key not in _RETRIEVAL_METRICS:
            raise DatasetError(
                f"unknown threshold key {key!r}; supported: "
                f"{sorted(SUPPORTED_METRICS)} or "
                f"<view>/<metric> with view in {sorted(SUPPORTED_VIEWS)}"
            )
        try:
            out[key] = float(value)
        except (TypeError, ValueError) as e:
            raise DatasetError(f"threshold {key!r} is not a number: {value!r}") from e
    return out
