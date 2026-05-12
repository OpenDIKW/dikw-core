"""Synth + distill task wiring.

These ops mirror ``ingest_op.make_ingest_runner``: build a
``TaskRunner`` closure that ``TaskManager.submit`` schedules. Each runner
owns its own provider clients (LLM, optional embedder) for the duration
of the task and returns a JSON-serialisable result dict that the manager
folds into the ``final`` event.

Synth + distill share the same provider-failure mode: a missing
``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` surfaces as a runner-level
``BadRequest``, so the task ends ``failed`` with a clear cause instead of
a partially-progressed terminal that hid the misconfiguration.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .. import api
from ..config import CONFIG_FILENAME, load_config
from ..progress import ProgressReporter
from ..providers import build_embedder, build_llm
from .errors import BadRequest

logger = logging.getLogger(__name__)


def make_synth_runner(
    *,
    wiki_root: Path,
    force_all: bool,
    no_embed: bool,
) -> Callable[[ProgressReporter], Awaitable[dict[str, Any]]]:
    """Build a ``TaskRunner`` that drives ``api.synthesize`` for one task."""

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        # Reload cfg from disk INSIDE the runner so providers stay
        # aligned with ``api.synthesize``'s own ``_with_storage`` reload.
        # See ``ingest_op.make_ingest_runner`` for the full reasoning.
        cfg = load_config(wiki_root / CONFIG_FILENAME)
        try:
            llm = build_llm(cfg.provider, wiki_base=wiki_root)
        except Exception as e:
            raise BadRequest(
                f"could not build LLM: {e}", code="llm_unavailable"
            ) from e

        embedder = None
        if not no_embed:
            try:
                embedder = build_embedder(cfg.provider)
            except Exception as e:
                # Synth without an embedder is a soft-degrade: pages
                # land on disk but new K-layer chunks miss the dense
                # leg. Surface this as a log so the operator sees it
                # but the task still succeeds.
                await reporter.log(
                    "WARN",
                    f"embedder unavailable, synth proceeds without dense indexing: {e}",
                )

        report = await api.synthesize(
            wiki_root,
            force_all=force_all,
            llm=llm,
            embedder=embedder,
            reporter=reporter,
        )
        return dataclasses.asdict(report)

    return _runner


def make_distill_runner(
    *,
    wiki_root: Path,
    pages_per_call: int,
) -> Callable[[ProgressReporter], Awaitable[dict[str, Any]]]:
    """Build a ``TaskRunner`` that drives ``api.distill`` for one task."""

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        cfg = load_config(wiki_root / CONFIG_FILENAME)
        try:
            llm = build_llm(cfg.provider, wiki_base=wiki_root)
        except Exception as e:
            raise BadRequest(
                f"could not build LLM: {e}", code="llm_unavailable"
            ) from e

        report = await api.distill(
            wiki_root,
            llm=llm,
            pages_per_call=pages_per_call,
            reporter=reporter,
        )
        return dataclasses.asdict(report)

    return _runner


def make_eval_runner(
    *,
    wiki_root: Path,
    dataset: str | None,
    mode: str,
    cache_mode: str,
    eval_modes: list[str] | None = None,
    judge: bool = False,
    judge_sample: int | None = None,
) -> Callable[[ProgressReporter], Awaitable[dict[str, Any]]]:
    """Build a ``TaskRunner`` that drives ``eval.run_eval`` (retrieval)
    and/or ``eval.run_synth_eval`` (K-layer) for one task.

    The runner builds an embedder + (optional) multimodal embedder from
    the server's wiki cfg so eval scores against the same vector space
    the live engine uses. ``dataset`` may be a registered name (resolved
    under the packaged datasets root), an explicit path, or ``None`` to
    run every packaged dataset back-to-back — preserving the
    ``dikw eval`` (no-arg) workflow that the in-process CLI shipped with.

    ``eval_modes`` (optional) restricts which families run per dataset.
    ``None`` falls back to each dataset's ``spec.modes`` declaration —
    so a retrieval-only dataset still runs retrieval, and a synth-
    capable dataset runs whatever it declared. Synth eval also drives
    the LLM judge layer when ``judge=True``.
    """

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        # Defer the eval imports — the eval module pulls in dataset
        # validators + corpus walkers we don't need on a server that
        # never runs eval.
        from ..eval.dataset import (
            DatasetError,
            iter_packaged_datasets,
            load_dataset,
        )
        from ..eval.runner import EvalError, run_eval, run_synth_eval
        from ..providers import build_llm, build_multimodal_embedder

        cfg = load_config(wiki_root / CONFIG_FILENAME)

        if dataset is None:
            # No-arg ``dikw eval`` ran every packaged dataset; preserve
            # that by enumerating them here. ``iter_packaged_datasets``
            # yields names suitable for ``load_dataset``.
            specs = []
            for name in iter_packaged_datasets():
                try:
                    specs.append(load_dataset(name))
                except DatasetError as e:
                    raise BadRequest(
                        f"could not load packaged dataset {name!r}: {e}",
                        code="dataset_not_found",
                    ) from e
        else:
            try:
                specs = [load_dataset(dataset)]
            except DatasetError as e:
                raise BadRequest(
                    f"could not load dataset {dataset!r}: {e}",
                    code="dataset_not_found",
                ) from e

        try:
            embedder = build_embedder(cfg.provider)
        except Exception as e:
            raise BadRequest(
                f"could not build embedder: {e}",
                code="embedder_unavailable",
            ) from e

        multimodal_embedder = None
        mm_cfg = cfg.assets.multimodal
        if mm_cfg is not None:
            try:
                multimodal_embedder = build_multimodal_embedder(
                    mm_cfg.provider,
                    base_url=mm_cfg.base_url,
                    batch=mm_cfg.batch,
                )
            except Exception as e:
                await reporter.log(
                    "WARN",
                    f"multimodal embedder unavailable, "
                    f"eval proceeds with text-only: {e}",
                )

        # LLM is only needed when synth mode is in play — lazy-built per
        # request, cached for the run so we don't pay setup twice on the
        # multi-dataset path.
        _llm = None

        def _get_llm() -> Any:
            nonlocal _llm
            if _llm is None:
                try:
                    _llm = build_llm(cfg.provider, wiki_base=wiki_root)
                except Exception as e:
                    raise BadRequest(
                        f"could not build LLM for synth eval: {e}",
                        code="llm_unavailable",
                    ) from e
            return _llm

        reports: list[dict[str, Any]] = []
        all_passed = True
        for spec in specs:
            modes_to_run = _resolve_eval_modes(spec, eval_modes)
            for em in modes_to_run:
                if em == "retrieval":
                    try:
                        report = await run_eval(
                            spec,
                            embedder=embedder,
                            provider_config=cfg.provider,
                            retrieval_config=cfg.retrieval,
                            assets_config=cfg.assets,
                            multimodal_embedder=multimodal_embedder,
                            mode=mode,  # type: ignore[arg-type]
                            cache_mode=cache_mode,  # type: ignore[arg-type]
                            reporter=reporter,
                        )
                    except EvalError as e:
                        raise BadRequest(
                            f"retrieval eval failed on dataset "
                            f"{spec.name!r}: {e}",
                            code="eval_error",
                        ) from e
                    dumped = report.model_dump(mode="json")
                    dumped["passed"] = report.passed
                    # ``mode: retrieval`` marks which family produced this
                    # row so the client renderer can dispatch.
                    dumped["mode"] = "retrieval"
                    all_passed = all_passed and report.passed
                    reports.append(dumped)
                else:  # em == "synth"
                    try:
                        synth_rep = await run_synth_eval(
                            spec,
                            llm=_get_llm(),
                            embedder=embedder,
                            provider_config=cfg.provider,
                            retrieval_config=cfg.retrieval,
                            judge=judge,
                            judge_sample=judge_sample,
                            reporter=reporter,
                        )
                    except EvalError as e:
                        raise BadRequest(
                            f"synth eval failed on dataset "
                            f"{spec.name!r}: {e}",
                            code="eval_error",
                        ) from e
                    dumped = synth_rep.model_dump(mode="json")
                    dumped["passed"] = synth_rep.passed
                    all_passed = all_passed and synth_rep.passed
                    reports.append(dumped)

        _ = wiki_root  # eval owns its own throwaway wiki tree
        # Single-report runs keep the legacy result shape so existing
        # client renderers (``render_eval_report``) Just Work; everything
        # else returns a ``{datasets: [...], passed: bool}`` envelope.
        if dataset is not None and len(reports) == 1:
            return reports[0]
        return {"datasets": reports, "passed": all_passed}

    return _runner


def _resolve_eval_modes(
    spec: Any, requested: list[str] | None
) -> list[str]:
    """Intersect requested eval_modes with the dataset's declared modes.

    ``None`` → use dataset's full ``spec.modes``; explicit list → keep
    only the modes the dataset also declares (silently drops mismatches
    so a multi-dataset ``--eval synth`` run can still cover the synth-
    capable ones without erroring on retrieval-only datasets)."""
    declared = list(spec.modes)
    if requested is None:
        return declared
    return [m for m in requested if m in declared]


__all__ = [
    "make_distill_runner",
    "make_eval_runner",
    "make_synth_runner",
]
