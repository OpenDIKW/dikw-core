"""Tau sweep for ``fact_grounding_ratio``.

Question: ``synth.grounding_threshold`` defaults to 0.65, but real-LLM
calibration (BASELINES.md 2026-05-13) showed only ~26% of synth-page
claims clear it on the mvp corpus — suspiciously low. The synth output
inspected by hand looks substantially grounded, so the suspicion is
that 0.65 is too tight for the embedder's similarity scale on natural-
language claims (Qwen3-Embedding-0.6B @ 1024-dim native).

This script answers: at what tau does ``fact_grounding_ratio`` best
match a human's "is this claim actually grounded in the source?"
judgement?

Approach: run a real-LLM ingest+synth on mvp ONCE, dump every claim's
peak cosine against its source chunks, then reduce ``fact_grounding_ratio``
at multiple tau values. Also emit the top-N highest- and lowest-cosine
claims so the author can manually label them (the operational tau is
"the value that classifies the 10-sample hand label correctly").

Usage::

    cd <base with codex auth + DIKW_EMBEDDING_API_KEY in .env>
    uv --directory <dikw-core> run python -m scripts.tau_sweep_grounding \\
        --base . --out /tmp/tau-sweep.json

The script does NOT mutate the wiki — it spins up a throwaway tempdir
identical to ``run_synth_eval``. Re-run as many times as you need;
nothing persists beyond the JSON dump.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

from dikw_core import api
from dikw_core.config import (
    CONFIG_FILENAME,
    SQLiteStorageConfig,
    dump_config_yaml,
    load_config,
)
from dikw_core.domains.knowledge.wiki import read_page
from dikw_core.eval.dataset import load_dataset
from dikw_core.eval.metrics import (
    compute_grounding_cosines,
    reduce_grounding_ratio,
)
from dikw_core.providers import build_embedder, build_llm
from dikw_core.schemas import Layer

logger = logging.getLogger("tau_sweep")

# Match the analysis taus listed in the 2026-05-13 BASELINES.md
# follow-up so the resulting numbers feed directly into the writeup.
_TAUS: tuple[float, ...] = (0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70)


async def _run(base: Path, dataset: str, out_path: Path) -> None:
    spec = load_dataset(dataset)
    cfg_template = load_config(base / "dikw.yml")

    with tempfile.TemporaryDirectory(prefix="dikw-tau-sweep-") as tmp:
        wiki = Path(tmp)
        # Mirror run_synth_eval's setup: copy corpus into sources/ +
        # write a fresh dikw.yml with the dataset's synth.page_types.
        (wiki / "sources").mkdir()
        for src in spec.corpus_dir.iterdir():
            if src.is_file():
                shutil.copy2(src, wiki / "sources" / src.name)
        cfg = cfg_template.model_copy(deep=True)
        # Force sqlite for the throwaway wiki — Postgres needs a live
        # database we don't want to provision per script run.
        cfg.storage = SQLiteStorageConfig(path=".dikw/index.sqlite")
        cfg.schema_.page_types = list(spec.synth.page_types)
        (wiki / CONFIG_FILENAME).write_text(
            dump_config_yaml(cfg), encoding="utf-8"
        )

        embedder = build_embedder(cfg.provider)
        llm = build_llm(cfg.provider, wiki_base=wiki)

        logger.info("ingesting corpus from %s", spec.corpus_dir)
        await api.ingest(wiki, embedder=embedder)

        logger.info("synthesising K-layer pages (real LLM)")
        await api.synthesize(
            wiki, force_all=True, llm=llm, embedder=embedder
        )

        # Pull pages + source chunks back via the storage adapter, same
        # path run_synth_eval uses.
        _cfg, _root, storage = await api._with_storage(wiki)
        try:
            wiki_docs = list(
                await storage.list_documents(layer=Layer.WIKI, active=True)
            )
            source_docs = list(
                await storage.list_documents(layer=Layer.SOURCE, active=True)
            )
            pages = []
            for doc in wiki_docs:
                try:
                    pages.append(read_page(wiki, doc.path))
                except FileNotFoundError:
                    continue
            chunks_by_source = {}
            for doc in source_docs:
                chunks_by_source[doc.path] = await storage.list_chunks(
                    doc.doc_id
                )
        finally:
            await storage.close()

        pages_with_sources = [
            (page, page.sources[0]) for page in pages if page.sources
        ]
        logger.info(
            "computing per-claim cosines (%d pages, %d sources)",
            len(pages_with_sources),
            len(chunks_by_source),
        )
        claims = await compute_grounding_cosines(
            pages_with_sources=pages_with_sources,
            chunks_by_source=chunks_by_source,
            embedder=embedder,
            embedding_model=cfg.provider.embedding_model,
        )

    # Reduce at each tau (pure math — no more LLM/embedding cost).
    ratios = {
        f"{tau:.2f}": reduce_grounding_ratio(
            claims, pages_with_sources=pages_with_sources, tau=tau
        )
        for tau in _TAUS
    }

    # Sort claims by cosine to make hand-labelling tractable: highest
    # (most-confidently-grounded) + lowest (most-confidently-ungrounded)
    # 10 each give a 20-sample dataset where the human signal is
    # strongest.
    by_cos = sorted(claims, key=lambda c: c.max_cosine, reverse=True)
    head = [_claim_to_dict(c) for c in by_cos[:10]]
    tail = [_claim_to_dict(c) for c in by_cos[-10:]]

    payload: dict[str, Any] = {
        "dataset": dataset,
        "embedding_model": cfg.provider.embedding_model,
        "embedding_dim": cfg.provider.embedding_dim,
        "n_pages": len(pages_with_sources),
        "n_claims": len(claims),
        "ratios_by_tau": ratios,
        "top_grounded_10": head,
        "least_grounded_10": tail,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("wrote %s", out_path)
    print(json.dumps({"ratios_by_tau": ratios}, indent=2))


def _claim_to_dict(c: Any) -> dict[str, Any]:
    return {
        "page": c.page_path,
        "source": c.source_path,
        "max_cosine": round(c.max_cosine, 4),
        "claim": c.claim,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Path to a base with codex auth + DIKW_EMBEDDING_API_KEY in .env",
    )
    ap.add_argument(
        "--dataset",
        default="mvp",
        help="Eval dataset name (default: mvp)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/tau-sweep.json"),
        help="Path to write the per-claim cosine dump",
    )
    args = ap.parse_args()

    if not args.base.is_dir():
        print(f"error: --base {args.base} is not a directory", file=sys.stderr)
        sys.exit(2)

    if load_dotenv is not None:
        load_dotenv(args.base / ".env")

    asyncio.run(_run(args.base, args.dataset, args.out))


if __name__ == "__main__":
    main()
