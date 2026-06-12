"""One elon-musk A/B run: wipe base K-state, ingest, synth --verify --judge, dump metrics.

Run from a dikw-core checkout root (branch decides the arm):

    uv run --env-file .env python "C:/Users/HE LE/Project/opendikw/scratch-elon-ab/run_elon.py" \
        --base "C:/Users/HE LE/Project/opendikw/scratch-codex-base" \
        --out  "C:/Users/HE LE/Project/opendikw/scratch-elon-ab/runs/<arm>-<n>.json"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from pathlib import Path

from dikw_core import api
from dikw_core.config import CONFIG_FILENAME, load_config
from dikw_core.domains.knowledge.links import parse_links
from dikw_core.providers import build_embedder, build_llm
from dikw_core.schemas import LinkType


def wipe(base: Path) -> None:
    """Reset K-layer state; keep dikw.yml, sources/, .dikw/auth.json."""
    for sub in ("knowledge", "trash"):
        p = base / sub
        if p.exists():
            shutil.rmtree(p)
    (base / "knowledge").mkdir()
    for f in (base / ".dikw").glob("index.sqlite*"):
        f.unlink()


async def run(base: Path, out: Path) -> None:
    cfg = load_config(base / CONFIG_FILENAME)
    llm = build_llm(cfg.provider, base_root=base)
    embedder = build_embedder(cfg.provider)

    t0 = time.time()
    await api.ingest(base, embedder=embedder)
    synth = await api.synthesize(
        base, llm=llm, embedder=embedder, verify=True, judge=True
    )
    elapsed = time.time() - t0

    total_wikilinks = 0
    page_count = 0
    for md in (base / "knowledge").rglob("*.md"):
        body = md.read_text(encoding="utf-8")
        total_wikilinks += sum(
            1 for link in parse_links(body) if link.kind is LinkType.WIKILINK
        )
        page_count += 1

    lint_report = await api.lint(base)
    lint_by_kind: dict[str, int] = {}
    for issue in lint_report.issues:
        lint_by_kind[issue.kind] = lint_by_kind.get(issue.kind, 0) + 1

    v = synth.verify
    result = {
        "elapsed_s": round(elapsed, 1),
        "created": synth.created,
        "updated": synth.updated,
        "pages_on_disk": page_count,
        "groups_processed": synth.groups_processed,
        "errors": synth.errors,
        "persist_errors": len(synth.persist_errors),
        "slug_merge_count": synth.slug_merge_count,
        "unresolved_wikilinks": synth.unresolved_wikilinks,
        "total_wikilinks": total_wikilinks,
        "wikilink_resolved_ratio": (
            (total_wikilinks - synth.unresolved_wikilinks) / total_wikilinks
            if total_wikilinks
            else None
        ),
        "lint_by_kind": lint_by_kind,
        "verify": None
        if v is None
        else {
            "passed": v.passed,
            "pages_checked": v.pages_checked,
            "lint_findings": len(v.lint_findings),
            "lint_kinds": sorted({f.kind for f in v.lint_findings}),
            "orphan_pages": len(v.orphan_pages),
            "duplicate_checked": v.duplicate_checked,
            "duplicate_ratio": v.duplicate_ratio,
            "grounding_checked": v.grounding_checked,
            "grounding_entailment_ratio": v.grounding_entailment_ratio,
            "grounding_ci": list(v.grounding_ci),
            "grounding_n_judged": v.grounding_n_judged,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = Path(args.base)
    wipe(base)
    asyncio.run(run(base, Path(args.out)))


if __name__ == "__main__":
    main()
