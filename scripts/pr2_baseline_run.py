"""Real-data baseline runner for PR2 lint propose.

Calls ``api.lint_propose`` directly (no server) against the
elon-musk-validation base so the result can be spot-checked without
spinning up ``dikw serve``. Prints a JSON summary suitable for
inclusion in ``evals/BASELINES.md``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Load .env from the base so OPENAI/codex tokens + DIKW_EMBEDDING_API_KEY
# resolve the same way ``dikw client`` would.
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

from dikw_core import api


async def _run(base: Path, rule: str, limit: int) -> None:
    if load_dotenv is not None:
        load_dotenv(base / ".env")
    report = await api.lint_propose(
        base, rule=rule, limit=limit, enable_llm=True  # type: ignore[arg-type]
    )
    summary = {
        "base": str(base),
        "rule": rule,
        "limit": limit,
        "proposals_count": len(report.proposals),
        "skipped_count": len(report.skipped),
        "proposals": [
            {
                "issue_path": p.issue_path,
                "issue_detail": p.issue_detail,
                "rationale": p.rationale,
                "source": p.source,
                "ops": [
                    {
                        "kind": op.kind,
                        "path": op.path,
                        "body_chars": len(op.new_body or ""),
                        "body_preview": (op.new_body or "")[:240],
                    }
                    for op in p.operations
                ],
            }
            for p in report.proposals[:5]
        ],
        "skipped_sample": report.skipped[:5],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    base = Path(os.environ.get("DIKW_PR2_BASELINE_BASE", "")).expanduser()
    if not base.is_dir():
        print(
            "set DIKW_PR2_BASELINE_BASE to the elon-musk-validation base path",
            file=sys.stderr,
        )
        return 2
    rule = os.environ.get("DIKW_PR2_BASELINE_RULE", "broken_wikilink")
    limit = int(os.environ.get("DIKW_PR2_BASELINE_LIMIT", "2"))
    asyncio.run(_run(base, rule, limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
