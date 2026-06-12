"""Post-run diagnostic: dump broken_wikilink lint details + snapshot the vault.

Run from a dikw-core checkout right after run_elon.py, BEFORE anything resets
the base. Usage: python post_elon_diag.py <base> <out_json> <snapshot_dir>
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path

from dikw_core import api


async def main() -> None:
    base = Path(sys.argv[1])
    out = Path(sys.argv[2])
    snap = Path(sys.argv[3])

    report = await api.lint(base)
    broken = [
        {"kind": str(i.kind), "path": i.path, "detail": i.detail, "line": i.line}
        for i in report.issues
        if str(i.kind) == "broken_wikilink" or getattr(i.kind, "value", "") == "broken_wikilink"
    ]
    out.write_text(json.dumps(broken, ensure_ascii=False, indent=2), encoding="utf-8")

    if snap.exists():
        shutil.rmtree(snap)
    shutil.copytree(base / "knowledge", snap)
    print(f"broken={len(broken)} -> {out}; vault snapshot -> {snap}")


asyncio.run(main())
