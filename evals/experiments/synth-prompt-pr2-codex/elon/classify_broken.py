"""Classify broken_wikilink targets from a diag run.

Usage (from a dikw-core checkout):
    uv run python classify_broken.py <broken_json> <vault_snapshot_dir>

Lint already ran exact->fuzzy->collision resolve, so every dumped target
failed both stages. Classes:

collision        — normalize key maps onto >=2 distinct pages (engine refuses
                   by design; Karpathy wrong-merge rule).
loose_near_miss  — beyond the fuzzy normalizer's reach but visibly close to an
                   existing title (substring containment either way): weakened
                   verbatim-copy discipline, the template-fixable class.
forward          — no resemblance to any existing title: a rule-3(c)
                   deliberate forward link (fixable lint debt by design).
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from dikw_core.domains.knowledge.links import _normalize_for_match

SUFFIX = " has no matching knowledge page"

broken = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
snap = Path(sys.argv[2])

titles: set[str] = set()
norm_to_titles: dict[str, set[str]] = {}
for p in snap.rglob("*.md"):
    m = re.search(r"^# (.+)$", p.read_text(encoding="utf-8"), flags=re.MULTILINE)
    if not m:
        continue
    title = m.group(1).strip()
    titles.add(title)
    norm_to_titles.setdefault(_normalize_for_match(title), set()).add(title)


def close_titles(t: str) -> list[str]:
    """Existing titles that contain t or are contained by t (len>=2 guard)."""
    out = []
    for x in titles:
        if len(t) >= 2 and len(x) >= 2 and (t in x or x in t) and t != x:
            out.append(x)
    return sorted(out)[:3]


classes = Counter()
rows = []
for issue in broken:
    t = issue["detail"].removesuffix(SUFFIX).strip().strip("[]").strip()
    hits = norm_to_titles.get(_normalize_for_match(t), set())
    near = close_titles(t)
    if len(hits) >= 2:
        cls = "collision"
    elif near:
        cls = "loose_near_miss"
    else:
        cls = "forward"
    classes[cls] += 1
    rows.append((cls, t, sorted(hits) if hits else near, issue["path"]))

for cls, t, ctx, path in sorted(rows):
    print(f"{cls:16s} [[{t}]]  ~{ctx}  in {path}")
print("\nsummary:", dict(classes), f"total={len(broken)}")
