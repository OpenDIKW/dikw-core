"""K-layer hygiene checker.

Reports four classes of issue that are safe to detect deterministically:

* ``broken_wikilink`` — wikilinks whose target title isn't a known K/W page.
* ``orphan_page`` — pages with no inbound wikilinks and no listing source.
* ``duplicate_title`` — more than one K-layer page with identical title.
* ``non_atomic_page`` — page body looks like multiple wikipage worth of
  content stuffed together (long body, many H2 sections, link-list-y).
  Layer-3 backstop for the Zettelkasten atomicity rule the synth prompt
  enforces in layer 1; the prompt can drift, this can't.

Phases 3+ may add semantic checks (stale claims, missing evidence for
approved wisdom items, etc.); this module intentionally stays lexical.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

from ...schemas import Layer, LinkType
from ...storage.base import Storage
from .links import parse_links

# Heuristic thresholds for ``non_atomic_page``. A page is flagged when ANY
# of these are exceeded — they're independent symptoms of "this page is
# really N pages glued together". Calibrated 2026-05-08 against
# elon-musk.md (1500-line subset, 70 LLM-synthesised K pages):
# - body 2500 chars: catches multi-language duplicate content (bilingual
#   pages); permissive enough that single-topic notes with substantive
#   narrative don't false-trigger
# - H2 ≥ 4: rare in atomic notes, common in MOC-style aggregations
# - wikilinks ≥ 15: entity-rich event pages routinely cite 8-12
#   participants without being non-atomic; only true index pages
#   accumulate 15+ distinct references
_ATOMIC_BODY_CHARS = 2500
_ATOMIC_H2_COUNT = 3
_ATOMIC_WIKILINK_COUNT = 15
# Tags using namespaces (``area/topic`` form) that span > 1 top-level
# area suggest the page straddles unrelated knowledge domains — almost
# always N atomic notes glued together. Flat tags (no "/") are *ignored*
# entirely: 2026-05-08 real-data validation on elon-musk.md showed that
# LLM-generated atomic pages routinely carry 3-5 flat tags
# (e.g. ``entrepreneur, biography, spacex, tesla``), so treating each
# flat tag as its own domain produced 100% false positives. The
# heuristic only fires when the wiki actually adopts namespaced tags.
_ATOMIC_TAG_DOMAIN_COUNT = 1

_H2_LINE = re.compile(r"^\s{0,3}##\s+\S", flags=re.MULTILINE)


@dataclass(frozen=True)
class LintIssue:
    kind: str            # broken_wikilink | orphan_page | duplicate_title
    path: str            # the doc exhibiting the issue
    detail: str
    line: int | None = None


@dataclass
class LintReport:
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for i in self.issues:
            counts[i.kind] = counts.get(i.kind, 0) + 1
        return counts


async def run_lint(storage: Storage, *, root: Path) -> LintReport:
    """Scan K-layer pages and return a structured report."""
    issues: list[LintIssue] = []

    wiki_docs = list(await storage.list_documents(layer=Layer.WIKI, active=True))
    title_to_paths: dict[str, list[str]] = defaultdict(list)
    inbound: Counter[str] = Counter()
    paths: list[str] = []

    for doc in wiki_docs:
        title = doc.title or Path(doc.path).stem
        title_to_paths[title].append(doc.path)
        paths.append(doc.path)

    title_set = set(title_to_paths)

    # broken wikilinks — re-parse on-disk bodies; storage.links only records
    # resolved links, so we must look at the raw text to find unresolved ones.
    for doc in wiki_docs:
        abs_path = (root / doc.path).resolve()
        if not abs_path.is_file():
            continue
        try:
            post = frontmatter.load(str(abs_path))
        except Exception:
            continue
        body = post.content
        page_links = parse_links(body)
        wikilink_count = 0
        for link in page_links:
            if link.kind is not LinkType.WIKILINK:
                continue
            wikilink_count += 1
            if (
                link.target not in title_set
                and not any(t.lower() == link.target.lower() for t in title_set)
            ):
                issues.append(
                    LintIssue(
                        kind="broken_wikilink",
                        path=doc.path,
                        detail=f"[[{link.target}]] has no matching wiki page",
                        line=link.line,
                    )
                )

        # atomicity check — independent symptoms; report once per page.
        violations: list[str] = []
        if len(body) > _ATOMIC_BODY_CHARS:
            violations.append(f"body {len(body)} chars > {_ATOMIC_BODY_CHARS}")
        h2_count = len(_H2_LINE.findall(body))
        if h2_count > _ATOMIC_H2_COUNT:
            violations.append(f"{h2_count} H2 sections > {_ATOMIC_H2_COUNT}")
        if wikilink_count > _ATOMIC_WIKILINK_COUNT:
            violations.append(
                f"{wikilink_count} wikilinks > {_ATOMIC_WIKILINK_COUNT}"
            )
        raw_tags = post.metadata.get("tags") or []
        tag_list = [t for t in raw_tags if isinstance(t, str) and t.strip()]
        namespaced = [t for t in tag_list if "/" in t]
        domains = sorted({t.split("/", 1)[0].strip() for t in namespaced})
        if len(domains) > _ATOMIC_TAG_DOMAIN_COUNT:
            violations.append(f"tags span {len(domains)} domains: {', '.join(domains)}")
        if violations:
            issues.append(
                LintIssue(
                    kind="non_atomic_page",
                    path=doc.path,
                    detail=(
                        "page looks like multiple atomic notes glued together: "
                        + "; ".join(violations)
                        + " — consider splitting via `dikw synth --force <path>`"
                    ),
                )
            )

        # accumulate inbound link counts (resolved links from storage)
        for stored in await storage.links_from(doc.doc_id):
            if stored.link_type is LinkType.WIKILINK:
                inbound[stored.dst_path] += 1

    # orphans — no inbound wikilinks AND not referenced from index.md/log.md
    orphan_exclusions = {"wiki/index.md", "wiki/log.md"}
    for doc in wiki_docs:
        if doc.path in orphan_exclusions:
            continue
        if inbound[doc.path] == 0:
            issues.append(
                LintIssue(
                    kind="orphan_page",
                    path=doc.path,
                    detail="no inbound wikilinks from other K-layer pages",
                )
            )

    # duplicate titles — reported per extra path beyond the first
    for title, dup_paths in title_to_paths.items():
        if len(dup_paths) > 1:
            primary = dup_paths[0]
            for extra in dup_paths[1:]:
                issues.append(
                    LintIssue(
                        kind="duplicate_title",
                        path=extra,
                        detail=f"title '{title}' also used by {primary}",
                    )
                )

    return LintReport(issues=issues)
