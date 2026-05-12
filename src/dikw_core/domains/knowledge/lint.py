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
from typing import Literal

import frontmatter

from ...schemas import Layer, LinkType
from ...storage.base import Storage
from .links import build_fuzzy_index, parse_links, resolve_links

# Heuristic thresholds for ``non_atomic_page``. A page is flagged when ANY
# of these are exceeded — they're independent symptoms of "this page is
# really N pages glued together":
# - body chars: catches bilingual/duplicate content; permissive enough
#   that single-topic notes with substantive narrative don't false-trigger
# - H2 count: rare in atomic notes, common in MOC-style aggregations
# - wikilink count: entity-rich event pages routinely cite 8-12
#   participants without being non-atomic; only true index pages
#   accumulate 15+ distinct references
# - tag-domain count: only namespaced tags (``area/topic``) count;
#   flat tags ignored, since LLM-generated atomic pages routinely carry
#   3-5 flat tags. See ``evals/BASELINES.md`` for calibration data.
_ATOMIC_BODY_CHARS = 2500
_ATOMIC_H2_COUNT = 3
_ATOMIC_WIKILINK_COUNT = 15
_ATOMIC_TAG_DOMAIN_COUNT = 1

_H1_LINE = re.compile(r"^\s{0,3}#\s+\S", flags=re.MULTILINE)
_H2_LINE = re.compile(r"^\s{0,3}##\s+\S", flags=re.MULTILINE)
# Strip ``` fenced blocks before counting headings — a code example
# like ``# install deps`` / ``## setup`` would otherwise inflate the
# H1/H2 counts and false-flag an atomic technical note.
_FENCED_CODE = re.compile(r"```[\s\S]*?```", flags=re.MULTILINE)


LintKind = Literal[
    "broken_wikilink",
    "orphan_page",
    "duplicate_title",
    "non_atomic_page",
]


@dataclass(frozen=True)
class LintIssue:
    kind: LintKind
    path: str
    detail: str
    line: int | None = None


@dataclass
class LintReport:
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def by_kind(self) -> dict[LintKind, int]:
        counts: dict[LintKind, int] = {}
        for i in self.issues:
            counts[i.kind] = counts.get(i.kind, 0) + 1
        return counts


@dataclass(frozen=True)
class AtomicityVerdict:
    """Outcome of ``check_atomicity`` — atomic flag + violation messages.

    The same heuristic powers two consumers: ``run_lint`` (loud surfacing of
    individual offending pages) and ``eval/metrics.atomicity_score``
    (aggregate quality gate). Keeping the implementation pure lets both
    callers share one source of truth for the thresholds.
    """

    atomic: bool
    violations: tuple[str, ...]


def check_atomicity(*, body: str, tags: list[str]) -> AtomicityVerdict:
    """Decide whether a wiki page body+tags violate the atomicity heuristic.

    A page is **non-atomic** when ANY of the independent symptoms below
    trigger; each contributes one entry to ``violations`` so callers can
    surface the specific reasons.

    * body chars > ``_ATOMIC_BODY_CHARS``
    * H1 count > 1 (atomic page should have exactly one title)
    * H2 count > ``_ATOMIC_H2_COUNT``
    * distinct wikilink targets > ``_ATOMIC_WIKILINK_COUNT``
    * namespaced-tag domains > ``_ATOMIC_TAG_DOMAIN_COUNT``

    Fenced code blocks are stripped before counting headings, so technical
    notes with inline shell snippets (``# install deps``) don't false-trigger.
    Wikilinks are counted by distinct target so a single-topic page that
    repeats one entity (``[[Elon Musk]]`` x16) stays atomic.
    """
    # Local imports avoid a circular dependency: ``links`` does not import
    # ``lint``, but keeping this lazy makes the dependency graph explicit.
    from ...schemas import LinkType
    from .links import parse_links

    violations: list[str] = []
    if len(body) > _ATOMIC_BODY_CHARS:
        violations.append(f"body {len(body)} chars > {_ATOMIC_BODY_CHARS}")
    prose = _FENCED_CODE.sub("", body)
    h1_count = len(_H1_LINE.findall(prose))
    if h1_count > 1:
        violations.append(
            f"{h1_count} H1 sections — atomic page should have exactly one"
        )
    h2_count = len(_H2_LINE.findall(prose))
    if h2_count > _ATOMIC_H2_COUNT:
        violations.append(f"{h2_count} H2 sections > {_ATOMIC_H2_COUNT}")
    page_links = parse_links(body)
    wikilink_targets = {
        link.target for link in page_links if link.kind is LinkType.WIKILINK
    }
    distinct_wikilinks = len(wikilink_targets)
    if distinct_wikilinks > _ATOMIC_WIKILINK_COUNT:
        violations.append(
            f"{distinct_wikilinks} distinct wikilinks > {_ATOMIC_WIKILINK_COUNT}"
        )
    namespaced = [t for t in tags if isinstance(t, str) and "/" in t]
    domains = sorted({t.split("/", 1)[0].strip() for t in namespaced})
    if len(domains) > _ATOMIC_TAG_DOMAIN_COUNT:
        violations.append(f"tags span {len(domains)} domains: {', '.join(domains)}")
    return AtomicityVerdict(atomic=not violations, violations=tuple(violations))


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

    # Share the same resolve semantics as engine persistence
    # (``_persist_wiki_page``): exact -> fuzzy normalize -> collision
    # refusal. Without this lint reports broken_wikilink on plurals that
    # storage already resolved, and silently swallows fuzzy collisions
    # that storage refused to guess.
    title_to_path: dict[str, str] = {
        t: dup_paths[0] for t, dup_paths in title_to_paths.items()
    }
    fuzzy_index = build_fuzzy_index(title_to_path)

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
        _, unresolved = resolve_links(
            doc.doc_id,
            page_links,
            title_to_path=title_to_path,
            fuzzy_index=fuzzy_index,
        )
        for u in unresolved:
            issues.append(
                LintIssue(
                    kind="broken_wikilink",
                    path=doc.path,
                    detail=f"{u.target_text} has no matching wiki page",
                    line=u.line,
                )
            )

        # atomicity check — delegate to the pure helper so eval/metrics
        # can apply the exact same thresholds.
        # Wiki pages are user-editable: a hand-written ``tags: 2024`` parses
        # as a scalar, not a list. Normalise before passing in.
        raw_tags = post.metadata.get("tags")
        if not isinstance(raw_tags, list):
            raw_tags = []
        tag_list = [t for t in raw_tags if isinstance(t, str) and t.strip()]
        verdict = check_atomicity(body=body, tags=tag_list)
        if not verdict.atomic:
            issues.append(
                LintIssue(
                    kind="non_atomic_page",
                    path=doc.path,
                    detail=(
                        "page looks like multiple atomic notes glued together: "
                        + "; ".join(verdict.violations)
                        + " — consider splitting the page by hand"
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
