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
from typing import Any, Literal, get_args

import frontmatter

from ...schemas import Layer, LinkType
from ...storage.base import Storage
from ..data.path_norm import normalize_path
from .links import build_fuzzy_index, parse_links, resolve_links
from .wiki import frontmatter_str_list

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
    "missing_provenance",
]


@dataclass(frozen=True)
class LintIssue:
    kind: LintKind
    path: str
    detail: str
    line: int | None = None


@dataclass(frozen=True)
class PageMeta:
    """Frontmatter slice ``run_lint`` already had to parse and that
    downstream callers (``lint_propose`` building ``WikiPageMeta``)
    would otherwise re-parse from disk. Keyed by ``LintReport.page_meta``.
    """

    sources: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass
class LintReport:
    issues: list[LintIssue] = field(default_factory=list)
    # Paths of pages that opted out of one or more lint rules via the
    # per-page ``lint: {skip: [<kind>, ...]}`` frontmatter block. The
    # rule is filtered out before ``issues`` is built, but the path
    # surfaces here so audit tools (and the JSON CLI output) can show
    # "N pages intentionally exempted" without scanning every page's
    # frontmatter a second time.
    acknowledged_leaves: list[str] = field(default_factory=list)
    # Cached frontmatter slice keyed by page path. Populated as a
    # by-product of the per-page frontmatter load already needed for
    # rule checks so downstream consumers (``lint_propose``) don't have
    # to re-parse the same files.
    page_meta: dict[str, PageMeta] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues

    def by_kind(self) -> dict[LintKind, int]:
        counts: dict[LintKind, int] = {}
        for i in self.issues:
            counts[i.kind] = counts.get(i.kind, 0) + 1
        return counts


def _read_lint_skip(metadata: dict[str, Any]) -> set[LintKind]:
    """Extract suppressed lint kinds from a page's ``lint:`` frontmatter.

    The frontmatter is user-editable, so robustness > strict validation:
    a malformed ``lint:`` block (list at top, non-list ``skip``, non-string
    kind) is silently ignored — the rule fires as if no suppression
    existed. Only well-formed ``lint: {skip: ["orphan_page", ...]}``
    entries take effect.
    """
    raw = metadata.get("lint")
    if not isinstance(raw, dict):
        return set()
    raw_skip = raw.get("skip")
    if not isinstance(raw_skip, list):
        return set()
    valid: set[LintKind] = set(get_args(LintKind))
    out: set[LintKind] = set()
    for k in raw_skip:
        if isinstance(k, str) and k in valid:
            out.add(k)
    return out


@dataclass(frozen=True)
class AtomicityVerdict:
    """Atomic-flag + ordered violation messages produced by
    ``check_atomicity``."""

    atomic: bool
    violations: tuple[str, ...]


def check_atomicity(*, body: str, tags: list[str]) -> AtomicityVerdict:
    """Decide whether a wiki page body+tags violate the atomicity heuristic.

    A page is **non-atomic** when ANY of these independent symptoms trigger;
    each contributes one entry to ``violations``:

    * body chars > ``_ATOMIC_BODY_CHARS``
    * H1 count > 1 (atomic page should have exactly one title)
    * H2 count > ``_ATOMIC_H2_COUNT``
    * distinct wikilink targets > ``_ATOMIC_WIKILINK_COUNT``
    * namespaced-tag domains > ``_ATOMIC_TAG_DOMAIN_COUNT``

    Fenced code blocks are stripped before counting headings so technical
    notes with inline shell snippets (``# install deps``) don't false-trigger.
    Wikilinks are counted by distinct target so a single-topic page that
    repeats one entity (``[[Elon Musk]]`` x16) stays atomic.
    """
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
    # path → set of LintKind the page opted out of via frontmatter.
    # Populated below in the same per-page frontmatter-load loop; consulted
    # both when appending per-page issues (broken_wikilink / non_atomic)
    # and when emitting orphan_page issues a second time later.
    suppressions: dict[str, set[LintKind]] = {}
    page_meta: dict[str, PageMeta] = {}

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
        skip_kinds = _read_lint_skip(post.metadata)
        if skip_kinds:
            suppressions[doc.path] = skip_kinds
        # Stash the frontmatter slice we already paid to parse so
        # ``lint_propose`` can build ``WikiPageMeta`` without a second
        # disk pass over the same K-layer pages. ``frontmatter_str_list``
        # is the shared malformed-shape guard (scalar / dict / null →
        # ``[]``) — same one ``persist_wiki_page`` and
        # ``MissingProvenanceFixer`` use, so this lint pass's view of
        # ``sources:`` matches what storage actually stored.
        sources_tuple = tuple(frontmatter_str_list(post.metadata, "sources"))
        tags_tuple = tuple(frontmatter_str_list(post.metadata, "tags"))
        page_meta[doc.path] = PageMeta(sources=sources_tuple, tags=tags_tuple)

        # Surface pages whose provenance table is out of sync with
        # frontmatter — typical on bases that existed before the
        # provenance feature shipped, or after a user hand-edits
        # ``sources:`` outside of synth / lint-apply. ``expected !=
        # existing`` catches five sub-cases with one comparison: zero
        # existing rows (never reconciled), partial rows (interrupted
        # reconcile), stale rows (frontmatter edited after a prior
        # reconcile), *cleared* sources (user removed the frontmatter
        # key but reconciled rows are still around), and *raw-spelling
        # drift* (the normalized keys still match, but the user edited
        # casing / NFC form so the stored raw ``source_path`` no longer
        # matches what the API now returns from frontmatter). All
        # resolve to the same fix — MissingProvenanceFixer is
        # deterministic, no LLM. See
        # docs/adr/0001-provenance-as-separate-edge.md.
        #
        # Comparison is dict (key -> raw) not set (keys only) because
        # the API contract preserves raw frontmatter spelling
        # faithfully — a key-only comparison would silently hide
        # ``Sources/Foo.md`` -> ``sources/foo.md`` edits, leaving the
        # forward-leg result drifting from the file.
        #
        # The storage probe is unconditional within the not-skipped
        # branch because the "frontmatter empty + table empty" case
        # only collapses cleanly *after* asking the table — we can't
        # short-circuit on ``sources_tuple`` alone without losing the
        # stale-rows-when-cleared case.
        if "missing_provenance" not in skip_kinds:
            existing_prov = await storage.provenance_from(doc.doc_id)
            existing_map = {
                e.source_path_key: e.source_path for e in existing_prov
            }
            expected_map = {normalize_path(s): s for s in sources_tuple}
            if existing_map != expected_map:
                issues.append(
                    LintIssue(
                        kind="missing_provenance",
                        path=doc.path,
                        detail=(
                            f"frontmatter declares {len(sources_tuple)} "
                            f"source(s); provenance table has "
                            f"{len(existing_prov)} matching row(s)"
                        ),
                    )
                )

        page_links = parse_links(body)
        _, unresolved = resolve_links(
            doc.doc_id,
            page_links,
            title_to_path=title_to_path,
            fuzzy_index=fuzzy_index,
        )
        if "broken_wikilink" not in skip_kinds:
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
        # can apply the exact same thresholds. ``tags_tuple`` already
        # absorbed the malformed-shape guard; drop the empty strings
        # ``check_atomicity`` doesn't want here so the helper itself
        # stays a pure "list of strings" extractor.
        tag_list = [t for t in tags_tuple if t.strip()]
        verdict = check_atomicity(body=body, tags=tag_list)
        if not verdict.atomic and "non_atomic_page" not in skip_kinds:
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
        if "orphan_page" in suppressions.get(doc.path, set()):
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
                if "duplicate_title" in suppressions.get(extra, set()):
                    continue
                issues.append(
                    LintIssue(
                        kind="duplicate_title",
                        path=extra,
                        detail=f"title '{title}' also used by {primary}",
                    )
                )

    return LintReport(
        issues=issues,
        acknowledged_leaves=sorted(suppressions),
        page_meta=page_meta,
    )
