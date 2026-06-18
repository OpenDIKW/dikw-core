"""K/W-layer hygiene checker.

Reports issue classes that are safe to detect deterministically (see the
``LintKind`` literal for the full set):

* ``broken_wikilink`` — wikilinks whose target title isn't a known K page.
* ``orphan_page`` — pages with no inbound wikilinks and no listing source.
* ``duplicate_title`` — more than one K-layer page with identical title.
* ``non_atomic_page`` — page body looks like multiple wikipage worth of
  content stuffed together (long body, many H2 sections, link-list-y).
  Layer-3 backstop for the Zettelkasten atomicity rule the synth prompt
  enforces in layer 1; the prompt can drift, this can't.
* ``missing_provenance`` — frontmatter ``sources:`` disagrees with the
  provenance table.
* ``invalid_wisdom_status`` — a wisdom page's ``status:`` isn't a known enum.
* ``uncategorized`` — a K page filed under the fallback category.
* ``title_slug_quality`` — K-page title/slug hygiene (missing/blank H1,
  title-vs-H1 drift, degenerate ``untitled`` slug).
* ``missing_file`` — an *active* ``documents`` row (D/K/W) whose backing file
  is gone from disk; the deterministic fixer purges the orphaned row
  (ADR-0005, the filesystem is the source of truth).
* ``stale_index`` — an *active* K/W row whose on-disk body hash no longer
  matches the indexed ``hash`` (a hand-edit outside dikw); the deterministic
  reindex fixer re-projects the current bytes without rewriting the file.
* ``untracked_file`` — a ``.md`` / ``.markdown`` file under ``knowledge/`` or
  ``wisdom/`` with no active row (hand-written or restored outside dikw); the
  reindex fixer indexes it, making hand-authored pages first-class.
* ``dangling_provenance`` — a K/W page whose ``sources:`` provenance edge points
  at a source file that no longer exists on disk. Read-only — surfaced, never
  auto-repaired (the frontmatter is the user's to edit; ADR-0005, ADR-0001).

``stale_index`` / ``untracked_file`` / ``dangling_provenance`` are K/W only —
D-layer source adds and edits are owned by ``ingest`` (zero overlap);
``missing_file`` spans D/K/W.

Later phases may add semantic checks; this module intentionally stays
lexical.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import frontmatter

from ...schemas import Layer, LinkType, WisdomStatus
from ...storage.base import Storage
from ..data.backends import supported_extensions
from ..data.backends.markdown import content_hash
from ..data.path_norm import normalize_path
from .links import build_title_indexes, parse_links, resolve_links
from .page import SLUG_FALLBACK, category_from_path, frontmatter_str_list

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

# ``title_slug_quality`` detection. ``_H1_CAPTURE`` grabs the first ATX H1's
# text (after fenced code is stripped); ``_TITLE_WORD`` is a Unicode word-char
# probe (CJK counts, so a Chinese title is never "punctuation-only").
# Two regex subtleties, both load-bearing for the no-false-positive contract:
#   * ``[ \t]`` not ``\s`` for the gap/indent — ``\s`` matches newlines, so a
#     blank ``#`` heading would greedily swallow the next paragraph as its title.
#   * the trailing ``[ \t]*#*[ \t]*$`` strips an ATX *closing* hash sequence
#     (``# Title #``), so this agrees byte-for-byte with synthesize.py's
#     ``_ATX_TITLE`` (``\s*#*\s*$``) — the regex that produced the frontmatter
#     ``title:`` we compare against. Without it, ``# Title #`` would read as
#     ``Title #`` here but ``Title`` in frontmatter and the title-drift leg
#     would false-fire on the engine's own correct output.
_H1_CAPTURE = re.compile(
    r"^[ \t]{0,3}#[ \t]+(.+?)[ \t]*#*[ \t]*$", flags=re.MULTILINE
)
_TITLE_WORD = re.compile(r"\w")
# An ASCII alphanumeric is what ``slugify`` keeps; a title with at least one is
# what tells a *degenerate* ``untitled`` slug (a pure-CJK title that collapsed)
# apart from a title legitimately spelled "Untitled" (which slugifies to the
# same literal but is a real slug). ``slugify``'s output alone can't distinguish
# them — both return ``"untitled"`` — so the lint inspects the title's content.
_ASCII_SLUGGABLE = re.compile(r"[A-Za-z0-9]")


LintKind = Literal[
    "broken_wikilink",
    "orphan_page",
    "duplicate_title",
    "non_atomic_page",
    "missing_provenance",
    "invalid_wisdom_status",
    "uncategorized",
    "title_slug_quality",
    "missing_file",
    "stale_index",
    "untracked_file",
    "dangling_provenance",
]

# 0.3.0 PR2 — frontmatter ``status:`` enum values the engine accepts on
# wisdom-layer pages. The markdown backend already collapses unknown
# values to ``DocumentRecord.status = None`` so ingest never blocks;
# this lint surfaces the divergence so the user sees the typo without
# having to grep their wisdom tree.
_VALID_WISDOM_STATUS_VALUES = {s.value for s in WisdomStatus}


@dataclass(frozen=True)
class LintIssue:
    kind: LintKind
    path: str
    detail: str
    line: int | None = None


@dataclass(frozen=True)
class PageMeta:
    """Frontmatter slice ``run_lint`` already had to parse and that
    downstream callers (``lint_propose`` building ``KnowledgePageMeta``)
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
    """Decide whether a knowledge page body+tags violate the atomicity heuristic.

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


def check_title_slug_quality(
    *, body: str, frontmatter_title: str | None, stem: str
) -> tuple[str, ...]:
    """Deterministic K-page title/slug hygiene — three zero-false-positive checks.

    Returns an ordered tuple of violation messages (empty == clean):

    * the body has no usable ``# Title`` ATX heading (absent, blank, or
      punctuation-only — a Chinese title is *not* punctuation-only, CJK counts
      as a word character).
    * ``frontmatter_title`` disagrees with the body ``# H1`` — the genuine
      title drift. ``write_page`` always serialises the two equal, so a
      divergence is a hand-edit to one side; storage indexes by frontmatter
      title while the user reads the H1, so the two silently disagreeing is a
      real hazard.
    * ``stem`` is the ``untitled`` slug fallback (:data:`page.SLUG_FALLBACK`),
      which is only reachable when ``slugify`` collapsed a non-ASCII title the
      LLM gave no ASCII/pinyin slug for. (There is no ``-NNN`` collision
      suffix — same-slug pages are merged by ``dedup_pages_by_slug``, never
      counter-suffixed — so an exact match is correct.)

    Deliberately NOT a ``slugify(title) == stem`` comparison: slugs are
    LLM-chosen and *intentionally* diverge from ``slugify(title)`` (stop-word
    dropping ``The DIKW Pyramid`` -> ``dikw-pyramid``, pinyin ``神经网络`` ->
    ``shen-jing-wang-luo``), and wikilinks resolve by title not slug, so that
    comparison would red-flag the engine's own correct output. Whether a
    well-formed title is *too generic* is a probabilistic judgement left to the
    LLM-judge leg, never this lexical lint.
    """
    violations: list[str] = []
    # Strip fenced code first so a ``# install deps`` comment inside a code
    # block isn't mistaken for the page heading (mirrors ``check_atomicity``).
    prose = _FENCED_CODE.sub("", body)
    m = _H1_CAPTURE.search(prose)
    h1: str | None = m.group(1).strip() if m else None
    if not h1:
        violations.append("body has no usable `# Title` heading")
        # Reset an empty-after-strip capture (a whitespace-only ``#   `` heading)
        # to None so the title-drift leg below doesn't then re-report it as a
        # frontmatter-vs-empty mismatch.
        h1 = None
    elif not _TITLE_WORD.search(h1):
        violations.append(
            f"`# Title` heading {h1!r} is punctuation-only (no word characters)"
        )
    fm = (frontmatter_title or "").strip()
    if fm and h1 is not None and fm != h1:
        violations.append(f"frontmatter title {fm!r} != body heading {h1!r}")
    # Only flag the *degenerate* fallback: a title with no ASCII-sluggable
    # character (pure CJK, no pinyin slug) that collapsed to ``untitled``. A page
    # legitimately titled "Untitled" slugifies to the same literal but its title
    # yielded a real slug, so it must not be flagged — check the title's content,
    # not just the stem (slugify's output is identical for both cases).
    if stem == SLUG_FALLBACK and not _ASCII_SLUGGABLE.search(fm or (h1 or "")):
        violations.append(
            f"filename slug {stem!r} is the `untitled` fallback — the title "
            "produced no usable ASCII/pinyin slug"
        )
    return tuple(violations)


async def run_lint(
    storage: Storage, *, root: Path, fallback: str = "未分类"
) -> LintReport:
    """Scan K-layer pages and return a structured report.

    ``fallback`` is ``SchemaConfig.fallback`` — knowledge pages filed under it
    are flagged ``uncategorized`` so a human can re-file them into a declared
    category (closed-set taxonomy contract, ADR-0003).
    """
    issues: list[LintIssue] = []
    root_resolved = root.resolve()
    # path → set of LintKind the page opted out of via frontmatter.
    # Populated below in the same per-page frontmatter-load loop; consulted
    # both when appending per-page issues (broken_wikilink / non_atomic)
    # and when emitting orphan_page issues a second time later.
    suppressions: dict[str, set[LintKind]] = {}
    page_meta: dict[str, PageMeta] = {}

    # ``page_docs`` aggregates the K + W layers because PR3 promoted
    # wisdom to a first-class document layer: a wikilink from a wisdom
    # page to a knowledge page (and vice versa) must surface here just like
    # wiki↔wiki, and an orphan_page check on wiki must credit incoming
    # backlinks from wisdom so a legitimate wisdom citation doesn't
    # trigger destructive lint apply on the knowledge page. SYNTH context
    # builds (``api._synth_pages_from_source``) deliberately keep
    # Layer.KNOWLEDGE only — wisdom is hand-written and not LLM-authored.
    page_docs = list(
        await storage.list_documents(layer=Layer.KNOWLEDGE, active=True)
    ) + list(
        await storage.list_documents(layer=Layer.WISDOM, active=True)
    )

    # ``missing_file`` (D/K/W) — an *active* document row whose backing file is
    # gone from disk. Detected up front, across all three layers, so a vanished
    # page is surfaced ONLY as ``missing_file`` and excluded from every other
    # pass below: disk is authoritative (ADR-0005), the row is a stale
    # projection pending purge, and a gone page can't meaningfully be an
    # ``orphan_page`` / ``uncategorized`` / ``duplicate_title`` (those
    # remediations contradict "purge the row"). The file is gone, so there is
    # no frontmatter to carry a ``lint: {skip}`` annotation — ``missing_file``
    # is never suppressible. D-layer ``sources/`` rows are never page_docs (no
    # wikilink / atomicity / title concerns), so they only ever reach lint via
    # this detector — closing the original "delete a source file, its row is
    # stuck active forever" gap. The deterministic ``MissingFileFixer`` purges
    # each orphaned row + its outgoing edges (D5: inbound edges from live pages
    # stay as ``broken_wikilink``).
    source_docs = list(
        await storage.list_documents(layer=Layer.SOURCE, active=True)
    )
    missing_paths: set[str] = set()
    for doc in [*source_docs, *page_docs]:
        if not (root_resolved / doc.path).resolve().is_file():
            missing_paths.add(doc.path)
    # Emit in sorted-path order. ``list_documents`` has no ORDER BY, so without
    # this the issue order — and thus which rows survive ``lint propose``'s
    # ``--limit`` cap on a base with many orphans — would be DB-arbitrary and
    # differ across runs / adapters. (Repeated propose→apply still drains them
    # all; this just makes each pass deterministic.)
    for path in sorted(missing_paths):
        issues.append(
            LintIssue(
                kind="missing_file",
                path=path,
                detail=(
                    "active document row has no backing file on disk — "
                    "purge the orphaned row (the file was deleted outside dikw)"
                ),
            )
        )

    # Normalized keys of active SOURCE docs whose backing file is present on
    # disk (``missing_file`` above already determined existence — reuse it, no
    # extra stat). ``dangling_provenance`` consults this so a cited source whose
    # frontmatter spelling drifts from the on-disk spelling only by case /
    # Unicode form (on a case-sensitive filesystem) is still treated as present:
    # the engine's source identity is the normalized path key, the same key
    # ``read_provenance`` resolves through, so the two surfaces agree.
    present_source_keys = {
        normalize_path(d.path) for d in source_docs if d.path not in missing_paths
    }

    # Every other K/W check runs over the *live* page set — pages whose file is
    # gone are excluded so they don't double-surface as orphan/duplicate/etc.
    # Sorted by path so the per-doc kinds emitted inside the main loop
    # (``stale_index`` / ``broken_wikilink`` / ``missing_provenance`` / …) — and
    # the ``duplicate_title`` "primary" pick — are deterministic across runs and
    # adapters (``list_documents`` has no ORDER BY), so ``lint propose --limit``
    # is reproducible (matching the already-sorted ``missing_file`` /
    # ``untracked_file`` passes).
    live_page_docs = sorted(
        (d for d in page_docs if d.path not in missing_paths), key=lambda d: d.path
    )

    title_to_paths: dict[str, list[str]] = defaultdict(list)
    inbound: Counter[str] = Counter()

    for doc in live_page_docs:
        title = doc.title or Path(doc.path).stem
        title_to_paths[title].append(doc.path)

    # Share the same resolve semantics as engine persistence
    # (``persist_knowledge``): exact -> fuzzy normalize -> collision refusal.
    # Without this lint reports ``broken_wikilink`` on plurals that
    # storage already resolved, and silently swallows fuzzy collisions
    # that storage refused to guess.
    #
    # ``build_title_indexes`` is the single source of truth for the
    # cross-layer collision policy — exact-title collisions across
    # KNOWLEDGE + WISDOM are dropped from ``title_to_path`` so the fuzzy
    # stage's ≥2-candidate refusal fires (Karpathy's wrong-merge rule).
    # The previous local ``{t: dup_paths[0]}`` shape silently let the
    # first-iterated layer win, contradicting what ``persist_knowledge``
    # actually wrote into storage and causing ``broken_wikilink`` and
    # ``duplicate_title`` lint to disagree with the persisted graph.
    # Built from ``live_page_docs`` so a missing page's title neither resolves
    # referrers' ``[[wikilink]]``s (they correctly surface as broken until the
    # row is purged or the file restored) nor masks a live same-titled page.
    title_to_path, fuzzy_index = build_title_indexes(
        (doc.title or Path(doc.path).stem, doc.path) for doc in live_page_docs
    )

    for doc in live_page_docs:
        abs_path = (root / doc.path).resolve()
        try:
            post = frontmatter.load(str(abs_path))
        except Exception:
            # File vanished between the up-front missing scan and here (a rare
            # race), or unparseable frontmatter — skip; the next lint pass
            # reflects the settled state.
            continue
        body = post.content
        skip_kinds = _read_lint_skip(post.metadata)
        if skip_kinds:
            suppressions[doc.path] = skip_kinds
        # Stash the frontmatter slice we already paid to parse so
        # ``lint_propose`` can build ``KnowledgePageMeta`` without a second
        # disk pass over the same K-layer pages. ``frontmatter_str_list``
        # is the shared malformed-shape guard (scalar / dict / null →
        # ``[]``) — same one ``persist_knowledge`` and
        # ``MissingProvenanceFixer`` use, so this lint pass's view of
        # ``sources:`` matches what storage actually stored.
        sources_tuple = tuple(frontmatter_str_list(post.metadata, "sources"))
        tags_tuple = tuple(frontmatter_str_list(post.metadata, "tags"))
        page_meta[doc.path] = PageMeta(sources=sources_tuple, tags=tags_tuple)

        # ``stale_index`` — the on-disk body drifted from the indexed copy
        # (a hand-edit outside dikw). The body was already loaded above for
        # the lexical checks, so the hash comparison costs nothing extra —
        # no separate mtime-prefiltered hashing pass is needed (the per-page
        # read the checks below require already dominates the scan). Disk is
        # authoritative (ADR-0005): the row + its chunks / links are the
        # lagging projection; the reindex fixer re-projects the current bytes
        # without rewriting the file. Compare the body hash the way
        # ``persist`` stored it (frontmatter-stripped, markdown
        # ``content_hash``). Scope note: this is BODY drift only — a
        # frontmatter-only edit (e.g. a wisdom ``status:`` change with the body
        # untouched) does not change the body hash, so it is not flagged here
        # (a rare case; the row's ``status``/``title`` projection stays stale
        # until the body is next edited or the page re-written).
        if "stale_index" not in skip_kinds and content_hash(body) != doc.hash:
            issues.append(
                LintIssue(
                    kind="stale_index",
                    path=doc.path,
                    detail=(
                        "on-disk body differs from the indexed copy — "
                        "re-project the page (it was hand-edited outside dikw)"
                    ),
                )
            )

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
        #
        # The provenance edges feed two checks (``missing_provenance`` here
        # and ``dangling_provenance`` below), so load them once and share —
        # the storage probe runs whenever *either* kind is wanted, not just
        # ``missing_provenance``.
        want_missing_prov = "missing_provenance" not in skip_kinds
        want_dangling_prov = "dangling_provenance" not in skip_kinds
        if want_missing_prov or want_dangling_prov:
            existing_prov = await storage.provenance_from(doc.doc_id)
            if want_missing_prov:
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
            if want_dangling_prov:
                # ``dangling_provenance`` — a provenance edge whose target
                # source file is gone from disk. Disk is authoritative
                # (ADR-0005): a citation whose backing file no longer exists is
                # drift. Read-only — surfaced, never auto-repaired (no fixer;
                # the frontmatter is the user's to edit — ADR-0001 non-cascade).
                #
                # Drives off the provenance *table* (``existing_prov``), i.e.
                # the reconciled edge, not the raw frontmatter list — the kind
                # names a "provenance edge", consistent with
                # ``read_provenance``. A page whose frontmatter cites a gone
                # source but whose table is not yet reconciled (empty / stale)
                # surfaces as ``missing_provenance`` first; once reconciled, the
                # edge lands here. So the two kinds can co-fire (table edge gone
                # + frontmatter drift) without either masking the other.
                #
                # Checks the *file*, not the D ``documents`` row: a source
                # present on disk but not yet ``ingest``-ed (no active row) is
                # NOT dangling — the fix there is ingest, not editing
                # frontmatter, so a row-based ``resolved=False`` test would
                # false-fire. (This is deliberately disk-existence, not
                # ``read_provenance``'s SOURCE-row resolution — it stays correct
                # for bases with a non-default source dir, where requiring a
                # ``sources/`` prefix would wrongly flag every edge.)
                #
                # A source counts as present (NOT dangling) when EITHER a file
                # exists at its path on disk OR its normalized key matches an
                # active source doc with a live file (``present_source_keys``).
                # The direct stat catches a not-yet-ingested source (no row);
                # the key match catches case / Unicode-form drift between the
                # frontmatter spelling and the on-disk spelling on a
                # case-sensitive filesystem (where the raw stat would miss).
                #
                # The raw ``source_path`` is normalized to forward slashes
                # before the join + key so a hand-edited Windows-style
                # ``sources\foo`` entry resolves the same on every platform
                # (matching the ``untracked_file`` pass and D-layer ``doc.path``);
                # without it the same base would lint clean on Windows but dirty
                # on Linux. An edge whose normalized path escapes the base can
                # never resolve to an in-base source, so it's dangling and the
                # escaping target's content is never ``is_file``-stat-ed (the
                # containment check short-circuits). Sorted by normalized key so
                # a multi-source page emits deterministically (``lint propose
                # --limit`` reproducibility, matching the other passes).
                for e in sorted(
                    existing_prov, key=lambda edge: edge.source_path_key
                ):
                    rel_src = e.source_path.replace("\\", "/")
                    abs_src = (root_resolved / rel_src).resolve()
                    present = (
                        abs_src.is_relative_to(root_resolved) and abs_src.is_file()
                    ) or normalize_path(rel_src) in present_source_keys
                    if present:
                        continue
                    issues.append(
                        LintIssue(
                            kind="dangling_provenance",
                            path=doc.path,
                            detail=(
                                f"declared source {e.source_path!r} has no file "
                                "on disk — edit this page's `sources:` "
                                "frontmatter (the source was deleted outside "
                                "dikw)"
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
                        detail=f"{u.target_text} has no matching knowledge page",
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

        # title_slug_quality — K-layer only (wisdom is hand-written and may
        # legitimately carry a frontmatter title distinct from its body H1).
        # All three sub-cases are deterministic and never fire on correct synth
        # output, so the kind is safe to gate in ``synth --verify``.
        if doc.layer is Layer.KNOWLEDGE and "title_slug_quality" not in skip_kinds:
            raw_title = post.metadata.get("title")
            tsq = check_title_slug_quality(
                body=body,
                frontmatter_title=raw_title if isinstance(raw_title, str) else None,
                stem=Path(doc.path).stem,
            )
            if tsq:
                issues.append(
                    LintIssue(
                        kind="title_slug_quality",
                        path=doc.path,
                        detail="; ".join(tsq),
                    )
                )

        # accumulate inbound link counts (resolved links from storage)
        for stored in await storage.links_from(doc.doc_id):
            if stored.link_type is LinkType.WIKILINK:
                inbound[stored.dst_path] += 1

        # ``invalid_wisdom_status`` is wisdom-only — the parser already
        # collapsed unknown frontmatter values to ``None`` before they
        # hit storage, so we re-read the raw spelling from disk to
        # quote back to the user. Folded into the main loop so the
        # frontmatter parse + suppressions populate happens once per
        # wisdom page, not twice (the previous trailing wisdom-only
        # loop clobbered ``suppressions[doc.path]`` and doubled disk I/O).
        if doc.layer is Layer.WISDOM and "invalid_wisdom_status" not in skip_kinds:
            raw = post.metadata.get("status")
            if raw is not None and (
                not isinstance(raw, str) or raw not in _VALID_WISDOM_STATUS_VALUES
            ):
                issues.append(
                    LintIssue(
                        kind="invalid_wisdom_status",
                        path=doc.path,
                        detail=(
                            f"status: {raw!r} is not in "
                            f"{sorted(_VALID_WISDOM_STATUS_VALUES)}"
                        ),
                    )
                )

    # orphans — no inbound wikilinks. Both K and W layer pages are scanned;
    # dikw-core no longer generates ``knowledge/index.md`` / ``knowledge/log.md``
    # scaffolds (ADR-0004), so there are no built-in pages to exclude.
    # ``live_page_docs`` excludes ``missing_file`` rows, so a vanished page is
    # never also reported as an orphan.
    for doc in live_page_docs:
        if "orphan_page" in suppressions.get(doc.path, set()):
            continue
        if inbound[doc.path] == 0:
            issues.append(
                LintIssue(
                    kind="orphan_page",
                    path=doc.path,
                    detail="no inbound wikilinks from other K- or W-layer pages",
                )
            )

    # uncategorized — pages synth filed under the fallback bucket because it
    # couldn't place them in a declared category. ``category_from_path`` only
    # matches ``knowledge/`` paths, so wisdom pages never trip this.
    for doc in live_page_docs:
        if "uncategorized" in suppressions.get(doc.path, set()):
            continue
        if category_from_path(doc.path) == fallback:
            issues.append(
                LintIssue(
                    kind="uncategorized",
                    path=doc.path,
                    detail=(
                        f"page is in the fallback category {fallback!r} — "
                        "re-file it under a declared category"
                    ),
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

    # ``untracked_file`` — markdown files on disk under ``knowledge/`` /
    # ``wisdom/`` with no active document row (hand-written, or restored
    # outside dikw). Disk is authoritative (ADR-0005): the file is real
    # content, the missing row is the lagging side, so the reindex fixer
    # indexes it. Walk is K/W only (D-layer source discovery is owned by
    # ``ingest``) and roots at each layer dir, so the sibling ``trash/`` /
    # ``.dikw/`` / ``assets/`` trees are naturally outside its scope. Only
    # backend-supported extensions count, so ``.gitkeep`` placeholders and
    # stray non-markdown files never trip. Inactive-only rows (no active
    # projection) intentionally surface here too — the reindex fixer's
    # ``upsert_document`` re-activates them, healing a deactivated page.
    # Deliberately NOT suppressible via ``lint: {skip}`` (unlike ``stale_index``):
    # detection is read-free (stat + membership), and honouring per-file
    # frontmatter would force a parse of every candidate; a scratch ``.md`` the
    # user doesn't want indexed belongs outside the ``knowledge/``/``wisdom/``
    # trees. The fix is opt-in (propose/apply), so an un-applied issue is just a
    # warning.
    tracked = {normalize_path(d.path) for d in page_docs}
    exts = supported_extensions()
    for layer_name in ("knowledge", "wisdom"):
        layer_dir = root_resolved / layer_name
        if not layer_dir.is_dir():
            continue
        for abs_file in sorted(layer_dir.rglob("*")):
            if not abs_file.is_file() or abs_file.suffix.lower() not in exts:
                continue
            # In-tree symlink / junction whose target escapes the base —
            # skip (mirrors ``iter_source_files``); reads stay under the tree.
            if not abs_file.resolve().is_relative_to(root_resolved):
                continue
            rel = abs_file.relative_to(root_resolved)
            # Dot-prefixed component (``.obsidian/``, editor swap dirs) — not
            # part of the managed knowledge/wisdom content.
            if any(part.startswith(".") for part in rel.parts):
                continue
            rel_str = str(rel).replace("\\", "/")
            if normalize_path(rel_str) in tracked:
                continue
            issues.append(
                LintIssue(
                    kind="untracked_file",
                    path=rel_str,
                    detail=(
                        "markdown file on disk has no active document row — "
                        "index it (hand-written or restored outside dikw)"
                    ),
                )
            )

    return LintReport(
        issues=issues,
        acknowledged_leaves=sorted(suppressions),
        page_meta=page_meta,
    )
