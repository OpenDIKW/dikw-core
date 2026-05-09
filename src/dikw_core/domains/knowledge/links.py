"""Link graph for K (and W) layers.

Parses three kinds of links out of markdown bodies:

* ``[[Target]]`` / ``[[Target|alias]]`` / ``[[Target#anchor]]`` — Obsidian
  wikilinks. Target resolution tries exact title match first, then a
  fuzzy normalize (NFKC + casefold + punctuation strip + trailing-plural
  stem). When normalize maps the link to a key that resolves to **two or
  more** distinct pages, we refuse to guess and let the wikilink stay
  broken so ``dikw lint`` can surface the ambiguity.
* ``[text](relative/path.md)`` — standard Markdown links. URLs and
  fragment-only references are classified as ``url`` or dropped.
* Bare URLs in the body — captured as ``url`` links with no target
  resolution.

Parsing is deliberately forgiving: we want best-effort discovery, not a CSS
parser. The output is a list of ``LinkRecord``s ready for
``Storage.upsert_link`` and a list of ``UnresolvedLink``s that ``lint``
surfaces to the user.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ...schemas import LinkRecord, LinkType

_WIKILINK = re.compile(r"\[\[([^\]\|\n]+?)(?:\|([^\]\n]+?))?\]\]")
_MD_LINK = re.compile(r"(?<!\!)\[([^\]\n]+?)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_URL = re.compile(r"(?<![\[\(])\b(https?://[^\s\)]+)")

# Strip these from the *boundaries* of each whitespace-separated token,
# never from the interior. Internal punctuation is load-bearing for
# technical titles — ``C++``, ``C#``, ``.NET``, ``Node.js`` — and a
# greedy strip would collapse them onto bare ``c``/``net``/``node`` and
# fuzzy-resolve to the wrong page when the index has only one of the
# variants. Trailing-comma / trailing-period style fragmentation is
# what we actually want to absorb.
_BOUNDARY_PUNCT = set(
    ".,!?;:\"'()[]{}<>"          # ASCII sentence + clause separators
    "。、《》〈〉「」『』【】"     # CJK
    + "“”‘’"                     # noqa: RUF001 - intentional smart quotes
)


def _stem_plural(word: str) -> str:
    """Trailing-plural stemmer for ASCII English nouns.

    ASCII-scoped on purpose: CJK titles don't pluralize with ``s`` and
    stronger morphology (``children`` -> ``child``) carries unbounded
    false-merge risk that the deterministic resolve path can't take.
    Broken wikilinks are recoverable via ``dikw lint``; wrong-merges
    are not.
    """
    if len(word) <= 3 or not word.isascii() or not word.isalpha():
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    for suffix in ("ses", "ches", "shes", "xes", "zes"):
        if word.endswith(suffix):
            return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _strip_boundary_punct(token: str) -> str:
    """Drop boundary punctuation from both ends of a whitespace-bounded token.

    Internal characters are untouched — ``c++``, ``c#``, ``.net`` keep
    their distinguishing symbols and remain distinct fuzzy keys.
    """
    while token and token[0] in _BOUNDARY_PUNCT:
        token = token[1:]
    while token and token[-1] in _BOUNDARY_PUNCT:
        token = token[:-1]
    return token


def _normalize_for_match(s: str) -> str:
    """Collapse a title or wikilink target onto a fuzzy-matchable key.

    Word boundaries are preserved (``Neural Network`` never becomes
    ``neuralnetwork``). Returns ``""`` for input that reduces to all
    boundary punctuation; callers treat empty as "no key" so an
    all-symbol wikilink can't accidentally collide with an empty-keyed
    page.
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    tokens = [t for t in (_strip_boundary_punct(t) for t in s.split()) if t]
    if not tokens:
        return ""
    tokens[-1] = _stem_plural(tokens[-1])
    return " ".join(tokens)


def build_fuzzy_index(title_to_path: dict[str, str]) -> dict[str, list[str]]:
    """Precompute the normalize-keyed lookup ``resolve_links`` needs.

    Hoisting this out of ``resolve_links`` lets a synth caller building
    a single ``title_to_path`` for a batch of pages avoid rebuilding
    the fuzzy index on each per-page resolve call (Stage A 1:N fan-out
    persists tens-to-hundreds of pages per source against the same
    title set).
    """
    index: dict[str, list[str]] = {}
    for title, path in title_to_path.items():
        key = _normalize_for_match(title)
        if not key:
            continue
        bucket = index.setdefault(key, [])
        if path not in bucket:
            bucket.append(path)
    return index


@dataclass(frozen=True)
class UnresolvedLink:
    """A wikilink whose target could not be resolved to a known path."""

    src_doc_id: str
    target_text: str  # raw text between ``[[ ... ]]``
    line: int


@dataclass(frozen=True)
class ParsedLink:
    """Intermediate structure before storage-path resolution."""

    kind: LinkType
    target: str          # for wikilinks: page title; for md: href; for url: the URL
    anchor: str | None
    line: int
    raw: str             # the matched substring (debug aid)


def parse_links(body: str) -> list[ParsedLink]:
    """Return every link discovered in ``body`` in source order."""
    results: list[ParsedLink] = []

    # Build a line-offset index so each match can report a 1-based line number.
    line_starts = _line_starts(body)

    # Wikilinks first so wikilink substrings inside body aren't mis-read as md links
    # (wikilink syntax uses `[[` which wouldn't match `(...)` anyway, so order is
    # mostly cosmetic — but we still keep one source-ordered list).
    matches = sorted(
        [("wikilink", m) for m in _WIKILINK.finditer(body)]
        + [("md", m) for m in _MD_LINK.finditer(body)]
        + [("url", m) for m in _URL.finditer(body)],
        key=lambda item: item[1].start(),
    )

    for kind, m in matches:
        line = _offset_to_line(m.start(), line_starts)
        if kind == "wikilink":
            raw_target = m.group(1).strip()
            target, anchor = _split_anchor(raw_target)
            results.append(
                ParsedLink(
                    kind=LinkType.WIKILINK,
                    target=target,
                    anchor=anchor,
                    line=line,
                    raw=m.group(0),
                )
            )
        elif kind == "md":
            href = m.group(2).strip()
            # Skip mailto and other pseudo-schemes quietly.
            if href.startswith(("mailto:", "tel:", "#")):
                continue
            if href.startswith(("http://", "https://")):
                target, anchor = _split_anchor(href)
                results.append(
                    ParsedLink(
                        kind=LinkType.URL,
                        target=target,
                        anchor=anchor,
                        line=line,
                        raw=m.group(0),
                    )
                )
            else:
                target, anchor = _split_anchor(href)
                results.append(
                    ParsedLink(
                        kind=LinkType.MARKDOWN,
                        target=target,
                        anchor=anchor,
                        line=line,
                        raw=m.group(0),
                    )
                )
        else:  # url
            target, anchor = _split_anchor(m.group(1))
            results.append(
                ParsedLink(
                    kind=LinkType.URL,
                    target=target,
                    anchor=anchor,
                    line=line,
                    raw=m.group(0),
                )
            )

    return results


def resolve_links(
    src_doc_id: str,
    links: list[ParsedLink],
    *,
    title_to_path: dict[str, str],
    fuzzy_index: dict[str, list[str]] | None = None,
) -> tuple[list[LinkRecord], list[UnresolvedLink]]:
    """Turn ``ParsedLink``s into storage records.

    ``title_to_path`` maps a K/W page title to its wiki-relative path.
    Wikilink resolution falls through three deterministic stages:
    exact title match → fuzzy normalize match (NFKC + casefold +
    punctuation strip + ASCII trailing-plural stem) → collision refusal
    (two-or-more normalize-equivalent paths leave the link broken so
    ``dikw lint`` surfaces the ambiguity rather than letting us guess).

    ``fuzzy_index`` is the output of ``build_fuzzy_index(title_to_path)``;
    callers persisting many pages against the same title set should hoist
    that build to amortize it. ``None`` means "build it here", which is
    fine for tests + one-shot callers.
    """
    fuzzy_to_paths = fuzzy_index if fuzzy_index is not None else build_fuzzy_index(title_to_path)

    resolved: list[LinkRecord] = []
    unresolved: list[UnresolvedLink] = []

    for link in links:
        if link.kind is LinkType.WIKILINK:
            target_path: str | None = title_to_path.get(link.target)
            if target_path is None:
                key = _normalize_for_match(link.target)
                candidates = fuzzy_to_paths.get(key, []) if key else []
                if len(candidates) == 1:
                    target_path = candidates[0]
            if target_path is None:
                unresolved.append(
                    UnresolvedLink(
                        src_doc_id=src_doc_id, target_text=link.raw, line=link.line
                    )
                )
                continue
            resolved.append(
                LinkRecord(
                    src_doc_id=src_doc_id,
                    dst_path=target_path,
                    link_type=LinkType.WIKILINK,
                    anchor=link.anchor,
                    line=link.line,
                )
            )
        else:
            resolved.append(
                LinkRecord(
                    src_doc_id=src_doc_id,
                    dst_path=link.target,
                    link_type=link.kind,
                    anchor=link.anchor,
                    line=link.line,
                )
            )

    return resolved, unresolved


def _split_anchor(s: str) -> tuple[str, str | None]:
    if "#" not in s:
        return s, None
    head, _, anchor = s.partition("#")
    return head, anchor or None


def _line_starts(body: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(body):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _offset_to_line(offset: int, line_starts: list[int]) -> int:
    # binary search would be faster but we rarely have huge docs in K layer
    line = 1
    for start in line_starts:
        if start > offset:
            return line - 1
        line += 1
    return len(line_starts)
