"""Client-side markdown pre-flight inspection + asset reference extraction.

Lives at the package root (next to ``schemas.py``) so both the client
upload command and the D-layer ingest backend can import it without
violating the client-doesn't-depend-on-engine layering invariant.
Dependencies are stdlib + ``python-frontmatter`` + ``schemas.AssetRef``
only — anything heavier belongs in ``domains/data/``.

Two surfaces:

* ``extract_image_refs(body)`` — the canonical regex pair for both
  standard ``![alt](path)`` and Obsidian ``![[file|alias]]`` embeds.
  ``domains/data/backends/markdown.py`` re-exports this.
* ``inspect_markdown(path, *, project_root)`` — used by ``dikw client
  upload`` to pre-flight every md before packaging. Returns every
  reason ingest would warn or fail (`frontmatter_error`,
  `asset_missing`, `empty_body`) plus the resolved absolute paths of
  every local asset reference so the packager can include them in the
  same package.
"""

from __future__ import annotations

import hashlib
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import frontmatter

from .schemas import AssetRef

IssueKind = Literal[
    "frontmatter_error",
    "asset_missing",
    "asset_symlink",
    "empty_body",
    "encoding_error",
]


# ---- file hashing (shared with D layer + server commit + tests) --------

_SHA256_CHUNK = 1 << 20


def sha256_file(path: Path, *, chunk_size: int = _SHA256_CHUNK) -> str:
    """Streaming SHA-256 hex digest of ``path`` (O(chunk_size) RAM)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def package_sha256(md_sha: str, asset_shas: Iterable[str]) -> str:
    """``sha256( sorted([md_sha, *asset_shas]).join("\\n") )`` — the
    digest the upload manifest carries per package; client + server +
    tests must use this single implementation or a typo means the
    server rejects every package as ``manifest_package_sha256_mismatch``."""
    joined = "\n".join(sorted([md_sha, *asset_shas]))
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


# ---- canonical asset-reference regex (kept here so both client + D layer
# share one source of truth) ---------------------------------------------

# Standard markdown image: ![alt](path) with optional "title" attribute.
# Path may contain spaces (Obsidian-style ``![](My Diagram.png)``); the
# lookahead pins the lazy match at the position where the title or the
# closing paren begins, so the path captures everything in between
# without swallowing the title or trailing whitespace.
_IMG_MD = re.compile(
    r"!\[([^\]]*)\]\(\s*([^)\n]+?)"
    r"(?=\s+\"[^\"\n]*\"\s*\)|\s*\))"
    r"(?:\s+\"[^\"\n]*\")?\s*\)"
)

# Obsidian image embed: ![[file]] with optional |alias.
_IMG_WIKILINK = re.compile(r"!\[\[([^\]|]+?)(?:\|([^\]]+))?\]\]")


def extract_image_refs(body: str) -> list[AssetRef]:
    """Find every image embed in ``body`` and return them in source order.

    Both ``![alt](path)`` (with optional ``"title"``) and
    ``![[file|alias]]`` are recognised. ``start`` / ``end`` are
    character offsets covering the literal reference syntax so the
    chunker can treat each embed as an atomic span.

    Remote URLs are still emitted here; ``inspect_markdown`` and the
    ingest-time materialize layer decide what to do with them.
    """
    refs: list[AssetRef] = []
    for m in _IMG_MD.finditer(body):
        refs.append(
            AssetRef(
                original_path=m.group(2),
                alt=m.group(1) or "",
                start=m.start(),
                end=m.end(),
                syntax="markdown",
            )
        )
    for m in _IMG_WIKILINK.finditer(body):
        refs.append(
            AssetRef(
                original_path=m.group(1),
                alt=m.group(2) or "",
                start=m.start(),
                end=m.end(),
                syntax="wikilink",
            )
        )
    refs.sort(key=lambda r: r.start)
    return refs


def _is_remote(original_path: str) -> bool:
    """Treat anything with a non-empty scheme (http, https, ftp, data, …)
    as remote. Plain relative paths and bare filenames stay local.
    Mirror of ``domains/data/assets._is_remote`` so the pre-flight + the
    materialize step agree on the same boundary."""
    parsed = urlparse(original_path)
    return bool(parsed.scheme) and parsed.scheme not in ("file",)


def _resolve_local(
    original_path: str, *, source_md_path: Path, project_root: Path
) -> Path | None:
    """Sibling-of-md → project-root two-stage lookup.

    Mirrors ``domains/data/assets._resolve_local`` so the upload
    packager and ingest agree on which file an embed reference points
    at."""
    candidate = (source_md_path.parent / original_path).resolve()
    if candidate.is_file():
        return candidate
    candidate = (project_root / original_path).resolve()
    if candidate.is_file():
        return candidate
    return None


def _candidate_is_symlink(
    original_path: str, *, source_md_path: Path, project_root: Path
) -> bool:
    """``True`` iff the raw lookup path for ``original_path`` (under
    sibling-of-md or project-root) lstats to a symlink.

    The pre-flight needs this because ``_resolve_local`` runs
    ``Path.resolve`` and silently follows the symlink — the upload
    would then archive the target's bytes under the symlink's name,
    breaking the md's reference and potentially leaking files outside
    the upload root.
    """
    for candidate in (
        source_md_path.parent / original_path,
        project_root / original_path,
    ):
        try:
            cst = candidate.lstat()
        except OSError:
            continue
        if stat.S_ISLNK(cst.st_mode):
            return True
    return False


# ---- pre-flight inspection ---------------------------------------------


@dataclass(frozen=True)
class InspectionIssue:
    """One reason a markdown file isn't safe to ship."""

    kind: IssueKind
    message: str


@dataclass(frozen=True)
class InspectionResult:
    """The output of ``inspect_markdown``.

    ``asset_paths`` lists the **resolved absolute** paths of every
    local asset embedded by the file (deduplicated, in source order).
    The upload packager uses it directly as the asset_paths of the
    package built around this md.
    """

    file_path: Path
    ok: bool
    issues: list[InspectionIssue] = field(default_factory=list)
    asset_paths: list[Path] = field(default_factory=list)


def inspect_markdown(
    path: Path, *, project_root: Path
) -> InspectionResult:
    """Inspect a markdown file for ingest readiness.

    Checks performed (in order so the report reads naturally):

    1. ``frontmatter_error`` — YAML front-matter doesn't parse. Ingest
       would crash on this file with ``parse_error``.
    2. ``empty_body`` — body (after front-matter strip + whitespace
       strip) is empty. Ingest would record a zero-chunk source, which
       pollutes the source set.
    3. ``asset_missing`` — every embed is checked with the same
       sibling-of-md → project-root fallback ingest uses; unresolved
       refs (excluding remote URLs) are reported one by one.

    Multiple issues accumulate so users see the full picture in one
    upload attempt rather than fixing-and-retrying drip-fed errors.
    """
    issues: list[InspectionIssue] = []
    asset_paths: list[Path] = []

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        return InspectionResult(
            file_path=path,
            ok=False,
            issues=[
                InspectionIssue(
                    kind="encoding_error",
                    message=f"file is not valid UTF-8: {e}",
                )
            ],
        )
    except OSError as e:
        # Caller normally checks existence first; surface the real
        # error so the upload packager can attribute it cleanly.
        return InspectionResult(
            file_path=path,
            ok=False,
            issues=[
                InspectionIssue(
                    kind="frontmatter_error",
                    message=f"could not read {path}: {e}",
                )
            ],
        )

    # Layer 1: frontmatter parse.
    body: str
    try:
        post = frontmatter.loads(text)
        body = post.content
    except Exception as e:
        # ``frontmatter.loads`` raises ``yaml.YAMLError`` for malformed
        # YAML; any other exception in the wild gets the same treatment
        # because the user-visible problem is the same: ingest can't
        # parse this file.
        issues.append(
            InspectionIssue(
                kind="frontmatter_error",
                message=f"YAML front-matter parse failed: {e}",
            )
        )
        # Without a parsed body we can't extract assets — return early
        # so we don't double-report half-resolvable refs.
        return InspectionResult(file_path=path, ok=False, issues=issues)

    # Layer 2: empty body.
    if not body.strip():
        issues.append(
            InspectionIssue(
                kind="empty_body",
                message="body is empty after front-matter / whitespace strip",
            )
        )

    # Layer 3: asset references. Even when body is empty we still try
    # to extract refs — there are none, so the loop is a no-op.
    seen: set[Path] = set()
    for ref in extract_image_refs(body):
        if _is_remote(ref.original_path):
            continue
        if _candidate_is_symlink(
            ref.original_path, source_md_path=path, project_root=project_root
        ):
            issues.append(
                InspectionIssue(
                    kind="asset_symlink",
                    message=(
                        f"asset reference {ref.original_path!r} is a symlink; "
                        "copy the file in place if you really mean to include it"
                    ),
                )
            )
            continue
        resolved = _resolve_local(
            ref.original_path,
            source_md_path=path,
            project_root=project_root,
        )
        if resolved is None:
            issues.append(
                InspectionIssue(
                    kind="asset_missing",
                    message=(
                        f"asset reference {ref.original_path!r} not found "
                        f"(looked in {path.parent} and {project_root})"
                    ),
                )
            )
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        asset_paths.append(resolved)

    return InspectionResult(
        file_path=path,
        ok=not issues,
        issues=issues,
        asset_paths=asset_paths,
    )


__all__ = [
    "InspectionIssue",
    "InspectionResult",
    "IssueKind",
    "extract_image_refs",
    "inspect_markdown",
    "package_sha256",
    "sha256_file",
]
