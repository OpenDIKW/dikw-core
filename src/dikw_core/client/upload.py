"""Local md+assets → tar.gz + per-md packages manifest.

The post-refactor client packages each markdown file together with its
referenced assets as a single logical "package" inside one tar.gz +
manifest. The server's ``POST /v1/upload/sources`` validates each
package independently, commits the well-formed ones into
``<base>/sources/`` directly, and reports per-package outcomes.

Wire shape::

    {
      "files":    [{"path": "sources/...", "size": ..., "sha256": ...}, ...],
      "packages": [
        {"id": 0, "md_path": "sources/note.md",
         "asset_paths": ["sources/diagram.png"],
         "package_sha256": "..."},
        ...
      ],
      "total_bytes": ...
    }

Pre-flight inspection (frontmatter parse, missing asset, empty body,
orphan asset, symlink) raises ``UploadError`` before any bytes are
tarred so a broken input fails locally — no network round trip.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import os
import stat
import tarfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import IO

from ..md_inspect import (
    InspectionResult,
    inspect_markdown,
    package_sha256,
    sha256_file,
)

# Spooled buffer threshold — 16 MiB matches the plan's recommendation.
# Below it, the tarball stays in RAM (zero disk I/O); above it, the
# spooled file rolls to a real tempfile so we don't OOM on big asset
# bundles.
_SPOOL_MAX_SIZE = 16 * 1024 * 1024

# Only ``.md`` is accepted for upload — the default ``dikw.yml``
# scans ``sources/**/*.md`` and ``.markdown`` files would commit but
# silently never get ingested. Users needing ``.markdown`` should
# rename + edit their config glob explicitly.
_DEFAULT_MD_EXTENSIONS = frozenset({".md"})
_DEFAULT_ASSET_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".pdf"}
)


@dataclass(frozen=True)
class ManifestEntry:
    """One file's metadata. Mirrors the server's ManifestEntry shape."""

    path: str  # POSIX, relative to the tarball root, e.g. ``sources/foo.md``
    size: int
    sha256: str  # lowercase hex


@dataclass(frozen=True)
class PackageEntry:
    """One md + its asset references, with the digest a server can recompute."""

    id: int
    md_path: str
    asset_paths: list[str]
    package_sha256: str


@dataclass
class UploadBundle:
    """A ready-to-send tar.gz + its manifest.

    ``payload`` is a file-like positioned at byte 0; the caller hands
    it to the transport. Owned by this dataclass — close it via
    :meth:`close` (or use as a context manager) when done.
    """

    payload: IO[bytes]
    manifest_json: str
    files_count: int
    bytes: int

    def close(self) -> None:
        # Tempfile cleanup races with delete-on-close on Windows; don't
        # let a swallowed cleanup error mask the real upload error.
        with contextlib.suppress(Exception):
            self.payload.close()

    def __enter__(self) -> UploadBundle:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class UploadError(Exception):
    """Surface client-side rejection (bad inputs, pre-flight lint, …)
    before any bytes leave the machine."""


def build_upload(src: Path) -> UploadBundle:
    """Pack ``src`` into a tar.gz + manifest the server will accept.

    ``src`` may be either a single ``.md`` file or a directory whose
    ``**/*.md`` tree becomes one package per file. Pre-flight
    inspection runs on every md (frontmatter parse, asset existence,
    non-empty body); orphan assets in the input tree are rejected so
    files don't get silently dropped.

    Raises :class:`UploadError` for any pre-flight failure with
    enough detail (file path, lint kind, missing ref) for the user to
    fix.
    """
    project_root, md_files, asset_files = _resolve_input(src)

    # Pre-flight inspection: collect packages or accumulate failures.
    packages: list[_PendingPackage] = []
    pre_flight_errors: list[str] = []
    for md_path in md_files:
        if md_path.is_symlink():
            pre_flight_errors.append(
                f"{md_path}: symlink (target: {os.readlink(md_path)!r}); "
                "copy the file in place if you really mean to include it"
            )
            continue
        result = inspect_markdown(md_path, project_root=project_root)
        if not result.ok:
            for issue in result.issues:
                pre_flight_errors.append(
                    f"{md_path}: {issue.kind}: {issue.message}"
                )
            continue
        packages.append(_pending_from_inspection(result, project_root))

    # Orphan asset check: any allowed-extension asset under the project
    # root that no md references. Skip it loudly so the user resolves
    # their intent (delete it, or reference it).
    if asset_files:
        all_referenced: set[Path] = {
            abs_path for pkg in packages for _, abs_path in pkg.assets
        }
        for asset in asset_files:
            if asset in all_referenced:
                continue
            pre_flight_errors.append(
                f"{asset}: orphan asset (not referenced by any md); "
                "remove it or embed it in a markdown file"
            )

    if pre_flight_errors:
        raise UploadError(
            "pre-flight inspection failed:\n  - "
            + "\n  - ".join(pre_flight_errors)
        )

    return _build_bundle(packages)


# ---- internals ---------------------------------------------------------


@dataclass(frozen=True)
class _PendingPackage:
    """In-progress package being assembled before the tarball write.

    ``assets`` pairs each asset's archive path (POSIX, ``sources/...``)
    with its on-disk absolute path — coupled in a single sequence so
    the two can never drift apart."""

    md_archive: str
    md_abs: Path
    assets: list[tuple[str, Path]]


def _resolve_input(
    src: Path,
) -> tuple[Path, list[Path], list[Path]]:
    """Inspect ``src`` once and return ``(project_root, md_files, asset_files)``.

    The symlink check has to run on the **user-supplied** path, not on
    the resolved one — ``Path.resolve`` strips the symlink for us, so
    a follow-up ``S_ISLNK`` would always say "regular file" and the
    pre-flight rejection promised in the docstring would silently
    leak the target's bytes into the upload.
    """
    try:
        st_raw = src.lstat()
    except OSError as e:
        raise UploadError(f"upload source does not exist: {src}") from e

    if stat.S_ISLNK(st_raw.st_mode):
        raise UploadError(
            f"refusing to upload symlink: {src} "
            f"(target: {os.readlink(src)!r})"
        )

    src = src.resolve()
    st = src.stat()

    if stat.S_ISREG(st.st_mode):
        if src.suffix.lower() not in _DEFAULT_MD_EXTENSIONS:
            raise UploadError(
                f"single-file upload only accepts markdown "
                f"({sorted(_DEFAULT_MD_EXTENSIONS)}); got {src.suffix!r}"
            )
        return src.parent, [src], []

    if stat.S_ISDIR(st.st_mode):
        # If the user points at a base-style tree (already contains a
        # ``sources/`` subdir), descend into it and treat ``src`` as the
        # project root. Without this, scanning + archiving from ``src``
        # would prepend a second ``sources/`` and commit md as
        # ``sources/sources/note.md``.
        scan_root = src / "sources" if (src / "sources").is_dir() else src
        md_files, asset_files = _discover_files(scan_root)
        if not md_files:
            raise UploadError(
                f"no markdown files found under {scan_root} "
                f"(expected ``**/*.md``)"
            )
        return src, md_files, asset_files

    raise UploadError(
        f"upload source is neither a file nor a directory: {src}"
    )


def _pending_from_inspection(
    result: InspectionResult, project_root: Path
) -> _PendingPackage:
    """Project the inspection result onto archive paths.

    Both md and assets get re-rooted under ``sources/`` preserving
    their relative position inside ``project_root``. Assets that
    resolve outside the project root (via ``_resolve_local``'s
    project-root fallback when the relative path escapes) are
    rejected — the upload root must self-contain.

    When the user pointed at a base-style tree, files already begin
    with ``sources/`` after relativising; ``_archive_path`` keeps a
    single prefix in either case.
    """
    md_archive = _archive_path(result.file_path, project_root)
    assets: list[tuple[str, Path]] = []
    for asset_abs in result.asset_paths:
        try:
            archive = _archive_path(asset_abs, project_root)
        except ValueError as e:
            raise UploadError(
                f"{result.file_path}: asset {asset_abs} resolves outside "
                f"the upload root ({project_root}); move the asset under "
                f"the upload root or invoke upload from a higher directory"
            ) from e
        assets.append((archive, asset_abs))
    return _PendingPackage(
        md_archive=md_archive,
        md_abs=result.file_path,
        assets=assets,
    )


def _archive_path(abs_path: Path, project_root: Path) -> str:
    """Compute the in-archive path for a file rooted at ``project_root``.

    ``project_root`` may itself sit one level above an existing
    ``sources/`` subtree (the base-style layout); in that case the
    relative path already begins with ``sources/`` and we don't add
    another prefix. Otherwise the relative path is assumed to live
    under the implicit ``sources/`` root and gets prefixed."""
    rel = abs_path.relative_to(project_root).as_posix()
    if rel == "sources" or rel.startswith("sources/"):
        return rel
    return "sources/" + rel


def _build_bundle(packages: list[_PendingPackage]) -> UploadBundle:
    """Pack the validated packages into a tar.gz + manifest.

    Files (md + asset) are deduped by archive path: the same logo
    referenced by two notes appears once in the archive, with both
    packages listing it in ``asset_paths``.
    """
    # Collect unique files preserving first-seen order (sorted later
    # so the manifest has a stable shape).
    abs_by_archive: dict[str, Path] = {}
    for pkg in packages:
        abs_by_archive.setdefault(pkg.md_archive, pkg.md_abs)
        for archive_path, abs_path in pkg.assets:
            abs_by_archive.setdefault(archive_path, abs_path)

    # Hash + stat every unique file once. ``stat`` is one syscall for
    # both size and mtime; sha256 streams the bytes.
    hash_by_archive: dict[str, tuple[str, int, float]] = {}
    for archive_path, abs_path in abs_by_archive.items():
        st = abs_path.stat()
        sha = sha256_file(abs_path)
        hash_by_archive[archive_path] = (sha, st.st_size, st.st_mtime)

    payload = SpooledTemporaryFile(  # noqa: SIM115 - returned to caller
        max_size=_SPOOL_MAX_SIZE, mode="w+b"
    )
    manifest_files: list[ManifestEntry] = []
    total_bytes = 0
    try:
        with gzip.GzipFile(fileobj=payload, mode="wb") as gz, tarfile.TarFile(
            fileobj=gz, mode="w"
        ) as tf:
            for archive_path in sorted(abs_by_archive):
                abs_path = abs_by_archive[archive_path]
                sha, size, mtime = hash_by_archive[archive_path]
                tarinfo = tarfile.TarInfo(name=archive_path)
                tarinfo.size = size
                tarinfo.mtime = int(mtime)
                tarinfo.mode = 0o644
                # Strip uid/gid/uname/gname so the archive is byte-stable
                # across users running the same upload — useful when CI
                # and a developer both upload the same tree.
                tarinfo.uid = 0
                tarinfo.gid = 0
                tarinfo.uname = ""
                tarinfo.gname = ""
                with abs_path.open("rb") as fh:
                    tf.addfile(tarinfo, fh)
                manifest_files.append(
                    ManifestEntry(path=archive_path, size=size, sha256=sha)
                )
                total_bytes += size
    except Exception:
        payload.close()
        raise

    payload.seek(0)
    pkg_entries: list[PackageEntry] = []
    for idx, pkg in enumerate(packages):
        md_sha = hash_by_archive[pkg.md_archive][0]
        asset_archives = [archive_path for archive_path, _ in pkg.assets]
        asset_shas = [hash_by_archive[ap][0] for ap in asset_archives]
        pkg_entries.append(
            PackageEntry(
                id=idx,
                md_path=pkg.md_archive,
                asset_paths=asset_archives,
                package_sha256=package_sha256(md_sha, asset_shas),
            )
        )
    manifest_json = json.dumps(
        {
            "files": [e.__dict__ for e in manifest_files],
            "packages": [e.__dict__ for e in pkg_entries],
            "total_bytes": total_bytes,
        }
    )
    return UploadBundle(
        payload=payload,
        manifest_json=manifest_json,
        files_count=len(manifest_files),
        bytes=total_bytes,
    )


def _discover_files(
    root: Path,
) -> tuple[list[Path], list[Path]]:
    """Walk ``root`` once, bucket entries into (md_files, asset_files).

    Hidden dirs (``.git/``, ``.dikw/``) and hidden files are skipped.
    Asset paths are resolved to absolute form so the orphan check
    matches what ``inspect_markdown`` returns."""
    md_files: list[Path] = []
    asset_files: list[Path] = []
    for parent, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.startswith("."):
                continue
            suffix = Path(name).suffix.lower()
            full = Path(parent) / name
            if suffix in _DEFAULT_MD_EXTENSIONS:
                md_files.append(full)
            elif suffix in _DEFAULT_ASSET_EXTENSIONS:
                asset_files.append(full.resolve())
    return md_files, asset_files


__all__ = [
    "ManifestEntry",
    "PackageEntry",
    "UploadBundle",
    "UploadError",
    "build_upload",
]
