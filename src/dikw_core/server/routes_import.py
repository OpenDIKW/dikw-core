"""Source import endpoint (multipart packages upload + per-package commit).

Wire format (``POST /v1/import``)::

    Content-Type: multipart/form-data
        payload  — tar.gz, every member's path must start with ``sources/``
        manifest — JSON ``{"files": [...], "packages": [...], "total_bytes": N}``

The server unpacks into a per-request staging directory under
``<base>/.dikw/staging/<import_id>/``, validates the manifest schema,
recomputes each file's sha256 and each package's ``package_sha256``,
then commits the well-formed packages straight into ``<base>/sources/``
(per-package via ``os.replace``) and ``rmtree``s the staging directory
before returning. Per-package failures (sha mismatch, commit error)
are reported in ``ImportResponse.rejected``; schema-level failures
(orphan file, missing packages, duplicate md_path) reject the whole
request.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import BaseModel, Field

from ..md_inspect import package_sha256, sha256_file
from ._time import isoformat_utc_ms
from .errors import BadRequest
from .runtime import ServerRuntime, get_runtime

logger = logging.getLogger(__name__)


# ``DIKW_SERVER_MAX_IMPORT_BYTES`` overrides the default at process start.
# 1 GiB is generous for an md+assets bundle but avoids the worst-case
# wedge on a runaway tarball.
_DEFAULT_MAX_IMPORT_BYTES = 1 * 1024 * 1024 * 1024
_ALLOWED_TOP_DIRS = ("sources",)

STAGING_DIRNAME = ".dikw/staging"


# ---- request / response models ------------------------------------------


class ManifestEntry(BaseModel):
    path: str
    size: int = Field(ge=0)
    sha256: str  # lowercase hex


class PackageEntry(BaseModel):
    id: int = Field(ge=0)
    md_path: str
    asset_paths: list[str] = Field(default_factory=list)
    package_sha256: str


class Manifest(BaseModel):
    files: list[ManifestEntry]
    packages: list[PackageEntry]
    total_bytes: int = Field(ge=0)


class RejectedPackage(BaseModel):
    id: int
    code: str
    detail: dict[str, Any] | None = None


class ImportResponse(BaseModel):
    import_id: str
    files_count: int
    bytes: int
    applied_at: str  # ISO8601 UTC
    committed: list[int]
    rejected: list[RejectedPackage]


# ---- router -------------------------------------------------------------


def make_router(*, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/v1", dependencies=[Depends(auth_dep)])

    @router.post("/import", response_model=ImportResponse)
    async def import_sources(
        request: Request,
        payload: UploadFile = File(..., description="tar.gz of sources/"),
        manifest: str = Form(..., description="JSON Manifest body."),
    ) -> ImportResponse:
        rt: ServerRuntime = get_runtime(request.app)
        max_bytes = _resolve_max_bytes()

        # Cheap content-length pre-check; the per-write counter inside
        # ``_save_payload`` is the authoritative limit.
        cl = request.headers.get("content-length")
        if cl is not None and int(cl) > max_bytes:
            raise BadRequest(
                f"import payload exceeds {max_bytes} bytes (content-length={cl})",
                code="import_too_large",
                detail={"max_bytes": max_bytes},
            )

        manifest_obj = _parse_manifest(manifest)

        import_id = uuid.uuid4().hex[:12]
        staging_root = rt.root / STAGING_DIRNAME / import_id
        staging_root.parent.mkdir(parents=True, exist_ok=True)
        staging_root.mkdir(parents=True, exist_ok=False)

        try:
            tarball_path = staging_root / "_payload.tar.gz"
            written = await _save_payload(payload, tarball_path, max_bytes)
            _extract_safely(tarball_path, staging_root)
            os.unlink(tarball_path)
            file_rejects = _verify_manifest(staging_root, manifest_obj)
            # Mutating ``<base>/sources/`` must serialize with ingest
            # (which scans that tree) and with other concurrent imports
            # (which would otherwise race on the same archive paths).
            # ``ingest_lock`` is the runtime-wide single-writer guard.
            async with rt.ingest_lock:
                committed, rejected = _commit_packages(
                    staging_root,
                    manifest_obj,
                    base_root=rt.root,
                    file_rejects=file_rejects,
                )
        finally:
            # staging is purely an internal staging area — clean it up
            # whether the request succeeds or fails so we never leak
            # disk. Per-package commit errors are already attributed
            # to ``rejected`` above.
            shutil.rmtree(staging_root, ignore_errors=True)

        return ImportResponse(
            import_id=import_id,
            files_count=len(manifest_obj.files),
            bytes=written,
            applied_at=isoformat_utc_ms(),
            committed=committed,
            rejected=rejected,
        )

    return router


# ---- helpers ------------------------------------------------------------


def _resolve_max_bytes() -> int:
    raw = os.environ.get("DIKW_SERVER_MAX_IMPORT_BYTES")
    if not raw:
        return _DEFAULT_MAX_IMPORT_BYTES
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MAX_IMPORT_BYTES


def _parse_manifest(raw: str) -> Manifest:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise BadRequest(
            f"manifest is not valid JSON: {e}", code="manifest_malformed"
        ) from e
    if not isinstance(data, dict) or "packages" not in data:
        raise BadRequest(
            "manifest must include a ``packages`` field; the legacy "
            "files-only shape is no longer supported",
            code="manifest_packages_missing",
        )
    try:
        return Manifest.model_validate(data)
    except Exception as e:
        raise BadRequest(
            f"manifest schema mismatch: {e}", code="manifest_invalid"
        ) from e


async def _save_payload(
    payload: UploadFile, dest: Path, max_bytes: int
) -> int:
    total = 0
    chunk_size = 1024 * 1024  # 1 MiB
    with dest.open("wb") as fh:
        while True:
            chunk = await payload.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise BadRequest(
                    f"import payload exceeds {max_bytes} bytes",
                    code="import_too_large",
                    detail={"max_bytes": max_bytes},
                )
            fh.write(chunk)
    return total


def _extract_safely(tarball: Path, target: Path) -> None:
    """Extract a tar.gz, rejecting anything that escapes ``target``.

    Two layers of defence:
      * member.name path-traversal check (``..`` segments, absolute
        paths) — common ``tarbomb`` defence.
      * post-resolve ``commonpath`` check — guards against symlink
        members and platform-specific edge cases (``\\`` on Windows).
    """
    with tarfile.open(tarball, mode="r:gz") as tf:
        members = tf.getmembers()
        for m in members:
            _check_member(m, target)
        # Python 3.12+ ships a ``filter='data'`` helper; we use it as a
        # belt to ``_check_member``'s suspenders.
        tf.extractall(target, filter="data")


def _check_member(member: tarfile.TarInfo, target: Path) -> None:
    name = member.name
    if not name:
        raise BadRequest(
            "tarball contains an empty path", code="tar_invalid"
        )
    if member.issym() or member.islnk():
        raise BadRequest(
            f"tarball symlinks/hardlinks are not allowed: {name!r}",
            code="tar_link_forbidden",
        )
    # Reject Windows-style separators so a Windows-built tar can't ship
    # ``..\\..\\etc\\foo`` and slip past POSIX path-resolution.
    if "\\" in name:
        raise BadRequest(
            f"tarball path uses backslashes: {name!r}",
            code="tar_invalid",
        )
    parts = Path(name).parts
    if any(p == ".." for p in parts) or Path(name).is_absolute():
        raise BadRequest(
            f"tarball path escapes root: {name!r}",
            code="tar_path_traversal",
        )
    top = parts[0] if parts else ""
    if member.isdir():
        return
    if top not in _ALLOWED_TOP_DIRS:
        raise BadRequest(
            f"tarball file outside sources/: {name!r}",
            code="tar_unexpected_path",
            detail={"allowed": list(_ALLOWED_TOP_DIRS)},
        )
    # Final resolved-path check.
    resolved = (target / name).resolve()
    base = target.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as e:
        raise BadRequest(
            f"tarball path escapes root after resolve: {name!r}",
            code="tar_path_traversal",
        ) from e


def _verify_manifest(
    staging_root: Path, manifest: Manifest
) -> dict[str, list[int]]:
    """Schema + sha256 verification.

    Returns a mapping ``{archive_path: [package_id, ...]}`` of files
    whose sha256 didn't match the on-disk content. Callers reject every
    package referencing those files.

    Schema-level errors raise :class:`BadRequest` directly so the
    request fails before any commit attempt.
    """
    declared = {entry.path: entry for entry in manifest.files}

    # On-disk inventory.
    on_disk: dict[str, Path] = {}
    for path in staging_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "_payload.tar.gz":
            continue
        rel = path.relative_to(staging_root).as_posix()
        on_disk[rel] = path

    missing = sorted(declared.keys() - on_disk.keys())
    if missing:
        raise BadRequest(
            f"manifest declares files not present in tarball: {missing}",
            code="manifest_missing_files",
            detail={"missing": missing[:10]},
        )
    extra = sorted(on_disk.keys() - declared.keys())
    if extra:
        raise BadRequest(
            f"tarball contains files not in manifest: {extra}",
            code="manifest_extra_files",
            detail={"extra": extra[:10]},
        )

    # Cross-reference: every package's md_path / asset_paths must
    # appear in declared.
    for pkg in manifest.packages:
        unknown = [
            p for p in [pkg.md_path, *pkg.asset_paths] if p not in declared
        ]
        if unknown:
            raise BadRequest(
                f"package {pkg.id} references files not in manifest: {unknown}",
                code="manifest_package_unknown_file",
                detail={"package_id": pkg.id, "unknown": unknown[:10]},
            )

    # Duplicate package id — must run before any per-pkg report keying
    # off id. Without this, two packages can share id=0 and the
    # response's ``committed`` / ``rejected`` lists become impossible
    # for the client to map back to its original packages.
    seen_id: set[int] = set()
    for pkg in manifest.packages:
        if pkg.id in seen_id:
            raise BadRequest(
                f"package id {pkg.id} appears in more than one package",
                code="manifest_duplicate_package_id",
                detail={"package_id": pkg.id},
            )
        seen_id.add(pkg.id)

    # Duplicate md_path.
    seen_md: dict[str, int] = {}
    for pkg in manifest.packages:
        if pkg.md_path in seen_md:
            raise BadRequest(
                f"md_path {pkg.md_path!r} appears in package "
                f"{seen_md[pkg.md_path]} and {pkg.id}",
                code="manifest_duplicate_md_path",
                detail={"md_path": pkg.md_path},
            )
        seen_md[pkg.md_path] = pkg.id

    # Orphan files: every declared file must belong to some package.
    referenced: set[str] = set()
    for pkg in manifest.packages:
        referenced.add(pkg.md_path)
        referenced.update(pkg.asset_paths)
    orphans = sorted(declared.keys() - referenced)
    if orphans:
        raise BadRequest(
            f"manifest contains files not referenced by any package: {orphans}",
            code="manifest_orphan_file",
            detail={"orphans": orphans[:10]},
        )

    # File-level sha verification → returns map of bad files →
    # affected package ids (per-package reject downstream).
    bad_files: dict[str, list[int]] = {}
    for rel, abs_path in on_disk.items():
        entry = declared[rel]
        if abs_path.stat().st_size != entry.size:
            bad_files[rel] = []
            continue
        h = sha256_file(abs_path)
        if h != entry.sha256.lower():
            bad_files[rel] = []
    # Backfill which packages each bad file belongs to.
    for pkg in manifest.packages:
        for f in (pkg.md_path, *pkg.asset_paths):
            if f in bad_files:
                bad_files[f].append(pkg.id)
    return bad_files


def _commit_packages(
    staging_root: Path,
    manifest: Manifest,
    *,
    base_root: Path,
    file_rejects: dict[str, list[int]],
) -> tuple[list[int], list[RejectedPackage]]:
    """Per-package commit.

    For each package not flagged by ``file_rejects`` and whose
    recomputed ``package_sha256`` matches, ``os.replace`` moves the
    md + assets into ``<base_root>/sources/`` preserving the staging
    layout (``sources/<rel-path>``).

    Returns (committed_ids, rejected_entries).
    """
    # Aggregate file-reject affected packages.
    package_rejected_for_file: dict[int, str] = {}
    for path, pkg_ids in file_rejects.items():
        for pid in pkg_ids:
            package_rejected_for_file[pid] = path

    file_sha: dict[str, str] = {e.path: e.sha256.lower() for e in manifest.files}

    committed: list[int] = []
    rejected: list[RejectedPackage] = []
    # Files (esp. assets) shared across packages get moved once: the
    # first package to commit takes the file out of staging, subsequent
    # packages just skip it. That's safe because the second package
    # references the *same* archive path which is now already in the
    # knowledge tree.
    already_moved: set[str] = set()
    for pkg in manifest.packages:
        if pkg.id in package_rejected_for_file:
            rejected.append(
                RejectedPackage(
                    id=pkg.id,
                    code="manifest_sha256_mismatch",
                    detail={"file": package_rejected_for_file[pkg.id]},
                )
            )
            continue

        # Recompute package_sha256 from declared file shas.
        md_sha = file_sha[pkg.md_path]
        asset_shas = [file_sha[ap] for ap in pkg.asset_paths]
        recomputed = package_sha256(md_sha, asset_shas)
        if recomputed != pkg.package_sha256.lower():
            rejected.append(
                RejectedPackage(
                    id=pkg.id,
                    code="manifest_package_sha256_mismatch",
                    detail={
                        "expected": recomputed,
                        "declared": pkg.package_sha256,
                    },
                )
            )
            continue

        # Move files into the knowledge tree. ``os.replace`` is atomic
        # per-file but the package as a whole isn't — a mid-package
        # failure would leave a half-applied package on disk and let
        # ingest see a md without its assets. We honour the per-
        # package contract two ways:
        #   * net-new files get unlinked on failure (see _rollback);
        #   * pre-existing files get backed up before overwrite, so
        #     a failure can restore the original byte-identical copy.
        moved: list[_MovedFile] = []
        try:
            for archive_path in (pkg.md_path, *pkg.asset_paths):
                if archive_path in already_moved:
                    continue
                moved.append(
                    _commit_one_file(staging_root, archive_path, base_root)
                )
                already_moved.add(archive_path)
        except OSError as e:
            _rollback_moved_files(moved, already_moved)
            rejected.append(
                RejectedPackage(
                    id=pkg.id,
                    code="package_commit_failed",
                    detail={"error": str(e)},
                )
            )
            continue
        # Success — drop backups (kept until now in case a later
        # file in the same package failed and we needed to restore).
        for entry in moved:
            entry.discard_backup()

        committed.append(pkg.id)

    return committed, rejected


@dataclass
class _MovedFile:
    """One file successfully ``os.replace``d into the knowledge tree, plus the
    backup that lets us undo the overwrite if a later file in the same
    package fails."""

    archive_path: str
    dst: Path
    backup: Path | None  # ``None`` if dst didn't exist before the commit

    def discard_backup(self) -> None:
        if self.backup is None:
            return
        with contextlib.suppress(OSError):
            self.backup.unlink()
        self.backup = None

    def restore(self) -> None:
        """Best-effort undo: restore backup over dst, or unlink dst if
        it was net-new."""
        if self.backup is not None:
            with contextlib.suppress(OSError):
                os.replace(self.backup, self.dst)
            self.backup = None
        else:
            with contextlib.suppress(OSError):
                self.dst.unlink()


def _rollback_moved_files(
    moved: list[_MovedFile], already_moved: set[str]
) -> None:
    """Undo a partial per-package commit, in reverse order.

    Pre-existing destinations are restored from their per-file backup;
    net-new destinations are unlinked. Reverse iteration matters when
    a single package targets the same path multiple times (it can't,
    given manifest dedup, but the order is the safe choice anyway)."""
    for entry in reversed(moved):
        entry.restore()
        already_moved.discard(entry.archive_path)


_BACKUP_SUFFIX = ".bak.import"


def _commit_one_file(
    staging_root: Path, archive_path: str, base_root: Path
) -> _MovedFile:
    """Move ``staging_root/<archive_path>`` to ``base_root/<archive_path>``,
    backing up any pre-existing destination so a sibling file's later
    failure can restore the byte-identical original.

    ``os.replace`` is atomic per-file: same-filesystem rename is
    metadata-only, and the staging tree lives under ``<base>/.dikw/``
    so the default layout always satisfies that constraint.
    """
    src = staging_root / archive_path
    dst = base_root / archive_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if dst.exists():
        # Use ``os.replace`` (rename) for the backup, not copy: it's
        # atomic and avoids reading the original bytes. The backup
        # path lives next to dst so it stays on the same filesystem.
        backup = dst.with_name(dst.name + _BACKUP_SUFFIX)
        os.replace(dst, backup)
    try:
        os.replace(src, dst)
    except OSError:
        # Couldn't put the new file in place — restore the backup
        # so dst is still in its pre-commit state, then re-raise
        # so the package-level handler sees the failure.
        if backup is not None:
            with contextlib.suppress(OSError):
                os.replace(backup, dst)
        raise
    return _MovedFile(archive_path=archive_path, dst=dst, backup=backup)


__all__ = [
    "STAGING_DIRNAME",
    "ImportResponse",
    "Manifest",
    "ManifestEntry",
    "PackageEntry",
    "RejectedPackage",
    "make_router",
]
