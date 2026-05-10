"""Shared fixtures for the server-side upload-route tests.

Both ``test_upload.py`` (tar shape + size limits + manifest schema)
and ``test_upload_packages.py`` (per-package commit semantics) need
the same primitives — packing a dict of ``{path: bytes}`` into a
gzipped tarball, computing per-file sha256, and assembling the
packages-aware manifest. Putting them here keeps the two suites in
sync: when the wire format moves, only this module changes.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from typing import Any


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_sha256(md_sha: str, asset_shas: list[str]) -> str:
    """Mirror of :func:`dikw_core.md_inspect.package_sha256` — kept
    here in test scope so a typo on either side surfaces as a test
    failure, not a runtime bug."""
    joined = "\n".join(sorted([md_sha, *asset_shas]))
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def tar_bytes(files: dict[str, bytes]) -> bytes:
    """Pack ``files`` into a gzipped tarball. Member metadata stays
    minimal (no uid/gid stamping) — the server's tar safety check
    operates on member.name only."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path, body in files.items():
            ti = tarfile.TarInfo(path)
            ti.size = len(body)
            tf.addfile(ti, io.BytesIO(body))
    return buf.getvalue()


def packages_manifest(
    files: dict[str, bytes],
    packages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the new manifest shape from raw file bytes + a list of
    ``{md_path, asset_paths}`` dicts. Computes ``package_sha256``
    and assigns ``id`` positionally."""
    files_entries = [
        {"path": p, "size": len(b), "sha256": sha256(b)}
        for p, b in files.items()
    ]
    sha_for = {p: sha256(b) for p, b in files.items()}
    pkg_entries: list[dict[str, Any]] = []
    for idx, pkg in enumerate(packages):
        md_sha = sha_for[pkg["md_path"]]
        asset_shas = [sha_for[ap] for ap in pkg.get("asset_paths", [])]
        pkg_entries.append(
            {
                "id": idx,
                "md_path": pkg["md_path"],
                "asset_paths": pkg.get("asset_paths", []),
                "package_sha256": package_sha256(md_sha, asset_shas),
            }
        )
    return {
        "files": files_entries,
        "packages": pkg_entries,
        "total_bytes": sum(len(b) for b in files.values()),
    }
