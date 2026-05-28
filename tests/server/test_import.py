"""HTTP-level tests for ``POST /v1/import`` — infrastructure layer.

This file covers the wire-format invariants that hold regardless of
the per-package commit model:

* tar.gz shape (path traversal, symlinks, allowed top dirs)
* manifest schema (json well-formed, files-vs-tarball mismatch)
* size limits

Per-package semantics (``packages``, ``package_sha256``, per-package
reject, commit-to-sources) live in ``test_import_packages.py``.
"""

from __future__ import annotations

import io
import json
import tarfile
from typing import Any

import httpx
import pytest

from ._import_helpers import packages_manifest, sha256, tar_bytes

# ---- happy path --------------------------------------------------------


@pytest.mark.asyncio
async def test_import_commits_into_sources(
    server_client: httpx.AsyncClient,
    base_root: Any,
) -> None:
    """Single-package import commits the md straight into
    ``<base>/sources/`` and returns committed=[0]."""
    files = {"sources/notes/x.md": b"# x\nhello\n"}
    manifest = packages_manifest(
        files,
        [{"md_path": "sources/notes/x.md", "asset_paths": []}],
    )
    payload = tar_bytes(files)
    resp = await server_client.post(
        "/v1/import",
        files={"payload": ("u.tar.gz", payload, "application/gzip")},
        data={"manifest": json.dumps(manifest)},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["committed"] == [0]
    assert body["rejected"] == []
    assert (base_root / "sources" / "notes" / "x.md").read_bytes() == files[
        "sources/notes/x.md"
    ]


# ---- manifest validation -----------------------------------------------


@pytest.mark.asyncio
async def test_manifest_missing_file_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    """A manifest entry pointing to a path that isn't in the tar fails
    schema validation — no per-package machinery can recover from it."""
    files = {"sources/x.md": b"hello"}
    manifest = packages_manifest(
        files, [{"md_path": "sources/x.md", "asset_paths": []}]
    )
    manifest["files"].append(
        {"path": "sources/ghost.md", "size": 5, "sha256": "0" * 64}
    )
    payload = tar_bytes(files)
    resp = await server_client.post(
        "/v1/import",
        files={"payload": ("u.tar.gz", payload, "application/gzip")},
        data={"manifest": json.dumps(manifest)},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "manifest_missing_files"


@pytest.mark.asyncio
async def test_manifest_extra_file_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    """A file in the tar but not in the manifest's ``files`` list is a
    schema-level error (every tar member must be declared)."""
    files = {"sources/x.md": b"hello", "sources/y.md": b"world"}
    manifest = packages_manifest(
        {"sources/x.md": files["sources/x.md"]},
        [{"md_path": "sources/x.md", "asset_paths": []}],
    )
    payload = tar_bytes(files)
    resp = await server_client.post(
        "/v1/import",
        files={"payload": ("u.tar.gz", payload, "application/gzip")},
        data={"manifest": json.dumps(manifest)},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "manifest_extra_files"


@pytest.mark.asyncio
async def test_manifest_invalid_json_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    payload = tar_bytes({"sources/x.md": b"hello"})
    resp = await server_client.post(
        "/v1/import",
        files={"payload": ("u.tar.gz", payload, "application/gzip")},
        data={"manifest": "{not json"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "manifest_malformed"


# ---- tarball shape validation -------------------------------------------


@pytest.mark.asyncio
async def test_tar_path_traversal_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    """A literal ``../escape.md`` member must never write outside the
    staging root. Build the tar by hand because tarfile's high-level
    API normalises leading separators — we want the literal escape."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        ti = tarfile.TarInfo("../escape.md")
        body = b"nope"
        ti.size = len(body)
        tf.addfile(ti, io.BytesIO(body))
    payload = buf.getvalue()
    files = {"../escape.md": b"nope"}
    manifest = {
        "files": [
            {
                "path": "../escape.md",
                "size": 4,
                "sha256": sha256(b"nope"),
            }
        ],
        "packages": [],
        "total_bytes": 4,
    }
    _ = files
    resp = await server_client.post(
        "/v1/import",
        files={"payload": ("u.tar.gz", payload, "application/gzip")},
        data={"manifest": json.dumps(manifest)},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "tar_path_traversal"


@pytest.mark.asyncio
async def test_tar_outside_allowed_top_dirs_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    """``sources/`` is the only allowed top-level dir in the new
    packages model (assets get co-located under ``sources/`` to
    preserve sibling-of-md asset resolution)."""
    files = {"knowledge/index.md": b"# index"}
    payload = tar_bytes(files)
    manifest = {
        "files": [
            {
                "path": "knowledge/index.md",
                "size": len(files["knowledge/index.md"]),
                "sha256": sha256(files["knowledge/index.md"]),
            }
        ],
        "packages": [],
        "total_bytes": len(files["knowledge/index.md"]),
    }
    resp = await server_client.post(
        "/v1/import",
        files={"payload": ("u.tar.gz", payload, "application/gzip")},
        data={"manifest": json.dumps(manifest)},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "tar_unexpected_path"


@pytest.mark.asyncio
async def test_tar_with_symlink_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        ti = tarfile.TarInfo("sources/link.md")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/passwd"
        tf.addfile(ti)
    payload = buf.getvalue()
    resp = await server_client.post(
        "/v1/import",
        files={"payload": ("u.tar.gz", payload, "application/gzip")},
        data={
            "manifest": json.dumps(
                {"files": [], "packages": [], "total_bytes": 0}
            )
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "tar_link_forbidden"


@pytest.mark.asyncio
async def test_import_too_large_rejected(
    server_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DIKW_SERVER_MAX_IMPORT_BYTES", "256")
    big = b"a" * 4096
    files = {"sources/big.md": big}
    manifest = packages_manifest(
        files, [{"md_path": "sources/big.md", "asset_paths": []}]
    )
    payload = tar_bytes(files)
    resp = await server_client.post(
        "/v1/import",
        files={"payload": ("u.tar.gz", payload, "application/gzip")},
        data={"manifest": json.dumps(manifest)},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "import_too_large"
