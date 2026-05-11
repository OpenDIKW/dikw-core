"""HTTP-level tests for the packages-aware ``POST /v1/import``.

The post-refactor import endpoint:

* requires a ``packages`` field in the manifest (one per md + its asset refs);
* does **per-package** sha256 verification + per-package commit straight
  into ``<base>/sources/`` (no separate ingest step);
* returns ``{committed: [pkg_id...], rejected: [{id, code, detail}...]}``;
* clears the staging directory after the request returns regardless of
  per-package outcome.

Schema-level manifest errors (missing ``packages``, orphan files,
duplicate ``md_path``) still 4xx because they indicate a broken client
implementation, not a transient transport problem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ._import_helpers import (
    packages_manifest as _manifest_with_packages,
)
from ._import_helpers import (
    sha256 as _sha256,
)
from ._import_helpers import (
    tar_bytes as _tar_bytes,
)


def _post(
    client: httpx.AsyncClient,
    files: dict[str, bytes],
    manifest: dict[str, Any],
) -> Any:
    payload = _tar_bytes(files)
    return client.post(
        "/v1/import",
        files={"payload": ("u.tar.gz", payload, "application/gzip")},
        data={"manifest": json.dumps(manifest)},
    )


# ---- happy path ---------------------------------------------------------


@pytest.mark.asyncio
async def test_single_package_md_only_committed(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    files = {"sources/note.md": b"# n\nbody\n"}
    manifest = _manifest_with_packages(
        files,
        [{"md_path": "sources/note.md", "asset_paths": []}],
    )

    resp = await _post(server_client, files, manifest)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["committed"] == [0]
    assert body["rejected"] == []
    assert (wiki_root / "sources" / "note.md").read_bytes() == files[
        "sources/note.md"
    ]


@pytest.mark.asyncio
async def test_two_packages_share_one_asset(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    """A logo embedded by two notes: one tar entry, two packages
    referencing the same archive path."""
    files = {
        "sources/note1.md": b"# 1\n![](logo.png)\n",
        "sources/note2.md": b"# 2\n![](logo.png)\n",
        "sources/logo.png": b"\x89PNG\r\n\x1a\n",
    }
    manifest = _manifest_with_packages(
        files,
        [
            {"md_path": "sources/note1.md", "asset_paths": ["sources/logo.png"]},
            {"md_path": "sources/note2.md", "asset_paths": ["sources/logo.png"]},
        ],
    )

    resp = await _post(server_client, files, manifest)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert sorted(body["committed"]) == [0, 1]
    assert body["rejected"] == []
    assert (wiki_root / "sources" / "note1.md").exists()
    assert (wiki_root / "sources" / "note2.md").exists()
    assert (wiki_root / "sources" / "logo.png").exists()


@pytest.mark.asyncio
async def test_cross_directory_asset_archive_layout(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    """``![](../shared/logo.png)`` in ``sources/sub/note.md`` packs the
    asset at ``sources/shared/logo.png`` (the relative path resolved
    against project_root, stripped of ``..``)."""
    files = {
        "sources/sub/note.md": b"# n\n![](../shared/logo.png)\n",
        "sources/shared/logo.png": b"\x89PNG\r\n\x1a\n",
    }
    manifest = _manifest_with_packages(
        files,
        [
            {
                "md_path": "sources/sub/note.md",
                "asset_paths": ["sources/shared/logo.png"],
            }
        ],
    )

    resp = await _post(server_client, files, manifest)
    assert resp.status_code == 200, resp.text
    assert (wiki_root / "sources" / "sub" / "note.md").exists()
    assert (wiki_root / "sources" / "shared" / "logo.png").exists()


# ---- per-package failure (200 with rejected) ---------------------------


@pytest.mark.asyncio
async def test_package_sha256_mismatch_rejects_only_that_package(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    """Tampering one package's ``package_sha256`` rejects that package
    alone; the other packages commit normally. Status is 200 because
    the manifest schema is well-formed."""
    files = {
        "sources/a.md": b"# a\n",
        "sources/b.md": b"# b\n",
    }
    manifest = _manifest_with_packages(
        files,
        [
            {"md_path": "sources/a.md", "asset_paths": []},
            {"md_path": "sources/b.md", "asset_paths": []},
        ],
    )
    manifest["packages"][1]["package_sha256"] = "0" * 64

    resp = await _post(server_client, files, manifest)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["committed"] == [0]
    assert len(body["rejected"]) == 1
    rej = body["rejected"][0]
    assert rej["id"] == 1
    assert rej["code"] == "manifest_package_sha256_mismatch"
    # a.md committed; b.md absent
    assert (wiki_root / "sources" / "a.md").exists()
    assert not (wiki_root / "sources" / "b.md").exists()


@pytest.mark.asyncio
async def test_file_sha256_mismatch_rejects_packages_referencing_it(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    """A bad file-level sha rejects every package that references that
    file (the asset shared by both packages → both rejected)."""
    files = {
        "sources/a.md": b"# a\n![](shared.png)\n",
        "sources/b.md": b"# b\n![](shared.png)\n",
        "sources/shared.png": b"\x89PNG\r\n\x1a\n",
    }
    manifest = _manifest_with_packages(
        files,
        [
            {"md_path": "sources/a.md", "asset_paths": ["sources/shared.png"]},
            {"md_path": "sources/b.md", "asset_paths": ["sources/shared.png"]},
        ],
    )
    # Tamper the shared asset's claimed sha.
    for entry in manifest["files"]:
        if entry["path"] == "sources/shared.png":
            entry["sha256"] = "0" * 64

    resp = await _post(server_client, files, manifest)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["committed"] == []
    assert sorted(r["id"] for r in body["rejected"]) == [0, 1]
    for r in body["rejected"]:
        assert r["code"] == "manifest_sha256_mismatch"
    assert not (wiki_root / "sources" / "a.md").exists()
    assert not (wiki_root / "sources" / "b.md").exists()


# ---- schema-level rejection (4xx) --------------------------------------


@pytest.mark.asyncio
async def test_packages_field_missing_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    """Old manifest shape (no ``packages``) is no longer accepted."""
    files = {"sources/x.md": b"# x\n"}
    manifest: dict[str, Any] = {
        "files": [
            {
                "path": "sources/x.md",
                "size": len(files["sources/x.md"]),
                "sha256": _sha256(files["sources/x.md"]),
            }
        ],
        "total_bytes": len(files["sources/x.md"]),
    }
    resp = await _post(server_client, files, manifest)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "manifest_packages_missing"


@pytest.mark.asyncio
async def test_orphan_file_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    """A file in the tar but not referenced by any package's md_path
    or asset_paths is an orphan; reject the whole import so the client
    fixes the bug instead of silently dropping bytes."""
    files = {
        "sources/note.md": b"# n\n",
        "sources/orphan.png": b"\x89PNG\r\n\x1a\n",
    }
    manifest = _manifest_with_packages(
        files,
        [{"md_path": "sources/note.md", "asset_paths": []}],
    )

    resp = await _post(server_client, files, manifest)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "manifest_orphan_file"
    assert "sources/orphan.png" in str(body["error"])


@pytest.mark.asyncio
async def test_duplicate_md_path_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    """Same md_path in two packages = bug."""
    files = {"sources/note.md": b"# n\n"}
    manifest = _manifest_with_packages(
        files,
        [
            {"md_path": "sources/note.md", "asset_paths": []},
            {"md_path": "sources/note.md", "asset_paths": []},
        ],
    )
    resp = await _post(server_client, files, manifest)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "manifest_duplicate_md_path"


@pytest.mark.asyncio
async def test_package_md_path_not_in_files_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    """A package referencing an md_path that isn't in the files list
    is also a manifest bug — the file would never make it onto disk."""
    files = {"sources/a.md": b"# a\n"}
    manifest = _manifest_with_packages(
        files,
        [{"md_path": "sources/a.md", "asset_paths": []}],
    )
    # Inject a package referring to a non-existent md.
    manifest["packages"].append(
        {
            "id": 1,
            "md_path": "sources/ghost.md",
            "asset_paths": [],
            "package_sha256": "0" * 64,
        }
    )

    resp = await _post(server_client, files, manifest)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "manifest_package_unknown_file"


# ---- staging hygiene ---------------------------------------------------


@pytest.mark.asyncio
async def test_staging_dir_cleared_after_success(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    files = {"sources/note.md": b"# n\n"}
    manifest = _manifest_with_packages(
        files,
        [{"md_path": "sources/note.md", "asset_paths": []}],
    )
    resp = await _post(server_client, files, manifest)
    assert resp.status_code == 200, resp.text

    staging_root = wiki_root / ".dikw" / "staging"
    # Either the parent dir doesn't exist, or it's empty — depends on
    # whether the server lazily creates it. Both states satisfy "no
    # leftover staging from this import."
    if staging_root.exists():
        assert list(staging_root.iterdir()) == []


@pytest.mark.asyncio
async def test_staging_dir_cleared_after_per_package_reject(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    """Even when some packages are rejected (200 with rejected list),
    the staging tree must be cleaned up — staging is an implementation
    detail, not user-visible."""
    files = {
        "sources/a.md": b"# a\n",
        "sources/b.md": b"# b\n",
    }
    manifest = _manifest_with_packages(
        files,
        [
            {"md_path": "sources/a.md", "asset_paths": []},
            {"md_path": "sources/b.md", "asset_paths": []},
        ],
    )
    manifest["packages"][1]["package_sha256"] = "0" * 64

    resp = await _post(server_client, files, manifest)
    assert resp.status_code == 200, resp.text

    staging_root = wiki_root / ".dikw" / "staging"
    if staging_root.exists():
        assert list(staging_root.iterdir()) == []


@pytest.mark.asyncio
async def test_response_does_not_leak_staging_path(
    server_client: httpx.AsyncClient,
) -> None:
    """``staging_path`` field is dead in the new model — staging is an
    implementation detail. Old client code reading the field should get
    a missing key, not an outdated path."""
    files = {"sources/note.md": b"# n\n"}
    manifest = _manifest_with_packages(
        files,
        [{"md_path": "sources/note.md", "asset_paths": []}],
    )
    resp = await _post(server_client, files, manifest)
    assert resp.status_code == 200, resp.text
    assert "staging_path" not in resp.json()
