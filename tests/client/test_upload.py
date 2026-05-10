"""``build_upload`` packaging tests (post-refactor: per-md packages).

These tests cover the new "one md = one package" packaging semantics
exposed by ``build_upload``. Each invocation walks the input (file
*or* directory), runs ``inspect_markdown`` on every md to learn its
asset references, and emits a tar.gz + manifest whose ``packages``
field carries the per-md grouping.

Pre-flight failures (frontmatter parse, missing asset, empty body,
orphan asset) raise ``UploadError`` before any bytes are tarred so the
caller sees the problem before kicking off a network round trip.

The CLI-level happy / sad path is in ``tests/client/test_upload_cli.py``;
this file isolates the packaging logic so a broken manifest builder
fails here, not deep in an HTTP test.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from dikw_core.client.upload import UploadError, build_upload


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _package_sha256(md_sha: str, asset_shas: list[str]) -> str:
    joined = "\n".join(sorted([md_sha, *asset_shas]))
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


# ---- happy path: packaging shape --------------------------------------


def test_build_single_md_file(tmp_path: Path) -> None:
    """``build_upload(<one.md>)`` packs that file as a single package."""
    note = tmp_path / "alpha.md"
    body = b"# A\nbody\n"
    # Use write_bytes so Windows doesn't translate \n → \r\n; sha256
    # must match what the server hashes off the wire.
    note.write_bytes(body)

    bundle = build_upload(note)
    try:
        manifest = json.loads(bundle.manifest_json)
        assert [e["path"] for e in manifest["files"]] == ["sources/alpha.md"]
        assert len(manifest["packages"]) == 1
        pkg = manifest["packages"][0]
        assert pkg["md_path"] == "sources/alpha.md"
        assert pkg["asset_paths"] == []
        # package_sha256 = sha256(md_sha) for asset-less packages.
        assert pkg["package_sha256"] == _package_sha256(_sha256(body), [])
    finally:
        bundle.close()


def test_build_directory_with_sibling_asset(tmp_path: Path) -> None:
    """A note that embeds ``diagram.png`` ships both files in one
    package; archive layout preserves the sibling-of-md convention."""
    src = tmp_path / "inbox"
    _write(src / "note.md", "# n\n![](diagram.png)\n")
    asset_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    (src / "diagram.png").write_bytes(asset_bytes)

    bundle = build_upload(src)
    try:
        manifest = json.loads(bundle.manifest_json)
        paths = sorted(e["path"] for e in manifest["files"])
        assert paths == ["sources/diagram.png", "sources/note.md"]
        assert len(manifest["packages"]) == 1
        pkg = manifest["packages"][0]
        assert pkg["md_path"] == "sources/note.md"
        assert pkg["asset_paths"] == ["sources/diagram.png"]
        # The actual tarball matches the manifest entries.
        with tarfile.open(fileobj=bundle.payload, mode="r:gz") as tf:
            tar_paths = sorted(m.name for m in tf.getmembers() if m.isfile())
            assert tar_paths == paths
    finally:
        bundle.close()


def test_build_two_mds_share_one_logo(tmp_path: Path) -> None:
    """Two notes embed the same logo: tar contains one logo entry,
    both packages reference it via ``asset_paths``."""
    src = tmp_path / "inbox"
    _write(src / "a.md", "# A\n![](logo.png)\n")
    _write(src / "b.md", "# B\n![](logo.png)\n")
    (src / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    bundle = build_upload(src)
    try:
        manifest = json.loads(bundle.manifest_json)
        paths = sorted(e["path"] for e in manifest["files"])
        assert paths == ["sources/a.md", "sources/b.md", "sources/logo.png"]
        # tar must dedupe the asset entry — duplicate paths are illegal in tar.
        with tarfile.open(fileobj=bundle.payload, mode="r:gz") as tf:
            tar_logo_count = sum(
                1 for m in tf.getmembers() if m.name == "sources/logo.png"
            )
            assert tar_logo_count == 1
        assert len(manifest["packages"]) == 2
        for pkg in manifest["packages"]:
            assert pkg["asset_paths"] == ["sources/logo.png"]
    finally:
        bundle.close()


def test_build_cross_directory_asset(tmp_path: Path) -> None:
    """``![](../shared/logo.png)`` from ``sources/sub/note.md`` resolves
    via project_root fallback; archive path normalises to
    ``sources/shared/logo.png``. The md body is **not** rewritten so
    Obsidian-style relative links keep working in the user's editor."""
    src = tmp_path / "inbox"
    _write(src / "sub" / "note.md", "# n\n![](../shared/logo.png)\n")
    (src / "shared").mkdir()
    (src / "shared" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    bundle = build_upload(src)
    try:
        manifest = json.loads(bundle.manifest_json)
        paths = sorted(e["path"] for e in manifest["files"])
        assert paths == [
            "sources/shared/logo.png",
            "sources/sub/note.md",
        ]
        # md body was not rewritten — the relative link survives.
        with tarfile.open(fileobj=bundle.payload, mode="r:gz") as tf:
            md_member = tf.extractfile("sources/sub/note.md")
            assert md_member is not None
            assert b"![](../shared/logo.png)" in md_member.read()
    finally:
        bundle.close()


# ---- pre-flight rejection ---------------------------------------------


def test_build_rejects_orphan_asset(tmp_path: Path) -> None:
    """A png in the input dir not referenced by any md: refuse to ship.
    Silently dropping it would surprise the user."""
    src = tmp_path / "inbox"
    _write(src / "note.md", "# n\nbody only\n")
    (src / "stray.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(UploadError, match=r"orphan|stray\.png"):
        build_upload(src)


def test_build_rejects_pre_flight_frontmatter_error(tmp_path: Path) -> None:
    src = tmp_path / "inbox"
    _write(src / "bad.md", "---\nfoo: : bar\n---\n# x\nbody\n")

    with pytest.raises(UploadError, match=r"frontmatter"):
        build_upload(src)


def test_build_rejects_pre_flight_asset_missing(tmp_path: Path) -> None:
    src = tmp_path / "inbox"
    _write(src / "note.md", "# x\n![](ghost.png)\n")

    with pytest.raises(UploadError, match=r"ghost\.png|asset"):
        build_upload(src)


def test_build_rejects_pre_flight_empty_body(tmp_path: Path) -> None:
    src = tmp_path / "inbox"
    _write(src / "empty.md", "---\ntitle: x\n---\n   \n")

    with pytest.raises(UploadError, match=r"empty"):
        build_upload(src)


def test_build_rejects_no_md_in_dir(tmp_path: Path) -> None:
    """An input dir with zero md files is a usage error."""
    src = tmp_path / "empty-dir"
    src.mkdir()

    with pytest.raises(UploadError, match=r"no .* md|no markdown|no files"):
        build_upload(src)


def test_build_rejects_input_path_not_found(tmp_path: Path) -> None:
    bogus = tmp_path / "nope"
    with pytest.raises(UploadError):
        build_upload(bogus)


def test_build_rejects_non_md_file_input(tmp_path: Path) -> None:
    """Single-file mode only accepts md (or extra_extensions); a bare
    .png input doesn't make sense."""
    asset = tmp_path / "logo.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(UploadError):
        build_upload(asset)


# ---- security ---------------------------------------------------------


def test_symlink_md_is_rejected(tmp_path: Path) -> None:
    """A symlinked md could be made to read /etc/passwd through the
    upload pipeline — the client refuses before the file leaves the
    machine."""
    src = tmp_path / "inbox"
    src.mkdir()
    real_file = tmp_path / "outside.md"
    real_file.write_text("# leaked\nbody\n", encoding="utf-8")
    link = src / "note.md"
    try:
        link.symlink_to(real_file)
    except (OSError, NotImplementedError) as e:  # pragma: no cover - Windows
        pytest.skip(f"symlinks unavailable in this environment: {e}")

    with pytest.raises(UploadError, match=r"symlink"):
        build_upload(src)


def test_symlink_asset_is_rejected(tmp_path: Path) -> None:
    """An md that resolves an asset reference through a symlink also
    leaks; pre-flight inspection refuses to follow it."""
    src = tmp_path / "inbox"
    src.mkdir()
    _write(src / "note.md", "# x\n![](leaked.png)\n")
    real_asset = tmp_path / "outside.png"
    real_asset.write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        (src / "leaked.png").symlink_to(real_asset)
    except (OSError, NotImplementedError) as e:  # pragma: no cover - Windows
        pytest.skip(f"symlinks unavailable in this environment: {e}")

    with pytest.raises(UploadError, match=r"symlink"):
        build_upload(src)


# ---- file integrity ---------------------------------------------------


def test_manifest_file_sha256_matches_disk(tmp_path: Path) -> None:
    """Every entry's sha256 in the ``files`` list matches the byte-for-byte
    content on disk so the server's ``manifest_sha256`` check passes."""
    src = tmp_path / "inbox"
    _write(src / "note.md", "# n\n![](logo.png)\n")
    asset_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 4
    (src / "logo.png").write_bytes(asset_bytes)

    bundle = build_upload(src)
    try:
        manifest = json.loads(bundle.manifest_json)
        for entry in manifest["files"]:
            top, rel = entry["path"].split("/", 1)
            on_disk = (src / rel).read_bytes()
            assert entry["sha256"] == _sha256(on_disk)
            assert entry["size"] == len(on_disk)
            assert top == "sources"
    finally:
        bundle.close()


def test_manifest_package_sha256_matches_formula(tmp_path: Path) -> None:
    """``package_sha256 = sha256(sorted([md_sha, *asset_shas]).join("\\n"))``;
    a server reading the manifest will recompute it the same way."""
    src = tmp_path / "inbox"
    src.mkdir(parents=True, exist_ok=True)
    md_body = b"# n\n![](logo.png)\n"
    asset_body = b"\x89PNG\r\n\x1a\n"
    # write_bytes — Windows write_text would translate \n to \r\n.
    (src / "note.md").write_bytes(md_body)
    (src / "logo.png").write_bytes(asset_body)

    bundle = build_upload(src)
    try:
        manifest = json.loads(bundle.manifest_json)
        pkg = manifest["packages"][0]
        expected = _package_sha256(_sha256(md_body), [_sha256(asset_body)])
        assert pkg["package_sha256"] == expected
    finally:
        bundle.close()
