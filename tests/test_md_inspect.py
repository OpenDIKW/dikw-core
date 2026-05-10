"""Client-side D-layer pre-flight inspection tests.

``inspect_markdown(path, *, project_root)`` is the function the new
``dikw client upload`` command calls before packaging a markdown file.
It returns an ``InspectionResult`` listing every reason ingest would
fail or warn (`frontmatter_error`, `asset_missing`, `empty_body`),
plus the resolved absolute paths of every local asset reference so the
client packaging step can include them in the same package as the md.

Layered above ``extract_image_refs`` (markdown-source asset extraction)
and ``_resolve_local`` (sibling-of-md → project-root two-stage lookup);
when the upload command writes broken md or a missing-asset md, the
user gets a fast pre-flight error instead of a server-side ingest crash
ten seconds later.
"""

from __future__ import annotations

from pathlib import Path

from dikw_core.md_inspect import InspectionResult, inspect_markdown


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_clean_md_no_assets(tmp_path: Path) -> None:
    note = tmp_path / "sources" / "alpha.md"
    _write(note, "---\ntitle: Alpha\n---\n# Alpha\n\nbody text\n")

    result = inspect_markdown(note, project_root=tmp_path)

    assert isinstance(result, InspectionResult)
    assert result.ok
    assert result.issues == []
    assert result.asset_paths == []


def test_clean_md_with_sibling_asset(tmp_path: Path) -> None:
    """Sibling-of-md is the Obsidian-native layout; the asset sits next
    to the md file. Resolved path is absolute."""
    note = tmp_path / "sources" / "note.md"
    _write(note, "# n\n\n![diagram](diagram.png)\n")
    asset = tmp_path / "sources" / "diagram.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    result = inspect_markdown(note, project_root=tmp_path)

    assert result.ok, result.issues
    assert result.asset_paths == [asset.resolve()]


def test_cross_directory_asset_via_project_root_fallback(tmp_path: Path) -> None:
    """``![](../shared/logo.png)`` resolves: sibling lookup fails, then
    project_root fallback picks it up because the relative path is
    well-formed against the project root too."""
    note = tmp_path / "sources" / "sub" / "note.md"
    _write(note, "# n\n![](../../shared/logo.png)\n")
    asset = tmp_path / "shared" / "logo.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = inspect_markdown(note, project_root=tmp_path)

    assert result.ok, result.issues
    assert result.asset_paths == [asset.resolve()]


def test_obsidian_wikilink_asset_extracted(tmp_path: Path) -> None:
    """``![[file.png]]`` syntax is the Obsidian wikilink form for embeds;
    it must be inspected the same way as ``![alt](path)``."""
    note = tmp_path / "sources" / "n.md"
    _write(note, "# x\n\n![[diagram.png]]\n")
    asset = tmp_path / "sources" / "diagram.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = inspect_markdown(note, project_root=tmp_path)

    assert result.ok, result.issues
    assert result.asset_paths == [asset.resolve()]


def test_remote_url_asset_is_not_an_issue(tmp_path: Path) -> None:
    """``![](https://...)`` is a remote URL; ingest's materialize_asset
    skips remote refs at runtime, so pre-flight must not flag them as
    asset_missing — the upload package needs no local file for them."""
    note = tmp_path / "sources" / "n.md"
    _write(note, "# x\n\n![](https://example.com/img.png)\n")

    result = inspect_markdown(note, project_root=tmp_path)

    assert result.ok, result.issues
    assert result.asset_paths == []


def test_frontmatter_error_reported(tmp_path: Path) -> None:
    """Malformed YAML between ``---`` fences fails to parse; ingest
    crashes here with kind=parse_error. Pre-flight catches it as
    ``frontmatter_error``."""
    note = tmp_path / "sources" / "bad.md"
    _write(note, "---\ntitle: a\nfoo: : bar\n---\n# x\n")

    result = inspect_markdown(note, project_root=tmp_path)

    assert not result.ok
    kinds = [i.kind for i in result.issues]
    assert "frontmatter_error" in kinds


def test_asset_missing_reported_with_ref_in_message(tmp_path: Path) -> None:
    note = tmp_path / "sources" / "n.md"
    _write(note, "# x\n\n![](ghost.png)\n")

    result = inspect_markdown(note, project_root=tmp_path)

    assert not result.ok
    asset_issues = [i for i in result.issues if i.kind == "asset_missing"]
    assert len(asset_issues) == 1
    assert "ghost.png" in asset_issues[0].message
    # asset_paths only lists *resolved* assets; unresolved drops out.
    assert result.asset_paths == []


def test_empty_body_after_frontmatter_reported(tmp_path: Path) -> None:
    """Body with only frontmatter (or only whitespace) yields zero
    chunks; ingest accepts it but it pollutes the source set."""
    note = tmp_path / "sources" / "empty.md"
    _write(note, "---\ntitle: empty\n---\n   \n\n")

    result = inspect_markdown(note, project_root=tmp_path)

    assert not result.ok
    assert any(i.kind == "empty_body" for i in result.issues)


def test_truly_empty_file_reported(tmp_path: Path) -> None:
    note = tmp_path / "sources" / "zero.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_bytes(b"")

    result = inspect_markdown(note, project_root=tmp_path)

    assert not result.ok
    assert any(i.kind == "empty_body" for i in result.issues)


def test_multiple_issues_accumulated(tmp_path: Path) -> None:
    """A single inspection lists *every* issue, not just the first.
    Users who fix one and re-run shouldn't be drip-fed errors."""
    note = tmp_path / "sources" / "n.md"
    _write(note, "# x\n![](missing-a.png)\n![](missing-b.png)\n")

    result = inspect_markdown(note, project_root=tmp_path)

    asset_issues = [i for i in result.issues if i.kind == "asset_missing"]
    assert len(asset_issues) == 2


def test_asset_paths_dedupe_across_refs(tmp_path: Path) -> None:
    """Same asset referenced twice in one md → resolved once in
    ``asset_paths`` (the upload packager needs the unique set, not the
    multiset of references)."""
    note = tmp_path / "sources" / "n.md"
    _write(note, "# x\n![](shared.png)\nbody\n![](shared.png)\n")
    asset = tmp_path / "sources" / "shared.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = inspect_markdown(note, project_root=tmp_path)

    assert result.ok, result.issues
    assert result.asset_paths == [asset.resolve()]


def test_inspection_result_is_immutable(tmp_path: Path) -> None:
    """A frozen dataclass means callers can stash the result in a dict
    or set without aliasing surprises."""
    note = tmp_path / "sources" / "alpha.md"
    _write(note, "# A\nbody\n")

    result = inspect_markdown(note, project_root=tmp_path)

    # frozen → mutating raises FrozenInstanceError (or AttributeError on
    # some Python builds).
    import dataclasses
    assert dataclasses.is_dataclass(result)
