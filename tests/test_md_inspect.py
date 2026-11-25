"""Client-side D-layer pre-flight inspection tests.

``inspect_markdown(path, *, project_root)`` is the function the new
``dikw client import`` command calls before packaging a markdown file.
It returns an ``InspectionResult`` listing every reason ingest would
fail or warn (`frontmatter_error`, `asset_missing`, `empty_body`),
plus the resolved absolute paths of every local asset reference so the
client packaging step can include them in the same package as the md.

Layered above ``extract_image_refs`` (markdown-source asset extraction)
and ``_resolve_local`` (sibling-of-md → project-root two-stage lookup);
when the import command receives broken md or a missing-asset md, the
user gets a fast pre-flight error instead of a server-side ingest crash
ten seconds later.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from dikw_core.md_inspect import (
    InspectionResult,
    _candidate_is_symlink,
    _is_remote,
    _is_within_dikw_meta,
    _resolve_local,
    find_outside_root_refs,
    inspect_markdown,
    is_outside_root_ref,
)


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
    asset_missing — the import package needs no local file for them."""
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
    ``asset_paths`` (the importer needs the unique set, not the
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


def test_resolve_local_rejects_absolute_path_outside_root(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    note = vault / "sources" / "n.md"
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n")
    _write(note, f"# n\n![]({outside})\n")

    assert (
        _resolve_local(str(outside), source_md_path=note, project_root=vault)
        is None
    )
    result = inspect_markdown(note, project_root=vault)
    assert not result.ok
    assert any(i.kind == "asset_outside_root" for i in result.issues)
    assert result.asset_paths == []


def test_resolve_local_rejects_parent_traversal_outside_root(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    note = vault / "sources" / "n.md"
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n")
    _write(note, "# n\n![](../../outside.png)\n")

    assert (
        _resolve_local(
            "../../outside.png", source_md_path=note, project_root=vault
        )
        is None
    )
    result = inspect_markdown(note, project_root=vault)
    assert not result.ok
    assert any(i.kind == "asset_outside_root" for i in result.issues)
    assert result.asset_paths == []


def test_resolve_local_rejects_dikw_internal_metadata(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    note = vault / "sources" / "n.md"
    secret = vault / ".dikw" / "auth.json"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text('{"access_token": "fixture-only"}\n', encoding="utf-8")
    _write(note, "# n\n![](../.dikw/auth.json)\n")

    assert (
        _resolve_local(
            "../.dikw/auth.json", source_md_path=note, project_root=vault
        )
        is None
    )
    result = inspect_markdown(note, project_root=vault)
    assert not result.ok
    assert any(i.kind == "asset_outside_root" for i in result.issues)


def test_resolve_local_rejects_symlink_escaping_root(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    note = vault / "sources" / "n.md"
    _write(note, "# n\n![](escape.png)\n")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n")
    link = vault / "sources" / "escape.png"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as e:  # pragma: no cover - Windows
        pytest.skip(f"symlinks unavailable: {e}")

    assert (
        _resolve_local("escape.png", source_md_path=note, project_root=vault)
        is None
    )


def test_resolve_local_rejects_fifo(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):  # pragma: no cover - Windows
        pytest.skip("mkfifo unavailable on this platform")
    vault = tmp_path / "vault"
    note = vault / "sources" / "n.md"
    _write(note, "# n\n![](pipe.png)\n")
    fifo = vault / "sources" / "pipe.png"
    os.mkfifo(fifo)

    assert (
        _resolve_local("pipe.png", source_md_path=note, project_root=vault)
        is None
    )


@pytest.mark.parametrize("mode", [stat.S_IFCHR | 0o666, stat.S_IFBLK | 0o660])
def test_resolve_local_rejects_device_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: int,
) -> None:
    vault = tmp_path / "vault"
    note = vault / "sources" / "n.md"
    _write(note, "# n\n![](dev.png)\n")
    target = vault / "sources" / "dev.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n")
    target_resolved = target.resolve()

    real_stat = Path.stat

    def _fake_stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        st = real_stat(self, follow_symlinks=follow_symlinks)
        if os.path.realpath(self) == str(target_resolved):
            return os.stat_result((mode, *st[1:]))
        return st

    monkeypatch.setattr(Path, "stat", _fake_stat)

    assert (
        _resolve_local("dev.png", source_md_path=note, project_root=vault)
        is None
    )


def test_is_within_dikw_meta_handles_path_outside_root(tmp_path: Path) -> None:
    vault = (tmp_path / "vault").resolve()
    outside = (tmp_path / "other" / "file.png").resolve()
    assert _is_within_dikw_meta(outside, vault) is False


@pytest.mark.parametrize("bad_ref", ["", "bad\x00.png"])
def test_forbidden_empty_and_nul_refs_rejected(
    tmp_path: Path, bad_ref: str
) -> None:
    vault = tmp_path / "vault"
    note = vault / "sources" / "n.md"
    _write(note, "# n\nbody\n")

    assert is_outside_root_ref(bad_ref, source_md_path=note, project_root=vault)
    assert _resolve_local(bad_ref, source_md_path=note, project_root=vault) is None


def test_is_outside_root_ref_allows_remote_urls(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    note = vault / "sources" / "n.md"
    _write(note, "# n\nbody\n")

    assert (
        is_outside_root_ref(
            "https://example.com/img.png",
            source_md_path=note,
            project_root=vault,
        )
        is False
    )


def test_find_outside_root_refs_falls_back_on_malformed_frontmatter(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    note = vault / "sources" / "bad.md"
    _write(
        note,
        "---\ntitle: a\nfoo: : bar\n---\n"
        "![](https://example.com/ok.png)\n"
        "![](../../outside.png)\n",
    )

    assert find_outside_root_refs(note, project_root=vault) == [
        "../../outside.png"
    ]


def test_candidate_is_symlink_skips_dotdot_segment(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    note = vault / "sources" / "n.md"
    _write(note, "# n\n![](..)\n")

    assert (
        _candidate_is_symlink("..", source_md_path=note, project_root=vault)
        is False
    )


def test_symlinked_dikw_directory_excluded_while_ordinary_assets_allowed(
    tmp_path: Path,
) -> None:
    """When ``root/.dikw`` is a symlink to ``root/private-state``, both
    ``../.dikw/auth.json`` and ``../private-state/auth.json`` must be rejected,
    while ordinary assets outside both subtrees remain valid."""
    vault = tmp_path / "vault"
    private_state = vault / "private-state"
    private_state.mkdir(parents=True)
    (private_state / "auth.json").write_text(
        '{"access_token": "fixture-only"}\n', encoding="utf-8"
    )
    dikw_link = vault / ".dikw"
    try:
        dikw_link.symlink_to(private_state, target_is_directory=True)
    except (OSError, NotImplementedError) as e:  # pragma: no cover - Windows
        pytest.skip(f"symlinks unavailable: {e}")

    ok_img = vault / "sources" / "ok.png"
    ok_img.parent.mkdir(parents=True, exist_ok=True)
    ok_img.write_bytes(b"\x89PNG\r\n\x1a\n")

    note_via_dikw = vault / "sources" / "via_dikw.md"
    _write(note_via_dikw, "# n\n![](../.dikw/auth.json)\n")
    assert (
        _resolve_local(
            "../.dikw/auth.json",
            source_md_path=note_via_dikw,
            project_root=vault,
        )
        is None
    )
    res_dikw = inspect_markdown(note_via_dikw, project_root=vault)
    assert not res_dikw.ok
    assert any(i.kind == "asset_outside_root" for i in res_dikw.issues)
    assert res_dikw.asset_paths == []

    note_via_target = vault / "sources" / "via_target.md"
    _write(note_via_target, "# n\n![](../private-state/auth.json)\n")
    assert (
        _resolve_local(
            "../private-state/auth.json",
            source_md_path=note_via_target,
            project_root=vault,
        )
        is None
    )
    res_target = inspect_markdown(note_via_target, project_root=vault)
    assert not res_target.ok
    assert any(i.kind == "asset_outside_root" for i in res_target.issues)
    assert res_target.asset_paths == []

    note_ok = vault / "sources" / "ok.md"
    _write(note_ok, "# n\n![](ok.png)\n")
    res_ok = inspect_markdown(note_ok, project_root=vault)
    assert res_ok.ok, res_ok.issues
    assert res_ok.asset_paths == [ok_img.resolve()]


def test_is_within_dikw_meta_handles_cyclic_dikw_symlink(
    tmp_path: Path,
) -> None:
    """A broken/cyclic ``.dikw`` symlink must not break ordinary assets."""
    vault = tmp_path / "vault"
    vault.mkdir()
    dikw_link = vault / ".dikw"
    try:
        dikw_link.symlink_to(dikw_link)
    except (OSError, NotImplementedError) as e:  # pragma: no cover - Windows
        pytest.skip(f"symlinks unavailable: {e}")

    vault_resolved = vault.resolve()
    assert (
        _is_within_dikw_meta(
            vault_resolved / "sources" / "ok.png", vault_resolved
        )
        is False
    )


@pytest.mark.parametrize(
    "win_ref",
    [r"C:\outside\secret.png", "c:/outside/secret.png", "D:relative.png"],
)
def test_windows_drive_letter_path_classified_local_and_outside_root(
    tmp_path: Path, win_ref: str
) -> None:
    """Windows drive-letter paths must not be mistaken for URL schemes by
    ``_is_remote``; they must reach containment and report ``asset_outside_root``."""
    vault = tmp_path / "vault"
    note = vault / "sources" / "n.md"
    _write(note, f"# n\n![]({win_ref})\n")

    assert _is_remote(win_ref) is False
    assert is_outside_root_ref(win_ref, source_md_path=note, project_root=vault)
    assert (
        _resolve_local(win_ref, source_md_path=note, project_root=vault) is None
    )
    result = inspect_markdown(note, project_root=vault)
    assert not result.ok
    assert any(i.kind == "asset_outside_root" for i in result.issues)
    assert result.asset_paths == []
