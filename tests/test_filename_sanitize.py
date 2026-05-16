"""Tests for the asset filename-sanitize + path-composition pipeline.

The pinned shape (asset path layout):

  assets/<h2>/<h8>-<sanitized-original-name>.<ext>

with sanitize() doing path-strip, NFC normalization, ASCII/CJK/JP/KR
character whitelisting, hyphen compression, byte-length truncation,
Windows reserved-name guarding, and empty-stem fallback.
"""

from __future__ import annotations

import pytest

from dikw_core.domains.data.assets import assets_relative_path

# 64-hex sha256 stub; only the first 8 chars affect the filename, but the full
# hash is required for the API contract.
H = "ab3f12ef" + "0" * 56


@pytest.mark.parametrize(
    "original, expected",
    [
        # Spaces and ASCII punctuation collapse to single hyphens.
        ("architecture diagram.png", "assets/ab/ab3f12ef-architecture-diagram.png"),
        # Pure CJK passes through.
        ("架构图.png", "assets/ab/ab3f12ef-架构图.png"),
        # Mixed ASCII + CJK + parens + dots in stem.
        (
            "系统架构 v2.1 (final).jpg",
            "assets/ab/ab3f12ef-系统架构-v2-1-final.jpg",
        ),
        # Em dash + double exclamation become hyphens; extension lower-cased.
        ("Figure 3 — Final!!.JPG", "assets/ab/ab3f12ef-Figure-3-Final.jpg"),
        # Japanese katakana passes through.
        (
            "スクリーンショット 2026.png",
            "assets/ab/ab3f12ef-スクリーンショット-2026.png",
        ),
        # Full-width quotes become hyphens, run-collapsed.
        (
            "“引用”的截图.png",
            "assets/ab/ab3f12ef-引用-的截图.png",
        ),
        # Empty stem after sanitize falls back to bare hash.
        ("   .png", "assets/ab/ab3f12ef.png"),
        # Windows reserved name gets underscore prefix.
        ("CON.png", "assets/ab/ab3f12ef-_CON.png"),
        ("aux.JPG", "assets/ab/ab3f12ef-_aux.jpg"),
        ("COM3.png", "assets/ab/ab3f12ef-_COM3.png"),
        # Path components are stripped (only basename considered).
        ("subdir/foo.PNG", "assets/ab/ab3f12ef-foo.png"),
        ("a/b/c/deep.gif", "assets/ab/ab3f12ef-deep.gif"),
        # Multi-dot stems: only the last segment is the extension.
        ("foo.tar.gz", "assets/ab/ab3f12ef-foo-tar.gz"),
        # Underscore is preserved in the whitelist.
        ("snake_case_name.png", "assets/ab/ab3f12ef-snake_case_name.png"),
        # Run of hyphens collapses to one.
        ("a   b   c.png", "assets/ab/ab3f12ef-a-b-c.png"),
        # Leading/trailing punctuation stripped from stem.
        ("---weird---.png", "assets/ab/ab3f12ef-weird.png"),
        # Korean Hangul passes through.
        (
            "화면-측체.png",
            "assets/ab/ab3f12ef-화면-측체.png",
        ),
        # Emoji & control chars become hyphens.
        ("hello\U0001f600world.png", "assets/ab/ab3f12ef-hello-world.png"),
    ],
)
def test_assets_relative_path(original: str, expected: str) -> None:
    assert assets_relative_path(hash_=H, original_path=original) == expected


def test_assets_relative_path_nfd_normalizes_to_nfc() -> None:
    """é written in NFD (e + U+0301) must NFC-normalize before sanitize.

    Source editors and Write-tool round-trips routinely normalize visible
    glyphs to NFC, so this composes the NFD form by explicit codepoint
    construction — the only way to guarantee the input is NOT already NFC.
    """
    import unicodedata as _u
    combining_acute = chr(0x0301)
    nfd = "re" + combining_acute + "sume" + combining_acute + ".png"
    # Sanity: the input really is NFD, not NFC.
    assert _u.normalize("NFD", nfd) == nfd
    assert _u.normalize("NFC", nfd) != nfd

    out = assets_relative_path(hash_=H, original_path=nfd)
    # Expected uses precomposed é (U+00E9), produced by NFC normalization.
    e_acute = chr(0x00E9)
    assert out == f"assets/ab/ab3f12ef-r{e_acute}sum{e_acute}.png"


def test_assets_relative_path_truncates_at_utf8_boundary() -> None:
    """A very long Chinese stem must truncate without splitting a multi-byte char."""
    # 60 CJK chars = 180 UTF-8 bytes, exceeds the 150-byte cap.
    very_long = "你好" * 30 + ".png"
    result = assets_relative_path(hash_=H, original_path=very_long)
    assert result.startswith("assets/ab/ab3f12ef-")
    assert result.endswith(".png")

    middle = result[len("assets/ab/ab3f12ef-") : -len(".png")]
    # Sanitized stem byte length capped at 150.
    assert len(middle.encode("utf-8")) <= 150
    # Round-trip — proves no half-character at the boundary.
    middle.encode("utf-8").decode("utf-8")
    # Must contain at least *some* of the original CJK content.
    assert "你" in middle or "好" in middle


def test_assets_relative_path_extension_lowercased() -> None:
    assert assets_relative_path(hash_=H, original_path="X.JPEG").endswith(".jpeg")
    assert assets_relative_path(hash_=H, original_path="x.WebP").endswith(".webp")
    assert assets_relative_path(hash_=H, original_path="x.GIF").endswith(".gif")


def test_assets_relative_path_no_extension() -> None:
    """Extensionless input → extensionless output. Materialize layer handles
    extension derivation from MIME if needed."""
    assert (
        assets_relative_path(hash_=H, original_path="screenshot")
        == "assets/ab/ab3f12ef-screenshot"
    )


def test_assets_relative_path_custom_dir() -> None:
    assert (
        assets_relative_path(hash_=H, original_path="x.png", dir_="media")
        == "media/ab/ab3f12ef-x.png"
    )


def test_assets_relative_path_uses_first_two_hash_chars_for_shard() -> None:
    h = "deadbeefcafef00d" + "0" * 48
    out = assets_relative_path(hash_=h, original_path="x.png")
    assert out == "assets/de/deadbeef-x.png"


def test_assets_relative_path_full_empty_input() -> None:
    """Both stem and ext empty → just <h8> with no name segment, no dot."""
    assert assets_relative_path(hash_=H, original_path="") == "assets/ab/ab3f12ef"
    assert assets_relative_path(hash_=H, original_path="...") == "assets/ab/ab3f12ef"
