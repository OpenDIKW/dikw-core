"""Multimedia asset materialization + path composition.

v1 scope: images only. Audio/video extension points are reserved on the
schemas (`AssetKind`, `MultimodalInput`) but not exercised here.

This module owns the contract between Markdown source references like
``![alt](./diagrams/foo.png)`` and the engine-managed on-disk layout at
``<project_root>/assets/<h2>/<h8>-<sanitized-name>.<ext>``. The path scheme
combines a 256-way sha256 prefix shard (FS-friendly under heavy ingest)
with the original file name preserved for human/Obsidian readability.
"""

from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable
from pathlib import Path

from ...md_inspect import _is_remote, _resolve_local
from ...schemas import AssetKind, AssetRecord, AssetRef, ImageMediaMeta
from .hashing import hash_bytes, hash_file

logger = logging.getLogger(__name__)

# Windows reserved device names. Compared case-insensitively against the
# sanitized stem so a literal `con.png` in markdown doesn't break a
# Windows-mounted vault.
_WIN_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

# Sanitized stem's max UTF-8 byte length. Combined with the ``<h8>-`` prefix
# (9 bytes), shard dir (3 bytes) and a few-byte extension this stays well
# under the 255-byte filename cap of every major filesystem (APFS, ext4,
# NTFS) — leaving headroom for cross-platform sync clients with their own
# stricter caps.
_STEM_BYTE_CAP = 150

_EXT_RE = re.compile(r"\.([A-Za-z0-9]+)$")
_HYPHEN_RUN = re.compile(r"-+")


def _split_basename_and_ext(name: str) -> tuple[str, str]:
    """Strip directories, then return (stem, ext) where ext is lower-case
    without the leading dot. Empty ext = no extension detected."""
    # Treat both POSIX and Windows separators robustly even though markdown
    # references should be POSIX-only in practice.
    base = os.path.basename(name.replace("\\", "/"))
    m = _EXT_RE.search(base)
    if m is None:
        return base, ""
    return base[: m.start()], m.group(1).lower()


def _is_allowed_char(ch: str) -> bool:
    """The sanitize whitelist. Anything not in here is replaced with ``-``.

    Allowed:
      - ``-`` and ``_`` (literal)
      - Any Unicode "Letter" (Lu/Ll/Lt/Lm/Lo) — covers ASCII letters, Latin
        Extended (é, ñ, ü), CJK, Hiragana, Katakana, Hangul, Cyrillic,
        Greek, Arabic, Hebrew, Devanagari, etc.
      - Any Unicode "Number" (Nd/Nl/No) — covers ASCII digits, Arabic-
        Indic digits, full-width digits, etc.

    Excluded by virtue of category mismatch: punctuation (P*), symbols
    (S*, including emoji), separators (Z*), marks (M*, includes leftover
    NFD combining accents that NFC didn't fold), and control codes (C*).
    """
    if ch in ("-", "_"):
        return True
    # Letters and Numbers cover all writing systems' name characters with
    # one rule, instead of an ever-growing range list. Mn/Mc (combining
    # marks) deliberately excluded so any NFD residue gets sanitized away.
    return unicodedata.category(ch)[0] in ("L", "N")


def _truncate_to_utf8_bytes(s: str, byte_cap: int) -> str:
    """Truncate ``s`` to at most ``byte_cap`` UTF-8 bytes without ever
    splitting a multi-byte character."""
    encoded = s.encode("utf-8")
    if len(encoded) <= byte_cap:
        return s
    encoded = encoded[:byte_cap]
    while encoded:
        try:
            return encoded.decode("utf-8")
        except UnicodeDecodeError:
            encoded = encoded[:-1]
    return ""


def _sanitize_stem(stem: str) -> str:
    """NFC-normalize, whitelist-filter, hyphen-collapse, byte-truncate,
    Windows-reserved guard, leading/trailing hyphen+dot strip."""
    # NFC folds APFS-style NFD so the same logical filename produces the
    # same sanitize output across platforms.
    stem = unicodedata.normalize("NFC", stem)
    out = "".join(ch if _is_allowed_char(ch) else "-" for ch in stem)
    out = _HYPHEN_RUN.sub("-", out)
    # Strip dots too — when the original ends with `.foo.bar`, splitting
    # off `.bar` as ext leaves a trailing `.` to clean up.
    out = out.strip("-.")
    out = _truncate_to_utf8_bytes(out, _STEM_BYTE_CAP)
    out = out.strip("-.")  # truncation may leave a dangling separator
    if out.upper() in _WIN_RESERVED:
        out = "_" + out
    return out


def assets_relative_path(
    *, hash_: str, original_path: str, dir_: str = "assets"
) -> str:
    """Compose the engine-managed relative storage path for an asset.

    Format::

        <dir_>/<h2>/<h8>[-<sanitized-stem>][.<ext>]

    where ``h2 = hash_[:2]`` is the 256-way shard and ``h8 = hash_[:8]`` is
    the per-file disambiguation prefix that lets two assets with identical
    sanitized names but different content live side by side without
    collision. The returned path is POSIX-style (``/`` separators) so it
    serializes uniformly into databases and cross-platform vaults.

    ``original_path`` may be an arbitrary reference string (relative path,
    bare filename, or even a path-with-subdirs); only the basename feeds
    the sanitized stem. ``dir_`` defaults to ``assets`` to match the
    convention written into ``dikw.yml`` under ``assets.dir``.
    """
    if len(hash_) < 8:
        raise ValueError(f"hash too short for sharding: {hash_!r}")
    h2 = hash_[:2]
    h8 = hash_[:8]
    stem, ext = _split_basename_and_ext(original_path)
    sanitized = _sanitize_stem(stem)
    name = h8
    if sanitized:
        name = f"{name}-{sanitized}"
    if ext:
        name = f"{name}.{ext}"
    return f"{dir_}/{h2}/{name}"


__all__ = [
    "assets_relative_path",
    "materialize_asset",
]


# ---- Image MIME + dimension probe (stdlib, no Pillow) --------------------

_EXT_TO_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "svg": "image/svg+xml",
}


def _detect_image_mime(data: bytes, ext_hint: str = "") -> str | None:
    """Detect image MIME from magic bytes; fall back to file extension.

    Returns ``None`` for anything that isn't a v1-supported image format,
    which the caller treats as a signal to skip materialization.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    # SVG is text-based; sniff the first ~200 bytes for the root tag.
    head = data[:200].lower()
    if b"<svg" in head or (b"<?xml" in head and b"svg" in data[:1000].lower()):
        return "image/svg+xml"
    return _EXT_TO_MIME.get(ext_hint.lower())


def _probe_dimensions(data: bytes, mime: str) -> tuple[int | None, int | None]:
    """Stdlib image (width, height) probe. Returns ``(None, None)`` for
    formats not parsed in v1 (SVG, WebP) or malformed data.

    PNG/JPEG/GIF only — no Pillow dependency. Each format's parser reads
    only the bytes it needs to resolve dimensions; nothing decodes pixels.
    """
    try:
        if mime == "image/png" and len(data) >= 24:
            # 8-byte signature, then IHDR chunk: 4-byte length, 4-byte type,
            # then width (uint32 BE), height (uint32 BE).
            w = int.from_bytes(data[16:20], "big")
            h = int.from_bytes(data[20:24], "big")
            return w, h
        if mime == "image/jpeg":
            # Walk segment headers (FF marker + 1 byte type + 2 byte length)
            # until a Start-of-Frame marker is found, then read precision +
            # height + width.
            i = 2  # skip SOI
            n = len(data)
            while i < n - 9:
                if data[i] != 0xFF:
                    return None, None
                marker = data[i + 1]
                # Restart markers (D0-D7) and SOI/EOI/etc carry no payload.
                if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                # SOFn: 0xC0..0xCF except 0xC4 (DHT), 0xC8 (JPG), 0xCC (DAC).
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    h = int.from_bytes(data[i + 5 : i + 7], "big")
                    w = int.from_bytes(data[i + 7 : i + 9], "big")
                    return w, h
                seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
                i += 2 + seg_len
            return None, None
        if mime == "image/gif" and len(data) >= 10:
            # Header: 6-byte signature + Logical Screen Descriptor's first
            # 4 bytes are width (uint16 LE) + height (uint16 LE).
            w = int.from_bytes(data[6:8], "little")
            h = int.from_bytes(data[8:10], "little")
            return w, h
    except (IndexError, ValueError):
        # Defensive: malformed image bytes shouldn't crash ingest.
        return None, None
    return None, None


# ---- Asset materialization -----------------------------------------------


async def _attach_or_return(
    existing: AssetRecord,
    ref: AssetRef,
    upsert_asset: Callable[[AssetRecord], Awaitable[None]],
) -> tuple[AssetRecord, bool]:
    """Append ``ref.original_path`` to ``existing.original_paths`` if it
    isn't already there, persist, and return ``(record, was_new=False)``.
    """
    if ref.original_path in existing.original_paths:
        return existing, False
    updated = existing.model_copy(
        update={
            "original_paths": [*existing.original_paths, ref.original_path],
        }
    )
    await upsert_asset(updated)
    return updated, False


async def materialize_asset(
    ref: AssetRef,
    *,
    source_md_path: Path,
    project_root: Path,
    get_asset: Callable[[str], Awaitable[AssetRecord | None]],
    upsert_asset: Callable[[AssetRecord], Awaitable[None]],
    dir_: str = "assets",
) -> tuple[AssetRecord, bool] | None:
    """Resolve a single ``AssetRef`` to an on-disk binary, copy it into the
    engine vault under ``project_root/<dir_>/…`` (deduped by sha256), probe
    its MIME + dimensions, and persist the metadata.

    Returns ``(record, was_newly_created)`` so the caller can avoid
    re-embedding assets that already lived in storage from a prior run.

    The function is idempotent by content hash:
      * Same hash already in storage → returns ``(existing_or_updated,
        False)`` after appending the current ``original_path`` to
        ``original_paths`` (deduplicating identical entries). The binary
        is *not* re-written.
      * New hash → writes ``<dir_>/<h2>/<h8>-<sanitized>.<ext>``
        atomically (``.tmp.<rand>`` → rename) and inserts a fresh
        ``AssetRecord``; returns ``(record, True)``.

    Returns ``None`` (and logs at WARNING) for remote URLs, missing files,
    or unrecognizable formats — the caller decides how to surface skips
    to the user.

    ``get_asset`` / ``upsert_asset`` are passed in as callables (rather
    than a full Storage handle) so this function stays decoupled from the
    storage Protocol additions, which land in a later phase.
    """
    if _is_remote(ref.original_path):
        logger.warning(
            "skipping remote image reference: %s", ref.original_path
        )
        return None

    abs_path = _resolve_local(
        ref.original_path,
        source_md_path=source_md_path,
        project_root=project_root,
    )
    if abs_path is None:
        logger.warning(
            "skipping unresolvable image reference: %r relative to %s",
            ref.original_path,
            source_md_path,
        )
        return None

    # Stream-hash first so cache hits never have to slurp the file.
    try:
        sha = hash_file(abs_path)
    except OSError as e:
        logger.warning("failed to hash %s: %s", abs_path, e)
        return None

    existing = await get_asset(sha)
    if existing is not None:
        # Revalidate against the current file before trusting the cache:
        # a concurrent rewrite between the hash and the lookup would
        # otherwise attach this path to a stale record. Re-streaming
        # keeps the memory bound; if the bytes shifted, fall through to
        # the slurp path which writes under the canonical hash.
        try:
            confirmed_sha = hash_file(abs_path)
        except OSError as e:
            logger.warning("failed to revalidate hash for %s: %s", abs_path, e)
            return None
        if confirmed_sha == sha:
            return await _attach_or_return(existing, ref, upsert_asset)
        sha = confirmed_sha
        existing = await get_asset(sha)
        if existing is not None:
            return await _attach_or_return(existing, ref, upsert_asset)

    try:
        data = abs_path.read_bytes()
    except OSError as e:
        logger.warning("failed to read %s: %s", abs_path, e)
        return None

    # Race window: the file may have changed between hash_file and the
    # slurp. Persist under the canonical hash of the bytes we actually
    # have, and re-check cache so a content-address dedup hit still wins.
    canonical_sha = hash_bytes(data)
    if canonical_sha != sha:
        sha = canonical_sha
        existing = await get_asset(sha)
        if existing is not None:
            return await _attach_or_return(existing, ref, upsert_asset)

    _, ext_hint = _split_basename_and_ext(ref.original_path)
    mime = _detect_image_mime(data, ext_hint=ext_hint)
    if mime is None:
        logger.warning(
            "skipping unrecognized image format at %s (ext=%r)",
            abs_path,
            ext_hint,
        )
        return None

    width, height = _probe_dimensions(data, mime)
    rel_stored = assets_relative_path(
        hash_=sha, original_path=ref.original_path, dir_=dir_
    )
    abs_stored = project_root / rel_stored
    if not abs_stored.exists():
        abs_stored.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: copy via temp file then rename so a crash in the
        # middle never leaves a half-written asset visible at stored_path.
        tmp = abs_stored.with_name(f".tmp.{os.urandom(6).hex()}{abs_stored.suffix}")
        tmp.write_bytes(data)
        os.replace(tmp, abs_stored)

    record = AssetRecord(
        asset_id=sha,
        kind=AssetKind.IMAGE,
        mime=mime,
        stored_path=rel_stored,
        original_paths=[ref.original_path],
        bytes=len(data),
        media_meta=(
            ImageMediaMeta(width=width, height=height)
            if (width is not None or height is not None)
            else None
        ),
        created_ts=time.time(),
    )
    await upsert_asset(record)
    return record, True
