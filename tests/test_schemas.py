"""Pydantic-shape tests for cross-Storage DTOs.

Pinned here so the discriminated-union ``MediaMeta`` contract on
``AssetRecord`` (and its image-only v1 incarnation) is the failing site
when someone reverts to the legacy ``width``/``height``/``caption`` columns.
"""

from __future__ import annotations

import time

from dikw_core.schemas import (
    AssetKind,
    AssetRecord,
    ImageMediaMeta,
)


def _bare_asset(**overrides: object) -> AssetRecord:
    """Build a minimal ``AssetRecord`` with all fields except media_meta set."""
    base: dict[str, object] = {
        "asset_id": "a" * 64,
        "kind": AssetKind.IMAGE,
        "mime": "image/png",
        "stored_path": "assets/aa/aaaaaaaa-img.png",
        "original_paths": ["img.png"],
        "bytes": 42,
        "created_ts": time.time(),
    }
    base.update(overrides)
    return AssetRecord(**base)  # type: ignore[arg-type]


def test_image_media_meta_kind_default() -> None:
    """ImageMediaMeta defaults to kind='image' and width/height=None."""
    m = ImageMediaMeta()
    assert m.kind == "image"
    assert m.width is None
    assert m.height is None


def test_image_media_meta_round_trip_json() -> None:
    """model_dump_json -> model_validate_json is byte-stable."""
    m = ImageMediaMeta(width=640, height=480)
    payload = m.model_dump_json()
    restored = ImageMediaMeta.model_validate_json(payload)
    assert restored == m
    # The kind field must persist on the wire so a future TypeAdapter dispatch
    # (audio/video) can route correctly.
    assert '"kind":"image"' in payload


def test_asset_record_drops_legacy_fields() -> None:
    """The legacy width/height/caption/caption_model columns must not exist
    on AssetRecord — they were intentionally folded into media_meta to keep
    the table schema modality-agnostic when audio/video lands."""
    fields = AssetRecord.model_fields
    for legacy in ("width", "height", "caption", "caption_model"):
        assert legacy not in fields, f"legacy field {legacy!r} resurfaced"


def test_asset_record_drops_hash_field() -> None:
    """``asset_id`` is itself the sha256 hex of the bytes, so a separate
    ``hash`` column would always carry the identical value. Pinned dead
    here so a future revert lands on a red test instead of silently
    reintroducing the duplicate UNIQUE column."""
    assert "hash" not in AssetRecord.model_fields


def test_asset_record_media_meta_defaults_to_none() -> None:
    """No media_meta keyword → field is None (matches DB NULL on probe miss)."""
    rec = _bare_asset()
    assert rec.media_meta is None


def test_asset_record_with_image_meta_round_trips() -> None:
    """An asset carrying an ImageMediaMeta must survive model_dump → validate."""
    rec = _bare_asset(media_meta=ImageMediaMeta(width=640, height=480))
    payload = rec.model_dump_json()
    restored = AssetRecord.model_validate_json(payload)
    assert restored.media_meta is not None
    assert isinstance(restored.media_meta, ImageMediaMeta)
    assert restored.media_meta.kind == "image"
    assert restored.media_meta.width == 640
    assert restored.media_meta.height == 480


