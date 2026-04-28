from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from _helpers import load_review_lib


def test_image_key_for_source_uses_canonical_content_hash_payload(tmp_path: Path) -> None:
    review_lib = load_review_lib()
    source = tmp_path / "source_a.png"
    source.write_bytes(b"same image bytes")
    render_profile = {"kind": "rgb_original"}

    payload = {
        "source_content_hash": "sha256:" + hashlib.sha256(b"same image bytes").hexdigest(),
        "render_profile": render_profile,
    }
    identity_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected = "img_" + hashlib.sha256(identity_bytes).hexdigest()[:16]

    assert review_lib.image_key_for_source(source, render_profile) == expected


def test_image_key_for_source_is_independent_of_filename(tmp_path: Path) -> None:
    review_lib = load_review_lib()
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"same image bytes")
    second.write_bytes(b"same image bytes")

    assert review_lib.image_key_for_source(first, {"kind": "rgb_original"}) == review_lib.image_key_for_source(second, {"kind": "rgb_original"})


def test_image_key_for_source_changes_by_render_profile(tmp_path: Path) -> None:
    review_lib = load_review_lib()
    source = tmp_path / "source.tiff"
    source.write_bytes(b"same source different render")

    true_color = review_lib.image_key_for_source(
        source,
        {"kind": "true_color", "bands": [1, 2, 3], "normalization": "current_loader_default", "output_colorspace": "RGB"},
    )
    false_color = review_lib.image_key_for_source(
        source,
        {"kind": "false_color", "bands": [4, 3, 2], "normalization": "current_loader_default", "output_colorspace": "RGB"},
    )

    assert true_color != false_color


def test_build_manifest_entry_uses_stable_image_key_and_keeps_filename(tmp_path: Path) -> None:
    review_lib = load_review_lib()
    source = tmp_path / "2020-01-01_tile_2_3.tiff"
    source.write_bytes(b"tile bytes")
    tile = review_lib.TileInfo(date="2020-01-01", row=2, col=3, path=source, mtime=123.0, size_bytes=len(b"tile bytes"))

    entry = review_lib.build_manifest_entry(
        date_value="2020-01-01",
        tile=tile,
        render_info={"path": "workspace/cache/example.png", "width": 100, "height": 80, "loader": "pillow", "rendered_at": "now"},
        model_info={"weights": {}, "confidence_threshold": 0.25},
        merge_info={"iou_threshold": 0.5},
    )

    assert entry["image_key"].startswith("img_")
    assert entry["image_key"] == review_lib.image_key_for_source(source, review_lib.TRUE_COLOR_RENDER_PROFILE)
    assert entry["image_filename"] == "2020-01-01_tile_2_3.png"
    assert entry["source_content_hash"].startswith("sha256:")
    assert entry["render_profile"] == review_lib.TRUE_COLOR_RENDER_PROFILE


def test_image_key_for_source_rejects_unknown_render_profile_kind(tmp_path: Path) -> None:
    review_lib = load_review_lib()
    source = tmp_path / "source.png"
    source.write_bytes(b"same image bytes")

    with pytest.raises(ValueError, match="Unknown render profile kind"):
        review_lib.image_key_for_source(source, {"kind": "rgb-orignal"})


def test_image_key_for_source_requires_geotiff_render_profile_fields(tmp_path: Path) -> None:
    review_lib = load_review_lib()
    source = tmp_path / "source.tiff"
    source.write_bytes(b"same image bytes")

    with pytest.raises(ValueError, match="missing required field"):
        review_lib.image_key_for_source(source, {"kind": "true_color", "bands": [1, 2, 3]})
