from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from _helpers import detection, load_review_lib, write_review


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_source_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (100, 80), color=(25, 50, 75))
    image.save(path)


def review_doc(cache_path: Path, *, review_label: str = "ship_non_sts") -> dict:
    class_by_label = {"ship_non_sts": (0, "ship"), "ships_sts": (1, "ships"), "neither": (0, "ship")}
    class_id, class_name = class_by_label[review_label]
    return {
        "schema_version": 1,
        "image_key": "img_abc123",
        "image_filename": "source.png",
        "reviewer": "gavin",
        "width": 100,
        "height": 80,
        "cache_path": str(cache_path),
        "review_status": "completed",
        "detections": [
            {
                "detection_id": "auto-0001",
                "source_type": "auto",
                "parent_detection_id": None,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": 0.8,
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.2,
                "height": 0.125,
                "review_label": review_label,
                "reviewed": True,
                "reviewed_at": "2026-04-28T00:00:00Z",
                "source_models": ["fixture"],
            }
        ],
    }


def test_generate_review_crop_writes_padded_png_and_sidecar(tmp_path: Path) -> None:
    review_lib = load_review_lib()
    source_path = tmp_path / "cache" / "source.png"
    review_path = tmp_path / "reviews" / "gavin" / "img_abc123.json"
    crop_root = tmp_path / "cache" / "review_crops"
    write_source_image(source_path)
    review_path.parent.mkdir(parents=True)
    review_path.write_text("{}\n")

    result = review_lib.generate_review_crop(
        review_doc(source_path),
        "auto-0001",
        review_path=review_path,
        crop_root=crop_root,
        padding_px=10,
        generated_at="2026-04-28T00:00:00Z",
    )

    crop_path = crop_root / "img_abc123" / "auto-0001.png"
    sidecar_path = crop_root / "img_abc123" / "auto-0001.json"
    assert result == {"crop_path": crop_path, "sidecar_path": sidecar_path}
    assert Image.open(crop_path).size == (40, 30)

    sidecar = read_json(sidecar_path)
    assert sidecar["schema_version"] == 1
    assert sidecar["image_key"] == "img_abc123"
    assert sidecar["detection_id"] == "auto-0001"
    assert sidecar["review_file"] == str(review_path)
    assert sidecar["crop_path"] == str(crop_path)
    assert sidecar["source_cache_path"] == str(source_path)
    assert sidecar["source_image_width"] == 100
    assert sidecar["source_image_height"] == 80
    assert sidecar["bbox"] == {"x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.125}
    assert sidecar["crop_pixels"] == {"x1": 30, "y1": 25, "x2": 70, "y2": 55, "padding_px": 10}
    assert sidecar["review"] == {
        "review_label": "ship_non_sts",
        "class_id": 0,
        "class_name": "ship",
        "source_type": "auto",
        "reviewed": True,
    }
    assert sidecar["generated_at"] == "2026-04-28T00:00:00Z"
    assert sidecar["fingerprints"]["source_image"].startswith("sha256:")
    assert sidecar["fingerprints"]["bbox_geometry"].startswith("sha256:")
    assert sidecar["fingerprints"]["crop_recipe"].startswith("sha256:")


def test_generate_review_crop_clamps_to_image_bounds(tmp_path: Path) -> None:
    review_lib = load_review_lib()
    source_path = tmp_path / "cache" / "source.png"
    write_source_image(source_path)
    review = review_doc(source_path)
    review["detections"][0].update({"x_center": 0.02, "y_center": 0.03, "width": 0.1, "height": 0.1})

    review_lib.generate_review_crop(
        review,
        "auto-0001",
        review_path=tmp_path / "reviews" / "gavin" / "img_abc123.json",
        crop_root=tmp_path / "cache" / "review_crops",
        padding_px=20,
        generated_at="2026-04-28T00:00:00Z",
    )

    sidecar = read_json(tmp_path / "cache" / "review_crops" / "img_abc123" / "auto-0001.json")
    assert sidecar["crop_pixels"]["x1"] == 0
    assert sidecar["crop_pixels"]["y1"] == 0


def test_generate_review_crop_preserves_neither_metadata_without_yolo_class(tmp_path: Path) -> None:
    review_lib = load_review_lib()
    source_path = tmp_path / "cache" / "source.png"
    write_source_image(source_path)

    review_lib.generate_review_crop(
        review_doc(source_path, review_label="neither"),
        "auto-0001",
        review_path=tmp_path / "reviews" / "gavin" / "img_abc123.json",
        crop_root=tmp_path / "cache" / "review_crops",
        padding_px=10,
        generated_at="2026-04-28T00:00:00Z",
    )

    sidecar = read_json(tmp_path / "cache" / "review_crops" / "img_abc123" / "auto-0001.json")
    assert sidecar["review"]["review_label"] == "neither"
    assert sidecar["review"]["class_id"] is None
    assert sidecar["review"]["class_name"] is None


def test_build_review_bbox_overview_collects_reviewed_crops_source_agnostic(tmp_path: Path, monkeypatch) -> None:
    review_lib = load_review_lib()
    monkeypatch.setattr(review_lib, 'WORKSPACE_DIR', tmp_path)
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    crop_root = tmp_path / "cache" / "review_crops"
    manual_ship = detection("manual-0001", "ship_non_sts")
    manual_ship["source_type"] = "manual"
    write_review(
        reviews_dir,
        cache_dir,
        reviewer="gavin",
        image_key="img_reviewed",
        detections=[manual_ship, detection("auto-0001", "ships_sts"), detection("auto-0002", "neither")],
    )

    overview = review_lib.build_review_bbox_overview(reviews_dir, crop_root=crop_root, padding_px=4, generate_crops=True)

    assert overview["counts"]["reviewed_bbox_count"] == 3
    assert overview["counts"]["reviewed_ship_bboxes"] == 1
    assert overview["counts"]["reviewed_ships_bboxes"] == 1
    assert overview["counts"]["neither_bboxes"] == 1
    assert overview["counts"]["generated_crop_count"] == 3
    manual_tile = next(tile for tile in overview["tiles"] if tile["detection_id"] == "manual-0001")
    assert manual_tile["source_type"] == "manual"
    assert manual_tile["class_name"] == "ship"
    assert Path(manual_tile["crop_path"]).is_file()
    from urllib.request import urlopen
    from scripts.training_pipeline_ui import TrainingPipelineHandler
    from test_training_pipeline_ui import UiServer
    with UiServer(TrainingPipelineHandler) as server:
        for tile in overview['tiles']:
            with urlopen(server.base_url + tile['crop_url']) as response:
                assert response.headers['Content-Type'] == 'image/png'
                assert response.read() == Path(tile['crop_path']).read_bytes()


def test_build_review_bbox_overview_flags_completed_duplicate_conflicts(tmp_path: Path) -> None:
    review_lib = load_review_lib()
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    write_review(reviews_dir, cache_dir, reviewer="gavin", image_key="img_duplicate", detections=[detection("auto-0001", "ship_non_sts")])
    write_review(reviews_dir, cache_dir, reviewer="alex", image_key="img_duplicate", detections=[detection("auto-0001", "ships_sts")])

    overview = review_lib.build_review_bbox_overview(reviews_dir, crop_root=tmp_path / "crops", generate_crops=False)

    assert overview["tiles"] == []
    assert overview["counts"]["conflict_image_count"] == 1
    assert overview["conflicts"][0]["image_key"] == "img_duplicate"
    assert sorted(overview["conflicts"][0]["reviewers"]) == ["alex", "gavin"]
