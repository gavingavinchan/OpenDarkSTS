from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from PIL import Image


def load_builder():
    try:
        return importlib.import_module("scripts.build_clean_sts_dataset")
    except ModuleNotFoundError:  # pragma: no cover - red state helper
        pytest.fail("scripts.build_clean_sts_dataset must provide the clean dataset builder")


def _write_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(path)


def _write_madhulika_file(root: Path, split: str, name: str, label_lines: list[str], size=(32, 32)) -> None:
    _write_image(root / split / "images" / name)
    label_path = root / split / "labels" / f"{Path(name).stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))


def test_parse_label_text_normalises_and_counts() -> None:
    builder = load_builder()
    lines, counts = builder.parse_label_text("0 0.5 0.5 0.1 0.2\n1 0.25 0.25 0.05 0.05\n\nbad line\n3 0.1 0.1 0.1 0.1\n")
    assert counts == {"ship": 1, "ships": 1}
    assert lines[0].startswith("0 0.50000000")
    assert len(lines) == 2


def test_madhulika_dedup_keeps_one_true_color_representative_per_scene(tmp_path: Path) -> None:
    builder = load_builder()
    root = tmp_path / "madhulika"
    scene = "2021-01-03-00_00_2021-01-03-23_59_Sentinel-2_L2A_True_color"
    # Three Roboflow augmentations of the same scene, spread over two splits.
    _write_madhulika_file(root, "train", f"{scene}_jpg.rf.aaa111.jpg", ["0 0.5 0.5 0.1 0.1"])
    _write_madhulika_file(root, "test", f"{scene}-1-_jpg.rf.bbb222.jpg", ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1"])
    _write_madhulika_file(root, "valid", f"{scene}_jpg.rf.ccc333.jpg", [])
    # A False_color render of another scene that also has a True_color render.
    other = "2023-04-13-00_00_2023-04-13-23_59_Sentinel-2_L2A"
    _write_madhulika_file(root, "train", f"{other}_False_color_jpg.rf.ddd444.jpg", ["0 0.4 0.4 0.1 0.1"])
    _write_madhulika_file(root, "train", f"{other}_True_color_jpg.rf.eee555.jpg", ["0 0.4 0.4 0.1 0.1"])

    items, summary = builder.collect_madhulika_items(root)

    assert summary["files_seen"] == 5
    assert summary["images_kept"] == 2
    assert summary["augmented_duplicates_dropped"] == 2
    assert summary["false_color_duplicates_dropped"] == 1
    # The three copies of one scene were spread across train/test/valid in Roboflow.
    assert summary["multi_split_scene_bases"]

    by_scene = {item.parsed.date: item for item in items}
    assert set(by_scene) == {"2021-01-03", "2023-04-13"}
    # The representative is the copy with the most labelled boxes.
    assert by_scene["2021-01-03"].class_counts == {"ship": 1, "ships": 1}
    assert by_scene["2023-04-13"].extra["render_profile"] == "true_color"
    # Every Madhulika image belongs to one Texas Gulf scene-date group.
    assert by_scene["2021-01-03"].group == "texas_gulf|2021-01-03"


def test_review_collection_keeps_negatives_and_drops_neither(tmp_path: Path) -> None:
    builder = load_builder()
    reviews = tmp_path / "reviews"
    cache = tmp_path / "cache" / "sentinel_tiles"
    _write_image(cache / "2020-02-08_tile_0_0.png")
    _write_image(cache / "2020-02-08_tile_1_1.png")

    def review(key: str, labels: list[str]) -> dict:
        return {
            "image_key": key,
            "image_filename": f"{key}.png",
            "cache_path": str(cache / f"{key}.png"),
            "review_status": "completed",
            "detections": [
                {
                    "review_label": label,
                    "x_center": 0.5,
                    "y_center": 0.5,
                    "width": 0.1,
                    "height": 0.1,
                }
                for label in labels
            ],
        }

    reviews.mkdir(parents=True)
    (reviews / "a.json").write_text(json.dumps(review("2020-02-08_tile_0_0", ["ship_non_sts", "ships_sts", "neither"])))
    (reviews / "b.json").write_text(json.dumps(review("2020-02-08_tile_1_1", [])))
    incomplete = review("2020-02-08_tile_0_0", ["ship_non_sts"])
    incomplete["review_status"] = "overview_1"
    incomplete["image_key"] = "2020-02-08_tile_9_9"
    (reviews / "c.json").write_text(json.dumps(incomplete))

    items, summary = builder.collect_review_items(reviews, tmp_path / "missing_manifest.json")

    assert summary["completed_reviews_used"] == 2
    assert summary["neither_boxes_dropped"] == 1
    assert summary["pure_negative_images"] == 1
    assert summary["skipped"] == {"not_completed": 1}
    # Both tiles are the same Gulf of Mexico observation, so they share one group.
    assert {item.group for item in items} == {"texas_gulf|2020-02-08"}
    assert sum(item.class_counts.get("ships", 0) for item in items) == 1


def test_region_for_import_entry_falls_back_to_source_path_then_album() -> None:
    builder = load_builder()
    manifest = {
        "with_bounds": {
            "album_ids": ["alb_sing"],
            "render": {"metadata": {"bounds": {"left": 103.9, "right": 104.0, "bottom": 1.1, "top": 1.2}}},
        }
    }
    album_regions = builder.album_region_index(manifest)
    assert album_regions == {"alb_sing": "singapore"}

    by_source_path = {
        "album_ids": [],
        "render": {"metadata": {"width": 10}},
        "source_path": "/home/gavin/Downloads/AIS data/Sentinel 2A/2020/ship_tiles new Apr 29/2021/2021-07-22_tile_5_10.tiff",
    }
    assert builder.region_for_import_entry(by_source_path, album_regions) == "texas_gulf"

    by_album = {"album_ids": ["alb_sing"], "render": {"metadata": {}}, "source_path": "/tmp/staging/x.tiff"}
    assert builder.region_for_import_entry(by_album, album_regions) == "singapore"

    unknown = {"album_ids": [], "render": {"metadata": {}}, "source_path": "/tmp/staging/x.tiff"}
    assert builder.region_for_import_entry(unknown, album_regions) == "unknown"
