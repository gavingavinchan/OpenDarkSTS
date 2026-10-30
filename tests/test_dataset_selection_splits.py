from __future__ import annotations

import json
from pathlib import Path

from _helpers import detection, load_training_lib, write_json, write_review, write_selection


def scene_manifest(tmp_path: Path, image_keys_to_names: dict[str, str]) -> Path:
    """Import manifest whose entries all share one region, so the date drives the group key."""
    images = {}
    for image_key, name in image_keys_to_names.items():
        images[image_key] = {
            "schema_version": 1,
            "image_key": image_key,
            "original_filename": name,
            "browser_relative_path": name,
            "source_path": "/Sentinel 2A/sentinel2/" + name,
            "album_ids": [],
        }
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    write_json(manifest_path, {"schema_version": 1, "images": images})
    return manifest_path


def test_auto_split_items_keeps_scene_groups_inside_one_split() -> None:
    training_lib = load_training_lib()
    # Three dates, four tiles each: an image-level shuffle splits every date.
    keys = []
    group_keys = {}
    for date in ("2020-01-14", "2020-02-15", "2020-03-16"):
        for tile in range(4):
            key = f"{date}_tile_{tile}"
            keys.append(key)
            group_keys[key] = f"texas_gulf|{date}"

    items = training_lib.auto_split_items(keys, seed=42, group_keys=group_keys, split_mode="group")
    summary = training_lib.split_summary_for_items(items)

    assert summary["groups_spanning_splits"] == 0
    assert summary["group_count"] == 3
    assert {item["group_key"] for item in items} == set(group_keys.values())

    legacy = training_lib.auto_split_items(keys, seed=42, group_keys=group_keys, split_mode="image")
    legacy_summary = training_lib.split_summary_for_items(legacy)
    assert legacy_summary["groups_spanning_splits"] > 0  # the escape hatch still leaks, by design


def test_save_dataset_selection_defaults_to_group_aware_splits(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    names = {}
    for date in ("2020-01-14", "2020-02-15", "2020-03-16"):
        for tile in range(4):
            key = f"img_{date}_{tile}"
            names[key] = f"sentinel2__S2_{date}_tile_{tile}_0.png"
            write_review(reviews_dir, cache_dir, reviewer="tester", image_key=key,
                         image_filename=names[key], detections=[detection("auto-0001", "ships_sts")])
    manifest_path = scene_manifest(tmp_path, names)

    result = training_lib.save_dataset_selection(
        dataset_name="Group Fixture",
        dataset_version="v1",
        image_keys=list(names),
        selections_dir=tmp_path / "dataset_selections",
        datasets_dir=tmp_path / "datasets",
        reviews_dir=reviews_dir,
        manifest_path=manifest_path,
        reviewer="tester",
    )

    selection = result["selection"]
    assert selection["split_policy"]["split_mode"] == "group"
    assert selection["split_policy"]["group_key_rule"] == "<region>|<date>"
    groups = {item["group_key"] for item in selection["items"]}
    assert groups == {"texas_gulf|2020-01-14", "texas_gulf|2020-02-15", "texas_gulf|2020-03-16"}
    assert result["split_summary"]["groups_spanning_splits"] == 0
    assert selection["split_summary"]["group_count"] == 3


def test_save_dataset_selection_image_mode_is_still_available(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    names = {}
    for tile in range(8):
        key = f"img_2020-01-14_{tile}"
        names[key] = f"sentinel2__S2_2020-01-14_tile_{tile}_0.png"
        write_review(reviews_dir, cache_dir, reviewer="tester", image_key=key, image_filename=names[key])
    manifest_path = scene_manifest(tmp_path, names)

    result = training_lib.save_dataset_selection(
        dataset_name="Image Mode Fixture",
        image_keys=list(names),
        selections_dir=tmp_path / "dataset_selections",
        datasets_dir=tmp_path / "datasets",
        reviews_dir=reviews_dir,
        manifest_path=manifest_path,
        reviewer="tester",
        split_mode="image",
    )

    assert result["selection"]["split_policy"]["split_mode"] == "image"
    # One date, eight tiles: the legacy mode spreads that single group over splits.
    assert result["split_summary"]["groups_spanning_splits"] >= 1


def test_preview_dataset_split_reports_group_counts(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    names = {}
    for date in ("2020-01-14", "2020-02-15", "2020-03-16", "2020-04-17"):
        for tile in range(3):
            key = f"img_{date}_{tile}"
            names[key] = f"sentinel2__S2_{date}_tile_{tile}_0.png"
            write_review(reviews_dir, cache_dir, reviewer="tester", image_key=key, image_filename=names[key])
    manifest_path = scene_manifest(tmp_path, names)

    preview = training_lib.preview_dataset_split(
        list(names), seed=42, reviews_dir=reviews_dir, manifest_path=manifest_path, reviewer="tester"
    )

    assert preview["split_mode"] == "group"
    assert preview["group_count"] == 4
    assert preview["groups_spanning_splits"] == 0
    assert sum(preview["split_counts"].values()) == 12


def test_export_dataset_selection_clears_obsolete_images_and_splits(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    dataset_dir = tmp_path / "datasets" / "fixture_dataset"
    for key in ("img_a", "img_b", "img_c"):
        write_review(reviews_dir, cache_dir, reviewer="gavin", image_key=key, detections=[detection("auto-0001", "ships_sts")])
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=reviews_dir,
        output_dir=dataset_dir,
        items=[
            {"image_key": "img_a", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None},
            {"image_key": "img_b", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None},
            {"image_key": "img_c", "split": "val", "include_labels": "all_exportable", "selected_detection_ids": None},
        ],
    )
    training_lib.export_dataset_selection(selection_path)

    # Drop img_b and move img_c from val to test, then re-export to the same directory.
    selection = json.loads(Path(selection_path).read_text())
    selection["items"] = [
        {"image_key": "img_a", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None},
        {"image_key": "img_c", "split": "test", "include_labels": "all_exportable", "selected_detection_ids": None},
    ]
    Path(selection_path).write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    manifest = training_lib.export_dataset_selection(selection_path)

    on_disk = sorted(
        str(path.relative_to(dataset_dir))
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.parts[-3] in {"images", "labels"}
    )
    assert on_disk == ["images/test/img_c.png", "images/train/img_a.png",
                       "labels/test/img_c.txt", "labels/train/img_a.txt"]
    assert manifest["exported_image_keys"] == ["img_a", "img_c"]
    assert not (dataset_dir / "images" / "val" / "img_c.png").exists()


def test_check_training_readiness_rejects_a_stale_export(tmp_path: Path) -> None:
    from _helpers import write_training_config

    training_lib = load_training_lib()
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    dataset_dir = tmp_path / "datasets" / "fixture_dataset"
    for key in ("img_train", "img_val", "img_extra"):
        write_review(reviews_dir, cache_dir, reviewer="gavin", image_key=key, detections=[detection("auto-0001", "ships_sts")])
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=reviews_dir,
        output_dir=dataset_dir,
        items=[
            {"image_key": "img_train", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None},
            {"image_key": "img_val", "split": "val", "include_labels": "all_exportable", "selected_detection_ids": None},
            {"image_key": "img_extra", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None},
        ],
    )
    training_lib.export_dataset_selection(selection_path)
    config_path = write_training_config(tmp_path / "training_configs" / "fixture_group.json", output_dir=tmp_path / "runs")
    assert training_lib.check_training_readiness(selection_path, config_path)["ready"] is True

    selection = json.loads(Path(selection_path).read_text())
    selection["items"] = [item for item in selection["items"] if item["image_key"] != "img_extra"]
    Path(selection_path).write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")

    result = training_lib.check_training_readiness(selection_path, config_path)
    assert result["ready"] is False
    assert any("out of date" in item for item in result["missing"])
    assert training_lib.export_selection_is_current(selection_path, dataset_dir) is False

    training_lib.export_dataset_selection(selection_path)
    assert training_lib.check_training_readiness(selection_path, config_path)["ready"] is True


def test_list_dataset_selections_flags_a_stale_export(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    dataset_dir = tmp_path / "datasets" / "fixture_dataset"
    for key in ("img_a", "img_b"):
        write_review(reviews_dir, cache_dir, reviewer="gavin", image_key=key, detections=[detection("auto-0001", "ships_sts")])
    selections_dir = tmp_path / "dataset_selections"
    selection_path = write_selection(
        selections_dir / "fixture_dataset.json",
        reviews_dir=reviews_dir,
        output_dir=dataset_dir,
        items=[
            {"image_key": "img_a", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None},
            {"image_key": "img_b", "split": "val", "include_labels": "all_exportable", "selected_detection_ids": None},
        ],
    )
    training_lib.export_dataset_selection(selection_path)
    assert training_lib.list_dataset_selections(selections_dir)[0]["export_is_current"] is True

    selection = json.loads(Path(selection_path).read_text())
    selection["items"][1]["split"] = "test"
    Path(selection_path).write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")

    listed = training_lib.list_dataset_selections(selections_dir)[0]
    assert listed["exported"] is True
    assert listed["export_is_current"] is False


def test_list_exported_datasets_finds_nested_datasets_and_skips_raw(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    datasets_dir = tmp_path / "datasets"
    for relative in ("top_level", "derived/sts_clean_v1", "raw/should_be_ignored", "extracted/also_ignored"):
        dataset_dir = datasets_dir / relative
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)
        (dataset_dir / "images" / "train" / "a.png").write_bytes(b"x")
        (dataset_dir / "data.yaml").write_text("names:\n  0: ship\n  1: ships\n")
        write_json(dataset_dir / "dataset_manifest.json", {
            "schema_version": 1,
            "dataset_name": relative.replace("/", "_"),
            "dataset_version": "v1",
            "exported_image_keys": ["a"],
        })

    names = {item["dataset_name"] for item in training_lib.list_exported_datasets(datasets_dir)}

    assert names == {"top_level", "derived_sts_clean_v1"}


def test_legacy_exported_dataset_count_uses_files_on_disk(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    dataset_dir = tmp_path / 'datasets' / 'legacy'
    for split in ['train', 'val', 'test']:
        directory = dataset_dir / 'images' / split
        directory.mkdir(parents=True)
        (directory / 'fixture.png').write_bytes(b'fixture')
    # Older derived datasets do not have exported_image_keys.
    write_json(dataset_dir / 'dataset_manifest.json', {'dataset_name': 'legacy'})
    record = training_lib.list_exported_datasets(tmp_path / 'datasets')[0]
    assert record['image_count'] == 3
    assert record['split_counts'] == {'train': 1, 'val': 1, 'test': 1}
