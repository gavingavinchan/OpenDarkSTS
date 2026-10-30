from __future__ import annotations

import importlib
from collections import Counter

import pytest


def load_splits():
    try:
        return importlib.import_module("scripts.dataset_splits")
    except ModuleNotFoundError:  # pragma: no cover - red state helper
        pytest.fail("scripts.dataset_splits must provide group-aware split helpers")


def test_parse_source_name_handles_plain_tile_names() -> None:
    splits = load_splits()
    parsed = splits.parse_source_name("2020-01-04_tile_0_1.png")
    assert parsed.date == "2020-01-04"
    assert parsed.scene == "scene0"
    assert (parsed.tile_row, parsed.tile_col) == (0, 1)


def test_parse_source_name_handles_scene_and_collection_prefix() -> None:
    splits = load_splits()
    parsed = splits.parse_source_name(
        "sentinel2_ship_tiles_new_apr29_2025__2025-06-23_scene1_tile_28_9.png"
    )
    assert parsed.collection == "sentinel2_ship_tiles_new_apr29_2025"
    assert parsed.date == "2025-06-23"
    assert parsed.scene == "scene1"
    assert (parsed.tile_row, parsed.tile_col) == (28, 9)


def test_parse_source_name_strips_roboflow_and_duplicate_markers() -> None:
    splits = load_splits()
    a = splits.parse_source_name(
        "2021-01-03-00_00_2021-01-03-23_59_Sentinel-2_L2A_True_color_jpg.rf.deadbeef01.jpg"
    )
    b = splits.parse_source_name(
        "2021-01-03-00_00_2021-01-03-23_59_Sentinel-2_L2A_True_color-1-_jpg.rf.0badc0de99.jpg"
    )
    assert a.date == b.date == "2021-01-03"
    assert a.base == b.base
    assert a.render_profile == "true_color"


def test_parse_source_name_treats_false_and_true_color_as_one_scene() -> None:
    splits = load_splits()
    true_color = splits.parse_source_name(
        "2023-04-13-00_00_2023-04-13-23_59_Sentinel-2_L2A_True_color_jpg.rf.aaa111.jpg"
    )
    false_color = splits.parse_source_name(
        "2023-04-13-00_00_2023-04-13-23_59_Sentinel-2_L2A_False_color_jpg.rf.bbb222.jpg"
    )
    assert true_color.base == false_color.base
    assert true_color.render_profile == "true_color"
    assert false_color.render_profile == "false_color"
    assert splits.group_key(true_color, "texas_gulf") == splits.group_key(false_color, "texas_gulf")


def test_parse_source_name_strips_zoom_qualifiers() -> None:
    splits = load_splits()
    plain = splits.parse_source_name(
        "2020-01-14-00_00_2020-01-14-23_59_Sentinel-2_L2A_True_color.tiff"
    )
    zoomed = splits.parse_source_name(
        "2020-01-14-00_00_2020-01-14-23_59_Sentinel-2_L2A_True_color zoomed-out_highres.tiff"
    )
    assert plain.base == zoomed.base


def test_parse_source_name_without_date_falls_back_to_base_key() -> None:
    splits = load_splits()
    parsed = splits.parse_source_name("img_00445766ca66d70f.png")
    assert parsed.date is None
    key = splits.group_key(parsed, "unknown")
    assert key.endswith("img_00445766ca66d70f")
    other = splits.group_key_for_name("img_ffffffffffffffff.png")
    assert key != other


def test_group_key_merges_tiles_of_the_same_region_and_date() -> None:
    splits = load_splits()
    keys = {
        splits.group_key_for_name("2020-02-08_tile_0_0.png", "texas_gulf"),
        splits.group_key_for_name("2020-02-08_tile_1_1.png", "texas_gulf"),
        splits.group_key_for_name(
            "2020-02-08-00_00_2020-02-08-23_59_Sentinel-2_L2A_True_color_jpg.rf.abc.jpg",
            "texas_gulf",
        ),
    }
    assert keys == {"texas_gulf|2020-02-08"}


def test_group_key_separates_regions_on_the_same_date() -> None:
    splits = load_splits()
    suez = splits.group_key_for_name("2025-07-01_tile_2_1.tiff", "suez")
    gibraltar = splits.group_key_for_name("2025-07-01_tile_2_1.tiff", "gibraltar")
    assert suez != gibraltar


def test_group_key_uses_scene_only_when_requested() -> None:
    splits = load_splits()
    a = "2020-01-24_tile_0_0.png"
    b = "2020-01-24_scene1_tile_0_3.png"
    assert splits.group_key_for_name(a, "texas_gulf") == splits.group_key_for_name(b, "texas_gulf")
    assert splits.group_key_for_name(a, "texas_gulf", use_scene=True) != splits.group_key_for_name(
        b, "texas_gulf", use_scene=True
    )


def test_region_lookup_from_bounds() -> None:
    splits = load_splits()
    assert splits.region_for_bounds({"left": -94.73, "right": -94.57, "bottom": 29.07, "top": 29.2}) == "texas_gulf"
    assert splits.region_for_bounds({"left": 32.565, "right": 32.6175, "bottom": 29.71, "top": 29.76}) == "suez"
    assert splits.region_for_bounds({"left": -5.4, "right": -5.3, "bottom": 35.9, "top": 36.0}) == "gibraltar"
    assert splits.region_for_bounds({"left": 103.9, "right": 104.0, "bottom": 1.1, "top": 1.2}) == "singapore"
    assert splits.region_for_bounds(None) == "unknown"
    assert splits.region_for_bounds((-94.73, 29.07, -94.57, 29.2)) == "texas_gulf"
    assert splits.region_for_bounds({"left": 12.0, "right": 12.1, "bottom": 55.0, "top": 55.1}).startswith("lon12_lat55")


def _make_groups(splits, spec):
    return [
        splits.GroupStats(key=key, images=images, boxes=Counter(boxes))
        for key, images, boxes in spec
    ]


def test_assignment_is_deterministic_and_seed_sensitive() -> None:
    splits = load_splits()
    spec = [(f"r|2020-01-{index:02d}", 3 + index % 4, {"ship": index, "ships": index % 3}) for index in range(1, 41)]
    groups = _make_groups(splits, spec)
    first = splits.assign_groups_to_splits(groups, seed=42)
    second = splits.assign_groups_to_splits(list(reversed(groups)), seed=42)
    assert first == second
    other = splits.assign_groups_to_splits(groups, seed=7)
    assert set(first) == set(other)
    assert first != other


def test_assignment_keeps_whole_groups_and_has_no_overlap() -> None:
    splits = load_splits()
    spec = [(f"r|2020-01-{index:02d}", 4, {"ship": 5, "ships": 2}) for index in range(1, 31)]
    groups = _make_groups(splits, spec)
    assignment = splits.assign_groups_to_splits(groups, seed=42)
    assert set(assignment) == {stats.key for stats in groups}
    assert set(assignment.values()) <= {"train", "val", "test"}

    item_groups = {}
    for stats in groups:
        for index in range(stats.images):
            item_groups[f"{stats.key}#{index}"] = (assignment[stats.key], stats.key)
    report = splits.check_no_group_overlap(item_groups)
    assert report["ok"] is True
    assert report["overlapping_groups"] == {}
    assert report["group_count"] == 30


def test_check_no_group_overlap_detects_a_leak() -> None:
    splits = load_splits()
    report = splits.check_no_group_overlap(
        {
            "a": ("train", "texas_gulf|2020-02-08"),
            "b": ("test", "texas_gulf|2020-02-08"),
            "c": ("val", "suez|2025-07-01"),
        }
    )
    assert report["ok"] is False
    assert report["overlapping_groups"] == {"texas_gulf|2020-02-08": ["test", "train"]}


def test_assignment_respects_target_ratios_roughly() -> None:
    splits = load_splits()
    spec = [(f"r|2020-{index // 28 + 1:02d}-{index % 28 + 1:02d}", 5, {"ship": 8, "ships": 3}) for index in range(100)]
    groups = _make_groups(splits, spec)
    assignment = splits.assign_groups_to_splits(groups, seed=42)
    counts = Counter(assignment.values())
    assert 60 <= counts["train"] <= 80
    assert 8 <= counts["val"] <= 22
    assert 8 <= counts["test"] <= 22


def test_min_requirements_force_extra_groups_into_test() -> None:
    splits = load_splits()
    spec = [(f"r|2020-01-{index:02d}", 2, {"ship": 4, "ships": 1}) for index in range(1, 41)]
    groups = _make_groups(splits, spec)
    by_key = {stats.key: stats for stats in groups}
    assignment = splits.assign_groups_to_splits(
        groups, seed=42, min_requirements={"test": {"ships": 20}}
    )
    test_ships = sum(by_key[key].boxes["ships"] for key, split in assignment.items() if split == "test")
    assert test_ships >= 20


def test_min_requirements_stop_when_impossible() -> None:
    splits = load_splits()
    groups = _make_groups(splits, [(f"r|2020-01-{index:02d}", 1, {"ships": 1}) for index in range(1, 6)])
    assignment = splits.assign_groups_to_splits(
        groups, seed=42, min_requirements={"test": {"ships": 999}}
    )
    assert len(assignment) == 5
