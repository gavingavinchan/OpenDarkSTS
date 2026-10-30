from __future__ import annotations

from pathlib import Path

from _helpers import load_review_lib, write_json


def test_mark_manual_detection_neither_preserves_box_for_review_visibility() -> None:
    review_lib = load_review_lib()
    review = {
        "image_key": "img_abc123",
        "review_status": "overview_1",
        "detail_cursor": None,
        "detections": [
            {
                "detection_id": "manual-0001",
                "source_type": "manual",
                "class_id": 0,
                "class_name": "ship",
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.2,
                "height": 0.1,
                "review_label": "ship_non_sts",
                "reviewed": False,
            }
        ],
    }

    review_lib.mark_detection_neither(review, "manual-0001")

    assert len(review["detections"]) == 1
    detection = review["detections"][0]
    assert detection["detection_id"] == "manual-0001"
    assert detection["review_label"] == "neither"
    assert detection["reviewed"] is True
    assert review_lib.active_detection(detection) is False


def test_delete_detection_removes_manual_box() -> None:
    review_lib = load_review_lib()
    review = {
        "image_key": "img_abc123",
        "review_status": "overview_1",
        "detail_cursor": None,
        "detections": [
            {
                "detection_id": "manual-0001",
                "source_type": "manual",
                "class_id": 0,
                "class_name": "ship",
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.2,
                "height": 0.1,
                "review_label": "ship_non_sts",
                "reviewed": False,
            }
        ],
    }

    review_lib.delete_detection(review, "manual-0001")

    assert review["detections"] == []


def test_geometry_update_without_class_preserves_saved_class() -> None:
    review_lib = load_review_lib()
    review = {
        "image_key": "img_abc123",
        "review_status": "overview_1",
        "detail_cursor": None,
        "detections": [
            {
                "detection_id": "auto-0001",
                "source_type": "auto",
                "class_id": 1,
                "class_name": "ships",
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.2,
                "height": 0.1,
                "review_label": "ships_sts",
                "reviewed": True,
            }
        ],
    }

    review_lib.update_detection_geometry(
        review,
        "auto-0001",
        x_center=0.55,
        y_center=0.45,
        width=0.25,
        height=0.15,
        class_id=None,
        user_edited=True,
    )

    detection = review["detections"][0]
    assert detection["class_id"] == 1
    assert detection["class_name"] == "ships"
    assert detection["review_label"] == "ships_sts"
    assert detection["x_center"] == 0.55


def test_tab_class_save_does_not_advance_detail_cursor() -> None:
    review_lib = load_review_lib()
    review = {
        "image_key": "img_abc123",
        "review_status": "detail",
        "detail_cursor": 0,
        "detections": [
            {
                "detection_id": "auto-0001",
                "source_type": "auto",
                "class_id": 0,
                "class_name": "ship",
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.2,
                "height": 0.1,
                "review_label": None,
                "reviewed": False,
            },
            {
                "detection_id": "auto-0002",
                "source_type": "auto",
                "class_id": 0,
                "class_name": "ship",
                "x_center": 0.7,
                "y_center": 0.7,
                "width": 0.2,
                "height": 0.1,
                "review_label": None,
                "reviewed": False,
            },
        ],
    }

    review_lib.set_detection_review_label(
        review,
        "auto-0001",
        "ships_sts",
        1,
        x_center=0.5,
        y_center=0.5,
        width=0.2,
        height=0.1,
    )

    assert review["review_status"] == "detail"
    assert review["detail_cursor"] == 0
    assert review["detections"][0]["review_label"] == "ships_sts"
    assert review["detections"][0]["class_id"] == 1


def test_review_document_enumeration_skips_underscore_archive_dirs(tmp_path: Path) -> None:
    review_lib = load_review_lib()
    active = tmp_path / "reviews" / "tester" / "img_active.json"
    archived = tmp_path / "reviews" / "_archived_unreviewed" / "2026-05-04T00-00-00-000Z" / "tester" / "img_archived.json"
    write_json(active, {"image_key": "img_active", "review_status": "overview_1", "detections": []})
    write_json(archived, {"image_key": "img_archived", "review_status": "overview_1", "detections": []})

    listed = review_lib.list_review_documents("tester", tmp_path / "reviews")
    collected = review_lib.collect_review_documents(tmp_path / "reviews")
    grouped = review_lib.review_file_groups(tmp_path / "reviews")

    assert [review["image_key"] for review in listed] == ["img_active"]
    assert [review["image_key"] for review in collected] == ["img_active"]
    assert sorted(grouped) == ["img_active"]
