from __future__ import annotations

from _helpers import load_review_lib


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
