import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import detect, pipeline, sts_utils, workspace
from scripts.correlate_ais import correlate_detections
from test_operational_pipeline import _detection, write_ais_csv, write_pil_geotiff


def test_stock_coco_model_is_rejected_before_inference(tmp_path):
    model = SimpleNamespace(names={0: "person", 1: "bicycle", 8: "boat"})
    with pytest.raises(ValueError, match="ships"):
        detect.detect_image(tmp_path / "missing.png", model)


def test_detection_ids_differ_for_same_filename_in_two_folders(tmp_path):
    scalar = lambda value: SimpleNamespace(item=lambda: value)
    box = SimpleNamespace(cls=scalar(1), conf=scalar(.9), xywh=[SimpleNamespace(tolist=lambda: [16, 16, 5, 5])])
    model = SimpleNamespace(names={0: "ship", 1: "ships"}, predict=lambda **kwargs: [SimpleNamespace(names={1: "ships"}, boxes=[box])])
    ids = []
    for folder in ["one", "two"]:
        path = tmp_path / folder / "2020-01-04_tile.png"
        path.parent.mkdir()
        Image.new("RGB", (32, 32)).save(path)
        ids.append(detect.detect_image(path, model)[0]["detection_id"])
    assert ids[0] != ids[1]


@pytest.mark.parametrize("content", ["", "latitude,longitude\n1,2\n", "MMSI,BaseDateTime,LAT,LON\n", "MMSI,BaseDateTime,LAT,LON\n1,2020-01-04T00:00:00,nan,2\n"])
def test_invalid_ais_never_becomes_fully_dark(tmp_path, content):
    (tmp_path / "AIS_2020_01_04.csv").write_text(content)
    result = correlate_detections([_detection("one", 1, 2)], tmp_path)[0]
    assert result["dark_status"] == "AIS_INVALID"
    assert result["ais_count"] is None


def test_missing_ais_and_ambiguous_ais_are_explicit(tmp_path):
    detections = [_detection("one", 1, 2)]
    assert correlate_detections(detections, tmp_path)[0]["dark_status"] == "AIS_MISSING"
    for folder in ["one", "two"]:
        write_ais_csv(tmp_path / folder / "AIS_2020_01_04.csv", [])
    with pytest.raises(ValueError, match="Multiple AIS files"):
        correlate_detections(detections, tmp_path)


def test_unknown_geotiff_crs_is_refused(tmp_path):
    image = tmp_path / "2020-01-04_tile.tiff"
    Image.new("RGB", (32, 32)).save(image, tiffinfo={33550: (.01, .01, 0.0), 33922: (0.0, 0.0, 0.0, 1.0, 2.0, 0.0)})
    with pytest.raises(ValueError, match="EPSG:4326"):
        sts_utils.load_bbox_for_image(image)


def test_current_model_tracks_promotion_and_refuses_modified_weight(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    workspace.set_current_model(first)
    assert workspace.current_model(tmp_path) == first
    workspace.set_current_model(second)
    assert workspace.current_model(tmp_path) == second
    second.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        workspace.current_model(tmp_path)


def test_zero_detection_run_still_records_provenance(tmp_path, monkeypatch):
    image = tmp_path / "2020-01-04_tile.tiff"
    write_pil_geotiff(image, 32, 32, (1.0, 1.0, 2.0, 2.0))
    weight = tmp_path / "fake.pt"
    weight.write_bytes(b"weight")
    monkeypatch.setattr(pipeline, "detect_images", lambda **kwargs: [])
    record = tmp_path / "run.json"
    assert pipeline.run_pipeline([str(image)], weights=weight, record_path=record) == []
    payload = json.loads(record.read_text())
    assert payload["weight_sha256"] == workspace.sha256(weight)
    assert payload["inputs"] == [str(image)]
    assert payload["status"] == "completed"


def test_area_filtered_ais_requires_coverage_of_entire_radius(tmp_path):
    from scripts.correlate_ais import correlate_detections
    from scripts.workspace import atomic_json, sha256
    csv = tmp_path / 'AIS_2020_01_04.csv'
    csv.write_text('MMSI,BaseDateTime,LAT,LON\n123,2020-01-04T12:00:00,29.15,-94.65\n')
    atomic_json(csv.with_suffix('.csv.json'), {'complete': True, 'identity': {'bbox': [-94.66,29.14,-94.64,29.16]}, 'sha256': sha256(csv)})
    result = correlate_detections([{'date':'2020-01-04', 'center_lat':29.15, 'center_lon':-94.65}], csv, radius_m=5000)[0]
    assert result['dark_status'] == 'AIS_INVALID'
    assert result['ais_count'] is None
    assert 'full search radius' in result['ais_error']
