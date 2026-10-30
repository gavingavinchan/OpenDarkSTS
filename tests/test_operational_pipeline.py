"""Tests for the operational (satellite-first) pipeline: detect -> georeference -> AIS."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:  # pragma: no cover - red state helper
        pytest.fail(f"{name} must be importable without ultralytics installed")


def write_pil_geotiff(path: Path, width: int, height: int, bbox: tuple[float, float, float, float]) -> None:
    """Write a minimal GeoTIFF using PIL tags (the rasterio-free fallback path)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    scale_x = (lon_max - lon_min) / width
    scale_y = (lat_max - lat_min) / height
    image = Image.new("RGB", (width, height), (10, 20, 30))
    image.save(
        path,
        tiffinfo={
            33550: (scale_x, scale_y, 0.0),
            33922: (0.0, 0.0, 0.0, lon_min, lat_max, 0.0),
            34735: (1, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326),
        },
    )


def write_ais_csv(path: Path, rows: list[dict[str, str]]) -> None:
    header = [
        "MMSI",
        "BaseDateTime",
        "LAT",
        "LON",
        "SOG",
        "COG",
        "Heading",
        "VesselName",
        "IMO",
        "CallSign",
        "VesselType",
        "Status",
        "Length",
        "Width",
        "Draft",
        "Cargo",
        "TransceiverClass",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(row.get(column, "") for column in header))
    path.write_text("\n".join(lines) + "\n")


# --- georeferencing math -------------------------------------------------------------


def test_pixel_to_lat_lon_maps_corners_and_center() -> None:
    sts_utils = load_module("scripts.sts_utils")
    bbox = (-94.73, 29.07, -94.57, 29.205)

    center_lat, center_lon = sts_utils.pixel_to_lat_lon(50.0, 25.0, 100, 50, bbox)
    assert center_lat == pytest.approx(29.1375)
    assert center_lon == pytest.approx(-94.65)

    top_left_lat, top_left_lon = sts_utils.pixel_to_lat_lon(0.0, 0.0, 100, 50, bbox)
    assert top_left_lat == pytest.approx(29.205)
    assert top_left_lon == pytest.approx(-94.73)

    bottom_right_lat, bottom_right_lon = sts_utils.pixel_to_lat_lon(100.0, 50.0, 100, 50, bbox)
    assert bottom_right_lat == pytest.approx(29.07)
    assert bottom_right_lon == pytest.approx(-94.57)


def test_pixel_to_lat_lon_rejects_empty_image() -> None:
    sts_utils = load_module("scripts.sts_utils")
    with pytest.raises(ValueError):
        sts_utils.pixel_to_lat_lon(1.0, 1.0, 0, 10, (0.0, 0.0, 1.0, 1.0))


def test_georeference_detection_uses_geotiff_tags(tmp_path: Path) -> None:
    georeference = load_module("scripts.georeference")
    bbox = (-94.73, 29.07, -94.57, 29.205)
    tiff_path = tmp_path / "2020-01-04_tile_0_0.tiff"
    write_pil_geotiff(tiff_path, 200, 100, bbox)

    detection = {
        "detection_id": "2020-01-04_tile_0_0:1",
        "image_path": str(tiff_path),
        "date": "2020-01-04",
        "cx_px": 100.0,
        "cy_px": 50.0,
        "w_px": 10.0,
        "h_px": 10.0,
        "image_width_px": 200,
        "image_height_px": 100,
    }
    enriched = georeference.georeference_detection(detection)
    assert enriched["center_lon"] == pytest.approx(-94.65, abs=1e-6)
    assert enriched["center_lat"] == pytest.approx(29.1375, abs=1e-6)


@pytest.mark.skipif(importlib.util.find_spec("rasterio") is None, reason="rasterio is not installed in this interpreter")
def test_georeference_detection_matches_rasterio_bounds(tmp_path: Path) -> None:
    import numpy
    import rasterio
    from rasterio.transform import from_bounds

    georeference = load_module("scripts.georeference")
    bbox = (-94.73, 29.07, -94.57, 29.205)
    tiff_path = tmp_path / "2020-01-04_tile_1_1.tiff"
    width, height = 200, 100
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
    data = numpy.zeros((3, height, width), dtype="uint8")
    with rasterio.open(
        tiff_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dataset:
        dataset.write(data)

    detection = {
        "detection_id": "2020-01-04_tile_1_1:1",
        "image_path": str(tiff_path),
        "date": "2020-01-04",
        "cx_px": 50.0,
        "cy_px": 25.0,
        "w_px": 8.0,
        "h_px": 8.0,
        "image_width_px": width,
        "image_height_px": height,
    }
    enriched = georeference.georeference_detection(detection)
    assert enriched["center_lon"] == pytest.approx(-94.69, abs=1e-6)
    assert enriched["center_lat"] == pytest.approx(29.17125, abs=1e-6)


# --- AIS file resolution -------------------------------------------------------------


def test_resolve_ais_file_finds_dated_csv_under_directory(tmp_path: Path) -> None:
    sts_utils = load_module("scripts.sts_utils")
    target = tmp_path / "AIS data 2020" / "all_csv" / "AIS_2020_01_04.csv"
    write_ais_csv(target, [])
    assert sts_utils.resolve_ais_file(tmp_path, "2020-01-04") == target


def test_resolve_ais_file_accepts_direct_file_and_rejects_mismatch(tmp_path: Path) -> None:
    sts_utils = load_module("scripts.sts_utils")
    target = tmp_path / "AIS_2020_01_04.csv"
    write_ais_csv(target, [{"MMSI": "1", "BaseDateTime": "2020-01-04T00:00:00", "LAT": "1", "LON": "1"}])
    assert sts_utils.resolve_ais_file(target, "2020-01-04") == target
    with pytest.raises(FileNotFoundError):
        sts_utils.resolve_ais_file(target, "2020-01-05")


def test_explicit_csv_date_is_established_by_contents(tmp_path: Path) -> None:
    sts_utils = load_module("scripts.sts_utils")
    target = tmp_path / "my-vessel-records.csv"
    write_ais_csv(target, [{"MMSI": "1", "BaseDateTime": "2020-01-04T00:00:00", "LAT": "1", "LON": "1"}])
    assert sts_utils.resolve_ais_file(target, "2020-01-04") == target
    write_ais_csv(target, [{"MMSI": "1", "BaseDateTime": "unknown", "LAT": "1", "LON": "1"}])
    with pytest.raises(ValueError, match="Cannot establish the AIS CSV date"):
        sts_utils.resolve_ais_file(target, "2020-01-04")


def test_malformed_explicit_csv_has_a_preflight_error(tmp_path: Path) -> None:
    sts_utils = load_module("scripts.sts_utils")
    target = tmp_path / "records.csv"
    target.write_text('MMSI,BaseDateTime,LAT,LON\n1,"unterminated')
    with pytest.raises(ValueError, match="Cannot read selected AIS CSV"):
        sts_utils.ais_content_dates(target)


def test_resolve_ais_file_raises_when_date_is_missing(tmp_path: Path) -> None:
    sts_utils = load_module("scripts.sts_utils")
    with pytest.raises(FileNotFoundError):
        sts_utils.resolve_ais_file(tmp_path, "2020-01-04")


# --- AIS correlation and dark status -------------------------------------------------


def _detection(detection_id: str, lat: float, lon: float) -> dict:
    return {
        "detection_id": detection_id,
        "image_path": f"/tmp/{detection_id.split(':')[0]}.tiff",
        "date": "2020-01-04",
        "cx_px": 10.0,
        "cy_px": 10.0,
        "w_px": 6.0,
        "h_px": 6.0,
        "confidence": 0.9,
        "center_lat": lat,
        "center_lon": lon,
    }


def test_correlate_detections_reports_all_three_dark_statuses(tmp_path: Path) -> None:
    correlate_ais = load_module("scripts.correlate_ais")
    ais_path = tmp_path / "AIS_2020_01_04.csv"
    write_ais_csv(
        ais_path,
        [
            # two vessels near 29.10 / -94.70 -> AIS_PRESENT
            {
                "MMSI": "111111111",
                "BaseDateTime": "2020-01-04T16:30:00",
                "LAT": "29.1000",
                "LON": "-94.7000",
                "SOG": "0.1",
                "COG": "12.3",
                "VesselName": "ALPHA",
                "VesselType": "80",
                "Length": "183",
            },
            {
                "MMSI": "222222222",
                "BaseDateTime": "2020-01-04T16:31:00",
                "LAT": "29.1002",
                "LON": "-94.7003",
                "SOG": "0.2",
                "COG": "15.0",
                "VesselName": "BRAVO",
                "VesselType": "70",
                "Length": "200",
            },
            # one vessel near 29.20 / -94.60 -> PARTIALLY_DARK
            {
                "MMSI": "333333333",
                "BaseDateTime": "2020-01-04T16:32:00",
                "LAT": "29.2001",
                "LON": "-94.6001",
                "SOG": "0.0",
                "COG": "90.0",
                "VesselName": "CHARLIE",
                "VesselType": "31",
                "Length": "50",
            },
            # far away, should never match
            {
                "MMSI": "444444444",
                "BaseDateTime": "2020-01-04T16:33:00",
                "LAT": "25.0000",
                "LON": "-90.0000",
                "VesselName": "DELTA",
            },
            # unparsable row, should be skipped without crashing
            {"MMSI": "555555555", "LAT": "", "LON": ""},
        ],
    )

    detections = [
        _detection("tile_a:1", 29.1000, -94.7000),
        _detection("tile_b:1", 29.2000, -94.6000),
        _detection("tile_c:1", 28.5000, -95.5000),
    ]
    results = correlate_ais.correlate_detections(detections=detections, ais_dir=tmp_path, radius_m=1000.0)
    by_id = {result["detection_id"]: result for result in results}

    assert by_id["tile_a:1"]["dark_status"] == "AIS_PRESENT"
    assert by_id["tile_a:1"]["ais_count"] == 2
    assert by_id["tile_b:1"]["dark_status"] == "PARTIALLY_DARK"
    assert by_id["tile_b:1"]["ais_count"] == 1
    assert by_id["tile_c:1"]["dark_status"] == "FULLY_DARK"
    assert by_id["tile_c:1"]["ais_vessels"] == []
    assert by_id["tile_a:1"]["ais_file"] == str(ais_path)


def test_correlate_detections_records_ais_attributes_and_timestamp(tmp_path: Path) -> None:
    correlate_ais = load_module("scripts.correlate_ais")
    ais_path = tmp_path / "AIS_2020_01_04.csv"
    write_ais_csv(
        ais_path,
        [
            {
                "MMSI": "111111111",
                "BaseDateTime": "2020-01-04T16:30:00",
                "LAT": "29.1000",
                "LON": "-94.7000",
                "SOG": "0.1",
                "COG": "12.3",
                "VesselName": "ALPHA",
                "VesselType": "80",
                "Length": "183",
            }
        ],
    )
    results = correlate_ais.correlate_detections(
        detections=[_detection("tile_a:1", 29.1000, -94.7000)], ais_dir=tmp_path, radius_m=1000.0
    )
    vessel = results[0]["ais_vessels"][0]
    assert vessel["BaseDateTime"] == "2020-01-04T16:30:00"
    assert vessel["VesselType"] == "80"
    assert vessel["SOG"] == "0.1"
    assert vessel["COG"] == "12.3"
    assert vessel["Length"] == "183"
    assert vessel["distance_m"] == pytest.approx(0.0, abs=1.0)


def test_correlate_detections_keeps_the_closest_ping_per_vessel(tmp_path: Path) -> None:
    correlate_ais = load_module("scripts.correlate_ais")
    ais_path = tmp_path / "AIS_2020_01_04.csv"
    write_ais_csv(
        ais_path,
        [
            {
                "MMSI": "111111111",
                "BaseDateTime": "2020-01-04T16:00:00",
                "LAT": "29.1050",
                "LON": "-94.7000",
                "VesselName": "ALPHA",
            },
            {
                "MMSI": "111111111",
                "BaseDateTime": "2020-01-04T16:30:00",
                "LAT": "29.1001",
                "LON": "-94.7000",
                "VesselName": "ALPHA",
            },
        ],
    )
    results = correlate_ais.correlate_detections(
        detections=[_detection("tile_a:1", 29.1000, -94.7000)], ais_dir=tmp_path, radius_m=2000.0
    )
    assert results[0]["ais_count"] == 1
    assert results[0]["ais_vessels"][0]["BaseDateTime"] == "2020-01-04T16:30:00"


def test_flatten_pipeline_result_exposes_vessel_attributes() -> None:
    sts_utils = load_module("scripts.sts_utils")
    result = {
        "date": "2020-01-04",
        "image_path": "/tmp/2020-01-04_tile_0_0.tiff",
        "detection_id": "2020-01-04_tile_0_0:1",
        "center_lat": 29.1,
        "center_lon": -94.7,
        "confidence": 0.8,
        "ais_count": 1,
        "dark_status": "PARTIALLY_DARK",
        "preview_png": "/tmp/preview.png",
        "ais_vessels": [
            {
                "MMSI": "111111111",
                "VesselName": "ALPHA",
                "BaseDateTime": "2020-01-04T16:30:00",
                "VesselType": "80",
                "SOG": "0.1",
                "COG": "12.3",
                "Length": "183",
                "distance_m": 42.0,
            }
        ],
    }
    row = sts_utils.flatten_pipeline_result(result)
    assert set(row) == set(sts_utils.PIPELINE_CSV_FIELDS)
    assert row["image_file"] == "2020-01-04_tile_0_0.tiff"
    assert row["vessel_1_time"] == "2020-01-04T16:30:00"
    assert row["vessel_1_type"] == "80"
    assert row["vessel_1_length"] == "183"
    assert row["vessel_2_mmsi"] == ""


# --- image selection: zero-byte skip, unreadable skip, date filter, scene1 names -----


def test_filter_readable_images_skips_zero_byte_and_corrupt_files(tmp_path: Path) -> None:
    sts_utils = load_module("scripts.sts_utils")
    good = tmp_path / "2020-01-04_tile_0_0.tiff"
    write_pil_geotiff(good, 32, 32, (-94.73, 29.07, -94.57, 29.205))
    empty = tmp_path / "2020-01-04_tile_0_1.tiff"
    empty.write_bytes(b"")
    corrupt = tmp_path / "2020-01-04_tile_1_0.tiff"
    corrupt.write_bytes(b"not a tiff at all")

    warnings: list[str] = []
    readable = sts_utils.filter_readable_images([good, empty, corrupt], warn=warnings.append)
    assert readable == [good]
    assert len(warnings) == 2
    assert any("zero-byte" in message for message in warnings)


def test_select_images_skips_zero_byte_rasters_in_a_directory(tmp_path: Path) -> None:
    pipeline = load_module("scripts.pipeline")
    good = tmp_path / "2020-01-04_tile_0_0.tiff"
    write_pil_geotiff(good, 32, 32, (-94.73, 29.07, -94.57, 29.205))
    (tmp_path / "2020-01-04_tile_0_1.tiff").write_bytes(b"")

    warnings: list[str] = []
    selected = pipeline.select_images([str(tmp_path)], warn=warnings.append)
    assert selected == [good.resolve()]
    assert warnings and "zero-byte" in warnings[0]


def test_select_images_handles_scene1_names_and_date_filter(tmp_path: Path) -> None:
    pipeline = load_module("scripts.pipeline")
    bbox = (-94.73, 29.07, -94.57, 29.205)
    scene1 = tmp_path / "2020-01-24_scene1_tile_1_2.tiff"
    plain = tmp_path / "2020-01-24_tile_0_0.tiff"
    other_date = tmp_path / "2020-02-08_tile_0_0.tiff"
    for path in (scene1, plain, other_date):
        write_pil_geotiff(path, 32, 32, bbox)

    selected = pipeline.select_images([str(tmp_path)], date="2020-01-24")
    assert selected == sorted([scene1.resolve(), plain.resolve()])

    sts_utils = load_module("scripts.sts_utils")
    assert sts_utils.parse_image_date(scene1) == "2020-01-24"


def test_select_images_warns_on_files_without_a_date(tmp_path: Path) -> None:
    pipeline = load_module("scripts.pipeline")
    undated = tmp_path / "mystery_tile.tiff"
    write_pil_geotiff(undated, 32, 32, (-94.73, 29.07, -94.57, 29.205))
    warnings: list[str] = []
    assert pipeline.select_images([str(tmp_path)], warn=warnings.append) == []
    assert warnings and "date" in warnings[0]


# --- previews and end-to-end wiring with YOLO mocked --------------------------------


def test_write_detection_preview_draws_box_and_lists_ais(tmp_path: Path) -> None:
    pipeline = load_module("scripts.pipeline")
    image_path = tmp_path / "2020-01-04_tile_0_0.tiff"
    Image.new("RGB", (400, 300), (10, 20, 30)).save(image_path)
    result = {
        "detection_id": "2020-01-04_tile_0_0:1",
        "image_path": str(image_path),
        "date": "2020-01-04",
        "cx_px": 200.0,
        "cy_px": 150.0,
        "w_px": 40.0,
        "h_px": 20.0,
        "confidence": 0.77,
        "center_lat": 29.1,
        "center_lon": -94.7,
        "dark_status": "PARTIALLY_DARK",
        "ais_count": 1,
        "ais_vessels": [
            {
                "MMSI": "111111111",
                "VesselName": "ALPHA",
                "BaseDateTime": "2020-01-04T16:30:00",
                "VesselType": "80",
                "SOG": "0.1",
                "COG": "12.3",
                "Length": "183",
                "distance_m": 42.0,
            }
        ],
    }
    destination = pipeline.write_detection_preview(result, tmp_path / "previews")
    assert destination.exists()
    with Image.open(destination) as preview:
        assert preview.size[0] > 0 and preview.size[1] > 0
        colors = {color for _, color in preview.convert("RGB").getcolors(maxcolors=100000)}
    assert (255, 0, 0) in colors, "the detection box should be drawn in red"

    captions = pipeline._caption_lines(result)
    assert any("111111111" in line for line in captions)
    assert any("2020-01-04T16:30:00" in line for line in captions)
    assert any("AIS vessels within" in line for line in captions)


def test_write_detection_preview_handles_a_box_at_the_image_edge(tmp_path: Path) -> None:
    pipeline = load_module("scripts.pipeline")
    image_path = tmp_path / "2020-01-04_tile_0_0.tiff"
    Image.new("RGB", (120, 90), (10, 20, 30)).save(image_path)
    result = {
        "detection_id": "edge:1",
        "image_path": str(image_path),
        "date": "2020-01-04",
        "cx_px": 3.0,
        "cy_px": 2.0,
        "w_px": 8.0,
        "h_px": 8.0,
        "confidence": 0.5,
        "center_lat": 29.1,
        "center_lon": -94.7,
        "dark_status": "FULLY_DARK",
        "ais_count": 0,
        "ais_vessels": [],
    }
    destination = pipeline.write_detection_preview(result, tmp_path / "previews")
    assert destination.exists()


def test_run_pipeline_end_to_end_with_mocked_yolo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = load_module("scripts.pipeline")
    bbox = (-94.73, 29.07, -94.57, 29.205)
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    good = tiles / "2020-01-04_tile_0_0.tiff"
    write_pil_geotiff(good, 200, 100, bbox)
    (tiles / "2020-01-04_tile_0_1.tiff").write_bytes(b"")

    ais_path = tmp_path / "ais" / "AIS_2020_01_04.csv"
    write_ais_csv(
        ais_path,
        [
            {
                "MMSI": "111111111",
                "BaseDateTime": "2020-01-04T16:30:00",
                "LAT": "29.1375",
                "LON": "-94.6500",
                "SOG": "0.1",
                "COG": "12.3",
                "VesselName": "ALPHA",
                "VesselType": "80",
                "Length": "183",
            }
        ],
    )

    def fake_detect_images(image_paths, weights, conf=0.25):
        assert [Path(path).name for path in image_paths] == ["2020-01-04_tile_0_0.tiff"]
        return [
            {
                "detection_id": "2020-01-04_tile_0_0:1",
                "image_path": str(image_paths[0]),
                "date": "2020-01-04",
                "cx_px": 100.0,
                "cy_px": 50.0,
                "w_px": 20.0,
                "h_px": 20.0,
                "confidence": 0.81,
                "class_id": 1,
                "class_name": "ships",
                "image_width_px": 200,
                "image_height_px": 100,
            }
        ]

    monkeypatch.setattr(pipeline, "detect_images", fake_detect_images)

    weight = tmp_path / "fixture.pt"
    weight.write_bytes(b"test checkpoint")
    results = pipeline.run_pipeline(
        image_inputs=[str(tiles)],
        weights=weight,
        record_path=tmp_path / "run.json",
        ais_dir=tmp_path / "ais",
        date="2020-01-04",
        preview_dir=tmp_path / "out" / "previews",
        warn=lambda message: None,
    )
    assert len(results) == 1
    assert results[0]["dark_status"] == "PARTIALLY_DARK"
    assert Path(results[0]["preview_png"]).exists()
    assert pipeline.summarise_results(results) == {
        "detections": 1,
        "FULLY_DARK": 0,
        "PARTIALLY_DARK": 1,
        "AIS_PRESENT": 0,
    }

    csv_path = tmp_path / "out" / "results.csv"
    json_path = tmp_path / "out" / "results.json"
    pipeline.write_pipeline_output(csv_path, results)
    pipeline.write_pipeline_output(json_path, results)
    header = csv_path.read_text().splitlines()[0]
    assert "vessel_1_type" in header and "vessel_1_time" in header
    assert "111111111" in csv_path.read_text()
    assert json_path.exists()


def test_run_pipeline_raises_when_no_readable_images(tmp_path: Path) -> None:
    pipeline = load_module("scripts.pipeline")
    (tmp_path / "2020-01-04_tile_0_0.tiff").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        pipeline.run_pipeline(image_inputs=[str(tmp_path)], ais_dir=tmp_path, warn=lambda message: None)


def test_default_weights_points_at_the_promoted_weight() -> None:
    pipeline = load_module("scripts.pipeline")
    assert pipeline.DEFAULT_WEIGHTS.name == "opendarksts_yolo11s_v4_best.pt"
    assert pipeline.DEFAULT_WEIGHTS.parent == REPO_ROOT / "workspace" / "model_weights"


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None,
    reason="ultralytics is not installed in this interpreter",
)
def test_integration_detect_on_a_real_tile_if_weights_exist(tmp_path: Path) -> None:
    pipeline = load_module("scripts.pipeline")
    if not pipeline.DEFAULT_WEIGHTS.exists():
        pytest.skip("promoted weight is not available")
    detect = load_module("scripts.detect")
    tile = REPO_ROOT / "workspace" / "inputs" / "2020_01_04"
    if not tile.exists():
        pytest.skip("no local tile inputs to run integration inference on")
    image_paths = pipeline.select_images([str(tile)], warn=lambda message: None)
    if not image_paths:
        pytest.skip("no readable tiles available")
    detections = detect.detect_images(image_paths=image_paths[:1], weights=pipeline.DEFAULT_WEIGHTS, conf=0.25)
    for detection in detections:
        assert detection["class_name"] == "ships"
