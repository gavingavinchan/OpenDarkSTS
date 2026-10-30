import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts import acquisition as acquire
from test_operational_pipeline import write_pil_geotiff


class Response(io.BytesIO):
    headers = {}


def zip_response(content):
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as archive:
        archive.writestr("AIS_2024_01_01.csv", content)
    return blob.getvalue()


def test_ais_full_and_area_caches_are_distinct_and_validated(tmp_path):
    data = "MMSI,BaseDateTime,LAT,LON\n1,2024-01-01T00:00:00,1,1\n2,2024-01-01T00:00:00,5,5\n"
    requests = []
    def opener(request, **kwargs):
        requests.append(request.full_url)
        return Response(zip_response(data))
    first = acquire.acquire_ais("2024-01-01", output_dir=tmp_path, bbox=[0, 0, 2, 2], opener=opener)
    second = acquire.acquire_ais("2024-01-01", output_dir=tmp_path, bbox=[4, 4, 6, 6], opener=opener)
    assert len(requests) == 1
    assert first["files"][0]["path"] != second["files"][0]["path"]
    assert "\n1," in Path(first["files"][0]["path"]).read_text()
    assert "\n2," in Path(second["files"][0]["path"]).read_text()
    full = tmp_path / "full" / "AIS_2024_01_01.csv"
    full.write_text("broken")
    acquire.acquire_ais("2024-01-01", output_dir=tmp_path, opener=opener)
    assert len(requests) == 2


def test_invalid_ais_download_is_not_published(tmp_path):
    with pytest.raises(ValueError, match="columns"):
        acquire.acquire_ais("2024-01-01", output_dir=tmp_path, opener=lambda *a, **kw: Response(zip_response("wrong,header\n1,2\n")))
    assert not list(tmp_path.rglob("*.csv"))
    assert not list(tmp_path.glob(".download-*"))


def test_current_noaa_zstandard_schema_is_normalized(tmp_path):
    import zstandard
    data = b"mmsi,base_date_time,latitude,longitude,vessel_name\n1,2025-01-01 00:00:00,1,2,Example\n"
    compressed = zstandard.ZstdCompressor().compress(data)
    result = acquire.acquire_ais("2025-01-01", output_dir=tmp_path, opener=lambda *a, **kw: Response(compressed))
    record = result["files"][0]
    assert record["identity"]["url"].endswith("csv2025/ais-2025-01-01.csv.zst")
    assert Path(record["path"]).read_text().startswith("MMSI,BaseDateTime,LAT,LON,VesselName\n1,2025-01-01T00:00:00")


def test_imagery_groups_acquisitions_and_preserves_metadata(tmp_path):
    calls = []
    class Client:
        def scenes(self, *args):
            return [{"id": name, "properties": {"datetime": timestamp}} for name, timestamp in [
                ("scene-a", "2024-01-01T10:00:00Z"), ("scene-b", "2024-01-01T10:00:00Z"),
                ("scene-c", "2024-01-01T11:00:00Z")]]
        def image(self, payload, destination, log):
            calls.append(payload)
            tiff = destination.parent / "tile.tiff"
            write_pil_geotiff(tiff, 32, 32, (1., 1., 1.01, 1.01))
            metadata = json.dumps({"tiles": [{"dataPath": "s3://source/tile", "date": "2024-01-01T10:00:00Z"}]}).encode()
            with tarfile.open(destination, "w") as archive:
                archive.add(tiff, arcname="default.tif")
                info = tarfile.TarInfo("userdata.json")
                info.size = len(metadata)
                archive.addfile(info, io.BytesIO(metadata))
    result = acquire.acquire_imagery("2024-01-01", bbox=[1, 1, 1.01, 1.01], output_dir=tmp_path, client=Client())
    assert len(calls) == 2
    assert result["acquisition_count"] == 2
    assert result["files"][0]["identity"]["catalog_scene_ids"] == ["scene-a", "scene-b"]
    assert calls[0]["input"]["data"][0]["dataFilter"]["timeRange"]["from"].endswith("09:59:59+00:00")
    assert result["files"][0]["acquisition"]["tiles"]
    acquire.acquire_imagery("2024-01-01", bbox=[1, 1, 1.01, 1.01], output_dir=tmp_path, client=Client())
    assert len(calls) == 2


def test_credentials_missing_is_clear_and_does_not_call_network(monkeypatch):
    monkeypatch.delenv("CDSE_CLIENT_ID", raising=False)
    monkeypatch.delenv("CDSE_CLIENT_SECRET", raising=False)
    with pytest.raises(ValueError, match="CDSE_CLIENT_ID"):
        acquire.CDSEClient().headers()


def test_imagery_tiles_obey_pixel_limit():
    tiles = list(acquire.imagery_tiles([-95, 28, -94, 29]))
    assert len(tiles) > 1
    assert all(1 <= n <= 2500 for _, size in tiles for n in size)


def test_download_reports_real_bytes_and_unknown_totals(tmp_path):
    for total in [True, False]:
        response = Response(b'abcdefgh')
        response.headers = {'Content-Length': '8'} if total else {}
        logs = []
        acquire.download('https://example.test', tmp_path / 'part', opener=lambda *a, **kw: response, log=logs.append)
        if total:
            assert any('8 / 8 bytes (100.0%)' in line for line in logs)
        else:
            assert any('8 bytes (total unknown)' in line for line in logs)


def test_ais_cancel_removes_only_partial_files(tmp_path):
    saved = tmp_path / 'full' / 'previous.csv'
    saved.parent.mkdir()
    saved.write_text('retained')
    calls = 0
    def check():
        nonlocal calls
        calls += 1
        if calls >= 5:
            raise acquire.Cancelled('cancelled')
    with pytest.raises(acquire.Cancelled), acquire.cancellation(check):
        acquire.acquire_ais('2024-01-01', output_dir=tmp_path, opener=lambda *a, **kw: Response(b'x' * 1000000))
    assert saved.read_text() == 'retained'
    assert list(tmp_path.rglob('*.csv')) == [saved]
    assert not list(tmp_path.glob('.download-*'))


def test_imagery_cancel_cleans_temporary_response(tmp_path):
    requested = False
    class Client:
        def scenes(self, *args):
            return [{'id': 'one', 'properties': {'datetime': '2024-01-01T10:00:00Z'}}]
        def image(self, payload, destination, log):
            nonlocal requested
            destination.write_bytes(b'partial')
            requested = True
            acquire.check_cancelled()
    def check():
        if requested:
            raise acquire.Cancelled('cancelled')
    with pytest.raises(acquire.Cancelled), acquire.cancellation(check):
        acquire.acquire_imagery('2024-01-01', bbox=[1, 1, 1.01, 1.01], output_dir=tmp_path, client=Client())
    assert not list(tmp_path.iterdir())
