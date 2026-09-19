"""Streaming NOAA AIS and acquisition-specific CDSE imagery downloads."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:
    from .workspace import WORKSPACE_DIR, atomic_json, contained, sha256
    from .correlate_ais import validate_ais_header, InvalidAIS
    from .sts_utils import load_bbox_for_image
except ImportError:
    from workspace import WORKSPACE_DIR, atomic_json, contained, sha256
    from correlate_ais import validate_ais_header, InvalidAIS
    from sts_utils import load_bbox_for_image

NOAA_BASE = "https://noaaocm.blob.core.windows.net/ais/csv2"
CDSE_BASE = "https://sh.dataspace.copernicus.eu"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


def date_range(start: str, end: str | None = None) -> list[str]:
    first, last = date.fromisoformat(start), date.fromisoformat(end or start)
    if last < first:
        raise ValueError("End date must be on or after start date")
    return [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]


def validate_bbox(bbox) -> list[float]:
    values = list(map(float, bbox))
    if len(values) != 4 or not all(map(math.isfinite, values)):
        raise ValueError("Bounds must contain four finite coordinates")
    west, south, east, north = values
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Bounds must be west, south, east, north in EPSG:4326")
    return values


def cache_key(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def download(url: str, destination: Path, *, headers=None, data=None, opener=None, log=print) -> None:
    """Retry whole transfers; a partial file is never a reusable download."""
    opener = opener or urllib.request.urlopen
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, data=data, headers=headers or {})
            with opener(request, timeout=120) as response, destination.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
                expected = response.headers.get("Content-Length")
                if expected is not None and output.tell() != int(expected):
                    raise OSError("Incomplete download")
            return
        except (OSError, urllib.error.URLError) as exc:
            destination.unlink(missing_ok=True)
            if attempt == 2 or isinstance(exc, urllib.error.HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"Download failed ({getattr(exc, 'code', type(exc).__name__)}); retry the job") from None
            log(f"Download interrupted; retry {attempt + 1}/2")
            time.sleep(attempt + 1)


def verified_cache(path: Path, identity: dict) -> dict | None:
    try:
        record = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        if record.get("complete") and record.get("identity") == identity and record["sha256"] == sha256(path):
            return record
    except (OSError, ValueError, KeyError):
        pass
    return None


def validate_ais_csv(path: Path, day: str) -> int:
    valid = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        validate_ais_header(reader.fieldnames)
        for row in reader:
            try:
                lat, lon = float(row["LAT"]), float(row["LON"])
                if row["MMSI"] and (row["BaseDateTime"] or "").startswith(day + "T") and -90 <= lat <= 90 and -180 <= lon <= 180:
                    valid += 1
            except (ValueError, TypeError):
                continue
    if not valid:
        raise InvalidAIS("Downloaded AIS has no valid positions for the requested day")
    return valid


def normalize_ais_csv(source: Path, destination: Path) -> None:
    aliases = {"mmsi": "MMSI", "base_date_time": "BaseDateTime", "latitude": "LAT", "longitude": "LON",
               "sog": "SOG", "cog": "COG", "heading": "Heading", "vessel_name": "VesselName",
               "imo": "IMO", "call_sign": "CallSign", "vessel_type": "VesselType", "status": "Status",
               "length": "Length", "width": "Width", "draft": "Draft", "cargo": "Cargo", "transceiver": "TransceiverClass"}
    with source.open(newline="", encoding="utf-8-sig") as handle, destination.open("w", newline="", encoding="utf-8") as output:
        reader = csv.DictReader(handle, strict=True)
        names = [aliases.get(name, name) for name in reader.fieldnames or []]
        validate_ais_header(names)
        writer = csv.DictWriter(output, fieldnames=names)
        writer.writeheader()
        for row in reader:
            normalized = {aliases.get(key, key): value for key, value in row.items() if key is not None}
            normalized["BaseDateTime"] = (normalized.get("BaseDateTime") or "").replace(" ", "T", 1)
            writer.writerow(normalized)


def acquire_ais(start: str, end: str | None = None, *, output_dir=None, bbox=None, opener=None, log=print) -> dict:
    root = contained(output_dir or WORKSPACE_DIR / "inputs" / "ais")
    bounds = validate_bbox(bbox) if bbox is not None else None
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for day in date_range(start, end):
        year = int(day[:4])
        if year < 2015:
            raise ValueError("Daily NOAA AIS downloads are supported from 2015 onward")
        name = f"AIS_{day.replace('-', '_')}.csv"
        archive_name = f"ais-{day}.csv.zst"
        url = f"{NOAA_BASE}/csv{year}/{archive_name}"
        identity = {"provider": "NOAA MarineCadastre", "date": day, "url": url, "bbox": None}
        full = contained(root / "full" / name)
        full.parent.mkdir(parents=True, exist_ok=True)
        record = verified_cache(full, identity)
        if record is None:
            log(f"Downloading NOAA AIS for {day}")
            with tempfile.TemporaryDirectory(prefix=".download-", dir=root) as temporary:
                archive, extracted = Path(temporary) / archive_name, Path(temporary) / name
                download(url, archive, opener=opener, log=log)
                if not zipfile.is_zipfile(archive):
                    import zstandard
                    with archive.open("rb") as source, extracted.open("wb") as output:
                        zstandard.ZstdDecompressor().copy_stream(source, output)
                else:
                    with zipfile.ZipFile(archive) as bundle:
                        members = [entry for entry in bundle.infolist() if not entry.is_dir() and Path(entry.filename).name == name]
                        if len(members) != 1:
                            raise InvalidAIS("Archive must contain exactly one matching AIS day CSV")
                        with bundle.open(members[0]) as source, extracted.open("wb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                normalized = Path(temporary) / "normalized.csv"
                normalize_ais_csv(extracted, normalized)
                normalized.replace(extracted)
                count = validate_ais_csv(extracted, day)
                record = {"identity": identity, "complete": True, "valid_rows": count, "sha256": sha256(extracted), "path": str(full)}
                extracted.replace(full)
                atomic_json(full.with_suffix(".csv.json"), record)
        if bounds is not None:
            target = contained(root / "regions" / cache_key(bounds) / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            region_identity = {**identity, "bbox": bounds, "source_sha256": record["sha256"]}
            filtered = verified_cache(target, region_identity)
            if filtered is None:
                with tempfile.TemporaryDirectory(prefix=".filter-", dir=root) as temporary:
                    part = Path(temporary) / name
                    count = 0
                    with full.open(newline="", encoding="utf-8-sig") as source, part.open("w", newline="", encoding="utf-8") as output:
                        reader = csv.DictReader(source)
                        writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                        writer.writeheader()
                        for row in reader:
                            try:
                                lat, lon = float(row["LAT"]), float(row["LON"])
                            except (ValueError, TypeError):
                                continue
                            if bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]:
                                writer.writerow(row)
                                count += 1
                    filtered = {"identity": region_identity, "complete": True, "row_count": count, "sha256": sha256(part), "path": str(target)}
                    part.replace(target)
                    atomic_json(target.with_suffix(".csv.json"), filtered)
            record = filtered
        records.append(record)
        log(f"AIS ready: {record['path']}")
    return {"files": records, "output_dir": str(root)}


EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["B02", "B03", "B04", "dataMask"], units: "REFLECTANCE"}],
          output: {bands: 3, sampleType: "UINT8"}, mosaicking: "TILE"};
}
function evaluatePixel(samples) {
  for (let s of samples) if (s.dataMask) return [s.B04, s.B03, s.B02].map(v => Math.min(255, Math.round(v * 3.5 * 255)));
  return [0, 0, 0];
}
function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
  outputMetadata.userData = {tiles: scenes.tiles, normalizationFactor: inputMetadata.normalizationFactor};
}
"""


class CDSEClient:
    def __init__(self, opener=None):
        self.opener = opener or urllib.request.urlopen
        self.token = None
        self.expires = 0

    def headers(self):
        if time.monotonic() >= self.expires:
            client_id, secret = os.environ.get("CDSE_CLIENT_ID"), os.environ.get("CDSE_CLIENT_SECRET")
            if not client_id or not secret:
                raise ValueError("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in the environment before acquiring imagery")
            body = urllib.parse.urlencode({"grant_type": "client_credentials", "client_id": client_id, "client_secret": secret}).encode()
            request = urllib.request.Request(TOKEN_URL, data=body)
            try:
                with self.opener(request, timeout=30) as response:
                    token = json.load(response)
                self.token = token["access_token"]
                self.expires = time.monotonic() + max(1, int(token.get("expires_in", 300)) - 30)
            except Exception:
                raise RuntimeError("CDSE authentication failed; check environment credentials") from None
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def scenes(self, bbox, start, end, max_cloud):
        payload = {"collections": ["sentinel-2-l2a"], "bbox": bbox,
                   "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 100,
                   "filter": f"eo:cloud_cover <= {max_cloud}", "filter-lang": "cql2-text"}
        scenes = []
        while True:
            request = urllib.request.Request(f"{CDSE_BASE}/catalog/v1/search", data=json.dumps(payload).encode(), headers=self.headers())
            with self.opener(request, timeout=120) as response:
                page = json.load(response)
            scenes.extend(page.get("features", []))
            next_token = page.get("context", {}).get("next")
            if next_token is None:
                return scenes
            payload["next"] = next_token

    def image(self, payload, destination, log):
        download(f"{CDSE_BASE}/process/v1", destination, headers={**self.headers(), "Accept": "application/tar"},
                 data=json.dumps(payload).encode(), opener=self.opener, log=log)


def imagery_tiles(bbox):
    west, south, east, north = bbox
    width = math.ceil((east - west) * 111320 * math.cos(math.radians((south + north) / 2)) / 10)
    height = math.ceil((north - south) * 111320 / 10)
    nx, ny = max(1, math.ceil(width / 2500)), max(1, math.ceil(height / 2500))
    for x in range(nx):
        for y in range(ny):
            yield [west + (east-west)*x/nx, south + (north-south)*y/ny,
                   west + (east-west)*(x+1)/nx, south + (north-south)*(y+1)/ny], [max(1, math.ceil(width/nx)), max(1, math.ceil(height/ny))]


def acquire_imagery(start: str, end: str | None = None, *, bbox, output_dir=None, max_cloud=30, client=None, log=print) -> dict:
    days = date_range(start, end)
    bounds = validate_bbox(bbox)
    if not math.isfinite(max_cloud) or not 0 <= max_cloud <= 100:
        raise ValueError("Maximum cloud cover must be between zero and 100")
    root = contained(output_dir or WORKSPACE_DIR / "inputs" / "imagery")
    root.mkdir(parents=True, exist_ok=True)
    client = client or CDSEClient()
    groups = {}
    for scene in client.scenes(bounds, days[0], days[-1], max_cloud):
        groups.setdefault(scene["properties"]["datetime"], []).append(scene["id"])
    records = []
    for acquired_at, scene_ids in sorted(groups.items()):
        instant = datetime.fromisoformat(acquired_at.replace("Z", "+00:00")).astimezone(timezone.utc)
        interval = {"from": (instant - timedelta(seconds=1)).isoformat(), "to": (instant + timedelta(seconds=1)).isoformat()}
        for tile_bbox, size in imagery_tiles(bounds):
            identity = {"catalog_scene_ids": sorted(set(scene_ids)), "acquired_at": acquired_at, "bbox": tile_bbox,
                        "size": size, "resolution_m": 10, "max_cloud": max_cloud, "renderer": "rgb-reflectance-3.5-v1"}
            target = contained(root / f"{instant.date()}_acq_{cache_key(identity)}.tiff")
            record = verified_cache(target, identity)
            if record is None:
                payload = {"input": {"bounds": {"bbox": tile_bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                                     "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": interval, "maxCloudCoverage": max_cloud}}]},
                           "output": {"width": size[0], "height": size[1], "responses": [
                               {"identifier": "default", "format": {"type": "image/tiff"}},
                               {"identifier": "userdata", "format": {"type": "application/json"}}]}, "evalscript": EVALSCRIPT}
                log(f"Downloading acquisition {acquired_at}, tile {target.name}")
                with tempfile.TemporaryDirectory(prefix=".scene-", dir=root) as temporary:
                    archive = Path(temporary) / "response.tar"
                    client.image(payload, archive, log)
                    with tarfile.open(archive) as bundle:
                        image_member = next((m for m in bundle.getmembers() if m.isfile() and m.name in {"default.tif", "default.tiff"}), None)
                        metadata_member = next((m for m in bundle.getmembers() if m.isfile() and m.name == "userdata.json"), None)
                        if image_member is None or metadata_member is None:
                            raise ValueError("CDSE response is missing imagery or acquisition metadata")
                        part = Path(temporary) / "image.tiff"
                        with bundle.extractfile(image_member) as source, part.open("wb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                        metadata = json.load(bundle.extractfile(metadata_member))
                    if not metadata.get("tiles"):
                        raise ValueError("CDSE returned no contributing scenes")
                    load_bbox_for_image(part)
                    record = {"identity": identity, "complete": True, "sha256": sha256(part), "path": str(target),
                              "acquisition": metadata, "composition": "first valid pixel across tiles at the requested acquisition time"}
                    part.replace(target)
                    atomic_json(target.with_suffix(".tiff.json"), record)
            records.append(record)
    return {"files": records, "output_dir": str(root), "acquisition_count": len(groups)}
