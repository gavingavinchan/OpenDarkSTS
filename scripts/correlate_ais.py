from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

try:
    from .sts_utils import (
        build_vessel_key,
        classify_dark_status,
        coarse_degree_window,
        haversine_m,
        read_json,
        resolve_ais_file,
        write_json,
    )
except ImportError:
    from sts_utils import (
        build_vessel_key,
        classify_dark_status,
        coarse_degree_window,
        haversine_m,
        read_json,
        resolve_ais_file,
        write_json,
    )


def _prepare_detection_windows(detections: list[dict[str, Any]], radius_m: float) -> tuple[list[dict[str, Any]], tuple[float, float, float, float]]:
    prepared: list[dict[str, Any]] = []
    min_lat = float("inf")
    max_lat = float("-inf")
    min_lon = float("inf")
    max_lon = float("-inf")

    for detection in detections:
        lat = float(detection["center_lat"])
        lon = float(detection["center_lon"])
        lat_delta, lon_delta = coarse_degree_window(lat=lat, radius_m=radius_m)
        prepared_detection = {
            "detection": dict(detection),
            "lat": lat,
            "lon": lon,
            "lat_min": lat - lat_delta,
            "lat_max": lat + lat_delta,
            "lon_min": lon - lon_delta,
            "lon_max": lon + lon_delta,
            "matches": {},
        }
        prepared.append(prepared_detection)
        min_lat = min(min_lat, prepared_detection["lat_min"])
        max_lat = max(max_lat, prepared_detection["lat_max"])
        min_lon = min(min_lon, prepared_detection["lon_min"])
        max_lon = max(max_lon, prepared_detection["lon_max"])

    return prepared, (min_lat, max_lat, min_lon, max_lon)


def correlate_detections(detections: list[dict[str, Any]], ais_dir: str | Path, radius_m: float = 1000.0) -> list[dict[str, Any]]:
    if not detections:
        return []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for detection in detections:
        grouped.setdefault(detection["date"], []).append(detection)

    correlated: list[dict[str, Any]] = []
    for date_str, date_detections in sorted(grouped.items()):
        ais_file = resolve_ais_file(ais_dir=ais_dir, date_str=date_str)
        prepared, global_window = _prepare_detection_windows(date_detections, radius_m=radius_m)
        global_lat_min, global_lat_max, global_lon_min, global_lon_max = global_window

        with ais_file.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                lat_text = (row.get("LAT") or "").strip()
                lon_text = (row.get("LON") or "").strip()
                if not lat_text or not lon_text:
                    continue

                try:
                    row_lat = float(lat_text)
                    row_lon = float(lon_text)
                except ValueError:
                    continue

                if row_lat < global_lat_min or row_lat > global_lat_max or row_lon < global_lon_min or row_lon > global_lon_max:
                    continue

                for candidate in prepared:
                    if row_lat < candidate["lat_min"] or row_lat > candidate["lat_max"]:
                        continue
                    if row_lon < candidate["lon_min"] or row_lon > candidate["lon_max"]:
                        continue

                    distance_m = haversine_m(candidate["lat"], candidate["lon"], row_lat, row_lon)
                    if distance_m > radius_m:
                        continue

                    vessel_key = build_vessel_key(row)
                    match_record = {
                        "MMSI": (row.get("MMSI") or "").strip(),
                        "VesselName": (row.get("VesselName") or "").strip(),
                        "BaseDateTime": (row.get("BaseDateTime") or "").strip(),
                        "LAT": row_lat,
                        "LON": row_lon,
                        "distance_m": round(distance_m, 3),
                    }

                    existing = candidate["matches"].get(vessel_key)
                    if existing is None or match_record["distance_m"] < existing["distance_m"]:
                        candidate["matches"][vessel_key] = match_record

        for candidate in prepared:
            ais_vessels = sorted(candidate["matches"].values(), key=lambda vessel: vessel["distance_m"])
            result = dict(candidate["detection"])
            result["ais_file"] = str(ais_file)
            result["ais_vessels"] = ais_vessels
            result["ais_count"] = len(ais_vessels)
            result["dark_status"] = classify_dark_status(result["ais_count"])
            correlated.append(result)

    return correlated


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate georeferenced detections against AIS CSV data")
    parser.add_argument("--detections", required=True, help="Path to georeferenced detections JSON")
    parser.add_argument("--ais-dir", required=True, help="AIS directory or specific AIS CSV file")
    parser.add_argument("--radius", type=float, default=1000.0, help="Search radius in meters")
    parser.add_argument("--output", required=True, help="Path to write correlated results JSON")
    args = parser.parse_args()

    detections = read_json(args.detections)
    if not isinstance(detections, list):
        raise ValueError("Detections JSON must contain a list")

    correlated = correlate_detections(detections=detections, ais_dir=args.ais_dir, radius_m=args.radius)
    write_json(args.output, correlated)
    print(f"Wrote {len(correlated)} correlated detections to {args.output}")


if __name__ == "__main__":
    main()
