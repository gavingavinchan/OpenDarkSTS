from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import statistics
import subprocess
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

try:
    from .sts_utils import ensure_parent_dir, haversine_m
except ImportError:
    from sts_utils import ensure_parent_dir, haversine_m


CRITERIA_VERSION = "ais_denmark_v1"
DEFAULT_BIN_MINUTES = 5
DEFAULT_MAX_TIME_GAP_MINUTES = 10
DEFAULT_INSERT_TIERS = {"HIGH", "MEDIUM"}
DEFAULT_MAX_DISTANCE_M = 250.0
DEFAULT_MEDIUM_DISTANCE_M = 150.0
DEFAULT_HIGH_DISTANCE_M = 100.0


@dataclass(slots=True)
class VesselObservation:
    timestamp: datetime
    mmsi: str
    lat: float
    lon: float
    sog_kn: float
    cog_deg: float | None
    heading_deg: float | None
    ship_type: str
    navigational_status: str
    name: str
    imo: str | None
    callsign: str | None
    length_m: float | None
    width_m: float | None


@dataclass(slots=True)
class PairObservation:
    event_timestamp: datetime
    vessel_1: VesselObservation
    vessel_2: VesselObservation
    separation_m: float
    heading_delta_deg: float | None


@dataclass(slots=True)
class ActiveEvent:
    vessel_1_mmsi: str
    vessel_2_mmsi: str
    points: list[PairObservation] = field(default_factory=list)
    last_seen: datetime | None = None


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%d/%m/%Y %H:%M:%S").replace(tzinfo=UTC)


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text in {"Unknown", "Undefined"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def clean_text(value: str | None) -> str:
    if value is None:
        return ""
    text = value.strip()
    if text in {"Unknown", "Undefined"}:
        return ""
    return text


def optional_text(value: str | None) -> str | None:
    text = clean_text(value)
    return text or None


def normalize_denmark_row(row: dict[str, str]) -> VesselObservation | None:
    if clean_text(row.get("Type of mobile")) != "Class A":
        return None

    mmsi = clean_text(row.get("MMSI"))
    lat = parse_float(row.get("Latitude"))
    lon = parse_float(row.get("Longitude"))
    sog_kn = parse_float(row.get("SOG"))
    timestamp_text = clean_text(row.get("# Timestamp"))
    if not mmsi or lat is None or lon is None or sog_kn is None or not timestamp_text:
        return None

    ship_type = clean_text(row.get("Ship type")) or "Undefined"
    return VesselObservation(
        timestamp=parse_timestamp(timestamp_text),
        mmsi=mmsi,
        lat=lat,
        lon=lon,
        sog_kn=sog_kn,
        cog_deg=parse_float(row.get("COG")),
        heading_deg=parse_float(row.get("Heading")),
        ship_type=ship_type,
        navigational_status=clean_text(row.get("Navigational status")) or "Unknown value",
        name=clean_text(row.get("Name")),
        imo=optional_text(row.get("IMO")),
        callsign=optional_text(row.get("Callsign")),
        length_m=parse_float(row.get("Length")),
        width_m=parse_float(row.get("Width")),
    )


def vessel_role(observation: VesselObservation) -> str | None:
    ship_type = observation.ship_type
    if ship_type == "Tanker":
        return "tanker"
    if ship_type == "Cargo":
        return "cargo"
    if ship_type in {"Undefined", "Other"}:
        if (observation.length_m is not None and observation.length_m >= 60.0) or observation.imo:
            return "fallback"
    return None


def pair_type_allowed(vessel_1: VesselObservation, vessel_2: VesselObservation) -> bool:
    role_1 = vessel_role(vessel_1)
    role_2 = vessel_role(vessel_2)
    if role_1 is None or role_2 is None:
        return False
    if role_1 == "fallback" and role_2 == "fallback":
        return False
    if "tanker" in {role_1, role_2}:
        return True
    return role_1 == "cargo" and role_2 in {"cargo", "fallback"} or role_2 == "cargo" and role_1 in {"cargo", "fallback"}


def heading_delta_deg(vessel_1: VesselObservation, vessel_2: VesselObservation) -> float | None:
    angle_1 = vessel_1.cog_deg if vessel_1.cog_deg is not None else vessel_1.heading_deg
    angle_2 = vessel_2.cog_deg if vessel_2.cog_deg is not None else vessel_2.heading_deg
    if angle_1 is None or angle_2 is None:
        return None
    diff = abs(angle_1 - angle_2) % 360.0
    return min(diff, 360.0 - diff)


def floor_to_bin(timestamp: datetime, bin_minutes: int) -> datetime:
    minute = (timestamp.minute // bin_minutes) * bin_minutes
    return timestamp.replace(minute=minute, second=0, microsecond=0)


def equirectangular_xy(lat: float, lon: float, ref_lat_deg: float) -> tuple[float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = math.cos(math.radians(ref_lat_deg)) * 111_320.0
    return lon * meters_per_deg_lon, lat * meters_per_deg_lat


def candidate_pairs_for_bin(observations: Iterable[VesselObservation], max_distance_m: float, max_time_gap: timedelta) -> list[PairObservation]:
    points = list(observations)
    if len(points) < 2:
        return []

    ref_lat = sum(point.lat for point in points) / len(points)
    grid: dict[tuple[int, int], list[tuple[VesselObservation, float, float]]] = {}
    for point in points:
        x, y = equirectangular_xy(lat=point.lat, lon=point.lon, ref_lat_deg=ref_lat)
        cell = (int(x // max_distance_m), int(y // max_distance_m))
        grid.setdefault(cell, []).append((point, x, y))

    seen_pairs: set[tuple[str, str]] = set()
    pair_observations: list[PairObservation] = []
    for cell, members in grid.items():
        neighbors: list[tuple[VesselObservation, float, float]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbors.extend(grid.get((cell[0] + dx, cell[1] + dy), []))

        for vessel_1, _, _ in members:
            for vessel_2, _, _ in neighbors:
                if vessel_1.mmsi >= vessel_2.mmsi:
                    continue
                pair_key = (vessel_1.mmsi, vessel_2.mmsi)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                if not pair_type_allowed(vessel_1, vessel_2):
                    continue
                if vessel_1.navigational_status == "Moored" and vessel_2.navigational_status == "Moored":
                    continue
                if abs(vessel_1.timestamp - vessel_2.timestamp) > max_time_gap:
                    continue
                if max(vessel_1.sog_kn, vessel_2.sog_kn) > 5.0:
                    continue
                if abs(vessel_1.sog_kn - vessel_2.sog_kn) > 2.0:
                    continue

                distance_m = haversine_m(vessel_1.lat, vessel_1.lon, vessel_2.lat, vessel_2.lon)
                if distance_m > max_distance_m:
                    continue

                pair_heading_delta = heading_delta_deg(vessel_1, vessel_2)
                if vessel_1.sog_kn > 1.0 and vessel_2.sog_kn > 1.0 and pair_heading_delta is not None and pair_heading_delta > 30.0:
                    continue

                pair_observations.append(
                    PairObservation(
                        event_timestamp=max(vessel_1.timestamp, vessel_2.timestamp),
                        vessel_1=vessel_1,
                        vessel_2=vessel_2,
                        separation_m=distance_m,
                        heading_delta_deg=pair_heading_delta,
                    )
                )

    return pair_observations


def make_event_id(source_file: str, vessel_1_mmsi: str, vessel_2_mmsi: str, start_time: datetime) -> str:
    payload = f"{source_file}|{vessel_1_mmsi}|{vessel_2_mmsi}|{start_time.isoformat()}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"evt_{digest}"


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_event(
    event: ActiveEvent,
    source_dataset: str,
    source_file: str,
) -> tuple[dict[str, object], list[dict[str, object]]] | None:
    if len(event.points) < 3:
        return None

    timestamps = [point.event_timestamp for point in event.points]
    start_time = min(timestamps)
    end_time = max(timestamps)
    duration_min = (end_time - start_time).total_seconds() / 60.0
    if duration_min < 15.0:
        return None

    separations = [point.separation_m for point in event.points]
    sog_1_values = [point.vessel_1.sog_kn for point in event.points]
    sog_2_values = [point.vessel_2.sog_kn for point in event.points]
    sog_diffs = [abs(point.vessel_1.sog_kn - point.vessel_2.sog_kn) for point in event.points]
    heading_deltas = [point.heading_delta_deg for point in event.points if point.heading_delta_deg is not None]

    vessel_1 = event.points[0].vessel_1
    vessel_2 = event.points[0].vessel_2
    if not pair_type_allowed(vessel_1, vessel_2):
        return None

    mean_sog_1 = sum(sog_1_values) / len(sog_1_values)
    mean_sog_2 = sum(sog_2_values) / len(sog_2_values)
    mean_sog_delta = sum(sog_diffs) / len(sog_diffs)
    mean_heading_delta = mean_or_none(heading_deltas)
    center_lat = sum((point.vessel_1.lat + point.vessel_2.lat) / 2.0 for point in event.points) / len(event.points)
    center_lon = sum((point.vessel_1.lon + point.vessel_2.lon) / 2.0 for point in event.points) / len(event.points)
    min_separation = min(separations)
    median_separation = statistics.median(separations)
    max_separation = max(separations)

    if mean_sog_1 > 2.0 or mean_sog_2 > 2.0:
        return None
    if max(sog_1_values) > 5.0 or max(sog_2_values) > 5.0:
        return None
    if mean_sog_delta > 1.5:
        return None
    if mean_sog_1 > 1.0 and mean_sog_2 > 1.0 and mean_heading_delta is not None and mean_heading_delta > 20.0:
        return None

    role_pair = {vessel_role(vessel_1), vessel_role(vessel_2)}
    is_tanker_pair = "tanker" in role_pair and role_pair.issubset({"tanker", "cargo"})
    if (
        max_separation <= DEFAULT_HIGH_DISTANCE_M
        and duration_min >= 30.0
        and mean_sog_1 <= 1.0
        and mean_sog_2 <= 1.0
        and is_tanker_pair
        and (mean_sog_1 <= 1.0 or mean_sog_2 <= 1.0 or mean_heading_delta is None or mean_heading_delta <= 15.0)
    ):
        confidence_tier = "HIGH"
    elif max_separation <= DEFAULT_MEDIUM_DISTANCE_M and duration_min >= 20.0:
        confidence_tier = "MEDIUM"
    elif max_separation <= DEFAULT_MAX_DISTANCE_M and duration_min >= 15.0:
        confidence_tier = "LOW"
    else:
        return None

    review_flags: list[str] = []
    statuses = {point.vessel_1.navigational_status for point in event.points} | {point.vessel_2.navigational_status for point in event.points}
    if "At anchor" in statuses:
        review_flags.append("anchor_status_present")
    if "Restricted maneuverability" in statuses:
        review_flags.append("restricted_maneuverability_present")
    if "Unknown value" in statuses:
        review_flags.append("unknown_status_present")

    event_id = make_event_id(source_file=source_file, vessel_1_mmsi=event.vessel_1_mmsi, vessel_2_mmsi=event.vessel_2_mmsi, start_time=start_time)
    event_row = {
        "event_id": event_id,
        "source_dataset": source_dataset,
        "source_file": source_file,
        "criteria_version": CRITERIA_VERSION,
        "vessel_1_mmsi": vessel_1.mmsi,
        "vessel_2_mmsi": vessel_2.mmsi,
        "vessel_1_name": vessel_1.name,
        "vessel_2_name": vessel_2.name,
        "vessel_1_ship_type": vessel_1.ship_type,
        "vessel_2_ship_type": vessel_2.ship_type,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_min": round(duration_min, 3),
        "center_lat": round(center_lat, 6),
        "center_lon": round(center_lon, 6),
        "observation_count": len(event.points),
        "min_separation_m": round(min_separation, 3),
        "median_separation_m": round(median_separation, 3),
        "max_separation_m": round(max_separation, 3),
        "vessel_1_mean_sog_kn": round(mean_sog_1, 3),
        "vessel_2_mean_sog_kn": round(mean_sog_2, 3),
        "mean_heading_delta_deg": None if mean_heading_delta is None else round(mean_heading_delta, 3),
        "confidence_tier": confidence_tier,
        "review_status": "pending_imagery",
        "review_notes": ",".join(review_flags),
        "created_at": datetime.now(tz=UTC).isoformat(),
    }

    point_rows = []
    for index, point in enumerate(event.points, start=1):
        point_rows.append(
            {
                "event_id": event_id,
                "point_index": index,
                "timestamp": point.event_timestamp.isoformat(),
                "vessel_1_lat": round(point.vessel_1.lat, 6),
                "vessel_1_lon": round(point.vessel_1.lon, 6),
                "vessel_2_lat": round(point.vessel_2.lat, 6),
                "vessel_2_lon": round(point.vessel_2.lon, 6),
                "separation_m": round(point.separation_m, 3),
                "vessel_1_sog_kn": round(point.vessel_1.sog_kn, 3),
                "vessel_2_sog_kn": round(point.vessel_2.sog_kn, 3),
                "vessel_1_cog_deg": point.vessel_1.cog_deg,
                "vessel_2_cog_deg": point.vessel_2.cog_deg,
            }
        )

    return event_row, point_rows


def sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def write_events_to_sqlite(
    db_path: Path,
    event_rows: list[dict[str, object]],
    point_rows: list[dict[str, object]],
    source_dataset: str,
    source_file: str,
) -> None:
    statements = [
        "PRAGMA foreign_keys = ON;",
        """
        CREATE TABLE IF NOT EXISTS ais_sts_candidates (
            event_id TEXT PRIMARY KEY,
            source_dataset TEXT NOT NULL,
            source_file TEXT NOT NULL,
            criteria_version TEXT NOT NULL,
            vessel_1_mmsi TEXT NOT NULL,
            vessel_2_mmsi TEXT NOT NULL,
            vessel_1_name TEXT,
            vessel_2_name TEXT,
            vessel_1_ship_type TEXT,
            vessel_2_ship_type TEXT,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            duration_min REAL NOT NULL,
            center_lat REAL NOT NULL,
            center_lon REAL NOT NULL,
            observation_count INTEGER NOT NULL,
            min_separation_m REAL NOT NULL,
            median_separation_m REAL NOT NULL,
            max_separation_m REAL NOT NULL,
            vessel_1_mean_sog_kn REAL NOT NULL,
            vessel_2_mean_sog_kn REAL NOT NULL,
            mean_heading_delta_deg REAL,
            confidence_tier TEXT NOT NULL,
            review_status TEXT NOT NULL,
            review_notes TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS ais_sts_candidate_points (
            event_id TEXT NOT NULL,
            point_index INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            vessel_1_lat REAL NOT NULL,
            vessel_1_lon REAL NOT NULL,
            vessel_2_lat REAL NOT NULL,
            vessel_2_lon REAL NOT NULL,
            separation_m REAL NOT NULL,
            vessel_1_sog_kn REAL NOT NULL,
            vessel_2_sog_kn REAL NOT NULL,
            vessel_1_cog_deg REAL,
            vessel_2_cog_deg REAL,
            PRIMARY KEY (event_id, point_index),
            FOREIGN KEY (event_id) REFERENCES ais_sts_candidates(event_id) ON DELETE CASCADE
        );
        """,
        "CREATE INDEX IF NOT EXISTS idx_ais_sts_candidates_time ON ais_sts_candidates (start_time, end_time);",
        "CREATE INDEX IF NOT EXISTS idx_ais_sts_candidates_tier ON ais_sts_candidates (confidence_tier);",
        "BEGIN;",
        (
            "DELETE FROM ais_sts_candidate_points WHERE event_id IN ("
            "SELECT event_id FROM ais_sts_candidates "
            f"WHERE source_dataset = {sql_literal(source_dataset)} "
            f"AND source_file = {sql_literal(source_file)} "
            f"AND criteria_version = {sql_literal(CRITERIA_VERSION)}"
            ");"
        ),
        (
            "DELETE FROM ais_sts_candidates "
            f"WHERE source_dataset = {sql_literal(source_dataset)} "
            f"AND source_file = {sql_literal(source_file)} "
            f"AND criteria_version = {sql_literal(CRITERIA_VERSION)};"
        ),
    ]

    for row in event_rows:
        columns = ", ".join(row.keys())
        values = ", ".join(sql_literal(row[column]) for column in row)
        statements.append(f"INSERT INTO ais_sts_candidates ({columns}) VALUES ({values});")

    for row in point_rows:
        columns = ", ".join(row.keys())
        values = ", ".join(sql_literal(row[column]) for column in row)
        statements.append(f"INSERT INTO ais_sts_candidate_points ({columns}) VALUES ({values});")

    statements.append("COMMIT;")
    sql_script = "\n".join(statements)
    subprocess.run(["sqlite3", str(db_path)], input=sql_script, text=True, check=True)


def scan_denmark_zip(
    zip_path: str | Path,
    db_path: str | Path,
    source_dataset: str,
    insert_tiers: set[str],
    bin_minutes: int = DEFAULT_BIN_MINUTES,
    max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
    max_time_gap_minutes: int = DEFAULT_MAX_TIME_GAP_MINUTES,
) -> dict[str, object]:
    zip_path = Path(zip_path)
    db_path = Path(db_path)
    ensure_parent_dir(db_path)

    max_time_gap = timedelta(minutes=max_time_gap_minutes)
    active_events: dict[tuple[str, str], ActiveEvent] = {}
    event_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    matched_pair_observations = 0
    rows_seen = 0
    usable_rows = 0
    bins_processed = 0

    def flush_finished_events(cutoff: datetime | None, force: bool = False) -> None:
        nonlocal event_rows, point_rows
        expired_keys: list[tuple[str, str]] = []
        for key, event in active_events.items():
            if force or (event.last_seen is not None and cutoff is not None and cutoff - event.last_seen > max_time_gap):
                summary = summarize_event(event=event, source_dataset=source_dataset, source_file=str(zip_path))
                if summary is not None:
                    event_row, rows = summary
                    if event_row["confidence_tier"] in insert_tiers:
                        event_rows.append(event_row)
                        point_rows.extend(rows)
                expired_keys.append(key)
        for key in expired_keys:
            del active_events[key]

    def process_bin(bin_records: dict[str, VesselObservation], cutoff: datetime) -> None:
        nonlocal matched_pair_observations, bins_processed
        bins_processed += 1
        pair_observations = candidate_pairs_for_bin(
            observations=bin_records.values(),
            max_distance_m=max_distance_m,
            max_time_gap=max_time_gap,
        )
        matched_pair_observations += len(pair_observations)
        for pair in pair_observations:
            key = (
                pair.vessel_1.mmsi,
                pair.vessel_2.mmsi,
            ) if pair.vessel_1.mmsi < pair.vessel_2.mmsi else (
                pair.vessel_2.mmsi,
                pair.vessel_1.mmsi,
            )
            event = active_events.get(key)
            if event is None:
                event = ActiveEvent(vessel_1_mmsi=key[0], vessel_2_mmsi=key[1])
                active_events[key] = event
            elif event.last_seen is not None and pair.event_timestamp - event.last_seen > max_time_gap:
                summary = summarize_event(event=event, source_dataset=source_dataset, source_file=str(zip_path))
                if summary is not None:
                    event_row, rows = summary
                    if event_row["confidence_tier"] in insert_tiers:
                        event_rows.append(event_row)
                        point_rows.extend(rows)
                active_events[key] = ActiveEvent(vessel_1_mmsi=key[0], vessel_2_mmsi=key[1])
                event = active_events[key]

            if pair.vessel_1.mmsi == key[0]:
                event.points.append(pair)
            else:
                event.points.append(
                    PairObservation(
                        event_timestamp=pair.event_timestamp,
                        vessel_1=pair.vessel_2,
                        vessel_2=pair.vessel_1,
                        separation_m=pair.separation_m,
                        heading_delta_deg=pair.heading_delta_deg,
                    )
                )
            event.last_seen = pair.event_timestamp

        flush_finished_events(cutoff=cutoff)

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected one CSV inside {zip_path}, found {len(names)} entries")
        with archive.open(names[0], "r") as raw_handle:
            text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", errors="replace", newline="")
            reader = csv.DictReader(text_handle)
            current_bin_start: datetime | None = None
            bin_records: dict[str, VesselObservation] = {}

            for row in reader:
                rows_seen += 1
                observation = normalize_denmark_row(row)
                if observation is None:
                    continue
                usable_rows += 1

                bin_start = floor_to_bin(observation.timestamp, bin_minutes=bin_minutes)
                if current_bin_start is None:
                    current_bin_start = bin_start
                elif bin_start != current_bin_start:
                    process_bin(bin_records=bin_records, cutoff=bin_start)
                    bin_records = {}
                    current_bin_start = bin_start

                existing = bin_records.get(observation.mmsi)
                if existing is None or observation.timestamp >= existing.timestamp:
                    bin_records[observation.mmsi] = observation

            if bin_records:
                process_bin(bin_records=bin_records, cutoff=(current_bin_start or datetime.now(tz=UTC)) + max_time_gap)

    flush_finished_events(cutoff=None, force=True)
    write_events_to_sqlite(
        db_path=db_path,
        event_rows=event_rows,
        point_rows=point_rows,
        source_dataset=source_dataset,
        source_file=str(zip_path),
    )

    tier_counts: dict[str, int] = {}
    for row in event_rows:
        tier = str(row["confidence_tier"])
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    return {
        "source_zip": str(zip_path),
        "db_path": str(db_path),
        "criteria_version": CRITERIA_VERSION,
        "rows_seen": rows_seen,
        "usable_rows": usable_rows,
        "bins_processed": bins_processed,
        "matched_pair_observations": matched_pair_observations,
        "inserted_events": len(event_rows),
        "inserted_point_rows": len(point_rows),
        "tier_counts": tier_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Denmark AIS zip files for likely STS candidate events and write them to SQLite")
    parser.add_argument("--zip", required=True, help="Path to the Denmark AIS zip file")
    parser.add_argument("--db", required=True, help="Path to the SQLite database to write")
    parser.add_argument("--source-dataset", default="AIS_DK_2026", help="Logical dataset name stored in the database")
    parser.add_argument("--insert-tiers", nargs="+", default=sorted(DEFAULT_INSERT_TIERS), choices=["HIGH", "MEDIUM", "LOW"], help="Confidence tiers to insert into the database")
    parser.add_argument("--bin-minutes", type=int, default=DEFAULT_BIN_MINUTES, help="Time bin size used when building pair tracks")
    parser.add_argument("--max-distance-m", type=float, default=DEFAULT_MAX_DISTANCE_M, help="Maximum separation allowed for low-confidence overflow pairing")
    parser.add_argument("--max-time-gap-minutes", type=int, default=DEFAULT_MAX_TIME_GAP_MINUTES, help="Maximum gap allowed between paired observations")
    args = parser.parse_args()

    summary = scan_denmark_zip(
        zip_path=args.zip,
        db_path=args.db,
        source_dataset=args.source_dataset,
        insert_tiers=set(args.insert_tiers),
        bin_minutes=args.bin_minutes,
        max_distance_m=args.max_distance_m,
        max_time_gap_minutes=args.max_time_gap_minutes,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
