"""Experimental candidate review and explicit human-label transfer to Improve."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, PngImagePlugin

from .workspace import WORKSPACE_DIR, atomic_json, contained, exclusive_lock, sha256
from .jobs import now, scan_description


def run_directory(value):
    path = contained(value, WORKSPACE_DIR)
    if not contained(path / "run.json", path).is_file() or not contained(path / "dark_sts.json", path).is_file():
        raise ValueError("Select a completed detection output directory")
    return path


def list_runs():
    records = []
    for path in (WORKSPACE_DIR / "reports" / "operational").glob("*/run.json"):
        try:
            directory = run_directory(path.parent)
            record = json.loads(path.read_text())
            description = scan_description({"images": record.get("inputs", []), "date": record.get("date"),
                                           "conf": record.get("confidence", .25), "radius": record.get("ais_radius_m", 1000)},
                                          record.get("started_at"))
            records.append({**record, "directory": str(directory), "description": description})
        except (ValueError, OSError):
            continue
    return sorted(records, key=lambda r: r.get("started_at", ""), reverse=True)


def read_results(directory):
    root = run_directory(directory)
    decision_path = contained(root / "decisions.json", root)
    decisions = json.loads(decision_path.read_text()) if decision_path.is_file() else {}
    return [{**row, "human_review": decisions.get(row["detection_id"])} for row in json.loads((root / "dark_sts.json").read_text())]


def find_detection(directory, detection_id):
    for row in read_results(directory):
        if row["detection_id"] == detection_id:
            return row
    raise ValueError("Detection not found in this run")


def review_detection(directory, detection_id, verdict, *, reviewer="default", training_label=None):
    if verdict not in {"real_sts", "not_sts", "unclear"}:
        raise ValueError("Choose real_sts, not_sts, or unclear")
    if not re.fullmatch(r"[\w-]{1,80}", reviewer):
        raise ValueError("Reviewer must be a simple name")
    label = "ships_sts" if verdict == "real_sts" else training_label
    if (verdict == "not_sts" and label not in {None, "ship_non_sts", "neither"}) or (verdict == "unclear" and label):
        raise ValueError("Invalid training label for this verdict")
    root = run_directory(directory)
    with exclusive_lock(root / ".review.lock", blocking=True):
        row = find_detection(root, detection_id)
        previous = row.get("human_review") or {}
        if previous.get("transfer"):
            raise ValueError("Already sent to Improve; edit the training review there")
        path = root / "decisions.json"
        decisions = json.loads(path.read_text()) if path.is_file() else {}
        decisions[detection_id] = {"verdict": verdict, "training_label": label, "reviewer": reviewer, "reviewed_at": now()}
        atomic_json(path, decisions)
        return decisions[detection_id]


def crop_detection(directory, detection_id, *, annotate=False, training_context=False):
    row = find_detection(directory, detection_id)
    source = Path(row["image_path"]).resolve()
    record = json.loads((run_directory(directory) / "run.json").read_text())
    if str(source) not in record["inputs"]:
        raise ValueError("Image is not an input of this run")
    expected_hash = record.get("input_sha256", {}).get(str(source))
    if expected_hash and sha256(source) != expected_hash:
        raise ValueError("Source image changed since detection; run it again before reviewing")
    with Image.open(source) as image:
        width, height = image.size
        cx, cy, w, h = [float(row[k]) for k in ["cx_px", "cy_px", "w_px", "h_px"]]
        if not all(math.isfinite(v) for v in [cx, cy, w, h]) or w <= 0 or h <= 0:
            raise ValueError("Invalid detection box")
        # Tight crop: its human verdict applies to this box, not to unrelated objects on the tile.
        left, top = max(0, math.floor(cx-w/2)), max(0, math.floor(cy-h/2))
        right, bottom = min(width, math.ceil(cx+w/2)), min(height, math.ceil(cy+h/2))
        if right <= left or bottom <= top:
            raise ValueError("Detection is outside the image")
        display_box = (max(0,left-24), max(0,top-24), min(width,right+24), min(height,bottom+24)) if annotate else (left,top,right,bottom)
        if training_context:
            # Native-resolution context, comparable to the 960 px training tiles.
            # Larger boxes retain at least four box-widths of context where the
            # source permits it. Clip the window at the original tile boundary.
            size = max(960, math.ceil(4 * w), math.ceil(4 * h))
            display_box = (max(0, math.floor(cx-size/2)), max(0, math.floor(cy-size/2)),
                           min(width, math.ceil(cx+size/2)), min(height, math.ceil(cy+size/2)))
        crop = image.convert("RGB").crop(display_box)
    if annotate:
        ImageDraw.Draw(crop).rectangle((left-display_box[0],top-display_box[1],right-display_box[0],bottom-display_box[1]), outline="#eab849", width=1)
    return crop, row, display_box if training_context else (left, top, right, bottom)


def export_results(directory, format):
    rows = read_results(directory)
    if format == "geojson":
        payload = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["center_lon"], r["center_lat"]]}, "properties": r} for r in rows]}
        return json.dumps(payload, allow_nan=False).encode()
    if format == "json":
        return json.dumps(rows, allow_nan=False).encode()
    if format != "csv":
        raise ValueError("Export format must be csv, json, or geojson")
    from .sts_utils import PIPELINE_CSV_FIELDS, flatten_pipeline_result
    stream = io.StringIO(newline="")
    review_fields = {"review_verdict": "verdict", "review_label": "training_label", "reviewer": "reviewer", "review_time": "reviewed_at"}
    writer = csv.DictWriter(stream, fieldnames=[*PIPELINE_CSV_FIELDS, *review_fields])
    writer.writeheader()
    for row in rows:
        decision = row.get("human_review") or {}
        writer.writerow({**flatten_pipeline_result(row), **{field: decision.get(key) or "" for field, key in review_fields.items()}})
    return stream.getvalue().encode()


def export_filename(directory, extension):
    root = run_directory(directory)
    record = json.loads((root / "run.json").read_text())
    timestamp = re.sub(r"[^0-9A-Za-z_-]", "-", str(record.get("started_at") or root.name))
    identity = hashlib.sha256(str(root).encode()).hexdigest()[:8]
    return f"reviewed_candidates_{timestamp[:60]}_{identity}.{extension}"


def send_to_improve(directory, detection_id):
    from . import training_pipeline_lib as training, visual_review_lib as review
    from .sts_utils import load_bbox_for_image
    root = run_directory(directory)
    with exclusive_lock(root / ".review.lock", blocking=True), exclusive_lock(WORKSPACE_DIR / ".results-import.lock", blocking=True):
        crop, row, bounds_px = crop_detection(root, detection_id, training_context=True)
        decision = row.get("human_review") or {}
        if decision.get("transfer"):
            return decision["transfer"]
        if decision.get("verdict") not in {"real_sts", "not_sts"} or not decision.get("training_label"):
            raise ValueError("A human training label is required; for not STS choose single ship or neither")
        token = hashlib.sha256(("context-v1:"+str(root)+detection_id).encode()).hexdigest()[:20]
        source = contained(WORKSPACE_DIR / "inputs" / "reviewed_candidates" / f"{row['date']}_{token}.png", WORKSPACE_DIR)
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            # Different candidates may share the same context pixels. Keep their
            # annotation identities distinct, and identical candidates deduplicable.
            info = PngImagePlugin.PngInfo()
            info.add_text("candidate", json.dumps([row["image_path"], *[row[k] for k in ("cx_px", "cy_px", "w_px", "h_px")]]))
            crop.save(source, pnginfo=info)
        manifest_path = WORKSPACE_DIR / "manifests" / "training_import_manifest.json"
        reviews_dir = WORKSPACE_DIR / "reviews"
        result = training.import_training_data([source], manifest_path=manifest_path,
                    cache_dir=WORKSPACE_DIR / "cache" / "imported_images", reviews_dir=reviews_dir,
                    reviewer=decision["reviewer"], album_name="Reviewed detection candidates", auto_detect=False)
        imported = result["imported"] + [item for item in result["skipped"] if item.get("status") == "already_imported"]
        if not imported:
            raise ValueError("Could not import reviewed crop: " + str(result.get("failed")))
        key = imported[0]["image_key"]
        manifest = training.load_import_manifest(manifest_path)
        entry = manifest["images"][key]
        path = review.review_file_path(key, decision["reviewer"], reviews_dir)
        doc = review.load_review_document(path)
        if doc["review_status"] == "completed":
            labels = {d.get("review_label") for d in doc["detections"] if d.get("reviewed")}
            if labels != {decision["training_label"]}:
                raise ValueError("This crop already has a different completed label in Improve; reconcile it there")
        # A duplicate crop keeps its original provenance and split group.
        if not entry.get("operational_source"):
            parent_bounds = load_bbox_for_image(row["image_path"])
            with Image.open(row["image_path"]) as parent:
                parent_width, parent_height = parent.size
            west, south, east, north = parent_bounds
            left, top, right, bottom = bounds_px
            crop_bounds = [west + left/parent_width*(east-west), north-bottom/parent_height*(north-south),
                           west + right/parent_width*(east-west), north-top/parent_height*(north-south)]
            entry.setdefault("render", {}).setdefault("metadata", {})["bounds"] = dict(zip(("left", "bottom", "right", "top"), crop_bounds))
            entry["date"] = row["date"]
            entry["operational_source"] = {"directory": str(root), "detection_id": detection_id, "image_path": row["image_path"],
                "source_sha256": sha256(Path(row["image_path"])), "source_bounds": list(parent_bounds),
                "crop_pixels": list(bounds_px), "bounds": crop_bounds, "crs": "EPSG:4326",
                "candidate_pixels": {key: row[key] for key in ("cx_px", "cy_px", "w_px", "h_px")}}
            atomic_json(source.with_suffix(".json"), {"bbox": crop_bounds, "crs": "EPSG:4326", "operational_source": entry["operational_source"]})
            training.write_json(manifest_path, manifest)
        if doc["review_status"] != "completed":
            label = decision["training_label"]
            class_id = 0 if label == "ship_non_sts" else 1
            doc.update(review_status="completed", detail_cursor=None, date=row["date"], operational_source=entry["operational_source"])
            doc["detections"] = [{"detection_id": "human-0001", "source_type": "manual", "parent_detection_id": None,
                "class_id": class_id, "class_name": review.CLASS_ID_TO_NAME[class_id], "confidence": None,
                "x_center": (row["cx_px"] - bounds_px[0]) / crop.width,
                "y_center": (row["cy_px"] - bounds_px[1]) / crop.height,
                "width": row["w_px"] / crop.width, "height": row["h_px"] / crop.height, "review_label": label,
                "reviewed": True, "reviewed_at": decision["reviewed_at"], "source_models": []}]
            review.save_review_document(doc, path)
        transfer = {"image_key": key, "review_path": str(path), "transferred_at": now()}
        decisions = json.loads((root / "decisions.json").read_text())
        decisions[detection_id]["transfer"] = transfer
        atomic_json(root / "decisions.json", decisions)
        return transfer
