"""
=============================================================
  OpenDarkSTS — Flask GUI v3 (Final)
  Detection flow:
    1. YOLO detects STS candidates from satellite image
    2. AIS correlation verifies if vessels are broadcasting
    3. Only AIS-verified events are flagged as confirmed
  Run:  python app.py → http://localhost:5000
=============================================================
"""
from __future__ import annotations

import csv
import json
import multiprocessing
import threading
import traceback
from pathlib import Path
from time import time

import torch
from flask import Flask, jsonify, render_template, request, send_file

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import numpy as np
except ImportError:
    pass

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    from detect import detect_images
    from georeference import georeference_detections
    from correlate_ais import correlate_detections
    from pipeline import run_pipeline, write_pipeline_output
    from sts_utils import (PIPELINE_CSV_FIELDS, flatten_pipeline_result,
                            resolve_image_inputs, write_json, read_json)
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR          = Path(r"C:\Users\Lenovo\Documents\data science project\data science dark fleet.v7i.yolov8 (1)")
OUTPUT_DIR        = BASE_DIR / "runs"
EXPORT_DIR        = OUTPUT_DIR / "exports"
UPLOAD_DIR        = OUTPUT_DIR / "uploads"
YAML_PATH         = BASE_DIR / "dataset.yaml"
BEST_WEIGHTS      = Path(__file__).parent / "weights" / "best.pt"
STATS_FILE        = EXPORT_DIR / "cumulative_stats.json"
SCAN_HISTORY_FILE = EXPORT_DIR / "scan_history.json"

for d in [OUTPUT_DIR, EXPORT_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["ship", "ships"]
COLORS_BGR  = {"ship": (0, 200, 0), "ships": (0, 0, 255)}

CONF_PRESETS = {
    "low":    0.10,
    "medium": 0.25,
    "high":   0.45,
}

MAX_BOX_AREA_FRAC = 0.25   # Lower = stricter filter for false positive large boxes
NMS_IOU_THRESHOLD = 0.30   # Lower = stricter suppression of overlapping boxes

# ─────────────────────────────────────────────────────────────
# DARK STATUS DEFINITIONS
# ─────────────────────────────────────────────────────────────
# DETECTED_ONLY  — STS found by YOLO but no AIS data provided (unverified)
# FULLY_DARK     — STS found + AIS checked + NO vessels nearby (confirmed dark)
# PARTIALLY_DARK — STS found + AIS checked + only 1 vessel nearby (suspicious)
# AIS_PRESENT    — STS found + AIS checked + 2+ vessels nearby (likely legitimate)
# NO_STS         — No STS detected in this image

_model            = None
_training_log     = []
_training_done    = True
_pipeline_results = []
_cumulative_stats = {
    "total_scans": 0, "total_detections": 0,
    "confirmed_dark": 0, "partially_dark": 0,
    "ais_present": 0, "detected_only": 0,
    "history": []
}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# ─────────────────────────────────────────────────────────────
# STATS PERSISTENCE
# ─────────────────────────────────────────────────────────────

def load_stats():
    global _cumulative_stats
    if STATS_FILE.exists():
        try:
            _cumulative_stats = json.loads(STATS_FILE.read_text())
        except Exception:
            pass

def save_stats():
    STATS_FILE.write_text(json.dumps(_cumulative_stats, indent=2))

def update_stats(results: list[dict]):
    global _cumulative_stats
    from datetime import datetime
    confirmed = sum(1 for r in results if r.get("dark_status") in ("FULLY_DARK", "PARTIALLY_DARK"))
    fully     = sum(1 for r in results if r.get("dark_status") == "FULLY_DARK")
    partial   = sum(1 for r in results if r.get("dark_status") == "PARTIALLY_DARK")
    ais_ok    = sum(1 for r in results if r.get("dark_status") == "AIS_PRESENT")
    det_only  = sum(1 for r in results if r.get("dark_status") == "DETECTED_ONLY")

    _cumulative_stats["total_scans"]      += 1
    _cumulative_stats["total_detections"] += len(results)
    _cumulative_stats["confirmed_dark"]   = _cumulative_stats.get("confirmed_dark", 0) + fully
    _cumulative_stats["partially_dark"]   = _cumulative_stats.get("partially_dark", 0) + partial
    _cumulative_stats["ais_present"]      = _cumulative_stats.get("ais_present", 0) + ais_ok
    _cumulative_stats["detected_only"]    = _cumulative_stats.get("detected_only", 0) + det_only
    _cumulative_stats["history"].append({
        "date":       datetime.now().strftime("%Y-%m-%d %H:%M"),
        "scan_id":    f"SCAN-{_cumulative_stats['total_scans']:04d}",
        "confirmed":  confirmed,
        "det_only":   det_only,
        "total":      len(results),
    })
    save_stats()


# ─────────────────────────────────────────────────────────────
# SCAN HISTORY PERSISTENCE
# ─────────────────────────────────────────────────────────────

def load_scan_history():
    global _pipeline_results
    if SCAN_HISTORY_FILE.exists():
        try:
            _pipeline_results = json.loads(SCAN_HISTORY_FILE.read_text())
        except Exception:
            pass

def save_scan_history():
    SCAN_HISTORY_FILE.write_text(json.dumps(_pipeline_results, indent=2))

def clear_scan_history():
    global _pipeline_results
    _pipeline_results = []
    if SCAN_HISTORY_FILE.exists():
        SCAN_HISTORY_FILE.unlink()


load_stats()
load_scan_history()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def get_model(weights_path: str | None = None):
    global _model
    path = Path(weights_path) if weights_path else BEST_WEIGHTS
    if path.exists():
        _model = YOLO(str(path)) if HAS_YOLO else None
    return _model


def is_valid_box(box, img_h: int, img_w: int) -> bool:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_h * img_w
    if img_area == 0:
        return True
    return (box_area / img_area) <= MAX_BOX_AREA_FRAC


def filter_overlapping_boxes(boxes, img_h: int, img_w: int, iou_threshold: float = 0.30):
    """
    Custom NMS to remove overlapping/nested boxes of the same class.
    Keeps the highest-confidence box when multiple boxes overlap significantly.
    Also removes boxes that fully contain smaller boxes (nested big-box false positives).
    """
    # First filter by area
    valid_boxes = [b for b in boxes if is_valid_box(b, img_h, img_w)]
    if len(valid_boxes) <= 1:
        return valid_boxes

    # Sort by confidence descending
    sorted_boxes = sorted(valid_boxes, key=lambda b: float(b.conf[0]), reverse=True)
    keep = []

    for box in sorted_boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cls = int(box.cls[0])
        box_area = (x2 - x1) * (y2 - y1)
        should_keep = True

        for kept in keep:
            kx1, ky1, kx2, ky2 = map(float, kept.xyxy[0])
            kcls = int(kept.cls[0])
            if kcls != cls:
                continue  # Only suppress same class

            # Compute IoU
            ix1 = max(x1, kx1); iy1 = max(y1, ky1)
            ix2 = min(x2, kx2); iy2 = min(y2, ky2)
            iw  = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
            inter = iw * ih
            kept_area = (kx2 - kx1) * (ky2 - ky1)
            union = box_area + kept_area - inter
            iou   = inter / union if union > 0 else 0

            # Containment ratio — if this box contains a kept box mostly inside
            contain_ratio = inter / min(box_area, kept_area) if min(box_area, kept_area) > 0 else 0

            if iou >= iou_threshold or contain_ratio >= 0.70:
                should_keep = False
                break

        if should_keep:
            keep.append(box)

    return keep


def draw_boxes(img_bgr, boxes, img_h: int, img_w: int):
    filtered = filter_overlapping_boxes(boxes, img_h, img_w, iou_threshold=NMS_IOU_THRESHOLD)
    for box in filtered:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        color  = COLORS_BGR.get(label, (255, 255, 0))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        tag = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img_bgr, tag, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img_bgr


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    return jsonify({
        "model_ready":    BEST_WEIGHTS.exists(),
        "pipeline_ready": HAS_PIPELINE,
        "gpu_available":  torch.cuda.is_available(),
        "gpu_name":       torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "results_loaded": len(_pipeline_results),
    })

@app.route("/api/stats")
def api_stats():
    return jsonify(_cumulative_stats)

@app.route("/api/stats/clear", methods=["POST"])
def api_clear_stats():
    global _cumulative_stats
    _cumulative_stats = {
        "total_scans": 0, "total_detections": 0,
        "confirmed_dark": 0, "partially_dark": 0,
        "ais_present": 0, "detected_only": 0,
        "history": []
    }
    save_stats()
    return jsonify({"status": "cleared"})


# ─────────────────────────────────────────────────────────────
# CORE SCAN ROUTE
# ─────────────────────────────────────────────────────────────

@app.route("/api/scan", methods=["POST"])
def api_scan():
    global _pipeline_results
    if not HAS_YOLO:
        return jsonify({"error": "ultralytics not installed"}), 500

    files   = request.files.getlist("images")
    ais_dir = request.form.get("ais_dir", "").strip()
    preset  = request.form.get("preset", "medium")
    conf    = CONF_PRESETS.get(preset, 0.25)
    radius  = float(request.form.get("radius", 1000.0))
    weights = request.form.get("weights", "").strip() or None

    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    model = get_model(weights)
    if model is None:
        return jsonify({"error": "Model not found. Please train first."}), 404

    has_ais = bool(ais_dir and Path(ais_dir).exists())
    scan_results = []

    for file in files:
        ts       = int(time() * 1000)
        img_path = UPLOAD_DIR / f"{ts}_{file.filename}"
        file.save(str(img_path))

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_h, img_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        results = model.predict(img_rgb, conf=conf, iou=NMS_IOU_THRESHOLD, verbose=False)
        r       = results[0]

        # Apply custom NMS to remove overlapping/nested duplicate boxes
        filtered_boxes = filter_overlapping_boxes(r.boxes, img_h, img_w, iou_threshold=NMS_IOU_THRESHOLD)

        annotated = draw_boxes(img_bgr.copy(), r.boxes, img_h, img_w)
        out_name  = f"det_{ts}_{img_path.stem[:40]}.jpg"
        cv2.imwrite(str(EXPORT_DIR / out_name), annotated)

        detections = []
        for box in filtered_boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class":      CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
                "confidence": round(float(box.conf[0]), 3),
                "is_sts":     cls_id == 1,
                "bbox":       list(map(int, box.xyxy[0].tolist())),
            })

        sts_detections = [d for d in detections if d["is_sts"]]
        ais_vessels    = []
        lat, lon       = None, None

        # ── STEP 1: Initial status based on detection only ──
        if not sts_detections:
            dark_status = "NO_STS"
        elif not has_ais:
            # STS detected but no AIS provided — unverified
            dark_status = "DETECTED_ONLY"
        else:
            # ── STEP 2: AIS correlation to verify ──
            dark_status = "DETECTED_ONLY"  # default until AIS confirms
            try:
                geo_results = run_pipeline(
                    image_inputs=[str(img_path)],
                    weights=weights or str(BEST_WEIGHTS),
                    ais_dir=ais_dir,
                    conf=conf,
                    radius_m=radius,
                )
                if geo_results:
                    first       = geo_results[0]
                    ais_vessels = first.get("ais_vessels", [])
                    lat         = first.get("center_lat")
                    lon         = first.get("center_lon")
                    ais_count   = len(ais_vessels)

                    # ── STEP 3: Classify based on AIS count ──
                    if ais_count == 0:
                        dark_status = "FULLY_DARK"       # confirmed dark fleet
                    elif ais_count == 1:
                        dark_status = "PARTIALLY_DARK"   # suspicious
                    else:
                        dark_status = "AIS_PRESENT"      # likely legitimate
            except Exception as e:
                dark_status = "AIS_ERROR"

        # Only flag as confirmed STS if AIS verified dark activity
        is_confirmed = dark_status in ("FULLY_DARK", "PARTIALLY_DARK")
        is_detected  = dark_status == "DETECTED_ONLY"

        scan_results.append({
            "filename":         file.filename,
            "result_image":     f"/exports/{out_name}",
            "total":            len(detections),
            "sts_count":        len(sts_detections),
            "ship_count":       len(detections) - len(sts_detections),
            "num_detections":   len(detections),
            "num_ships":        len(detections) - len(sts_detections),
            "num_sts":          len(sts_detections),
            "conf_thresh":      conf,
            "detections":       detections,
            "ais_vessels":      ais_vessels,
            "dark_status":      dark_status,
            "lat":              lat,
            "lon":              lon,
            "is_confirmed_sts": is_confirmed,
            "is_detected_only": is_detected,
            "ais_provided":     has_ais,
            "preset":           preset,
        })

    # Append to history and persist
    _pipeline_results = _pipeline_results + scan_results
    save_scan_history()
    update_stats([r for r in scan_results if r["dark_status"] != "NO_STS"])

    confirmed = sum(1 for r in scan_results if r["is_confirmed_sts"])
    detected  = sum(1 for r in scan_results if r["is_detected_only"])

    return jsonify({
        "total_images":   len(scan_results),
        "confirmed_sts":  confirmed,
        "detected_only":  detected,
        "results":        scan_results,
    })


@app.route("/api/scan/clear", methods=["POST"])
def api_clear_scans():
    clear_scan_history()
    return jsonify({"status": "cleared"})


@app.route("/api/scan/history")
def api_scan_history():
    return jsonify(_pipeline_results)


# ─────────────────────────────────────────────────────────────
# DEVELOPER DETECT
# ─────────────────────────────────────────────────────────────

@app.route("/api/detect", methods=["POST"])
def api_detect():
    if not HAS_YOLO:
        return jsonify({"error": "ultralytics not installed"}), 500

    file    = request.files.get("image")
    preset  = request.form.get("preset", "medium")
    conf    = CONF_PRESETS.get(preset, 0.25)
    weights = request.form.get("weights", "").strip() or None

    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    model = get_model(weights)
    if model is None:
        return jsonify({"error": "Model not found."}), 404

    ts       = int(time() * 1000)
    img_path = UPLOAD_DIR / f"{ts}_{file.filename}"
    file.save(str(img_path))

    img_bgr      = cv2.imread(str(img_path))
    img_h, img_w = img_bgr.shape[:2]
    img_rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results      = model.predict(img_rgb, conf=conf, iou=NMS_IOU_THRESHOLD, verbose=False)
    r            = results[0]

    annotated = draw_boxes(img_bgr.copy(), r.boxes, img_h, img_w)
    out_name  = f"dev_{ts}.jpg"
    cv2.imwrite(str(EXPORT_DIR / out_name), annotated)

    detections = []
    for box in r.boxes:
        if not is_valid_box(box, img_h, img_w):
            continue
        cls_id = int(box.cls[0])
        detections.append({
            "class":      CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
            "confidence": round(float(box.conf[0]), 3),
            "status":     "STS Candidate" if cls_id == 1 else "Normal Ship",
            "bbox":       list(map(int, box.xyxy[0].tolist())),
        })

    return jsonify({
        "detections":   detections,
        "total":        len(detections),
        "sts_count":    sum(1 for d in detections if d["class"] == "ships"),
        "ship_count":   sum(1 for d in detections if d["class"] == "ship"),
        "result_image": f"/exports/{out_name}",
        "conf_used":    conf,
        "preset":       preset,
        "note":         "These are YOLO candidates only — use Scan Images with AIS data for verification",
    })


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    global _training_log, _training_done
    _training_log  = ["⏳ Preparing training…"]
    _training_done = False

    data      = request.get_json()
    data_yaml = data.get("data_yaml", str(YAML_PATH))
    out_dir   = data.get("output_dir", str(OUTPUT_DIR / "model_compare"))
    epochs    = int(data.get("epochs", 50))
    imgsz     = int(data.get("imgsz", 1280))
    batch     = int(data.get("batch", 8))
    patience  = int(data.get("patience", 0))
    seed      = int(data.get("seed", 42))

    if not Path(data_yaml).exists():
        return jsonify({"error": f"data.yaml not found: {data_yaml}"}), 400

    def train_thread():
        global _training_log, _training_done
        try:
            device = "0" if torch.cuda.is_available() else "cpu"
            _training_log.append(f"🖥️  Device: {'GPU ✅' if device=='0' else 'CPU ⚠️'}")
            _training_log.append(f"📋 Epochs={epochs} ImgSz={imgsz} Batch={batch}")
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            m = YOLO(str(BEST_WEIGHTS) if BEST_WEIGHTS.exists() else "yolov8l.pt")
            start = time()
            m.train(
                data=data_yaml, epochs=epochs, imgsz=imgsz,
                batch=batch, device=device, project=out_dir,
                name="retrain", exist_ok=True, workers=0,
                patience=patience, cos_lr=True, lr0=0.0005, lrf=0.001,
                warmup_epochs=10, degrees=90, flipud=0.5, fliplr=0.5,
                mosaic=1.0, scale=0.5, plots=True, save=True, verbose=True,
            )
            _training_log.append(f"\n✅ Done in {(time()-start)/60:.1f} min!")
        except Exception as e:
            _training_log.append(f"\n❌ Error: {e}\n{traceback.format_exc()}")
        finally:
            _training_done = True

    threading.Thread(target=train_thread, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/train/log")
def api_train_log():
    return jsonify({"log": "\n".join(_training_log), "done": _training_done})


# ─────────────────────────────────────────────────────────────
# EXPORT & STATIC
# ─────────────────────────────────────────────────────────────

@app.route("/api/export/<fmt>")
def api_export(fmt):
    if not _pipeline_results:
        return jsonify({"error": "No results. Run a scan first."}), 404

    if fmt == "json":
        out = EXPORT_DIR / "export_results.json"
        out.write_text(json.dumps(_pipeline_results, indent=2))
    else:
        out = EXPORT_DIR / "export_results.csv"
        keys = ["filename", "total", "sts_count", "ship_count",
                "dark_status", "lat", "lon", "preset", "ais_provided"]
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(_pipeline_results)

    return send_file(str(out), as_attachment=True)


@app.route("/exports/<filename>")
def serve_export(filename):
    return send_file(str(EXPORT_DIR / filename))


# ─────────────────────────────────────────────────────────────
# AIS DATA MANAGER ROUTES
# ─────────────────────────────────────────────────────────────

AIS_DIR         = BASE_DIR.parent / "AIS_Downloads"
AIS_DIR.mkdir(parents=True, exist_ok=True)

_ais_download_log    = []
_ais_download_done   = True
_ais_download_total  = 0
_ais_download_done_n = 0

AIS_BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/ais-{date}.csv.zst"


def _generate_dates(year=None, month=None, date_from=None, date_to=None, single=None):
    import datetime
    dates = []
    if single:
        dates = [single]
    elif date_from and date_to:
        d = datetime.date.fromisoformat(date_from)
        e = datetime.date.fromisoformat(date_to)
        while d <= e:
            dates.append(d.isoformat())
            d += datetime.timedelta(days=1)
    elif year and month:
        import calendar
        _, days = calendar.monthrange(int(year), int(month))
        for day in range(1, days + 1):
            dates.append(f"{year}-{int(month):02d}-{day:02d}")
    elif year:
        d = datetime.date(int(year), 1, 1)
        e = datetime.date(int(year), 12, 31)
        while d <= e:
            dates.append(d.isoformat())
            d += datetime.timedelta(days=1)
    return dates


@app.route("/api/ais/download", methods=["POST"])
def api_ais_download():
    global _ais_download_log, _ais_download_done, _ais_download_total, _ais_download_done_n

    data      = request.get_json()
    mode      = data.get("mode", "single")   # single | range | month | year
    year      = data.get("year")
    month     = data.get("month")
    date_from = data.get("date_from")
    date_to   = data.get("date_to")
    single    = data.get("single")

    if mode == "single":
        dates = _generate_dates(single=single)
    elif mode == "range":
        dates = _generate_dates(date_from=date_from, date_to=date_to)
    elif mode == "month":
        dates = _generate_dates(year=year, month=month)
    elif mode == "year":
        dates = _generate_dates(year=year)
    else:
        return jsonify({"error": "Invalid mode"}), 400

    if not dates:
        return jsonify({"error": "No dates generated — check inputs"}), 400

    _ais_download_log    = [f"📡 Preparing to download {len(dates)} file(s)…"]
    _ais_download_done   = False
    _ais_download_total  = len(dates)
    _ais_download_done_n = 0

    def download_thread():
        global _ais_download_done, _ais_download_done_n
        import urllib.request
        try:
            try:
                import zstandard as zstd
                import pandas as pd
                HAS_ZST = True
            except ImportError:
                HAS_ZST = False
                _ais_download_log.append("⚠️  zstandard/pandas not installed. Run: pip install zstandard pandas")

            # Get coordinate filter if provided
            lat_min = data.get("lat_min")
            lat_max = data.get("lat_max")
            lon_min = data.get("lon_min")
            lon_max = data.get("lon_max")
            has_filter = all(v is not None for v in [lat_min, lat_max, lon_min, lon_max])

            if has_filter:
                _ais_download_log.append(f"📍 Filtering to: Lat {lat_min}°→{lat_max}°, Lon {lon_min}°→{lon_max}°")
            else:
                _ais_download_log.append("🌍 No coordinate filter — downloading full dataset")

            import zipfile

            failed = []
            for i, date in enumerate(dates, 1):
                yr        = int(date[:4])
                date_us   = date.replace('-', '_')   # YYYY_MM_DD
                csv_name  = f"AIS_{date_us}.csv"
                csv_path  = AIS_DIR / csv_name

                if csv_path.exists():
                    _ais_download_log.append(f"[{i}/{len(dates)}] ⏭️  Already exists: {csv_name}")
                    _ais_download_done_n += 1
                    continue

                # Pick URL format based on year
                # 2015-2024: AIS_YYYY_MM_DD.zip
                # 2025+:     ais-YYYY-MM-DD.csv.zst
                if yr >= 2025:
                    url      = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{yr}/ais-{date}.csv.zst"
                    dl_name  = f"ais-{date}.csv.zst"
                    use_zst  = True
                else:
                    url      = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{yr}/AIS_{date_us}.zip"
                    dl_name  = f"AIS_{date_us}.zip"
                    use_zst  = False

                dl_path = AIS_DIR / dl_name

                try:
                    _ais_download_log.append(f"[{i}/{len(dates)}] ⬇️  Downloading {dl_name}…")

                    # Use urlopen with longer timeout and chunked download with retries
                    import socket
                    socket.setdefaulttimeout(120)  # 2 minute timeout per chunk

                    max_retries = 3
                    last_err    = None
                    for attempt in range(max_retries):
                        try:
                            with urllib.request.urlopen(url, timeout=120) as response:
                                total_size = int(response.headers.get('Content-Length', 0))
                                downloaded = 0
                                chunk_size = 1024 * 1024  # 1 MB chunks
                                with open(dl_path, 'wb') as f_out:
                                    while True:
                                        chunk = response.read(chunk_size)
                                        if not chunk:
                                            break
                                        f_out.write(chunk)
                                        downloaded += len(chunk)
                                        # Log progress every 50MB
                                        if downloaded % (50 * 1024 * 1024) < chunk_size:
                                            pct = (downloaded * 100 // total_size) if total_size else 0
                                            _ais_download_log.append(
                                                f"[{i}/{len(dates)}]    {downloaded/1024**2:.0f}MB / {total_size/1024**2:.0f}MB ({pct}%)"
                                            )
                            last_err = None
                            break  # success
                        except Exception as e:
                            last_err = e
                            if dl_path.exists():
                                dl_path.unlink()
                            _ais_download_log.append(f"[{i}/{len(dates)}] ⚠️  Attempt {attempt+1} failed: {e} — retrying…")

                    if last_err:
                        raise last_err

                    raw_mb = dl_path.stat().st_size / 1024**2

                    # Decompress based on format
                    temp_path = AIS_DIR / f"_temp_{date}.csv"

                    if use_zst:
                        if not HAS_ZST:
                            _ais_download_log.append(f"[{i}/{len(dates)}] ⚠️  Install zstandard to auto-decompress")
                            _ais_download_done_n += 1
                            continue
                        _ais_download_log.append(f"[{i}/{len(dates)}] 📦 Decompressing zst {raw_mb:.1f}MB…")
                        with open(dl_path, "rb") as f_in:
                            dctx = zstd.ZstdDecompressor()
                            with open(temp_path, "wb") as f_out:
                                dctx.copy_stream(f_in, f_out)
                    else:
                        _ais_download_log.append(f"[{i}/{len(dates)}] 📦 Extracting zip {raw_mb:.1f}MB…")
                        with zipfile.ZipFile(dl_path, 'r') as zf:
                            # Find CSV inside zip
                            csv_inside = next((n for n in zf.namelist() if n.endswith('.csv')), None)
                            if not csv_inside:
                                raise Exception("No CSV found inside ZIP")
                            with zf.open(csv_inside) as f_in, open(temp_path, "wb") as f_out:
                                f_out.write(f_in.read())

                    dl_path.unlink()  # remove archive

                    if has_filter:
                        _ais_download_log.append(f"[{i}/{len(dates)}] 📍 Filtering to coordinates…")
                        df = pd.read_csv(temp_path, low_memory=False)
                        total_rows = len(df)

                        # Detect column names — handles LAT/LON, lat/lon, latitude/longitude
                        cols = {c.strip().lower(): c.strip() for c in df.columns}
                        lat_col = cols.get('lat') or cols.get('latitude')
                        lon_col = cols.get('lon') or cols.get('long') or cols.get('longitude')

                        if not lat_col or not lon_col:
                            _ais_download_log.append(f"[{i}/{len(dates)}] ⚠️  Could not find LAT/LON columns — saving unfiltered")
                            temp_path.rename(csv_path)
                        else:
                            filtered = df[
                                (df[lat_col] >= float(lat_min)) & (df[lat_col] <= float(lat_max)) &
                                (df[lon_col] >= float(lon_min)) & (df[lon_col] <= float(lon_max))
                            ]
                            filtered.to_csv(csv_path, index=False)
                            temp_path.unlink()
                            csv_mb = csv_path.stat().st_size / 1024**2
                            _ais_download_log.append(
                                f"[{i}/{len(dates)}] ✅ {csv_name} — "
                                f"{len(filtered):,}/{total_rows:,} rows kept ({csv_mb:.1f}MB)"
                            )
                    else:
                        temp_path.rename(csv_path)
                        csv_mb = csv_path.stat().st_size / 1024**2
                        _ais_download_log.append(f"[{i}/{len(dates)}] ✅ {csv_name} ({csv_mb:.1f}MB)")

                    _ais_download_done_n += 1

                except Exception as e:
                    _ais_download_log.append(f"[{i}/{len(dates)}] ❌ Failed: {e}")
                    failed.append(date)
                    for p in [dl_path, AIS_DIR / f"_temp_{date}.csv"]:
                        if p.exists():
                            p.unlink()

            _ais_download_log.append(f"\n✅ Complete! {_ais_download_done_n}/{len(dates)} files downloaded")
            if failed:
                _ais_download_log.append(f"❌ Failed: {', '.join(failed)}")
            _ais_download_log.append(f"📁 Saved to: {AIS_DIR}")

        except Exception as e:
            _ais_download_log.append(f"\n❌ Error: {e}")
        finally:
            _ais_download_done = True

    threading.Thread(target=download_thread, daemon=True).start()
    return jsonify({"status": "started", "total": len(dates)})


@app.route("/api/ais/log")
def api_ais_log():
    return jsonify({
        "log":   "\n".join(_ais_download_log),
        "done":  _ais_download_done,
        "total": _ais_download_total,
        "done_n": _ais_download_done_n,
    })


@app.route("/api/ais/files")
def api_ais_files():
    files = []
    for f in sorted(AIS_DIR.glob("AIS_*.csv")):
        files.append({
            "name":    f.name,
            "path":    str(f.parent),
            "size_mb": round(f.stat().st_size / 1024**2, 1),
            "date":    f.stem.replace("AIS_", "").replace("_", "-"),
        })
    return jsonify(files)


@app.route("/api/ais/delete", methods=["POST"])
def api_ais_delete():
    data     = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "No filename provided"}), 400
    path = AIS_DIR / filename
    if path.exists():
        path.unlink()
        return jsonify({"status": "deleted"})
    return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("=" * 55)
    print("  🛸 OpenDarkSTS GUI v3")
    print("  Open: http://localhost:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)
