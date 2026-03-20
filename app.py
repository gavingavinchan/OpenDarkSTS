"""
=============================================================
  OpenDarkSTS — Flask GUI v3 (Rebuilt)
  Run:  python app.py
  Then open:  http://localhost:5000
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
from typing import Any

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
BASE_DIR     = Path(r"C:\Users\Lenovo\Documents\data science project\data science dark fleet.v7i.yolov8 (1)")
OUTPUT_DIR   = BASE_DIR / "runs"
EXPORT_DIR   = OUTPUT_DIR / "exports"
UPLOAD_DIR   = OUTPUT_DIR / "uploads"
YAML_PATH    = BASE_DIR / "dataset.yaml"
BEST_WEIGHTS = OUTPUT_DIR / "dark_fleet_yolov8l" / "weights" / "best.pt"
STATS_FILE   = EXPORT_DIR / "cumulative_stats.json"

for d in [OUTPUT_DIR, EXPORT_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["ship", "ships"]
COLORS_BGR  = {"ship": (0, 200, 0), "ships": (0, 0, 255)}

# Confidence presets — Low/Medium/High
CONF_PRESETS = {
    "low":    0.10,   # catches more but may have false positives
    "medium": 0.25,   # balanced — recommended
    "high":   0.45,   # precise but may miss faint detections
}

MAX_BOX_AREA_FRAC = 0.50   # filter boxes covering >50% of image

# Global state
_model            = None
_training_log     = []
_training_done    = True
_pipeline_results = []
_cumulative_stats = {
    "total_scans": 0,
    "total_detections": 0,
    "confirmed_sts": 0,
    "fully_dark": 0,
    "partially_dark": 0,
    "ais_present": 0,
    "history": []
}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# ─────────────────────────────────────────────────────────────
# STATS
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
    sts     = sum(1 for r in results if r.get("is_confirmed_sts"))
    fully   = sum(1 for r in results if r.get("dark_status") == "FULLY_DARK")
    partial = sum(1 for r in results if r.get("dark_status") == "PARTIALLY_DARK")
    ais_ok  = sum(1 for r in results if r.get("dark_status") == "AIS_PRESENT")

    _cumulative_stats["total_scans"]      += 1
    _cumulative_stats["total_detections"] += len(results)
    _cumulative_stats["confirmed_sts"]    += sts
    _cumulative_stats["fully_dark"]       += fully
    _cumulative_stats["partially_dark"]   += partial
    _cumulative_stats["ais_present"]      += ais_ok
    _cumulative_stats["history"].append({
        "date":       datetime.now().strftime("%Y-%m-%d %H:%M"),
        "scan_id":    f"SCAN-{_cumulative_stats['total_scans']:04d}",
        "sts_count":  sts,
        "total":      len(results),
        "fully_dark": fully,
    })
    save_stats()

load_stats()


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


def draw_boxes(img_bgr, boxes, img_h: int, img_w: int):
    for box in boxes:
        if not is_valid_box(box, img_h, img_w):
            continue
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


@app.route("/api/scan", methods=["POST"])
def api_scan():
    global _pipeline_results
    if not HAS_YOLO:
        return jsonify({"error": "ultralytics not installed"}), 500

    files      = request.files.getlist("images")
    ais_dir    = request.form.get("ais_dir", "").strip()
    preset     = request.form.get("preset", "medium")
    conf       = CONF_PRESETS.get(preset, 0.25)
    radius     = float(request.form.get("radius", 1000.0))
    weights    = request.form.get("weights", "").strip() or None

    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    model = get_model(weights)
    if model is None:
        return jsonify({"error": "Model not found. Please train first."}), 404

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

        results = model.predict(img_rgb, conf=conf, verbose=False)
        r       = results[0]

        annotated = draw_boxes(img_bgr.copy(), r.boxes, img_h, img_w)
        out_name  = f"det_{ts}_{img_path.stem[:40]}.jpg"
        out_path  = EXPORT_DIR / out_name
        cv2.imwrite(str(out_path), annotated)

        detections = []
        for box in r.boxes:
            if not is_valid_box(box, img_h, img_w):
                continue
            cls_id = int(box.cls[0])
            detections.append({
                "class":      CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
                "confidence": round(float(box.conf[0]), 3),
                "is_sts":     cls_id == 1,
                "bbox":       list(map(int, box.xyxy[0].tolist())),
            })

        sts_detections = [d for d in detections if d["is_sts"]]
        ais_vessels, dark_status, lat, lon = [], "UNKNOWN", None, None

        if ais_dir and HAS_PIPELINE and sts_detections:
            try:
                geo_results = run_pipeline(
                    image_inputs=[str(img_path)],
                    weights=weights or str(BEST_WEIGHTS),
                    ais_dir=ais_dir,
                    conf=conf,
                    radius_m=radius,
                )
                if geo_results:
                    first        = geo_results[0]
                    ais_vessels  = first.get("ais_vessels", [])
                    dark_status  = first.get("dark_status", "UNKNOWN")
                    lat          = first.get("center_lat")
                    lon          = first.get("center_lon")
            except Exception:
                dark_status = "AIS_ERROR"

        scan_results.append({
            "filename":         file.filename,
            "result_image":     f"/exports/{out_name}",
            "total":            len(detections),
            "sts_count":        len(sts_detections),
            "ship_count":       len(detections) - len(sts_detections),
            "detections":       detections,
            "ais_vessels":      ais_vessels,
            "dark_status":      dark_status,
            "lat":              lat,
            "lon":              lon,
            "is_confirmed_sts": len(sts_detections) > 0,
            "preset":           preset,
        })

    _pipeline_results = scan_results
    update_stats(scan_results)

    return jsonify({
        "total_images":  len(scan_results),
        "confirmed_sts": sum(1 for r in scan_results if r["is_confirmed_sts"]),
        "results":       scan_results,
    })


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

    img_bgr          = cv2.imread(str(img_path))
    img_h, img_w     = img_bgr.shape[:2]
    img_rgb          = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results          = model.predict(img_rgb, conf=conf, verbose=False)
    r                = results[0]

    annotated = draw_boxes(img_bgr.copy(), r.boxes, img_h, img_w)
    out_name  = f"dev_{ts}.jpg"
    out_path  = EXPORT_DIR / out_name
    cv2.imwrite(str(out_path), annotated)

    detections = []
    for box in r.boxes:
        if not is_valid_box(box, img_h, img_w):
            continue
        cls_id = int(box.cls[0])
        detections.append({
            "class":      CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
            "confidence": round(float(box.conf[0]), 3),
            "status":     "STS / Dark Fleet" if cls_id == 1 else "Normal Ship",
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
    })


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
                batch=batch, device=device,
                project=out_dir, name="retrain",
                exist_ok=True, workers=0,
                patience=patience, cos_lr=True,
                lr0=0.0005, lrf=0.001,
                warmup_epochs=10,
                degrees=90, flipud=0.5, fliplr=0.5,
                mosaic=1.0, scale=0.5,
                plots=True, save=True, verbose=True,
            )
            elapsed = time() - start
            _training_log.append(f"\n✅ Done in {elapsed/60:.1f} min!")
        except Exception as e:
            _training_log.append(f"\n❌ Error: {e}\n{traceback.format_exc()}")
        finally:
            _training_done = True

    threading.Thread(target=train_thread, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/train/log")
def api_train_log():
    return jsonify({"log": "\n".join(_training_log), "done": _training_done})


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
                "dark_status", "lat", "lon", "preset"]
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(_pipeline_results)

    return send_file(str(out), as_attachment=True)


@app.route("/exports/<filename>")
def serve_export(filename):
    return send_file(str(EXPORT_DIR / filename))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("=" * 55)
    print("  🛸 OpenDarkSTS GUI v3")
    print("  Open: http://localhost:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)
