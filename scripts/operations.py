"""Application state and HTTP-facing helpers for the Detect workflow."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .workspace import WORKSPACE_DIR, atomic_json, current_model, exclusive_lock, default_manifest
from .acquisition import validate_bbox, date_range, cache_key
from . import jobs

PROJECT_PATH = WORKSPACE_DIR / "detection_project.json"
PRESETS = {
    "Gulf of Mexico (US waters)": [-98.0, 24.0, -80.0, 31.0],
    "Bundled example (Gulf of Mexico)": [-94.36, 28.62, -94.29, 28.69],
    "Galveston offshore": [-94.73, 29.07, -94.57, 29.205],
    "Los Angeles / Long Beach": [-118.4, 33.55, -118.05, 33.8],
    "New York approaches": [-74.15, 40.3, -73.7, 40.6],
    "San Francisco approaches": [-123.0, 37.5, -122.4, 37.9],
}
SENSITIVITY_PRESETS = {"Low": .10, "Medium": .25, "High": .45}


def project():
    if PROJECT_PATH.is_file():
        return json.loads(PROJECT_PATH.read_text())
    return {"bbox": PRESETS["Bundled example (Gulf of Mexico)"], "start": "2020-01-24", "end": "2020-01-24",
            "images": str(WORKSPACE_DIR / "inputs" / "imagery"), "ais_dir": str(WORKSPACE_DIR / "inputs" / "ais" / "full"),
            "confidence": .25, "radius": 1000, "max_cloud": 35}


def save_project(payload):
    with exclusive_lock(PROJECT_PATH.with_suffix('.lock'), blocking=True):
        return _save_project(payload)


def apply_project_update(update):
    """Select acquired inputs only if the user has not edited the project meanwhile."""
    with exclusive_lock(PROJECT_PATH.with_suffix('.lock'), blocking=True):
        if project() != update['expected_project']:
            return False
        _save_project(update['updates'])
        return True


def _save_project(payload):
    allowed = set(project())
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("Unknown project settings")
    settings = {**project(), **payload}
    settings["bbox"] = validate_bbox(settings["bbox"])
    date_range(settings["start"], settings["end"])
    for key, lower, upper in [("confidence", 0, 1), ("radius", 1, 100000), ("max_cloud", 0, 100)]:
        settings[key] = float(settings[key])
        if not lower <= settings[key] <= upper:
            raise ValueError(f"Invalid {key}")
    for key in ["images", "ais_dir"]:
        value = str(settings[key]).strip()
        if value:
            path = Path(value).expanduser()
            settings[key] = str((path if path.is_absolute() else WORKSPACE_DIR / path).resolve())
    atomic_json(PROJECT_PATH, settings)
    return settings


def selected_detection_images(settings):
    """Use the same dated image selection for readiness and job submission."""
    from .sts_utils import resolve_image_inputs, parse_image_date
    days = set(date_range(settings["start"], settings["end"]))
    images = []
    for path in resolve_image_inputs([str(Path(settings["images"]).expanduser())]):
        try:
            if parse_image_date(path) in days:
                images.append(str(path))
        except ValueError:
            continue
    return images


def detection_readiness(settings=None):
    settings = settings or project()
    images = selected_detection_images(settings)
    blockers, warnings = [], []
    if not images:
        blockers.append("No image filenames match the selected dates. Choose an imagery folder and check Area & dates.")
    try:
        if not current_model().is_file():
            blockers.append("No detection model is available. Initialize the workspace or select a model in Improve.")
    except (ValueError, OSError) as exc:
        blockers.append(str(exc))
    if any(download_overlaps_inputs(job, settings) for job in jobs.list_jobs()):
        blockers.append("Wait for the download of the selected inputs to finish, or cancel it in Jobs before starting detection.")
    ais = Path(settings["ais_dir"]).expanduser()
    has_ais = bool(str(settings["ais_dir"]).strip()) and ((ais.is_file() and ais.suffix.lower() == ".csv") or (ais.is_dir() and any(
        path.is_file() and path.suffix.lower() == ".csv" for path in ais.rglob("*"))))
    if not has_ais:
        warnings.append("No AIS CSV files are available at the selected source. Detection can run, but vessel presence will be unknown. Select AIS on Data.")
    elif ais.is_file():
        from .sts_utils import ais_content_dates, parse_image_date
        try:
            dates = ais_content_dates(ais)
            missing = {parse_image_date(path) for path in images} - dates
            if missing:
                blockers.append("Selected AIS CSV has no records for image date(s): " + ", ".join(sorted(missing)) + ". Select a matching daily CSV before scanning.")
        except (ValueError, OSError) as exc:
            blockers.append(str(exc))
    return {"ready": not blockers, "image_count": len(images), "blockers": blockers, "warnings": warnings}


def download_overlaps_inputs(job, settings):
    """Only acquisition jobs writing the selected source and dates block a scan."""
    kind = job["kind"]
    if kind not in {"ais", "imagery"} or job["status"] in jobs.TERMINAL:
        return False
    options = job["options"]
    if not set(date_range(options["start"], options.get("end"))) & set(date_range(settings["start"], settings["end"])):
        return False
    default = WORKSPACE_DIR / "inputs" / ("ais" if kind == "ais" else "imagery")
    output = Path(options.get("output_dir") or default).expanduser().resolve()
    selected = Path(settings["ais_dir" if kind == "ais" else "images"]).expanduser().resolve()
    return selected.is_relative_to(output) or output.is_relative_to(selected)


def imagery_credentials_ready():
    return bool(os.environ.get("CDSE_CLIENT_ID") and os.environ.get("CDSE_CLIENT_SECRET"))


def use_bundled_example():
    root = (WORKSPACE_DIR / "inputs" / "example").resolve()
    if not (root / "2020-01-24_example.tiff").is_file() or not (root / "AIS_2020_01_24.csv").is_file():
        raise ValueError("The bundled example is not installed. Run opendarksts init for this workspace first.")
    return save_project({"bbox": PRESETS["Bundled example (Gulf of Mexico)"],
                         "start": "2020-01-24", "end": "2020-01-24",
                         "images": str(root), "ais_dir": str(root / "AIS_2020_01_24.csv"),
                         "confidence": .25, "radius": 1000})


def model_information():
    model = None
    model_error = None
    try:
        path = current_model()
        if path.is_file():
            record = path.parent / "current.json"
            model = {"filename": path.name, "path": str(path), **(json.loads(record.read_text()) if record.is_file() else {})}
            manifest = default_manifest()
            bundled = model.get("sha256") == manifest["sha256"]
            model["name"] = "YOLO11s" if bundled else model.get("architecture") or path.stem
            model["model_card_url"] = "/model-card" if bundled else None
        else:
            model_error = "Initialize the workspace or promote a model in Improve."
    except (ValueError, OSError) as exc:
        model_error = str(exc)
    return {"model": model, "model_error": model_error}


def dashboard():
    model_info = model_information()
    settings = project()
    inventory = {}
    for key in ["images", "ais_dir"]:
        root = Path(settings[key]).expanduser()
        suffixes = {".tif", ".tiff", ".png", ".jpg", ".jpeg"} if key == "images" else {".csv"}
        paths = [root] if root.is_file() else root.rglob("*") if root.is_dir() else []
        inventory[key] = sum(1 for item in paths if item.is_file() and item.suffix.lower() in suffixes)
    pending_reviews = 0
    training_images, completed_images = set(), set()
    manifest = WORKSPACE_DIR / "manifests" / "training_import_manifest.json"
    try:
        imported = json.loads(manifest.read_text())
        if isinstance(imported, dict) and isinstance(imported.get("images"), dict):
            training_images.update(imported["images"])
    except (OSError, ValueError):
        pass
    for path in (WORKSPACE_DIR / "reviews").glob("*/*.json"):
        try:
            review = json.loads(path.read_text())
            if not isinstance(review, dict):
                continue
            key = review.get("image_key") or path.stem
            training_images.add(key)
            if review.get("review_status") != "completed":
                pending_reviews += 1
            else:
                completed_images.add(key)
        except (OSError, ValueError):
            continue
    history = jobs.list_jobs(include_training=True)
    completed = [job for job in history if job["kind"] == "run" and job["status"] == "completed"]
    detection_summary = {"runs": len(completed), "detections": 0, "FULLY_DARK": 0, "PARTIALLY_DARK": 0,
                         "AIS_PRESENT": 0, "AIS_MISSING": 0, "AIS_INVALID": 0}
    for job in completed:
        result = job.get("result")
        summary = result.get("summary") if isinstance(result, dict) else None
        for key, value in (summary if isinstance(summary, dict) else {}).items():
            if key in detection_summary and key != "runs" and type(value) is int and value >= 0:
                detection_summary[key] += value
    return {"pending_reviews": pending_reviews, "project": settings, "presets": PRESETS,
            "sensitivity_presets": SENSITIVITY_PRESETS, "detection_summary": detection_summary,
            **model_info, "imagery_credentials_ready": imagery_credentials_ready(),
            "training_inventory": {"images": len(training_images), "completed": len(completed_images),
                                   "albums": len(list((WORKSPACE_DIR / "albums").glob("alb_*.json")))},
            "readiness": detection_readiness(settings),
            "inventory": inventory, "jobs": history, "workspace": str(WORKSPACE_DIR)}


def submit_project_job(kind):
    settings = project()
    if kind == "run":
        readiness = detection_readiness(settings)
        if not readiness["ready"]:
            raise ValueError(" ".join(readiness["blockers"]))
        images = selected_detection_images(settings)
        options = {"images": images, "ais_dir": settings["ais_dir"], "conf": settings["confidence"], "radius": settings["radius"]}
    elif kind in {"ais", "imagery"}:
        options = {"start": settings["start"], "end": settings["end"]}
        if kind == "imagery":
            if not imagery_credentials_ready():
                raise ValueError("Configure Copernicus credentials and restart the server before downloading imagery. See Data / AIS Manager.")
            options.update(bbox=settings["bbox"], max_cloud=settings["max_cloud"], output_dir=str(WORKSPACE_DIR / "inputs" / "imagery" / cache_key(settings["bbox"])))
        else:
            options["output_dir"] = str(WORKSPACE_DIR / "inputs" / "ais")
    else:
        raise ValueError("Unknown job type")
    updates = None
    if kind == "imagery":
        updates = {"images": options["output_dir"]}
    elif kind == "ais":
        updates = {"ais_dir": str(WORKSPACE_DIR / "inputs" / "ais" / "full")}
    return jobs.submit_job(kind, options, project_update={"expected_project": settings, "updates": updates} if updates else None)
