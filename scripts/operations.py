"""Application state and HTTP-facing helpers for the Detect workflow."""
from __future__ import annotations

import json
from pathlib import Path

from .workspace import WORKSPACE_DIR, atomic_json, current_model
from .acquisition import validate_bbox, date_range, cache_key
from . import jobs

PROJECT_PATH = WORKSPACE_DIR / "detection_project.json"
PRESETS = {
    "Galveston offshore": [-94.73, 29.07, -94.57, 29.205],
    "Los Angeles / Long Beach": [-118.4, 33.55, -118.05, 33.8],
    "New York approaches": [-74.15, 40.3, -73.7, 40.6],
    "San Francisco approaches": [-123.0, 37.5, -122.4, 37.9],
}


def project():
    if PROJECT_PATH.is_file():
        return json.loads(PROJECT_PATH.read_text())
    return {"bbox": PRESETS["Galveston offshore"], "start": "2020-01-04", "end": "2020-01-04",
            "images": str(WORKSPACE_DIR / "inputs" / "imagery"), "ais_dir": str(WORKSPACE_DIR / "inputs" / "ais" / "full"),
            "confidence": .25, "radius": 1000, "max_cloud": 30}


def save_project(payload):
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
    atomic_json(PROJECT_PATH, settings)
    return settings


def dashboard():
    model = None
    model_error = None
    try:
        path = current_model()
        if path.is_file():
            record = path.parent / "current.json"
            model = {"filename": path.name, "path": str(path), **(json.loads(record.read_text()) if record.is_file() else {})}
        else:
            model_error = "Initialize the workspace or promote a model in Improve."
    except (ValueError, OSError) as exc:
        model_error = str(exc)
    settings = project()
    inventory = {}
    for key in ["images", "ais_dir"]:
        root = Path(settings[key]).expanduser()
        suffixes = {".tif", ".tiff", ".png", ".jpg", ".jpeg"} if key == "images" else {".csv"}
        paths = [root] if root.is_file() else root.rglob("*") if root.is_dir() else []
        inventory[key] = sum(1 for item in paths if item.is_file() and item.suffix.lower() in suffixes)
    pending_reviews = 0
    for path in (WORKSPACE_DIR / "reviews").glob("*/*.json"):
        try:
            if json.loads(path.read_text()).get("review_status") != "completed":
                pending_reviews += 1
        except (OSError, ValueError):
            continue
    return {"pending_reviews": pending_reviews, "project": settings, "presets": PRESETS, "model": model, "model_error": model_error,
            "inventory": inventory, "jobs": jobs.list_jobs(include_training=True), "workspace": str(WORKSPACE_DIR)}


def submit_project_job(kind):
    settings = project()
    if kind == "run":
        if any(job["kind"] in {"ais", "imagery"} and job["status"] not in jobs.TERMINAL for job in jobs.list_jobs()):
            raise ValueError("Wait for the active downloads to finish before starting detection")
        # The whole saved date interval is enforced by selecting its dated images.
        from .sts_utils import resolve_image_inputs, parse_image_date
        days = set(date_range(settings["start"], settings["end"]))
        images = []
        for path in resolve_image_inputs([settings["images"]]):
            try:
                if parse_image_date(path) in days:
                    images.append(str(path))
            except ValueError:
                continue
        options = {"images": images, "ais_dir": settings["ais_dir"], "conf": settings["confidence"], "radius": settings["radius"]}
    elif kind in {"ais", "imagery"}:
        options = {"start": settings["start"], "end": settings["end"]}
        if kind == "imagery":
            options.update(bbox=settings["bbox"], max_cloud=settings["max_cloud"], output_dir=str(WORKSPACE_DIR / "inputs" / "imagery" / cache_key(settings["bbox"])))
        else:
            options["output_dir"] = str(WORKSPACE_DIR / "inputs" / "ais")
    else:
        raise ValueError("Unknown job type")
    job = jobs.submit_job(kind, options)
    if kind == "imagery":
        save_project({"images": options["output_dir"]})
    elif kind == "ais":
        save_project({"ais_dir": str(WORKSPACE_DIR / "inputs" / "ais" / "full")})
    return job
