#!/usr/bin/python3
from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from .visual_review_lib import (
        DEFAULT_MANIFEST_PATH,
        DEFAULT_REVIEWER,
        DEFAULT_REVIEWS_DIR,
        WORKSPACE_DIR,
        build_review_bbox_overview,
        CLASS_ID_TO_NAME,
        REVIEW_LABEL_TO_CLASS,
        complete_image_review,
        ensure_cache_for_image_key,
        create_manual_detection,
        delete_detection,
        detection_payload,
        generate_review_crop,
        label_detection,
        list_review_documents,
        load_manifest,
        load_review_document,
        mark_all_active_detections_reviewed,
        mark_detection_neither,
        navigate_detail_cursor,
        queue_progress,
        repo_relative,
        repo_resolve,
        review_counts,
        review_file_path,
        save_review_document,
        set_review_status,
        start_detail_mode,
        sync_review_status,
        utcnow_iso,
        update_detection_geometry,
    )
except ImportError:
    from visual_review_lib import (
        DEFAULT_MANIFEST_PATH,
        DEFAULT_REVIEWER,
        DEFAULT_REVIEWS_DIR,
        WORKSPACE_DIR,
        build_review_bbox_overview,
        CLASS_ID_TO_NAME,
        REVIEW_LABEL_TO_CLASS,
        complete_image_review,
        ensure_cache_for_image_key,
        create_manual_detection,
        delete_detection,
        detection_payload,
        generate_review_crop,
        label_detection,
        list_review_documents,
        load_manifest,
        load_review_document,
        mark_all_active_detections_reviewed,
        mark_detection_neither,
        navigate_detail_cursor,
        queue_progress,
        repo_relative,
        repo_resolve,
        review_counts,
        review_file_path,
        save_review_document,
        set_review_status,
        start_detail_mode,
        sync_review_status,
        utcnow_iso,
        update_detection_geometry,
    )


MANIFEST_PATH = DEFAULT_MANIFEST_PATH
REVIEWS_DIR = DEFAULT_REVIEWS_DIR
REVIEWER = DEFAULT_REVIEWER
QUEUE_WINDOW_RADIUS = 80
QUEUE_CACHE_LOCK = threading.Lock()
QUEUE_CACHE: list[dict] | None = None

# DELETE BEFORE SHIP: temporary interaction logger for diagnosing missed UI saves.
DEBUG_INTERACTION_LOG_PATH = repo_resolve("workspace/reports/visual_review_interactions_DELETE_BEFORE_SHIP.jsonl")


def render_bbox_overview_html() -> str:
    try:
        from .training_pipeline_ui import render_app_html
    except ImportError:
        from training_pipeline_ui import render_app_html
    return render_app_html()


def json_body(handler: BaseHTTPRequestHandler) -> dict:
    content_length = int(handler.headers.get("Content-Length", 0))
    if content_length <= 0:
        return {}
    raw = handler.rfile.read(content_length)
    return json.loads(raw.decode("utf-8")) if raw else {}


def append_debug_interaction_event(payload: dict) -> None:
    # DELETE BEFORE SHIP: temporary click/key logger. Remove this helper and endpoint before release.
    DEBUG_INTERACTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "server_received_at": utcnow_iso(),
        "reviewer": REVIEWER,
        "event": payload,
    }
    with DEBUG_INTERACTION_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def invalidate_review_queue_cache() -> None:
    global QUEUE_CACHE
    with QUEUE_CACHE_LOCK:
        QUEUE_CACHE = None


def query_bool(parsed, name: str, default: bool) -> bool:
    values = parse_qs(parsed.query).get(name)
    if not values:
        return default
    return values[0].strip().lower() not in {"0", "false", "no", "off"}


def pick_image_id(queue_rows, requested_image_id: str | None) -> str | None:
    if not queue_rows:
        return None
    queue_ids = [str(row["image_key"]) for row in queue_rows]
    if requested_image_id in queue_ids:
        return requested_image_id
    for row in queue_rows:
        if row["review_status"] not in {"completed", "overview_2"}:
            return str(row["image_key"])
    for row in queue_rows:
        if row["review_status"] != "completed":
            return str(row["image_key"])
    return str(queue_rows[0]["image_key"])


def load_review_by_image_id(image_id: str) -> tuple[dict, Path]:
    path = review_file_path(image_id, REVIEWER, REVIEWS_DIR)
    if not path.exists():
        raise KeyError(f"Review file not found for image_id={image_id} reviewer={REVIEWER}")
    return load_review_document(path), path


def save_review(review: dict, review_path: Path) -> None:
    sync_review_status(review)
    save_review_document(review, review_path)
    invalidate_review_queue_cache()


def refresh_detection_crop(review: dict, review_path: Path, detection_id: str) -> None:
    try:
        generate_review_crop(review, detection_id, review_path=review_path)
    except Exception as exc:
        print(f"Warning: could not generate review crop for {review.get('image_key')} {detection_id}: {exc}")


def synced_review_queue() -> list[dict]:
    global QUEUE_CACHE
    with QUEUE_CACHE_LOCK:
        if QUEUE_CACHE is not None:
            return QUEUE_CACHE

    reviews = list_review_documents(REVIEWER, REVIEWS_DIR)
    for review in reviews:
        before = (review.get("review_status"), review.get("detail_cursor"))
        sync_review_status(review)
        after = (review.get("review_status"), review.get("detail_cursor"))
        if before != after and review.get("_path") is not None:
            save_review_document(review, review["_path"])
    with QUEUE_CACHE_LOCK:
        QUEUE_CACHE = reviews
    return reviews


def queue_payload_window(queue_rows: list[dict], manifest_images: dict, current_index: int) -> tuple[list[dict], int, int]:
    start = max(0, current_index - QUEUE_WINDOW_RADIUS)
    end = min(len(queue_rows), current_index + QUEUE_WINDOW_RADIUS + 1)
    payload = []
    for index, row in enumerate(queue_rows[start:end], start=start):
        payload.append(
            {
                "image_id": row["image_key"],
                "image_key": row["image_key"],
                "image_filename": manifest_display_filename(manifest_images, row),
                "date": row.get("date"),
                "tile_row": int(row.get("tile_row") or 0),
                "tile_col": int(row.get("tile_col") or 0),
                "review_status": row["review_status"],
                "queue_index": index,
            }
        )
    return payload, start, end


def manifest_display_filename(manifest_images: dict, review: dict) -> str | None:
    entry = manifest_images.get(review.get("image_key")) or {}
    return entry.get("original_filename") or entry.get("display_filename") or review.get("display_filename") or review.get("original_filename") or entry.get("image_filename") or review.get("image_filename")


def cache_path_for_review(review: dict) -> Path:
    cache_path = review.get("cache_path")
    if cache_path:
        return repo_resolve(cache_path)
    return ensure_cache_for_image_key(review["image_key"], MANIFEST_PATH)


def build_state(image_id: str | None) -> dict:
    queue_rows = synced_review_queue()
    manifest_images = load_manifest(MANIFEST_PATH).get("images", {})
    progress = queue_progress(queue_rows)
    selected_image_id = pick_image_id(queue_rows, image_id)

    if selected_image_id is None:
        return {
            "queue": {
                "images": [],
                "window_start": 0,
                "window_end": 0,
                "current_index": None,
                "prev_image_id": None,
                "next_image_id": None,
                **progress,
            },
            "image": None,
            "detections": [],
            "detail_queue": [],
            "current_detection_id": None,
            "counts": {
                "total_detections": 0,
                "active_detections": 0,
                "reviewed_active_detections": 0,
                "pending_active_detections": 0,
                "neither_detections": 0,
            },
        }

    review, review_path = load_review_by_image_id(selected_image_id)
    before = (review.get("review_status"), review.get("detail_cursor"))
    sync_review_status(review)
    after = (review.get("review_status"), review.get("detail_cursor"))
    if before != after:
        save_review(review, review_path)
        queue_rows = synced_review_queue()
    image = next((row for row in queue_rows if row["image_key"] == selected_image_id), None)
    if image is None:
        raise KeyError(f"Image {selected_image_id} not found")

    cache_path = cache_path_for_review(review)

    detections = [detection_payload(row, int(review["width"]), int(review["height"]), selected_image_id) for row in review["detections"]]
    detail_payload = [item for item in detections if item["is_active"]]

    current_index = next(index for index, row in enumerate(queue_rows) if row["image_key"] == selected_image_id)
    prev_image_id = queue_rows[current_index - 1]["image_key"] if current_index > 0 else None
    next_image_id = queue_rows[current_index + 1]["image_key"] if current_index < len(queue_rows) - 1 else None
    queue_payload, queue_window_start, queue_window_end = queue_payload_window(queue_rows, manifest_images, current_index)

    current_detection_id = None
    detail_cursor = review.get("detail_cursor")
    if review["review_status"] == "detail" and isinstance(detail_cursor, int) and 0 <= detail_cursor < len(review["detections"]):
        current_candidate = review["detections"][detail_cursor]
        if current_candidate.get("review_label") != "neither":
            current_detection_id = current_candidate["detection_id"]

    return {
        "queue": {
            "images": queue_payload,
            "window_start": queue_window_start,
            "window_end": queue_window_end,
            "current_index": current_index,
            "prev_image_id": prev_image_id,
            "next_image_id": next_image_id,
            **progress,
        },
        "image": {
            "image_id": review["image_key"],
            "image_key": review["image_key"],
            "image_filename": manifest_display_filename(manifest_images, review),
            "date": review.get("date"),
            "tile_row": int(review.get("tile_row") or 0),
            "tile_col": int(review.get("tile_col") or 0),
            "width": int(review["width"]),
            "height": int(review["height"]),
            "tile_count": 1,
            "review_status": review["review_status"],
            "detail_cursor": review.get("detail_cursor"),
            "cache_url": "/cache/" + repo_relative(cache_path),
        },
        "detections": detections,
        "detail_queue": detail_payload,
        "current_detection_id": current_detection_id,
        "counts": review_counts(review),
    }


def parse_image_id(payload: dict) -> str:
    image_id = payload.get("image_id")
    if image_id is None:
        raise ValueError("image_id is required")
    value = str(image_id).strip()
    if not value:
        raise ValueError("image_id must be non-empty")
    return value


def parse_detection_geometry(payload: dict) -> tuple[float, float, float, float]:
    return (
        float(payload["x_center"]),
        float(payload["y_center"]),
        float(payload["width"]),
        float(payload["height"]),
    )


def reopen_if_completed(review: dict) -> None:
    if review.get("review_status") == "completed":
        set_review_status(review, "overview_2", None)


def render_app_html(default_image_id: str | None, default_detection_id: str | None = None) -> str:
    initial_image = json.dumps(default_image_id)
    initial_detection = json.dumps(default_detection_id)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Visual Detection Reviewer</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js"></script>
  <style>
    :root {{
      --ink: #102330;
      --muted: #5e7380;
      --paper: #f4efe5;
      --paper-strong: #fffaf2;
      --panel: rgba(255, 250, 242, 0.9);
      --line: rgba(16, 35, 48, 0.14);
      --accent: #0f766e;
      --accent-strong: #115e59;
      --ship: #2563eb;
      --ships: #c81e1e;
      --pending: #b93824;
      --neither: #7c6f64;
      --shadow: 0 18px 48px rgba(16, 35, 48, 0.14);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 34%),
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.12), transparent 28%),
        linear-gradient(180deg, #d9e8eb 0%, #f1ebdd 52%, #efe6d5 100%);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      min-height: 100vh;
      overflow: hidden;
    }}
    .shell {{ display: grid; grid-template-rows: auto 1fr; height: 100vh; overflow: hidden; }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 16px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 250, 242, 0.82);
      backdrop-filter: blur(10px);
    }}
    .pipeline-rail {{ display: flex; gap: 6px; align-items: center; justify-content: center; flex-wrap: wrap; }}
    .pipeline-stage {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 10px;
      background: rgba(255, 250, 242, 0.62);
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.03em;
      text-decoration: none;
      text-transform: uppercase;
      white-space: nowrap;
    }}
    .pipeline-stage.active {{ background: var(--ink); border-color: var(--ink); color: white; }}
    .pipeline-stage.enabled:hover {{ border-color: rgba(15, 118, 110, 0.4); color: var(--accent-strong); transform: translateY(-1px); }}
    .brand h1 {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      font-size: 25px;
      letter-spacing: 0.01em;
    }}
    .brand p {{ margin: 4px 0 0; color: var(--muted); font-size: 13px; }}
    .queue-meta {{ display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }}
    .pill {{
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 12px;
      background: rgba(16, 35, 48, 0.07);
      color: var(--ink);
    }}
    .layout {{ display: grid; grid-template-columns: 390px 1fr; min-height: 0; overflow: hidden; }}
    .sidebar {{
      padding: 20px;
      border-right: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,250,242,0.88), rgba(250,244,232,0.78));
      overflow-y: auto;
      overflow-x: hidden;
    }}
    .viewer {{ position: relative; min-height: 520px; overflow: hidden; }}
    #map {{ width: 100%; height: calc(100vh - 88px); }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(16, 35, 48, 0.08);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 16px;
      margin-bottom: 16px;
    }}
    .card h2, .card h3 {{
      margin: 0 0 10px;
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      font-weight: 700;
    }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .meta-block {{ padding: 10px 12px; border-radius: 14px; background: rgba(16, 35, 48, 0.04); }}
    .meta-block label {{ display: block; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }}
    .meta-block strong {{ font-size: 15px; }}
    .controls {{ display: grid; gap: 10px; }}
    .button-row {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
    button, select {{
      border: 0;
      border-radius: 12px;
      padding: 11px 12px;
      font: inherit;
    }}
    .class-picker {{
      display: grid;
      gap: 6px;
    }}
    .class-picker select {{
      width: 100%;
      padding: 14px 16px;
      font-size: 16px;
      font-weight: 700;
      border-radius: 14px;
      color: white;
      cursor: pointer;
      appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' fill='white'%3E%3Cpath d='M1 1l5 5 5-5'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 14px center;
      transition: background-color 180ms ease;
    }}
    .class-picker select.cls-ship {{ background-color: var(--ship); }}
    .class-picker select.cls-ships {{ background-color: var(--ships); }}
    .class-picker .shortcut-hint {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }}
    .class-picker .shortcut-hint kbd {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 5px;
      border: 1px solid rgba(16, 35, 48, 0.18);
      background: rgba(16, 35, 48, 0.06);
      font-family: inherit;
      font-size: 11px;
      font-weight: 700;
      line-height: 1.4;
    }}
    button {{
      cursor: pointer;
      background: var(--ink);
      color: white;
      transition: transform 120ms ease, opacity 120ms ease, background 120ms ease;
    }}
    button:hover {{ transform: translateY(-1px); }}
    button.secondary {{ background: rgba(16, 35, 48, 0.09); color: var(--ink); }}
    button.accent {{ background: var(--accent); }}
    button.ship {{ background: var(--ship); }}
    button.ships {{ background: var(--ships); }}
    button.warn {{ background: var(--pending); }}
    button.flat {{ background: rgba(16, 35, 48, 0.08); color: var(--ink); }}
    button:disabled {{ opacity: 0.45; cursor: not-allowed; transform: none; }}
    [data-tooltip] {{ position: relative; }}
    [data-tooltip]:hover::after {{
      content: attr(data-tooltip);
      position: absolute;
      bottom: calc(100% + 8px);
      left: 50%;
      transform: translateX(-50%);
      padding: 7px 11px;
      border-radius: 9px;
      background: var(--ink);
      color: white;
      font-size: 12px;
      font-weight: 500;
      width: max-content;
      max-width: min(340px, calc(100vw - 32px));
      white-space: normal;
      line-height: 1.35;
      text-align: left;
      z-index: 9999;
      pointer-events: none;
      box-shadow: 0 4px 12px rgba(16, 35, 48, 0.25);
    }}
    [data-tooltip].tooltip-right:hover::after {{
      left: auto;
      right: 0;
      transform: none;
    }}
    [data-tooltip]:disabled:hover::after {{ display: none; }}
    .hotkey-overlay {{
      display: none;
      position: fixed;
      inset: 0;
      z-index: 10000;
      align-items: center;
      justify-content: center;
      background: rgba(16, 35, 48, 0.5);
      backdrop-filter: blur(4px);
    }}
    .hotkey-overlay.visible {{ display: flex; }}
    .hotkey-card {{
      background: var(--paper-strong);
      border-radius: 20px;
      padding: 28px 32px;
      box-shadow: 0 24px 64px rgba(16, 35, 48, 0.28);
      max-width: 520px;
      width: 90%;
    }}
    .hotkey-card h2 {{ margin: 0 0 16px; font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif; }}
    .hotkey-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 24px; font-size: 13px; }}
    .hotkey-row {{ display: flex; justify-content: space-between; align-items: center; padding: 4px 0; }}
    .hotkey-row kbd {{
      display: inline-block;
      min-width: 24px;
      padding: 2px 7px;
      border-radius: 5px;
      border: 1px solid rgba(16, 35, 48, 0.18);
      background: rgba(16, 35, 48, 0.06);
      font-family: inherit;
      font-size: 11px;
      font-weight: 700;
      line-height: 1.4;
      text-align: center;
    }}
    .status-line {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; color: var(--muted); font-size: 13px; }}
    .badge {{ border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 700; }}
    .badge.overview_1 {{ background: rgba(185, 56, 36, 0.12); color: var(--pending); }}
    .badge.detail {{ background: rgba(15, 118, 110, 0.12); color: var(--accent-strong); }}
    .badge.overview_2 {{ background: rgba(37, 99, 235, 0.12); color: var(--ship); }}
    .badge.completed {{ background: rgba(22, 128, 60, 0.12); color: #16803c; }}
    .legend {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; font-size: 13px; }}
    .legend div {{ display: flex; align-items: center; gap: 8px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 3px; }}
    .note {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
    .selection {{ font-size: 13px; color: var(--muted); line-height: 1.45; }}
    .selection strong {{ color: var(--ink); }}
    .empty {{ padding: 40px 20px; text-align: center; color: var(--muted); }}
    .footer-nav {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
    .queue-list {{ display: grid; gap: 6px; max-height: 220px; overflow: auto; }}
    .queue-item {{
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid transparent;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      cursor: pointer;
      background: rgba(16, 35, 48, 0.04);
    }}
    .queue-item.completed {{ border-color: rgba(22, 101, 52, 0.24); background: rgba(34, 197, 94, 0.16); }}
    .queue-item.active {{ border-color: rgba(15, 118, 110, 0.36); background: rgba(15, 118, 110, 0.1); }}
    .queue-item.completed.active {{ border-color: rgba(22, 101, 52, 0.4); background: rgba(34, 197, 94, 0.22); }}
    .queue-item small {{ color: var(--muted); }}
    .leaflet-container {{ background: #0f172a; }}
    .bbox-label-anchor {{ background: transparent; border: 0; width: auto !important; height: auto !important; }}
    .bbox-label {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 4px 8px;
      border-radius: 8px;
      border: 2px solid currentColor;
      background: rgba(255, 250, 242, 0.96);
      box-shadow: 0 8px 20px rgba(16, 35, 48, 0.18);
      font-size: 12px;
      font-weight: 700;
      line-height: 1;
      white-space: nowrap;
      transform: translate(0, -100%);
      transform-origin: left bottom;
    }}
    .bbox-label.ship {{ color: #2563eb; }}
    .bbox-label.ships {{ color: #c81e1e; }}
    .bbox-label.neither {{ color: #7c6f64; }}
    .bbox-label.selected {{
      background: #fff;
      box-shadow: 0 0 0 3px rgba(255, 250, 242, 0.85), 0 10px 24px rgba(16, 35, 48, 0.28);
    }}
    .viewer-hint {{
      position: absolute;
      right: 16px;
      bottom: 16px;
      z-index: 500;
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid rgba(16, 35, 48, 0.08);
      border-radius: 14px;
      padding: 10px 12px;
      box-shadow: var(--shadow);
      max-width: 320px;
      font-size: 13px;
      line-height: 1.45;
    }}
    .map-nav-buttons {{
      position: absolute;
      right: 16px;
      bottom: 80px;
      z-index: 500;
      display: flex;
      gap: 6px;
      align-items: center;
    }}
    .map-nav-buttons button {{
      min-width: 40px;
      height: 40px;
      border-radius: 999px;
      border: 1px solid rgba(16, 35, 48, 0.15);
      background: rgba(255, 250, 242, 0.92);
      padding: 0 13px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      box-shadow: var(--shadow);
      color: var(--fg);
    }}
    .map-nav-buttons button:hover {{
      background: var(--accent);
      color: #fff;
    }}
    .review-progress {{
      position: fixed;
      left: 50%;
      bottom: 14px;
      transform: translateX(-50%);
      z-index: 900;
      width: min(640px, calc(100vw - 32px));
      padding: 10px 14px;
      border: 1px solid rgba(16, 35, 48, 0.08);
      border-radius: 16px;
      background: rgba(255, 250, 242, 0.95);
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: center;
      gap: 12px;
      pointer-events: none;
    }}
    .review-progress-track {{
      display: grid;
      grid-auto-flow: column;
      grid-auto-columns: minmax(8px, 1fr);
      gap: 4px;
      align-items: center;
    }}
    .review-progress-segment {{
      height: 14px;
      border-radius: 999px;
      background: rgba(16, 35, 48, 0.11);
      border: 1px solid rgba(16, 35, 48, 0.08);
    }}
    .review-progress-segment.filled {{
      background: linear-gradient(180deg, #0f766e, #115e59);
      border-color: rgba(15, 118, 110, 0.3);
    }}
    .review-progress-segment.current {{
      box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.45), 0 0 14px rgba(245, 158, 11, 0.45);
      border-color: #f59e0b;
    }}
    .review-progress-count {{
      text-align: right;
      font-size: 15px;
      font-weight: 700;
      color: var(--ink);
      white-space: nowrap;
    }}
    .review-progress.empty-state {{
      width: 220px;
    }}
    @media (max-width: 980px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      #map {{ height: 60vh; }}
      .topbar {{ flex-direction: column; align-items: flex-start; }}
      .pipeline-rail {{ justify-content: flex-start; }}
      .queue-meta {{ justify-content: flex-start; }}
      .review-progress {{
        width: min(420px, calc(100vw - 24px));
        bottom: 10px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div class="brand">
        <h1>Visual Detection Reviewer</h1>
        <p>Review Sentinel tile imagery, correct detections, and export clean YOLO labels.</p>
      </div>
      <nav class="pipeline-rail" aria-label="Training pipeline stages">
        <a class="pipeline-stage enabled" href="/import">Import Data</a>
        <a class="pipeline-stage active enabled" href="/manual-review">Manual Review</a>
        <a class="pipeline-stage enabled" href="/bbox-overview">BBox Overview</a>
        <span class="pipeline-stage">Dataset Selection</span>
        <span class="pipeline-stage">Training Setup</span>
        <span class="pipeline-stage">Train YOLO</span>
        <span class="pipeline-stage">Evaluate</span>
      </nav>
      <div class="queue-meta" id="queueMeta"></div>
    </div>
    <div class="layout">
      <aside class="sidebar" id="sidebar"></aside>
      <main class="viewer">
        <div id="map"></div>
        <div class="viewer-hint" id="viewerHint">Load an image to review detections. Draw new boxes for misses, select a box to edit it, and use detail mode for labeling.</div>
        <div class="review-progress" id="reviewProgress"></div>
        <div class="map-nav-buttons">
          <button onclick="navigateImageButton(-1, event)" data-tooltip="Previous image. Hold Shift while clicking to mark this image fully reviewed first.">Previous Image</button>
          <button onclick="retreatReview()" data-tooltip="Previous bbox [Left arrow]">Previous</button>
          <button onclick="advanceReview()" data-tooltip="Accept current bbox class and advance [Right arrow / Space]">Next</button>
          <button class="tooltip-right" onclick="navigateImageButton(1, event)" data-tooltip="Next image. Hold Shift while clicking to mark this image fully reviewed first.">Next Image</button>
        </div>
      </main>
    </div>
  </div>

  <div class="hotkey-overlay" id="hotkeyOverlay" onclick="if(event.target===this)toggleHotkeyOverlay()">
    <div class="hotkey-card">
      <h2>Keyboard Shortcuts</h2>
      <div class="hotkey-grid">
        <div>
          <div class="hotkey-row"><span>Label Ships (STS)</span><kbd>S</kbd></div>
          <div class="hotkey-row"><span>Label Ship (non-STS)</span><kbd>D</kbd></div>
          <div class="hotkey-row"><span>Mark Neither (background)</span><kbd>N</kbd> / <kbd>Backspace</kbd></div>
          <div class="hotkey-row"><span>Delete box permanently</span><kbd>Del</kbd></div>
          <div class="hotkey-row"><span>Cycle selected class</span><kbd>Tab</kbd> ship / ships / neither</div>
          <div class="hotkey-row"><span>Start Detail Review</span><kbd>R</kbd></div>
          <div class="hotkey-row"><span>Show / hide this</span><kbd>?</kbd></div>
        </div>
        <div>
          <div class="hotkey-row"><span>Advance review</span><kbd>Space</kbd></div>
          <div class="hotkey-row"><span>Prev detection</span><kbd>&#9664;</kbd></div>
          <div class="hotkey-row"><span>Next detection</span><kbd>&#9654;</kbd></div>
          <div class="hotkey-row"><span>Previous image</span><kbd>[</kbd></div>
          <div class="hotkey-row"><span>Next image</span><kbd>]</kbd></div>
          <div class="hotkey-row"><span>Mark image reviewed + navigate</span><kbd>Shift</kbd> + image button</div>
          <div class="hotkey-row"><span>Draw Box</span><kbd>B</kbd></div>
          <div class="hotkey-row"><span>Edit selected</span><kbd>E</kbd></div>
          <div class="hotkey-row"><span>Save Geometry</span><kbd>G</kbd></div>
          <div class="hotkey-row"><span>Cancel / Deselect</span><kbd>Esc</kbd></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const initialImageId = {initial_image};
    const initialDetectionId = {initial_detection};
    const statusLabels = {{ pending: 'Pending', overview_1: 'Overview 1', detail: 'Detail', overview_2: 'Overview 2', completed: 'Completed' }};
    const defaultClassOptions = [{{value: 0, label: 'Ship'}}, {{value: 1, label: 'Ships'}}];

    let state = null;
    let map = null;
    let overlay = null;
    let detectionGroup = null;
    let selectedDetectionId = null;
    let renderedImageId = null;
    let lastFocusedDetectionId = null;
    let drawHandler = null;
    let isEditing = false;
    let layerIndex = new Map();
    let labelIndex = new Map();
    let requestedDetectionId = null;

    function colorForDetection(detection) {{
      if (detection.pending_review_label === 'neither' || detection.review_label === 'neither' || !detection.is_active) return '#7c6f64';
      if (detection.class_id === 1) return '#c81e1e';
      return '#2563eb';
    }}

    function detectionName(detection) {{
      if (detection.pending_review_label === 'neither' || detection.review_label === 'neither') return 'neither';
      return detection.class_id === 1 ? 'ships' : 'ship';
    }}

    function getImageIdFromUrl() {{
      const params = new URLSearchParams(window.location.search);
      const raw = params.get('image_id');
      if (!raw) return initialImageId;
      return raw;
    }}

    function getDetectionIdFromUrl() {{
      const params = new URLSearchParams(window.location.search);
      const raw = params.get('detection_id');
      if (!raw) return initialDetectionId;
      return raw;
    }}

    function setImageIdInUrl(imageId) {{
      const params = new URLSearchParams(window.location.search);
      if (imageId === null || imageId === undefined) params.delete('image_id');
      else params.set('image_id', imageId);
      params.delete('detection_id');
      const next = `${{window.location.pathname}}?${{params.toString()}}`;
      window.history.replaceState({{}}, '', next);
    }}

    async function fetchState(imageId) {{
      const query = imageId ? `?image_id=${{encodeURIComponent(imageId)}}` : '';
      const response = await fetch('/api/state' + query);
      state = await response.json();
      syncReviewCursorFromState();
      if (state.image) setImageIdInUrl(state.image.image_id);
      renderAll();
      if (requestedDetectionId) {{
        const target = requestedDetectionId;
        requestedDetectionId = null;
        await focusDetectionFromUrl(target);
      }}
    }}

    async function focusDetectionFromUrl(detectionId) {{
      const layer = findLayerByDetectionId(detectionId);
      if (!layer) return;
      await selectLayer(layer, true);
      focusLayer(layer);
      lastFocusedDetectionId = detectionId;
    }}

    function renderAll() {{
      renderTopbar();
      renderSidebar();
      renderMap();
      renderReviewProgress();
    }}

    function getPersistedTraversalCount() {{
      if (!state || !state.image) return 0;
      const total = state.counts?.active_detections || 0;
      const reviewed = state.counts?.reviewed_active_detections || 0;
      if (total <= 0) return 0;
      if (state.image.review_status === 'overview_2' || state.image.review_status === 'completed') return total;
      if (state.image.review_status === 'detail') {{
        const currentIndex = (state.detail_queue || []).findIndex(item => item.detection_id === state.current_detection_id);
        if (currentIndex >= 0) return Math.max(reviewed, Math.min(currentIndex + 1, total));
      }}
      return Math.min(reviewed, total);
    }}

    function syncReviewCursorFromState() {{
      if (!state || !state.image) {{
        reviewCursorIndex = -1;
        return;
      }}
      const total = state.counts?.active_detections || 0;
      const reviewed = state.counts?.reviewed_active_detections || 0;
      if (total <= 0) {{
        reviewCursorIndex = -1;
        return;
      }}
      if (state.image.review_status === 'overview_2' || state.image.review_status === 'completed') {{
        reviewCursorIndex = total;
        return;
      }}
      if (state.image.review_status === 'detail') {{
        const currentIndex = (state.detail_queue || []).findIndex(item => item.detection_id === state.current_detection_id);
        if (currentIndex >= 0) {{
          reviewCursorIndex = currentIndex;
          return;
        }}
      }}
      if (reviewed > 0) {{
        reviewCursorIndex = Math.min(reviewed - 1, total - 1);
        return;
      }}
      reviewCursorIndex = -1;
    }}

    function getCycleReviewedCount() {{
      if (!state || !state.counts) return 0;
      const total = state.counts.active_detections || 0;
      if (total <= 0) return 0;
      const persistedCount = getPersistedTraversalCount();
      const liveCount = reviewCursorIndex < 0 ? 0 : Math.min(reviewCursorIndex + 1, total);
      return Math.max(persistedCount, liveCount);
    }}

    function renderReviewProgress() {{
      const host = document.getElementById('reviewProgress');
      if (!host || !state || !state.image) return;
      const total = state.counts?.active_detections || 0;
      const reviewed = getCycleReviewedCount();
      const currentIndex = Math.max(-1, Math.min(reviewCursorIndex, total - 1));
      if (total <= 0) {{
        host.className = 'review-progress empty-state';
        host.innerHTML = `
          <div class="review-progress-count">0/0</div>
        `;
        return;
      }}

      host.className = 'review-progress';
      host.innerHTML = `
        <div class="review-progress-track">
          ${{Array.from({{ length: total }}, (_, index) => `<span class="review-progress-segment ${{index < reviewed ? 'filled' : ''}} ${{index === currentIndex ? 'current' : ''}}"></span>`).join('')}}
        </div>
        <div class="review-progress-count">${{reviewed}}/${{total}}</div>
      `;
    }}

    function renderTopbar() {{
      const host = document.getElementById('queueMeta');
      if (!state || !state.queue) {{
        host.innerHTML = '';
        return;
      }}
      const currentIndex = state.queue.current_index === null ? '-' : state.queue.current_index + 1;
      host.innerHTML = `
        <span class="pill">${{state.queue.completed_images}} / ${{state.queue.total_images}} completed</span>
        <span class="pill">Queue position ${{currentIndex}}</span>
        <span class="pill">Ship reviewed ${{state.queue.reviewed_ship_bboxes || 0}}</span>
        <span class="pill">Ships reviewed ${{state.queue.reviewed_ships_bboxes || 0}}</span>
      `;
    }}

    function imageDisplayName(image) {{
      if (!image) return 'Unknown image';
      return image.image_filename || image.date || image.image_key || image.image_id || 'Unknown image';
    }}

    function imageLocationText(image) {{
      if (!image) return 'Unknown image';
      if (image.date) return `${{image.date}} (${{image.tile_row}},${{image.tile_col}})`;
      return imageDisplayName(image);
    }}

    function renderSidebar() {{
      const host = document.getElementById('sidebar');
      if (!state.image) {{
        host.innerHTML = '<div class="empty">No preprocessed images found. Run <code>scripts/preprocess_sentinel_tiles.py</code> first.</div>';
        return;
      }}

      const current = state.image;
      const counts = state.counts;
      const currentDetection = state.detail_queue.find(item => item.detection_id === state.current_detection_id) || null;
      const currentDetectionIndex = currentDetection ? state.detail_queue.findIndex(item => item.detection_id === currentDetection.detection_id) + 1 : 0;
      const selectedDetection = state.detections.find(item => item.detection_id === selectedDetectionId) || null;
      const defaultClassValue = selectedDetection ? selectedDetection.class_id : (currentDetection ? currentDetection.class_id : 0);

      host.innerHTML = `
        <div class="card">
          <div class="status-line">
            <span class="badge ${{current.review_status}}">${{statusLabels[current.review_status] || current.review_status}}</span>
            <span>${{imageDisplayName(current)}}</span>
          </div>
          <h2 style="margin-top:12px">${{imageLocationText(current)}}</h2>
          <div class="meta-grid">
            <div class="meta-block"><label>${{current.date ? 'Grid' : 'Image Key'}}</label><strong>${{current.date ? `(${{current.tile_row}},${{current.tile_col}})` : current.image_key}}</strong></div>
            <div class="meta-block"><label>Size</label><strong>${{current.width}} x ${{current.height}}</strong></div>
            <div class="meta-block"><label>Active</label><strong>${{counts.active_detections}}</strong></div>
            <div class="meta-block"><label>Reviewed</label><strong>${{counts.reviewed_active_detections}}</strong></div>
          </div>
        </div>

        <div class="card">
          <h3>Legend</h3>
          <div class="legend">
            <div><span class="swatch" style="background:#2563eb"></span> ship</div>
            <div><span class="swatch" style="background:#c81e1e"></span> ships</div>
            <div><span class="swatch" style="background:#7c6f64"></span> neither</div>
            <div><span class="swatch" style="background:#f8fafc;border:1px solid rgba(16,35,48,0.18)"></span> Selected / editing</div>
          </div>
          <p class="note" style="margin-top:12px">Click any box to jump straight into edit mode. Use <kbd>N</kbd> or <kbd>Backspace</kbd> to mark it as neither, or <kbd>Del</kbd> to permanently delete it.</p>
        </div>

        <div class="card controls">
          <h3>Overview Controls</h3>
          <div class="class-picker">
            <select id="newClassSelect" class="cls-${{defaultClassValue === 1 ? 'ships' : 'ship'}}" onchange="onClassSelectChange(this)">${{defaultClassOptions.map(opt => `<option value="${{opt.value}}" ${{opt.value === defaultClassValue ? 'selected' : ''}}>${{opt.label}}</option>`).join('')}}</select>
            <div class="shortcut-hint"><kbd>Tab</kbd> cycle Ship / Ships / Neither locally</div>
          </div>
          <p class="selection">${{selectedDetection ? `Selected <strong>#${{selectedDetection.detection_id}}</strong> - ${{detectionName(selectedDetection)}}` : 'No detection selected. Click a box to enter edit mode.'}}</p>
          <div class="button-row">
            <button class="accent" onclick="startDraw()" data-tooltip="Draw a new bounding box [B]">Draw Box</button>
            <button class="secondary" onclick="toggleEditSelected()" data-tooltip="Toggle geometry editing [E]">${{isEditing ? 'Pause Edit' : 'Edit Selected'}}</button>
          </div>
          <div class="button-row">
            <button class="flat" onclick="saveSelectedGeometry()" data-tooltip="Save box position [G]">Save Geometry</button>
            <button class="secondary" onclick="clearSelection()" data-tooltip="Deselect current box [Esc]">Clear Selection</button>
          </div>
          <div class="button-row">
            <button class="warn" onclick="markSelectedNeither()" data-tooltip="Mark as background/false-positive for training [N]">Mark Neither</button>
            <button onclick="deleteSelectedDetection()" style="background:#7f1d1d;color:#fca5a5" data-tooltip="Permanently remove this box [Del]">Delete Box</button>
          </div>
          <p class="note">Image completion is automatic once every active bbox has been reviewed.</p>
          <hr style="border:none;border-top:1px solid rgba(16,35,48,0.08);margin:4px 0">
          <div class="button-row">
            <button class="accent" onclick="startDetail()" data-tooltip="Enter per-detection review [R]">Start Detail Review</button>
          </div>
        </div>

        <div class="card controls">
          <h3>Detail Controls</h3>
          <div class="status-line">
            <span>${{currentDetection ? `Detection ${{currentDetectionIndex}} / ${{state.detail_queue.length}}` : 'No active detail target'}}</span>
            <span>${{counts.pending_active_detections}} pending</span>
          </div>
          <div class="button-row">
            <button class="secondary" onclick="navigateDetail('prev')" ${{current.review_status === 'detail' ? '' : 'disabled'}} data-tooltip="Previous detection [Left arrow]">&#9664; Back</button>
            <button class="secondary" onclick="advanceReview()" ${{current.review_status === 'detail' ? '' : 'disabled'}} data-tooltip="Accept current class and advance [Right arrow / Space]">Accept / Next &#9654;</button>
          </div>
          <div class="button-row">
            <button class="ships" onclick="labelCurrent('ships_sts')" ${{currentDetection ? '' : 'disabled'}} data-tooltip="Label as STS pair [S]">Ships (STS)</button>
            <button class="ship" onclick="labelCurrent('ship_non_sts')" ${{currentDetection ? '' : 'disabled'}} data-tooltip="Label as single ship [D]">Ship (non-STS)</button>
          </div>
          <div class="button-row">
            <button class="warn" onclick="labelCurrent('neither')" ${{currentDetection ? '' : 'disabled'}} data-tooltip="Mark as false positive [N]">Neither</button>
            <button class="flat" onclick="saveSelectedGeometry()" data-tooltip="Save box position [G]">Save Box</button>
          </div>
          <p class="note">Need to split one detection into two? Draw replacement boxes, then mark the old one as neither.</p>
        </div>

        <div class="card">
          <h3>Queue</h3>
          <p class="note" style="margin-top:0;margin-bottom:8px"><kbd>[</kbd> / <kbd>]</kbd> navigate images &middot; showing #${{(state.queue.window_start || 0) + 1}}-${{state.queue.window_end || 0}} of ${{state.queue.total_images}} &middot; <kbd>?</kbd> all shortcuts</p>
          <div class="queue-list">
            ${{state.queue.images.map(item => `
              <div class="queue-item ${{item.review_status === 'completed' ? 'completed' : ''}} ${{item.image_id === current.image_id ? 'active' : ''}}" onclick='openImage(${{JSON.stringify(item.image_id)}})'>
                <div>
                  <strong>${{imageLocationText(item)}}</strong><br>
                  <small>${{statusLabels[item.review_status] || item.review_status}}</small>
                </div>
                <small>#${{item.queue_index + 1}}</small>
              </div>
            `).join('')}}
          </div>
        </div>

        <div class="footer-nav">
          <button class="secondary" onclick='navigateImageButton(-1, event)' ${{state.queue.prev_image_id ? '' : 'disabled'}} data-tooltip="Previous image. Hold Shift while clicking to mark this image fully reviewed first.">Previous Image</button>
          <button class="secondary tooltip-right" onclick='navigateImageButton(1, event)' ${{state.queue.next_image_id ? '' : 'disabled'}} data-tooltip="Next image. Hold Shift while clicking to mark this image fully reviewed first.">Next Image</button>
        </div>
      `;
    }}

    function renderMap() {{
      if (!map) {{
        map = L.map('map', {{ crs: L.CRS.Simple, minZoom: -4, zoomSnap: 0.25, attributionControl: false }});
        detectionGroup = L.featureGroup().addTo(map);
        map.on('draw:created', onDrawCreated);
        map.on('click', event => {{
          if (event.originalEvent && event.originalEvent.target && event.originalEvent.target.closest('.leaflet-interactive')) return;
          if (!drawHandler || !drawHandler._enabled) clearSelection();
        }});
        map.on('mousemove', updateSelectedLabelPosition);
        map.on('mouseup', () => clampSelectedLayerToImageBounds());
      }}
      if (!state.image) return;

      const imageChanged = renderedImageId !== state.image.image_id;
      const bounds = [[0, 0], [state.image.height, state.image.width]];
      if (imageChanged) {{
        if (overlay) overlay.remove();
        overlay = L.imageOverlay(state.image.cache_url, bounds).addTo(map);
        map.fitBounds(bounds, {{ padding: [24, 24] }});
        renderedImageId = state.image.image_id;
        lastFocusedDetectionId = null;
      }}

      detectionGroup.clearLayers();
      layerIndex = new Map();
      labelIndex = new Map();

      sortedDetectionsForDisplay(state.detections).forEach(detection => {{
        const layer = L.rectangle(
          [[detection.y1_px, detection.x1_px], [detection.y2_px, detection.x2_px]],
          {{ color: colorForDetection(detection), weight: detection.detection_id === selectedDetectionId ? 5 : 3, opacity: 1, fillOpacity: detection.detection_id === selectedDetectionId ? 0.08 : 0.03, dashArray: detection.detection_id === selectedDetectionId ? '10 6' : null }}
        );
        layer.detection = detection;
        layer.on('click', () => selectLayer(layer));
        layer.on('edit drag', () => updateLabelPosition(layer));
        layer.on('dragend moveend', () => clampLayerToImageBounds(layer));
        detectionGroup.addLayer(layer);
        layerIndex.set(detection.detection_id, layer);

        const labelClass = detection.pending_review_label === 'neither' || detection.review_label === 'neither' || !detection.is_active
          ? 'neither'
          : (detection.class_id === 1 ? 'ships' : 'ship');
        const labelMarker = L.marker(labelPositionForBounds(layer.getBounds()), {{
          interactive: false,
          keyboard: false,
            zIndexOffset: labelZIndexForDetection(detection),
          icon: L.divIcon({{
            className: 'bbox-label-anchor',
            html: `<div class="bbox-label ${{labelClass}} ${{detection.detection_id === selectedDetectionId ? 'selected' : ''}}">${{detectionName(detection)}}</div>`,
          }}),
        }});
        labelMarker.addTo(detectionGroup);
        labelIndex.set(detection.detection_id, labelMarker);
      }});

      if (selectedDetectionId) {{
        const selectedLayer = findLayerByDetectionId(selectedDetectionId);
        if (selectedLayer) selectLayer(selectedLayer, false);
      }}

      if (state.image.review_status === 'detail' && state.current_detection_id) {{
        const currentLayer = findLayerByDetectionId(state.current_detection_id);
        if (currentLayer) {{
          selectLayer(currentLayer, false);
          if (imageChanged || lastFocusedDetectionId !== state.current_detection_id) {{
            focusLayer(currentLayer);
            lastFocusedDetectionId = state.current_detection_id;
          }}
        }}
      }}
    }}

    async function selectLayer(layer, rerender=true) {{
      await autoSaveGeometryIfNeeded();
      disableAllEditing();
      selectedDetectionId = layer.detection.detection_id;
      layerIndex.forEach((item, detectionId) => {{
        const selected = item === layer;
        item.setStyle({{
          color: colorForDetection(item.detection),
          weight: selected ? 5 : 3,
          opacity: 1,
          fillOpacity: selected ? 0.08 : 0.03,
          dashArray: selected ? '10 6' : null,
        }});
        const labelMarker = labelIndex.get(detectionId);
        if (labelMarker) {{
          const labelClass = item.detection.pending_review_label === 'neither' || item.detection.review_label === 'neither' || !item.detection.is_active
            ? 'neither'
            : (item.detection.class_id === 1 ? 'ships' : 'ship');
          labelMarker.setIcon(L.divIcon({{
            className: 'bbox-label-anchor',
            html: `<div class="bbox-label ${{labelClass}} ${{selected ? 'selected' : ''}}">${{detectionName(item.detection)}}</div>`,
          }}));
          if (labelMarker.getElement()) labelMarker.getElement().style.zIndex = String(labelZIndexForDetection(item.detection));
        }}
      }});
      if (layer.editing) layer.editing.enable();
      isEditing = true;
      document.getElementById('viewerHint').textContent = `Selected detection #${{layer.detection.detection_id}}. Drag corners or the center handle to resize or move it. Press Backspace to mark it as neither, or Delete to remove it.`;
      if (rerender) renderSidebar();
    }}

    async function clearSelection() {{
      await autoSaveGeometryIfNeeded();
      disableAllEditing();
      selectedDetectionId = null;
      isEditing = false;
      layerIndex.forEach((item, detectionId) => {{
        item.setStyle({{ color: colorForDetection(item.detection), weight: 3, opacity: 1, fillOpacity: 0.03, dashArray: null }});
        const labelMarker = labelIndex.get(detectionId);
        if (labelMarker) {{
          const labelClass = item.detection.pending_review_label === 'neither' || item.detection.review_label === 'neither' || !item.detection.is_active
            ? 'neither'
            : (item.detection.class_id === 1 ? 'ships' : 'ship');
          labelMarker.setIcon(L.divIcon({{
            className: 'bbox-label-anchor',
            html: `<div class="bbox-label ${{labelClass}}">${{detectionName(item.detection)}}</div>`,
          }}));
        }}
      }});
      document.getElementById('viewerHint').textContent = 'Click a box to enter edit mode immediately, or draw a new box for a missed detection.';
      renderSidebar();
    }}

    function findLayerByDetectionId(detectionId) {{
      return layerIndex.get(detectionId) || null;
    }}

    function detectionArea(detection) {{
      return Math.max(0, Number(detection.width) || 0) * Math.max(0, Number(detection.height) || 0);
    }}

    function sortedDetectionsForDisplay(detections) {{
      return [...detections].sort((left, right) => {{
        const areaDelta = detectionArea(right) - detectionArea(left);
        if (Math.abs(areaDelta) > 1e-12) return areaDelta;
        return String(left.detection_id).localeCompare(String(right.detection_id));
      }});
    }}

    function labelZIndexForDetection(detection) {{
      return Math.max(1, Math.round(1000000 - detectionArea(detection) * 1000000));
    }}

    function labelPositionForBounds(bounds) {{
      const top = Math.min(bounds.getNorth(), bounds.getSouth());
      const right = Math.max(bounds.getEast(), bounds.getWest());
      return [top, right];
    }}

    function updateLabelPosition(layer) {{
      if (!layer || !layer.detection) return;
      const labelMarker = labelIndex.get(layer.detection.detection_id);
      if (labelMarker) labelMarker.setLatLng(labelPositionForBounds(layer.getBounds()));
    }}

    function updateSelectedLabelPosition() {{
      if (!selectedDetectionId || !isEditing) return;
      updateLabelPosition(findLayerByDetectionId(selectedDetectionId));
    }}

    function disableAllEditing() {{
      layerIndex.forEach(layer => {{
        if (layer.editing) layer.editing.disable();
      }});
      isEditing = false;
    }}

    function focusLayer(layer) {{
      map.fitBounds(layer.getBounds(), {{ padding: [80, 80], maxZoom: 2 }});
    }}

    async function zoomToOverview() {{
      if (!map || !state.image) return;
      await clearSelection();
      const bounds = [[0, 0], [state.image.height, state.image.width]];
      map.fitBounds(bounds, {{ padding: [24, 24] }});
      lastFocusedDetectionId = null;
    }}

    // Review cycle state: -1 = initial overview, 0..n-1 = detections traversed, n = final overview
    let reviewCursorIndex = -1;

    function getActiveDetections() {{
      return state.detections.filter(d => d.is_active && d.review_label !== 'neither');
    }}

    function reviewLabelForClassId(classId) {{
      return Number(classId) === 1 ? 'ships_sts' : 'ship_non_sts';
    }}

    function getCurrentReviewTarget() {{
      const targetId = selectedDetectionId || state.current_detection_id;
      if (!targetId) return null;
      return state.detections.find(item => item.detection_id === targetId) || null;
    }}

    async function acceptCurrentReviewChoice() {{
      // Space/Right persists a pending Tab choice even for already-reviewed bboxes.
      const target = getCurrentReviewTarget();
      if (!target || target.review_label === 'neither' || !target.is_active) return false;
      if (target.pending_review_label === 'neither') {{
        await labelCurrent('neither');
        return true;
      }}
      if (target.pending_review_label && target.pending_review_label !== target.review_label) {{
        await labelCurrent(target.pending_review_label);
        return true;
      }}
      if (target.reviewed) return false;
      await labelCurrent(reviewLabelForClassId(target.class_id));
      return true;
    }}

    async function advanceReview() {{
      if (!state || !state.image) return;
      if (await acceptCurrentReviewChoice()) return;
      const active = getActiveDetections();
      reviewCursorIndex++;

      if (active.length === 0) {{
        // No detections — skip to next image
        await navigateQueue(1);
        return;
      }}

      if (reviewCursorIndex >= 0 && reviewCursorIndex < active.length) {{
        const det = active[reviewCursorIndex];
        const layer = findLayerByDetectionId(det.detection_id);
        renderReviewProgress();
        if (layer) {{
          await selectLayer(layer, true);
          focusLayer(layer);
          lastFocusedDetectionId = det.detection_id;
        }}
        return;
      }}

      if (reviewCursorIndex === active.length) {{
        // Final overview pass
        renderReviewProgress();
        await zoomToOverview();
        return;
      }}

      // Past final overview — go to next image
      await navigateQueue(1);
    }}

    async function retreatReview() {{
      if (!state || !state.image) return;
      const active = getActiveDetections();

      if (reviewCursorIndex <= 0) {{
        // Already at or before overview — go to previous image
        await navigateQueue(-1);
        return;
      }}

      reviewCursorIndex--;

      if (reviewCursorIndex < 0) {{
        // Back to initial overview
        reviewCursorIndex = -1;
        renderReviewProgress();
        await zoomToOverview();
        return;
      }}

      const det = active[reviewCursorIndex];
      renderReviewProgress();
      if (det) {{
        const layer = findLayerByDetectionId(det.detection_id);
        if (layer) {{
          await selectLayer(layer, true);
          focusLayer(layer);
          lastFocusedDetectionId = det.detection_id;
        }}
      }}
    }}

    async function startDraw() {{
      if (!map || !state.image) return;
      await clearSelection();
      if (drawHandler) {{
        drawHandler.disable();
        drawHandler = null;
      }}
      drawHandler = new L.Draw.Rectangle(map, {{ shapeOptions: {{ color: '#0f766e', weight: 2 }} }});
      drawHandler.enable();
    }}

    function toggleEditSelected() {{
      const layer = selectedDetectionId ? findLayerByDetectionId(selectedDetectionId) : null;
      if (!layer || !layer.editing) return;
      if (isEditing) {{
        layer.editing.disable();
        isEditing = false;
      }} else {{
        layer.editing.enable();
        isEditing = true;
      }}
      renderSidebar();
    }}

    function boundsToGeometry(bounds) {{
      const pixelBounds = clampedPixelBounds(bounds);
      const west = pixelBounds.west;
      const east = pixelBounds.east;
      const north = pixelBounds.north;
      const south = pixelBounds.south;
      return {{
        x_center: ((west + east) / 2) / state.image.width,
        y_center: 1.0 - ((north + south) / 2) / state.image.height,
        width: (east - west) / state.image.width,
        height: (north - south) / state.image.height,
      }};
    }}

    function clampedPixelBounds(bounds) {{
      const minBoxSizePx = 1;
      let west = Math.min(bounds.getWest(), bounds.getEast());
      let east = Math.max(bounds.getWest(), bounds.getEast());
      let south = Math.min(bounds.getNorth(), bounds.getSouth());
      let north = Math.max(bounds.getNorth(), bounds.getSouth());
      west = Math.max(0, Math.min(state.image.width, west));
      east = Math.max(0, Math.min(state.image.width, east));
      south = Math.max(0, Math.min(state.image.height, south));
      north = Math.max(0, Math.min(state.image.height, north));
      if (east <= west) east = Math.min(state.image.width, west + minBoxSizePx);
      if (east <= west) west = Math.max(0, east - minBoxSizePx);
      if (north <= south) north = Math.min(state.image.height, south + minBoxSizePx);
      if (north <= south) south = Math.max(0, north - minBoxSizePx);
      return {{ west, east, south, north }};
    }}

    function boundsNeedClamping(bounds, pixelBounds) {{
      const tolerance = 0.001;
      return Math.abs(Math.min(bounds.getWest(), bounds.getEast()) - pixelBounds.west) > tolerance
        || Math.abs(Math.max(bounds.getWest(), bounds.getEast()) - pixelBounds.east) > tolerance
        || Math.abs(Math.min(bounds.getNorth(), bounds.getSouth()) - pixelBounds.south) > tolerance
        || Math.abs(Math.max(bounds.getNorth(), bounds.getSouth()) - pixelBounds.north) > tolerance;
    }}

    function clampLayerToImageBounds(layer) {{
      if (!layer || !state.image || typeof layer.getBounds !== 'function' || typeof layer.setBounds !== 'function') return;
      const bounds = layer.getBounds();
      const pixelBounds = clampedPixelBounds(bounds);
      if (boundsNeedClamping(bounds, pixelBounds)) {{
        layer.setBounds([[pixelBounds.south, pixelBounds.west], [pixelBounds.north, pixelBounds.east]]);
      }}
      updateLabelPosition(layer);
    }}

    function clampSelectedLayerToImageBounds() {{
      const targetId = selectedDetectionId || state.current_detection_id;
      if (!targetId) return;
      clampLayerToImageBounds(findLayerByDetectionId(targetId));
    }}

    function geometryMatchesDetection(geometry, detection) {{
      const tolerance = 0.0000005;
      return Math.abs(geometry.x_center - detection.x_center) <= tolerance
        && Math.abs(geometry.y_center - detection.y_center) <= tolerance
        && Math.abs(geometry.width - detection.width) <= tolerance
        && Math.abs(geometry.height - detection.height) <= tolerance;
    }}

    async function onDrawCreated(event) {{
      const geometry = boundsToGeometry(event.layer.getBounds());
      const classId = parseInt(document.getElementById('newClassSelect')?.value || '0', 10);
      await postJson('/api/save-detection', {{
        image_id: state.image.image_id,
        class_id: classId,
        x_center: geometry.x_center,
        y_center: geometry.y_center,
        width: geometry.width,
        height: geometry.height,
        parent_detection_id: state.current_detection_id,
      }});
      await fetchState(state.image.image_id);
    }}

    async function saveSelectedGeometry() {{
      if (!selectedDetectionId) return;
      const layer = findLayerByDetectionId(selectedDetectionId);
      if (!layer) return;
      const geometry = boundsToGeometry(layer.getBounds());
      const classId = parseInt(document.getElementById('newClassSelect')?.value || String(layer.detection.class_id), 10);
      await postJson('/api/save-detection', {{
        image_id: state.image.image_id,
        detection_id: selectedDetectionId,
        class_id: classId,
        x_center: geometry.x_center,
        y_center: geometry.y_center,
        width: geometry.width,
        height: geometry.height,
      }});
      isEditing = false;
      await fetchState(state.image.image_id);
    }}

    async function autoSaveGeometryIfNeeded() {{
      const targetId = selectedDetectionId || state.current_detection_id;
      if (!targetId) return;
      const layer = findLayerByDetectionId(targetId);
      if (!layer) return;
      const geometry = boundsToGeometry(layer.getBounds());
      const pendingLabel = layer.detection.pending_review_label || null;
      if (pendingLabel === 'neither') {{
        try {{
          await postJson('/api/mark-neither', {{ image_id: state.image.image_id, detection_id: targetId }});
          layer.detection.review_label = 'neither';
          layer.detection.reviewed = true;
          layer.detection.is_active = false;
          delete layer.detection.pending_review_label;
        }} catch (e) {{
          console.warn('Autosave pending neither failed:', e);
        }}
        return;
      }}
      const classId = pendingLabel
        ? (pendingLabel === 'ships_sts' ? 1 : 0)
        : parseInt(document.getElementById('newClassSelect')?.value || String(layer.detection.class_id), 10);
      const reviewLabel = reviewLabelForClassId(classId);
      const pendingClassChange = pendingLabel && pendingLabel !== layer.detection.review_label;
      if (!pendingClassChange && geometryMatchesDetection(geometry, layer.detection) && classId === layer.detection.class_id) return;
      try {{
        await postJson('/api/save-detection', {{
          image_id: state.image.image_id,
          detection_id: targetId,
          class_id: classId,
          x_center: geometry.x_center,
          y_center: geometry.y_center,
          width: geometry.width,
          height: geometry.height,
        }});
        delete layer.detection.pending_review_label;
      }} catch (e) {{
        console.warn('Autosave failed:', e);
      }}
    }}

    async function markSelectedNeither() {{
      const targetId = selectedDetectionId || state.current_detection_id;
      if (!targetId) return;
      await postJson('/api/mark-neither', {{ image_id: state.image.image_id, detection_id: targetId }});
      selectedDetectionId = null;
      isEditing = false;
      await fetchState(state.image.image_id);
    }}

    async function deleteSelectedDetection() {{
      const targetId = selectedDetectionId || state.current_detection_id;
      if (!targetId) return;
      await postJson('/api/delete-detection', {{ image_id: state.image.image_id, detection_id: targetId }});
      selectedDetectionId = null;
      isEditing = false;
      await fetchState(state.image.image_id);
    }}

    async function startDetail() {{
      await postJson('/api/start-detail', {{ image_id: state.image.image_id }});
      await fetchState(state.image.image_id);
    }}

    async function navigateDetail(direction) {{
      await autoSaveGeometryIfNeeded();
      await postJson('/api/navigate-detail', {{ image_id: state.image.image_id, direction }});
      await fetchState(state.image.image_id);
    }}

    function refreshLayerVisual(layer) {{
      if (!layer) return;
      const detection = layer.detection;
      const selected = detection.detection_id === selectedDetectionId;
      layer.setStyle({{
        color: colorForDetection(detection),
        weight: selected ? 5 : 3,
        opacity: 1,
        fillOpacity: selected ? 0.08 : 0.03,
        dashArray: selected ? '10 6' : null,
      }});
      const labelMarker = labelIndex.get(detection.detection_id);
      if (labelMarker) {{
        const labelClass = detection.pending_review_label === 'neither' || detection.review_label === 'neither' || !detection.is_active
          ? 'neither'
          : (detection.class_id === 1 ? 'ships' : 'ship');
        labelMarker.setIcon(L.divIcon({{
          className: 'bbox-label-anchor',
          html: `<div class="bbox-label ${{labelClass}} ${{selected ? 'selected' : ''}}">${{detectionName(detection)}}</div>`,
        }}));
      }}
    }}

    async function cycleSelectedClassOnly() {{
      const targetId = selectedDetectionId || state.current_detection_id;
      if (!targetId) return;
      const layer = findLayerByDetectionId(targetId);
      const det = layer ? layer.detection : (state.detail_queue || []).find(d => d.detection_id === targetId);
      if (!det) return;
      const current = det.pending_review_label || det.review_label || reviewLabelForClassId(det.class_id);
      const cycle = ['ship_non_sts', 'ships_sts', 'neither'];
      const nextLabel = cycle[(Math.max(0, cycle.indexOf(current)) + 1) % cycle.length];
      const geometry = layer ? boundsToGeometry(layer.getBounds()) : det;
      det.pending_review_label = nextLabel;
      if (nextLabel !== 'neither') {{
        const nextClassId = nextLabel === 'ships_sts' ? 1 : 0;
        det.class_id = nextClassId;
        det.class_name = nextClassId === 1 ? 'ships' : 'ship';
      }}
      const selectEl = document.getElementById('newClassSelect');
      if (selectEl) {{
        selectEl.value = String(det.class_id);
        onClassSelectChange(selectEl);
      }}
      if (layer) refreshLayerVisual(layer);
      renderSidebar();
      try {{
        if (nextLabel === 'neither') {{
          await postJson('/api/mark-neither', {{ image_id: state.image.image_id, detection_id: targetId }});
        }} else {{
          await postJson('/api/label-detection', {{
            image_id: state.image.image_id,
            detection_id: targetId,
            review_label: nextLabel,
            class_id: nextLabel === 'ships_sts' ? 1 : 0,
            x_center: geometry.x_center,
            y_center: geometry.y_center,
            width: geometry.width,
            height: geometry.height,
          }});
        }}
        selectedDetectionId = targetId;
        await fetchState(state.image.image_id);
      }} catch (error) {{
        alert(`Could not save Tab label change: ${{error.message}}`);
      }}
    }}

    async function labelCurrent(reviewLabel) {{
      const targetId = selectedDetectionId || state.current_detection_id;
      if (!targetId) return;
      if (reviewLabel === 'neither') {{
        await postJson('/api/mark-neither', {{ image_id: state.image.image_id, detection_id: targetId }});
        selectedDetectionId = null;
        await fetchState(state.image.image_id);
        return;
      }}

      const layer = findLayerByDetectionId(targetId);
      const geometry = layer ? boundsToGeometry(layer.getBounds()) : state.detail_queue.find(item => item.detection_id === targetId);
      const classId = reviewLabel === 'ships_sts' ? 1 : 0;
      await postJson('/api/label-detection', {{
        image_id: state.image.image_id,
        detection_id: targetId,
        review_label: reviewLabel,
        class_id: classId,
        x_center: geometry.x_center,
        y_center: geometry.y_center,
        width: geometry.width,
        height: geometry.height,
      }});
      selectedDetectionId = null;
      isEditing = false;
      await fetchState(state.image.image_id);
    }}

    async function completeImage() {{
      try {{
        await postJson('/api/complete-image', {{ image_id: state.image.image_id }});
        await fetchState(state.image.image_id);
      }} catch (error) {{
        alert(error.message);
      }}
    }}

    async function openImage(imageId) {{
      if (!imageId) return;
      await autoSaveGeometryIfNeeded();
      selectedDetectionId = null;
      isEditing = false;
      reviewCursorIndex = -1;
      await fetchState(imageId);
    }}

    async function markCurrentImageReviewed() {{
      if (!state || !state.image) return false;
      await postJson('/api/mark-image-reviewed', {{ image_id: state.image.image_id }});
      return true;
    }}

    async function navigateImageButton(direction, event) {{
      if (event && event.shiftKey) {{
        const marked = await markCurrentImageReviewed();
        if (!marked) return;
      }}
      await navigateQueue(direction);
    }}

    async function postJson(url, payload) {{
      const response = await fetch(url, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload),
      }});
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || 'Request failed');
      return data;
    }}

    // DELETE BEFORE SHIP: temporary full click/key logger for diagnosing missed review saves.
    // DELETE BEFORE SHIP: remove logDebugInteraction plus both capture listeners before release.
    function summarizeTarget(target) {{
      if (!target) return null;
      const text = (target.innerText || target.textContent || '').replace(/\\s+/g, ' ').trim();
      return {{
        tag: target.tagName || null,
        id: target.id || null,
        class_name: typeof target.className === 'string' ? target.className : null,
        name: target.getAttribute ? target.getAttribute('name') : null,
        type: target.getAttribute ? target.getAttribute('type') : null,
        aria_label: target.getAttribute ? target.getAttribute('aria-label') : null,
        title: target.getAttribute ? target.getAttribute('title') : null,
        text: text.slice(0, 120),
      }};
    }}

    function logDebugInteraction(kind, event) {{
      // DELETE BEFORE SHIP: this intentionally logs every click and keydown.
      const payload = {{
        kind,
        browser_time: new Date().toISOString(),
        url: window.location.href,
        image_id: state?.image?.image_id || null,
        review_status: state?.image?.review_status || null,
        selected_detection_id: selectedDetectionId || null,
        current_detection_id: state?.current_detection_id || null,
        target: summarizeTarget(event.target),
        event_phase: event.eventPhase,
        default_prevented_at_capture: event.defaultPrevented,
      }};
      if (kind === 'click') {{
        payload.button = event.button;
        payload.buttons = event.buttons;
        payload.client_x = event.clientX;
        payload.client_y = event.clientY;
        payload.ctrl_key = event.ctrlKey;
        payload.shift_key = event.shiftKey;
        payload.alt_key = event.altKey;
        payload.meta_key = event.metaKey;
      }} else if (kind === 'keydown') {{
        payload.key = event.key;
        payload.code = event.code;
        payload.repeat = event.repeat;
        payload.ctrl_key = event.ctrlKey;
        payload.shift_key = event.shiftKey;
        payload.alt_key = event.altKey;
        payload.meta_key = event.metaKey;
        payload.active_tag = document.activeElement ? document.activeElement.tagName : null;
      }}

      console.debug('[DELETE BEFORE SHIP visual-review-interaction]', payload);
      const body = JSON.stringify(payload);
      if (navigator.sendBeacon) {{
        navigator.sendBeacon('/api/debug-interaction', new Blob([body], {{ type: 'application/json' }}));
        return;
      }}
      fetch('/api/debug-interaction', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body,
        keepalive: true,
      }}).catch(() => {{}});
    }}

    function onClassSelectChange(selectEl) {{
      selectEl.className = selectEl.value === '1' ? 'cls-ships' : 'cls-ship';
    }}

    function toggleClassSelect() {{
      const selectEl = document.getElementById('newClassSelect');
      if (!selectEl) return;
      selectEl.value = selectEl.value === '1' ? '0' : '1';
      onClassSelectChange(selectEl);
    }}

    function toggleHotkeyOverlay() {{
      const overlay = document.getElementById('hotkeyOverlay');
      if (!overlay) return;
      overlay.classList.toggle('visible');
    }}

    async function navigateQueue(direction) {{
      if (!state.queue || !state.image) return;
      const nextImageId = direction < 0 ? state.queue.prev_image_id : state.queue.next_image_id;
      if (nextImageId) await openImage(nextImageId);
    }}

    // DELETE BEFORE SHIP: temporary capture-phase diagnostics for every click and keydown.
    document.addEventListener('click', event => logDebugInteraction('click', event), true);
    document.addEventListener('keydown', event => logDebugInteraction('keydown', event), true);

    document.addEventListener('keydown', async event => {{
      const activeTag = document.activeElement ? document.activeElement.tagName : '';
      if (activeTag === 'INPUT' || activeTag === 'TEXTAREA' || activeTag === 'SELECT') return;

      const key = event.key;

      if (key === '?') {{
        event.preventDefault();
        toggleHotkeyOverlay();
        return;
      }}
      if (key === 'Tab') {{
        event.preventDefault();
        if (selectedDetectionId || state.current_detection_id) {{
          await cycleSelectedClassOnly();
        }} else {{
          toggleClassSelect();
        }}
        return;
      }}
      if (key === 'Escape') {{
        event.preventDefault();
        if (drawHandler) {{
          drawHandler.disable();
          drawHandler = null;
        }} else {{
          await clearSelection();
          reviewCursorIndex = -1;
        }}
        return;
      }}

      // Space: advance through review cycle
      if (key === ' ') {{
        event.preventDefault();
        await advanceReview();
        return;
      }}

      // Navigation: [ previous image, ] next image
      if (key === '[') {{
        event.preventDefault();
        await navigateQueue(-1);
        return;
      }}
      if (key === ']') {{
        event.preventDefault();
        await navigateQueue(1);
        return;
      }}

      // Left/Right arrows: prev/next detection
      if (key === 'ArrowLeft') {{
        event.preventDefault();
        await retreatReview();
        return;
      }}
      if (key === 'ArrowRight') {{
        event.preventDefault();
        await advanceReview();
        return;
      }}

      // Labelling: S = Ships (STS), D = Ship (non-STS), N = Neither
      if (key === 's' || key === 'S') {{
        event.preventDefault();
        await labelCurrent('ships_sts');
        return;
      }}
      if (key === 'd' || key === 'D') {{
        event.preventDefault();
        await labelCurrent('ship_non_sts');
        return;
      }}
      if (key === 'n' || key === 'N') {{
        event.preventDefault();
        await labelCurrent('neither');
        return;
      }}

      // Actions
      if (key === 'r' || key === 'R') {{
        event.preventDefault();
        await startDetail();
        return;
      }}
      if (key === 'b' || key === 'B') {{
        event.preventDefault();
        await startDraw();
        return;
      }}
      if (key === 'e' || key === 'E') {{
        event.preventDefault();
        toggleEditSelected();
        return;
      }}
      if (key === 'g' || key === 'G') {{
        event.preventDefault();
        await saveSelectedGeometry();
        return;
      }}
      if (key === 'Backspace' && (selectedDetectionId || state.current_detection_id)) {{
        event.preventDefault();
        await markSelectedNeither();
        return;
      }}
      if (key === 'Delete' && (selectedDetectionId || state.current_detection_id)) {{
        event.preventDefault();
        await deleteSelectedDetection();
      }}
    }});

    requestedDetectionId = getDetectionIdFromUrl();
    fetchState(getImageIdFromUrl());
  </script>
</body>
</html>"""


class ReviewHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "", "/manual-review", "/manual-review/"}:
            queue_rows = synced_review_queue()
            default_image_id = pick_image_id(queue_rows, self._query_image_id(parsed))
            self._respond_html(render_app_html(default_image_id, self._query_detection_id(parsed)))
            return

        if path == "/bbox-overview":
            self._respond_html(render_bbox_overview_html())
            return

        if path == "/api/overview":
            payload = build_review_bbox_overview(
                REVIEWS_DIR,
                crop_root=WORKSPACE_DIR / "cache" / "review_crops",
                include_incomplete=False,
                generate_crops=query_bool(parsed, "generate", True),
            )
            self._respond_json(payload)
            return

        if path == "/api/state":
            payload = build_state(self._query_image_id(parsed))
            self._respond_json(payload)
            return

        if path.startswith("/cache/"):
            self._serve_cache_file(path[len("/cache/") :])
            return

        self._respond_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        payload = json_body(self)

        if parsed.path == "/api/debug-interaction":
            # DELETE BEFORE SHIP: temporary diagnostic endpoint for click/key logging.
            append_debug_interaction_event(payload)
            self._respond_json({"ok": True})
            return

        try:
            image_id = parse_image_id(payload)
            review, review_path = load_review_by_image_id(image_id)
            if parsed.path == "/api/start-detail":
                start_detail_mode(review)
                save_review(review, review_path)
                self._respond_json(build_state(image_id))
            elif parsed.path == "/api/navigate-detail":
                direction = payload.get("direction", "next")
                navigate_detail_cursor(review, direction)
                save_review(review, review_path)
                self._respond_json(build_state(image_id))
            elif parsed.path == "/api/save-detection":
                reopen_if_completed(review)
                x_center, y_center, width, height = parse_detection_geometry(payload)
                detection_id = payload.get("detection_id")
                class_id = int(payload.get("class_id", 0))
                if detection_id is None:
                    detection_id = create_manual_detection(
                        review,
                        class_id=class_id,
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height,
                        parent_detection_id=payload.get("parent_detection_id"),
                    )
                else:
                    detection_id = str(detection_id)
                    update_detection_geometry(review, str(detection_id), x_center, y_center, width, height, class_id=class_id)
                refresh_detection_crop(review, review_path, str(detection_id))
                save_review(review, review_path)
                self._respond_json(build_state(image_id))
            elif parsed.path == "/api/label-detection":
                reopen_if_completed(review)
                detection_id = str(payload["detection_id"])
                review_label = payload["review_label"]
                class_id = int(payload.get("class_id", REVIEW_LABEL_TO_CLASS[review_label]))
                x_center, y_center, width, height = parse_detection_geometry(payload)
                label_detection(review, detection_id, review_label, class_id, x_center, y_center, width, height)
                refresh_detection_crop(review, review_path, detection_id)
                save_review(review, review_path)
                self._respond_json(build_state(image_id))
            elif parsed.path == "/api/mark-neither":
                reopen_if_completed(review)
                detection_id = str(payload["detection_id"])
                mark_detection_neither(review, detection_id)
                refresh_detection_crop(review, review_path, detection_id)
                save_review(review, review_path)
                self._respond_json(build_state(image_id))
            elif parsed.path == "/api/delete-detection":
                reopen_if_completed(review)
                detection_id = str(payload["detection_id"])
                delete_detection(review, detection_id)
                save_review(review, review_path)
                self._respond_json(build_state(image_id))
            elif parsed.path == "/api/mark-image-reviewed":
                changed = mark_all_active_detections_reviewed(review)
                for detection in review.get("detections", []):
                    if detection.get("reviewed") and detection.get("review_label") != "neither":
                        refresh_detection_crop(review, review_path, str(detection.get("detection_id")))
                save_review(review, review_path)
                response = build_state(image_id)
                response["marked_reviewed_count"] = changed
                self._respond_json(response)
            elif parsed.path == "/api/complete-image":
                complete_image_review(review)
                save_review(review, review_path)
                self._respond_json(build_state(image_id))
            elif parsed.path == "/api/reopen-image":
                set_review_status(review, "overview_2", None)
                save_review(review, review_path)
                self._respond_json(build_state(image_id))
            else:
                self._respond_json({"error": "not found"}, status=404)
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=400)

    def _query_image_id(self, parsed) -> str | None:
        values = parse_qs(parsed.query).get("image_id")
        if not values:
            return None
        value = values[0].strip()
        return value or None

    def _query_detection_id(self, parsed) -> str | None:
        values = parse_qs(parsed.query).get("detection_id")
        if not values:
            return None
        value = values[0].strip()
        return value or None

    def _serve_cache_file(self, relative_path: str) -> None:
        target = repo_resolve(relative_path)
        cache_root = repo_resolve("workspace/cache")
        if not target.exists() or cache_root not in target.parents:
            self._respond_json({"error": "cache file not found"}, status=404)
            return

        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(target.read_bytes())

    def _respond_html(self, html: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _respond_json(self, payload: dict, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual detection reviewer web UI")
    parser.add_argument("--port", type=int, default=8766, help="Port to serve on")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Preprocess manifest path")
    parser.add_argument("--reviews-dir", default=str(DEFAULT_REVIEWS_DIR), help="Root directory containing reviewer subdirectories")
    parser.add_argument("--reviewer", default=DEFAULT_REVIEWER, help="Reviewer name whose review files should be loaded")
    args = parser.parse_args()

    global MANIFEST_PATH, REVIEWS_DIR, REVIEWER
    MANIFEST_PATH = args.manifest
    REVIEWS_DIR = Path(args.reviews_dir)
    REVIEWER = args.reviewer

    progress = queue_progress(synced_review_queue())

    print("Visual Detection Reviewer")
    print(f"  Manifest: {repo_resolve(MANIFEST_PATH)}")
    print(f"  Reviews: {repo_resolve(REVIEWS_DIR)}")
    print(f"  Reviewer: {REVIEWER}")
    print(f"  Queue: {progress['completed_images']} / {progress['total_images']} completed")
    print(f"  Open http://localhost:{args.port}")

    server = HTTPServer(("0.0.0.0", args.port), ReviewHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
