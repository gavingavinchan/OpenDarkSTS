#!/usr/bin/python3
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import secrets
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from .visual_review_lib import (
        DEFAULT_REVIEWER,
        DEFAULT_REVIEWS_DIR,
        WORKSPACE_DIR,
        build_review_bbox_overview,
        repo_resolve,
    )
    from . import training_pipeline_lib as pipeline
    from . import review_visual_detections_ui as review_ui
except ImportError:
    from visual_review_lib import (
        DEFAULT_REVIEWER,
        DEFAULT_REVIEWS_DIR,
        WORKSPACE_DIR,
        build_review_bbox_overview,
        repo_resolve,
    )
    import training_pipeline_lib as pipeline
    import review_visual_detections_ui as review_ui


REVIEWS_DIR = DEFAULT_REVIEWS_DIR
CROP_ROOT = WORKSPACE_DIR / "cache" / "review_crops"
PADDING_PX = 48
INCLUDE_INCOMPLETE = False
GENERATE_CROPS = True
MANUAL_REVIEW_URL = "/manual-review"

IMPORT_MANIFEST_PATH = pipeline.DEFAULT_IMPORT_MANIFEST_PATH
IMPORT_CACHE_DIR = pipeline.DEFAULT_IMPORT_CACHE_DIR
IMPORT_STAGING_ROOT = WORKSPACE_DIR / "cache" / "import_staging"
IMPORT_REVIEWER = DEFAULT_REVIEWER
TRAINING_UI_INTERACTION_LOG_PATH = WORKSPACE_DIR / "reports" / "training_pipeline_ui_interactions.jsonl"

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
JOB_RETENTION_SECONDS = 60 * 30
REVIEW_POST_PATHS = {
    "/api/debug-interaction",
    "/api/start-detail",
    "/api/navigate-detail",
    "/api/save-detection",
    "/api/label-detection",
    "/api/set-detection-class",
    "/api/mark-neither",
    "/api/delete-detection",
    "/api/mark-image-reviewed",
    "/api/complete-image",
    "/api/reopen-image",
}


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def query_bool(parsed, name: str, default: bool) -> bool:
    values = parse_qs(parsed.query).get(name)
    if not values:
        return default
    return values[0].strip().lower() not in {"0", "false", "no", "off"}


def parse_multipart_files(body: bytes, content_type: str) -> list[tuple[str, bytes]]:
    boundary_match = re.search(r"boundary=([^;]+)", content_type or "")
    if not boundary_match:
        raise ValueError("multipart upload missing boundary")
    boundary = b"--" + boundary_match.group(1).strip().strip('"').encode("latin-1")
    chunks = body.split(boundary)
    files: list[tuple[str, bytes]] = []
    for chunk in chunks[1:-1]:
        chunk = chunk.lstrip(b"\r\n")
        header_end = chunk.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        header_blob = chunk[:header_end].decode("latin-1", errors="replace")
        payload = chunk[header_end + 4:]
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        filename_match = re.search(r'filename="([^"]*)"', header_blob)
        if not filename_match:
            continue
        filename = filename_match.group(1)
        if not filename:
            continue
        files.append((filename, payload))
    return files


def safe_relative_path(raw: str) -> Path:
    cleaned = raw.replace("\\", "/").lstrip("/")
    parts: list[str] = []
    for segment in cleaned.split("/"):
        if not segment or segment in {".", ".."}:
            continue
        parts.append(segment)
    if not parts:
        raise ValueError(f"refusing unsafe upload path: {raw!r}")
    return Path(*parts)


def new_job_id() -> str:
    return "imp_" + secrets.token_hex(6)


def reap_old_jobs() -> None:
    cutoff = time.time() - JOB_RETENTION_SECONDS
    with JOBS_LOCK:
        stale = [
            job_id for job_id, job in JOBS.items()
            if job["state"] in {"done", "failed"} and job.get("finished_epoch", 0) < cutoff
        ]
        for job_id in stale:
            JOBS.pop(job_id, None)


def update_job(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(fields)


def append_training_ui_interaction(payload: dict) -> None:
    TRAINING_UI_INTERACTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "server_received_at": utcnow_iso(),
        "reviewer": IMPORT_REVIEWER,
        "event": payload,
    }
    with TRAINING_UI_INTERACTION_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def progress_caption(event: dict) -> str:
    phase = event.get("phase")
    if phase == "discovered":
        total = event.get("total", 0)
        skipped_unsupported = event.get("discovered_failed", 0)
        if total == 0:
            return "No supported images found in the upload."
        suffix = f" ({skipped_unsupported} unsupported skipped)" if skipped_unsupported else ""
        return f"Discovered {total} files{suffix}. Beginning import."
    if phase == "processing":
        index = event.get("index", 0) + 1
        total = event.get("total", 0)
        current = event.get("current", "")
        return f"Importing {index} of {total}: {current}"
    if phase == "saving":
        return "Writing manifest..."
    if phase == "done":
        return (
            f"Done. {event.get('imported_count', 0)} imported, "
            f"{event.get('skipped_count', 0)} already imported, "
            f"{event.get('failed_count', 0)} failed."
        )
    return ""


def query_text(parsed, name: str) -> str | None:
    values = parse_qs(parsed.query).get(name)
    if not values:
        return None
    value = values[0].strip()
    return value or None


def run_import_job(
    job_id: str,
    staging_dir: Path,
    *,
    force: bool,
    recursive: bool,
    album_name: str | None,
    album_description: str | None,
    source_display_name: str | None,
    append_to_album_id: str | None = None,
) -> None:
    def callback(event: dict) -> None:
        phase = event.get("phase")
        fields = {
            "phase": phase,
            "current_step": progress_caption(event),
            "total_count": event.get("total", 0),
        }
        if phase == "processing":
            fields["processed_count"] = event.get("index", 0)
            fields["current_filename"] = event.get("current", "")
            fields["imported_count"] = event.get("imported_count", 0)
            fields["skipped_count"] = event.get("skipped_count", 0)
            fields["failed_count"] = event.get("failed_count", 0)
        elif phase == "done":
            fields["processed_count"] = event.get("total", 0)
            fields["imported_count"] = event.get("imported_count", 0)
            fields["skipped_count"] = event.get("skipped_count", 0)
            fields["failed_count"] = event.get("failed_count", 0)
        update_job(job_id, **fields)

    update_job(job_id, state="importing", phase="starting", current_step="Scanning uploaded files...")
    try:
        result = pipeline.import_training_data(
            sources=[str(staging_dir)],
            manifest_path=IMPORT_MANIFEST_PATH,
            cache_dir=IMPORT_CACHE_DIR,
            album_name=album_name,
            album_description=album_description,
            source_display_name=source_display_name,
            source_kind="browser_upload",
            append_to_album_id=append_to_album_id,
            import_job_id=job_id,
            reviews_dir=REVIEWS_DIR,
            reviewer=IMPORT_REVIEWER,
            auto_detect=True,
            recursive=recursive,
            force=force,
            progress_callback=callback,
        )
        review_ui.invalidate_review_queue_cache()
        update_job(
            job_id,
            state="done",
            phase="done",
            current_step=progress_caption({
                "phase": "done",
                "imported_count": result["imported_count"],
                "skipped_count": result["skipped_count"],
                "failed_count": result["failed_count"],
            }),
            result=result,
            album=result.get("album"),
            finished_at=utcnow_iso(),
            finished_epoch=time.time(),
        )
    except Exception as exc:
        update_job(
            job_id,
            state="failed",
            phase="failed",
            current_step=f"Import failed: {exc}",
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            finished_at=utcnow_iso(),
            finished_epoch=time.time(),
        )


def start_import_job(
    uploads: list[tuple[str, bytes]],
    *,
    force: bool,
    album_name: str | None,
    album_description: str | None,
    source_display_name: str | None,
    append_to_album_id: str | None = None,
) -> str:
    job_id = new_job_id()
    staging_dir = IMPORT_STAGING_ROOT / job_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for raw_name, blob in uploads:
        try:
            relative = safe_relative_path(raw_name)
        except ValueError:
            continue
        target = staging_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        saved += 1
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "state": "queued",
            "phase": "queued",
            "current_step": f"Queued {saved} uploaded files for import.",
            "total_count": saved,
            "processed_count": 0,
            "imported_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "current_filename": "",
            "started_at": utcnow_iso(),
            "finished_at": None,
            "finished_epoch": 0,
            "error": None,
            "result": None,
            "album": None,
            "staging_dir": str(staging_dir),
            "force": bool(force),
            "album_name": album_name,
            "album_description": album_description,
            "source_display_name": source_display_name,
            "append_to_album_id": append_to_album_id,
        }
    thread = threading.Thread(
        target=run_import_job,
        args=(job_id, staging_dir),
        kwargs={
            "force": force,
            "recursive": True,
            "album_name": album_name,
            "album_description": album_description,
            "source_display_name": source_display_name,
            "append_to_album_id": append_to_album_id,
        },
        daemon=True,
        name=f"import-job-{job_id}",
    )
    thread.start()
    return job_id


def cache_url_for(cache_path: str | None) -> str | None:
    if not cache_path:
        return None
    try:
        resolved = repo_resolve(cache_path).resolve()
        cache_root = repo_resolve("workspace/cache").resolve()
    except OSError:
        return None
    if resolved == cache_root or cache_root in resolved.parents:
        return "/cache/" + pipeline.repo_relative(resolved)
    return None


def review_summary_from_document(review: dict, review_path: Path) -> dict:
    can_archive, reason = pipeline.review_archivable_status(review)
    summary = {
        "review_path": pipeline.repo_relative(review_path),
        "review_queue_filter": "unreviewed" if can_archive else "protected",
        "unreviewed_archivable": bool(can_archive),
        "review_protected_reason": reason,
        "review_status": review.get("review_status", "overview_1"),
        "review_counts": {},
    }
    try:
        counts = pipeline.review_lib.review_counts(review)
    except Exception:
        counts = {}
    summary["review_counts"] = counts
    if can_archive:
        return summary
    if review.get("review_status") == "completed":
        summary["review_queue_filter"] = "reviewed"
    elif int(counts.get("reviewed_active_detections") or 0) > 0 or any(
        isinstance(detection, dict) and detection.get("review_label") is not None
        for detection in review.get("detections", [])
    ):
        summary["review_queue_filter"] = "partially_reviewed"
    return summary


def manifest_summary() -> dict:
    manifest_file = repo_resolve(IMPORT_MANIFEST_PATH)
    album_index_path = pipeline.default_albums_dir_for_manifest(manifest_file) / "index.json"
    albums: list[dict] = []
    album_names: dict[str, str] = {}
    if album_index_path.is_file():
        try:
            album_index = pipeline.load_json(album_index_path)
            albums = [album for album in album_index.get("albums") or [] if isinstance(album, dict) and album.get("album_id")]
            album_names = {str(album["album_id"]): str(album.get("album_name") or album["album_id"]) for album in albums}
        except Exception:
            albums = []
            album_names = {}
    review_docs = pipeline.review_lib.list_review_documents(IMPORT_REVIEWER, REVIEWS_DIR)
    reviews_by_key = {str(review.get("image_key")): review for review in review_docs if review.get("image_key")}
    if not manifest_file.exists():
        images = []
        for review in review_docs:
            image_key = str(review.get("image_key") or "")
            review_path = Path(review.get("_path")) if review.get("_path") else pipeline.review_lib.review_file_path(image_key, IMPORT_REVIEWER, REVIEWS_DIR)
            images.append({
                "image_key": image_key,
                "original_filename": review.get("original_filename") or review.get("display_filename") or review.get("image_filename") or image_key,
                "browser_relative_path": None,
                "source_path": None,
                "source_collection": "review_only",
                "source_display_name": "Review queue",
                "album_ids": [],
                "album_names": [],
                "source_size_bytes": None,
                "width": review.get("width"),
                "height": review.get("height"),
                "import_status": "review_only",
                "imported_at": None,
                "updated_at": review.get("updated_at"),
                "render_profile_kind": None,
                "loader": None,
                "warnings": ["not_in_import_manifest"],
                "cache_url": cache_url_for(review.get("cache_path")),
                "cache_path": review.get("cache_path"),
                **review_summary_from_document(review, review_path),
            })
        return {
            "manifest_exists": False,
            "manifest_path": str(manifest_file),
            "image_count": len(images),
            "images": images,
            "albums": albums,
            "status_counts": {"review_only": len(images)} if images else {},
            "updated_at": None,
        }
    manifest = pipeline.load_import_manifest(manifest_file)
    images_map = manifest.get("images", {}) or {}
    images: list[dict] = []
    status_counts: dict[str, int] = {}
    for image_key, entry in images_map.items():
        status = entry.get("import_status", "imported")
        status_counts[status] = status_counts.get(status, 0) + 1
        cache_path = entry.get("cache_path")
        cache_url = cache_url_for(cache_path)
        album_ids = entry.get("album_ids") or []
        review = reviews_by_key.pop(str(image_key), None)
        if review is not None:
            review_path = Path(review.get("_path")) if review.get("_path") else pipeline.review_lib.review_file_path(image_key, IMPORT_REVIEWER, REVIEWS_DIR)
            review_summary = review_summary_from_document(review, review_path)
        else:
            review_summary = pipeline.review_queue_filter_summary(image_key, IMPORT_REVIEWER, REVIEWS_DIR)
        images.append({
            "image_key": image_key,
            "original_filename": entry.get("original_filename"),
            "browser_relative_path": entry.get("browser_relative_path"),
            "source_path": entry.get("source_path"),
            "source_collection": entry.get("source_collection"),
            "source_display_name": entry.get("source_display_name"),
            "album_ids": album_ids,
            "album_names": [album_names.get(str(album_id), str(album_id)) for album_id in album_ids],
            "source_size_bytes": entry.get("source_size_bytes"),
            "width": entry.get("width"),
            "height": entry.get("height"),
            "import_status": status,
            "imported_at": entry.get("imported_at"),
            "updated_at": entry.get("updated_at"),
            "render_profile_kind": (entry.get("render_profile") or {}).get("kind"),
            "loader": entry.get("loader"),
            "warnings": entry.get("warnings") or [],
            "cache_url": cache_url,
            "cache_path": cache_path,
            **review_summary,
        })
    for image_key, review in reviews_by_key.items():
        review_path = Path(review.get("_path")) if review.get("_path") else pipeline.review_lib.review_file_path(image_key, IMPORT_REVIEWER, REVIEWS_DIR)
        status_counts["review_only"] = status_counts.get("review_only", 0) + 1
        images.append({
            "image_key": image_key,
            "original_filename": review.get("original_filename") or review.get("display_filename") or review.get("image_filename") or image_key,
            "browser_relative_path": None,
            "source_path": None,
            "source_collection": "review_only",
            "source_display_name": "Review queue",
            "album_ids": [],
            "album_names": [],
            "source_size_bytes": None,
            "width": review.get("width"),
            "height": review.get("height"),
            "import_status": "review_only",
            "imported_at": None,
            "updated_at": review.get("updated_at"),
            "render_profile_kind": None,
            "loader": None,
            "warnings": ["not_in_import_manifest"],
            "cache_url": cache_url_for(review.get("cache_path")),
            "cache_path": review.get("cache_path"),
            **review_summary_from_document(review, review_path),
        })
    images.sort(key=lambda item: item.get("imported_at") or "", reverse=True)
    return {
        "manifest_exists": True,
        "manifest_path": str(manifest_file),
        "image_count": len(images),
        "images": images,
        "albums": albums,
        "status_counts": status_counts,
        "updated_at": manifest.get("updated_at"),
    }


def render_bbox_overview_html() -> str:
    return (
        BBOX_OVERVIEW_HTML
        .replace("__MANUAL_REVIEW_URL__", json.dumps(MANUAL_REVIEW_URL))
        .replace("__MANUAL_REVIEW_HREF__", MANUAL_REVIEW_URL.rstrip("/") + "/")
    )


def render_app_html() -> str:
    return render_bbox_overview_html()


def render_import_html() -> str:
    return IMPORT_HTML.replace("__MANUAL_REVIEW_HREF__", MANUAL_REVIEW_URL.rstrip("/") + "/")


def render_dataset_selection_html() -> str:
    return DATASET_SELECTION_HTML.replace("__MANUAL_REVIEW_HREF__", MANUAL_REVIEW_URL.rstrip("/") + "/")


DATASET_SELECTION_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenDarkSTS / Dataset Selection</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@9..144,300..900,0..100&family=IBM+Plex+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --ink: #111713;
      --ink-soft: #2a3530;
      --muted: #6f7a6e;
      --paper: #ece7da;
      --paper-warm: #e6dfca;
      --paper-cool: #d9e1d7;
      --card: #f8f4e8;
      --card-deep: #efe7d2;
      --line: rgba(17, 23, 19, 0.18);
      --line-strong: rgba(17, 23, 19, 0.42);
      --green: #0f6b4f;
      --red: #a83226;
      --blue: #2458a6;
      --amber: #b7791f;
      --stamp: #7a1f12;
      --stamp-deep: #4d1109;
      --rule: rgba(17, 23, 19, 0.12);
      --shadow: 0 18px 48px rgba(17, 23, 19, 0.16);
      --serif: "Fraunces", "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      --sans: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      --mono: "IBM Plex Mono", "Cascadia Mono", monospace;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(1200px 600px at 12% -10%, rgba(122, 31, 18, 0.06), transparent 60%),
        radial-gradient(900px 700px at 110% 110%, rgba(15, 107, 79, 0.07), transparent 60%),
        linear-gradient(90deg, rgba(17, 23, 19, 0.045) 1px, transparent 1px) 0 0 / 36px 36px,
        linear-gradient(180deg, #ded7c5 0%, #f2eddf 48%, #d9e1d7 100%);
      font-family: var(--sans);
      font-feature-settings: "ss01", "cv11";
    }
    body::before {
      content: "";
      position: fixed; inset: 0; pointer-events: none; z-index: 0;
      background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.06 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
      mix-blend-mode: multiply;
      opacity: 0.55;
    }
    .shell { position: relative; z-index: 1; min-height: 100vh; display: grid; grid-template-rows: auto auto 1fr; }
    header {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) auto;
      gap: 28px;
      align-items: end;
      padding: 22px 36px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(248, 244, 232, 0.72);
      backdrop-filter: blur(14px);
      position: sticky;
      top: 0;
      z-index: 6;
    }
    .titleblock { display: grid; grid-template-columns: auto 1fr; gap: 16px; align-items: center; }
    .filenum {
      font-family: var(--serif);
      font-weight: 300;
      font-style: italic;
      font-size: clamp(56px, 6vw, 96px);
      line-height: 0.85;
      color: var(--stamp);
      letter-spacing: -0.04em;
      transform: translateY(4px);
    }
    .filenum sup { font-size: 0.42em; vertical-align: super; color: var(--muted); font-style: normal; letter-spacing: 0.18em; font-family: var(--sans); font-weight: 700; padding-left: 4px; }
    h1 {
      margin: 0;
      font-family: var(--serif);
      font-weight: 500;
      font-size: clamp(30px, 3.4vw, 50px);
      letter-spacing: -0.035em;
      line-height: 1.0;
    }
    h1 em { font-style: italic; font-weight: 400; color: var(--ink-soft); }
    .subtitle { margin: 8px 0 0; color: var(--muted); font-size: 13px; max-width: 56ch; letter-spacing: 0.01em; }
    .rail { display: flex; gap: 7px; align-items: center; justify-content: flex-end; flex-wrap: wrap; }
    .stage {
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 11px;
      background: rgba(255, 255, 255, 0.36);
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      text-decoration: none;
    }
    .stage.active { background: var(--ink); color: var(--card); border-color: var(--ink); }
    .stage:hover:not(.active):not(span) { background: rgba(255, 255, 255, 0.8); color: var(--ink); }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto auto auto;
      gap: 10px;
      align-items: center;
      padding: 14px 36px;
      border-bottom: 1px solid var(--line);
      background: rgba(239, 231, 210, 0.72);
      backdrop-filter: blur(8px);
      position: sticky;
      top: 92px;
      z-index: 4;
    }
    input, select, textarea, button {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 12px;
      font: inherit;
      font-size: 13px;
      background: rgba(248, 244, 232, 0.92);
      color: var(--ink);
    }
    input:focus, select:focus, textarea:focus { outline: 2px solid var(--green); outline-offset: -1px; border-color: var(--green); }
    textarea { min-height: 72px; resize: vertical; }
    button { cursor: pointer; font-weight: 700; letter-spacing: 0.02em; border-radius: 6px; transition: background 120ms ease, border-color 120ms ease; }
    button:hover:not(:disabled) { background: rgba(248, 244, 232, 1); }
    button.primary { background: var(--ink); border-color: var(--ink); color: var(--card); }
    button.primary:hover:not(:disabled) { background: var(--ink-soft); }
    button.secondary { background: rgba(36, 88, 166, 0.10); border-color: rgba(36, 88, 166, 0.24); color: var(--blue); }
    button.secondary:hover:not(:disabled) { background: rgba(36, 88, 166, 0.18); }
    button:disabled { cursor: not-allowed; opacity: 0.38; }
    .layout {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr) 360px;
      min-height: 0;
    }
    .panel, aside {
      border-right: 1px solid var(--line);
      background: rgba(248, 244, 232, 0.60);
      padding: 22px;
    }
    aside { border-right: 0; border-left: 1px solid var(--line); }
    .sticky { position: sticky; top: 154px; max-height: calc(100vh - 154px); overflow: auto; }
    .panel h2, aside h2 {
      margin: 0 0 14px;
      font-family: var(--serif);
      font-weight: 500;
      font-size: 22px;
      letter-spacing: -0.03em;
    }
    .section { display: grid; gap: 12px; margin-bottom: 24px; }
    .checkrow { display: flex; gap: 9px; align-items: center; font-size: 13px; color: var(--muted); cursor: pointer; }
    .album-list { display: grid; gap: 6px; }
    .album-choice {
      display: flex; gap: 9px; align-items: flex-start;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.28);
      padding: 10px;
      cursor: pointer;
      font-size: 13px;
      color: var(--muted);
      transition: background 120ms ease, border-color 120ms ease;
    }
    .album-choice:hover { background: rgba(255, 255, 255, 0.56); border-color: var(--line-strong); }
    .album-choice:has(input:checked) { background: rgba(15, 107, 79, 0.08); border-color: rgba(15, 107, 79, 0.36); }
    .album-choice strong { display: block; color: var(--ink); }
    .album-choice small { display: block; margin-top: 2px; }
    .contact-sheet { padding: 22px 36px 40px; min-width: 0; }
    .stats { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 16px; }
    .stat {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: rgba(248, 244, 232, 0.82);
      color: var(--muted);
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.02em;
    }
    .stat strong { color: var(--ink); font-weight: 700; font-family: var(--mono); font-size: 12px; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px;
      align-items: start;
    }
    .tile {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--card);
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(17, 23, 19, 0.06);
      cursor: pointer;
      transition: transform 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
    }
    .tile:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(17, 23, 19, 0.12); }
    .tile.selected { border-color: var(--green); box-shadow: 0 0 0 2px rgba(15, 107, 79, 0.22), 0 6px 20px rgba(17, 23, 19, 0.12); }
    .thumb { aspect-ratio: 1.35 / 1; display: grid; place-items: center; background: #14201a; color: rgba(255, 250, 240, 0.5); font-size: 12px; overflow: hidden; }
    .thumb img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .tile-meta { padding: 10px 12px; display: grid; gap: 6px; }
    .tile-head { display: flex; justify-content: space-between; gap: 8px; align-items: flex-start; }
    .tile-title { font-weight: 700; font-size: 12px; overflow-wrap: anywhere; line-height: 1.3; }
    .tile-sub { color: var(--muted); font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .badge-row { display: flex; gap: 4px; flex-wrap: wrap; }
    .badge { border-radius: 4px; padding: 2px 6px; font-size: 10px; font-weight: 700; letter-spacing: 0.03em; }
    .badge.ship { background: rgba(36, 88, 166, 0.12); color: var(--blue); }
    .badge.ships { background: rgba(168, 50, 38, 0.12); color: var(--red); }
    .badge.neither { background: rgba(17, 23, 19, 0.07); color: var(--muted); }
    .badge.album { background: rgba(15, 107, 79, 0.10); color: var(--green); }
    .composer { display: grid; gap: 16px; }
    .field { display: grid; gap: 5px; }
    .field label { color: var(--muted); font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
    .split-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .saved-list { display: grid; gap: 6px; }
    .saved-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.28);
      padding: 10px 12px;
      display: grid;
      gap: 6px;
      transition: background 120ms ease, border-color 120ms ease;
    }
    .saved-item:hover { background: rgba(255, 255, 255, 0.48); }
    .saved-item.active { border-color: var(--green); background: rgba(15, 107, 79, 0.06); }
    .saved-title { font-weight: 700; font-size: 13px; }
    .saved-meta { color: var(--muted); font-size: 12px; }
    .hint { color: var(--muted); font-size: 12px; font-style: italic; }
    .empty { color: var(--muted); font-size: 13px; }
    .mono { font-family: var(--mono); font-size: 11px; overflow-wrap: anywhere; color: var(--muted); }
    .toast {
      position: fixed;
      left: 50%;
      bottom: 22px;
      transform: translateX(-50%) translateY(18px);
      opacity: 0;
      pointer-events: none;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--ink);
      color: var(--card);
      padding: 10px 16px;
      font-size: 13px;
      font-weight: 600;
      box-shadow: var(--shadow);
      transition: opacity 160ms ease, transform 160ms ease;
      z-index: 20;
      max-width: 420px;
      text-align: center;
    }
    .toast.show { opacity: 1; transform: translateX(-50%) translateY(0); pointer-events: auto; }
    .toast.error { background: var(--red); }
    @media (max-width: 1120px) {
      header, .toolbar { position: static; grid-template-columns: 1fr; }
      .rail { justify-content: flex-start; }
      .layout { grid-template-columns: 1fr; }
      .panel, aside { border: 0; border-bottom: 1px solid var(--line); }
      .sticky { position: static; max-height: none; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="titleblock">
        <div class="filenum">04<sup>/07</sup></div>
        <div>
          <h1>Dataset <em>Selection</em></h1>
          <p class="subtitle">Assemble a reproducible image-level YOLO dataset from completed reviews. Filter by album, inspect class balance, save a manifest, then export.</p>
        </div>
      </div>
      <nav class="rail" aria-label="Training pipeline stages">
        <a class="stage" href="/import">Import Data</a>
        <a class="stage" href="__MANUAL_REVIEW_HREF__">Manual Review</a>
        <a class="stage" href="/bbox-overview">BBox Overview</a>
        <a class="stage active" href="/dataset-selection">Dataset Selection</a>
        <a class="stage" href="/training-setup">Training Setup</a>
        <a class="stage" href="/train-yolo">Train YOLO</a>
        <span class="stage">Evaluate</span>
      </nav>
    </header>

    <section class="toolbar">
      <input id="search" placeholder="Search image, album, source" autocomplete="off">
      <select id="labelFilter">
        <option value="all">All labels</option>
        <option value="exportable">Exportable only</option>
        <option value="ship">Ship only</option>
        <option value="ships">Ships only</option>
        <option value="neither">Neither only</option>
      </select>
      <label class="checkrow"><input id="includeIncomplete" type="checkbox"> Include incomplete</label>
      <button class="secondary" id="reloadBtn">Refresh</button>
    </section>

    <main class="layout">
      <section class="panel"><div class="sticky">
        <div class="section">
          <h2>Albums</h2>
          <p class="hint">Select one or more albums. Leave blank to use every completed review.</p>
          <div class="album-list" id="albumList"></div>
        </div>
      </div></section>

      <section class="contact-sheet">
        <div class="stats" id="stats"></div>
        <div class="actions">
          <button id="selectVisibleBtn">Select Visible</button>
          <button id="clearBtn">Clear</button>
        </div>
        <div class="grid" id="grid"></div>
      </section>

      <aside><div class="sticky composer">
        <section class="section">
          <h2>Selection Manifest</h2>
          <div class="stats" id="selectionStats"></div>
          <div class="field"><label for="datasetName">Dataset name</label><input id="datasetName" value="manual_reviewed"></div>
          <div class="field"><label for="datasetVersion">Version</label><input id="datasetVersion" value="v1"></div>
          <div class="field"><label for="description">Description</label><textarea id="description" placeholder="Optional notes for future retraining"></textarea></div>
          <div class="split-row">
            <div class="field"><label for="trainRatio">Train</label><input id="trainRatio" type="number" min="0" max="1" step="0.05" value="0.8"></div>
            <div class="field"><label for="valRatio">Val</label><input id="valRatio" type="number" min="0" max="1" step="0.05" value="0.1"></div>
            <div class="field"><label for="testRatio">Test</label><input id="testRatio" type="number" min="0" max="1" step="0.05" value="0.1"></div>
          </div>
          <button class="primary" id="saveBtn">Save Selection</button>
          <button id="exportBtn" disabled>Export YOLO Dataset</button>
          <button id="continueBtn" disabled>Continue To Training Setup</button>
          <p class="hint" id="selectionPathHint">Save a selection to enable export.</p>
        </section>
        <section class="section">
          <h2>Saved</h2>
          <div class="saved-list" id="savedList"></div>
        </section>
      </div></aside>
    </main>
  </div>
  <div class="toast" id="toast"></div>

  <script>
    const stateKey = 'opendarksts.datasetSelection.state.v1';
    const state = {
      albums: [],
      candidates: [],
      saved: [],
      selectedKeys: [],
      selectedAlbumIds: [],
      search: '',
      label: 'all',
      includeIncomplete: false,
      currentSelectionPath: null,
    };

    function escape(value) {
      return String(value == null ? '' : value).replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      })[ch]);
    }
    function showToast(message, kind) {
      const toast = document.getElementById('toast');
      toast.textContent = message;
      toast.classList.toggle('error', kind === 'error');
      toast.classList.add('show');
      clearTimeout(showToast._t);
      showToast._t = setTimeout(() => toast.classList.remove('show'), 2600);
    }
    function logAction(action, details) {
      const payload = {
        action,
        timestamp: new Date().toISOString(),
        page: 'dataset-selection',
        filter: { album_ids: state.selectedAlbumIds, label: state.label, include_incomplete: state.includeIncomplete, search: state.search },
        selected_count: state.selectedKeys.length,
        details: details || {},
      };
      const body = JSON.stringify(payload);
      try {
        if (navigator.sendBeacon) {
          navigator.sendBeacon('/api/training-ui-interaction', new Blob([body], { type: 'application/json' }));
          return;
        }
      } catch (e) {}
      fetch('/api/training-ui-interaction', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }).catch(() => {});
    }
    function persistState() {
      localStorage.setItem(stateKey, JSON.stringify({
        selectedKeys: state.selectedKeys,
        selectedAlbumIds: state.selectedAlbumIds,
        search: state.search,
        label: state.label,
        includeIncomplete: state.includeIncomplete,
        currentSelectionPath: state.currentSelectionPath,
      }));
    }
    function restoreState() {
      try {
        const raw = JSON.parse(localStorage.getItem(stateKey) || '{}');
        if (Array.isArray(raw.selectedKeys)) state.selectedKeys = raw.selectedKeys.filter(x => typeof x === 'string');
        if (Array.isArray(raw.selectedAlbumIds)) state.selectedAlbumIds = raw.selectedAlbumIds.filter(x => typeof x === 'string');
        if (typeof raw.search === 'string') state.search = raw.search;
        if (typeof raw.label === 'string') state.label = raw.label;
        if (typeof raw.includeIncomplete === 'boolean') state.includeIncomplete = raw.includeIncomplete;
        if (typeof raw.currentSelectionPath === 'string') state.currentSelectionPath = raw.currentSelectionPath;
      } catch (e) {}
    }
    function candidateCacheUrl(candidate) {
      if (candidate.cache_url) return candidate.cache_url;
      const raw = candidate.cache_path || '';
      if (!raw || raw.startsWith('/')) return '';
      return '/cache/' + raw.replace(/^[/]+/, '');
    }
    function filteredCandidates() {
      const q = state.search.trim().toLowerCase();
      if (!q) return state.candidates;
      return state.candidates.filter(candidate => [
        candidate.image_key,
        candidate.image_filename,
        candidate.source_display_name,
        ...(candidate.album_names || []),
      ].join(' ').toLowerCase().includes(q));
    }
    function selectedCandidates() {
      const selected = new Set(state.selectedKeys);
      return state.candidates.filter(candidate => selected.has(candidate.image_key));
    }
    function countBboxes(candidates) {
      return candidates.reduce((acc, candidate) => {
        const counts = candidate.counts || {};
        acc.ship += Number(counts.ship || 0);
        acc.ships += Number(counts.ships || 0);
        acc.neither += Number(counts.neither || 0);
        acc.total += Number(counts.total || 0);
        return acc;
      }, { ship: 0, ships: 0, neither: 0, total: 0 });
    }
    function stat(label, value) {
      return `<span class="stat"><strong>${escape(value)}</strong> ${escape(label)}</span>`;
    }
    function renderAlbums() {
      const host = document.getElementById('albumList');
      if (!state.albums.length) {
        host.innerHTML = '<p class="empty">No albums yet.</p>';
        return;
      }
      host.innerHTML = state.albums.map(album => {
        const checked = state.selectedAlbumIds.includes(album.album_id) ? 'checked' : '';
        const reviewed = album.reviewed_count == null ? '-' : album.reviewed_count;
        return `<label class="album-choice">
          <input type="checkbox" data-album-id="${escape(album.album_id)}" ${checked}>
          <span><strong>${escape(album.album_name || album.album_id)}</strong><small>${escape(album.image_count || 0)} images, ${escape(reviewed)} reviewed</small></span>
        </label>`;
      }).join('');
      host.querySelectorAll('input[data-album-id]').forEach(input => {
        input.onchange = () => {
          const albumId = input.getAttribute('data-album-id');
          const set = new Set(state.selectedAlbumIds);
          if (input.checked) set.add(albumId); else set.delete(albumId);
          state.selectedAlbumIds = Array.from(set);
          logAction('album_filter_changed', { album_ids: state.selectedAlbumIds });
          persistState();
          loadCandidates();
        };
      });
    }
    function renderStats() {
      const visible = filteredCandidates();
      const selected = selectedCandidates();
      const visibleCounts = countBboxes(visible);
      const selectedCounts = countBboxes(selected);
      document.getElementById('stats').innerHTML = [
        stat('visible images', visible.length),
        stat('ship boxes', visibleCounts.ship),
        stat('ships boxes', visibleCounts.ships),
        stat('neither boxes', visibleCounts.neither),
      ].join('');
      document.getElementById('selectionStats').innerHTML = [
        stat('selected images', selected.length),
        stat('ship', selectedCounts.ship),
        stat('ships', selectedCounts.ships),
        stat('total boxes', selectedCounts.total),
      ].join('');
      document.getElementById('saveBtn').disabled = selected.length === 0;
      document.getElementById('exportBtn').disabled = !state.currentSelectionPath;
      document.getElementById('continueBtn').disabled = !state.currentSelectionPath;
      document.getElementById('selectionPathHint').textContent = state.currentSelectionPath ? state.currentSelectionPath : 'Save a selection to enable export.';
    }
    function renderGrid() {
      const visible = filteredCandidates();
      const selected = new Set(state.selectedKeys);
      const host = document.getElementById('grid');
      if (!visible.length) {
        host.innerHTML = '<p class="empty">No dataset candidates match these filters.</p>';
        renderStats();
        return;
      }
      host.innerHTML = visible.map(candidate => {
        const counts = candidate.counts || {};
        const checked = selected.has(candidate.image_key) ? 'checked' : '';
        const selectedClass = checked ? ' selected' : '';
        const imageUrl = candidateCacheUrl(candidate);
        const albums = (candidate.album_names || []).slice(0, 3).map(name => `<span class="badge album">${escape(name)}</span>`).join('');
        return `<article class="tile${selectedClass}" data-image-key="${escape(candidate.image_key)}">
          <div class="thumb">${imageUrl ? `<img src="${escape(imageUrl)}" alt="">` : 'No preview'}</div>
          <div class="tile-meta">
            <div class="tile-head"><div class="tile-title">${escape(candidate.image_filename || candidate.image_key)}</div><input type="checkbox" ${checked} aria-label="Select ${escape(candidate.image_key)}"></div>
            <div class="tile-sub mono">${escape(candidate.image_key)}</div>
            <div class="badge-row">
              <span class="badge ship">ship ${escape(counts.ship || 0)}</span>
              <span class="badge ships">ships ${escape(counts.ships || 0)}</span>
              <span class="badge neither">neither ${escape(counts.neither || 0)}</span>
              ${albums}
            </div>
          </div>
        </article>`;
      }).join('');
      host.querySelectorAll('.tile').forEach(tile => {
        tile.onclick = event => {
          if (event.target && event.target.tagName === 'INPUT') event.stopPropagation();
          const key = tile.getAttribute('data-image-key');
          toggleSelection(key);
        };
        const box = tile.querySelector('input[type="checkbox"]');
        box.onclick = event => { event.stopPropagation(); toggleSelection(tile.getAttribute('data-image-key')); };
      });
      renderStats();
    }
    function renderSaved() {
      const host = document.getElementById('savedList');
      if (!state.saved.length) {
        host.innerHTML = '<p class="empty">No saved selections yet.</p>';
        return;
      }
      host.innerHTML = state.saved.map(item => {
        const active = item.selection_path === state.currentSelectionPath ? ' active' : '';
        return `<article class="saved-item${active}" data-selection-path="${escape(item.selection_path)}">
          <div class="saved-title">${escape(item.dataset_name || item.name || 'Dataset')}</div>
          <div class="saved-meta">${escape(item.dataset_version || '')} · ${escape(item.item_count || 0)} images</div>
          <div class="mono">${escape(item.selection_path || '')}</div>
          <button>Select</button>
        </article>`;
      }).join('');
      host.querySelectorAll('.saved-item button').forEach(button => {
        button.onclick = () => {
          state.currentSelectionPath = button.closest('.saved-item').getAttribute('data-selection-path');
          logAction('saved_selection_selected', { selection_path: state.currentSelectionPath });
          persistState();
          renderSaved();
          renderStats();
        };
      });
    }
    function render() {
      document.getElementById('search').value = state.search;
      document.getElementById('labelFilter').value = state.label;
      document.getElementById('includeIncomplete').checked = state.includeIncomplete;
      renderAlbums();
      renderGrid();
      renderSaved();
    }
    function toggleSelection(key) {
      const set = new Set(state.selectedKeys);
      if (set.has(key)) set.delete(key); else set.add(key);
      state.selectedKeys = Array.from(set);
      logAction('candidate_selection_toggled', { image_key: key, selected: set.has(key) });
      persistState();
      renderGrid();
    }
    function selectVisible() {
      const set = new Set(state.selectedKeys);
      for (const candidate of filteredCandidates()) set.add(candidate.image_key);
      state.selectedKeys = Array.from(set);
      logAction('select_visible', { visible_count: filteredCandidates().length });
      persistState();
      renderGrid();
    }
    function clearSelection() {
      state.selectedKeys = [];
      state.currentSelectionPath = null;
      logAction('clear_selection');
      persistState();
      renderGrid();
      renderSaved();
    }
    function queryForCandidates() {
      const params = new URLSearchParams();
      for (const albumId of state.selectedAlbumIds) params.append('album_id', albumId);
      if (state.label && state.label !== 'all') params.append('label', state.label);
      if (state.includeIncomplete) params.set('include_incomplete', '1');
      return params.toString() ? '?' + params.toString() : '';
    }
    async function fetchJson(url, options) {
      const response = await fetch(url, options);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || response.statusText);
      return payload;
    }
    async function loadAlbums() {
      const payload = await fetchJson('/api/albums');
      state.albums = payload.albums || [];
    }
    async function loadSelections() {
      const payload = await fetchJson('/api/dataset-selections');
      state.saved = payload.selections || [];
    }
    async function loadCandidates() {
      try {
        const payload = await fetchJson('/api/dataset-candidates' + queryForCandidates());
        state.candidates = payload.candidates || [];
        const known = new Set(state.candidates.map(candidate => candidate.image_key));
        state.selectedKeys = state.selectedKeys.filter(key => known.has(key));
        persistState();
        render();
        logAction('candidates_loaded', { candidate_count: state.candidates.length });
      } catch (error) {
        showToast('Could not load candidates: ' + error.message, 'error');
        logAction('candidates_load_error', { message: error.message });
      }
    }
    function splitRatios() {
      return {
        train: Number(document.getElementById('trainRatio').value || 0),
        val: Number(document.getElementById('valRatio').value || 0),
        test: Number(document.getElementById('testRatio').value || 0),
      };
    }
    async function saveSelection() {
      const imageKeys = selectedCandidates().map(candidate => candidate.image_key);
      const payload = {
        dataset_name: document.getElementById('datasetName').value.trim(),
        dataset_version: document.getElementById('datasetVersion').value.trim() || 'v1',
        description: document.getElementById('description').value.trim() || null,
        image_keys: imageKeys,
        album_ids: state.selectedAlbumIds,
        split_ratios: splitRatios(),
      };
      if (!payload.dataset_name) {
        showToast('Dataset name is required', 'error');
        return;
      }
      try {
        logAction('save_selection_started', { image_count: imageKeys.length, dataset_name: payload.dataset_name, dataset_version: payload.dataset_version });
        const result = await fetchJson('/api/dataset-selections', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        state.currentSelectionPath = result.selection_path;
        await loadSelections();
        persistState();
        renderSaved();
        renderStats();
        showToast('Selection saved');
        logAction('save_selection_done', { selection_path: result.selection_path, image_count: imageKeys.length });
      } catch (error) {
        showToast('Save failed: ' + error.message, 'error');
        logAction('save_selection_error', { message: error.message });
      }
    }
    async function exportSelection() {
      if (!state.currentSelectionPath) return;
      try {
        logAction('export_selection_started', { selection_path: state.currentSelectionPath });
        const result = await fetchJson('/api/dataset-selections/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ selection_path: state.currentSelectionPath }) });
        showToast(`Exported ${result.exported_image_count || 0} images`);
        logAction('export_selection_done', result);
      } catch (error) {
        showToast('Export failed: ' + error.message, 'error');
        logAction('export_selection_error', { message: error.message });
      }
    }
    async function boot() {
      restoreState();
      document.getElementById('search').value = state.search;
      document.getElementById('labelFilter').value = state.label;
      document.getElementById('includeIncomplete').checked = state.includeIncomplete;
      document.getElementById('search').oninput = event => { state.search = event.target.value; persistState(); renderGrid(); logAction('search_changed', { length: state.search.length }); };
      document.getElementById('labelFilter').onchange = event => { state.label = event.target.value; persistState(); logAction('label_filter_changed', { label: state.label }); loadCandidates(); };
      document.getElementById('includeIncomplete').onchange = event => { state.includeIncomplete = event.target.checked; persistState(); logAction('include_incomplete_changed', { include_incomplete: state.includeIncomplete }); loadCandidates(); };
      document.getElementById('reloadBtn').onclick = () => { logAction('reload_pressed'); loadCandidates(); };
      document.getElementById('selectVisibleBtn').onclick = selectVisible;
      document.getElementById('clearBtn').onclick = clearSelection;
      document.getElementById('saveBtn').onclick = saveSelection;
      document.getElementById('exportBtn').onclick = exportSelection;
      document.getElementById('continueBtn').onclick = () => {
        logAction('continue_to_training_setup', { selection_path: state.currentSelectionPath });
        const target = '/training-setup?selection_path=' + encodeURIComponent(state.currentSelectionPath || '');
        window.location.href = target;
      };
      try {
        await Promise.all([loadAlbums(), loadSelections()]);
        await loadCandidates();
      } catch (error) {
        showToast('Dataset selection failed to initialize: ' + error.message, 'error');
        logAction('boot_error', { message: error.message });
      }
    }
    boot();
  </script>
</body>
</html>"""


def render_training_setup_html() -> str:
    return TRAINING_SETUP_HTML.replace("__MANUAL_REVIEW_HREF__", MANUAL_REVIEW_URL.rstrip("/") + "/")


def render_train_yolo_html() -> str:
    return TRAIN_YOLO_HTML.replace("__MANUAL_REVIEW_HREF__", MANUAL_REVIEW_URL.rstrip("/") + "/")


TRAIN_YOLO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenDarkSTS / Train YOLO</title>
  <style>
    :root { --ink: #121817; --paper: #ebe6d8; --card: #f8f4e8; --line: rgba(18,24,23,0.18); --muted: #68736f; --blue: #2458a6; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--paper); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .shell { max-width: 1180px; margin: 0 auto; padding: 28px; }
    header { display: grid; grid-template-columns: 1fr auto; gap: 20px; align-items: start; margin-bottom: 28px; }
    h1 { margin: 0; font-size: clamp(34px, 6vw, 74px); letter-spacing: -0.06em; line-height: 0.9; }
    em { font-family: Georgia, serif; font-weight: 400; }
    .subtitle { color: var(--muted); max-width: 680px; font-size: 16px; line-height: 1.5; }
    .filenum { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted); font-weight: 700; margin-bottom: 8px; }
    .rail { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
    .stage { color: inherit; text-decoration: none; border: 1px solid var(--line); background: rgba(255,255,255,0.35); border-radius: 999px; padding: 8px 11px; font-size: 12px; font-weight: 700; }
    .stage:hover { border-color: var(--blue); color: var(--blue); }
    .stage.active { background: var(--ink); border-color: var(--ink); color: white; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 22px; padding: 26px; box-shadow: 0 18px 40px rgba(18,24,23,0.08); }
    .card h2 { margin: 0 0 10px; font-size: 24px; }
    .actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
    .button { display: inline-block; color: white; background: var(--blue); text-decoration: none; border-radius: 12px; padding: 11px 14px; font-weight: 700; }
    .button.secondary { color: var(--ink); background: transparent; border: 1px solid var(--line); }
    @media (max-width: 860px) { header { grid-template-columns: 1fr; } .rail { justify-content: flex-start; } }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <div class="filenum">06<sup>/07</sup></div>
        <h1>Train <em>YOLO</em></h1>
        <p class="subtitle">Launch and monitor YOLO training from pipeline-managed configs. The full Step 6 run controls are being wired next; this page is available so the stage rail can navigate here from every previous step.</p>
      </div>
      <nav class="rail" aria-label="Training pipeline stages">
        <a class="stage" href="/import">Import Data</a>
        <a class="stage" href="__MANUAL_REVIEW_HREF__">Manual Review</a>
        <a class="stage" href="/bbox-overview">BBox Overview</a>
        <a class="stage" href="/dataset-selection">Dataset Selection</a>
        <a class="stage" href="/training-setup">Training Setup</a>
        <a class="stage active" href="/train-yolo">Train YOLO</a>
        <span class="stage">Evaluate</span>
      </nav>
    </header>
    <main class="card">
      <h2>Step 6 Placeholder</h2>
      <p class="subtitle">Use this page as the navigation target while the Step 6 backend, CLI, and active-run UI are implemented. Training setup remains the source for saved configs.</p>
      <div class="actions">
        <a class="button" href="/training-setup">Back To Training Setup</a>
        <a class="button secondary" href="/dataset-selection">Dataset Selection</a>
      </div>
    </main>
  </div>
</body>
</html>"""


TRAINING_SETUP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenDarkSTS / Training Setup</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@9..144,300..900,0..100&family=IBM+Plex+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --ink: #111713;
      --ink-soft: #2a3530;
      --muted: #6f7a6e;
      --paper: #ece7da;
      --paper-warm: #e6dfca;
      --paper-cool: #d9e1d7;
      --card: #f8f4e8;
      --card-deep: #efe7d2;
      --line: rgba(17, 23, 19, 0.18);
      --line-strong: rgba(17, 23, 19, 0.42);
      --green: #0f6b4f;
      --red: #a83226;
      --blue: #2458a6;
      --amber: #b7791f;
      --stamp: #7a1f12;
      --stamp-deep: #4d1109;
      --rule: rgba(17, 23, 19, 0.12);
      --shadow: 0 18px 48px rgba(17, 23, 19, 0.16);
      --serif: "Fraunces", "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      --sans: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      --mono: "IBM Plex Mono", "Cascadia Mono", monospace;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(1200px 600px at 12% -10%, rgba(122, 31, 18, 0.06), transparent 60%),
        radial-gradient(900px 700px at 110% 110%, rgba(15, 107, 79, 0.07), transparent 60%),
        linear-gradient(90deg, rgba(17, 23, 19, 0.045) 1px, transparent 1px) 0 0 / 36px 36px,
        linear-gradient(180deg, #ded7c5 0%, #f2eddf 48%, #d9e1d7 100%);
      font-family: var(--sans);
      font-feature-settings: "ss01", "cv11";
    }
    body::before {
      content: "";
      position: fixed; inset: 0; pointer-events: none; z-index: 0;
      background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.06 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
      mix-blend-mode: multiply;
      opacity: 0.55;
    }
    .shell { position: relative; z-index: 1; min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
    header {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) auto;
      gap: 28px;
      align-items: end;
      padding: 22px 36px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(248, 244, 232, 0.72);
      backdrop-filter: blur(14px);
      position: sticky;
      top: 0;
      z-index: 6;
    }
    .titleblock { display: grid; grid-template-columns: auto 1fr; gap: 16px; align-items: center; }
    .filenum {
      font-family: var(--serif);
      font-weight: 300;
      font-style: italic;
      font-size: clamp(56px, 6vw, 96px);
      line-height: 0.85;
      color: var(--stamp);
      letter-spacing: -0.04em;
      transform: translateY(4px);
    }
    .filenum sup { font-size: 0.42em; vertical-align: super; color: var(--muted); font-style: normal; letter-spacing: 0.18em; font-family: var(--sans); font-weight: 700; padding-left: 4px; }
    h1 {
      margin: 0;
      font-family: var(--serif);
      font-weight: 500;
      font-size: clamp(30px, 3.4vw, 50px);
      letter-spacing: -0.035em;
      line-height: 1.0;
    }
    h1 em { font-style: italic; font-weight: 400; color: var(--ink-soft); }
    .subtitle { margin: 8px 0 0; color: var(--muted); font-size: 13px; max-width: 56ch; letter-spacing: 0.01em; }
    .rail { display: flex; gap: 7px; align-items: center; justify-content: flex-end; flex-wrap: wrap; }
    .stage {
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 11px;
      background: rgba(255, 255, 255, 0.36);
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      text-decoration: none;
    }
    .stage.active { background: var(--ink); color: var(--card); border-color: var(--ink); }
    .stage:hover:not(.active):not(span) { background: rgba(255, 255, 255, 0.8); color: var(--ink); }
    input, select, textarea, button {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 12px;
      font: inherit;
      font-size: 13px;
      background: rgba(248, 244, 232, 0.92);
      color: var(--ink);
    }
    input:focus, select:focus, textarea:focus { outline: 2px solid var(--green); outline-offset: -1px; border-color: var(--green); }
    textarea { min-height: 72px; resize: vertical; }
    button { cursor: pointer; font-weight: 700; letter-spacing: 0.02em; border-radius: 6px; transition: background 120ms ease, border-color 120ms ease; }
    button:hover:not(:disabled) { background: rgba(248, 244, 232, 1); }
    button.primary { background: var(--ink); border-color: var(--ink); color: var(--card); }
    button.primary:hover:not(:disabled) { background: var(--ink-soft); }
    button.secondary { background: rgba(36, 88, 166, 0.10); border-color: rgba(36, 88, 166, 0.24); color: var(--blue); }
    button.secondary:hover:not(:disabled) { background: rgba(36, 88, 166, 0.18); }
    button:disabled { cursor: not-allowed; opacity: 0.38; }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      min-height: 0;
    }
    .form-pane { padding: 28px 36px 60px; min-width: 0; display: grid; gap: 26px; align-content: start; }
    aside.composer { border-left: 1px solid var(--line); background: rgba(248, 244, 232, 0.60); padding: 22px; }
    .sticky { position: sticky; top: 116px; max-height: calc(100vh - 116px); overflow: auto; display: grid; gap: 22px; align-content: start; }
    section.card {
      background: rgba(248, 244, 232, 0.78);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 20px 22px;
      box-shadow: 0 2px 8px rgba(17, 23, 19, 0.05);
      display: grid;
      gap: 14px;
    }
    section.card h2 {
      margin: 0;
      font-family: var(--serif);
      font-weight: 500;
      font-size: 22px;
      letter-spacing: -0.03em;
    }
    section.card h2 em { font-style: italic; color: var(--ink-soft); font-weight: 400; }
    section.card .hint { color: var(--muted); font-size: 12px; font-style: italic; margin: 0; }
    .field { display: grid; gap: 5px; }
    .field label { color: var(--muted); font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
    .field-help { color: var(--muted); font-size: 11px; font-style: italic; margin-top: 1px; }
    .field-label-row { display: flex; align-items: baseline; gap: 6px; }
    .field-label-row label { flex: 1; }
    .recommend { color: var(--green); font-size: 10px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
    .hp-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px 18px; }
    .weight-list { display: grid; gap: 6px; max-height: 320px; overflow: auto; padding-right: 4px; }
    .weight-row {
      display: grid;
      grid-template-columns: 22px 1fr auto;
      align-items: center;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.32);
      padding: 9px 11px;
      cursor: pointer;
      transition: background 120ms ease, border-color 120ms ease;
    }
    .weight-row:hover { background: rgba(255, 255, 255, 0.6); }
    .weight-row:has(input:checked) { background: rgba(15, 107, 79, 0.08); border-color: rgba(15, 107, 79, 0.36); }
    .weight-row input[type="checkbox"] { margin: 0; }
    .weight-name { font-family: var(--mono); font-size: 12px; color: var(--ink); }
    .weight-meta { font-size: 11px; color: var(--muted); margin-top: 2px; }
    .prov {
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      padding: 3px 7px;
      border-radius: 4px;
      white-space: nowrap;
    }
    .prov.documented { background: rgba(15, 107, 79, 0.14); color: var(--green); }
    .prov.undocumented { background: rgba(183, 121, 31, 0.18); color: var(--amber); }
    .preset-row { display: flex; gap: 6px; flex-wrap: wrap; }
    .preset-chip {
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.42);
      color: var(--muted);
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      cursor: pointer;
      transition: background 120ms ease, color 120ms ease, border-color 120ms ease;
    }
    .preset-chip:hover { background: rgba(255, 255, 255, 0.85); color: var(--ink); }
    .preset-chip.active { background: var(--ink); border-color: var(--ink); color: var(--card); }
    .aug-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px 14px; }
    details.advanced { border: 1px dashed var(--line); border-radius: 8px; padding: 10px 14px; background: rgba(255, 255, 255, 0.22); }
    details.advanced summary { cursor: pointer; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
    details.advanced[open] summary { margin-bottom: 10px; color: var(--ink); }
    .run-name-list {
      display: grid;
      gap: 4px;
      max-height: 200px;
      overflow: auto;
      padding: 8px 10px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.22);
      font-family: var(--mono);
      font-size: 11px;
      color: var(--ink-soft);
    }
    .run-name-list .empty { color: var(--muted); font-style: italic; font-family: var(--sans); }
    .check-list { display: grid; gap: 6px; }
    .check-item { display: flex; gap: 8px; align-items: flex-start; font-size: 12px; line-height: 1.35; }
    .check-item .mark { font-family: var(--mono); font-weight: 700; line-height: 1.35; flex-shrink: 0; }
    .check-item.pass .mark { color: var(--green); }
    .check-item.warn .mark { color: var(--amber); }
    .check-item.block .mark { color: var(--red); }
    .check-item.pass { color: var(--ink-soft); }
    .check-item.warn { color: var(--ink-soft); }
    .check-item.block { color: var(--ink); font-weight: 600; }
    .saved-list { display: grid; gap: 6px; }
    .saved-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.28);
      padding: 9px 11px;
      display: grid;
      gap: 4px;
      cursor: pointer;
      transition: background 120ms ease, border-color 120ms ease;
    }
    .saved-item:hover { background: rgba(255, 255, 255, 0.55); border-color: var(--line-strong); }
    .saved-item.active { border-color: var(--green); background: rgba(15, 107, 79, 0.07); }
    .saved-title { font-weight: 700; font-size: 12px; color: var(--ink); }
    .saved-meta { color: var(--muted); font-size: 11px; }
    .footer-actions { display: grid; gap: 8px; padding-top: 4px; border-top: 1px solid var(--rule); margin-top: 6px; }
    .footer-actions .secondary-row { display: flex; gap: 6px; }
    .footer-actions .secondary-row button { flex: 1; }
    .dataset-detail {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.32);
      padding: 12px 14px;
      display: grid;
      gap: 6px;
    }
    .dataset-detail .stat-row { display: flex; gap: 6px; flex-wrap: wrap; }
    .dataset-stat {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      background: rgba(248, 244, 232, 0.82);
      color: var(--muted);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.04em;
    }
    .dataset-stat strong { color: var(--ink); font-family: var(--mono); padding-right: 4px; }
    .checkbox-row { display: flex; gap: 8px; align-items: center; font-size: 12px; color: var(--muted); cursor: pointer; }
    .toast {
      position: fixed;
      left: 50%;
      bottom: 22px;
      transform: translateX(-50%) translateY(18px);
      opacity: 0;
      pointer-events: none;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--ink);
      color: var(--card);
      padding: 10px 16px;
      font-size: 13px;
      font-weight: 600;
      box-shadow: var(--shadow);
      transition: opacity 160ms ease, transform 160ms ease;
      z-index: 20;
      max-width: 420px;
      text-align: center;
    }
    .toast.show { opacity: 1; transform: translateX(-50%) translateY(0); pointer-events: auto; }
    .toast.error { background: var(--red); }
    .empty { color: var(--muted); font-size: 12px; font-style: italic; }
    @media (max-width: 1120px) {
      header { position: static; grid-template-columns: 1fr; }
      .rail { justify-content: flex-start; }
      .layout { grid-template-columns: 1fr; }
      aside.composer { border-left: 0; border-top: 1px solid var(--line); }
      .sticky { position: static; max-height: none; }
      .hp-grid, .aug-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 720px) {
      .hp-grid, .aug-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="titleblock">
        <div class="filenum">05<sup>/07</sup></div>
        <div>
          <h1>Training <em>Setup</em></h1>
          <p class="subtitle">Pick an exported dataset, choose starting weights, set hyperparameters and augmentation, and save a reproducible training config. Launching the run is the next stage.</p>
        </div>
      </div>
      <nav class="rail" aria-label="Training pipeline stages">
        <a class="stage" href="/import">Import Data</a>
        <a class="stage" href="__MANUAL_REVIEW_HREF__">Manual Review</a>
        <a class="stage" href="/bbox-overview">BBox Overview</a>
        <a class="stage" href="/dataset-selection">Dataset Selection</a>
        <a class="stage active" href="/training-setup">Training Setup</a>
        <a class="stage" href="/train-yolo">Train YOLO</a>
        <span class="stage">Evaluate</span>
      </nav>
    </header>

    <main class="layout">
      <div class="form-pane">

        <section class="card">
          <h2>Exported <em>Dataset</em></h2>
          <p class="hint">Only dataset selections that have been exported to YOLO format appear here. Export from the previous stage if your dataset is missing.</p>
          <div class="field">
            <label for="datasetPick">Dataset</label>
            <select id="datasetPick"><option value="">Loading datasets...</option></select>
          </div>
          <div class="dataset-detail" id="datasetDetail"><p class="empty">Pick a dataset to see split and class counts.</p></div>
        </section>

        <section class="card">
          <h2>Starting <em>Weights</em></h2>
          <p class="hint">Tick one weight for a single-model run, or several to launch a sequential suite. Run names are derived per weight.</p>
          <div class="weight-list" id="weightList"><p class="empty">Loading available weights...</p></div>
        </section>

        <section class="card">
          <h2>Hyper<em>parameters</em></h2>
          <p class="hint">Defaults are rule-of-thumb starting points. Hover or focus a field for what it controls.</p>
          <div class="hp-grid">
            <div class="field">
              <div class="field-label-row"><label for="imageSize">Image size</label><span class="recommend">rec 960</span></div>
              <input id="imageSize" type="number" min="32" step="32" value="960">
              <span class="field-help">Square training resolution in pixels. 640 / 960 / 1280 are common; match your annotation scale.</span>
            </div>
            <div class="field">
              <div class="field-label-row"><label for="epochs">Epochs</label><span class="recommend">rec 150</span></div>
              <input id="epochs" type="number" min="1" step="1" value="150">
              <span class="field-help">Full passes over the dataset. 100&ndash;300 is typical for transfer learning; early-stop trims it.</span>
            </div>
            <div class="field">
              <div class="field-label-row"><label for="batch">Batch</label><span class="recommend">rec 8</span></div>
              <input id="batch" type="number" min="-1" step="1" value="8">
              <span class="field-help">Images per gradient step. Largest that fits in GPU memory; use -1 to auto-fit.</span>
            </div>
            <div class="field">
              <div class="field-label-row"><label for="patience">Patience</label><span class="recommend">rec 20</span></div>
              <input id="patience" type="number" min="0" step="1" value="20">
              <span class="field-help">Early-stop after N epochs without val mAP improvement. 20&ndash;50 is typical.</span>
            </div>
            <div class="field">
              <div class="field-label-row"><label for="seed">Seed</label><span class="recommend">rec 42</span></div>
              <input id="seed" type="number" min="0" step="1" value="42">
              <span class="field-help">Random seed. Reuse the same seed to reproduce a run.</span>
            </div>
            <div class="field">
              <div class="field-label-row"><label for="device">Device</label><span class="recommend" id="deviceRec">auto</span></div>
              <input id="device" value="auto">
              <span class="field-help">GPU index (<code>0</code>), multi-GPU (<code>0,1</code>), or <code>cpu</code>. Auto-detected on load.</span>
            </div>
            <div class="field">
              <div class="field-label-row"><label for="workers">Workers</label><span class="recommend">rec 4</span></div>
              <input id="workers" type="number" min="0" step="1" value="4">
              <span class="field-help">DataLoader worker processes. 4&ndash;8 is fine on most machines.</span>
            </div>
          </div>
        </section>

        <section class="card">
          <h2>Aug<em>mentation</em></h2>
          <p class="hint">Pick a preset to fill the fields below, then tweak. Unknown keys are rejected by the backend whitelist.</p>
          <div class="preset-row" id="augPresets">
            <button type="button" class="preset-chip" data-preset="none">None</button>
            <button type="button" class="preset-chip" data-preset="light">Light</button>
            <button type="button" class="preset-chip active" data-preset="default">Default</button>
            <button type="button" class="preset-chip" data-preset="heavy">Heavy</button>
          </div>
          <div class="aug-grid" id="augCommon">
            <div class="field"><label for="aug_mosaic">mosaic</label><input id="aug_mosaic" type="number" min="0" max="1" step="0.05" value="1.0"></div>
            <div class="field"><label for="aug_mixup">mixup</label><input id="aug_mixup" type="number" min="0" max="1" step="0.01" value="0.05"></div>
            <div class="field"><label for="aug_fliplr">fliplr</label><input id="aug_fliplr" type="number" min="0" max="1" step="0.05" value="0.5"></div>
            <div class="field"><label for="aug_flipud">flipud</label><input id="aug_flipud" type="number" min="0" max="1" step="0.05" value="0.5"></div>
            <div class="field"><label for="aug_degrees">degrees</label><input id="aug_degrees" type="number" min="0" max="360" step="1" value="180"></div>
            <div class="field"><label for="aug_scale">scale</label><input id="aug_scale" type="number" min="0" max="1" step="0.05" value="0.4"></div>
            <div class="field"><label for="aug_translate">translate</label><input id="aug_translate" type="number" min="0" max="1" step="0.05" value="0.1"></div>
          </div>
          <details class="advanced" id="augAdvanced">
            <summary>Advanced augmentation</summary>
            <div class="aug-grid">
              <div class="field"><label for="aug_hsv_h">hsv_h</label><input id="aug_hsv_h" type="number" min="0" max="1" step="0.005" value="0"></div>
              <div class="field"><label for="aug_hsv_s">hsv_s</label><input id="aug_hsv_s" type="number" min="0" max="1" step="0.05" value="0.35"></div>
              <div class="field"><label for="aug_hsv_v">hsv_v</label><input id="aug_hsv_v" type="number" min="0" max="1" step="0.05" value="0.25"></div>
              <div class="field"><label for="aug_shear">shear</label><input id="aug_shear" type="number" min="0" max="45" step="1" value="0"></div>
              <div class="field"><label for="aug_perspective">perspective</label><input id="aug_perspective" type="number" min="0" max="0.001" step="0.0001" value="0"></div>
              <div class="field"><label for="aug_copy_paste">copy_paste</label><input id="aug_copy_paste" type="number" min="0" max="1" step="0.05" value="0"></div>
              <div class="field"><label for="aug_erasing">erasing</label><input id="aug_erasing" type="number" min="0" max="1" step="0.05" value="0"></div>
              <div class="field"><label for="aug_crop_fraction">crop_fraction</label><input id="aug_crop_fraction" type="number" min="0" max="1" step="0.05" value="0"></div>
            </div>
          </details>
        </section>

        <section class="card">
          <h2>Run <em>Identity</em></h2>
          <p class="hint">The run group is the experiment label. Run names are derived per selected weight as <code>&lt;weight_stem&gt;_&lt;run_group&gt;</code>.</p>
          <div class="field">
            <label for="runGroup">Run group</label>
            <input id="runGroup" placeholder="e.g. manual_reviewed_gavin_balanced_1to1_v1_e150_img960">
          </div>
          <div class="field">
            <label>Derived run names</label>
            <div class="run-name-list" id="runNameList"><span class="empty">Tick at least one weight and enter a run group.</span></div>
          </div>
          <label class="checkbox-row"><input id="overwrite" type="checkbox"> Overwrite existing run directories</label>
        </section>

      </div>

      <aside class="composer">
        <div class="sticky">

          <section class="card">
            <h2>Readiness</h2>
            <p class="hint" id="readinessHint">Fill in a run group, dataset, and at least one weight to see readiness.</p>
            <div class="check-list" id="readinessList"></div>
          </section>

          <section class="card">
            <h2>Saved <em>Configs</em></h2>
            <div class="saved-list" id="savedList"><p class="empty">Loading saved configs...</p></div>
          </section>

          <section class="card">
            <h2>Actions</h2>
            <div class="footer-actions">
              <button class="primary" id="saveBtn" disabled>Save Config</button>
              <div class="secondary-row">
                <button id="resetBtn">Reset</button>
                <button id="continueBtn" disabled title="Available in Step 6">Continue to Train YOLO</button>
              </div>
              <p class="hint" id="savePathHint">Save a config to use it for training.</p>
            </div>
          </section>

        </div>
      </aside>
    </main>
  </div>
  <div class="toast" id="toast"></div>

  <script>
    const stateKey = 'opendarksts.trainingSetup.state.v1';
    const AUG_KEYS_COMMON = ['mosaic','mixup','fliplr','flipud','degrees','scale','translate'];
    const AUG_KEYS_ADVANCED = ['hsv_h','hsv_s','hsv_v','shear','perspective','copy_paste','erasing','crop_fraction'];
    const AUG_KEYS = AUG_KEYS_COMMON.concat(AUG_KEYS_ADVANCED);
    const AUG_PRESETS = {
      none: { mosaic:0, mixup:0, fliplr:0, flipud:0, degrees:0, scale:0, translate:0, hsv_h:0, hsv_s:0, hsv_v:0, shear:0, perspective:0, copy_paste:0, erasing:0, crop_fraction:0 },
      light: { mosaic:0.5, mixup:0, fliplr:0.5, flipud:0, degrees:0, scale:0.2, translate:0.05, hsv_h:0, hsv_s:0.2, hsv_v:0.2, shear:0, perspective:0, copy_paste:0, erasing:0, crop_fraction:0 },
      default: { mosaic:1.0, mixup:0.05, fliplr:0.5, flipud:0.5, degrees:180, scale:0.4, translate:0.1, hsv_h:0.015, hsv_s:0.35, hsv_v:0.25, shear:0, perspective:0, copy_paste:0, erasing:0, crop_fraction:0 },
      heavy: { mosaic:1.0, mixup:0.10, fliplr:0.5, flipud:0.5, degrees:180, scale:0.6, translate:0.2, hsv_h:0.015, hsv_s:0.4, hsv_v:0.3, shear:0, perspective:0, copy_paste:0.1, erasing:0.4, crop_fraction:0 },
    };
    const HP_DEFAULTS = { image_size:960, epochs:150, batch:8, patience:20, seed:42, device:'auto', workers:4 };

    const state = {
      datasets: [],
      weights: [],
      saved: [],
      deviceInfo: null,
      selectedSelectionPath: '',
      selectedWeights: [],
      lastSavedRunGroup: null,
      lastSavedConfigPath: null,
      readiness: null,
    };

    function escape(value) {
      return String(value == null ? '' : value).replace(/[&<>"']/g, ch => ({
        '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
      })[ch]);
    }
    function weightStem(name) {
      const base = (name || '').split('/').pop();
      const dot = base.lastIndexOf('.');
      return dot >= 0 ? base.slice(0, dot) : base;
    }
    function showToast(message, kind) {
      const toast = document.getElementById('toast');
      toast.textContent = message;
      toast.classList.toggle('error', kind === 'error');
      toast.classList.add('show');
      clearTimeout(showToast._t);
      showToast._t = setTimeout(() => toast.classList.remove('show'), 2600);
    }
    function logAction(action, details) {
      const payload = {
        action,
        timestamp: new Date().toISOString(),
        page: 'training-setup',
        selected_weights: state.selectedWeights,
        selection_path: state.selectedSelectionPath,
        details: details || {},
      };
      const body = JSON.stringify(payload);
      try {
        if (navigator.sendBeacon) {
          navigator.sendBeacon('/api/training-ui-interaction', new Blob([body], { type: 'application/json' }));
          return;
        }
      } catch (e) {}
      fetch('/api/training-ui-interaction', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }).catch(() => {});
    }
    function persistState() {
      localStorage.setItem(stateKey, JSON.stringify({
        selectedSelectionPath: state.selectedSelectionPath,
        selectedWeights: state.selectedWeights,
        lastSavedRunGroup: state.lastSavedRunGroup,
        lastSavedConfigPath: state.lastSavedConfigPath,
        runGroup: document.getElementById('runGroup').value,
        overwrite: document.getElementById('overwrite').checked,
        hp: collectHyperparams(),
        aug: collectAugmentation(),
      }));
    }
    function restoreState() {
      try {
        const raw = JSON.parse(localStorage.getItem(stateKey) || '{}');
        if (typeof raw.selectedSelectionPath === 'string') state.selectedSelectionPath = raw.selectedSelectionPath;
        if (Array.isArray(raw.selectedWeights)) state.selectedWeights = raw.selectedWeights.filter(x => typeof x === 'string');
        if (typeof raw.lastSavedRunGroup === 'string') state.lastSavedRunGroup = raw.lastSavedRunGroup;
        if (typeof raw.lastSavedConfigPath === 'string') state.lastSavedConfigPath = raw.lastSavedConfigPath;
        if (raw.runGroup) document.getElementById('runGroup').value = raw.runGroup;
        if (typeof raw.overwrite === 'boolean') document.getElementById('overwrite').checked = raw.overwrite;
        if (raw.hp) applyHyperparams(raw.hp);
        if (raw.aug) applyAugmentation(raw.aug);
      } catch (e) {}
    }
    function applyHyperparams(hp) {
      if (hp.image_size != null) document.getElementById('imageSize').value = hp.image_size;
      if (hp.epochs != null) document.getElementById('epochs').value = hp.epochs;
      if (hp.batch != null) document.getElementById('batch').value = hp.batch;
      if (hp.patience != null) document.getElementById('patience').value = hp.patience;
      if (hp.seed != null) document.getElementById('seed').value = hp.seed;
      if (hp.device != null) document.getElementById('device').value = hp.device;
      if (hp.workers != null) document.getElementById('workers').value = hp.workers;
    }
    function applyAugmentation(aug) {
      for (const key of AUG_KEYS) {
        if (aug[key] != null) {
          const input = document.getElementById('aug_' + key);
          if (input) input.value = aug[key];
        }
      }
    }
    function applyPreset(presetName) {
      const preset = AUG_PRESETS[presetName];
      if (!preset) return;
      applyAugmentation(preset);
      document.querySelectorAll('#augPresets .preset-chip').forEach(chip => {
        chip.classList.toggle('active', chip.dataset.preset === presetName);
      });
      logAction('aug_preset_selected', { preset: presetName });
      persistState();
      requestReadiness();
    }
    function collectHyperparams() {
      return {
        image_size: Number(document.getElementById('imageSize').value || HP_DEFAULTS.image_size),
        epochs: Number(document.getElementById('epochs').value || HP_DEFAULTS.epochs),
        batch: Number(document.getElementById('batch').value || HP_DEFAULTS.batch),
        patience: Number(document.getElementById('patience').value || HP_DEFAULTS.patience),
        seed: Number(document.getElementById('seed').value || HP_DEFAULTS.seed),
        device: document.getElementById('device').value.trim() || HP_DEFAULTS.device,
        workers: Number(document.getElementById('workers').value || HP_DEFAULTS.workers),
      };
    }
    function collectAugmentation() {
      const out = {};
      for (const key of AUG_KEYS) {
        const input = document.getElementById('aug_' + key);
        if (!input) continue;
        const value = Number(input.value);
        if (Number.isFinite(value)) out[key] = value;
      }
      return out;
    }
    function selectedDataset() {
      return state.datasets.find(d => d.selection_path === state.selectedSelectionPath) || null;
    }
    function selectedWeightEntries() {
      const selected = new Set(state.selectedWeights);
      return state.weights.filter(w => selected.has(w.path));
    }
    function derivedRunNames() {
      const group = document.getElementById('runGroup').value.trim();
      if (!group) return [];
      return selectedWeightEntries().map(w => weightStem(w.path) + '_' + group);
    }

    function renderDatasetPicker() {
      const host = document.getElementById('datasetPick');
      if (!state.datasets.length) {
        host.innerHTML = '<option value="">No exported datasets found</option>';
        return;
      }
      const options = state.datasets.map(d => {
        const label = (d.dataset_name || d.name || 'dataset') + (d.dataset_version ? ' / ' + d.dataset_version : '');
        const suffix = d.available === false ? ' (unavailable)' : '';
        const disabled = d.available === false ? ' disabled' : '';
        const selected = d.selection_path === state.selectedSelectionPath ? ' selected' : '';
        return '<option value="' + escape(d.selection_path) + '"' + disabled + selected + '>' + escape(label + suffix) + '</option>';
      });
      host.innerHTML = '<option value="">Select a dataset...</option>' + options.join('');
    }
    function renderDatasetDetail() {
      const host = document.getElementById('datasetDetail');
      const dataset = selectedDataset();
      if (!dataset) {
        host.innerHTML = '<p class="empty">Pick a dataset to see split and class counts.</p>';
        return;
      }
      if (dataset.available === false) {
        host.innerHTML = '<p class="empty">' + escape(dataset.unavailable_reason || 'This dataset is not ready for training.') + '</p>';
        return;
      }
      const splits = dataset.split_counts || dataset.splits || {};
      const counts = dataset.bbox_counts || dataset.counts || {};
      const rows = [
        ['train', splits.train || 0],
        ['val', splits.val || 0],
        ['test', splits.test || 0],
        ['ship boxes', counts.ship || 0],
        ['ships boxes', counts.ships || 0],
      ];
      const stats = rows.map(([label, value]) => '<span class="dataset-stat"><strong>' + escape(value) + '</strong>' + escape(label) + '</span>').join('');
      host.innerHTML = '<div class="stat-row">' + stats + '</div><div style="font-family: var(--mono); font-size: 11px; color: var(--muted);">' + escape(dataset.selection_path) + '</div>';
    }
    function renderWeightList() {
      const host = document.getElementById('weightList');
      if (!state.weights.length) {
        host.innerHTML = '<p class="empty">No starting weights discovered. Add a .pt file to the repo root or workspace/model_weights/.</p>';
        return;
      }
      const selected = new Set(state.selectedWeights);
      host.innerHTML = state.weights.map(w => {
        const checked = selected.has(w.path) ? 'checked' : '';
        const documented = w.provenance && w.provenance.documented;
        const prov = documented
          ? '<span class="prov documented" title="' + escape(w.provenance.source_doc || 'documented') + '">documented</span>'
          : '<span class="prov undocumented" title="No entry in docs/yolo_weight_recreation.md">undocumented</span>';
        const sizeMb = w.size_bytes ? (w.size_bytes / (1024 * 1024)).toFixed(1) + ' MB' : '';
        return '<label class="weight-row" data-weight-path="' + escape(w.path) + '">'
          + '<input type="checkbox" ' + checked + '>'
          + '<span><div class="weight-name">' + escape(w.basename || w.path) + '</div>'
          + '<div class="weight-meta">' + escape(w.path) + (sizeMb ? ' &middot; ' + escape(sizeMb) : '') + '</div></span>'
          + prov
          + '</label>';
      }).join('');
      host.querySelectorAll('.weight-row input[type="checkbox"]').forEach(input => {
        input.onchange = () => {
          const path = input.closest('.weight-row').getAttribute('data-weight-path');
          const set = new Set(state.selectedWeights);
          if (input.checked) set.add(path); else set.delete(path);
          state.selectedWeights = Array.from(set);
          logAction('weight_toggled', { path, selected: input.checked });
          persistState();
          renderRunNames();
          renderActions();
          requestReadiness();
        };
      });
    }
    function renderRunNames() {
      const host = document.getElementById('runNameList');
      const names = derivedRunNames();
      if (!names.length) {
        host.innerHTML = '<span class="empty">Tick at least one weight and enter a run group.</span>';
        return;
      }
      host.innerHTML = names.map(n => '<div>' + escape(n) + '</div>').join('');
    }
    function renderSavedList() {
      const host = document.getElementById('savedList');
      if (!state.saved.length) {
        host.innerHTML = '<p class="empty">No saved configs yet.</p>';
        return;
      }
      host.innerHTML = state.saved.map(item => {
        const active = item.run_group === state.lastSavedRunGroup ? ' active' : '';
        const modelCount = item.model_count || (item.models ? item.models.length : 1);
        const datasetLabel = item.dataset_name ? escape(item.dataset_name) + (item.dataset_version ? ' / ' + escape(item.dataset_version) : '') : 'unknown dataset';
        return '<article class="saved-item' + active + '" data-run-group="' + escape(item.run_group) + '" data-config-path="' + escape(item.config_path || '') + '">'
          + '<div class="saved-title">' + escape(item.run_group) + '</div>'
          + '<div class="saved-meta">' + datasetLabel + ' &middot; ' + escape(modelCount) + ' model' + (modelCount === 1 ? '' : 's') + '</div>'
          + '</article>';
      }).join('');
      host.querySelectorAll('.saved-item').forEach(item => {
        item.onclick = () => {
          const runGroup = item.getAttribute('data-run-group');
          loadSavedConfig(runGroup);
        };
      });
    }
    function renderReadiness() {
      const host = document.getElementById('readinessList');
      const hint = document.getElementById('readinessHint');
      const data = state.readiness;
      if (!data) {
        host.innerHTML = '';
        hint.textContent = 'Fill in a run group, dataset, and at least one weight to see readiness.';
        return;
      }
      hint.textContent = data.ready ? 'Ready to save.' : 'Resolve blockers before saving.';
      const items = [];
      for (const msg of (data.missing || [])) items.push({ level: 'block', message: msg });
      for (const msg of (data.warnings || [])) items.push({ level: 'warn', message: msg });
      if (!items.length && data.ready) items.push({ level: 'pass', message: 'All checks pass.' });
      host.innerHTML = items.map(item => {
        const mark = item.level === 'pass' ? '✓' : (item.level === 'warn' ? '!' : '✗');
        return '<div class="check-item ' + item.level + '"><span class="mark">' + mark + '</span><span>' + escape(item.message) + '</span></div>';
      }).join('');
      if (data.runs && data.runs.length > 1) {
        const perRun = data.runs.map(run => {
          const status = run.ready ? 'pass' : 'block';
          const mark = run.ready ? '✓' : '✗';
          const detail = (run.missing && run.missing.length) ? ': ' + run.missing.join('; ') : '';
          return '<div class="check-item ' + status + '"><span class="mark">' + mark + '</span><span>' + escape(run.run_name || run.model || 'run') + escape(detail) + '</span></div>';
        }).join('');
        host.innerHTML += '<div style="margin-top:8px;border-top:1px solid var(--rule);padding-top:8px;font-size:11px;color:var(--muted);letter-spacing:0.06em;text-transform:uppercase;font-weight:700;">Per-run</div>' + perRun;
      }
    }
    function renderActions() {
      const hasDataset = !!state.selectedSelectionPath;
      const hasWeights = state.selectedWeights.length > 0;
      const hasGroup = document.getElementById('runGroup').value.trim().length > 0;
      const ready = state.readiness && state.readiness.ready;
      document.getElementById('saveBtn').disabled = !(hasDataset && hasWeights && hasGroup && ready);
      const pathHint = document.getElementById('savePathHint');
      if (state.lastSavedConfigPath) {
        pathHint.textContent = 'Saved: ' + state.lastSavedConfigPath;
      } else if (!hasDataset) {
        pathHint.textContent = 'Pick a dataset to begin.';
      } else if (!hasWeights) {
        pathHint.textContent = 'Tick at least one starting weight.';
      } else if (!hasGroup) {
        pathHint.textContent = 'Name the run group.';
      } else if (!ready) {
        pathHint.textContent = 'Resolve readiness blockers to save.';
      } else {
        pathHint.textContent = 'Ready to save.';
      }
    }

    async function fetchJson(url, options) {
      const response = await fetch(url, options);
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || payload.message || response.statusText);
      return payload;
    }

    async function loadDatasets() {
      try {
        const payload = await fetchJson('/api/exported-datasets');
        state.datasets = payload.datasets || payload.result?.datasets || [];
        renderDatasetPicker();
        renderDatasetDetail();
      } catch (error) {
        document.getElementById('datasetPick').innerHTML = '<option value="">Failed to load datasets</option>';
        showToast('Could not load exported datasets: ' + error.message, 'error');
        logAction('datasets_load_error', { message: error.message });
      }
    }
    async function loadWeights() {
      try {
        const payload = await fetchJson('/api/starting-weights');
        state.weights = payload.weights || payload.result?.weights || [];
        renderWeightList();
      } catch (error) {
        document.getElementById('weightList').innerHTML = '<p class="empty">Could not load starting weights: ' + escape(error.message) + '</p>';
        logAction('weights_load_error', { message: error.message });
      }
    }
    async function loadSavedConfigs() {
      try {
        const payload = await fetchJson('/api/training-configs');
        state.saved = payload.configs || payload.result?.configs || [];
        renderSavedList();
      } catch (error) {
        document.getElementById('savedList').innerHTML = '<p class="empty">Could not load saved configs: ' + escape(error.message) + '</p>';
        logAction('saved_configs_load_error', { message: error.message });
      }
    }
    async function loadDeviceInfo() {
      try {
        const payload = await fetchJson('/api/system/device-info');
        state.deviceInfo = payload.device_info || payload.result || payload;
        const recommended = state.deviceInfo.recommended_device || 'cpu';
        document.getElementById('deviceRec').textContent = 'rec ' + recommended;
        const input = document.getElementById('device');
        if (input.value === 'auto' || input.value === '') input.value = recommended;
        if (state.deviceInfo.gpus && state.deviceInfo.gpus.length) {
          const gpu = state.deviceInfo.gpus[0];
          input.title = 'Detected: ' + (gpu.name || 'GPU ' + gpu.index) + (gpu.memory_gb ? ' (' + gpu.memory_gb + ' GB)' : '');
        }
      } catch (error) {
        document.getElementById('deviceRec').textContent = 'cpu (probe failed)';
        document.getElementById('device').value = 'cpu';
      }
    }
    async function loadSavedConfig(runGroup) {
      try {
        const payload = await fetchJson('/api/training-configs/' + encodeURIComponent(runGroup));
        const config = payload.config || payload.result || payload;
        applyLoadedConfig(config);
        showToast('Loaded ' + runGroup);
        logAction('saved_config_loaded', { run_group: runGroup });
      } catch (error) {
        showToast('Load failed: ' + error.message, 'error');
        logAction('saved_config_load_error', { run_group: runGroup, message: error.message });
      }
    }
    function applyLoadedConfig(config) {
      if (!config) return;
      if (config.selection_path) state.selectedSelectionPath = config.selection_path;
      const models = Array.isArray(config.models) ? config.models : (config.model ? [config.model] : []);
      state.selectedWeights = models.map(name => {
        const match = state.weights.find(w => w.basename === name || w.path === name);
        return match ? match.path : name;
      });
      document.getElementById('runGroup').value = config.run_group || '';
      document.getElementById('overwrite').checked = !!config.overwrite;
      applyHyperparams({
        image_size: config.image_size,
        epochs: config.epochs,
        batch: config.batch,
        patience: config.patience,
        seed: config.seed,
        device: config.device,
        workers: config.workers,
      });
      applyAugmentation(config.augmentation || {});
      state.lastSavedRunGroup = config.run_group || null;
      renderDatasetPicker();
      renderDatasetDetail();
      renderWeightList();
      renderRunNames();
      renderSavedList();
      persistState();
      requestReadiness();
    }
    function buildConfigPayload() {
      const hp = collectHyperparams();
      const aug = collectAugmentation();
      const runGroup = document.getElementById('runGroup').value.trim();
      const selected = selectedWeightEntries();
      const dataset = selectedDataset();
      return {
        schema_version: 1,
        selection_path: state.selectedSelectionPath,
        dataset_name: dataset ? dataset.dataset_name : null,
        dataset_version: dataset ? dataset.dataset_version : null,
        models: selected.map(w => w.basename || w.path.split('/').pop()),
        image_size: hp.image_size,
        epochs: hp.epochs,
        batch: hp.batch,
        patience: hp.patience,
        seed: hp.seed,
        device: hp.device,
        workers: hp.workers,
        augmentation: aug,
        run_group: runGroup,
        overwrite: document.getElementById('overwrite').checked,
      };
    }

    let readinessTimer = null;
    function requestReadiness() {
      if (readinessTimer) clearTimeout(readinessTimer);
      readinessTimer = setTimeout(() => {
        readinessTimer = null;
        doReadinessCheck();
      }, 250);
    }
    async function doReadinessCheck() {
      const hasDataset = !!state.selectedSelectionPath;
      const hasWeights = state.selectedWeights.length > 0;
      const hasGroup = document.getElementById('runGroup').value.trim().length > 0;
      if (!(hasDataset && hasWeights && hasGroup)) {
        state.readiness = null;
        renderReadiness();
        renderActions();
        return;
      }
      try {
        const config = buildConfigPayload();
        const payload = await fetchJson('/api/training-config/check-ready', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config, selection_path: state.selectedSelectionPath }),
        });
        state.readiness = payload.result || payload;
        renderReadiness();
        renderActions();
      } catch (error) {
        state.readiness = { ready: false, missing: ['Readiness check failed: ' + error.message], warnings: [] };
        renderReadiness();
        renderActions();
      }
    }

    async function saveConfig() {
      const config = buildConfigPayload();
      if (!config.run_group) {
        showToast('Run group is required', 'error');
        return;
      }
      if (!config.models.length) {
        showToast('Pick at least one starting weight', 'error');
        return;
      }
      try {
        logAction('save_started', { run_group: config.run_group, model_count: config.models.length });
        const payload = await fetchJson('/api/training-configs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config),
        });
        const saved = payload.config || payload.result || payload;
        state.lastSavedRunGroup = saved.run_group || config.run_group;
        state.lastSavedConfigPath = payload.config_path || payload.result?.config_path || null;
        const warnings = payload.provenance_warnings || saved.provenance_warnings || [];
        if (warnings.length) {
          showToast('Saved with ' + warnings.length + ' provenance warning(s)');
        } else {
          showToast('Config saved');
        }
        await loadSavedConfigs();
        persistState();
        document.getElementById('continueBtn').disabled = false;
        document.getElementById('continueBtn').title = 'Available in Step 6';
        renderActions();
        logAction('save_done', { run_group: state.lastSavedRunGroup, config_path: state.lastSavedConfigPath });
      } catch (error) {
        showToast('Save failed: ' + error.message, 'error');
        logAction('save_error', { message: error.message });
      }
    }
    function resetToDefaults() {
      applyHyperparams(HP_DEFAULTS);
      applyPreset('default');
      document.getElementById('runGroup').value = '';
      document.getElementById('overwrite').checked = false;
      state.selectedWeights = [];
      state.selectedSelectionPath = '';
      state.readiness = null;
      state.lastSavedConfigPath = null;
      state.lastSavedRunGroup = null;
      renderDatasetPicker();
      renderDatasetDetail();
      renderWeightList();
      renderRunNames();
      renderReadiness();
      renderActions();
      persistState();
      if (state.deviceInfo && state.deviceInfo.recommended_device) {
        document.getElementById('device').value = state.deviceInfo.recommended_device;
      }
      logAction('reset_clicked');
    }

    function attachInputHandlers() {
      document.getElementById('datasetPick').onchange = (event) => {
        state.selectedSelectionPath = event.target.value;
        renderDatasetDetail();
        persistState();
        logAction('dataset_selected', { selection_path: state.selectedSelectionPath });
        requestReadiness();
        renderActions();
      };
      document.getElementById('runGroup').oninput = () => {
        renderRunNames();
        renderActions();
        persistState();
        requestReadiness();
      };
      document.getElementById('overwrite').onchange = () => {
        persistState();
        logAction('overwrite_toggled', { overwrite: document.getElementById('overwrite').checked });
        requestReadiness();
      };
      const hpIds = ['imageSize','epochs','batch','patience','seed','device','workers'];
      for (const id of hpIds) {
        document.getElementById(id).oninput = () => { persistState(); requestReadiness(); };
      }
      for (const key of AUG_KEYS) {
        const input = document.getElementById('aug_' + key);
        if (input) input.oninput = () => {
          document.querySelectorAll('#augPresets .preset-chip').forEach(chip => chip.classList.remove('active'));
          persistState();
        };
      }
      document.querySelectorAll('#augPresets .preset-chip').forEach(chip => {
        chip.onclick = () => applyPreset(chip.dataset.preset);
      });
      document.getElementById('saveBtn').onclick = saveConfig;
      document.getElementById('resetBtn').onclick = resetToDefaults;
      document.getElementById('continueBtn').onclick = () => {
        logAction('continue_to_train_yolo_stub', { run_group: state.lastSavedRunGroup });
        showToast('Train YOLO is the next stage and is not implemented yet.');
      };
    }

    function applyUrlParams() {
      const params = new URLSearchParams(window.location.search);
      const selectionPath = params.get('selection_path');
      if (selectionPath) state.selectedSelectionPath = selectionPath;
    }

    async function boot() {
      attachInputHandlers();
      applyUrlParams();
      restoreState();
      renderRunNames();
      renderSavedList();
      renderReadiness();
      renderActions();
      try {
        await Promise.all([loadDatasets(), loadWeights(), loadSavedConfigs(), loadDeviceInfo()]);
        renderDatasetPicker();
        renderDatasetDetail();
        renderWeightList();
        renderActions();
        requestReadiness();
        logAction('page_open', { selection_path: state.selectedSelectionPath });
      } catch (error) {
        showToast('Training setup failed to initialize: ' + error.message, 'error');
        logAction('boot_error', { message: error.message });
      }
    }
    boot();
  </script>
</body>
</html>"""


BBOX_OVERVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenDarkSTS / BBox Overview</title>
  <style>
    :root {
      --ink: #111713;
      --muted: #667164;
      --paper: #ece7da;
      --card: #f8f4e8;
      --line: rgba(17, 23, 19, 0.16);
      --green: #0f6b4f;
      --red: #a83226;
      --blue: #2458a6;
      --amber: #b7791f;
      --slate: #44515a;
      --shadow: 0 18px 48px rgba(17, 23, 19, 0.16);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(17, 23, 19, 0.045) 1px, transparent 1px) 0 0 / 36px 36px,
        linear-gradient(180deg, #ded7c5 0%, #f2eddf 48%, #d9e1d7 100%);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
    }
    .shell { min-height: 100vh; display: grid; grid-template-rows: auto auto 1fr; }
    header {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto;
      gap: 24px;
      align-items: end;
      padding: 22px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: rgba(248, 244, 232, 0.78);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 4;
    }
    h1 {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      font-size: clamp(28px, 3vw, 44px);
      letter-spacing: -0.03em;
    }
    .subtitle { margin: 6px 0 0; color: var(--muted); font-size: 14px; }
    .rail { display: flex; gap: 7px; align-items: center; justify-content: flex-end; flex-wrap: wrap; }
    .stage {
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 11px;
      background: rgba(255, 255, 255, 0.34);
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      text-decoration: none;
    }
    .stage.active { background: var(--ink); color: var(--card); border-color: var(--ink); }
    .controls {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) repeat(3, auto);
      gap: 10px;
      align-items: center;
      padding: 14px 28px;
      border-bottom: 1px solid var(--line);
      background: rgba(236, 231, 218, 0.72);
      position: sticky;
      top: 92px;
      z-index: 3;
    }
    input, select, button {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      font: inherit;
      background: rgba(248, 244, 232, 0.92);
      color: var(--ink);
    }
    button { cursor: pointer; font-weight: 800; }
    button.primary { color: #fffaf0; background: var(--green); border-color: var(--green); }
    .size-control { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 13px; }
    .main { display: grid; grid-template-columns: minmax(0, 1fr) 340px; min-height: 0; }
    .contact-sheet { padding: 24px 28px 36px; min-width: 0; }
    .stats { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
    .stat {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 11px;
      background: rgba(248, 244, 232, 0.8);
      font-size: 12px;
      color: var(--muted);
    }
    .stat strong { color: var(--ink); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(var(--tile-size, 170px), 1fr));
      gap: 14px;
      align-items: start;
    }
    .tile {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      background: rgba(248, 244, 232, 0.92);
      box-shadow: 0 10px 26px rgba(17, 23, 19, 0.08);
      cursor: pointer;
      transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
    }
    .tile:hover, .tile.selected { transform: translateY(-2px); border-color: rgba(15, 107, 79, 0.55); box-shadow: var(--shadow); }
    .thumb {
      aspect-ratio: 1 / 0.72;
      display: grid;
      place-items: center;
      background: #17211b;
      color: rgba(255, 250, 240, 0.78);
      overflow: hidden;
    }
    .thumb img { width: 100%; height: 100%; object-fit: contain; display: block; image-rendering: auto; }
    .tile-meta { padding: 10px 11px 12px; display: grid; gap: 7px; }
    .tile-title { display: flex; justify-content: space-between; gap: 8px; align-items: center; font-size: 13px; font-weight: 900; }
    .tile-sub { color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .badge-row { display: flex; gap: 5px; flex-wrap: wrap; }
    .badge { border-radius: 999px; padding: 4px 7px; font-size: 11px; font-weight: 900; }
    .badge.ship { background: rgba(36, 88, 166, 0.14); color: var(--blue); }
    .badge.ships { background: rgba(168, 50, 38, 0.14); color: var(--red); }
    .badge.neither { background: rgba(68, 81, 90, 0.14); color: var(--slate); }
    .badge.warn { background: rgba(183, 121, 31, 0.18); color: #7a4d0e; }
    aside {
      border-left: 1px solid var(--line);
      background: rgba(248, 244, 232, 0.86);
      padding: 22px;
      position: sticky;
      top: 154px;
      height: calc(100vh - 154px);
      overflow: auto;
    }
    .panel { display: grid; gap: 16px; }
    .panel h2 { margin: 0; font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif; font-size: 25px; }
    .kv { display: grid; grid-template-columns: 110px 1fr; gap: 7px 10px; font-size: 13px; }
    .kv span:nth-child(odd) { color: var(--muted); }
    .mono { font-family: "Cascadia Mono", "IBM Plex Mono", monospace; font-size: 12px; overflow-wrap: anywhere; }
    .empty { padding: 56px 20px; text-align: center; color: var(--muted); }
    .conflicts { margin-bottom: 16px; display: grid; gap: 8px; }
    .conflict { border: 1px solid rgba(168, 50, 38, 0.28); border-radius: 14px; padding: 11px; background: rgba(168, 50, 38, 0.08); }
    @media (max-width: 980px) {
      header, .controls { position: static; grid-template-columns: 1fr; }
      .rail { justify-content: flex-start; }
      .main { grid-template-columns: 1fr; }
      aside { position: static; height: auto; border-left: 0; border-top: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>BBox Overview</h1>
        <p class="subtitle">Reviewed crop contact sheet for final visual QA before dataset selection.</p>
      </div>
      <nav class="rail" aria-label="Training pipeline stages">
        <a class="stage" href="/import">Import Data</a>
        <a class="stage" href="__MANUAL_REVIEW_HREF__">Manual Review</a>
        <a class="stage active" href="/bbox-overview">BBox Overview</a>
        <a class="stage" href="/dataset-selection">Dataset Selection</a>
        <a class="stage" href="/training-setup">Training Setup</a>
        <a class="stage" href="/train-yolo">Train YOLO</a>
        <span class="stage">Evaluate</span>
      </nav>
    </header>
    <section class="controls">
      <input id="search" placeholder="Search image, detection, reviewer, model" oninput="renderAndPersist()">
      <select id="classFilter" onchange="renderAndPersist()">
        <option value="all">All reviewed bboxes</option>
        <option value="ship_non_sts">Ship</option>
        <option value="ships_sts">Ships</option>
        <option value="neither">Neither</option>
        <option value="warnings">Warnings</option>
        <option value="conflicts">Needs Reconciliation</option>
      </select>
      <select id="sortMode" onchange="renderAndPersist()">
        <option value="image">Sort by image</option>
        <option value="class">Sort by class</option>
        <option value="area_desc">Largest boxes first</option>
        <option value="confidence_desc">Highest confidence first</option>
      </select>
      <label class="size-control">Tile <input id="tileSize" type="range" min="120" max="280" value="170" oninput="renderAndPersist()"></label>
      <button class="primary" onclick="reloadOverview(true)">Refresh Crops</button>
    </section>
    <main class="main">
      <section class="contact-sheet">
        <div class="stats" id="stats"></div>
        <div class="conflicts" id="conflicts"></div>
        <div class="grid" id="grid"></div>
      </section>
      <aside><div class="panel" id="sidebar"></div></aside>
    </main>
  </div>
  <script>
    const manualReviewUrl = __MANUAL_REVIEW_URL__;
    const overviewStateKey = 'opendarksts.bboxOverview.state.v1';
    let overview = { counts: {}, tiles: [], conflicts: [] };
    let selectedTileId = null;
    let pendingScrollY = null;
    let scrollSaveTimer = null;

    function readStoredOverviewState() {
      try {
        return JSON.parse(localStorage.getItem(overviewStateKey) || sessionStorage.getItem(overviewStateKey) || '{}');
      } catch (error) {
        console.warn('Could not read overview state:', error);
        return {};
      }
    }

    function restoreOverviewState() {
      const stored = readStoredOverviewState();
      if (typeof stored.search === 'string') document.getElementById('search').value = stored.search;
      if (typeof stored.classFilter === 'string') document.getElementById('classFilter').value = stored.classFilter;
      if (typeof stored.sortMode === 'string') document.getElementById('sortMode').value = stored.sortMode;
      if (typeof stored.tileSize === 'string') document.getElementById('tileSize').value = stored.tileSize;
      if (typeof stored.selectedTileId === 'string') selectedTileId = stored.selectedTileId;
      if (Number.isFinite(Number(stored.scrollY))) pendingScrollY = Number(stored.scrollY);
    }

    function currentOverviewState() {
      return {
        search: document.getElementById('search').value,
        classFilter: document.getElementById('classFilter').value,
        sortMode: document.getElementById('sortMode').value,
        tileSize: document.getElementById('tileSize').value,
        selectedTileId,
        scrollY: window.scrollY,
      };
    }

    function persistOverviewState() {
      const serialized = JSON.stringify(currentOverviewState());
      localStorage.setItem(overviewStateKey, serialized);
      sessionStorage.setItem(overviewStateKey, serialized);
    }

    function renderAndPersist() {
      render();
      persistOverviewState();
    }

    function restoreScrollIfNeeded() {
      if (pendingScrollY === null) return;
      const target = pendingScrollY;
      pendingScrollY = null;
      requestAnimationFrame(() => {
        window.scrollTo(0, target);
        persistOverviewState();
      });
    }

    async function reloadOverview(forceGenerate=false) {
      const response = await fetch('/api/overview' + (forceGenerate ? '?generate=1' : ''));
      overview = await response.json();
      if (!overview.tiles.some(tile => tile.tile_id === selectedTileId)) selectedTileId = null;
      render();
      restoreScrollIfNeeded();
    }

    function labelClass(tile) {
      if (tile.review_label === 'ships_sts') return 'ships';
      if (tile.review_label === 'neither') return 'neither';
      return 'ship';
    }

    function labelText(tile) {
      if (tile.review_label === 'ships_sts') return 'ships';
      if (tile.review_label === 'neither') return 'neither';
      return 'ship';
    }

    function filteredTiles() {
      const filter = document.getElementById('classFilter').value;
      if (filter === 'conflicts') return [];
      const query = document.getElementById('search').value.trim().toLowerCase();
      let tiles = overview.tiles.filter(tile => {
        if (filter === 'warnings' && (!tile.warnings || tile.warnings.length === 0)) return false;
        if (filter !== 'all' && filter !== 'warnings' && tile.review_label !== filter) return false;
        if (!query) return true;
        const haystack = [tile.image_key, tile.image_filename, tile.detection_id, tile.reviewer, tile.source_type, ...(tile.source_models || [])].join(' ').toLowerCase();
        return haystack.includes(query);
      });
      const sort = document.getElementById('sortMode').value;
      tiles.sort((left, right) => {
        if (sort === 'class') return String(left.review_label).localeCompare(String(right.review_label)) || String(left.image_key).localeCompare(String(right.image_key));
        if (sort === 'area_desc') return Number(right.bbox_area || 0) - Number(left.bbox_area || 0);
        if (sort === 'confidence_desc') return Number(right.confidence || -1) - Number(left.confidence || -1);
        return String(left.image_key).localeCompare(String(right.image_key)) || String(left.detection_id).localeCompare(String(right.detection_id));
      });
      return tiles;
    }

    function renderStats(tiles) {
      const counts = overview.counts || {};
      document.getElementById('stats').innerHTML = `
        <span class="stat"><strong>${tiles.length}</strong> visible</span>
        <span class="stat"><strong>${counts.reviewed_bbox_count || 0}</strong> reviewed bboxes</span>
        <span class="stat"><strong>${counts.reviewed_image_count || 0}</strong> images</span>
        <span class="stat"><strong>${counts.reviewed_ship_bboxes || 0}</strong> ship</span>
        <span class="stat"><strong>${counts.reviewed_ships_bboxes || 0}</strong> ships</span>
        <span class="stat"><strong>${counts.neither_bboxes || 0}</strong> neither</span>
        <span class="stat"><strong>${counts.warning_tile_count || 0}</strong> warning tiles</span>
        <span class="stat"><strong>${counts.conflict_image_count || 0}</strong> conflicts</span>
      `;
    }

    function renderConflicts() {
      const host = document.getElementById('conflicts');
      const showOnly = document.getElementById('classFilter').value === 'conflicts';
      const conflicts = overview.conflicts || [];
      if (!conflicts.length || !showOnly) {
        host.innerHTML = '';
        return;
      }
      host.innerHTML = conflicts.map(conflict => `
        <div class="conflict">
          <strong>${conflict.image_key}</strong><br>
          <span class="mono">${(conflict.review_files || []).join('<br>')}</span>
        </div>
      `).join('');
    }

    function renderGrid(tiles) {
      const grid = document.getElementById('grid');
      const tileSize = document.getElementById('tileSize').value;
      grid.style.setProperty('--tile-size', `${tileSize}px`);
      if (document.getElementById('classFilter').value === 'conflicts') {
        grid.innerHTML = '';
        return;
      }
      if (!tiles.length) {
        grid.innerHTML = '<div class="empty">No reviewed bbox crops match the current filters.</div>';
        return;
      }
      grid.innerHTML = tiles.map(tile => `
        <article class="tile ${tile.tile_id === selectedTileId ? 'selected' : ''}" onclick="selectTile('${tile.tile_id}')" ondblclick="openManualReview('${tile.image_key}', '${tile.detection_id}')">
          <div class="thumb">${tile.crop_url ? `<img loading="lazy" src="${tile.crop_url}" alt="${tile.image_key} ${tile.detection_id}">` : 'No crop'}</div>
          <div class="tile-meta">
            <div class="tile-title"><span>${tile.detection_id}</span><span class="badge ${labelClass(tile)}">${labelText(tile)}</span></div>
            <div class="tile-sub">${tile.image_filename || tile.image_key}</div>
            <div class="badge-row">
              <span class="badge">${tile.source_type || 'unknown'}</span>
              ${(tile.warnings || []).map(warning => `<span class="badge warn">${warning}</span>`).join('')}
            </div>
          </div>
        </article>
      `).join('');
    }

    function selectedTile() {
      return overview.tiles.find(tile => tile.tile_id === selectedTileId) || null;
    }

    function renderSidebar() {
      const tile = selectedTile();
      const host = document.getElementById('sidebar');
      if (!tile) {
        host.innerHTML = '<h2>No Tile Selected</h2><p class="subtitle">Single-click a crop to inspect metadata. Double-click a crop to open the source image in manual review.</p>';
        return;
      }
      host.innerHTML = `
        <h2>${labelText(tile)} <span class="badge ${labelClass(tile)}">${tile.detection_id}</span></h2>
        <button class="primary" onclick="openManualReview('${tile.image_key}', '${tile.detection_id}')">Edit In Manual Review</button>
        <div class="kv">
          <span>Image</span><span class="mono">${tile.image_filename || tile.image_key}</span>
          <span>Image Key</span><span class="mono">${tile.image_key}</span>
          <span>Reviewer</span><span>${tile.reviewer || 'unknown'}</span>
          <span>Source</span><span>${tile.source_type || 'unknown'}</span>
          <span>Confidence</span><span>${tile.confidence === null || tile.confidence === undefined ? '-' : tile.confidence}</span>
          <span>Models</span><span>${(tile.source_models || []).join(', ') || '-'}</span>
          <span>Dataset</span><span>not assigned yet</span>
          <span>Warnings</span><span>${(tile.warnings || []).join(', ') || 'none'}</span>
          <span>BBox</span><span class="mono">x=${tile.bbox.x_center.toFixed(6)} y=${tile.bbox.y_center.toFixed(6)} w=${tile.bbox.width.toFixed(6)} h=${tile.bbox.height.toFixed(6)}</span>
          <span>Review File</span><span class="mono">${tile.review_file}</span>
        </div>
      `;
    }

    function selectTile(tileId) {
      selectedTileId = tileId;
      renderAndPersist();
    }

    function openManualReview(imageKey, detectionId) {
      persistOverviewState();
      const url = `${manualReviewUrl.replace(/[/]$/, '')}/?image_id=${encodeURIComponent(imageKey)}&detection_id=${encodeURIComponent(detectionId)}`;
      window.location.href = url;
    }

    function render() {
      const tiles = filteredTiles();
      renderStats(tiles);
      renderConflicts();
      renderGrid(tiles);
      renderSidebar();
    }

    window.addEventListener('scroll', () => {
      if (scrollSaveTimer) window.clearTimeout(scrollSaveTimer);
      scrollSaveTimer = window.setTimeout(persistOverviewState, 120);
    }, { passive: true });
    window.addEventListener('pagehide', persistOverviewState);
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') persistOverviewState();
    });

    restoreOverviewState();
    reloadOverview(false);
  </script>
</body>
</html>"""


IMPORT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenDarkSTS / Intake</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@9..144,300..900,0..100&family=IBM+Plex+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --ink: #111713;
      --ink-soft: #2a3530;
      --muted: #6f7a6e;
      --paper: #ece7da;
      --paper-warm: #e6dfca;
      --paper-cool: #d9e1d7;
      --card: #f8f4e8;
      --card-deep: #efe7d2;
      --line: rgba(17, 23, 19, 0.18);
      --line-strong: rgba(17, 23, 19, 0.42);
      --green: #0f6b4f;
      --red: #a83226;
      --blue: #2458a6;
      --amber: #b7791f;
      --stamp: #7a1f12;
      --stamp-deep: #4d1109;
      --rule: rgba(17, 23, 19, 0.12);
      --shadow: 0 18px 48px rgba(17, 23, 19, 0.16);
      --serif: "Fraunces", "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      --sans: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      --mono: "IBM Plex Mono", "Cascadia Mono", monospace;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(1200px 600px at 12% -10%, rgba(122, 31, 18, 0.06), transparent 60%),
        radial-gradient(900px 700px at 110% 110%, rgba(15, 107, 79, 0.07), transparent 60%),
        linear-gradient(90deg, rgba(17, 23, 19, 0.045) 1px, transparent 1px) 0 0 / 36px 36px,
        linear-gradient(180deg, #ded7c5 0%, #f2eddf 48%, #d9e1d7 100%);
      font-family: var(--sans);
      font-feature-settings: "ss01", "cv11";
    }
    body::before {
      content: "";
      position: fixed; inset: 0; pointer-events: none; z-index: 0;
      background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.06 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
      mix-blend-mode: multiply;
      opacity: 0.55;
    }
    .shell { position: relative; z-index: 1; min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }

    header {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) auto;
      gap: 28px;
      align-items: end;
      padding: 22px 36px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(248, 244, 232, 0.72);
      backdrop-filter: blur(14px);
      position: sticky; top: 0; z-index: 6;
    }
    .titleblock { display: grid; grid-template-columns: auto 1fr; gap: 16px; align-items: center; }
    .filenum {
      font-family: var(--serif);
      font-weight: 300;
      font-style: italic;
      font-size: clamp(56px, 6vw, 96px);
      line-height: 0.85;
      color: var(--stamp);
      letter-spacing: -0.04em;
      transform: translateY(4px);
    }
    .filenum sup { font-size: 0.42em; vertical-align: super; color: var(--muted); font-style: normal; letter-spacing: 0.18em; font-family: var(--sans); font-weight: 700; padding-left: 4px; }
    h1 {
      margin: 0;
      font-family: var(--serif);
      font-weight: 500;
      font-size: clamp(30px, 3.4vw, 50px);
      letter-spacing: -0.035em;
      line-height: 1.0;
    }
    h1 em { font-style: italic; font-weight: 400; color: var(--ink-soft); }
    .subtitle {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 13px;
      max-width: 56ch;
      letter-spacing: 0.01em;
    }
    .rail { display: flex; gap: 7px; align-items: center; justify-content: flex-end; flex-wrap: wrap; }
    .stage {
      display: inline-flex; align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 11px;
      background: rgba(255, 255, 255, 0.36);
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      text-decoration: none;
    }
    .stage.active { background: var(--ink); color: var(--card); border-color: var(--ink); }
    .stage:hover:not(.active):not(span) { background: rgba(255, 255, 255, 0.8); color: var(--ink); }

    main { padding: 0; min-height: 0; }

    /* ==================== empty state: intake card ==================== */
    .intake {
      display: grid;
      grid-template-columns: minmax(280px, 580px);
      justify-content: center;
      padding: 64px 28px 80px;
    }
    .intake-card {
      position: relative;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 56px 56px 44px;
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.6) inset,
        0 28px 64px rgba(17, 23, 19, 0.18),
        0 80px 120px rgba(17, 23, 19, 0.06);
      overflow: hidden;
    }
    .intake-card::before {
      content: "";
      position: absolute; inset: 12px;
      border: 1px solid var(--rule);
      pointer-events: none;
      border-radius: 1px;
    }
    .intake-card::after {
      content: "FORM 01-A — TRAINING INTAKE";
      position: absolute; top: 22px; left: 50%;
      transform: translateX(-50%);
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.32em;
      color: var(--muted);
    }
    .intake-stamp {
      position: absolute;
      top: 36px; right: 28px;
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.22em;
      color: var(--stamp);
      border: 1.5px solid var(--stamp);
      padding: 6px 10px;
      transform: rotate(6deg);
      opacity: 0.78;
      border-radius: 3px;
    }
    .intake-eyebrow {
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.32em;
      color: var(--muted);
      text-transform: uppercase;
      margin: 20px 0 6px;
    }
    .intake-title {
      font-family: var(--serif);
      font-weight: 400;
      font-size: clamp(36px, 4.8vw, 60px);
      line-height: 0.95;
      letter-spacing: -0.035em;
      margin: 0 0 8px;
    }
    .intake-title em {
      font-style: italic;
      color: var(--stamp);
    }
    .intake-lede {
      color: var(--muted);
      font-size: 15px;
      line-height: 1.55;
      margin: 0 0 28px;
      max-width: 46ch;
    }
    .intake-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-bottom: 28px;
    }
    .pick-btn {
      position: relative;
      border: 1.5px solid var(--ink);
      background: var(--ink);
      color: var(--card);
      font-family: var(--sans);
      font-weight: 700;
      letter-spacing: 0.04em;
      padding: 18px 18px 16px;
      border-radius: 2px;
      cursor: pointer;
      text-align: left;
      display: grid;
      gap: 4px;
      transition: transform 120ms ease, box-shadow 120ms ease;
    }
    .pick-btn:hover { transform: translateY(-2px); box-shadow: 0 12px 22px rgba(17, 23, 19, 0.22); }
    .pick-btn:active { transform: translateY(0); }
    .pick-btn .glyph { font-family: var(--serif); font-style: italic; font-size: 22px; font-weight: 300; opacity: 0.9; }
    .pick-btn .label { font-size: 13px; }
    .pick-btn .hint { font-size: 11px; opacity: 0.7; font-family: var(--mono); letter-spacing: 0.06em; }
    .pick-btn.secondary { background: transparent; color: var(--ink); }
    .pick-btn.secondary:hover { background: rgba(17, 23, 19, 0.06); }
    .intake-options {
      display: flex; align-items: center; gap: 14px;
      font-size: 12px; color: var(--muted);
      padding-top: 16px; border-top: 1px dashed var(--rule);
    }
    .intake-options label { display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }
    .intake-options input { accent-color: var(--stamp); }
    .album-fields { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 18px 0; }
    .album-fields label { display: grid; gap: 5px; color: var(--muted); font-family: var(--mono); font-size: 11px; letter-spacing: 0.04em; text-transform: uppercase; }
    .album-fields input { width: 100%; border: 1px solid var(--line); border-radius: 3px; background: rgba(255,255,255,0.52); padding: 9px 10px; color: var(--ink); font: 13px var(--mono); }
    .album-fields .wide { grid-column: 1 / -1; }
    .intake-footer {
      margin-top: 22px;
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.04em;
      color: var(--muted);
      line-height: 1.6;
    }
    .intake-footer code { color: var(--ink); background: rgba(17, 23, 19, 0.06); padding: 1px 5px; border-radius: 2px; }

    /* ==================== albums section ==================== */
    .albums-block {
      margin: 22px 0 6px;
      padding-top: 18px;
      border-top: 1px dashed var(--rule);
    }
    .albums-block .albums-head {
      display: flex; align-items: baseline; justify-content: space-between;
      gap: 10px; margin-bottom: 10px;
    }
    .albums-block .albums-eyebrow {
      font-family: var(--mono); font-size: 10px; letter-spacing: 0.28em;
      text-transform: uppercase; color: var(--muted); font-weight: 700;
    }
    .albums-block .albums-meta {
      font-family: var(--mono); font-size: 11px; color: var(--muted);
    }
    .albums-target {
      display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
      padding: 9px 12px; margin-bottom: 12px;
      background: rgba(15, 107, 79, 0.07);
      border: 1px solid rgba(15, 107, 79, 0.32);
      border-radius: 2px;
      font-family: var(--mono); font-size: 11px; color: var(--ink-soft);
      letter-spacing: 0.04em;
    }
    .albums-target.new-album { background: rgba(122, 31, 18, 0.06); border-color: rgba(122, 31, 18, 0.32); color: var(--stamp-deep); }
    .albums-target strong { color: var(--ink); font-weight: 700; letter-spacing: 0.06em; }
    .albums-target .clear-target {
      border: 1px solid currentColor; background: transparent; color: inherit;
      font: inherit; padding: 4px 8px; border-radius: 2px; cursor: pointer;
      font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase;
    }
    .albums-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 10px;
    }
    .album-card {
      position: relative;
      background: var(--card-deep);
      border: 1px solid var(--line);
      border-radius: 2px;
      padding: 14px 14px 12px;
      display: grid; gap: 6px;
    }
    .album-card.empty-tile {
      background: transparent;
      border-style: dashed;
      cursor: pointer;
      align-items: center;
      justify-items: center;
      text-align: center;
      color: var(--muted);
      transition: border-color 120ms ease, color 120ms ease, background 120ms ease;
    }
    .album-card.empty-tile:hover { border-color: var(--ink); color: var(--ink); background: rgba(255,255,255,0.4); }
    .album-card.targeted { border-color: var(--green); box-shadow: 0 0 0 2px rgba(15,107,79,0.18); }
    .album-card.focused { border-color: var(--stamp); box-shadow: 0 0 0 2px rgba(122,31,18,0.18); }
    .album-card .album-name {
      font-family: var(--serif); font-size: 17px; font-weight: 500;
      letter-spacing: -0.015em; color: var(--ink);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .album-card .album-desc {
      font-family: var(--sans); font-size: 12px; color: var(--ink-soft);
      line-height: 1.35; min-height: 1em;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .album-card .album-counts {
      display: flex; flex-wrap: wrap; gap: 8px; row-gap: 4px;
      font-family: var(--mono); font-size: 10px; color: var(--muted);
      letter-spacing: 0.06em; margin-top: 2px;
    }
    .album-card .album-counts strong { color: var(--ink); font-weight: 700; }
    .album-card .album-actions {
      display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px;
    }
    .album-card .album-actions button,
    .album-card .album-actions a {
      border: 1px solid var(--ink);
      background: var(--ink);
      color: var(--card);
      font-family: var(--sans); font-weight: 700;
      font-size: 11px; letter-spacing: 0.06em;
      padding: 7px 9px; border-radius: 2px; cursor: pointer;
      text-decoration: none;
    }
    .album-card .album-actions .ghost { background: transparent; color: var(--ink); }
    .album-card .album-actions .ghost:hover { background: rgba(17, 23, 19, 0.06); }
    .album-card .album-actions .filter-on { background: var(--green); border-color: var(--green); }
    .album-card .album-id {
      font-family: var(--mono); font-size: 9px; color: var(--muted);
      letter-spacing: 0.04em;
    }

    /* ==================== progress block ==================== */
    .progress {
      margin-top: 28px;
      padding: 22px 22px 18px;
      border: 1px solid var(--line);
      background: rgba(248, 244, 232, 0.85);
      border-radius: 3px;
      display: none;
      animation: rise 240ms ease-out;
    }
    .progress.active { display: block; }
    .progress.failed { border-color: rgba(168, 50, 38, 0.45); background: rgba(168, 50, 38, 0.06); }
    @keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    .progress-head {
      display: flex; align-items: baseline; justify-content: space-between;
      gap: 12px; margin-bottom: 12px;
    }
    .progress-label {
      font-family: var(--mono); font-size: 11px; letter-spacing: 0.22em;
      text-transform: uppercase; color: var(--stamp); font-weight: 700;
    }
    .progress-pct {
      font-family: var(--serif); font-style: italic; font-weight: 400;
      font-size: 22px; color: var(--ink); line-height: 1;
    }
    .progress-bar {
      position: relative; height: 12px;
      background:
        repeating-linear-gradient(45deg, rgba(17,23,19,0.05) 0 6px, transparent 6px 12px),
        var(--card-deep);
      border: 1px solid var(--line);
      border-radius: 2px;
      overflow: hidden;
    }
    .progress-fill {
      position: absolute; inset: 0 auto 0 0;
      width: 0%;
      background: linear-gradient(90deg, var(--green), #137a5a);
      transition: width 220ms ease;
    }
    .progress-fill.indeterminate {
      width: 30% !important;
      animation: slide 1.4s ease-in-out infinite;
    }
    @keyframes slide { 0% { transform: translateX(-110%); } 100% { transform: translateX(360%); } }
    .progress-caption {
      margin-top: 10px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--ink-soft);
      letter-spacing: 0.01em;
      min-height: 1.2em;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .progress-counts {
      margin-top: 10px;
      display: flex; gap: 14px;
      font-family: var(--mono); font-size: 11px;
      color: var(--muted);
      letter-spacing: 0.04em;
    }
    .progress-counts strong { color: var(--ink); font-weight: 700; }
    .progress-counts .fail strong { color: var(--stamp); }

    /* ==================== populated state ==================== */
    .ledger { display: grid; grid-template-rows: auto auto 1fr; min-height: calc(100vh - 110px); }

    .compact-toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px; align-items: center;
      padding: 14px 28px;
      border-bottom: 1px solid var(--line);
      background: rgba(236, 231, 218, 0.72);
      position: sticky; top: 102px; z-index: 4;
    }
    .compact-toolbar .pill {
      font-family: var(--mono); font-size: 11px; letter-spacing: 0.18em;
      text-transform: uppercase; color: var(--muted);
      padding: 8px 12px; border: 1px dashed var(--rule); border-radius: 2px;
    }
    .compact-toolbar .pill strong { color: var(--ink); font-weight: 700; }
    .toolbar-btn {
      border: 1px solid var(--ink);
      background: var(--ink);
      color: var(--card);
      font-family: var(--sans);
      font-weight: 700;
      font-size: 12px;
      padding: 10px 14px;
      border-radius: 2px;
      cursor: pointer;
      letter-spacing: 0.04em;
    }
    .toolbar-btn.ghost { background: transparent; color: var(--ink); }
    .toolbar-btn.ghost:hover { background: rgba(17, 23, 19, 0.06); }
    .toolbar-btn.danger { background: var(--stamp); border-color: var(--stamp); }
    .toolbar-btn:disabled { opacity: 0.45; cursor: not-allowed; }
    .toolbar-search {
      border: 1px solid var(--line);
      background: var(--card);
      padding: 10px 12px;
      border-radius: 2px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--ink);
    }
    .toolbar-search:focus { outline: 2px solid var(--stamp); outline-offset: 1px; }
    .view-toggle { display: inline-flex; border: 1px solid var(--line); border-radius: 2px; overflow: hidden; }
    .view-toggle button {
      border: 0; background: transparent;
      font-family: var(--mono); font-size: 11px; letter-spacing: 0.18em;
      text-transform: uppercase;
      padding: 10px 12px; cursor: pointer; color: var(--muted);
    }
    .view-toggle button.active { background: var(--ink); color: var(--card); }
    .sort-select {
      border: 1px solid var(--line);
      background: var(--card);
      padding: 10px 12px;
      border-radius: 2px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--ink);
    }
    .filter-select { min-width: 190px; }
    .selection-select { min-width: 210px; }

    .layout {
      display: grid; grid-template-columns: minmax(0, 1fr) 360px;
      min-height: 0;
    }
    .canvas { padding: 24px 28px 56px; min-width: 0; }
    .canvas .progress { margin-bottom: 22px; margin-top: 0; max-width: 720px; }

    /* ledger view */
    .ledger-table {
      width: 100%; border-collapse: collapse;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 2px;
      overflow: hidden;
      font-family: var(--mono);
      font-size: 12px;
    }
    .ledger-table thead th {
      text-align: left;
      font-family: var(--mono);
      font-weight: 700;
      font-size: 10px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--muted);
      padding: 12px 12px;
      border-bottom: 1.5px solid var(--ink);
      background: var(--card-deep);
    }
    .select-cell { width: 42px; text-align: center; }
    .select-cell input { accent-color: var(--green); }
    .ledger-table tbody td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--rule);
      vertical-align: middle;
      color: var(--ink-soft);
    }
    .ledger-table tbody tr {
      cursor: pointer;
      transition: background 100ms ease;
    }
    .ledger-table tbody tr:hover { background: rgba(122, 31, 18, 0.04); }
    .ledger-table tbody tr.selected { background: rgba(15, 107, 79, 0.08); }
    .ledger-table tbody tr.selected td:first-child { box-shadow: inset 3px 0 0 var(--green); }
    .row-num { width: 56px; color: var(--muted); font-variant-numeric: tabular-nums; }
    .row-thumb { width: 64px; }
    .row-thumb img {
      width: 56px; height: 36px; object-fit: cover;
      border: 1px solid var(--rule);
      background: #17211b;
      display: block;
    }
    .row-name { font-family: var(--sans); font-weight: 500; color: var(--ink); }
    .row-status { width: 130px; }
    .status-tag {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 1px;
      font-size: 10px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-weight: 700;
    }
    .status-tag.imported { background: rgba(15, 107, 79, 0.14); color: var(--green); border: 1px solid rgba(15, 107, 79, 0.32); }
    .status-tag.regenerated_cache { background: rgba(36, 88, 166, 0.14); color: var(--blue); border: 1px solid rgba(36, 88, 166, 0.32); }
    .status-tag.already_imported { background: rgba(68, 81, 90, 0.14); color: var(--ink-soft); border: 1px solid var(--rule); }
    .status-tag.failed { background: rgba(168, 50, 38, 0.14); color: var(--red); border: 1px solid rgba(168, 50, 38, 0.32); }

    /* contact sheet view */
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 14px;
    }
    .tile {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 2px;
      overflow: hidden;
      cursor: pointer;
      transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
    }
    .tile:hover { transform: translateY(-2px); box-shadow: 0 14px 28px rgba(17, 23, 19, 0.12); border-color: rgba(15, 107, 79, 0.4); }
    .tile.selected { border-color: var(--green); box-shadow: 0 14px 28px rgba(15, 107, 79, 0.2); }
    .tile-select { position: absolute; top: 8px; left: 8px; z-index: 2; accent-color: var(--green); }
    .tile { position: relative; }
    .tile-img {
      aspect-ratio: 1.2 / 1;
      background: #17211b;
      display: grid; place-items: center;
      color: rgba(255, 250, 240, 0.62);
      overflow: hidden;
    }
    .tile-img img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .tile-meta { padding: 10px 12px 12px; display: grid; gap: 4px; }
    .tile-name {
      font-family: var(--sans); font-weight: 600; font-size: 12px;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .tile-sub {
      font-family: var(--mono); font-size: 10px; color: var(--muted);
      letter-spacing: 0.04em;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }

    /* sidebar */
    aside.detail {
      border-left: 1px solid var(--line);
      background: rgba(248, 244, 232, 0.86);
      padding: 26px 24px;
      position: sticky; top: 168px;
      height: calc(100vh - 168px);
      overflow: auto;
    }
    .detail-empty {
      font-family: var(--serif); font-style: italic; font-weight: 300;
      color: var(--muted); font-size: 18px; line-height: 1.4;
      padding: 30px 0;
    }
    .detail-empty::before {
      content: "—";
      display: block; font-size: 36px; color: var(--rule);
      margin-bottom: 14px;
    }
    .detail-eyebrow {
      font-family: var(--mono); font-size: 10px; letter-spacing: 0.28em;
      text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
    }
    .detail-title {
      font-family: var(--serif); font-weight: 500; font-size: 22px;
      letter-spacing: -0.02em; margin: 0 0 4px; word-break: break-word;
    }
    .detail-key {
      font-family: var(--mono); font-size: 11px; color: var(--stamp);
      letter-spacing: 0.04em; margin-bottom: 18px;
    }
    .detail-thumb {
      width: 100%;
      aspect-ratio: 1.2 / 1;
      background: #17211b;
      border: 1px solid var(--line);
      margin-bottom: 18px;
      display: grid; place-items: center;
    }
    .detail-thumb img { width: 100%; height: 100%; object-fit: contain; }
    .detail-actions { display: grid; gap: 8px; margin: 0 0 18px; }
    .detail-action {
      border: 1px solid var(--green);
      background: var(--green);
      color: var(--card);
      border-radius: 2px;
      padding: 11px 12px;
      font-family: var(--mono);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      cursor: pointer;
    }
    .detail-action:hover { filter: brightness(0.94); }
    .kv {
      display: grid; grid-template-columns: 110px 1fr;
      gap: 8px 12px; font-size: 12px;
    }
    .kv dt {
      font-family: var(--mono); font-size: 10px; letter-spacing: 0.16em;
      text-transform: uppercase; color: var(--muted); margin: 0;
    }
    .kv dd {
      margin: 0; color: var(--ink-soft);
      font-family: var(--mono); font-size: 11px;
      word-break: break-word;
    }
    .kv dd.value-strong { font-family: var(--sans); font-size: 13px; color: var(--ink); font-weight: 500; }

    /* sidebar — album dossier mode */
    .album-dossier { position: relative; }
    .album-dossier::before {
      content: "DOSSIER";
      position: absolute; top: -10px; right: -4px;
      font-family: var(--mono); font-size: 9px; letter-spacing: 0.36em;
      color: var(--stamp); border: 1.5px solid var(--stamp);
      padding: 4px 8px; transform: rotate(4deg); opacity: 0.8;
      border-radius: 3px; background: rgba(248, 244, 232, 0.95);
    }
    .album-dossier .clear-focus {
      position: absolute; top: -2px; left: -2px;
      border: 0; background: transparent; color: var(--muted);
      font-family: var(--mono); font-size: 10px; letter-spacing: 0.18em;
      cursor: pointer; padding: 4px 6px;
    }
    .album-dossier .clear-focus:hover { color: var(--ink); }
    .album-dossier .detail-eyebrow { margin-top: 14px; }
    .album-dossier h3.detail-title { padding-right: 12px; }
    .album-meta-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 10px 14px;
      padding: 14px 0;
      border-top: 1px solid var(--rule);
      border-bottom: 1px solid var(--rule);
      margin: 14px 0 16px;
    }
    .album-meta-grid .cell {
      display: grid; gap: 3px;
    }
    .album-meta-grid .cell .label {
      font-family: var(--mono); font-size: 9px; letter-spacing: 0.22em;
      text-transform: uppercase; color: var(--muted);
    }
    .album-meta-grid .cell .value {
      font-family: var(--mono); font-size: 11px; color: var(--ink-soft);
      word-break: break-word;
    }
    .album-meta-grid .cell .value strong { color: var(--ink); }
    .album-meta-grid .cell.span2 { grid-column: 1 / -1; }
    .album-counts-block {
      margin: 0 0 18px;
      border: 1px solid var(--rule);
      background: rgba(255, 255, 255, 0.4);
      padding: 12px 14px;
      border-radius: 2px;
    }
    .album-counts-block .header {
      font-family: var(--mono); font-size: 10px; letter-spacing: 0.22em;
      text-transform: uppercase; color: var(--muted);
      margin-bottom: 10px;
      display: flex; justify-content: space-between;
    }
    .album-counts-block .row {
      display: grid; grid-template-columns: 1fr auto;
      gap: 6px 10px; font-family: var(--mono); font-size: 11px;
      padding: 3px 0; color: var(--ink-soft);
    }
    .album-counts-block .row.zero { color: var(--muted); }
    .album-counts-block .row strong { color: var(--ink); font-weight: 700; }
    .album-rename-form {
      display: grid; gap: 10px;
      padding: 14px;
      border: 1px solid var(--ink);
      background: var(--card);
      margin: 0 0 18px;
      border-radius: 2px;
      box-shadow: 0 6px 18px rgba(17, 23, 19, 0.08);
    }
    .album-rename-form label {
      display: grid; gap: 5px;
      font-family: var(--mono); font-size: 10px; letter-spacing: 0.22em;
      text-transform: uppercase; color: var(--muted);
    }
    .album-rename-form input,
    .album-rename-form textarea {
      width: 100%; border: 1px solid var(--line); border-radius: 2px;
      background: rgba(255, 255, 255, 0.7); padding: 9px 10px;
      color: var(--ink); font: 13px var(--mono);
    }
    .album-rename-form textarea { min-height: 64px; resize: vertical; }
    .album-rename-form .row {
      display: flex; gap: 8px; justify-content: flex-end; align-items: center;
    }
    .album-rename-form button {
      border: 1px solid var(--ink);
      background: var(--ink); color: var(--card);
      padding: 9px 12px; border-radius: 2px; cursor: pointer;
      font-family: var(--mono); font-size: 11px; letter-spacing: 0.16em;
      text-transform: uppercase; font-weight: 700;
    }
    .album-rename-form button.ghost { background: transparent; color: var(--ink); }
    .album-rename-form button.ghost:hover { background: rgba(17, 23, 19, 0.06); }
    .album-rename-form .err {
      font-family: var(--mono); font-size: 11px; color: var(--red);
    }

    /* inline album-naming card (replaces window.prompt) */
    .modal-backdrop.naming-backdrop {
      position: fixed; inset: 0; z-index: 60;
      background: rgba(17, 23, 19, 0.34);
      display: grid; place-items: center;
      padding: 24px;
      animation: fadeIn 160ms ease-out;
    }
    @keyframes fadeIn { from { opacity: 0 } to { opacity: 1 } }
    .naming-card {
      width: min(520px, 100%);
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 38px 42px 32px;
      position: relative;
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.6) inset,
        0 28px 64px rgba(17, 23, 19, 0.32);
      animation: rise 220ms ease-out;
    }
    .naming-card::before {
      content: ""; position: absolute; inset: 12px;
      border: 1px solid var(--rule); pointer-events: none; border-radius: 1px;
    }
    .naming-card::after {
      content: "FORM 01-A.2 — NEW ALBUM";
      position: absolute; top: 22px; left: 50%; transform: translateX(-50%);
      font-family: var(--mono); font-size: 10px; letter-spacing: 0.32em;
      color: var(--muted);
    }
    .naming-card .stamp {
      position: absolute; top: 30px; right: 26px;
      font-family: var(--mono); font-size: 9px; letter-spacing: 0.22em;
      color: var(--stamp); border: 1.5px solid var(--stamp);
      padding: 5px 8px; transform: rotate(6deg); opacity: 0.78;
      border-radius: 3px;
    }
    .naming-card .eyebrow {
      font-family: var(--mono); font-size: 11px; letter-spacing: 0.32em;
      text-transform: uppercase; color: var(--muted); margin: 22px 0 6px;
    }
    .naming-card h3 {
      font-family: var(--serif); font-weight: 400; font-size: 32px;
      letter-spacing: -0.02em; margin: 0 0 6px;
    }
    .naming-card h3 em { font-style: italic; color: var(--stamp); }
    .naming-card .lede {
      color: var(--muted); font-size: 13px; line-height: 1.55; margin: 0 0 18px;
      max-width: 46ch;
    }
    .naming-card label {
      display: grid; gap: 5px; margin: 10px 0;
      font-family: var(--mono); font-size: 10px; letter-spacing: 0.24em;
      text-transform: uppercase; color: var(--muted);
    }
    .naming-card label input,
    .naming-card label textarea {
      width: 100%; border: 1px solid var(--line); border-radius: 2px;
      background: rgba(255, 255, 255, 0.6); padding: 10px 12px;
      color: var(--ink); font: 13px var(--mono);
    }
    .naming-card label textarea { min-height: 60px; resize: vertical; }
    .naming-card .pick-row {
      display: grid; grid-template-columns: 1fr 1fr;
      gap: 12px; margin-top: 18px;
    }
    .naming-card .pick-row button {
      border: 1.5px solid var(--ink);
      background: var(--ink); color: var(--card);
      padding: 14px 14px 12px; border-radius: 2px; cursor: pointer;
      text-align: left; display: grid; gap: 4px;
      font-family: var(--sans);
    }
    .naming-card .pick-row button.secondary { background: transparent; color: var(--ink); }
    .naming-card .pick-row button .glyph {
      font-family: var(--serif); font-style: italic; font-size: 18px; font-weight: 300; opacity: 0.9;
    }
    .naming-card .pick-row button .label { font-size: 12px; font-weight: 700; letter-spacing: 0.04em; }
    .naming-card .pick-row button .hint { font-size: 10px; opacity: 0.7; font-family: var(--mono); letter-spacing: 0.06em; }
    .naming-card .footer-row {
      display: flex; justify-content: space-between; align-items: center;
      margin-top: 18px; padding-top: 14px; border-top: 1px dashed var(--rule);
      font-family: var(--mono); font-size: 11px; color: var(--muted);
    }
    .naming-card .footer-row button.cancel {
      border: 0; background: transparent; color: var(--muted);
      font-family: var(--mono); font-size: 11px; letter-spacing: 0.18em;
      cursor: pointer; padding: 6px 8px;
    }
    .naming-card .footer-row button.cancel:hover { color: var(--ink); }
    .naming-card .err {
      font-family: var(--mono); font-size: 11px; color: var(--red);
      margin-top: 6px;
    }

    .toast {
      position: fixed; bottom: 24px; left: 50%;
      transform: translateX(-50%) translateY(20px);
      background: var(--ink); color: var(--card);
      font-family: var(--mono); font-size: 12px;
      padding: 10px 16px; border-radius: 2px;
      box-shadow: 0 10px 24px rgba(17, 23, 19, 0.3);
      opacity: 0; transition: all 200ms ease;
      pointer-events: none; z-index: 50;
      letter-spacing: 0.04em;
    }
    .toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
    .toast.error { background: var(--stamp); }

    .modal-backdrop {
      position: fixed; inset: 0;
      background: rgba(17, 23, 19, 0.48);
      display: grid; place-items: center;
      z-index: 80;
      padding: 24px;
    }
    .conflict-modal {
      width: min(760px, 100%);
      background: var(--card);
      border: 1px solid var(--ink);
      box-shadow: 0 30px 90px rgba(17, 23, 19, 0.36);
      padding: 26px;
      display: grid;
      gap: 18px;
    }
    .conflict-modal h3 {
      margin: 0;
      font-family: var(--serif);
      font-size: 30px;
      font-weight: 400;
      letter-spacing: -0.03em;
    }
    .conflict-modal p { margin: 0; color: var(--muted); line-height: 1.45; }
    .conflict-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .conflict-card { border: 1px solid var(--line); background: var(--card-deep); padding: 12px; display: grid; gap: 10px; }
    .conflict-card img { width: 100%; aspect-ratio: 1.4 / 1; object-fit: contain; background: #17211b; border: 1px solid var(--line); }
    .conflict-label { font-family: var(--mono); font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted); }
    .conflict-name { font-family: var(--mono); font-size: 12px; word-break: break-word; color: var(--ink); }
    .conflict-actions { display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
    .conflict-actions button { border-radius: 2px; font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; }
    .conflict-actions .danger { background: var(--stamp); border-color: var(--stamp); color: var(--card); }
    .conflict-actions .primary { background: var(--green); border-color: var(--green); color: var(--card); }

    @media (max-width: 980px) {
      header, .compact-toolbar { position: static; grid-template-columns: 1fr; }
      .rail { justify-content: flex-start; }
      .layout { grid-template-columns: 1fr; }
      aside.detail { position: static; height: auto; border-left: 0; border-top: 1px solid var(--line); }
      .intake-card { padding: 36px 28px 28px; }
      .intake-actions { grid-template-columns: 1fr; }
      .album-fields { grid-template-columns: 1fr; }
      .conflict-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <input type="file" id="fileInput" multiple accept=".png,.jpg,.jpeg,.tif,.tiff" style="display:none">
  <input type="file" id="folderInput" webkitdirectory directory multiple style="display:none">

  <div class="shell">
    <header>
      <div class="titleblock">
        <div class="filenum">01<sup>/07</sup></div>
        <div>
          <h1>Intake <em>&amp;</em> Cache</h1>
          <p class="subtitle">Catalog source imagery, render browser-ready RGB cache, and seed the manual review queue. Source files are never modified.</p>
        </div>
      </div>
      <nav class="rail" aria-label="Training pipeline stages">
        <a class="stage active" href="/import">Import Data</a>
        <a class="stage" href="__MANUAL_REVIEW_HREF__">Manual Review</a>
        <a class="stage" href="/bbox-overview">BBox Overview</a>
        <a class="stage" href="/dataset-selection">Dataset Selection</a>
        <a class="stage" href="/training-setup">Training Setup</a>
        <a class="stage" href="/train-yolo">Train YOLO</a>
        <span class="stage">Evaluate</span>
      </nav>
    </header>

    <main id="main"></main>
  </div>

  <div class="toast" id="toast"></div>

  <script>
    const stateKey = 'opendarksts.import.state.v1';
    const POLL_MS = 700;

    const state = {
      manifest: null,
      albums: [],
      targetAlbumId: null,
      focusedAlbumId: null,
      albumEditing: false,
      albumDraftError: null,
      job: null,
      pollTimer: null,
      view: 'ledger',
      sort: 'imported_desc',
      filter: 'all',
      search: '',
      selectedKey: null,
      selectedKeys: [],
      lastSelectedKey: null,
      selectionPersistenceWarned: false,
      forceRefresh: false,
      albumName: '',
      albumDescription: '',
      sourceDisplayName: '',
    };

    function restoreState() {
      try {
        const raw = JSON.parse(localStorage.getItem(stateKey) || '{}');
        if (raw.view === 'grid' || raw.view === 'ledger') state.view = raw.view;
        if (typeof raw.sort === 'string') state.sort = raw.sort;
        if (typeof raw.filter === 'string') state.filter = raw.filter;
        if (typeof raw.search === 'string') state.search = raw.search;
        if (typeof raw.selectedKey === 'string') state.selectedKey = raw.selectedKey;
        if (Array.isArray(raw.selectedKeys)) state.selectedKeys = raw.selectedKeys.filter(x => typeof x === 'string');
        if (typeof raw.lastSelectedKey === 'string') state.lastSelectedKey = raw.lastSelectedKey;
        if (typeof raw.forceRefresh === 'boolean') state.forceRefresh = raw.forceRefresh;
        if (typeof raw.albumName === 'string') state.albumName = raw.albumName;
        if (typeof raw.albumDescription === 'string') state.albumDescription = raw.albumDescription;
        if (typeof raw.sourceDisplayName === 'string') state.sourceDisplayName = raw.sourceDisplayName;
        if (typeof raw.targetAlbumId === 'string') state.targetAlbumId = raw.targetAlbumId;
        if (typeof raw.focusedAlbumId === 'string') state.focusedAlbumId = raw.focusedAlbumId;
      } catch (e) {}
    }
    function persistState() {
      const persistSelection = state.selectedKeys.length <= 50000;
      localStorage.setItem(stateKey, JSON.stringify({
        view: state.view, sort: state.sort, filter: state.filter, search: state.search,
        selectedKey: state.selectedKey, selectedKeys: persistSelection ? state.selectedKeys : [], lastSelectedKey: state.lastSelectedKey, forceRefresh: state.forceRefresh,
        selectionPersistenceDropped: !persistSelection,
        albumName: state.albumName, albumDescription: state.albumDescription,
        sourceDisplayName: state.sourceDisplayName,
        targetAlbumId: state.targetAlbumId,
        focusedAlbumId: state.focusedAlbumId,
      }));
    }

    function escape(value) {
      return String(value == null ? '' : value).replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      })[ch]);
    }
    function formatBytes(n) {
      if (!n && n !== 0) return '-';
      const units = ['B', 'KB', 'MB', 'GB'];
      let i = 0; let v = n;
      while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
      return v.toFixed(v >= 100 || i === 0 ? 0 : 1) + ' ' + units[i];
    }
    function formatTime(iso) {
      if (!iso) return '-';
      try {
        const d = new Date(iso);
        return d.toISOString().replace('T', ' ').replace('Z', 'Z').slice(0, 19) + 'Z';
      } catch (e) { return iso; }
    }
    function showToast(message, kind) {
      const toast = document.getElementById('toast');
      toast.textContent = message;
      toast.classList.toggle('error', kind === 'error');
      toast.classList.add('show');
      clearTimeout(showToast._t);
      showToast._t = setTimeout(() => toast.classList.remove('show'), 2400);
    }

    function logAction(action, details) {
      const payload = {
        action,
        timestamp: new Date().toISOString(),
        page: 'import',
        filter: state.filter,
        sort: state.sort,
        search: state.search,
        selected_count: state.selectedKeys.length,
        selected_key: state.selectedKey,
        details: details || {},
      };
      const body = JSON.stringify(payload);
      try {
        if (navigator.sendBeacon) {
          navigator.sendBeacon('/api/training-ui-interaction', new Blob([body], { type: 'application/json' }));
          return;
        }
      } catch (e) {}
      fetch('/api/training-ui-interaction', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }).catch(() => {});
    }

    function uploadNameForFile(file) {
      return file.webkitRelativePath || file.name;
    }

    function duplicateUploadName(name, usedNames) {
      const slash = name.lastIndexOf('/');
      const dir = slash >= 0 ? name.slice(0, slash + 1) : '';
      const filename = slash >= 0 ? name.slice(slash + 1) : name;
      const dot = filename.lastIndexOf('.');
      const stem = dot > 0 ? filename.slice(0, dot) : filename;
      const suffix = dot > 0 ? filename.slice(dot) : '';
      let index = 1;
      let candidate = `${dir}${stem} (${index})${suffix}`;
      while (usedNames.has(candidate)) {
        index += 1;
        candidate = `${dir}${stem} (${index})${suffix}`;
      }
      return candidate;
    }

    function filePreviewUrl(file) {
      if (!file || !file.type || !file.type.startsWith('image/')) return '';
      return URL.createObjectURL(file);
    }

    function resolveUploadConflict(existing, incoming, conflictNumber) {
      return new Promise(resolve => {
        const oldUrl = filePreviewUrl(existing.file);
        const newUrl = filePreviewUrl(incoming.file);
        const backdrop = document.createElement('div');
        backdrop.className = 'modal-backdrop';
        backdrop.innerHTML = `
          <section class="conflict-modal" role="dialog" aria-modal="true" aria-labelledby="conflictTitle">
            <div>
              <p class="detail-eyebrow">Upload conflict ${conflictNumber}</p>
              <h3 id="conflictTitle">Duplicate filename found</h3>
            </div>
            <p>The selected upload contains another file named <strong>${escape(incoming.uploadName)}</strong>. Choose how to stage this incoming file. This does not modify your source files.</p>
            <div class="conflict-grid">
              <article class="conflict-card">
                <div class="conflict-label">Already staged</div>
                ${oldUrl ? `<img src="${oldUrl}" alt="">` : '<div class="detail-thumb">No preview</div>'}
                <div class="conflict-name">${escape(existing.uploadName)}</div>
              </article>
              <article class="conflict-card">
                <div class="conflict-label">Incoming file</div>
                ${newUrl ? `<img src="${newUrl}" alt="">` : '<div class="detail-thumb">No preview</div>'}
                <div class="conflict-name">${escape(incoming.uploadName)}</div>
              </article>
            </div>
            <div class="conflict-actions">
              <button id="conflictSkip">Skip</button>
              <button class="danger" id="conflictReplace">Replace</button>
              <button class="primary" id="conflictDuplicate">Duplicate</button>
            </div>
          </section>
        `;
        const cleanup = (choice) => {
          if (oldUrl) URL.revokeObjectURL(oldUrl);
          if (newUrl) URL.revokeObjectURL(newUrl);
          backdrop.remove();
          logAction('upload_duplicate_choice', { choice, upload_name: incoming.uploadName });
          resolve(choice);
        };
        document.body.appendChild(backdrop);
        backdrop.querySelector('#conflictSkip').onclick = () => cleanup('skip');
        backdrop.querySelector('#conflictReplace').onclick = () => cleanup('replace');
        backdrop.querySelector('#conflictDuplicate').onclick = () => cleanup('duplicate');
      });
    }

    async function resolveDuplicateUploads(fileList) {
      const accepted = [];
      const positions = new Map();
      let conflictNumber = 0;
      for (const file of Array.from(fileList || [])) {
        const uploadName = uploadNameForFile(file);
        const incoming = { file, uploadName };
        if (!positions.has(uploadName)) {
          positions.set(uploadName, accepted.length);
          accepted.push(incoming);
          continue;
        }
        conflictNumber += 1;
        const existingIndex = positions.get(uploadName);
        const choice = await resolveUploadConflict(accepted[existingIndex], incoming, conflictNumber);
        if (choice === 'skip') continue;
        if (choice === 'replace') {
          accepted[existingIndex] = incoming;
          continue;
        }
        if (choice === 'duplicate') {
          const duplicateName = duplicateUploadName(uploadName, positions);
          positions.set(duplicateName, accepted.length);
          accepted.push({ file, uploadName: duplicateName });
        }
      }
      return accepted;
    }

    async function loadAlbums() {
      try {
        const resp = await fetch('/api/albums');
        if (!resp.ok) return;
        const data = await resp.json();
        state.albums = Array.isArray(data.albums) ? data.albums : [];
      } catch (e) {
        state.albums = [];
      }
      if (state.targetAlbumId && !state.albums.some(a => a.album_id === state.targetAlbumId)) {
        state.targetAlbumId = null;
        persistState();
      }
      if (state.focusedAlbumId && !state.albums.some(a => a.album_id === state.focusedAlbumId)) {
        state.focusedAlbumId = null;
        state.albumEditing = false;
        persistState();
      }
    }

    async function loadManifest() {
      await loadAlbums();
      const resp = await fetch('/api/import');
      state.manifest = await resp.json();
      const known = new Set((state.manifest.images || []).map(im => im.image_key));
      state.selectedKeys = state.selectedKeys.filter(key => known.has(key));
      if (state.selectedKey && !known.has(state.selectedKey)) state.selectedKey = null;
      render();
    }

    function render() {
      const main = document.getElementById('main');
      const empty = !state.manifest || state.manifest.image_count === 0;
      const jobActive = state.job && (state.job.state === 'queued' || state.job.state === 'importing');
      if (empty && !jobActive && !(state.job && state.job.state === 'failed')) {
        main.innerHTML = renderEmptyState();
        wireIntakeControls();
      } else if (empty) {
        main.innerHTML = renderEmptyState();
        wireIntakeControls();
        renderProgressInto(document.querySelector('.intake-card'));
      } else {
        main.innerHTML = renderPopulatedState();
        wireToolbarControls();
        renderProgressInto(document.querySelector('.canvas'), { atTop: true });
      }
    }

    function targetAlbum() {
      if (!state.targetAlbumId) return null;
      return state.albums.find(a => a.album_id === state.targetAlbumId) || null;
    }

    function renderTargetBar() {
      const target = targetAlbum();
      if (target) {
        return `
          <div class="albums-target">
            <span>Target:</span>
            <strong>${escape(target.album_name || target.album_id)}</strong>
            <span>(append)</span>
            <span class="album-id">${escape(target.album_id)}</span>
            <button class="clear-target" data-action="clear-target">New album instead</button>
          </div>
        `;
      }
      const name = state.albumName.trim();
      return `
        <div class="albums-target new-album">
          <span>Target:</span>
          <strong>New album</strong>
          <span>${name ? '→ ' + escape(name) : '(name required at upload time)'}</span>
        </div>
      `;
    }

    function renderAlbumCard(album) {
      const isTarget = album.album_id === state.targetAlbumId;
      const isFocused = album.album_id === state.focusedAlbumId;
      const filterValue = 'album:' + album.album_id;
      const isFilterOn = state.filter === filterValue;
      const reviewLine = album.review_counts || {};
      const updated = album.updated_at ? formatTime(album.updated_at).slice(0, 10) : '';
      return `
        <div class="album-card ${isTarget ? 'targeted' : ''} ${isFocused ? 'focused' : ''}" data-album-id="${escape(album.album_id)}">
          <div data-action="focus-album" data-album-id="${escape(album.album_id)}" style="cursor:pointer;display:grid;gap:6px;">
            <div class="album-name">${escape(album.album_name || album.album_id)}</div>
            <div class="album-desc">${escape(album.album_description || '')}</div>
            <div class="album-counts">
              <span><strong>${album.image_count || 0}</strong> img</span>
              <span><strong>${reviewLine.review_completed || 0}</strong>/${reviewLine.review_total || 0} reviewed</span>
              <span>${escape(updated || '—')}</span>
              <span class="album-id">${escape(album.album_id)}</span>
            </div>
          </div>
          <div class="album-actions">
            <button data-action="add-to-album" data-album-id="${escape(album.album_id)}">Add Photos</button>
            <a href="/manual-review?album_id=${encodeURIComponent(album.album_id)}" data-action="review-album" data-album-id="${escape(album.album_id)}">Review</a>
            <button class="ghost ${isFilterOn ? 'filter-on' : ''}" data-action="filter-album" data-album-id="${escape(album.album_id)}">${isFilterOn ? 'Clear filter' : 'Filter'}</button>
          </div>
        </div>
      `;
    }

    function renderAlbumsSection() {
      const albums = state.albums || [];
      const cards = albums.map(renderAlbumCard).join('');
      const newTile = `
        <div class="album-card empty-tile" data-action="create-album" role="button" tabindex="0">
          <div class="album-name" style="color:inherit;">+ New empty album</div>
          <div class="album-desc">Create an album you can fill later.</div>
        </div>
      `;
      return `
        <section class="albums-block">
          <div class="albums-head">
            <span class="albums-eyebrow">Albums</span>
            <span class="albums-meta">${albums.length} album${albums.length === 1 ? '' : 's'}</span>
          </div>
          ${renderTargetBar()}
          <div class="albums-grid">
            ${newTile}
            ${cards}
          </div>
        </section>
      `;
    }

    function renderEmptyState() {
      const target = targetAlbum();
      const primaryLabel = target ? 'Add Photos To Album' : 'Pick Images';
      const primaryHint = target ? 'append to ' + (target.album_name || target.album_id) : '.png .jpg .tif — one or many';
      return `
        <div class="intake">
          <div class="intake-card">
            <div class="intake-stamp">UNFILED</div>
            <p class="intake-eyebrow">Step 01 / Training Pipeline</p>
            <h2 class="intake-title">Bring your <em>imagery</em> into the workspace.</h2>
            <p class="intake-lede">Pick image files or a folder. We will compute a stable <code>image_key</code>, render a browser-friendly cache, and create review stubs so manual review is ready immediately.</p>
            <div class="intake-actions">
              <button class="pick-btn" id="pickFiles">
                <span class="glyph">⌥</span>
                <span class="label">${escape(primaryLabel)}</span>
                <span class="hint">${escape(primaryHint)}</span>
              </button>
              <button class="pick-btn secondary" id="pickFolder">
                <span class="glyph">▢</span>
                <span class="label">${target ? 'Add Folder To Album' : 'Pick Folder'}</span>
                <span class="hint">recursive scan, hidden skipped</span>
              </button>
            </div>
            ${target ? '' : `<div class="intake-actions" style="grid-template-columns:1fr;margin-top:-12px;"><button class="pick-btn secondary" id="newAlbumImport"><span class="glyph">＋</span><span class="label">Import to a NEW album…</span><span class="hint">name it first, then choose files or folder</span></button></div>`}
            ${target ? '' : renderAlbumFields()}
            ${renderAlbumsSection()}
            <div class="intake-options">
              <label><input type="checkbox" id="forceRefresh" ${state.forceRefresh ? 'checked' : ''}> Force re-render existing cache</label>
            </div>
            <p class="intake-footer">For very large batches, prefer the CLI:<br><code>.venv/bin/python scripts/training_pipeline_cli.py import-data --source &lt;folder&gt; --recursive</code></p>
          </div>
        </div>
      `;
    }

    function renderPopulatedState() {
      const m = state.manifest;
      const sc = m.status_counts || {};
      const selectedShown = filteredImages().filter(im => state.selectedKeys.includes(im.image_key)).length;
      return `
        <div class="ledger">
          <div class="compact-toolbar">
            <button class="toolbar-btn" id="pickFiles">+ Pick Images</button>
            <button class="toolbar-btn ghost" id="pickFolder">+ Pick Folder</button>
            <button class="toolbar-btn ghost" id="newAlbumImport" title="Name a new album, then pick files or folder">+ New Album…</button>
            <span class="pill"><strong>${m.image_count}</strong> images · ${(sc.imported || 0)}+${(sc.regenerated_cache || 0)}+${(sc.already_imported || 0)} imported · ${(sc.review_only || 0)} review-only</span>
            <span class="pill"><strong>${state.selectedKeys.length}</strong> selected · ${selectedShown} shown</span>
            <select class="sort-select filter-select" id="filter">
              ${renderFilterOptions()}
            </select>
            <select class="sort-select selection-select" id="selectionAction">
              <option value="">Selection actions</option>
              <option value="select_matching">Select all matching filter</option>
              <option value="invert_filtered">Invert current filtered selection</option>
              <option value="clear">Clear selection</option>
            </select>
            <button class="toolbar-btn danger" id="removeSelected" ${state.selectedKeys.length ? '' : 'disabled'}>Remove From Workspace</button>
            <button class="toolbar-btn ghost" id="removeFromAlbum" ${state.selectedKeys.length && state.filter.startsWith('album:') ? '' : 'disabled'} title="Remove the selected images from the active album filter only">Remove From Album</button>
            <input class="toolbar-search" id="albumNameToolbar" placeholder="new album name" value="${escape(state.albumName)}" ${targetAlbum() ? 'disabled' : ''}>
            <input class="toolbar-search" id="search" placeholder="search filename, key, status" value="${escape(state.search)}">
            <select class="sort-select" id="sort">
              <option value="imported_desc" ${state.sort === 'imported_desc' ? 'selected' : ''}>Newest first</option>
              <option value="imported_asc" ${state.sort === 'imported_asc' ? 'selected' : ''}>Oldest first</option>
              <option value="name_asc" ${state.sort === 'name_asc' ? 'selected' : ''}>Filename A→Z</option>
              <option value="size_desc" ${state.sort === 'size_desc' ? 'selected' : ''}>Largest first</option>
            </select>
            <div class="view-toggle">
              <button class="${state.view === 'ledger' ? 'active' : ''}" id="viewLedger">Ledger</button>
              <button class="${state.view === 'grid' ? 'active' : ''}" id="viewGrid">Sheet</button>
            </div>
            <label style="font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:0.06em;display:inline-flex;gap:6px;align-items:center;">
              <input type="checkbox" id="forceRefresh" ${state.forceRefresh ? 'checked' : ''}> force
            </label>
          </div>
          <div class="layout">
            <section class="canvas" id="canvas">
              <div class="albums-canvas-wrap" style="margin-bottom:18px;">${renderAlbumsSection()}</div>
              ${state.view === 'ledger' ? renderLedger() : renderGrid()}
            </section>
            <aside class="detail" id="detail">${renderDetail()}</aside>
          </div>
        </div>
      `;
    }

    function renderFilterOptions() {
      const albums = state.manifest?.albums || [];
      const options = [
        ['all', 'Filter: All images'],
        ['unreviewed', 'Filter: Unreviewed images'],
        ['reviewed', 'Filter: Reviewed images'],
        ['partially_reviewed', 'Filter: Partially reviewed images'],
      ];
      if (albums.length) {
        options.push(['no_album', 'Filter: No album']);
        for (const album of albums) {
          options.push(['album:' + album.album_id, 'Filter: Album: ' + (album.album_name || album.album_id)]);
        }
      }
      if (!options.some(([value]) => value === state.filter)) state.filter = 'all';
      return options.map(([value, label]) => `<option value="${escape(value)}" ${state.filter === value ? 'selected' : ''}>${escape(label)}</option>`).join('');
    }

    function renderAlbumFields() {
      return `
        <div class="album-fields">
          <label>Album name<input id="albumName" value="${escape(state.albumName)}" placeholder="defaults from selected folder"></label>
          <label>Source label<input id="sourceDisplayName" value="${escape(state.sourceDisplayName)}" placeholder="e.g. Singapore / 2025"></label>
          <label class="wide">Description<input id="albumDescription" value="${escape(state.albumDescription)}" placeholder="optional notes for this album"></label>
        </div>
      `;
    }

    function filteredImages() {
      const images = (state.manifest?.images || []).slice();
      let filtered = images;
      if (state.filter === 'unreviewed') {
        filtered = filtered.filter(im => im.unreviewed_archivable === true);
      } else if (state.filter === 'reviewed') {
        filtered = filtered.filter(im => im.review_queue_filter === 'reviewed');
      } else if (state.filter === 'partially_reviewed') {
        filtered = filtered.filter(im => im.review_queue_filter === 'partially_reviewed');
      } else if (state.filter === 'no_album') {
        filtered = filtered.filter(im => !im.album_ids || im.album_ids.length === 0);
      } else if (state.filter.startsWith('album:')) {
        const albumId = state.filter.slice('album:'.length);
        filtered = filtered.filter(im => (im.album_ids || []).includes(albumId));
      }
      const q = state.search.trim().toLowerCase();
      filtered = q ? filtered.filter(im => {
        const hay = [
          im.image_key, im.original_filename, im.import_status, im.source_collection,
          im.source_path, im.source_display_name, im.review_queue_filter,
          ...(im.album_names || []), ...(im.album_ids || [])
        ].join(' ').toLowerCase();
        return hay.includes(q);
      }) : filtered;
      const sortKey = state.sort;
      filtered.sort((a, b) => {
        if (sortKey === 'imported_asc') return String(a.imported_at || '').localeCompare(String(b.imported_at || ''));
        if (sortKey === 'name_asc') return String(a.original_filename || '').localeCompare(String(b.original_filename || ''));
        if (sortKey === 'size_desc') return Number(b.source_size_bytes || 0) - Number(a.source_size_bytes || 0);
        return String(b.imported_at || '').localeCompare(String(a.imported_at || ''));
      });
      return filtered;
    }

    function renderLedger() {
      const images = filteredImages();
      if (!images.length) return '<p style="font-family:var(--serif);font-style:italic;color:var(--muted);padding:40px 0;">No imported images match the current search.</p>';
      const selected = new Set(state.selectedKeys);
      const rows = images.map((im, idx) => `
        <tr class="${im.image_key === state.selectedKey ? 'selected' : ''}" onclick="selectImage('${escape(im.image_key)}')">
          <td class="select-cell"><input type="checkbox" ${selected.has(im.image_key) ? 'checked' : ''} onclick="toggleImageSelection('${escape(im.image_key)}', ${idx}, event)"></td>
          <td class="row-num">${String(idx + 1).padStart(4, '0')}</td>
          <td class="row-thumb">${im.cache_url ? `<img loading="lazy" src="${escape(im.cache_url)}" alt="">` : ''}</td>
          <td class="row-name">${escape(im.original_filename || im.image_key)}</td>
          <td>${im.width || '?'}×${im.height || '?'}</td>
          <td>${formatBytes(im.source_size_bytes)}</td>
          <td>${escape(im.render_profile_kind || '-')}</td>
          <td class="row-status"><span class="status-tag ${escape(im.import_status)}">${escape(im.import_status)}</span></td>
          <td>${formatTime(im.imported_at).slice(0, 19)}</td>
        </tr>
      `).join('');
      return `
        <table class="ledger-table">
          <thead>
            <tr>
              <th class="select-cell"></th>
              <th class="row-num">№</th>
              <th></th>
              <th>Filename</th>
              <th>Dims</th>
              <th>Size</th>
              <th>Profile</th>
              <th>Status</th>
              <th>Imported</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      `;
    }

    function renderGrid() {
      const images = filteredImages();
      if (!images.length) return '<p style="font-family:var(--serif);font-style:italic;color:var(--muted);padding:40px 0;">No imported images match the current search.</p>';
      const selected = new Set(state.selectedKeys);
      return '<div class="grid">' + images.map(im => `
        <article class="tile ${im.image_key === state.selectedKey ? 'selected' : ''}" onclick="selectImage('${escape(im.image_key)}')">
          <input class="tile-select" type="checkbox" ${selected.has(im.image_key) ? 'checked' : ''} onclick="toggleImageSelection('${escape(im.image_key)}', ${images.indexOf(im)}, event)">
          <div class="tile-img">${im.cache_url ? `<img loading="lazy" src="${escape(im.cache_url)}" alt="">` : 'no cache'}</div>
          <div class="tile-meta">
            <div class="tile-name">${escape(im.original_filename || im.image_key)}</div>
            <div class="tile-sub">${im.width || '?'}×${im.height || '?'} · ${escape(im.import_status)}${im.unreviewed_archivable ? ' · unreviewed' : ''}</div>
          </div>
        </article>
      `).join('') + '</div>';
    }

    function focusedAlbum() {
      if (!state.focusedAlbumId) return null;
      return state.albums.find(a => a.album_id === state.focusedAlbumId) || null;
    }

    function renderAlbumDossier(album) {
      const counts = album.counts || {};
      const review = album.review_counts || {};
      if (state.albumEditing) {
        const draftName = state.albumDraftName ?? (album.album_name || '');
        const draftDesc = state.albumDraftDesc ?? (album.album_description || '');
        const err = state.albumDraftError ? `<div class="err">${escape(state.albumDraftError)}</div>` : '';
        return `
          <section class="album-dossier">
            <button class="clear-focus" data-action="album-focus-clear" title="Close album panel">× CLOSE</button>
            <p class="detail-eyebrow">Edit album details</p>
            <h3 class="detail-title">Renaming…</h3>
            <div class="detail-key">${escape(album.album_id)}</div>
            <form class="album-rename-form" data-action="album-rename-form" autocomplete="off">
              <label>Album name<input id="albumDraftName" value="${escape(draftName)}" required></label>
              <label>Description<textarea id="albumDraftDesc">${escape(draftDesc)}</textarea></label>
              ${err}
              <div class="row">
                <button type="button" class="ghost" data-action="album-edit-cancel">Cancel</button>
                <button type="submit">Save</button>
              </div>
            </form>
          </section>
        `;
      }
      const countRows = [
        ['Uploaded', counts.uploaded],
        ['Discovered', counts.discovered],
        ['Imported', counts.imported],
        ['Skipped', counts.skipped],
        ['Failed', counts.failed],
        ['Members imported', counts.members_added_imported],
        ['Members existing', counts.members_added_existing],
        ['Members already present', counts.members_already_present],
        ['Members failed', counts.members_failed],
        ['Review stubs created', counts.review_stubs_created],
        ['Review stubs existing', counts.review_stubs_existing],
        ['Review stubs skipped', counts.review_stubs_skipped],
        ['Auto detect created', counts.auto_detect_created_review],
        ['Auto detect updated', counts.auto_detect_updated_existing_review],
        ['Auto detect protected', counts.auto_detect_protected_existing_review],
        ['Auto detect skipped', counts.auto_detect_skipped],
      ].map(([label, value]) => {
        const v = value == null ? 0 : value;
        return `<div class="row ${v ? '' : 'zero'}"><span>${escape(label)}</span><strong>${v}</strong></div>`;
      }).join('');
      const filterValue = 'album:' + album.album_id;
      const isFilterOn = state.filter === filterValue;
      const isTarget = state.targetAlbumId === album.album_id;
      return `
        <section class="album-dossier">
          <button class="clear-focus" data-action="album-focus-clear" title="Close album panel">× CLOSE</button>
          <p class="detail-eyebrow">Album dossier</p>
          <h3 class="detail-title">${escape(album.album_name || album.album_id)}</h3>
          <div class="detail-key">${escape(album.album_id)}</div>
          ${album.album_description ? `<p style="font-family:var(--serif);font-style:italic;color:var(--ink-soft);font-size:14px;margin:0 0 12px;">${escape(album.album_description)}</p>` : ''}
          <div class="album-meta-grid">
            <div class="cell"><span class="label">Images</span><span class="value"><strong>${album.image_count || 0}</strong></span></div>
            <div class="cell"><span class="label">Reviewed</span><span class="value"><strong>${review.review_completed || 0}</strong> / ${review.review_total || 0}</span></div>
            <div class="cell"><span class="label">Created</span><span class="value">${escape(formatTime(album.created_at))}</span></div>
            <div class="cell"><span class="label">Updated</span><span class="value">${escape(formatTime(album.updated_at))}</span></div>
            <div class="cell"><span class="label">Source kind</span><span class="value">${escape(album.source_kind || '—')}</span></div>
            <div class="cell"><span class="label">Source label</span><span class="value">${escape(album.source_display_name || '—')}</span></div>
            <div class="cell span2"><span class="label">Latest import job</span><span class="value">${escape(album.latest_import_job_id || '—')}</span></div>
          </div>
          <div class="detail-actions">
            <button class="detail-action" data-action="album-rename-open">Rename / edit details</button>
            <button class="detail-action" data-action="add-to-album" data-album-id="${escape(album.album_id)}" style="background:var(--ink);border-color:var(--ink);">${isTarget ? 'Targeted — pick more' : 'Add Photos to Album'}</button>
            <a class="detail-action" href="/manual-review?album_id=${encodeURIComponent(album.album_id)}" data-action="review-album" data-album-id="${escape(album.album_id)}" style="text-decoration:none;display:block;text-align:center;background:var(--blue);border-color:var(--blue);">Review This Album →</a>
            <button class="detail-action" data-action="filter-album" data-album-id="${escape(album.album_id)}" style="background:transparent;color:var(--ink);border-color:var(--ink);">${isFilterOn ? 'Clear album filter' : 'Filter ledger to album'}</button>
          </div>
          <div class="album-counts-block">
            <div class="header"><span>Counts</span><span>since first import</span></div>
            ${countRows}
          </div>
        </section>
      `;
    }

    function renderDetail() {
      const album = focusedAlbum();
      if (album) return renderAlbumDossier(album);
      const im = (state.manifest?.images || []).find(x => x.image_key === state.selectedKey);
      if (!im) return '<div class="detail-empty">Select an album or image to inspect its dossier, manifest record, and counts.</div>';
      const editAction = `<div class="detail-actions"><button class="detail-action" onclick="editInManualReview('${escape(im.image_key)}')">Edit in manual review</button></div>`;
      return `
        <p class="detail-eyebrow">Manifest record</p>
        <h3 class="detail-title">${escape(im.original_filename || im.image_key)}</h3>
        <div class="detail-key">${escape(im.image_key)}</div>
        <div class="detail-thumb">${im.cache_url ? `<img src="${escape(im.cache_url)}" alt="">` : ''}</div>
        ${editAction}
        <dl class="kv">
          <dt>Status</dt><dd class="value-strong"><span class="status-tag ${escape(im.import_status)}">${escape(im.import_status)}</span></dd>
          <dt>Dimensions</dt><dd>${im.width || '?'} × ${im.height || '?'}</dd>
          <dt>Source size</dt><dd>${formatBytes(im.source_size_bytes)}</dd>
          <dt>Render profile</dt><dd>${escape(im.render_profile_kind || '-')}</dd>
          <dt>Loader</dt><dd>${escape(im.loader || '-')}</dd>
          <dt>Imported at</dt><dd>${formatTime(im.imported_at)}</dd>
          <dt>Updated at</dt><dd>${formatTime(im.updated_at)}</dd>
          <dt>Album ids</dt><dd>${(im.album_ids || []).map(escape).join(', ') || '-'}</dd>
          <dt>Albums</dt><dd>${(im.album_names || []).map(escape).join(', ') || '-'}</dd>
          <dt>Source label</dt><dd>${escape(im.source_display_name || '-')}</dd>
          <dt>Review queue</dt><dd>${escape(im.review_queue_filter || '-')}</dd>
          <dt>Review status</dt><dd>${escape(im.review_status || '-')}</dd>
          <dt>Source path</dt><dd>${escape(im.source_path || im.browser_relative_path || '-')}</dd>
          <dt>Source collection</dt><dd>${escape(im.source_collection || '-')}</dd>
          <dt>Cache path</dt><dd>${escape(im.cache_path || '-')}</dd>
          <dt>Warnings</dt><dd>${(im.warnings || []).map(escape).join(', ') || 'none'}</dd>
        </dl>
      `;
    }

    function editInManualReview(imageKey) {
      window.location.href = '/manual-review?image_id=' + encodeURIComponent(imageKey);
    }

    function renderProgressInto(host, opts) {
      if (!host || !state.job) return;
      const job = state.job;
      const pct = job.total_count > 0
        ? Math.min(100, Math.round((job.processed_count / job.total_count) * 100))
        : (job.state === 'done' ? 100 : 0);
      const indeterminate = job.state === 'queued' || job.state === 'starting' || (job.total_count === 0 && job.state !== 'done' && job.state !== 'failed');
      const failed = job.state === 'failed';
      const block = document.createElement('div');
      block.className = 'progress active' + (failed ? ' failed' : '');
      block.innerHTML = `
        <div class="progress-head">
          <span class="progress-label">${failed ? 'Import failed' : (job.state === 'done' ? 'Import complete' : 'Importing')}</span>
          <span class="progress-pct">${pct}%</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill ${indeterminate ? 'indeterminate' : ''}" style="width:${pct}%"></div>
        </div>
        <div class="progress-caption">${escape(job.current_step || '')}</div>
        <div class="progress-counts">
          <span><strong>${job.imported_count || 0}</strong> imported</span>
          <span><strong>${job.skipped_count || 0}</strong> already imported</span>
          <span class="fail"><strong>${job.failed_count || 0}</strong> failed</span>
          <span style="margin-left:auto;">${job.processed_count || 0} / ${job.total_count || '?'} files</span>
        </div>
      `;
      if (opts && opts.atTop) host.prepend(block); else host.appendChild(block);
    }

    function wireIntakeControls() {
      const fileBtn = document.getElementById('pickFiles');
      const folderBtn = document.getElementById('pickFolder');
      const newAlbumBtn = document.getElementById('newAlbumImport');
      const force = document.getElementById('forceRefresh');
      const launchPicker = (kind) => {
        if (!targetAlbum() && !state.albumName.trim()) {
          openNewAlbumImportPanel({ mode: 'name-then-import' });
          return;
        }
        const input = document.getElementById(kind === 'folder' ? 'folderInput' : 'fileInput');
        input.value = '';
        logAction(kind === 'folder' ? 'pick_folder_pressed' : 'pick_files_pressed', { target_album_id: state.targetAlbumId });
        input.click();
      };
      if (fileBtn) fileBtn.onclick = () => launchPicker('files');
      if (folderBtn) folderBtn.onclick = () => launchPicker('folder');
      if (newAlbumBtn) newAlbumBtn.onclick = () => openNewAlbumImportPanel({ mode: 'name-then-import' });
      if (force) force.onchange = (e) => { state.forceRefresh = e.target.checked; persistState(); logAction('force_refresh_changed', { checked: state.forceRefresh }); };
      const albumName = document.getElementById('albumName');
      const albumNameToolbar = document.getElementById('albumNameToolbar');
      const albumDescription = document.getElementById('albumDescription');
      const sourceDisplayName = document.getElementById('sourceDisplayName');
      if (albumName) albumName.oninput = (e) => { state.albumName = e.target.value; persistState(); };
      if (albumNameToolbar) albumNameToolbar.oninput = (e) => { state.albumName = e.target.value; persistState(); refreshAlbumsSection(); };
      if (albumDescription) albumDescription.oninput = (e) => { state.albumDescription = e.target.value; persistState(); };
      if (sourceDisplayName) sourceDisplayName.oninput = (e) => { state.sourceDisplayName = e.target.value; persistState(); };
      wireAlbumDelegation();
    }

    function refreshAlbumsSection() {
      document.querySelectorAll('.albums-block').forEach(block => {
        const replacement = document.createElement('div');
        replacement.innerHTML = renderAlbumsSection();
        block.replaceWith(replacement.firstElementChild);
      });
      wireAlbumDelegation();
    }

    function wireAlbumDelegation() {
      document.querySelectorAll('.albums-block').forEach(block => {
        if (block.dataset.wired === '1') return;
        block.dataset.wired = '1';
        block.addEventListener('click', onAlbumsBlockClick);
        block.addEventListener('keydown', (event) => {
          if (event.key !== 'Enter' && event.key !== ' ') return;
          if (event.target instanceof HTMLElement && event.target.dataset.action === 'create-album') {
            event.preventDefault();
            onAlbumsBlockClick({ target: event.target });
          }
        });
      });
      const detail = document.getElementById('detail');
      if (detail && detail.dataset.wired !== '1') {
        detail.dataset.wired = '1';
        detail.addEventListener('click', onDossierClick);
        detail.addEventListener('submit', (event) => {
          const form = event.target && event.target.closest && event.target.closest('[data-action="album-rename-form"]');
          if (!form) return;
          event.preventDefault();
          submitAlbumRename(form);
        });
      }
    }

    function onDossierClick(event) {
      const target = event.target && event.target.closest ? event.target.closest('[data-action]') : null;
      if (!target) return;
      const action = target.dataset.action;
      const albumId = target.dataset.albumId || state.focusedAlbumId;
      if (action === 'album-focus-clear') {
        event.preventDefault();
        clearAlbumFocus();
      } else if (action === 'album-rename-open') {
        event.preventDefault();
        openAlbumRename();
      } else if (action === 'album-edit-cancel') {
        event.preventDefault();
        cancelAlbumRename();
      } else if (action === 'add-to-album' && albumId) {
        event.preventDefault();
        setTargetAlbum(albumId);
        const input = document.getElementById('fileInput');
        input.value = '';
        logAction('album_add_photos_pressed', { album_id: albumId, source: 'dossier' });
        input.click();
      } else if (action === 'review-album' && albumId) {
        logAction('album_review_pressed', { album_id: albumId, source: 'dossier' });
        // default link follows
      } else if (action === 'filter-album' && albumId) {
        event.preventDefault();
        const filterValue = 'album:' + albumId;
        state.filter = state.filter === filterValue ? 'all' : filterValue;
        persistState();
        logAction('album_filter_pressed', { album_id: albumId, filter: state.filter, source: 'dossier' });
        render();
      }
    }

    async function onAlbumsBlockClick(event) {
      const target = event.target && event.target.closest ? event.target.closest('[data-action]') : null;
      if (!target) return;
      const action = target.dataset.action;
      const albumId = target.dataset.albumId || (target.closest('[data-album-id]')?.dataset?.albumId);
      if (action === 'create-album') {
        openNewAlbumImportPanel({ mode: 'create-empty' });
      } else if (action === 'create-empty-album') {
        openNewAlbumImportPanel({ mode: 'create-empty' });
      } else if (action === 'focus-album' && albumId) {
        focusAlbum(albumId);
      } else if (action === 'add-to-album' && albumId) {
        setTargetAlbum(albumId);
        const input = document.getElementById('fileInput');
        input.value = '';
        logAction('album_add_photos_pressed', { album_id: albumId });
        input.click();
      } else if (action === 'review-album' && albumId) {
        logAction('album_review_pressed', { album_id: albumId });
        // default link follows
      } else if (action === 'filter-album' && albumId) {
        const filterValue = 'album:' + albumId;
        state.filter = state.filter === filterValue ? 'all' : filterValue;
        persistState();
        logAction('album_filter_pressed', { album_id: albumId, filter: state.filter });
        render();
      } else if (action === 'clear-target') {
        setTargetAlbum(null);
        logAction('album_target_cleared');
      }
    }

    function focusAlbum(albumId) {
      state.focusedAlbumId = albumId || null;
      state.albumEditing = false;
      state.albumDraftName = null;
      state.albumDraftDesc = null;
      state.albumDraftError = null;
      persistState();
      logAction('album_focus_set', { album_id: state.focusedAlbumId });
      render();
    }

    function clearAlbumFocus() {
      state.focusedAlbumId = null;
      state.albumEditing = false;
      state.albumDraftName = null;
      state.albumDraftDesc = null;
      state.albumDraftError = null;
      persistState();
      logAction('album_focus_cleared');
      render();
    }

    function openAlbumRename() {
      const album = focusedAlbum();
      if (!album) return;
      state.albumEditing = true;
      state.albumDraftName = album.album_name || '';
      state.albumDraftDesc = album.album_description || '';
      state.albumDraftError = null;
      logAction('album_rename_opened', { album_id: album.album_id });
      render();
    }

    function cancelAlbumRename() {
      state.albumEditing = false;
      state.albumDraftName = null;
      state.albumDraftDesc = null;
      state.albumDraftError = null;
      logAction('album_rename_cancelled');
      render();
    }

    async function submitAlbumRename(form) {
      const album = focusedAlbum();
      if (!album) return;
      const name = (form.querySelector('#albumDraftName')?.value || '').trim();
      const description = (form.querySelector('#albumDraftDesc')?.value || '').trim();
      if (!name) {
        state.albumDraftError = 'Album name must not be blank';
        render();
        return;
      }
      const body = { album_name: name, album_description: description };
      logAction('album_rename_submitted', { album_id: album.album_id, name_changed: name !== (album.album_name || ''), description_changed: description !== (album.album_description || '') });
      try {
        const resp = await fetch(`/api/albums/${encodeURIComponent(album.album_id)}`, {
          method: 'PATCH', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'rename failed');
        showToast(`Renamed → ${data.album_name}`);
        state.albumEditing = false;
        state.albumDraftName = null;
        state.albumDraftDesc = null;
        state.albumDraftError = null;
        await loadAlbums();
        render();
      } catch (err) {
        state.albumDraftError = err.message;
        logAction('album_rename_failed', { album_id: album.album_id, message: err.message });
        render();
      }
    }

    function setTargetAlbum(albumId) {
      state.targetAlbumId = albumId || null;
      persistState();
      logAction('album_target_set', { album_id: state.targetAlbumId });
      render();
    }

    function openNewAlbumImportPanel(opts) {
      const mode = (opts && opts.mode) || 'create-empty';
      logAction('album_naming_card_opened', { mode });
      const existing = document.querySelector('.modal-backdrop.naming-backdrop');
      if (existing) existing.remove();
      const backdrop = document.createElement('div');
      backdrop.className = 'modal-backdrop naming-backdrop';
      backdrop.innerHTML = `
        <section class="naming-card" role="dialog" aria-modal="true" aria-labelledby="namingTitle">
          <div class="stamp">UNFILED</div>
          <p class="eyebrow">Step 01.A — New Album</p>
          <h3 id="namingTitle">Name <em>this album</em>.</h3>
          <p class="lede">An album bundles a batch of images. The name shows up in the dossier and album list; the <code>album_id</code> is generated automatically and is immutable.</p>
          <label>Album name<input id="namingAlbumName" autocomplete="off" required></label>
          <label>Description<textarea id="namingAlbumDesc" rows="2" placeholder="optional"></textarea></label>
          <div class="err" id="namingErr" style="display:none;"></div>
          <div class="pick-row">
            <button data-naming-action="pick-files">
              <span class="glyph">⌥</span>
              <span class="label">Pick Images &amp; Import</span>
              <span class="hint">.png .jpg .tif</span>
            </button>
            <button class="secondary" data-naming-action="pick-folder">
              <span class="glyph">▢</span>
              <span class="label">Pick Folder &amp; Import</span>
              <span class="hint">recursive scan, hidden skipped</span>
            </button>
          </div>
          <div class="footer-row">
            <button class="cancel" data-naming-action="cancel">× Cancel</button>
            <button data-naming-action="save-empty" style="border:1px solid var(--ink);background:transparent;color:var(--ink);font-family:var(--mono);font-size:11px;letter-spacing:0.18em;text-transform:uppercase;padding:9px 14px;border-radius:2px;cursor:pointer;">Save Empty Album</button>
          </div>
        </section>
      `;
      document.body.appendChild(backdrop);
      const nameInput = backdrop.querySelector('#namingAlbumName');
      const descInput = backdrop.querySelector('#namingAlbumDesc');
      nameInput.value = state.albumName || '';
      descInput.value = state.albumDescription || '';
      setTimeout(() => nameInput.focus(), 30);

      function close(reason) {
        backdrop.remove();
        logAction('album_naming_card_closed', { reason });
      }

      function showErr(message) {
        const err = backdrop.querySelector('#namingErr');
        err.textContent = message;
        err.style.display = message ? 'block' : 'none';
      }

      async function saveEmpty() {
        const name = nameInput.value.trim();
        const description = descInput.value.trim();
        if (!name) { showErr('Album name must not be blank'); return; }
        logAction('album_create_pressed', { name_length: name.length, has_description: !!description, mode: 'save_empty' });
        try {
          const resp = await fetch('/api/albums', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ album_name: name, album_description: description || null })
          });
          const data = await resp.json();
          if (!resp.ok) throw new Error(data.error || 'create album failed');
          logAction('album_create_succeeded', { album_id: data.album_id, mode: 'save_empty' });
          showToast(`Album "${data.album_name}" created`);
          state.albumName = '';
          state.albumDescription = '';
          persistState();
          close('saved-empty');
          await loadAlbums();
          focusAlbum(data.album_id);
        } catch (err) {
          logAction('album_create_failed', { message: err.message });
          showErr('Create failed: ' + err.message);
        }
      }

      function pickAndImport(kind) {
        const name = nameInput.value.trim();
        if (!name) { showErr('Album name must not be blank'); return; }
        const description = descInput.value.trim();
        state.albumName = name;
        state.albumDescription = description;
        state.targetAlbumId = null;
        persistState();
        logAction('album_create_pressed', { name_length: name.length, has_description: !!description, mode: 'pick_' + kind });
        close('pick-' + kind);
        const inputId = kind === 'folder' ? 'folderInput' : 'fileInput';
        const input = document.getElementById(inputId);
        if (!input) return;
        input.value = '';
        input.click();
      }

      backdrop.addEventListener('click', (event) => {
        if (event.target === backdrop) {
          close('backdrop');
        }
        const btn = event.target && event.target.closest ? event.target.closest('[data-naming-action]') : null;
        if (!btn) return;
        const action = btn.dataset.namingAction;
        if (action === 'cancel') close('cancel');
        else if (action === 'save-empty') saveEmpty();
        else if (action === 'pick-files') pickAndImport('files');
        else if (action === 'pick-folder') pickAndImport('folder');
      });
      backdrop.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') close('escape');
      });
    }

    // Backward-compatible alias for callers expecting the old name.
    const promptCreateAlbum = () => openNewAlbumImportPanel({ mode: 'create-empty' });

    function wireToolbarControls() {
      wireIntakeControls();
      const search = document.getElementById('search');
      if (search) {
        search.oninput = (e) => { state.search = e.target.value; persistState(); logAction('search_changed', { length: state.search.length }); render(); };
      }
      const sort = document.getElementById('sort');
      if (sort) {
        sort.onchange = (e) => { state.sort = e.target.value; persistState(); logAction('sort_changed', { sort: state.sort }); render(); };
      }
      const filter = document.getElementById('filter');
      if (filter) {
        filter.onchange = (e) => { state.filter = e.target.value; persistState(); logAction('filter_changed', { filter: state.filter }); render(); };
      }
      const selectionAction = document.getElementById('selectionAction');
      if (selectionAction) {
        selectionAction.onchange = (e) => {
          applySelectionAction(e.target.value);
          logAction('selection_action_changed', { action: e.target.value });
          e.target.value = '';
        };
      }
      const vL = document.getElementById('viewLedger');
      const vG = document.getElementById('viewGrid');
      if (vL) vL.onclick = () => { state.view = 'ledger'; persistState(); logAction('view_changed', { view: state.view }); rerenderCanvas(true); };
      if (vG) vG.onclick = () => { state.view = 'grid'; persistState(); logAction('view_changed', { view: state.view }); rerenderCanvas(true); };
      document.getElementById('removeSelected')?.addEventListener('click', removeSelectedFromReviewQueue);
      document.getElementById('removeFromAlbum')?.addEventListener('click', removeSelectedFromAlbum);
    }

    async function removeSelectedFromAlbum() {
      const imageKeys = state.selectedKeys.slice();
      if (!imageKeys.length || !state.filter.startsWith('album:')) return;
      const albumId = state.filter.slice('album:'.length);
      const album = state.albums.find(a => a.album_id === albumId);
      const albumLabel = album ? album.album_name || albumId : albumId;
      logAction('remove_from_album_pressed', { album_id: albumId, selected_count: imageKeys.length });
      const confirmed = window.confirm(`Remove ${imageKeys.length} selected image(s) from album "${albumLabel}"? This only removes album membership; the images stay in the workspace and review queue.`);
      logAction('remove_from_album_confirmation', { album_id: albumId, confirmed, selected_count: imageKeys.length });
      if (!confirmed) return;
      try {
        const resp = await fetch(`/api/albums/${encodeURIComponent(albumId)}/remove-images`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_keys: imageKeys })
        });
        const result = await resp.json();
        if (!resp.ok) throw new Error(result.error || 'remove from album failed');
        showToast(`Removed ${result.removed_count} image(s) from album`);
        await loadManifest();
      } catch (err) {
        showToast('Remove from album failed: ' + err.message, 'error');
      }
    }

    function rerenderCanvas(includeToggle) {
      const canvas = document.getElementById('canvas');
      const detail = document.getElementById('detail');
      if (canvas) canvas.innerHTML = state.view === 'ledger' ? renderLedger() : renderGrid();
      if (detail) detail.innerHTML = renderDetail();
      if (includeToggle) {
        document.getElementById('viewLedger')?.classList.toggle('active', state.view === 'ledger');
        document.getElementById('viewGrid')?.classList.toggle('active', state.view === 'grid');
      }
    }

    function selectImage(key) {
      state.selectedKey = key;
      persistState();
      logAction('image_selected', { image_key: key });
      document.querySelectorAll('.ledger-table tbody tr, .tile').forEach(el => el.classList.remove('selected'));
      const detail = document.getElementById('detail');
      if (detail) detail.innerHTML = renderDetail();
      rerenderCanvas();
    }

    function setSelectedKeys(keys) {
      const seen = new Set();
      state.selectedKeys = [];
      for (const key of keys) {
        if (!key || seen.has(key)) continue;
        seen.add(key);
        state.selectedKeys.push(key);
      }
      if (state.selectedKeys.length > 50000) {
        if (!state.selectionPersistenceWarned) {
          showToast('Selection exceeds 50,000 images; it will not be restored after refresh', 'error');
          state.selectionPersistenceWarned = true;
        }
      } else {
        state.selectionPersistenceWarned = false;
      }
      persistState();
    }

    function toggleImageSelection(key, index, event) {
      if (event) event.stopPropagation();
      const selected = new Set(state.selectedKeys);
      const images = filteredImages();
      if (event?.shiftKey && state.lastSelectedKey) {
        const start = images.findIndex(im => im.image_key === state.lastSelectedKey);
        const end = images.findIndex(im => im.image_key === key);
        if (start >= 0 && end >= 0) {
          for (const im of images.slice(Math.min(start, end), Math.max(start, end) + 1)) selected.add(im.image_key);
        } else if (selected.has(key)) {
          selected.delete(key);
        } else {
          selected.add(key);
        }
      } else if (selected.has(key)) {
        selected.delete(key);
      } else {
        selected.add(key);
      }
      state.lastSelectedKey = key;
      setSelectedKeys(Array.from(selected));
      logAction('image_checkbox_toggled', { image_key: key, selected_count: state.selectedKeys.length, shift: !!event?.shiftKey });
      render();
    }

    function selectAllMatching() {
      const selected = new Set(state.selectedKeys);
      for (const im of filteredImages()) selected.add(im.image_key);
      setSelectedKeys(Array.from(selected));
      logAction('select_all_matching', { matching_count: filteredImages().length, selected_count: state.selectedKeys.length });
      render();
    }

    function invertCurrentFilteredSelection() {
      const selected = new Set(state.selectedKeys);
      for (const im of filteredImages()) {
        if (selected.has(im.image_key)) selected.delete(im.image_key);
        else selected.add(im.image_key);
      }
      setSelectedKeys(Array.from(selected));
      logAction('invert_filtered_selection', { matching_count: filteredImages().length, selected_count: state.selectedKeys.length });
      render();
    }

    function clearSelection() {
      setSelectedKeys([]);
      logAction('clear_selection');
      render();
    }

    function applySelectionAction(action) {
      if (action === 'select_matching') selectAllMatching();
      else if (action === 'invert_filtered') invertCurrentFilteredSelection();
      else if (action === 'clear') clearSelection();
    }

    async function removeSelectedFromReviewQueue() {
      const imageKeys = state.selectedKeys.slice();
      if (!imageKeys.length) return;
      logAction('remove_selected_pressed', { selected_count: imageKeys.length });
      try {
        const previewResp = await fetch('/api/images/remove/preview', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_keys: imageKeys, filters: { search: state.search, sort: state.sort, filter: state.filter } })
        });
        const preview = await previewResp.json();
        if (!previewResp.ok) throw new Error(preview.error || 'preview failed');
        const message = `Remove ${preview.removable_count} selected images from this workspace? This removes import records, album membership, active review files, archived review files, and workspace cache/staging copies. Original external source files are untouched.`;
        if (!preview.removable_count) return;
        const confirmed = window.confirm(message);
        logAction('remove_selected_confirmation', { confirmed, removable_count: preview.removable_count, missing_count: preview.missing_count });
        if (!confirmed) return;
        const confirmResp = await fetch('/api/images/remove', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_keys: imageKeys, confirmed_removable_count: preview.removable_count, filters: { search: state.search, sort: state.sort, filter: state.filter } })
        });
        const result = await confirmResp.json();
        if (!confirmResp.ok) throw new Error(result.error || 'remove failed');
        const removed = new Set((result.removed || []).map(item => item.image_key));
        setSelectedKeys(state.selectedKeys.filter(key => !removed.has(key)));
        showToast(`Removed ${result.removed_count || 0} images`);
        await loadManifest();
      } catch (err) {
        showToast('Remove failed: ' + err.message, 'error');
      }
    }

    async function uploadFiles(fileList) {
      if (!fileList || fileList.length === 0) return;
      logAction('upload_files_selected', { selected_file_count: fileList.length });
      const resolvedFiles = await resolveDuplicateUploads(fileList);
      if (!resolvedFiles.length) {
        logAction('upload_cancelled_no_files_after_conflicts', { original_file_count: fileList.length });
        showToast('No files selected for import after duplicate choices', 'error');
        return;
      }
      const formData = new FormData();
      let topLevel = '';
      for (const item of resolvedFiles) {
        const f = item.file;
        const name = item.uploadName;
        if (!topLevel && name.includes('/')) {
          topLevel = name.split('/')[0];
        }
        formData.append('files', f, name);
      }
      const target = targetAlbum();
      let albumName = state.albumName.trim();
      const sourceDisplayName = state.sourceDisplayName.trim() || topLevel;
      if (!target && !albumName && !sourceDisplayName) {
        // Should not be reachable: the naming card runs before the file dialog. Guard anyway.
        showToast('Name a new album first via + Import to new album', 'error');
        logAction('album_name_missing_at_upload', { staged_count: resolvedFiles.length });
        return;
      }
      const params = new URLSearchParams();
      if (state.forceRefresh) params.set('force', '1');
      if (target) {
        params.set('append_to_album_id', target.album_id);
      } else {
        if (albumName.trim()) params.set('album_name', albumName.trim());
        if (state.albumDescription.trim()) params.set('album_description', state.albumDescription.trim());
        if (sourceDisplayName) params.set('source_display_name', sourceDisplayName);
      }
      const url = '/api/import/upload' + (params.toString() ? '?' + params.toString() : '');
      try {
        logAction('upload_started', { file_count: resolvedFiles.length, force: state.forceRefresh, album_name: target ? null : !!albumName.trim(), append_to_album_id: target ? target.album_id : null, source_display_name: target ? null : (sourceDisplayName || null) });
        const resp = await fetch(url, { method: 'POST', body: formData });
        if (!resp.ok) {
          const text = await resp.text();
          showToast('Upload failed: ' + text.slice(0, 120), 'error');
          logAction('upload_failed_response', { status: resp.status, message: text.slice(0, 200) });
          return;
        }
        const data = await resp.json();
        state.job = { state: 'queued', current_step: 'Uploaded. Queued for import.', total_count: 0, processed_count: 0 };
        logAction('upload_queued', { job_id: data.job_id, queued_files: data.queued_files });
        render();
        startPolling(data.job_id);
      } catch (err) {
        showToast('Upload error: ' + err.message, 'error');
        logAction('upload_error', { message: err.message });
      }
    }

    function startPolling(jobId) {
      stopPolling();
      const tick = async () => {
        try {
          const resp = await fetch('/api/import/jobs/' + encodeURIComponent(jobId));
          if (!resp.ok) { stopPolling(); return; }
          const job = await resp.json();
          state.job = job;
          render();
          if (job.state === 'done' || job.state === 'failed') {
            stopPolling();
            if (job.state === 'done') {
              showToast(`Imported ${job.imported_count}, skipped ${job.skipped_count}, failed ${job.failed_count}`);
              logAction('import_job_done', { job_id: jobId, imported_count: job.imported_count, skipped_count: job.skipped_count, failed_count: job.failed_count });
              if (state.targetAlbumId) { state.targetAlbumId = null; persistState(); }
              await loadManifest();
              setTimeout(() => { state.job = null; render(); }, 4000);
            } else {
              logAction('import_job_failed', { job_id: jobId, error: job.error || job.current_step });
            }
          }
        } catch (err) {
          stopPolling();
          logAction('import_job_poll_error', { job_id: jobId, message: err.message });
        }
      };
      tick();
      state.pollTimer = setInterval(tick, POLL_MS);
    }
    function stopPolling() {
      if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
    }

    document.getElementById('fileInput').addEventListener('change', async (e) => { await uploadFiles(e.target.files); e.target.value = ''; });
    document.getElementById('folderInput').addEventListener('change', async (e) => { await uploadFiles(e.target.files); e.target.value = ''; });

    document.addEventListener('click', event => {
      const target = event.target && event.target.closest ? event.target.closest('button,a,input,select') : null;
      if (!target) return;
      logAction('gui_click', {
        tag: target.tagName,
        id: target.id || null,
        text: (target.innerText || target.value || '').slice(0, 80),
        href: target.getAttribute && target.getAttribute('href'),
      });
    }, true);

    window.selectImage = selectImage;
    window.toggleImageSelection = toggleImageSelection;
    window.editInManualReview = editInManualReview;

    restoreState();
    loadManifest();
  </script>
</body>
</html>"""


class TrainingPipelineHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", ""}:
            self._respond_html(render_import_html())
            return
        if path == "/bbox-overview":
            self._respond_html(render_bbox_overview_html())
            return
        if path == "/dataset-selection":
            self._respond_html(render_dataset_selection_html())
            return
        if path == "/training-setup":
            self._respond_html(render_training_setup_html())
            return
        if path == "/train-yolo":
            self._respond_html(render_train_yolo_html())
            return
        if path == "/import":
            self._respond_html(render_import_html())
            return
        if path in {"/manual-review", "/manual-review/"}:
            album_id = self._query_album_id(parsed)
            queue_rows = review_ui.filter_queue_by_album(review_ui.synced_review_queue(), album_id)
            default_image_id = review_ui.pick_image_id(queue_rows, self._query_image_id(parsed))
            self._respond_html(review_ui.render_app_html(default_image_id, self._query_detection_id(parsed), album_id=album_id))
            return
        if path == "/api/overview":
            generate = query_bool(parsed, "generate", GENERATE_CROPS)
            payload = build_review_bbox_overview(
                REVIEWS_DIR,
                crop_root=CROP_ROOT,
                padding_px=PADDING_PX,
                include_incomplete=INCLUDE_INCOMPLETE,
                generate_crops=generate,
            )
            self._respond_json(payload)
            return
        if path == "/api/state":
            self._respond_json(review_ui.build_state(self._query_image_id(parsed), album_id=self._query_album_id(parsed)))
            return
        if path == "/api/import":
            self._respond_json(manifest_summary())
            return
        if path == "/api/albums":
            self._handle_list_albums()
            return
        if path == "/api/dataset-candidates":
            self._handle_dataset_candidates(parsed)
            return
        if path == "/api/dataset-selections":
            self._handle_list_dataset_selections()
            return
        if path == "/api/exported-datasets":
            self._handle_list_exported_datasets()
            return
        if path == "/api/starting-weights":
            self._handle_list_starting_weights()
            return
        if path == "/api/training-configs":
            self._handle_list_training_configs()
            return
        if path == "/api/system/device-info":
            self._handle_device_info()
            return
        if path.startswith("/api/training-configs/"):
            run_group = path[len("/api/training-configs/"):].strip("/")
            if run_group and "/" not in run_group:
                self._handle_show_training_config(run_group)
                return
        if path.startswith("/api/albums/"):
            album_id = path[len("/api/albums/"):].strip("/")
            if album_id and "/" not in album_id:
                self._handle_show_album(album_id)
                return
        if path.startswith("/api/import/jobs/"):
            job_id = path[len("/api/import/jobs/"):]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                payload = dict(job) if job else None
            if payload is None:
                self._respond_json({"error": "job not found"}, status=404)
            else:
                self._respond_json(payload)
            return
        if path.startswith("/cache/"):
            self._serve_cache_file(path[len("/cache/"):])
            return
        self._respond_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/training-ui-interaction":
            self._handle_training_ui_interaction()
            return
        if parsed.path == "/api/import/upload":
            self._handle_upload(parsed)
            return
        if parsed.path == "/api/albums":
            self._handle_create_album()
            return
        if parsed.path == "/api/dataset-selections":
            self._handle_save_dataset_selection()
            return
        if parsed.path == "/api/dataset-selections/export":
            self._handle_export_dataset_selection()
            return
        if parsed.path == "/api/training-configs":
            self._handle_save_training_config()
            return
        if parsed.path == "/api/training-config/check-ready":
            self._handle_training_config_check_ready()
            return
        if parsed.path.startswith("/api/albums/") and parsed.path.endswith("/add-images"):
            album_id = parsed.path[len("/api/albums/"):-len("/add-images")].strip("/")
            if album_id and "/" not in album_id:
                self._handle_add_album_images(album_id)
                return
        if parsed.path.startswith("/api/albums/") and parsed.path.endswith("/remove-images"):
            album_id = parsed.path[len("/api/albums/"):-len("/remove-images")].strip("/")
            if album_id and "/" not in album_id:
                self._handle_remove_album_images(album_id)
                return
        if parsed.path == "/api/images/remove/preview":
            self._handle_remove_preview()
            return
        if parsed.path == "/api/images/remove":
            self._handle_remove_confirm()
            return
        if parsed.path in REVIEW_POST_PATHS:
            review_ui.ReviewHandler.do_POST(self)
            return
        self._respond_json({"error": "not found"}, status=404)

    def do_PATCH(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/albums/"):
            album_id = parsed.path[len("/api/albums/"):].strip("/")
            if album_id and "/" not in album_id:
                self._handle_rename_album(album_id)
                return
        self._respond_json({"error": "not found"}, status=404)

    def _handle_upload(self, parsed) -> None:
        reap_old_jobs()
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._respond_json({"error": "empty upload"}, status=400)
            return
        body = self.rfile.read(length)
        try:
            files = parse_multipart_files(body, content_type)
        except ValueError as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        if not files:
            self._respond_json({"error": "no files in upload"}, status=400)
            return
        force = query_bool(parsed, "force", False)
        album_name = query_text(parsed, "album_name")
        album_description = query_text(parsed, "album_description")
        source_display_name = query_text(parsed, "source_display_name")
        append_to_album_id = query_text(parsed, "append_to_album_id")
        try:
            job_id = start_import_job(
                files,
                force=force,
                album_name=album_name,
                album_description=album_description,
                source_display_name=source_display_name,
                append_to_album_id=append_to_album_id,
            )
        except Exception as exc:
            self._respond_json({"error": f"could not stage upload: {exc}"}, status=500)
            return
        self._respond_json({"job_id": job_id, "queued_files": len(files)}, status=202)

    def _handle_training_ui_interaction(self) -> None:
        try:
            payload = self._json_body()
            append_training_ui_interaction(payload)
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        self._respond_json({"ok": True})

    def _json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _handle_prune_preview(self) -> None:
        try:
            payload = self._json_body()
            result = pipeline.preview_review_queue_prune(
                reviewer=str(payload.get("reviewer") or IMPORT_REVIEWER),
                reviews_dir=REVIEWS_DIR,
                image_keys=[str(value) for value in payload.get("image_keys") or []],
                filters=payload.get("filters") or {},
            )
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        self._respond_json(result)

    def _handle_prune_confirm(self) -> None:
        try:
            payload = self._json_body()
            result = pipeline.prune_review_queue(
                reviewer=str(payload.get("reviewer") or IMPORT_REVIEWER),
                reviews_dir=REVIEWS_DIR,
                image_keys=[str(value) for value in payload.get("image_keys") or []],
                filters=payload.get("filters") or {},
                confirmed_archivable_count=int(payload.get("confirmed_archivable_count")),
            )
            review_ui.invalidate_review_queue_cache()
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        self._respond_json(result)

    def _handle_list_albums(self) -> None:
        try:
            summaries = pipeline.list_albums_summary(
                pipeline.default_albums_dir_for_manifest(IMPORT_MANIFEST_PATH),
                manifest_path=IMPORT_MANIFEST_PATH,
                reviews_dir=REVIEWS_DIR,
                reviewer=IMPORT_REVIEWER,
            )
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._respond_json({"albums": summaries})

    def _handle_dataset_candidates(self, parsed) -> None:
        query = parse_qs(parsed.query)
        try:
            result = pipeline.list_dataset_candidates(
                reviews_dir=REVIEWS_DIR,
                manifest_path=IMPORT_MANIFEST_PATH,
                albums_dir=pipeline.default_albums_dir_for_manifest(IMPORT_MANIFEST_PATH),
                reviewer=IMPORT_REVIEWER,
                album_ids=[str(value) for value in query.get("album_id") or [] if str(value).strip()],
                required_labels=[str(value) for value in query.get("label") or [] if str(value).strip()],
                completed_only=not query_bool(parsed, "include_incomplete", False),
            )
            for candidate in result.get("candidates") or []:
                if isinstance(candidate, dict):
                    candidate["cache_url"] = cache_url_for(candidate.get("cache_path"))
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        self._respond_json(result)

    def _handle_list_dataset_selections(self) -> None:
        try:
            selections = pipeline.list_dataset_selections(pipeline.DEFAULT_DATASET_SELECTIONS_DIR)
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._respond_json({"selections": selections})


    def _handle_list_exported_datasets(self) -> None:
        try:
            datasets = pipeline.list_exported_datasets(pipeline.DEFAULT_DATASET_SELECTIONS_DIR)
        except Exception as exc:
            self._respond_json({"ok": False, "error": str(exc)}, status=500)
            return
        self._respond_json({"ok": True, "datasets": datasets})


    def _handle_list_starting_weights(self) -> None:
        try:
            weights = pipeline.list_starting_weights()
        except Exception as exc:
            self._respond_json({"ok": False, "error": str(exc)}, status=500)
            return
        self._respond_json({"ok": True, "weights": weights})


    def _handle_list_training_configs(self) -> None:
        try:
            configs = pipeline.list_training_configs(pipeline.DEFAULT_TRAINING_CONFIGS_DIR)
        except Exception as exc:
            self._respond_json({"ok": False, "error": str(exc)}, status=500)
            return
        self._respond_json({"ok": True, "configs": configs})


    def _handle_show_training_config(self, run_group: str) -> None:
        try:
            path = pipeline.training_config_path(pipeline.DEFAULT_TRAINING_CONFIGS_DIR, run_group)
            if not path.is_file():
                self._respond_json({"ok": False, "error": "training config not found"}, status=404)
                return
            config = pipeline.load_json(path)
        except Exception as exc:
            self._respond_json({"ok": False, "error": str(exc)}, status=500)
            return
        self._respond_json({"ok": True, "config_path": pipeline.repo_relative(path), "config": config})


    def _handle_device_info(self) -> None:
        try:
            result = pipeline.probe_training_devices()
        except Exception as exc:
            self._respond_json({"ok": False, "error": str(exc)}, status=500)
            return
        payload = {"ok": True}
        payload.update(result)
        self._respond_json(payload)

    def _handle_save_dataset_selection(self) -> None:
        try:
            payload = self._json_body()
            result = pipeline.save_dataset_selection(
                dataset_name=str(payload.get("dataset_name") or "").strip(),
                dataset_version=str(payload.get("dataset_version") or "v1").strip() or "v1",
                description=payload.get("description"),
                image_keys=[str(value) for value in (payload.get("image_keys") or []) if str(value).strip()],
                selections_dir=pipeline.DEFAULT_DATASET_SELECTIONS_DIR,
                datasets_dir=pipeline.DEFAULT_DATASETS_DIR,
                reviews_dir=REVIEWS_DIR,
                manifest_path=IMPORT_MANIFEST_PATH,
                reviewer=IMPORT_REVIEWER,
                album_ids=[str(value) for value in (payload.get("album_ids") or []) if str(value).strip()],
                seed=int(payload.get("seed") or 42),
                split_ratios=payload.get("split_ratios"),
                created_by=IMPORT_REVIEWER,
            )
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        self._respond_json({
            "selection_path": result["selection_path"],
            "selection": result["selection"],
        }, status=201)

    def _handle_export_dataset_selection(self) -> None:
        try:
            payload = self._json_body()
            selection_path = payload.get("selection_path")
            if not selection_path:
                self._respond_json({"error": "selection_path is required"}, status=400)
                return
            manifest = pipeline.export_dataset_selection(
                selection_path,
                output_dir_override=payload.get("output_dir"),
                include_incomplete_override=payload.get("include_incomplete_reviews"),
            )
            selection = pipeline.load_json(selection_path)
            dataset_dir = pipeline.repo_resolve(payload.get("output_dir") or selection.get("export", {}).get("output_dir"))
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        self._respond_json({
            "dataset_dir": str(dataset_dir),
            "dataset_manifest": str(dataset_dir / "dataset_manifest.json"),
            "data_yaml": str(dataset_dir / "data.yaml"),
            "exported_image_count": len(manifest.get("exported_image_keys") or []),
            "warnings": list(manifest.get("warnings") or []),
        })


    def _handle_save_training_config(self) -> None:
        try:
            payload = self._json_body()
            config = payload.get("config") if isinstance(payload.get("config"), dict) else payload
            append_training_ui_interaction({"action": "training_config_save_attempted", "run_group": config.get("run_group") if isinstance(config, dict) else None})
            if not isinstance(config, dict):
                raise ValueError("config JSON object is required")
            result = pipeline.save_training_config(config, configs_dir=pipeline.DEFAULT_TRAINING_CONFIGS_DIR)
        except Exception as exc:
            append_training_ui_interaction({"action": "training_config_save_failed", "error": str(exc)})
            self._respond_json({"ok": False, "error": str(exc)}, status=400)
            return
        append_training_ui_interaction({"action": "training_config_save_succeeded", "config_path": result.get("config_path"), "run_group": result.get("config", {}).get("run_group")})
        self._respond_json({
            "ok": True,
            "config_path": result["config_path"],
            "config": result["config"],
            "provenance_warnings": result.get("provenance_warnings") or [],
        }, status=201)


    def _handle_training_config_check_ready(self) -> None:
        try:
            payload = self._json_body()
            config = payload.get("config") if isinstance(payload.get("config"), dict) else payload
            if not isinstance(config, dict):
                raise ValueError("config JSON object is required")
            selection_path = payload.get("selection_path") or config.get("selection_path")
            readiness = pipeline.check_training_readiness(selection_path, config)
        except Exception as exc:
            self._respond_json({"ok": False, "error": str(exc)}, status=400)
            return
        self._respond_json({
            "ok": True,
            "ready": readiness["ready"],
            "missing": readiness["missing"],
            "warnings": readiness.get("warnings") or [],
            "runs": readiness.get("runs") or [],
        })

    def _handle_show_album(self, album_id: str) -> None:
        try:
            detail = pipeline.load_album_detail(
                pipeline.default_albums_dir_for_manifest(IMPORT_MANIFEST_PATH),
                album_id,
                manifest_path=IMPORT_MANIFEST_PATH,
                reviews_dir=REVIEWS_DIR,
                reviewer=IMPORT_REVIEWER,
            )
        except ValueError as exc:
            self._respond_json({"error": str(exc)}, status=404)
            return
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._respond_json(detail)

    def _handle_create_album(self) -> None:
        try:
            payload = self._json_body()
            album_name = str(payload.get("album_name") or "").strip()
            if not album_name:
                self._respond_json({"error": "album_name is required"}, status=400)
                return
            record = pipeline.create_empty_album(
                albums_dir=pipeline.default_albums_dir_for_manifest(IMPORT_MANIFEST_PATH),
                manifest_path=IMPORT_MANIFEST_PATH,
                cache_dir=IMPORT_CACHE_DIR,
                reviews_dir=REVIEWS_DIR,
                album_name=album_name,
                album_description=payload.get("album_description"),
                source_display_name=payload.get("source_display_name"),
                reviewer=IMPORT_REVIEWER,
            )
        except ValueError as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._respond_json({
            "album_id": record["album_id"],
            "album_name": record["album_name"],
            "album_description": record["album_description"],
            "source_kind": record["source_kind"],
            "image_count": 0,
        }, status=201)

    def _handle_rename_album(self, album_id: str) -> None:
        try:
            payload = self._json_body()
            has_name = "album_name" in payload
            has_description = "album_description" in payload
            if not has_name and not has_description:
                self._respond_json({"error": "album_name and/or album_description is required"}, status=400)
                return
            record = pipeline.rename_album(
                albums_dir=pipeline.default_albums_dir_for_manifest(IMPORT_MANIFEST_PATH),
                manifest_path=IMPORT_MANIFEST_PATH,
                album_id=album_id,
                album_name=payload.get("album_name") if has_name else None,
                album_description=payload.get("album_description") if has_description else None,
            )
        except ValueError as exc:
            message = str(exc)
            status = 404 if "not found" in message.lower() else 400
            self._respond_json({"error": message}, status=status)
            return
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._respond_json({
            "album_id": record["album_id"],
            "album_name": record["album_name"],
            "album_description": record["album_description"],
            "updated_at": record["updated_at"],
            "created_at": record["created_at"],
        })

    def _handle_remove_album_images(self, album_id: str) -> None:
        try:
            payload = self._json_body()
            keys = [str(value) for value in (payload.get("image_keys") or []) if str(value).strip()]
            if not keys:
                self._respond_json({"error": "image_keys is required"}, status=400)
                return
            result = pipeline.remove_image_keys_from_album(
                albums_dir=pipeline.default_albums_dir_for_manifest(IMPORT_MANIFEST_PATH),
                manifest_path=IMPORT_MANIFEST_PATH,
                album_id=album_id,
                image_keys=keys,
            )
        except ValueError as exc:
            self._respond_json({"error": str(exc)}, status=404)
            return
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._respond_json(result)

    def _handle_add_album_images(self, album_id: str) -> None:
        try:
            payload = self._json_body()
            keys = [str(value) for value in (payload.get("image_keys") or []) if str(value).strip()]
            if not keys:
                self._respond_json({"error": "image_keys is required"}, status=400)
                return
            result = pipeline.add_existing_image_keys_to_album(
                albums_dir=pipeline.default_albums_dir_for_manifest(IMPORT_MANIFEST_PATH),
                manifest_path=IMPORT_MANIFEST_PATH,
                album_id=album_id,
                image_keys=keys,
                reviewer=IMPORT_REVIEWER,
            )
        except ValueError as exc:
            self._respond_json({"error": str(exc)}, status=404)
            return
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        if int(result.get("members_failed") or 0) == int(result.get("requested_count") or 0):
            payload = dict(result)
            payload["error"] = "No image_key could be added to album"
            self._respond_json(payload, status=400)
            return
        self._respond_json(result)

    def _handle_remove_preview(self) -> None:
        try:
            payload = self._json_body()
            result = pipeline.preview_remove_images(
                reviewer=str(payload.get("reviewer") or IMPORT_REVIEWER),
                reviews_dir=REVIEWS_DIR,
                manifest_path=IMPORT_MANIFEST_PATH,
                image_keys=[str(value) for value in payload.get("image_keys") or []],
                filter_mode=payload.get("filter"),
                filters=payload.get("filters") or {},
            )
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        self._respond_json(result)

    def _handle_remove_confirm(self) -> None:
        try:
            payload = self._json_body()
            result = pipeline.remove_images(
                reviewer=str(payload.get("reviewer") or IMPORT_REVIEWER),
                reviews_dir=REVIEWS_DIR,
                manifest_path=IMPORT_MANIFEST_PATH,
                image_keys=[str(value) for value in payload.get("image_keys") or []],
                filter_mode=payload.get("filter"),
                filters=payload.get("filters") or {},
                confirmed_removable_count=int(payload.get("confirmed_removable_count")),
            )
            review_ui.invalidate_review_queue_cache()
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        self._respond_json(result)

    def _serve_cache_file(self, relative_path: str) -> None:
        cleaned = relative_path.lstrip("/")
        candidates = [repo_resolve(cleaned)] if cleaned.startswith("workspace/") else [repo_resolve("workspace") / cleaned]
        cache_root = repo_resolve("workspace/cache").resolve()
        target_resolved = None
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved.exists() and (resolved == cache_root or cache_root in resolved.parents):
                target_resolved = resolved
                break
        if target_resolved is None:
            self._respond_json({"error": "cache file not found"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(target_resolved))[0] or "application/octet-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(target_resolved.read_bytes())

    def _query_image_id(self, parsed) -> str | None:
        values = parse_qs(parsed.query).get("image_id")
        if not values:
            return None
        value = values[0].strip()
        return value or None

    def _query_album_id(self, parsed) -> str | None:
        values = parse_qs(parsed.query).get("album_id")
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
    parser = argparse.ArgumentParser(description="OpenDarkSTS training pipeline UI")
    parser.add_argument("--port", type=int, default=8766, help="Port to serve on")
    parser.add_argument("--reviews-dir", default=str(DEFAULT_REVIEWS_DIR), help="Root directory containing review JSON files")
    parser.add_argument("--crop-root", default=str(WORKSPACE_DIR / "cache" / "review_crops"), help="Root directory for bbox crop thumbnails")
    parser.add_argument("--padding-px", type=int, default=48, help="Padding around generated bbox crop thumbnails")
    parser.add_argument("--manual-review-url", default="/manual-review", help="Base URL for the step 2 manual reviewer")
    parser.add_argument("--include-incomplete", action="store_true", help="Include reviewed bboxes from incomplete review files")
    parser.add_argument("--no-generate-crops", action="store_true", help="Do not generate missing or stale overview crop files")
    parser.add_argument("--import-manifest", default=str(pipeline.DEFAULT_IMPORT_MANIFEST_PATH), help="Path to the training import manifest")
    parser.add_argument("--import-cache-dir", default=str(pipeline.DEFAULT_IMPORT_CACHE_DIR), help="Directory holding browser-ready cache PNGs")
    parser.add_argument("--reviewer", default=DEFAULT_REVIEWER, help="Reviewer used when seeding new review stubs")
    args = parser.parse_args()

    global REVIEWS_DIR, CROP_ROOT, PADDING_PX, INCLUDE_INCOMPLETE, GENERATE_CROPS, MANUAL_REVIEW_URL
    global IMPORT_MANIFEST_PATH, IMPORT_CACHE_DIR, IMPORT_REVIEWER
    REVIEWS_DIR = Path(args.reviews_dir)
    CROP_ROOT = Path(args.crop_root)
    PADDING_PX = args.padding_px
    INCLUDE_INCOMPLETE = args.include_incomplete
    GENERATE_CROPS = not args.no_generate_crops
    MANUAL_REVIEW_URL = args.manual_review_url
    IMPORT_MANIFEST_PATH = Path(args.import_manifest)
    IMPORT_CACHE_DIR = Path(args.import_cache_dir)
    IMPORT_REVIEWER = args.reviewer
    IMPORT_STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    review_ui.MANIFEST_PATH = IMPORT_MANIFEST_PATH
    review_ui.REVIEWS_DIR = REVIEWS_DIR
    review_ui.REVIEWER = IMPORT_REVIEWER

    print("OpenDarkSTS Training Pipeline UI")
    print(f"  Import manifest: {repo_resolve(IMPORT_MANIFEST_PATH)}")
    print(f"  Import cache:    {repo_resolve(IMPORT_CACHE_DIR)}")
    print(f"  Reviews:         {repo_resolve(REVIEWS_DIR)}")
    print(f"  Reviewer:        {IMPORT_REVIEWER}")
    print(f"  Open http://localhost:{args.port}")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), TrainingPipelineHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
