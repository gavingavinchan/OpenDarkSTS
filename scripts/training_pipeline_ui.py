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


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = TEMPLATES_DIR / "static"
TEMPLATE_CACHE: dict[str, str] = {}

PIPELINE_STAGE_LABELS: list[str] = [
    "Import Data",
    "Manual Review",
    "BBox Overview",
    "Dataset Selection",
    "Training Setup",
    "Train YOLO",
    "Evaluate",
]


def pipeline_stages() -> list[tuple[str, str]]:
    """Stage hrefs are resolved per render because MANUAL_REVIEW_URL is configurable."""
    hrefs = [
        "/import",
        MANUAL_REVIEW_URL.rstrip("/") + "/",
        "/bbox-overview",
        "/datasets",
        "/training-setup",
        "/training",
        "/evaluation",
    ]
    return list(zip(PIPELINE_STAGE_LABELS, hrefs))


def load_templates() -> dict[str, str]:
    """Read every template from disk once so page renders are plain substitutions."""
    TEMPLATE_CACHE.clear()
    for path in sorted(TEMPLATES_DIR.glob("*.html")):
        TEMPLATE_CACHE[path.stem] = path.read_text(encoding="utf-8")
    return TEMPLATE_CACHE


def template_text(name: str) -> str:
    if name not in TEMPLATE_CACHE:
        load_templates()
    if name not in TEMPLATE_CACHE:
        raise KeyError(f"Unknown template: {name}")
    return TEMPLATE_CACHE[name]


def render_template(name: str, /, **substitutions: str) -> str:
    """Substitute __UPPER_SNAKE__ placeholders in a template loaded from scripts/templates."""
    html = template_text(name)
    for key, value in substitutions.items():
        html = html.replace(f"__{key.upper()}__", str(value))
    return html


def render_stage_nav(active_stage: int, *, rail_class: str = "rail", stage_class: str = "stage") -> str:
    """Build the global 7-stage pipeline progress bar shared by every page."""
    parts = [f'<nav class="{rail_class}" aria-label="Training pipeline stages">']
    for index, (label, href) in enumerate(pipeline_stages(), start=1):
        classes = [stage_class]
        if index == active_stage:
            classes.append("active")
        elif index < active_stage:
            classes.append("visited")
        parts.append(
            f'<a class="{" ".join(classes)}" href="{href}" data-stage="{index}">'
            f'<span class="stage-num">{index}</span>{label}</a>'
        )
    parts.append("</nav>")
    return "\n        ".join(parts)


def render_bbox_overview_html() -> str:
    return render_template(
        "bbox_overview",
        manual_review_url=json.dumps(MANUAL_REVIEW_URL),
        manual_review_href=MANUAL_REVIEW_URL.rstrip("/") + "/",
        stage_nav=render_stage_nav(3),
    )


def render_app_html() -> str:
    return render_bbox_overview_html()


def render_import_html() -> str:
    return render_template(
        "import",
        manual_review_href=MANUAL_REVIEW_URL.rstrip("/") + "/",
        stage_nav=render_stage_nav(1),
    )


def render_datasets_html() -> str:
    return render_template("datasets", stage_nav=render_stage_nav(4))


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
        if path == "/import":
            self._respond_html(render_import_html())
            return
        if path in {"/datasets", "/dataset-selection"}:
            self._respond_html(render_datasets_html())
            return
        if path.startswith("/static/"):
            self._serve_static_file(path[len("/static/"):])
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
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
        if path == "/api/training/exported-datasets":
            self._handle_list_exported_datasets()
            return
        if path == "/api/training/configs":
            self._handle_list_training_configs()
            return
        if path == "/api/training/starting-weights":
            self._handle_list_starting_weights()
            return
        if path == "/api/training/runs/active":
            self._handle_list_active_runs(parsed)
            return
        if path.startswith("/api/jobs/"):
            self._handle_job_status(path[len("/api/jobs/"):])
            return
        if path.startswith("/api/training/runs/") and path.endswith("/status"):
            run_ref = self._training_run_ref(path[len("/api/training/runs/"):-len("/status")])
            if run_ref:
                self._handle_training_run_status(*run_ref, parsed)
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
        if parsed.path == "/api/training/configs":
            self._handle_save_training_config()
            return
        if parsed.path == "/api/training/readiness":
            self._handle_training_readiness()
            return
        if parsed.path == "/api/training/command-preview":
            self._handle_training_command_preview()
            return
        if parsed.path == "/api/training/runs/start":
            self._handle_start_training_run()
            return
        if parsed.path == "/api/training/evaluation-report":
            self._handle_write_evaluation_report()
            return
        if parsed.path == "/api/training/promote-weight":
            self._handle_promote_weight()
            return
        if parsed.path.startswith("/api/training/runs/") and parsed.path.endswith("/stop"):
            run_ref = self._training_run_ref(parsed.path[len("/api/training/runs/"):-len("/stop")])
            if run_ref:
                self._handle_stop_training_run(*run_ref)
                return
        if parsed.path.startswith("/api/training/runs/") and parsed.path.endswith("/test-split"):
            run_ref = self._training_run_ref(parsed.path[len("/api/training/runs/"):-len("/test-split")])
            if run_ref:
                self._handle_run_test_split(*run_ref)
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

    def _training_run_ref(self, raw: str) -> tuple[str, str] | None:
        parts = [part for part in raw.strip("/").split("/") if part]
        if len(parts) != 2 or any(part in {".", ".."} for part in parts):
            return None
        return parts[0], parts[1]

    def _log_training_action(self, action: str, details: dict) -> None:
        try:
            append_training_ui_interaction({"source": "training_api", "action": action, **details})
        except Exception:
            pass

    def _handle_job_status(self, job_id: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id.strip("/"))
            payload = dict(job) if job else None
        if payload is None:
            self._respond_json({"error": "job not found"}, status=404)
            return
        self._respond_json(payload)

    def _handle_list_exported_datasets(self) -> None:
        try:
            datasets = pipeline.list_exported_datasets(pipeline.DEFAULT_DATASETS_DIR)
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._respond_json({"datasets": datasets})

    def _handle_list_training_configs(self) -> None:
        try:
            configs = pipeline.list_training_configs(pipeline.DEFAULT_TRAINING_CONFIGS_DIR)
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._respond_json({"configs": configs})

    def _handle_list_starting_weights(self) -> None:
        try:
            weights = pipeline.list_starting_weights(
                root_dir=pipeline.ROOT_DIR,
                model_weights_dir=pipeline.DEFAULT_MODEL_WEIGHTS_DIR,
                runs_dir=pipeline.DEFAULT_RUNS_DIR,
            )
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        warnings = [entry["provenance_warning"] for entry in weights if entry.get("provenance_warning")]
        self._respond_json({"weights": weights, "warnings": warnings})

    def _handle_save_training_config(self) -> None:
        try:
            payload = self._json_body()
            result = pipeline.save_training_config(
                dataset_name=str(payload.get("dataset_name") or "").strip(),
                dataset_version=payload.get("dataset_version") or "v1",
                model=str(payload.get("model") or "").strip(),
                run_group=str(payload.get("run_group") or "").strip(),
                run_name=str(payload.get("run_name") or "").strip(),
                image_size=int(payload.get("image_size") or 960),
                epochs=int(payload.get("epochs") or 150),
                batch=int(payload.get("batch") or 8),
                patience=payload.get("patience"),
                seed=int(payload.get("seed") or 42),
                device=payload.get("device", "cpu"),
                workers=payload.get("workers"),
                augmentation=payload.get("augmentation") or {},
                overwrite=bool(payload.get("overwrite")),
                selection_path=payload.get("selection_path"),
                config_name=payload.get("config_name"),
                configs_dir=pipeline.DEFAULT_TRAINING_CONFIGS_DIR,
                output_dir=pipeline.DEFAULT_RUNS_DIR,
            )
        except (ValueError, TypeError) as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._log_training_action("save_training_config", {
            "config_path": result["config_path"],
            "run_group": result["config"]["run_group"],
            "run_name": result["config"]["run_name"],
            "warnings": result.get("warnings") or [],
        })
        self._respond_json(result, status=201)

    def _handle_training_readiness(self) -> None:
        try:
            payload = self._json_body()
            result = pipeline.check_training_readiness(payload["selection_path"], payload["config_path"])
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        self._respond_json(result)

    def _handle_training_command_preview(self) -> None:
        try:
            payload = self._json_body()
            config = pipeline.load_json(payload["config_path"])
            missing = pipeline.config_missing_items(config)
            if missing:
                self._respond_json({"error": "; ".join(missing), "missing": missing}, status=400)
                return
            selection = pipeline.load_json(payload["selection_path"])
            dataset_dir = pipeline.repo_resolve(selection.get("export", {}).get("output_dir"))
            command_line = pipeline.build_yolo_train_command(config, dataset_dir)
            run_dir = pipeline.run_dir_for_config(config)
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        self._respond_json({
            "command_line": command_line,
            "run_dir": pipeline.repo_relative(run_dir),
            "data_yaml": pipeline.repo_relative(dataset_dir / "data.yaml"),
        })

    def _handle_start_training_run(self) -> None:
        try:
            payload = self._json_body()
            selection_path = payload["selection_path"]
            config_path = payload["config_path"]
            readiness = pipeline.check_training_readiness(selection_path, config_path)
            if not readiness["ready"]:
                self._log_training_action("start_training_blocked", {"config_path": str(config_path), "missing": readiness["missing"]})
                self._respond_json({"error": "Training is not ready", "missing": readiness["missing"]}, status=409)
                return
            result = pipeline.start_training_run(selection_path, config_path)
        except (KeyError, ValueError) as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._log_training_action("start_training", {"run_dir": result["run_dir"], "pid": result["pid"]})
        self._respond_json(result, status=202)

    def _handle_training_run_status(self, run_group: str, run_name: str, parsed) -> None:
        run_dir = pipeline.repo_resolve(pipeline.DEFAULT_RUNS_DIR) / run_group / run_name
        try:
            result = pipeline.training_status(run_dir)
        except ValueError as exc:
            self._respond_json({"error": str(exc)}, status=404)
            return
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._respond_json(result)

    def _handle_stop_training_run(self, run_group: str, run_name: str) -> None:
        run_dir = pipeline.repo_resolve(pipeline.DEFAULT_RUNS_DIR) / run_group / run_name
        try:
            result = pipeline.stop_training_run(run_dir)
        except ValueError as exc:
            self._respond_json({"error": str(exc)}, status=404)
            return
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._log_training_action("stop_training", {"run_dir": result["run_dir"], "status": result["status"]})
        self._respond_json(result)

    def _handle_list_active_runs(self, parsed) -> None:
        try:
            runs = pipeline.list_training_runs(pipeline.DEFAULT_RUNS_DIR, active_only=not query_bool(parsed, "all", False))
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._respond_json({"runs": runs})

    def _handle_run_test_split(self, run_group: str, run_name: str) -> None:
        run_dir = pipeline.repo_resolve(pipeline.DEFAULT_RUNS_DIR) / run_group / run_name
        try:
            payload = self._json_body()
            result = pipeline.run_test_split(
                run_dir,
                dataset_dir=payload.get("dataset_dir"),
                device=payload.get("device", "cpu"),
                image_size=payload.get("image_size"),
            )
        except (ValueError, FileNotFoundError) as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._log_training_action("run_test_split", {"run_dir": result["run_dir"]})
        self._respond_json(result)

    def _handle_write_evaluation_report(self) -> None:
        try:
            payload = self._json_body()
            result = pipeline.build_run_group_evaluation_report(
                run_group=str(payload.get("run_group") or "").strip(),
                runs_dir=pipeline.DEFAULT_RUNS_DIR,
                promoted_run_name=payload.get("promoted_run_name"),
                promotion_reason=payload.get("promotion_reason"),
            )
        except ValueError as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._log_training_action("write_evaluation_report", {
            "run_group": result["run_group"],
            "report_path": result["report_path"],
            "run_count": result["run_count"],
        })
        self._respond_json(result, status=201)

    def _handle_promote_weight(self) -> None:
        payload: dict = {}
        try:
            payload = self._json_body()
            result = pipeline.promote_weight(
                run_group=str(payload.get("run_group") or "").strip(),
                run_name=str(payload.get("run_name") or "").strip(),
                target_name=str(payload.get("target_name") or "").strip(),
                recreation_doc_updated=bool(payload.get("recreation_doc_updated")),
                overwrite=bool(payload.get("overwrite")),
                runs_dir=pipeline.DEFAULT_RUNS_DIR,
                model_weights_dir=pipeline.DEFAULT_MODEL_WEIGHTS_DIR,
            )
        except ValueError as exc:
            self._log_training_action("promote_weight_refused", {
                "run_group": str(payload.get("run_group") or ""),
                "target_name": str(payload.get("target_name") or ""),
                "error": str(exc),
            })
            self._respond_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=500)
            return
        self._log_training_action("promote_weight", {
            "run_group": result["run_group"],
            "run_name": result["run_name"],
            "promoted_path": result["promoted_path"],
        })
        self._respond_json(result, status=201)

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

    def _serve_static_file(self, relative_path: str) -> None:
        cleaned = relative_path.lstrip("/")
        static_root = STATIC_DIR.resolve()
        try:
            resolved = (static_root / cleaned).resolve()
        except OSError:
            resolved = None
        if resolved is None or not resolved.is_file() or static_root not in resolved.parents:
            self._respond_json({"error": "static file not found"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(resolved))[0] or "application/octet-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(resolved.read_bytes())

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
