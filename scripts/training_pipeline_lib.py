from __future__ import annotations

import hashlib
import csv
import copy
import json
import math
import os
import platform
import random
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from . import visual_review_lib as review_lib
    from .workspace import current_model, set_current_model, model_selection_status, restore_default_model, default_manifest, WORKSPACE_DIR, exclusive_lock, contained
    from . import process_support
except ImportError:
    import visual_review_lib as review_lib
    from workspace import current_model, set_current_model, model_selection_status, restore_default_model, default_manifest, WORKSPACE_DIR, exclusive_lock, contained
    import process_support

try:
    from .dataset_splits import GroupStats, assign_groups_to_splits, check_no_group_overlap, group_key, parse_source_name
    from . import build_clean_sts_dataset as clean_dataset_lib
except ImportError:
    from dataset_splits import GroupStats, assign_groups_to_splits, check_no_group_overlap, group_key, parse_source_name  # type: ignore
    import build_clean_sts_dataset as clean_dataset_lib  # type: ignore


ROOT_DIR = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")
SUPPORTED_IMPORT_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_IMPORT_MANIFEST_PATH = WORKSPACE_DIR / "manifests" / "training_import_manifest.json"
DEFAULT_IMPORT_CACHE_DIR = WORKSPACE_DIR / "cache" / "imported_images"
DEFAULT_ALBUMS_DIR = WORKSPACE_DIR / "albums"
DEFAULT_IMPORT_JOBS_DIR = WORKSPACE_DIR / "reports" / "import_jobs"
DEFAULT_REVIEW_QUEUE_PRUNE_REPORTS_DIR = WORKSPACE_DIR / "reports" / "review_queue_prune"
DEFAULT_IMAGE_REMOVAL_REPORTS_DIR = WORKSPACE_DIR / "reports" / "image_removals"
DEFAULT_DATASET_SELECTIONS_DIR = WORKSPACE_DIR / "dataset_selections"
DEFAULT_DATASETS_DIR = WORKSPACE_DIR / "datasets"
DEFAULT_AUTO_DETECT_CONFIDENCE = 0.25
DEFAULT_AUTO_DETECT_IOU = 0.5
RGB_ORIGINAL_RENDER_PROFILE = {"kind": "rgb_original"}
TRUE_COLOR_RENDER_PROFILE = {
    "kind": "true_color",
    "bands": [1, 2, 3],
    "normalization": "current_loader_default",
    "output_colorspace": "RGB",
}
FALSE_COLOR_RENDER_PROFILE = {
    "kind": "false_color",
    "bands": [4, 3, 2],
    "normalization": "current_loader_default",
    "output_colorspace": "RGB",
}
DOCUMENTED_IMPORT_STATUSES = {"imported", "already_imported", "regenerated_cache", "failed"}
REQUIRED_IMPORT_ENTRY_FIELDS = {
    "schema_version",
    "image_key",
    "original_filename",
    "source_path",
    "browser_relative_path",
    "source_collection",
    "source_size_bytes",
    "source_mtime",
    "source_content_hash",
    "width",
    "height",
    "cache_path",
    "cache_width",
    "cache_height",
    "cache_content_hash",
    "render_profile",
    "loader",
    "import_status",
    "imported_at",
    "updated_at",
    "warnings",
}
REVIEW_LABEL_TO_CLASS = {"ship_non_sts": 0, "ships_sts": 1}
CLASS_ID_TO_NAME = {0: "ship", 1: "ships"}
RECOGNIZED_RESULTS_COLUMNS = {
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
}
REQUIRED_TRAINING_CONFIG_FIELDS = {
    "dataset_name": "dataset name/version",
    "dataset_version": "dataset name/version",
    "model": "model or starting weights",
    "image_size": "image size",
    "epochs": "epochs",
    "batch": "batch",
    "device": "device",
    "seed": "seed",
    "run_name": "output run name",
    "augmentation": "augmentation settings",
}


def repo_resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def repo_relative(path: str | Path) -> str:
    candidate = Path(path).resolve()
    try:
        return str(candidate.relative_to(ROOT_DIR))
    except ValueError:
        return str(candidate)


def slugify(value: str, fallback: str = "untitled") -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "").strip()).strip("._-").lower()
    return slug or fallback


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def load_json(path: str | Path) -> dict[str, Any]:
    target = repo_resolve(path)
    payload = json.loads(target.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {target}")
    return payload


def load_json_or_dict(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return load_json(value)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = contained(repo_resolve(path), WORKSPACE_DIR)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
            temp = Path(handle.name)
            handle.write(content)
        temp.replace(target)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def load_import_manifest(manifest_path: str | Path = DEFAULT_IMPORT_MANIFEST_PATH) -> dict[str, Any]:
    target = repo_resolve(manifest_path)
    now = utcnow_iso_for_training()
    if not target.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": "training_import",
            "created_at": now,
            "updated_at": now,
            "cache_dir": repo_relative(DEFAULT_IMPORT_CACHE_DIR),
            "images": {},
        }
    if not target.is_file():
        raise ValueError(f"Import manifest path is not a file: {target}")
    manifest = json.loads(target.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"Import manifest must be a JSON object: {target}")
    if int(manifest.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ValueError("Import manifest must use schema_version 1")
    images = manifest.setdefault("images", {})
    if not isinstance(images, dict):
        raise ValueError("Import manifest images field must be a JSON object")
    manifest.setdefault("schema_version", SCHEMA_VERSION)
    manifest.setdefault("manifest_type", "training_import")
    manifest.setdefault("created_at", now)
    manifest.setdefault("updated_at", now)
    manifest.setdefault("cache_dir", repo_relative(DEFAULT_IMPORT_CACHE_DIR))
    return manifest


def save_import_manifest(manifest_path: str | Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utcnow_iso_for_training()
    write_json(manifest_path, manifest)


def append_jsonl(path: str | Path, event: dict[str, Any]) -> None:
    target = repo_resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def default_albums_dir_for_manifest(manifest_path: str | Path) -> Path:
    manifest_file = repo_resolve(manifest_path)
    if manifest_file == repo_resolve(DEFAULT_IMPORT_MANIFEST_PATH):
        return DEFAULT_ALBUMS_DIR
    return manifest_file.parent.parent / "albums"


def default_import_jobs_dir_for_manifest(manifest_path: str | Path) -> Path:
    manifest_file = repo_resolve(manifest_path)
    if manifest_file == repo_resolve(DEFAULT_IMPORT_MANIFEST_PATH):
        return DEFAULT_IMPORT_JOBS_DIR
    return manifest_file.parent.parent / "reports" / "import_jobs"


def new_album_id(albums_dir: str | Path) -> str:
    root = repo_resolve(albums_dir)
    while True:
        album_id = "alb_" + secrets.token_hex(5)
        if not (root / f"{album_id}.json").exists():
            return album_id


def new_import_job_id(import_jobs_dir: str | Path) -> str:
    root = repo_resolve(import_jobs_dir)
    while True:
        job_id = "imp_" + secrets.token_hex(6)
        if not (root / f"{job_id}.summary.json").exists() and not (root / f"{job_id}.jsonl").exists():
            return job_id


def load_album_record(albums_dir: str | Path, album_id: str) -> dict[str, Any]:
    validate_run_segment(album_id, "album_id")
    target = repo_resolve(Path(albums_dir) / f"{album_id}.json")
    if not target.is_file():
        raise ValueError(f"Album id not found: {album_id}")
    album = json.loads(target.read_text())
    if not isinstance(album, dict):
        raise ValueError(f"Album record must be a JSON object: {target}")
    return album


def derive_album_name(sources: list[str | Path], source_kind: str, source_display_name: str | None) -> str:
    if source_kind == "browser_upload":
        if source_display_name:
            return Path(source_display_name).name or source_display_name
        return "Untitled " + utcnow_iso_for_training().replace("T", " ")[:16]
    if sources:
        first = repo_resolve(sources[0])
        return first.name or str(first)
    return "Untitled " + utcnow_iso_for_training().replace("T", " ")[:16]


def infer_import_source_kind(sources: list[str | Path]) -> str:
    if len(sources) == 1:
        source = repo_resolve(sources[0])
        if source.is_file():
            return "local_file"
        if source.is_dir():
            return "local_folder"
    return "local_folder"


def default_source_display_name(sources: list[str | Path], source_kind: str) -> str | None:
    if source_kind == "browser_upload":
        return None
    if len(sources) == 1:
        return str(repo_resolve(sources[0]))
    return None


def append_unique(values: list[Any], value: Any) -> bool:
    if value in (None, "") or value in values:
        return False
    values.append(value)
    return True


def add_album_id_to_entry(entry: dict[str, Any], album_id: str) -> bool:
    album_ids = entry.setdefault("album_ids", [])
    if not isinstance(album_ids, list):
        album_ids = []
        entry["album_ids"] = album_ids
    return append_unique(album_ids, album_id)


def browser_relative_path_for(source_path: Path, roots: list[Path], source_kind: str) -> str | None:
    if source_kind != "browser_upload":
        return None
    for root in roots:
        try:
            return source_path.relative_to(root).as_posix()
        except ValueError:
            continue
    return source_path.name


def album_image_record(
    entry: dict[str, Any] | None,
    item: dict[str, Any],
    *,
    import_status: str,
    member_status: str,
) -> dict[str, Any]:
    return {
        "image_key": item.get("image_key") or (entry or {}).get("image_key"),
        "original_filename": (entry or {}).get("original_filename") or Path(str(item.get("source") or "")).name or None,
        "browser_relative_path": (entry or {}).get("browser_relative_path"),
        "source_path": (entry or {}).get("source_path") or item.get("source"),
        "cache_path": (entry or {}).get("cache_path") or item.get("cache_path"),
        "review_stub_path": item.get("review_stub_path"),
        "import_status": import_status,
        "member_status": member_status,
        "reason": item.get("reason"),
    }


def merge_album_images(existing_images: list[dict[str, Any]], new_images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for image in existing_images:
        key = image.get("image_key")
        if key:
            positions[key] = len(merged)
        merged.append(image)
    for image in new_images:
        key = image.get("image_key")
        if key and key in positions:
            merged[positions[key]] = image
        else:
            if key:
                positions[key] = len(merged)
            merged.append(image)
    return merged


def write_album_index(albums_dir: str | Path, album: dict[str, Any]) -> None:
    index_path = repo_resolve(Path(albums_dir) / "index.json")
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if not isinstance(index, dict):
            raise ValueError(f"Album index must be a JSON object: {index_path}")
    else:
        index = {"schema_version": SCHEMA_VERSION, "albums": []}
    albums = index.setdefault("albums", [])
    if not isinstance(albums, list):
        albums = []
        index["albums"] = albums
    latest_job_id = None
    job_ids = album.get("import_job_ids") or []
    if job_ids:
        latest_job_id = job_ids[-1]
    summary = {
        "album_id": album.get("album_id"),
        "album_name": album.get("album_name"),
        "album_description": album.get("album_description"),
        "source_display_name": album.get("source_display_name"),
        "source_kind": album.get("source_kind"),
        "image_count": len([image for image in album.get("images") or [] if image.get("image_key")]),
        "counts": album.get("counts") or {},
        "created_at": album.get("created_at"),
        "updated_at": album.get("updated_at"),
        "latest_import_job_id": latest_job_id,
    }
    replaced = False
    for idx, current in enumerate(albums):
        if current.get("album_id") == summary["album_id"]:
            albums[idx] = summary
            replaced = True
            break
    if not replaced:
        albums.append(summary)
    index["updated_at"] = utcnow_iso_for_training()
    write_json(index_path, index)


EMPTY_ALBUM_COUNTS = {
    "uploaded": 0,
    "discovered": 0,
    "imported": 0,
    "skipped": 0,
    "failed": 0,
    "members_added_imported": 0,
    "members_added_existing": 0,
    "members_already_present": 0,
    "members_failed": 0,
    "review_stubs_created": 0,
    "review_stubs_existing": 0,
    "review_stubs_skipped": 0,
    "auto_detect_created_review": 0,
    "auto_detect_updated_existing_review": 0,
    "auto_detect_protected_existing_review": 0,
    "auto_detect_skipped": 0,
}


def review_completion_map(reviewer: str, reviews_dir: str | Path) -> dict[str, str]:
    completion: dict[str, str] = {}
    try:
        reviews = review_lib.list_review_documents(reviewer, reviews_dir)
    except Exception:
        return completion
    for review in reviews:
        key = str(review.get("image_key") or "")
        if not key:
            continue
        completion[key] = str(review.get("review_status") or "overview_1")
    return completion


def album_review_counts(album_images: list[dict[str, Any]], completion: dict[str, str]) -> dict[str, int]:
    completed = 0
    pending = 0
    total_with_review = 0
    for image in album_images or []:
        key = str((image or {}).get("image_key") or "")
        if not key:
            continue
        status = completion.get(key)
        if status is None:
            continue
        total_with_review += 1
        if status == "completed":
            completed += 1
        else:
            pending += 1
    return {
        "review_completed": completed,
        "review_pending": pending,
        "review_total": total_with_review,
    }


def list_albums_summary(
    albums_dir: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_IMPORT_MANIFEST_PATH,
    reviews_dir: str | Path = review_lib.DEFAULT_REVIEWS_DIR,
    reviewer: str = review_lib.DEFAULT_REVIEWER,
) -> list[dict[str, Any]]:
    album_root = repo_resolve(albums_dir)
    completion = review_completion_map(reviewer, reviews_dir)
    summaries: list[dict[str, Any]] = []
    index_path = album_root / "index.json"
    if index_path.is_file():
        try:
            index = load_json(index_path)
            index_albums = [album for album in index.get("albums") or [] if isinstance(album, dict) and album.get("album_id")]
        except Exception:
            index_albums = []
    else:
        index_albums = []
    index_by_id = {str(album.get("album_id")): album for album in index_albums}
    seen: set[str] = set()
    for album_path in sorted(album_root.glob("alb_*.json")):
        try:
            album = load_json(album_path)
        except Exception:
            continue
        album_id = str(album.get("album_id") or album_path.stem)
        if album_id in seen:
            continue
        seen.add(album_id)
        review_counts = album_review_counts(album.get("images") or [], completion)
        summary_from_index = index_by_id.get(album_id) or {}
        counts = album.get("counts") or summary_from_index.get("counts") or {}
        latest_job_id = None
        job_ids = album.get("import_job_ids") or []
        if job_ids:
            latest_job_id = job_ids[-1]
        elif "latest_import_job_id" in summary_from_index:
            latest_job_id = summary_from_index.get("latest_import_job_id")
        summaries.append({
            "album_id": album_id,
            "album_name": album.get("album_name") or album_id,
            "album_description": album.get("album_description"),
            "source_display_name": album.get("source_display_name"),
            "source_kind": album.get("source_kind"),
            "image_count": len([image for image in album.get("images") or [] if image.get("image_key")]),
            "counts": counts,
            "review_counts": review_counts,
            "created_at": album.get("created_at"),
            "updated_at": album.get("updated_at"),
            "latest_import_job_id": latest_job_id,
        })
    summaries.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return summaries


def create_empty_album(
    *,
    albums_dir: str | Path | None = None,
    manifest_path: str | Path = DEFAULT_IMPORT_MANIFEST_PATH,
    cache_dir: str | Path = DEFAULT_IMPORT_CACHE_DIR,
    reviews_dir: str | Path = review_lib.DEFAULT_REVIEWS_DIR,
    album_name: str,
    album_description: str | None = None,
    source_display_name: str | None = None,
    reviewer: str = review_lib.DEFAULT_REVIEWER,
) -> dict[str, Any]:
    if not album_name or not str(album_name).strip():
        raise ValueError("album_name is required")
    manifest_file = repo_resolve(manifest_path)
    album_root = repo_resolve(albums_dir or default_albums_dir_for_manifest(manifest_file))
    album_root.mkdir(parents=True, exist_ok=True)
    album_id = new_album_id(album_root)
    now = utcnow_iso_for_training()
    album_record = {
        "schema_version": SCHEMA_VERSION,
        "album_id": album_id,
        "album_name": str(album_name).strip(),
        "album_description": (album_description or None),
        "source_display_name": source_display_name,
        "source_kind": "manual_album",
        "import_job_ids": [],
        "created_at": now,
        "updated_at": now,
        "created_by": reviewer,
        "reviewer": reviewer,
        "manifest_path": repo_relative(manifest_file),
        "cache_dir": repo_relative(cache_dir),
        "reviews_dir": repo_relative(reviews_dir),
        "options": {},
        "counts": dict(EMPTY_ALBUM_COUNTS),
        "images": [],
    }
    album_path = album_root / f"{album_id}.json"
    write_json(album_path, album_record)
    write_album_index(album_root, album_record)
    return album_record


def add_existing_image_keys_to_album(
    *,
    albums_dir: str | Path | None = None,
    manifest_path: str | Path = DEFAULT_IMPORT_MANIFEST_PATH,
    album_id: str,
    image_keys: list[str],
    reviewer: str = review_lib.DEFAULT_REVIEWER,
) -> dict[str, Any]:
    if not album_id:
        raise ValueError("album_id is required")
    manifest_file = repo_resolve(manifest_path)
    album_root = repo_resolve(albums_dir or default_albums_dir_for_manifest(manifest_file))
    album = load_album_record(album_root, album_id)
    manifest = load_import_manifest(manifest_file)
    images_map = manifest.get("images", {}) or {}
    existing_album_images = album.get("images") or []
    existing_keys = {image.get("image_key") for image in existing_album_images if image.get("image_key")}
    new_album_images: list[dict[str, Any]] = []
    added_existing = 0
    already_present = 0
    failed: list[dict[str, Any]] = []
    requested = [str(key) for key in (image_keys or []) if str(key).strip()]
    manifest_changed = False
    for image_key in requested:
        entry = images_map.get(image_key)
        if not entry:
            failed.append({"image_key": image_key, "reason": "not in import manifest"})
            continue
        if image_key in existing_keys:
            already_present += 1
            new_album_images.append(album_image_record(entry, {"image_key": image_key}, import_status=entry.get("import_status") or "imported", member_status="already_present"))
            continue
        if add_album_id_to_entry(entry, album_id):
            manifest_changed = True
        new_album_images.append(album_image_record(entry, {"image_key": image_key}, import_status=entry.get("import_status") or "imported", member_status="added_existing"))
        existing_keys.add(image_key)
        added_existing += 1
    counts = album.setdefault("counts", dict(EMPTY_ALBUM_COUNTS))
    for key, default in EMPTY_ALBUM_COUNTS.items():
        counts.setdefault(key, default)
    counts["members_added_existing"] = int(counts.get("members_added_existing") or 0) + added_existing
    counts["members_already_present"] = int(counts.get("members_already_present") or 0) + already_present
    counts["members_failed"] = int(counts.get("members_failed") or 0) + len(failed)
    album["images"] = merge_album_images(existing_album_images, new_album_images)
    album["updated_at"] = utcnow_iso_for_training()
    album_path = album_root / f"{album_id}.json"
    write_json(album_path, album)
    write_album_index(album_root, album)
    if manifest_changed:
        save_import_manifest(manifest_file, manifest)
    return {
        "album_id": album_id,
        "album_path": repo_relative(album_path),
        "manifest_path": repo_relative(manifest_file),
        "requested_count": len(requested),
        "members_added_existing": added_existing,
        "members_already_present": already_present,
        "members_failed": len(failed),
        "failed": failed,
        "image_count": len([image for image in album["images"] if image.get("image_key")]),
    }


def rename_album(
    *,
    albums_dir: str | Path | None = None,
    manifest_path: str | Path = DEFAULT_IMPORT_MANIFEST_PATH,
    album_id: str,
    album_name: str | None = None,
    album_description: str | None = None,
) -> dict[str, Any]:
    if not album_id:
        raise ValueError("album_id is required")
    if album_name is None and album_description is None:
        raise ValueError("rename_album requires album_name and/or album_description")
    manifest_file = repo_resolve(manifest_path)
    album_root = repo_resolve(albums_dir or default_albums_dir_for_manifest(manifest_file))
    album = load_album_record(album_root, album_id)
    if album_name is not None:
        cleaned = str(album_name).strip()
        if not cleaned:
            raise ValueError("album_name must not be blank")
        album["album_name"] = cleaned
    if album_description is not None:
        cleaned_description = str(album_description).strip()
        album["album_description"] = cleaned_description or None
    album["updated_at"] = utcnow_iso_for_training()
    album_path = album_root / f"{album_id}.json"
    write_json(album_path, album)
    write_album_index(album_root, album)
    return album


def remove_image_keys_from_album(
    *,
    albums_dir: str | Path | None = None,
    manifest_path: str | Path = DEFAULT_IMPORT_MANIFEST_PATH,
    album_id: str,
    image_keys: list[str],
) -> dict[str, Any]:
    if not album_id:
        raise ValueError("album_id is required")
    manifest_file = repo_resolve(manifest_path)
    album_root = repo_resolve(albums_dir or default_albums_dir_for_manifest(manifest_file))
    album = load_album_record(album_root, album_id)
    requested = [str(key) for key in (image_keys or []) if str(key).strip()]
    requested_set = set(requested)
    existing_album_images = album.get("images") or []
    existing_keys = {image.get("image_key") for image in existing_album_images if image.get("image_key")}
    not_in_album = sorted(requested_set - existing_keys)
    removed_keys = sorted(requested_set & existing_keys)
    album["images"] = [image for image in existing_album_images if image.get("image_key") not in requested_set]
    album["updated_at"] = utcnow_iso_for_training()
    counts = album.setdefault("counts", dict(EMPTY_ALBUM_COUNTS))
    for key, default in EMPTY_ALBUM_COUNTS.items():
        counts.setdefault(key, default)
    manifest = load_import_manifest(manifest_file)
    manifest_changed = False
    for image_key in removed_keys:
        entry = manifest.get("images", {}).get(image_key)
        if not entry:
            continue
        ids = entry.get("album_ids") or []
        if album_id in ids:
            entry["album_ids"] = [value for value in ids if value != album_id]
            manifest_changed = True
    album_path = album_root / f"{album_id}.json"
    write_json(album_path, album)
    write_album_index(album_root, album)
    if manifest_changed:
        save_import_manifest(manifest_file, manifest)
    return {
        "album_id": album_id,
        "album_path": repo_relative(album_path),
        "manifest_path": repo_relative(manifest_file),
        "requested_count": len(requested),
        "removed_count": len(removed_keys),
        "not_in_album_count": len(not_in_album),
        "removed_image_keys": removed_keys,
        "not_in_album_image_keys": not_in_album,
        "image_count": len([image for image in album["images"] if image.get("image_key")]),
    }


def load_album_detail(
    albums_dir: str | Path,
    album_id: str,
    *,
    manifest_path: str | Path = DEFAULT_IMPORT_MANIFEST_PATH,
    reviews_dir: str | Path = review_lib.DEFAULT_REVIEWS_DIR,
    reviewer: str = review_lib.DEFAULT_REVIEWER,
) -> dict[str, Any]:
    album_root = repo_resolve(albums_dir)
    album = load_album_record(album_root, album_id)
    completion = review_completion_map(reviewer, reviews_dir)
    review_counts = album_review_counts(album.get("images") or [], completion)
    image_keys = [str(image.get("image_key")) for image in album.get("images") or [] if image.get("image_key")]
    return {
        "album": album,
        "album_path": repo_relative(album_root / f"{album_id}.json"),
        "image_keys": image_keys,
        "image_count": len(image_keys),
        "review_counts": review_counts,
    }


def default_render_profile(kind: str) -> dict[str, Any]:
    profiles = {
        "rgb_original": RGB_ORIGINAL_RENDER_PROFILE,
        "true_color": TRUE_COLOR_RENDER_PROFILE,
        "false_color": FALSE_COLOR_RENDER_PROFILE,
    }
    if kind not in profiles:
        raise ValueError(f"Unknown render profile kind: {kind!r}")
    return copy.deepcopy(profiles[kind])


def resolve_import_render_profile(render_profile_value: str, source_path: str | Path) -> dict[str, Any]:
    value = str(render_profile_value or "auto")
    source_suffix = Path(source_path).suffix.lower()
    if value == "auto":
        if source_suffix in {".png", ".jpg", ".jpeg"}:
            profile = default_render_profile("rgb_original")
        elif source_suffix in {".tif", ".tiff"}:
            profile = default_render_profile("true_color")
        else:
            raise ValueError(f"Unsupported image suffix for auto render profile: {source_suffix}")
    else:
        profile_path = repo_resolve(value)
        if profile_path.is_file():
            profile = load_json(profile_path)
        else:
            profile = default_render_profile(value)
    review_lib.validate_render_profile(profile)
    return profile


def is_hidden_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts if part not in {".", ".."})


def scan_import_source(source: str | Path, *, recursive: bool) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    source_path = repo_resolve(source)
    if not source_path.exists():
        return [], {
            "image_key": None,
            "source": str(source_path),
            "cache_path": None,
            "render_profile_kind": None,
            "status": "failed",
            "reason": "Source path not found",
        }
    if source_path.is_file():
        if source_path.name.startswith(".") or source_path.suffix.lower() not in SUPPORTED_IMPORT_SUFFIXES:
            return [], None
        return [{"source_path": source_path, "source_collection": str(source_path)}], None
    if not source_path.is_dir():
        return [], {
            "image_key": None,
            "source": str(source_path),
            "cache_path": None,
            "render_profile_kind": None,
            "status": "failed",
            "reason": "Source path is neither a file nor a directory",
        }

    pattern = "**/*" if recursive else "*"
    files: list[dict[str, Any]] = []
    for candidate in sorted(source_path.glob(pattern)):
        try:
            relative = candidate.relative_to(source_path)
        except ValueError:
            relative = Path(candidate.name)
        if is_hidden_path(relative):
            continue
        if candidate.is_dir():
            continue
        if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_IMPORT_SUFFIXES:
            continue
        files.append({"source_path": candidate, "source_collection": str(source_path)})
    return files, None


def discover_import_files(sources: list[str | Path], *, recursive: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for source in sources:
        discovered, failure = scan_import_source(source, recursive=recursive)
        if failure is not None:
            failed.append(failure)
        for candidate in discovered:
            path = repo_resolve(candidate["source_path"])
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append({"source_path": resolved, "source_collection": candidate["source_collection"]})
    return files, failed


def render_profile_bands(render_profile: dict[str, Any]) -> list[int]:
    bands = render_profile.get("bands")
    if not isinstance(bands, list) or not bands:
        raise ValueError("render_profile bands must be a non-empty list")
    parsed = [int(value) for value in bands[:3]]
    while len(parsed) < 3:
        parsed.append(parsed[-1])
    return parsed


def load_import_image(source_path: str | Path, render_profile: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    path = repo_resolve(source_path)
    kind = str(render_profile.get("kind"))
    if kind == "rgb_original":
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return rgb.copy(), {"loader": "pillow", "mode": image.mode, "width": rgb.width, "height": rgb.height}

    rasterio = review_lib.optional_rasterio()
    if rasterio is not None:
        with rasterio.open(path) as dataset:
            band_count = max(int(dataset.count), 1)
            indexes = [max(1, min(index, band_count)) for index in render_profile_bands(render_profile)]
            data = dataset.read(indexes)
            import numpy as np

            if data.ndim == 2:
                data = np.stack([data, data, data], axis=0)
            normalized = np.stack([review_lib.normalize_band_to_uint8(data[index]) for index in range(3)], axis=-1)
            image = Image.fromarray(normalized, mode="RGB")
            metadata = {
                "loader": "rasterio",
                "dtype": str(data.dtype),
                "band_count": band_count,
                "bands": indexes,
                "width": int(dataset.width),
                "height": int(dataset.height),
                "bounds": {
                    "left": float(dataset.bounds.left),
                    "bottom": float(dataset.bounds.bottom),
                    "right": float(dataset.bounds.right),
                    "top": float(dataset.bounds.top),
                },
            }
            return image, metadata

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return rgb.copy(), {
            "loader": "pillow",
            "mode": image.mode,
            "width": rgb.width,
            "height": rgb.height,
            "render_profile_approximated": kind,
        }


def render_import_cache(source_path: str | Path, cache_path: str | Path, render_profile: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    output = repo_resolve(cache_path)
    image, metadata = load_import_image(source_path, render_profile)
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, format="PNG")
    return {
        "path": repo_relative(output),
        "width": int(image.width),
        "height": int(image.height),
        "cache_content_hash": None if dry_run else file_hash(output),
        "loader": metadata.get("loader"),
        "metadata": metadata,
        "rendered_at": utcnow_iso_for_training(),
    }


def import_manifest_entry(
    source_path: Path,
    cache_path: Path,
    render_profile: dict[str, Any],
    render_info: dict[str, Any],
    *,
    source_collection: str,
    source_kind: str,
    source_display_name: str | None,
    browser_relative_path: str | None,
    album_id: str | None,
    import_status: str,
    previous_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if import_status not in DOCUMENTED_IMPORT_STATUSES:
        raise ValueError(f"Unknown import status: {import_status}")
    now = utcnow_iso_for_training()
    source_hash = review_lib.source_content_hash(source_path)
    image_key = review_lib.image_key_for_identity_payload(
        {
            "source_content_hash": source_hash,
            "render_profile": render_profile,
        }
    )
    stat = source_path.stat()
    entry = {
        "schema_version": SCHEMA_VERSION,
        "image_key": image_key,
        "image_filename": f"{image_key}.png",
        "original_filename": source_path.name,
        "display_filename": source_path.name,
        "source_path": str(source_path),
        "browser_relative_path": browser_relative_path,
        "source_collection": source_collection,
        "source_display_name": source_display_name,
        "source_kind": source_kind,
        "date": None,
        "tile_row": None,
        "tile_col": None,
        "source_content_hash": source_hash,
        "source_size_bytes": stat.st_size,
        "source_mtime": stat.st_mtime,
        "width": int(render_info["width"]),
        "height": int(render_info["height"]),
        "cache_path": repo_relative(cache_path),
        "cache_width": int(render_info["width"]),
        "cache_height": int(render_info["height"]),
        "cache_content_hash": render_info.get("cache_content_hash"),
        "render_profile": render_profile,
        "loader": render_info.get("loader"),
        "import_status": import_status,
        "imported_at": (previous_entry or {}).get("imported_at") or now,
        "updated_at": now,
        "warnings": [],
        "render": {
            "loader": render_info.get("loader"),
            "rendered_at": render_info.get("rendered_at"),
            "output_format": "png",
            "metadata": render_info.get("metadata") or {},
        },
    }
    if previous_entry:
        for key in ("album_ids",):
            if key in previous_entry:
                entry[key] = copy.deepcopy(previous_entry[key])
    if album_id:
        add_album_id_to_entry(entry, album_id)
    return entry


def import_result_item(entry: dict[str, Any], status: str) -> dict[str, Any]:
    if status not in DOCUMENTED_IMPORT_STATUSES:
        raise ValueError(f"Unknown import status: {status}")
    return {
        "image_key": entry["image_key"],
        "source": entry.get("source_path") or entry.get("source"),
        "cache_path": entry["cache_path"],
        "render_profile_kind": entry.get("render_profile", {}).get("kind"),
        "status": status,
    }


def import_entry_for_review_stub(entry: dict[str, Any]) -> dict[str, Any]:
    adapted = dict(entry)
    adapted.setdefault("image_filename", f"{entry['image_key']}.png")
    adapted.setdefault("display_filename", entry.get("original_filename") or adapted["image_filename"])
    adapted.setdefault("date", None)
    adapted.setdefault("tile_row", None)
    adapted.setdefault("tile_col", None)
    return adapted


def attach_review_stub_result(
    item: dict[str, Any],
    entry: dict[str, Any],
    *,
    reviewer: str,
    reviews_dir: str | Path,
    skip: bool,
    dry_run: bool,
) -> dict[str, Any]:
    review_path = review_lib.review_file_path(entry["image_key"], reviewer, reviews_dir)
    item["review_stub_path"] = repo_relative(review_path)
    if skip or dry_run:
        item["review_stub_status"] = "skipped"
        return item
    if review_path.exists():
        item["review_stub_status"] = "existing"
        return item
    try:
        review_stub = review_lib.build_review_stub(import_entry_for_review_stub(entry), reviewer, merged_detections=[])
        review_lib.write_review_stub(review_doc=review_stub, reviewer=reviewer, reviews_dir=reviews_dir, overwrite=False)
        item["review_stub_status"] = "created"
    except Exception as exc:
        item["review_stub_status"] = "failed"
        item["review_stub_error"] = str(exc)
    return item


def normalize_auto_detect_model_paths(model_paths: dict[str, str | Path] | None = None) -> dict[str, str | Path]:
    paths = model_paths or {"current": current_model(WORKSPACE_DIR / "model_weights")}
    return {str(name): path for name, path in paths.items() if str(name).strip() and str(path).strip()}


def load_auto_detect_models(model_paths: dict[str, str | Path] | None = None) -> dict[str, Any]:
    resolved_paths = normalize_auto_detect_model_paths(model_paths)
    if not resolved_paths:
        raise ValueError("At least one YOLO model path is required for auto detection")
    return review_lib.load_yolo_models(resolved_paths)


def review_safe_for_auto_detect_update(review: dict[str, Any]) -> bool:
    if review.get("review_status") not in {None, "overview_1", "overview_2"}:
        return False
    for detection in review.get("detections") or []:
        if detection.get("source_type") == "manual":
            return False
        if detection.get("reviewed"):
            return False
        if detection.get("review_label") not in {None, ""}:
            return False
    return True


def auto_detect_review_detections(
    entry: dict[str, Any],
    *,
    reviewer: str,
    models: dict[str, Any],
    confidence: float = DEFAULT_AUTO_DETECT_CONFIDENCE,
    iou: float = DEFAULT_AUTO_DETECT_IOU,
) -> dict[str, Any]:
    raw_detections = review_lib.run_model_predictions(entry["cache_path"], models=models, conf_threshold=confidence)
    merged_detections = review_lib.merge_detections_class_aware(raw_detections, iou_threshold=iou)
    review_stub = review_lib.build_review_stub(import_entry_for_review_stub(entry), reviewer, merged_detections)
    return {
        "detections": review_stub["detections"],
        "raw_detection_count": len(raw_detections),
        "merged_detection_count": len(merged_detections),
    }


def attach_review_stub_with_optional_auto_detect(
    item: dict[str, Any],
    entry: dict[str, Any],
    *,
    reviewer: str,
    reviews_dir: str | Path,
    skip: bool,
    dry_run: bool,
    auto_detect_models: dict[str, Any] | None,
    auto_detect_confidence: float = DEFAULT_AUTO_DETECT_CONFIDENCE,
    auto_detect_iou: float = DEFAULT_AUTO_DETECT_IOU,
) -> dict[str, Any]:
    review_path = review_lib.review_file_path(entry["image_key"], reviewer, reviews_dir)
    item["review_stub_path"] = repo_relative(review_path)
    item["auto_detect_status"] = "skipped"
    item["auto_detection_count"] = 0
    item["auto_raw_detection_count"] = 0
    if skip or dry_run:
        item["review_stub_status"] = "skipped"
        return item

    try:
        existing_review = None
        if review_path.exists():
            item["review_stub_status"] = "existing"
            existing_review = review_lib.load_review_document(review_path)
            if auto_detect_models is not None and not review_safe_for_auto_detect_update(existing_review):
                item["auto_detect_status"] = "protected_existing_review"
                return item

        detection_result = None
        if auto_detect_models is not None:
            detection_result = auto_detect_review_detections(
                entry,
                reviewer=reviewer,
                models=auto_detect_models,
                confidence=auto_detect_confidence,
                iou=auto_detect_iou,
            )
            item["auto_detection_count"] = detection_result["merged_detection_count"]
            item["auto_raw_detection_count"] = detection_result["raw_detection_count"]

        if existing_review is not None:
            if detection_result is None:
                return item
            review = existing_review
            review["detections"] = detection_result["detections"]
            review["detail_cursor"] = 0 if detection_result["detections"] else None
            review["review_status"] = "overview_1"
            review_lib.save_review_document(review, review_path)
            item["auto_detect_status"] = "updated_existing_review"
            return item

        merged_detections = detection_result["detections"] if detection_result else []
        review_stub = review_lib.build_review_stub(import_entry_for_review_stub(entry), reviewer, [])
        review_stub["detections"] = merged_detections
        review_stub["detail_cursor"] = 0 if merged_detections else None
        review_lib.write_review_stub(review_doc=review_stub, reviewer=reviewer, reviews_dir=reviews_dir, overwrite=False)
        item["review_stub_status"] = "created"
        item["auto_detect_status"] = "created_review" if detection_result is not None else "skipped"
    except Exception as exc:
        item["review_stub_status"] = "failed"
        item["review_stub_error"] = str(exc)
    return item


def review_stub_failure_item(item: dict[str, Any]) -> dict[str, Any]:
    failed_item = dict(item)
    failed_item["status"] = "failed"
    failed_item["reason"] = f"Review stub failed: {item.get('review_stub_error') or 'unknown error'}"
    return failed_item


def unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def review_archivable_status(review: dict[str, Any]) -> tuple[bool, str | None]:
    image_key = str(review.get("image_key") or "").strip()
    if not image_key:
        return False, "missing image_key"
    status = review.get("review_status", "overview_1")
    if status != "overview_1":
        return False, f"review_status is {status!r}"
    if review.get("completed_at") or review.get("reviewed_at") or review.get("image_reviewed") is True:
        return False, "image-level review completion metadata is present"
    detections = review.get("detections", [])
    if not isinstance(detections, list):
        return False, "detections is not a list"
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            return False, f"detection {index} is not an object"
        if detection.get("reviewed") is True:
            return False, f"detection {index} has reviewed=true"
        if detection.get("review_label") is not None:
            return False, f"detection {index} has review_label"
        if detection.get("source_type") == "manual":
            return False, f"detection {index} is manual"
    return True, None


def review_archivable_for_image_key(image_key: str, reviewer: str, reviews_dir: str | Path) -> tuple[bool, str | None, Path]:
    review_path = review_lib.review_file_path(image_key, reviewer, reviews_dir)
    if not review_path.exists():
        return False, "review file missing", review_path
    try:
        review = review_lib.load_review_document(review_path)
    except Exception as exc:
        return False, f"review file unreadable: {exc}", review_path
    review_image_key = str(review.get("image_key") or "")
    if review_image_key != image_key:
        return False, f"review image_key mismatch: {review_image_key or 'missing'}", review_path
    archivable, reason = review_archivable_status(review)
    return archivable, reason, review_path


def latest_archived_review_path(image_key: str, reviewer: str, reviews_dir: str | Path) -> Path | None:
    archive_root = repo_resolve(reviews_dir) / "_archived_unreviewed"
    if not image_key or not reviewer or not archive_root.exists():
        return None
    matches = sorted(archive_root.glob(f"*/{reviewer}/{image_key}.json"), key=lambda path: str(path.parent.parent.name))
    return matches[-1] if matches else None


def archived_review_paths_by_key(reviewer: str, reviews_dir: str | Path) -> dict[str, Path]:
    archive_root = repo_resolve(reviews_dir) / "_archived_unreviewed"
    if not reviewer or not archive_root.exists():
        return {}
    archived: dict[str, Path] = {}
    for path in sorted(archive_root.glob(f"*/{reviewer}/*.json"), key=lambda item: str(item.parent.parent.name)):
        archived[path.stem] = path
    return archived


def review_queue_filter_summary(
    image_key: str,
    reviewer: str,
    reviews_dir: str | Path,
    archived_reviews: dict[str, Path] | None = None,
) -> dict[str, Any]:
    can_archive, reason, review_path = review_archivable_for_image_key(image_key, reviewer, reviews_dir)
    summary: dict[str, Any] = {
        "review_path": repo_relative(review_path),
        "review_queue_filter": "unreviewed" if can_archive else "protected",
        "unreviewed_archivable": bool(can_archive),
        "review_protected_reason": reason,
        "review_status": None,
        "review_counts": {},
    }
    if can_archive:
        return summary
    if reason == "review file missing":
        summary["review_queue_filter"] = "missing_review"
        return summary

    try:
        review = review_lib.load_review_document(review_path)
    except Exception:
        return summary

    summary["review_status"] = review.get("review_status", "overview_1")
    try:
        counts = review_lib.review_counts(review)
    except Exception:
        counts = {}
    summary["review_counts"] = counts
    if review.get("review_status") == "completed":
        summary["review_queue_filter"] = "reviewed"
    elif int(counts.get("reviewed_active_detections") or 0) > 0 or any(
        isinstance(detection, dict) and detection.get("review_label") is not None
        for detection in review.get("detections", [])
    ):
        summary["review_queue_filter"] = "partially_reviewed"
    return summary


def list_reviewer_review_image_keys(reviewer: str, reviews_dir: str | Path) -> list[str]:
    reviewer_dir = repo_resolve(Path(reviews_dir) / reviewer)
    if not reviewer_dir.exists():
        return []
    keys: list[str] = []
    for path in sorted(reviewer_dir.glob("*.json")):
        try:
            review = review_lib.load_review_document(path)
        except Exception:
            keys.append(path.stem)
            continue
        keys.append(str(review.get("image_key") or path.stem))
    return unique_preserving_order(keys)


def preview_review_queue_prune(
    *,
    reviewer: str,
    reviews_dir: str | Path = review_lib.DEFAULT_REVIEWS_DIR,
    image_keys: list[str] | None = None,
    filter_mode: str | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if filter_mode not in {None, "unreviewed"}:
        raise ValueError(f"Unsupported review queue prune filter: {filter_mode}")
    explicit_keys = unique_preserving_order(image_keys or [])
    if filter_mode is None and not explicit_keys:
        raise ValueError("Provide at least one --image-key or --filter unreviewed")

    all_keys = list_reviewer_review_image_keys(reviewer, reviews_dir)
    if filter_mode == "unreviewed" and explicit_keys:
        requested_keys = explicit_keys
    elif filter_mode == "unreviewed":
        requested_keys = all_keys
    else:
        requested_keys = explicit_keys

    archivable: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    for image_key in requested_keys:
        can_archive, reason, review_path = review_archivable_for_image_key(image_key, reviewer, reviews_dir)
        item = {
            "image_key": image_key,
            "review_path": repo_relative(review_path),
        }
        if can_archive:
            archivable.append(item)
        elif filter_mode == "unreviewed" and not explicit_keys:
            continue
        else:
            protected.append({**item, "reason": reason or "protected"})

    return {
        "requested_count": len(requested_keys),
        "archivable_count": len(archivable),
        "protected_count": len(protected),
        "reviewer": reviewer,
        "reviews_dir": repo_relative(reviews_dir),
        "filters": filters or ({"filter": filter_mode} if filter_mode else {}),
        "requested_image_keys": requested_keys,
        "archivable": archivable,
        "protected": protected,
    }


def utc_timestamp_ms_for_path() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")[:23] + "Z"


def prune_review_queue(
    *,
    reviewer: str,
    reviews_dir: str | Path = review_lib.DEFAULT_REVIEWS_DIR,
    image_keys: list[str] | None = None,
    filter_mode: str | None = None,
    filters: dict[str, Any] | None = None,
    dry_run: bool = False,
    confirmed_archivable_count: int | None = None,
    reports_dir: str | Path = DEFAULT_REVIEW_QUEUE_PRUNE_REPORTS_DIR,
) -> dict[str, Any]:
    preview = preview_review_queue_prune(
        reviewer=reviewer,
        reviews_dir=reviews_dir,
        image_keys=image_keys,
        filter_mode=filter_mode,
        filters=filters,
    )
    if dry_run:
        return preview
    if confirmed_archivable_count is not None and int(confirmed_archivable_count) != int(preview["archivable_count"]):
        raise ValueError(
            "confirmed_archivable_count does not match current preview "
            f"({confirmed_archivable_count} != {preview['archivable_count']})"
        )

    timestamp = utc_timestamp_ms_for_path()
    archive_root = repo_resolve(reviews_dir) / "_archived_unreviewed" / timestamp / reviewer
    report_path = repo_resolve(reports_dir) / f"{timestamp}.json"
    archived: list[dict[str, Any]] = []
    for item in preview["archivable"]:
        image_key = item["image_key"]
        source_path = review_lib.review_file_path(image_key, reviewer, reviews_dir)
        target_path = archive_root / f"{image_key}.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(target_path))
        archived.append(
            {
                "image_key": image_key,
                "review_path": repo_relative(source_path),
                "archive_path": repo_relative(target_path),
            }
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp,
        "reviewer": reviewer,
        "reviews_dir": repo_relative(reviews_dir),
        "requested_image_keys": preview["requested_image_keys"],
        "archived": archived,
        "protected": preview["protected"],
        "filters": preview.get("filters") or {},
    }
    write_json(report_path, report)

    return {
        **preview,
        "archived_count": len(archived),
        "archived": archived,
        "report_path": repo_relative(report_path),
    }


def removable_image_keys_for_filter(
    *,
    manifest: dict[str, Any],
    reviewer: str,
    reviews_dir: str | Path,
    filter_mode: str | None,
) -> list[str]:
    if filter_mode is None:
        return []
    if filter_mode != "inactive-review":
        raise ValueError(f"Unsupported image removal filter: {filter_mode}")
    keys: list[str] = []
    for image_key in (manifest.get("images") or {}).keys():
        review_path = review_lib.review_file_path(str(image_key), reviewer, reviews_dir)
        if not review_path.exists():
            keys.append(str(image_key))
    return keys


def archived_review_paths_for_image_key(image_key: str, reviewer: str, reviews_dir: str | Path) -> list[Path]:
    archive_root = repo_resolve(reviews_dir) / "_archived_unreviewed"
    if not image_key or not reviewer or not archive_root.exists():
        return []
    return sorted(archive_root.glob(f"*/{reviewer}/{image_key}.json"))


def safe_workspace_file(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None
    path = repo_resolve(path_value)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if path_is_relative_to(resolved, WORKSPACE_DIR):
        return resolved
    return None


def safe_file_in_roots(path_value: str | Path | None, roots: list[str | Path]) -> Path | None:
    if not path_value:
        return None
    path = repo_resolve(path_value)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for root in roots:
        if path_is_relative_to(resolved, repo_resolve(root)):
            return resolved
    return None


def remove_empty_parents(path: Path, stop_at: Path) -> None:
    stop = stop_at.resolve()
    current = path.parent
    while current != stop and path_is_relative_to(current, stop):
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def preview_remove_images(
    *,
    reviewer: str,
    reviews_dir: str | Path = review_lib.DEFAULT_REVIEWS_DIR,
    manifest_path: str | Path = DEFAULT_IMPORT_MANIFEST_PATH,
    albums_dir: str | Path | None = None,
    image_keys: list[str] | None = None,
    filter_mode: str | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_file = repo_resolve(manifest_path)
    album_root = repo_resolve(albums_dir or default_albums_dir_for_manifest(manifest_file))
    manifest = load_import_manifest(manifest_file)
    explicit_keys = unique_preserving_order(image_keys or [])
    requested_keys = explicit_keys or removable_image_keys_for_filter(
        manifest=manifest,
        reviewer=reviewer,
        reviews_dir=reviews_dir,
        filter_mode=filter_mode,
    )
    if not requested_keys and filter_mode is None:
        raise ValueError("Provide at least one --image-key or --filter inactive-review")

    removable: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    images = manifest.get("images") or {}
    for image_key in requested_keys:
        entry = images.get(image_key)
        review_path = review_lib.review_file_path(image_key, reviewer, reviews_dir)
        review: dict[str, Any] | None = None
        if review_path.exists():
            try:
                review = review_lib.load_review_document(review_path)
            except Exception:
                review = None
        cache_paths: list[str] = []
        source_paths: list[str] = []
        if isinstance(entry, dict):
            if entry.get("cache_path"):
                cache_paths.append(repo_relative(repo_resolve(entry["cache_path"])))
            source_path = safe_workspace_file(entry.get("source_path"))
            if source_path and path_is_relative_to(source_path, WORKSPACE_DIR / "cache" / "import_staging"):
                source_paths.append(repo_relative(source_path))
        if review and review.get("cache_path"):
            cache_path = safe_workspace_file(review.get("cache_path"))
            if cache_path:
                cache_paths.append(repo_relative(cache_path))
        archive_paths = [repo_relative(path) for path in archived_review_paths_for_image_key(image_key, reviewer, reviews_dir)]
        if entry is None and not review_path.exists() and not archive_paths:
            missing.append({"image_key": image_key, "reason": "not found"})
            continue
        removable.append(
            {
                "image_key": image_key,
                "in_manifest": entry is not None,
                "active_review_path": repo_relative(review_path) if review_path.exists() else None,
                "archived_review_paths": archive_paths,
                "cache_paths": unique_preserving_order(cache_paths),
                "source_paths": unique_preserving_order(source_paths),
                "album_ids": list((entry or {}).get("album_ids") or []),
            }
        )

    return {
        "requested_count": len(requested_keys),
        "removable_count": len(removable),
        "missing_count": len(missing),
        "reviewer": reviewer,
        "reviews_dir": repo_relative(reviews_dir),
        "manifest_path": repo_relative(manifest_file),
        "albums_dir": repo_relative(album_root),
        "filters": filters or ({"filter": filter_mode} if filter_mode else {}),
        "requested_image_keys": requested_keys,
        "removable": removable,
        "missing": missing,
    }


def remove_images(
    *,
    reviewer: str,
    reviews_dir: str | Path = review_lib.DEFAULT_REVIEWS_DIR,
    manifest_path: str | Path = DEFAULT_IMPORT_MANIFEST_PATH,
    albums_dir: str | Path | None = None,
    image_keys: list[str] | None = None,
    filter_mode: str | None = None,
    filters: dict[str, Any] | None = None,
    dry_run: bool = False,
    confirmed_removable_count: int | None = None,
    reports_dir: str | Path = DEFAULT_IMAGE_REMOVAL_REPORTS_DIR,
) -> dict[str, Any]:
    manifest_file = repo_resolve(manifest_path)
    album_root = repo_resolve(albums_dir or default_albums_dir_for_manifest(manifest_file))
    preview = preview_remove_images(
        reviewer=reviewer,
        reviews_dir=reviews_dir,
        manifest_path=manifest_file,
        albums_dir=album_root,
        image_keys=image_keys,
        filter_mode=filter_mode,
        filters=filters,
    )
    if dry_run:
        return preview
    if confirmed_removable_count is not None and int(confirmed_removable_count) != int(preview["removable_count"]):
        raise ValueError(
            "confirmed_removable_count does not match current preview "
            f"({confirmed_removable_count} != {preview['removable_count']})"
        )

    manifest = load_import_manifest(manifest_file)
    images = manifest.get("images") or {}
    cache_root = repo_resolve(manifest.get("cache_dir") or DEFAULT_IMPORT_CACHE_DIR)
    allowed_delete_roots = [WORKSPACE_DIR]
    removed_keys = {item["image_key"] for item in preview["removable"]}
    removed: list[dict[str, Any]] = []
    deleted_files: list[str] = []
    already_missing_files: list[str] = []

    for item in preview["removable"]:
        image_key = item["image_key"]
        images.pop(image_key, None)
        file_candidates = []
        file_candidates.extend(item.get("cache_paths") or [])
        file_candidates.extend(item.get("source_paths") or [])
        if item.get("active_review_path"):
            file_candidates.append(item["active_review_path"])
        file_candidates.extend(item.get("archived_review_paths") or [])
        item_deleted: list[str] = []
        for path_text in unique_preserving_order(file_candidates):
            target = safe_file_in_roots(path_text, allowed_delete_roots)
            if target is None:
                continue
            if target.exists() and target.is_file():
                target.unlink()
                deleted = repo_relative(target)
                deleted_files.append(deleted)
                item_deleted.append(deleted)
                if "_archived_unreviewed" in target.parts:
                    remove_empty_parents(target, repo_resolve(reviews_dir) / "_archived_unreviewed")
            else:
                already_missing_files.append(repo_relative(target))
        removed.append({**item, "deleted_files": item_deleted})

    for album_path in sorted(album_root.glob("alb_*.json")):
        album = load_json(album_path)
        original_images = album.get("images") or []
        filtered_images = [image for image in original_images if image.get("image_key") not in removed_keys]
        if len(filtered_images) == len(original_images):
            continue
        album["images"] = filtered_images
        album["updated_at"] = utcnow_iso_for_training()
        write_json(album_path, album)
        write_album_index(album_root, album)

    save_import_manifest(manifest_file, manifest)
    timestamp = utc_timestamp_ms_for_path()
    report_path = repo_resolve(reports_dir) / f"{timestamp}.json"
    report = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp,
        "reviewer": reviewer,
        "reviews_dir": repo_relative(reviews_dir),
        "manifest_path": repo_relative(manifest_file),
        "albums_dir": repo_relative(album_root),
        "requested_image_keys": preview["requested_image_keys"],
        "removed": removed,
        "missing": preview["missing"],
        "deleted_files": deleted_files,
        "already_missing_files": unique_preserving_order(already_missing_files),
        "filters": preview.get("filters") or {},
    }
    write_json(report_path, report)

    return {
        **preview,
        "removed_count": len(removed),
        "removed": removed,
        "deleted_files": deleted_files,
        "already_missing_files": unique_preserving_order(already_missing_files),
        "report_path": repo_relative(report_path),
    }


def import_entry_contract_complete(entry: dict[str, Any]) -> bool:
    if not REQUIRED_IMPORT_ENTRY_FIELDS.issubset(entry):
        return False
    if entry.get("import_status") not in DOCUMENTED_IMPORT_STATUSES - {"failed"}:
        return False
    if not entry.get("cache_content_hash"):
        return False
    return True


def import_training_data(
    sources: list[str | Path],
    *,
    manifest_path: str | Path = DEFAULT_IMPORT_MANIFEST_PATH,
    cache_dir: str | Path = DEFAULT_IMPORT_CACHE_DIR,
    albums_dir: str | Path | None = None,
    import_jobs_dir: str | Path | None = None,
    album_name: str | None = None,
    album_description: str | None = None,
    source_display_name: str | None = None,
    source_kind: str | None = None,
    append_to_album_id: str | None = None,
    import_job_id: str | None = None,
    resume_job_id: str | None = None,
    render_profile_value: str = "auto",
    reviews_dir: str | Path = review_lib.DEFAULT_REVIEWS_DIR,
    reviewer: str = review_lib.DEFAULT_REVIEWER,
    create_review_stubs: bool = True,
    auto_detect: bool = False,
    auto_detect_model_paths: dict[str, str | Path] | None = None,
    auto_detect_confidence: float = DEFAULT_AUTO_DETECT_CONFIDENCE,
    auto_detect_iou: float = DEFAULT_AUTO_DETECT_IOU,
    recursive: bool = False,
    force: bool = False,
    dry_run: bool = False,
    progress_callback=None,
) -> dict[str, Any]:
    if not sources:
        raise ValueError("At least one --source is required")
    manifest_file = repo_resolve(manifest_path)
    cache_root = contained(repo_resolve(cache_dir), WORKSPACE_DIR)
    contained(repo_resolve(manifest_path), WORKSPACE_DIR)
    contained(repo_resolve(reviews_dir), WORKSPACE_DIR)
    album_root = repo_resolve(albums_dir or default_albums_dir_for_manifest(manifest_file))
    job_root = repo_resolve(import_jobs_dir or default_import_jobs_dir_for_manifest(manifest_file))
    resolved_source_kind = source_kind or infer_import_source_kind(sources)
    resolved_source_display_name = source_display_name if source_display_name not in {""} else None
    if resolved_source_display_name is None:
        resolved_source_display_name = default_source_display_name(sources, resolved_source_kind)
    if manifest_file.exists() and not manifest_file.is_file():
        raise ValueError(f"Import manifest path is not a file: {manifest_file}")
    if cache_root.exists() and not cache_root.is_dir():
        raise ValueError(f"Import cache path is not a directory: {cache_root}")
    if resume_job_id and append_to_album_id:
        raise ValueError("Use either --resume-job or --append-to-album-id, not both")

    if resume_job_id:
        summary_path = job_root / f"{resume_job_id}.summary.json"
        if not summary_path.is_file():
            raise ValueError(f"Import job summary not found for resume: {resume_job_id}")
        summary = json.loads(summary_path.read_text())
        album_summary = summary.get("album") or summary.get("result", {}).get("album") or {}
        append_to_album_id = album_summary.get("album_id")
        if not append_to_album_id:
            raise ValueError(f"Import job summary has no album id: {resume_job_id}")
        import_job_id = resume_job_id

    if import_job_id is None:
        import_job_id = new_import_job_id(job_root)

    if append_to_album_id:
        existing_album = load_album_record(album_root, append_to_album_id)
        album_id = append_to_album_id
        resolved_album_name = existing_album.get("album_name") or album_name or album_id
        resolved_album_description = existing_album.get("album_description")
        if album_description not in {None, ""}:
            resolved_album_description = album_description
        if resolved_source_display_name is None:
            resolved_source_display_name = existing_album.get("source_display_name")
        existing_album_images = existing_album.get("images") or []
    else:
        existing_album = None
        album_id = new_album_id(album_root)
        resolved_album_name = album_name or derive_album_name(sources, resolved_source_kind, resolved_source_display_name)
        resolved_album_description = album_description or None
        existing_album_images = []

    album_existing_keys = {image.get("image_key") for image in existing_album_images if image.get("image_key")}
    log_path = job_root / f"{import_job_id}.jsonl"
    summary_path = job_root / f"{import_job_id}.summary.json"

    def log_event(event: dict[str, Any]) -> None:
        if dry_run:
            return
        payload = {
            "logged_at": utcnow_iso_for_training(),
            "job_id": import_job_id,
            "album_id": album_id,
            **event,
        }
        append_jsonl(log_path, payload)

    manifest = load_import_manifest(manifest_file)
    manifest["manifest_type"] = "training_import"
    manifest["cache_dir"] = repo_relative(cache_root)
    files, failed = discover_import_files(sources, recursive=recursive)
    initial_failed_count = len(failed)
    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    album_images: list[dict[str, Any]] = []
    review_stub_counts = {"created": 0, "existing": 0, "skipped": 0}
    auto_detect_counts = {"created_review": 0, "updated_existing_review": 0, "protected_existing_review": 0, "skipped": 0}
    member_counts = {"added_imported": 0, "added_existing": 0, "already_present": 0, "failed": 0}
    manifest_changed = False
    browser_roots = [repo_resolve(source).resolve() for source in sources]
    auto_detect_models = None
    if auto_detect and create_review_stubs and not dry_run:
        auto_detect_models = load_auto_detect_models(auto_detect_model_paths)

    def emit(event: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(event)
        except Exception:
            pass

    log_event({
        "phase": "queued",
        "sources": [str(repo_resolve(source)) for source in sources],
        "source_kind": resolved_source_kind,
        "source_display_name": resolved_source_display_name,
    })

    discovered_event = {
        "phase": "discovered",
        "total": len(files),
        "discovered_failed": len(failed),
    }
    emit(discovered_event)
    log_event(discovered_event)

    for item in failed:
        member_counts["failed"] += 1
        album_images.append(album_image_record(None, item, import_status="failed", member_status="failed"))
        log_event({"phase": "failed", "source": item.get("source"), "reason": item.get("reason")})

    for index, candidate in enumerate(files):
        source_path = repo_resolve(candidate["source_path"])
        source_collection = str(candidate["source_collection"])
        browser_relative_path = browser_relative_path_for(source_path, browser_roots, resolved_source_kind)
        processing_event = {
            "phase": "processing",
            "index": index,
            "total": len(files),
            "current": source_path.name,
            "browser_relative_path": browser_relative_path,
            "imported_count": len(imported),
            "skipped_count": len(skipped),
            "failed_count": len(failed),
        }
        emit(processing_event)
        log_event(processing_event)
        try:
            render_profile = resolve_import_render_profile(render_profile_value, source_path)
            image_key = review_lib.image_key_for_source(source_path, render_profile)
            cache_path = cache_root / f"{image_key}.png"
            existing = manifest.get("images", {}).get(image_key)
            cache_exists = cache_path.exists()
            existing_complete = bool(existing and import_entry_contract_complete(existing))
            if existing and cache_exists and existing_complete and not force:
                item = import_result_item(existing, "already_imported")
                item = attach_review_stub_with_optional_auto_detect(
                    item,
                    existing,
                    reviewer=reviewer,
                    reviews_dir=reviews_dir,
                    skip=not create_review_stubs,
                    dry_run=dry_run,
                    auto_detect_models=auto_detect_models,
                    auto_detect_confidence=auto_detect_confidence,
                    auto_detect_iou=auto_detect_iou,
                )
                if item["review_stub_status"] == "failed":
                    failed.append(review_stub_failure_item(item))
                    member_counts["failed"] += 1
                    album_images.append(album_image_record(existing, item, import_status="failed", member_status="failed"))
                    log_event({"phase": "failed", "image_key": image_key, "current": source_path.name, "reason": item.get("review_stub_error")})
                    continue
                if add_album_id_to_entry(existing, album_id):
                    manifest_changed = True
                if resolved_source_display_name and not existing.get("source_display_name"):
                    existing["source_display_name"] = resolved_source_display_name
                    manifest_changed = True
                if browser_relative_path and not existing.get("browser_relative_path"):
                    existing["browser_relative_path"] = browser_relative_path
                    manifest_changed = True
                review_stub_counts[item["review_stub_status"]] += 1
                auto_detect_counts[item.get("auto_detect_status") or "skipped"] += 1
                member_status = "already_present" if image_key in album_existing_keys else "added_existing"
                member_counts[member_status] += 1
                item["album_id"] = album_id
                item["member_status"] = member_status
                album_images.append(album_image_record(existing, item, import_status="already_imported", member_status=member_status))
                skipped.append(item)
                log_event({"phase": "skipped", "image_key": image_key, "current": source_path.name, "status": "already_imported", "member_status": member_status})
                continue

            status = "regenerated_cache" if existing else "imported"
            render_info = render_import_cache(source_path, cache_path, render_profile, dry_run=dry_run)
            entry = import_manifest_entry(
                source_path,
                cache_path,
                render_profile,
                render_info,
                source_collection=source_collection,
                source_kind=resolved_source_kind,
                source_display_name=resolved_source_display_name,
                browser_relative_path=browser_relative_path,
                album_id=album_id,
                import_status=status,
                previous_entry=existing,
            )
            item = import_result_item(entry, status)
            item = attach_review_stub_with_optional_auto_detect(
                item,
                entry,
                reviewer=reviewer,
                reviews_dir=reviews_dir,
                skip=not create_review_stubs,
                dry_run=dry_run,
                auto_detect_models=auto_detect_models,
                auto_detect_confidence=auto_detect_confidence,
                auto_detect_iou=auto_detect_iou,
            )
            if item["review_stub_status"] == "failed":
                failed.append(review_stub_failure_item(item))
                member_counts["failed"] += 1
                album_images.append(album_image_record(entry, item, import_status="failed", member_status="failed"))
                log_event({"phase": "failed", "image_key": image_key, "current": source_path.name, "reason": item.get("review_stub_error")})
                continue
            review_stub_counts[item["review_stub_status"]] += 1
            auto_detect_counts[item.get("auto_detect_status") or "skipped"] += 1
            member_status = "added_imported" if status == "imported" else "added_existing"
            if image_key in album_existing_keys:
                member_status = "already_present"
            member_counts[member_status] += 1
            item["album_id"] = album_id
            item["member_status"] = member_status
            album_images.append(album_image_record(entry, item, import_status=status, member_status=member_status))
            imported.append(item)
            if not dry_run:
                manifest.setdefault("images", {})[image_key] = entry
                manifest_changed = True
            log_event({"phase": "imported", "image_key": image_key, "current": source_path.name, "status": status, "member_status": member_status})
        except Exception as exc:
            render_profile_kind = None
            try:
                render_profile_kind = resolve_import_render_profile(render_profile_value, source_path).get("kind")
            except Exception:
                pass
            item = {
                "image_key": None,
                "source": str(source_path),
                "cache_path": None,
                "render_profile_kind": render_profile_kind,
                "status": "failed",
                "reason": str(exc),
            }
            failed.append(item)
            member_counts["failed"] += 1
            album_images.append(album_image_record(None, item, import_status="failed", member_status="failed"))
            log_event({"phase": "failed", "current": source_path.name, "source": str(source_path), "reason": str(exc)})

    if not dry_run and manifest_changed:
        emit({"phase": "saving", "total": len(files)})
        log_event({"phase": "manifest-saving", "total": len(files)})
        save_import_manifest(manifest_file, manifest)
        log_event({"phase": "manifest-saved", "manifest_path": repo_relative(manifest_file)})

    album_counts = {
        "uploaded": len(files) + initial_failed_count,
        "discovered": len(files),
        "imported": len(imported),
        "skipped": len(skipped),
        "failed": len(failed),
        "members_added_imported": member_counts["added_imported"],
        "members_added_existing": member_counts["added_existing"],
        "members_already_present": member_counts["already_present"],
        "members_failed": member_counts["failed"],
        "review_stubs_created": review_stub_counts["created"],
        "review_stubs_existing": review_stub_counts["existing"],
        "review_stubs_skipped": review_stub_counts["skipped"],
        "auto_detect_created_review": auto_detect_counts["created_review"],
        "auto_detect_updated_existing_review": auto_detect_counts["updated_existing_review"],
        "auto_detect_protected_existing_review": auto_detect_counts["protected_existing_review"],
        "auto_detect_skipped": auto_detect_counts["skipped"],
    }

    album_record = {
        "schema_version": SCHEMA_VERSION,
        "album_id": album_id,
        "album_name": resolved_album_name,
        "album_description": resolved_album_description,
        "source_display_name": resolved_source_display_name,
        "source_kind": resolved_source_kind,
        "import_job_ids": list((existing_album or {}).get("import_job_ids") or []),
        "created_at": (existing_album or {}).get("created_at") or utcnow_iso_for_training(),
        "updated_at": utcnow_iso_for_training(),
        "created_by": (existing_album or {}).get("created_by") or reviewer,
        "reviewer": reviewer,
        "manifest_path": repo_relative(manifest_file),
        "cache_dir": repo_relative(cache_root),
        "reviews_dir": repo_relative(reviews_dir),
        "options": {
            "recursive": bool(recursive),
            "force": bool(force),
            "render_profile": render_profile_value,
            "create_review_stubs": bool(create_review_stubs),
        },
        "counts": album_counts,
        "images": merge_album_images(existing_album_images, album_images),
    }
    append_unique(album_record["import_job_ids"], import_job_id)
    album_path = album_root / f"{album_id}.json"
    if not dry_run:
        write_json(album_path, album_record)
        write_album_index(album_root, album_record)
        log_event({"phase": "album-saved", "album_path": repo_relative(album_path)})

    done_event = {
        "phase": "done",
        "total": len(files),
        "imported_count": len(imported),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
    }
    emit(done_event)
    log_event(done_event)

    album_summary = {
        "album_id": album_id,
        "album_name": resolved_album_name,
        "album_description": resolved_album_description,
        "album_path": repo_relative(album_path),
        "album_index_path": repo_relative(album_root / "index.json"),
        "source_display_name": resolved_source_display_name,
        "source_kind": resolved_source_kind,
        "image_count": len([image for image in album_record["images"] if image.get("image_key")]),
    }
    result = {
        "import_job_id": import_job_id,
        "import_log_path": repo_relative(log_path),
        "import_summary_path": repo_relative(summary_path),
        "album": album_summary,
        "manifest_path": repo_relative(manifest_file),
        "cache_dir": repo_relative(cache_root),
        "processed_count": len(imported) + len(skipped) + len(failed),
        "imported_count": len(imported),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "review_stubs_created_count": review_stub_counts["created"],
        "review_stubs_existing_count": review_stub_counts["existing"],
        "review_stubs_skipped_count": review_stub_counts["skipped"],
        "auto_detect_created_review_count": auto_detect_counts["created_review"],
        "auto_detect_updated_existing_review_count": auto_detect_counts["updated_existing_review"],
        "auto_detect_protected_existing_review_count": auto_detect_counts["protected_existing_review"],
        "auto_detect_skipped_count": auto_detect_counts["skipped"],
        "imported": imported,
        "skipped": skipped,
        "failed": failed,
    }
    if not dry_run:
        write_json(summary_path, {"schema_version": SCHEMA_VERSION, "job_id": import_job_id, "album": album_summary, "result": result})
    return result


def run_auto_detect_on_imported_reviews(
    *,
    manifest_path: str | Path = DEFAULT_IMPORT_MANIFEST_PATH,
    reviews_dir: str | Path = review_lib.DEFAULT_REVIEWS_DIR,
    reviewer: str = review_lib.DEFAULT_REVIEWER,
    image_keys: list[str] | None = None,
    limit: int = 0,
    model_paths: dict[str, str | Path] | None = None,
    confidence: float = DEFAULT_AUTO_DETECT_CONFIDENCE,
    iou: float = DEFAULT_AUTO_DETECT_IOU,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest_file = repo_resolve(manifest_path)
    manifest = load_import_manifest(manifest_file)
    images = manifest.get("images") or {}
    requested = unique_preserving_order(image_keys or list(images.keys()))
    if limit and limit > 0:
        requested = requested[:limit]

    counts = {
        "created_review": 0,
        "updated_existing_review": 0,
        "protected_existing_review": 0,
        "skipped": 0,
        "missing": 0,
        "failed": 0,
    }
    items: list[dict[str, Any]] = []
    models = None if dry_run else load_auto_detect_models(model_paths)

    for image_key in requested:
        entry = images.get(image_key)
        if not isinstance(entry, dict):
            counts["missing"] += 1
            items.append({"image_key": image_key, "auto_detect_status": "missing", "reason": "image key is not in import manifest"})
            continue
        review_path = review_lib.review_file_path(image_key, reviewer, reviews_dir)
        if dry_run:
            status = "created_review"
            if review_path.exists():
                review = review_lib.load_review_document(review_path)
                status = "updated_existing_review" if review_safe_for_auto_detect_update(review) else "protected_existing_review"
            counts[status] += 1
            items.append({"image_key": image_key, "review_stub_path": repo_relative(review_path), "auto_detect_status": status})
            continue

        item = import_result_item(entry, "already_imported")
        item = attach_review_stub_with_optional_auto_detect(
            item,
            entry,
            reviewer=reviewer,
            reviews_dir=reviews_dir,
            skip=False,
            dry_run=False,
            auto_detect_models=models,
            auto_detect_confidence=confidence,
            auto_detect_iou=iou,
        )
        status = item.get("auto_detect_status") or "skipped"
        if item.get("review_stub_status") == "failed":
            status = "failed"
        counts[status] += 1
        items.append(item)

    return {
        "manifest_path": repo_relative(manifest_file),
        "reviews_dir": repo_relative(reviews_dir),
        "reviewer": reviewer,
        "requested_count": len(requested),
        "created_review_count": counts["created_review"],
        "updated_existing_review_count": counts["updated_existing_review"],
        "protected_existing_review_count": counts["protected_existing_review"],
        "skipped_count": counts["skipped"],
        "missing_count": counts["missing"],
        "failed_count": counts["failed"],
        "items": items,
    }


def album_name_map(albums_dir: str | Path) -> dict[str, str]:
    names: dict[str, str] = {}
    for album in list_albums_summary(albums_dir):
        album_id = str(album.get("album_id") or "")
        if album_id:
            names[album_id] = str(album.get("album_name") or album_id)
    return names


def review_label_counts(review: dict[str, Any]) -> dict[str, int]:
    counts = {"ship": 0, "ships": 0, "neither": 0, "exportable": 0, "total": 0}
    for detection in review.get("detections") or []:
        if not isinstance(detection, dict):
            continue
        counts["total"] += 1
        label = detection.get("review_label")
        if label == "ship_non_sts":
            counts["ship"] += 1
            counts["exportable"] += 1
        elif label == "ships_sts":
            counts["ships"] += 1
            counts["exportable"] += 1
        elif label == "neither":
            counts["neither"] += 1
    return counts


def list_dataset_candidates(
    *,
    reviews_dir: str | Path = review_lib.DEFAULT_REVIEWS_DIR,
    manifest_path: str | Path = DEFAULT_IMPORT_MANIFEST_PATH,
    albums_dir: str | Path | None = None,
    reviewer: str = review_lib.DEFAULT_REVIEWER,
    album_ids: list[str] | None = None,
    required_labels: list[str] | None = None,
    completed_only: bool = True,
) -> dict[str, Any]:
    """List review-backed image candidates; album filters match manifest album_ids only."""
    manifest_file = repo_resolve(manifest_path)
    album_root = repo_resolve(albums_dir or default_albums_dir_for_manifest(manifest_file))
    manifest_images = load_import_manifest(manifest_file).get("images", {}) if manifest_file.exists() else {}
    names_by_album = album_name_map(album_root) if album_root.exists() else {}
    album_filter = {str(value) for value in (album_ids or []) if str(value).strip()}
    label_filter = {str(value) for value in (required_labels or []) if str(value).strip()}
    candidates: list[dict[str, Any]] = []
    split_counts = {"ship": 0, "ships": 0, "neither": 0, "exportable": 0, "total": 0}

    for review in review_lib.list_review_documents(reviewer, reviews_dir):
        image_key = str(review.get("image_key") or "")
        if not image_key:
            continue
        if completed_only and review.get("review_status") != "completed":
            continue
        entry = manifest_images.get(image_key) or {}
        candidate_album_ids = [str(value) for value in (entry.get("album_ids") or []) if str(value).strip()]
        if album_filter and not album_filter.intersection(candidate_album_ids):
            continue
        counts = review_label_counts(review)
        if label_filter:
            matched = False
            if "ship" in label_filter and counts["ship"] > 0:
                matched = True
            if "ships" in label_filter and counts["ships"] > 0:
                matched = True
            if "neither" in label_filter and counts["neither"] > 0:
                matched = True
            if "exportable" in label_filter and counts["exportable"] > 0:
                matched = True
            if not matched:
                continue
        for key, value in counts.items():
            split_counts[key] += int(value)
        review_path = Path(review.get("_path")) if review.get("_path") else review_lib.review_file_path(image_key, reviewer, reviews_dir)
        candidates.append({
            "image_key": image_key,
            "image_filename": review.get("image_filename") or entry.get("original_filename") or f"{image_key}.png",
            "display_filename": entry.get("original_filename") or review.get("display_filename") or review.get("image_filename") or f"{image_key}.png",
            "review_status": review.get("review_status"),
            "review_path": repo_relative(review_path),
            "cache_path": review.get("cache_path") or entry.get("cache_path"),
            "album_ids": candidate_album_ids,
            "album_names": [names_by_album.get(album_id, album_id) for album_id in candidate_album_ids],
            "source_display_name": entry.get("source_display_name"),
            "counts": counts,
        })

    candidates.sort(key=lambda item: (str(item.get("image_filename") or ""), str(item.get("image_key") or "")))
    return {
        "schema_version": SCHEMA_VERSION,
        "reviews_dir": repo_relative(reviews_dir),
        "manifest_path": repo_relative(manifest_file),
        "reviewer": reviewer,
        "candidate_count": len(candidates),
        "bbox_counts": split_counts,
        "candidates": candidates,
    }


SPLIT_MODES = ("group", "image")
GROUP_KEY_RULE = "<region>|<date>"


def resolve_selection_group_keys(
    image_keys: list[str],
    *,
    reviews_dir: str | Path = review_lib.DEFAULT_REVIEWS_DIR,
    manifest_path: str | Path | None = DEFAULT_IMPORT_MANIFEST_PATH,
    reviewer: str = review_lib.DEFAULT_REVIEWER,
) -> dict[str, str]:
    """Map image keys to `<region>|<date>` split groups, exactly as the clean builder does.

    Region/date come from the import manifest entry when there is one (raster bounds,
    then source path, then album majority), otherwise from the review document's cache
    collection and its original filename.
    """
    manifest_file = repo_resolve(manifest_path) if manifest_path else None
    import_images: dict[str, Any] = {}
    if manifest_file is not None and manifest_file.exists():
        import_images = load_import_manifest(manifest_file).get("images", {}) or {}
    album_regions = clean_dataset_lib.album_region_index(import_images)
    resolved: dict[str, str] = {}
    for raw_key in image_keys:
        image_key = str(raw_key)
        entry = import_images.get(image_key)
        if entry:
            original = (
                entry.get("browser_relative_path")
                or entry.get("original_filename")
                or entry.get("image_filename")
                or image_key
            )
            region = clean_dataset_lib.region_for_import_entry(entry, album_regions)
        else:
            review: dict[str, Any] | None = None
            try:
                review, _ = find_review_for_image(image_key, reviews_dir, include_incomplete=True)
            except Exception:
                review = None
            if review:
                cache_dir = Path(str(review.get("cache_path") or "")).parent.name
                region = clean_dataset_lib.CACHE_DIR_REGIONS.get(cache_dir, "unknown")
                original = str(review.get("image_filename") or image_key)
            else:
                region = "unknown"
                original = image_key
        resolved[image_key] = group_key(parse_source_name(str(original)), region)
    return resolved


def split_items_by_image(unique_keys: list[str], *, seed: int, ratios: dict[str, float]) -> dict[str, str]:
    """Legacy image-level split: shuffle image keys and cut by ratio. Leaks scene groups."""
    keys = list(unique_keys)
    rng = random.Random(seed)
    rng.shuffle(keys)
    total = len(keys)
    train_count = int(round(total * float(ratios.get("train", 0.8))))
    val_count = int(round(total * float(ratios.get("val", 0.1))))
    if total >= 2 and val_count == 0:
        val_count = 1
    if train_count + val_count > total:
        train_count = max(0, total - val_count)
    split_by_key: dict[str, str] = {}
    for index, image_key in enumerate(keys):
        if index < train_count:
            split = "train"
        elif index < train_count + val_count:
            split = "val"
        else:
            split = "test"
        split_by_key[image_key] = split
    return split_by_key


def auto_split_items(
    image_keys: list[str],
    *,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
    group_keys: dict[str, str] | None = None,
    split_mode: str = "group",
) -> list[dict[str, Any]]:
    """Assign splits to image keys.

    ``split_mode="group"`` (the default) keeps every `<region>|<date>` group inside one
    split so adjacent tiles of the same scene cannot leak between train and val/test.
    ``split_mode="image"`` is the old image-level shuffle, kept as an escape hatch.
    """
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got {split_mode!r}")
    ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    unique_keys = list(dict.fromkeys(str(key) for key in image_keys if str(key).strip()))
    if not unique_keys:
        return []

    groups_by_key = {key: str((group_keys or {}).get(key) or key) for key in unique_keys}
    if split_mode == "image":
        split_by_key = split_items_by_image(unique_keys, seed=seed, ratios=ratios)
    else:
        stats: dict[str, GroupStats] = {}
        for key in unique_keys:
            stats.setdefault(groups_by_key[key], GroupStats(key=groups_by_key[key])).add()
        assignment = assign_groups_to_splits(stats.values(), ratios=ratios, seed=seed, metrics=["images"])
        split_by_key = {key: assignment[groups_by_key[key]] for key in unique_keys}

    return [
        {
            "image_key": image_key,
            "split": split_by_key[image_key],
            "group_key": groups_by_key[image_key],
            "include_labels": "all_exportable",
            "selected_detection_ids": None,
        }
        for image_key in sorted(unique_keys)
    ]


def split_summary_for_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise a selection's splits, including how many groups straddle two splits."""
    split_counts = {split: 0 for split in SPLITS}
    item_groups: dict[str, tuple[str, str]] = {}
    for item in items:
        split = str((item or {}).get("split") or "train")
        if split in split_counts:
            split_counts[split] += 1
        image_key = str((item or {}).get("image_key") or "")
        item_groups[image_key] = (split, str((item or {}).get("group_key") or image_key))
    overlap = check_no_group_overlap(item_groups)
    return {
        "image_count": len(items),
        "split_counts": split_counts,
        "group_count": int(overlap["group_count"]),
        "groups_spanning_splits": len(overlap["overlapping_groups"]),
        "overlapping_groups": sorted(overlap["overlapping_groups"]),
    }


def preview_dataset_split(
    image_keys: list[str],
    *,
    seed: int = 42,
    split_ratios: dict[str, float] | None = None,
    split_mode: str = "group",
    reviews_dir: str | Path = review_lib.DEFAULT_REVIEWS_DIR,
    manifest_path: str | Path | None = DEFAULT_IMPORT_MANIFEST_PATH,
    reviewer: str = review_lib.DEFAULT_REVIEWER,
) -> dict[str, Any]:
    """Backend split preview for step 4: the exact splits `save_dataset_selection` would write."""
    keys = list(dict.fromkeys(str(key) for key in image_keys if str(key).strip()))
    groups = resolve_selection_group_keys(keys, reviews_dir=reviews_dir, manifest_path=manifest_path, reviewer=reviewer)
    items = auto_split_items(keys, seed=int(seed), ratios=split_ratios, group_keys=groups, split_mode=split_mode)
    summary = split_summary_for_items(items)
    summary.update({
        "split_mode": split_mode,
        "group_key_rule": GROUP_KEY_RULE,
        "seed": int(seed),
        "ratios": split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1},
        "items": items,
    })
    return summary


def selection_filename(dataset_name: str, dataset_version: str | int) -> str:
    name_slug = slugify(dataset_name, "dataset")
    version_slug = slugify(str(dataset_version), "v1")
    return f"{name_slug}_{version_slug}.json"


def save_dataset_selection(
    *,
    dataset_name: str,
    dataset_version: str | int = "v1",
    description: str | None = None,
    image_keys: list[str],
    selections_dir: str | Path = DEFAULT_DATASET_SELECTIONS_DIR,
    datasets_dir: str | Path = DEFAULT_DATASETS_DIR,
    reviews_dir: str | Path = review_lib.DEFAULT_REVIEWS_DIR,
    manifest_path: str | Path = DEFAULT_IMPORT_MANIFEST_PATH,
    reviewer: str = review_lib.DEFAULT_REVIEWER,
    album_ids: list[str] | None = None,
    seed: int = 42,
    split_ratios: dict[str, float] | None = None,
    split_mode: str = "group",
    created_by: str | None = None,
) -> dict[str, Any]:
    clean_name = str(dataset_name or "").strip()
    if not clean_name:
        raise ValueError("dataset_name is required")
    keys = list(dict.fromkeys(str(key) for key in image_keys if str(key).strip()))
    if not keys:
        raise ValueError("At least one image_key is required")
    version = str(dataset_version or "v1").strip() or "v1"
    ratios = split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    output_name = slugify(f"{clean_name}_{version}", "dataset_v1")
    now = utcnow_iso_for_training()
    group_keys = resolve_selection_group_keys(keys, reviews_dir=reviews_dir, manifest_path=manifest_path, reviewer=reviewer)
    selection = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": clean_name,
        "dataset_version": version,
        "description": description or None,
        "created_at": now,
        "updated_at": now,
        "created_by": created_by or reviewer,
        "selection_mode": "image",
        "split_summary": None,
        "classes": {str(class_id): class_name for class_id, class_name in CLASS_ID_TO_NAME.items()},
        "sources": {
            "reviews_dir": repo_relative(reviews_dir),
            "manifest": repo_relative(manifest_path),
            "reviewer": reviewer,
            "album_ids": [str(value) for value in (album_ids or []) if str(value).strip()],
        },
        "split_policy": {
            "mode": "auto_with_manual_override",
            "split_mode": split_mode,
            "group_key_rule": GROUP_KEY_RULE if split_mode == "group" else None,
            "default_ratios": ratios,
            "seed": int(seed),
        },
        "items": auto_split_items(
            keys,
            seed=int(seed),
            ratios=ratios,
            group_keys=group_keys,
            split_mode=split_mode,
        ),
        "background_negatives": {"enabled": False},
        "export": {
            "output_dir": repo_relative(repo_resolve(datasets_dir) / output_name),
            "include_incomplete_reviews": False,
        },
    }
    summary = split_summary_for_items(selection["items"])
    summary["split_mode"] = split_mode
    summary["group_key_rule"] = GROUP_KEY_RULE if split_mode == "group" else None
    selection["split_summary"] = summary
    target = repo_resolve(selections_dir) / selection_filename(clean_name, version)
    write_json(target, selection)
    return {"selection_path": repo_relative(target), "selection": selection, "split_summary": summary}


def list_dataset_selections(selections_dir: str | Path = DEFAULT_DATASET_SELECTIONS_DIR) -> list[dict[str, Any]]:
    root = repo_resolve(selections_dir)
    selections: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            selection = load_json(path)
        except Exception:
            continue
        items = selection.get("items") or []
        summary = split_summary_for_items(items)
        split_counts = summary["split_counts"]
        export_dir = selection.get("export", {}).get("output_dir")
        export_path = repo_resolve(export_dir) if export_dir else None
        selections.append({
            "split_mode": (selection.get("split_policy") or {}).get("split_mode") or "image",
            "group_count": summary["group_count"],
            "groups_spanning_splits": summary["groups_spanning_splits"],
            "export_is_current": export_selection_is_current(path, export_path),
            "selection_path": repo_relative(path),
            "dataset_name": selection.get("dataset_name"),
            "dataset_version": selection.get("dataset_version"),
            "description": selection.get("description"),
            "item_count": len(items),
            "split_counts": split_counts,
            "export_output_dir": export_dir,
            "exported": bool(export_path and (export_path / "dataset_manifest.json").is_file()),
            "updated_at": selection.get("updated_at") or selection.get("created_at"),
        })
    selections.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return selections


def data_yaml_classes(data_yaml: Path) -> dict[int, str]:
    """Read the names block mapping produced by the dataset exporter."""
    if not data_yaml.is_file():
        return {}
    classes: dict[int, str] = {}
    in_names = False
    for line in data_yaml.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            in_names = line.split("#", 1)[0].strip() == "names:"
            continue
        if not in_names:
            continue
        key, separator, value = line.strip().partition(":")
        if not separator or not key.isdigit() or int(key) in classes:
            return {}
        try:
            tokens = shlex.split(value, comments=True)
        except ValueError:
            return {}
        if len(tokens) != 1:
            return {}
        classes[int(key)] = tokens[0]
    return classes


def yaml_defines_required_classes(data_yaml: Path) -> bool:
    return data_yaml_classes(data_yaml) == CLASS_ID_TO_NAME


def likely_duplicate_class_name_warnings(class_names: list[str]) -> list[str]:
    warnings: list[str] = []
    normalized = [name.strip().lower() for name in class_names if name.strip()]
    for index, left in enumerate(normalized):
        for right in normalized[index + 1:]:
            if left == right or left.rstrip("s") == right.rstrip("s"):
                warning = f"Class names may be singular/plural duplicates: {left!r} and {right!r}"
                if {left, right} == {"ship", "ships"}:
                    warning += "; allowed as intentional project classes"
                warnings.append(warning)
    return list(dict.fromkeys(warnings))


def resolve_training_run_dir(run_dir: str | Path) -> Path:
    resolved = repo_resolve(run_dir).resolve()
    runs_root = repo_resolve(WORKSPACE_DIR / "runs").resolve()
    if not path_is_relative_to(resolved, runs_root):
        raise ValueError(f"run_dir must be inside {repo_relative(runs_root)}: {run_dir}")
    return resolved


def validate_run_segment(value: str, label: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value or Path(value).drive:
        raise ValueError(f"{label} must be a single path segment: {value!r}")
    return value


def resolve_training_run_ref(runs_dir: str | Path, run_group: str, run_name: str | None = None) -> Path:
    root = resolve_training_run_dir(runs_dir)
    group = validate_run_segment(str(run_group), "run_group")
    target = root / group
    if run_name is not None:
        target /= validate_run_segment(str(run_name), "run_name")
    resolved = resolve_training_run_dir(target)
    if not path_is_relative_to(resolved, root):
        raise ValueError(f"Run reference must stay inside {repo_relative(root)}")
    return resolved


def split_has_images(dataset_dir: Path, split: str) -> bool:
    image_dir = dataset_dir / "images" / split
    if not image_dir.is_dir():
        return False
    return any(path.is_file() for path in image_dir.iterdir())


def config_missing_items(config: dict[str, Any]) -> list[str]:
    missing = []
    emitted = set()
    for field, label in REQUIRED_TRAINING_CONFIG_FIELDS.items():
        if field in config and config[field] not in (None, ""):
            continue
        if label not in emitted:
            missing.append(f"Missing required training config field: {label}")
            emitted.add(label)
    return missing


def export_selection_is_current(selection_path: str | Path, dataset_dir: str | Path | None) -> bool:
    """True when the exported dataset was built from the selection file as it stands now."""
    if dataset_dir is None:
        return False
    manifest_path = repo_resolve(dataset_dir) / "dataset_manifest.json"
    selection_file = repo_resolve(selection_path)
    if not manifest_path.is_file() or not selection_file.is_file():
        return False
    try:
        recorded = str(load_json(manifest_path).get("dataset_selection_manifest_sha256") or "")
    except Exception:
        return False
    return bool(recorded) and recorded == file_hash(selection_file)


def check_training_readiness(selection_path: str | Path, training_config: str | Path | dict[str, Any]) -> dict[str, Any]:
    selection = load_json(selection_path)
    config = load_json_or_dict(training_config)
    output_dir = selection.get("export", {}).get("output_dir")
    dataset_dir = repo_resolve(output_dir) if output_dir else None
    missing: list[str] = []
    warnings: list[str] = []

    if dataset_dir is None:
        missing.append("Dataset output dir is not configured")
    elif not dataset_dir.exists():
        missing.append("Dataset has not been exported yet")
    elif not (dataset_dir / "dataset_manifest.json").is_file():
        missing.append("Exported dataset is missing dataset_manifest.json")
    if dataset_dir is None or not (dataset_dir / "data.yaml").is_file():
        missing.append("Exported dataset is missing data.yaml")
    elif not yaml_defines_required_classes(dataset_dir / "data.yaml"):
        missing.append("data.yaml must define class 0: ship and class 1: ships")
        warnings.append("data.yaml class names do not match the project classes (0: ship, 1: ships)")
    if dataset_dir is not None:
        classes = data_yaml_classes(dataset_dir / "data.yaml")
        warnings.extend(likely_duplicate_class_name_warnings(list(classes.values())))

    for split in ("train", "val"):
        if dataset_dir is None or not (dataset_dir / "images" / split).is_dir():
            missing.append(f"Exported dataset is missing images/{split}")
        if dataset_dir is None or not (dataset_dir / "labels" / split).is_dir():
            missing.append(f"Exported dataset is missing labels/{split}")
        if dataset_dir is None or not split_has_images(dataset_dir, split):
            missing.append(f"{split} split must contain at least one image")

    if dataset_dir is not None and (dataset_dir / "dataset_manifest.json").is_file() \
            and not export_selection_is_current(selection_path, dataset_dir):
        missing.append(
            "Exported dataset is out of date: it was built from a different version of this "
            "selection manifest; re-export the dataset in step 4"
        )

    if dataset_dir is not None and not split_has_images(dataset_dir, "test"):
        warnings.append("test split has no images; final test metrics will be unavailable")

    missing.extend(config_missing_items(config))

    run_name = config.get("run_name")
    run_group = config.get("run_group") or "default"
    output_root = config.get("output_dir") or WORKSPACE_DIR / "runs"
    if run_name:
        run_dir = resolve_training_run_ref(output_root, str(run_group), str(run_name))
        if run_dir.exists() and not bool(config.get("overwrite")):
            missing.append("output run name would overwrite an existing run; enable overwrite to proceed")

    return {"ready": not missing, "missing": missing, "warnings": warnings}


def config_value(config: dict[str, Any], key: str, fallback: Any = None) -> Any:
    value = config.get(key, fallback)
    if value is None:
        return fallback
    return value


def build_yolo_train_command(training_config: dict[str, Any], dataset_dir: str | Path) -> list[str]:
    dataset_path = repo_resolve(dataset_dir)
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    run_group = str(config_value(training_config, "run_group", "default"))
    run_name = str(training_config["run_name"])
    output_root = repo_resolve(config_value(training_config, "output_dir", WORKSPACE_DIR / "runs"))
    resolve_training_run_ref(output_root, run_group, run_name)
    model = str(training_config['model'])
    local_model = repo_resolve(Path(model).expanduser())
    # Workers run in the workspace; local checkpoints are relative to the repo.
    # Keep upstream aliases available when there is no matching local file.
    if local_model.is_file() or Path(model).parent != Path('.'):
        model = str(local_model)
    command = [
        "yolo",
        "detect",
        "train",
        f"model={model}",
        f"data={data_yaml}",
        f"imgsz={training_config['image_size']}",
        f"epochs={training_config['epochs']}",
        f"batch={training_config['batch']}",
        f"device={resolve_training_device(training_config['device'])}",
        f"seed={training_config['seed']}",
        f"project={output_root / run_group}",
        f"name={run_name}",
    ]
    for optional_key in ("patience", "workers"):
        if optional_key in training_config and training_config[optional_key] is not None:
            command.append(f"{optional_key}={training_config[optional_key]}")
    for key, value in sorted((training_config.get("augmentation") or {}).items()):
        command.append(f"{key}={value}")
    if bool(training_config.get("overwrite")):
        command.append("exist_ok=True")
    return command


def parse_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_epoch(value: Any, fallback: int) -> int:
    parsed = parse_float(value)
    if parsed is None:
        return fallback
    return int(parsed)


def load_training_results_series(results_csv: str | Path) -> dict[str, Any]:
    target = repo_resolve(results_csv)
    if not target.exists():
        return {"status": "waiting", "message": "Waiting for first epoch", "series": {}}

    with target.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [field.strip() for field in (reader.fieldnames or [])]
        recognized = [field for field in fieldnames if field in RECOGNIZED_RESULTS_COLUMNS]
        series: dict[str, list[list[float | int]]] = {field: [] for field in recognized}
        for fallback_epoch, row in enumerate(reader, start=1):
            normalized_row = {str(key).strip(): value for key, value in row.items()}
            epoch = parse_epoch(normalized_row.get("epoch"), fallback_epoch)
            for field in recognized:
                value = parse_float(normalized_row.get(field))
                if value is not None:
                    series[field].append([epoch, value])

    return {"status": "ready", "message": None, "series": series}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def optional_package_version(package_name: str) -> str | None:
    try:
        module = __import__(package_name)
    except ModuleNotFoundError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def package_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package_name in ("ultralytics", "torch"):
        version = optional_package_version(package_name)
        if version is not None:
            versions[package_name] = version
    return versions


def git_info() -> dict[str, Any]:
    info = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return info
    info["commit"] = commit
    info["dirty"] = bool(status)
    return info


def find_review_for_image(image_key: str, reviews_dir: str | Path, *, include_incomplete: bool = False) -> tuple[dict[str, Any], Path]:
    root = repo_resolve(reviews_dir)
    candidates: list[tuple[dict[str, Any], Path]] = []
    for path in sorted(root.glob("**/*.json")):
        review = load_json(path)
        if str(review.get("image_key") or "") != image_key:
            continue
        if include_incomplete or review.get("review_status") == "completed":
            candidates.append((review, path))

    if not candidates:
        raise ValueError(f"No completed review found for image_key={image_key}")
    if len(candidates) > 1:
        paths = ", ".join(str(path) for _, path in candidates)
        raise ValueError(f"Reconciliation conflict for image_key={image_key}: {paths}")
    return candidates[0]


# Canonical implementations live in visual_review_lib so the legacy export script and the
# training pipeline always produce byte-identical labels and data.yaml files.
exportable_detections = review_lib.exportable_detections
write_yolo_label_file = review_lib.write_yolo_label_file


def write_data_yaml(dataset_dir: Path) -> None:
    contained(dataset_dir / "data.yaml", WORKSPACE_DIR)
    review_lib.write_data_yaml(dataset_dir, split_layout=True)


def relative_export_path(dataset_dir: Path, path: Path) -> str:
    return str(path.relative_to(dataset_dir))


def record_exported_file(dataset_dir: Path, path: Path) -> dict[str, Any]:
    return {
        "path": relative_export_path(dataset_dir, path),
        "sha256": file_hash(path),
        "size_bytes": path.stat().st_size,
    }


def background_negative_limit(positive_count: int, requested_count: int, max_fraction: float | None) -> int:
    if max_fraction is None:
        return requested_count
    fraction = float(max_fraction)
    if fraction <= 0.0 or positive_count <= 0:
        return 0
    if fraction >= 1.0:
        return requested_count
    return min(requested_count, math.floor((fraction * positive_count) / (1.0 - fraction)))


def ensure_dataset_dirs(dataset_dir: Path) -> None:
    for split in SPLITS:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def reset_dataset_split_dirs(dataset_dir: Path) -> None:
    """Empty every images/<split> and labels/<split> dir so a re-export equals the selection.

    Without this, an image dropped from the selection (or moved to another split) stays on
    disk and YOLO keeps training on it.
    """
    for kind in ("images", "labels"):
        for split in SPLITS:
            target = contained(dataset_dir / kind / split, WORKSPACE_DIR)
            if target.is_dir():
                shutil.rmtree(target)
    ensure_dataset_dirs(dataset_dir)


def copy_review_image(review: dict[str, Any], image_target: Path) -> None:
    source = repo_resolve(str(review["cache_path"]))
    if not source.exists():
        raise FileNotFoundError(f"Review cache image missing: {source}")
    image_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, image_target)


def export_review_item(
    *,
    dataset_dir: Path,
    review: dict[str, Any],
    item: dict[str, Any],
    detections: list[dict[str, Any]],
) -> list[Path]:
    split = str(item.get("split") or "train")
    if split not in SPLITS:
        raise ValueError(f"Invalid split for image_key={item.get('image_key')}: {split}")

    image_filename = str(review.get("image_filename") or f"{review['image_key']}.png")
    validate_run_segment(image_filename, "image_filename")
    image_stem = Path(image_filename).stem
    image_target = contained(dataset_dir / "images" / split / image_filename, WORKSPACE_DIR)
    label_target = contained(dataset_dir / "labels" / split / f"{image_stem}.txt", WORKSPACE_DIR)
    copy_review_image(review, image_target)
    write_yolo_label_file(label_target, detections)
    return [image_target, label_target]


def export_dataset_selection(
    selection_path: str | Path,
    *,
    output_dir_override: str | Path | None = None,
    include_incomplete_override: bool | None = None,
) -> dict[str, Any]:
    """Build a complete replacement before publishing any exported dataset files."""
    selection_file = repo_resolve(selection_path)
    selection = load_json(selection_file)
    output_dir = output_dir_override or selection.get("export", {}).get("output_dir")
    if not output_dir:
        raise ValueError("Dataset selection manifest is missing export.output_dir")
    dataset_dir = contained(repo_resolve(output_dir), WORKSPACE_DIR)
    if dataset_dir == WORKSPACE_DIR:
        raise ValueError("Dataset export requires its own subdirectory")
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(str(dataset_dir).encode()).hexdigest()
    with exclusive_lock(WORKSPACE_DIR / "datasets" / ".export_locks" / token, blocking=True):
        with tempfile.TemporaryDirectory(prefix=".dataset_export_", dir=dataset_dir.parent) as temporary:
            staging = Path(temporary) / "new"
            manifest = _export_dataset_selection_into(
                selection_file, staging, dataset_dir,
                output_dir_override=output_dir_override,
                include_incomplete_override=include_incomplete_override,
            )
            _publish_dataset_export(staging, dataset_dir, Path(temporary) / "previous")
            return manifest


def _publish_dataset_export(staging: Path, dataset_dir: Path, backup: Path) -> None:
    """Replace managed output entries, restoring the prior export on publish failure."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    backup.mkdir()
    names = ("images", "labels", "data.yaml", "dataset_manifest.json")
    for name in names:
        if (dataset_dir / name).is_symlink():
            raise ValueError(f"Dataset output entry must not be a symbolic link: {dataset_dir / name}")
    replaced, saved = [], []
    try:
        for name in names:
            target = contained(dataset_dir / name, WORKSPACE_DIR)
            if target.exists():
                target.replace(backup / name)
                saved.append(name)
            (staging / name).replace(target)
            replaced.append(name)
    except Exception:
        for name in reversed(replaced):
            target = dataset_dir / name
            target.replace(staging / name)
        for name in reversed(saved):
            (backup / name).replace(dataset_dir / name)
        raise


def _export_dataset_selection_into(
    selection_path: str | Path,
    dataset_dir: Path,
    published_dir: Path,
    *,
    output_dir_override: str | Path | None = None,
    include_incomplete_override: bool | None = None,
) -> dict[str, Any]:
    selection_file = repo_resolve(selection_path)
    selection = load_json(selection_file)
    if int(selection.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("Dataset selection manifest must use schema_version 1")
    if output_dir_override is not None:
        selection.setdefault("export", {})["output_dir"] = str(output_dir_override)
    if include_incomplete_override is not None:
        selection.setdefault("export", {})["include_incomplete_reviews"] = bool(include_incomplete_override)

    reviews_dir = selection.get("sources", {}).get("reviews_dir")
    if not reviews_dir:
        raise ValueError("Dataset selection manifest is missing sources.reviews_dir")
    output_dir = selection.get("export", {}).get("output_dir")
    if not output_dir:
        raise ValueError("Dataset selection manifest is missing export.output_dir")

    include_incomplete = bool(selection.get("export", {}).get("include_incomplete_reviews", False))
    ensure_dataset_dirs(dataset_dir)

    exported_files: list[dict[str, Any]] = []
    exported_image_keys: list[str] = []
    warnings: list[str] = []
    item_count = 0

    for item in selection.get("items", []):
        image_key = str(item.get("image_key") or "")
        if not image_key:
            raise ValueError("Dataset selection item is missing image_key")
        review, _ = find_review_for_image(image_key, reviews_dir, include_incomplete=include_incomplete)
        detections = exportable_detections(review, item)
        for path in export_review_item(dataset_dir=dataset_dir, review=review, item=item, detections=detections):
            exported_files.append(record_exported_file(dataset_dir, path))
        exported_image_keys.append(image_key)
        item_count += 1

    background = selection.get("background_negatives") or {}
    if background.get("enabled"):
        if background.get("mode") != "whole_image_empty_label":
            raise ValueError(f"Unsupported background negative mode: {background.get('mode')}")
        requested_keys = [str(image_key) for image_key in background.get("image_keys") or []]
        limit = background_negative_limit(item_count, len(requested_keys), background.get("max_fraction"))
        if limit < len(requested_keys):
            warnings.append(
                f"Truncated background negatives from {len(requested_keys)} to {limit} because of max_fraction={background.get('max_fraction')}"
            )
        for image_key in requested_keys[:limit]:
            review, _ = find_review_for_image(image_key, reviews_dir, include_incomplete=include_incomplete)
            item = {"image_key": image_key, "split": "train", "include_labels": "selected_only", "selected_detection_ids": []}
            for path in export_review_item(dataset_dir=dataset_dir, review=review, item=item, detections=[]):
                exported_files.append(record_exported_file(dataset_dir, path))
            exported_image_keys.append(image_key)

    write_data_yaml(dataset_dir)
    yaml_path = dataset_dir / "data.yaml"
    # The temporary build directory disappears after publication; YOLO must use
    # the final dataset location recorded here and in the exported checksum.
    yaml_path.write_text(yaml_path.read_text().replace(f"path: {dataset_dir}\n", f"path: {published_dir}\n", 1))
    exported_files.append(record_exported_file(dataset_dir, dataset_dir / "data.yaml"))
    warnings.extend(likely_duplicate_class_name_warnings(list(CLASS_ID_TO_NAME.values())))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": selection.get("dataset_name"),
        "dataset_version": selection.get("dataset_version"),
        "dataset_selection_manifest": str(selection_file),
        "dataset_selection_manifest_sha256": file_hash(selection_file),
        "classes": {str(class_id): class_name for class_id, class_name in CLASS_ID_TO_NAME.items()},
        "exported_image_keys": exported_image_keys,
        "exported_files": exported_files,
        "warnings": warnings,
    }
    write_json(dataset_dir / "dataset_manifest.json", manifest)
    return manifest


def write_training_metadata(
    *,
    run_dir: str | Path,
    training_config_path: str | Path,
    dataset_manifest_path: str | Path,
    command_line: list[str],
    status: str,
    started_at: str,
    ended_at: str | None = None,
) -> Path:
    output_run_dir = resolve_training_run_dir(run_dir)
    output_run_dir.mkdir(parents=True, exist_ok=True)
    config_file = repo_resolve(training_config_path)
    dataset_manifest_file = repo_resolve(dataset_manifest_path)
    data_yaml = dataset_manifest_file.parent / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found next to dataset manifest: {data_yaml}")

    training_config = load_json(config_file)
    starting_weight = repo_resolve(training_config["model"]) if training_config.get("model") else None
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "training_config_path": str(config_file),
        "training_config_sha256": file_hash(config_file),
        "training_config": training_config,
        "starting_weight_sha256": file_hash(starting_weight) if starting_weight and starting_weight.is_file() else None,
        "dataset_manifest_path": str(dataset_manifest_file),
        "dataset_manifest_sha256": file_hash(dataset_manifest_file),
        "data_yaml_path": str(data_yaml),
        "data_yaml_sha256": file_hash(data_yaml),
        "command_line": list(command_line),
        "started_at": started_at,
        "ended_at": ended_at,
        "output_run_dir": str(output_run_dir),
        "package_versions": package_versions(),
        "git": git_info(),
    }
    metadata_path = output_run_dir / "training_metadata.json"
    write_json(metadata_path, metadata)
    return metadata_path


def metric_value(metrics: dict[str, Any] | None, name: str) -> Any:
    if not isinstance(metrics, dict):
        return ""
    return metrics.get(name, "")


def evaluation_markdown(report: dict[str, Any]) -> str:
    promotion = report["promotion"]
    promoted_text = promotion["run_name"] if promotion["promoted"] else "none"
    lines = [
        f"# Evaluation: {report['run_group']}",
        "",
        f"Generated: {report['generated_at']}",
        f"Recommended/selected: {report['recommendation']['run_name'] or 'none'}",
        f"Promoted: {promoted_text}",
        f"Selection reason: {report['recommendation']['reason']}",
        "",
        "## Runs",
        "",
        "Validation columns describe the best.pt epoch (highest mAP50-95, the Ultralytics "
        "fitness for detection), not the final epoch.",
        "",
        "| Run | Best epoch | Val mAP50 | Final val mAP50 | Test mAP50 | Test Precision | Test Recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report.get("runs", []):
        validation = run.get("validation_metrics") or {}
        final_validation = run.get("final_validation_metrics") or {}
        test = run.get("test_metrics") or {}
        best_epoch = run.get("best_validation_epoch")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run.get("run_name", "")),
                    "" if best_epoch is None else str(best_epoch),
                    str(metric_value(validation, "mAP50")),
                    str(metric_value(final_validation, "mAP50")),
                    str(metric_value(test, "mAP50")),
                    str(metric_value(test, "precision")),
                    str(metric_value(test, "recall")),
                ]
            )
            + " |"
        )
    if not report.get("runs"):
        lines.append("| none |  |  |  |  |  |  |")
    if report.get("baseline") is not None:
        lines.extend(["", "## Baseline", "", f"`{json.dumps(report['baseline'], sort_keys=True)}`"])
    return "\n".join(lines) + "\n"


def write_evaluation_report(
    *,
    run_group: str,
    runs: list[dict[str, Any]],
    reports_dir: str | Path = WORKSPACE_DIR / "reports",
    baseline: dict[str, Any] | None = None,
    promoted_run_name: str | None,
    promotion_reason: str,
    generated_at: str | None = None,
) -> dict[str, Path]:
    validate_run_segment(str(run_group), "run_group")
    output_dir = contained(repo_resolve(reports_dir), WORKSPACE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_run = None
    record_path = WORKSPACE_DIR / "model_weights" / "current.json"
    if record_path.is_file():
        try:
            record = load_json(record_path)
            if record.get("run_group") == run_group:
                current_model(WORKSPACE_DIR / "model_weights")
                actual_run = record.get("run_name")
        except (ValueError, OSError):
            pass
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_group": run_group,
        "generated_at": generated_at or utcnow_iso_for_training(),
        "runs": runs,
        "baseline": baseline,
        "recommendation": {"run_name": promoted_run_name, "reason": promotion_reason},
        "promotion": {"promoted": actual_run is not None, "run_name": actual_run,
                      "reason": "Current operational model" if actual_run else "No current promotion from this run group"},
    }
    json_path = output_dir / f"{run_group}_evaluation.json"
    markdown_path = contained(output_dir / f"{run_group}_evaluation.md", WORKSPACE_DIR)
    write_json(json_path, report)
    markdown_path.write_text(evaluation_markdown(report))
    return {"json_path": json_path, "markdown_path": markdown_path}


def utcnow_iso_for_training() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Step 5: training setup
# ---------------------------------------------------------------------------

DEFAULT_TRAINING_CONFIGS_DIR = WORKSPACE_DIR / "training_configs"
DEFAULT_RUNS_DIR = WORKSPACE_DIR / "runs"
DEFAULT_MODEL_WEIGHTS_DIR = WORKSPACE_DIR / "model_weights"
WEIGHT_RECREATION_DOC = ROOT_DIR / "docs" / "yolo_weight_recreation.md"
if not WEIGHT_RECREATION_DOC.is_file():
    WEIGHT_RECREATION_DOC = WORKSPACE_DIR / "model_weights" / "weight_recreation.md"
ALLOWED_AUGMENTATION_KEYS = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
    "copy_paste",
    "erasing",
    "crop_fraction",
)


def normalize_augmentation(augmentation: dict[str, Any] | None) -> dict[str, Any]:
    values = augmentation or {}
    if not isinstance(values, dict):
        raise ValueError("augmentation must be an object of Ultralytics train args")
    unknown = sorted(str(key) for key in values if str(key) not in ALLOWED_AUGMENTATION_KEYS)
    if unknown:
        raise ValueError(
            "Unknown augmentation key(s): " + ", ".join(unknown) + ". Allowed keys: " + ", ".join(ALLOWED_AUGMENTATION_KEYS)
        )
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        parsed = parse_float(value)
        if parsed is None:
            raise ValueError(f"augmentation value for {key} must be numeric")
        normalized[str(key)] = parsed
    return normalized


def augmentation_presets() -> dict[str, dict[str, float]]:
    """Explicit, editable online augmentation values shared by GUI and CLI."""
    none = dict.fromkeys(ALLOWED_AUGMENTATION_KEYS, 0.0)
    none["crop_fraction"] = 1.0
    default = {**none, "hsv_h": .015, "hsv_s": .7, "hsv_v": .4, "degrees": 10.,
               "translate": .1, "scale": .5, "fliplr": .5, "mosaic": 1.}
    return {"none": none,
            "light": {**none, "hsv_h": .01, "hsv_s": .2, "hsv_v": .2, "degrees": 5.,
                      "translate": .05, "scale": .2, "fliplr": .5},
            "default": default,
            "heavy": {**default, "degrees": 180., "flipud": .5, "mixup": .05}}


def doc_mentions_weight_name(weight_name: str, doc_text: str) -> bool:
    """True when `doc_text` names this exact weight file, not merely a longer name.

    A substring test lets `26n.pt` match an existing `yolo26n.pt` entry, so promotion
    would report `documented: true` with no recipe for the promoted destination.
    """
    pattern = r"(?<![A-Za-z0-9_.\-])" + re.escape(str(weight_name)) + r"(?![A-Za-z0-9_.\-])"
    return re.search(pattern, doc_text) is not None


def weight_recreation_doc_text(doc_path: str | Path = WEIGHT_RECREATION_DOC) -> str:
    target = repo_resolve(doc_path)
    if not target.is_file():
        return ""
    return target.read_text()


def weight_is_documented(weight_path: str | Path, *, doc_text: str | None = None, doc_path: str | Path = WEIGHT_RECREATION_DOC) -> bool:
    text = weight_recreation_doc_text(doc_path) if doc_text is None else doc_text
    candidate = repo_resolve(weight_path)
    record_path = candidate.with_suffix(".recreation.json")
    if record_path.is_file() and candidate.is_file():
        try:
            record = load_json(record_path)
            if record.get("sha256") == file_hash(candidate) and (record.get("training_metadata") or record.get("legacy_recreation_document")):
                return True
        except (ValueError, OSError):
            pass
    if not text:
        return False
    candidate = Path(str(weight_path))
    needles = {repo_relative(repo_resolve(candidate)), str(candidate)}
    if candidate.name not in {"best.pt", "last.pt"}:
        # Generic checkpoint names would match unrelated rows, so only match them by path.
        needles.add(candidate.name)
    relative = repo_relative(repo_resolve(candidate))
    parts = Path(relative).parts
    if len(parts) >= 5 and parts[0] == "workspace" and parts[1] == "runs" and parts[-2] == "weights":
        # The recreation doc documents whole run groups with a `<run_name>` placeholder.
        needles.add(str(Path(*parts[:3], "<run_name>", *parts[-2:])))
    return any(needle and needle in text for needle in needles)


def starting_weight_entry(path: Path, source: str, doc_text: str) -> dict[str, Any]:
    documented = weight_is_documented(path, doc_text=doc_text)
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = None
    entry: dict[str, Any] = {
        "path": repo_relative(path),
        "name": path.name,
        "source": source,
        "size_bytes": size_bytes,
        "documented": documented,
        "provenance_warning": None,
    }
    if not documented:
        entry["provenance_warning"] = (
            f"{repo_relative(path)} is not mentioned in docs/yolo_weight_recreation.md; document it before promoting or reusing it"
        )
    return entry


def list_starting_weights(
    *,
    root_dir: str | Path = ROOT_DIR,
    model_weights_dir: str | Path = DEFAULT_MODEL_WEIGHTS_DIR,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    doc_path: str | Path = WEIGHT_RECREATION_DOC,
) -> list[dict[str, Any]]:
    doc_text = weight_recreation_doc_text(doc_path)
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(path: Path, source: str) -> None:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen or not resolved.is_file():
            return
        seen.add(key)
        entries.append(starting_weight_entry(resolved, source, doc_text))

    root = repo_resolve(root_dir)
    for path in sorted(root.glob("*.pt")):
        add(path, "repo_root")
    weights_root = repo_resolve(model_weights_dir)
    if weights_root.is_dir():
        for path in sorted(weights_root.glob("*.pt")):
            add(path, "model_weights")
    runs_root = resolve_training_run_dir(runs_dir)
    if runs_root.is_dir():
        for path in sorted(runs_root.glob("**/weights/best.pt")):
            if path_is_relative_to(path, runs_root):
                add(path, "run_best")
    return entries


# Import staging areas under workspace/datasets: never exported YOLO datasets themselves.
EXCLUDED_DATASET_SUBDIRS = ("raw", "extracted")
DATASET_DISCOVERY_DEPTH = 2


def list_exported_datasets(datasets_dir: str | Path = DEFAULT_DATASETS_DIR) -> list[dict[str, Any]]:
    root = repo_resolve(datasets_dir)
    datasets: list[dict[str, Any]] = []
    if not root.is_dir():
        return datasets
    manifest_paths: list[Path] = []
    for depth in range(1, DATASET_DISCOVERY_DEPTH + 1):
        manifest_paths.extend(root.glob("/".join(["*"] * depth) + "/dataset_manifest.json"))
    seen: set[Path] = set()
    for manifest_path in sorted(set(manifest_paths)):
        dataset_dir = manifest_path.parent
        if dataset_dir in seen:
            continue
        seen.add(dataset_dir)
        relative_parts = dataset_dir.relative_to(root).parts
        if relative_parts[0] in EXCLUDED_DATASET_SUBDIRS:
            continue
        try:
            manifest = load_json(manifest_path)
        except Exception:
            continue
        split_counts = {}
        for split in SPLITS:
            image_dir = dataset_dir / "images" / split
            split_counts[split] = sum(1 for path in image_dir.iterdir() if path.is_file()) if image_dir.is_dir() else 0
        datasets.append({
            "dataset_dir": repo_relative(dataset_dir),
            "dataset_name": manifest.get("dataset_name"),
            "dataset_version": manifest.get("dataset_version"),
            "dataset_manifest": repo_relative(manifest_path),
            "data_yaml": repo_relative(dataset_dir / "data.yaml"),
            "data_yaml_exists": (dataset_dir / "data.yaml").is_file(),
            "image_count": sum(split_counts.values()),
            "split_counts": split_counts,
            "warnings": list(manifest.get("warnings") or []),
            "selection_path": repo_relative(repo_resolve(manifest["dataset_selection_manifest"]))
                if manifest.get("dataset_selection_manifest") and repo_resolve(manifest["dataset_selection_manifest"]).is_file() else None,
        })
    datasets.sort(key=lambda item: str(item.get("dataset_dir") or ""))
    return datasets


def training_config_filename(run_group: str, config_name: str | None = None) -> str:
    slug = slugify(config_name or run_group, "training_config")
    return f"{slug}.json"


TRAINING_COMPUTE_SNIPPET = """
import json, sys, torch
cuda = torch.cuda.is_available()
mps = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
print(json.dumps({'device': 0 if cuda else 'mps' if mps else 'cpu',
    'backend': 'cuda' if cuda else 'mps' if mps else 'cpu',
    'label': torch.cuda.get_device_name(0) if cuda else 'Apple GPU' if mps else 'CPU',
    'python_executable': sys.executable, 'torch_version': torch.__version__}))
"""


def training_compute(python_executable: str | Path | None = None) -> dict[str, Any]:
    """Inspect the interpreter used for training; prefer a GPU when available."""
    interpreter = str(python_executable or sys.executable)
    if os.path.abspath(interpreter) != os.path.abspath(sys.executable):
        completed = subprocess.run([interpreter, "-c", TRAINING_COMPUTE_SNIPPET],
                                   capture_output=True, text=True, timeout=30)
        if completed.returncode:
            raise ValueError("Could not inspect training hardware in " + interpreter + ": " + completed.stderr.strip())
        return json.loads(completed.stdout)
    try:
        import torch
    except ImportError as exc:
        raise ValueError("Automatic hardware selection requires PyTorch in the training Python environment; install the application dependencies or choose CPU explicitly") from exc
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return {"device": 0 if cuda else "mps" if mps else "cpu",
            "backend": "cuda" if cuda else "mps" if mps else "cpu",
            "label": torch.cuda.get_device_name(0) if cuda else "Apple GPU" if mps else "CPU",
            "python_executable": sys.executable, "torch_version": torch.__version__}


def resolve_training_device(device: str | int | None, *, python_executable: str | Path | None = None) -> str | int:
    if device is None or str(device).strip().lower() in {"", "auto"}:
        return training_compute(python_executable)["device"]
    return device


def training_defaults() -> dict[str, Any]:
    compute = training_compute()
    recommended = None
    try:
        model = current_model(DEFAULT_MODEL_WEIGHTS_DIR)
        if model.is_file() and weight_is_documented(model):
            recommended = repo_relative(model)
    except (OSError, ValueError):
        pass
    return {"defaults": {"image_size": 960, "epochs": 150, "batch": 8, "patience": 30,
                         "seed": 42, "workers": 4, "device": "auto", "augmentation": {}},
            "compute": compute, "recommended_model": recommended, "augmentation_presets": augmentation_presets()}


def save_training_config(
    *,
    dataset_name: str,
    dataset_version: str | int,
    model: str,
    run_group: str,
    run_name: str,
    image_size: int = 960,
    epochs: int = 150,
    batch: int = 8,
    patience: int | None = 30,
    seed: int = 42,
    device: str | int = "auto",
    workers: int | None = 4,
    augmentation: dict[str, Any] | None = None,
    overwrite: bool = False,
    output_dir: str | Path = DEFAULT_RUNS_DIR,
    configs_dir: str | Path = DEFAULT_TRAINING_CONFIGS_DIR,
    config_name: str | None = None,
    selection_path: str | Path | None = None,
    doc_path: str | Path = WEIGHT_RECREATION_DOC,
) -> dict[str, Any]:
    clean_group = str(run_group or "").strip()
    clean_run_name = str(run_name or "").strip()
    clean_model = str(model or "").strip()
    if not clean_group:
        raise ValueError("run_group is required")
    if not clean_run_name:
        raise ValueError("run_name is required")
    if not clean_model:
        raise ValueError("model (starting weights) is required")
    # Run group/name become URL path segments and directory names; whitespace makes both
    # ambiguous, so it is rejected at the source rather than escaped everywhere downstream.
    for label, value in (("run_group", clean_group), ("run_name", clean_run_name)):
        if any(character.isspace() for character in value):
            raise ValueError(f"{label} must not contain whitespace: {value!r}")
        if "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError(f"{label} must be a single path segment: {value!r}")
    if not str(dataset_name or "").strip():
        raise ValueError("dataset_name is required")

    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utcnow_iso_for_training(),
        "dataset_name": str(dataset_name).strip(),
        "dataset_version": dataset_version,
        "model": clean_model,
        "image_size": int(image_size),
        "epochs": int(epochs),
        "batch": int(batch),
        "patience": None if patience is None else int(patience),
        "seed": int(seed),
        "device": resolve_training_device(device),
        "device_request": device,
        "training_python": sys.executable,
        "selection_path": repo_relative(repo_resolve(selection_path)) if selection_path is not None else None,
        "workers": None if workers is None else int(workers),
        "augmentation": normalize_augmentation(augmentation),
        "run_group": clean_group,
        "run_name": clean_run_name,
        "output_dir": repo_relative(repo_resolve(output_dir)),
        "overwrite": bool(overwrite),
    }

    run_dir_for_config(config)
    warnings: list[str] = []
    model_path = repo_resolve(clean_model)
    if model_path.is_file():
        if not weight_is_documented(model_path, doc_path=doc_path):
            warnings.append(
                f"Starting weights {repo_relative(model_path)} are not mentioned in docs/yolo_weight_recreation.md"
            )
    elif clean_model.endswith(".pt"):
        warnings.append(f"Starting weights {clean_model} were not found in the repository; Ultralytics will try to resolve them")

    readiness: dict[str, Any] | None = None
    if selection_path is not None:
        readiness = check_training_readiness(selection_path, config)
        warnings.extend(readiness.get("warnings") or [])
    else:
        missing = config_missing_items(config)
        readiness = {"ready": not missing, "missing": missing, "warnings": []}

    target = repo_resolve(configs_dir) / training_config_filename(clean_group, config_name)
    write_json(target, config)
    return {
        "config_path": repo_relative(target),
        "config": config,
        "readiness": readiness,
        "warnings": warnings,
    }


def list_training_configs(configs_dir: str | Path = DEFAULT_TRAINING_CONFIGS_DIR) -> list[dict[str, Any]]:
    root = repo_resolve(configs_dir)
    configs: list[dict[str, Any]] = []
    if not root.is_dir():
        return configs
    for path in sorted(root.glob("*.json")):
        try:
            config = load_json(path)
        except Exception:
            continue
        configs.append({
            "config_path": repo_relative(path),
            "dataset_name": config.get("dataset_name"),
            "dataset_version": config.get("dataset_version"),
            "model": config.get("model"),
            "run_group": config.get("run_group"),
            "run_name": config.get("run_name"),
            "image_size": config.get("image_size"),
            "epochs": config.get("epochs"),
            "batch": config.get("batch"),
            "device": config.get("device"),
            "device_request": config.get("device_request"),
            "selection_path": config.get("selection_path"),
            "seed": config.get("seed"),
            "patience": config.get("patience"),
            "workers": config.get("workers"),
            "augmentation": dict(config.get("augmentation") or {}),
            "output_dir": config.get("output_dir"),
            "overwrite": bool(config.get("overwrite")),
            "missing": config_missing_items(config),
        })
    configs.sort(key=lambda item: str(item.get("config_path") or ""))
    return configs


# ---------------------------------------------------------------------------
# Step 6: launching and monitoring training runs
# ---------------------------------------------------------------------------

PROCESS_RECORD_FILENAME = "process.json"
TRAINING_LOG_FILENAME = "train.log"
TRAINING_METADATA_FILENAME = "training_metadata.json"
TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}
TRAINING_COMPLETION_MARKERS = ("Results saved to", "epochs completed in")
# Artifacts a launch regenerates. A relaunch into the same run directory rotates them
# aside so a failed relaunch can never be judged `completed` from the previous launch.
ROTATED_RUN_ARTIFACTS = (TRAINING_LOG_FILENAME, "results.csv", "test_metrics.json", "test_eval.log")


class TrainingRunActiveError(RuntimeError):
    """Raised when a launch targets a run directory whose previous process is still alive."""

    def __init__(self, message: str, *, run_dir: str, pid: int | None) -> None:
        super().__init__(message)
        self.run_dir = run_dir
        self.pid = pid


def read_process_start_ticks(pid: int) -> int | None:
    """Platform-specific process identity, used to guard against PID reuse."""
    if process_support.IS_WINDOWS:
        return process_support.windows_start_ticks(pid)
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
    except (OSError, ValueError):
        return None
    tail = raw.rpartition(")")[2].split()
    # /proc stat field 22 (1-indexed) is starttime; after `comm` it is offset 19.
    if len(tail) < 20:
        return None
    try:
        return int(tail[19])
    except ValueError:
        return None


def process_state(pid: int) -> str | None:
    """Single-character Linux process state, e.g. `R`, `S`, `Z`."""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
    except (OSError, ValueError):
        return None
    tail = raw.rpartition(")")[2].split()
    return tail[0] if tail else None


def reap_if_child(pid: int) -> None:
    """Best-effort reap so a terminated direct child does not linger as a zombie."""
    try:
        os.waitpid(int(pid), os.WNOHANG)
    except (ChildProcessError, OSError, ValueError):
        pass


def process_is_alive(pid: int | None, start_ticks: int | None = None) -> bool:
    try:
        pid = int(pid)
        start_ticks = int(start_ticks) if start_ticks is not None else None
    except (TypeError, ValueError, OverflowError):
        return False
    if pid <= 0 or pid > sys.maxsize:
        return False
    if process_support.IS_WINDOWS:
        return process_support.windows_process_alive(pid, start_ticks)
    if process_state(int(pid)) == "Z":
        reap_if_child(int(pid))
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except (OSError, ValueError, OverflowError):
        return False
    if start_ticks is None:
        return True
    current = read_process_start_ticks(int(pid))
    if current is None:
        return True
    return current == int(start_ticks)


def resolve_yolo_executable(executable: str | Path | None = None) -> str:
    if executable:
        return str(executable)
    venv_yolo = Path(sys.executable).parent / ("yolo.exe" if os.name == "nt" else "yolo")
    if venv_yolo.is_file():
        return str(venv_yolo)
    found = shutil.which("yolo")
    if found:
        return found
    raise FileNotFoundError("Could not find a `yolo` executable; pass yolo_executable explicitly")


def run_dir_for_config(config: dict[str, Any]) -> Path:
    run_group = str(config_value(config, "run_group", "default"))
    run_name = str(config["run_name"])
    output_root = repo_resolve(config_value(config, "output_dir", DEFAULT_RUNS_DIR))
    return resolve_training_run_ref(output_root, run_group, run_name)


def read_log_tail(log_path: Path, max_lines: int = 40) -> list[str]:
    if not log_path.is_file():
        return []
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return lines[-max_lines:]


def active_run_process(run_dir: Path) -> dict[str, Any] | None:
    """Return the recorded process for `run_dir` when it is still alive, else None."""
    run_dir = resolve_training_run_dir(run_dir)
    process_path = run_dir / PROCESS_RECORD_FILENAME
    if not process_path.is_file():
        return None
    try:
        record = load_json(process_path)
    except Exception:
        return None
    if str(record.get("status") or "running") in TERMINAL_RUN_STATUSES:
        return None
    if not process_is_alive(record.get("pid"), record.get("start_ticks")):
        return None
    return record


def new_launch_id() -> str:
    return f"{utcnow_iso_for_training().replace(':', '').replace('-', '')}_{secrets.token_hex(3)}"


def rotate_previous_launch_artifacts(run_dir: Path, previous_launch_id: str | None) -> list[str]:
    """Move the previous launch's regenerable artifacts aside so this launch starts clean."""
    run_dir = resolve_training_run_dir(run_dir)
    suffix = str(previous_launch_id or "previous")
    rotated: list[str] = []
    for filename in ROTATED_RUN_ARTIFACTS:
        source = run_dir / filename
        if not source.is_file():
            continue
        target = run_dir / f"{source.stem}.{suffix}{source.suffix}"
        index = 1
        while target.exists():
            target = run_dir / f"{source.stem}.{suffix}-{index}{source.suffix}"
            index += 1
        source.rename(target)
        rotated.append(target.name)
    return rotated


def start_training_run(selection_path, training_config_path, *, yolo_executable=None, extra_env=None):
    run_dir = run_dir_for_config(load_json(repo_resolve(training_config_path)))
    token = hashlib.sha256(str(run_dir).encode()).hexdigest()
    try:
        with exclusive_lock(WORKSPACE_DIR / "runs" / ".launch_locks" / token):
            return _start_training_run(selection_path, training_config_path, yolo_executable=yolo_executable, extra_env=extra_env)
    except BlockingIOError:
        raise TrainingRunActiveError("Another launch is in progress for this run", run_dir=repo_relative(run_dir), pid=None) from None


def _start_training_run(
    selection_path: str | Path,
    training_config_path: str | Path,
    *,
    yolo_executable: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    config_file = repo_resolve(training_config_path)
    config = load_json(config_file)
    run_dir = run_dir_for_config(config)
    live = active_run_process(run_dir)
    if live is not None:
        raise TrainingRunActiveError(
            f"A training process (pid {live.get('pid')}) is still running in {repo_relative(run_dir)}; "
            "stop it before launching this run again",
            run_dir=repo_relative(run_dir),
            pid=live.get("pid"),
        )
    readiness = check_training_readiness(selection_path, config)
    if not readiness["ready"]:
        raise ValueError("Training is not ready: " + "; ".join(readiness["missing"]))

    selection = load_json(selection_path)
    dataset_dir = repo_resolve(selection.get("export", {}).get("output_dir"))
    command_line = build_yolo_train_command(config, dataset_dir)
    if "exist_ok=True" not in command_line:
        # The pipeline owns and pre-creates the run directory for train.log/process.json, so
        # Ultralytics must reuse it instead of auto-incrementing the run name to `<run_name>2`.
        command_line.append("exist_ok=True")
    executable = resolve_yolo_executable(yolo_executable)
    prefix = [sys.executable, executable] if Path(executable).suffix == ".py" else [executable]
    argv = [*prefix, *command_line[1:]]

    run_dir.mkdir(parents=True, exist_ok=True)
    previous_record: dict[str, Any] = {}
    previous_process_path = run_dir / PROCESS_RECORD_FILENAME
    if previous_process_path.is_file():
        try:
            previous_record = load_json(previous_process_path)
        except Exception:
            previous_record = {}
    rotated = rotate_previous_launch_artifacts(run_dir, previous_record.get("launch_id"))
    launch_id = new_launch_id()
    log_path = run_dir / TRAINING_LOG_FILENAME
    started_at = utcnow_iso_for_training()

    env = dict(os.environ)
    env.update(extra_env or {})
    # `w`, never `a`: a relaunch must not inherit the previous launch's completion markers.
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"# start_training_run {started_at} launch_id={launch_id}\n# {' '.join(argv)}\n")
        log_handle.flush()
        process = subprocess.Popen(
            argv,
            cwd=str(WORKSPACE_DIR),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            **process_support.detached_options(),
            env=env,
        )

    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "launch_id": launch_id,
        "rotated_previous_artifacts": rotated,
        "pid": process.pid,
        "pgid": process.pid,
        "start_ticks": read_process_start_ticks(process.pid),
        "started_at": started_at,
        "ended_at": None,
        "return_code": None,
        "reason": None,
        "command_line": list(command_line),
        "argv": list(argv),
        "cwd": str(WORKSPACE_DIR),
        "log_path": repo_relative(log_path),
        "run_dir": repo_relative(run_dir),
        "training_config_path": repo_relative(config_file),
        "selection_path": repo_relative(repo_resolve(selection_path)),
        "expected_epochs": int(config.get("epochs") or 0),
    }
    write_json(run_dir / PROCESS_RECORD_FILENAME, record)
    write_training_metadata(
        run_dir=run_dir,
        training_config_path=config_file,
        dataset_manifest_path=dataset_dir / "dataset_manifest.json",
        command_line=command_line,
        status="running",
        started_at=started_at,
    )
    return {
        "run_dir": repo_relative(run_dir),
        "log_path": repo_relative(log_path),
        "process_path": repo_relative(run_dir / PROCESS_RECORD_FILENAME),
        "training_metadata_path": repo_relative(run_dir / TRAINING_METADATA_FILENAME),
        "pid": process.pid,
        "status": "running",
        "launch_id": launch_id,
        "rotated_previous_artifacts": rotated,
        "started_at": started_at,
        "command_line": list(command_line),
        "argv": list(argv),
        "warnings": list(readiness.get("warnings") or []),
    }


def update_training_metadata_status(run_dir: Path, status: str, *, ended_at: str | None = None, reason: str | None = None) -> None:
    run_dir = resolve_training_run_dir(run_dir)
    metadata_path = run_dir / TRAINING_METADATA_FILENAME
    if not metadata_path.is_file():
        return
    try:
        metadata = load_json(metadata_path)
    except Exception:
        return
    metadata["status"] = status
    if ended_at is not None:
        metadata["ended_at"] = ended_at
    if reason is not None:
        metadata["status_reason"] = reason
    weights_dir = run_dir / "weights"
    metadata["best_weights_path"] = repo_relative(weights_dir / "best.pt") if (weights_dir / "best.pt").is_file() else None
    metadata["last_weights_path"] = repo_relative(weights_dir / "last.pt") if (weights_dir / "last.pt").is_file() else None
    write_json(metadata_path, metadata)


def results_summary(series: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    run_dir = resolve_training_run_dir(run_dir)
    data = series.get("series") or {}
    epochs = [point[0] for points in data.values() for point in points]
    current_epoch = max(epochs) if epochs else 0
    best: dict[str, Any] = {}
    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"):
        points = data.get(key) or []
        if points:
            best_point = max(points, key=lambda point: point[1])
            best[key] = {"epoch": best_point[0], "value": best_point[1]}
    latest: dict[str, Any] = {}
    for key, points in data.items():
        if points:
            latest[key] = points[-1][1]
    return {
        "current_epoch": current_epoch,
        "best_metrics": best,
        "latest_metrics": latest,
        "results_csv": repo_relative(run_dir / "results.csv") if (run_dir / "results.csv").is_file() else None,
    }


def training_completed_successfully(run_dir: Path, log_lines: list[str], summary: dict[str, Any], expected_epochs: int) -> bool:
    run_dir = resolve_training_run_dir(run_dir)
    if any(marker in line for line in log_lines for marker in TRAINING_COMPLETION_MARKERS):
        return True
    best_weights = run_dir / "weights" / "best.pt"
    if best_weights.is_file() and expected_epochs and int(summary.get("current_epoch") or 0) >= int(expected_epochs):
        return True
    return False


def training_status(run_dir: str | Path, *, log_lines: int = 40) -> dict[str, Any]:
    target = resolve_training_run_dir(run_dir)
    process_path = target / PROCESS_RECORD_FILENAME
    if not process_path.is_file():
        raise ValueError(f"No pipeline-managed training run found at {repo_relative(target)}")
    warnings: list[str] = []
    record_readable = True
    try:
        record = load_json(process_path)
    except (OSError, ValueError):
        record = {}
        record_readable = False
        warnings.append("process.json could not be parsed; inferred terminal state from run artifacts")

    log_path = target / TRAINING_LOG_FILENAME
    if record.get("log_path"):
        recorded_log = safe_file_in_roots(record["log_path"], [target])
        if recorded_log is None:
            warnings.append("process log_path is outside the run directory; using default train.log")
        else:
            log_path = recorded_log
    safe_log = safe_file_in_roots(log_path, [target])
    if safe_log is None:
        warnings.append("train.log is outside the run directory; log unavailable")
    # A completion check needs the whole log, but only a tail is returned.
    all_lines = read_log_tail(safe_log, max_lines=100000) if safe_log is not None else []
    tail = all_lines[-log_lines:] if log_lines else []
    series = load_training_results_series(target / "results.csv")
    summary = results_summary(series, target)

    status = str(record.get("status") or "running")
    alive = process_is_alive(record.get("pid"), record.get("start_ticks"))
    changed = False
    if status == "running" and not alive:
        expected_epochs = int(record.get("expected_epochs") or 0)
        if training_completed_successfully(target, all_lines, summary, expected_epochs):
            status = "completed"
            record["reason"] = "training_process_finished"
        else:
            status = "failed"
            record["reason"] = "process_missing"
        record["status"] = status
        record["ended_at"] = record.get("ended_at") or utcnow_iso_for_training()
        changed = True
    if changed and record_readable:
        write_json(process_path, record)
        update_training_metadata_status(target, status, ended_at=record.get("ended_at"), reason=record.get("reason"))

    weights_dir = target / "weights"
    return {
        "run_dir": repo_relative(target),
        "run_group": target.parent.name,
        "run_name": target.name,
        "status": status,
        "terminal": status in TERMINAL_RUN_STATUSES,
        "alive": alive,
        "pid": record.get("pid"),
        "reason": record.get("reason"),
        "warnings": warnings,
        "started_at": record.get("started_at"),
        "ended_at": record.get("ended_at"),
        "expected_epochs": record.get("expected_epochs"),
        "command_line": list(record.get("command_line") or []),
        "log_path": repo_relative(safe_log) if safe_log is not None and safe_log.is_file() else None,
        "log_tail": tail,
        "results": series,
        "current_epoch": summary["current_epoch"],
        "best_metrics": summary["best_metrics"],
        "latest_metrics": summary["latest_metrics"],
        "results_csv": summary["results_csv"],
        "best_weights_path": repo_relative(weights_dir / "best.pt") if (weights_dir / "best.pt").is_file() else None,
        "last_weights_path": repo_relative(weights_dir / "last.pt") if (weights_dir / "last.pt").is_file() else None,
        "training_metadata_path": repo_relative(target / TRAINING_METADATA_FILENAME) if (target / TRAINING_METADATA_FILENAME).is_file() else None,
    }


def stop_training_run(run_dir: str | Path, *, grace_seconds: float = 10.0) -> dict[str, Any]:
    target = resolve_training_run_dir(run_dir)
    process_path = target / PROCESS_RECORD_FILENAME
    if not process_path.is_file():
        raise ValueError(f"No pipeline-managed training run found at {repo_relative(target)}")
    try:
        record = load_json(process_path)
    except (OSError, ValueError) as exc:
        raise ValueError("process.json could not be parsed; no process can safely be stopped") from exc
    pid = record.get("pid")
    status = str(record.get("status") or "running")
    if status in TERMINAL_RUN_STATUSES:
        return {"run_dir": repo_relative(target), "status": status, "stopped": False, "reason": "run already terminal"}
    if not process_is_alive(pid, record.get("start_ticks")):
        record["status"] = "failed"
        record["reason"] = "process_missing"
        record["ended_at"] = record.get("ended_at") or utcnow_iso_for_training()
        write_json(process_path, record)
        update_training_metadata_status(target, "failed", ended_at=record["ended_at"], reason="process_missing")
        return {"run_dir": repo_relative(target), "status": "failed", "stopped": False, "reason": "process_missing"}

    if process_support.IS_WINDOWS:
        escalated = process_support.stop_windows_tree(int(pid), record.get("start_ticks"), float(grace_seconds))
    else:
        escalated = False
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(int(pid), signal.SIGTERM)
            except OSError:
                pass
        deadline = time.monotonic() + float(grace_seconds)
        while time.monotonic() < deadline:
            if not process_is_alive(pid, record.get("start_ticks")):
                break
            time.sleep(0.2)
        if process_is_alive(pid, record.get("start_ticks")):
            escalated = True
            try:
                os.killpg(int(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except OSError:
                    pass
            time.sleep(0.5)

    ended_at = utcnow_iso_for_training()
    record["status"] = "cancelled"
    record["reason"] = "stopped_by_user"
    record["ended_at"] = ended_at
    write_json(process_path, record)
    update_training_metadata_status(target, "cancelled", ended_at=ended_at, reason="stopped_by_user")
    return {
        "run_dir": repo_relative(target),
        "status": "cancelled",
        "stopped": True,
        "escalated_to_sigkill": escalated,
        "pid": pid,
        "ended_at": ended_at,
    }


def list_training_runs(runs_dir: str | Path = DEFAULT_RUNS_DIR, *, active_only: bool = False) -> list[dict[str, Any]]:
    root = resolve_training_run_dir(runs_dir)
    runs: list[dict[str, Any]] = []
    if not root.is_dir():
        return runs
    for process_path in sorted(root.glob("*/*/" + PROCESS_RECORD_FILENAME)):
        if not path_is_relative_to(process_path, root):
            continue
        try:
            status = training_status(process_path.parent, log_lines=5)
        except Exception:
            continue
        if active_only and status["terminal"]:
            continue
        runs.append({
            "run_dir": status["run_dir"],
            "run_group": status["run_group"],
            "run_name": status["run_name"],
            "status": status["status"],
            "alive": status["alive"],
            "pid": status["pid"],
            "started_at": status["started_at"],
            "ended_at": status["ended_at"],
            "current_epoch": status["current_epoch"],
            "expected_epochs": status["expected_epochs"],
        })
    runs.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
    return runs


# ---------------------------------------------------------------------------
# Step 7: evaluation and weight promotion
# ---------------------------------------------------------------------------

TEST_METRICS_FILENAME = "test_metrics.json"
RESULTS_METRIC_ALIASES = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
}
VAL_METRICS_SNIPPET = """
import json, sys
from ultralytics import YOLO

weights, data_yaml, imgsz, device, split, output = sys.argv[1:7]
metrics = YOLO(weights).val(data=data_yaml, imgsz=int(imgsz), device=device, split=split, verbose=False)
results = {}
for key, value in (getattr(metrics, "results_dict", {}) or {}).items():
    try:
        results[str(key)] = float(value)
    except (TypeError, ValueError):
        results[str(key)] = str(value)
payload = {
    "split": split,
    "weights": weights,
    "data_yaml": data_yaml,
    "imgsz": int(imgsz),
    "device": device,
    "results_dict": results,
    "metrics": {
        "precision": results.get("metrics/precision(B)"),
        "recall": results.get("metrics/recall(B)"),
        "mAP50": results.get("metrics/mAP50(B)"),
        "mAP50-95": results.get("metrics/mAP50-95(B)"),
    },
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\\n")
"""


def read_results_rows(results_csv: str | Path) -> list[dict[str, str]]:
    target = repo_resolve(results_csv)
    if not target.is_file():
        return []
    with target.open(newline="") as handle:
        return [{str(key).strip(): value for key, value in row.items()} for row in csv.DictReader(handle)]


def row_metrics(row: dict[str, str] | None) -> dict[str, float]:
    if not row:
        return {}
    metrics: dict[str, float] = {}
    for column, alias in RESULTS_METRIC_ALIASES.items():
        value = parse_float(row.get(column))
        if value is not None:
            metrics[alias] = value
    return metrics


def final_results_row(results_csv: str | Path) -> dict[str, float]:
    rows = read_results_rows(results_csv)
    return row_metrics(rows[-1] if rows else None)


def best_results_row(results_csv: str | Path) -> tuple[dict[str, float], int | None]:
    """Return the validation row for the epoch `best.pt` was saved from, and its epoch.

    Ultralytics saves `best.pt` on the highest *fitness*, which for detection is
    mAP50-95 (weights [0, 0, 0, 1]); ties keep the earliest epoch because the trainer
    only overwrites on a strict improvement. Reporting the last row instead makes a run
    that peaked early and collapsed look as bad as its final epoch, while the promoted
    checkpoint is the good one.
    """
    rows = read_results_rows(results_csv)
    if not rows:
        return {}, None

    def fitness(row: dict[str, str]) -> float | None:
        for column in ("metrics/mAP50-95(B)", "metrics/mAP50(B)"):
            value = parse_float(row.get(column))
            if value is not None:
                return value
        return None

    best_index = None
    best_value = None
    for index, row in enumerate(rows):
        value = fitness(row)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_index = index
    if best_index is None:
        return row_metrics(rows[-1]), parse_epoch(rows[-1].get("epoch"), len(rows))
    best_row = rows[best_index]
    return row_metrics(best_row), parse_epoch(best_row.get("epoch"), best_index + 1)


def run_evaluation_summary(run_dir: str | Path) -> dict[str, Any]:
    target = resolve_training_run_dir(run_dir)
    weights = target / "weights" / "best.pt"
    test_metrics_path = target / TEST_METRICS_FILENAME
    test_metrics = None
    if test_metrics_path.is_file():
        try:
            test_metrics = (load_json(test_metrics_path).get("metrics") or None)
        except Exception:
            test_metrics = None
    best_metrics, best_epoch = best_results_row(target / "results.csv")
    return {
        "run_name": target.name,
        "run_dir": repo_relative(target),
        "weights_path": repo_relative(weights) if weights.is_file() else None,
        # `validation_metrics` describes the checkpoint that is offered for promotion
        # (best.pt), so it is the best epoch's row, not the last one.
        "validation_metrics": best_metrics,
        "best_validation_epoch": best_epoch,
        "final_validation_metrics": final_results_row(target / "results.csv"),
        "test_metrics": test_metrics,
        "training_metadata_path": repo_relative(target / TRAINING_METADATA_FILENAME) if (target / TRAINING_METADATA_FILENAME).is_file() else None,
    }


def list_run_groups(runs_dir: str | Path = DEFAULT_RUNS_DIR) -> list[dict[str, Any]]:
    """List evaluatable run groups: any runs_dir child holding at least one run with results.csv."""
    root = resolve_training_run_dir(runs_dir)
    groups: list[dict[str, Any]] = []
    if not root.is_dir():
        return groups
    for group_dir in sorted(root.iterdir()):
        if not path_is_relative_to(group_dir, root) or not group_dir.is_dir():
            continue
        run_names = [child.name for child in sorted(group_dir.iterdir()) if path_is_relative_to(child, root) and child.is_dir() and (child / "results.csv").is_file()]
        if not run_names:
            continue
        report_path = repo_resolve(WORKSPACE_DIR / "reports") / f"{group_dir.name}_evaluation.md"
        groups.append({
            "run_group": group_dir.name,
            "run_dir": repo_relative(group_dir),
            "run_count": len(run_names),
            "run_names": run_names,
            "promoted_weight_count": sum(1 for name in run_names if (group_dir / name / "weights" / "best.pt").is_file()),
            "test_metrics_count": sum(1 for name in run_names if (group_dir / name / TEST_METRICS_FILENAME).is_file()),
            "report_markdown_path": repo_relative(report_path) if report_path.is_file() else None,
        })
    groups.sort(key=lambda item: str(item.get("run_group") or ""))
    return groups


def collect_run_group_summaries(run_group: str, runs_dir: str | Path = DEFAULT_RUNS_DIR) -> list[dict[str, Any]]:
    group_dir = resolve_training_run_ref(runs_dir, run_group)
    if not group_dir.is_dir():
        raise ValueError(f"Run group not found: {repo_relative(group_dir)}")
    summaries = []
    for child in sorted(group_dir.iterdir()):
        if path_is_relative_to(child, group_dir) and child.is_dir() and (child / "results.csv").is_file():
            summaries.append(run_evaluation_summary(child))
    return summaries


def best_run_by_metric(runs: list[dict[str, Any]]) -> tuple[str | None, str]:
    def score(run: dict[str, Any]) -> tuple[int, float]:
        test = (run.get("test_metrics") or {}).get("mAP50")
        if test is not None:
            return (1, float(test))
        validation = (run.get("validation_metrics") or {}).get("mAP50")
        return (0, float(validation)) if validation is not None else (0, -1.0)

    scored = [(score(run), run) for run in runs]
    scored = [item for item in scored if item[0][1] >= 0.0]
    if not scored:
        return None, "No run in this group reported mAP50 metrics."
    best = max(scored, key=lambda item: item[0])
    basis = "test mAP50" if best[0][0] == 1 else "validation mAP50"
    return str(best[1]["run_name"]), f"Highest {basis} ({best[0][1]:.4f}) in run group."


def build_run_group_evaluation_report(
    *,
    run_group: str,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    reports_dir: str | Path | None = None,
    baseline: dict[str, Any] | None = None,
    promoted_run_name: str | None = None,
    promotion_reason: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    runs = collect_run_group_summaries(run_group, runs_dir)
    if promoted_run_name is None:
        promoted_run_name, auto_reason = best_run_by_metric(runs)
    else:
        auto_reason = f"Run {promoted_run_name} selected by the caller."
    paths = write_evaluation_report(
        run_group=run_group,
        runs=runs,
        reports_dir=reports_dir or WORKSPACE_DIR / "reports",
        baseline=baseline,
        promoted_run_name=promoted_run_name,
        promotion_reason=promotion_reason or auto_reason,
        generated_at=generated_at,
    )
    return {
        "run_group": run_group,
        "run_count": len(runs),
        "runs": runs,
        "selected_run_name": promoted_run_name,
        "promoted_run_name": load_json(paths["json_path"])["promotion"]["run_name"],
        "promotion_reason": promotion_reason or auto_reason,
        "report_path": repo_relative(paths["json_path"]),
        "report_markdown_path": repo_relative(paths["markdown_path"]),
    }


def run_test_split(
    run_dir: str | Path,
    *,
    dataset_dir: str | Path | None = None,
    device: str | int = "auto",
    image_size: int | None = None,
    python_executable: str | Path | None = None,
    timeout_seconds: float | None = 3600.0,
) -> dict[str, Any]:
    target = resolve_training_run_dir(run_dir)
    weights = target / "weights" / "best.pt"
    if not weights.is_file():
        raise ValueError(f"Run has no best.pt to evaluate: {repo_relative(weights)}")
    metadata_path = target / TRAINING_METADATA_FILENAME
    if dataset_dir is not None:
        data_yaml = repo_resolve(dataset_dir) / "data.yaml"
        imgsz = image_size or 640
    else:
        if not metadata_path.is_file():
            raise ValueError(f"Run is missing {TRAINING_METADATA_FILENAME}; pass dataset_dir explicitly")
        metadata = load_json(metadata_path)
        data_yaml = repo_resolve(metadata["data_yaml_path"])
        imgsz = image_size or int((metadata.get("training_config") or {}).get("image_size") or 640)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    interpreter = str(python_executable or sys.executable)
    device = resolve_training_device(device, python_executable=interpreter)
    output_path = target / TEST_METRICS_FILENAME
    log_path = target / "test_eval.log"
    if safe_file_in_roots(log_path, [target]) is None:
        raise ValueError("test_eval.log must be inside the run directory")
    argv = [interpreter, "-c", VAL_METRICS_SNIPPET, str(weights), str(data_yaml), str(imgsz), str(device), "test", str(output_path)]
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"# run_test_split {utcnow_iso_for_training()}\n")
        log_handle.flush()
        completed = subprocess.run(
            argv,
            cwd=str(WORKSPACE_DIR),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(f"Test-split evaluation failed (exit {completed.returncode}); see {repo_relative(log_path)}")
    payload = load_json(output_path)
    return {
        "run_dir": repo_relative(target),
        "test_metrics_path": repo_relative(output_path),
        "log_path": repo_relative(log_path),
        "metrics": payload.get("metrics") or {},
    }


def promote_weight(
    *,
    run_group: str,
    run_name: str,
    target_name: str,
    recreation_doc_updated: bool = False,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    model_weights_dir: str | Path = DEFAULT_MODEL_WEIGHTS_DIR,
    doc_path: str | Path = WEIGHT_RECREATION_DOC,
    overwrite: bool = False,
) -> dict[str, Any]:
    clean_target = str(target_name or "").strip()
    if not clean_target:
        raise ValueError("target_name is required")
    if not clean_target.lower().endswith(".pt"):
        clean_target += ".pt"
    if clean_target.casefold() == default_manifest()["filename"].casefold():
        raise ValueError("The bundled default is reserved; choose a different target_name")
    if "/" in clean_target or "\\" in clean_target:
        raise ValueError("target_name must be a bare filename, not a path")

    run_dir = resolve_training_run_ref(runs_dir, run_group, run_name)
    source = resolve_training_run_dir(run_dir / "weights" / "best.pt")
    if not source.is_file():
        raise ValueError(f"Run weights not found: {repo_relative(source)}")

    destination = contained(repo_resolve(model_weights_dir) / clean_target, WORKSPACE_DIR)
    metadata_path = run_dir / TRAINING_METADATA_FILENAME
    if metadata_path.is_file():
        metadata = load_json(metadata_path)
        required = ("command_line", "training_config", "training_config_sha256",
                    "dataset_manifest_sha256", "data_yaml_sha256", "git", "package_versions")
        missing = [key for key in required if not metadata.get(key)]
        if missing:
            raise ValueError("Training recreation metadata is incomplete: " + ", ".join(missing))
        provenance = {"training_metadata": metadata}
        gaps = ["Exact numerical reproduction can vary with hardware and library versions."]
        # Preserve available inputs as well as their recorded training-time hashes.
        for name, path_key, hash_key in (
            ("dataset_manifest", "dataset_manifest_path", "dataset_manifest_sha256"),
            ("data_yaml", "data_yaml_path", "data_yaml_sha256"),
        ):
            input_path = repo_resolve(metadata[path_key]) if metadata.get(path_key) else None
            if input_path and input_path.is_file() and file_hash(input_path) == metadata[hash_key]:
                provenance[name] = input_path.read_text(encoding="utf-8")
            else:
                gaps.append(f"{name} is unavailable or has changed; retain the original input matching its recorded hash.")
        if not metadata["git"].get("commit") or metadata["git"].get("dirty"):
            gaps.append("Source revision is unavailable or includes uncommitted changes; retain the original application source.")
        if not metadata.get("starting_weight_sha256"):
            gaps.append("Starting weight hash was not recorded by this training version; retain the model named in training_config.")
    else:
        # Older/manual runs may still use an explicit existing recreation recipe.
        doc_text = weight_recreation_doc_text(doc_path)
        if not recreation_doc_updated or not doc_mentions_weight_name(clean_target, doc_text):
            raise ValueError("Training recreation metadata is missing; train through Improve to record provenance automatically, "
                             "or supply a documented legacy recreation recipe with recreation_doc_updated")
        provenance = {"legacy_recreation_document": doc_text}
        gaps = ["Legacy run: training metadata was not recorded; see the supplied recreation recipe."]
    if destination.exists() and not overwrite:
        raise ValueError(f"Refusing to overwrite existing weight {repo_relative(destination)}; pass overwrite to replace it")

    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = file_hash(source)
    recreation_path = contained(destination.with_suffix(".recreation.json"), WORKSPACE_DIR)
    write_json(recreation_path, {
        "schema_version": SCHEMA_VERSION, "created_at": utcnow_iso_for_training(),
        "filename": clean_target, "sha256": digest, "source_path": repo_relative(source),
        "run_group": run_group, "run_name": run_name, "reproducibility_gaps": gaps, **provenance,
    })
    shutil.copy2(source, destination)
    set_current_model(destination, recreation_path=repo_relative(recreation_path), run_group=run_group, run_name=run_name, metrics=run_evaluation_summary(run_dir)["validation_metrics"])
    return {
        "run_group": str(run_group),
        "run_name": str(run_name),
        "source_path": repo_relative(source),
        "promoted_path": repo_relative(destination),
        "recreation_path": repo_relative(recreation_path),
        "sha256": file_hash(destination),
        "size_bytes": destination.stat().st_size,
        "documented": True,
    }
