from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
try:
    from .workspace import WORKSPACE_DIR, contained, atomic_json
except ImportError:
    from workspace import WORKSPACE_DIR, contained, atomic_json
DEFAULT_SOURCE_DIR = WORKSPACE_DIR / "inputs" / "imagery"
DEFAULT_CACHE_DIR = WORKSPACE_DIR / "cache" / "sentinel_tiles"
DEFAULT_REVIEWS_DIR = WORKSPACE_DIR / "reviews"
DEFAULT_MANIFEST_PATH = WORKSPACE_DIR / "manifests" / "visual_preprocess_manifest.json"
DEFAULT_EXPORT_DIR = WORKSPACE_DIR / "datasets" / "sentinel_reviewed"
DEFAULT_REVIEWER = os.environ.get("USER") or "default"

TILE_NAME_SUFFIXES = {".tif", ".tiff"}
TILE_NAME_TEMPLATE = "{date}_tile_{row}_{col}.tiff"
CLASS_ID_TO_NAME = {0: "ship", 1: "ships"}
CLASS_NAME_TO_ID = {value: key for key, value in CLASS_ID_TO_NAME.items()}
REVIEW_LABEL_TO_CLASS = {"ship_non_sts": 0, "ships_sts": 1}
CLASS_ID_TO_REVIEW_LABEL = {value: key for key, value in REVIEW_LABEL_TO_CLASS.items()}
EXPORTABLE_REVIEW_LABELS = set(REVIEW_LABEL_TO_CLASS)
RENDER_PROFILE_REQUIRED_FIELDS = {
    "rgb_original": {"kind"},
    "true_color": {"kind", "bands", "normalization", "output_colorspace"},
    "false_color": {"kind", "bands", "normalization", "output_colorspace"},
}
ALLOWED_RENDER_PROFILE_KINDS = set(RENDER_PROFILE_REQUIRED_FIELDS)
TRUE_COLOR_RENDER_PROFILE = {
    "kind": "true_color",
    "bands": [1, 2, 3],
    "normalization": "current_loader_default",
    "output_colorspace": "RGB",
}


@dataclass(frozen=True)
class TileInfo:
    date: str
    row: int
    col: int
    path: Path
    mtime: float
    size_bytes: int


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clamp_bbox_to_unit(x_center: float, y_center: float, width: float, height: float) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(1.0, float(x_center) - float(width) / 2.0))
    x2 = max(0.0, min(1.0, float(x_center) + float(width) / 2.0))
    y1 = max(0.0, min(1.0, float(y_center) - float(height) / 2.0))
    y2 = max(0.0, min(1.0, float(y_center) + float(height) / 2.0))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def repo_relative(path: str | Path) -> str:
    candidate = Path(path).resolve()
    try:
        return str(candidate.relative_to(ROOT_DIR))
    except ValueError:
        return str(candidate)


def repo_resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def cache_url_for(path):
    target = repo_resolve(path).resolve()
    root = (WORKSPACE_DIR / "cache").resolve()
    if not target.is_relative_to(root):
        return None
    return "/cache/workspace/cache/" + quote(target.relative_to(root).as_posix(), safe="/")


def resolve_cache_resource(resource):
    cleaned = unquote(resource).replace("\\", "/")
    if cleaned.startswith("workspace/"):
        cleaned = cleaned[len("workspace/"):]
    target = (WORKSPACE_DIR / cleaned).resolve()
    if not target.is_relative_to((WORKSPACE_DIR / "cache").resolve()) or not target.is_file():
        return None
    return target


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=json_default)


def source_content_hash(source_path: str | Path) -> str:
    target = repo_resolve(source_path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def validate_render_profile(render_profile: dict[str, Any]) -> None:
    if not isinstance(render_profile, dict):
        raise ValueError("render_profile must be a JSON object")
    kind = render_profile.get("kind")
    if kind not in ALLOWED_RENDER_PROFILE_KINDS:
        raise ValueError(f"Unknown render profile kind: {kind!r}")
    required = RENDER_PROFILE_REQUIRED_FIELDS[str(kind)]
    missing = sorted(field for field in required if field not in render_profile)
    if missing:
        raise ValueError(f"render_profile kind={kind!r} is missing required field(s): {', '.join(missing)}")


def canonical_identity_payload(source_path: str | Path, render_profile: dict[str, Any]) -> dict[str, Any]:
    validate_render_profile(render_profile)
    return {
        "source_content_hash": source_content_hash(source_path),
        "render_profile": render_profile,
    }


def image_key_for_identity_payload(payload: dict[str, Any]) -> str:
    validate_render_profile(payload.get("render_profile"))
    identity_bytes = canonical_json(payload).encode("utf-8")
    return "img_" + hashlib.sha256(identity_bytes).hexdigest()[:16]


def image_key_for_source(source_path: str | Path, render_profile: dict[str, Any]) -> str:
    return image_key_for_identity_payload(canonical_identity_payload(source_path, render_profile))


def load_json(path: str | Path, default: Any | None = None) -> Any:
    target = repo_resolve(path)
    if not target.exists():
        return {} if default is None else default
    return json.loads(target.read_text())


def write_json(path: str | Path, payload: Any) -> None:
    target = contained(repo_resolve(path), WORKSPACE_DIR)
    atomic_json(target, json.loads(json.dumps(payload, default=json_default)))


def parse_tile_name(tile_path: str | Path) -> TileInfo:
    path = Path(tile_path).resolve()
    suffix = path.suffix.lower()
    if suffix not in TILE_NAME_SUFFIXES:
        raise ValueError(f"Unsupported tile suffix for {path.name}")

    stem_parts = path.stem.split("_tile_")
    if len(stem_parts) != 2:
        raise ValueError(f"Tile name does not match {TILE_NAME_TEMPLATE}: {path.name}")
    date_part, grid_part = stem_parts
    grid_tokens = grid_part.split("_")
    if len(grid_tokens) != 2:
        raise ValueError(f"Tile name does not match {TILE_NAME_TEMPLATE}: {path.name}")

    row = int(grid_tokens[0])
    col = int(grid_tokens[1])
    stat = path.stat()
    return TileInfo(date=date_part, row=row, col=col, path=path, mtime=stat.st_mtime, size_bytes=stat.st_size)


def discover_tiles(source_dir: str | Path = DEFAULT_SOURCE_DIR) -> dict[str, list[TileInfo]]:
    source_root = repo_resolve(source_dir)
    groups: dict[str, list[TileInfo]] = {}
    for candidate in sorted(source_root.rglob("*.tif*")):
        if not candidate.is_file():
            continue
        tile = parse_tile_name(candidate)
        groups.setdefault(tile.date, []).append(tile)

    for tiles in groups.values():
        tiles.sort(key=lambda item: (item.row, item.col, item.path.name))
    return dict(sorted(groups.items()))


def optional_rasterio():
    try:
        return importlib.import_module("rasterio")
    except ModuleNotFoundError:
        return None


def normalize_band_to_uint8(band) -> Any:
    import numpy as np

    if band.dtype == np.uint8:
        return band

    band = band.astype("float32")
    finite_mask = np.isfinite(band)
    if not finite_mask.any():
        return np.zeros_like(band, dtype=np.uint8)

    valid = band[finite_mask]
    low = float(valid.min())
    high = float(valid.max())
    if math.isclose(low, high):
        output = np.zeros_like(band, dtype=np.uint8)
        output[finite_mask] = 255
        return output

    scaled = (band - low) / (high - low)
    scaled = np.clip(scaled * 255.0, 0.0, 255.0)
    scaled[~finite_mask] = 0.0
    return scaled.astype(np.uint8)


def load_tile_image(tile_path: str | Path) -> tuple[Image.Image, dict[str, Any]]:
    path = repo_resolve(tile_path)
    rasterio = optional_rasterio()

    if rasterio is not None:
        with rasterio.open(path) as dataset:
            band_count = max(int(dataset.count), 1)
            indexes = [min(index, band_count) for index in (1, 2, 3)]
            data = dataset.read(indexes)
            import numpy as np

            if data.ndim == 2:
                data = np.stack([data, data, data], axis=0)
            elif data.shape[0] == 1:
                data = np.repeat(data, 3, axis=0)
            elif data.shape[0] == 2:
                data = np.concatenate([data, data[:1]], axis=0)

            normalized = np.stack([normalize_band_to_uint8(data[index]) for index in range(3)], axis=-1)
            image = Image.fromarray(normalized, mode="RGB")
            metadata = {
                "loader": "rasterio",
                "dtype": str(data.dtype),
                "band_count": band_count,
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
        metadata = {
            "loader": "pillow",
            "mode": image.mode,
            "width": rgb.width,
            "height": rgb.height,
        }
        return rgb.copy(), metadata


def render_tile_png(tile: TileInfo, output_path: str | Path) -> dict[str, Any]:
    output = repo_resolve(output_path)
    ensure_parent(output)
    image, metadata = load_tile_image(tile.path)
    image.save(output, format="PNG")
    return {
        "path": repo_relative(output),
        "width": image.width,
        "height": image.height,
        "tile_row": tile.row,
        "tile_col": tile.col,
        "tile_path": str(tile.path),
        "loader": metadata.get("loader"),
        "rendered_at": utcnow_iso(),
    }


def load_manifest(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    manifest = load_json(manifest_path, default={})
    if not isinstance(manifest, dict):
        manifest = {}
    manifest.setdefault("schema_version", 1)
    manifest.setdefault("updated_at", utcnow_iso())
    manifest.setdefault("source_dir", str(DEFAULT_SOURCE_DIR))
    manifest.setdefault("images", {})
    return manifest


def save_manifest(manifest_path: str | Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utcnow_iso()
    write_json(manifest_path, manifest)


def image_key_for_filename(filename: str) -> str:
    return Path(filename).stem


def build_tile_image_key(date_value: str, tile_row: int, tile_col: int) -> str:
    return image_key_for_filename(f"{date_value}_tile_{tile_row}_{tile_col}.png")


def manifest_fingerprint(entry: dict[str, Any]) -> str:
    payload = canonical_json(entry).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def build_manifest_entry(
    *,
    date_value: str,
    tile: TileInfo,
    render_info: dict[str, Any],
    model_info: dict[str, Any],
    merge_info: dict[str, Any],
) -> dict[str, Any]:
    image_filename = f"{date_value}_tile_{tile.row}_{tile.col}.png"
    source_hash = source_content_hash(tile.path)
    render_profile = dict(TRUE_COLOR_RENDER_PROFILE)
    image_key = image_key_for_identity_payload(
        {
            "source_content_hash": source_hash,
            "render_profile": render_profile,
        }
    )
    return {
        "image_key": image_key,
        "image_filename": image_filename,
        "source_content_hash": source_hash,
        "render_profile": render_profile,
        "source_kind": "tile",
        "date": date_value,
        "tile_row": tile.row,
        "tile_col": tile.col,
        "width": int(render_info["width"]),
        "height": int(render_info["height"]),
        "cache_path": render_info["path"],
        "source_tiles": [
            {
                "path": str(tile.path),
                "row": tile.row,
                "col": tile.col,
                "mtime": tile.mtime,
                "size_bytes": tile.size_bytes,
                "source_content_hash": source_hash,
            }
        ],
        "render": {
            "loader": render_info.get("loader"),
            "rendered_at": render_info.get("rendered_at"),
            "output_format": "png",
        },
        "model_weights": dict(model_info.get("weights", {})),
        "inference_settings": {
            "confidence_threshold": model_info.get("confidence_threshold"),
        },
        "merge_settings": dict(merge_info),
        "updated_at": utcnow_iso(),
    }


def update_tile_manifest_entry(
    manifest_path: str | Path,
    date_value: str,
    tile: TileInfo,
    render_info: dict[str, Any],
    model_info: dict[str, Any],
    merge_info: dict[str, Any],
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    entry = build_manifest_entry(
        date_value=date_value,
        tile=tile,
        render_info=render_info,
        model_info=model_info,
        merge_info=merge_info,
    )
    manifest["images"][entry["image_key"]] = entry
    save_manifest(manifest_path, manifest)
    return entry


def sorted_manifest_entries(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    entries = list(manifest.get("images", {}).values())
    entries.sort(key=lambda item: (item.get("date") or "", int(item.get("tile_row") or 0), int(item.get("tile_col") or 0), item["image_key"]))
    return entries


def get_manifest_entry(image_key: str, manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    entry = manifest.get("images", {}).get(image_key)
    if entry is None:
        raise KeyError(f"No manifest entry found for {image_key}")
    return entry


def ensure_cache_for_image_key(image_key: str, manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> Path:
    entry = get_manifest_entry(image_key, manifest_path)
    cache_path = repo_resolve(entry["cache_path"])
    if cache_path.exists():
        return cache_path
    if entry.get("source_kind") != "tile":
        raise FileNotFoundError(f"Cache missing for {image_key} and source_kind={entry.get('source_kind')} is not rebuildable here")
    source_tiles = entry.get("source_tiles") or []
    if len(source_tiles) != 1:
        raise FileNotFoundError(f"Expected exactly one source tile for {image_key}")
    source = source_tiles[0]
    tile = parse_tile_name(source["path"])
    render_tile_png(tile, cache_path)
    return cache_path


def review_file_path(image_key: str, reviewer: str, reviews_dir: str | Path = DEFAULT_REVIEWS_DIR) -> Path:
    for value in [str(image_key), str(reviewer)]:
        if not value or value in {".", ".."} or any(c in value for c in "/\\:"):
            raise ValueError("Review identifiers must be single path segments")
    return contained(repo_resolve(Path(reviews_dir) / reviewer / f"{image_key}.json"), WORKSPACE_DIR)


def detection_source_models(merged_detection: dict[str, Any]) -> list[str]:
    models = {source_id.split(":", 1)[0] for source_id in merged_detection.get("source_raw_ids", [])}
    return sorted(models)


def build_review_stub(entry: dict[str, Any], reviewer: str, merged_detections: list[dict[str, Any]]) -> dict[str, Any]:
    fingerprint = manifest_fingerprint(entry)
    detections = []
    for index, merged in enumerate(merged_detections, start=1):
        detections.append(
            {
                "detection_id": f"auto-{index:04d}",
                "source_type": "auto",
                "parent_detection_id": None,
                "class_id": int(merged["class_id"]),
                "class_name": str(merged["class_name"]),
                "confidence": None if merged.get("confidence") is None else float(merged["confidence"]),
                "x_center": float(merged["x_center"]),
                "y_center": float(merged["y_center"]),
                "width": float(merged["width"]),
                "height": float(merged["height"]),
                "review_label": None,
                "reviewed": False,
                "reviewed_at": None,
                "source_models": detection_source_models(merged),
            }
        )
    return {
        "schema_version": 1,
        "image_key": entry["image_key"],
        "image_filename": entry["image_filename"],
        "original_filename": entry.get("original_filename"),
        "display_filename": entry.get("display_filename") or entry.get("original_filename") or entry["image_filename"],
        "reviewer": reviewer,
        "preprocess_fingerprint": fingerprint,
        "date": entry.get("date"),
        "tile_row": entry.get("tile_row"),
        "tile_col": entry.get("tile_col"),
        "width": int(entry["width"]),
        "height": int(entry["height"]),
        "cache_path": entry["cache_path"],
        "review_status": "overview_1",
        "detail_cursor": 0,
        "updated_at": utcnow_iso(),
        "detections": detections,
    }


def write_review_stub(
    *,
    review_doc: dict[str, Any],
    reviewer: str,
    reviews_dir: str | Path = DEFAULT_REVIEWS_DIR,
    overwrite: bool = False,
) -> Path:
    target = review_file_path(review_doc["image_key"], reviewer, reviews_dir)
    if target.exists() and not overwrite:
        return target
    write_json(target, review_doc)
    return target


def load_review_document(review_path: str | Path) -> dict[str, Any]:
    review = load_json(review_path, default={})
    if not isinstance(review, dict):
        raise ValueError(f"Review file is not a JSON object: {review_path}")
    review.setdefault("schema_version", 1)
    review.setdefault("review_status", "overview_1")
    review.setdefault("detail_cursor", 0)
    review.setdefault("detections", [])
    review.setdefault("updated_at", utcnow_iso())
    return review


def save_review_document(review: dict[str, Any], review_path: str | Path) -> None:
    review_path = contained(repo_resolve(review_path), WORKSPACE_DIR)
    review["updated_at"] = utcnow_iso()
    payload = {key: value for key, value in review.items() if not str(key).startswith("_")}
    write_json(review_path, payload)


def list_review_documents(reviewer: str, reviews_dir: str | Path = DEFAULT_REVIEWS_DIR) -> list[dict[str, Any]]:
    root = repo_resolve(Path(reviews_dir) / reviewer)
    if root.name.startswith("_"):
        return []
    if not root.exists():
        return []
    reviews = []
    for path in sorted(root.glob("*.json")):
        review = load_review_document(path)
        review["_path"] = path
        reviews.append(review)
    reviews.sort(key=lambda item: (item.get("date") or "", int(item.get("tile_row") or 0), int(item.get("tile_col") or 0), item["image_key"]))
    return reviews


def collect_review_documents(reviews_dir: str | Path = DEFAULT_REVIEWS_DIR) -> list[dict[str, Any]]:
    root = repo_resolve(reviews_dir)
    if not root.exists():
        return []
    reviews = []
    seen_by_key: dict[str, Path] = {}
    for path in sorted(root.glob("**/*.json")):
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            relative_parts = path.parts
        if any(part.startswith("_") for part in relative_parts[:-1]):
            continue
        review = load_review_document(path)
        image_key = str(review.get("image_key") or "")
        if not image_key:
            raise ValueError(f"Review file is missing image_key: {path}")
        previous = seen_by_key.get(image_key)
        if previous is not None:
            raise ValueError(f"Duplicate review file for image_key={image_key}: {previous} and {path}")
        seen_by_key[image_key] = path
        review["_path"] = path
        reviews.append(review)
    reviews.sort(key=lambda item: (item.get("date") or "", int(item.get("tile_row") or 0), int(item.get("tile_col") or 0), item["image_key"]))
    return reviews


def clamp_norm(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def active_detection(detection: dict[str, Any]) -> bool:
    return detection.get("review_label") != "neither"


def active_detections(review: dict[str, Any]) -> list[dict[str, Any]]:
    return [detection for detection in review.get("detections", []) if active_detection(detection)]


def next_detection_id(review: dict[str, Any], prefix: str) -> str:
    highest = 0
    for detection in review.get("detections", []):
        detection_id = str(detection.get("detection_id") or "")
        if not detection_id.startswith(f"{prefix}-"):
            continue
        try:
            highest = max(highest, int(detection_id.split("-", 1)[1]))
        except ValueError:
            continue
    return f"{prefix}-{highest + 1:04d}"


def detection_payload(detection: dict[str, Any], image_width: int, image_height: int, image_key: str) -> dict[str, Any]:
    x_center = float(detection["x_center"])
    y_center = float(detection["y_center"])
    width = float(detection["width"])
    height = float(detection["height"])
    return {
        "detection_id": str(detection["detection_id"]),
        "image_id": image_key,
        "source_type": detection["source_type"],
        "parent_detection_id": detection.get("parent_detection_id"),
        "class_id": int(detection["class_id"]),
        "class_name": detection["class_name"],
        "confidence": None if detection.get("confidence") is None else float(detection["confidence"]),
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
        "x1_px": round((x_center - width / 2.0) * image_width, 3),
        "y1_px": round((1.0 - y_center - height / 2.0) * image_height, 3),
        "x2_px": round((x_center + width / 2.0) * image_width, 3),
        "y2_px": round((1.0 - y_center + height / 2.0) * image_height, 3),
        "review_label": detection.get("review_label"),
        "is_active": active_detection(detection),
        "reviewed": bool(detection.get("reviewed")),
        "reviewed_at": detection.get("reviewed_at"),
        "source_models": list(detection.get("source_models") or []),
        "updated_at": detection.get("updated_at", detection.get("reviewed_at")),
    }


def first_unreviewed_active_index(review: dict[str, Any]) -> int | None:
    for index, detection in enumerate(review.get("detections", [])):
        if active_detection(detection) and not bool(detection.get("reviewed")):
            return index
    return None


def next_active_index(review: dict[str, Any], start: int) -> int | None:
    for index in range(start + 1, len(review.get("detections", []))):
        detection = review["detections"][index]
        if active_detection(detection):
            return index
    return None


def prev_active_index(review: dict[str, Any], start: int) -> int | None:
    for index in range(start - 1, -1, -1):
        detection = review["detections"][index]
        if active_detection(detection):
            return index
    return None


def sync_review_status(review: dict[str, Any]) -> tuple[str, int | None]:
    current_status = review.get("review_status", "overview_1")
    pending_index = first_unreviewed_active_index(review)
    detection_count = len(review.get("detections", []))

    if current_status == "completed":
        if pending_index is None:
            review["detail_cursor"] = None
            return "completed", None
        review["review_status"] = "detail"
        review["detail_cursor"] = pending_index
        return "detail", pending_index
    if pending_index is None:
        if detection_count == 0:
            review["review_status"] = "overview_1"
            review["detail_cursor"] = None
            return "overview_1", None
        review["review_status"] = "completed"
        review["detail_cursor"] = None
        return "completed", None
    if current_status == "detail":
        cursor = review.get("detail_cursor")
        if not isinstance(cursor, int) or cursor < 0 or cursor >= len(review.get("detections", [])):
            review["detail_cursor"] = pending_index
            return "detail", pending_index
        current_detection = review["detections"][cursor]
        if not active_detection(current_detection):
            review["detail_cursor"] = pending_index
            return "detail", pending_index
        return "detail", cursor
    if current_status == "overview_2" and pending_index is not None:
        review["review_status"] = "detail"
        review["detail_cursor"] = pending_index
        return "detail", pending_index
    if current_status not in {"overview_1", "overview_2"}:
        review["review_status"] = "overview_1"
    return review["review_status"], review.get("detail_cursor")


def set_review_status(review: dict[str, Any], review_status: str, detail_cursor: int | None) -> None:
    review["review_status"] = review_status
    review["detail_cursor"] = detail_cursor


def start_detail_mode(review: dict[str, Any]) -> tuple[str, int | None]:
    pending_index = first_unreviewed_active_index(review)
    if pending_index is None:
        set_review_status(review, "completed", None)
        return "completed", None
    set_review_status(review, "detail", pending_index)
    return "detail", pending_index


def advance_detail_cursor(review: dict[str, Any]) -> tuple[str, int | None]:
    pending_index = first_unreviewed_active_index(review)
    if pending_index is None:
        set_review_status(review, "completed", None)
        return "completed", None
    set_review_status(review, "detail", pending_index)
    return "detail", pending_index


def navigate_detail_cursor(review: dict[str, Any], direction: str) -> tuple[str, int | None]:
    if review.get("review_status") != "detail":
        return review.get("review_status", "overview_1"), review.get("detail_cursor")
    detections = review.get("detections", [])
    if not detections:
        set_review_status(review, "completed", None)
        return "completed", None

    current = review.get("detail_cursor")
    if not isinstance(current, int):
        current = first_unreviewed_active_index(review)
    if current is None:
        set_review_status(review, "completed", None)
        return "completed", None

    if direction == "prev":
        previous = prev_active_index(review, current)
        if previous is None:
            set_review_status(review, "overview_1", None)
            return "overview_1", None
        set_review_status(review, "detail", previous)
        return "detail", previous

    following = next_active_index(review, current)
    if following is None:
        return sync_review_status(review)
    set_review_status(review, "detail", following)
    return "detail", following


def find_detection(review: dict[str, Any], detection_id: str) -> dict[str, Any]:
    for detection in review.get("detections", []):
        if str(detection.get("detection_id")) == str(detection_id):
            return detection
    raise KeyError(f"Detection {detection_id} not found in image {review.get('image_key')}")


def create_manual_detection(
    review: dict[str, Any],
    *,
    class_id: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    parent_detection_id: str | None = None,
) -> str:
    detection_id = next_detection_id(review, "manual")
    reviewed_at = utcnow_iso()
    x_center, y_center, width, height = clamp_bbox_to_unit(x_center, y_center, width, height)
    review.setdefault("detections", []).append(
        {
            "detection_id": detection_id,
            "source_type": "manual",
            "parent_detection_id": parent_detection_id,
            "class_id": int(class_id),
            "class_name": CLASS_ID_TO_NAME[int(class_id)],
            "confidence": None,
            "x_center": clamp_norm(x_center),
            "y_center": clamp_norm(y_center),
            "width": clamp_norm(width),
            "height": clamp_norm(height),
            "review_label": CLASS_ID_TO_REVIEW_LABEL[int(class_id)],
            "reviewed": True,
            "reviewed_at": reviewed_at,
            "source_models": [],
        }
    )
    return detection_id


def update_detection_geometry(
    review: dict[str, Any],
    detection_id: str,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    class_id: int | None = None,
    *,
    user_edited: bool = False,
) -> None:
    """Save geometry without treating display/rounding saves as human decisions.

    Callers must identify a user-made edit explicitly. Existing decisions, including
    their timestamps and rejected labels, remain intact for automatic saves.
    """
    detection = find_detection(review, detection_id)
    if not user_edited:
        return
    x_center, y_center, width, height = clamp_bbox_to_unit(x_center, y_center, width, height)
    detection["x_center"] = clamp_norm(x_center)
    detection["y_center"] = clamp_norm(y_center)
    detection["width"] = clamp_norm(width)
    detection["height"] = clamp_norm(height)
    if class_id is not None:
        class_changed = int(class_id) != int(detection["class_id"])
        detection["class_id"] = int(class_id)
        detection["class_name"] = CLASS_ID_TO_NAME[int(class_id)]
        if class_changed or detection.get("review_label") not in {*EXPORTABLE_REVIEW_LABELS, "neither"}:
            detection["review_label"] = CLASS_ID_TO_REVIEW_LABEL[int(class_id)]
    elif detection.get("review_label") not in {*EXPORTABLE_REVIEW_LABELS, "neither"}:
        detection["review_label"] = CLASS_ID_TO_REVIEW_LABEL[int(detection["class_id"])]
    detection["reviewed"] = True
    detection["reviewed_at"] = utcnow_iso()


def label_detection(
    review: dict[str, Any],
    detection_id: str,
    review_label: str,
    class_id: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> tuple[str, int | None]:
    detection = find_detection(review, detection_id)
    reviewed_at = utcnow_iso()
    x_center, y_center, width, height = clamp_bbox_to_unit(x_center, y_center, width, height)
    detection["class_id"] = int(class_id)
    detection["class_name"] = CLASS_ID_TO_NAME[int(class_id)]
    detection["review_label"] = review_label
    detection["reviewed"] = True
    detection["reviewed_at"] = reviewed_at
    detection["x_center"] = clamp_norm(x_center)
    detection["y_center"] = clamp_norm(y_center)
    detection["width"] = clamp_norm(width)
    detection["height"] = clamp_norm(height)
    return advance_detail_cursor(review)


def set_detection_review_label(
    review: dict[str, Any],
    detection_id: str,
    review_label: str,
    class_id: int | None,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> None:
    detection = find_detection(review, detection_id)
    reviewed_at = utcnow_iso()
    x_center, y_center, width, height = clamp_bbox_to_unit(x_center, y_center, width, height)
    if review_label != "neither":
        if class_id is None:
            class_id = REVIEW_LABEL_TO_CLASS[review_label]
        detection["class_id"] = int(class_id)
        detection["class_name"] = CLASS_ID_TO_NAME[int(class_id)]
    detection["review_label"] = review_label
    detection["reviewed"] = True
    detection["reviewed_at"] = reviewed_at
    detection["x_center"] = clamp_norm(x_center)
    detection["y_center"] = clamp_norm(y_center)
    detection["width"] = clamp_norm(width)
    detection["height"] = clamp_norm(height)


def remove_detection(review: dict[str, Any], detection_id: str) -> tuple[str, int | None]:
    detections = review.get("detections", [])
    index = next((idx for idx, detection in enumerate(detections) if str(detection.get("detection_id")) == str(detection_id)), None)
    if index is None:
        raise KeyError(f"Detection {detection_id} not found in image {review.get('image_key')}")
    detections.pop(index)
    cursor = review.get("detail_cursor")
    if isinstance(cursor, int):
        if cursor > index:
            review["detail_cursor"] = cursor - 1
        elif cursor >= len(detections):
            review["detail_cursor"] = len(detections) - 1 if detections else None
    if review.get("review_status") != "detail":
        return review.get("review_status", "overview_1"), review.get("detail_cursor")
    return advance_detail_cursor(review)


def mark_detection_neither(review: dict[str, Any], detection_id: str) -> tuple[str, int | None]:
    detection = find_detection(review, detection_id)
    timestamp = utcnow_iso()
    detection["review_label"] = "neither"
    detection["reviewed"] = True
    detection["reviewed_at"] = timestamp
    if review.get("review_status") != "detail":
        return review.get("review_status", "overview_1"), review.get("detail_cursor")
    return advance_detail_cursor(review)


def delete_detection(review: dict[str, Any], detection_id: str) -> tuple[str, int | None]:
    return remove_detection(review, detection_id)


def mark_all_active_detections_reviewed(review: dict[str, Any]) -> int:
    timestamp = utcnow_iso()
    changed = 0
    if not review.get("detections"):
        set_review_status(review, "completed", None)
        return changed
    for detection in review.get("detections", []):
        if not active_detection(detection):
            continue
        if not detection.get("reviewed"):
            changed += 1
        detection["reviewed"] = True
        detection["reviewed_at"] = timestamp
        if detection.get("review_label") not in EXPORTABLE_REVIEW_LABELS:
            detection["review_label"] = CLASS_ID_TO_REVIEW_LABEL[int(detection["class_id"])]
    sync_review_status(review)
    return changed


def complete_image_review(review: dict[str, Any]) -> None:
    pending_count = sum(1 for detection in review.get("detections", []) if active_detection(detection) and not bool(detection.get("reviewed")))
    if pending_count:
        raise ValueError("Cannot complete image while active detections remain unlabeled")
    set_review_status(review, "completed", None)


def review_counts(review: dict[str, Any]) -> dict[str, int]:
    detections = review.get("detections", [])
    active = [detection for detection in detections if active_detection(detection)]
    reviewed_active = [detection for detection in active if bool(detection.get("reviewed"))]
    neither = [detection for detection in detections if detection.get("review_label") == "neither"]
    return {
        "total_detections": len(detections),
        "active_detections": len(active),
        "reviewed_active_detections": len(reviewed_active),
        "pending_active_detections": len(active) - len(reviewed_active),
        "neither_detections": len(neither),
    }


def file_fingerprint(path: str | Path) -> str:
    target = repo_resolve(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def crop_pixel_bounds(detection: dict[str, Any], image_width: int, image_height: int, padding_px: int) -> dict[str, int]:
    bbox_x1 = int(round((float(detection["x_center"]) - float(detection["width"]) / 2.0) * image_width))
    bbox_y1 = int(round((float(detection["y_center"]) - float(detection["height"]) / 2.0) * image_height))
    bbox_x2 = int(round((float(detection["x_center"]) + float(detection["width"]) / 2.0) * image_width))
    bbox_y2 = int(round((float(detection["y_center"]) + float(detection["height"]) / 2.0) * image_height))
    x1 = max(0, min(image_width, bbox_x1 - int(padding_px)))
    y1 = max(0, min(image_height, bbox_y1 - int(padding_px)))
    x2 = max(0, min(image_width, bbox_x2 + int(padding_px)))
    y2 = max(0, min(image_height, bbox_y2 + int(padding_px)))
    if x2 <= x1:
        x2 = min(image_width, x1 + 1)
    if y2 <= y1:
        y2 = min(image_height, y1 + 1)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "padding_px": int(padding_px)}


def crop_review_metadata(detection: dict[str, Any]) -> dict[str, Any]:
    review_label = detection.get("review_label")
    if review_label in REVIEW_LABEL_TO_CLASS:
        class_id = REVIEW_LABEL_TO_CLASS[str(review_label)]
        class_name = CLASS_ID_TO_NAME[class_id]
    else:
        class_id = None
        class_name = None
    return {
        "review_label": review_label,
        "class_id": class_id,
        "class_name": class_name,
        "source_type": detection.get("source_type"),
        "reviewed": bool(detection.get("reviewed")),
    }


def generate_review_crop(
    review: dict[str, Any],
    detection_id: str,
    *,
    review_path: str | Path,
    crop_root: str | Path = WORKSPACE_DIR / "cache" / "review_crops",
    padding_px: int = 48,
    generated_at: str | None = None,
) -> dict[str, Path]:
    detection = find_detection(review, detection_id)
    image_key = str(review["image_key"])
    source_path = repo_resolve(str(review["cache_path"]))
    crop_dir = repo_resolve(crop_root) / image_key
    crop_path = crop_dir / f"{detection_id}.png"
    sidecar_path = crop_dir / f"{detection_id}.json"
    crop_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as source:
        image_width, image_height = source.size
        crop_pixels = crop_pixel_bounds(detection, image_width, image_height, padding_px)
        crop = source.crop((crop_pixels["x1"], crop_pixels["y1"], crop_pixels["x2"], crop_pixels["y2"]))
        crop.save(crop_path, format="PNG")

    bbox = {
        "x_center": float(detection["x_center"]),
        "y_center": float(detection["y_center"]),
        "width": float(detection["width"]),
        "height": float(detection["height"]),
    }
    bbox_fingerprint = f"sha256:{hashlib.sha256(canonical_json(bbox).encode('utf-8')).hexdigest()}"
    crop_recipe = {"bbox": bbox, "crop_pixels": crop_pixels, "padding_px": int(padding_px)}
    crop_recipe_fingerprint = f"sha256:{hashlib.sha256(canonical_json(crop_recipe).encode('utf-8')).hexdigest()}"
    sidecar = {
        "schema_version": 1,
        "image_key": image_key,
        "detection_id": str(detection_id),
        "review_file": str(repo_resolve(review_path)),
        "crop_path": str(crop_path),
        "source_cache_path": str(source_path),
        "source_image_width": int(image_width),
        "source_image_height": int(image_height),
        "bbox": bbox,
        "crop_pixels": crop_pixels,
        "review": crop_review_metadata(detection),
        "fingerprints": {
            "source_image": file_fingerprint(source_path),
            "bbox_geometry": bbox_fingerprint,
            "crop_recipe": crop_recipe_fingerprint,
        },
        "generated_at": generated_at or utcnow_iso(),
    }
    write_json(sidecar_path, sidecar)
    return {"crop_path": crop_path, "sidecar_path": sidecar_path}


def reviewer_for_review_path(review_path: str | Path, reviews_dir: str | Path = DEFAULT_REVIEWS_DIR) -> str | None:
    target = repo_resolve(review_path)
    root = repo_resolve(reviews_dir)
    try:
        relative = target.relative_to(root)
    except ValueError:
        return None
    return relative.parts[0] if len(relative.parts) > 1 else None


def review_crop_paths(image_key: str, detection_id: str, crop_root: str | Path = WORKSPACE_DIR / "cache" / "review_crops") -> tuple[Path, Path]:
    crop_dir = repo_resolve(crop_root) / str(image_key)
    return crop_dir / f"{detection_id}.png", crop_dir / f"{detection_id}.json"


def bbox_matches_sidecar(sidecar_bbox: Any, detection: dict[str, Any]) -> bool:
    if not isinstance(sidecar_bbox, dict):
        return False
    for key in ("x_center", "y_center", "width", "height"):
        try:
            if abs(float(sidecar_bbox.get(key)) - float(detection[key])) > 1e-9:
                return False
        except (TypeError, ValueError, KeyError):
            return False
    return True


def crop_sidecar_warnings(crop_path: Path, sidecar_path: Path, detection: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not crop_path.exists():
        warnings.append("missing_crop")
    if not sidecar_path.exists():
        warnings.append("missing_crop_metadata")
        return warnings

    try:
        sidecar = load_json(sidecar_path, default={})
    except json.JSONDecodeError:
        warnings.append("invalid_crop_metadata")
        return warnings

    if not bbox_matches_sidecar(sidecar.get("bbox"), detection):
        warnings.append("stale_crop_geometry")
    if sidecar.get("review") != crop_review_metadata(detection):
        warnings.append("stale_crop_review")
    source_cache_path = sidecar.get("source_cache_path")
    if source_cache_path and not repo_resolve(str(source_cache_path)).exists():
        warnings.append("missing_source_cache")
    return warnings


def review_file_groups(reviews_dir: str | Path = DEFAULT_REVIEWS_DIR) -> dict[str, list[tuple[dict[str, Any], Path]]]:
    root = repo_resolve(reviews_dir)
    groups: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    if not root.exists():
        return groups
    for path in sorted(root.glob("**/*.json")):
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            relative_parts = path.parts
        if any(part.startswith("_") for part in relative_parts[:-1]):
            continue
        review = load_review_document(path)
        image_key = str(review.get("image_key") or path.stem)
        review.setdefault("image_key", image_key)
        groups.setdefault(image_key, []).append((review, path))
    return groups


def overview_class_name(review_label: Any) -> str | None:
    if review_label in REVIEW_LABEL_TO_CLASS:
        return CLASS_ID_TO_NAME[REVIEW_LABEL_TO_CLASS[str(review_label)]]
    if review_label == "neither":
        return "neither"
    return None


def review_overview_tile(
    review: dict[str, Any],
    review_path: Path,
    detection: dict[str, Any],
    *,
    reviews_dir: str | Path,
    crop_root: str | Path,
    padding_px: int,
    generate_crops: bool,
) -> tuple[dict[str, Any], bool]:
    image_key = str(review["image_key"])
    detection_id = str(detection["detection_id"])
    crop_path, sidecar_path = review_crop_paths(image_key, detection_id, crop_root)
    warnings = crop_sidecar_warnings(crop_path, sidecar_path, detection)
    generated = False

    if generate_crops and warnings:
        try:
            generate_review_crop(review, detection_id, review_path=review_path, crop_root=crop_root, padding_px=padding_px)
            generated = True
            warnings = crop_sidecar_warnings(crop_path, sidecar_path, detection)
        except Exception as exc:
            warnings = [warning for warning in warnings if warning != "missing_crop"]
            warnings.append(f"crop_error:{exc}")

    review_label = detection.get("review_label")
    class_id = REVIEW_LABEL_TO_CLASS.get(str(review_label)) if review_label in REVIEW_LABEL_TO_CLASS else None
    area = max(0.0, float(detection.get("width") or 0.0)) * max(0.0, float(detection.get("height") or 0.0))
    tile = {
        "tile_id": f"{image_key}:{detection_id}",
        "image_key": image_key,
        "image_filename": review.get("image_filename") or f"{image_key}.png",
        "date": review.get("date"),
        "tile_row": review.get("tile_row"),
        "tile_col": review.get("tile_col"),
        "detection_id": detection_id,
        "reviewer": review.get("reviewer") or reviewer_for_review_path(review_path, reviews_dir),
        "review_status": review.get("review_status"),
        "review_file": str(review_path),
        "crop_path": str(crop_path) if crop_path.exists() else None,
        "crop_url": cache_url_for(crop_path) if crop_path.exists() else None,
        "review_label": review_label,
        "class_id": class_id,
        "class_name": overview_class_name(review_label),
        "source_type": detection.get("source_type"),
        "confidence": detection.get("confidence"),
        "source_models": list(detection.get("source_models") or []),
        "reviewed": bool(detection.get("reviewed")),
        "reviewed_at": detection.get("reviewed_at"),
        "bbox": {
            "x_center": float(detection.get("x_center") or 0.0),
            "y_center": float(detection.get("y_center") or 0.0),
            "width": float(detection.get("width") or 0.0),
            "height": float(detection.get("height") or 0.0),
        },
        "bbox_area": area,
        "warnings": warnings,
    }
    return tile, generated


def build_review_bbox_overview(
    reviews_dir: str | Path = DEFAULT_REVIEWS_DIR,
    *,
    crop_root: str | Path = WORKSPACE_DIR / "cache" / "review_crops",
    padding_px: int = 48,
    include_incomplete: bool = False,
    generate_crops: bool = True,
) -> dict[str, Any]:
    groups = review_file_groups(reviews_dir)
    tiles: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    generated_crop_count = 0

    for image_key, group in sorted(groups.items()):
        completed = [(review, path) for review, path in group if review.get("review_status") == "completed"]
        if len(completed) > 1:
            conflicts.append(
                {
                    "image_key": image_key,
                    "review_files": [str(path) for _, path in completed],
                    "reviewers": [review.get("reviewer") or reviewer_for_review_path(path, reviews_dir) for review, path in completed],
                    "status": "needs_reconciliation",
                }
            )
            continue
        if completed:
            review, review_path = completed[0]
        elif include_incomplete and len(group) == 1:
            review, review_path = group[0]
        else:
            continue

        for detection in review.get("detections", []):
            if not bool(detection.get("reviewed")):
                continue
            tile, generated = review_overview_tile(
                review,
                review_path,
                detection,
                reviews_dir=reviews_dir,
                crop_root=crop_root,
                padding_px=padding_px,
                generate_crops=generate_crops,
            )
            generated_crop_count += 1 if generated else 0
            tiles.append(tile)

    tiles.sort(
        key=lambda item: (
            str(item.get("date") or ""),
            int(item.get("tile_row") or 0),
            int(item.get("tile_col") or 0),
            str(item.get("image_key") or ""),
            str(item.get("detection_id") or ""),
        )
    )
    counts = {
        "review_file_count": sum(len(group) for group in groups.values()),
        "reviewed_image_count": len({tile["image_key"] for tile in tiles}),
        "reviewed_bbox_count": len(tiles),
        "reviewed_ship_bboxes": sum(1 for tile in tiles if tile.get("review_label") == "ship_non_sts"),
        "reviewed_ships_bboxes": sum(1 for tile in tiles if tile.get("review_label") == "ships_sts"),
        "neither_bboxes": sum(1 for tile in tiles if tile.get("review_label") == "neither"),
        "warning_tile_count": sum(1 for tile in tiles if tile.get("warnings")),
        "generated_crop_count": generated_crop_count,
        "conflict_image_count": len(conflicts),
    }
    return {"schema_version": 1, "counts": counts, "tiles": tiles, "conflicts": conflicts}


def queue_progress(reviews: list[dict[str, Any]]) -> dict[str, int]:
    reviewed_ship_count = 0
    reviewed_ships_count = 0
    for review in reviews:
        for detection in review.get("detections", []):
            if not detection.get("reviewed"):
                continue
            if detection.get("review_label") == "ship_non_sts":
                reviewed_ship_count += 1
            elif detection.get("review_label") == "ships_sts":
                reviewed_ships_count += 1
    return {
        "total_images": len(reviews),
        "completed_images": sum(1 for review in reviews if review.get("review_status") == "completed"),
        "reviewed_ship_bboxes": reviewed_ship_count,
        "reviewed_ships_bboxes": reviewed_ships_count,
    }


def exportable_detections(review: dict[str, Any], item: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Canonical exportable-label filter shared by the legacy export and the training pipeline.

    Reviewed positive labels are ground truth regardless of auto/manual origin. When a
    dataset-selection ``item`` is supplied and it requests ``include_labels="selected_only"``,
    only the listed ``selected_detection_ids`` are exported.
    """
    selected: set[str] | None = None
    if item is not None and item.get("include_labels") == "selected_only":
        selected_ids = item.get("selected_detection_ids")
        if not isinstance(selected_ids, list):
            raise ValueError(f"selected_only requires selected_detection_ids for image_key={item.get('image_key')}")
        selected = {str(detection_id) for detection_id in selected_ids}

    detections = []
    for detection in review.get("detections", []):
        if detection.get("review_label") not in EXPORTABLE_REVIEW_LABELS:
            continue
        if selected is not None and str(detection.get("detection_id")) not in selected:
            continue
        detections.append(detection)
    return detections


def write_yolo_label_file(label_path: str | Path, detections: list[dict[str, Any]]) -> None:
    target = repo_resolve(label_path)
    ensure_parent(target)
    lines = []
    for detection in detections:
        review_label = detection["review_label"]
        class_id = REVIEW_LABEL_TO_CLASS[review_label]
        lines.append(
            " ".join(
                [
                    str(class_id),
                    f"{float(detection['x_center']):.8f}",
                    f"{float(detection['y_center']):.8f}",
                    f"{float(detection['width']):.8f}",
                    f"{float(detection['height']):.8f}",
                ]
            )
        )
    target.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_data_yaml(dataset_dir: str | Path, *, split_layout: bool = False) -> None:
    """Canonical data.yaml writer.

    ``split_layout=False`` keeps the legacy flat single-directory export layout.
    ``split_layout=True`` writes the train/val/test layout used by dataset selections.
    """
    target = repo_resolve(Path(dataset_dir) / "data.yaml")
    ensure_parent(target)
    if split_layout:
        splits = "train: images/train\nval: images/val\ntest: images/test\n"
    else:
        splits = "train: images\nval: images\n"
    content = (
        f"path: {repo_resolve(dataset_dir)}\n"
        f"{splits}"
        "names:\n"
        "  0: ship\n"
        "  1: ships\n"
    )
    target.write_text(content)


def normalize_names_mapping(names: Any) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if isinstance(names, dict):
        iterable = names.items()
    else:
        iterable = enumerate(names or [])

    for raw_key, raw_value in iterable:
        try:
            key = int(raw_key)
        except (TypeError, ValueError):
            continue
        mapping[key] = str(raw_value)
    return mapping


def infer_class_name(class_id: int, names: dict[int, str]) -> str:
    class_name = names.get(class_id, CLASS_ID_TO_NAME.get(class_id, str(class_id)))
    return str(class_name)


def load_yolo_models(model_paths: dict[str, str | Path]) -> dict[str, Any]:
    try:
        yolo_module = importlib.import_module("ultralytics")
    except ModuleNotFoundError as exc:
        raise RuntimeError("ultralytics is required for preprocessing") from exc

    models = {}
    for model_name, model_path in model_paths.items():
        resolved = repo_resolve(model_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Model weights not found for {model_name}: {resolved}")
        models[model_name] = yolo_module.YOLO(str(resolved))
        try:
            from .detect import require_ships_class
        except ImportError:
            from detect import require_ships_class
        require_ships_class(models[model_name].names)
    return models


def run_model_predictions(image_path: str | Path, models: dict[str, Any], conf_threshold: float) -> list[dict[str, Any]]:
    image = repo_resolve(image_path)
    with Image.open(image) as handle:
        image_width, image_height = handle.size

    raw_detections: list[dict[str, Any]] = []
    for model_name, model in models.items():
        results = model.predict(source=str(image), conf=conf_threshold, verbose=False)
        if not results:
            continue
        result = results[0]
        names = normalize_names_mapping(getattr(result, "names", {}) or {})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        for index, box in enumerate(boxes, start=1):
            class_id = int(box.cls.item())
            class_name = infer_class_name(class_id, names)
            if class_name not in {"ship", "ships"}:
                continue
            xywhn = [float(value) for value in box.xywhn[0].tolist()]
            raw_detections.append(
                {
                    "temp_id": f"{model_name}:{index}",
                    "source_model": model_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(float(box.conf.item()), 6),
                    "x_center": round(xywhn[0], 8),
                    "y_center": round(xywhn[1], 8),
                    "width": round(xywhn[2], 8),
                    "height": round(xywhn[3], 8),
                    "image_width": image_width,
                    "image_height": image_height,
                }
            )
    return raw_detections


def xywh_to_xyxy(box: dict[str, Any]) -> tuple[float, float, float, float]:
    half_width = box["width"] / 2.0
    half_height = box["height"] / 2.0
    return (
        box["x_center"] - half_width,
        box["y_center"] - half_height,
        box["x_center"] + half_width,
        box["y_center"] + half_height,
    )


def box_iou(box_a: dict[str, Any], box_b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def merge_detections_class_aware(raw_detections: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    next_index = 1

    for class_id in sorted({detection["class_id"] for detection in raw_detections}):
        class_items = [item for item in raw_detections if item["class_id"] == class_id]
        class_items.sort(key=lambda item: (-item["confidence"], item["source_model"], item["temp_id"]))

        consumed: set[str] = set()
        for candidate in class_items:
            if candidate["temp_id"] in consumed:
                continue

            source_items = [candidate]
            consumed.add(candidate["temp_id"])
            for other in class_items:
                if other["temp_id"] in consumed:
                    continue
                if box_iou(candidate, other) >= iou_threshold:
                    source_items.append(other)
                    consumed.add(other["temp_id"])

            best = max(source_items, key=lambda item: item["confidence"])
            merged.append(
                {
                    "temp_id": f"merged:{next_index}",
                    "class_id": best["class_id"],
                    "class_name": best["class_name"],
                    "confidence": best["confidence"],
                    "x_center": best["x_center"],
                    "y_center": best["y_center"],
                    "width": best["width"],
                    "height": best["height"],
                    "source_raw_ids": [item["temp_id"] for item in source_items],
                }
            )
            next_index += 1

    merged.sort(key=lambda item: (-item["confidence"], item["class_id"], item["temp_id"]))
    return merged
