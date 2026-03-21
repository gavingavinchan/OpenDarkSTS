from __future__ import annotations

import importlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = Path("/home/gavin/Downloads/AIS data/Sentinel 2A/sentinel2/")
WORKSPACE_DIR = ROOT_DIR / "workspace"
DEFAULT_CACHE_DIR = WORKSPACE_DIR / "cache" / "sentinel_tiles"
DEFAULT_DATABASE_PATH = WORKSPACE_DIR / "databases" / "visual_detections.sqlite"
DEFAULT_MANIFEST_PATH = WORKSPACE_DIR / "manifests" / "visual_preprocess_manifest.json"
DEFAULT_EXPORT_DIR = WORKSPACE_DIR / "datasets" / "sentinel_reviewed"

TILE_NAME_SUFFIXES = {".tif", ".tiff"}
TILE_NAME_TEMPLATE = "{date}_tile_{row}_{col}.tiff"
CLASS_ID_TO_NAME = {0: "ship", 1: "ships"}
CLASS_NAME_TO_ID = {value: key for key, value in CLASS_ID_TO_NAME.items()}
REVIEW_LABEL_TO_CLASS = {"ship_non_sts": 0, "ships_sts": 1}


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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def require_lastrowid(cursor: sqlite3.Cursor) -> int:
    if cursor.lastrowid is None:
        raise RuntimeError("Expected SQLite insert to produce lastrowid")
    return int(cursor.lastrowid)


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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_json(path: str | Path, default: Any | None = None) -> Any:
    target = repo_resolve(path)
    if not target.exists():
        return {} if default is None else default
    return json.loads(target.read_text())


def write_json(path: str | Path, payload: Any) -> None:
    target = repo_resolve(path)
    ensure_parent(target)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def get_db(db_path: str | Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    target = repo_resolve(db_path)
    ensure_parent(target)
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def initialize_database(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            tile_row INTEGER NOT NULL DEFAULT 0,
            tile_col INTEGER NOT NULL DEFAULT 0,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            tile_count INTEGER NOT NULL DEFAULT 1,
            cache_path TEXT,
            review_status TEXT NOT NULL DEFAULT 'pending',
            detail_cursor INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(date, tile_row, tile_col)
        );

        CREATE TABLE IF NOT EXISTS image_sources (
            image_source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            tile_path TEXT NOT NULL,
            tile_row INTEGER NOT NULL,
            tile_col INTEGER NOT NULL,
            tile_mtime REAL NOT NULL,
            tile_size_bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE,
            UNIQUE(image_id, tile_path)
        );

        CREATE TABLE IF NOT EXISTS detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            source_type TEXT NOT NULL,
            source_model TEXT,
            parent_detection_id INTEGER,
            class_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL,
            x_center REAL NOT NULL,
            y_center REAL NOT NULL,
            width REAL NOT NULL,
            height REAL NOT NULL,
            review_label TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            reviewed INTEGER NOT NULL DEFAULT 0,
            review_order INTEGER,
            reviewed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE,
            FOREIGN KEY (parent_detection_id) REFERENCES detections(detection_id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS detection_sources (
            detection_source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            raw_detection_id INTEGER NOT NULL,
            source_model TEXT NOT NULL,
            FOREIGN KEY (detection_id) REFERENCES detections(detection_id) ON DELETE CASCADE,
            FOREIGN KEY (raw_detection_id) REFERENCES detections(detection_id) ON DELETE CASCADE,
            UNIQUE(detection_id, raw_detection_id)
        );

        CREATE INDEX IF NOT EXISTS idx_images_review_status ON images(review_status);
        CREATE INDEX IF NOT EXISTS idx_images_date_tile ON images(date, tile_row, tile_col);
        CREATE INDEX IF NOT EXISTS idx_detections_image_source ON detections(image_id, source_type);
        CREATE INDEX IF NOT EXISTS idx_detections_image_queue ON detections(image_id, is_active, review_order);
        """
    )
    conn.commit()


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


def build_stitch_recipe(tiles: list[TileInfo]) -> dict[str, Any]:
    if not tiles:
        raise ValueError("Cannot build stitch recipe without tiles")

    rows = sorted({tile.row for tile in tiles})
    cols = sorted({tile.col for tile in tiles})

    tile_sizes: dict[tuple[int, int], tuple[int, int]] = {}
    tile_metadata: dict[str, dict[str, Any]] = {}
    for tile in tiles:
        with Image.open(tile.path) as image:
            tile_sizes[(tile.row, tile.col)] = image.size
        tile_metadata[str(tile.path)] = {
            "row": tile.row,
            "col": tile.col,
            "mtime": tile.mtime,
            "size_bytes": tile.size_bytes,
        }

    column_widths = {
        col: max(tile_sizes[(tile.row, tile.col)][0] for tile in tiles if tile.col == col)
        for col in cols
    }
    row_heights = {
        row: max(tile_sizes[(tile.row, tile.col)][1] for tile in tiles if tile.row == row)
        for row in rows
    }

    x_offsets: dict[int, int] = {}
    running_x = 0
    for col in cols:
        x_offsets[col] = running_x
        running_x += column_widths[col]

    y_offsets: dict[int, int] = {}
    running_y = 0
    for row in rows:
        y_offsets[row] = running_y
        running_y += row_heights[row]

    placements = []
    for tile in tiles:
        width, height = tile_sizes[(tile.row, tile.col)]
        placements.append(
            {
                "path": str(tile.path),
                "row": tile.row,
                "col": tile.col,
                "x": x_offsets[tile.col],
                "y": y_offsets[tile.row],
                "width": width,
                "height": height,
                "mtime": tile.mtime,
                "size_bytes": tile.size_bytes,
            }
        )

    recipe = {
        "date": tiles[0].date,
        "tile_count": len(tiles),
        "rows": rows,
        "cols": cols,
        "width": running_x,
        "height": running_y,
        "placements": placements,
        "tile_metadata": tile_metadata,
    }
    return recipe


def render_stitched_png(recipe: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    output = repo_resolve(output_path)
    ensure_parent(output)

    canvas = Image.new("RGB", (int(recipe["width"]), int(recipe["height"])))
    tile_render_metadata = []
    for placement in recipe["placements"]:
        image, metadata = load_tile_image(placement["path"])
        canvas.paste(image, (int(placement["x"]), int(placement["y"])))
        tile_render_metadata.append(
            {
                "path": placement["path"],
                "row": placement["row"],
                "col": placement["col"],
                "x": placement["x"],
                "y": placement["y"],
                "width": placement["width"],
                "height": placement["height"],
                "loader": metadata.get("loader"),
            }
        )

    canvas.save(output, format="PNG")
    return {
        "path": repo_relative(output),
        "width": canvas.width,
        "height": canvas.height,
        "tile_render_metadata": tile_render_metadata,
        "rendered_at": utcnow_iso(),
        "render_settings": {
            "output_format": "png",
            "color_mode": "RGB",
            "conversion": "preserve_uint8_or_minmax_per_band",
        },
    }


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
    manifest.setdefault("version", 1)
    manifest.setdefault("updated_at", utcnow_iso())
    manifest.setdefault("source_dir", str(DEFAULT_SOURCE_DIR))
    manifest.setdefault("images", {})
    return manifest


def save_manifest(manifest_path: str | Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utcnow_iso()
    write_json(manifest_path, manifest)


def update_manifest_entry(
    manifest_path: str | Path,
    recipe: dict[str, Any],
    render_info: dict[str, Any],
    model_info: dict[str, Any],
    merge_info: dict[str, Any],
) -> None:
    manifest = load_manifest(manifest_path)
    manifest["images"][recipe["date"]] = {
        "date": recipe["date"],
        "tile_count": recipe["tile_count"],
        "rows": recipe["rows"],
        "cols": recipe["cols"],
        "width": recipe["width"],
        "height": recipe["height"],
        "placements": recipe["placements"],
        "cache": render_info,
        "models": model_info,
        "merge": merge_info,
        "updated_at": utcnow_iso(),
    }
    save_manifest(manifest_path, manifest)


def update_tile_manifest_entry(
    manifest_path: str | Path,
    date_value: str,
    tile: TileInfo,
    render_info: dict[str, Any],
    model_info: dict[str, Any],
    merge_info: dict[str, Any],
) -> None:
    manifest = load_manifest(manifest_path)
    key = f"{date_value}_tile_{tile.row}_{tile.col}"
    manifest["images"][key] = {
        "date": date_value,
        "tile_row": tile.row,
        "tile_col": tile.col,
        "tile_path": str(tile.path),
        "width": render_info["width"],
        "height": render_info["height"],
        "cache": render_info,
        "models": model_info,
        "merge": merge_info,
        "updated_at": utcnow_iso(),
    }
    save_manifest(manifest_path, manifest)


def get_manifest_entry(date_value: str, manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    entry = manifest.get("images", {}).get(date_value)
    if entry is None:
        raise KeyError(f"No manifest entry found for {date_value}")
    return entry


def ensure_cache_for_date(
    date_value: str,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> Path:
    entry = get_manifest_entry(date_value, manifest_path)
    cache_path_value = entry.get("cache", {}).get("path")
    cache_path = repo_resolve(cache_path_value) if cache_path_value else repo_resolve(Path(cache_dir) / f"{date_value}.png")
    if cache_path.exists():
        return cache_path

    recipe = {
        "date": entry["date"],
        "tile_count": entry["tile_count"],
        "rows": entry["rows"],
        "cols": entry["cols"],
        "width": entry["width"],
        "height": entry["height"],
        "placements": entry["placements"],
    }
    render_info = render_stitched_png(recipe, cache_path)
    update_manifest_entry(manifest_path, recipe, render_info, entry.get("models", {}), entry.get("merge", {}))
    return cache_path


def ensure_cache_for_tile(
    date_value: str,
    tile_row: int,
    tile_col: int,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> Path:
    cache_path = repo_resolve(Path(cache_dir) / f"{date_value}_tile_{tile_row}_{tile_col}.png")
    if cache_path.exists():
        return cache_path

    grouped = discover_tiles(source_dir)
    tiles = grouped.get(date_value, [])
    match = next((t for t in tiles if t.row == tile_row and t.col == tile_col), None)
    if match is None:
        raise FileNotFoundError(f"Tile ({tile_row},{tile_col}) for {date_value} not found in {source_dir}")
    render_tile_png(match, cache_path)
    return cache_path


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
    for index, detection in enumerate(merged, start=1):
        detection["review_order"] = index
    return merged


def delete_image_by_date(conn: sqlite3.Connection, date_value: str) -> None:
    conn.execute("DELETE FROM images WHERE date = ?", (date_value,))
    conn.commit()


def delete_image_by_tile(conn: sqlite3.Connection, date_value: str, tile_row: int, tile_col: int) -> None:
    conn.execute(
        "DELETE FROM images WHERE date = ? AND tile_row = ? AND tile_col = ?",
        (date_value, tile_row, tile_col),
    )
    conn.commit()


def image_exists(conn: sqlite3.Connection, date_value: str) -> bool:
    row = conn.execute("SELECT 1 FROM images WHERE date = ?", (date_value,)).fetchone()
    return row is not None


def tile_exists(conn: sqlite3.Connection, date_value: str, tile_row: int, tile_col: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM images WHERE date = ? AND tile_row = ? AND tile_col = ?",
        (date_value, tile_row, tile_col),
    ).fetchone()
    return row is not None


def insert_preprocessed_image(
    conn: sqlite3.Connection,
    date_value: str,
    cache_path: str | Path,
    width: int,
    height: int,
    tiles: list[TileInfo],
    raw_detections: list[dict[str, Any]],
    merged_detections: list[dict[str, Any]],
) -> int:
    timestamp = utcnow_iso()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO images (date, width, height, tile_count, cache_path, review_status, detail_cursor, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 'pending', NULL, ?, ?)
        """,
        (date_value, width, height, len(tiles), repo_relative(cache_path), timestamp, timestamp),
    )
    image_id = require_lastrowid(cursor)

    for tile in tiles:
        cursor.execute(
            """
            INSERT INTO image_sources (image_id, tile_path, tile_row, tile_col, tile_mtime, tile_size_bytes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (image_id, str(tile.path), tile.row, tile.col, tile.mtime, tile.size_bytes, timestamp),
        )

    raw_detection_ids: dict[str, int] = {}
    for raw in raw_detections:
        cursor.execute(
            """
            INSERT INTO detections (
                image_id, source_type, source_model, parent_detection_id, class_id, class_name,
                confidence, x_center, y_center, width, height, review_label, is_active,
                reviewed, review_order, reviewed_at, created_at, updated_at
            ) VALUES (?, 'raw_model', ?, NULL, ?, ?, ?, ?, ?, ?, ?, NULL, 0, 0, NULL, NULL, ?, ?)
            """,
            (
                image_id,
                raw["source_model"],
                raw["class_id"],
                raw["class_name"],
                raw["confidence"],
                raw["x_center"],
                raw["y_center"],
                raw["width"],
                raw["height"],
                timestamp,
                timestamp,
            ),
        )
        raw_detection_ids[raw["temp_id"]] = require_lastrowid(cursor)

    for merged in merged_detections:
        cursor.execute(
            """
            INSERT INTO detections (
                image_id, source_type, source_model, parent_detection_id, class_id, class_name,
                confidence, x_center, y_center, width, height, review_label, is_active,
                reviewed, review_order, reviewed_at, created_at, updated_at
            ) VALUES (?, 'merged', NULL, NULL, ?, ?, ?, ?, ?, ?, ?, NULL, 1, 0, ?, NULL, ?, ?)
            """,
            (
                image_id,
                merged["class_id"],
                merged["class_name"],
                merged["confidence"],
                merged["x_center"],
                merged["y_center"],
                merged["width"],
                merged["height"],
                merged["review_order"],
                timestamp,
                timestamp,
            ),
        )
        merged_id = require_lastrowid(cursor)
        for raw_id in merged["source_raw_ids"]:
            cursor.execute(
                """
                INSERT INTO detection_sources (detection_id, raw_detection_id, source_model)
                VALUES (?, ?, ?)
                """,
                (merged_id, raw_detection_ids[raw_id], raw_id.split(":", 1)[0]),
            )

    conn.commit()
    return image_id


def insert_preprocessed_tile(
    conn: sqlite3.Connection,
    date_value: str,
    tile: TileInfo,
    cache_path: str | Path,
    width: int,
    height: int,
    raw_detections: list[dict[str, Any]],
    merged_detections: list[dict[str, Any]],
) -> int:
    timestamp = utcnow_iso()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO images (date, tile_row, tile_col, width, height, tile_count, cache_path, review_status, detail_cursor, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 1, ?, 'pending', NULL, ?, ?)
        """,
        (date_value, tile.row, tile.col, width, height, repo_relative(cache_path), timestamp, timestamp),
    )
    image_id = require_lastrowid(cursor)

    cursor.execute(
        """
        INSERT INTO image_sources (image_id, tile_path, tile_row, tile_col, tile_mtime, tile_size_bytes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (image_id, str(tile.path), tile.row, tile.col, tile.mtime, tile.size_bytes, timestamp),
    )

    raw_detection_ids: dict[str, int] = {}
    for raw in raw_detections:
        cursor.execute(
            """
            INSERT INTO detections (
                image_id, source_type, source_model, parent_detection_id, class_id, class_name,
                confidence, x_center, y_center, width, height, review_label, is_active,
                reviewed, review_order, reviewed_at, created_at, updated_at
            ) VALUES (?, 'raw_model', ?, NULL, ?, ?, ?, ?, ?, ?, ?, NULL, 0, 0, NULL, NULL, ?, ?)
            """,
            (
                image_id,
                raw["source_model"],
                raw["class_id"],
                raw["class_name"],
                raw["confidence"],
                raw["x_center"],
                raw["y_center"],
                raw["width"],
                raw["height"],
                timestamp,
                timestamp,
            ),
        )
        raw_detection_ids[raw["temp_id"]] = require_lastrowid(cursor)

    for merged in merged_detections:
        cursor.execute(
            """
            INSERT INTO detections (
                image_id, source_type, source_model, parent_detection_id, class_id, class_name,
                confidence, x_center, y_center, width, height, review_label, is_active,
                reviewed, review_order, reviewed_at, created_at, updated_at
            ) VALUES (?, 'merged', NULL, NULL, ?, ?, ?, ?, ?, ?, ?, NULL, 1, 0, ?, NULL, ?, ?)
            """,
            (
                image_id,
                merged["class_id"],
                merged["class_name"],
                merged["confidence"],
                merged["x_center"],
                merged["y_center"],
                merged["width"],
                merged["height"],
                merged["review_order"],
                timestamp,
                timestamp,
            ),
        )
        merged_id = require_lastrowid(cursor)
        for raw_id in merged["source_raw_ids"]:
            cursor.execute(
                """
                INSERT INTO detection_sources (detection_id, raw_detection_id, source_model)
                VALUES (?, ?, ?)
                """,
                (merged_id, raw_detection_ids[raw_id], raw_id.split(":", 1)[0]),
            )

    conn.commit()
    return image_id


def get_image_queue(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT image_id, date, tile_row, tile_col, review_status, detail_cursor, width, height, tile_count, cache_path
        FROM images
        ORDER BY date ASC, tile_row ASC, tile_col ASC
        """
    ).fetchall()


def get_image_row(conn: sqlite3.Connection, image_id: int) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT image_id, date, tile_row, tile_col, review_status, detail_cursor, width, height, tile_count, cache_path,
               created_at, updated_at
        FROM images
        WHERE image_id = ?
        """,
        (image_id,),
    ).fetchone()


def get_image_row_by_date(conn: sqlite3.Connection, date_value: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT image_id, date, tile_row, tile_col, review_status, detail_cursor, width, height, tile_count, cache_path FROM images WHERE date = ?",
        (date_value,),
    ).fetchone()


def get_reviewable_detections(conn: sqlite3.Connection, image_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT detection_id, image_id, source_type, source_model, parent_detection_id, class_id, class_name,
               confidence, x_center, y_center, width, height, review_label, is_active, reviewed,
               review_order, reviewed_at, created_at, updated_at
        FROM detections
        WHERE image_id = ? AND source_type IN ('merged', 'manual')
        ORDER BY COALESCE(review_order, 999999), detection_id
        """,
        (image_id,),
    ).fetchall()


def get_active_detection_queue(conn: sqlite3.Connection, image_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT detection_id, image_id, source_type, source_model, parent_detection_id, class_id, class_name,
               confidence, x_center, y_center, width, height, review_label, is_active, reviewed,
               review_order, reviewed_at, created_at, updated_at
        FROM detections
        WHERE image_id = ? AND source_type IN ('merged', 'manual') AND is_active = 1
        ORDER BY COALESCE(review_order, 999999), detection_id
        """,
        (image_id,),
    ).fetchall()


def set_image_status(conn: sqlite3.Connection, image_id: int, review_status: str, detail_cursor: int | None) -> None:
    conn.execute(
        "UPDATE images SET review_status = ?, detail_cursor = ?, updated_at = ? WHERE image_id = ?",
        (review_status, detail_cursor, utcnow_iso(), image_id),
    )
    conn.commit()


def get_next_review_order(conn: sqlite3.Connection, image_id: int) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(review_order), 0) FROM detections WHERE image_id = ? AND source_type IN ('merged', 'manual')",
        (image_id,),
    ).fetchone()
    return int(row[0] or 0) + 1


def clamp_norm(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def detection_payload(row: sqlite3.Row, image_width: int, image_height: int) -> dict[str, Any]:
    x_center = float(row["x_center"])
    y_center = float(row["y_center"])
    width = float(row["width"])
    height = float(row["height"])
    return {
        "detection_id": int(row["detection_id"]),
        "image_id": int(row["image_id"]),
        "source_type": row["source_type"],
        "source_model": row["source_model"],
        "parent_detection_id": row["parent_detection_id"],
        "class_id": int(row["class_id"]),
        "class_name": row["class_name"],
        "confidence": None if row["confidence"] is None else float(row["confidence"]),
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
        "x1_px": round((x_center - width / 2.0) * image_width, 3),
        "y1_px": round((1.0 - y_center - height / 2.0) * image_height, 3),
        "x2_px": round((x_center + width / 2.0) * image_width, 3),
        "y2_px": round((1.0 - y_center + height / 2.0) * image_height, 3),
        "review_label": row["review_label"],
        "is_active": bool(row["is_active"]),
        "reviewed": bool(row["reviewed"]),
        "review_order": row["review_order"],
        "reviewed_at": row["reviewed_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def first_unreviewed_active_index(conn: sqlite3.Connection, image_id: int) -> int | None:
    active = get_active_detection_queue(conn, image_id)
    for index, row in enumerate(active):
        if not bool(row["reviewed"]):
            return index
    return None


def sync_image_review_status(conn: sqlite3.Connection, image_id: int) -> tuple[str, int | None]:
    image = get_image_row(conn, image_id)
    if image is None:
        raise KeyError(f"Image {image_id} not found")

    current_status = image["review_status"]
    active = get_active_detection_queue(conn, image_id)
    pending_index = first_unreviewed_active_index(conn, image_id)
    reviewed_count = sum(1 for row in active if bool(row["reviewed"]))

    if current_status == "completed":
        return current_status, image["detail_cursor"]
    if current_status == "pending":
        set_image_status(conn, image_id, "overview_1", None)
        return "overview_1", None
    if current_status == "detail":
        if pending_index is None:
            set_image_status(conn, image_id, "overview_2", None)
            return "overview_2", None
        cursor = image["detail_cursor"]
        if cursor is None or cursor < 0 or cursor >= len(active):
            cursor = pending_index
            set_image_status(conn, image_id, "detail", cursor)
            return "detail", cursor
        return "detail", int(cursor)
    if current_status == "overview_2" and reviewed_count < len(active):
        set_image_status(conn, image_id, "detail", pending_index)
        return "detail", pending_index
    return current_status, image["detail_cursor"]


def create_manual_detection(
    conn: sqlite3.Connection,
    image_id: int,
    class_id: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    parent_detection_id: int | None = None,
) -> int:
    timestamp = utcnow_iso()
    review_order = get_next_review_order(conn, image_id)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO detections (
            image_id, source_type, source_model, parent_detection_id, class_id, class_name,
            confidence, x_center, y_center, width, height, review_label, is_active,
            reviewed, review_order, reviewed_at, created_at, updated_at
        ) VALUES (?, 'manual', NULL, ?, ?, ?, NULL, ?, ?, ?, ?, NULL, 1, 0, ?, NULL, ?, ?)
        """,
        (
            image_id,
            parent_detection_id,
            class_id,
            CLASS_ID_TO_NAME[class_id],
            clamp_norm(x_center),
            clamp_norm(y_center),
            clamp_norm(width),
            clamp_norm(height),
            review_order,
            timestamp,
            timestamp,
        ),
    )
    detection_id = require_lastrowid(cursor)
    conn.commit()
    return detection_id


def update_detection_geometry(
    conn: sqlite3.Connection,
    detection_id: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    class_id: int | None = None,
) -> None:
    if class_id is None:
        conn.execute(
            """
            UPDATE detections
            SET x_center = ?, y_center = ?, width = ?, height = ?, updated_at = ?
            WHERE detection_id = ?
            """,
            (clamp_norm(x_center), clamp_norm(y_center), clamp_norm(width), clamp_norm(height), utcnow_iso(), detection_id),
        )
    else:
        conn.execute(
            """
            UPDATE detections
            SET class_id = ?, class_name = ?, x_center = ?, y_center = ?, width = ?, height = ?, updated_at = ?
            WHERE detection_id = ?
            """,
            (
                class_id,
                CLASS_ID_TO_NAME[class_id],
                clamp_norm(x_center),
                clamp_norm(y_center),
                clamp_norm(width),
                clamp_norm(height),
                utcnow_iso(),
                detection_id,
            ),
        )
    conn.commit()


def label_detection(
    conn: sqlite3.Connection,
    image_id: int,
    detection_id: int,
    review_label: str,
    class_id: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> tuple[str, int | None]:
    reviewed_at = utcnow_iso()
    conn.execute(
        """
        UPDATE detections
        SET class_id = ?, class_name = ?, review_label = ?, reviewed = 1, is_active = 1,
            x_center = ?, y_center = ?, width = ?, height = ?, reviewed_at = ?, updated_at = ?
        WHERE detection_id = ? AND image_id = ?
        """,
        (
            class_id,
            CLASS_ID_TO_NAME[class_id],
            review_label,
            clamp_norm(x_center),
            clamp_norm(y_center),
            clamp_norm(width),
            clamp_norm(height),
            reviewed_at,
            reviewed_at,
            detection_id,
            image_id,
        ),
    )
    conn.commit()
    return advance_detail_cursor(conn, image_id)


def mark_detection_neither(conn: sqlite3.Connection, image_id: int, detection_id: int) -> tuple[str, int | None]:
    timestamp = utcnow_iso()
    image = get_image_row(conn, image_id)
    conn.execute(
        """
        UPDATE detections
        SET review_label = 'neither', reviewed = 1, is_active = 0, reviewed_at = ?, updated_at = ?
        WHERE detection_id = ? AND image_id = ?
        """,
        (timestamp, timestamp, detection_id, image_id),
    )
    conn.commit()
    if image is None:
        raise KeyError(f"Image {image_id} not found")
    if image["review_status"] != "detail":
        return image["review_status"], image["detail_cursor"]
    return advance_detail_cursor(conn, image_id)


def start_detail_mode(conn: sqlite3.Connection, image_id: int) -> tuple[str, int | None]:
    pending_index = first_unreviewed_active_index(conn, image_id)
    if pending_index is None:
        set_image_status(conn, image_id, "overview_2", None)
        return "overview_2", None
    set_image_status(conn, image_id, "detail", pending_index)
    return "detail", pending_index


def advance_detail_cursor(conn: sqlite3.Connection, image_id: int) -> tuple[str, int | None]:
    pending_index = first_unreviewed_active_index(conn, image_id)
    if pending_index is None:
        set_image_status(conn, image_id, "overview_2", None)
        return "overview_2", None
    set_image_status(conn, image_id, "detail", pending_index)
    return "detail", pending_index


def navigate_detail_cursor(conn: sqlite3.Connection, image_id: int, direction: str) -> tuple[str, int | None]:
    image = get_image_row(conn, image_id)
    if image is None:
        raise KeyError(f"Image {image_id} not found")
    active = get_active_detection_queue(conn, image_id)
    if not active:
        set_image_status(conn, image_id, "overview_2", None)
        return "overview_2", None

    current = image["detail_cursor"] or 0
    if direction == "prev":
        next_cursor = max(0, current - 1)
    else:
        next_cursor = min(len(active) - 1, current + 1)
    set_image_status(conn, image_id, "detail", next_cursor)
    return "detail", next_cursor


def complete_image_review(conn: sqlite3.Connection, image_id: int) -> None:
    pending_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM detections
        WHERE image_id = ? AND source_type IN ('merged', 'manual') AND is_active = 1 AND reviewed = 0
        """,
        (image_id,),
    ).fetchone()[0]
    if pending_count:
        raise ValueError("Cannot complete image while active detections remain unlabeled")
    set_image_status(conn, image_id, "completed", None)


def review_counts(conn: sqlite3.Connection, image_id: int) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total_detections,
            SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) AS active_detections,
            SUM(CASE WHEN is_active = 1 AND reviewed = 1 THEN 1 ELSE 0 END) AS reviewed_active_detections,
            SUM(CASE WHEN is_active = 1 AND reviewed = 0 THEN 1 ELSE 0 END) AS pending_active_detections,
            SUM(CASE WHEN review_label = 'neither' THEN 1 ELSE 0 END) AS neither_detections
        FROM detections
        WHERE image_id = ? AND source_type IN ('merged', 'manual')
        """,
        (image_id,),
    ).fetchone()
    return {
        "total_detections": int(row["total_detections"] or 0),
        "active_detections": int(row["active_detections"] or 0),
        "reviewed_active_detections": int(row["reviewed_active_detections"] or 0),
        "pending_active_detections": int(row["pending_active_detections"] or 0),
        "neither_detections": int(row["neither_detections"] or 0),
    }


def queue_progress(conn: sqlite3.Connection) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total_images,
            SUM(CASE WHEN review_status = 'completed' THEN 1 ELSE 0 END) AS completed_images
        FROM images
        """
    ).fetchone()
    return {
        "total_images": int(row["total_images"] or 0),
        "completed_images": int(row["completed_images"] or 0),
    }


def exportable_detections(conn: sqlite3.Connection, image_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT detection_id, class_id, class_name, x_center, y_center, width, height, review_label
        FROM detections
        WHERE image_id = ? AND source_type IN ('merged', 'manual') AND is_active = 1 AND review_label IN ('ship_non_sts', 'ships_sts')
        ORDER BY COALESCE(review_order, 999999), detection_id
        """,
        (image_id,),
    ).fetchall()


def write_yolo_label_file(label_path: str | Path, detections: list[sqlite3.Row]) -> None:
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


def write_data_yaml(dataset_dir: str | Path) -> None:
    target = repo_resolve(Path(dataset_dir) / "data.yaml")
    ensure_parent(target)
    content = (
        f"path: {repo_resolve(dataset_dir)}\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        "  0: ship\n"
        "  1: ships\n"
    )
    target.write_text(content)
