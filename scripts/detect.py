from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO

try:
    from .sts_utils import load_image_size, parse_image_date, resolve_image_inputs, write_json
except ImportError:
    from sts_utils import load_image_size, parse_image_date, resolve_image_inputs, write_json


def detect_image(image_path: str | Path, model: YOLO, conf: float = 0.25) -> list[dict[str, Any]]:
    image_path = Path(image_path).resolve()
    image_width, image_height = load_image_size(image_path)
    image_date = parse_image_date(image_path)

    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    if not results:
        return []

    result = results[0]
    names = getattr(result, "names", {}) or {}
    ships_ids = {class_id for class_id, class_name in names.items() if class_name == "ships"}
    if not ships_ids and 1 in names:
        ships_ids = {1}
    if not ships_ids:
        raise ValueError("The loaded YOLO model does not expose a 'ships' class")

    detections: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for index, box in enumerate(boxes, start=1):
        class_id = int(box.cls.item())
        if class_id not in ships_ids:
            continue

        xywh = box.xywh[0].tolist()
        detections.append(
            {
                "detection_id": f"{image_path.stem}:{index}",
                "image_path": str(image_path),
                "date": image_date,
                "cx_px": round(float(xywh[0]), 4),
                "cy_px": round(float(xywh[1]), 4),
                "w_px": round(float(xywh[2]), 4),
                "h_px": round(float(xywh[3]), 4),
                "confidence": round(float(box.conf.item()), 6),
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "image_width_px": image_width,
                "image_height_px": image_height,
            }
        )

    return detections


def detect_images(image_paths: list[str | Path], weights: str | Path, conf: float = 0.25) -> list[dict[str, Any]]:
    model = YOLO(str(weights))
    detections: list[dict[str, Any]] = []
    for image_path in image_paths:
        detections.extend(detect_image(image_path=image_path, model=model, conf=conf))
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference and keep only 'ships' detections")
    parser.add_argument("--image", nargs="+", required=True, help="Image path(s), directories, or glob patterns")
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", required=True, help="Path to write detections JSON")
    args = parser.parse_args()

    image_paths = resolve_image_inputs(args.image)
    if not image_paths:
        raise FileNotFoundError("No input images were found")

    detections = detect_images(image_paths=image_paths, weights=args.weights, conf=args.conf)
    write_json(args.output, detections)
    print(f"Wrote {len(detections)} 'ships' detections to {args.output}")


if __name__ == "__main__":
    main()
