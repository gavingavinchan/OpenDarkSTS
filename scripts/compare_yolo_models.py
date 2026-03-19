from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

from ultralytics import YOLO


def extract_metrics(result) -> dict:
    metrics = {}
    results_dict = getattr(result, "results_dict", {}) or {}
    for key, value in results_dict.items():
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue

    box = getattr(result, "box", None)
    if box is not None:
        for key in ("map", "map50", "map75", "mp", "mr"):
            value = getattr(box, key, None)
            if value is not None:
                metrics[f"box_{key}"] = float(value)

    speed = getattr(result, "speed", None)
    if speed is not None:
        metrics["speed_ms"] = {k: float(v) for k, v in speed.items()}

    return metrics


def train_and_evaluate(model_name: str, args: argparse.Namespace, output_dir: Path) -> dict:
    run_name = f"{Path(model_name).stem}_e{args.epochs}_img{args.imgsz}"
    model = YOLO(model_name)

    start = time()
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(output_dir),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        seed=args.seed,
        deterministic=True,
        workers=args.workers,
        patience=args.patience,
        plots=True,
        verbose=True,
    )
    train_seconds = time() - start

    best_path = Path(model.trainer.best)
    best_model = YOLO(str(best_path))
    test_result = best_model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(output_dir),
        name=f"{run_name}_test",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    return {
        "model": model_name,
        "run_name": run_name,
        "train_seconds": round(train_seconds, 2),
        "best_weights": str(best_path),
        "test_metrics": extract_metrics(test_result),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 and YOLO26 and compare test metrics.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--output-dir", default="workspace/runs/model_compare", help="Training output directory")
    parser.add_argument("--report", default="workspace/reports/yolo_compare_summary.json", help="Path to JSON summary")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    models = ["yolov8n.pt", "yolo26n.pt"]
    results = []
    for model_name in models:
        results.append(train_and_evaluate(model_name, args, output_dir))

    summary = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "seed": args.seed,
        "results": results,
    }
    report_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
