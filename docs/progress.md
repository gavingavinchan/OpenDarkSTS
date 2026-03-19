# Progress Tracking

## Current Dataset Workspace

- Source copied from read-only folder: `/home/gavin/Downloads/AIS data/Madhulika/data science dark fleet.v7i.yolov8 (1).zip`
- Working copy stored at `workspace/datasets/raw/data_science_dark_fleet_v7i_yolov8.zip`
- Extracted dataset stored at `workspace/datasets/extracted/madhulika_yolov8`
- Dataset split sizes: 336 train, 27 valid, 18 test
- Current classes in `workspace/datasets/extracted/madhulika_yolov8/data.yaml`: `ship`, `ships`

## Environment

- Local Python environment created at `.venv/`
- Setup notes stored in `docs/local-venv.md`
- Comparison script stored in `scripts/compare_yolo_models.py`

## YOLO Benchmark: 50 Epochs

- Report: `workspace/reports/yolo_compare_summary.json`
- YOLOv8 test metrics: precision 0.7475, recall 0.6987, mAP50 0.7951, mAP50-95 0.4934
- YOLO26 test metrics: precision 0.7295, recall 0.7575, mAP50 0.7902, mAP50-95 0.4933
- Initial conclusion: results were very close; YOLOv8 was slightly better overall and faster, while YOLO26 had slightly better recall

## YOLO Benchmark: 150 Epochs Max

- Report: `workspace/reports/yolo_compare_summary_150.json`
- YOLOv8 run directory: `runs/detect/workspace/runs/model_compare_150/yolov8n_e150_img960`
- YOLO26 run directory: `runs/detect/workspace/runs/model_compare_150/yolo26n_e150_img960`

### Early Stopping Behavior

- YOLOv8 stopped early after epoch 112; best validation mAP50-95 was at epoch 92
- YOLO26 stopped early after epoch 87; best validation mAP50-95 was at epoch 67
- Conclusion: both models plateaued before 150 epochs under `patience=20`

### Best Validation Scores

- YOLOv8: best val mAP50 0.9042, best val mAP50-95 0.5556
- YOLO26: best val mAP50 0.8758, best val mAP50-95 0.5568

### Test Set Results From Best Weights

- YOLOv8: precision 0.8753, recall 0.6421, mAP50 0.8079, mAP50-95 0.4970
- YOLO26: precision 0.8601, recall 0.7020, mAP50 0.8263, mAP50-95 0.5251

### Current Conclusion

- With longer training, YOLO26 is the stronger model on the current test split
- YOLOv8 keeps slightly higher precision
- YOLO26 now has better recall and better mAP, which is promising for candidate STS event detection

## Known Dataset Issue

- Ultralytics warned that the dataset appears to mix boxes and segments
- Training proceeded using boxes only
- This dataset should be cleaned or standardized before treating current metrics as final

## Current Best Weights

- YOLOv8 best weights: `runs/detect/workspace/runs/model_compare_150/yolov8n_e150_img960/weights/best.pt`
- YOLO26 best weights: `runs/detect/workspace/runs/model_compare_150/yolo26n_e150_img960/weights/best.pt`

## Suggested Next Steps

- Inspect false positives and false negatives from the best YOLO26 model
- Clean the mixed box/segment dataset issue and rerun the benchmark
- Decide whether to force a full 150-epoch run with early stopping disabled for curve inspection
