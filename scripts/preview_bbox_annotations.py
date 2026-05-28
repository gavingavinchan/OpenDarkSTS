import cv2
import random
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────
IMAGE_DIR = Path(r"C:\Users\Lenovo\Documents\data science project\data science dark fleet.v7i.yolov8 (1)\train\images")
LABEL_DIR = Path(r"C:\Users\Lenovo\Documents\data science project\data science dark fleet.v7i.yolov8 (1)\train\labels")
OUTPUT_DIR = Path(r"C:\Users\Lenovo\Documents\OpenDarkSTS\bbox_preview")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES  = ["ship", "ships"]
COLORS       = [(0, 200, 0), (0, 0, 255)]  # green=ship, red=ships
NUM_SAMPLES  = 10  # how many images to preview

# ── DRAW BOXES ────────────────────────────────────────────────
image_files = sorted(IMAGE_DIR.glob("*.jpg")) + sorted(IMAGE_DIR.glob("*.png"))
samples     = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))

for img_path in samples:
    lbl_path = LABEL_DIR / (img_path.stem + ".txt")
    if not lbl_path.exists():
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]

    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])

        # Convert YOLO normalized to pixel coords
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        color = COLORS[cls_id] if cls_id < len(COLORS) else (255, 255, 0)
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_path = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(out_path), img)
    print(f"✅ Saved: {out_path.name}")

print(f"\n✅ Done! Check: {OUTPUT_DIR}")