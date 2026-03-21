from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────
LABEL_DIRS = [
    r"C:\Users\Lenovo\Documents\data science project\data science dark fleet.v7i.yolov8 (1)\train\labels",
    r"C:\Users\Lenovo\Documents\data science project\data science dark fleet.v7i.yolov8 (1)\valid\labels",
    r"C:\Users\Lenovo\Documents\data science project\data science dark fleet.v7i.yolov8 (1)\test\labels",
]

converted = 0
skipped   = 0

for label_dir in LABEL_DIRS:
    label_dir = Path(label_dir)
    if not label_dir.exists():
        print(f"⚠️  Not found: {label_dir}")
        continue

    for txt_file in label_dir.glob("*.txt"):
        lines     = txt_file.read_text().splitlines()
        new_lines = []
        changed   = False

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            cls_id = parts[0]
            coords = list(map(float, parts[1:]))

            if len(coords) == 4:
                # Already a bounding box (cx, cy, w, h) — keep as is
                new_lines.append(line)
                continue

            if len(coords) >= 6 and len(coords) % 2 == 0:
                # Polygon: x1 y1 x2 y2 x3 y3 ... → convert to bbox
                xs = coords[0::2]
                ys = coords[1::2]

                x_min = min(xs)
                x_max = max(xs)
                y_min = min(ys)
                y_max = max(ys)

                # Convert to YOLO format (cx, cy, w, h) normalized
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                w  = x_max - x_min
                h  = y_max - y_min

                new_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                changed = True
            else:
                # Unknown format — keep as is
                new_lines.append(line)

        if changed:
            txt_file.write_text("\n".join(new_lines) + "\n")
            converted += 1
        else:
            skipped += 1

print(f"\n✅ Conversion complete!")
print(f"   Converted : {converted} label files")
print(f"   Unchanged : {skipped} label files (already bounding boxes)")
print(f"\nNow retrain with the corrected labels!")