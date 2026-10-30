"""Recreate the documented small release example from its source tile and AIS day."""
import argparse
import csv
from pathlib import Path

from .workspace import WORKSPACE_DIR, atomic_json, contained, sha256


def build_example(image, ais, output):
    import rasterio
    from rasterio.windows import Window
    output = contained(output, WORKSPACE_DIR)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "2020-01-24_example.tiff"
    with rasterio.open(image) as source:
        window = Window(0, 320, 1024, 1024)
        profile = source.profile.copy()
        profile.update(width=1024, height=1024, transform=source.window_transform(window), compress="deflate")
        with rasterio.open(target, "w", **profile) as destination:
            destination.write(source.read(window=window))
    with rasterio.open(target) as dataset:
        b = dataset.bounds
        bounds = [b.left-.02, b.bottom-.02, b.right+.02, b.top+.02]
    csv_path = output / "AIS_2020_01_24.csv"
    count = 0
    with Path(ais).open(newline="", encoding="utf-8-sig") as source, csv_path.open("w", newline="", encoding="utf-8") as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            try:
                lon, lat = float(row["LON"]), float(row["LAT"])
            except (ValueError, TypeError):
                continue
            if bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]:
                writer.writerow(row)
                count += 1
    metadata = {"complete": True, "identity": {"date": "2020-01-24", "bbox": bounds, "source_sha256": sha256(ais)},
                "sha256": sha256(csv_path), "row_count": count,
                "source_url": "https://noaaocm.blob.core.windows.net/ais/csv2/csv2020/ais-2020-01-24.csv.zst"}
    atomic_json(csv_path.with_suffix(".csv.json"), metadata)
    atomic_json(output / "provenance.json", {"image_source_filename": Path(image).name,
        "image_source_sha256": sha256(image), "image_sha256": sha256(target), "crop_pixels": [0,320,1024,1024],
        "attribution": "Contains modified Copernicus Sentinel data 2020; NOAA Office for Coastal Management AIS data.",
        "acquisition_date": "2020-01-24", "product_id": None,
        "known_gaps": "Original Sentinel product identifier, acquisition time and rendering request were not retained. Date is from the source filename. No inference performance claim is made from this example.",
        "ais": metadata})
    return {"output": str(output), "ais_rows": count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--ais", required=True)
    parser.add_argument("--output", default=str(WORKSPACE_DIR / "exports" / "release_example"))
    args = parser.parse_args()
    print(build_example(args.image, args.ais, args.output))
