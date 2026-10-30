"""Prepare a local, unpublished annotation/split bundle for contributor review."""
import argparse
import json
import zipfile
from pathlib import Path

from .workspace import WORKSPACE_DIR, atomic_json, contained, sha256


def prepare(dataset, output):
    dataset = Path(dataset).resolve()
    output = contained(output, WORKSPACE_DIR)
    source_manifest = dataset / "dataset_manifest.json"
    manifest = json.loads(source_manifest.read_text())
    if not manifest.get("summary", {}).get("leak_check", {}).get("ok"):
        raise ValueError("A passing split leakage check is required")
    records = []
    members = []
    for item in manifest["items"]:
        label = contained(dataset / item["output_label"], dataset)
        image = contained(dataset / item["output_image"], dataset)
        image_hash = sha256(image)
        if item["sha256"].removeprefix("sha256:") != image_hash:
            raise ValueError(f"Dataset image changed: {item['item_id']}")
        record = {key: item[key] for key in ["item_id", "original_name", "source", "split", "group", "region", "date", "class_counts", "output_image", "output_label"]}
        record.update(image_sha256=image_hash, label_sha256=sha256(label), rights_status="pending contributor/source review")
        records.append(record)
        members.append((label, item["output_label"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Choose a new annotation bundle output path")
    review_manifest = {"dataset_name": manifest["dataset_name"], "classes": manifest["classes"],
          "summary": manifest["summary"], "split_rule": manifest["split_rule"], "items": records,
          "source_manifest_sha256": sha256(source_manifest), "publication_status": "NOT CLEARED FOR PUBLICATION"}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path, member in members:
            bundle.write(path, member)
        bundle.writestr("split_manifest.json", json.dumps(review_manifest, indent=2))
        bundle.writestr("README.txt", "Local review candidate only. Contributor/source rights review is pending. This bundle contains labels and split metadata; source imagery and model weights are excluded. Image and label hashes allow verification against the documented dataset.\n")
    record = {"path": str(output), "sha256": sha256(output), "annotation_files": len(members), "publication_status": "NOT CLEARED FOR PUBLICATION"}
    atomic_json(output.with_suffix(".json"), record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.output)))
