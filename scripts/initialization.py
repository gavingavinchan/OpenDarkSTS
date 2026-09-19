"""First-run workspace creation and checksummed default model acquisition."""
from __future__ import annotations

import json
import shutil
import tempfile
from importlib.resources import files
from pathlib import Path

from .workspace import WORKSPACE_DIR, atomic_json, contained, current_model, exclusive_lock, set_current_model, sha256


def default_manifest():
    return json.loads(files("opendarksts").joinpath("assets/default_model.json").read_text(encoding="utf-8"))


def initialize(*, weights=None, no_model=False, example=True):
    manifest = default_manifest()
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(WORKSPACE_DIR / ".init.lock", blocking=True):
        for name in ["inputs", "model_weights", "reports", "jobs", "reviews", "runs", "datasets", "manifests"]:
            contained(WORKSPACE_DIR / name, WORKSPACE_DIR).mkdir(parents=True, exist_ok=True)
        installed = None
        model_root = WORKSPACE_DIR / "model_weights"
        if not no_model:
            target = contained(model_root / manifest["filename"], WORKSPACE_DIR)
            if not target.is_file() or sha256(target) != manifest["sha256"]:
                with tempfile.TemporaryDirectory(prefix="init_", dir=model_root) as temporary:
                    candidate = Path(temporary) / manifest["filename"]
                    if weights:
                        shutil.copyfile(Path(weights).expanduser(), candidate)
                    else:
                        from .acquisition import download
                        try:
                            download(manifest["url"], candidate)
                        except RuntimeError as exc:
                            raise RuntimeError(f"Default model download failed. {manifest['release_status']} Supply --weights with the documented checkpoint for local testing.") from exc
                    if sha256(candidate) != manifest["sha256"]:
                        raise ValueError("Default model checksum mismatch; existing model was not replaced")
                    candidate.replace(target)
            for filename in ["model_card.md", "weight_recreation.md"]:
                resource = files("opendarksts").joinpath("assets", filename)
                if resource.is_file():
                    (model_root / filename).write_text(resource.read_text(encoding="utf-8"), encoding="utf-8")
            # Re-running init must not reset a model explicitly promoted in Improve.
            if not (model_root / "current.json").exists():
                set_current_model(target, metrics=manifest["metrics"], source_url=manifest["url"])
            installed = str(current_model(model_root))
        example_dir = contained(WORKSPACE_DIR / "inputs" / "example", WORKSPACE_DIR)
        if example:
            example_dir.mkdir(parents=True, exist_ok=True)
            for source in files("opendarksts").joinpath("example").iterdir():
                if source.is_file():
                    target = contained(example_dir / source.name, WORKSPACE_DIR)
                    if target.exists() and target.read_bytes() != source.read_bytes():
                        raise ValueError(f"Example file has local changes: {target}")
                    if not target.exists():
                        target.write_bytes(source.read_bytes())
        return {"workspace": str(WORKSPACE_DIR), "model": installed, "example": str(example_dir) if example else None}
