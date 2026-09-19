"""Shared workspace paths and small persistent records."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = Path(os.environ.get("OPENDARKSTS_WORKSPACE", REPO_ROOT / "workspace")).expanduser().resolve()
DEFAULT_WEIGHT_NAME = "yolo26n_sts_clean_v1_e300_img960_2026-09-15_best.pt"


def contained(path: str | Path, root: str | Path = WORKSPACE_DIR) -> Path:
    target = Path(path).expanduser().resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Output path must stay inside {root}: {path}")
    return target


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    content = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def current_model(model_weights_dir: str | Path | None = None) -> Path:
    root = Path(model_weights_dir or WORKSPACE_DIR / "model_weights").resolve()
    record = root / "current.json"
    if not record.is_file():
        return root / DEFAULT_WEIGHT_NAME
    payload = json.loads(record.read_text(encoding="utf-8"))
    weight = contained(root / payload["filename"], root)
    if not weight.is_file() or sha256(weight) != payload["sha256"]:
        raise ValueError("The current model is missing or its checksum changed; initialize or promote a model again")
    return weight


def set_current_model(weight: str | Path, **metadata: Any) -> dict[str, Any]:
    path = Path(weight).resolve()
    record = {**metadata, "filename": path.name, "sha256": sha256(path)}
    atomic_json(path.parent / "current.json", record)
    return record


@contextmanager
def exclusive_lock(path: str | Path, *, blocking=False):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
