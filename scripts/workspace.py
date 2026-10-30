"""Shared workspace paths and small persistent records."""
from __future__ import annotations

import hashlib
import errno
import json
import os
import tempfile
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = Path(os.environ.get("OPENDARKSTS_WORKSPACE", Path.cwd() / "workspace")).expanduser().resolve()


def default_manifest() -> dict[str, Any]:
    # Direct script entry points put scripts/, rather than the project root, on
    # sys.path. Prefer the adjacent package so another installed checkout cannot
    # supply a stale manifest. Installed wheels retain the same sibling layout.
    adjacent = REPO_ROOT / "opendarksts" / "assets" / "default_model.json"
    if adjacent.is_file():
        return json.loads(adjacent.read_text(encoding="utf-8"))
    return json.loads(files("opendarksts").joinpath("assets/default_model.json").read_text(encoding="utf-8"))


DEFAULT_WEIGHT_NAME = default_manifest()["filename"]


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
    target = contained(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise BlockingIOError("Workspace lock is held by another process") from exc
                raise
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


def model_selection_status(model_weights_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(model_weights_dir or WORKSPACE_DIR / "model_weights").resolve()
    manifest = default_manifest()
    selected = current_model(root)
    default = contained(root / manifest["filename"], root)
    available = default.is_file() and sha256(default) == manifest["sha256"]
    return {"current_model": str(selected), "default_model": manifest["filename"],
            "default_available": available, "is_default": available and selected == default}


def restore_default_model(model_weights_dir: str | Path | None = None) -> dict[str, Any]:
    root = contained(model_weights_dir or WORKSPACE_DIR / "model_weights")
    manifest = default_manifest()
    default = contained(root / manifest["filename"], root)
    if not default.is_file() or sha256(default) != manifest["sha256"]:
        raise ValueError("Bundled default is missing or its checksum changed; run opendarksts init to restore it")
    set_current_model(default, source_url=manifest["url"], metrics=manifest["metrics"])
    return model_selection_status(root)
