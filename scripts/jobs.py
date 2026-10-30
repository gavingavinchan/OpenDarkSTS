"""Persistent background jobs shared by the HTTP and command-line interfaces."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .workspace import WORKSPACE_DIR, atomic_json, contained, exclusive_lock
from . import acquisition, pipeline, process_support
from . import training_pipeline_lib as training
from .training_pipeline_lib import process_is_alive, read_process_start_ticks

JOB_ROOT = WORKSPACE_DIR / "jobs"
TERMINAL = {"completed", "failed", "cancelled"}


def now():
    return datetime.now(timezone.utc).isoformat()


def job_dir(job_id):
    if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        raise ValueError("Invalid job id")
    return contained(JOB_ROOT / job_id, JOB_ROOT)


def validate_options(kind, options):
    allowed = {"ais": {"start", "end", "bbox", "output_dir"},
               "imagery": {"start", "end", "bbox", "output_dir", "max_cloud"},
               "run": {"images", "ais_dir", "date", "weights", "conf", "radius", "output_dir"}}
    if kind not in allowed or not isinstance(options, dict) or set(options) - allowed[kind]:
        raise ValueError("Invalid job kind or options")
    if options.get("output_dir"):
        contained(options["output_dir"])
    if kind in {"ais", "imagery"}:
        acquisition.date_range(options.get("start", ""), options.get("end"))
        if kind == "imagery" or options.get("bbox") is not None:
            acquisition.validate_bbox(options.get("bbox") or [])
    elif not isinstance(options.get("images"), list) or not options["images"]:
        raise ValueError("Select at least one imagery folder or file")
    return dict(options)


def _read_job(job_id, *, reconcile=True):
    directory = job_dir(job_id)
    record = json.loads(contained(directory / "job.json", directory).read_text())
    if (not isinstance(record, dict) or record.get("job_id") != job_id
            or not isinstance(record.get("kind"), str) or not isinstance(record.get("status"), str)
            or record.get("kind") not in {"ais", "imagery", "run"}
            or record.get("status") not in TERMINAL | {"queued", "running"}
            or not isinstance(record.get("options"), dict)
            or not isinstance(record.get("created_at"), str)):
        raise ValueError(f"Invalid job record: {job_id}")
    try:
        created_at = datetime.fromisoformat(record["created_at"])
        if created_at.tzinfo is None:
            raise ValueError("Job timestamp must include a timezone")
    except ValueError as exc:
        raise ValueError(f"Invalid job creation timestamp: {job_id}") from exc
    launch = contained(directory / "launch.json", directory)
    if launch.is_file():
        launch_record = json.loads(launch.read_text())
        if (not isinstance(launch_record, dict) or set(launch_record) - {"pid", "start_ticks"}
                or type(launch_record.get("pid")) is not int or launch_record["pid"] <= 0
                or (launch_record.get("start_ticks") is not None and type(launch_record["start_ticks"]) is not int)):
            raise ValueError(f"Invalid job launch record: {job_id}")
        record.update(launch_record)
    if reconcile and record["status"] not in TERMINAL and record.get("pid"):
        if not process_is_alive(record["pid"], record.get("start_ticks")):
            record.update(status="failed", error="Job process exited without a completion record", ended_at=now())
            atomic_json(directory / "job.json", record)
    if reconcile and record["status"] == "queued" and not record.get("pid"):
        age = (datetime.now(timezone.utc) - created_at).total_seconds()
        if age > 30:
            record.update(status="failed", error="Submission interrupted before worker launch", ended_at=now())
            atomic_json(directory / "job.json", record)
    log = contained(directory / "job.log", directory)
    if log.is_file():
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 24000))
            record["log_tail"] = handle.read().decode("utf-8", errors="replace").splitlines()[-100:]
    return record


def read_job(job_id, *, reconcile=True):
    if str(job_id).startswith("training:"):
        parts = job_id[len("training:"):].split("/")
        if len(parts) != 2:
            raise ValueError("Invalid training job reference")
        target = training.resolve_training_run_ref(WORKSPACE_DIR / "runs", *parts)
        status = training.training_status(target)
        return {**status, "job_id": job_id, "kind": "training", "created_at": status.get("started_at") or ""}
    directory = job_dir(job_id)
    if not (directory / "job.json").is_file():
        raise FileNotFoundError("Job not found")
    with exclusive_lock(directory / ".state.lock", blocking=True):
        return _read_job(job_id, reconcile=reconcile)


def write_job(job_id, record):
    directory = job_dir(job_id)
    with exclusive_lock(directory / ".state.lock", blocking=True):
        atomic_json(directory / "job.json", record)


def list_jobs(*, include_training=False):
    records = []
    if JOB_ROOT.is_dir():
        for directory in JOB_ROOT.iterdir():
            if not directory.is_dir():
                continue
            try:
                records.append(read_job(directory.name))
            except (OSError, ValueError):
                continue
    if include_training:
        for run in training.list_training_runs(WORKSPACE_DIR / "runs"):
            records.append({**run, "job_id": f"training:{run['run_group']}/{run['run_name']}",
                            "kind": "training", "created_at": run.get("started_at") or ""})
    return sorted(records, key=lambda item: item["created_at"], reverse=True)


def submit_job(kind, options, *, project_update=None):
    options = validate_options(kind, options)
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        with exclusive_lock(JOB_ROOT / ".submit.lock"):
            for existing in list_jobs():
                if existing["status"] not in TERMINAL and existing["kind"] == kind and existing["options"] == options:
                    raise ValueError(f"This job is already running: {existing['job_id']}")
            job_id = uuid.uuid4().hex
            directory = job_dir(job_id)
            directory.mkdir()
            record = {"job_id": job_id, "kind": kind, "options": options, "status": "queued", "created_at": now()}
            if project_update:
                record['project_update'] = project_update
            atomic_json(directory / "job.json", record)
            try:
                with (directory / "job.log").open("w", encoding="utf-8") as output:
                    kwargs = process_support.detached_options()
                    process = subprocess.Popen([sys.executable, "-m", "scripts.jobs", job_id], stdin=subprocess.DEVNULL,
                                               stdout=output, stderr=subprocess.STDOUT, **kwargs)
            except OSError as exc:
                record.update(status="failed", ended_at=now(), error=f"Could not launch job: {exc}")
                write_job(job_id, record)
                raise
            record.update(pid=process.pid, start_ticks=read_process_start_ticks(process.pid))
            atomic_json(directory / "launch.json", {"pid": record["pid"], "start_ticks": record["start_ticks"]})
            # The worker waits for launch.json, so submission cannot overwrite its result.
            return record
    except BlockingIOError:
        raise ValueError("Another job is being submitted; retry shortly") from None


def execute_job(kind, options, log=print):
    options = validate_options(kind, options)
    if kind in {"ais", "imagery"}:
        return getattr(acquisition, "acquire_" + kind)(**options, log=log)
    output = contained(options.get("output_dir") or WORKSPACE_DIR / "reports" / "operational" / uuid.uuid4().hex)
    with exclusive_lock(output / ".run.lock"):
        if any(path.name != ".run.lock" for path in output.iterdir()):
            raise ValueError("Detection output directory must be empty; choose a new directory")
        return _execute_pipeline(options, output, log)


def _execute_pipeline(options, output, log):
    results = pipeline.run_pipeline(options["images"], weights=options.get("weights"),
                ais_dir=options.get("ais_dir", WORKSPACE_DIR / "inputs" / "ais" / "full"), date=options.get("date"),
                conf=float(options.get("conf", .25)), radius_m=float(options.get("radius", 1000)),
                preview_dir=output / "previews", record_path=output / "run.json", warn=log)
    pipeline.write_pipeline_output(output / "dark_sts.json", results)
    pipeline.write_pipeline_output(output / "dark_sts.csv", results)
    return {"output_dir": str(output), "summary": pipeline.summarise_results(results)}


def worker(job_id):
    import time
    directory = job_dir(job_id)
    for _ in range(100):
        if (directory / "launch.json").is_file():
            break
        time.sleep(.05)
    record = read_job(job_id, reconcile=False)
    record.update(json.loads((directory / "launch.json").read_text()))
    record.update(status="running", started_at=now())
    write_job(job_id, record)
    try:
        record["result"] = execute_job(record["kind"], record["options"], log=lambda message: print(message, flush=True))
        if record.get('project_update') and record['result'].get('files'):
            from .operations import apply_project_update
            record['project_update_applied'] = apply_project_update(record['project_update'])
        record["status"] = "completed"
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        print(str(exc), flush=True)
    finally:
        record["ended_at"] = now()
        record.pop("log_tail", None)
        write_job(job_id, record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    worker(parser.parse_args().job_id)
