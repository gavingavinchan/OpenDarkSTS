"""One CLI for acquisition, detection, and the existing training workflows."""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(prog="opendarksts")
    parser.add_argument("--workspace", help="Workspace directory (or OPENDARKSTS_WORKSPACE)")
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="Prepare a workspace, default model, and bundled example")
    initialize.add_argument("--weights", help="Local copy of the documented default checkpoint")
    initialize.add_argument("--no-model", action="store_true", help="Prepare workspace and example without weights")
    initialize.add_argument("--no-example", action="store_true")
    serve = commands.add_parser("serve", help="Start the local GUI in the background")
    serve.add_argument("--port", type=int, default=8766, help="Local port (default: 8766)")
    commands.add_parser("stop", help="Stop the recorded GUI server for this workspace; background jobs continue")
    acquire = commands.add_parser("acquire", help="Download US AIS records or Sentinel-2 imagery").add_subparsers(dest="kind", required=True)
    for kind in ["ais", "imagery"]:
        command = acquire.add_parser(kind)
        command.add_argument("--start", required=True)
        command.add_argument("--end")
        command.add_argument("--bbox", nargs=4, type=float, required=kind == "imagery")
        command.add_argument("--output-dir")
        command.add_argument("--background", action="store_true")
        if kind == "imagery":
            command.add_argument("--max-cloud", type=float, default=35)
    run = commands.add_parser("run", help="Detect candidate pairs and compare same-day AIS")
    run.add_argument("--images", nargs="+", required=True)
    run.add_argument("--ais-dir", help="AIS folder or a single daily CSV file")
    run.add_argument("--date")
    run.add_argument("--weights")
    run.add_argument("--conf", type=float, default=.25)
    run.add_argument("--radius", type=float, default=1000)
    run.add_argument("--output-dir")
    run.add_argument("--background", action="store_true")
    jobs = commands.add_parser("jobs", help="Inspect background work or cancel a download")
    jobs.add_argument("job_id", nargs="?")
    jobs.add_argument("--cancel", action="store_true", help="Cancel a download and remove its partial files")
    project = commands.add_parser("project", help="Inspect or save the Detect settings")
    project.add_argument("--set", dest="settings", help="JSON object of project settings")
    project.add_argument("--start-job", choices=["ais", "imagery", "run"])
    project.add_argument("--check-ready", action="store_true", help="Check detection inputs for the saved dates")
    results = commands.add_parser("results", help="Review and export detection results")
    results.add_argument("directory", nargs="?")
    results.add_argument("--detection")
    results.add_argument("--verdict", choices=["real_sts", "not_sts", "unclear"])
    results.add_argument("--reviewer", default="default")
    results.add_argument("--training-label", choices=["ship_non_sts", "neither"])
    results.add_argument("--send-to-improve", action="store_true")
    results.add_argument("--format", choices=["csv", "json", "geojson"])
    results.add_argument("--output")
    training = commands.add_parser("training", help="Prepare datasets, train, and evaluate models", add_help=False)
    training.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def execute(args):
    if args.command == "init":
        from .initialization import initialize
        return initialize(weights=args.weights, no_model=args.no_model, example=not args.no_example)
    if args.command == "serve":
        from .service import serve
        return serve(args.port)
    if args.command == "stop":
        from .service import stop
        return stop()
    from . import jobs, operations
    if args.command == "results":
        from . import results
        if not args.directory:
            return {"runs": results.list_runs()}
        if args.verdict:
            results.review_detection(args.directory, args.detection, args.verdict, reviewer=args.reviewer, training_label=args.training_label)
        if args.send_to_improve:
            return results.send_to_improve(args.directory, args.detection)
        if args.format:
            if not args.output:
                raise ValueError("An export requires --output")
            target = jobs.contained(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(results.export_results(args.directory, args.format))
            return {"output": str(target)}
        return {"detections": results.read_results(args.directory)}
    if args.command == "project":
        if args.settings:
            operations.save_project(json.loads(args.settings))
        if args.check_ready:
            return operations.detection_readiness()
        return operations.submit_project_job(args.start_job) if args.start_job else operations.project()
    log = lambda message: print(message, file=sys.stderr, flush=True)
    if args.command == "jobs":
        if args.cancel:
            if not args.job_id:
                raise ValueError("Select a job id to cancel")
            return jobs.cancel_job(args.job_id)
        return jobs.read_job(args.job_id) if args.job_id else {"jobs": jobs.list_jobs(include_training=True)}
    kind = args.kind if args.command == "acquire" else args.command
    options = {key: value for key, value in vars(args).items()
               if key not in {"command", "kind", "workspace", "background"} and value is not None}
    return jobs.submit_job(kind, options) if args.background else jobs.run_foreground(kind, options, log=log)



def main(argv=None):
    try:
        parser = build_parser()
        args, forwarded = parser.parse_known_args(argv)
        if forwarded and args.command != "training":
            parser.error("unrecognized arguments: " + " ".join(forwarded))
    except SystemExit as exc:
        code = int(exc.code or 0)
        if code:
            print(json.dumps({"ok": False, "command": "", "result": {}, "errors": ["Invalid command-line arguments"]}))
        return code
    if args.workspace:
        os.environ["OPENDARKSTS_WORKSPACE"] = str(Path(args.workspace).expanduser().resolve())
    if args.command == "training":
        from .training_pipeline_cli import main as training_main
        return training_main(forwarded + args.arguments)
    try:
        with redirect_stdout(sys.stderr):
            result = execute(args)
        payload, code = {"ok": True, "command": args.command, "result": result, "errors": []}, 0
    except Exception as exc:
        payload = {"ok": False, "command": args.command, "result": {}, "errors": [str(exc)]}
        code = 2 if isinstance(exc, ValueError) else 1
    print(json.dumps(payload))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
