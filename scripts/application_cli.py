"""One CLI for acquisition, detection, and the existing training workflows."""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(prog="opendarksts")
    parser.add_argument("--workspace", help="Workspace directory (or OPENDARKSTS_WORKSPACE)")
    commands = parser.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser("acquire").add_subparsers(dest="kind", required=True)
    for kind in ["ais", "imagery"]:
        command = acquire.add_parser(kind)
        command.add_argument("--start", required=True)
        command.add_argument("--end")
        command.add_argument("--bbox", nargs=4, type=float, required=kind == "imagery")
        command.add_argument("--output-dir")
        if kind == "imagery":
            command.add_argument("--max-cloud", type=float, default=30)
    run = commands.add_parser("run")
    run.add_argument("--images", nargs="+", required=True)
    run.add_argument("--ais-dir")
    run.add_argument("--date")
    run.add_argument("--weights")
    run.add_argument("--conf", type=float, default=.25)
    run.add_argument("--radius", type=float, default=1000)
    run.add_argument("--output-dir")
    training = commands.add_parser("training", add_help=False)
    training.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def execute(args):
    from . import acquisition, pipeline
    from .workspace import WORKSPACE_DIR, contained
    log = lambda message: print(message, file=sys.stderr, flush=True)
    if args.command == "acquire":
        options = {"output_dir": args.output_dir, "bbox": args.bbox, "log": log}
        if args.kind == "imagery":
            options["max_cloud"] = args.max_cloud
        return getattr(acquisition, "acquire_" + args.kind)(args.start, args.end, **options)
    if args.command == "run":
        output = contained(args.output_dir or WORKSPACE_DIR / "reports" / "operational" / uuid.uuid4().hex)
        results = pipeline.run_pipeline(args.images, weights=args.weights,
                    ais_dir=args.ais_dir or WORKSPACE_DIR / "inputs" / "ais" / "full",
                    date=args.date, conf=args.conf, radius_m=args.radius,
                    preview_dir=output / "previews", record_path=output / "run.json", warn=log)
        pipeline.write_pipeline_output(output / "dark_sts.json", results)
        pipeline.write_pipeline_output(output / "dark_sts.csv", results)
        return {"output_dir": str(output), "summary": pipeline.summarise_results(results)}
    raise ValueError("Unknown command")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.workspace:
        os.environ["OPENDARKSTS_WORKSPACE"] = str(Path(args.workspace).expanduser().resolve())
    if args.command == "training":
        from .training_pipeline_cli import main as training_main
        return training_main(args.arguments)
    try:
        result = execute(args)
        payload, code = {"ok": True, "command": args.command, "result": result, "errors": []}, 0
    except Exception as exc:
        payload = {"ok": False, "command": args.command, "result": {}, "errors": [str(exc)]}
        code = 2 if isinstance(exc, ValueError) else 1
    print(json.dumps(payload))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
