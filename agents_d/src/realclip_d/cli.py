"""Single-sample workflow and research-only batches; mock is the default."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from .config import load_settings
from .inputs import image_manifest, load_sample


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="realclip-d", description="D independent research → observation workflow")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("validate", "research", "describe", "run"):
        command = sub.add_parser(name)
        command.add_argument("--input", required=True, type=Path, help="Explicit SampleInput JSON; relative files resolve beside it")
        command.add_argument("--config", type=Path)
        if name != "validate":
            command.add_argument("--output", required=True, type=Path)
            command.add_argument("--mode", choices=("mock", "replay", "live"), default="mock")
            command.add_argument("--budget-usd", type=float)
            command.add_argument("--replay-from", type=Path)
            if name in {"describe", "run"}:
                command.add_argument("--handoff", type=Path, help="Reviewed, research-hash-bound observation handoff revision JSON")
            if name == "describe":
                command.add_argument("--research", type=Path)
    batch = sub.add_parser("research-batch", help="Run complete research agents concurrently across compounds")
    batch.add_argument("--inputs", nargs="+", required=True, type=Path)
    batch.add_argument("--output", required=True, type=Path)
    batch.add_argument("--config", type=Path)
    batch.add_argument("--mode", choices=("mock", "live"), default="mock")
    batch.add_argument("--workers", type=int, default=10)
    batch.add_argument("--budget-usd", type=float)
    batch.add_argument("--credential-file", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "research-batch":
            from .research_batch import batch_settings, run_batch

            settings = load_settings(args.config) if args.config else batch_settings()
            if args.budget_usd is not None:
                settings = type(settings).model_validate({**settings.model_dump(), "budget_usd": args.budget_usd})
            result = run_batch(args.inputs, args.output, settings, workers=args.workers,
                               mode=args.mode, credential_file=args.credential_file)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if all(row["status"] == "complete" for row in result["samples"]) else 2
        settings = load_settings(args.config)
        if args.command == "validate":
            from .tools import CPTable

            sample = load_sample(args.input)
            manifest = image_manifest(sample)
            cp = CPTable(sample.cp_file, preprocessing=sample.cp_preprocessing, metadata=sample.cp_metadata)
            result = {"status": "valid", "sample_id": sample.sample_id, "images": len(manifest),
                      "cp_rows": len(cp.rows), "cp_available": sample.cp_file is not None,
                      "model": settings.model, "network_calls": 0, "model_calls": 0}
        else:
            from .workflow import execute

            if args.budget_usd is not None:
                settings = type(settings).model_validate({**settings.model_dump(), "budget_usd": args.budget_usd})
            result = asyncio.run(execute(args.command, args.input, args.output, settings, args.mode,
                                         getattr(args, "research", None), args.replay_from,
                                         getattr(args, "handoff", None)))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"realclip-d: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
