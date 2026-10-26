"""Exercise current research handoffs with explicitly synthetic images and no network.

This is a plumbing check, not a biological observation experiment. There is no
live-mode option and no credential loading. Model responses are local scripts.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from pathlib import Path
import socket

from PIL import Image, ImageDraw

from .config import Settings
from .contracts import ImageInput
from .inputs import load_sample
from .storage import file_hash, freeze_json, read_json, run_lock
from .workflow import _describe, _read_research


@contextmanager
def network_disabled():
    original = socket.socket.connect, socket.socket.connect_ex, socket.create_connection

    def forbidden(*args, **kwargs):
        raise RuntimeError("Offline observation verification forbids network connections")

    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = forbidden
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = original


async def verify(research_run: Path, handoff_dir: Path, output: Path) -> dict:
    research_run, output = research_run.resolve(), output.resolve()
    if research_run == output or research_run in output.parents or output in research_run.parents:
        raise ValueError("Offline output and research source must be separate directories")
    manifest = read_json(research_run / "manifest.json")
    report = {"status": "offline_pipeline_complete", "synthetic": True,
              "biological_observation_performed": False, "network_calls": 0,
              "paid_model_calls": 0, "cost_usd": 0, "samples": []}
    with network_disabled(), run_lock(output):
        for entry in manifest["samples"]:
            sample_id = entry["sample_id"]
            sample = load_sample(research_run / "inputs" / f"{sample_id}.json")
            stage = research_run / "samples" / sample_id / "research"
            research = _read_research(stage, sample)
            source_hash = file_hash(stage / "result.json")
            directory = output / "samples" / sample_id
            (directory / "images").mkdir(parents=True, exist_ok=True)
            sample.images = []
            sample.cp_file = None
            sample.cp_preprocessing = None
            for role in ("treatment", "control"):
                for channel in ("DNA", "AGP", "Mito"):
                    for field in ("1", "2"):
                        image_id = f"synthetic_{role}_{channel}_{field}"
                        path = directory / "images" / f"{image_id}.png"
                        image = Image.new("RGB", (48, 48), color="black")
                        ImageDraw.Draw(image).ellipse((12, 12, 30, 30), fill="gray")
                        image.save(path)
                        sample.images.append(ImageInput(image_id=image_id, path=path, role=role,
                            channel=channel, field=field, well_id=f"synthetic_{role}_well",
                            alignment_id=f"synthetic_{role}_{field}",
                            stain="SYNTHETIC geometric fixture; no biological staining.",
                            display_settings="Identical fictional display; not microscopy evidence."))
            freeze_json(directory / "synthetic_input.json", sample.model_dump(mode="json"))
            freeze_json(directory / "research_reference.json", {
                "directory": str(stage), "result_sha256": source_hash,
                "note": "Research read locally; source files are not modified or presented as a new retrieval."})
            result = await _describe(sample, research, Settings(), directory / "observation", "mock", None,
                                     handoff_dir / f"{sample_id}.json")
            if file_hash(stage / "result.json") != source_hash:
                raise ValueError("Research source changed during offline verification")
            report["samples"].append({"sample_id": sample_id, "hypotheses": len(result.hypothesis_assessments),
                "images": len(sample.images), "cp_available": False,
                "local_scripted_requests": len(list((directory / "observation/requests").glob("*.json"))),
                "description": result.description,
                "assessments": [row.assessment for row in result.hypothesis_assessments],
                "result": str(directory / "observation/result.json")})
        freeze_json(output / "summary.json", report)
        lines = ["# Observation Agent offline verification", "",
                 "SYNTHETIC PIPELINE CHECK. Images are geometric drawings. Local scripted model responses",
                 "perform no biological interpretation; all hypotheses remain uncertain and descriptions are empty.", "",
                 "Network connections are disabled. Paid model calls: 0. Cost: $0.", "",
                 "| Sample | Hypotheses | Synthetic images | Local scripted requests |",
                 "|---|---:|---:|---:|"]
        lines.extend(f"| {row['sample_id']} | {row['hypotheses']} | {row['images']} | {row['local_scripted_requests']} |"
                     for row in report["samples"])
        (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-run", type=Path, required=True)
    parser.add_argument("--handoff-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import json
    print(json.dumps(asyncio.run(verify(args.research_run, args.handoff_dir, args.output)), indent=2))


if __name__ == "__main__":
    main()
