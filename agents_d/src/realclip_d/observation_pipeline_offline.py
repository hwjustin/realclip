"""Verify four Observation phases and native crops using local scripted models.

Real prepared images exercise transport and provenance, not biological image
interpretation. There is no live option or credential loading. Network
connections are blocked throughout execution and post-run verification.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from .config import Settings
from .inputs import load_sample
from .observation_cp_offline import verify as verify_cp_inputs
from .observation_offline import network_disabled
from .observation_pipeline import PHASES, load_inspected_images
from .storage import file_hash, read_json, run_lock, verify_stage, write_json


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _nodes(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nodes(child)


def _phase_input(directory: Path) -> tuple[dict, list[dict]]:
    messages = read_json(directory / "request_input.json")
    _require(len(messages) == 1 and messages[0]["role"] == "user", "Unexpected phase input envelope")
    content = messages[0]["content"]
    _require(content[0]["type"] == "input_text", "Missing phase context")
    return json.loads(content[0]["text"]), content


def _image_hashes(value: Any) -> set[str]:
    return {row["image_url"]["image_sha256"] for row in _nodes(value)
            if row.get("type") == "input_image" and isinstance(row.get("image_url"), dict)}


def _verify_native_crops(directory: Path, sample, originals: list[dict], settings: Settings) -> dict:
    crops = load_inspected_images(directory, originals)
    manifest = read_json(directory / "crops" / "manifest.json")
    _require(manifest["limits"] == {"max_crops": settings.observation_max_crops,
                                  "max_pixels_per_crop": settings.observation_crop_max_pixels},
             "Saved crop limits differ from settings")
    _require(0 < len(crops) <= settings.observation_max_crops,
             "The default offline script must exercise at least one bounded native crop")
    source_paths = {row.image_id: row.path for row in sample.images}
    transport_hashes = set()
    for path in sorted((directory / "phases").glob("*/requests/*.json")):
        transport_hashes.update(_image_hashes(read_json(path)))
    for row in crops:
        x, y, width, height = row["crop_box"]
        _require(width * height <= settings.observation_crop_max_pixels, "Crop exceeds pixel limit")
        _require(row["pixel_transform"] == "native_pixel_crop_only", "Unexpected crop transform")
        crop_path = directory / "crops" / (row["image_id"] + ".png")
        with Image.open(source_paths[row["parent_image_id"]]) as source, Image.open(crop_path) as crop:
            source.load()
            crop.load()
            expected = source.crop((x, y, x + width, y + height))
            _require(crop.mode == expected.mode and crop.size == expected.size
                     and crop.tobytes() == expected.tobytes(), "Generated crop pixels differ from source")
        data_url = "data:image/png;base64," + base64.b64encode(crop_path.read_bytes()).decode()
        _require(hashlib.sha256(data_url.encode()).hexdigest() in transport_hashes,
                 "Native crop image was not present in an actual scripted model request")
    _require(any(row["status"] == "created" for row in manifest["requests"]),
             "No successful crop tool request was recorded")
    _require(all(row["question"].strip() and row["selection_reason"].strip()
                 for row in manifest["requests"]), "Missing crop question or selection reason")
    return {"crop_count": len(crops), "native_pixels_match_source": True,
            "actual_crop_image_transport_verified": True, "crop_bounds_and_limits_valid": True}


def _verify_case(directory: Path) -> dict:
    verify_stage(directory)
    view = read_json(directory / "input.json")
    sample = load_sample(directory.parent / "sample.json")
    settings = Settings.model_validate(read_json(directory / "settings.json"))
    pipeline = read_json(directory / "pipeline.json")
    _require(pipeline["mode"] == "mock", "Offline runner produced a live stage")
    _require([row["phase"] for row in pipeline["phases"]] == list(PHASES), "Wrong phase order")
    _require(all(row["status"] == "validated" for row in pipeline["phases"]), "A phase was not validated")
    _require(pipeline["max_model_turns"] == settings.observation_max_turns,
             "Saved model-turn bound differs from settings")
    calls = _jsonl(directory / "calls.jsonl")
    _require(len(calls) == pipeline["model_turns"] <= settings.observation_max_turns,
             "Shared model-turn bound failed")
    _require(all(row["mode"] == "mock" and row["status"] == "complete"
                 and row["request_id"] is None for row in calls), "Unexpected paid or failed model call")
    _require(pipeline["tool_calls"] <= settings.max_tool_calls, "Shared tool-call bound failed")
    expected_tools = {"baseline": {"inspect_image"}, "guided": {"inspect_image", "query_cp"},
                      "writer": set(), "reviewer": {"inspect_image"}}
    payloads, contents, outputs = {}, {}, {}
    for plan in pipeline["phases"]:
        name = plan["phase"]
        phase = directory / "phases" / name
        payloads[name], contents[name] = _phase_input(phase)
        outputs[name] = read_json(phase / "output.json")
        requests = sorted((phase / "requests").glob("*.json"))
        local_calls = _jsonl(phase / "calls.jsonl")
        _require(1 <= len(requests) == len(local_calls) <= plan["max_turns"],
                 "Phase calls exceed its allocated turn bound")
        _require(plan["research_guidance_visible"] == (name == "guided")
                 and plan["images_visible"] == (name != "writer"), "Incorrect phase visibility record")
        for path in requests:
            request = read_json(path)
            _require({row["name"] for row in request["tools"]} <= expected_tools[name],
                     "A tool escaped its phase boundary")
            _require(not request["previous_response_id"] and not request["conversation_id"],
                     "Phase used a stored conversation continuation")
    _require(set(payloads["baseline"]) == {"sample_id", "context", "images", "cp_available"},
             "Research guidance or CP catalogue entered baseline")
    _require(payloads["baseline"]["images"] == view["images"], "Baseline did not receive all originals")
    _require(payloads["guided"]["hypotheses"] == view["hypotheses"]
             and payloads["guided"]["cp_catalogue"] == view["cp_catalogue"]
             and payloads["guided"]["baseline"] == outputs["baseline"], "Guided handoff differs")
    baseline_observations = outputs["baseline"]["observations"]
    _require(outputs["guided"]["observations"][:len(baseline_observations)] == baseline_observations,
             "Guided modified the baseline prefix")
    writing_evidence = {key: value for key, value in outputs["guided"].items()
                        if key != "hypothesis_assessments"}
    _require(payloads["writer"] == {"evidence": writing_evidence},
             "Writer received content other than observation evidence")
    _require(set(payloads["reviewer"]) == {"evidence", "draft", "images"}
             and payloads["reviewer"]["evidence"] == writing_evidence
             and payloads["reviewer"]["draft"] == outputs["writer"]
             and payloads["reviewer"]["images"] == view["images"], "Reviewer handoff differs")
    originals = {row["image_id"] for row in view["images"]}
    reviewed = outputs["baseline"]["reviewed_image_ids"]
    _require(len(reviewed) == len(originals) and set(reviewed) == originals,
             "Baseline original-image coverage is incomplete")
    exposure = read_json(directory / "image_exposure.json")
    _require(exposure["writer"] == [] and not _image_hashes(contents["writer"]),
             "Images entered the text-only writer")
    for name in ("baseline", "guided", "reviewer"):
        _require(originals <= set(exposure[name]), "A visual phase omitted supplied original images")
        metadata_ids = {json.loads(item["text"])["image_id"] for item in contents[name][1:]
                        if item.get("type") == "input_text"}
        _require(originals <= metadata_ids, "Original metadata was omitted from a visual request")
        _require(sum(item.get("type") == "input_image" for item in contents[name]) >= len(originals),
                 "Original images were omitted from a visual request")
    result = read_json(directory / "result.json")
    _require(result["description"] == "" and not result["description_sentences"]
             and not result["observations"], "Offline script produced a biological appearance claim")
    _require(all(row["assessment"] == "uncertain" for row in result["hypothesis_assessments"]),
             "Offline script produced a biological hypothesis judgment")
    crop_check = _verify_native_crops(directory, sample, view["images"], settings)
    _require(pipeline["crop_count"] == crop_check["crop_count"], "Saved crop count differs")
    return {"sample_id": view["sample_id"], "phases": list(PHASES),
            "local_scripted_model_calls": len(calls), "model_turn_limit": settings.observation_max_turns,
            "tool_calls": pipeline["tool_calls"], "phase_context_isolation_valid": True,
            "all_original_images_supplied_to_visual_phases": True,
            "baseline_coverage_declaration_valid": True, "baseline_prefix_preserved": True,
            "sealed_stage_valid": True, "description": "", **crop_check}


async def verify(prepared: Path, output: Path) -> dict:
    prepared, output = prepared.resolve(), output.resolve()
    _require(prepared != output and prepared not in output.parents and output not in prepared.parents,
             "Offline output must be separate from prepared input data")
    pairs = sorted(prepared.glob("*/pairing.json"))
    _require(bool(pairs), "No prepared treatment/control cases found")
    source_hashes: dict[Path, str] = {}
    research_stages: list[Path] = []
    for path in pairs:
        pair = read_json(path)
        research = Path(pair["research_stage"])
        verify_stage(research)
        research_stages.append(research)
        for source in (path, path.parent / "sample.json", Path(pair["research_input"]),
                       research / "result.json", research / "complete.json"):
            source_hashes[source] = file_hash(source)
    with network_disabled():
        cp_report = await verify_cp_inputs(prepared, output)
        with run_lock(output):
            cases = [_verify_case(output / row["sample_id"] / "observation") for row in cp_report["samples"]]
            for path, before in source_hashes.items():
                _require(file_hash(path) == before, "A prepared or research source artifact changed")
            for stage in research_stages:
                verify_stage(stage)
            report = {"status": "offline_pipeline_complete", "network_calls": 0,
                      "paid_model_calls": 0, "cost_usd": 0, "biological_observation_performed": False,
                      "model": "local scripted SDK model; no biological image interpretation",
                      "source_research_and_prepared_inputs_unchanged": True,
                      "samples": cases, "cp_input_checks": cp_report["samples"]}
            write_json(output / "cp_input_checks.json", cp_report)
            write_json(output / "summary.json", report)
            lines = ["# Observation pipeline: offline verification", "",
                     "All four phases ran with local scripted models against the prepared real well pairs.",
                     "These scripts perform no biological image interpretation: descriptions and observations remain empty,",
                     "and all research hypotheses remain uncertain. Coverage checks are software checks, not biological labels.", "",
                     "Network connections were blocked. Paid model calls: 0. Cost: $0. Source research and prepared inputs are unchanged.", "",
                     "| Sample | Phases | Scripted calls | Native crops |",
                     "|---|---:|---:|---:|"]
            lines.extend(f"| {row['sample_id']} | 4 | {row['local_scripted_model_calls']} | {row['crop_count']} |"
                         for row in cases)
            lines += ["", "Verified phase order and tool boundaries, separate input contexts, all-original visual input coverage,",
                      "unchanged baseline handoff, total turn/tool/crop limits, CP input checks, and completion seals.",
                      "Every generated crop matches its original decoded pixels and appeared as an actual image in a recorded SDK request.",
                      "The text-only writer receives no images or hypothesis assessments; the reviewer receives no research checklist.",
                      "No scientific description-quality improvement is asserted by this verifier."]
            (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(asyncio.run(verify(args.prepared, args.output)), indent=2))


if __name__ == "__main__":
    main()
