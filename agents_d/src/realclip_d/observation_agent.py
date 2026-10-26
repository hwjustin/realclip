"""Observation agent orchestration: reviewed observation, then plain Summarize.

The four image/evidence modules retain their existing contracts. Their sealed
report is an internal artifact; the agent's public description is the final
plain-text Summarize output. Both stages share one run budget ledger.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from .config import Settings
from .contracts import Contract, NonEmpty, ResearchOutput, SampleInput
from .inputs import image_manifest
from .observation_protocol import HandoffRevision, prepare_observation
from .observation_stages import ObservationPipelineResult
from .storage import digest, file_hash, freeze_json, read_json, seal_stage, verify_stage
from .summarize import prompt_path, run_summarize
from .summarize_contracts import SummarizeConstraints
from .summarize_inputs import build_request, make_provisional_manifest


MODULES = ("baseline", "guided", "writer", "reviewer", "summarize")
_ARTIFACTS = {"reviewed": "reviewed/result.json", "summarize": "summarize/result.json"}
_MOCK_SUMMARY = ("SYNTHETIC MOCK OUTPUT: no biological description was generated; "
                 "this is a software integration fixture.")


class ObservationAgentResult(Contract):
    schema_version: Literal["5"] = "5"
    sample_id: NonEmpty
    description: NonEmpty
    full_description: str
    modules: list[NonEmpty]
    module_artifacts: dict[str, str]
    limitations: list[NonEmpty]

    @model_validator(mode="after")
    def fixed_module_boundary(self):
        if self.modules != list(MODULES) or self.module_artifacts != _ARTIFACTS:
            raise ValueError("Observation agent modules and relative artifact paths must match the pipeline")
        return self


def preflight_observation_agent(sample: SampleInput, settings: Settings) -> list[str]:
    """Validate image bytes/metadata and select available requested channels.

    The configured channels are a preference set: missing channels are omitted,
    supporting wells with a smaller supplied fluorescence collection. At least
    one original treatment channel must remain. No channel or well is invented.
    """
    if settings.observation_max_turns < len(MODULES):
        raise ValueError("Observation requires at least five total model turns including Summarize")
    images = image_manifest(sample, include_metadata=True)
    treatment = [row for row in images if row["role"] == "treatment"]
    wells = {row.get("well_id") for row in treatment}
    if (not treatment or len(wells) != 1
            or any(not isinstance(well, str) or not well.strip() for well in wells)):
        raise ValueError("Observation must identify exactly one explicit treatment well")
    available = {row["channel"] for row in treatment if row.get("parent_image_id") is None}
    channels = [channel for channel in settings.observation_summary_channels if channel in available]
    if not channels:
        raise ValueError("No configured Summarize channels are available in original treatment views")
    return channels


def _source_binding(directory: Path, manifest: dict, request: dict) -> dict:
    return {
        "observation_result_sha256": file_hash(directory / "result.json"),
        "observation_receipt_sha256": file_hash(directory / "complete.json"),
        "input_sha256": digest(request), "manifest_sha256": digest(manifest),
        "prompt_sha256": file_hash(prompt_path()), "upstream_read_only": True,
    }


def _read_summary(reviewed: Path, summary: Path, *, manifest: dict | None = None,
                  settings: Settings | None = None, mode: str | None = None,
                  check_current_prompt: bool = False) -> dict:
    verify_stage(summary)
    saved_manifest = read_json(summary / "visual_manifest.json")
    if manifest is not None and saved_manifest != manifest:
        raise ValueError("Saved Summarize visual inputs differ; use a new output directory")
    saved_request = read_json(summary / "input.json")
    request = build_request(reviewed, saved_manifest, SummarizeConstraints(),
                            request_version=saved_request.get("request_version", "1"))
    if saved_request != request:
        raise ValueError("Saved Summarize request differs from the reviewed source")
    binding = read_json(summary / "binding.json")
    expected = _source_binding(reviewed, saved_manifest, request)
    # A completed receipt is replayable after copying its directory. Its saved
    # absolute source path is provenance only, not a location-dependent identity.
    if not check_current_prompt:
        expected["prompt_sha256"] = file_hash(summary / "instructions.md")
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved Summarize source, configuration or prompt binding differs")
    if settings is not None:
        expected_settings = {**settings.model_dump(mode="json"), "mode": mode}
        if read_json(summary / "settings.json") != expected_settings:
            raise ValueError("Saved Summarize settings differ; use a new output directory")
    result = read_json(summary / "result.json")
    if (result.get("schema_version") != "2" or result.get("sample_id") != request["target_treatment_well"]["sample_id"]
            or not isinstance(result.get("description"), str) or not result["description"].strip()
            or (mode is not None and result.get("mode") != mode)):
        raise ValueError("Saved Summarize result belongs to a different source or output contract")
    if (summary / "description.txt").read_text(encoding="utf-8") != result["description"] + "\n":
        raise ValueError("Saved Summarize plain description differs from its result")
    return result


def read_observation_agent(directory: Path) -> ObservationAgentResult:
    """Read and validate an entire sealed agent result without calling a model."""
    directory = Path(directory)
    verify_stage(directory)
    reviewed, summary = directory / "reviewed", directory / "summarize"
    verify_stage(reviewed)
    verify_stage(summary)
    binding = read_json(directory / "binding.json")
    for name in ("reviewed", "summarize"):
        if binding.get("module_receipt_sha256", {}).get(name) != file_hash(directory / name / "complete.json"):
            raise ValueError(f"Observation module completion receipt changed: {name}")
    initial_binding = read_json(directory / "run_binding.json")
    if initial_binding != {key: value for key, value in binding.items() if key != "module_receipt_sha256"}:
        raise ValueError("Observation final binding differs from its original run binding")
    saved_settings = read_json(directory / "settings.json")
    reviewed_settings = {**saved_settings,
        "observation_max_turns": saved_settings["observation_max_turns"] - 1}
    if read_json(reviewed / "settings.json") != reviewed_settings:
        raise ValueError("Reviewed module settings differ from the agent turn allocation")
    summary_settings = read_json(summary / "settings.json")
    if ({key: value for key, value in summary_settings.items() if key not in {"budget_usd", "mode"}} != saved_settings
            or summary_settings.get("mode") != binding.get("mode")):
        raise ValueError("Summarize module settings differ from the agent settings")
    if (binding.get("settings_sha256") != digest(saved_settings)
            or binding.get("input_sha256") != digest(read_json(directory / "input.json"))
            or binding.get("handoff_sha256") != digest(read_json(directory / "handoff.json"))):
        raise ValueError("Observation saved configuration or inputs differ from their run binding")
    reviewed_binding = read_json(reviewed / "binding.json")
    if (any(reviewed_binding.get(key) != binding.get(key)
            for key in ("mode", "input_sha256", "prompts", "cp_sha256", "research_sha256", "handoff_sha256"))
            or reviewed_binding.get("settings_sha256") != digest(reviewed_settings)):
        raise ValueError("Reviewed module binding differs from the agent source or settings")
    full = ObservationPipelineResult.model_validate(read_json(reviewed / "result.json"))
    short = _read_summary(reviewed, summary, mode=binding["mode"])
    result = ObservationAgentResult.model_validate(read_json(directory / "result.json"))
    if (result.sample_id != full.sample_id or result.sample_id != short["sample_id"]
            or result.full_description != full.description or result.description != short["description"]
            or result.limitations != full.limitations):
        raise ValueError("Observation agent result differs from its module outputs")
    if read_json(directory / "input.json") != read_json(reviewed / "input.json"):
        raise ValueError("Observation agent input differs from reviewed module input")
    if read_json(directory / "handoff.json") != read_json(reviewed / "handoff.json"):
        raise ValueError("Observation agent handoff differs from reviewed module handoff")
    if (directory / "description.txt").read_text(encoding="utf-8") != result.description + "\n":
        raise ValueError("Observation agent plain description differs from its result")
    return result


def _aggregate_artifacts(directory: Path, settings: Settings, mode: str) -> None:
    """Keep convenient agent-wide usage and raw receipts without hiding modules."""
    calls = []
    for name in ("reviewed", "summarize"):
        source = directory / name
        calls_path = source / "calls.jsonl"
        module_calls = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
        for row in module_calls:
            calls.append({"module": row.get("phase", "summarize" if name == "summarize" else "reviewed"), **row})
        for kind in ("requests", "responses", "input_token_counts"):
            for path in sorted((source / kind).glob("*.json")):
                prefix = "summarize_" if name == "summarize" else ""
                freeze_json(directory / kind / (prefix + path.name), read_json(path))
    (directory / "calls.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in calls), encoding="utf-8")
    freeze_json(directory / "usage.json", {"requests": len(calls), **{
        key: sum((row.get("usage") or {}).get(key, 0) for row in calls)
        for key in ("input_tokens", "output_tokens", "total_tokens")}})
    reviewed_pipeline = read_json(directory / "reviewed/pipeline.json")
    phases = [*reviewed_pipeline["phases"], {
        "phase": "summarize", "status": "complete", "max_turns": 1,
        "ordinary_max_turns": 1, "correction_max_turns": 0, "corrections": [],
        "research_guidance_visible": False, "images_visible": False,
    }]
    freeze_json(directory / "pipeline.json", {
        "schema_version": "2", "mode": mode, "modules": list(MODULES),
        "phases": phases, "model_turns": len(calls),
        "max_model_turns": settings.observation_max_turns,
        "reserved_summarize_turns": 1,
        "tool_calls": reviewed_pipeline["tool_calls"],
        "max_tool_calls": settings.max_tool_calls,
        "crop_count": reviewed_pipeline["crop_count"],
        "biological_accuracy_validated": False,
    })


async def run_observation_agent(sample: SampleInput, research: ResearchOutput,
                                settings: Settings, directory: Path, mode: str,
                                budget, handoff_path: Path | None = None) -> ObservationAgentResult:
    """Run all five modules; a failed summary never reruns sealed image work."""
    # Local imports avoid cycling with workflow's public orchestration entry.
    from .workflow import _binding, _describe, _settings

    directory = Path(directory)
    # Validate the complete outer receipt before considering any child work.
    # Deleted child artifacts must never look like permission to regenerate a
    # module inside an otherwise completed paid run.
    if (directory / "complete.json").exists():
        read_observation_agent(directory)
    channels = preflight_observation_agent(sample, settings)
    reviewed, summary = directory / "reviewed", directory / "summarize"
    saved_handoff = reviewed / "handoff.json"
    revision = (HandoffRevision.model_validate(read_json(saved_handoff)["revision"])
                if saved_handoff.exists() and handoff_path is None else None)
    view, audit = prepare_observation(sample, research, handoff_path, revision)
    payload = view.model_dump(mode="json")
    binding = _binding(mode, payload, settings)
    binding.update(research_sha256=digest(research.model_dump(mode="json")),
                   cp_sha256=file_hash(sample.cp_file) if sample.cp_file else None,
                   handoff_sha256=digest(audit), summary_channels=channels)
    # Freeze an initial binding separately: final binding also seals the nested
    # completion receipts, whose hashes do not exist before the modules finish.
    freeze_json(directory / "run_binding.json", binding)
    freeze_json(directory / "input.json", payload)
    freeze_json(directory / "settings.json", _settings(settings))
    freeze_json(directory / "handoff.json", audit)
    freeze_json(directory / "sources.json", [])
    core_settings = settings.model_copy(update={"observation_max_turns": settings.observation_max_turns - 1})
    full = await _describe(sample, research, core_settings, reviewed, mode, budget, handoff_path)
    manifest = make_provisional_manifest(reviewed, channels)
    if (summary / "complete.json").exists():
        short = _read_summary(reviewed, summary, manifest=manifest, settings=settings,
                              mode=mode, check_current_prompt=True)
    else:
        if mode == "live" and (any((summary / "requests").glob("*.json"))
                               or (summary / "calls.jsonl").exists()
                               or (summary / "result.json").exists()):
            raise ValueError("Incomplete live Summarize already has model attempts; saved receipts are preserved and will not be resubmitted")
        short = await run_summarize(reviewed, manifest, summary, settings,
            mode=mode, budget=budget, fixture=_MOCK_SUMMARY if mode == "mock" else None)
    if (directory / "complete.json").exists():
        return read_observation_agent(directory)
    result = ObservationAgentResult(sample_id=sample.sample_id,
        description=short["description"], full_description=full.description,
        modules=list(MODULES), module_artifacts=dict(_ARTIFACTS), limitations=full.limitations)
    freeze_json(directory / "result.json", result.model_dump(mode="json"))
    (directory / "description.txt").write_text(result.description + "\n", encoding="utf-8")
    _aggregate_artifacts(directory, settings, mode)
    freeze_json(directory / "binding.json", {**binding, "module_receipt_sha256": {
        name: file_hash(directory / name / "complete.json") for name in ("reviewed", "summarize")}})
    freeze_json(directory / "validation.json", {
        "reviewed_stage_validated": True, "summary_source_binding_valid": True,
        "summary_citation_validation": False, "summary_corrective_calls": False,
        "biological_accuracy_validated": False})
    seal_stage(directory)
    return read_observation_agent(directory)
