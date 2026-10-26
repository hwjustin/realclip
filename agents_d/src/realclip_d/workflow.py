"""Application-owned, single-direction orchestration with independent stages."""
from __future__ import annotations

import importlib.metadata
import os
import shutil
from pathlib import Path
from typing import Literal

from .config import Settings
from .contracts import ResearchResultV4
from .contracts import (ResearchOutput, SampleInput, Source,
                        parse_research_result, validate_research_references)
from .inputs import load_sample, research_input
from .observation_contracts import ObservationResult, validate_observation_references
from .observation_stages import ObservationPipelineResult, validate_pipeline_references
from .observation_pipeline import cp_query_results, load_inspected_images
from .observation_agent import (
    preflight_observation_agent, read_observation_agent, run_observation_agent,
)
from .observation_protocol import HandoffRevision, prepare_observation, check_observation_text
from .research_protocol import research_receipts, validate_research_protocol
from .storage import (digest, file_hash, freeze_json, read_json, run_lock,
                      seal_stage, verify_stage, write_json)

Mode = Literal["mock", "replay", "live"]


def _prompt_hashes() -> dict[str, str]:
    directory = Path(__file__).parent / "prompts"
    return {path.name: file_hash(path) for path in sorted(directory.glob("*.md"))}


def _settings(settings: Settings) -> dict:
    return settings.model_dump(mode="json", exclude={"budget_usd"})


def _read_research(directory: Path, sample: SampleInput) -> ResearchOutput:
    verify_stage(directory)
    if read_json(directory / "input.json") != research_input(sample):
        raise ValueError("Saved research belongs to different compound, MOA or context")
    result = parse_research_result(read_json(directory / "result.json"))
    sources = [Source.model_validate(row) for row in read_json(directory / "sources.json")]
    if not isinstance(result, ResearchResultV4):
        validate_research_references(result, sources)
        validate_research_protocol(result, sample, research_receipts(directory))
    if result.sample_id != sample.sample_id:
        raise ValueError("Saved research belongs to another sample")
    return result


def _binding(mode: str, payload: dict, settings: Settings) -> dict:
    return {"mode": mode, "input_sha256": digest(payload),
            "settings_sha256": digest(_settings(settings)), "prompts": _prompt_hashes()}


async def _research(sample: SampleInput, settings: Settings, directory: Path,
                    mode: str, budget, *, pubchem_limiter=None) -> ResearchOutput:
    from .agent_runtime import run_research

    payload = research_input(sample)
    binding = _binding(mode, payload, settings)
    if (directory / "complete.json").exists():
        verify_stage(directory)
        if read_json(directory / "binding.json") != binding:
            raise ValueError("Saved research configuration differs; use a new output directory")
        return _read_research(directory, sample)
    if mode == "live" and any((directory / "requests").glob("*.json")):
        raise ValueError("Incomplete live research already has model attempts; saved receipts are preserved and will not be resubmitted")
    freeze_json(directory / "binding.json", binding)
    freeze_json(directory / "input.json", payload)
    freeze_json(directory / "settings.json", _settings(settings))
    result = await run_research(sample, settings, directory, mode, budget,
                                **({"pubchem_limiter": pubchem_limiter} if pubchem_limiter is not None else {}))
    freeze_json(directory / "candidate_result.json", result.model_dump(mode="json"))
    sources = [Source.model_validate(row) for row in read_json(directory / "sources.json")]
    if not isinstance(result, ResearchResultV4):
        validate_research_references(result, sources)
        validate_research_protocol(result, sample, research_receipts(directory))
    if result.sample_id != sample.sample_id:
        raise ValueError("Research agent returned a different sample ID")
    freeze_json(directory / "result.json", result.model_dump(mode="json"))
    validation = ({"structure_valid": True, "source_records_saved": True, "reference_validation": "not_required",
                   "biological_accuracy_validated": False} if isinstance(result, ResearchResultV4) else
                  {"references_valid": True, "biological_accuracy_validated": False})
    freeze_json(directory / "validation.json", validation)
    seal_stage(directory)
    return result


def _description_evidence(directory: Path) -> set[str]:
    evidence_path = directory / "evidence.json"
    if not evidence_path.exists():
        raise ValueError("Missing observation evidence exposure record")
    return set(read_json(evidence_path).get("cp_feature_ids", []))


async def _describe(sample: SampleInput, research: ResearchOutput,
                    settings: Settings, directory: Path, mode: str, budget,
                    handoff_path: Path | None = None) -> ObservationPipelineResult:
    from .agent_runtime import run_observation
    from .tools import CPTable

    saved_handoff = directory / "handoff.json"
    revision = (HandoffRevision.model_validate(read_json(saved_handoff)["revision"])
                if saved_handoff.exists() and handoff_path is None else None)
    view, audit = prepare_observation(sample, research, handoff_path, revision)
    images = view.images
    cp = CPTable(sample.cp_file, preprocessing=sample.cp_preprocessing, metadata=sample.cp_metadata)
    # Bind exact measurement bytes without sending the private path to the agent.
    cp_hash = file_hash(sample.cp_file) if sample.cp_file else None
    binding = _binding(mode, view.model_dump(mode="json"), settings)
    binding.update(research_sha256=digest(research.model_dump(mode="json")), cp_sha256=cp_hash,
                   handoff_sha256=digest(audit))
    if (directory / "complete.json").exists():
        verify_stage(directory)
        if read_json(directory / "binding.json") != binding:
            raise ValueError("Saved observation inputs/configuration differ; use a new output directory")
        result = ObservationPipelineResult.model_validate(read_json(directory / "result.json"))
    else:
        if mode == "live" and (any((directory / "requests").glob("*.json"))
                               or any((directory / "phases").glob("*/requests/*.json"))):
            raise ValueError("Incomplete live observation already has model attempts; saved receipts are preserved and will not be resubmitted")
        freeze_json(directory / "binding.json", binding)
        freeze_json(saved_handoff, audit)
        freeze_json(directory / "input.json", view.model_dump(mode="json"))
        freeze_json(directory / "settings.json", _settings(settings))
        result = await run_observation(sample, view, settings, directory, mode, budget)
    if result.sample_id != sample.sample_id:
        raise ValueError("Observation agent returned a different sample ID")
    shown_cp = _description_evidence(directory)
    if not shown_cp <= {row.feature_id for row in cp.rows}:
        raise ValueError("Exposure record names unknown CP features")
    images = images + load_inspected_images(directory, images)
    validate_pipeline_references(result, view, images, shown_cp, cp,
                                 cp_query_results=cp_query_results(directory))
    check_observation_text(result.description, sample)
    freeze_json(directory / "result.json", result.model_dump(mode="json"))
    freeze_json(directory / "description_evidence.json", {"sentences": [
        sentence.model_dump(mode="json") for sentence in result.description_sentences], "observations": [
        {"observation_id": oid, "comparison_kind": row.comparison_kind, "text": row.text, "scope": row.scope,
         "image_ids": row.image_ids, "cp_feature_ids": row.cp_feature_ids,
         "cp_assessment": row.cp_assessment.model_dump(mode="json")}
        for oid in result.description_observation_ids for row in result.observations if row.id == oid]})
    freeze_json(directory / "validation.json", {"references_valid": True, "hypothesis_coverage_valid": True,
                "description_assembly_valid": True, "cp_integration_structure_valid": True,
                "phase_boundaries_valid": True, "sentence_review_complete": True,
                "biological_accuracy_validated": False})
    if not (directory / "complete.json").exists():
        seal_stage(directory)
    return result


def _replay(sample: SampleInput, source: Path, output: Path, command: str) -> dict:
    if source.resolve() == output.resolve() or source.resolve() in output.resolve().parents or output.resolve() in source.resolve().parents:
        raise ValueError("Replay source and output must be separate directories")
    research = _read_research(source / "research", sample)
    stages = ["research"]
    if command != "research":
        observation = source / "observation"
        final = read_observation_agent(observation)
        reviewed = observation / "reviewed"
        audit = read_json(reviewed / "handoff.json")
        view, expected_audit = prepare_observation(sample, research, revision=HandoffRevision.model_validate(audit["revision"]))
        if read_json(reviewed / "input.json") != view.model_dump(mode="json") or audit != expected_audit:
            raise ValueError("Replay image metadata or hashes differ")
        if read_json(reviewed / "binding.json")["cp_sha256"] != (file_hash(sample.cp_file) if sample.cp_file else None):
            raise ValueError("Replay CP input differs")
        if read_json(reviewed / "binding.json")["research_sha256"] != digest(research.model_dump(mode="json")):
            raise ValueError("Replay Research result differs from the bound Observation input")
        result = ObservationPipelineResult.model_validate(read_json(reviewed / "result.json"))
        from .tools import CPTable
        cp = CPTable(sample.cp_file, preprocessing=sample.cp_preprocessing, metadata=sample.cp_metadata)
        validate_pipeline_references(result, view, view.images + load_inspected_images(reviewed, view.images),
                                     _description_evidence(reviewed), cp,
                                     cp_query_results=cp_query_results(reviewed))
        check_observation_text(result.description, sample)
        if result.sample_id != sample.sample_id or final.sample_id != sample.sample_id:
            raise ValueError("Replay description belongs to another sample")
        stages.append("observation")
    for name in stages:
        target = output / name
        if target.exists():
            read_observation_agent(target) if name == "observation" else verify_stage(target)
            if read_json(target / "complete.json") != read_json(source / name / "complete.json"):
                raise ValueError("Replay output already contains different artifacts")
        else:
            shutil.copytree(source / name, target)
    status = {"status": "complete", "mode": "replay", "sample_id": sample.sample_id,
              "source": str(source.resolve()), "replayed_stages": stages,
              "model_calls": 0, "network_calls": 0, "synthetic": read_json(source / "research/binding.json")["mode"] == "mock"}
    if command != "research":
        status.update(description_result=str(output / "observation/result.json"),
                      description_text=str(output / "observation/description.txt"),
                      agents=["research", "observation"], observation_modules=final.modules)
    freeze_json(output / "replay.json", status)
    write_json(output / "status.json", status)
    return status


async def execute(command: str, input_path: Path, output: Path, settings: Settings,
                  mode: Mode = "mock", research_path: Path | None = None,
                  replay_from: Path | None = None, handoff_path: Path | None = None) -> dict:
    sample = load_sample(input_path)
    output = output.resolve()
    if mode == "replay" and replay_from is not None:
        replay_source = replay_from.resolve()
        if replay_source == output or replay_source in output.parents or output in replay_source.parents:
            raise ValueError("Replay source and output must be separate directories")
    if command == "describe" and research_path is not None and mode != "replay":
        source = (research_path.parent if research_path.is_file() else research_path).resolve()
        destination = output / "research"
        if source != destination and (source == output or source in output.parents or output in source.parents):
            raise ValueError("Research source and output must not overlap; reuse output/research or choose a separate output")
    if command != "research" and mode != "replay":
        # Resolve well/site/channel input before either paid functional agent.
        preflight_observation_agent(sample, settings)
    with run_lock(output):
        if mode == "replay":
            if replay_from is None:
                raise ValueError("Replay requires --replay-from with a completed run")
            return _replay(sample, replay_from.resolve(), output, command)
        if mode == "live" and (settings.budget_usd is None or (settings.provider == "openai" and not os.environ.get(settings.api_key_env or "OPENAI_API_KEY"))):
            raise ValueError("Live mode requires OPENAI_API_KEY and an explicit positive --budget-usd")
        from .agent_runtime import BudgetLedger

        budget = BudgetLedger(output / "budget.json", settings.budget_usd, settings)
        freeze_json(output / "runtime.json", {
            "schema_version": "1", "mode": mode, "sample_id": sample.sample_id,
            "model": settings.model, "reasoning_effort": settings.reasoning_effort,
            "packages": {name: importlib.metadata.version(name) for name in ("openai-agents", "openai", "pydantic")},
        })
        try:
            if command == "describe":
                if research_path is None:
                    raise ValueError("describe requires --research pointing to a completed research stage or result.json")
                source = research_path.parent if research_path.is_file() else research_path
                research = _read_research(source, sample)
                if mode == "live" and read_json(source / "binding.json")["mode"] != "live":
                    raise ValueError("Paid observation requires live research, not synthetic research")
                destination = output / "research"
                if source.resolve() != destination.resolve():
                    if not destination.exists():
                        shutil.copytree(source, destination)
                    elif verify_stage(destination) != verify_stage(source):
                        raise ValueError("Output already has different research artifacts")
            else:
                research = await _research(sample, settings, output / "research", mode, budget)
            observation = None
            if command != "research":
                observation = await run_observation_agent(
                    sample, research, settings, output / "observation", mode, budget, handoff_path)
            status = {"status": "complete", "sample_id": sample.sample_id, "mode": mode,
                      "synthetic": mode == "mock", "research_result": str(output / "research/result.json"),
                      "description_result": str(output / "observation/result.json") if command != "research" else None,
                      "description_text": str(output / "observation/description.txt") if command != "research" else None,
                      "agents": ["research", "observation"] if observation is not None else ["research"],
                      "observation_modules": observation.modules if observation is not None else [],
                      "biological_accuracy_validated": False}
            write_json(output / "status.json", status)
            return status
        except Exception as error:
            write_json(output / "status.json", {"status": "failed", "mode": mode,
                       "sample_id": sample.sample_id, "error_type": type(error).__name__, "error": str(error)})
            raise
