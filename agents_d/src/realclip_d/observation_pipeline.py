"""Image-first observation, guided evidence, writing and independent image review.

Each phase starts a new SDK conversation. Only explicitly projected artifacts
cross the phase boundary; budgets and tool limits remain shared by the run.
"""
from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any

from agents import Agent, AgentOutputSchema, RunConfig, Runner, function_tool
from agents.run_error_handlers import RunErrorHandlerResult
from agents.tool import ToolOutputImage, ToolOutputText
from pydantic import ValidationError

from .config import Settings
from .contracts import SampleInput
from .inputs import image_manifest
from .language import validate_agent_input, validate_agent_output
from .observation_contracts import ObservationRequest
from .observation_protocol import check_observation_text
from .observation_stages import (
    BaselineOutput, DescriptionDraft, DescriptionReview, GuidanceEvidenceOutput,
    ObservationPipelineResult, apply_review, validate_baseline, validate_draft,
    validate_evidence, validate_pipeline_references,
)
from .storage import read_json, write_json
from .tools.cp import CPTable
from .tools.image_inspection import ImageInspection


PHASES = ("baseline", "guided", "writer", "reviewer")


def cp_query_results(directory: Path) -> list[dict]:
    """Read actual query receipts for current execution and sealed replay checks."""
    path = directory / "tools.jsonl"
    return [row for line in path.read_text().splitlines()
            if (row := json.loads(line)).get("tool") == "query_cp"] if path.exists() else []


def _changed_output_paths(before: Any, after: Any, path: str = "") -> list[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        return [changed for key in sorted(before.keys() | after.keys())
                for changed in (_changed_output_paths(before[key], after[key], f"{path}/{key}")
                                if key in before and key in after else [f"{path}/{key}"])]
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [changed for index, (left, right) in enumerate(zip(before, after))
                for changed in _changed_output_paths(left, right, f"{path}/{index}")]
    return [] if before == after else [path or "/"]


def _image_content(metadata: dict, path: Path) -> list[dict]:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != metadata["sha256"]:
        raise ValueError("Image bytes changed after observation input preparation")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return [{"type": "input_text", "text": json.dumps(metadata, ensure_ascii=False)},
            {"type": "input_image", "image_url": f"data:{mime};base64," + base64.b64encode(data).decode(),
             "detail": "high"}]


def phase_messages(payload: dict, sample: SampleInput | None = None,
                   inspection: ImageInspection | None = None) -> list[dict]:
    """Project explicit phase context; writer never receives images/hypotheses."""
    validate_agent_input(payload)
    content: list[dict] = [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]
    if sample is not None:
        for row, original in zip(image_manifest(sample, include_metadata=True), sample.images):
            content.extend(_image_content(row, original.path))
    if inspection is not None:
        for row in inspection.images:
            content.extend(_image_content(row, inspection.path_for(row["image_id"])))
    return [{"role": "user", "content": content}]


def mock_model_for_phase(phase: str, steps: list, final: dict):
    """Separate injection point for deterministic tests of all phase boundaries."""
    from . import agent_runtime as runtime
    return runtime.ScriptedModel(steps, final)


def _mock_final(phase: str, sample: SampleInput, view: ObservationRequest, baseline=None) -> dict:
    limitation = "SYNTHETIC MOCK OUTPUT: no biological image interpretation was performed; the visual description is empty."
    if phase == "baseline":
        return {"sample_id": sample.sample_id, "observations": [],
                "reviewed_image_ids": [row["image_id"] for row in view.images], "limitations": [limitation]}
    if phase == "guided":
        return {"sample_id": sample.sample_id, "observations": baseline.observations if baseline else [],
                "hypothesis_assessments": [{"hypothesis_id": h.hypothesis_id, "assessment": "uncertain",
                    "checks": [{"check": h.change, "assessment": "uncertain", "observation_ids": [],
                        "reason": "The local scripted model performs no biological image interpretation.",
                        "contrast": {"treatment": "Not interpreted by the scripted model.",
                                     "control": "Not interpreted by the scripted model.", "difference": "unassessable"}}],
                    "reason": "Synthetic mock only; no visual hypothesis assessment was performed."}
                    for h in view.hypotheses], "limitations": [limitation]}
    return {"sample_id": sample.sample_id, "sentences" if phase == "writer" else "reviews": [],
            "limitations": [limitation]}


def load_inspected_images(directory: Path, originals: list[dict]) -> list[dict]:
    """Validate saved crops for final validation and replay, including pixel lineage."""
    from io import BytesIO
    from PIL import Image
    from .storage import file_hash

    manifest = read_json(directory / "crops" / "manifest.json")
    if manifest["originals"] != originals:
        raise ValueError("Saved crop parents differ from the bound original images")
    by_id = {row["image_id"]: row for row in originals}
    crops = []
    for entry in manifest["crops"]:
        row = {key: value for key, value in entry.items() if key != "relative_path"}
        parent = by_id.get(row["parent_image_id"])
        if parent is None or row["image_id"] in by_id or any(c["image_id"] == row["image_id"] for c in crops):
            raise ValueError("Invalid generated crop identity or parent")
        for key in ("role", "channel", "field", "well_id", "alignment_id", "display_settings", "acquisition_settings", "stain", "pixel_size_um"):
            if row.get(key) != parent.get(key):
                raise ValueError("Generated crop metadata differs from its parent")
        x, y, width, height = row["crop_box"]
        if (any(type(v) is not int for v in (x, y, width, height)) or min(x, y) < 0 or min(width, height) < 1
                or x + width > parent["width"] or y + height > parent["height"]
                or (row["width"], row["height"]) != (width, height)
                or row["parent_sha256"] != parent["sha256"]):
            raise ValueError("Invalid saved crop bounds or source hash")
        relative = entry["relative_path"]
        if relative != row["image_id"] + ".png" or Path(relative).name != relative:
            raise ValueError("Invalid saved crop path")
        path = directory / "crops" / relative
        if file_hash(path) != row["sha256"]:
            raise ValueError("Saved crop image changed")
        with Image.open(BytesIO(path.read_bytes())) as loaded:
            loaded.load()
            if loaded.size != (width, height):
                raise ValueError("Saved crop dimensions differ from metadata")
        crops.append(row)
    return crops


async def run_pipeline(sample: SampleInput, view: ObservationRequest, settings: Settings,
                       stage_dir: Path, mode: str = "mock", budget=None) -> ObservationPipelineResult:
    from . import agent_runtime as runtime

    # Complete read-only checks before constructing a paid client.
    validate_agent_input(view.model_dump(mode="json"))
    check_observation_text(view.model_dump(mode="json"), sample)
    if view.sample_id != sample.sample_id or view.context != sample.context:
        raise ValueError("Observation view belongs to different sample/context")
    if view.images != image_manifest(sample, include_metadata=True):
        raise ValueError("Observation view differs from supplied image bytes/metadata")
    cp = CPTable(sample.cp_file, preprocessing=sample.cp_preprocessing, metadata=sample.cp_metadata)
    check_observation_text([row.model_dump(exclude={"value"}) for row in cp.rows], sample)
    check_observation_text(cp.metadata.model_dump(mode="json"), sample)
    if view.cp_catalogue != cp.catalogue() or view.cp_available != (sample.cp_file is not None):
        raise ValueError("Observation CP catalogue differs from supplied measurements")
    if settings.observation_max_turns < len(PHASES):
        raise ValueError("Observation requires at least four total model turns for its four phases")
    if mode not in {"mock", "live"}:
        raise ValueError("Runtime mode must be mock or live")
    if any((stage_dir / "phases").glob("*/requests/*.json")):
        raise ValueError("Incomplete observation has model attempts; use a new directory instead of resubmitting")
    stage_dir.mkdir(parents=True, exist_ok=True)
    recorder = runtime._ToolRecorder(stage_dir, settings.max_tool_calls)
    inspection = ImageInspection(sample, stage_dir / "crops", max_crops=settings.observation_max_crops,
                                 max_pixels=settings.observation_crop_max_pixels)
    write_json(stage_dir / "sources.json", [])
    write_json(stage_dir / "evidence.json", {"cp_feature_ids": []})
    write_json(stage_dir / "output_contract.json", {"name": "ObservationPipelineResult", "schema_version": "4",
        "description_assembly": "ordered_reviewed_sentences_with_observation_references"})
    write_json(stage_dir / "output_schema.json", AgentOutputSchema(ObservationPipelineResult).json_schema())
    exposure: dict[str, list[str]] = {}
    plans: list[dict] = []
    used_turns = 0
    active_phase = "baseline"
    client = None

    def save_state():
        write_json(stage_dir / "image_exposure.json", exposure)
        write_json(stage_dir / "pipeline.json", {"schema_version": "1", "mode": mode,
            "phases": plans, "model_turns": used_turns, "max_model_turns": settings.observation_max_turns,
            "tool_calls": recorder.calls, "max_tool_calls": settings.max_tool_calls,
            "crop_count": len(inspection.images), "biological_accuracy_validated": False})

    @function_tool
    async def inspect_image(image_id: str, x: int, y: int, width: int, height: int,
                            question: str, selection_reason: str) -> list[ToolOutputText | ToolOutputImage]:
        """Inspect native pixels from one original image. Coordinates are x,y,width,height in pixels.

        Specify the detail to verify and why this region matters. Returns the actual
        lossless crop image and its parent/site/channel metadata. No resizing,
        enhancement, new optical resolution or automatic channel registration.
        Other sites and treatment/control are different cells, not matching objects.
        """
        arguments = dict(image_id=image_id, x=x, y=y, width=width, height=height,
                         question=question, selection_reason=selection_reason)
        recorder.calls += 1
        try:
            validate_agent_output({"question": question, "selection_reason": selection_reason})
            if recorder.calls > recorder.max_calls:
                raise ValueError("Run tool-call limit reached")
            metadata = inspection.crop(**arguments)
            content = _image_content(metadata, inspection.path_for(metadata["image_id"]))
            if metadata["image_id"] not in exposure[active_phase]:
                exposure[active_phase].append(metadata["image_id"])
            result = {"status": "ok", "data": metadata}
            outputs = [ToolOutputText(text=json.dumps(result, ensure_ascii=False)),
                       ToolOutputImage(image_url=content[1]["image_url"], detail="high")]
        except ValueError as exc:
            result = {"status": "error", "message": str(exc)}
            outputs = [ToolOutputText(text=json.dumps(result))]
        runtime._append(stage_dir / "tools.jsonl", {"phase": active_phase, "tool": "inspect_image",
            "keyword_arguments": arguments, "result": result, "model_result": result})
        save_state()
        return outputs

    @function_tool
    async def query_cp(compartment: str = "all", channel: str = "all", family: str = "all",
                       offset: int = 0, limit: int = 40, query: str = "",
                       feature_names: list[str] | None = None) -> str:
        """Return paired treatment/control features; offset and limit count comparisons.

        Discover actual feature names using query, then select exact feature_names.
        Choose filter combinations from cp_catalogue; do not invent a family.
        On no match, read query_diagnostic and its navigation suggestions; these
        names contain no measurement evidence. No match does not mean no CP.
        Respect comparison_ready, definitions, scope and interpretation_gaps.
        Numeric order alone does not establish physical size or a per-cell change.
        """
        return await recorder.call("query_cp", cp.query, compartment=compartment, channel=channel,
            family=family, offset=offset, limit=limit, query=query, feature_names=feature_names)

    def aggregate_phase(phase: str, directory: Path):
        """Portable root receipts preserve phase-local input counters and raw responses."""
        for item in (directory / "calls.jsonl").read_text().splitlines() if (directory / "calls.jsonl").exists() else []:
            runtime._append(stage_dir / "calls.jsonl", {"phase": phase, **json.loads(item)})
        for kind in ("requests", "responses", "input_token_counts"):
            for source in sorted((directory / kind).glob("*.json")):
                target = stage_dir / kind / f"{phase}_{source.name}"
                write_json(target, read_json(source))
        calls_path = stage_dir / "calls.jsonl"
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
        write_json(stage_dir / "usage.json", {"requests": len(calls), **{
            key: sum((row.get("usage") or {}).get(key, 0) for row in calls)
            for key in ("input_tokens", "output_tokens", "total_tokens")}})

    async def phase(name: str, payload: dict, schema: type, tools: list, *, images: bool,
                    baseline=None):
        nonlocal used_turns, active_phase, client
        active_phase = name
        index = PHASES.index(name)
        remaining = settings.observation_max_turns - used_turns
        reserved_later = len(PHASES) - index - 1
        preferred = {"baseline": 2, "guided": 3, "writer": 1, "reviewer": 2}[name]
        turns = min(preferred, remaining - reserved_later)
        if turns < 1:
            raise ValueError("Observation model turn budget exhausted before completing all phases")
        correction_turns = min(1, remaining - reserved_later - turns)
        phase_limit = turns + correction_turns
        directory = stage_dir / "phases" / name
        directory.mkdir(parents=True, exist_ok=True)
        exposure[name] = ([row["image_id"] for row in view.images] +
                          [row["image_id"] for row in inspection.images]) if images else []
        messages = phase_messages(payload, sample if images else None, inspection if images else None)
        prompt = runtime._prompt("observation_" + name, "v1")
        final = _mock_final(name, sample, view, baseline)
        if mode == "live":
            inner, client = runtime._live_model(settings)
        else:
            first = sample.images[0]
            size = next(row for row in view.images if row["image_id"] == first.image_id)
            steps = ([[('inspect_image', dict(image_id=first.image_id, x=0, y=0,
                width=min(128, size["width"]), height=min(128, size["height"]),
                question="Exercise native image transport without interpreting cells.",
                selection_reason="Synthetic offline tool plumbing only."))]] if name == "baseline" and turns > 1
                and settings.observation_max_crops else
                [[('query_cp', {})]] if name == "guided" and turns > 1 else [])
            inner = mock_model_for_phase(name, steps, runtime._jsonable(final))
        phase_settings = settings.model_copy(update={"observation_max_turns": phase_limit})

        class PhaseRecordedModel(runtime.RecordedModel):
            async def get_response(self, *args, **kwargs):
                if self.calls >= phase_limit:
                    raise runtime.BudgetError("Observation phase model-call limit reached")
                if self.calls + 1 >= turns:
                    self.force_final = True
                return await super().get_response(*args, **kwargs)

        model = PhaseRecordedModel(inner, phase_settings, directory, "observation", budget, mode == "live", False)
        agent = Agent(name="D Observation " + name, instructions=prompt, model=model,
            model_settings=replace(runtime._model_settings(settings, False), truncation="disabled"),
            tools=tools, output_type=schema)
        (directory / "instructions.md").write_text(prompt)
        write_json(directory / "output_schema.json", AgentOutputSchema(schema).json_schema())
        write_json(directory / "request_input.json", runtime.sanitize_request(messages))
        plans.append({"phase": name, "status": "running", "max_turns": phase_limit,
                      "ordinary_max_turns": turns, "correction_max_turns": correction_turns,
                      "corrections": [],
                      "research_guidance_visible": name == "guided", "images_visible": images})
        save_state()

        async def correct_invalid_output(failure):
            """One metered schema correction; never retry transport/usage failures."""
            response = runtime._jsonable(failure.run_data.raw_responses[-1])
            messages = [item for item in response.get("output", []) if item.get("type") == "message"]
            text = "".join(part["text"] for part in messages[-1].get("content", [])
                           if part.get("type") == "output_text") if messages else ""
            # Recover only syntactically valid JSON that fails our typed contract.
            # Refusals, truncation and unrelated SDK failures keep their original error.
            try:
                candidate = json.loads(text)
            except (ValueError, TypeError):
                raise failure.error
            try:
                schema.model_validate_json(text, strict=True)
            except ValidationError as exc:
                errors = exc.errors(include_url=False, include_context=False, include_input=False)
            else:
                raise failure.error
            correction = {"failed_call": model.calls, "invalid_output": candidate,
                          "validation_errors": errors, "status": "not_attempted"}
            record = directory / "corrections" / "001.json"
            plans[-1]["corrections"].append({"failed_call": model.calls, "status": "not_attempted"})
            write_json(record, correction)
            save_state()
            if (not correction_turns or model.calls >= phase_limit
                    or used_turns + model.calls + 1 + reserved_later > settings.observation_max_turns):
                raise ValueError(f"{name} output failed typed validation; no correction turn remains: {errors}")
            if mode == "live" and (budget is None or budget.state["blocked"]):
                raise runtime.UnknownUsageError("Cannot correct output with an unavailable or blocked budget")
            feedback = {"output_validation_errors": errors,
                "instruction": "Correct only the contract errors in your previous JSON output. Preserve all valid observations, image/CP references, baseline records and findings. Do not invent evidence or change a finding to satisfy validation. No tools are available. Return the complete corrected output using the same schema."}
            correction.update(status="attempted", correction_call=model.calls + 1)
            plans[-1]["corrections"][-1].update(status="attempted", correction_call=model.calls + 1)
            write_json(record, correction)
            save_state()
            try:
                corrected = await Runner.run(agent.clone(tools=[]), [*failure.run_data.history,
                    {"role": "user", "content": json.dumps(feedback, ensure_ascii=False)}],
                    max_turns=1, run_config=RunConfig(tracing_disabled=True))
                corrected_output = runtime._jsonable(corrected.final_output)
                correction.update(status="corrected", corrected_output=corrected_output,
                                  changed_paths=_changed_output_paths(candidate, corrected_output))
                plans[-1]["corrections"][-1]["status"] = "corrected"
                return RunErrorHandlerResult(final_output=corrected.final_output, include_in_history=False)
            except Exception as exc:
                correction.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                plans[-1]["corrections"][-1]["status"] = "failed"
                raise
            finally:
                write_json(record, correction)
                save_state()

        try:
            response = await Runner.run(agent, messages, max_turns=turns, run_config=RunConfig(tracing_disabled=True),
                error_handlers={"invalid_final_output": correct_invalid_output})
            raw = runtime._jsonable(response.final_output)
            write_json(directory / "candidate_output.json", raw)
            write_json(stage_dir / "candidate_output.json", {"phase": name, "output": raw})
            output = schema.model_validate(raw)
            validate_agent_output(output.model_dump(mode="json"))
            return output
        finally:
            used_turns += model.calls
            aggregate_phase(name, directory)
            calls = [json.loads(line) for line in (directory / "calls.jsonl").read_text().splitlines()
                     ] if (directory / "calls.jsonl").exists() else []
            write_json(directory / "usage.json", {"requests": len(calls), **{
                key: sum((row.get("usage") or {}).get(key, 0) for row in calls)
                for key in ("input_tokens", "output_tokens", "total_tokens")}})
            if client is not None:
                await client.close()
                client = None
            save_state()

    def finish(name: str, output):
        write_json(stage_dir / "phases" / name / "output.json", output.model_dump(mode="json"))
        plans[-1]["status"] = "validated"
        save_state()

    baseline_payload = {"sample_id": view.sample_id, "context": view.context.model_dump(mode="json"),
                        "images": view.images, "cp_available": view.cp_available}
    baseline = await phase("baseline", baseline_payload, BaselineOutput, [inspect_image], images=True)
    validate_baseline(baseline, view, view.images + inspection.images)
    check_observation_text([r.text for r in baseline.observations], sample)
    finish("baseline", baseline)

    guided_payload = {**view.model_dump(mode="json"), "baseline": baseline.model_dump(mode="json")}
    evidence = await phase("guided", guided_payload, GuidanceEvidenceOutput, [inspect_image, query_cp],
                           images=True, baseline=baseline)
    validate_evidence(evidence, view, view.images + inspection.images, recorder.cp_ids, cp, baseline,
                      cp_query_results=cp_query_results(stage_dir))
    check_observation_text([r.text for r in evidence.observations], sample)
    finish("guided", evidence)

    # Writing/review receive observations, not the Research hypothesis checklist.
    writing_evidence = evidence.model_dump(mode="json", exclude={"hypothesis_assessments"})
    draft = await phase("writer", {"evidence": writing_evidence}, DescriptionDraft, [], images=False)
    validate_draft(draft, evidence)
    check_observation_text([r.text for r in draft.sentences], sample)
    finish("writer", draft)

    review = await phase("reviewer", {"evidence": writing_evidence, "draft": draft.model_dump(mode="json"),
        "images": view.images}, DescriptionReview, [inspect_image], images=True)
    result = apply_review(draft, review, evidence)
    validate_pipeline_references(result, view, view.images + inspection.images, recorder.cp_ids, cp,
                                 cp_query_results=cp_query_results(stage_dir))
    check_observation_text(result.description, sample)
    finish("reviewer", review)
    write_json(stage_dir / "candidate_output.json", result.model_dump(mode="json"))
    write_json(stage_dir / "observations.json", evidence.model_dump(mode="json"))
    write_json(stage_dir / "request_input.json", {name: read_json(stage_dir / "phases" / name / "request_input.json") for name in PHASES})
    return result
