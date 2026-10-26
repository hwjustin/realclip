"""Exercise all observation phases through the real SDK with local models only."""
from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
import json
import socket

from PIL import Image
from agents.exceptions import ModelBehaviorError, UserError
import pytest

from realclip_d import agent_runtime as runtime
from realclip_d import observation_pipeline as pipeline
from realclip_d.config import Settings
from realclip_d.contracts import ImageInput, SampleInput
from realclip_d.inputs import image_manifest
from realclip_d.observation_contracts import ObservationRequest
from realclip_d.observation_stages import ObservationPipelineResult
from realclip_d.storage import digest, read_json
from realclip_d.tools.cp import CPTable


GUIDANCE_MARKER = "GUIDANCE_ONLY_6351"
CP_VALUE = 987654.125


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Pipeline integration tests forbid network or paid clients")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(runtime, "AsyncOpenAI", forbidden)
    monkeypatch.setattr(runtime, "_live_model", forbidden)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def case(tmp_path):
    sample = SampleInput(sample_id="PIPELINE_SYNTHETIC", compound={"name": "PRIVATE_COMPOUND"})
    for role in ("treatment", "control"):
        for channel in ("DNA", "AGP"):
            path = tmp_path / f"private-{role}-{channel}.png"
            image = Image.new("RGB", (20, 16))
            image.putdata([(x * 11, y * 13, (x + y) * 7) for y in range(16) for x in range(20)])
            image.save(path)
            sample.images.append(ImageInput(image_id=f"{role}_{channel}", path=path,
                role=role, channel=channel, field="site_1", well_id=f"{role}_well"))
    sample.cp_file = tmp_path / "cp.csv"
    sample.cp_file.write_text(
        "feature_id,feature_name,compartment,channel,family,role,value,unit,scope\n"
        f"cp_t,Cells_AreaShape_Area,Cells,all,AreaShape,treatment,{CP_VALUE},unknown,well\n"
        "cp_c,Cells_AreaShape_Area,Cells,all,AreaShape,control,2,unknown,well\n")
    table = CPTable(sample.cp_file)
    view = ObservationRequest(sample_id=sample.sample_id, context=sample.context,
        images=image_manifest(sample, include_metadata=True), cp_available=True,
        cp_catalogue=table.catalogue(), hypotheses=[{
            "hypothesis_id": "H1", "change": GUIDANCE_MARKER,
            "applicability": "Synthetic offline fixture only.",
            "where_to_look": "Supplied DNA views.",
            "what_to_check": "Compare the supplied outline patterns.",
            "how_to_compare": "Inspect both roles across sites; preserve uncertainty.",
        }])
    return sample, view


def observation():
    return {"id": "O1", "comparison_kind": "treatment_control", "text": "Synthetic outline patterns are visible in both roles.",
        "scope": "Synthetic supplied site only; no biological interpretation.",
        "image_ids": ["treatment_DNA", "control_DNA"], "cp_feature_ids": [],
        "visually_confirmed": True, "cp_assessment": {"status": "not_checked",
            "reason": "Measurements were not used for this synthetic visual observation.",
            "effect_on_description": "Keep the statement limited to the supplied image patterns."}}


def phase_finals(view):
    obs = observation()
    limitation = "Synthetic offline fixture; no biological assessment is made."
    assessment = {"hypothesis_id": "H1", "assessment": "uncertain", "checks": [{
        "check": "Compare the supplied outline patterns.", "assessment": "uncertain",
        "observation_ids": ["O1"], "reason": "Synthetic fixture only.",
        "contrast": {"treatment": "Synthetic patterns.", "control": "Synthetic patterns.",
                     "difference": "unassessable"}}], "reason": "Synthetic fixture only."}
    sentence = {"id": "S1", "text": "Both supplied roles contain synthetic outline patterns.",
                "observation_ids": ["O1"]}
    return {
        "baseline": {"sample_id": view.sample_id, "observations": [obs],
            "reviewed_image_ids": [row["image_id"] for row in view.images], "limitations": [limitation]},
        "guided": {"sample_id": view.sample_id, "observations": [deepcopy(obs)],
            "hypothesis_assessments": [assessment], "limitations": [limitation]},
        "writer": {"sample_id": view.sample_id, "sentences": [sentence], "limitations": [limitation]},
        "reviewer": {"sample_id": view.sample_id, "reviews": [{"sentence_id": "S1", "action": "keep",
            "text": sentence["text"], "observation_ids": ["O1"], "reason": "Synthetic review fixture."}],
            "limitations": [limitation]},
    }


def capture_models(monkeypatch, *, finals=None, scripts=None):
    captured = {}

    def factory(phase, suggested_steps, suggested_final):
        captured[phase] = []

        class CapturingModel(runtime.ScriptedModel):
            async def get_response(self, *args, **kwargs):
                captured[phase].append(runtime._jsonable(kwargs.get("input", args[1] if len(args) > 1 else None)))
                return await super().get_response(*args, **kwargs)

        return CapturingModel(
            deepcopy(scripts.get(phase, []) if scripts is not None else suggested_steps),
            deepcopy(finals[phase] if finals is not None else suggested_final))

    monkeypatch.setattr(pipeline, "mock_model_for_phase", factory)
    return captured


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def images_in(value):
    return [row for row in walk(value) if row.get("type") == "input_image"]


def initial_payload(request):
    return json.loads(request[0]["content"][0]["text"])


def crop_arguments(**updates):
    return {"image_id": "treatment_DNA", "x": 2, "y": 3, "width": 5, "height": 4,
        "question": "Are adjacent outlines distinguishable?",
        "selection_reason": "Check a local detail without inferring whole-well prevalence.", **updates}


@pytest.mark.asyncio
async def test_phase_boundaries_and_reviewed_description(case, tmp_path, monkeypatch):
    sample, view = case
    finals = phase_finals(view)
    captured = capture_models(monkeypatch, finals=finals, scripts={})
    stage = tmp_path / "observation"
    result = await pipeline.run_pipeline(sample, view, Settings(), stage)
    assert isinstance(result, ObservationPipelineResult)
    assert result.schema_version == "4"
    assert result.description == finals["writer"]["sentences"][0]["text"]
    assert result.description_observation_ids == ["O1"]

    baseline = initial_payload(captured["baseline"][0])
    assert "hypotheses" not in baseline and "cp_catalogue" not in baseline
    assert GUIDANCE_MARKER not in json.dumps(captured["baseline"])
    assert str(CP_VALUE) not in json.dumps(captured["baseline"])
    assert len(images_in(captured["baseline"][0])) == len(sample.images)
    guided = initial_payload(captured["guided"][0])
    assert guided["baseline"] == finals["baseline"]
    assert guided["hypotheses"][0]["change"] == GUIDANCE_MARKER
    for name in ("writer", "reviewer"):
        assert GUIDANCE_MARKER not in json.dumps(captured[name])
        assert "hypothesis_assessments" not in json.dumps(captured[name])
    assert images_in(captured["writer"]) == []
    assert len(images_in(captured["reviewer"][0])) == len(sample.images)
    assert initial_payload(captured["reviewer"][0])["draft"] == finals["writer"]
    assert [row["action"] for row in read_json(stage / "phases/reviewer/output.json")["reviews"]] == ["keep"]
    state = read_json(stage / "pipeline.json")
    assert state["model_turns"] == 4
    assert all(row["status"] == "validated" for row in state["phases"])


@pytest.mark.asyncio
async def test_crop_tool_transports_actual_native_pixels_to_next_sdk_request(case, tmp_path, monkeypatch):
    sample, view = case
    captured = capture_models(monkeypatch, finals=phase_finals(view),
        scripts={"baseline": [[("inspect_image", crop_arguments())]]})
    stage = tmp_path / "observation"
    await pipeline.run_pipeline(sample, view, Settings(), stage)
    assert len(captured["baseline"]) == 2
    crop_meta = read_json(stage / "crops/manifest.json")["crops"][0]
    data = (stage / "crops" / crop_meta["relative_path"]).read_bytes()
    expected_url = "data:image/png;base64," + base64.b64encode(data).decode()
    # The image returned by the tool enters the real SDK conversation as an
    # image, in addition to the textual provenance returned alongside it.
    returned = [row for row in images_in(captured["baseline"][1]) if row.get("image_url") == expected_url]
    assert returned and returned[0]["detail"] == "high"
    with Image.open(BytesIO(base64.b64decode(expected_url.split(",", 1)[1]))) as local, Image.open(sample.images[0].path) as original:
        assert local.size == (5, 4)
        assert local.tobytes() == original.crop((2, 3, 7, 7)).tobytes()
    exposures = read_json(stage / "image_exposure.json")
    assert crop_meta["image_id"] in exposures["baseline"]
    assert crop_meta["image_id"] in exposures["guided"]
    assert crop_meta["image_id"] in exposures["reviewer"]
    assert exposures["writer"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("updates,message", [
    ({"image_id": "not_supplied"}, "original supplied image ID"),
    ({"x": 100}, "exceeds the original image bounds"),
])
async def test_inspection_errors_are_returned_without_crops_or_exposure(case, tmp_path, monkeypatch, updates, message):
    sample, view = case
    captured = capture_models(monkeypatch, finals=phase_finals(view),
        scripts={"baseline": [[("inspect_image", crop_arguments(**updates))]]})
    stage = tmp_path / "observation"
    await pipeline.run_pipeline(sample, view, Settings(), stage)
    assert read_json(stage / "crops/manifest.json")["crops"] == []
    tools = [json.loads(line) for line in (stage / "tools.jsonl").read_text().splitlines()]
    assert tools[0]["result"]["status"] == "error"
    assert message in tools[0]["result"]["message"]
    assert len(images_in(captured["baseline"][1])) == len(sample.images)


@pytest.mark.asyncio
async def test_baseline_cannot_cite_a_crop_only_planned_for_later(case, tmp_path, monkeypatch):
    sample, view = case
    original = view.images[0]
    future_id = "crop_" + digest({"parent_image_id": original["image_id"],
        "parent_sha256": original["sha256"], "crop_box": [2, 3, 5, 4]})[:24]
    finals = phase_finals(view)
    finals["baseline"]["observations"][0]["image_ids"][0] = future_id
    captured = capture_models(monkeypatch, finals=finals,
        scripts={"guided": [[("inspect_image", crop_arguments())]]})
    stage = tmp_path / "observation"
    with pytest.raises(ValueError, match="unknown image"):
        await pipeline.run_pipeline(sample, view, Settings(), stage)
    assert set(captured) == {"baseline"}
    assert read_json(stage / "phases/baseline/candidate_output.json") == finals["baseline"]
    assert not (stage / "phases/baseline/output.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["missing_observation", "O2"])
async def test_writer_rejects_unknown_or_measurement_only_references_and_preserves_candidate(case, tmp_path, monkeypatch, reference):
    sample, view = case
    finals = phase_finals(view)
    finals["guided"]["observations"].append({"id": "O2", "comparison_kind": "measurement_only", "text": "Synthetic measurement only.",
        "scope": "Supplied well profile only.", "image_ids": [], "cp_feature_ids": ["cp_t", "cp_c"],
        "visually_confirmed": False, "cp_assessment": {"status": "measurement_only",
            "reason": "Only measurement rows support this fixture.",
            "effect_on_description": "Do not include this measurement-only observation in a visual description."}})
    finals["writer"]["sentences"][0]["observation_ids"] = [reference]
    captured = capture_models(monkeypatch, finals=finals,
        scripts={"guided": [[("query_cp", {})]]})
    stage = tmp_path / "observation"
    with pytest.raises(ValueError, match="only existing visual observations"):
        await pipeline.run_pipeline(sample, view, Settings(), stage)
    assert "reviewer" not in captured
    assert read_json(stage / "phases/writer/candidate_output.json") == finals["writer"]
    assert read_json(stage / "candidate_output.json")["phase"] == "writer"
    assert not (stage / "phases/writer/output.json").exists()


@pytest.mark.asyncio
async def test_reviewer_must_make_one_decision_for_every_draft_sentence(case, tmp_path, monkeypatch):
    sample, view = case
    finals = phase_finals(view)
    finals["reviewer"]["reviews"] = []
    capture_models(monkeypatch, finals=finals, scripts={})
    stage = tmp_path / "observation"
    with pytest.raises(ValueError, match="Every draft sentence requires exactly one review"):
        await pipeline.run_pipeline(sample, view, Settings(), stage)
    assert read_json(stage / "phases/reviewer/candidate_output.json") == finals["reviewer"]
    assert not (stage / "phases/reviewer/output.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("max_turns", [4, 5, 6, 8])
async def test_model_turn_limit_is_shared_across_all_four_phases(case, tmp_path, monkeypatch, max_turns):
    sample, view = case
    captured = capture_models(monkeypatch)
    stage = tmp_path / "observation"
    result = await pipeline.run_pipeline(sample, view, Settings(observation_max_turns=max_turns), stage)
    state = read_json(stage / "pipeline.json")
    actual_calls = sum(len(requests) for requests in captured.values())
    assert 4 <= actual_calls <= max_turns
    assert state["model_turns"] == actual_calls
    assert set(captured) == set(pipeline.PHASES)
    assert result.description == ""  # Default mock never invents biology.


@pytest.mark.asyncio
async def test_tool_limit_is_shared_across_inspection_and_cp_and_phases(case, tmp_path, monkeypatch):
    sample, view = case
    capture_models(monkeypatch, finals=phase_finals(view), scripts={
        "baseline": [[("inspect_image", crop_arguments())]],
        "guided": [[("query_cp", {})]],
        "reviewer": [[("inspect_image", crop_arguments(x=3))]],
    })
    stage = tmp_path / "observation"
    await pipeline.run_pipeline(sample, view, Settings(max_tool_calls=1), stage)
    tools = [json.loads(line) for line in (stage / "tools.jsonl").read_text().splitlines()]
    assert [row["tool"] for row in tools] == ["inspect_image", "query_cp", "inspect_image"]
    assert [row["result"]["status"] for row in tools] == ["ok", "error", "error"]
    assert all("tool-call limit" in row["result"]["message"] for row in tools[1:])
    assert len(read_json(stage / "crops/manifest.json")["crops"]) == 1
    assert read_json(stage / "evidence.json")["cp_feature_ids"] == []
    assert read_json(stage / "pipeline.json")["tool_calls"] == 3


@pytest.mark.asyncio
async def test_too_small_turn_budget_fails_before_model_construction(case, tmp_path, monkeypatch):
    sample, view = case
    captured = capture_models(monkeypatch)
    with pytest.raises(ValueError, match="at least four total model turns"):
        await pipeline.run_pipeline(sample, view, Settings(observation_max_turns=3), tmp_path / "observation")
    assert captured == {}


@pytest.mark.asyncio
async def test_invalid_cp_status_gets_one_metered_correction_with_original_evidence(case, tmp_path, monkeypatch):
    sample, view = case
    finals = phase_finals(view)
    extra = deepcopy(finals["guided"]["observations"][0])
    extra["id"] = "O2"
    extra["cp_assessment"]["status"] = "no_matching_measurement"
    extra["cp_assessment"]["reason"] = "The Cells/Number query returned no matching measurement."
    finals["guided"]["observations"].append(extra)
    invalid = deepcopy(finals["guided"])
    invalid["observations"][-1]["cp_assessment"]["status"] = "not_informative"
    requests = []

    def factory(phase, steps, final):
        if phase != "guided":
            return runtime.ScriptedModel([], finals[phase])

        class CorrectingModel(runtime.ScriptedModel):
            async def get_response(self, *args, **kwargs):
                requests.append({"input": runtime._jsonable(kwargs.get("input")),
                    "model_settings": runtime._jsonable(kwargs.get("model_settings")),
                    "tools": [tool.name for tool in kwargs.get("tools", [])]})
                self.final = invalid if self.calls < 3 else finals[phase]
                return await super().get_response(*args, **kwargs)

        return CorrectingModel([[('query_cp', {'family': 'AreaShape'})],
                                [('query_cp', {'compartment': 'Cells', 'family': 'Number'})]], invalid)

    monkeypatch.setattr(pipeline, "mock_model_for_phase", factory)
    stage = tmp_path / "observation"
    result = await pipeline.run_pipeline(sample, view, Settings(observation_max_turns=12), stage)
    assert result.observations[-1].cp_assessment.status == "no_matching_measurement"
    assert len(requests) == 4
    assert requests[-1]["tools"] == []
    assert requests[2]["model_settings"]["tool_choice"] == "none"
    repair = read_json(stage / "phases/guided/corrections/001.json")
    assert repair["failed_call"] == 3 and repair["correction_call"] == 4
    assert repair["status"] == "corrected"
    assert repair["invalid_output"] == invalid
    assert repair["changed_paths"] == ["/observations/1/cp_assessment/status"]
    assert repair["validation_errors"][0]["loc"] == ["observations", 1]
    assert "output_validation_errors" in json.dumps(requests[-1]["input"])
    assert len(images_in(requests[-1]["input"])) == len(sample.images)
    assert read_json(stage / "phases/guided/usage.json")["requests"] == 4
    assert read_json(stage / "usage.json")["requests"] == 7
    assert read_json(stage / "pipeline.json")["model_turns"] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("turns,expected_calls", [(4, 1), (12, 2)])
async def test_invalid_output_correction_is_bounded_and_never_seals_failure(case, tmp_path, monkeypatch,
                                                                         turns, expected_calls):
    sample, view = case
    finals = phase_finals(view)
    finals["baseline"]["observations"][0]["cp_assessment"]["status"] = "not_informative"
    captured = capture_models(monkeypatch, finals=finals, scripts={})
    stage = tmp_path / "observation"
    # The SDK redacts exceptions escaping an error handler into UserError.
    with pytest.raises((ValueError, ModelBehaviorError, UserError)):
        await pipeline.run_pipeline(sample, view, Settings(observation_max_turns=turns), stage)
    assert len(captured["baseline"]) == expected_calls
    assert set(captured) == {"baseline"}
    assert not (stage / "phases/baseline/output.json").exists()
    assert not (stage / "complete.json").exists()
    assert read_json(stage / "pipeline.json")["model_turns"] == expected_calls


@pytest.mark.asyncio
async def test_provider_failure_never_gets_schema_correction(case, tmp_path, monkeypatch):
    sample, view = case
    attempts = []

    def factory(phase, steps, final):
        class BrokenModel(runtime.ScriptedModel):
            async def get_response(self, *args, **kwargs):
                attempts.append(phase)
                raise TimeoutError("Synthetic transport failure; do not retry.")
        return BrokenModel([], final)

    monkeypatch.setattr(pipeline, "mock_model_for_phase", factory)
    stage = tmp_path / "observation"
    with pytest.raises(TimeoutError):
        await pipeline.run_pipeline(sample, view, Settings(observation_max_turns=12), stage)
    assert attempts == ["baseline"]
    assert not (stage / "phases/baseline/corrections").exists()
