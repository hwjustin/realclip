"""New Research V3 -> observation integration. All imagery and CP are synthetic."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import socket

from agents import AgentOutputSchema
from PIL import Image
import pytest

from realclip_d import agent_runtime as runtime
from realclip_d.config import Settings
from realclip_d.contracts import ImageInput, ResearchResultV3, SampleInput
from realclip_d.inputs import image_manifest, load_sample
from realclip_d.observation_contracts import (ObservationOutput, ObservationResult, assemble_result,
                                             validate_observation_references)
from realclip_d.observation_protocol import (HandoffRevision, check_observation_text,
                                            prepare_observation)
from realclip_d.storage import digest, read_json, verify_stage, write_json
from realclip_d.tools.cp import CPTable
from realclip_d.workflow import _describe

PROJECT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Observation tests forbid network/client creation")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(runtime, "AsyncOpenAI", forbidden)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def payload():
    value = read_json(PROJECT / "docs/observation_agent_design/output_example.synthetic.json")
    for index, row in enumerate(value["hypothesis_assessments"], 1):
        row["hypothesis_id"] = f"H{index}"
    return value


def add_images(sample, directory):
    sample.images = []
    for role in ("treatment", "control"):
        for view, channel in (("overview", "DNA"), ("local", "AGP")):
            path = directory / f"private-file-{role}-{view}.png"
            Image.new("RGB", (16, 16), color="black").save(path)
            sample.images.append(ImageInput(image_id=f"synthetic_{role}_{view}", path=path,
                role=role, channel=channel, field="F1", view=view, well_id=f"{role}_well",
                alignment_id=f"{role}_F1", stain="Synthetic format-test stain",
                display_settings="Same synthetic display settings; no biological content."))
    return sample


@pytest.fixture
def sample(tmp_path):
    value = add_images(SampleInput(sample_id="SYNTHETIC_OBSERVATION_DESIGN",
                                  compound={"name": "PRIVATE_COMPOUND"}), tmp_path)
    value.cp_file = tmp_path / "measurements.csv"
    value.cp_preprocessing = "Synthetic standardized measurements; not raw physical values."
    value.cp_file.write_text(
        "feature_id,feature_name,compartment,channel,family,role,value,unit,scope\n"
        "synthetic_treatment_area,AreaShape_Area,Cells,AGP,AreaShape,treatment,1,standardized,well\n"
        "synthetic_control_area,AreaShape_Area,Cells,AGP,AreaShape,control,2,standardized,well\n")
    return value


@pytest.fixture
def research(sample):
    changes = ["Fewer visible cells and rounder cells with brighter nuclei.",
               "Smaller projected cell area.", "Lower field-level cell abundance."]
    return ResearchResultV3(sample_id=sample.sample_id,
        compound_identification={"summary": "Synthetic identity fixture.", "source_ids": []},
        compound_research_and_analysis={"findings": [], "research_gaps": "Synthetic format fixture only."},
        morphology_hypotheses=[{"id": f"H{i}", "change": change,
            "basis": "Synthetic input premise; no biological claim.",
            "applicability": "Synthetic images only; no scientific applicability.", "source_ids": []}
            for i, change in enumerate(changes, 1)],
        observation_guidance=[{"hypothesis_id": f"H{i}", "where_to_look": "Supplied fields and local views.",
            "what_to_check": change, "how_to_compare": "Compare actual treatment/control views and retain uncertainty."}
            for i, change in enumerate(changes, 1)], sources=[])


def test_model_schema_is_five_fields_without_generated_description_or_version(payload):
    schema = AgentOutputSchema(ObservationOutput).json_schema()
    assert set(schema["properties"]) == {
        "sample_id", "observations", "hypothesis_assessments", "description_observation_ids", "limitations"}
    assert schema["additionalProperties"] is False
    output = ObservationOutput.model_validate(payload)
    result = assemble_result(output)
    assert result.schema_version == "3"
    assert result.description == " ".join(row["text"] for row in payload["observations"][:2])
    assert "lower cell area" not in result.description
    assert result.hypothesis_assessments[0].assessment == "partially_supported"
    # H1 and H3 deliberately reuse O1; it is only selected once.
    assert result.description_observation_ids.count("O1") == 1


@pytest.mark.parametrize("mutation,match", [
    ("duplicate_observation", "Observation IDs"),
    ("duplicate_assessment", "assessments must be unique"),
    ("unknown_check_ref", "existing observation"),
    ("duplicate_selection", "unique existing"),
    ("unknown_selection", "unique existing"),
    ("cp_only_selection", "Measurement-only"),
    ("cp_only_supported", "require visual observations"),
    ("all_uncertain_partial", "Overall assessment"),
    ("empty_without_reason", "empty description"),
])
def test_invalid_links_selection_and_judgments_are_rejected(payload, mutation, match):
    if mutation == "duplicate_observation": payload["observations"].append(deepcopy(payload["observations"][0]))
    elif mutation == "duplicate_assessment": payload["hypothesis_assessments"].append(deepcopy(payload["hypothesis_assessments"][0]))
    elif mutation == "unknown_check_ref": payload["hypothesis_assessments"][0]["checks"][0]["observation_ids"] = ["unknown"]
    elif mutation == "duplicate_selection": payload["description_observation_ids"] = ["O1", "O1"]
    elif mutation == "unknown_selection": payload["description_observation_ids"] = ["unknown"]
    elif mutation == "cp_only_selection": payload["description_observation_ids"] = ["O3"]
    elif mutation == "cp_only_supported":
        payload["hypothesis_assessments"][2]["checks"][0]["observation_ids"] = ["O3"]
    elif mutation == "all_uncertain_partial":
        for check in payload["hypothesis_assessments"][0]["checks"]: check["assessment"] = "uncertain"
    elif mutation == "empty_without_reason":
        payload["description_observation_ids"] = []
        payload["limitations"] = []
    with pytest.raises(ValueError, match=match): ObservationOutput.model_validate(payload)


@pytest.mark.parametrize("mutation,match", [
    ("missing_hypothesis", "Every supplied hypothesis"),
    ("unknown_hypothesis", "Every supplied hypothesis"),
    ("unknown_image", "unknown image"),
    ("unreturned_cp", "not returned"),
    ("no_control", "treatment and control"),
    ("wrong_channel", "common channel"),
    ("wrong_sample", "another sample"),
])
def test_evidence_and_complete_hypothesis_coverage(payload, sample, research, mutation, match):
    images = image_manifest(sample, include_metadata=True)
    cp_ids = {"synthetic_control_area", "synthetic_treatment_area"}
    if mutation == "missing_hypothesis": payload["hypothesis_assessments"].pop()
    elif mutation == "unknown_hypothesis": payload["hypothesis_assessments"][0]["hypothesis_id"] = "unknown"
    elif mutation == "unknown_image": payload["observations"][0]["image_ids"].append("unknown")
    elif mutation == "unreturned_cp": cp_ids.remove("synthetic_control_area")
    elif mutation == "no_control": payload["observations"][0]["image_ids"] = ["synthetic_treatment_overview"]
    elif mutation == "wrong_channel": payload["observations"][0]["image_ids"] = ["synthetic_treatment_overview", "synthetic_control_local"]
    elif mutation == "wrong_sample": payload["sample_id"] = "OTHER"
    value = ObservationOutput.model_validate(payload)
    with pytest.raises(ValueError, match=match): validate_observation_references(value, research, images, cp_ids)


def test_abstention_and_description_tampering(payload):
    payload["description_observation_ids"] = []
    result = assemble_result(ObservationOutput.model_validate(payload))
    assert result.description == ""
    saved = result.model_dump(mode="json")
    saved["description"] = "An invented summary claim."
    with pytest.raises(ValueError, match="must equal"): ObservationResult.model_validate(saved)


def test_unexpected_visual_change_can_be_selected_without_a_hypothesis(sample, research, payload):
    payload["observations"].append({**payload["observations"][0], "id": "O4",
        "text": "A subset of cells shows uneven peripheral staining in the synthetic treated views."})
    payload["description_observation_ids"] = ["O4"]
    value = ObservationOutput.model_validate(payload)
    validate_observation_references(value, research, image_manifest(sample, include_metadata=True),
                                    {"synthetic_treatment_area", "synthetic_control_area"})
    assert assemble_result(value).description == payload["observations"][-1]["text"]


def test_field_conflict_is_retained_in_uncertain_assessment(payload):
    item = payload["hypothesis_assessments"][2]
    item["assessment"] = item["checks"][0]["assessment"] = "uncertain"
    item["checks"][0]["reason"] = "Opposite patterns in supplied fields prevent a pooled conclusion."
    item["reason"] = item["checks"][0]["reason"]
    value = ObservationOutput.model_validate(payload)
    assert "Opposite patterns" in value.hypothesis_assessments[2].reason


@pytest.mark.parametrize("sample_id", ["D_R001", "D_R002", "D_R003"])
def test_actual_new_research_handoffs_with_synthetic_image_material(tmp_path, sample_id):
    fixtures = PROJECT / "tests/fixtures/observation_research_v3"
    sample = add_images(load_sample(fixtures / f"{sample_id}.input.json"), tmp_path)
    research = ResearchResultV3.model_validate(read_json(fixtures / f"{sample_id}.research.json"))
    original = research.model_dump(mode="json")
    with pytest.raises(ValueError, match="Observation view contains"):
        prepare_observation(sample, research)
    revision_path = PROJECT / "inputs/observation_handoffs" / f"{sample_id}.json"
    view, audit = prepare_observation(sample, research, revision_path)
    assert len(view.hypotheses) == (2 if sample_id == "D_R003" else 3)
    assert audit["revision"]["research_sha256"] == digest(original)
    assert audit["revision"]["edits"]
    assert view.cp_available is False
    assert research.model_dump(mode="json") == original
    text = view.model_dump_json()
    for private in (sample.compound.name, "mTORC1", "mitotic arrest", "dying", "source_ids", "basis"):
        assert private not in text
    if sample_id == "D_R003":
        assert "fewer or smaller" in view.hypotheses[0].change


@pytest.mark.parametrize("mutation", ["research_hash", "sample", "original", "duplicate"])
def test_handoff_revisions_reject_stale_or_ambiguous_edits(tmp_path, mutation):
    fixtures = PROJECT / "tests/fixtures/observation_research_v3"
    sample = add_images(load_sample(fixtures / "D_R002.input.json"), tmp_path)
    research = ResearchResultV3.model_validate(read_json(fixtures / "D_R002.research.json"))
    revision = read_json(PROJECT / "inputs/observation_handoffs/D_R002.json")
    if mutation == "research_hash": revision["research_sha256"] = "wrong"
    elif mutation == "sample": revision["sample_id"] = "OTHER"
    elif mutation == "original": revision["edits"][0]["original"] = "Changed after review."
    else: revision["edits"].append(deepcopy(revision["edits"][0]))
    with pytest.raises(ValueError):
        prepare_observation(sample, research, revision=HandoffRevision.model_validate(revision))


def test_cp_catalogue_is_metadata_not_exposure_and_reports_real_preprocessing(sample):
    table = CPTable(sample.cp_file, preprocessing=sample.cp_preprocessing)
    catalogue = table.catalogue()
    assert catalogue["groups"][0]["roles"] == ["control", "treatment"]
    assert "synthetic_control_area" not in json.dumps(catalogue)
    assert catalogue["input_preprocessing"] == sample.cp_preprocessing
    page = table.query(limit=1)
    assert page.data["next_offset"] is None
    assert len(page.data["rows"]) == 2
    assert page.data["pagination_unit"] == "feature_comparisons"
    assert page.data["role_totals"] == {"treatment": 1, "control": 1}
    assert page.data["tool_transform"] == "none"
    assert CPTable(sample.cp_file).query().data["input_preprocessing"] == "unknown"


def test_private_cp_labels_are_rejected_before_they_can_enter_a_tool_response(sample, research):
    sample.cp_file.write_text(sample.cp_file.read_text().replace("synthetic_control_area", "PRIVATE_COMPOUND_area"))
    with pytest.raises(ValueError, match="private compound"):
        prepare_observation(sample, research)


def test_image_metadata_includes_dimensions_without_private_paths(sample):
    data = image_manifest(sample, include_metadata=True)
    assert data[0]["width"] == data[0]["height"] == 16
    assert data[0]["well_id"] == "treatment_well"
    assert data[0]["alignment_id"] == "treatment_F1"
    assert "path" not in data[0]
    assert "private-file" not in json.dumps(data)


def test_invalid_crop_and_parent_relationships(sample):
    data = sample.model_dump(mode="json")
    data["images"][1]["parent_image_id"] = data["images"][0]["image_id"]
    with pytest.raises(ValueError, match="share role, channel"):
        SampleInput.model_validate(data)
    data["images"][1]["channel"] = "DNA"
    data["images"][1]["crop_box"] = [10, 10, 10, 10]
    value = SampleInput.model_validate(data)
    with pytest.raises(ValueError, match="exceeds"):
        image_manifest(value, include_metadata=True)


def scripted_result(monkeypatch, payload, *, both_pages=True):
    from realclip_d import observation_pipeline as pipeline
    original = runtime.ScriptedModel
    steps = [[("query_cp", {"offset": 0, "limit": 1,
               **({} if both_pages else {"feature_names": ["not_supplied"]})})]]
    def model(phase, _steps, final):
        if phase == "guided":
            guided = {k: deepcopy(v) for k, v in payload.items() if k != "description_observation_ids"}
            for row in guided['observations']:
                row['comparison_kind'] = 'treatment_control' if row['visually_confirmed'] else 'measurement_only'
            return original(steps, guided)
        if phase == "writer":
            by_id = {row['id']: row for row in payload['observations']}
            return original([], {'sample_id': payload['sample_id'], 'sentences': [
                {'id': f'S{i}', 'text': by_id[oid]['text'], 'observation_ids': [oid]}
                for i, oid in enumerate(payload['description_observation_ids'], 1)], 'limitations': []})
        if phase == "reviewer":
            by_id = {row['id']: row for row in payload['observations']}
            return original([], {'sample_id': payload['sample_id'], 'reviews': [
                {'sentence_id': f'S{i}', 'action': 'keep', 'text': by_id[oid]['text'],
                 'observation_ids': [oid], 'reason': 'Synthetic transport fixture only.'}
                for i, oid in enumerate(payload['description_observation_ids'], 1)], 'limitations': []})
        return original(_steps, final)
    monkeypatch.setattr(pipeline, "mock_model_for_phase", model)


@pytest.mark.asyncio
async def test_sdk_tool_loop_partial_support_evidence_and_sentence_assembly(
    sample, research, payload, tmp_path, monkeypatch,
):
    scripted_result(monkeypatch, payload)
    directory = tmp_path / "observation"
    result = await _describe(sample, research, Settings(), directory, "mock", None)
    assert result.description == " ".join(o["text"] for o in payload["observations"][:2])
    assert result.hypothesis_assessments[0].assessment == "partially_supported"
    receipt = verify_stage(directory)
    assert {"handoff.json", "output_contract.json", "candidate_output.json"} <= set(receipt["files"])
    assert read_json(directory / "evidence.json")["cp_feature_ids"] == ["synthetic_control_area", "synthetic_treatment_area"]
    request = read_json(directory / "phases/guided/requests/001.json")
    assert [tool["name"] for tool in request["tools"]] == ["inspect_image", "query_cp"]
    assert request["input"][0]["role"] == "user"
    assert "input_image" in json.dumps(request)
    assert "PRIVATE_COMPOUND" not in json.dumps(request)
    assert "private-file" not in json.dumps(request)
    assert set(request["output_schema"]["properties"]) == {
        'sample_id', 'observations', 'hypothesis_assessments', 'limitations'}
    assert read_json(directory / "validation.json")["hypothesis_coverage_valid"] is True


@pytest.mark.asyncio
async def test_unreturned_cp_is_rejected_after_raw_candidate_is_preserved(sample, research, payload, tmp_path, monkeypatch):
    scripted_result(monkeypatch, payload, both_pages=False)
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError, match="not returned"):
        await _describe(sample, research, Settings(), directory, "mock", None)
    assert (directory / "phases/guided/responses/002.json").exists()
    assert (directory / "candidate_output.json").exists()
    assert not (directory / "result.json").exists()
    assert not (directory / "complete.json").exists()


@pytest.mark.asyncio
async def test_missing_images_and_private_handoff_stop_before_client(sample, research, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail("Preflight must happen before any paid client")
    monkeypatch.setattr(runtime, "_live_model", forbidden)
    view, _ = prepare_observation(sample, research)
    view.hypotheses[0].how_to_compare = "mTORC1 inhibition"
    with pytest.raises(ValueError, match="research/biological"):
        await runtime.run_observation(sample, view, Settings(budget_usd=1), tmp_path / "private", mode="live")
    sample.images = []
    with pytest.raises(ValueError, match="requires treatment and control images"):
        prepare_observation(sample, research)


@pytest.mark.asyncio
async def test_missing_cp_mock_abstains_for_every_real_new_hypothesis(tmp_path):
    fixtures = PROJECT / "tests/fixtures/observation_research_v3"
    sample = add_images(load_sample(fixtures / "D_R003.input.json"), tmp_path)
    research = ResearchResultV3.model_validate(read_json(fixtures / "D_R003.research.json"))
    result = await _describe(sample, research, Settings(), tmp_path / "no-cp", "mock", None,
                             PROJECT / "inputs/observation_handoffs/D_R003.json")
    assert result.description == ""
    assert len(result.hypothesis_assessments) == 2
    assert {h.assessment for h in result.hypothesis_assessments} == {"uncertain"}
    assert "SYNTHETIC" in result.limitations[0]


@pytest.mark.parametrize("text", ["PRIVATE_COMPOUND causes a change.", "mTORC1 inhibition is visible.",
                                  "The cells are apoptotic.", "The dying cells are round."])
def test_private_and_mechanistic_description_guard(sample, text):
    with pytest.raises(ValueError): check_observation_text(text, sample)
