"""Offline tests for Summarize's read-only source and image-identity boundary."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import socket

import pytest

from realclip_d.observation_contracts import ObservationRequest
from realclip_d.observation_stages import ObservationPipelineResult
from realclip_d.storage import seal_stage, write_json
from realclip_d.summarize_contracts import SummarizeConstraints
from realclip_d.summarize_inputs import build_request, make_provisional_manifest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Summarize input tests must never access the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def image(key, role="treatment", channel="DNA", field="1"):
    return {"image_id": key, "role": role, "well_id": "W1" if role == "treatment" else "CW1",
            "channel": channel, "field": field, "view": "full_field", "width": 16, "height": 16,
            "sha256": hashlib.sha256(key.encode()).hexdigest(), "parent_image_id": None,
            "crop_box": None, "alignment_id": None, "stain": "Synthetic test channel.",
            "display_settings": "Synthetic byte-preserved view.", "acquisition_settings": None,
            "path": "/PRIVATE_LOCAL_PATH/SECRET_COMPOUND.png"}


def observation(key, image_ids, kind="treatment_only", text="Oval profiles are present."):
    visual = kind != "measurement_only"
    return {"id": key, "text": text, "scope": "Synthetic supplied field only.",
            "image_ids": image_ids, "cp_feature_ids": [] if visual else ["CP_PRIVATE"],
            "visually_confirmed": visual, "comparison_kind": kind,
            "cp_assessment": {"status": "not_checked" if visual else "measurement_only",
                              "reason": "Synthetic fixture only.",
                              "effect_on_description": "Retain the supplied scope."}}


@pytest.fixture
def case(tmp_path):
    images = [image("tDNA1"), image("tDNA2", field="2"), image("tAGP", channel="AGP"),
              image("tBF", channel="Brightfield_L"), image("cDNA", "control"),
              image("cAGP", "control", channel="AGP")]
    view = ObservationRequest(sample_id="SYNTHETIC_CASE", images=images, cp_available=True,
        context={"cell_line": "U2OS", "dose": "PRIVATE_DOSE", "notes": ["SECRET_COMPOUND"]},
        hypotheses=[{"hypothesis_id": "H1", "change": "PRIVATE_HYPOTHESIS", "applicability": "Test.",
                     "where_to_look": "Test.", "what_to_check": "Test.", "how_to_compare": "Test."}],
        cp_catalogue={"PRIVATE_CP_VALUE": 12345})
    observations = [observation("O1", ["tDNA1"]), observation("O2", ["tDNA2"], text="Sparse fields are present."),
                    observation("O3", ["cDNA"], "control_only", "Control fields are dense."),
                    observation("O4", ["tBF"], text="Occasional thin extensions are present."),
                    observation("O5", ["tDNA1"], text="PRIVATE_UNSELECTED_FINDING"),
                    observation("O6", ["tAGP", "cAGP"], "treatment_control", "Both roles show linear staining."),
                    observation("O7", [], "measurement_only", "PRIVATE_CP_ONLY_FINDING")]
    sentences = [{"id": "S1", "text": "Oval profiles are present.", "observation_ids": ["O1"]},
                 {"id": "S2", "text": "Sparse fields are present.", "observation_ids": ["O2"]},
                 {"id": "S3", "text": "Control fields are dense.", "observation_ids": ["O3"]},
                 {"id": "S4", "text": "Occasional thin extensions are present.", "observation_ids": ["O4"]},
                 {"id": "S5", "text": "Both roles show linear staining.", "observation_ids": ["O6"]}]
    result = ObservationPipelineResult(sample_id=view.sample_id, observations=observations,
        hypothesis_assessments=[], description_sentences=sentences,
        description_observation_ids=["O1", "O2", "O3", "O4", "O6"],
        description=" ".join(row["text"] for row in sentences), limitations=["Synthetic fixture only."])
    for name, value in {"input.json": view.model_dump(mode="json"), "result.json": result.model_dump(mode="json"),
                        "settings.json": {}, "sources.json": [], "binding.json": {}}.items():
        write_json(tmp_path / name, value)
    seal_stage(tmp_path)
    return tmp_path


def request_for(case, channels=None, constraints=None, request_version="2"):
    manifest = make_provisional_manifest(case, channels if channels is not None else ["DNA", "AGP"])
    return build_request(case, manifest, constraints or SummarizeConstraints(), request_version=request_version)


def reseal(case):
    (case / "complete.json").unlink()
    seal_stage(case)


def test_legacy_projection_excludes_private_fields_and_unselected_findings(case):
    request = request_for(case, request_version="1")
    encoded = json.dumps(request)
    for marker in ("SECRET_COMPOUND", "PRIVATE_LOCAL_PATH", "PRIVATE_DOSE", "PRIVATE_HYPOTHESIS",
                   "PRIVATE_CP_VALUE", "PRIVATE_UNSELECTED_FINDING", "PRIVATE_CP_ONLY_FINDING", "CP_PRIVATE"):
        assert marker not in encoded
    assert set(request) == {"shared_task_context", "target_treatment_well", "description",
                            "selected_observations", "clip_visual_manifest", "limitations", "constraints"}
    assert request["description"] == json.loads((case / "result.json").read_text())["description"]
    assert "description_sentences" not in request
    assert [row["id"] for row in request["selected_observations"]] == ["O1", "O2", "O3", "O4", "O6"]
    selected = {row["id"]: row for row in request["selected_observations"]}
    assert selected["O4"]["eligible_treatment_image_ids"] == []
    assert selected["O3"]["image_evidence"][0]["role"] == "control"
    assert request["clip_visual_manifest"]["status"] == "provisional"
    assert all(row["visibility"] == "source_native" for row in request["clip_visual_manifest"]["images"])


def test_target_projection_excludes_unsupported_sources_and_raw_description(case):
    request = request_for(case)
    encoded = json.dumps(request)
    for marker in ("SECRET_COMPOUND", "PRIVATE_LOCAL_PATH", "PRIVATE_DOSE", "PRIVATE_HYPOTHESIS",
                   "PRIVATE_CP_VALUE", "PRIVATE_UNSELECTED_FINDING", "PRIVATE_CP_ONLY_FINDING", "CP_PRIVATE",
                   "Occasional thin extensions are present.", "Control fields are dense."):
        assert marker not in encoded
    assert set(request) == {"request_version", "shared_task_context", "target_treatment_well",
                            "reviewed_description_context", "eligible_observations", "clip_visual_manifest",
                            "limitations", "constraints"}
    assert request["request_version"] == "2"
    assert "description" not in request
    assert "selected_observations" not in request
    assert [row["id"] for row in request["eligible_observations"]] == ["O1", "O2", "O6"]
    assert [row["sentence_id"] for row in request["reviewed_description_context"]] == ["S1", "S2", "S5"]
    assert all(row["eligibility"] == "fully_eligible" for row in request["reviewed_description_context"])
    assert all(row["use"] == "only_clauses_supported_by_eligible_observations"
               for row in request["reviewed_description_context"])
    for row in request["eligible_observations"]:
        assert row["image_ids"]
        assert all(image["role"] == "treatment" for image in row["image_evidence"])
        assert all(image["channel"] in {"DNA", "AGP"} for image in row["image_evidence"])
        assert "cp_feature_ids" not in row


def test_target_projection_preserves_field_scope_without_inventing_prevalence(case):
    request = request_for(case)
    row = next(row for row in request["eligible_observations"] if row["id"] == "O1")
    assert row["scope"] == "Synthetic supplied field only."
    assert row["target_scope"]["observed_fields"] == ["1"]
    assert row["target_scope"]["available_fields"] == ["1", "2"]
    assert row["target_scope"]["observed_fields_by_channel"] == {"DNA": ["1"]}
    assert row["target_scope"]["available_fields_by_channel"]["DNA"] == ["1", "2"]
    assert row["target_scope"]["views"] == ["full_field"]
    assert row["target_scope"]["sampling_limit"]
    assert row["reviewed_sentence_ids"] == ["S1"]


def test_comparison_observation_keeps_only_treatment_evidence_for_absolute_rewrite(case):
    request = request_for(case)
    row = next(row for row in request["eligible_observations"] if row["id"] == "O6")
    assert row["comparison_kind"] == "treatment_control"
    assert row["text"] == "Both roles show linear staining."
    assert row["image_ids"] == ["tAGP"]
    assert [image["image_id"] for image in row["image_evidence"]] == ["tAGP"]
    assert "cAGP" not in json.dumps(row)
    assert row["target_scope"]["observed_fields_by_channel"] == {"AGP": ["1"]}


def update_result(case, mutate):
    path = case / "result.json"
    payload = json.loads(path.read_text())
    mutate(payload)
    payload["description"] = " ".join(row["text"] for row in payload["description_sentences"])
    payload["description_observation_ids"] = list(dict.fromkeys(
        key for row in payload["description_sentences"] for key in row["observation_ids"]))
    write_json(path, payload)
    reseal(case)


def test_mixed_sentence_is_partial_context_and_never_wholesale_eligible(case):
    text = "Oval DNA profiles are present, and thin extensions are seen in Brightfield_L."

    def mutate(result):
        result["description_sentences"] = [{"id": "S_mixed", "text": text,
                                             "observation_ids": ["O1", "O4"]}]

    update_result(case, mutate)
    request = request_for(case)
    assert "description" not in request
    assert [row["id"] for row in request["eligible_observations"]] == ["O1"]
    assert request["reviewed_description_context"] == [{
        "sentence_id": "S_mixed", "text": text, "eligible_observation_ids": ["O1"],
        "eligibility": "partial", "use": "only_clauses_supported_by_eligible_observations"}]


def test_observation_with_unselected_treatment_channel_is_not_partly_admitted(case):
    marker = "PRIVATE_MIXED_BF_APPEARANCE"

    def mutate(result):
        result["observations"][0]["image_ids"] = ["tDNA1", "tBF"]
        result["observations"][0]["text"] = marker
        result["description_sentences"][0]["text"] = marker

    update_result(case, mutate)
    request = request_for(case)
    assert "O1" not in {row["id"] for row in request["eligible_observations"]}
    assert marker not in json.dumps(request)


def test_outside_manifest_field_is_excluded_even_when_channel_matches(case):
    manifest = make_provisional_manifest(case, ["DNA"])
    manifest["images"] = [row for row in manifest["images"] if row["field"] == "1"]
    request = build_request(case, manifest, SummarizeConstraints())
    assert [row["id"] for row in request["eligible_observations"]] == ["O1"]
    assert "Sparse fields are present." not in json.dumps(request)
    assert request["eligible_observations"][0]["target_scope"]["available_fields"] == ["1"]


def test_no_eligible_reviewed_content_does_not_import_unselected_observations(case):
    def mutate(result):
        result["description_sentences"] = [{"id": "S_BF", "text": "BF-only outlines.",
                                             "observation_ids": ["O4"]}]

    update_result(case, mutate)
    request = request_for(case)
    assert request["eligible_observations"] == []
    assert request["reviewed_description_context"] == []
    assert "PRIVATE_UNSELECTED_FINDING" not in json.dumps(request)


@pytest.mark.parametrize("mutation", ["role", "well", "hash", "source_hash", "source", "mapping", "dimension"])
def test_manifest_identity_and_lineage_are_checked(case, mutation):
    manifest = make_provisional_manifest(case, ["DNA"])
    first = manifest["images"][0]
    if mutation == "role":
        first["role"] = "control"
    elif mutation == "well":
        first["well_id"] = "OTHER_WELL"
    elif mutation == "hash":
        first["sha256"] = "a" * 64
    elif mutation == "source_hash":
        first["source_sha256"] = "b" * 64
    elif mutation == "source":
        first["source_image_ids"] = ["missing_source"]
    elif mutation == "mapping":
        first["source_image_ids"] = ["tAGP"]
    else:
        first["width"] = 4
    with pytest.raises(ValueError):
        build_request(case, manifest, SummarizeConstraints())


@pytest.mark.parametrize("channels", [[], ["DNA", "DNA"], ["MISSING"], [" "]])
def test_no_implicit_channel_selection(case, channels):
    with pytest.raises(ValueError):
        make_provisional_manifest(case, channels)


def test_changed_source_artifact_cannot_be_used(case):
    with (case / "result.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="missing or changed"):
        make_provisional_manifest(case, ["DNA"])


@pytest.mark.parametrize("well", [None, "OTHER_TREATMENT_WELL"])
def test_ambiguous_or_unknown_treatment_well_stops_preflight(case, well):
    path = case / "input.json"
    payload = json.loads(path.read_text())
    payload["images"][1]["well_id"] = well
    write_json(path, payload)
    reseal(case)
    with pytest.raises(ValueError, match="exactly one treatment well"):
        make_provisional_manifest(case, ["DNA"])


def test_local_tokenizer_path_stays_outside_model_request(case):
    constraints = SummarizeConstraints(tokenizer_path="/PRIVATE_TOKENIZER_PATH/tokenizer.json", max_text_tokens=77)
    request = request_for(case, constraints=constraints)
    assert "PRIVATE_TOKENIZER_PATH" not in json.dumps(request)
    assert request["constraints"]["tokenizer_configured"] is True
    assert request["constraints"]["max_text_tokens"] == 77


def test_unbound_or_partial_source_is_rejected(case):
    receipt_path = case / "complete.json"
    receipt = json.loads(receipt_path.read_text())
    del receipt["files"]["result.json"]
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="does not bind result"):
        make_provisional_manifest(case, ["DNA"])


def add_saved_crop(case):
    from PIL import Image

    originals = json.loads((case / "input.json").read_text())["images"]
    parent = originals[0]
    crop = {key: value for key, value in parent.items() if key != "path"}
    crop.update(image_id="crop_1", parent_image_id=parent["image_id"], crop_box=[0, 0, 8, 8],
                width=8, height=8, view="local_crop", parent_sha256=parent["sha256"],
                relative_path="crop_1.png")
    path = case / "crops" / "crop_1.png"
    path.parent.mkdir()
    Image.new("L", (8, 8), 100).save(path)
    crop["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(case / "crops" / "manifest.json", {"schema_version": "1", "originals": originals, "crops": [crop]})
    result = json.loads((case / "result.json").read_text())
    result["observations"][0]["image_ids"] = ["crop_1"]
    write_json(case / "result.json", result)
    reseal(case)
    return crop


def test_full_field_does_not_substitute_for_excluded_crop_evidence(case):
    add_saved_crop(case)
    request = request_for(case)
    assert "O1" not in {row["id"] for row in request["eligible_observations"]}
    assert "Oval profiles are present." not in json.dumps(request)


def test_explicit_byte_identical_crop_can_map_to_its_own_view(case):
    crop = add_saved_crop(case)
    manifest = make_provisional_manifest(case, ["DNA"])
    crop_view = deepcopy(manifest["images"][0])
    for key in ("width", "height", "view", "sha256", "parent_image_id", "crop_box"):
        crop_view[key] = crop[key]
    crop_view.update(image_id="clip_crop_alias", source_image_ids=["crop_1"], source_sha256=crop["sha256"])
    manifest["images"].append(crop_view)
    request = build_request(case, manifest, SummarizeConstraints())
    row = next(row for row in request["eligible_observations"] if row["id"] == "O1")
    assert row["eligible_treatment_image_ids"] == ["clip_crop_alias"]
    assert row["image_ids"] == ["crop_1"]
    assert row["image_evidence"][0]["image_id"] == "crop_1"
    assert row["target_scope"]["views"] == ["local_crop"]
    assert row["target_scope"]["observed_fields"] == ["1"]
    assert row["target_scope"]["available_fields"] == ["1", "2"]
    assert row["scope"] == "Synthetic supplied field only."


def test_unknown_request_version_is_rejected_before_model_input_is_built(case):
    with pytest.raises(ValueError, match="version"):
        request_for(case, request_version="future")


def test_saved_crop_parent_hash_is_checked_after_receipt_verification(case):
    add_saved_crop(case)
    path = case / "crops" / "manifest.json"
    payload = json.loads(path.read_text())
    payload["crops"][0]["parent_sha256"] = "a" * 64
    write_json(path, payload)
    reseal(case)
    with pytest.raises(ValueError, match="parent hash"):
        make_provisional_manifest(case, ["DNA"])
