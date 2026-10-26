"""Empty CP queries retain image evidence without inventing measurement support."""
from copy import deepcopy
import csv

import pytest

from realclip_d.observation_contracts import AppearanceObservation, ObservationRequest
from realclip_d.observation_stages import (
    BaselineOutput, DescriptionDraft, DescriptionReview, GuidanceEvidenceOutput,
    apply_review, validate_baseline, validate_evidence, validate_pipeline_references,
)
from realclip_d.tools.cp import CPTable


def observation(identifier="O2", *, status="no_matching_measurement", cp_ids=None):
    return dict(id=identifier, text="Synthetic treatment fields have more open regions than control fields.",
        scope="Supplied synthetic fields; coverage is not a count of individual cells.",
        comparison_kind="treatment_control", image_ids=["T1", "C1"],
        cp_feature_ids=cp_ids or [], visually_confirmed=True,
        cp_assessment=dict(status=status,
            reason="A query for a total-count measurement matched no supplied features.",
            effect_on_description="Retain the image-based coverage comparison without a quantitative count claim."))


@pytest.fixture
def material(tmp_path):
    path = tmp_path / "cp.csv"
    rows = [dict(feature_id=f"{role}_Area", feature_name="Cells_AreaShape_Area",
        compartment="Cells", channel="AGP", family="AreaShape", role=role,
        value=value, unit="unknown", scope="well")
        for role, value in (("treatment", 2), ("control", 1))]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = CPTable(path)
    images = [dict(image_id=identifier, role=role, channel="AGP", field="1")
              for identifier, role in (("T1", "treatment"), ("C1", "control"))]
    view = ObservationRequest(sample_id="S1", context={}, hypotheses=[], images=images,
        cp_available=True, cp_catalogue=table.catalogue())
    arguments = {"query": "TotalCount"}
    result = table.query(**arguments).model_dump(mode="json")
    assert result["status"] == "not_found" and result["data"]["total"] == 0
    receipt = dict(tool="query_cp", arguments=[], keyword_arguments=arguments,
                   result=result, model_result=deepcopy(result))
    return table, view, images, receipt


def evidence():
    return GuidanceEvidenceOutput(sample_id="S1", observations=[observation()],
        hypothesis_assessments=[], limitations=[])


def test_empty_query_status_can_continue_through_writer_review_and_final_validation(material):
    table, view, images, receipt = material
    baseline = BaselineOutput(sample_id="S1", observations=[observation("O1", status="not_checked")],
        reviewed_image_ids=["T1", "C1"], limitations=[])
    validate_baseline(baseline, view, images)
    output = GuidanceEvidenceOutput(sample_id="S1",
        observations=[*baseline.observations, observation()], hypothesis_assessments=[], limitations=[])
    validate_evidence(output, view, images, set(), table, baseline, cp_query_results=[receipt])
    draft = DescriptionDraft(sample_id="S1", sentences=[dict(id="SENT1",
        text="Synthetic treatment fields have more open regions than control fields.",
        observation_ids=["O2"])], limitations=[])
    review = DescriptionReview(sample_id="S1", reviews=[dict(sentence_id="SENT1", action="keep",
        text=draft.sentences[0].text, observation_ids=["O2"], reason="Synthetic visual evidence only.")],
        limitations=[])
    final = apply_review(draft, review, output)
    validate_pipeline_references(final, view, images, set(), table, cp_query_results=[receipt])
    assert final.description == draft.sentences[0].text
    assert final.observations[0] == baseline.observations[0]
    assert final.observations[1].cp_feature_ids == []


@pytest.mark.parametrize("receipts", [None, []])
def test_model_claim_without_actual_query_receipt_is_rejected(material, receipts):
    table, view, images, _ = material
    with pytest.raises(ValueError, match="actual query_cp zero-match receipt"):
        validate_evidence(evidence(), view, images, set(), table, cp_query_results=receipts)


def test_empty_query_receipt_without_bound_table_is_rejected(material):
    _, view, images, receipt = material
    with pytest.raises(ValueError, match="actual query_cp zero-match receipt"):
        validate_evidence(evidence(), view, images, set(), cp_query_results=[receipt])


def test_explicit_empty_feature_selection_is_not_a_missing_measurement(material):
    table, view, images, _ = material
    assert table.rows  # Real measurements exist; choosing no names exposes none.
    arguments = {"feature_names": []}
    result = table.query(**arguments).model_dump(mode="json")
    receipt = dict(tool="query_cp", arguments=[], keyword_arguments=arguments,
                   result=result, model_result=deepcopy(result))
    with pytest.raises(ValueError, match="actual query_cp zero-match receipt"):
        validate_evidence(evidence(), view, images, set(), table, cp_query_results=[receipt])


def test_empty_feature_selection_diagnostic_cannot_authorize_missing_measurement(material):
    table, view, images, original = material
    receipt = deepcopy(original)
    receipt["result"]["data"].setdefault("query_diagnostic", {})["outcome"] = "empty_feature_selection"
    receipt["model_result"] = deepcopy(receipt["result"])
    with pytest.raises(ValueError, match="actual query_cp zero-match receipt"):
        validate_evidence(evidence(), view, images, set(), table, cp_query_results=[receipt])


@pytest.mark.parametrize("change", [
    "wrong_tool", "no_tool", "error", "unavailable", "ok", "remaining_matches", "missing_total",
    "boolean_total", "nonempty_rows", "nonempty_comparisons", "nonzero_total_rows",
    "not_exposed_to_model", "missing_arguments", "positional_arguments", "invalid_arguments",
    "forged_no_match", "empty_page",
])
def test_only_verified_zero_match_queries_authorize_empty_state(material, change):
    table, view, images, original = material
    receipt = deepcopy(original)
    if change == "wrong_tool": receipt["tool"] = "inspect_image"
    elif change == "no_tool": receipt.pop("tool")
    elif change in {"error", "unavailable", "ok"}: receipt["result"]["status"] = change
    elif change == "remaining_matches": receipt["result"]["data"]["total"] = 1
    elif change == "missing_total": receipt["result"]["data"].pop("total")
    elif change == "boolean_total": receipt["result"]["data"]["total"] = False
    elif change == "nonempty_rows": receipt["result"]["data"]["rows"] = [{"feature_id": "CP1"}]
    elif change == "nonempty_comparisons": receipt["result"]["data"]["comparisons"] = [{}]
    elif change == "nonzero_total_rows": receipt["result"]["data"]["total_rows"] = 1
    elif change == "not_exposed_to_model": receipt["model_result"]["status"] = "error"
    elif change == "missing_arguments": receipt.pop("keyword_arguments")
    elif change == "positional_arguments": receipt["arguments"] = ["Cells"]
    elif change == "invalid_arguments": receipt["keyword_arguments"] = {"not_a_query_parameter": "X"}
    elif change == "forged_no_match": receipt["keyword_arguments"] = {"query": "Area"}
    elif change == "empty_page":
        receipt["keyword_arguments"] = {"query": "Area", "offset": 100}
        receipt["result"] = table.query(**receipt["keyword_arguments"]).model_dump(mode="json")
        receipt["model_result"] = deepcopy(receipt["result"])
        assert receipt["result"]["data"]["rows"] == []
        assert receipt["result"]["data"]["total"] > 0
    with pytest.raises(ValueError, match="actual query_cp zero-match receipt"):
        validate_evidence(evidence(), view, images, set(), table, cp_query_results=[receipt])


def test_empty_status_cannot_hide_absent_cp(material):
    table, view, images, receipt = material
    no_cp = view.model_copy(update={"cp_available": False})
    with pytest.raises(ValueError, match="without CP requires not_available"):
        validate_evidence(evidence(), no_cp, images, set(), table, cp_query_results=[receipt])
    # A forged empty receipt cannot turn an unavailable query into a genuine zero match.
    with pytest.raises(ValueError, match="actual query_cp zero-match receipt"):
        validate_evidence(evidence(), view, images, set(), CPTable(None), cp_query_results=[receipt])


def test_baseline_cannot_imply_a_query_happened(material):
    _, view, images, _ = material
    baseline = BaselineOutput(sample_id="S1", observations=[observation()],
        reviewed_image_ids=["T1", "C1"], limitations=[])
    with pytest.raises(ValueError, match="Baseline CP status must be not_checked"):
        validate_baseline(baseline, view, images)


def test_no_match_remains_visual_only_and_cannot_cite_measurements():
    row = observation()
    row.pop("comparison_kind")
    AppearanceObservation.model_validate(row)
    with pytest.raises(ValueError, match="cannot cite measurement evidence"):
        AppearanceObservation.model_validate({**row, "cp_feature_ids": ["treatment_Area"]})
    with pytest.raises(ValueError, match="Measurement-only observations require measurement_only"):
        AppearanceObservation.model_validate({**row, "visually_confirmed": False})


@pytest.mark.parametrize("status", ["not_informative", "supports", "contradicts", "mixed", "measurement_only"])
def test_existing_measurement_statuses_still_require_feature_references(status):
    row = observation(status=status)
    row.pop("comparison_kind")
    with pytest.raises(ValueError, match="requires CP feature references"):
        AppearanceObservation.model_validate(row)


def test_actual_noninformative_measurements_still_require_exposed_ids(material):
    table, view, images, _ = material
    ids = ["treatment_Area", "control_Area"]
    output = GuidanceEvidenceOutput(sample_id="S1", observations=[observation(status="not_informative", cp_ids=ids)],
        hypothesis_assessments=[], limitations=[])
    with pytest.raises(ValueError, match="not returned to this agent"):
        validate_evidence(output, view, images, set(), table)
    validate_evidence(output, view, images, set(ids), table)
