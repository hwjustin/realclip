"""Offline stage boundaries: coverage, evidence retention, and reviewed prose."""
from types import SimpleNamespace

from agents import AgentOutputSchema
import pytest

from realclip_d.observation_contracts import ObservationOutput, ObservationRequest, validate_observation_references
from realclip_d.observation_stages import (
    BaselineOutput, DescriptionDraft, DescriptionReview, DescriptionSentence,
    GuidanceEvidenceOutput, ObservationPipelineResult, PipelineAppearanceObservation, apply_review,
    validate_baseline, validate_draft, validate_evidence, validate_pipeline_references,
)


def visual(identifier="O1", status="not_checked", cp_ids=None):
    return dict(id=identifier, text="Synthetic rounded profiles occur in a subset.",
        scope="Synthetic paired fields; no well-wide prevalence estimate.",
        comparison_kind="treatment_control",
        image_ids=["T1", "C1"], cp_feature_ids=cp_ids or [], visually_confirmed=True,
        cp_assessment=dict(status=status, reason="Synthetic fixture only.",
            effect_on_description="Keep the subset qualification."))


@pytest.fixture
def stages():
    images = [dict(image_id=identifier, channel="DNA", role=role, field="1")
              for identifier, role in (("T1", "treatment"), ("C1", "control"))]
    view = ObservationRequest(sample_id="S1", context={}, images=images,
        cp_available=True, cp_catalogue={}, hypotheses=[dict(hypothesis_id="H1",
            change="Synthetic rounding.", applicability="Synthetic.", where_to_look="DNA.",
            what_to_check="Contours.", how_to_compare="Across synthetic roles.")])
    baseline = BaselineOutput(sample_id="S1", observations=[visual()],
        reviewed_image_ids=["T1", "C1"], limitations=["Synthetic example."])
    assessment = dict(hypothesis_id="H1", assessment="uncertain", reason="Synthetic uncertainty.",
        checks=[dict(check="Synthetic contours.", assessment="uncertain", observation_ids=["O1"],
            reason="No biological evaluation.", contrast=dict(treatment="Synthetic subset.",
                control="Synthetic subset.", difference="unassessable"))])
    evidence = GuidanceEvidenceOutput(sample_id="S1", observations=[visual(), visual("O2")],
        hypothesis_assessments=[assessment], limitations=["Synthetic example."])
    draft = DescriptionDraft(sample_id="S1", sentences=[
        dict(id="SENT1", text="Synthetic profiles vary.", observation_ids=["O2", "O1"]),
        dict(id="SENT2", text="A synthetic subset is rounded.", observation_ids=["O1"])],
        limitations=["Synthetic example.", "Draft limitation."])
    review = DescriptionReview(sample_id="S1", reviews=[
        dict(sentence_id="SENT2", action="revise", text="A synthetic subset remains compact.",
            observation_ids=["O2"], reason="Preserve the subset scope."),
        dict(sentence_id="SENT1", action="keep", text="Synthetic profiles vary.",
            observation_ids=["O2", "O1"], reason="Retain the evidence-linked summary.")],
        limitations=["Review limitation."])
    return SimpleNamespace(view=view, images=images, baseline=baseline,
                           evidence=evidence, draft=draft, review=review)


def test_strict_stage_schemas_keep_hypotheses_out_of_baseline(stages):
    schema = AgentOutputSchema(BaselineOutput).json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {
        "sample_id", "observations", "reviewed_image_ids", "limitations"}
    assert "HypothesisAssessment" not in schema.get("$defs", {})
    with pytest.raises(ValueError, match="Extra inputs"):
        BaselineOutput.model_validate({**stages.baseline.model_dump(), "hypothesis_assessments": []})
    for contract in (GuidanceEvidenceOutput, DescriptionDraft, DescriptionReview):
        AgentOutputSchema(contract)


def test_baseline_requires_all_original_images_without_duplicates(stages):
    validate_baseline(stages.baseline, stages.view, stages.images, set())
    bad = stages.baseline.model_copy(deep=True)
    bad.reviewed_image_ids = ["T1"]
    with pytest.raises(ValueError, match="every original"):
        validate_baseline(bad, stages.view, stages.images, set())
    bad.reviewed_image_ids = ["T1", "C1", "T1"]
    with pytest.raises(ValueError, match="must be unique"):
        validate_baseline(bad, stages.view, stages.images, set())
    bad.reviewed_image_ids = ["T1", "C1", "CROP1"]
    with pytest.raises(ValueError, match="every original"):
        validate_baseline(bad, stages.view, stages.images, set())


def test_baseline_cannot_query_or_cite_cp_and_handles_absent_cp(stages):
    with pytest.raises(ValueError, match="cannot receive queried CP"):
        validate_baseline(stages.baseline, stages.view, stages.images, {"CP1"})
    with pytest.raises(ValueError, match="cannot cite CP"):
        BaselineOutput(sample_id="S1", observations=[visual(status="not_informative", cp_ids=["CP1"])],
            reviewed_image_ids=["T1", "C1"], limitations=[])
    view = stages.view.model_copy(update={"cp_available": False})
    with pytest.raises(ValueError, match="not_available"):
        validate_baseline(stages.baseline, view, stages.images, set())
    stages.baseline.observations[0].cp_assessment.status = "not_available"
    validate_baseline(stages.baseline, view, stages.images, set())


def test_baseline_still_checks_sample_images_and_treatment_control(stages):
    for mutation, message in (("sample", "another sample"), ("image", "unknown image"),
                              ("control", "treatment and control")):
        bad = stages.baseline.model_copy(deep=True)
        if mutation == "sample": bad.sample_id = "OTHER"
        elif mutation == "image": bad.observations[0].image_ids.append("UNKNOWN")
        else: bad.observations[0].image_ids = ["T1"]
        with pytest.raises(ValueError, match=message):
            validate_baseline(bad, stages.view, stages.images, set())


@pytest.mark.parametrize("change", ["text", "scope", "images", "cp", "kind", "delete", "reorder"])
def test_guidance_cannot_rewrite_or_lose_baseline_observations(stages, change):
    bad = stages.evidence.model_copy(deep=True)
    if change == "text": bad.observations[0].text = "A contradictory replacement."
    elif change == "scope": bad.observations[0].scope = "All wells."
    elif change == "images": bad.observations[0].image_ids.reverse()
    elif change == "cp": bad.observations[0].cp_assessment.reason = "Different reason."
    elif change == "kind": bad.observations[0].comparison_kind = "treatment_only"
    elif change == "delete":
        bad.observations.pop(0)
        bad.hypothesis_assessments[0].checks[0].observation_ids = ["O2"]
    else: bad.observations.reverse()
    with pytest.raises(ValueError, match="baseline observation prefix unchanged"):
        validate_evidence(bad, stages.view, stages.images, set(), baseline=stages.baseline)


def test_guidance_preserves_baseline_and_requires_complete_hypothesis_coverage(stages):
    validate_evidence(stages.evidence, stages.view, stages.images, set(), baseline=stages.baseline)
    bad = stages.evidence.model_copy(deep=True)
    bad.hypothesis_assessments = []
    with pytest.raises(ValueError, match="Every supplied hypothesis"):
        validate_evidence(bad, stages.view, stages.images, set(), baseline=stages.baseline)
    no_hypotheses = stages.view.model_copy(update={"hypotheses": []})
    validate_evidence(bad, no_hypotheses, stages.images, set(), baseline=stages.baseline)


def test_guidance_reuses_exposed_pair_and_cp_readiness_requirements(stages):
    stages.evidence.observations[1] = stages.evidence.observations[1].model_validate(
        visual("O2", "supports", ["T_AREA", "C_AREA"]))
    pair = dict(treatment_feature_id="T_AREA", control_feature_id="C_AREA", comparison_ready=True)
    table = SimpleNamespace(comparisons=lambda: [pair])
    with pytest.raises(ValueError, match="not returned"):
        validate_evidence(stages.evidence, stages.view, stages.images, {"T_AREA"}, table)
    pair["comparison_ready"] = False
    with pytest.raises(ValueError, match="interpretation gaps"):
        validate_evidence(stages.evidence, stages.view, stages.images, {"T_AREA", "C_AREA"}, table)
    pair["comparison_ready"] = True
    validate_evidence(stages.evidence, stages.view, stages.images, {"T_AREA", "C_AREA"}, table)


def test_draft_allows_rewriting_and_merging_only_existing_visual_evidence(stages):
    validate_draft(stages.draft, stages.evidence)
    for refs in (["UNKNOWN"], ["CP_ONLY"]):
        bad = stages.draft.model_copy(deep=True)
        bad.sentences[0].observation_ids = refs
        measurement = visual("CP_ONLY", "measurement_only", ["CP1"])
        measurement.update(visually_confirmed=False, image_ids=[], comparison_kind="measurement_only")
        evidence = GuidanceEvidenceOutput.model_validate({**stages.evidence.model_dump(),
            "observations": [*stages.evidence.model_dump()["observations"], measurement]})
        with pytest.raises(ValueError, match="only existing visual observations"):
            validate_draft(bad, evidence)
    stages.draft.sample_id = "OTHER"
    with pytest.raises(ValueError, match="different samples"):
        validate_draft(stages.draft, stages.evidence)


def test_description_ids_and_references_cannot_be_empty_or_duplicated(stages):
    with pytest.raises(ValueError, match="at least 1 item"):
        DescriptionSentence(id="S1", text="Synthetic.", observation_ids=[])
    with pytest.raises(ValueError, match="must be unique"):
        DescriptionSentence(id="S1", text="Synthetic.", observation_ids=["O1", "O1"])
    bad = stages.draft.model_copy(deep=True)
    bad.sentences.append(bad.sentences[0])
    with pytest.raises(ValueError, match="sentence IDs must be unique"):
        validate_draft(bad, stages.evidence)


def test_review_keeps_draft_order_and_assembles_reference_union(stages):
    result = apply_review(stages.draft, stages.review, stages.evidence)
    assert result.schema_version == "4"
    assert [row.id for row in result.description_sentences] == ["SENT1", "SENT2"]
    assert result.description == "Synthetic profiles vary. A synthetic subset remains compact."
    assert result.description_observation_ids == ["O2", "O1"]
    assert result.limitations == ["Synthetic example.", "Draft limitation.", "Review limitation."]
    assert result.observations == stages.evidence.observations
    assert result.hypothesis_assessments == stages.evidence.hypothesis_assessments
    validate_pipeline_references(result, stages.view, stages.images, set())


@pytest.mark.parametrize("change", ["missing", "unknown", "duplicate", "sample"])
def test_review_requires_exactly_one_decision_per_draft_sentence(stages, change):
    bad = stages.review.model_copy(deep=True)
    if change == "missing": bad.reviews.pop()
    elif change == "unknown": bad.reviews[0].sentence_id = "UNKNOWN"
    elif change == "duplicate": bad.reviews.append(bad.reviews[0])
    else: bad.sample_id = "OTHER"
    with pytest.raises(ValueError, match="exactly one review|must be unique|different samples"):
        apply_review(stages.draft, bad, stages.evidence)


@pytest.mark.parametrize("change", ["text", "references"])
def test_keep_means_exact_text_and_reference_order(stages, change):
    bad = stages.review.model_copy(deep=True)
    decision = bad.reviews[1]
    if change == "text": decision.text += " Modified."
    else: decision.observation_ids.reverse()
    with pytest.raises(ValueError, match="exact text and observation references"):
        apply_review(stages.draft, bad, stages.evidence)


def test_review_cannot_add_unknown_observations(stages):
    bad = stages.review.model_copy(deep=True)
    bad.reviews[0].observation_ids = ["NEW_FINDING"]
    with pytest.raises(ValueError, match="only existing visual observations"):
        apply_review(stages.draft, bad, stages.evidence)


def test_delete_has_no_surviving_text_or_references(stages):
    deleted = stages.review.model_copy(deep=True)
    deleted.reviews[0].action = "delete"
    with pytest.raises(ValueError, match="empty text and observation references"):
        apply_review(stages.draft, deleted, stages.evidence)
    deleted.reviews[0].text = ""
    deleted.reviews[0].observation_ids = []
    result = apply_review(stages.draft, deleted, stages.evidence)
    assert result.description == "Synthetic profiles vary."
    assert result.description_observation_ids == ["O2", "O1"]


def test_abstention_requires_explicit_limitations_and_can_delete_all_sentences(stages):
    with pytest.raises(ValueError, match="abstention limitation"):
        BaselineOutput(sample_id="S1", observations=[], reviewed_image_ids=[], limitations=[])
    with pytest.raises(ValueError, match="abstention limitation"):
        DescriptionDraft(sample_id="S1", sentences=[], limitations=[])
    for decision in stages.review.reviews:
        decision.action = "delete"
        decision.text = ""
        decision.observation_ids = []
    result = apply_review(stages.draft, stages.review, stages.evidence)
    assert result.description == "" and result.description_observation_ids == []
    assert result.limitations


@pytest.mark.parametrize("mutation", ["description", "union", "sentence", "image"])
def test_saved_pipeline_cannot_diverge_from_reviewed_sentences_or_evidence(stages, mutation):
    result = apply_review(stages.draft, stages.review, stages.evidence)
    if mutation == "description": result.description = "Unreviewed text."
    elif mutation == "union": result.description_observation_ids.reverse()
    elif mutation == "sentence": result.description_sentences[0].observation_ids = ["UNKNOWN"]
    else: result.observations[0].image_ids.append("UNKNOWN")
    with pytest.raises(ValueError, match="ordered reviewed|ordered sentence reference union|only existing visual|unknown image"):
        validate_pipeline_references(result, stages.view, stages.images, set())


def test_pipeline_comparison_kind_is_required_and_preserved_in_saved_result(stages):
    schema = AgentOutputSchema(BaselineOutput).json_schema()
    assert "comparison_kind" in schema["$defs"]["PipelineAppearanceObservation"]["required"]
    missing = visual()
    missing.pop("comparison_kind")
    with pytest.raises(ValueError, match="comparison_kind"):
        PipelineAppearanceObservation.model_validate(missing)
    result = apply_review(stages.draft, stages.review, stages.evidence)
    assert all(row["comparison_kind"] == "treatment_control"
               for row in result.model_dump()["observations"])


@pytest.mark.parametrize("kind,image_id", [("treatment_only", "T1"), ("control_only", "C1")])
def test_single_role_morphology_records_are_valid_without_invented_comparisons(stages, kind, image_id):
    baseline = stages.baseline.model_copy(deep=True)
    observation = baseline.observations[0]
    observation.comparison_kind = kind
    observation.image_ids = [image_id]
    observation.text = "Synthetic visible outlines vary within this role."
    validate_baseline(baseline, stages.view, stages.images)
    evidence = stages.evidence.model_copy(deep=True)
    evidence.observations[0] = observation
    validate_evidence(evidence, stages.view, stages.images, set(), baseline=baseline)


@pytest.mark.parametrize("kind", ["treatment_only", "control_only"])
def test_single_role_declaration_cannot_cover_both_roles(stages, kind):
    bad = stages.baseline.model_copy(deep=True)
    bad.observations[0].comparison_kind = kind
    with pytest.raises(ValueError, match=f"{kind} observation must cite only"):
        validate_baseline(bad, stages.view, stages.images)


def test_treatment_control_declaration_still_requires_a_common_channel(stages):
    images = [dict(row) for row in stages.images]
    images[1]["channel"] = "RNA"
    with pytest.raises(ValueError, match="treatment and control evidence in a common channel"):
        validate_baseline(stages.baseline, stages.view, images)


@pytest.mark.parametrize("visual_confirmation,kind", [(True, "measurement_only"), (False, "treatment_only"),
                                                    (False, "control_only"), (False, "treatment_control")])
def test_measurement_kind_must_match_visual_confirmation(visual_confirmation, kind):
    data = visual()
    if not visual_confirmation:
        data.update(visually_confirmed=False, image_ids=[], cp_feature_ids=["CP1"],
            cp_assessment=dict(status="measurement_only", reason="Synthetic measurement.",
                               effect_on_description="Not visual evidence."))
    data["comparison_kind"] = kind
    with pytest.raises(ValueError, match="Measurement-only comparison kind"):
        PipelineAppearanceObservation.model_validate(data)


def test_measurement_only_record_keeps_existing_cp_validation(stages):
    data = visual("CP_ONLY", "measurement_only", ["CP1"])
    data.update(visually_confirmed=False, image_ids=[], comparison_kind="measurement_only")
    evidence = GuidanceEvidenceOutput.model_validate({**stages.evidence.model_dump(),
        "observations": [*stages.evidence.model_dump()["observations"], data]})
    with pytest.raises(ValueError, match="not returned"):
        validate_evidence(evidence, stages.view, stages.images, set())
    validate_evidence(evidence, stages.view, stages.images, {"CP1"})


@pytest.mark.parametrize("assessment,difference", [("supported", "observed"), ("not_supported", "not_observed")])
def test_decisive_hypothesis_requires_a_cited_comparison_not_a_single_role(stages, assessment, difference):
    evidence = stages.evidence.model_copy(deep=True)
    evidence.observations[1].comparison_kind = "treatment_only"
    evidence.observations[1].image_ids = ["T1"]
    hypothesis = evidence.hypothesis_assessments[0]
    hypothesis.assessment = assessment
    check = hypothesis.checks[0]
    check.assessment = assessment
    check.contrast.difference = difference
    check.observation_ids = ["O2"]  # A valid comparison O1 exists but is not cited.
    with pytest.raises(ValueError, match="cited treatment_control visual comparison"):
        validate_evidence(evidence, stages.view, stages.images, set())
    check.observation_ids = ["O1", "O2"]
    validate_evidence(evidence, stages.view, stages.images, set())


def test_two_separate_single_role_records_do_not_implicitly_establish_a_contrast(stages):
    evidence = stages.evidence.model_copy(deep=True)
    for row, kind, image_id in zip(evidence.observations,
                                  ("treatment_only", "control_only"), ("T1", "C1")):
        row.comparison_kind, row.image_ids = kind, [image_id]
    check = evidence.hypothesis_assessments[0].checks[0]
    check.observation_ids = ["O1", "O2"]
    validate_evidence(evidence, stages.view, stages.images, set())  # Uncertain is appropriate.
    evidence.hypothesis_assessments[0].assessment = check.assessment = "supported"
    check.contrast.difference = "observed"
    with pytest.raises(ValueError, match="cited treatment_control visual comparison"):
        validate_evidence(evidence, stages.view, stages.images, set())


def test_legacy_default_remains_strict_and_new_kind_projection_must_be_complete(stages):
    data = visual()
    data.pop("comparison_kind")
    data["image_ids"] = ["T1"]
    legacy = ObservationOutput(sample_id="S1", observations=[data], hypothesis_assessments=[],
        description_observation_ids=["O1"], limitations=[])
    view = stages.view.model_copy(update={"hypotheses": []})
    with pytest.raises(ValueError, match="treatment and control"):
        validate_observation_references(legacy, view, stages.images, set())
    for mapping in ({}, {"O1": "treatment_only", "OTHER": "control_only"}, {"O1": "unrestricted"}):
        with pytest.raises(ValueError, match="map every observation ID exactly once"):
            validate_observation_references(legacy, view, stages.images, set(), comparison_kinds=mapping)
    validate_observation_references(legacy, view, stages.images, set(),
                                    comparison_kinds={"O1": "treatment_only"})
    with pytest.raises(ValueError, match="Measurement-only comparison kind"):
        validate_observation_references(legacy, view, stages.images, set(),
                                        comparison_kinds={"O1": "measurement_only"})
