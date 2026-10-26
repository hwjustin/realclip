"""Boundaries for image-first observation, description writing, and image review.

References and stage order are checked here. These checks cannot establish the
visual truth of a sentence or guarantee that its wording follows its evidence.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .contracts import Contract, NonEmpty
from .observation_contracts import (
    AppearanceObservation,
    HypothesisAssessment,
    ObservationOutput,
    ObservationRequest,
    validate_observation_references,
)


def _unique(values: list[str], what: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{what} must be unique")


def _visual_ids(observations: list[AppearanceObservation]) -> list[str]:
    return [row.id for row in observations if row.visually_confirmed]


class PipelineAppearanceObservation(AppearanceObservation):
    """Declare whether an image record describes one role or compares roles.

    The kind is explicit model output. Actual image roles/channels are checked
    against provenance by the stage reference validator, not guessed from prose.
    """

    comparison_kind: Literal["treatment_control", "treatment_only", "control_only", "measurement_only"]

    @model_validator(mode="after")
    def comparison_kind_matches_evidence(self):
        if (self.comparison_kind == "measurement_only") == self.visually_confirmed:
            raise ValueError("Measurement-only comparison kind requires a nonvisual observation, and visual kinds require visual confirmation")
        return self


def _comparison_kinds(value: Any) -> dict[str, str]:
    return {row.id: row.comparison_kind for row in value.observations}


def _evidence_output(value: Any) -> ObservationOutput:
    return ObservationOutput(
        sample_id=value.sample_id,
        # The legacy contract stays unchanged. Preserve the new explicit kinds
        # separately when calling reference validation below.
        observations=[AppearanceObservation.model_validate(row.model_dump(exclude={"comparison_kind"}))
                      for row in value.observations],
        hypothesis_assessments=value.hypothesis_assessments,
        description_observation_ids=_visual_ids(value.observations),
        limitations=value.limitations,
    )


class BaselineOutput(Contract):
    """Only image observations; this stage has no hypothesis assessments."""

    sample_id: NonEmpty
    observations: list[PipelineAppearanceObservation]
    reviewed_image_ids: list[NonEmpty]
    limitations: list[NonEmpty]

    @model_validator(mode="after")
    def valid_baseline(self):
        _unique([row.id for row in self.observations], "Observation IDs")
        _unique(self.reviewed_image_ids, "Reviewed image IDs")
        if not self.observations and not self.limitations:
            raise ValueError("An empty baseline requires an abstention limitation")
        if any(row.cp_feature_ids or not row.visually_confirmed for row in self.observations):
            raise ValueError("Baseline observations require visual evidence and cannot cite CP")
        return self


class GuidanceEvidenceOutput(Contract):
    sample_id: NonEmpty
    observations: list[PipelineAppearanceObservation]
    hypothesis_assessments: list[HypothesisAssessment]
    limitations: list[NonEmpty]

    @model_validator(mode="after")
    def valid_evidence(self):
        _evidence_output(self)
        return self


class DescriptionSentence(Contract):
    id: NonEmpty
    text: NonEmpty
    observation_ids: list[NonEmpty] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_references(self):
        _unique(self.observation_ids, "Sentence observation references")
        return self


class DescriptionDraft(Contract):
    sample_id: NonEmpty
    sentences: list[DescriptionSentence]
    limitations: list[NonEmpty]

    @model_validator(mode="after")
    def valid_sentences(self):
        _unique([row.id for row in self.sentences], "Description sentence IDs")
        if not self.sentences and not self.limitations:
            raise ValueError("An empty description requires an abstention limitation")
        return self


class SentenceReview(Contract):
    sentence_id: NonEmpty
    action: Literal["keep", "revise", "delete"]
    text: str
    observation_ids: list[NonEmpty]
    reason: NonEmpty

    @model_validator(mode="after")
    def action_fields_match(self):
        _unique(self.observation_ids, "Reviewed sentence observation references")
        if self.action == "delete":
            if self.text != "" or self.observation_ids:
                raise ValueError("Deleting a sentence requires empty text and observation references")
        elif not self.text.strip() or not self.observation_ids:
            raise ValueError("Keeping or revising a sentence requires text and observation references")
        return self


class DescriptionReview(Contract):
    sample_id: NonEmpty
    reviews: list[SentenceReview]
    limitations: list[NonEmpty]

    @model_validator(mode="after")
    def unique_reviews(self):
        _unique([row.sentence_id for row in self.reviews], "Reviewed sentence IDs")
        return self


def _sentence_references(sentences: list[DescriptionSentence]) -> list[str]:
    return list(dict.fromkeys(key for row in sentences for key in row.observation_ids))


def _check_sentence_evidence(sentences: list[DescriptionSentence], evidence: GuidanceEvidenceOutput) -> None:
    available = set(_visual_ids(evidence.observations))
    for row in sentences:
        if not set(row.observation_ids) <= available:
            raise ValueError("Description sentences may reference only existing visual observations")


class ObservationPipelineResult(GuidanceEvidenceOutput):
    schema_version: Literal["4"] = "4"
    description_sentences: list[DescriptionSentence]
    description_observation_ids: list[NonEmpty]
    description: str

    @model_validator(mode="after")
    def description_matches_sentences(self):
        _unique([row.id for row in self.description_sentences], "Description sentence IDs")
        _check_sentence_evidence(self.description_sentences, self)
        if self.description != " ".join(row.text for row in self.description_sentences):
            raise ValueError("Description must equal the ordered reviewed sentence texts")
        if self.description_observation_ids != _sentence_references(self.description_sentences):
            raise ValueError("Description observation IDs must equal the ordered sentence reference union")
        if not self.description_sentences and not self.limitations:
            raise ValueError("An empty description requires an abstention limitation")
        return self


def validate_baseline(output: BaselineOutput, view: ObservationRequest,
                      images: list[dict[str, Any]], cp_ids: set[str] | None = None,
                      cp_table: Any = None) -> None:
    """Require all original image IDs; CP is unavailable to this stage."""
    output = BaselineOutput.model_validate(output.model_dump(mode="json"))
    if cp_ids:
        raise ValueError("Baseline stage cannot receive queried CP feature IDs")
    expected_cp_status = "not_checked" if view.cp_available else "not_available"
    if any(row.cp_assessment.status != expected_cp_status for row in output.observations):
        raise ValueError(f"Baseline CP status must be {expected_cp_status}, without implying a CP check")
    expected = [row["image_id"] for row in view.images]
    if set(output.reviewed_image_ids) != set(expected):
        raise ValueError("Baseline must review every original supplied image and no others")
    # Use the established image/CP checks without making hypotheses available.
    blind_view = view.model_copy(update={"hypotheses": []})
    legacy = ObservationOutput(sample_id=output.sample_id, observations=[
        AppearanceObservation.model_validate(row.model_dump(exclude={"comparison_kind"}))
        for row in output.observations],
        hypothesis_assessments=[], description_observation_ids=_visual_ids(output.observations),
        limitations=output.limitations)
    validate_observation_references(legacy, blind_view, images, set(), cp_table,
                                    comparison_kinds=_comparison_kinds(output))


def validate_evidence(output: GuidanceEvidenceOutput, view: ObservationRequest,
                      images: list[dict[str, Any]], cp_ids: set[str], cp_table: Any = None,
                      baseline: BaselineOutput | None = None, *,
                      cp_query_results: list[dict[str, Any]] | None = None) -> None:
    output = GuidanceEvidenceOutput.model_validate(output.model_dump(mode="json"))
    if baseline is not None:
        baseline = BaselineOutput.model_validate(baseline.model_dump(mode="json"))
        if baseline.sample_id != output.sample_id:
            raise ValueError("Baseline and guidance evidence belong to different samples")
        prefix = output.observations[:len(baseline.observations)]
        if [row.model_dump(mode="json") for row in prefix] != [
            row.model_dump(mode="json") for row in baseline.observations
        ]:
            raise ValueError("Guidance evidence must preserve the complete baseline observation prefix unchanged")
    validate_observation_references(_evidence_output(output), view, images, cp_ids, cp_table,
                                    comparison_kinds=_comparison_kinds(output),
                                    cp_query_results=cp_query_results)
    observations = {row.id: row for row in output.observations}
    for assessment in output.hypothesis_assessments:
        for check in assessment.checks:
            if check.assessment in {"supported", "not_supported"} and not any(
                observations[key].visually_confirmed
                and observations[key].comparison_kind == "treatment_control"
                for key in check.observation_ids
            ):
                raise ValueError("A decisive hypothesis check requires a cited treatment_control visual comparison, not single-role presence")


def validate_draft(draft: DescriptionDraft, evidence: GuidanceEvidenceOutput) -> None:
    draft = DescriptionDraft.model_validate(draft.model_dump(mode="json"))
    evidence = GuidanceEvidenceOutput.model_validate(evidence.model_dump(mode="json"))
    if draft.sample_id != evidence.sample_id:
        raise ValueError("Description and evidence belong to different samples")
    _check_sentence_evidence(draft.sentences, evidence)


def apply_review(draft: DescriptionDraft, review: DescriptionReview,
                 evidence: GuidanceEvidenceOutput) -> ObservationPipelineResult:
    """Apply exactly one decision per draft sentence, retaining draft order.

    A reviewer can revise wording and references, but new observations must go
    back through the evidence stage instead of being invented during review.
    """
    draft = DescriptionDraft.model_validate(draft.model_dump(mode="json"))
    review = DescriptionReview.model_validate(review.model_dump(mode="json"))
    evidence = GuidanceEvidenceOutput.model_validate(evidence.model_dump(mode="json"))
    validate_draft(draft, evidence)
    if review.sample_id != draft.sample_id:
        raise ValueError("Review and description belong to different samples")
    reviews = {row.sentence_id: row for row in review.reviews}
    if set(reviews) != {row.id for row in draft.sentences}:
        raise ValueError("Every draft sentence requires exactly one review; unknown sentences are forbidden")
    sentences = []
    for row in draft.sentences:
        decision = reviews[row.id]
        if decision.action == "delete":
            continue
        if decision.action == "keep" and (
            decision.text != row.text or decision.observation_ids != row.observation_ids
        ):
            raise ValueError("A kept sentence must preserve its exact text and observation references")
        sentences.append(DescriptionSentence(id=row.id, text=decision.text,
                                             observation_ids=decision.observation_ids))
    _check_sentence_evidence(sentences, evidence)
    return ObservationPipelineResult(
        sample_id=evidence.sample_id, observations=evidence.observations,
        hypothesis_assessments=evidence.hypothesis_assessments,
        description_sentences=sentences, description_observation_ids=_sentence_references(sentences),
        description=" ".join(row.text for row in sentences),
        limitations=list(dict.fromkeys([*evidence.limitations, *draft.limitations, *review.limitations])),
    )


def validate_pipeline_references(result: ObservationPipelineResult, view: ObservationRequest,
                                 images: list[dict[str, Any]], cp_ids: set[str], cp_table: Any = None, *,
                                 cp_query_results: list[dict[str, Any]] | None = None) -> None:
    result = ObservationPipelineResult.model_validate(result.model_dump(mode="json"))
    evidence = GuidanceEvidenceOutput.model_validate(result.model_dump(include={
        "sample_id", "observations", "hypothesis_assessments", "limitations"}))
    validate_evidence(evidence, view, images, cp_ids, cp_table,
                      cp_query_results=cp_query_results)
