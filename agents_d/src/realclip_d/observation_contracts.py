"""Model-authored observations and application-assembled descriptions."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .contracts import Contract, ExperimentContext, HypothesisViewV3, NonEmpty, ResearchResultV3, ResearchResultV4


class ObservationRequest(Contract):
    sample_id: NonEmpty
    context: ExperimentContext
    hypotheses: list[HypothesisViewV3]
    images: list[dict[str, Any]]
    cp_available: bool
    cp_catalogue: dict[str, Any]


class CPAssessment(Contract):
    status: Literal["supports", "contradicts", "mixed", "not_informative",
                    "not_available", "not_checked", "no_matching_measurement", "measurement_only"]
    reason: NonEmpty
    effect_on_description: NonEmpty


class AppearanceObservation(Contract):
    id: NonEmpty
    text: NonEmpty
    scope: NonEmpty
    image_ids: list[NonEmpty]
    cp_feature_ids: list[NonEmpty]
    visually_confirmed: bool
    cp_assessment: CPAssessment

    @model_validator(mode="after")
    def evidence_exists(self):
        for ids in (self.image_ids, self.cp_feature_ids):
            if len(ids) != len(set(ids)):
                raise ValueError("Observation evidence IDs must be unique")
        if self.visually_confirmed and not self.image_ids:
            raise ValueError("Visual confirmation requires image evidence, not CP alone")
        if not self.image_ids and not self.cp_feature_ids:
            raise ValueError("An observation requires image or measurement evidence")
        status = self.cp_assessment.status
        if status in {"not_available", "not_checked", "no_matching_measurement"}:
            if self.cp_feature_ids:
                raise ValueError("Unavailable, unchecked or unmatched CP cannot cite measurement evidence")
        elif not self.cp_feature_ids:
            raise ValueError("A CP evidence assessment requires CP feature references")
        if not self.visually_confirmed and status != "measurement_only":
            raise ValueError("Measurement-only observations require measurement_only CP status")
        if self.visually_confirmed and status == "measurement_only":
            raise ValueError("Visual observations must explain the CP relationship, not measurement_only")
        return self


class VisualContrast(Contract):
    treatment: NonEmpty
    control: NonEmpty
    difference: Literal["observed", "not_observed", "unassessable"]


class AppearanceCheck(Contract):
    check: NonEmpty
    assessment: Literal["supported", "not_supported", "uncertain"]
    observation_ids: list[NonEmpty]
    reason: NonEmpty
    contrast: VisualContrast

    @model_validator(mode="after")
    def assessment_requires_comparison(self):
        if self.assessment == "supported" and self.contrast.difference != "observed":
            raise ValueError("Support requires an observed treatment-control difference, not presence alone")
        if self.assessment == "not_supported" and self.contrast.difference != "not_observed":
            raise ValueError("An unassessable comparison is uncertain, not not_supported")
        return self


class HypothesisAssessment(Contract):
    hypothesis_id: NonEmpty
    assessment: Literal["supported", "partially_supported", "not_supported", "uncertain"]
    checks: list[AppearanceCheck] = Field(min_length=1)
    reason: NonEmpty

    @model_validator(mode="after")
    def aggregate_matches_checks(self):
        statuses = {check.assessment for check in self.checks}
        expected = ("supported" if statuses == {"supported"} else
                    "not_supported" if statuses == {"not_supported"} else
                    "partially_supported" if "supported" in statuses else "uncertain")
        if self.assessment != expected:
            raise ValueError("Overall assessment must match component assessments")
        return self


class ObservationOutput(Contract):
    """Exactly five model-generated fields. No generated description/version."""
    sample_id: NonEmpty
    observations: list[AppearanceObservation]
    hypothesis_assessments: list[HypothesisAssessment]
    description_observation_ids: list[NonEmpty]
    limitations: list[NonEmpty]

    @model_validator(mode="after")
    def links_and_selection_match(self):
        observations = {row.id: row for row in self.observations}
        if len(observations) != len(self.observations):
            raise ValueError("Observation IDs must be unique")
        hypotheses = [row.hypothesis_id for row in self.hypothesis_assessments]
        if len(hypotheses) != len(set(hypotheses)):
            raise ValueError("Hypothesis assessments must be unique")
        for item in self.hypothesis_assessments:
            for check in item.checks:
                ids = check.observation_ids
                if len(ids) != len(set(ids)) or not set(ids) <= set(observations):
                    raise ValueError("Checks must reference unique existing observation IDs")
                if check.assessment != "uncertain" and not any(
                    observations[key].visually_confirmed for key in ids
                ):
                    raise ValueError("Supported or unsupported visual checks require visual observations, not CP alone")
                cp_from_measurements = {fid for key in ids if not observations[key].visually_confirmed
                                        for fid in observations[key].cp_feature_ids}
                cp_in_visual = {fid for key in ids if observations[key].visually_confirmed
                                for fid in observations[key].cp_feature_ids}
                if cp_from_measurements and any(observations[key].visually_confirmed for key in ids):
                    if not cp_from_measurements <= cp_in_visual:
                        raise ValueError("CP used in a visual check must be integrated into its visual observations")
        selected = self.description_observation_ids
        if len(selected) != len(set(selected)) or not set(selected) <= set(observations):
            raise ValueError("Description selection must reference unique existing observations")
        if any(not observations[key].visually_confirmed for key in selected):
            raise ValueError("Measurement-only observations cannot enter the visual description")
        if not selected and not self.limitations:
            raise ValueError("An empty description selection requires a limitation explaining abstention")
        return self

    def assemble_description(self) -> str:
        observations = {row.id: row for row in self.observations}
        return " ".join(observations[key].text for key in self.description_observation_ids)


class ObservationResult(ObservationOutput):
    schema_version: Literal["3"] = "3"
    description: str

    @model_validator(mode="after")
    def description_matches_selection(self):
        if self.description != self.assemble_description():
            raise ValueError("Description must equal the ordered selected observation sentences")
        return self


def assemble_result(output: ObservationOutput) -> ObservationResult:
    return ObservationResult(**output.model_dump(mode="json"), description=output.assemble_description())


def _has_verified_empty_cp_query(cp_query_results: list[dict[str, Any]] | None,
                                 cp_table: Any) -> bool:
    """Verify an actual zero-match query, not a model claim or an empty page.

    Receipts must come from this run's application-written tools.jsonl. Repeating
    the read-only local query checks that the saved filters really match nothing
    in the bound CP table. This establishes query provenance, not the semantic
    relevance of that query to a particular visual observation.
    """
    if cp_table is None:
        return False

    def empty(result: Any) -> bool:
        if not isinstance(result, dict) or result.get("status") != "not_found":
            return False
        data = result.get("data")
        return (isinstance(data, dict) and type(data.get("total")) is int
                and data["total"] == 0 and data.get("rows") == []
                and data.get("comparisons") == []
                and ("total_rows" not in data or data["total_rows"] == 0)
                and (not isinstance(data.get("query_diagnostic"), dict)
                     or data["query_diagnostic"].get("outcome") != "empty_feature_selection"))

    for receipt in cp_query_results or []:
        if (not isinstance(receipt, dict) or receipt.get("tool") != "query_cp"
                or not empty(receipt.get("result"))
                or ("model_result" in receipt and not empty(receipt["model_result"]))
                or receipt.get("arguments", []) != []):
            continue
        arguments = receipt.get("keyword_arguments")
        if not isinstance(arguments, dict) or arguments.get("feature_names") == []:
            continue
        try:
            reproduced = cp_table.query(**arguments).model_dump(mode="json")
        except (TypeError, ValueError):
            continue
        if empty(reproduced):
            return True
    return False


def validate_observation_references(result: ObservationOutput, research: ResearchResultV3 | ResearchResultV4 | ObservationRequest,
                                    images: list[dict[str, Any]], cp_ids: set[str], cp_table: Any = None,
                                    *, comparison_kinds: dict[str, str] | None = None,
                                    cp_query_results: list[dict[str, Any]] | None = None) -> None:
    # Revalidate mutated instances, too.
    if isinstance(result, ObservationResult):
        ObservationResult.model_validate(result.model_dump(mode="json"))
    result = ObservationOutput.model_validate(result.model_dump(exclude={"schema_version", "description"}))
    # Legacy callers retain the existing paired-evidence requirement. New-stage
    # callers must supply a complete typed-observation projection, never a flag
    # that simply disables the treatment/control guard.
    if comparison_kinds is not None:
        allowed = {"treatment_control", "treatment_only", "control_only", "measurement_only"}
        if (not isinstance(comparison_kinds, dict)
                or set(comparison_kinds) != {row.id for row in result.observations}
                or any(not isinstance(kind, str) or kind not in allowed for kind in comparison_kinds.values())):
            raise ValueError("Comparison kinds must map every observation ID exactly once to a valid kind")
    if result.sample_id != research.sample_id:
        raise ValueError("Observation belongs to another sample")
    expected = ({row.id for row in research.morphology_hypotheses} if isinstance(research, (ResearchResultV3, ResearchResultV4))
                else {row.hypothesis_id for row in research.hypotheses})
    if {row.hypothesis_id for row in result.hypothesis_assessments} != expected:
        raise ValueError("Every supplied hypothesis requires exactly one assessment; no unknown hypotheses")
    image_map = {row["image_id"]: row for row in images}
    if any(row.cp_assessment.status == "no_matching_measurement" for row in result.observations):
        if not _has_verified_empty_cp_query(cp_query_results, cp_table):
            raise ValueError("no_matching_measurement requires an actual query_cp zero-match receipt verified against the supplied CP table")
    for row in result.observations:
        kind = comparison_kinds[row.id] if comparison_kinds is not None else "treatment_control"
        if not set(row.image_ids) <= set(image_map):
            raise ValueError("Observation cites an unknown image")
        if not set(row.cp_feature_ids) <= cp_ids:
            raise ValueError("Observation cites a CP feature not returned to this agent")
        if isinstance(research, ObservationRequest) and not research.cp_available:
            if row.cp_assessment.status != "not_available":
                raise ValueError("An input without CP requires not_available CP assessments")
        if comparison_kinds is not None and (kind == "measurement_only") == row.visually_confirmed:
            raise ValueError("Measurement-only comparison kind requires a nonvisual observation, and visual kinds require visual confirmation")
        if row.cp_assessment.status in {"supports", "contradicts", "mixed"}:
            if cp_table is None:
                raise ValueError("Decisive CP integration requires supplied comparison metadata")
            cited = set(row.cp_feature_ids)
            matched = set()
            for pair in cp_table.comparisons():
                ids = {pair["treatment_feature_id"], pair["control_feature_id"]}
                if None not in ids and ids <= cited:
                    if not pair["comparison_ready"]:
                        raise ValueError("CP comparison has unresolved interpretation gaps; use not_informative")
                    matched.update(ids)
            if cited != matched:
                raise ValueError("Decisive CP integration requires complete treatment/control pairs of the same measurement")
        if row.visually_confirmed:
            roles = {image_map[key]["role"] for key in row.image_ids}
            if kind in {"treatment_only", "control_only"}:
                expected_role = "treatment" if kind == "treatment_only" else "control"
                if roles != {expected_role}:
                    raise ValueError(f"A {kind} observation must cite only {expected_role} images")
                continue
            treatment = {image_map[key]["channel"] for key in row.image_ids if image_map[key]["role"] == "treatment"}
            control = {image_map[key]["channel"] for key in row.image_ids if image_map[key]["role"] == "control"}
            if not treatment & control:
                raise ValueError("A visual comparison requires treatment and control evidence in a common channel")
