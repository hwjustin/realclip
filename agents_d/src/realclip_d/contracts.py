"""Versioned, explicit data boundaries shared by D's two agents."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CompoundIdentity(Contract):
    name: str | None = None
    inchikey: str | None = None
    smiles: str | None = None
    pubchem_cid: int | None = Field(default=None, gt=0)
    chembl_id: str | None = None

    @field_validator("inchikey")
    @classmethod
    def valid_key(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", value):
            raise ValueError("A complete uppercase InChIKey is required")
        return value

    @model_validator(mode="after")
    def has_identifier(self):
        if not any([self.name, self.inchikey, self.smiles, self.pubchem_cid, self.chembl_id]):
            raise ValueError("At least one compound identifier is required")
        return self


class ExperimentContext(Contract):
    cell_line: str | None = None
    dose: str | None = None
    treatment_duration: str | None = None
    notes: list[str] = Field(default_factory=list)


class ImageInput(Contract):
    image_id: str = Field(min_length=1)
    path: Path
    role: Literal["treatment", "control"]
    channel: str = Field(min_length=1)
    field: str = Field(min_length=1)
    view: str = "overview"
    well_id: str | None = None
    parent_image_id: str | None = None
    crop_box: tuple[int, int, int, int] | None = None
    alignment_id: str | None = None
    stain: str | None = None
    pixel_size_um: float | None = Field(default=None, gt=0)
    acquisition_settings: str | None = None
    display_settings: str | None = None

    @model_validator(mode="after")
    def valid_crop(self):
        if self.crop_box is not None:
            x, y, width, height = self.crop_box
            if x < 0 or y < 0 or width <= 0 or height <= 0 or not self.parent_image_id:
                raise ValueError("crop_box requires a parent image, nonnegative origin and positive size")
        return self


class CPFeatureMetadata(Contract):
    """Supplied definitions; no automatic interpretation inferred from names."""
    feature_name: NonEmpty
    definition: NonEmpty
    scale: str | None = None
    raw_unit: str | None = None
    higher_value_meaning: str | None = None


class CPMetadata(Contract):
    value_kind: Literal["raw", "preprocessed", "unknown"] = "unknown"
    normalization_reference: str | None = None
    comparison_context: str | None = None
    aggregation_method: str | None = None
    within_feature_order_preserved: bool | None = None
    quality_notes: list[str] = Field(default_factory=list)
    features: list[CPFeatureMetadata] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_definitions(self):
        names = [row.feature_name for row in self.features]
        if len(names) != len(set(names)):
            raise ValueError("CP feature definitions must have unique feature names")
        return self


class SampleInput(Contract):
    schema_version: Literal["1"] = "1"
    sample_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    compound: CompoundIdentity
    known_moa: list[str] = Field(default_factory=list)
    context: ExperimentContext = Field(default_factory=ExperimentContext)
    images: list[ImageInput] = Field(default_factory=list)
    cp_file: Path | None = None
    cp_preprocessing: str | None = None
    cp_metadata: CPMetadata | None = None

    @model_validator(mode="after")
    def unique_images(self):
        ids = [image.image_id for image in self.images]
        if len(ids) != len(set(ids)):
            raise ValueError("Image IDs must be unique")
        by_id = {image.image_id: image for image in self.images}
        for image in self.images:
            if image.parent_image_id:
                parent = by_id.get(image.parent_image_id)
                if parent is None or parent is image:
                    raise ValueError("Image parent must reference another supplied image")
                if (parent.role, parent.channel, parent.field, parent.well_id) != (
                    image.role, image.channel, image.field, image.well_id
                ):
                    raise ValueError("Crop and parent must share role, channel, field and well")
                seen = {image.image_id}
                while parent:
                    if parent.image_id in seen:
                        raise ValueError("Image parent relationships cannot contain cycles")
                    seen.add(parent.image_id)
                    parent = by_id.get(parent.parent_image_id)
        return self


class Source(Contract):
    source_id: str
    url: str
    retrieved_at: str
    cache_ref: str | None = None
    kind: Literal["official_api", "web_search", "mock"] = "official_api"
    title: str | None = None


class ToolResult(Contract):
    status: Literal["ok", "not_found", "ambiguous", "unavailable", "error", "rate_limited"]
    data: dict[str, Any] = Field(default_factory=dict)
    sources: list[Source] = Field(default_factory=list)
    message: str | None = None


class ResearchFact(Contract):
    fact_id: str
    statement: str
    source_ids: list[str]
    evidence_level: Literal["direct_morphology", "other_context", "biochemical", "mechanism_inference", "database_annotation", "unknown"]
    conditions: str
    limitations: str


class MorphologyHypothesis(Contract):
    hypothesis_id: str
    change: str
    where_to_check: list[str]
    applicable_conditions: str
    uncertainty: str
    fact_ids: list[str]
    disconfirming_observation: str


class ResearchResult(Contract):
    schema_version: Literal["1"] = "1"
    sample_id: str
    identity_summary: str
    sources: list[Source]
    facts: list[ResearchFact]
    hypotheses: list[MorphologyHypothesis]
    limitations: list[str]


NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
HypothesisBasis = Literal["direct_literature", "indirect_literature", "moa_inference"]


class ResearchFactV2(ResearchFact):
    statement: NonEmpty
    conditions: NonEmpty
    limitations: NonEmpty
    evidence_level: Literal["direct_morphology", "other_context", "biochemical", "mechanism_inference", "database_annotation", "provided_moa", "unknown"]
    origin: Literal["literature", "database", "provided_moa", "inference", "unknown"]
    compound_scope: Literal["exact_compound", "related_compound", "target_class", "unknown"]
    access: Literal["abstract", "full_text", "annotation", "input", "not_available"]

    @model_validator(mode="after")
    def evidence_is_not_promoted(self):
        if self.origin in {"literature", "database"} and not self.source_ids:
            raise ValueError("Retrieved facts require source IDs")
        if self.origin == "literature" and self.access not in {"abstract", "full_text"}:
            raise ValueError("Literature facts require an actually read abstract or full text")
        if self.origin == "database" and self.access != "annotation":
            raise ValueError("Database facts must retain annotation access")
        if self.evidence_level == "direct_morphology" and self.origin != "literature":
            raise ValueError("Direct morphology requires a literature observation, not a mechanism or database summary")
        if self.origin == "provided_moa" or self.evidence_level == "provided_moa":
            if self.origin != "provided_moa" or self.evidence_level != "provided_moa" or self.access != "input" or self.source_ids:
                raise ValueError("Provided MOA must remain an input prior without fabricated source citations")
        return self


class CPQueryHint(Contract):
    compartment: NonEmpty
    channel: NonEmpty
    family: NonEmpty
    interpretation: NonEmpty


class VerificationPlan(Contract):
    comparison: NonEmpty
    image_checks: list[NonEmpty] = Field(min_length=1)
    cp_checks: list[CPQueryHint]
    supporting_observation: NonEmpty
    not_assessable_when: list[NonEmpty] = Field(min_length=1)


class MorphologyHypothesisV2(MorphologyHypothesis):
    change: NonEmpty
    where_to_check: list[NonEmpty] = Field(min_length=1)
    applicable_conditions: NonEmpty
    uncertainty: NonEmpty
    fact_ids: list[str] = Field(min_length=1)
    disconfirming_observation: NonEmpty
    basis: HypothesisBasis
    rationale: NonEmpty
    verification: VerificationPlan


class IdentityContextReview(Contract):
    status: Literal["verified", "resolved", "ambiguous", "unresolved"]
    source_ids: list[str]
    context: ExperimentContext
    unknown_conditions: list[Literal["cell_line", "dose", "treatment_duration"]]
    conflicts: list[str]


class LiteratureReview(Contract):
    status: Literal["direct_observations_found", "indirect_observations_only", "no_morphology_found", "unavailable", "not_searched"]
    search_summary: NonEmpty
    fact_ids: list[str]
    limitations: list[str]


class MechanismReview(Contract):
    status: Literal["hypotheses_proposed", "no_supported_inference", "not_applicable"]
    fact_ids: list[str]
    summary: NonEmpty
    limitations: list[str]


class ResearchResultV2(ResearchResult):
    """Four explicit steps; V1 remains a separate, readable artifact contract."""
    schema_version: Literal["2"] = "2"
    facts: list[ResearchFactV2]
    hypotheses: list[MorphologyHypothesisV2]
    identity_context: IdentityContextReview
    literature_review: LiteratureReview
    mechanism_review: MechanismReview

    @model_validator(mode="after")
    def classifications_match_grounding(self):
        facts = {fact.fact_id: fact for fact in self.facts}
        if len(facts) != len(self.facts):
            raise ValueError("Research fact IDs must be unique")
        if not set(self.identity_context.source_ids) <= {source.source_id for source in self.sources}:
            raise ValueError("Identity review cites an unknown source")
        literature = {fact.fact_id for fact in self.facts if fact.origin == "literature"}
        if set(self.literature_review.fact_ids) != literature:
            raise ValueError("Literature review must list exactly the literature facts")
        direct = [fact for fact in self.facts if fact.evidence_level == "direct_morphology"
                  and fact.compound_scope == "exact_compound"]
        indirect = [fact for fact in self.facts if fact.evidence_level == "direct_morphology"
                    and fact.compound_scope in {"related_compound", "target_class"}]
        status = self.literature_review.status
        if bool(direct) != (status == "direct_observations_found"):
            raise ValueError("Direct literature status requires an exact-compound morphology observation")
        if not direct and bool(indirect) != (status == "indirect_observations_only"):
            raise ValueError("Indirect literature status must match related-compound observations")
        if not set(self.mechanism_review.fact_ids) <= set(facts):
            raise ValueError("Mechanism review cites an unknown fact")
        inferred = []
        for hypothesis in self.hypotheses:
            if not set(hypothesis.fact_ids) <= set(facts):
                raise ValueError("Hypothesis cites an unknown fact")
            linked = [facts[identifier] for identifier in hypothesis.fact_ids]
            if hypothesis.basis == "direct_literature" and not any(
                fact.origin == "literature" and fact.evidence_level == "direct_morphology"
                and fact.compound_scope == "exact_compound" for fact in linked
            ):
                raise ValueError("Direct literature hypotheses require an exact-compound literature morphology fact")
            if hypothesis.basis == "indirect_literature" and not any(
                fact.origin == "literature" and fact.evidence_level == "direct_morphology"
                and fact.compound_scope in {"related_compound", "target_class"} for fact in linked
            ):
                raise ValueError("Indirect literature hypotheses require related-compound morphology evidence")
            if hypothesis.basis == "moa_inference":
                inferred.append(hypothesis)
                if not any(fact.origin not in {"inference", "unknown"} and fact.evidence_level in {
                    "provided_moa", "biochemical", "mechanism_inference", "database_annotation"} for fact in linked):
                    raise ValueError("MOA inference requires a supplied MOA or retrieved mechanistic premise")
                if not set(hypothesis.fact_ids) <= set(self.mechanism_review.fact_ids):
                    raise ValueError("MOA hypothesis grounds must appear in the mechanism review")
        if bool(inferred) != (self.mechanism_review.status == "hypotheses_proposed"):
            raise ValueError("Mechanism review status must match MOA hypotheses")
        return self


class ResearchSource(Contract):
    id: NonEmpty
    title: str | None
    url: NonEmpty


class CompoundIdentification(Contract):
    summary: NonEmpty
    source_ids: list[NonEmpty]


class SourceFinding(Contract):
    finding: NonEmpty
    reported_conditions: NonEmpty
    source_ids: list[NonEmpty] = Field(min_length=1)


class CompoundResearchAndAnalysis(Contract):
    findings: list[SourceFinding]
    research_gaps: NonEmpty


class MorphologyPrediction(Contract):
    id: NonEmpty
    change: NonEmpty
    basis: NonEmpty
    applicability: NonEmpty
    source_ids: list[NonEmpty]


class ObservationGuidance(Contract):
    hypothesis_id: NonEmpty
    where_to_look: NonEmpty
    what_to_check: NonEmpty
    how_to_compare: NonEmpty


class ResearchResultV3(Contract):
    """Compound identification, source findings, morphology hypotheses, and observation guidance."""
    sample_id: NonEmpty
    compound_identification: CompoundIdentification
    compound_research_and_analysis: CompoundResearchAndAnalysis
    morphology_hypotheses: list[MorphologyPrediction]
    observation_guidance: list[ObservationGuidance]
    sources: list[ResearchSource]

    @model_validator(mode="after")
    def references_match(self):
        sources = [source.id for source in self.sources]
        hypotheses = [hypothesis.id for hypothesis in self.morphology_hypotheses]
        guidance = [item.hypothesis_id for item in self.observation_guidance]
        if len(sources) != len(set(sources)) or len(hypotheses) != len(set(hypotheses)):
            raise ValueError("Research source and hypothesis IDs must be unique")
        if len(guidance) != len(set(guidance)) or set(guidance) != set(hypotheses):
            raise ValueError("Each morphology hypothesis requires exactly one matching observation guidance entry")
        links = [self.compound_identification.source_ids]
        links.extend(finding.source_ids for finding in self.compound_research_and_analysis.findings)
        links.extend(hypothesis.source_ids for hypothesis in self.morphology_hypotheses)
        for ids in links:
            if len(ids) != len(set(ids)) or not set(ids) <= set(sources):
                raise ValueError("Research source references must be unique and cite declared sources")
        return self


class Finding(Contract):
    finding: NonEmpty
    reported_conditions: NonEmpty


class FindingsAndAnalysis(Contract):
    findings: list[Finding]
    research_gaps: NonEmpty


class CompoundSummary(Contract):
    summary: NonEmpty


class MorphologyHypothesis(Contract):
    id: NonEmpty
    change: NonEmpty
    basis: NonEmpty
    applicability: NonEmpty


class ResearchResultV4(Contract):
    """The original four research stages, with source provenance saved separately."""
    sample_id: NonEmpty
    compound_identification: CompoundSummary
    compound_research_and_analysis: FindingsAndAnalysis
    morphology_hypotheses: list[MorphologyHypothesis]
    observation_guidance: list[ObservationGuidance]

    @model_validator(mode="after")
    def guidance_matches(self):
        hypotheses = [row.id for row in self.morphology_hypotheses]
        guidance = [row.hypothesis_id for row in self.observation_guidance]
        if len(hypotheses) != len(set(hypotheses)):
            raise ValueError("Morphology hypothesis IDs must be unique")
        if len(guidance) != len(set(guidance)) or set(guidance) != set(hypotheses):
            raise ValueError("Each morphology hypothesis requires one matching observation guidance entry")
        return self


ResearchOutput = ResearchResultV4 | ResearchResultV3 | ResearchResult


def research_counts(result: ResearchOutput) -> dict[str, int]:
    if isinstance(result, (ResearchResultV3, ResearchResultV4)):
        return {"hypotheses": len(result.morphology_hypotheses),
                "findings": len(result.compound_research_and_analysis.findings)}
    return {"hypotheses": len(result.hypotheses), "facts": len(result.facts)}


def parse_research_result(value: Any) -> ResearchOutput:
    if isinstance(value, (ResearchResultV4, ResearchResultV3, ResearchResult)):
        return value
    if "compound_identification" in value or "morphology_hypotheses" in value:
        return (ResearchResultV3 if "sources" in value else ResearchResultV4).model_validate(value)
    if value.get("schema_version", "1") == "2":
        return ResearchResultV2.model_validate(value)
    return ResearchResult.model_validate(value)


class HypothesisView(Contract):
    """Only appearance-facing fields may cross into the observation request."""
    hypothesis_id: str
    change: str
    where_to_check: list[str]
    applicable_conditions: str
    uncertainty: str
    disconfirming_observation: str


class HypothesisViewV2(HypothesisView):
    basis: HypothesisBasis
    verification: VerificationPlan


class HypothesisViewV3(Contract):
    """Appearance and guidance only; research findings and basis remain private."""
    hypothesis_id: str
    change: str
    applicability: str
    where_to_look: str
    what_to_check: str
    how_to_compare: str


class ObservationInput(Contract):
    sample_id: str
    context: ExperimentContext
    hypotheses: list[HypothesisViewV3 | HypothesisViewV2 | HypothesisView]
    images: list[dict[str, str]]
    cp_available: bool


class Observation(Contract):
    observation_id: str
    text: str
    assessment: Literal["supported", "not_supported", "uncertain"]
    hypothesis_ids: list[str]
    image_ids: list[str]
    cp_feature_ids: list[str]
    visually_confirmed: bool
    scope: str

    @model_validator(mode="after")
    def visual_evidence_exists(self):
        if self.visually_confirmed and not self.image_ids:
            raise ValueError("Visual confirmation requires image evidence, not CP alone")
        return self


class DescriptionResult(Contract):
    schema_version: Literal["1"] = "1"
    sample_id: str
    description: str = Field(min_length=1)
    observations: list[Observation]
    limitations: list[str]

    @field_validator("description")
    @classmethod
    def english_description(cls, text: str) -> str:
        if re.search(r"[\u3400-\u9fff\uf900-\ufaff]", text):
            raise ValueError("The training description must be English")
        return text


class CPRow(Contract):
    feature_id: str
    feature_name: str
    compartment: str
    channel: str
    family: str
    role: Literal["treatment", "control"]
    value: float | None
    unit: str
    scope: str


def validate_research_references(result: ResearchOutput, sources: list[Source]) -> None:
    if isinstance(result, ResearchResultV4):
        ResearchResultV4.model_validate(result.model_dump(mode="json"))
        return  # Source records are collected by the application, not authored by the model.
    if isinstance(result, ResearchResultV3):
        ResearchResultV3.model_validate(result.model_dump(mode="json"))
        authentic = {source.source_id: source for source in sources}
        for source in result.sources:
            receipt = authentic.get(source.id)
            if receipt is None or source.url != receipt.url or source.title != receipt.title:
                raise ValueError("Research sources must match actual tool or web receipts (ID, title and URL)")
        return
    source_ids = {source.source_id for source in sources}
    declared = [source.source_id for source in result.sources]
    facts = [fact.fact_id for fact in result.facts]
    hypotheses = [h.hypothesis_id for h in result.hypotheses]
    if len(set(declared)) != len(declared) or len(set(facts)) != len(facts) or len(set(hypotheses)) != len(hypotheses):
        raise ValueError("Research IDs must be unique")
    authentic = {source.source_id: source for source in sources}
    if any(source.source_id not in authentic or source != authentic[source.source_id] for source in result.sources):
        raise ValueError("Research sources must match actual tool or web receipts")
    for fact in result.facts:
        if not set(fact.source_ids) <= set(declared) & source_ids:
            raise ValueError("Research fact cites an unknown source")
    for hypothesis in result.hypotheses:
        if not set(hypothesis.fact_ids) <= set(facts):
            raise ValueError("Hypothesis cites an unknown fact")


def validate_description_references(result: DescriptionResult, research: ResearchOutput,
                                    image_ids: set[str], cp_ids: set[str]) -> None:
    hypotheses = ({h.id for h in research.morphology_hypotheses} if isinstance(research, ResearchResultV3)
                  else {h.hypothesis_id for h in research.hypotheses})
    observations = [o.observation_id for o in result.observations]
    if len(observations) != len(set(observations)):
        raise ValueError("Observation IDs must be unique")
    for row in result.observations:
        if not set(row.hypothesis_ids) <= hypotheses:
            raise ValueError("Observation cites an unknown hypothesis")
        if not set(row.image_ids) <= image_ids:
            raise ValueError("Observation cites an unknown image")
        if not set(row.cp_feature_ids) <= cp_ids:
            raise ValueError("Observation cites an unknown CP feature")
