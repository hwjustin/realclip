"""Evidence boundary tests; no model key, public network or biological data."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from realclip_d.config import load_settings
from realclip_d.contracts import (
    CompoundIdentity,
    DescriptionResult,
    Observation,
    ResearchResult,
    SampleInput,
    Source,
    validate_description_references,
    validate_research_references,
)


@pytest.fixture
def source() -> Source:
    return Source(
        source_id="source_1",
        url="https://example.org/synthetic-evidence",
        retrieved_at="2026-01-01T00:00:00Z",
        kind="mock",
        title="Synthetic source",
    )


@pytest.fixture
def research(source: Source) -> ResearchResult:
    return ResearchResult(
        sample_id="TEST",
        identity_summary="Synthetic fixture",
        sources=[source],
        facts=[{
            "fact_id": "fact_1",
            "statement": "Synthetic measurement information only",
            "source_ids": [source.source_id],
            "evidence_level": "unknown",
            "conditions": "No biological experiment",
            "limitations": "Not evidence of an actual phenotype",
        }],
        hypotheses=[{
            "hypothesis_id": "hypothesis_1",
            "change": "Larger visible objects may be present",
            "where_to_check": ["DNA overview"],
            "applicable_conditions": "Synthetic images only",
            "uncertainty": "Software test fixture",
            "fact_ids": ["fact_1"],
            "disconfirming_observation": "Objects are equally sized or smaller",
        }],
        limitations=["Synthetic fixture"],
    )


@pytest.fixture
def description() -> DescriptionResult:
    return DescriptionResult(
        sample_id="TEST",
        description="Object appearance is uncertain in the synthetic fixture.",
        observations=[Observation(
            observation_id="observation_1",
            text="A CP-only difference lacks visual confirmation.",
            assessment="uncertain",
            hypothesis_ids=["hypothesis_1"],
            image_ids=[],
            cp_feature_ids=["cp_1"],
            visually_confirmed=False,
            scope="synthetic_well",
        )],
        limitations=["Synthetic fixture"],
    )


def test_cp_only_observation_cannot_be_visually_confirmed(description: DescriptionResult):
    payload = description.observations[0].model_dump()
    payload["visually_confirmed"] = True
    with pytest.raises(ValidationError, match="requires image evidence"):
        Observation.model_validate(payload)


def test_real_receipts_required_for_research_sources(research: ResearchResult, source: Source):
    validate_research_references(research, [source])
    changed = source.model_copy(update={"url": "https://example.org/altered"})
    with pytest.raises(ValueError, match="actual tool or web receipts"):
        validate_research_references(research, [changed])


@pytest.mark.parametrize("reference", ["source", "fact"])
def test_unknown_research_reference_rejected(research: ResearchResult, source: Source, reference: str):
    if reference == "source":
        research.facts[0].source_ids = ["unknown_source"]
    else:
        research.hypotheses[0].fact_ids = ["unknown_fact"]
    with pytest.raises(ValueError, match=f"unknown {reference}"):
        validate_research_references(research, [source])


@pytest.mark.parametrize("duplicate", ["source", "fact", "hypothesis"])
def test_duplicate_research_ids_rejected(research: ResearchResult, source: Source, duplicate: str):
    rows = {"source": research.sources, "fact": research.facts, "hypothesis": research.hypotheses}[duplicate]
    rows.append(rows[0].model_copy(deep=True))
    with pytest.raises(ValueError, match="must be unique"):
        validate_research_references(research, [source])


@pytest.mark.parametrize(
    ("field", "unknown", "message"),
    [
        ("hypothesis_ids", "unknown_hypothesis", "unknown hypothesis"),
        ("image_ids", "unknown_image", "unknown image"),
        ("cp_feature_ids", "unknown_feature", "unknown CP feature"),
    ],
)
def test_unknown_observation_reference_rejected(
    description: DescriptionResult, research: ResearchResult, field: str, unknown: str, message: str,
):
    setattr(description.observations[0], field, [unknown])
    with pytest.raises(ValueError, match=message):
        validate_description_references(description, research, {"image_1"}, {"cp_1"})


def test_valid_cp_only_observation_references(description: DescriptionResult, research: ResearchResult):
    validate_description_references(description, research, {"image_1"}, {"cp_1"})


def test_unexpected_morphology_needs_no_hypothesis_reference(
    description: DescriptionResult, research: ResearchResult,
):
    description.observations[0].hypothesis_ids = []
    validate_description_references(description, research, set(), {"cp_1"})


def test_no_research_findings_is_valid(source: Source):
    result = ResearchResult(
        sample_id="TEST", identity_summary="Unknown", sources=[source], facts=[], hypotheses=[],
        limitations=["No supporting literature was found"],
    )
    validate_research_references(result, [source])


def test_duplicate_observations_rejected(description: DescriptionResult, research: ResearchResult):
    description.observations.append(description.observations[0].model_copy(deep=True))
    with pytest.raises(ValueError, match="must be unique"):
        validate_description_references(description, research, set(), {"cp_1"})


def test_input_ids_unique():
    image = {"image_id": "same", "path": "fixture.png", "role": "control", "channel": "DNA", "field": "1"}
    with pytest.raises(ValidationError, match="Image IDs must be unique"):
        SampleInput(sample_id="TEST", compound={"name": "Synthetic"}, images=[image, image])


def test_compound_identity_requires_identifier():
    with pytest.raises(ValidationError, match="At least one compound"):
        CompoundIdentity()


def test_contracts_reject_unknown_fields():
    with pytest.raises(ValidationError, match="Extra inputs"):
        CompoundIdentity(name="Synthetic", unrecognized="value")


def test_training_description_requires_english(description: DescriptionResult):
    payload = description.model_dump()
    payload["description"] = "细胞形态描述"
    with pytest.raises(ValidationError, match="must be English"):
        DescriptionResult.model_validate(payload)


def test_examples_are_self_contained_and_explicitly_synthetic():
    examples = Path(__file__).resolve().parents[1] / "examples"
    sample = SampleInput.model_validate_json((examples / "sample.json").read_text())
    assert "SYNTHETIC" in sample.sample_id
    assert "SYNTHETIC" in sample.compound.name
    assert sample.cp_file is not None
    assert (examples / sample.cp_file).is_file()
    assert {image.role for image in sample.images} == {"treatment", "control"}
    assert all((examples / image.path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for image in sample.images)


def test_example_config_is_valid():
    root = Path(__file__).resolve().parents[1]
    settings = load_settings(root / "config.example.json")
    assert settings.model == "gpt-6-luna"
    assert settings.reasoning_effort == "high"
    assert settings.budget_usd is None
    assert json.loads((root / "config.example.json").read_text())["pubchem_requests_per_second"] == 2
