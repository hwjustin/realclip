"""Synthetic four-step research tests; no real compound, paper or paid call."""
from __future__ import annotations

from copy import deepcopy
import json

from agents import Agent, RunConfig, Runner
import pytest
from pydantic import ValidationError

from realclip_d import agent_runtime as runtime
from realclip_d.contracts import (
    DescriptionResult,
    ResearchResult,
    ResearchResultV2,
    SampleInput,
    Source,
    parse_research_result,
    validate_research_references,
)
from realclip_d.inputs import hypothesis_view
from realclip_d.research_protocol import mock_research_v2, validate_research_protocol


@pytest.fixture
def sample() -> SampleInput:
    return SampleInput(
        sample_id="SYNTHETIC_TEST",
        compound={"name": "SYNTHETIC_PRIVATE_COMPOUND"},
        known_moa=["SYNTHETIC_PRIVATE_MOA"],
        context={"cell_line": "Synthetic test cells", "dose": "synthetic dose"},
    )


@pytest.fixture
def sources() -> list[Source]:
    return [
        Source(source_id="synthetic_identity", url="https://example.invalid/identity",
               retrieved_at="2000-01-01T00:00:00Z", kind="mock"),
        Source(source_id="synthetic_article", url="https://example.invalid/article",
               retrieved_at="2000-01-01T00:00:00Z", kind="mock"),
    ]


@pytest.fixture
def payload(sample: SampleInput, sources: list[Source]) -> dict:
    """Manually invented observations and mechanism; never scientific evidence."""
    verification = {
        "comparison": "Compare treatment and control in matched DNA overviews, retaining field differences.",
        "image_checks": ["Inspect nuclear outlines and compare visible object sizes in both fields."],
        "cp_checks": [{"compartment": "Nuclei", "channel": "DNA", "family": "AreaShape",
                       "interpretation": "Use area measurements as corroboration; well values cannot confirm a crop."}],
        "supporting_observation": "Nuclear outlines are consistently larger in treatment than matched controls.",
        "not_assessable_when": ["DNA images are absent, blurred or saturated.",
                                "Fields disagree without a consistent directional difference."],
    }
    hypothesis = {
        "hypothesis_id": "direct_check",
        "change": "Larger nuclear outlines may be visible.",
        "where_to_check": ["DNA overview", "DNA local view"],
        "applicable_conditions": "Synthetic test cells and synthetic dose; treatment duration is unknown.",
        "uncertainty": "Synthetic fixture; transfer to the supplied duration cannot be evaluated.",
        "fact_ids": ["synthetic_direct"],
        "disconfirming_observation": "Nuclear outlines are the same size or smaller in treatment.",
        "basis": "direct_literature",
        "rationale": "SYNTHETIC_PRIVATE_COMPOUND was described in an invented abstract; test fixture only.",
        "verification": verification,
    }
    inferred = deepcopy(hypothesis)
    inferred.update(
        hypothesis_id="inferred_check", basis="moa_inference", fact_ids=["synthetic_prior"],
        change="More rounded object outlines may be visible.",
        rationale="SYNTHETIC_PRIVATE_MOA may affect a structural process and thereby alter object outlines; invented reasoning.",
        disconfirming_observation="Treatment outlines remain elongated to the same degree as controls.",
    )
    inferred["verification"]["image_checks"] = ["Compare outline roundness without treating intensity changes as shape."]
    inferred["verification"]["supporting_observation"] = "More treatment outlines are rounded across matched fields."
    return {
        "schema_version": "2", "sample_id": sample.sample_id,
        "identity_summary": "SYNTHETIC_PRIVATE_COMPOUND identity fixture only.",
        "sources": [source.model_dump(mode="json") for source in sources],
        "facts": [
            {"fact_id": "synthetic_direct", "statement": "Invented abstract reports larger nuclear outlines.",
             "source_ids": [sources[1].source_id], "evidence_level": "direct_morphology",
             "conditions": "Invented study: different synthetic cells, unknown time and dose.",
             "limitations": "Synthetic evidence; conditions differ from the sample.",
             "origin": "literature", "compound_scope": "exact_compound", "access": "abstract"},
            {"fact_id": "synthetic_prior", "statement": sample.known_moa[0], "source_ids": [],
             "evidence_level": "provided_moa", "conditions": "Supplied sample metadata; no assay was read.",
             "limitations": "User-provided prior, not independently established by a paper.",
             "origin": "provided_moa", "compound_scope": "exact_compound", "access": "input"},
        ],
        "hypotheses": [hypothesis, inferred],
        "limitations": ["Entirely synthetic test fixture, not biological evidence."],
        "identity_context": {"status": "verified", "source_ids": [sources[0].source_id],
                             "context": sample.context.model_dump(mode="json"),
                             "unknown_conditions": ["treatment_duration"], "conflicts": []},
        "literature_review": {"status": "direct_observations_found", "search_summary": "Invented literature read for tests.",
                              "fact_ids": ["synthetic_direct"], "limitations": ["Only an invented abstract."]},
        "mechanism_review": {"status": "hypotheses_proposed", "fact_ids": ["synthetic_prior"],
                             "summary": "Separate MOA-to-structure-to-appearance reasoning.",
                             "limitations": ["No biological conclusion can be drawn."]},
    }


@pytest.fixture
def receipts(sources: list[Source]) -> list[dict]:
    return [
        {"tool": "resolve_compound", "result": {"status": "ok", "data": {"identity_verified": True},
                                                 "sources": [sources[0].model_dump(mode="json")]}},
        {"tool": "read_article", "result": {"status": "ok", "data": {
            "abstract": "Invented abstract reports larger nuclear outlines.",
            "body_text": None, "fulltext_status": "unavailable"},
            "sources": [sources[1].model_dump(mode="json")]}},
    ]


def test_v2_distinguishes_direct_observation_and_moa_inference(payload, sample, receipts, sources):
    result = ResearchResultV2.model_validate(payload)
    validate_research_references(result, sources)
    validate_research_protocol(result, sample, receipts)
    assert [hypothesis.basis for hypothesis in result.hypotheses] == ["direct_literature", "moa_inference"]
    assert result.facts[0].access == "abstract"
    assert result.facts[0].conditions != result.hypotheses[0].applicable_conditions


@pytest.mark.parametrize("cite_xml", [True, False])
def test_fulltext_access_allows_same_article_metadata_only_with_cited_xml(payload, sample, receipts, cite_xml):
    xml = Source(source_id="synthetic_xml", url="https://example.invalid/PMC123/fullTextXML",
                 retrieved_at="2000-01-01T00:00:00Z", kind="mock").model_dump(mode="json")
    payload["sources"].append(xml)
    payload["facts"][0]["access"] = "full_text"
    if cite_xml:
        payload["facts"][0]["source_ids"].append(xml["source_id"])
    receipts[1]["result"]["sources"].append(xml)
    receipts[1]["result"]["data"].update(fulltext_status="accessible", body_text="Synthetic full article body.")
    result = ResearchResultV2.model_validate(payload)
    if cite_xml:
        validate_research_protocol(result, sample, receipts)
    else:
        with pytest.raises(ValueError, match="matching read_article"):
            validate_research_protocol(result, sample, receipts)


def test_fulltext_cocitation_cannot_promote_another_articles_abstract(payload, sample, receipts):
    xml = Source(source_id="synthetic_xml", url="https://example.invalid/PMC123/fullTextXML",
                 retrieved_at="2000-01-01T00:00:00Z", kind="mock").model_dump(mode="json")
    other = Source(source_id="other_article", url="https://example.invalid/other",
                   retrieved_at="2000-01-01T00:00:00Z", kind="mock").model_dump(mode="json")
    payload["sources"].extend([xml, other])
    payload["facts"][0].update(access="full_text", source_ids=[xml["source_id"], other["source_id"]])
    receipts[1]["result"]["sources"].append(xml)
    receipts[1]["result"]["data"].update(fulltext_status="accessible", body_text="Synthetic full body.")
    receipts.append({"tool": "read_article", "result": {"status": "ok", "sources": [other],
                    "data": {"abstract": "Another synthetic abstract", "fulltext_status": "unavailable"}}})
    with pytest.raises(ValueError, match="matching read_article"):
        validate_research_protocol(ResearchResultV2.model_validate(payload), sample, receipts)


@pytest.mark.parametrize("field,value", [
    ("fact_ids", []), ("applicable_conditions", "  "), ("where_to_check", []),
    ("rationale", " "), ("disconfirming_observation", ""),
])
def test_hypotheses_require_grounds_conditions_and_check_locations(payload, field, value):
    payload["hypotheses"][0][field] = value
    with pytest.raises(ValidationError):
        ResearchResultV2.model_validate(payload)


@pytest.mark.parametrize("field,value", [
    ("comparison", ""), ("image_checks", []), ("supporting_observation", " "),
    ("not_assessable_when", []),
])
def test_verification_must_explain_comparison_support_and_failure_to_assess(payload, field, value):
    payload["hypotheses"][0]["verification"][field] = value
    with pytest.raises(ValidationError):
        ResearchResultV2.model_validate(payload)


def test_cp_hints_require_interpretation_limits(payload):
    payload["hypotheses"][0]["verification"]["cp_checks"][0]["interpretation"] = ""
    with pytest.raises(ValidationError):
        ResearchResultV2.model_validate(payload)


def test_biochemical_fact_cannot_be_promoted_to_direct_observation(payload):
    payload["facts"][0].update(origin="database", access="annotation", evidence_level="biochemical")
    payload["literature_review"].update(status="no_morphology_found", fact_ids=[])
    with pytest.raises(ValidationError, match="Direct literature"):
        ResearchResultV2.model_validate(payload)


def test_parent_or_related_compound_observation_is_not_direct_exact_compound(payload):
    payload["facts"][0]["compound_scope"] = "related_compound"
    payload["literature_review"]["status"] = "indirect_observations_only"
    with pytest.raises(ValidationError, match="Direct literature"):
        ResearchResultV2.model_validate(payload)
    payload["hypotheses"][0]["basis"] = "indirect_literature"
    assert ResearchResultV2.model_validate(payload).hypotheses[0].basis == "indirect_literature"


def test_moa_inference_requires_a_retrieved_or_supplied_premise(payload):
    payload["facts"][1].update(origin="inference", evidence_level="mechanism_inference", access="not_available")
    with pytest.raises(ValidationError, match="premise"):
        ResearchResultV2.model_validate(payload)


def test_unknown_fact_reference_is_rejected(payload):
    payload["hypotheses"][0]["fact_ids"] = ["not_a_fact"]
    with pytest.raises(ValidationError, match="unknown fact"):
        ResearchResultV2.model_validate(payload)


def test_article_search_hit_is_not_actual_article_read(payload, sample, receipts):
    receipts[1]["tool"] = "search_literature"
    with pytest.raises(ValueError):
        validate_research_protocol(ResearchResultV2.model_validate(payload), sample, receipts)


def test_abstract_receipt_cannot_be_upgraded_to_full_text(payload, sample, receipts):
    payload["facts"][0]["access"] = "full_text"
    with pytest.raises(ValueError):
        validate_research_protocol(ResearchResultV2.model_validate(payload), sample, receipts)


@pytest.mark.parametrize("status,body", [("unavailable", None), ("accessible", ""), ("invalid_xml", "partial")])
def test_incomplete_or_unavailable_full_text_is_not_full_text_evidence(payload, sample, receipts, status, body):
    payload["facts"][0]["access"] = "full_text"
    receipts[1]["result"]["sources"][0]["url"] += "/fullTextXML"
    payload["sources"][1]["url"] += "/fullTextXML"
    receipts[1]["result"]["data"].update(fulltext_status=status, body_text=body)
    with pytest.raises(ValueError):
        validate_research_protocol(ResearchResultV2.model_validate(payload), sample, receipts)


def test_complete_full_text_receipt_can_support_full_text_fact(payload, sample, receipts):
    payload["facts"][0]["access"] = "full_text"
    receipts[1]["result"]["sources"][0]["url"] += "/fullTextXML"
    payload["sources"][1]["url"] += "/fullTextXML"
    receipts[1]["result"]["data"].update(fulltext_status="accessible", body_text="Invented complete article body.")
    validate_research_protocol(ResearchResultV2.model_validate(payload), sample, receipts)


def test_abstract_source_cannot_claim_full_text_access_from_another_xml_source(payload, sample, receipts):
    xml_source = deepcopy(payload["sources"][1])
    xml_source.update(source_id="synthetic_xml", url="https://example.invalid/PMC1/fullTextXML")
    payload["sources"].append(xml_source)
    receipts[1]["result"]["sources"].append(deepcopy(xml_source))
    receipts[1]["result"]["data"].update(fulltext_status="accessible", body_text="Invented complete body.")
    payload["facts"][0]["access"] = "full_text"
    with pytest.raises(ValueError):
        validate_research_protocol(ResearchResultV2.model_validate(payload), sample, receipts)
    payload["facts"][0]["source_ids"] = ["synthetic_xml"]
    validate_research_protocol(ResearchResultV2.model_validate(payload), sample, receipts)


def test_provided_moa_must_be_attributable_to_actual_sample(payload, sample, receipts):
    payload["facts"][1]["statement"] = "MOA not supplied in the sample"
    with pytest.raises(ValueError):
        validate_research_protocol(ResearchResultV2.model_validate(payload), sample, receipts)


def test_provided_moa_cannot_fabricate_external_citation(payload, sources):
    payload["facts"][1]["source_ids"] = [sources[0].source_id]
    with pytest.raises(ValidationError, match="Provided MOA"):
        ResearchResultV2.model_validate(payload)


def test_unverified_identity_cannot_back_a_direct_compound_hypothesis(payload, sample, receipts):
    payload["identity_context"]["status"] = "ambiguous"
    receipts[0]["result"].update(status="ambiguous", data={"identity_verified": False})
    with pytest.raises(ValueError):
        validate_research_protocol(ResearchResultV2.model_validate(payload), sample, receipts)


def test_v2_cannot_skip_identity_tool_even_with_no_findings(sample, sources):
    result = mock_research_v2(sample, sources)
    with pytest.raises(ValueError):
        validate_research_protocol(result, sample, [])


def test_default_mock_keeps_no_scientific_hypotheses(sample, sources):
    result = mock_research_v2(sample, sources)
    assert result.schema_version == "2"
    assert result.hypotheses == []
    assert [fact.statement for fact in result.facts] == sample.known_moa
    assert all(fact.origin == "provided_moa" and fact.access == "input" for fact in result.facts)
    assert "SYNTHETIC" in " ".join(result.limitations)


def test_v1_reading_does_not_silently_upgrade_evidence():
    legacy = ResearchResult(sample_id="LEGACY", identity_summary="Legacy synthetic fixture",
                            sources=[], facts=[], hypotheses=[], limitations=[])
    parsed = parse_research_result(legacy.model_dump(mode="json"))
    assert type(parsed) is ResearchResult
    assert parsed.schema_version == "1"


@pytest.mark.asyncio
async def test_sdk_nonempty_handoff_retains_verification_but_blinds_research(payload, sample):
    """Actual SDK typed-output and model boundary, entirely local and scripted."""
    research_agent = Agent(
        name="Synthetic research integration", model=runtime.ScriptedModel([], payload),
        output_type=ResearchResultV2,
    )
    response = await Runner.run(research_agent, "Synthetic research only", run_config=RunConfig(tracing_disabled=True))
    research = ResearchResultV2.model_validate(response.final_output)
    view = hypothesis_view(sample, research, [])
    request = json.loads(view.model_dump_json())
    first = request["hypotheses"][0]
    assert first["basis"] == "direct_literature"
    assert first["verification"] == payload["hypotheses"][0]["verification"]
    serialized = json.dumps(request)
    assert "SYNTHETIC_PRIVATE_COMPOUND" not in serialized
    assert "SYNTHETIC_PRIVATE_MOA" not in serialized
    assert "rationale" not in first and "fact_ids" not in first and "sources" not in request
    description = DescriptionResult(sample_id=sample.sample_id,
                                    description="Synthetic fixture; no biological interpretation was performed.",
                                    observations=[], limitations=["Synthetic integration only."])
    model = runtime.ScriptedModel([], description.model_dump(mode="json"))
    observer = Agent(name="Synthetic observation integration", model=model, output_type=DescriptionResult)
    observed = await Runner.run(observer, view.model_dump_json(), run_config=RunConfig(tracing_disabled=True))
    assert DescriptionResult.model_validate(observed.final_output) == description
