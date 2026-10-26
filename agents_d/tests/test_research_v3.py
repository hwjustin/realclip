"""Four-stage contract, source integrity and compatibility; no real model calls."""
from copy import deepcopy
import json
from pathlib import Path

from agents import AgentOutputSchema
import pytest

from realclip_d.contracts import (
    DescriptionResult, ResearchResultV3, SampleInput, Source, parse_research_result,
    validate_description_references, validate_research_references,
)
from realclip_d.inputs import hypothesis_view
from realclip_d.research_protocol import mock_research_v3, research_receipts, validate_research_protocol
from realclip_d.storage import write_json


@pytest.fixture
def payload():
    return json.loads((Path(__file__).parents[1] / "examples/research_v3_synthetic_example.json").read_text())


@pytest.fixture
def sample(payload):
    return SampleInput(sample_id=payload["sample_id"], compound={"name": "PRIVATE_COMPOUND"},
                       known_moa=["PRIVATE_MOA"], context={"cell_line": "U2OS"})


def test_schema_and_saved_schema_have_only_the_approved_stages():
    schema = AgentOutputSchema(ResearchResultV3).json_schema()
    saved = json.loads((Path(__file__).parents[1] / "docs/research_output_schema_v3.json").read_text())
    assert schema == saved
    assert set(schema["properties"]) == {
        "sample_id", "compound_identification", "compound_research_and_analysis",
        "morphology_hypotheses", "observation_guidance", "sources",
    }
    serialized = json.dumps(schema)
    for old_field in ("schema_version", "literature_review", "mechanism_review", "fact_ids",
                      "evidence_level", "compound_scope", "verification", "cp_checks", "cache_ref"):
        assert old_field not in serialized


def test_source_findings_and_predictions_keep_distinct_text_and_conditions(payload):
    result = parse_research_result(payload)
    assert isinstance(result, ResearchResultV3)
    assert result.model_dump(mode="json") == payload
    assert "reports" in result.compound_research_and_analysis.findings[0].finding
    assert "inference" in result.morphology_hypotheses[0].basis
    assert result.compound_research_and_analysis.findings[0].reported_conditions != result.morphology_hypotheses[0].applicability


@pytest.mark.parametrize("invalid", ["missing", "extra", "duplicate"])
def test_every_hypothesis_has_exactly_one_guidance_entry(payload, invalid):
    if invalid == "missing":
        payload["observation_guidance"] = []
    elif invalid == "extra":
        payload["observation_guidance"][0]["hypothesis_id"] = "unknown"
    else:
        payload["observation_guidance"].append(deepcopy(payload["observation_guidance"][0]))
    with pytest.raises(ValueError, match="matching observation guidance"):
        ResearchResultV3.model_validate(payload)


@pytest.mark.parametrize("where", ["identity", "finding", "hypothesis"])
def test_all_stage_source_references_must_exist(payload, where):
    owner = {"identity": payload["compound_identification"],
             "finding": payload["compound_research_and_analysis"]["findings"][0],
             "hypothesis": payload["morphology_hypotheses"][0]}[where]
    owner["source_ids"] = ["invented_source"]
    with pytest.raises(ValueError, match="declared sources"):
        ResearchResultV3.model_validate(payload)


@pytest.mark.parametrize("kind", ["source", "hypothesis"])
def test_duplicate_ids_are_rejected(payload, kind):
    rows = payload["sources" if kind == "source" else "morphology_hypotheses"]
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError, match="IDs must be unique"):
        ResearchResultV3.model_validate(payload)


@pytest.mark.parametrize("field", ["id", "title", "url"])
def test_minimal_source_must_match_authentic_receipt(payload, field):
    original = payload["sources"][0]
    receipt = Source(source_id=original["id"], title=original["title"], url=original["url"],
                     retrieved_at="2000-01-01T00:00:00Z", cache_ref="cache/original.json")
    result = ResearchResultV3.model_validate(payload)
    validate_research_references(result, [receipt])
    setattr(result.sources[0], field, "changed")
    with pytest.raises(ValueError):
        validate_research_references(result, [receipt])


def test_empty_research_and_input_moa_inference_need_no_dummy_facts(sample):
    result = mock_research_v3(sample, [])
    assert result.compound_research_and_analysis.findings == []
    assert result.morphology_hypotheses == result.observation_guidance == []
    payload = result.model_dump(mode="json")
    payload["morphology_hypotheses"] = [{"id": "H1", "change": "Outlines may be rounder.",
        "basis": "An inference from the supplied PRIVATE_MOA; no retrieved morphology observation.",
        "applicability": "Unknown dose and duration limit the U2OS prediction.", "source_ids": []}]
    payload["observation_guidance"] = [{"hypothesis_id": "H1", "where_to_look": "Visible outlines.",
        "what_to_check": "Outline roundness.", "how_to_compare": "Compare treated and control outlines."}]
    parsed = ResearchResultV3.model_validate(payload)
    validate_research_references(parsed, [])
    validate_research_protocol(parsed, sample, [{"tool": "resolve_compound", "result": {
        "status": "ambiguous", "sources": [], "data": {"candidates": []}}}])


@pytest.mark.parametrize("old_field", ["schema_version", "facts", "identity_context", "mechanism_review"])
def test_old_engineering_fields_cannot_enter_new_output(payload, old_field):
    payload[old_field] = "legacy"
    with pytest.raises(ValueError, match="Extra inputs"):
        parse_research_result(payload)


def test_blinded_projection_contains_guidance_but_no_basis_or_sources(payload, sample):
    payload["compound_identification"]["summary"] = "PRIVATE_COMPOUND identity."
    payload["morphology_hypotheses"][0]["basis"] = "PRIVATE_MOA from PRIVATE_COMPOUND; prediction only."
    result = ResearchResultV3.model_validate(payload)
    view = hypothesis_view(sample, result, [])
    serialized = view.model_dump_json()
    for private in ("PRIVATE_COMPOUND", "PRIVATE_MOA", "source_ids", "synthetic_source", "basis"):
        assert private not in serialized
    assert view.hypotheses[0].how_to_compare == result.observation_guidance[0].how_to_compare
    description = DescriptionResult(sample_id=sample.sample_id, description="Synthetic description.",
                                    observations=[], limitations=[])
    validate_description_references(description, result, set(), set())


def test_identity_citations_require_identity_receipts_and_sample_match(payload, sample):
    result = ResearchResultV3.model_validate(payload)
    with pytest.raises(ValueError, match="identity-tool attempt"):
        validate_research_protocol(result, sample, [])
    unrelated = [{"tool": "resolve_compound", "result": {"status": "not_found", "sources": []}}]
    with pytest.raises(ValueError, match="identity-tool receipts"):
        validate_research_protocol(result, sample, unrelated)
    wrong = sample.model_copy(update={"sample_id": "OTHER"})
    with pytest.raises(ValueError, match="another sample"):
        validate_research_protocol(result, wrong, unrelated)


def test_completed_web_annotations_are_valid_receipts_without_python_tool_call(tmp_path, sample):
    source = Source(source_id="web_fixture", title="Synthetic web source", url="https://example.invalid/page",
                    retrieved_at="2000-01-01T00:00:00Z", kind="web_search", cache_ref="responses/001.json")
    payload = mock_research_v3(sample, [source]).model_dump(mode="json")
    payload["compound_identification"]["source_ids"] = []
    payload["compound_research_and_analysis"]["findings"] = [{
        "finding": "A fictional source finding.", "reported_conditions": "Unreported.",
        "source_ids": [source.source_id]}]
    write_json(tmp_path / "sources.json", [source.model_dump(mode="json")])
    write_json(tmp_path / "responses/001.json", {"output": [
        {"type": "web_search_call", "status": "completed", "action": {"type": "search"}},
    ]})
    (tmp_path / "tools.jsonl").write_text(json.dumps({"tool": "resolve_compound", "result": {
        "status": "not_found", "sources": [], "data": {}}}) + "\n")
    result = ResearchResultV3.model_validate(payload)
    validate_research_references(result, [source])
    validate_research_protocol(result, sample, research_receipts(tmp_path))


def test_failed_only_sources_cannot_become_evidence_through_list_sources(payload, sample):
    result = ResearchResultV3.model_validate(payload)
    source = payload["sources"][0]
    receipt = {"source_id": source["id"], "title": source["title"], "url": source["url"]}
    receipts = [{"tool": "resolve_compound", "result": {"status": "not_found", "sources": [receipt]}},
                {"tool": "list_sources", "result": {"status": "ok", "sources": [receipt]}}]
    with pytest.raises(ValueError, match="failed-only sources"):
        validate_research_protocol(result, sample, receipts)
