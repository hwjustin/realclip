"""One research run, unchanged scientific prompt, backend provenance; offline only."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket

from agents import AgentOutputSchema
import httpx
import pytest

from realclip_d import agent_runtime as runtime, workflow
from realclip_d.config import Settings
from realclip_d.contracts import ResearchResultV4, SampleInput, Source, ToolResult, parse_research_result
from realclip_d.language import validate_model_request
from realclip_d.pilot_report import generate_report
from realclip_d.storage import read_json, verify_stage, write_json


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("This test must not make a network request")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def scientific_result(sample_id="fixture"):
    return ResearchResultV4(
        sample_id=sample_id,
        compound_identification={"summary": "Synthetic compound identity; not real chemical evidence."},
        compound_research_and_analysis={"findings": [
            {"finding": "The fictional source reports target activity of 3 nM.",
             "reported_conditions": "Cell-free assay; not a cellular morphology measurement."},
            {"finding": "A separate fictional experiment reports smaller cell outlines.",
             "reported_conditions": "Synthetic cells; 1 uM for 24 hours."},
            {"finding": "Another fictional condition showed no outline change.",
             "reported_conditions": "Synthetic cells; 0.1 uM for 2 hours."}],
            "research_gaps": "Fictional fixtures only. Transfer to U2OS is unknown."},
        morphology_hypotheses=[{"id": "H1", "change": "Cell outlines may be smaller.",
            "basis": "The outline observation motivates an inference; the 3 nM assay alone does not establish morphology.",
            "applicability": "Unknown dose and time limit transfer to the current cells."}],
        observation_guidance=[{"hypothesis_id": "H1", "where_to_look": "Whole-cell outlines.",
            "what_to_check": "Compare apparent cell area across multiple cells.",
            "how_to_compare": "Compare matched treated and control fields; equal sizes challenge the prediction."}],
    )


def test_scientific_prompt_is_byte_identical_to_the_approved_version():
    prompt = Path(runtime.__file__).parent / "prompts/research_v3.md"
    assert hashlib.sha256(prompt.read_bytes()).hexdigest() == "2ee97f9ea07e849090b30fad70db9ac0a8b3ac2c4a59b821b843a5c463175fad"


def test_larger_explicit_input_reservation_keeps_the_full_research_context():
    assert Settings().max_input_tokens == 200000
    assert Settings(max_input_tokens=1000000).max_input_tokens == 1000000
    with pytest.raises(ValueError):
        Settings(max_input_tokens=1000001)


def test_schema_has_four_stages_without_citation_or_notebook_fields():
    schema = AgentOutputSchema(ResearchResultV4).json_schema()
    assert set(schema["properties"]) == {"sample_id", "compound_identification",
        "compound_research_and_analysis", "morphology_hypotheses", "observation_guidance"}
    for field in ("source_ids", "sources", "evidence_ids", "notes", "reading_tasks"):
        assert field not in json.dumps(schema)
    assert isinstance(parse_research_result(scientific_result().model_dump()), ResearchResultV4)


@pytest.mark.parametrize("mutation", ["missing_guidance", "duplicate_hypothesis", "empty_finding"])
def test_basic_output_structure_is_still_checked(mutation):
    payload = scientific_result().model_dump(mode="json")
    if mutation == "missing_guidance":
        payload["observation_guidance"] = []
    elif mutation == "duplicate_hypothesis":
        payload["morphology_hypotheses"].append(deepcopy(payload["morphology_hypotheses"][0]))
    else:
        payload["compound_research_and_analysis"]["findings"][0]["finding"] = ""
    with pytest.raises(ValueError):
        ResearchResultV4.model_validate(payload)


def test_exactly_one_runner_run_saves_the_actual_findings_without_another_pass(tmp_path, monkeypatch):
    calls = []
    original_run = runtime.Runner.run
    original_scripted = runtime.ScriptedModel
    expected = scientific_result()
    def scripted(steps, final):
        return original_scripted(steps, expected.model_dump(mode="json"))
    async def observe(agent, *args, **kwargs):
        calls.append({"agent": agent.name, "instructions": agent.instructions, "input": deepcopy(args[0]),
                      "filter": kwargs["run_config"].call_model_input_filter})
        return await original_run(agent, *args, **kwargs)
    monkeypatch.setattr(runtime, "ScriptedModel", scripted)
    monkeypatch.setattr(runtime.Runner, "run", observe)
    monkeypatch.setattr(runtime.Agent, "clone", lambda *args, **kwargs: pytest.fail("Research must not split into cloned passes"))
    sample = SampleInput(sample_id="fixture", compound={"name": "Synthetic"},
                         context={"cell_line": "U2OS", "dose": None, "treatment_duration": None})
    result = asyncio.run(runtime.run_research(sample, Settings(research_context_bytes=8000), tmp_path))
    assert result == expected and len(calls) == 1
    assert calls[0]["instructions"] == runtime._prompt("research")
    assert calls[0]["filter"] is None
    assert read_json(tmp_path / "findings.json") == expected.compound_research_and_analysis.model_dump(mode="json")
    assert not (tmp_path / "research_memory").exists()
    assert not (tmp_path / "synthesis_input.json").exists()
    assert not (tmp_path / "findings_format.md").exists()
    assert len(list((tmp_path / "requests").glob("*.json"))) == 3
    request = read_json(tmp_path / "requests/001.json")
    assert {tool["name"] for tool in request["tools"]} == {"resolve_compound", "get_pubchem_annotations",
        "get_chembl_evidence", "search_literature", "read_article"}
    assert len(request["input"]) == 2
    assert json.loads(request["input"][1]["content"])["context"] == {
        "cell_line": "U2OS", "dose": None, "treatment_duration": None}


def test_source_metadata_is_saved_in_backend_and_does_not_gate_new_results(tmp_path, monkeypatch):
    def no_citation_gate(*args, **kwargs):
        pytest.fail("New research must not run legacy citation validators")
    monkeypatch.setattr(workflow, "validate_research_references", no_citation_gate)
    monkeypatch.setattr(workflow, "validate_research_protocol", no_citation_gate)
    class Knowledge(runtime.MockKnowledge):
        def __init__(self):
            super().__init__()
            self.source = self.source.model_copy(update={"source_id": "backend_only_source",
                "title": "\u540e\u53f0\u539f\u59cb\u6765\u6e90\u6807\u9898"})
    monkeypatch.setattr(runtime, "MockKnowledge", Knowledge)
    sample = SampleInput(sample_id="fixture", compound={"name": "Synthetic"})
    stage = tmp_path / "research"
    result = asyncio.run(workflow._research(sample, Settings(), stage, "mock", None))
    assert "sources" not in result.model_dump()
    assert read_json(stage / "sources.json")[0]["source_id"] == "backend_only_source"
    tools = [json.loads(line) for line in (stage / "tools.jsonl").read_text().splitlines()]
    assert tools[0]["result"]["sources"][0]["title"] == "\u540e\u53f0\u539f\u59cb\u6765\u6e90\u6807\u9898"
    assert tools[0]["model_result"]["sources"] == []
    for path in (stage / "requests").glob("*.json"):
        request = read_json(path)
        validate_model_request(request)
        assert "backend_only_source" not in json.dumps(request)
    assert read_json(stage / "validation.json")["reference_validation"] == "not_required"
    verify_stage(stage)
    assert workflow._read_research(stage, sample) == result


def test_report_accepts_structurally_valid_research_with_backend_sources(tmp_path):
    stage = tmp_path / "samples/fixture/research"
    expected = scientific_result()
    for name, value in {"result": expected.model_dump(mode="json"), "input": {"sample_id": "fixture"},
                        "complete": {"files": {}}, "validation": {"structure_valid": True, "reference_validation": "not_required"},
                        "sources": []}.items():
        write_json(stage / (name + ".json"), value)
    write_json(tmp_path / "manifest.json", {"samples": [{"sample_id": "fixture", "name": "Synthetic"}]})
    summary = generate_report(tmp_path)
    assert summary["samples"][0]["completed"]
    assert summary["samples"][0]["result"] == expected.model_dump(mode="json")
    assert "Source provenance is saved separately" in (tmp_path / "index.html").read_text()
