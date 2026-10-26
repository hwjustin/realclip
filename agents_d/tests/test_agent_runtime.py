"""SDK integration tests use local scripted models or HTTP MockTransport only."""
import asyncio
import json
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from PIL import Image
import pytest

from realclip_d import agent_runtime as runtime
from realclip_d.config import Settings
from realclip_d.contracts import CompoundIdentity, DescriptionResult, ImageInput
from realclip_d.contracts import ResearchResult, ResearchResultV2, ResearchResultV3, ResearchResultV4, SampleInput
from realclip_d.contracts import Source, ToolResult
from realclip_d.inputs import hypothesis_view, image_manifest
from realclip_d.observation_stages import ObservationPipelineResult
from realclip_d.observation_protocol import prepare_observation
from realclip_d.research_protocol import mock_research_v2, mock_research_v3


def as_v4(payload):
    """Convert synthetic historical fixtures to the source-free current contract."""
    payload = json.loads(json.dumps(payload))
    payload.pop("sources", None)
    payload["compound_identification"].pop("source_ids", None)
    for row in payload["compound_research_and_analysis"]["findings"] + payload["morphology_hypotheses"]:
        row.pop("source_ids", None)
    return ResearchResultV4.model_validate(payload)


def sample_with_images(tmp_path):
    images = []
    for role in ("treatment", "control"):
        path = tmp_path / f"private-drug-{role}.png"
        Image.new("RGB", (8, 8), color="black").save(path)
        images.append(ImageInput(image_id=role, path=path, role=role,
                                 channel="DNA", field="1"))
    return SampleInput(sample_id="fixture", compound=CompoundIdentity(name="private-drug"),
                       known_moa=["private-mechanism"], images=images)


def test_mock_runs_actual_sdk_tools_without_network(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Mock must not instantiate an OpenAI client")
    monkeypatch.setattr(runtime, "AsyncOpenAI", no_network)
    sample = sample_with_images(tmp_path)
    research_dir, observation_dir = tmp_path / "research", tmp_path / "observation"
    research = asyncio.run(runtime.run_research(sample, Settings(), research_dir))
    view, _ = prepare_observation(sample, research)
    description = asyncio.run(runtime.run_observation(sample, view, Settings(), observation_dir))
    assert isinstance(research, ResearchResultV4)
    assert "schema_version" not in research.model_dump()
    assert isinstance(description, ObservationPipelineResult)
    assert "SYNTHETIC" in research.compound_research_and_analysis.research_gaps
    assert description.description == ""
    assert "SYNTHETIC" in description.limitations[0]
    research_tools = [json.loads(line)["tool"] for line in (research_dir / "tools.jsonl").read_text().splitlines()]
    assert research_tools == ["resolve_compound", "search_literature"]
    observation_tools = [json.loads(line)["tool"] for line in (observation_dir / "tools.jsonl").read_text().splitlines()]
    assert observation_tools == ["inspect_image", "query_cp"]
    assert len(list((research_dir / "responses").glob("*.json"))) == 3
    assert len(list((observation_dir / "responses").glob("*.json"))) == 6
    text = (observation_dir / "request_input.json").read_text()
    assert "private-drug" not in text and "private-mechanism" not in text
    assert "image_sha256" in text and "base64," not in text
    assert json.loads((observation_dir / "sources.json").read_text()) == []
    first_request = json.loads((research_dir / "requests" / "001.json").read_text())
    assert first_request["system_instructions"].startswith("You are a compound research agent")
    output_format = (research_dir / "output_format.md").read_text()
    assert first_request["input"][0] == {"role": "developer", "content": output_format}
    assert first_request["input"][1]["role"] == "user"
    assert json.loads(first_request["input"][1]["content"]) == json.loads(
        (research_dir / "request_input.json").read_text())
    assert output_format not in first_request["system_instructions"]
    assert "input_image" not in json.dumps(first_request["input"])
    assert "cp_file" not in json.dumps(first_request["input"])
    observer_request = json.loads((observation_dir / "phases/baseline/requests/001.json").read_text())
    assert "(observation_v4)" not in observer_request["system_instructions"]
    assert "hypotheses" not in observer_request["input"][0]["content"][0]["text"]
    assert output_format not in json.dumps(observer_request)


def test_observation_sends_real_image_bytes_with_labels(tmp_path):
    sample = sample_with_images(tmp_path)
    research = ResearchResult(sample_id=sample.sample_id, identity_summary="unused",
                              sources=[], facts=[], hypotheses=[], limitations=[])
    view = hypothesis_view(sample, research, image_manifest(sample))
    messages = runtime.observation_messages(sample, view)
    content = messages[0]["content"]
    assert sum(item["type"] == "input_image" for item in content) == 2
    assert all(item["image_url"].startswith("data:image/png;base64,")
               for item in content if item["type"] == "input_image")
    assert "private-drug" not in json.dumps(messages)


def test_budget_blocks_pending_and_unknown_requests(tmp_path):
    settings = Settings(budget_usd=1)
    ledger = runtime.BudgetLedger(tmp_path / "budget.json", 1, settings)
    ledger.reserve("research", False)
    restarted = runtime.BudgetLedger(tmp_path / "budget.json", 1, settings)
    with pytest.raises(runtime.UnknownUsageError):
        restarted.reserve("research", False)
    assert len(restarted.state["requests"]) == 1


def test_budget_preflight_and_known_usage(tmp_path):
    settings = Settings(budget_usd=1)
    ledger = runtime.BudgetLedger(tmp_path / "budget.json", 1, settings)
    request = ledger.reserve("research", True)
    ledger.complete(request, {"input_tokens": 100, "output_tokens": 40}, 1)
    assert ledger.state["spent_estimate_usd"] == pytest.approx(0.0100325)
    small = runtime.BudgetLedger(tmp_path / "small.json", 0.000001, settings)
    with pytest.raises(runtime.BudgetError):
        small.reserve("research", False)
    assert not small.state["requests"]


def test_web_sources_only_from_hosted_results_and_annotations(tmp_path):
    model = runtime.RecordedModel(runtime.ScriptedModel([], {}), Settings(), tmp_path,
                                  "research", None, False, True)
    model._capture_web_sources({"output": [
        {"type": "web_search_call", "action": {"sources": [{"url": "https://official.example/article", "title": "Article"}]}},
        {"type": "message", "content": [{"type": "output_text", "text": '{"url":"https://invented.example"}',
                                           "annotations": [{"type": "url_citation", "url": "https://official.example/second", "title": "Second"}]}]},
    ]}, "responses/001.json")
    assert {source.url for source in model.web_sources} == {
        "https://official.example/article", "https://official.example/second"}
    assert all(source.cache_ref == "responses/001.json" for source in model.web_sources)


def test_tool_sources_are_portable_with_saved_stage(tmp_path):
    stage = tmp_path / "research"
    recorder = runtime._ToolRecorder(stage, 2)
    source = Source(source_id="api_fixture", url="https://official.example/data",
                    retrieved_at="2000-01-01T00:00:00+00:00",
                    cache_ref=str(stage / "cache" / "digest.json"))
    result = asyncio.run(recorder.call("fixture", lambda: ToolResult(status="ok", sources=[source])))
    assert json.loads(result)["sources"][0]["cache_ref"] == "cache/digest.json"
    assert recorder.sources["api_fixture"].cache_ref == "cache/digest.json"


def test_actual_responses_sdk_transport_is_recorded_and_budgeted(tmp_path, monkeypatch):
    """Exercise the real SDK converter and HTTP request, with zero network."""
    sample = sample_with_images(tmp_path)
    final = as_v4(mock_research_v3(sample, []).model_dump(mode="json"))
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={
            "id": "resp_fixture", "object": "response", "created_at": 1700000000,
            "status": "completed", "model": "gpt-6-luna",
            "output": [{"id": "msg_fixture", "type": "message", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text",
                        "text": final.model_dump_json(), "annotations": []}]}],
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 20}},
            "parallel_tool_calls": False, "tools": [], "tool_choice": "auto",
        })

    def fixture_model(settings):
        client = runtime._DAsyncOpenAI(api_key="fixture-not-a-real-key", max_retries=0,
                            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        assert client._platform is not None
        assert client.with_options(max_retries=0)._platform is not None
        assert client.max_retries == 0
        return runtime.OpenAIResponsesModel(model=settings.model, openai_client=client), client

    monkeypatch.setattr(runtime, "_live_model", fixture_model)
    settings = Settings(budget_usd=1, web_search=True)
    ledger = runtime.BudgetLedger(tmp_path / "budget.json", 1, settings)
    stage = tmp_path / "live-fixture"
    result = asyncio.run(runtime.run_research(sample, settings, stage, "live", ledger))
    assert result == final
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "gpt-6-luna"
    assert request["reasoning"]["effort"] == "high"
    assert request["store"] is False
    assert request["service_tier"] == "default"
    assert request["max_output_tokens"] == settings.max_output_tokens
    assert request["max_tool_calls"] == settings.web_max_calls
    assert "web_search_call.action.sources" in request["include"]
    assert request["text"]["format"]["strict"] is True
    output_format = (stage / "output_format.md").read_text()
    assert request["input"][0] == {"role": "developer", "content": output_format}
    assert output_format not in request["instructions"]
    schema = request["text"]["format"]["schema"]
    assert set(schema["properties"]) == {"sample_id", "compound_identification", "compound_research_and_analysis",
                                         "morphology_hypotheses", "observation_guidance"}
    hypothesis_schema = schema["$defs"]["MorphologyHypothesis"]
    assert set(hypothesis_schema["required"]) == {"id", "change", "basis", "applicability"}
    assert json.loads((stage / "output_schema.json").read_text()) == schema
    assert json.loads((stage / "output_contract.json").read_text())["schema_version"] == "4"
    assert "private-drug" in json.dumps(request["input"])
    assert not any(item.get("type") == "input_image" for item in request["input"])
    assert ledger.state["requests"][0]["status"] == "complete"
    raw = json.loads((stage / "responses" / "001.json").read_text())
    assert raw["raw_usage"]["input_tokens"] == 100
    assert raw["raw_usage"]["output_tokens_details"]["reasoning_tokens"] == 20


def test_v2_prompts_preserve_the_four_stages_and_blinded_checklist():
    prompt = runtime._prompt("research")
    headings = ["1. COMPOUND IDENTIFICATION", "2. COMPOUND RESEARCH AND ANALYSIS",
                "3. MORPHOLOGY HYPOTHESES", "4. OBSERVATION GUIDANCE"]
    positions = [prompt.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "EVIDENCE AND HANDOFF RULES" not in prompt
    assert "OUTPUT FORMAT" not in prompt
    observation = runtime._prompt("observation")
    for requirement in ("image_checks", "cp_checks", "not_assessable_when",
                        "visually_confirmed=false", "uncertain", "unexpected"):
        assert requirement in observation
    assert "research_v1" in runtime._prompt("research", "v1")
    assert "observation_v1" in runtime._prompt("observation", "v1")


@pytest.mark.parametrize("article_readable", [True, False])
def test_workflow_validates_nonempty_sdk_research_before_sealing(tmp_path, monkeypatch, article_readable):
    """Exercise real workflow + SDK tool loop, with invented scientific content."""
    from realclip_d import workflow

    sample = sample_with_images(tmp_path)
    identity_source = runtime.MockKnowledge().source
    article_source = Source(source_id="synthetic_read_article",
                            url="https://example.invalid/synthetic-abstract",
                            retrieved_at="2000-01-01T00:00:00+00:00", kind="mock")
    sources = [identity_source, article_source]
    final = mock_research_v3(sample, sources).model_dump(mode="json")
    final["compound_identification"]["source_ids"] = [identity_source.source_id]
    final["compound_research_and_analysis"]["findings"] = [{
        "finding": "An invented abstract reports larger outlines.",
        "reported_conditions": "Synthetic cells; dose and duration unreported.",
        "source_ids": [article_source.source_id],
    }]
    final["morphology_hypotheses"] = [{
        "id": "outline", "change": "Larger outlines may be visible.",
        "basis": "The invented source reports larger outlines; applicability here is an inference.",
        "applicability": "Unknown dose and duration limit this synthetic prediction.",
        "source_ids": [article_source.source_id],
    }]
    final["observation_guidance"] = [{
        "hypothesis_id": "outline", "where_to_look": "DNA views if available.",
        "what_to_check": "Inspect outline sizes across fields.",
        "how_to_compare": "Compare matched treatment and control fields; similar outlines challenge this prediction.",
    }]
    final = as_v4(final).model_dump(mode="json")
    if not article_readable:
        final["compound_research_and_analysis"] = {"findings": [], "research_gaps": "The synthetic article is unavailable."}
        final["morphology_hypotheses"] = []
        final["observation_guidance"] = []

    class SyntheticKnowledge(runtime.MockKnowledge):
        def resolve(self, identity):
            result = super().resolve(identity)
            return result.model_copy(update={"data": {**result.data, "identity_verified": True,
                "compound": {"CID": 1, "InChIKey": "AAAAAAAAAAAAAA-BBBBBBBBBB-C"}}})

        def read(self, article_id, source, **kwargs):
            return ToolResult(status="ok" if article_readable else "unavailable", sources=[article_source],
                              data={"abstract": "Invented larger-outline observation." if article_readable else None,
                                    "body_text": None, "fulltext_status": "unavailable"})

    original_scripted = runtime.ScriptedModel
    monkeypatch.setattr(runtime, "MockKnowledge", SyntheticKnowledge)
    monkeypatch.setattr(runtime, "ScriptedModel", lambda _steps, _final: original_scripted([
        [("resolve_compound", {})],
        [("search_literature", {"query": "synthetic morphology", "limit": 2})],
        [("read_article", {"article_id": "SYNTHETIC", "source": "MED"})],
    ], final))

    def no_network(*args, **kwargs):
        raise AssertionError("Synthetic workflow must not instantiate an OpenAI client")

    monkeypatch.setattr(runtime, "AsyncOpenAI", no_network)
    input_path = tmp_path / "sample.json"
    input_path.write_text(sample.model_dump_json())
    output = tmp_path / "workflow"
    execute = workflow.execute("research", input_path, output, Settings())
    status = asyncio.run(execute)
    assert status["status"] == "complete"
    saved = workflow._read_research(output / "research", sample)
    assert [item.id for item in saved.morphology_hypotheses] == (["outline"] if article_readable else [])
    assert (output / "research/complete.json").exists()
    assert (output / "research/findings.json").exists()
    assert json.loads((output / "research/validation.json").read_text())["reference_validation"] == "not_required"


def test_unrelated_chembl_query_cannot_verify_the_supplied_compound(tmp_path):
    from realclip_d.research_protocol import validate_research_protocol

    sample = sample_with_images(tmp_path)
    source = Source(source_id="unrelated_chembl", url="https://example.invalid/unrelated-molecule",
                    retrieved_at="2000-01-01T00:00:00+00:00", kind="mock")
    payload = mock_research_v2(sample, [source]).model_dump(mode="json")
    payload["identity_context"].update(status="verified", source_ids=[source.source_id])
    result = ResearchResultV2.model_validate(payload)
    receipts = [{
        "tool": "get_chembl_evidence", "keyword_arguments": {"chembl_id": "CHEMBL999"},
        "result": {"status": "ok", "sources": [source.model_dump(mode="json")], "data": {
            "identity_verified": True, "exact_molecule_id": "CHEMBL999",
            "molecule": {"molecule_chembl_id": "CHEMBL999",
                         "molecule_structures": {"standard_inchi_key": "AAAAAAAAAAAAAA-BBBBBBBBBB-C"}},
        }},
    }, {"tool": "search_literature", "result": {"status": "not_found", "sources": [], "data": {}}}]
    with pytest.raises(ValueError, match="identity|Identity"):
        validate_research_protocol(result, sample, receipts)


def test_unknown_usage_blocks_without_automatic_retry(tmp_path):
    settings = Settings(budget_usd=1)
    ledger = runtime.BudgetLedger(tmp_path / "budget.json", 1, settings)
    inner = runtime.ScriptedModel([], {"ok": True})
    model = runtime.RecordedModel(inner, settings, tmp_path / "calls", "research", ledger, True, False)
    with pytest.raises(runtime.UnknownUsageError):
        asyncio.run(model.get_response(system_instructions="fixture", input="fixture",
                    model_settings=runtime._model_settings(settings, False), tools=[],
                    output_schema=None, handoffs=[], tracing=None,
                    previous_response_id=None, conversation_id=None, prompt=None))
    assert inner.calls == 1
    assert ledger.state["blocked"]
    with pytest.raises(runtime.UnknownUsageError):
        ledger.reserve("observation", False)
