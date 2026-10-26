"""Saved-evidence synthesis uses real SDK conversion but no network or real key."""
from __future__ import annotations

from copy import deepcopy
import json

import httpx
import pytest

from realclip_d import agent_runtime as runtime
from realclip_d import pilot_continuation as continuation
from realclip_d.contracts import ResearchResultV2, ResearchResultV3, SampleInput, Source, ToolResult
from realclip_d.research_pilot import pilot_settings
from realclip_d.research_protocol import mock_research_v2, mock_research_v3
from realclip_d.storage import file_hash, read_json, verify_stage, write_json


@pytest.fixture
def saved_pilot(tmp_path, monkeypatch):
    """Invented identity and abstract are plumbing fixtures, not real evidence."""
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-before-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/before-test")
    directory = tmp_path / "pilot"
    settings = pilot_settings()
    sample = SampleInput(
        sample_id="SYNTHETIC_PILOT",
        compound={"name": "SYNTHETIC_COMPOUND", "pubchem_cid": 1234,
                  "inchikey": "AAAAAAAAAAAAAA-BBBBBBBBBB-C"},
        known_moa=["Synthetic supplied mechanism"],
    )
    stage = directory / "samples" / sample.sample_id / "research"
    input_path = directory / "inputs" / f"{sample.sample_id}.json"
    write_json(input_path, sample.model_dump(mode="json"))
    identity = Source(source_id="synthetic_identity", url="https://example.invalid/identity",
                      retrieved_at="2000-01-01T00:00:00Z", kind="mock")
    article = Source(source_id="synthetic_article", url="https://example.invalid/article",
                     retrieved_at="2000-01-01T00:00:00Z", kind="mock")
    sources = [identity, article]
    receipts = []
    for name, result in [
        ("resolve_compound", ToolResult(status="ok", sources=[identity], data={
            "identity_verified": True, "compound": {"CID": 1234,
                "InChIKey": sample.compound.inchikey}})),
        ("read_article", ToolResult(status="ok", sources=[article], data={
            "abstract": "Invented abstract reports more rounded nuclear outlines.",
            "body_text": None, "fulltext_status": "unavailable"})),
    ]:
        encoded = result.model_dump(mode="json")
        receipts.append({"tool": name, "arguments": [], "keyword_arguments": {},
                         "result": encoded, "model_result": deepcopy(encoded)})
    stage.mkdir(parents=True)
    (stage / "tools.jsonl").write_text("\n".join(json.dumps(row) for row in receipts) + "\n")
    write_json(stage / "input.json", sample.model_dump(mode="json"))
    write_json(stage / "settings.json", settings.model_dump(mode="json"))
    write_json(stage / "sources.json", [source.model_dump(mode="json") for source in sources])
    write_json(stage / "binding.json", {"fixture": "original research attempt"})
    write_json(stage / "requests" / "001.json", {"fixture": "original model request",
                                               "system_instructions": runtime._prompt("research", "v2")})
    write_json(stage / "responses" / "001.json", {"output": [], "fixture": "original model response"})
    original_status = {"sample_id": sample.sample_id, "name": "Synthetic compound", "status": "failed",
                       "error_type": "MaxTurnsExceeded", "error": "Synthetic earlier attempt reached its limit"}
    write_json(stage.parent / "status.json", original_status)
    manifest = {"settings": settings.model_dump(mode="json"), "budget_cap_usd": 5,
                "second_agent_enabled": False,
                "samples": [{"sample_id": sample.sample_id, "name": "Synthetic compound",
                             "input_path": str(input_path), "input_sha256": file_hash(input_path)}]}
    write_json(directory / "manifest.json", manifest)
    key_file = tmp_path / "authorized.env"
    key_file.write_text("OPENAI_API_KEY=synthetic-offline-test-key\n")

    payload = mock_research_v2(sample, sources).model_dump(mode="json")
    payload["identity_context"].update(status="verified", source_ids=[identity.source_id], conflicts=[])
    payload["facts"].append({"fact_id": "synthetic_direct", "statement": "Invented abstract reports rounder nuclei.",
        "source_ids": [article.source_id], "evidence_level": "direct_morphology", "origin": "literature",
        "compound_scope": "exact_compound", "access": "abstract",
        "conditions": "Invented experimental conditions; not the current sample.",
        "limitations": "Synthetic abstract only; no full text or matching sample conditions."})
    payload["literature_review"].update(status="direct_observations_found", fact_ids=["synthetic_direct"])
    payload["hypotheses"] = [{"hypothesis_id": "synthetic_roundness", "change": "Nuclear outlines may be rounder.",
        "where_to_check": ["DNA channel nuclear outlines"], "applicable_conditions": "Unknown cell line, dose and time.",
        "uncertainty": "Synthetic abstract fixture with unknown transferability.", "fact_ids": ["synthetic_direct"],
        "disconfirming_observation": "Matched treatment nuclei have the same or lower roundness.",
        "basis": "direct_literature", "rationale": "Invented abstract directly described the outline difference.",
        "verification": {"comparison": "Compare matched treatment and control fields.",
            "image_checks": ["Compare nuclear outlines across all visible fields."], "cp_checks": [],
            "supporting_observation": "Consistently rounder nuclei across treatment fields.",
            "not_assessable_when": ["No DNA images, no controls, or incompatible fields."]}}]
    final = ResearchResultV2.model_validate(payload)
    ledger = runtime.BudgetLedger(directory / "budget.json", 5, settings)
    old_request = ledger.reserve("research", True)
    ledger.complete(old_request, {"input_tokens": 200, "output_tokens": 100}, 1)
    return {"directory": directory, "settings": settings, "sample": sample, "stage": stage,
            "key_file": key_file, "candidate": final, "manifest": manifest, "ledger": ledger,
            "original_status": original_status, "original_request_id": old_request}


def _forbid_model(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("A continuation guard must run before constructing any model or API client")
    monkeypatch.setattr(continuation, "_live_model", forbidden)


def _mock_responses(monkeypatch, candidate):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={"id": "resp_synthetic_synthesis", "object": "response",
            "created_at": 1700000000, "status": "completed", "model": "gpt-6-luna",
            "output": [{"id": "msg_synthetic", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": candidate.model_dump_json(), "annotations": []}]}],
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 20}},
            "parallel_tool_calls": False, "tools": [], "tool_choice": "auto"})

    def fixture_model(settings):
        client = runtime._DAsyncOpenAI(api_key="synthetic-offline-transport-key", max_retries=0,
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        return runtime.OpenAIResponsesModel(model=settings.model, openai_client=client), client

    monkeypatch.setattr(continuation, "_live_model", fixture_model)
    return requests


@pytest.mark.asyncio
async def test_four_stage_saved_candidate_revalidates_locally_without_model_or_schema_upgrade(saved_pilot, monkeypatch):
    from agents import AgentOutputSchema

    fixture = saved_pilot
    stage = fixture["stage"]
    request = read_json(stage / "requests/001.json")
    request["system_instructions"] = runtime._prompt("research")
    request["output_schema"] = AgentOutputSchema(ResearchResultV3).json_schema()
    write_json(stage / "requests/001.json", request)
    write_json(stage / "output_contract.json", {"name": "ResearchResultV3", "schema_version": "3"})
    source = Source.model_validate(read_json(stage / "sources.json")[0])
    candidate = mock_research_v3(fixture["sample"], [source])
    write_json(stage / "candidate_result.json", candidate.model_dump(mode="json"))
    before = (fixture["directory"] / "budget.json").read_bytes()
    _forbid_model(monkeypatch)
    result = await continuation.complete_saved(fixture["directory"], fixture["key_file"])
    assert result["samples"][0]["findings"] == 0
    assert read_json(stage / "result.json") == candidate.model_dump(mode="json")
    assert (fixture["directory"] / "budget.json").read_bytes() == before
    assert not (stage / "continuation_01").exists()
    verify_stage(stage)


@pytest.mark.asyncio
async def test_four_stage_saved_synthesis_uses_original_contract_and_format_with_fake_transport(saved_pilot, monkeypatch):
    from agents import AgentOutputSchema

    fixture = saved_pilot
    stage = fixture["stage"]
    request = read_json(stage / "requests/001.json")
    request["system_instructions"] = runtime._prompt("research")
    request["output_schema"] = AgentOutputSchema(ResearchResultV3).json_schema()
    write_json(stage / "requests/001.json", request)
    write_json(stage / "output_contract.json", {"name": "ResearchResultV3", "schema_version": "3"})
    saved_format = runtime._prompt("research_output_format", "v2")
    (stage / "output_format.md").write_text(saved_format)
    source = Source.model_validate(read_json(stage / "sources.json")[0])
    candidate = mock_research_v3(fixture["sample"], [source])
    requests = _mock_responses(monkeypatch, candidate)
    original = (stage / "requests/001.json").read_bytes()
    outcome = await continuation.complete_saved(fixture["directory"], fixture["key_file"])
    assert len(requests) == 1 and outcome["samples"][0]["status"] == "complete"
    assert requests[0]["text"]["format"]["schema"] == request["output_schema"]
    assert requests[0]["input"][0] == {"role": "developer", "content": saved_format}
    assert "full_text literature fact" not in json.dumps(requests[0]["input"])
    assert (stage / "requests/001.json").read_bytes() == original
    assert "schema_version" not in read_json(stage / "result.json")
    verify_stage(stage)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending", "unknown"])
async def test_unreconciled_shared_usage_blocks_before_any_model(saved_pilot, monkeypatch, state):
    fixture = saved_pilot
    ledger = fixture["ledger"]
    request = ledger.reserve("research", False)
    if state == "unknown":
        ledger.unknown(request, "Synthetic interrupted response")
    before = (fixture["directory"] / "budget.json").read_bytes()
    _forbid_model(monkeypatch)
    with pytest.raises(ValueError, match="Unknown usage"):
        await continuation.complete_saved(fixture["directory"], fixture["key_file"])
    assert (fixture["directory"] / "budget.json").read_bytes() == before
    assert not (fixture["stage"] / "continuation_01").exists()


@pytest.mark.asyncio
async def test_existing_continuation_is_never_implicitly_resubmitted(saved_pilot, monkeypatch):
    fixture = saved_pilot
    sentinel = fixture["stage"] / "continuation_01" / "requests" / "001.json"
    write_json(sentinel, {"fixture": "an already submitted continuation"})
    original = sentinel.read_bytes()
    _forbid_model(monkeypatch)
    with pytest.raises(ValueError, match="already exists"):
        await continuation.complete_saved(fixture["directory"], fixture["key_file"])
    assert sentinel.read_bytes() == original
    assert len(read_json(fixture["directory"] / "budget.json")["requests"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["per_sample", "shared"])
async def test_original_request_limits_include_the_continuation(saved_pilot, monkeypatch, limit):
    fixture = saved_pilot
    if limit == "per_sample":
        for number in range(2, 17):
            write_json(fixture["stage"] / "requests" / f"{number:03d}.json", {"fixture": number})
    else:
        ledger = fixture["ledger"]
        for _ in range(47):
            request = ledger.reserve("research", False)
            ledger.complete(request, {"input_tokens": 1, "output_tokens": 1}, 0)
    _forbid_model(monkeypatch)
    with pytest.raises(ValueError, match="request-count limit"):
        await continuation.complete_saved(fixture["directory"], fixture["key_file"])
    assert not (fixture["stage"] / "continuation_01").exists()


@pytest.mark.asyncio
async def test_exact_saved_valid_candidate_is_revalidated_with_zero_additional_requests(saved_pilot, monkeypatch):
    fixture = saved_pilot
    stage, directory = fixture["stage"], fixture["directory"]
    # Local validation remains possible after all originally permitted turns;
    # the request cap guards new spending, not access to a result already paid for.
    for number in range(2, 17):
        write_json(stage / "requests" / f"{number:03d}.json", {"fixture": number})
    candidate_path = stage / "candidate_result.json"
    candidate_path.write_text(json.dumps(fixture["candidate"].model_dump(mode="json"), indent=4) + "\n")
    original_candidate = candidate_path.read_bytes()
    original_ledger = (directory / "budget.json").read_bytes()
    original_tools = (stage / "tools.jsonl").read_bytes()
    _forbid_model(monkeypatch)
    outcome = await continuation.complete_saved(directory, fixture["key_file"])
    assert outcome["samples"][0]["status"] == "complete"
    assert outcome["samples"][0]["offline_candidate_revalidated"] is True
    assert outcome["second_agent_calls"] == 0
    assert candidate_path.read_bytes() == original_candidate
    assert (stage / "tools.jsonl").read_bytes() == original_tools
    assert (directory / "budget.json").read_bytes() == original_ledger
    assert not list(stage.glob("continuation_*"))
    assert read_json(stage / "result.json") == fixture["candidate"].model_dump(mode="json")
    validation = read_json(stage / "validation.json")
    assert validation["offline_candidate_revalidated"] is True and validation["additional_model_calls"] == 0
    assert validation["biological_accuracy_validated"] is False
    assert read_json(stage.parent / "initial_status.json") == fixture["original_status"]
    verify_stage(stage)


@pytest.mark.asyncio
@pytest.mark.parametrize("has_separate_format", [False, True])
async def test_saved_evidence_synthesis_validates_and_seals_without_retrieval_or_observer(saved_pilot, monkeypatch, has_separate_format):
    fixture = saved_pilot
    stage, directory = fixture["stage"], fixture["directory"]
    saved_format = "OUTPUT FORMAT\nUse the saved schema and source IDs.\n"
    if has_separate_format:
        (stage / "output_format.md").write_text(saved_format)
    original_paths = [stage / "requests" / "001.json", stage / "responses" / "001.json",
                      stage / "tools.jsonl", stage / "input.json", stage / "settings.json", stage / "binding.json"]
    original_artifacts = {path: path.read_bytes() for path in original_paths}
    earlier_invalid = {"fixture": "earlier invalid final result remains available"}
    write_json(stage / "candidate_result.json", earlier_invalid)
    requests = _mock_responses(monkeypatch, fixture["candidate"])

    def forbidden(*_args, **_kwargs):
        pytest.fail("Saved synthesis must not retrieve new knowledge or run an observer")

    monkeypatch.setattr(runtime, "run_observation", forbidden)
    from realclip_d import workflow
    from realclip_d.tools import PubChemClient, ChEMBLClient, EuropePMCClient
    monkeypatch.setattr(workflow, "_describe", forbidden)
    monkeypatch.setattr(PubChemClient, "resolve_compound", forbidden)
    monkeypatch.setattr(ChEMBLClient, "get_evidence", forbidden)
    monkeypatch.setattr(EuropePMCClient, "read_article", forbidden)

    outcome = await continuation.complete_saved(directory, fixture["key_file"])
    assert len(requests) == 1
    sent = requests[0]
    assert sent["model"] == "gpt-6-luna" and sent["reasoning"]["effort"] == "high"
    assert sent["tools"] == [] and "max_tool_calls" not in sent
    format_messages = [item for item in sent["input"] if item.get("role") == "developer"]
    assert format_messages == ([{"role": "developer", "content": saved_format}] if has_separate_format else [])
    serialized = json.dumps(sent["input"])
    assert "saved_evidence" in serialized and "Invented abstract reports" in serialized
    assert "input_image" not in serialized and "cp_file" not in serialized
    assert outcome["second_agent_calls"] == 0 and outcome["budget_cap_usd"] == 5
    assert outcome["samples"][0]["status"] == "complete"
    assert read_json(stage / "result.json") == fixture["candidate"].model_dump(mode="json")
    assert read_json(stage / "candidate_result.json") == earlier_invalid
    assert read_json(stage / "continuation_01" / "candidate_result.json") == fixture["candidate"].model_dump(mode="json")
    assert read_json(stage.parent / "initial_status.json") == fixture["original_status"]
    assert read_json(stage.parent / "status.json")["initial_error"] == fixture["original_status"]
    assert read_json(stage / "validation.json")["completed_from_saved_evidence"] is True
    assert read_json(stage / "validation.json")["biological_accuracy_validated"] is False
    for path, data in original_artifacts.items():
        assert path.read_bytes() == data
    ledger = read_json(directory / "budget.json")
    assert ledger["budget_usd"] == 5 and len(ledger["requests"]) == 2
    assert ledger["requests"][0]["request_id"] == fixture["original_request_id"]
    assert ledger["spent_estimate_usd"] == pytest.approx(0.010125 + 0.0000625)
    assert ledger["requests"][1]["web_calls"] == 0
    verify_stage(stage)
    # A completed stage is read-only on later invocations; no new model request.
    _forbid_model(monkeypatch)
    second = await continuation.complete_saved(directory, fixture["key_file"])
    assert second["spent_estimate_usd"] == outcome["spent_estimate_usd"]
    assert len(read_json(directory / "budget.json")["requests"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["unread_article", "fabricated_source"])
async def test_invalid_synthesis_is_saved_but_cannot_become_a_completed_result(saved_pilot, monkeypatch, invalid):
    fixture = saved_pilot
    stage = fixture["stage"]
    candidate = fixture["candidate"]
    if invalid == "unread_article":
        rows = [json.loads(line) for line in (stage / "tools.jsonl").read_text().splitlines()]
        rows[1]["tool"] = "search_literature"
        for key in ("result", "model_result"):
            rows[1][key]["data"] = {"results": [{"id": "synthetic", "title": "A search hit only"}]}
        (stage / "tools.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    else:
        payload = candidate.model_dump(mode="json")
        payload["sources"][1]["url"] = "https://example.invalid/fabricated-url"
        candidate = ResearchResultV2.model_validate(payload)
    requests = _mock_responses(monkeypatch, candidate)
    outcome = await continuation.complete_saved(fixture["directory"], fixture["key_file"])
    assert len(requests) == 1
    assert outcome["samples"][0]["status"] == "failed"
    assert outcome["samples"][0]["error_type"] == "ValueError"
    assert (stage / "continuation_01" / "candidate_result.json").is_file()
    assert not (stage / "result.json").exists() and not (stage / "complete.json").exists()
    ledger = read_json(fixture["directory"] / "budget.json")
    assert len(ledger["requests"]) == 2 and all(row["status"] == "complete" for row in ledger["requests"])
    _forbid_model(monkeypatch)
    with pytest.raises(ValueError, match="already exists"):
        await continuation.complete_saved(fixture["directory"], fixture["key_file"])
