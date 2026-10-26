"""Language checks at actual SDK boundaries; no paid or network calls."""
from __future__ import annotations

import json

import pytest

from realclip_d import agent_runtime as runtime
from realclip_d import research_protocol
from realclip_d.config import Settings
from realclip_d.contracts import SampleInput


@pytest.mark.asyncio
async def test_non_english_research_input_stops_before_live_client_creation(tmp_path, monkeypatch):
    def forbidden(_settings):
        pytest.fail("Rejected input must not construct a paid API client")

    monkeypatch.setattr(runtime, "_live_model", forbidden)
    sample = SampleInput(sample_id="LANGUAGE_TEST", compound={"name": "synthetic"},
                         known_moa=["\u4e2d\u6587"])
    with pytest.raises(ValueError, match=r"English-only policy violation at \$\.known_moa\[0\]"):
        await runtime.run_research(sample, Settings(budget_usd=1), tmp_path / "stage", "live")
    assert not list(tmp_path.rglob("requests/*.json"))


@pytest.mark.asyncio
async def test_non_english_generated_output_is_rejected_with_raw_response_preserved(tmp_path, monkeypatch):
    original = research_protocol.mock_research_v3

    def invalid_narrative(sample, sources):
        value = original(sample, sources)
        value.compound_identification.summary = "\u4e2d\u6587"
        return value

    monkeypatch.setattr(research_protocol, "mock_research_v3", invalid_narrative)
    sample = SampleInput(sample_id="LANGUAGE_TEST", compound={"name": "synthetic"})
    with pytest.raises(ValueError, match=r"English-only policy violation at \$\.compound_identification.summary"):
        await runtime.run_research(sample, Settings(), tmp_path / "stage", "mock")
    raw = json.loads((tmp_path / "stage/responses/003.json").read_text())
    final = next(item for item in raw["output"] if item["type"] == "message")
    assert json.loads(final["content"][0]["text"])["compound_identification"]["summary"] == "\u4e2d\u6587"
    assert not (tmp_path / "stage/result.json").exists()


def test_research_prompt_explains_purpose_without_project_version_or_language_policy():
    current = runtime._prompt("research")
    assert current.startswith("You are a compound research agent")
    assert "testable hypotheses about cellular\nmorphological changes" in current
    assert "D's" not in current
    assert "prompt version" not in current and "research_v3" not in current
    assert "LANGUAGE POLICY" not in current and "Use English" not in current
    historical = runtime._prompt("research", "v2")
    assert "research_v2" in historical


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["prompt", "tool_source", "escaped_tool_source"])
async def test_complete_request_language_guard_runs_before_any_paid_reservation(tmp_path, location):
    settings = Settings(budget_usd=5)
    ledger = runtime.BudgetLedger(tmp_path / "budget.json", 5, settings)
    inner = runtime.ScriptedModel([], {})
    recorded = runtime.RecordedModel(inner, settings, tmp_path / "stage", "research", ledger, True, False)
    prompt = "Use English."
    tool_data = {"sources": [{"title": "English source"}]}
    if location == "prompt":
        prompt = "\u4e2d\u6587"
    else:
        tool_data["sources"][0]["title"] = "\u4e2d\u6587"
    tool_output = json.dumps(tool_data, ensure_ascii=location == "escaped_tool_source")
    with pytest.raises(ValueError, match="English-only model input required"):
        await recorded.get_response(system_instructions=prompt, input=[
            {"type": "function_call_output", "call_id": "synthetic", "output": tool_output}])
    assert inner.calls == 0 and ledger.state["requests"] == []
    assert not (tmp_path / "budget.json").exists()
