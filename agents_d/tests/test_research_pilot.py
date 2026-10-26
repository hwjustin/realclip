"""Shared pilot spending and explicit credential loading; entirely offline."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from realclip_d import research_pilot as pilot
from realclip_d.agent_runtime import BudgetError, BudgetLedger
from realclip_d.contracts import ToolResult


def test_frozen_three_case_plan_fits_one_five_dollar_budget(tmp_path):
    settings = pilot.pilot_settings()
    assert settings.model == "gpt-6-luna"
    assert settings.reasoning_effort == "high"
    assert settings.budget_usd == 5
    reservation = (settings.max_input_tokens * settings.input_usd_per_million / 1_000_000
                   + settings.max_output_tokens * settings.output_usd_per_million / 1_000_000
                   + settings.web_max_calls * settings.web_search_usd_per_call)
    calls = len(pilot.CASES) * settings.research_max_turns
    assert calls == 48
    assert calls * reservation == pytest.approx(4.872)
    assert calls * reservation <= settings.budget_usd
    ledger = BudgetLedger(tmp_path / "budget.json", 5, settings)
    for _ in range(calls):
        request = ledger.reserve("research", web_enabled=True)
        ledger.complete(request, {"input_tokens": settings.max_input_tokens,
                                  "output_tokens": settings.max_output_tokens},
                        settings.web_max_calls)
    assert ledger.state["spent_estimate_usd"] == pytest.approx(4.872)
    assert not ledger.state["blocked"]
    # Even if the planned turn count were accidentally exceeded, the common
    # ledger remains authoritative rather than granting $5 per compound.
    ledger.reserve("research", web_enabled=True)
    with pytest.raises(BudgetError, match="cannot reserve"):
        ledger.reserve("research", web_enabled=True)


@pytest.mark.parametrize("entry,expected", [
    ("OPENAI_API_KEY=synthetic-key", "synthetic-key"),
    ('export OPENAI_API_KEY="synthetic key" # comment', "synthetic key"),
    ("  OPENAI_API_KEY = 'synthetic-key'", "synthetic-key"),
])
def test_key_parser_reads_only_authorized_entry_and_clears_base_url(tmp_path, monkeypatch, entry, expected):
    monkeypatch.setenv("OPENAI_API_KEY", "preexisting-synthetic-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/synthetic-proxy")
    monkeypatch.setenv("UNRELATED_STATE", "original")
    credential = tmp_path / "authorized.env"
    credential.write_text('UNRELATED_STATE="changed"\nOPENAI_BASE_URL=https://example.invalid\n'
                          'UNPARSEABLE_OTHER_KEY="unterminated\n' + entry + "\n")
    pilot.load_authorized_key(credential)
    assert os.environ["OPENAI_API_KEY"] == expected
    assert "OPENAI_BASE_URL" not in os.environ
    assert os.environ["UNRELATED_STATE"] == "original"


def test_key_parser_never_executes_shell_substitution(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-old")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid")
    sentinel = tmp_path / "must-not-be-created"
    literal = f"$(touch {sentinel})"
    credential = tmp_path / "authorized.env"
    credential.write_text(f"OPENAI_API_KEY='{literal}'\nOTHER=$(touch {sentinel})\n")
    pilot.load_authorized_key(credential)
    assert os.environ["OPENAI_API_KEY"] == literal
    assert not sentinel.exists()


@pytest.mark.parametrize("contents", [
    "OTHER=synthetic\n",
    "# OPENAI_API_KEY=commented-out\n",
    "OPENAI_API_KEY=\n",
    "OPENAI_API_KEY=''\n",
    "OPENAI_API_KEY= # comment\n",
    "OPENAI_API_KEY='unterminated\n",
    "OPENAI_API_KEY=two tokens\n",
    "OPENAI_API_KEY=one\nOPENAI_API_KEY=two\n",
    "OPENAI_API_KEY=one\nexport OPENAI_API_KEY=one\n",
])
def test_key_parser_rejects_missing_empty_malformed_or_duplicate_entries(tmp_path, monkeypatch, contents):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-old")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/original")
    credential = tmp_path / "authorized.env"
    credential.write_text(contents)
    with pytest.raises(ValueError):
        pilot.load_authorized_key(credential)
    assert os.environ["OPENAI_API_KEY"] == "synthetic-old"
    assert os.environ["OPENAI_BASE_URL"] == "https://example.invalid/original"


def test_selected_cases_have_distinct_priors_and_explicit_unknown_conditions():
    assert [case["name"] for case in pilot.CASES] == ["Nocodazole", "Staurosporine", "Rapamycin"]
    assert len({tuple(case["known_moa"]) for case in pilot.CASES}) == 3
    assert len({case["cid"] for case in pilot.CASES}) == 3
    assert all(case["selection_reason"] and case["selection_source"] for case in pilot.CASES)


@pytest.mark.asyncio
async def test_pilot_runs_only_research_sequentially_with_one_shared_ledger(tmp_path, monkeypatch):
    class SyntheticHTTP:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    class SyntheticPubChem:
        def __init__(self, _http):
            pass

        def resolve_compound(self, identity):
            # Synthetic structural identifier; only parser/plumbing validation.
            letter = {4122: "A", 44259: "B", 5284616: "C"}[identity.pubchem_cid]
            return ToolResult(status="ok", data={"compound": {
                "CID": identity.pubchem_cid, "InChIKey": f"{letter * 14}-{letter * 10}-{letter}"}})

    calls = []

    async def synthetic_research(sample, settings, directory, mode, budget):
        calls.append((sample, directory, mode, budget))
        return SimpleNamespace(hypotheses=[], facts=[])

    async def forbidden_observer(*_args, **_kwargs):
        pytest.fail("Research-only pilot must never run the observer")

    monkeypatch.setattr(pilot, "CachedHTTP", SyntheticHTTP)
    monkeypatch.setattr(pilot, "PubChemClient", SyntheticPubChem)
    monkeypatch.setattr(pilot, "_research", synthetic_research)
    from realclip_d import agent_runtime, workflow
    monkeypatch.setattr(agent_runtime, "run_observation", forbidden_observer)
    monkeypatch.setattr(workflow, "_describe", forbidden_observer)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-old")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid")
    directory = tmp_path / "pilot"
    manifest = pilot.prepare(directory)
    assert manifest["maximum_reserved_plan_usd"] == pytest.approx(4.872)
    assert manifest["second_agent_enabled"] is False
    assert manifest["images_or_cp_supplied"] is False
    key_file = tmp_path / "authorized.env"
    key_file.write_text("OPENAI_API_KEY=synthetic-offline-test\n")
    outcome = await pilot.run(directory, key_file)
    assert len(calls) == 3
    assert len({id(call[3]) for call in calls}) == 1
    assert all(call[2] == "live" for call in calls)
    assert all(call[3].state["budget_usd"] == 5 for call in calls)
    for sample, stage, _, _ in calls:
        assert not sample.images and sample.cp_file is None
        assert sample.context.cell_line == "U2OS"
        assert sample.context.dose is None
        assert sample.context.treatment_duration is None
        assert sample.context.notes == []
        prepared = pilot.read_json(directory / "inputs" / f"{sample.sample_id}.json")
        assert "images" not in prepared and "cp_file" not in prepared
        assert prepared["context"] == {"cell_line": "U2OS", "dose": None, "treatment_duration": None}
        assert stage.name == "research"
    assert outcome["second_agent_calls"] == 0
    assert outcome["spent_estimate_usd"] == 0
    assert all(row["status"] == "complete" for row in outcome["samples"])
    assert not list(directory.rglob("observation"))
