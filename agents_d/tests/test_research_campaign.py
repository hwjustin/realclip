"""Unattended planning, cross-run reuse and honest cost/status records; offline."""
from pathlib import Path
import json
from types import SimpleNamespace
import threading

import pytest

from realclip_d.config import Settings
from realclip_d.contracts import SampleInput
from realclip_d.inputs import research_input
from realclip_d.research_campaign import CampaignProgress, completed_research, cost_summary, prepare_plan
from realclip_d.research_batch import run_batch
from realclip_d.storage import read_json, write_json


KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


def prior_run(root, mode="live", complete=True):
    path = root / "prior"
    inp = path / "input.json"
    sample = SampleInput(sample_id="old", compound={"inchikey": KEY},
                         context={"cell_line": "U2OS", "dose": None, "treatment_duration": None})
    write_json(inp, sample.model_dump(mode="json", exclude={"images": True, "cp_file": True}))
    # The SDK mock builds real stage seals without a paid request. Relabel only
    # the synthetic fixture's mode and re-seal it to exercise live reuse logic.
    outcome = run_batch([inp], path, Settings(), workers=1)
    assert outcome["samples"][0]["status"] == "complete"
    stage = path / "samples/old/research"
    binding = read_json(stage / "binding.json"); binding["mode"] = mode
    write_json(stage / "binding.json", binding)
    (stage / "complete.json").unlink()
    if complete:
        from realclip_d.storage import seal_stage
        seal_stage(stage)
    return stage


def test_reuse_is_based_on_verified_live_identity_and_context_not_sample_id(tmp_path):
    runs = tmp_path / "runs"
    stage = prior_run(runs)
    records = [{"compound_id": "JCP2022_1", "inchikey": KEY, "known_moa": ["Example mechanism"]},
               {"compound_id": "JCP2022_2", "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N", "known_moa": []}]
    result = prepare_plan(records, runs, runs / "new")
    assert len(result["reused"]) == len(result["scheduled"]) == 1
    assert result["reused"][0]["stage"] == str(stage)
    assert not (runs / "new/prepared_inputs/JCP2022_1.json").exists()
    payload = read_json(runs / "new/prepared_inputs/JCP2022_2.json")
    assert "images" not in payload and "cp_file" not in payload
    assert payload["context"] == {"cell_line": "U2OS", "dose": None, "treatment_duration": None}


def test_incomplete_live_attempt_is_held_and_mock_completion_is_not_reused(tmp_path):
    runs = tmp_path / "runs"
    prior_run(runs, complete=False)
    records = [{"compound_id": "JCP2022_1", "inchikey": KEY}]
    result = prepare_plan(records, runs, runs / "held")
    assert len(result["held_for_review"]) == 1 and not result["scheduled"]
    other = tmp_path / "mock_runs"
    prior_run(other, mode="mock")
    result = prepare_plan(records, other, other / "new")
    assert not result["reused"] and len(result["scheduled"]) == 1


def test_tampered_completion_is_not_reused_or_silently_repeated(tmp_path):
    runs = tmp_path / "runs"
    stage = prior_run(runs)
    with (stage / "result.json").open("a") as stream:
        stream.write(" ")
    result = prepare_plan([{"compound_id": "JCP2022_1", "inchikey": KEY}], runs, runs / "new")
    assert len(result["held_for_review"]) == 1 and not result["scheduled"]


def test_costs_include_cache_long_context_and_keep_unknown_usage_separate():
    ledger = {"budget_usd": 1000, "blocked": True, "spent_estimate_usd": 9,
        "requests": [{"status": "complete", "reserved_usd": 1, "web_calls": 2,
                      "usage": {"input_tokens": 300000, "output_tokens": 10000,
                                "input_tokens_details": {"cached_tokens": 100000, "cache_write_tokens": 50000}}},
                     {"status": "unknown", "reserved_usd": .3}, {"status": "pending", "reserved_usd": .3}]}
    result = cost_summary(ledger)
    expected = ((150000 * .1 + 100000 * .01 + 50000 * .125) * 2 + 10000 * .5 * 1.5) / 1e6 + .02
    assert result["known_standard_price_estimate_usd"] == pytest.approx(expected)
    assert result["unsettled_reserved_usd"] == pytest.approx(.6)
    assert result["request_statuses"] == {"complete": 1, "unknown": 1, "pending": 1}
    assert not result["billing_reconciled"] and result["blocked"]


def test_progress_persists_partial_failure_rate_limits_and_result_index(tmp_path):
    plan = {"workers": 32, "dataset_compounds": 3, "scheduled": [{}, {}],
            "reused": [{"compound_id": "previous", "result_path": "prior/result.json"}], "held_for_review": []}
    monitor = SimpleNamespace(_lock=threading.Lock(), http_events=[
        {"host": "api.openai.com", "status_code": 429, "transport_error_type": None,
         "response_limit_headers": {"retry-after": "3"}}])
    write_json(tmp_path / "status.json", {"samples": [
        {"sample_id": "one", "status": "complete"}, {"sample_id": "two", "status": "failed"}]})
    with CampaignProgress(tmp_path, plan, monitor, interval=999):
        pass
    assert read_json(tmp_path / "progress.json")["phase"] == "finished_with_errors"
    assert read_json(tmp_path / "progress.json")["completed_total"] == 2
    assert read_json(tmp_path / "rate_limits.json")["http_429_by_host"] == {"api.openai.com": 1}
    assert len(read_json(tmp_path / "failed_samples.json")) == 1
    assert len(read_json(tmp_path / "result_index.json")) == 3
