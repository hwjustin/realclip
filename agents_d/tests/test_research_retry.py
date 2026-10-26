"""Explicit retry preparation without any live calls or old artifact writes."""
from copy import deepcopy

import pytest

from realclip_d.contracts import SampleInput
from realclip_d.inputs import research_input
from realclip_d.research_retry import prepare_technical_rerun
from realclip_d.storage import read_json, seal_stage, write_json


KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


def attempted(runs):
    stage = runs / "old/samples/retry_one/research"
    payload = research_input(SampleInput(sample_id="retry_one", compound={"inchikey": KEY},
                                         context={"cell_line": "U2OS", "dose": None, "treatment_duration": None}))
    write_json(stage / "input.json", payload)
    write_json(stage / "binding.json", {"mode": "live"})
    write_json(stage / "requests/001.json", {"old_paid_attempt": True})
    write_json(stage / "cache/raw.json", {"original": "Exact cached source bytes"})
    candidate = {"compound_id": "retry_one", "inchikey": KEY, "original_stage": str(stage),
                 "category": "mixed_language_tool_content"}
    return candidate, stage


def test_retry_preserves_original_input_and_copies_only_cache(tmp_path):
    runs = tmp_path / "runs"
    candidate, old = attempted(runs)
    old_bytes = {str(p.relative_to(old)): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    policy = {"compound_id": "held", "category": "provider_policy_rejection"}
    output = runs / "new"
    plan = prepare_technical_rerun([candidate], [candidate, policy], runs, output)
    assert len(plan["scheduled"]) == 1 and not plan["already_complete_excluded"]
    assert plan["provider_rejected_cases_held"] == ["held"]
    assert read_json(output / "prepared_inputs/retry_one.json") == read_json(old / "input.json")
    assert (output / "samples/retry_one/research/cache/raw.json").read_bytes() == (old / "cache/raw.json").read_bytes()
    assert not list(output.glob("samples/*/research/requests/*.json"))
    assert old_bytes == {str(p.relative_to(old)): p.read_bytes() for p in old.rglob("*") if p.is_file()}


@pytest.mark.parametrize("kind", ["policy", "unlisted", "duplicate", "wrong_identity"])
def test_unapproved_or_policy_cases_rejected_before_preparation(tmp_path, kind):
    runs = tmp_path / "runs"
    candidate, _ = attempted(runs)
    failures = [deepcopy(candidate)]
    candidates = [deepcopy(candidate)]
    if kind == "policy":
        failures[0]["category"] = "provider_policy_rejection"
    elif kind == "unlisted":
        candidates[0]["compound_id"] = "other"
    elif kind == "duplicate":
        candidates *= 2
    else:
        candidates[0]["inchikey"] = "RYYVLZVUVIJVGH-UHFFFAOYSA-N"
    with pytest.raises(ValueError):
        prepare_technical_rerun(candidates, failures, runs, runs / "new")
    assert not (runs / "new").exists()


def test_completed_identity_under_another_id_is_excluded(tmp_path):
    runs = tmp_path / "runs"
    candidate, old = attempted(runs)
    complete = runs / "complete/samples/different_id/research"
    payload = read_json(old / "input.json"); payload["sample_id"] = "different_id"
    for name, data in {"input": payload, "binding": {"mode": "live"}, "result": {},
                       "settings": {}, "sources": []}.items():
        write_json(complete / f"{name}.json", data)
    seal_stage(complete)
    output = runs / "new"
    plan = prepare_technical_rerun([candidate], [candidate], runs, output)
    assert not plan["scheduled"] and len(plan["already_complete_excluded"]) == 1
    assert not (output / "prepared_inputs").exists()
    assert not (output / "samples").exists()
