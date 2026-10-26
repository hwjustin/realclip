"""Report only synthetic saved artifacts; no network, model or credentials."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import pytest

from realclip_d.pilot_report import _href, generate_report


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def pilot(tmp_path):
    run = tmp_path / "pilot"
    entry = {"sample_id": "SYNTHETIC_1", "name": "Synthetic compound <script>alert(1)</script>",
             "selection_reason": "Synthetic report fixture only", "known_moa": ["Synthetic MOA"]}
    write(run / "manifest.json", {"samples": [entry], "settings": {"model": "synthetic"}, "budget_cap_usd": 5})
    stage = run / "samples/SYNTHETIC_1/research"
    write(stage / "input.json", {"sample_id": "SYNTHETIC_1", "compound": {"name": "Synthetic"},
                                 "known_moa": ["Synthetic MOA"], "context": {"dose": None}})
    source = {"source_id": "synthetic_article", "title": "Synthetic paper", "url": "https://example.invalid/paper",
              "cache_ref": "cache/raw.json", "retrieved_at": "2000-01-01T00:00:00Z"}
    write(stage / "cache/raw.json", {"synthetic": True})
    write(stage / "sources.json", [source])
    result = {"sample_id": "SYNTHETIC_1", "identity_summary": "Synthetic identity.",
              "sources": [source], "facts": [{"fact_id": "f1", "statement": "Exact English <unsafe> statement.",
                                              "origin": "literature", "evidence_level": "direct_morphology"}],
              "hypotheses": [{"hypothesis_id": "h1", "change": "Synthetic outline may enlarge.",
                              "basis": "direct_literature", "rationale": "Invented fixture only.",
                              "fact_ids": ["f1"], "applicable_conditions": "Unknown synthetic dose and time.",
                              "where_to_check": ["Synthetic DNA overview"], "uncertainty": "Synthetic.",
                              "disconfirming_observation": "Equal synthetic outlines.",
                              "verification": {"comparison": "Compare synthetic treatment and control.",
                                               "supporting_observation": "Larger synthetic outlines.",
                                               "not_assessable_when": ["Blurry synthetic input."], "cp_checks": []}}],
              "limitations": ["Synthetic; no biological conclusion."],
              "literature_review": {"status": "direct_observations_found"},
              "mechanism_review": {"status": "no_supported_inference"},
              "identity_context": {"status": "verified"}}
    write(stage / "result.json", result)
    write(stage / "complete.json", {"synthetic_seal": True})
    write(stage / "validation.json", {"references_valid": True, "biological_accuracy_validated": False})
    write(stage.parent / "status.json", {"status": "complete", "mode": "synthetic"})
    tool = {"tool": "read_article", "arguments": ["SYNTHETIC_ID", "MED"], "keyword_arguments": {},
            "result": {"status": "ok", "data": {"article": {"id": "SYNTHETIC_ID", "title": "Synthetic paper"},
                        "abstract": "Synthetic abstract.", "fulltext_status": "unavailable", "body_text": None},
                        "sources": [source]}}
    (stage / "tools.jsonl").write_text(json.dumps(tool) + "\n")
    (stage / "calls.jsonl").write_text(json.dumps({"call": 1, "request_id": "synthetic_req", "status": "complete"}) + "\n")
    write(stage / "requests/001.json", {"input": [{"role": "user", "content": "Synthetic input"}]})
    write(stage / "responses/001.json", {"output": [
        {"type": "function_call", "name": "read_article", "call_id": "call_synthetic",
         "arguments": '{"article_id":"SYNTHETIC_ID","source":"MED"}'},
        {"type": "web_search_call", "status": "completed", "action": {"type": "search", "query": "synthetic query"}}],
        "raw_usage": {"input_tokens": 123, "output_tokens": 12}})
    write(run / "budget.json", {"budget_usd": 5, "spent_estimate_usd": .001, "blocked": False,
                               "requests": [{"request_id": "synthetic_req", "cost_estimate_usd": .001, "status": "complete"}]})
    return run, stage, result


def snapshots(root: Path):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file() and path.name not in {"index.html", "REPORT.md", "summary.json"}}


def test_four_stage_report_displays_all_stages_without_old_classification_counts(pilot):
    run, stage, _ = pilot
    example = json.loads((Path(__file__).parents[1] / "examples/research_v3_synthetic_example.json").read_text())
    example["sample_id"] = "SYNTHETIC_1"
    write(stage / "result.json", example)
    before = snapshots(run)
    summary = generate_report(run)
    html = (run / "index.html").read_text()
    assert summary["samples"][0]["hypothesis_count"] == 1
    assert summary["samples"][0]["basis_counts"] == {}
    for stage_name in ("Compound Identification", "Compound Research and Analysis", "Morphology Hypotheses", "Observation Guidance"):
        assert stage_name in html
    assert "No morphological hypotheses." not in html
    assert summary["samples"][0]["result"] == example
    assert snapshots(run) == before


def test_complete_report_preserves_english_receipts_cost_and_raw_artifacts(pilot):
    run, stage, result = pilot
    before = snapshots(run)
    summary = generate_report(run)
    assert snapshots(run) == before
    sample = summary["samples"][0]
    assert sample["result"] == result
    assert sample["completed"] and sample["output_state"] == "completed"
    assert not sample["biological_accuracy_validated"]
    assert sample["basis_counts"] == {"direct_literature": 1, "indirect_literature": 0, "moa_inference": 0}
    assert sample["cost_estimate_usd"] == .001
    receipt = sample["timeline"][0]["events"][0]["receipt"]
    assert receipt["article_title"] == "Synthetic paper"
    assert receipt["fulltext_status"] == "unavailable"
    assert receipt["abstract_available"] and not receipt["body_text_available"]
    assert sample["timeline"][0]["events"][1]["action"]["query"] == "synthetic query"
    assert sample["timeline"][0]["usage"]["input_tokens"] == 123
    html = (run / "index.html").read_text()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "Exact English &lt;unsafe&gt; statement." in html
    assert 'href="samples/SYNTHETIC_1/research/cache/raw.json"' in html
    assert result["hypotheses"][0]["change"] in (run / "REPORT.md").read_text()
    assert json.loads((run / "summary.json").read_text())["samples"][0]["result"] == result
    assert '<html lang="en">' in html
    for filename in ("index.html", "REPORT.md", "summary.json"):
        assert re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", (run / filename).read_text()) is None
    assert summary["rationale_visibility"]["private_internal_reasoning_available"] is False
    assert "not a transcript of private internal reasoning" in html
    assert "not a transcript of private internal reasoning" in (run / "REPORT.md").read_text()


def test_candidate_failure_is_shown_without_repair_or_filtering(pilot):
    run, stage, result = pilot
    (stage / "complete.json").unlink()
    (stage / "result.json").unlink()
    (stage / "validation.json").unlink()
    candidate = deepcopy(result)
    candidate["facts"][0]["source_ids"] = ["fabricated_fixture_source"]
    write(stage / "candidate_result.json", candidate)
    write(stage.parent / "status.json", {"status": "failed", "error": "Synthetic reference validation failure"})
    summary = generate_report(run)
    sample = summary["samples"][0]
    assert not sample["completed"] and sample["output_state"] == "unvalidated_candidate"
    assert sample["result"] == candidate
    assert "Validation incomplete" in (run / "index.html").read_text()
    assert "fabricated_fixture_source" in (run / "REPORT.md").read_text()


def test_failed_parse_recovers_raw_json_candidate_from_last_response(pilot):
    run, stage, result = pilot
    (stage / "result.json").unlink()
    (stage / "complete.json").unlink()
    write(stage / "responses/002.json", {"output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps(result)}]}]})
    sample = generate_report(run)["samples"][0]
    assert sample["result"] == result and sample["output_state"] == "unvalidated_candidate"


def test_absent_output_receipt_and_pending_response_are_explicit(pilot):
    run, stage, result = pilot
    (stage / "result.json").unlink()
    (stage / "complete.json").unlink()
    (stage / "tools.jsonl").unlink()
    write(stage / "requests/002.json", {"input": "Synthetic pending request"})
    summary = generate_report(run)
    sample = summary["samples"][0]
    assert sample["output_state"] == "no_output"
    assert any("saved receipt is missing" in error for error in sample["errors"])
    assert any("No displayable structured output was saved" in error for error in sample["errors"])
    assert sample["timeline"][1]["response_missing"]


def test_report_detects_observer_and_image_cp_boundary_violations(pilot):
    run, stage, result = pilot
    (stage.parent / "observation").mkdir()
    write(stage / "requests/001.json", {"input": [{"type": "input_image", "image_url": "data:image/png;base64,synthetic"}]})
    summary = generate_report(run)
    assert not summary["boundaries"]["research_only"]
    assert summary["samples"][0]["boundary_violations"]


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html,test", "file:///tmp/secret",
                                     "//evil.example/x", "https://user:password@example.invalid", "http://[",
                                     "../outside.json", "%2e%2e/outside.json", "\\outside", "https://example.invalid/\nfoo"])
def test_unsafe_links_are_not_clickable(url):
    assert _href(url) is None


def test_external_and_confined_relative_links_allowed():
    assert _href("https://example.invalid/paper") == "https://example.invalid/paper"
    assert _href("samples/SYNTHETIC_1/research/result.json") == "samples/SYNTHETIC_1/research/result.json"


def test_source_injection_escaped_cache_escape_rejected(pilot):
    run, stage, result = pilot
    write(stage / "sources.json", [{"source_id": "bad", "title": '<img src=x onerror="alert(1)">',
                                   "url": "javascript:alert(1)", "cache_ref": "../../../../outside.json"}])
    generate_report(run)
    html = (run / "index.html").read_text()
    assert 'href="javascript:' not in html and "<img src=x" not in html
    assert "Cache missing or outside this run directory" in html


def test_no_arbitrary_manifest_input_read_and_symlink_is_rejected(pilot, tmp_path):
    run, stage, result = pilot
    secret = tmp_path / "synthetic.env"
    secret.write_text("SYNTHETIC_SECRET_DO_NOT_DISPLAY")
    (stage / "input.json").unlink()
    (stage / "input.json").symlink_to(secret)
    summary = generate_report(run)
    assert any("symlink" in error for error in summary["samples"][0]["errors"])
    assert "SYNTHETIC_SECRET_DO_NOT_DISPLAY" not in (run / "index.html").read_text()


def test_manifest_path_traversal_is_rejected(pilot):
    run, stage, result = pilot
    write(run / "manifest.json", {"samples": [{"sample_id": "../../outside"}]})
    with pytest.raises(ValueError, match="safe path"):
        generate_report(run)


def test_schema_invalid_candidate_stays_visible_and_does_not_crash(pilot):
    run, stage, result = pilot
    (stage / "result.json").unlink()
    (stage / "complete.json").unlink()
    candidate = {"hypotheses": "SYNTHETIC_INVALID_LIST", "facts": None, "extra": "SYNTHETIC_EXTRA_FIELD"}
    write(stage / "candidate_result.json", candidate)
    summary = generate_report(run)
    assert summary["samples"][0]["result"] == candidate
    html = (run / "index.html").read_text()
    assert "SYNTHETIC_INVALID_LIST" in html and "SYNTHETIC_EXTRA_FIELD" in html


def test_report_output_symlink_cannot_overwrite_raw_file(pilot):
    run, stage, result = pilot
    target = stage / "result.json"
    original = target.read_bytes()
    (run / "summary.json").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        generate_report(run)
    assert target.read_bytes() == original


def test_retrieved_fulltext_is_distinct_from_exact_windows_shown_to_model(pilot):
    run, stage, result = pilot
    row = json.loads((stage / "tools.jsonl").read_text())
    raw_body = "SYNTHETIC_FULL_BODY_NOT_SHOWN_IN_FULL. " * 100
    window = {"start": 0, "end": 35, "text": raw_body[:35]}
    row["result"]["data"].update(fulltext_status="accessible", body_text=raw_body)
    row["model_result"] = {"status": "ok", "data": {
        "article": row["result"]["data"]["article"], "abstract": "Synthetic abstract excerpt.",
        "fulltext_status": "accessible", "body_windows": [window],
        "_model_view": {"full_text_retrieved": True, "model_read_entire_article": False,
                        "normalized_body_characters": len(raw_body), "fulltext_characters_shown": 35,
                        "fulltext_content_omitted": True, "additional_omissions": []}}}
    (stage / "tools.jsonl").write_text(json.dumps(row) + "\n")
    summary = generate_report(run)
    receipt = summary["samples"][0]["timeline"][0]["events"][0]["receipt"]
    assert receipt["fulltext_status"] == "accessible" and receipt["body_text_available"]
    exposure = receipt["model_exposure"]
    assert exposure["metadata"]["model_read_entire_article"] is False
    assert exposure["body_windows_shown"] == [window]
    assert exposure["metadata"]["normalized_body_characters"] > exposure["metadata"]["fulltext_characters_shown"]
    assert raw_body not in (run / "index.html").read_text()
    assert "Successful full-text retrieval does not mean the model read the entire paper" in (run / "index.html").read_text()


def test_legacy_receipt_does_not_claim_entire_article_was_shown(pilot):
    run, stage, result = pilot
    sample = generate_report(run)["samples"][0]
    exposure = sample["timeline"][0]["events"][0]["receipt"]["model_exposure"]
    assert not exposure["model_result_recorded"]
    assert exposure["metadata"] is None
    assert "does not establish" in exposure["note"]


def test_continuation_turns_cost_and_initial_failure_are_included(pilot):
    run, stage, result = pilot
    continuation = stage / "continuation_01"
    write(stage.parent / "initial_status.json", {"status": "failed", "error_type": "SyntheticBoundError"})
    write(stage.parent / "status.json", {"status": "completed_from_saved_evidence", "initial_error": "SyntheticBoundError"})
    write(continuation / "input.json", {"saved_evidence": "Synthetic already-seen excerpts only."})
    write(continuation / "binding.json", {"tool_access": False, "reason": "synthetic continuation fixture"})
    write(continuation / "requests/001.json", {"input": "Synthetic finalization request"})
    write(continuation / "responses/001.json", {"output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps(result)}]}], "raw_usage": {"input_tokens": 23, "output_tokens": 45}})
    (continuation / "calls.jsonl").write_text(json.dumps({"call": 1, "request_id": "synthetic_continuation", "status": "complete"}) + "\n")
    ledger = json.loads((run / "budget.json").read_text())
    ledger["requests"].append({"request_id": "synthetic_continuation", "cost_estimate_usd": .002, "status": "complete"})
    ledger["spent_estimate_usd"] = .003
    write(run / "budget.json", ledger)
    before = snapshots(run)
    summary = generate_report(run)
    sample = summary["samples"][0]
    assert snapshots(run) == before
    assert sample["model_requests"] == 2
    assert [row["turn"] for row in sample["timeline"]] == ["001", "continuation_01/001"]
    assert sample["timeline"][1]["request"].endswith("continuation_01/requests/001.json")
    assert sample["cost_estimate_usd"] == .003
    assert sample["initial_status"]["error_type"] == "SyntheticBoundError"
    assert len(sample["continuations"]) == 1
    assert "continuation_01/001" in (run / "index.html").read_text()
    assert "SyntheticBoundError" in (run / "REPORT.md").read_text()


def test_optional_audit_preserved_separately_with_escaping(pilot):
    run, stage, result = pilot
    audit = {"scope": "Synthetic excerpt audit only", "samples": [
        {"sample_id": "SYNTHETIC_1", "findings": ["<script>synthetic injection</script>"]}]}
    write(run / "audit.json", audit)
    summary = generate_report(run)
    assert summary["audit"] == audit
    assert summary["samples"][0]["result"] == result
    html = (run / "index.html").read_text()
    assert "Independent offline evidence review" in html
    assert "<script>synthetic injection</script>" not in html
    assert 'href="audit.json"' in html
    assert not summary["samples"][0]["biological_accuracy_validated"]


def test_english_presentation_and_review_preserve_original_records(pilot):
    run, stage, result = pilot
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["samples"][0]["selection_reason"] = "\u5386\u53f2\u9009\u62e9\u539f\u56e0"
    write(run / "manifest.json", manifest)
    write(run / "audit.json", {"findings": ["\u5386\u53f2\u5ba1\u9605\u8bb0\u5f55"]})
    english_audit = {"findings": ["Historical review record"], "preserves_original_findings": True}
    write(run / "audit.en.json", english_audit)
    write(run / "presentation.en.json", {
        "language": "en",
        "selection_reasons": {"SYNTHETIC_1": "English selection reason", "UNKNOWN_SAMPLE": "Ignored reason"},
        "result": {"hypotheses": []},
    })
    before = snapshots(run)
    summary = generate_report(run)
    assert snapshots(run) == before
    assert summary["samples"][0]["result"] == result
    assert summary["samples"][0]["selection_reason"] == "English selection reason"
    assert summary["audit"] == english_audit
    assert summary["audit_filename"] == "audit.en.json"
    assert summary["presentation_provenance"] == {
        "file": "presentation.en.json", "selection_reason_overrides": ["SYNTHETIC_1"],
        "original_manifest_unchanged": True,
    }
    for filename in ("index.html", "REPORT.md", "summary.json"):
        text = (run / filename).read_text()
        assert re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text) is None
        assert "Ignored reason" not in text
    assert 'href="audit.en.json"' in (run / "index.html").read_text()
    assert "[audit.en.json](audit.en.json)" in (run / "REPORT.md").read_text()


def test_invalid_english_presentation_is_explicitly_rejected(pilot):
    run, stage, result = pilot
    write(run / "presentation.en.json", {"language": "en", "selection_reasons": []})
    with pytest.raises(ValueError, match="presentation.en.json"):
        generate_report(run)
