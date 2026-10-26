"""Regression coverage for the twelve authorized technical-failure reruns."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from realclip_d.agent_runtime import _ToolRecorder
from realclip_d.contracts import ToolResult
from realclip_d.language import validate_model_request
from realclip_d.tool_language import UNAVAILABLE
from realclip_d.tool_views import build_model_view, research_view
from realclip_d.tools import CachedHTTP, EuropePMCClient
from realclip_d.tools.article_reading import parse_article


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "runs/research_failed30_audit_20261021"


def test_only_structure_indexes_deferred_all_scientific_records_and_components_retained():
    component = {"accession": "P12345", "component_type": "PROTEIN", "organism": "Homo sapiens",
                 "target_component_synonyms": [{"component_synonym": "Functional name"}],
                 "target_component_xrefs": [
                     {"xref_src_db": db, "xref_id": f"id_{i}"}
                     for db in ("PDB", "PDBe") for i in range(1000)] + [
                     {"xref_src_db": db, "xref_id": db + "_function"} for db in ("GO", "Reactome", "UnknownDB")]}
    data = {"activity_scope": "exact_molecule", "mechanism_scope": "parent_aggregated",
            "activities": [{"standard_value": "4.2", "standard_units": "nM", "standard_relation": "<",
                            "assay_description": "U2OS; 24 hours; 37 degrees C"}],
            "mechanisms": [{"mechanism_of_action": "Enzyme inhibition", "direct_interaction": True}],
            "assays": [{"assay_chembl_id": "CHEMBL1", "description": "Complete methods " * 3000}],
            "documents": [{"pubmed_id": "123"}],
            "targets": [{"pref_name": "Multicomponent target", "target_components": [deepcopy(component) for _ in range(82)]}]}
    raw = ToolResult(status="ok", data=data)
    before = deepcopy(raw.model_dump())
    view = build_model_view("get_chembl_evidence", raw)
    for key in ("activities", "mechanisms", "assays", "documents", "activity_scope", "mechanism_scope"):
        assert view.data[key] == data[key]
    assert len(view.data["targets"]) == 1
    assert len(view.data["targets"][0]["target_components"]) == 82
    for result in view.data["targets"][0]["target_components"]:
        assert {k: v for k, v in result.items() if k not in {
            "target_component_xrefs", "structure_xrefs_deferred"}} == {
            k: v for k, v in component.items() if k != "target_component_xrefs"}
        assert result["target_component_xrefs"] == component["target_component_xrefs"][-3:]
        assert result["structure_xrefs_deferred"] == {"PDB": 1000, "PDBe": 1000}
    assert view.data["_model_view"]["structure_index_view"]["deferred_rows_by_database"] == {
        "PDB": 82000, "PDBe": 82000}
    assert raw.model_dump() == before


def test_bilingual_search_titles_use_source_english_and_mixed_abstract_is_unavailable():
    raw = ToolResult(status="ok", data={"results": [
        {"id": "1", "title": "中文标题 Effects and mechanism of compound in mice", "abstractText": "English abstract."},
        {"id": "2", "title": "English paper", "abstractText": "Mechanism changes C. elegans的壽命有多種方式。"}]})
    view = research_view("search_literature", raw)
    assert [r["id"] for r in view.data["candidates"]] == ["1", "2"]
    assert view.data["candidates"][0]["title"] == "Effects and mechanism of compound in mice"
    assert view.data["results"][0]["abstractText"] == "English abstract."
    assert view.data["results"][1]["abstractText"] is None
    assert not view.data["results"][1]["abstract_available"]
    assert view.data["results"][1]["abstract_language_status"] == "unavailable_in_English"
    assert "的壽命" in raw.data["results"][1]["abstractText"]
    assert not view.data["_language_view"]["translation_performed"]
    validate_model_request({"input": view.model_dump_json()})


def test_large_shared_functional_annotations_reconstruct_every_value_and_component_order():
    rows = [{"xref_src_db": "Reactome", "xref_id": f"R-HSA-{i}", "xref_name": "Complete pathway description " * 10}
            for i in range(120)]
    rows.append({"xref_src_db": "FutureDB", "new_field": None, "nested": {"value": "Unrecognized annotation kept"}})
    components = [{"accession": f"P{i}", "target_component_xrefs": deepcopy(rows[i:] + rows[:i] + rows[:2])}
                  for i in range(10)]
    raw = ToolResult(status="ok", data={"targets": [{"target_components": components}]})
    before = deepcopy(raw.model_dump())
    view = build_model_view("get_chembl_evidence", raw)
    assert len(view.model_dump_json()) < len(raw.model_dump_json()) * .3
    table = view.data["target_annotation_records"]
    for original, shown in zip(components, view.data["targets"][0]["target_components"], strict=True):
        assert original["accession"] == shown["accession"]
        assert [table[i] for i in shown["target_component_xrefs"]["record_indexes"]] == original["target_component_xrefs"]
    assert view.data["_model_view"]["shared_target_annotations"]["all_functional_annotations_retained"]
    assert raw.model_dump() == before


XML = '''<article><front><article-meta><abstract><p>Source English abstract.</p></abstract>
<trans-abstract xml:lang="zh"><p>这是已提供英文的翻译摘要。</p></trans-abstract></article-meta></front>
<body><sec><title>Ifupinostat (贝特琳; Betlin): Key Points</title>
<p>Complete English mechanism and conditions.</p><sec><title>Late result</title>
<p>Later English observation.</p></sec></sec></body></article>'''


def article_client(tmp_path, xml=XML):
    calls = []
    def respond(request):
        calls.append(str(request.url))
        if request.url.path.endswith("search"):
            return httpx.Response(200, json={"hitCount": 1, "resultList": {"result": [{
                "id": "1", "source": "MED", "pmcid": "PMC1", "title": "English title",
                "abstractText": "Source English abstract."}]}})
        return httpx.Response(200, text=xml)
    return EuropePMCClient(CachedHTTP(tmp_path, client=httpx.Client(transport=httpx.MockTransport(respond)))), calls


@pytest.mark.parametrize("mode", ["overview", "sections", "full_text"])
def test_english_article_pagination_has_stable_ids_exact_offsets_and_honest_coverage(tmp_path, mode):
    client, calls = article_client(tmp_path)
    outline = build_model_view("read_article", client.read_article("1")).data["sections"]
    assert [s["id"] for s in outline] == [s["id"] for s in parse_article(XML)["sections"]]
    assert "Ifupinostat (Betlin): Key Points" in [s["title"] for s in outline]
    ids = [outline[-2]["id"]] if mode == "sections" else None
    args = dict(article_id="1", mode=mode, section_ids=ids, max_chars=19)
    texts, end, checksums = [], 0, set()
    while args:
        raw = client.read_article(**args)
        assert raw.status == "ok"
        assert "翻译摘要" in raw.data["document"]["text"]
        view = research_view("read_article", raw)
        validate_model_request({"input": view.model_dump_json()})
        page = view.data["reading"]
        assert page["offset"] == end
        assert len(page["text"]) == page["end"] - page["offset"] <= 19
        assert not page["full_text_complete_in_this_response"]
        texts.append(page["text"])
        end = page["end"]
        checksums.add(page["content_sha256"])
        args = page["next_request"]
    text = "".join(texts)
    assert len(checksums) == 1 and hashlib.sha256(text.encode()).hexdigest() in checksums
    assert "English" in text
    if mode != "overview":
        assert "Later English observation." in text
    if mode == "full_text":
        assert "Translated abstract omitted" in text
    assert len(calls) == 2


def test_missing_english_counterpart_is_not_silently_stripped_or_called_full_read(tmp_path):
    xml = '<article><body><sec><title>Results</title><p>English mechanism.</p><p>中文重要机制。</p></sec></body></article>'
    client, _ = article_client(tmp_path, xml)
    recorder = _ToolRecorder(tmp_path / "stage", 10, research=True)
    output = json.loads(asyncio.run(recorder.call("read_article", client.read_article, "1", mode="full_text")))
    view = output["data"]
    assert UNAVAILABLE in view["reading"]["text"]
    assert "English mechanism." in view["reading"]["text"]
    assert not view["reading"]["full_text_complete_in_this_response"]
    assert not view["_model_view"]["full_text_complete"]
    assert view["_model_view"]["English_presentation_complete"]
    saved = json.loads((tmp_path / "stage/tools.jsonl").read_text())
    assert "中文重要机制" in saved["result"]["data"]["fulltext_xml"]
    assert saved["model_result"] == output
    validate_model_request({"input": output})


def test_translated_only_abstract_does_not_invent_an_english_counterpart(tmp_path):
    xml = '<article><front><article-meta><trans-abstract xml:lang="zh"><p>没有英文版本。</p></trans-abstract></article-meta></front><body><p>English body.</p></body></article>'
    projected = parse_article(xml, english_view=True)
    assert UNAVAILABLE in projected["abstract_text"]
    assert "omitted_English_abstract_available" not in json.dumps(projected)


@pytest.mark.parametrize("compound_id", [
    "JCP2022_002490", "JCP2022_019949", "JCP2022_020544", "JCP2022_023282",
    "JCP2022_028914", "JCP2022_042944", "JCP2022_049322", "JCP2022_055874",
    "JCP2022_065189", "JCP2022_073458", "JCP2022_087170", "JCP2022_111882"])
def test_saved_failed_case_tool_views_are_english_and_originals_unchanged(compound_id):
    if not AUDIT.is_dir():
        pytest.skip("Optional saved campaign receipts are absent")
    candidates = json.loads((AUDIT / "retry_candidates.json").read_text())["candidates"]
    case = next(c for c in candidates if c["compound_id"] == compound_id)
    tools = Path(case["original_stage"]) / "tools.jsonl"
    for line in tools.read_text().splitlines():
        row = json.loads(line)
        raw = ToolResult.model_validate(row["result"])
        before = deepcopy(raw.model_dump())
        view = research_view(row["tool"], raw)
        validate_model_request({"input": view.model_dump_json()})
        assert raw.model_dump() == before
        if row["tool"] == "get_chembl_evidence":
            for key in ("activities", "mechanisms", "assays", "documents"):
                assert view.data.get(key) == raw.data.get(key)
            assert [len(t.get("target_components", [])) for t in view.data["targets"]] == [
                len(t.get("target_components", [])) for t in raw.data["targets"]]


@pytest.mark.parametrize("compound_id", ["JCP2022_019949", "JCP2022_023282", "JCP2022_065189", "JCP2022_073458"])
def test_failed_size_guard_request_reconstructed_offline_fits_original_bound(compound_id):
    if not AUDIT.is_dir():
        pytest.skip("Optional saved campaign receipts are absent")
    case = next(c for c in json.loads((AUDIT / "retry_candidates.json").read_text())["candidates"]
                if c["compound_id"] == compound_id)
    stage = Path(case["original_stage"])
    last = sorted((stage / "requests").glob("*.json"))[-1]
    request = json.loads(last.read_text())
    receipts = [json.loads(line) for line in (stage / "tools.jsonl").read_text().splitlines()]
    canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)
    by_original = {canonical(r["model_result"]): r for r in receipts}
    used = set()
    for item in request["input"]:
        if item.get("type") != "function_call_output":
            continue
        key = canonical(json.loads(item["output"]))
        assert key in by_original
        used.add(key)
        row = by_original[key]
        item["output"] = research_view(row["tool"], ToolResult.model_validate(row["result"])).model_dump_json()
    # The failed next request was never saved/sent. Add the final response and
    # pending tool outputs to the last saved conversation without summarizing it.
    request["input"].extend(json.loads((stage / "responses" / last.name).read_text())["output"])
    for key, row in by_original.items():
        if key not in used:
            request["input"].append({"type": "function_call_output", "call_id": "offline-reconstruction",
                "output": research_view(row["tool"], ToolResult.model_validate(row["result"])).model_dump_json()})
    validate_model_request(request)
    bound = len(json.dumps(request, ensure_ascii=False).encode()) + 16384
    assert bound < 1_000_000, (compound_id, bound)


def test_new_ataluren_failed_request_fits_with_lossless_functional_annotation_sharing():
    stage = ROOT / "runs/research_retry_12_20261021_090500/samples/JCP2022_065189/research"
    if not stage.is_dir():
        pytest.skip("Optional saved Ataluren rerun receipts are absent")
    receipts = [json.loads(line) for line in (stage / "tools.jsonl").read_text().splitlines()]
    last = sorted((stage / "requests").glob("*.json"))[-1]
    request = json.loads(last.read_text())
    canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)
    originals = {canonical(r["model_result"]): r for r in receipts}
    used = set()
    for item in request["input"]:
        if item.get("type") == "function_call_output":
            key = canonical(json.loads(item["output"])); used.add(key)
            row = originals[key]
            item["output"] = research_view(row["tool"], ToolResult.model_validate(row["result"])).model_dump_json()
    request["input"].extend(json.loads((stage / "responses" / last.name).read_text())["output"])
    for key, row in originals.items():
        if key not in used:
            request["input"].append({"type": "function_call_output", "call_id": "offline-reconstruction",
                "output": research_view(row["tool"], ToolResult.model_validate(row["result"])).model_dump_json()})
    assert len(json.dumps(request, ensure_ascii=False).encode()) + 16384 < 1_000_000
    chembl = next(r for r in receipts if r["tool"] == "get_chembl_evidence")
    view = research_view("get_chembl_evidence", ToolResult.model_validate(chembl["result"]))
    table = view.data["target_annotation_records"]
    for target, original in zip(view.data["targets"], chembl["result"]["data"]["targets"], strict=True):
        for component, raw in zip(target["target_components"], original["target_components"], strict=True):
            refs = component["target_component_xrefs"]
            reconstructed = [table[i] for i in refs["record_indexes"]] if isinstance(refs, dict) else refs
            assert reconstructed == [r for r in raw["target_component_xrefs"] if r["xref_src_db"] not in {"PDB", "PDBe"}]
