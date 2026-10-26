"""Synthesis evidence packets use saved views only; no model/network/env calls."""
from copy import deepcopy
import json

import pytest

from realclip_d.contracts import Source, ToolResult
from realclip_d.pilot_evidence import PACKET_MAX_BYTES, build_evidence_packet


def source(identifier="synthetic_source"):
    return Source(source_id=identifier, url=f"https://example.invalid/{identifier}",
                  retrieved_at="2026-10-20T00:00:00Z", cache_ref="cache/record.json").model_dump(mode="json")


def receipt(tool, data, *, original=None, sources=None, arguments=None):
    sources = sources if sources is not None else [source()]
    result = ToolResult(status="ok", data=data, sources=sources).model_dump(mode="json")
    raw = deepcopy(result)
    if original is not None:
        raw["data"] = original
    return {"tool": tool, "arguments": arguments if arguments is not None else [],
            "keyword_arguments": {}, "result": raw, "model_result": result}


def save(tmp_path, rows):
    (tmp_path / "tools.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_exact_seen_article_excerpts_sources_identity_and_arguments_preserved(tmp_path, monkeypatch):
    identity = {"compound": {"InChIKey": "SYNTHETIC", "CID": 123}, "synonyms": ["Synthetic"], "identity_verified": True}
    article = {"article": {"title": "Synthetic title"}, "abstract": "SEEN_ABSTRACT",
               "fulltext_status": "accessible", "body_windows": [{"start": 99, "end": 110, "text": "SEEN_WINDOW"}],
               "_model_view": {"model_read_entire_article": False, "fulltext_content_omitted": True}}
    rows = [receipt("resolve_compound", identity),
            receipt("read_article", article, original={"body_text": "UNSEEN_RAW_BODY_SENTINEL", "fulltext_xml": "UNSEEN_XML"},
                    arguments=["SYNTHETIC_ARTICLE", "MED"]),
            receipt("list_sources", {}, sources=[source()])]
    save(tmp_path, rows)
    monkeypatch.setenv("OPENAI_API_KEY", "CREDENTIAL_MUST_NOT_APPEAR")
    packet = build_evidence_packet(tmp_path)
    assert packet["receipts"][0]["data"] == identity
    assert packet["receipts"][1]["data"] == article
    assert packet["receipts"][1]["receipt_index"] == 2
    assert packet["receipts"][1]["arguments"] == ["SYNTHETIC_ARTICLE", "MED"]
    assert packet["receipts"][1]["source_ids"] == ["synthetic_source"]
    assert packet["sources"] == [source()]
    serialized = json.dumps(packet, ensure_ascii=False)
    assert "UNSEEN" not in serialized
    assert "CREDENTIAL_MUST_NOT_APPEAR" not in serialized
    assert len(serialized.encode()) <= PACKET_MAX_BYTES


def test_search_retains_queries_titles_counts_but_never_unread_abstracts(tmp_path):
    data = {"query": "exact synthetic query", "hit_count": 77,
            "results": [{"id": "1", "title": "Paper title", "abstractText": "NOT_A_READ_ABSTRACT", "source": "MED"}]}
    save(tmp_path, [receipt("search_literature", data, arguments=["exact synthetic query", 10])])
    packet = build_evidence_packet(tmp_path)
    item = packet["receipts"][0]
    assert item["evidence_kind"] == "search_discovery_only"
    assert item["data"]["query"] == "exact synthetic query"
    assert item["data"]["hit_count"] == 77
    assert item["data"]["articles"] == [{"id": "1", "title": "Paper title", "source": "MED"}]
    assert "NOT_A_READ_ABSTRACT" not in json.dumps(packet)


def test_chembl_preserves_mechanisms_activities_and_identity_scope(tmp_path):
    data = {"molecule": {"molecule_chembl_id": "CHEMBL1", "molecule_structures": {
                "standard_inchi_key": "KEY", "molfile": "STRUCTURE_SENTINEL"}},
            "exact_molecule_id": "CHEMBL1", "parent_molecule_id": "CHEMBL2",
            "mechanism_scope": "parent_aggregated", "activity_scope": "exact_molecule",
            "mechanisms": [{"action_type": "SYNTHETIC_INHIBITOR"}],
            "activities": [{"activity": "ACTIVITY_SENTINEL"}]}
    save(tmp_path, [receipt("get_chembl_evidence", data)])
    packet = build_evidence_packet(tmp_path)
    evidence = packet["receipts"][0]["data"]
    assert evidence["mechanisms"] == data["mechanisms"]
    assert evidence["mechanism_scope"] == "parent_aggregated"
    assert evidence["activity_scope"] == "exact_molecule"
    assert evidence["molecule"]["molecule_structures"]["standard_inchi_key"] == "KEY"
    assert evidence["activities"] == data["activities"]
    assert "ACTIVITY_SENTINEL" in json.dumps(packet)
    assert "STRUCTURE_SENTINEL" not in json.dumps(packet)


@pytest.mark.parametrize("contents", ["not json\n", "{}\n", "[]\n"])
def test_malformed_receipt_fails_explicitly(tmp_path, contents):
    (tmp_path / "tools.jsonl").write_text(contents)
    with pytest.raises(ValueError, match="Malformed research receipt"):
        build_evidence_packet(tmp_path)


def test_missing_model_view_is_not_reconstructed_from_raw_body(tmp_path):
    row = receipt("read_article", {"body_text": "UNSEEN_RAW"})
    del row["model_result"]
    save(tmp_path, [row])
    with pytest.raises(ValueError, match="Malformed research receipt"):
        build_evidence_packet(tmp_path)


def test_tampered_model_source_is_rejected(tmp_path):
    row = receipt("read_article", {"abstract": "Synthetic"})
    row["model_result"]["sources"][0]["url"] = "https://example.invalid/tampered"
    save(tmp_path, [row])
    with pytest.raises(ValueError, match="sources differ"):
        build_evidence_packet(tmp_path)


def test_cannot_silently_shorten_seen_article_to_fit(tmp_path):
    article = {"abstract": "合成" * 20000, "body_windows": [{"text": "EXACT_WINDOW", "start": 0, "end": 12}]}
    save(tmp_path, [receipt("read_article", article)])
    with pytest.raises(ValueError, match="exceeds 100000"):
        build_evidence_packet(tmp_path)


def test_large_optional_search_context_compacts_without_touching_article(tmp_path):
    article = {"abstract": "EXACT_ABSTRACT", "body_windows": [{"text": "EXACT_WINDOW", "start": 0, "end": 12}]}
    searches = [{"id": str(i), "title": "合成" * 10000} for i in range(5)]
    save(tmp_path, [receipt("read_article", article),
                    receipt("search_literature", {"query": "synthetic", "hit_count": 5, "results": searches})])
    packet = build_evidence_packet(tmp_path)
    assert packet["receipts"][0]["data"] == article
    assert packet["receipts"][1]["data"]["omission_note"]
    assert len(json.dumps(packet, ensure_ascii=False).encode()) <= PACKET_MAX_BYTES


def test_missing_or_empty_receipts_fail_without_external_reads(tmp_path):
    with pytest.raises(ValueError, match="required"):
        build_evidence_packet(tmp_path)
    (tmp_path / "tools.jsonl").write_text("\n")
    with pytest.raises(ValueError, match="no receipts"):
        build_evidence_packet(tmp_path)


def test_chembl_packet_preserves_all_scientific_categories_units_and_linked_records(tmp_path):
    data = {"molecule": {"molecule_properties": {"alogp": "3.1"}},
            "mechanism_scope": "parent_aggregated", "activity_scope": "exact_molecule",
            "mechanisms": [{"mechanism_of_action": "Synthetic inhibition"}],
            "activities": [{"standard_value": "7", "standard_units": "nM", "assay_chembl_id": "A1"}],
            "assays": [{"id": "A1", "description": "Synthetic cell-free assay"}],
            "targets": [{"target_components": [{"accession": "SYNTHETIC"}]}],
            "documents": [{"doi": "synthetic-doi"}], "truncated": {"activities": True}}
    save(tmp_path, [receipt("get_chembl_evidence", data)])
    evidence = build_evidence_packet(tmp_path)["receipts"][0]["data"]
    for key, value in data.items():
        assert evidence[key] == value


def test_oversized_mechanism_evidence_is_not_silently_dropped_to_fit(tmp_path):
    save(tmp_path, [receipt("get_chembl_evidence", {"activities": [{"assay_description": "x" * 110000}]})])
    with pytest.raises(ValueError, match="exceeds 100000"):
        build_evidence_packet(tmp_path)


def test_new_workspace_tool_receipts_remain_readable_without_unseen_evidence(tmp_path):
    rows = [receipt("record_research_notes", {"saved_notes": [{"statement": "Already shown finding"}]}),
            receipt("read_research_memory", {"text": "Already shown local page"}),
            receipt("stop_article_reading", {"reading_task": {"status": "stopped", "stop_reason": "Irrelevant"}})]
    save(tmp_path, rows)
    packet = build_evidence_packet(tmp_path)
    assert [item["data"] for item in packet["receipts"]] == [row["model_result"]["data"] for row in rows]
