"""Model-view bounds, provenance, and honest excerpt access; no API calls."""
import json

from realclip_d.contracts import Source, ToolResult
from realclip_d.tool_views import DATA_CHARACTER_LIMIT, build_model_view


def receipt(data):
    return ToolResult(status="ok", data=data, sources=[Source(
        source_id="complete_source_identifier", url="https://example.invalid/evidence",
        retrieved_at="2026-10-20T00:00:00Z", cache_ref="cache/full-response.json")])


def test_article_overview_does_not_select_body_by_keywords_or_expose_raw_xml():
    original = receipt({"article": {"id": "1", "title": "Actual article title", "abstractText": "duplicate"},
                        "abstract": "Abstract. " * 3000, "fulltext_xml": "SECRET_RAW_XML_SENTINEL",
                        "body_text": "Nocodazole produced rounded cells. Important mechanism elsewhere.",
                        "fulltext_status": "accessible"})
    before = original.model_dump()
    view = build_model_view("read_article", original)
    serialized = view.model_dump_json()
    assert "SECRET_RAW_XML_SENTINEL" not in serialized
    assert "rounded cells" not in serialized
    assert "body_text" not in view.data and "body_windows" not in view.data
    assert "abstractText" not in view.data["article"]
    reading = view.data["reading"]
    assert reading["mode"] == "overview" and reading["has_more"]
    assert reading["next_request"]["offset"] == reading["end"]
    assert view.data["_model_view"]["full_text_retrieved"]
    assert view.data["_model_view"]["full_text_complete"] is False
    assert original.model_dump() == before
    assert view.sources == original.sources


def test_chembl_preserves_all_scientific_records_and_exact_parent_scope():
    original = receipt({"exact_molecule_id": "CHEMBL1", "parent_molecule_id": "CHEMBL2",
                        "activity_scope": "exact_molecule", "mechanism_scope": "parent_aggregated",
                        "molecule": {"molecule_structures": {"molfile": "MOLFILE_SENTINEL",
                                                               "standard_inchi_key": "UNCHANGED"}},
                        **{key: [{"id": index, "description": "x" * 1000} for index in range(25)]
                           for key in ["activities", "mechanisms", "assays", "targets", "documents"]},
                        "truncated": {"activities": True}})
    view = build_model_view("get_chembl_evidence", original)
    assert "MOLFILE_SENTINEL" not in view.model_dump_json()
    for category in ("activities", "mechanisms", "assays", "targets", "documents"):
        assert view.data[category] == original.data[category]
    assert view.data["activity_scope"] == "exact_molecule"
    assert view.data["mechanism_scope"] == "parent_aggregated"
    assert view.data["truncated"]["activities"] is True
    assert view.data["_model_view"]["row_counts"]["assays"]["omitted_rows"] == 0
    assert view.sources[0].model_dump() == original.sources[0].model_dump()


def test_pubchem_large_tree_is_valid_bounded_json_with_explicit_omissions():
    original = receipt({"cid": 123, "record": {"Section": [
        {"TOCHeading": f"section {index}", "Information": [
            {"Value": {"StringWithMarkup": [{"String": "very large annotation " * 5000}]}}
            for _ in range(20)]} for index in range(30)]}})
    view = build_model_view("get_pubchem_annotations", original)
    serialized = json.dumps(view.data, ensure_ascii=False)
    assert len(serialized) <= DATA_CHARACTER_LIMIT
    assert json.loads(serialized) == view.data
    assert view.data["_model_view"]["additional_omissions"]
    assert view.data["_model_view"]["complete_tool_receipt_retained"]
    assert view.sources == original.sources


def test_search_keeps_all_candidates_and_does_not_become_a_read_receipt():
    original = receipt({"results": [{"id": str(index), "title": "Title"} for index in range(100)],
                        "hit_count": 999, "next_cursor": "cursor"})
    view = build_model_view("search_literature", original)
    assert len(view.data["results"]) == len(view.data["candidates"]) == 100
    assert view.data["hit_count"] == 999
    assert view.data["next_cursor"] == "cursor"
    assert view.data["_model_view"]["search_results_are_not_article_read_receipts"]
    assert view.data["_model_view"]["omitted_rows"] == 0


def test_unhandled_tools_unchanged_and_sources_never_clipped():
    original = receipt({"source_id": "id" * 3000, "source_ids": ["abc" * 3000]})
    assert build_model_view("resolve_compound", original) is original
    view = build_model_view("get_pubchem_annotations", original)
    assert view.data["source_id"] == original.data["source_id"]
    assert view.data["source_ids"] == original.data["source_ids"]
    assert view.sources == original.sources


def test_missing_fulltext_remains_unavailable():
    original = receipt({"article": {"title": "Abstract only"}, "abstract": "Short abstract.",
                        "fulltext_status": "unavailable", "fulltext_error": "HTTP 404", "body_text": None})
    view = build_model_view("read_article", original)
    assert view.data["reading"]["scope"] == "abstract"
    assert view.data["sections"] == []
    assert view.data["fulltext_status"] == "unavailable"
    assert view.data["fulltext_error"] == "HTTP 404"
    assert view.data["_model_view"]["full_text_retrieved"] is False
