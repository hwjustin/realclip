"""Lossless candidate discovery and bounded automatic article expansion; offline."""
from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest

from realclip_d.contracts import ToolResult
from realclip_d.tool_views import build_model_view, DATA_CHARACTER_LIMIT
from realclip_d.tools import CachedHTTP, EuropePMCClient
from realclip_d.tools.article_reading import AUTO_READ_MAX_CHARS, parse_article, reading_page


def test_large_metadata_cannot_hide_later_mechanism_or_morphology_papers():
    rows = [{"id": str(i), "title": f"Complete candidate title {i}", "abstractText": "Entire abstract. " * 50,
             "authorList": {"author": [{"affiliation": "Irrelevant metadata " * 500}] * 20}}
            for i in range(20)]
    rows[5]["title"] = "Compound-specific microtubule imaging study"
    rows[6]["title"] = "Direct measurement of nuclear size"
    raw = ToolResult(status="ok", data={"query": "compound", "limit": 20, "results": rows, "hit_count": 100})
    before = deepcopy(raw.model_dump())
    view = build_model_view("search_literature", raw).data
    assert [r["title"] for r in view["candidates"]] == [r["title"] for r in rows]
    assert [r["abstractText"] for r in view["results"]] == [r["abstractText"] for r in rows]
    assert "authorList" not in json.dumps(view)
    assert view["next_request"] is None and view["_model_view"]["omitted_rows"] == 0
    assert raw.model_dump() == before


def test_long_abstracts_page_without_losing_candidates_or_scientific_text():
    rows = [{"id": str(i), "title": f"Study {i}", "abstractText": str(i) + " Mechanism. " * 1000}
            for i in range(20)]
    offset, actual = 0, []
    while True:
        raw = ToolResult(status="ok", data={"query": "q", "limit": 20, "cursor": "page-1",
                         "next_cursor": "page-2", "results": rows, "result_offset": offset})
        view = build_model_view("search_literature", raw).data
        assert len(view["candidates"]) == 20
        assert len(json.dumps(view, ensure_ascii=False)) <= DATA_CHARACTER_LIMIT
        actual.extend(view["results"])
        assert view["next_page_request"] == {"query": "q", "limit": 20, "cursor": "page-2", "result_offset": 0}
        if view["next_request"] is None:
            break
        assert view["next_request"]["result_offset"] > offset
        offset = view["next_request"]["result_offset"]
    assert [r["abstractText"] for r in actual] == [r["abstractText"] for r in rows]


def test_single_oversized_abstract_has_explicit_overview_access():
    raw = ToolResult(status="ok", data={"results": [{"id": "1", "title": "Intact title", "abstractText": "x" * 100000}]})
    view = build_model_view("search_literature", raw).data
    assert view["candidates"][0]["title"] == "Intact title"
    assert view["results"][0]["abstract_in_overview"]
    assert "abstractText" not in view["results"][0]
    assert view["next_request"] is None


def test_search_cursor_and_abstract_offset_use_cached_official_page(tmp_path):
    calls = []
    def respond(request):
        calls.append(request)
        cursor = request.url.params.get("cursorMark")
        return httpx.Response(200, json={"hitCount": 6, "nextCursorMark": "second" if cursor is None else cursor,
            "resultList": {"result": [{"id": str(i), "title": "Study"} for i in range(3)]}})
    reader = EuropePMCClient(CachedHTTP(tmp_path, client=httpx.Client(transport=httpx.MockTransport(respond))))
    assert reader.search("q", 3).data["next_cursor"] == "second"
    assert reader.search("q", 3, result_offset=2).data["result_offset"] == 2
    assert len(calls) == 1
    assert reader.search("q", 3, cursor="second").data["next_cursor"] is None
    assert calls[-1].url.params["cursorMark"] == "second"
    assert reader.search("q", 3, result_offset=4).status == "error"
    assert reader.search("q", 3, result_offset=-1).status == "error"


@pytest.mark.parametrize("mode", ["sections", "full_text"])
def test_automatic_read_returns_late_mechanism_exactly_once_without_more_http(tmp_path, mode):
    xml = '<article><body><sec><title>Results</title><p>' + 'Earlier result. ' * 2000 + '</p><p>LATE_MECHANISM_SENTINEL.</p></sec></body></article>'
    calls = []
    def respond(request):
        calls.append(request)
        if request.url.path.endswith("search"):
            return httpx.Response(200, json={"hitCount": 1, "resultList": {"result": [{"id": "1", "pmcid": "PMC1"}]}})
        return httpx.Response(200, text=xml)
    reader = EuropePMCClient(CachedHTTP(tmp_path, client=httpx.Client(transport=httpx.MockTransport(respond))))
    document = parse_article(xml)
    ids = [next(s["id"] for s in document["sections"] if s["title"] == "Results")] if mode == "sections" else None
    result = reader.read_article("1", mode=mode, section_ids=ids, max_chars=12000, auto_read=True)
    page = result.data["reading"]
    expected = reading_page(result.data, "1", "MED", mode, ids, 0, 96000)["text"]
    assert len(page["text"]) > 24000 and page["text"] == expected
    assert page["text"].count("LATE_MECHANISM_SENTINEL") == 1
    assert not page["has_more"] and page["selection_complete_in_this_response"]
    assert page["full_text_complete_in_this_response"] == (mode == "full_text")
    assert page["auto_read_additional_characters"] > 0
    assert build_model_view("read_article", result).data["reading"] == page
    reader.read_article("1", mode=mode, section_ids=ids, auto_read=True)
    assert len(calls) == 2


def test_overlong_selection_keeps_exact_valid_continuation_and_no_false_completion(tmp_path):
    xml = '<article><body><p>' + 'Long paragraph without breaks. ' * 8000 + '</p></body></article>'
    def respond(request):
        if request.url.path.endswith("search"):
            return httpx.Response(200, json={"hitCount": 1, "resultList": {"result": [{"id": "1", "pmcid": "PMC1"}]}})
        return httpx.Response(200, text=xml)
    reader = EuropePMCClient(CachedHTTP(tmp_path, client=httpx.Client(transport=httpx.MockTransport(respond))))
    arguments = dict(article_id="1", mode="full_text", max_chars=12000)
    texts = []
    while True:
        result = reader.read_article(**arguments, auto_read=True)
        assert result.status == "ok"
        page = result.data["reading"]
        assert len(page["text"]) <= AUTO_READ_MAX_CHARS
        texts.append(page["text"])
        if not page["has_more"]:
            assert not page["selection_complete_in_this_response"]
            break
        assert page["continuation_reason"]
        assert not page["full_text_complete_in_this_response"]
        arguments = page["next_request"]
        assert arguments["max_chars"] == 12000
    assert "".join(texts) == parse_article(xml)["text"]


def test_saved_four_partial_selections_now_complete_without_changing_selected_sections():
    root = Path(__file__).resolve().parents[1] / "runs/research_batch_16_20261021_003900"
    if not root.is_dir():
        pytest.skip("Optional saved-run regression records are absent")
    from realclip_d.tools.article_reading import reading_selection
    cases = []
    for sid in ("D_R015", "D_R017", "D_R021", "D_R025"):
        for line in (root / f"samples/{sid}/research/tools.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["tool"] == "read_article" and r.get("model_result", {}).get("data", {}).get("reading", {}).get("has_more"):
                page = reading_selection(r["result"]["data"], *r["arguments"], **r["keyword_arguments"], auto_read=True)
                assert not page["has_more"] and page["selection_complete_in_this_response"]
                assert page["section_ids"] == r["keyword_arguments"]["section_ids"]
                assert page["end"] == page["total_characters"]
                cases.append(sid)
    assert cases == ["D_R015", "D_R017", "D_R021", "D_R025"]


def test_saved_parbendazole_search_exposes_previously_hidden_direct_studies():
    path = Path(__file__).resolve().parents[1] / "runs/research_batch_16_20261021_003900/samples/D_R021/research/tools.jsonl"
    if not path.is_file():
        pytest.skip("Optional saved-run regression records are absent")
    matched = False
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["tool"] != "search_literature":
            continue
        raw = ToolResult.model_validate(row["result"])
        ids = {r.get("id") for r in raw.data.get("results", [])}
        if {"31861153", "35199530"} <= ids:
            view = build_model_view("search_literature", raw).data
            assert {"31861153", "35199530"} <= {r.get("id") for r in view["candidates"]}
            assert view["_model_view"]["omitted_rows"] == 0
            matched = True
    assert matched
