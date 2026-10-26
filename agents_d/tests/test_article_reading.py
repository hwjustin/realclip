"""Agent-chosen article reading, exact paging and exposure with local fixtures only."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from realclip_d import agent_runtime as runtime
from realclip_d.config import Settings
from realclip_d.contracts import SampleInput
from realclip_d.tool_views import build_model_view
from realclip_d.tools import CachedHTTP, EuropePMCClient
from realclip_d.tools.article_reading import parse_article


XML = '''<article xmlns:xlink="http://www.w3.org/1999/xlink">
<front><article-meta><abstract><p>XML abstract of the fictional study.</p></abstract></article-meta></front>
<body>
<p>Unsectioned introductory material.</p>
<sec><title>Methods</title><p>Unfiltered culture and exposure details.</p>
  <sec><title>Measurement</title><p>Acquisition setup.</p></sec></sec>
<sec><title>Results</title>
  <sec><title>Signaling</title><p>ENZYME_ALPHA activates FACTOR_BETA.</p></sec>
  <sec><title>Appearance</title><p>Nocodazole produced rounded cells in this fictional example.</p>
  <fig><label>Figure 1</label><caption><title>Mechanism figure</title><p>Signaling diagram caption.</p></caption></fig>
  <table-wrap><label>Table 1</label><caption><p>Measurement summary.</p></caption>
    <table><thead><tr><th>Condition</th><th>Value</th></tr></thead>
    <tbody><tr><td>Exposure</td><td>42</td></tr></tbody></table>
    <table-wrap-foot><p>Table footnote.</p></table-wrap-foot></table-wrap>
  </sec></sec>
<sec><p>Untitled final section.</p></sec>
</body>
<back><ref-list><title>References</title><ref>Fictional bibliography entry.</ref></ref-list>
<supplementary-material xlink:href="supplement.pdf"><caption><p>Supplementary methods.</p></caption></supplementary-material></back>
<floats-group><fig><caption><p>Floating figure caption.</p></caption></fig></floats-group>
</article>'''


@pytest.fixture
def client(tmp_path):
    requests = []

    def handler(request):
        requests.append(str(request.url))
        if request.url.path.endswith("search"):
            return httpx.Response(200, json={"hitCount": 1, "resultList": {"result": [{
                "id": "1", "source": "MED", "pmcid": "PMC1", "title": "Fictional study",
                "abstractText": "Database abstract of the fictional study."}]}})
        return httpx.Response(200, text=XML)

    http = CachedHTTP(tmp_path / "cache", client=httpx.Client(transport=httpx.MockTransport(handler)))
    return EuropePMCClient(http), requests


def test_overview_has_abstract_and_all_nested_and_untitled_sections(client):
    reader, requests = client
    raw = reader.read_article("1")
    view = build_model_view("read_article", raw)
    assert raw.status == "ok" and view.data["mode"] == "overview"
    titles = [entry["title"] for entry in view.data["sections"]]
    assert {"Methods", "Measurement", "Results", "Signaling", "Appearance", "Untitled sec", "References"} <= set(titles)
    assert view.data["reading"]["text"] == "Database abstract of the fictional study."
    assert "ENZYME_ALPHA" not in json.dumps(view.data)
    assert "Nocodazole produced" not in json.dumps(view.data)
    assert "body_text" not in view.data and "fulltext_xml" not in view.data
    assert not view.data["_model_view"]["full_text_complete"]
    assert raw.data["fulltext_xml"] == XML and "ENZYME_ALPHA" in raw.data["document"]["text"]
    assert view.sources == raw.sources
    assert len(requests) == 2


def test_agent_can_read_mechanism_without_any_compound_or_morphology_keyword(client):
    reader, _ = client
    outline = reader.read_article("1").data["document"]["sections"]
    identifier = next(section["id"] for section in outline if section["title"] == "Signaling")
    result = reader.read_article("1", mode="sections", section_ids=[identifier])
    page = result.data["reading"]
    assert page["text"] == "Signaling\n\nENZYME_ALPHA activates FACTOR_BETA."
    assert page["selection_complete_in_this_response"]
    assert not page["full_text_complete_in_this_response"]
    assert "Nocodazole" not in page["text"]


def test_parent_and_child_selection_is_deduplicated_and_in_document_order(client):
    reader, requests = client
    overview = reader.read_article("1")
    sections = {item["title"]: item for item in overview.data["document"]["sections"]}
    chosen = [sections[title]["id"] for title in ("Signaling", "Results", "Methods")]
    result = reader.read_article("1", mode="sections", section_ids=chosen)
    text = result.data["reading"]["text"]
    assert text.index("Methods") < text.index("Results") < text.index("Signaling")
    assert text.count("ENZYME_ALPHA") == 1
    assert "Acquisition setup." in text
    assert "Unsectioned introductory material." not in text
    assert len(requests) == 2  # Same cached official responses, no repeat HTTP calls.


def test_full_text_can_be_requested_directly_and_retains_tables_captions_and_back_matter(client):
    reader, _ = client
    result = reader.read_article("1", mode="full_text", max_chars=24000)
    text = result.data["reading"]["text"]
    for value in ("XML abstract", "Unsectioned introductory", "ENZYME_ALPHA", "Mechanism figure",
                  "Signaling diagram caption.", "Condition | Value", "Exposure | 42", "Table footnote.",
                  "Untitled final section.", "Fictional bibliography", "Floating figure caption."):
        assert value in text
    assert result.data["reading"]["full_text_complete_in_this_response"]
    view = build_model_view("read_article", result)
    assert view.data["reading"] == result.data["reading"]
    assert view.data["linked_materials"][0]["links"] == ["supplement.pdf"]
    assert view.data["linked_materials"][0]["status"] == "not_retrieved"


@pytest.mark.parametrize("mode", ["overview", "sections", "full_text"])
def test_following_next_request_reconstructs_exact_text_without_gaps(client, mode):
    reader, _ = client
    section_ids = None
    if mode == "sections":
        sections = reader.read_article("1").data["document"]["sections"]
        section_ids = [next(item["id"] for item in sections if item["title"] == "Results")]
    expected = reader.read_article("1", mode=mode, section_ids=section_ids, max_chars=24000).data["reading"]["text"]
    arguments = {"article_id": "1", "mode": mode, "section_ids": section_ids, "max_chars": 23}
    pages = []
    end = 0
    while arguments is not None:
        result = reader.read_article(**arguments)
        assert result.status == "ok"
        page = result.data["reading"]
        assert page["offset"] == end and 0 < len(page["text"]) <= 23
        assert page["text"] == expected[page["offset"]:page["end"]]
        pages.append(page["text"])
        end = page["end"]
        arguments = page["next_request"]
    assert "".join(pages) == expected


def test_long_paragraph_and_late_mechanism_are_reachable_without_truncation(client):
    reader, _ = client
    first = reader.read_article("1", mode="full_text", max_chars=7).data["reading"]
    next_page = reader.read_article(**first["next_request"]).data["reading"]
    assert first["text"] + next_page["text"] == reader.read_article("1", mode="full_text").data["reading"]["text"][:14]


@pytest.mark.parametrize("kwargs", [
    {"mode": "bad"}, {"mode": "sections"}, {"offset": -1}, {"max_chars": 0}, {"max_chars": 24001},
    {"mode": "sections", "section_ids": ["x", "x"]}, {"mode": "full_text", "section_ids": ["x"]},
])
def test_invalid_parameters_stop_before_retrieval(client, kwargs):
    reader, requests = client
    result = reader.read_article("1", **kwargs)
    assert result.status == "error" and result.message
    assert requests == []


def test_unknown_sections_and_invalid_offset_return_explicit_errors(client):
    reader, _ = client
    assert "Unknown section" in reader.read_article("1", mode="sections", section_ids=["missing"]).message
    assert "offset exceeds" in reader.read_article("1", offset=10**6).message


def test_namespaced_xml_and_missing_database_abstract_have_readable_xml_abstract(tmp_path):
    xml = XML.replace('<article xmlns:xlink=', '<article xmlns="urn:jats" xmlns:xlink=')

    def handler(request):
        if request.url.path.endswith("search"):
            return httpx.Response(200, json={"hitCount": 1, "resultList": {"result": [{"id": "1", "pmcid": "PMC1"}]}})
        return httpx.Response(200, text=xml)

    http = CachedHTTP(tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = EuropePMCClient(http).read_article("1")
    assert result.status == "ok"
    assert result.data["reading"]["text"] == "XML abstract of the fictional study."
    assert any(item["title"] == "Signaling" for item in result.data["document"]["sections"])


@pytest.mark.parametrize("fulltext,status", [(None, "not_in_pmc"), ("404", "unavailable"),
                                            ("<bad", "invalid_xml"), ("<article/>", "no_readable_body")])
def test_unavailable_fulltext_still_has_overview_but_cannot_claim_full_read(tmp_path, fulltext, status):
    def handler(request):
        if request.url.path.endswith("search"):
            article = {"id": "1", "abstractText": "Available abstract."}
            if fulltext is not None:
                article["pmcid"] = "PMC1"
            return httpx.Response(200, json={"hitCount": 1, "resultList": {"result": [article]}})
        return httpx.Response(404 if fulltext == "404" else 200, text=fulltext)

    reader = EuropePMCClient(CachedHTTP(tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler))))
    overview = reader.read_article("1")
    assert overview.status == "ok" and overview.data["fulltext_status"] == status
    assert overview.data["reading"]["text"] == "Available abstract."
    full = reader.read_article("1", mode="full_text")
    assert full.status == "unavailable"
    assert not build_model_view("read_article", full).data["_model_view"]["full_text_complete"]


def test_complete_outline_and_pages_are_not_clipped_by_generic_view_limits():
    xml = "<article><body>" + "".join(f"<sec><title>Section {i}</title><p>Content {i}.</p></sec>" for i in range(40)) + "</body></article>"
    document = parse_article(xml)
    from realclip_d.contracts import ToolResult
    from realclip_d.tools.article_reading import reading_page
    data = {"article": {}, "abstract": "", "mode": "full_text", "document": document, "fulltext_status": "accessible"}
    data["reading"] = reading_page(data, "1", "MED", "full_text", None, 0, 24000)
    raw = ToolResult(status="ok", data=data)
    original = deepcopy(raw.model_dump())
    view = build_model_view("read_article", raw)
    assert len(view.data["sections"]) == 41
    assert view.data["reading"]["text"] == document["text"]
    assert "Section 39" in view.data["reading"]["text"]
    assert raw.model_dump() == original


def test_exposure_coverage_requires_all_pages_and_does_not_double_count(client, tmp_path):
    reader, _ = client
    recorder = runtime._ToolRecorder(tmp_path / "stage", 100)
    async def invoke(**kwargs):
        return json.loads(await recorder.call("read_article", reader.read_article, "1", **kwargs))["data"]
    first = asyncio.run(invoke(mode="full_text", max_chars=70))
    again = asyncio.run(invoke(mode="full_text", max_chars=70))
    assert first["_model_view"]["full_text_characters_provided"] == again["_model_view"]["full_text_characters_provided"]
    assert not first["_model_view"]["full_text_complete"]
    following = first["reading"]["next_request"]
    last = first
    while following is not None:
        args = {key: value for key, value in following.items() if key != "article_id"}
        last = asyncio.run(invoke(**args))
        following = last["reading"]["next_request"]
    assert last["_model_view"]["full_text_complete"]
    assert last["_model_view"]["full_text_characters_provided"] == last["reading"]["total_characters"]
    # This last page alone was not the full article; cumulative exposure is distinct.
    assert not last["reading"]["full_text_complete_in_this_response"]
    assert asyncio.run(invoke())["_model_view"]["full_text_complete"]


def test_starting_at_later_page_does_not_claim_full_text_complete(client, tmp_path):
    reader, _ = client
    recorder = runtime._ToolRecorder(tmp_path / "stage", 2)
    data = json.loads(asyncio.run(recorder.call("read_article", reader.read_article, "1", mode="full_text", offset=20)))
    assert not data["data"]["_model_view"]["full_text_complete"]


def test_sdk_tool_exposes_modes_parameter_descriptions_and_selected_reading(client, tmp_path, monkeypatch):
    reader, _ = client
    methods = reader.read_article("1").data["document"]["sections"]
    identifier = next(item["id"] for item in methods if item["title"] == "Signaling")

    class Knowledge(runtime.MockKnowledge):
        def read(self, article_id, source, **kwargs):
            return reader.read_article(article_id, source, **kwargs)

    original = runtime.ScriptedModel
    steps = [[("resolve_compound", {})],
             [("read_article", {"article_id": "1", "source": "MED", "mode": "overview"})],
             [("read_article", {"article_id": "1", "source": "MED", "mode": "sections", "section_ids": [identifier]})],
             [("read_article", {"article_id": "1", "source": "MED", "mode": "full_text"})]]
    monkeypatch.setattr(runtime, "MockKnowledge", Knowledge)
    monkeypatch.setattr(runtime, "ScriptedModel", lambda _steps, final: original(steps, final))
    sample = SampleInput(sample_id="TOOL_TEST", compound={"name": "Synthetic"})
    asyncio.run(runtime.run_research(sample, Settings(), tmp_path / "research"))
    request = json.loads((tmp_path / "research/requests/001.json").read_text())
    tool = next(item for item in request["tools"] if item["name"] == "read_article")
    properties = tool["parameters"]["properties"]
    assert properties["mode"]["enum"] == ["overview", "sections", "full_text"]
    for name in ("article_id", "source", "mode", "section_ids", "offset", "max_chars"):
        assert properties[name]["description"]
    calls = [json.loads(line) for line in (tmp_path / "research/tools.jsonl").read_text().splitlines()]
    reads = [row for row in calls if row["tool"] == "read_article"]
    assert [row["model_result"]["data"]["mode"] for row in reads] == ["overview", "sections", "full_text"]
    assert "ENZYME_ALPHA" in reads[1]["model_result"]["data"]["reading"]["text"]
    assert "ENZYME_ALPHA" not in reads[0]["model_result"]["data"]["reading"]["text"]
