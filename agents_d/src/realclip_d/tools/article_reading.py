"""Deterministic article sections and lossless paging, without topic filtering."""
from __future__ import annotations

import hashlib
from html import unescape
from html.parser import HTMLParser
from typing import Any, Literal
import xml.etree.ElementTree as ET

ArticleMode = Literal["overview", "sections", "full_text"]
DEFAULT_MAX_CHARS = 12_000
MAX_PAGE_CHARS = 24_000
AUTO_READ_MAX_CHARS = 96_000


class _AbstractText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str):
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs):
        if tag in {"p", "div", "h3", "h4", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str):
        if tag in {"p", "div", "h3", "h4"}:
            self.parts.append("\n")


def abstract_text(value: str | None) -> str:
    parser = _AbstractText()
    parser.feed(value or "")
    return "\n\n".join(" ".join(line.split()) for line in "".join(parser.parts).splitlines() if line.strip())


def _text(node: ET.Element) -> str:
    parts: list[str] = []

    def walk(element: ET.Element):
        parts.append(element.text or "")
        for child in element:
            block = child.tag in {"p", "title", "label", "list-item", "td", "th", "tr"}
            if block:
                parts.append(" ")
            walk(child)
            if block:
                parts.append(" ")
            parts.append(child.tail or "")

    walk(node)
    return " ".join(unescape("".join(parts)).split())


def parse_article(xml_text: str, abstract: str | None = None, *, english_view: bool = False) -> dict[str, Any]:
    """Render available JATS text and retain every section in document order.

    Abstracts, body, back matter and floating figures/tables are included. No
    topic, compound, mechanism or morphology words influence selection.
    """
    root = ET.fromstring(xml_text)
    from ..tool_language import UNAVAILABLE, english_label, has_han
    for node in root.iter():
        if isinstance(node.tag, str):
            node.tag = node.tag.rsplit("}", 1)[-1]
    blocks: list[str] = []
    sections: list[dict[str, Any]] = []
    omissions: list[dict[str, str]] = []
    suppressed = 0
    english_abstract_available = any(
        _text(node) and not has_han(_text(node))
        for node in root.findall("./front/article-meta/abstract"))
    containers = {"abstract", "trans-abstract", "body", "sec", "back", "app", "ref-list",
                  "ack", "notes", "fn-group", "glossary", "floats-group"}

    def emit(value: str):
        if suppressed:
            return
        value = value.strip()
        if english_view and has_han(value):
            omissions.append({"kind": "scientific_passage", "status": "unavailable_in_English"})
            value = UNAVAILABLE
        if value:
            blocks.append(value)

    def visit(node: ET.Element, parent: str | None = None, depth: int = 0):
        nonlocal suppressed
        tag = node.tag
        if tag in containers:
            heading = " ".join(_text(child) for child in node if child.tag in {"label", "title"}).strip()
            if english_view and has_han(heading):
                heading = english_label(heading)
                omissions.append({"kind": "section_heading", "status": "English_display_label"})
            identifier = f"section_{len(sections) + 1}"
            entry = {"id": identifier, "title": heading or {
                "abstract": "Abstract", "trans-abstract": "Translated abstract", "body": "Article body",
                "back": "Back matter", "ref-list": "References", "floats-group": "Figures and tables",
            }.get(tag, f"Untitled {tag}"), "parent_id": parent, "level": depth,
                     "_start": len(blocks), "_end": 0}
            sections.append(entry)
            emit(heading)
            translated = (english_view and tag == "trans-abstract" and english_abstract_available
                          and node.get("{http://www.w3.org/XML/1998/namespace}lang", "").lower()
                          not in {"", "en", "eng", "en-us", "en-gb"})
            if translated:
                omissions.append({"kind": "translated_abstract", "status": "omitted_English_abstract_available"})
                emit("[Translated abstract omitted; source-provided English abstract is available.]")
                suppressed += 1
            emit(" ".join((node.text or "").split()))
            for child in node:
                if child.tag not in {"label", "title"}:
                    visit(child, identifier, depth + 1)
                emit(" ".join((child.tail or "").split()))
            if translated:
                suppressed -= 1
            entry["_end"] = len(blocks)
        elif tag == "table":
            rows = []
            for row in node.iter("tr"):
                cells = [_text(cell) for cell in row if cell.tag in {"td", "th"}]
                if cells:
                    rows.append(" | ".join(cells))
            emit("\n".join(rows) if rows else _text(node))
        elif tag in {"p", "title", "label", "caption", "ref", "disp-formula", "def-item", "list-item"}:
            emit(_text(node))
        elif len(node):
            emit(" ".join((node.text or "").split()))
            for child in node:
                visit(child, parent, depth)
                emit(" ".join((child.tail or "").split()))
        else:
            emit(_text(node))

    abstracts = root.findall("./front/article-meta/abstract") + root.findall("./front/article-meta/trans-abstract")
    for node in abstracts:
        visit(node)
    if not abstracts and abstract_text(abstract):
        fallback = ET.Element("abstract")
        ET.SubElement(fallback, "p").text = abstract_text(abstract)
        visit(fallback)
    available_abstract = "\n\n".join(blocks)
    body = root.find("./body")
    body_entry = None
    for node in root:
        if node.tag in {"body", "back", "floats-group"}:
            if node is body:
                body_entry = len(sections)
            visit(node)
    text = "\n\n".join(blocks)
    starts = []
    position = 0
    for block in blocks:
        starts.append(position)
        position += len(block) + 2
    for entry in sections:
        first, last = entry.pop("_start"), entry.pop("_end")
        start = starts[first] if first < len(starts) else len(text)
        end = starts[last - 1] + len(blocks[last - 1]) if last > first else start
        entry.update(start=start, end=end, characters=end - start)
    body_range = sections[body_entry] if body_entry is not None else None
    linked_materials = []
    for material in root.iter("supplementary-material"):
        links = [value for node in material.iter() for key, value in node.attrib.items()
                 if key.rsplit("}", 1)[-1] == "href"]
        linked_materials.append({"label": _text(material), "links": links, "status": "not_retrieved"})
    document = {"text": text, "sections": sections, "abstract_text": available_abstract,
            "body_text": text[body_range["start"]:body_range["end"]] if body_range else "",
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "linked_materials": linked_materials,
            "scope": "Available XML abstracts, body, back matter, and floating figure/table text; "
                     "linked supplements and image pixels are not downloaded."}
    if english_view and omissions:
        document["scope"] += " English presentation: see language_projection for unavailable original content."
        document["language_projection"] = {"changes": omissions, "translation_performed": False,
                                            "original_content_complete": False}
    return document


def validate_reading(mode: str, section_ids: list[str] | None, offset: int, max_chars: int) -> str | None:
    if mode not in {"overview", "sections", "full_text"}:
        return "mode must be overview, sections or full_text"
    if type(offset) is not int or offset < 0:
        return "offset must be a nonnegative integer"
    if type(max_chars) is not int or not 1 <= max_chars <= MAX_PAGE_CHARS:
        return f"max_chars must be 1..{MAX_PAGE_CHARS}"
    if section_ids is not None and (not isinstance(section_ids, list) or
                                   any(not isinstance(sid, str) for sid in section_ids)):
        return "section_ids must be a list of section IDs or null"
    if mode == "sections" and (not section_ids or len(set(section_ids)) != len(section_ids)):
        return "sections mode requires nonempty, unique section_ids from the overview"
    if mode != "sections" and section_ids:
        return "section_ids is only used in sections mode"
    return None


def reading_page(data: dict[str, Any], article_id: str, source: str, mode: ArticleMode,
                 section_ids: list[str] | None, offset: int, max_chars: int) -> dict[str, Any]:
    """Offsets apply to the selected text, with no skipped characters between pages."""
    document = data.get("document") or {"text": "", "sections": []}
    if mode == "overview":
        selected = abstract_text(data.get("abstract"))
        scope = "abstract"
    elif mode == "full_text":
        selected = document["text"]
        scope = document["scope"]
    else:
        by_id = {entry["id"]: entry for entry in document["sections"]}
        unknown = set(section_ids or []) - set(by_id)
        if unknown:
            raise ValueError("Unknown section IDs: " + ", ".join(sorted(unknown)))
        ranges: list[list[int]] = []
        for entry in sorted((by_id[sid] for sid in section_ids or []), key=lambda row: (row["start"], -row["end"])):
            start, end = entry["start"], entry["end"]
            if ranges and start <= ranges[-1][1]:
                ranges[-1][1] = max(ranges[-1][1], end)
            else:
                ranges.append([start, end])
        selected = "\n\n".join(document["text"][start:end] for start, end in ranges)
        scope = "Selected sections in document order, including their subsections; overlapping selections appear once."
    if offset > len(selected):
        raise ValueError("offset exceeds the selected text length")
    end = min(len(selected), offset + max_chars)
    if end < len(selected):
        boundary = selected.rfind("\n\n", offset, end)
        if boundary > offset:
            end = boundary + 2
    more = end < len(selected)
    continuation = {"article_id": article_id, "source": source, "mode": mode,
                    "section_ids": section_ids, "offset": end, "max_chars": max_chars} if more else None
    return {"mode": mode, "section_ids": section_ids or [], "scope": scope,
            "offset": offset, "end": end, "total_characters": len(selected),
            "content_sha256": hashlib.sha256(selected.encode()).hexdigest(),
            "text": selected[offset:end], "has_more": more, "next_request": continuation,
            "selection_complete_in_this_response": offset == 0 and not more,
            "full_text_complete_in_this_response": mode == "full_text" and bool(selected) and offset == 0 and not more,
            "offset_reference": "Character offsets in the selected normalized text; start inclusive, end exclusive. "
                                "Long paragraphs may span pages. Coverage records exposure, not comprehension."}


def reading_selection(data: dict[str, Any], article_id: str, source: str, mode: ArticleMode,
                      section_ids: list[str] | None, offset: int, max_chars: int,
                      *, auto_read: bool = False) -> dict[str, Any]:
    """Expand an already retrieved selection locally, without another HTTP/model call.

    The research tool opts in. Low-level callers can retain explicit small pages.
    A hard response limit still exposes an exact, valid continuation request.
    """
    page = reading_page(data, article_id, source, mode, section_ids, offset, max_chars)
    initial_end = page["end"]
    if auto_read and mode != "overview" and page["has_more"]:
        page = reading_page(data, article_id, source, mode, section_ids, offset, AUTO_READ_MAX_CHARS)
        if page["next_request"]:
            page["next_request"]["max_chars"] = max_chars
    page.update(auto_read_enabled=auto_read and mode != "overview",
                auto_read_additional_characters=page["end"] - initial_end,
                response_character_limit=AUTO_READ_MAX_CHARS if auto_read and mode != "overview" else max_chars)
    projection = (data.get("document") or {}).get("language_projection")
    if projection:
        page.update(language_projection=projection,
                    presentation_complete_in_this_response=page["full_text_complete_in_this_response"],
                    full_text_complete_in_this_response=False)
    if auto_read and mode != "overview" and page["has_more"]:
        page["continuation_reason"] = "Selected text exceeds the automatic response limit. Follow next_request to finish the selection, or explain why further reading is unnecessary."
    return page
