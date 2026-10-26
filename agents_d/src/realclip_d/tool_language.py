"""English presentation of external records; originals remain in tool receipts.

This does not translate scientific text. Source-provided English labels may be
displayed; mixed-language scientific passages are explicitly unavailable.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from .language import _HAN

UNAVAILABLE = "[Source text unavailable in English; original retained in the tool receipt.]"


def has_han(value: str) -> bool:
    return bool(_HAN.search(value))


def english_label(value: str) -> str:
    """Keep an existing English title/label, never translate a scientific claim."""
    if not has_han(value):
        return value
    matches = list(_HAN.finditer(value))
    # Bilingual bibliographies place the English title after the translated one.
    suffix = value[matches[-1].end():].strip(" \t\n.;:,-()[]")
    prefix = value[:matches[0].start()].strip()
    if not re.search(r"[A-Za-z]{2,}", prefix):
        suffix = re.sub(r"^[^A-Za-z]+", "", suffix)
        return suffix if len(re.findall(r"[A-Za-z]{2,}", suffix)) >= 3 else UNAVAILABLE
    # Restrict inline alias removal to parenthesized labels, e.g. a brand name.
    def alias(match):
        text = match.group(0)
        if not has_han(text):
            return text
        text = _HAN.sub("", text)[1:-1].strip(" ;,:-/")
        return f"({text})" if text else ""
    label = re.sub(r"\([^()]*\)", alias, value)
    if has_han(label):
        return UNAVAILABLE
    return re.sub(r" {2,}", " ", label).strip()


def english_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Project external metadata/fields, leaving already paged article text alone.

    Article text must be projected before offsets are computed. Search abstracts
    without a complete English counterpart are unavailable, not partial reads.
    """
    notes: list[dict[str, str]] = []

    def walk(value, path="data", key=""):
        if key == "reading":
            return deepcopy(value)
        if isinstance(value, dict):
            result = {k: walk(v, f"{path}.{k}", k) for k, v in value.items()}
            if isinstance(value.get("abstractText"), str) and has_han(value["abstractText"]):
                result.update(abstractText=None, abstract_language_status="unavailable_in_English")
            return result
        if isinstance(value, list):
            return [walk(v, f"{path}[{i}]", key) for i, v in enumerate(value)]
        if isinstance(value, str) and has_han(value):
            label = key in {"title", "label", "TOCHeading", "journalTitle"}
            projected = english_label(value) if label else UNAVAILABLE
            notes.append({"path": path, "status": "source_English_label" if projected != UNAVAILABLE
                          else "unavailable_in_English"})
            return projected
        return value

    projected = walk(data)
    if notes:
        projected["_language_view"] = {"changes": notes, "translation_performed": False,
                                      "original_retained_in_tool_receipt": True}
    return projected


def article_presentation(data: dict[str, Any]) -> dict[str, Any]:
    """Build an English document before paging, retaining source section IDs.

    The returned copy is solely for presentation; data/document/fulltext_xml in
    the original retrieval receipt are never modified.
    """
    from .tools.article_reading import parse_article

    document = data.get("document") or {}
    if not (has_han(document.get("text", "")) or has_han(data.get("abstract") or "")):
        return data
    projected = dict(data)
    if data.get("fulltext_xml"):
        projected["document"] = parse_article(data["fulltext_xml"], data.get("abstract"), english_view=True)
    if has_han(data.get("abstract") or ""):
        # Prefer the original English XML abstract; never concatenate a partially
        # unavailable passage and call it an intact abstract.
        import xml.etree.ElementTree as ET
        candidate = None
        if data.get("fulltext_xml"):
            root = ET.fromstring(data["fulltext_xml"])
            for node in root.iter():
                node.tag = node.tag.rsplit("}", 1)[-1]
            for node in root.findall("./front/article-meta/abstract"):
                text = " ".join("".join(node.itertext()).split())
                if text and not has_han(text):
                    candidate = text
                    break
        projected["abstract"] = candidate
        projected["abstract_language_status"] = "source_English_XML_abstract" if candidate else "unavailable_in_English"
    return projected
