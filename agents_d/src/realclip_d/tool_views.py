"""Bound model-facing evidence while retaining the full tool receipt elsewhere.

These are presentation views, not replacement retrieval records. All Source
records are preserved verbatim. Article pages use offsets in the selected
normalized text, never original XML byte positions, and bypass generic clipping.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from .contracts import ToolResult
from .tool_language import article_presentation, english_fields

DATA_CHARACTER_LIMIT = 40_000
_SUPPORTED = {"get_chembl_evidence", "read_article", "search_literature", "get_pubchem_annotations"}
def _compact(value: Any, *, string_limit: int, row_limit: int,
             omissions: list[dict[str, Any]], path: str = "data", depth: int = 0) -> Any:
    def note(reason: str, **counts: Any) -> None:
        if len(omissions) < 80:
            omissions.append({"path": path, "reason": reason, **counts})

    if isinstance(value, str):
        if len(value) > string_limit:
            note("string_excerpt", original_characters=len(value), shown_characters=string_limit)
            return value[:string_limit]
        return value
    if isinstance(value, (dict, list)) and depth >= 12:
        note("depth_limit")
        return {"omitted": "See the complete recorded tool receipt."}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"molfile", "fulltext_xml"}:
                note("raw_structure_or_xml_omitted", field=key)
                continue
            # Never clip provenance identifiers or reference records. Source
            # records themselves live outside data and bypass compaction.
            result[key] = deepcopy(item) if key in {"source_id", "source_ids"} else _compact(
                item, string_limit=string_limit, row_limit=row_limit,
                omissions=omissions, path=f"{path}.{key}", depth=depth + 1)
        return result
    if isinstance(value, list):
        if len(value) > row_limit:
            note("rows_omitted", original_rows=len(value), shown_rows=row_limit)
        return [_compact(item, string_limit=string_limit, row_limit=row_limit,
                         omissions=omissions, path=f"{path}[{index}]", depth=depth + 1)
                for index, item in enumerate(value[:row_limit])]
    return value


def _bound(data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    for string_limit, row_limit in [(6000, 12), (2500, 10), (1500, 8), (800, 4), (300, 2), (100, 1)]:
        omissions: list[dict[str, Any]] = []
        compacted = _compact(data, string_limit=string_limit, row_limit=row_limit, omissions=omissions)
        compacted["_model_view"] = {**metadata, "additional_omissions": omissions,
                                    "complete_tool_receipt_retained": True}
        if len(json.dumps(compacted, ensure_ascii=False)) <= DATA_CHARACTER_LIMIT:
            return compacted
    # An unusually wide upstream object may still be large after per-field
    # limits. Omit top-level payloads explicitly instead of returning invalid
    # sliced JSON. This does not affect the complete disk receipt or Sources.
    return {"_model_view": {**metadata, "complete_tool_receipt_retained": True,
                            "payload_omitted": "Upstream object exceeds the model-view size limit.",
                            "available_data_keys": list(data)[:100]}}


def _search_view(data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Expose every candidate; page complete abstracts rather than clipping papers."""
    rows = data.get("results", [])
    fields = {"id", "source", "pmid", "pmcid", "doi", "title", "pubYear",
              "isOpenAccess", "inPMC", "inEPMC", "hasPDF"}
    candidates = [{key: value for key, value in row.items() if key in fields} for row in rows]
    start = data.get("result_offset", 0)
    query, limit, cursor = data.get("query", ""), data.get("limit", len(rows) or 10), data.get("cursor")
    view = {key: data[key] for key in ("query", "limit", "cursor", "hit_count", "next_cursor") if key in data}
    view.update(candidates=candidates, results=[], result_offset=start,
                next_request=None, next_page_request=None)
    # Leave room for pagination metadata. Full titles and IDs are never clipped.
    size = len(json.dumps(view, ensure_ascii=False)) + 2000
    end = start
    for index in range(start, len(rows)):
        row = {**candidates[index], "abstractText": rows[index].get("abstractText"),
               "abstract_available": bool(rows[index].get("abstractText"))}
        if rows[index].get("abstract_language_status"):
            row["abstract_language_status"] = rows[index]["abstract_language_status"]
        extra = len(json.dumps(row, ensure_ascii=False)) + 2
        if size + extra > DATA_CHARACTER_LIMIT:
            if view["results"]:
                break
            # An individual oversized abstract stays intact in overview access.
            row.pop("abstractText")
            row["abstract_in_overview"] = True
            extra = len(json.dumps(row, ensure_ascii=False)) + 2
        view["results"].append(row)
        size += extra
        end = index + 1
    if end < len(rows):
        view["next_request"] = {"query": query, "limit": limit, "cursor": cursor, "result_offset": end}
    if data.get("next_cursor"):
        view["next_page_request"] = {"query": query, "limit": limit,
                                     "cursor": data["next_cursor"], "result_offset": 0}
    view["_model_view"] = {**metadata, "retrieved_rows": len(rows), "shown_rows": len(candidates),
                           "omitted_rows": 0, "abstract_rows_this_response": len(view["results"]),
                           "deferred_abstract_rows": len(rows) - end,
                           "search_results_are_not_article_read_receipts": True,
                           "complete_tool_receipt_retained": True,
                           "note": "All retrieved candidates are listed. Full abstracts are paginated in results; use next_request for remaining abstracts, next_page_request for more database hits, or read_article overview for a chosen paper. Open-access flags do not establish successful full-text access."}
    if "_language_view" in data:
        view["_language_view"] = data["_language_view"]
    return view


def _share_target_annotations(data: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Losslessly share repeated functional annotations within this one response.

    Each component keeps its complete ordered list of indexes. No memory tool,
    previous request, omitted scientific record, or extra retrieval is involved.
    """
    components = [c for t in data.get("targets", []) for c in t.get("target_components", [])
                  if isinstance(c.get("target_component_xrefs"), list)]
    original_size = sum(len(json.dumps(c["target_component_xrefs"])) for c in components)
    if original_size < 80_000:
        return
    records, indexes, links = [], {}, []
    for component in components:
        references = []
        for row in component["target_component_xrefs"]:
            key = json.dumps(row, sort_keys=True, ensure_ascii=False)
            if key not in indexes:
                indexes[key] = len(records)
                records.append(row)
            references.append(indexes[key])
        links.append(references)
    if len(json.dumps(records)) + len(json.dumps(links)) >= original_size * .8:
        return
    for component, references in zip(components, links, strict=True):
        component["target_component_xrefs"] = {"record_indexes": references,
                                               "lookup": "target_annotation_records"}
    data["target_annotation_records"] = records
    metadata["shared_target_annotations"] = {
        "unique_records": len(records), "component_references": sum(map(len, links)),
        "all_functional_annotations_retained": True,
        "note": "Lossless within-response representation. Each component's target_component_xrefs.record_indexes lists zero-based indexes into data.target_annotation_records, in original order including duplicates. Every annotation value and its component association is retained; no extra tool call is needed."}


def build_model_view(tool_name: str, result: ToolResult) -> ToolResult:
    """Return a bounded research-tool view; never mutate full result or Sources."""
    if tool_name not in _SUPPORTED:
        return result
    data = deepcopy(result.data)
    metadata: dict[str, Any] = {"tool": tool_name, "presentation_only": True,
                                "original_data_characters": len(json.dumps(data, ensure_ascii=False))}
    if tool_name == "read_article":
        # Pages and complete outlines bypass generic string/list clipping. Every
        # omitted page has an explicit continuation; raw XML stays on disk.
        from .tools.article_reading import DEFAULT_MAX_CHARS, reading_page, reading_selection

        projected = article_presentation(data)
        if projected is not data:
            # Render before selecting pages: section IDs, offsets and hashes must
            # consistently refer to the same English presentation on every call.
            data = projected
            old = data.get("reading")
            if old:
                limit = (old.get("next_request") or {}).get("max_chars") or (
                    DEFAULT_MAX_CHARS if old.get("auto_read_enabled") else old.get("response_character_limit", DEFAULT_MAX_CHARS))
                data["reading"] = reading_selection(data, (data.get("article") or {}).get("id", ""),
                    (data.get("article") or {}).get("source", "MED"), old["mode"],
                    old.get("section_ids") or None, old["offset"], limit,
                    auto_read=old.get("auto_read_enabled", False))

        article = data.get("article") or {}
        article_fields = {"id", "source", "pmcid", "doi", "title", "authorString", "pubYear",
                          "journalTitle", "journalInfo", "isOpenAccess", "fullTextUrlList"}
        reading = data.get("reading")
        if reading is None and data.get("mode", "overview") == "overview" and result.status == "ok":
            # Older synthetic tool fixtures have no section document. Show only
            # their abstract, sequentially; never reconstruct unseen excerpts.
            reading = reading_page(data, article.get("id", ""), article.get("source", "MED"),
                                   "overview", None, 0, DEFAULT_MAX_CHARS)
        metadata.update(
            fulltext_xml_omitted=True,
            full_text_retrieved=data.get("fulltext_status") == "accessible",
            full_text_complete=bool(reading and reading.get("full_text_complete_in_this_response")),
            text_selection="Agent-selected mode/sections, sequential paging; no keyword filtering.",
            complete_tool_receipt_retained=True,
            full_text_content_sha256=(data.get("document") or {}).get("sha256"),
            full_text_total_characters=len((data.get("document") or {}).get("text", "")),
            language_projection=(data.get("document") or {}).get("language_projection"),
        )
        return result.model_copy(update={"data": english_fields({
            "article": {key: value for key, value in article.items() if key in article_fields},
            "mode": data.get("mode", "overview"),
            "abstract_available": bool(data.get("abstract")),
            "abstract_language_status": data.get("abstract_language_status"),
            "fulltext_status": data.get("fulltext_status"),
            "fulltext_error": data.get("fulltext_error"),
            "sections": deepcopy((data.get("document") or {}).get("sections", [])),
            "linked_materials": deepcopy((data.get("document") or {}).get("linked_materials", [])),
            "reading": reading,
            "_model_view": metadata,
        })})
    elif tool_name == "get_chembl_evidence":
        # Scientific records are indivisible here: no row caps, string excerpts,
        # or generic bounding of activity values, conditions and linked records.
        # The runtime archives this view and the complete original tool receipt.
        structures = (data.get("molecule") or {}).get("molecule_structures") or {}
        structures.pop("molfile", None)
        deferred = {"PDB": 0, "PDBe": 0}
        for target in data.get("targets", []):
            for component in target.get("target_components", []):
                xrefs = component.get("target_component_xrefs")
                if not isinstance(xrefs, list):
                    continue
                keep, counts = [], {"PDB": 0, "PDBe": 0}
                for row in xrefs:
                    source = row.get("xref_src_db")
                    if source in counts:
                        counts[source] += 1
                        deferred[source] += 1
                    else:
                        keep.append(row)
                if any(counts.values()):
                    component["target_component_xrefs"] = keep
                    component["structure_xrefs_deferred"] = counts
        metadata["structure_index_view"] = {
            "deferred_rows_by_database": deferred,
            "note": "Only PDB/PDBe structure-index entries are deferred to the complete backend receipt. All target components, functional annotations, activities, mechanisms, assays, documents and experimental conditions are retained."}
        _share_target_annotations(data, metadata)
        metadata.update(row_counts={key: {"retrieved_rows": len(data[key]),
                                          "shown_rows": len(data[key]), "omitted_rows": 0}
                                    for key in ("activities", "mechanisms", "assays", "targets", "documents")
                                    if isinstance(data.get(key), list)},
                        molfile_omitted=True,
                        scope_note="Retain exact activity scope and parent-aggregated mechanism scope; no equivalence inferred.",
                        complete_tool_receipt_retained=True)
        data["_model_view"] = metadata
        return result.model_copy(update={"data": english_fields(data)})
    elif tool_name == "search_literature":
        return result.model_copy(update={"data": _search_view(english_fields(data), metadata)})
    else:
        metadata["annotation_view"] = "Bounded structured annotation; omitted fields/rows require the full tool receipt."
    return result.model_copy(update={"data": english_fields(_bound(data, metadata))})


def research_view(tool_name: str, raw: ToolResult, article_view: ToolResult | None = None) -> ToolResult:
    """Keep the existing tool content; provenance is recorded by the application."""
    view = article_view if article_view is not None else build_model_view(tool_name, raw)
    return view.model_copy(update={"sources": [], "data": english_fields(view.data)})
