"""Reuse already-seen pilot evidence for explicit research-only synthesis.

No retrieval, model call, environment access, or inferred scientific content is
performed here. Read-article excerpts are copied from the recorded model view,
not rebuilt from the larger raw retrieval result.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .contracts import ToolResult

PACKET_MAX_BYTES = 100_000
_TOOLS = {"resolve_compound", "get_pubchem_annotations", "get_chembl_evidence",
          "search_literature", "read_article", "list_sources", "record_research_notes",
          "read_research_memory", "stop_article_reading"}


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _small(value: Any, *, depth: int = 0) -> Any:
    """Bound optional annotation context, with explicit truncation markers."""
    if isinstance(value, str):
        return value if len(value) <= 700 else {
            "excerpt": value[:700], "original_characters": len(value), "truncated": True}
    if isinstance(value, (dict, list)) and depth >= 8:
        return {"omitted": "Additional nesting is in the recorded model_result."}
    if isinstance(value, dict):
        return {key: _small(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        shown = [_small(item, depth=depth + 1) for item in value[:4]]
        return shown if len(value) <= 4 else {"shown_rows": shown, "original_rows": len(value),
                                           "omitted_rows": len(value) - 4}
    return value


def _search(data: dict[str, Any]) -> dict[str, Any]:
    rows = data.get("results", [])
    if not isinstance(rows, list):
        raise ValueError("Malformed search receipt: results must be a list")
    selected = {"id", "source", "pmcid", "doi", "title"}
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Malformed search receipt: article must be an object")
    return {"query": data.get("query"), "hit_count": data.get("hit_count"),
            "model_view_result_count": len(rows),
            "articles": [{key: value for key, value in row.items() if key in selected} for row in rows],
            "access_note": "Search discovery only; no abstract or full-text reading is established.",
            "original_view_metadata": deepcopy(data.get("_model_view", {}))}


def _chembl(data: dict[str, Any]) -> dict[str, Any]:
    molecule = data.get("molecule") or {}
    if not isinstance(molecule, dict):
        raise ValueError("Malformed ChEMBL receipt: molecule must be an object")
    structure = molecule.get("molecule_structures") or {}
    if not isinstance(structure, dict):
        raise ValueError("Malformed ChEMBL receipt: structures must be an object")
    projected = deepcopy(data)
    projected["molecule"] = deepcopy(molecule)
    if "molecule_structures" in projected["molecule"]:
        projected["molecule"]["molecule_structures"] = deepcopy(structure)
        projected["molecule"]["molecule_structures"].pop("molfile", None)
    projected["original_view_metadata"] = projected.pop("_model_view", {})
    return projected



def _receipt(line: str, index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("tool") not in _TOOLS:
            raise ValueError("Unknown or non-research tool")
        if not isinstance(row.get("arguments"), list) or not isinstance(row.get("keyword_arguments"), dict):
            raise ValueError("Missing exact recorded tool arguments")
        raw = row["result"]
        viewed = row["model_result"]
        ToolResult.model_validate(raw)
        ToolResult.model_validate(viewed)
        if raw.get("status") != viewed.get("status") or raw.get("sources", []) != viewed.get("sources", []):
            raise ValueError("Model result status or sources differ from the raw receipt")
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Malformed research receipt at tools.jsonl line {index}: {error}") from error
    tool = row["tool"]
    data = deepcopy(viewed.get("data", {}))
    if tool == "search_literature":
        data = _search(data)
    elif tool == "get_chembl_evidence":
        data = _chembl(data)
    elif tool == "get_pubchem_annotations":
        if _size(data) > 8000:
            data = {"bounded_annotation_context": _small(data),
                    "omission_note": "Further reduced from the already-seen annotation view; full model_result retained on disk."}
        if _size(data) > 8000:
            data = {"cid": data.get("cid"), "heading": data.get("heading"),
                    "omission_note": "Annotation context omitted for packet size; original model_result remains in tools.jsonl."}
    elif tool == "list_sources":
        data = {"note": "Authentic source records deduplicated once in the packet sources list."}
    # read_article and resolve_compound data are intentionally copied intact.
    summary = {"receipt_index": index, "receipt_reference": f"tools.jsonl#L{index}",
               "tool": tool, "arguments": deepcopy(row["arguments"]),
               "keyword_arguments": deepcopy(row["keyword_arguments"]),
               "status": viewed["status"], "source_ids": [source["source_id"] for source in viewed.get("sources", [])],
               "evidence_kind": "already_seen_article_excerpt" if tool == "read_article" else
                                "search_discovery_only" if tool == "search_literature" else "recorded_tool_evidence",
               "data": data}
    if viewed.get("message"):
        summary["message"] = viewed["message"]
    return summary, deepcopy(viewed.get("sources", []))


def build_evidence_packet(stage_dir: Path) -> dict[str, Any]:
    """Build <=100k UTF-8 bytes of prior evidence or fail without losing excerpts.

The count uses ``json.dumps(packet, ensure_ascii=False)``. The complete identity
views, all article model views and all source records are indivisible. If these
cannot fit, callers must stop rather than silently weaken the evidence packet.
"""
    path = Path(stage_dir) / "tools.jsonl"
    if not path.is_file():
        raise ValueError("Saved research tools.jsonl is required")
    receipts = []
    sources: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        summary, records = _receipt(line, index)
        receipts.append(summary)
        for source in records:
            identifier = source["source_id"]
            if identifier in sources and sources[identifier] != source:
                raise ValueError(f"Conflicting authentic source records for {identifier}")
            sources[identifier] = source
    if not receipts:
        raise ValueError("Saved research tools.jsonl contains no receipts")
    packet = {"schema_version": "research-evidence-packet-v1",
              "receipt_indexing": "1-based physical tools.jsonl line numbers",
              "notes": ["This packet contains only already-recorded evidence; no retrieval or model call was made.",
                        "Article reading pages/outlines (or historical abstract/body_windows) are exactly the previously shown model_result, not unseen raw body or XML.",
                        "Full-text retrieval does not imply complete-paper model reading; excerpts may omit important context.",
                        "Search results establish discovery only. Identity and article views are not silently shortened."],
              "receipts": receipts, "sources": list(sources.values())}
    if _size(packet) > PACKET_MAX_BYTES:
        # Search bibliographies and annotation context are optional synthesis
        # aids. Reduce them before refusing an oversized packet. Keep actual
        # article evidence, identity, exact query arguments and Sources intact.
        for receipt in receipts:
            if receipt["tool"] == "search_literature":
                data = receipt["data"]
                receipt["data"] = {"query": data["query"], "hit_count": data["hit_count"],
                    "model_view_result_count": data["model_view_result_count"],
                    "article_ids": [{key: value for key, value in item.items() if key != "title"}
                                    for item in data["articles"]],
                    "omission_note": "Discovery titles omitted for size; this remains search-only evidence."}
            elif receipt["tool"] == "get_pubchem_annotations":
                receipt["data"] = {"omission_note": "Optional annotation context omitted for packet size; full receipt remains on disk."}
    if _size(packet) > PACKET_MAX_BYTES:
        raise ValueError("Evidence packet exceeds 100000 UTF-8 bytes; preserving exact identity/article/mechanism views and source records requires an explicit smaller evidence selection")
    return packet
