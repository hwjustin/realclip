"""Practical English-only boundary checks for agent-authored text.

The guard detects Han/CJK ideographs, including supplementary extensions. It is
not a language classifier and does not prove that accepted text is fluent English.
Greek letters, scientific symbols, and accented scientific names remain valid.

Only normal agent inputs and structured outputs belong here. External tool
receipts are not rewritten or checked: faithful source records, identifiers,
URLs, and file paths must remain exactly as retrieved or supplied. In particular,
the top-level output ``sources`` collection is exempt. No claim is made about a
model's private reasoning; any exposed rationale is ordinary checked output.
"""
from __future__ import annotations

import json
import re
from typing import Any


_HAN = re.compile(
    "[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "\U00020000-\U0002ee5f\U0002f800-\U0002fa1f"
    "\U00030000-\U000323af]"
)

# Explicit contract fields avoid broad '*id*' or '*path*' rules that could
# accidentally exempt narrative such as identity_summary or pathway effects.
_IMMUTABLE_FIELDS = frozenset({
    "schema_version", "sample_id", "source_id", "source_ids", "fact_id",
    "fact_ids", "hypothesis_id", "hypothesis_ids", "observation_id",
    "image_id", "image_ids", "cp_feature_id", "cp_feature_ids", "feature_id",
    "feature_name", "inchikey", "smiles", "pubchem_cid", "chembl_id",
    "path", "cp_file", "url", "image_url", "cache_ref", "retrieved_at",
    "web_retrieved_at",
})


def _field_path(parent: str, key: str) -> str:
    if key.isascii() and key.isidentifier():
        return f"{parent}.{key}"
    return f"{parent}[{json.dumps(key, ensure_ascii=True)}]"


def _validate(payload: dict[str, Any], *, output: bool) -> None:
    def walk(value: Any, path: str, *, root: bool = False) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in _IMMUTABLE_FIELDS or (output and root and key == "sources"):
                    continue
                walk(child, _field_path(path, str(key)))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str) and _HAN.search(value):
            # Deliberately report only a field path, never the offending value.
            raise ValueError(f"English-only policy violation at {path}")

    if not isinstance(payload, dict):
        raise ValueError("Agent payload must be an object at $")
    walk(payload, "$", root=True)


def validate_agent_input(payload: dict[str, Any]) -> None:
    """Reject Han/CJK in input names and narrative; retain identifiers and paths."""
    _validate(payload, output=False)


def validate_agent_output(payload: dict[str, Any]) -> None:
    """Reject Han/CJK in authored output, retaining authentic Source records."""
    _validate(payload, output=True)


def validate_model_request(payload: dict[str, Any]) -> None:
    """Check the complete model-visible request, including JSON tool returns.

    Unlike the structured narrative checks, this has no Source/identifier
    exemptions: non-English source material must not silently enter a new
    model request. Original retrieval receipts remain unchanged on disk.
    """
    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if _HAN.search(str(key)):
                    raise ValueError(f"English-only model input required at {path}")
                walk(child, _field_path(path, str(key)))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str):
            if _HAN.search(value):
                raise ValueError(f"English-only model input required at {path}")
            if value.lstrip().startswith(("{", "[")):
                try:
                    decoded = json.loads(value)
                except (ValueError, RecursionError):
                    return
                if isinstance(decoded, (dict, list)):
                    walk(decoded, path)

    walk(payload, "$")
