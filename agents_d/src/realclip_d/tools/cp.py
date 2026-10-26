"""Local CellProfiler measurements. Values are exposed without normalization."""
from __future__ import annotations

import csv
from difflib import SequenceMatcher
import math
import hashlib
import json
from pathlib import Path

from ..contracts import CPMetadata, CPRow, ToolResult


_COUNT_NAME_CAUTION = (
    "Names containing Number, Object_Number, or Count do not by themselves establish a cell count. "
    "Use supplied feature definitions, scope, units and preprocessing; an undefined name is not a count measurement."
)
_DISCOVERY_NOTE = (
    "Navigation metadata only: no suggested query has been executed. Candidate names and groups are not "
    "measurement evidence and contain no measurement values or feature IDs. Query an actual candidate before assessing CP evidence."
)


def _known(value: str | None) -> bool:
    return bool(value and value.strip().casefold() not in
                {"", "unknown", "unspecified", "unverified", "not verified", "none", "null", "n/a"})


class CPTable:
    def __init__(self, path: Path | None, *, preprocessing: str | None = None,
                 metadata: CPMetadata | None = None):
        self.path = Path(path) if path is not None else None
        self.preprocessing = preprocessing
        self.metadata = metadata or CPMetadata()
        self.rows: list[CPRow] = []
        if self.path is None:
            return
        seen_ids: set[str] = set()
        with self.path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = set(CPRow.model_fields)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("CP CSV requires columns: " + ", ".join(sorted(required)))
            for number, record in enumerate(reader, start=2):
                if None in record:
                    raise ValueError(f"Malformed CP CSV row {number}")
                values = {key: record[key] for key in required}
                raw = values["value"].strip()
                if raw.lower() in {"", "na", "nan", "none", "null", "unknown"}:
                    values["value"] = None
                else:
                    try:
                        value = float(raw)
                    except ValueError as exc:
                        raise ValueError(f"Invalid CP value on CSV row {number}") from exc
                    values["value"] = value if math.isfinite(value) else None
                row = CPRow.model_validate(values)
                if not row.feature_id.strip() or not row.feature_name.strip():
                    raise ValueError(f"CP CSV row {number} needs feature identity")
                if row.feature_id in seen_ids:
                    raise ValueError(f"Duplicate CP feature_id on CSV row {number}: {row.feature_id}")
                seen_ids.add(row.feature_id)
                self.rows.append(row)
        keys = [(row.feature_name, row.compartment, row.channel, row.family, row.scope, row.role)
                for row in self.rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate CP measurement identity within a role; supply distinct measurement scopes")
        names = {row.feature_name for row in self.rows}
        if any(row.feature_name not in names for row in self.metadata.features):
            raise ValueError("CP feature definition does not match a supplied measurement")

    @staticmethod
    def measurement_key(row: CPRow) -> tuple[str, ...]:
        return row.feature_name, row.compartment, row.channel, row.family, row.scope

    def comparisons(self) -> list[dict]:
        """Keep both roles together, including explicit missing counterparts."""
        groups: dict[tuple[str, ...], dict[str, CPRow]] = {}
        definitions = {row.feature_name: row.model_dump(mode="json") for row in self.metadata.features}
        for row in self.rows:
            groups.setdefault(self.measurement_key(row), {})[row.role] = row
        result = []
        for key, roles in sorted(groups.items()):
            treatment, control = roles.get("treatment"), roles.get("control")
            gaps = []
            if treatment is None or control is None:
                gaps.append("A treatment or control counterpart is missing for this measurement and scope.")
            elif treatment.value is None or control.value is None:
                gaps.append("A treatment or control value is missing or nonfinite.")
            if treatment and control and treatment.unit != control.unit:
                gaps.append("Treatment and control units do not match.")
            if any(row.unit.strip().lower() in {"", "unknown"} for row in roles.values()):
                gaps.append("Measurement units are unknown.")
            if key[4].strip().lower() in {"", "unknown"}:
                gaps.append("Measurement scope is unknown.")
            if not _known(self.metadata.comparison_context):
                gaps.append("Shared processing/acquisition comparison context is unspecified.")
            if not _known(self.metadata.aggregation_method):
                gaps.append("Measurement aggregation method is unspecified.")
            if key[0] not in definitions or not _known(definitions[key[0]]["definition"]):
                gaps.append("Feature definition is unspecified; do not infer its biological meaning from the name alone.")
            if self.metadata.value_kind == "unknown":
                gaps.append("Whether values are raw or preprocessed is unspecified.")
            if self.metadata.value_kind == "preprocessed":
                if not _known(self.metadata.normalization_reference):
                    gaps.append("Normalization reference is unspecified.")
                if self.metadata.within_feature_order_preserved is not True:
                    gaps.append("Preservation of within-feature order is not confirmed.")
            if self.metadata.within_feature_order_preserved is False:
                gaps.append("Processing does not preserve within-feature order.")
            numeric_order = None
            if treatment and control and treatment.value is not None and control.value is not None:
                numeric_order = ("equal" if treatment.value == control.value else
                                 "treatment_higher" if treatment.value > control.value else "treatment_lower")
            result.append({"comparison_id": "cp_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20],
                **dict(zip(("feature_name", "compartment", "channel", "family", "scope"), key)),
                "definition": definitions.get(key[0]),
                "treatment_feature_id": treatment.feature_id if treatment else None,
                "control_feature_id": control.feature_id if control else None,
                "numeric_order": numeric_order, "comparison_ready": not gaps,
                "interpretation_gaps": gaps,
                "interpretation_limit": "Numeric order is not statistical significance, a fold change, a physical effect size, or a particular cell's appearance."})
        return result

    def catalogue(self) -> dict:
        """Actual group metadata only; catalogue exposure does not expose values/IDs."""
        groups: dict[tuple[str, str, str], list[CPRow]] = {}
        for row in self.rows:
            groups.setdefault((row.compartment, row.channel, row.family), []).append(row)
        entries = [{"compartment": key[0], "channel": key[1], "family": key[2],
                    "roles": sorted({row.role for row in rows}), "rows": len(rows),
                    "units": sorted({row.unit for row in rows}), "scopes": sorted({row.scope for row in rows})}
                   for key, rows in sorted(groups.items())]
        return {"available": self.path is not None, "total_rows": len(self.rows),
                "groups": entries[:200], "total_groups": len(entries),
                "groups_omitted": max(0, len(entries) - 200),
                "input_preprocessing": self.preprocessing or "unknown",
                "tool_transform": "none",
                "measurement_metadata": self.metadata.model_dump(mode="json", exclude={"features"}),
                "filter_dimensions": {key: self._bounded_values(key) for key in ("compartment", "channel", "family")},
                "count_name_caution": _COUNT_NAME_CAUTION,
                "pagination_unit": "feature_comparisons",
                "note": "Catalogue contains group metadata, not measurement evidence; query_cp exposes rows."}

    def _bounded_values(self, key: str, limit: int = 200) -> dict:
        values = sorted({getattr(row, key) for row in self.rows})
        return {"values": values[:limit], "total": len(values), "omitted": max(0, len(values) - limit)}

    def _query_navigation(self, filters: dict[str, str], grouped: list[dict],
                          query: str, feature_names: list[str] | None) -> dict:
        """Suggest existing names/groups without exposing their values or evidence IDs.

        Suggestions never change the query or the evidence exposure set. A mismatch
        in an exact name or one dimension must remain an empty measurement result.
        """
        groups: dict[tuple[str, str, str], set[str]] = {}
        for row in self.rows:
            groups.setdefault((row.compartment, row.channel, row.family), set()).add(row.feature_name)
        keys = ("compartment", "channel", "family")
        # Prefer the requested family when a guessed compartment/family combination
        # does not exist. These weights affect navigation order only, never filtering.
        weights = {"family": 3, "compartment": 2, "channel": 1}
        def rank(key):
            return -sum(weights[dimension] for dimension, value in zip(keys, key)
                        if filters[dimension].casefold() != "all"
                        and value.casefold() == filters[dimension].casefold()), key
        alternative_groups = []
        for key in sorted(groups, key=rank)[:8]:
            names = sorted(groups[key])
            dimensions = dict(zip(keys, key))
            alternative_groups.append({**dimensions,
                "changed_filters": [dimension for dimension, value in dimensions.items()
                                    if filters[dimension].casefold() != "all"
                                    and filters[dimension].casefold() != value.casefold()],
                "example_feature_names": names[:3], "total_feature_names": len(names),
                "suggested_query": dimensions})
        # When dimensions exist, discover within them; otherwise offer names from
        # the explicitly labelled alternative groups, never silently a wider query.
        if grouped:
            names = sorted({row["feature_name"] for row in grouped})
            candidate_scope = "groups_matching_requested_dimensions"
        else:
            chosen_keys = [tuple(group[key] for key in keys) for group in alternative_groups]
            names = sorted({name for key in chosen_keys for name in groups[key]})
            candidate_scope = "alternative_groups_shown_above"
        hints = [value.casefold() for value in ([query] if query else []) + (feature_names or [])]
        if hints:
            def name_rank(name):
                contained, similarity = max((int(hint in name.casefold()),
                    SequenceMatcher(None, hint, name.casefold()).ratio()) for hint in hints)
                return -contained, -similarity, name
            names.sort(key=name_rank)
        return {"note": _DISCOVERY_NOTE, "available_group_count": len(groups),
                "suggested_groups": alternative_groups,
                "suggested_groups_omitted": max(0, len(groups) - len(alternative_groups)),
                "candidate_feature_names": names[:20], "candidate_name_scope": candidate_scope,
                "candidate_feature_names_total": len(names), "candidate_feature_names_omitted": max(0, len(names) - 20),
                "count_name_caution": _COUNT_NAME_CAUTION}

    def query(self, compartment: str = "all", channel: str = "all", family: str = "all",
              offset: int = 0, limit: int = 40, query: str = "",
              feature_names: list[str] | None = None) -> ToolResult:
        if self.path is None:
            return ToolResult(status="unavailable", data={"rows": [], "comparisons": [], "total": 0,
                "query_diagnostic": {"outcome": "no_cp_file", "measurements_returned": False,
                    "reason": "No CellProfiler CSV was supplied. This is different from a query with no matching feature.",
                    "assessment_guidance": "Use cp_assessment.status='not_available' with cp_feature_ids=[]; no CP CSV was supplied."}},
                message="No CellProfiler CSV was supplied")
        if offset < 0 or not 1 <= limit <= 200:
            return ToolResult(status="error", message="offset must be >=0 and limit must be 1..200")
        filters = {"compartment": compartment, "channel": channel, "family": family}
        grouped = [row for row in self.comparisons() if all(
            value.lower() == "all" or row[key].casefold() == value.casefold()
            for key, value in filters.items())]
        selected = [row for row in grouped if
            (not query or query.casefold() in row["feature_name"].casefold()) and
            (feature_names is None or row["feature_name"] in feature_names)]
        page = selected[offset:offset+limit]
        by_id = {row.feature_id: row for row in self.rows}
        def rows_for(comparisons):
            return [by_id[c[key]] for c in comparisons for key in ("treatment_feature_id", "control_feature_id")
                    if c[key] is not None]
        selected_rows, page_rows = rows_for(selected), rows_for(page)
        if page:
            outcome, reason = "measurements_returned", "Measurement evidence is limited to the returned rows and comparisons."
        elif selected:
            outcome, reason = "offset_out_of_range", "The query matches measurements, but this page offset is beyond the result set; no measurement evidence was returned on this page."
        elif feature_names == []:
            outcome, reason = "empty_feature_selection", "feature_names=[] explicitly selects no features. This empty selection does not establish that any applicable measurement is missing."
        elif not self.rows:
            outcome, reason = "empty_table", "A CellProfiler CSV was supplied but contains no measurement rows."
        elif not grouped:
            outcome, reason = "group_not_found", "CellProfiler data exists, but no group matches all requested compartment/channel/family filters. The query was not broadened."
        else:
            outcome, reason = "feature_filter_no_match", "The requested compartment/channel/family group exists, but the feature-name filters match no measurement. The query was not broadened."
        diagnostic = {"outcome": outcome, "reason": reason, "measurements_returned": bool(page),
            "applied_filters": {**filters, "query": query, "feature_names": feature_names},
            "count_name_caution": _COUNT_NAME_CAUTION}
        if outcome == "offset_out_of_range":
            diagnostic["suggested_offset"] = ((len(selected) - 1) // limit) * limit
            diagnostic["assessment_guidance"] = (
                "Retry at a valid offset to obtain measurement evidence. This query matched features, so this empty page "
                "does not justify no_matching_measurement. Do not claim a CP assessment without returned measurements."
            )
        elif outcome == "empty_feature_selection":
            diagnostic["navigation"] = self._query_navigation(filters, grouped, query, feature_names)
            diagnostic["assessment_guidance"] = (
                "Retry with feature_names=null to discover actual features, or select a nonempty list of actual names. "
                "The caller selected an empty set, so this result does not justify no_matching_measurement. "
                "Do not claim a CP assessment without returned measurements."
            )
        elif not page:
            diagnostic["navigation"] = self._query_navigation(filters, grouped, query, feature_names)
            diagnostic["assessment_guidance"] = (
                "Use actual navigation candidates for any follow-up query. If no applicable measurement is obtained, "
                "use cp_assessment.status='no_matching_measurement' with cp_feature_ids=[] and state the failed query; "
                "do not use not_informative without an assessed measurement, or not_available when a CP CSV was supplied."
            )
        return ToolResult(status="ok" if selected else "not_found", data={
            "rows": [row.model_dump(mode="json") for row in page_rows],
            "comparisons": page, "pagination_unit": "feature_comparisons",
            "total_rows": len(selected_rows),
            "total": len(selected), "offset": offset,
            "next_offset": offset + limit if offset + limit < len(selected) else None,
            "tool_transform": "none", "input_preprocessing": self.preprocessing or "unknown",
            "measurement_metadata": self.metadata.model_dump(mode="json", exclude={"features"}),
            "role_totals": {role: sum(row.role == role for row in selected_rows) for role in ("treatment", "control")},
            "returned_role_totals": {role: sum(row.role == role for row in page_rows) for role in ("treatment", "control")},
            "measurement_scope": "As explicitly supplied in each row; unknown values remain null",
            "query_diagnostic": diagnostic,
        }, message=reason if not page else None)
