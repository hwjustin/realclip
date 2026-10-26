"""CP discovery is not evidence, and an empty query is not absent CP data."""
import csv
import json

import pytest

from realclip_d.contracts import CPRow
from realclip_d.tools.cp import CPTable


def write_table(path, features):
    rows = [dict(feature_id=f"private_{role}_{index}", feature_name=name,
                 compartment=compartment, channel=channel, family=family,
                 role=role, value=987654.125 + index, unit="preprocessed", scope="well")
            for index, (name, compartment, channel, family) in enumerate(features)
            for role in ("treatment", "control")]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CPRow.model_fields))
        writer.writeheader()
        writer.writerows(rows)
    return CPTable(path, preprocessing="Provided well-level preprocessing; physical units unspecified.")


@pytest.fixture
def table(tmp_path):
    return write_table(tmp_path / "cp.csv", [
        ("Cells_AreaShape_Area", "Cells", "all", "AreaShape"),
        ("Cells_AreaShape_Compactness", "Cells", "all", "AreaShape"),
        ("Nuclei_Number_Object_Number", "Nuclei", "all", "Number"),
        ("Cytoplasm_Number_Object_Number", "Cytoplasm", "all", "Number"),
        ("Cells_Intensity_MeanIntensity_Mito", "Cells", "Mito", "Intensity"),
    ])


def assert_discovery_has_no_evidence(discovery):
    serialized = json.dumps(discovery)
    assert "private_" not in serialized
    assert "987654" not in serialized
    assert '"comparisons"' not in serialized
    assert '"feature_id"' not in serialized
    assert '"value"' not in serialized


def test_absent_cp_and_supplied_empty_table_are_different(tmp_path):
    absent = CPTable(None).query()
    supplied = write_table(tmp_path / "empty.csv", []).query()
    assert absent.status == "unavailable"
    assert absent.data["query_diagnostic"]["outcome"] == "no_cp_file"
    assert "status='not_available'" in absent.data["query_diagnostic"]["assessment_guidance"]
    assert supplied.status == "not_found"
    assert supplied.data["query_diagnostic"]["outcome"] == "empty_table"
    assert not supplied.data["query_diagnostic"]["measurements_returned"]
    assert "status='no_matching_measurement'" in supplied.data["query_diagnostic"]["assessment_guidance"]
    assert supplied.data["query_diagnostic"]["navigation"]["suggested_groups"] == []


def test_nonexistent_group_offers_actual_groups_without_exposing_measurements(table):
    result = table.query(compartment="Cells", family="Number")
    assert result.status == "not_found"
    assert result.data["rows"] == result.data["comparisons"] == []
    assert result.data["total"] == 0
    diagnostic = result.data["query_diagnostic"]
    assert diagnostic["outcome"] == "group_not_found"
    assert diagnostic["applied_filters"]["compartment"] == "Cells"
    assert diagnostic["applied_filters"]["family"] == "Number"
    navigation = diagnostic["navigation"]
    assert_discovery_has_no_evidence(navigation)
    assert navigation["suggested_groups"][0]["family"] == "Number"
    for group in navigation["suggested_groups"]:
        suggestion = table.query(**group["suggested_query"])
        assert suggestion.status == "ok"
        assert set(group["example_feature_names"]).issubset(
            {row["feature_name"] for row in suggestion.data["comparisons"]})
    assert "Nuclei_Number_Object_Number" in navigation["candidate_feature_names"]
    assert "no suggested query has been executed" in navigation["note"]


@pytest.mark.parametrize("kwargs", [dict(query="Aera"), dict(feature_names=["Cells_AreaShape_Aera"])])
def test_existing_group_wrong_name_stays_empty_and_discovers_only_its_names(table, kwargs):
    result = table.query(compartment="Cells", family="AreaShape", **kwargs)
    assert result.status == "not_found" and result.data["rows"] == []
    diagnostic = result.data["query_diagnostic"]
    assert diagnostic["outcome"] == "feature_filter_no_match"
    navigation = diagnostic["navigation"]
    assert set(navigation["candidate_feature_names"]) == {
        "Cells_AreaShape_Area", "Cells_AreaShape_Compactness"}
    assert navigation["candidate_name_scope"] == "groups_matching_requested_dimensions"
    assert_discovery_has_no_evidence(navigation)


@pytest.mark.parametrize("compartment", ["Cells", "absent"])
def test_empty_exact_selection_is_not_evidence_of_missing_measurements(table, compartment):
    result = table.query(compartment=compartment, family="AreaShape", feature_names=[])
    assert result.status == "not_found"
    assert result.data["rows"] == result.data["comparisons"] == []
    assert result.data["total"] == 0
    diagnostic = result.data["query_diagnostic"]
    assert diagnostic["outcome"] == "empty_feature_selection"
    assert "feature_names=null" in diagnostic["assessment_guidance"]
    assert "does not justify no_matching_measurement" in diagnostic["assessment_guidance"]
    assert_discovery_has_no_evidence(diagnostic["navigation"])
    if compartment == "Cells":
        assert table.query(compartment=compartment, family="AreaShape", feature_names=None).data["rows"]


def test_empty_selection_on_empty_table_requires_an_actual_query(tmp_path):
    table = write_table(tmp_path / "empty.csv", [])
    result = table.query(feature_names=[])
    assert result.data["query_diagnostic"]["outcome"] == "empty_feature_selection"
    assert "does not justify no_matching_measurement" in result.data["query_diagnostic"]["assessment_guidance"]
    assert table.query(feature_names=None).data["query_diagnostic"]["outcome"] == "empty_table"


def test_name_discovery_never_makes_case_insensitive_exact_names(table):
    bad = table.query(compartment="cells", family="areashape", feature_names=["cells_areashape_area"])
    assert bad.status == "not_found"
    assert bad.data["query_diagnostic"]["outcome"] == "feature_filter_no_match"
    good = table.query(compartment="cells", family="areashape", feature_names=["Cells_AreaShape_Area"])
    assert good.status == "ok" and good.data["total"] == 1


def test_out_of_range_page_is_not_a_missing_measurement_query(table):
    result = table.query(compartment="Cells", family="AreaShape", offset=2, limit=1)
    assert result.status == "ok"
    assert result.data["total"] == 2 and result.data["total_rows"] == 4
    assert result.data["rows"] == result.data["comparisons"] == []
    assert result.data["query_diagnostic"]["outcome"] == "offset_out_of_range"
    assert result.data["query_diagnostic"]["suggested_offset"] == 1
    assert not result.data["query_diagnostic"]["measurements_returned"]
    assert "navigation" not in result.data["query_diagnostic"]
    assert "does not justify no_matching_measurement" in result.data["query_diagnostic"]["assessment_guidance"]
    assert table.query(compartment="Cells", family="AreaShape", offset=1, limit=1).data["rows"]


def test_count_looking_names_are_not_given_invented_definitions(table):
    result = table.query(compartment="Cells", family="Number")
    diagnostic = result.data["query_diagnostic"]
    assert "do not by themselves establish a cell count" in diagnostic["count_name_caution"]
    assert all("Count" not in name for name in diagnostic["navigation"]["candidate_feature_names"])
    actual = table.query(compartment="Nuclei", family="Number")
    comparison = actual.data["comparisons"][0]
    assert comparison["definition"] is None and comparison["comparison_ready"] is False
    assert "Feature definition is unspecified" in " ".join(comparison["interpretation_gaps"])


def test_catalogue_lists_only_actual_bounded_dimensions_and_no_evidence(table):
    catalogue = table.catalogue()
    assert catalogue["filter_dimensions"]["compartment"] == {
        "values": ["Cells", "Cytoplasm", "Nuclei"], "total": 3, "omitted": 0}
    assert catalogue["filter_dimensions"]["family"]["values"] == ["AreaShape", "Intensity", "Number"]
    assert_discovery_has_no_evidence(catalogue)


def test_navigation_and_catalogue_limits_are_real_and_explicit(tmp_path):
    table = write_table(tmp_path / "many.csv", [
        (f"Feature_{group}_{index}", f"Compartment_{group}", "DNA", "AreaShape")
        for group in range(205) for index in range(4)])
    catalogue = table.catalogue()
    assert len(catalogue["groups"]) == 200 and catalogue["groups_omitted"] == 5
    assert catalogue["filter_dimensions"]["compartment"]["omitted"] == 5
    discovery = table.query(compartment="absent").data["query_diagnostic"]["navigation"]
    assert len(discovery["suggested_groups"]) == 8
    assert discovery["suggested_groups_omitted"] == 197
    assert len(discovery["candidate_feature_names"]) == 20
    assert discovery["candidate_feature_names_total"] == 32
    assert discovery["candidate_feature_names_omitted"] == 12
    assert all(len(group["example_feature_names"]) == 3 for group in discovery["suggested_groups"])
    assert_discovery_has_no_evidence(discovery)


def test_successful_query_preserves_pairs_values_and_preprocessing(table):
    result = table.query(compartment="Cells", family="AreaShape", limit=1)
    assert result.status == "ok"
    assert result.data["total"] == 2 and result.data["total_rows"] == 4
    assert result.data["next_offset"] == 1
    assert result.data["pagination_unit"] == "feature_comparisons"
    assert {row["role"] for row in result.data["rows"]} == {"treatment", "control"}
    assert [row["value"] for row in result.data["rows"]] == [987654.125, 987654.125]
    assert result.data["tool_transform"] == "none"
    assert result.data["input_preprocessing"] == table.preprocessing
    assert result.data["query_diagnostic"]["outcome"] == "measurements_returned"
    assert result.data["query_diagnostic"]["measurements_returned"]
    assert "navigation" not in result.data["query_diagnostic"]
