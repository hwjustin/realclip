"""Contract tests using official-shaped responses and no external network."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from realclip_d.contracts import CompoundIdentity
from realclip_d.tools import CPTable, CachedHTTP, ChEMBLClient, EuropePMCClient, PubChemClient, TransportError


KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
OTHER_KEY = "RYYVLZVUVIJVGH-UHFFFAOYSA-N"


def transport(tmp_path, handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return CachedHTTP(tmp_path / "cache", client=client, sleep=lambda _: None, **kwargs)


def test_pubchem_exact_structure_and_provenance(tmp_path):
    requests = []

    def respond(request):
        requests.append(request)
        if "synonyms" in request.url.path:
            return httpx.Response(200, json={"InformationList": {"Information": [{"CID": 2244, "Synonym": ["aspirin"]}]}})
        return httpx.Response(200, json={"PropertyTable": {"Properties": [{"CID": 2244, "InChIKey": KEY, "Title": "Aspirin"}]}})

    client = PubChemClient(transport(tmp_path, respond))
    result = client.resolve_compound(CompoundIdentity(inchikey=KEY, pubchem_cid=2244))
    assert result.status == "ok"
    assert result.data["identity_verified"]
    assert result.data["synonyms"] == ["aspirin"]
    assert len(requests) == len(result.sources) == 2
    assert result.sources == client.sources
    snapshot_path = Path(result.sources[0].cache_ref)
    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot["body_sha256"]
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == snapshot_path.stem


@pytest.mark.parametrize("identity,properties", [
    (CompoundIdentity(name="ambiguous"), [{"CID": 1, "InChIKey": KEY}, {"CID": 2, "InChIKey": OTHER_KEY}]),
    (CompoundIdentity(inchikey=KEY), [{"CID": 2, "InChIKey": OTHER_KEY}]),
    (CompoundIdentity(inchikey=KEY, pubchem_cid=7), [{"CID": 2, "InChIKey": KEY}]),
])
def test_pubchem_never_silently_chooses_candidate(tmp_path, identity, properties):
    client = PubChemClient(transport(tmp_path, lambda _: httpx.Response(200, json={"PropertyTable": {"Properties": properties}})))
    assert client.resolve_compound(identity).status == "ambiguous"


def test_pug_view_keeps_annotations_and_references(tmp_path):
    payload = {"Record": {"Section": [{"TOCHeading": "Pharmacology", "Information": [{"ReferenceNumber": 1}]}], "Reference": [{"ReferenceNumber": 1, "SourceName": "Example"}]}}
    client = PubChemClient(transport(tmp_path, lambda _: httpx.Response(200, json=payload)))
    result = client.get_annotations(2244)
    assert result.status == "ok"
    assert result.data["record"] == payload["Record"]
    assert "heading=" in result.sources[0].url


def test_unexpected_api_payload_preserves_source_receipt(tmp_path):
    client = PubChemClient(transport(tmp_path, lambda _: httpx.Response(200, json={"PropertyTable": {"Properties": [{"Title": "Missing CID"}]}})))
    result = client.resolve_compound(CompoundIdentity(name="example"))
    assert result.status == "error" and len(result.sources) == 1
    assert "KeyError" in result.message


def test_chembl_parent_scope_pagination_and_linked_evidence(tmp_path):
    seen = []

    def respond(request):
        seen.append(str(request.url))
        path = request.url.path
        if path.endswith("molecule.json"):
            return httpx.Response(200, json={"molecules": [{"molecule_chembl_id": "CHEMBL1", "molecule_structures": {"standard_inchi_key": KEY}, "molecule_hierarchy": {"parent_chembl_id": "CHEMBL2"}}]})
        if path.endswith("mechanism.json"):
            assert request.url.params["molecule_chembl_id"] == "CHEMBL2"
            return httpx.Response(200, json={"mechanisms": [{"mechanism_of_action": "example", "target_chembl_id": "CHEMBL20"}]})
        if path.endswith("activity.json"):
            if request.url.params.get("offset") == "1":
                return httpx.Response(200, json={"activities": [{"activity_id": 2, "assay_chembl_id": "CHEMBL10", "document_chembl_id": "CHEMBL30"}], "page_meta": {"next": None}})
            assert request.url.params["molecule_chembl_id"] == "CHEMBL1"
            return httpx.Response(200, json={"activities": [{"activity_id": 1}], "page_meta": {"next": "/chembl/api/data/activity.json?molecule_chembl_id=CHEMBL1&offset=1"}})
        if "/assay/" in path:
            return httpx.Response(200, json={"assay_chembl_id": "CHEMBL10", "description": "binding assay"})
        if "/target/" in path:
            return httpx.Response(200, json={"target_chembl_id": "CHEMBL20"})
        if "/document/" in path:
            return httpx.Response(200, json={"document_chembl_id": "CHEMBL30"})
        raise AssertionError(str(request.url))

    result = ChEMBLClient(transport(tmp_path, respond)).get_evidence(inchikey=KEY)
    assert result.status == "ok"
    assert result.data["mechanism_scope"] == "parent_aggregated"
    assert result.data["activity_scope"] == "exact_molecule"
    assert [row["activity_id"] for row in result.data["activities"]] == [1, 2]
    assert len(result.data["assays"]) == len(result.data["targets"]) == len(result.data["documents"]) == 1
    assert result.data["identity_verified"]


def test_chembl_pagination_rejects_external_host(tmp_path):
    def respond(request):
        return httpx.Response(200, json={"molecules": [], "page_meta": {"next": "https://untrusted.example/query"}})
    result = ChEMBLClient(transport(tmp_path, respond)).get_evidence(inchikey=KEY)
    assert result.status == "error"


def test_chembl_conflicting_key_and_id_are_explicit(tmp_path):
    client = ChEMBLClient(transport(tmp_path, lambda _: httpx.Response(200, json={"molecule_chembl_id": "CHEMBL1", "molecule_structures": {"standard_inchi_key": OTHER_KEY}})))
    result = client.get_evidence(inchikey=KEY, chembl_id="CHEMBL1")
    assert result.status == "ambiguous" and result.data["identity_verified"] is False
    assert "InChIKey" in result.message


def test_chembl_checks_returned_molecule_id(tmp_path):
    client = ChEMBLClient(transport(tmp_path, lambda _: httpx.Response(200, json={"molecule_chembl_id": "CHEMBL2", "molecule_structures": {"standard_inchi_key": KEY}})))
    result = client.get_evidence(inchikey=KEY, chembl_id="CHEMBL1")
    assert result.status == "ambiguous" and "ChEMBL ID" in result.message


@pytest.mark.parametrize("status,expected", [(404, "unavailable"), (200, "accessible")])
def test_article_abstract_is_distinguished_from_fulltext(tmp_path, status, expected):
    def respond(request):
        if request.url.path.endswith("search"):
            return httpx.Response(200, json={"hitCount": 1, "resultList": {"result": [{"id": "1", "source": "MED", "pmcid": "PMC1", "abstractText": "Abstract only"}]}})
        return httpx.Response(status, content=b"<article><body><sec><p>Full body text</p></sec></body></article>")
    result = EuropePMCClient(transport(tmp_path, respond)).read_article("1")
    assert result.status == "ok"
    assert result.data["abstract"] == "Abstract only"
    assert result.data["fulltext_status"] == expected
    assert (result.data["body_text"] == "Full body text") == (expected == "accessible")


def test_no_literature_is_a_normal_result(tmp_path):
    client = EuropePMCClient(transport(tmp_path, lambda _: httpx.Response(200, json={"hitCount": 0, "resultList": {"result": []}})))
    assert client.search("missing compound").status == "not_found"
    assert client.read_article("missing").status == "not_found"


def test_rate_limit_retry_after_raw_cache_and_offline_replay(tmp_path):
    count = 0
    pauses = []

    def respond(request):
        nonlocal count
        count += 1
        return httpx.Response(429 if count == 1 else 200, headers={"Retry-After": "2"}, json={"ok": count > 1})

    client = httpx.Client(transport=httpx.MockTransport(respond))
    http = CachedHTTP(tmp_path, client=client, sleep=pauses.append, monotonic=lambda: 10.0)
    first = http.get("https://pubchem.ncbi.nlm.nih.gov/example")
    assert first.status_code == 200 and count == 2
    assert 2.0 in pauses and 0.5 in pauses
    replay = CachedHTTP(tmp_path, allow_network=False)
    replayed = replay.get("https://pubchem.ncbi.nlm.nih.gov/example")
    assert first == replayed and count == 2
    assert len(list(tmp_path.glob("*.json"))) == 3  # two immutable snapshots plus URL index
    with pytest.raises(TransportError, match="cache miss"):
        replay.get("https://pubchem.ncbi.nlm.nih.gov/uncached")
    replay.close()


def test_corrupt_cache_is_rejected(tmp_path):
    http = transport(tmp_path, lambda _: httpx.Response(200, json={"ok": True}))
    record = http.get("https://www.ebi.ac.uk/test")
    Path(record.cache_ref).write_text("{}")
    with pytest.raises(TransportError, match="digest"):
        http.get("https://www.ebi.ac.uk/test")


def test_only_official_https_hosts_and_no_redirect_following(tmp_path):
    calls = []

    def respond(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://untrusted.example/"})

    http = transport(tmp_path, respond)
    result = EuropePMCClient(http).search("example")
    assert result.status == "unavailable" and len(result.sources) == 1
    assert len(calls) == 1
    with pytest.raises(TransportError, match="official API hosts"):
        http.get("https://untrusted.example/")
    with pytest.raises(TransportError, match="official API hosts"):
        http.get("http://www.ebi.ac.uk/test")


def test_retry_is_bounded_and_large_retry_after_is_not_violated(tmp_path):
    count = 0

    def respond(request):
        nonlocal count
        count += 1
        return httpx.Response(429, headers={"Retry-After": "120"})

    http = transport(tmp_path, respond)
    result = EuropePMCClient(http).search("example")
    assert result.status == "rate_limited" and count == 1


def cp_file(path, rows):
    fields = ["feature_id", "feature_name", "compartment", "channel", "family", "role", "value", "unit", "scope"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_cp_query_keeps_raw_values_scope_roles_and_unknowns(tmp_path):
    base = dict(feature_name="AreaShape_Area", compartment="Nuclei", channel="DNA", family="AreaShape", unit="pixels^2", scope="whole_well")
    path = cp_file(tmp_path / "cp.csv", [dict(base, feature_id="treatment-area", role="treatment", value="42.5"), dict(base, feature_id="control-area", role="control", value="NaN")])
    table = CPTable(path)
    result = table.query(compartment="nuclei", channel="dna", family="AreaShape", query="Area", limit=1)
    assert result.status == "ok"
    assert result.data["total"] == 1 and result.data["next_offset"] is None
    assert result.data["rows"][0]["value"] == 42.5
    assert result.data["rows"][0]["scope"] == "whole_well"
    assert result.data["rows"][1]["value"] is None
    assert result.data["comparisons"][0]["comparison_ready"] is False
    assert table.query(channel="ER").status == "not_found"
    assert CPTable(None).query().status == "unavailable"


def test_invalid_cp_value_and_missing_file_are_reported(tmp_path):
    base = dict(feature_id="f", feature_name="Area", compartment="Nuclei", channel="DNA", family="AreaShape", role="treatment", value="broken", unit="pixels", scope="whole_well")
    with pytest.raises(ValueError, match="Invalid CP value"):
        CPTable(cp_file(tmp_path / "cp.csv", [base]))
    with pytest.raises(FileNotFoundError):
        CPTable(tmp_path / "absent.csv")
    base["value"] = "42"
    with pytest.raises(ValueError, match="Duplicate CP feature_id"):
        CPTable(cp_file(tmp_path / "duplicate.csv", [base, base]))
