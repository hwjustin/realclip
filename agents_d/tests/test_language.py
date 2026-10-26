from copy import deepcopy

import pytest

from realclip_d.language import validate_agent_input, validate_agent_output


@pytest.mark.parametrize("field,value,expected", [
    ("compound", {"name": "\u5316\u5408\u7269"}, "$.compound.name"),
    ("known_moa", ["Kinase inhibition", "\u673a\u5236"], "$.known_moa[1]"),
    ("context", {"notes": ["\u6761\u4ef6"]}, "$.context.notes[0]"),
    ("images", [{"channel": "DNA", "field": "\u89c6\u91ce"}], "$.images[0].field"),
    ("metadata", {"narrative": "\u89e3\u91ca"}, "$.metadata.narrative"),
])
def test_input_narrative_is_checked_recursively(field, value, expected):
    with pytest.raises(ValueError) as error:
        validate_agent_input({field: value})
    assert str(error.value) == f"English-only policy violation at {expected}"


@pytest.mark.parametrize("payload,expected", [
    ({"facts": [{"statement": "\u76f4\u63a5\u89c2\u5bdf"}]}, "$.facts[0].statement"),
    ({"hypotheses": [{"rationale": "\u63a8\u6d4b"}]}, "$.hypotheses[0].rationale"),
    ({"identity_summary": "\u8eab\u4efd"}, "$.identity_summary"),
    ({"literature_review": {"search_summary": "\u68c0\u7d22"}}, "$.literature_review.search_summary"),
    ({"mechanism_review": {"summary": "\u673a\u5236"}}, "$.mechanism_review.summary"),
    ({"description": "\u5f62\u6001"}, "$.description"),
    ({"observations": [{"text": "\u89c2\u5bdf"}]}, "$.observations[0].text"),
    ({"limitations": ["\u9650\u5236"]}, "$.limitations[0]"),
    ({"hypotheses": [{"verification": {"image_checks": ["\u6838\u819c"]}}]},
     "$.hypotheses[0].verification.image_checks[0]"),
    ({"hypotheses": [{"applicable_conditions": "\u6761\u4ef6"}]},
     "$.hypotheses[0].applicable_conditions"),
    ({"hypotheses": [{"disconfirming_observation": "\u53cd\u8bc1"}]},
     "$.hypotheses[0].disconfirming_observation"),
    ({"metadata": {"sources": [{"title": "\u4e0d\u662f\u5916\u90e8\u6765\u6e90"}]}},
     "$.metadata.sources[0].title"),
])
def test_output_narrative_and_exposed_rationale_are_checked(payload, expected):
    with pytest.raises(ValueError) as error:
        validate_agent_output(payload)
    assert str(error.value) == f"English-only policy violation at {expected}"


def test_error_contains_only_field_path_not_sensitive_or_non_english_value():
    secret_value = "private-prefix-\u654f\u611f-private-suffix"
    with pytest.raises(ValueError) as error:
        validate_agent_output({"facts": [{"statement": secret_value}]})
    assert str(error.value) == "English-only policy violation at $.facts[0].statement"
    assert secret_value not in str(error.value)
    assert "private-prefix" not in str(error.value)


def test_sources_identifiers_and_paths_are_preserved_exactly():
    payload = {
        "sample_id": "\u539f\u6837\u672c",
        "compound": {"name": "Rapamycin", "inchikey": "\u539f\u6807\u8bc6",
                     "smiles": "\u539f\u7ed3\u6784", "chembl_id": "\u539f\u7f16\u53f7"},
        "images": [{"image_id": "\u56fe\u7247", "path": "/data/\u56fe\u7247.tif",
                    "field": "Field 1", "channel": "DNA"}],
        "cp_file": "/data/\u6d4b\u91cf.csv",
        "url": "https://example.org/\u6587\u732e",
        "facts": [{"source_ids": ["\u6765\u6e90"], "statement": "Cells were round."}],
        "sources": [{"source_id": "original-source", "title": "\u539f\u59cb\u6807\u9898",
                     "url": "https://example.org/\u539f\u59cb", "cache_ref": "cache/\u539f.json"}],
    }
    original = deepcopy(payload)
    validate_agent_output(payload)
    assert payload == original
    # Input is an agent payload, not an external receipt collection.
    with pytest.raises(ValueError, match=r"\$\.sources\[0\]\.title"):
        validate_agent_input(payload)


def test_scientific_symbols_and_english_text_are_allowed():
    payload = {
        "compound": {"name": "Nocodazole"},
        "known_moa": ["Microtubule destabilizer"],
        "context": {"dose": "2 \u00b5M", "notes": ["\u03b1-tubulin; 37 \u00b0C; \u2265 30 min"]},
        "hypotheses": [{"rationale": "Compare DNA intensity and nuclear area (\u00b5m\u00b2)."}],
    }
    validate_agent_input(payload)
    validate_agent_output(payload)


@pytest.mark.parametrize("text", ["\u3400", "\uf900", "\U00020000", "\U0002f800", "\U00030000"])
def test_cjk_extensions_are_detected(text):
    with pytest.raises(ValueError, match=r"\$\.description"):
        validate_agent_output({"description": text})


def test_a_leading_slash_does_not_exempt_narrative():
    with pytest.raises(ValueError, match=r"\$\.description"):
        validate_agent_output({"description": "/\u4e2d\u6587"})


def test_validator_does_not_claim_complete_language_identification():
    # This deliberately documents the practical guard's limitation.
    validate_agent_output({"description": "Les cellules sont rondes."})


@pytest.mark.parametrize("validator", [validate_agent_input, validate_agent_output])
def test_invalid_payload_shape_has_an_english_error(validator):
    with pytest.raises(ValueError, match=r"Agent payload must be an object at \$"):
        validator([])
