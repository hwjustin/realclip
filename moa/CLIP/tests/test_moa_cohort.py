"""Checks against alias-key collisions and repeated-parent support inflation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_moa_probe import raw_labels, support_rows, unique_lookup, well_key


def test_key_normalization_rejects_duplicate_aliases():
    assert well_key("source_1", "plate", "a1") == ("source_1", "plate", "A01")
    with pytest.raises(ValueError, match="Duplicate"):
        unique_lookup([{"s": "s", "p": "p", "w": "a1"}, {"s": "s", "p": "p", "w": "A01"}], ("s", "p", "w"))


def test_multilabel_delimiter_preserves_biological_names():
    assert raw_labels("dopamine receptor antagonist|serotonin receptor antagonist") == ["dopamine receptor antagonist", "serotonin receptor antagonist"]
    with pytest.raises(ValueError):
        raw_labels("negative_control")


def test_parent_support_does_not_count_duplicate_wells():
    samples = [{"parent_id": "p", "split": "train", "labels": ["x"]}] * 10
    samples += [{"parent_id": "q", "split": "val", "labels": ["x"]}, {"parent_id": "r", "split": "test", "labels": ["x"]}]
    row = support_rows(samples, ["x"])[0]
    assert row["train_positive_parents"] == 1
    assert not row["primary_eligible"] and not row["exploratory_eligible"]
