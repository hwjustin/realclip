"""Ensure expanded training invalidates prior test parents and entire plates."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_expanded_moa_probe import overlap_reasons


def test_exclude_whole_plate_even_if_parent_was_not_added():
    sample = {"parent_id": "new_parent", "source": "s", "plate": "p"}
    parents = {"train": {"added_parent"}, "val": set(), "test": set()}
    plates = {"train": {("s", "p")}, "val": set(), "test": set()}
    hashes = {"train": set(), "val": set(), "test": set()}
    assert overlap_reasons(sample, parents, plates, hashes, {"new_sha"}) == ["expanded_train_plate"]


def test_exclude_validation_parent_and_duplicate_image_on_different_plate():
    sample = {"parent_id": "seen", "source": "s", "plate": "new_plate"}
    parents = {"train": set(), "val": {"seen"}, "test": set()}
    plates = {"train": set(), "val": set(), "test": set()}
    hashes = {"train": {"same_sha"}, "val": set(), "test": set()}
    assert overlap_reasons(sample, parents, plates, hashes, {"same_sha"}) == ["expanded_train_raw_image_sha", "expanded_val_parent"]
