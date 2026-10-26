"""Checks on fixed-vocabulary metrics and stage membership for expanded MoA."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CLIP/scripts"))
from run_expanded_moa_probe import safe_metrics, stage_indices


def test_unsupported_class_is_explicit_and_full_macro_is_null():
    y = np.array([[1, 0], [0, 0], [0, 0]])
    scores = np.array([[.9, .4], [.2, .3], [.1, .1]])
    result = safe_metrics(y, scores, ["present", "absent"])
    assert result["macro_ap"] is None and result["macro_roc_auc"] is None
    assert result["unsupported_labels"] == ["absent"]
    assert result["supported_only_macro_ap"] == 1
    assert result["per_label"][1]["average_precision"] is None


def test_complete_class_vocabulary_retains_macro_metrics():
    y = np.array([[1, 0], [0, 1], [0, 0]])
    result = safe_metrics(y, y * .8 + .1, ["a", "b"])
    assert result["all_labels_supported"] and result["macro_ap"] == 1


def test_fixed306_stage_preserves_original_order_and_excludes_holdout():
    ids = np.array([f"s{i:03d}" for i in range(735)])
    splits = np.array(["train"] * 603 + ["val"] * 66 + ["test"] * 66)
    original = [{"sample_id": str(s), "split": "train"} for s in ids[:306][::-1]]
    result = stage_indices({"sample_ids": ids, "splits": splits}, original)
    np.testing.assert_array_equal(result["fixed306"], np.arange(306)[::-1])
    assert len(result["expanded603"]) == 603
    bad = original.copy()
    bad[0] = {"sample_id": "s734", "split": "train"}
    with pytest.raises(ValueError):
        stage_indices({"sample_ids": ids, "splits": splits}, bad)
