"""Protect fixed class scope and paired bootstrap semantics in the expansion."""
from pathlib import Path
import sys
import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_expanded_moa_probe import metrics, weighted_ap, bootstrap_weights, model_family


def test_absent_class_never_becomes_a_smaller_macro():
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    p = np.array([[.7, .2, .1], [.1, .7, .2], [.3, .3, .4]])
    result = metrics(y, p)
    assert result["status"] == "unsupported"
    assert result["macro_ap"] is None
    assert result["per_label_ap"] == [1., 1., None]


def test_weighted_tied_ap_matches_explicit_resampling():
    y = np.array([0, 1, 0, 1])
    p = np.array([.8, .8, .2, .1])
    weights = np.array([[2, 3, 0, 1], [0, 1, 2, 4]])
    expected = []
    for row in weights:
        indices = np.repeat(np.arange(4), row)
        expected.append(average_precision_score(y[indices], p[indices]))
    np.testing.assert_allclose(weighted_ap(y, p, weights), expected, atol=1e-15)


def test_cluster_sampling_and_version_families_remain_distinct():
    weights, count = bootstrap_weights(np.array(["a", "a", "b", "c"]), 30, 42)
    assert count == 3
    np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
    np.testing.assert_array_equal(weights[:, [0, 2, 3]].sum(axis=1), np.full(30, 3))
    assert model_family("old_matched_seed42") == "old_matched"
    assert model_family("new_matched_seed42") == "new_matched"
    assert model_family("dino_plus_new_matched_seed42") == "fusion_new_matched"
