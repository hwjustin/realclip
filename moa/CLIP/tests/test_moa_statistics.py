"""Independent checks of paired, clustered AP uncertainty."""
from pathlib import Path
import sys
import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_moa_probe import bootstrap_weights, weighted_ap


def test_weighted_ap_matches_duplicated_rows_with_ties_and_zero_weights():
    y = np.array([1, 0, 1, 0, 0])
    score = np.array([.8, .8, .3, .2, .2])
    weights = np.array([[2, 1, 3, 0, 1], [0, 2, 1, 4, 0]], dtype=float)
    expected = []
    for row in weights:
        indices = np.repeat(np.arange(len(y)), row.astype(int))
        expected.append(average_precision_score(y[indices], score[indices]))
    np.testing.assert_allclose(weighted_ap(y, score, weights), expected, atol=1e-15)


def test_bootstrap_preserves_whole_plate_and_group_draw_count():
    weights, count = bootstrap_weights(np.array(["a", "a", "b", "c", "c"]), 100, 42)
    assert count == 3
    np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
    np.testing.assert_array_equal(weights[:, 3], weights[:, 4])
    np.testing.assert_array_equal(weights[:, [0, 2, 3]].sum(axis=1), np.full(100, 3))


def test_constant_prediction_ap_is_weighted_prevalence():
    y = np.array([1, 0, 0, 1])
    weights = np.array([[1, 2, 3, 1], [2, 0, 1, 3]], dtype=float)
    np.testing.assert_allclose(weighted_ap(y, np.ones(4), weights), weights @ y / weights.sum(axis=1))
