"""Guard validation isolation, fusion scale and fixed low-label comparisons."""
from pathlib import Path
import sys
import numpy as np
import pytest

pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_moa_linear_probe import fit_design, transform_design, fit_and_select, nested_subsets


def test_holdout_transformation_does_not_change_training_statistics():
    training = {"dino": np.array([[0., 2.], [2., 4.]])}
    _, preprocessing = fit_design("dino", training, np.array(["a", "a"]))
    transform_design(preprocessing, {"dino": np.array([[1e9, -1e9]])}, np.array(["unseen"]))
    np.testing.assert_array_equal(preprocessing["blocks"][0]["scaler"].mean_, [1, 3])


def test_fusion_has_equal_expected_energy_despite_different_dimensions():
    rng = np.random.default_rng(42)
    store = {"dino": rng.normal(size=(20, 12)), "matched_seed42": rng.normal(size=(20, 3))}
    x, _ = fit_design("dino_plus_matched_seed42", store, np.array(["a"] * 20))
    np.testing.assert_allclose(np.mean((x[:, :12] ** 2).sum(axis=1)), .5)
    np.testing.assert_allclose(np.mean((x[:, 12:] ** 2).sum(axis=1)), .5)


def test_validation_tie_retains_smallest_C():
    x = {"dino": np.zeros((4, 2))}
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    _, audit = fit_and_select("dino", x, np.array(["a"] * 4), y,
                             x, np.array(["a"] * 4), y, [1., .1, .01])
    assert audit["selected_C"] == .01


def test_low_label_indices_are_nested_and_only_training_rows():
    train = np.arange(100, 200)
    y = np.zeros((100, 2), dtype=int)
    y[:20, 0] = 1
    y[10:35, 1] = 1
    result = nested_subsets(train, y, [42, 43, 44], [.25, .5])
    for small, large in zip(result[::2], result[1::2]):
        assert set(small["indices"]) <= set(large["indices"]) <= set(train)
        assert (y[small["indices"] - 100].sum(axis=0) >= 2).all()
