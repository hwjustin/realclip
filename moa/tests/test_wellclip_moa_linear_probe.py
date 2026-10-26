"""Fit boundaries and controlled feature transforms for the WellCLIP MoA study."""
from pathlib import Path
import sys

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CLIP/scripts"))
from run_moa_linear_probe import fit_and_select, fit_design, nested_subsets, onehot, predict_saved, transform_design


def test_scaler_uses_training_rows_and_fusion_equal_energy():
    rng = np.random.default_rng(3)
    store = {"dino": rng.normal(size=(20, 10)), "matched_seed42": rng.normal(size=(20, 4))}
    source = np.repeat("source_a", 20)
    design, model = fit_design("dino_plus_matched_seed42", store, source)
    assert np.allclose(np.mean(design[:, :10] ** 2) * 10, .5)
    assert np.allclose(np.mean(design[:, 10:] ** 2) * 4, .5)
    means = model["blocks"][0]["scaler"].mean_.copy()
    changed = {key: value + 10000 for key, value in store.items()}
    transform_design(model, changed, source)
    np.testing.assert_array_equal(model["blocks"][0]["scaler"].mean_, means)


def test_pca_adapts_to_small_training_set_and_unseen_source_is_zero():
    rng = np.random.default_rng(2)
    _, model = fit_design("dino_pca128", {"dino": rng.normal(size=(8, 20))}, np.repeat("a", 8))
    assert model["blocks"][0]["dimension"] == 7
    assert onehot(np.array(["unseen"]), ["a", "b"]).sum() == 0


def test_nested_subsets_keep_train_support_and_share_membership():
    y = np.zeros((40, 2), dtype=int)
    y[:15, 0] = 1
    y[15:30, 1] = 1
    rows = nested_subsets(np.arange(40) + 100, y, [42], [.25, .5])
    assert set(rows[0]["indices"]) <= set(rows[1]["indices"])
    assert rows[0]["indices"].min() >= 100
    selected = y[rows[0]["indices"] - 100]
    assert np.all(selected.sum(axis=0) >= 2)


def test_exact_tie_selects_smallest_c_and_saved_models_load_without_custom_class(tmp_path):
    # Constant validation features force equal rankings and AP for each C.
    train = {"dino": np.array([[-2.], [-1.], [1.], [2.]])}
    val = {"dino": np.zeros((2, 1))}
    model, audit = fit_and_select("dino", train, np.repeat("a", 4), np.array([[0], [0], [1], [1]]),
                                  val, np.repeat("a", 2), np.array([[0], [1]]), [1., .01])
    assert audit["selected_C"] == .01
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    reloaded = joblib.load(path)
    np.testing.assert_allclose(predict_saved(model, val, np.repeat("a", 2)),
                               predict_saved(reloaded, val, np.repeat("a", 2)))
