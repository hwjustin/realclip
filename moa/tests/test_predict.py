from pathlib import Path
import sys

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict import feature_mapping, predict
from run_moa_linear_probe import fit_and_select, hash_file, predict_saved


def fitted(tmp_path):
    x = np.array([[-3., 1.], [-2., 2.], [-1., 1.], [1., 2.], [2., 1.], [3., 2.]])
    y = np.array([[0], [0], [0], [1], [1], [1]])
    sources = np.repeat("a", len(x))
    model, _ = fit_and_select("clip", {"clip": x}, sources, y,
                              {"clip": x}, sources, y, [.01, 1.])
    model["labels"] = ["mechanism"]
    model_path = tmp_path / "readout.joblib"
    joblib.dump(model, model_path)
    features = tmp_path / "features.npz"
    np.savez_compressed(features, sample_ids=np.array([f"s{i}" for i in range(len(x))]),
                        image_embeddings=x)
    return model_path, features, model, x, sources


def test_real_fit_save_reload_prediction_preserves_labels_rows_and_inputs(tmp_path):
    model_path, features, model, x, sources = fitted(tmp_path)
    before = [hash_file(model_path), hash_file(features)]
    output = tmp_path / "predictions.npz"
    receipt = predict(model_path, features, output, {"clip": "image_embeddings"})
    with np.load(output, allow_pickle=False) as actual:
        np.testing.assert_array_equal(actual["sample_ids"], [f"s{i}" for i in range(len(x))])
        assert actual["labels"].tolist() == ["mechanism"]
        np.testing.assert_array_equal(actual["probabilities"], predict_saved(model, {"clip": x}, sources))
    assert before == [hash_file(model_path), hash_file(features)]
    assert receipt["no_targets_read"] and receipt["no_fitting"]
    with pytest.raises(FileExistsError):
        predict(model_path, features, output, {"clip": "image_embeddings"})


@pytest.mark.parametrize("bad", [np.ones((6, 3)), np.full((6, 2), np.nan)])
def test_rejects_feature_dimension_or_finiteness_changes_before_output(tmp_path, bad):
    model_path, features, _, _, _ = fitted(tmp_path)
    np.savez(features, sample_ids=np.array([f"s{i}" for i in range(6)]), image_embeddings=bad)
    output = tmp_path / "invalid.npz"
    with pytest.raises(ValueError, match="shape/values"):
        predict(model_path, features, output, {"clip": "image_embeddings"})
    assert not output.exists()


def test_duplicate_row_ids_and_unused_mapping_are_rejected(tmp_path):
    model_path, features, _, x, _ = fitted(tmp_path)
    np.savez(features, sample_ids=np.repeat("duplicate", len(x)), image_embeddings=x)
    with pytest.raises(ValueError, match="unique"):
        predict(model_path, features, tmp_path / "bad.npz", {"clip": "image_embeddings"})
    with pytest.raises(ValueError, match="unused"):
        predict(model_path, features, tmp_path / "bad.npz", {"typo": "image_embeddings"})
    with pytest.raises(ValueError):
        feature_mapping(["clip=x", "clip=y"])


def test_prevalence_model_needs_no_feature_matrix_or_targets(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump({"kind": "prevalence", "labels": ["a", "b"],
                 "probability": np.array([.2, .7])}, model_path)
    features = tmp_path / "features.npz"
    np.savez(features, sample_ids=np.array(["new1", "new2"]))
    output = tmp_path / "p.npz"
    predict(model_path, features, output)
    with np.load(output) as result:
        np.testing.assert_array_equal(result["probabilities"], [[.2, .7], [.2, .7]])


def test_pca_model_uses_saved_transform_and_source_only_requires_sources(tmp_path):
    from run_dimension_moa_probe import fit_dimension_readout, predict_dimension_saved

    rng = np.random.default_rng(8)
    x = rng.normal(size=(10, 4))
    y = (np.arange(10) % 2)[:, None]
    sources = np.repeat("s", len(x))
    spec = {"key": "raw__pca2", "bank": "raw", "stage": "pca", "dimension": 2}
    model, _ = fit_dimension_readout(spec, {"raw": x}, sources, y,
                                    {"raw": x}, sources, y, [.1])
    model["labels"] = ["a"]
    model_path = tmp_path / "pca.joblib"
    joblib.dump(model, model_path)
    features = tmp_path / "x.npz"
    np.savez(features, sample_ids=np.array([f"s{i}" for i in range(len(x))]), raw=x)
    output = tmp_path / "pca.npz"
    predict(model_path, features, output)
    with np.load(output) as result:
        np.testing.assert_array_equal(result["probabilities"], predict_dimension_saved(model, {"raw": x}, sources))
    source_model, _ = fit_and_select("source_only", {}, sources, y, {}, sources, y, [.1])
    source_model["labels"] = ["a"]
    joblib.dump(source_model, model_path)
    with pytest.raises(ValueError, match="requires sources"):
        predict(model_path, features, tmp_path / "source.npz")
