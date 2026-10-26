from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


def probe_module():
    pytest.importorskip("sklearn")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_full_biomed_moa_probe
    return run_full_biomed_moa_probe


def test_declared_features_keep_native_pre_and_post_dimensions(tmp_path):
    probe = probe_module()
    specs = [{"key": "biomed2560", "stage": "raw", "dimension": 2560},
             {"key": "native512", "stage": "embedding", "dimension": 512},
             {"key": "bottleneck_seed42__hidden160", "stage": "hidden", "dimension": 160}]
    probe.write_json(tmp_path / "features.json", specs)
    assert probe.feature_specs(tmp_path) == specs
    probe.write_json(tmp_path / "features.json", specs + [dict(specs[0])])
    with pytest.raises(ValueError, match="Duplicate"):
        probe.feature_specs(tmp_path)
    for key in ("source_only", "prevalence", "dino_pca128", "dino_plus_matched"):
        probe.write_json(tmp_path / "features.json", [{"key": key, "stage": "raw", "dimension": 10}])
        with pytest.raises(ValueError, match="Reserved"):
            probe.feature_specs(tmp_path)
    probe.write_json(tmp_path / "features.json", [{"key": "native512", "stage": "embedding", "dimension": True}])
    with pytest.raises(ValueError, match="positive integers"):
        probe.feature_specs(tmp_path)


def test_readout_scaler_fits_training_rows_and_reload_is_exact(tmp_path):
    probe = probe_module()
    rng = np.random.default_rng(71)
    train = {"native512": rng.normal(size=(18, 7))}
    val = {"native512": 10 + rng.normal(size=(6, 7))}
    y_train = (np.arange(18) % 2)[:, None]
    y_val = (np.arange(6) % 2)[:, None]
    sources, val_sources = np.asarray(["s"] * 18), np.asarray(["s"] * 6)
    spec = {"key": "native512", "stage": "embedding", "dimension": 7}
    with probe.threadpool_limits(limits=1):
        model, audit = probe.fit_full_biomed_readout(spec, train, sources, y_train,
            val, val_sources, y_val, [0.01, 1.0])
        expected = probe.predict_dimension_saved(model, val, val_sources)
    np.testing.assert_allclose(model["blocks"][0]["scaler"].mean_, train["native512"].mean(0))
    assert model["blocks"][0]["pca"] is None
    assert audit["preprocessing"][0]["fit_split"] == "train"
    assert audit["preprocessing"][0]["n_fit_rows"] == 18
    path = tmp_path / "readout.joblib"
    probe.joblib.dump(model, path)
    actual = probe.predict_dimension_saved(probe.joblib.load(path), val, val_sources)
    np.testing.assert_array_equal(actual, expected)


def test_export_rejects_missing_binding_and_changed_feature_archive(tmp_path):
    probe = probe_module()
    probe.write_json(tmp_path / "binding.json", {"complete_frozen_towers": True})
    for name in probe.REQUIRED_EXPORTS:
        (tmp_path / name).write_bytes(b"feature-identity")
    complete = {"binding_sha256": probe.hash_file(tmp_path / "binding.json"),
                "moa_labels_used_for_export": False,
                "artifacts": {name: probe.hash_file(tmp_path / name) for name in probe.REQUIRED_EXPORTS}}
    probe.write_json(tmp_path / "complete.json", complete)
    assert probe.validate_export(tmp_path) == complete
    del complete["artifacts"]["samples.json"]
    probe.write_json(tmp_path / "complete.json", complete)
    with pytest.raises(ValueError, match="every required"):
        probe.validate_export(tmp_path)
    complete["artifacts"]["samples.json"] = probe.hash_file(tmp_path / "samples.json")
    probe.write_json(tmp_path / "complete.json", complete)
    (tmp_path / "primary_store.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        probe.validate_export(tmp_path)


def test_readout_selection_seal_blocks_model_or_selection_mutation(tmp_path):
    probe = probe_module()
    identity = {"protocol_sha256": "bound-protocol"}
    (tmp_path / "models").mkdir()
    selected = []
    for index in range(2):
        path = tmp_path / "models" / f"readout{index}.joblib"
        path.write_bytes(f"fitted-model-{index}".encode())
        selected.append({"id": f"primary__r{index}", "identity": identity,
                         "model_path": str(path.relative_to(tmp_path)),
                         "model_sha256": probe.hash_file(path), "test_used_for_model_selection": False})
    probe.seal_selection(tmp_path, identity, selected)
    loaded, seal = probe.load_sealed_selection(tmp_path, identity)
    assert loaded == selected and seal["n_readouts"] == 2
    (tmp_path / selected[0]["model_path"]).write_bytes(b"refit-after-test")
    with pytest.raises(ValueError, match="after selection seal"):
        probe.load_sealed_selection(tmp_path, identity)
    (tmp_path / selected[0]["model_path"]).write_bytes(b"fitted-model-0")
    selection = probe.read_json(tmp_path / "selection.json")
    selection["runs"].pop()
    probe.write_json(tmp_path / "selection.json", selection)
    with pytest.raises(ValueError, match="seal changed"):
        probe.load_sealed_selection(tmp_path, identity)
