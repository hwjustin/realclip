"""Run the six sealed MoA adapters on new, label-free feature rows only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from run_moa_capacity_probe import ImageMoA
from wellclip.common import hash_file, read_json, utc_now, write_json


@torch.inference_mode()
def predict(capacity: Path, feature_path: Path) -> dict:
    torch.set_num_threads(2)
    results_path, seal_path = capacity / "results.json", capacity / "selection_seal.json"
    results_hash, seal_hash = hash_file(results_path), hash_file(seal_path)
    results, seal = read_json(results_path), read_json(seal_path)
    assert seal_hash == results["selection_seal_sha256"]
    assert seal["all_six_sealed_before_test_predictions"]
    with np.load(feature_path, allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].copy()
        features = archive["image_features"].copy().astype(np.float32)
    if (features.shape != (len(sample_ids), 5, 384) or not np.isfinite(features).all()
            or len(set(sample_ids.tolist())) != len(sample_ids)):
        raise ValueError("Expected finite, uniquely identified complete-well DINO features")
    arrays = {"sample_ids": sample_ids, "splits": np.repeat("test_supplemental", len(sample_ids))}
    checkpoints = {}
    for record in results["runs"]:
        name = record["name"]
        directory = capacity / name
        assert hash_file(directory / "complete.json") == seal["runs"][name]
        receipt = read_json(directory / "complete.json")
        assert receipt["identity"] == results["identity"]
        checkpoint_path = capacity / record["checkpoint"]
        assert hash_file(checkpoint_path) == receipt["artifacts"]["best.pt"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert checkpoint["labels"] == record["labels"]
        model = ImageMoA(len(record["labels"]))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        arrays[name] = torch.sigmoid(model(torch.from_numpy(features))).numpy()
        arrays[f"{record['label_set']}_labels"] = np.asarray(record["labels"])
        checkpoints[name] = hash_file(checkpoint_path)
    output_path = capacity / "supplemental_probabilities.npz"
    np.savez_compressed(output_path, **arrays)
    # Original selection and checkpoint completion records are immutable.
    assert hash_file(results_path) == results_hash and hash_file(seal_path) == seal_hash
    receipt = {"completed_at": utc_now(), "sample_count": len(sample_ids), "no_training": True,
               "no_labels_read_by_this_script": True,
               "feature_archive": str(feature_path.resolve()), "feature_archive_sha256": hash_file(feature_path),
               "original_capacity_results_sha256": results_hash, "original_selection_seal_sha256": seal_hash,
               "checkpoint_sha256": checkpoints,
               "script_sha256": hash_file(__file__),
               "probability_archive": output_path.name, "probability_archive_sha256": hash_file(output_path),
               "schema": "sample_ids/splits identify all supplemental rows; six named [n,K] probability arrays and per-label-set column names"}
    write_json(capacity / "supplemental_prediction_receipt.json", receipt)
    print(f"Predicted {len(sample_ids)} supplemental wells with all six unchanged adapters", flush=True)
    return receipt


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "artifacts/moa_probe_v1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=Path, default=root / "capacity")
    parser.add_argument("--features", type=Path, default=root / "supplemental/features/features.npz")
    args = parser.parse_args()
    predict(args.capacity, args.features)
