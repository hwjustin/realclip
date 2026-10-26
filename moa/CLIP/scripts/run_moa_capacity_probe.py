"""Train the original image-adapter architecture on MoA labels only.

This is a capacity control for the image-text adapters, not a new CLIP model.
Six train/validation choices are sealed before any test probabilities are made.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wellclip.common import hash_file, read_json, stable_hash, utc_now, write_json
from wellclip.training import load_features


class ImageMoA(nn.Module):
    """Exactly the WellCLIP image branch followed by a MoA linear readout."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.channel_projection = nn.Sequential(nn.Linear(384, 32), nn.GELU())
        self.image_projection = nn.Sequential(nn.Dropout(.1), nn.Linear(160, 128))
        self.classifier = nn.Linear(128, n_classes)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 3 or images.shape[1:] != (5, 384):
            raise ValueError("Expected complete five-channel frozen DINO features")
        hidden = self.channel_projection(images).flatten(start_dim=1)
        return F.normalize(self.image_projection(hidden), dim=-1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(images))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold-grouped average precision, including correct treatment of ties."""
    y, score = np.asarray(y, dtype=np.int64), np.asarray(score, dtype=np.float64)
    if y.ndim != 1 or score.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected aligned nonempty binary targets and scores")
    if not np.isfinite(score).all():
        raise ValueError("Scores must be finite")
    positives = int(y.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_score) != 0), len(y) - 1]
    true_positives = np.cumsum(y[order])[ends]
    increments = np.diff(np.r_[0, true_positives])
    return float(np.sum(increments * true_positives / (ends + 1)) / positives)


def macro_ap(y: np.ndarray, score: np.ndarray) -> float:
    if y.ndim != 2 or score.shape != y.shape or not y.shape[1]:
        raise ValueError("Expected nonempty aligned multilabel arrays")
    return float(np.mean([average_precision(y[:, i], score[:, i]) for i in range(y.shape[1])]))


def better_checkpoint(candidate: float, best: float) -> bool:
    # Evaluations are visited in ascending step order, preserving earliest ties.
    return candidate > best


def fit_adapter(x_train: torch.Tensor, y_train: torch.Tensor,
                x_val: torch.Tensor, y_val: np.ndarray, seed: int,
                steps: int, eval_every: int) -> tuple[ImageMoA, list[dict], dict]:
    """Train with no test arrays or test targets in this function's interface."""
    if x_train.shape[0] != y_train.shape[0] or x_val.shape[0] != len(y_val):
        raise ValueError("Features and split-specific targets differ")
    positive = y_train.sum(dim=0)
    negative = len(y_train) - positive
    if not torch.all((positive > 0) & (negative > 0)):
        raise ValueError("Every training class requires both positive and negative examples")
    torch.manual_seed(seed)
    model = ImageMoA(y_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    weights = negative / positive
    best_score, best_step, best_state = -float("inf"), None, None
    history = []
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(x_train), y_train, pos_weight=weights)
        loss.backward()
        optimizer.step()
        row = {"step": step, "training_bce": float(loss.detach())}
        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                validation_probability = torch.sigmoid(model(x_val)).numpy()
            score = macro_ap(y_val, validation_probability)
            row["validation_macro_ap"] = score
            if better_checkpoint(score, best_score):
                best_score, best_step = score, step
                best_state = copy.deepcopy(model.state_dict())
        history.append(row)
    if best_state is None:
        raise AssertionError("No validation checkpoint evaluated")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        check = macro_ap(y_val, torch.sigmoid(model(x_val)).numpy())
    assert check == best_score
    return model, history, {
        "seed": seed, "best_step": best_step, "validation_macro_ap": best_score,
        "train_positive_counts": positive.to(torch.int64).tolist(),
        "train_negative_counts": negative.to(torch.int64).tolist(),
        "positive_weights": weights.tolist(), "weight_fit_split": "train",
        "image_adapter_parameters": sum(p.numel() for name, p in model.named_parameters() if not name.startswith("classifier.")),
        "classifier_parameters": sum(p.numel() for p in model.classifier.parameters()),
        "total_trainable_parameters": sum(p.numel() for p in model.parameters()),
        "selection": "validation macro AP, earliest exact tie", "test_labels_used_for_selection": False,
    }


def run(work: Path, output: Path, threads: int = 2) -> dict:
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    manifest, features, metadata = load_features(work / "manifest.json", work / "features")
    parent = output.parent
    protocol_path, cohort_path = parent / "protocol.json", parent / "cohort_arrays.npz"
    protocol = read_json(protocol_path)
    config = protocol["capacity_baseline"]
    assert hash_file(cohort_path) == protocol["cohort_arrays_sha256"]
    assert hash_file(work / "features/features.npz") == protocol["feature_archive_sha256"]
    with np.load(cohort_path, allow_pickle=False) as archive:
        sample_ids, splits = archive["sample_ids"].copy(), archive["splits"].copy()
        labels = archive["label_names"].tolist()
        targets = archive["y_all"].copy()
        parents = archive["parent_ids"].copy()
    assert sample_ids.tolist() == features["sample_ids"].tolist()
    assert len(set(parents)) == len(parents), "This fixed protocol expects one independent parent per well"
    assert splits.tolist() == [sample["split"] for sample in manifest["samples"]]
    train_index, val_index = [np.flatnonzero(splits == split) for split in ["train", "val"]]
    x_train = torch.from_numpy(features["image_features"][train_index])
    x_val = torch.from_numpy(features["image_features"][val_index])
    identity = {"protocol_sha256": hash_file(protocol_path), "cohort_arrays_sha256": hash_file(cohort_path),
                "features_sha256": hash_file(work / "features/features.npz"),
                "script_sha256": hash_file(__file__)}
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for label_set in config["sets_trained_separately"]:
        chosen_labels = protocol[f"{label_set}_labels"]
        label_index = [labels.index(label) for label in chosen_labels]
        train_targets = torch.from_numpy(targets[np.ix_(train_index, label_index)].astype(np.float32))
        val_targets = targets[np.ix_(val_index, label_index)]
        for seed in config["seeds"]:
            name = f"{label_set}_seed{seed}"
            directory = output / name
            directory.mkdir(exist_ok=True)
            receipt_path = directory / "complete.json"
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                assert receipt["identity"] == identity, "Completed capacity run belongs to another identity"
                for filename, expected in receipt["artifacts"].items():
                    assert hash_file(directory / filename) == expected
                selection = read_json(directory / "selection.json")
            else:
                model, history, selection = fit_adapter(x_train, train_targets, x_val, val_targets, seed,
                                                        config["steps"], config["eval_every"])
                selection.update(label_set=label_set, labels=chosen_labels, training_sample_ids=sample_ids[train_index].tolist(),
                                 validation_sample_ids=sample_ids[val_index].tolist())
                torch.save({"identity": identity, "name": name, "labels": chosen_labels,
                            "state_dict": model.state_dict(), "selection": selection}, directory / "best.pt")
                write_json(directory / "history.json", history)
                write_json(directory / "selection.json", selection)
                write_json(receipt_path, {"identity": identity, "completed_at": utc_now(),
                    "artifacts": {filename: hash_file(directory / filename) for filename in ["best.pt", "history.json", "selection.json"]}})
            records.append({"name": name, "label_set": label_set, "seed": seed,
                            "labels": chosen_labels, "best_step": selection["best_step"],
                            "validation_macro_ap": selection["validation_macro_ap"],
                            "total_trainable_parameters": selection["total_trainable_parameters"],
                            "checkpoint": f"{name}/best.pt", "receipt_sha256": hash_file(receipt_path)})
            print(f"SEALED {name} step={selection['best_step']} val_AP={selection['validation_macro_ap']:.6f}", flush=True)
    assert len(records) == 6
    seal = {"identity": identity, "all_six_sealed_before_test_predictions": True,
            "completed_at": utc_now(), "runs": {record["name"]: record["receipt_sha256"] for record in records}}
    write_json(output / "selection_seal.json", seal)
    # No training or model selection occurs after this point. Test target values
    # are never accessed: the parent evaluator handles downstream test metrics.
    predictions = {"sample_ids": sample_ids, "splits": splits}
    all_images = torch.from_numpy(features["image_features"])
    for record in records:
        checkpoint = torch.load(output / record["checkpoint"], map_location="cpu", weights_only=True)
        model = ImageMoA(len(record["labels"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(all_images)).numpy()
        assert probabilities.shape == (len(sample_ids), len(record["labels"]))
        predictions[record["name"]] = probabilities
    for label_set in config["sets_trained_separately"]:
        predictions[f"{label_set}_labels"] = np.asarray(protocol[f"{label_set}_labels"])
    np.savez_compressed(output / "probabilities.npz", **predictions)
    result = {"identity": identity, "completed_at": utc_now(), "manifest_id": manifest["manifest_id"],
              "feature_id": metadata["feature_id"], "method": "Pure DINO features, original trainable image-adapter architecture, MoA-only supervision",
              "image_adapter_parameters": 32928, "test_targets_evaluated_by_this_script": False,
              "selection_seal_sha256": hash_file(output / "selection_seal.json"),
              "probability_archive": "probabilities.npz", "probability_archive_sha256": hash_file(output / "probabilities.npz"),
              "probability_schema": "sample_ids/splits align with all original 438 rows; each primary_seed42 etc. array is [438, K]; primary_labels/exploratory_labels give column order",
              "runs": records}
    write_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=root / "artifacts/prototype_v1")
    parser.add_argument("--output", type=Path, default=root / "artifacts/moa_probe_v1/capacity")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.work, args.output, args.threads)
