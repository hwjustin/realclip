"""Predict MoA from exported features using the existing fitted readout models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent / "CLIP/scripts"))
from run_moa_linear_probe import hash_file, predict_saved


def feature_mapping(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, archive_key = value.partition("=")
        if not separator or not key or not archive_key or key in result:
            raise ValueError("Use unique --feature-map MODEL_FEATURE=NPZ_KEY values")
        result[key] = archive_key
    return result


def predict(model_path: Path, features_path: Path, output: Path,
            mapping: dict[str, str] | None = None, threads: int = 2) -> dict:
    """No targets, fitting, threshold selection, or checkpoint selection."""
    if threads < 1:
        raise ValueError("threads must be positive")
    output = Path(output)
    receipt_path = output.with_suffix(".json")
    if output.suffix != ".npz":
        raise ValueError("Output must have a .npz extension")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("Choose a new output; existing predictions are preserved")
    input_hashes = {"model_sha256": hash_file(model_path),
                    "features_sha256": hash_file(features_path)}
    model = joblib.load(model_path)
    if not isinstance(model, dict) or model.get("kind") not in ("logistic", "prevalence"):
        raise ValueError("Expected an existing fitted linear-probe dictionary model")
    labels = model.get("labels")
    if (not isinstance(labels, list) or not labels or
            any(not isinstance(label, str) or not label.strip() for label in labels) or
            len(set(labels)) != len(labels)):
        raise ValueError("Model must contain unique ordered labels")
    mapping = dict(mapping or {})
    required = {b["feature"]: b["input_dimension"] for b in model.get("blocks", [])
                if b["feature"] != "source_only"}
    if "input_pca" in model:
        transform = model["input_pca"]
        required = {transform["feature"]: transform["scaler"].n_features_in_}
    if set(mapping) - set(required):
        raise ValueError("Feature mapping contains keys unused by the saved model")
    needs_sources = any(b["feature"] == "source_only" for b in model.get("blocks", []))
    with np.load(features_path, allow_pickle=False) as archive:
        ids = archive["sample_ids"].copy()
        if (ids.ndim != 1 or ids.dtype.kind not in "US" or not len(ids) or
                len(set(ids.tolist())) != len(ids) or any(not x.strip() for x in ids.tolist())):
            raise ValueError("Expected unique nonempty string sample_ids")
        if needs_sources and "sources" not in archive:
            raise ValueError("The source-only control requires sources in the feature archive")
        sources = archive["sources"].copy() if "sources" in archive else np.repeat("", len(ids))
        if sources.shape != ids.shape or sources.dtype.kind not in "US":
            raise ValueError("sources must be strings aligned with sample_ids")
        store = {}
        for name, dimension in required.items():
            key = mapping.get(name, name)
            if key not in archive:
                raise ValueError(f"Missing feature {key}; map the saved model feature {name}")
            value = np.asarray(archive[key], dtype=np.float64)
            if value.shape != (len(ids), dimension) or not np.isfinite(value).all():
                raise ValueError(f"Invalid feature shape/values for {name}: expected [{len(ids)}, {dimension}]")
            store[name] = value
    with threadpool_limits(limits=threads):
        if "input_pca" in model:
            transform = model["input_pca"]
            store = {transform["output_feature"]: transform["pca"].transform(
                transform["scaler"].transform(store[transform["feature"]]))}
        probability = predict_saved(model, store, sources)
    if (probability.shape != (len(ids), len(labels)) or not np.isfinite(probability).all()
            or not ((probability >= 0) & (probability <= 1)).all()):
        raise ValueError("Saved model returned invalid probabilities or label columns")
    if input_hashes != {"model_sha256": hash_file(model_path),
                        "features_sha256": hash_file(features_path)}:
        raise ValueError("Prediction inputs changed while being read")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, sample_ids=ids, labels=np.asarray(labels),
                            probabilities=probability)
    receipt = {"schema_version": 1, **input_hashes, "sample_count": len(ids),
               "labels": labels, "feature_mapping": mapping,
               "no_fitting": True, "no_targets_read": True,
               "probabilities_sha256": hash_file(output)}
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Locally fitted .joblib readout")
    parser.add_argument("--features", required=True, type=Path, help="NPZ with sample_ids and feature matrices")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-map", action="append", default=[], metavar="MODEL_FEATURE=NPZ_KEY")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    result = predict(args.model, args.features, args.output,
                     feature_mapping(args.feature_map), args.threads)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
