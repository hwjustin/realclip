"""Sealed retrieval and pre/post-head exports for the DINO last-two-block pilot.

This module never fits a backbone or a downstream readout. MOA fitting lives in
``scripts/run_finetune_moa_probe.py`` so it can use the existing sklearn env.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import hash_file, load_manifest, model_identity, read_json, stable_hash, write_json
from .features import _atomic_npz, pool_well_features
from .training import load_features, random_expectation, retrieval_metrics, split_indices

ARMS = ("frozen_matched", "frozen_shuffled", "tail_matched", "tail_shuffled")


def assert_supplemental_isolation(primary: list[dict], supplemental: list[dict]) -> None:
    """Extra test wells must not share a training/validation parent or plate."""
    fitted = [sample for sample in primary if sample["split"] in ("train", "val")]
    parents = {sample["parent_id"] for sample in fitted}
    plates = {(sample["source"], sample["plate"]) for sample in fitted}
    ids = {sample["sample_id"] for sample in primary}
    wells = {sample["well_id"] for sample in primary}
    for sample in supplemental:
        if sample["parent_id"] in parents:
            raise ValueError("Supplemental chemical parent overlaps train/validation")
        if (sample["source"], sample["plate"]) in plates:
            raise ValueError("Supplemental plate overlaps train/validation")
        if sample["sample_id"] in ids or sample["well_id"] in wells:
            raise ValueError("Supplemental well duplicates the primary cohort")
    if len({s["sample_id"] for s in supplemental}) != len(supplemental):
        raise ValueError("Duplicate supplemental sample")


def encode_pooled(head, pooled: np.ndarray, text: np.ndarray | None, device: str = "cpu") -> dict:
    """Return the morphology representation before, and alignment after, the head."""
    pooled = np.asarray(pooled, dtype=np.float32)
    if pooled.ndim != 3 or pooled.shape[1:] != (5, 384) or not np.isfinite(pooled).all():
        raise ValueError("Expected complete-well pooled features [N,5,384]")
    if not np.allclose(np.linalg.norm(pooled, axis=-1), 1.0, atol=1e-5):
        raise ValueError("Each pre-head channel must have unit norm")
    head.eval()
    with torch.inference_mode():
        image = head.encode_image(torch.from_numpy(pooled).to(device)).float().cpu().numpy()
        result = {"pre_projection": pooled.reshape(len(pooled), 1920).copy(),
                  "image_embeddings": image}
        if text is not None:
            text = np.asarray(text, dtype=np.float32)
            if text.shape != (len(pooled), 512) or not np.isfinite(text).all():
                raise ValueError("Expected unchanged frozen text features [N,512]")
            result["text_embeddings"] = head.encode_text(torch.from_numpy(text).to(device)).float().cpu().numpy()
    for key in ("image_embeddings", "text_embeddings"):
        if key in result and (result[key].shape != (len(pooled), 128) or
                              not np.isfinite(result[key]).all() or
                              not np.allclose(np.linalg.norm(result[key], axis=1), 1.0, atol=1e-5)):
            raise ValueError(f"Invalid unit 128-dimensional {key}")
    return result


def encode_tail_well(split, tokens: np.ndarray, images: list[dict], device: str,
                     view_batch_size: int = 8) -> np.ndarray:
    """All views contribute once; site order is irrelevant to channel pooling."""
    if view_batch_size < 1 or len(tokens) != len(images):
        raise ValueError("Invalid view batch or cached view membership")
    split.tail.eval()
    cls = []
    with torch.inference_mode():
        for start in range(0, len(tokens), view_batch_size):
            # Memmaps are read-only; copy just this batch before torch wraps it.
            batch = torch.from_numpy(np.array(tokens[start:start + view_batch_size], copy=True)).to(device)
            cls.append(split.tail(batch).float().cpu().numpy())
    return pool_well_features(np.concatenate(cls), images)


def _load_supplemental_features(manifest: dict, directory: Path) -> tuple[dict, dict]:
    metadata = read_json(directory / "metadata.json")
    if hash_file(directory / "features.npz") != metadata.get("artifacts", {}).get("features.npz"):
        raise ValueError("Supplemental frozen feature checksum mismatch")
    with np.load(directory / "features.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if arrays["sample_ids"].tolist() != [s["sample_id"] for s in manifest["samples"]]:
        raise ValueError("Supplemental frozen features have different well order")
    pooled = arrays["image_features"]
    if pooled.shape != (len(manifest["samples"]), 5, 384) or not np.isfinite(pooled).all():
        raise ValueError("Invalid supplemental image features")
    # This is an authenticated retained subset of the old image-only cache. Its
    # old manifest has a different schema from the new raw-TIFF prefix manifest.
    return arrays, metadata


def _tail_pool(split, manifest: dict, cache_dir: Path, output: Path, binding: dict,
               device: str, view_batch_size: int) -> np.ndarray:
    from .finetune_cache import load_well_tokens

    values = []
    for index, sample in enumerate(manifest["samples"]):
        path = output / "wells" / f"{sample['sample_id']}.npz"
        receipt_path = path.with_suffix(".json")
        identity = stable_hash({"binding": binding, "sample": sample})
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt.get("identity") != identity or hash_file(path) != receipt.get("sha256"):
                raise ValueError("Fine-tuned pooled feature cache identity changed")
            with np.load(path, allow_pickle=False) as archive:
                pooled = archive["image_features"].copy()
        else:
            tokens = load_well_tokens(cache_dir, sample["sample_id"])
            pooled = encode_tail_well(split, tokens, sample["images"], device, view_batch_size)
            _atomic_npz(path, image_features=pooled)
            write_json(receipt_path, {"identity": identity, "sha256": hash_file(path)})
        values.append(pooled)
        if (index + 1) % 25 == 0 or index + 1 == len(manifest["samples"]):
            print(f"EXPORTED {output.name} {index + 1}/{len(manifest['samples'])}", flush=True)
    return np.stack(values)


def export_representations(manifest_path: str | Path, frozen_features_dir: str | Path,
                           prefix_cache_dir: str | Path, training_dir: str | Path,
                           output_dir: str | Path, dino_path: str | Path,
                           device: str = "cuda", view_batch_size: int = 8,
                           supplemental_manifest_path: str | Path | None = None,
                           supplemental_cache_dir: str | Path | None = None,
                           supplemental_features_dir: str | Path | None = None,
                           legacy_store_dir: str | Path | None = None) -> dict:
    """Export only after every arm is selected, without consulting MOA labels.

    ``legacy_store_dir`` is the existing expanded MOA ``precomputed`` folder;
    its two authoritative frozen seed-42 arrays are retained as references.
    """
    from .finetune_training import load_finetune_run, validate_finetune_training
    from .finetune_cache import validate_prefix_cache

    manifest, arrays, metadata = load_features(manifest_path, frozen_features_dir)
    if {name: len(rows) for name, rows in split_indices(manifest).items()} != {"train": 603, "val": 66, "test": 66}:
        raise ValueError("Use the fixed603/66/66 fine-tuning cohort")
    training_dir, output_dir = Path(training_dir), Path(output_dir)
    protocol, receipts = validate_finetune_training(training_dir)
    if {r.get("arm", r.get("name")) for r in receipts} != set(ARMS):
        raise ValueError("All four predetermined arms must be complete before export")
    if protocol.get("manifest_id", manifest["manifest_id"]) != manifest["manifest_id"]:
        raise ValueError("Fine-tuning used a different manifest")
    if protocol.get("feature_id", metadata["feature_id"]) != metadata["feature_id"]:
        raise ValueError("Fine-tuning used different frozen text/image features")
    if model_identity(dino_path)["sha256"] != metadata["model_identities"]["dino"]["sha256"]:
        raise ValueError("Evaluation DINO weights differ from original frozen features")
    primary_cache = validate_prefix_cache(prefix_cache_dir, manifest_path, verify_arrays=False)
    if protocol.get("prefix_cache_id") != primary_cache["cache_id"]:
        raise ValueError("Fine-tuning used a different frozen prefix cache")
    if primary_cache["identity"]["model_identity"]["sha256"] != metadata["model_identities"]["dino"]["sha256"]:
        raise ValueError("Prefix cache DINO differs from original frozen features")
    supplemental_args = (supplemental_manifest_path, supplemental_cache_dir, supplemental_features_dir)
    if any(value is not None for value in supplemental_args) and not all(value is not None for value in supplemental_args):
        raise ValueError("Supplemental manifest, prefix cache and frozen features are all required")
    scopes = {"primary": (manifest, arrays, Path(prefix_cache_dir))}
    if supplemental_manifest_path is not None:
        extra_manifest = load_manifest(supplemental_manifest_path)
        assert_supplemental_isolation(manifest["samples"], extra_manifest["samples"])
        extra_cache = validate_prefix_cache(supplemental_cache_dir, supplemental_manifest_path, verify_arrays=False)
        for key in ("model_identity", "preprocessing", "split_after_block", "tail_blocks", "prefix_compute_dtype", "array_dtype"):
            if extra_cache["identity"][key] != primary_cache["identity"][key]:
                raise ValueError(f"Supplemental prefix cache differs in {key}")
        extra_arrays, _ = _load_supplemental_features(extra_manifest, Path(supplemental_features_dir))
        scopes["supplemental"] = (extra_manifest, extra_arrays, Path(supplemental_cache_dir))
    binding = {"version": "last-two-DINO-evaluation-v1", "protocol_id": protocol["protocol_id"],
               "manifest_id": manifest["manifest_id"], "feature_id": metadata["feature_id"],
               "training_complete_sha256": hash_file(training_dir / "complete.json"),
               "evaluation_code_sha256": hash_file(__file__),
               "checkpoints": {arm: hash_file(training_dir / arm / "best.pt") for arm in ARMS},
               "prefix_metadata": {scope: hash_file(cache / "metadata.json") for scope, (_, _, cache) in scopes.items()},
               "supplemental_manifest_id": scopes.get("supplemental", ({},))[0].get("manifest_id"),
               "frozen_archives": {"primary": hash_file(Path(frozen_features_dir) / "features.npz")}}
    binding["frozen_metadata"] = {"primary": hash_file(Path(frozen_features_dir) / "metadata.json")}
    if supplemental_features_dir is not None:
        binding["frozen_archives"]["supplemental"] = hash_file(Path(supplemental_features_dir) / "features.npz")
        binding["frozen_metadata"]["supplemental"] = hash_file(Path(supplemental_features_dir) / "metadata.json")
    if legacy_store_dir is not None:
        binding["legacy_stores"] = {scope: hash_file(Path(legacy_store_dir) / f"{scope}_store.npz") for scope in scopes}
    output_dir.mkdir(parents=True, exist_ok=True)
    binding_path = output_dir / "binding.json"
    if binding_path.exists() and read_json(binding_path) != binding:
        raise ValueError("Representation output belongs to different inputs or code")
    write_json(binding_path, binding)
    complete_path = output_dir / "complete.json"
    if complete_path.exists():
        complete = read_json(complete_path)
        if complete["binding_sha256"] != hash_file(binding_path):
            raise ValueError("Export completion binding changed")
        for name, digest in complete["artifacts"].items():
            if hash_file(output_dir / name) != digest:
                raise ValueError(f"Exported representation changed: {name}")
        return complete
    stores = {scope: {"sample_ids": data["sample_ids"],
                      "dino": np.asarray(data["image_features"], dtype=np.float32).reshape(len(data["sample_ids"]), 1920)}
              for scope, (_, data, _) in scopes.items()}
    if legacy_store_dir is not None:
        for scope, store in stores.items():
            with np.load(Path(legacy_store_dir) / f"{scope}_store.npz", allow_pickle=False) as archive:
                if archive["sample_ids"].tolist() != store["sample_ids"].tolist():
                    raise ValueError("Legacy MOA store well order differs")
                for arm in ("matched", "shuffled"):
                    store[f"legacy_{arm}_seed42"] = archive[f"new_{arm}_seed42"].copy()
    artifacts = {}
    for arm in ARMS:
        split, head, checkpoint = load_finetune_run(training_dir / arm / "best.pt", dino_path, device)
        for key, expected in (("arm", arm), ("protocol_id", protocol["protocol_id"]),
                              ("manifest_id", manifest["manifest_id"]), ("feature_id", metadata["feature_id"]),
                              ("prefix_cache_id", primary_cache["cache_id"])):
            if checkpoint.get(key) != expected:
                raise ValueError(f"Checkpoint {key} differs from selected run inputs")
        for scope, (scope_manifest, data, cache) in scopes.items():
            pooled = data["image_features"] if arm.startswith("frozen_") else _tail_pool(
                split, scope_manifest, cache, output_dir / "pooled" / scope / arm,
                {"export": binding, "arm": arm, "scope": scope}, device, view_batch_size)
            encoded = encode_pooled(head, pooled, data.get("text_features") if scope == "primary" else None, device)
            stores[scope][f"{arm}_pre1920"] = encoded["pre_projection"]
            stores[scope][f"{arm}_post128"] = encoded["image_embeddings"]
            gallery = output_dir / f"{scope}__{arm}_gallery.npz"
            _atomic_npz(gallery, sample_ids=data["sample_ids"], **encoded)
            artifacts[gallery.name] = hash_file(gallery)
        del split, head
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    for scope, store in stores.items():
        path = output_dir / f"{scope}_store.npz"
        _atomic_npz(path, **store)
        artifacts[path.name] = hash_file(path)
    sample_path = output_dir / "samples.json"
    write_json(sample_path, {scope: [{key: row[key] for key in ("sample_id", "well_id", "source", "plate", "parent_id", "split", "description") if key in row}
                                  for row in scope_manifest["samples"]]
                            for scope, (scope_manifest, _, _) in scopes.items()})
    artifacts[sample_path.name] = hash_file(sample_path)
    complete = {"binding_sha256": hash_file(binding_path), "artifacts": artifacts,
                "arms": list(ARMS), "sample_counts": {scope: len(store["sample_ids"]) for scope, store in stores.items()},
                "pre_projection": "channel-wise unit mean-site DINO CLS, DNA/RNA/ER/Mito/AGP concatenation1920",
                "post_projection": "unit128 image/text embeddings", "moa_labels_used_for_export": False}
    write_json(complete_path, complete)
    return complete


def evaluate_retrieval(export_dir: str | Path, output_dir: str | Path | None = None) -> dict:
    """Score the original fixed test gallery after all four checkpoints are sealed."""
    from .evaluation import source_metrics, topk_cases

    export_dir = Path(export_dir)
    output_dir = Path(output_dir) if output_dir is not None else export_dir.parent / "retrieval"
    complete = read_json(export_dir / "complete.json")
    if complete["binding_sha256"] != hash_file(export_dir / "binding.json"):
        raise ValueError("Export binding changed")
    for name, digest in complete["artifacts"].items():
        if hash_file(export_dir / name) != digest:
            raise ValueError("Exported gallery or representation checksum mismatch")
    samples = read_json(export_dir / "samples.json")["primary"]
    splits = split_indices({"samples": samples})
    test = splits["test"]
    test_samples = [samples[i] for i in test]
    result, cases = [], {}
    for arm in ARMS:
        with np.load(export_dir / f"primary__{arm}_gallery.npz", allow_pickle=False) as archive:
            if archive["sample_ids"].tolist() != [s["sample_id"] for s in samples]:
                raise ValueError("Gallery sample order differs")
            similarities = archive["image_embeddings"][test] @ archive["text_embeddings"][test].T
        result.append({"arm": arm, "test": retrieval_metrics(similarities),
                       "within_source": source_metrics(similarities, test_samples)})
        cases[arm] = {direction: topk_cases(similarities, test_samples, direction)
                      for direction in ("image_to_text", "text_to_image")}
    metrics = {row["arm"]: row["test"]["mean_mrr"] for row in result}
    evaluation = {"scope": "fixed-seed pilot; held-out plates; original internal test gallery",
                  "export_complete_sha256": hash_file(export_dir / "complete.json"),
                  "split_counts": {key: len(value) for key, value in splits.items()}, "runs": result,
                  "random_expectation": random_expectation(len(test)),
                  "mean_mrr_differences": {
                      "tail_minus_frozen_matched": metrics["tail_matched"] - metrics["frozen_matched"],
                      "tail_minus_frozen_shuffled": metrics["tail_shuffled"] - metrics["frozen_shuffled"],
                      "matched_minus_shuffled_frozen": metrics["frozen_matched"] - metrics["frozen_shuffled"],
                      "matched_minus_shuffled_tail": metrics["tail_matched"] - metrics["tail_shuffled"]}}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evaluation.json", evaluation)
    write_json(output_dir / "top5.json", cases)
    return evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    for name in ("manifest-path", "frozen-features-dir", "prefix-cache-dir", "training-dir", "output-dir", "dino-path"):
        export.add_argument(f"--{name}", type=Path, required=True)
    for name in ("supplemental-manifest-path", "supplemental-cache-dir", "supplemental-features-dir", "legacy-store-dir"):
        export.add_argument(f"--{name}", type=Path)
    export.add_argument("--device", default="cuda")
    export.add_argument("--view-batch-size", type=int, default=8)
    evaluate = subparsers.add_parser("retrieval")
    evaluate.add_argument("--export-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    export_representations(**args) if command == "export" else evaluate_retrieval(**args)
