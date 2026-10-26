"""Authenticate old/new CLIP heads and precompute frozen MoA feature stores.

Uses the CLIP torch environment. Does not read MoA target values or fit weights.
The plan is bound to cohort/artifact hashes before any downstream model fitting.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wellclip.common import hash_file, read_json, utc_now, write_json
from wellclip.training import load_features, load_run_model, validate_training


def assert_isolation(primary: dict, supplemental: dict) -> None:
    known = np.isin(primary["splits"], ["train", "val"])
    parent_overlap = set(primary["parent_ids"][known]) & set(supplemental["parent_ids"])
    plate_overlap = set(zip(primary["sources"][known], primary["plates"][known])) & set(zip(supplemental["sources"], supplemental["plates"]))
    if parent_overlap or plate_overlap:
        raise ValueError("Supplemental cohort overlaps expanded training/validation parents or plates")


def ids_only(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in ["sample_ids", "splits", "parent_ids", "sources", "plates"]}


def freeze_protocol(output: Path, old: Path, expanded: Path) -> dict:
    plan_path = output / "protocol_plan.json"
    plan = read_json(plan_path)
    old_protocol = read_json(old.parent / "moa_probe_v1/protocol.json")
    for key in ["primary_labels", "exploratory_labels"]:
        assert plan[key] == old_protocol[key]
    paths = {
        "protocol_plan": plan_path, "cohort": output / "cohort.json", "cohort_arrays": output / "cohort_arrays.npz",
        "supplemental_cohort": output / "supplemental/cohort.json",
        "supplemental_cohort_arrays": output / "supplemental/cohort_arrays.npz",
        "expanded_features": expanded / "features/features.npz",
        "supplemental_features": output / "supplemental/features/features.npz",
        "old_manifest": old / "manifest.json", "expanded_manifest": expanded / "manifest.json",
        "original_moa_protocol": old.parent / "moa_probe_v1/protocol.json",
        "precompute_script": Path(__file__), "run_script": Path(__file__).with_name("run_expanded_moa_probe.py"),
        "fit_helper": Path(__file__).with_name("run_moa_linear_probe.py"),
    }
    bindings = {key: {"path": str(path.resolve()), "sha256": hash_file(path)} for key, path in paths.items()}
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        assert protocol["bindings"] == bindings, "Frozen expanded MoA inputs changed; use a new output directory"
        return protocol
    protocol = {**plan, "status": "frozen before downstream fitting and test scoring", "frozen_at_utc": utc_now(),
                "bindings": bindings, "n_representations": 22, "n_model_choices": 88}
    write_json(protocol_path, protocol)
    return protocol


@torch.inference_mode()
def run(old: Path, expanded: Path, output: Path) -> dict:
    torch.set_num_threads(2)
    protocol = freeze_protocol(output, old, expanded)
    cohort = ids_only(output / "cohort_arrays.npz")
    supplemental = ids_only(output / "supplemental/cohort_arrays.npz")
    assert_isolation(cohort, supplemental)
    original_manifest, original_arrays, old_metadata = load_features(old / "manifest.json", old / "features")
    manifest, arrays, new_metadata = load_features(expanded / "manifest.json", expanded / "features")
    assert cohort["sample_ids"].tolist() == arrays["sample_ids"].tolist()
    assert [s["split"] for s in manifest["samples"]] == cohort["splits"].tolist()
    assert len(cohort["sample_ids"]) == 735 and len(supplemental["sample_ids"]) == 67
    assert {k: int(np.sum(cohort["splits"] == k)) for k in ["train", "val", "test"]} == {"train": 603, "val": 66, "test": 66}
    lookup = {sample: i for i, sample in enumerate(cohort["sample_ids"])}
    original_rows = np.asarray([lookup[s] for s in original_arrays["sample_ids"]])
    np.testing.assert_array_equal(original_arrays["image_features"], arrays["image_features"][original_rows])
    for sample, row in zip(original_manifest["samples"], original_rows):
        assert sample["split"] == cohort["splits"][row]
    for key in ["model_identities", "preprocessing", "implementation_sha256"]:
        assert old_metadata["identity"][key] == new_metadata["identity"][key]
    with np.load(output / "supplemental/features/features.npz", allow_pickle=False) as archive:
        supplemental_features = archive["image_features"].copy()
        assert archive["sample_ids"].tolist() == supplemental["sample_ids"].tolist()
    assert supplemental_features.shape == (67, 5, 384) and np.isfinite(supplemental_features).all()
    stores = {"primary": {"sample_ids": arrays["sample_ids"], "dino": arrays["image_features"].reshape(735, 1920)},
              "supplemental": {"sample_ids": supplemental["sample_ids"], "dino": supplemental_features.reshape(67, 1920)}}
    models = []
    for version, work, metadata in [("old", old, old_metadata), ("new", expanded, new_metadata)]:
        _, receipts = validate_training(work / "training", metadata["manifest_id"], metadata["feature_id"])
        evaluation = read_json(work / "evaluation/evaluation.json")
        evaluation_seal = read_json(work / "evaluation/complete.json")
        assert hash_file(work / "evaluation/evaluation.json") == evaluation_seal["evaluation_sha256"]
        for receipt in receipts:
            name = receipt["name"]
            full_name = f"{version}_{name}"
            checkpoint_path = work / "training" / name / "best.pt"
            model, checkpoint = load_run_model(checkpoint_path, "cpu")
            primary_projection = model.encode_image(torch.from_numpy(arrays["image_features"])).numpy()
            supplemental_projection = model.encode_image(torch.from_numpy(supplemental_features)).numpy()
            gallery_path = work / "evaluation" / f"{name}_gallery.npz"
            assert hash_file(gallery_path) == evaluation["artifacts"][gallery_path.name]
            with np.load(gallery_path, allow_pickle=False) as gallery:
                gallery_rows = np.asarray([lookup[s] for s in gallery["sample_ids"]])
                authoritative = gallery["image_embeddings"].copy()
            error = float(np.max(np.abs(primary_projection[gallery_rows] - authoritative)))
            np.testing.assert_allclose(primary_projection[gallery_rows], authoritative, atol=2e-5, rtol=2e-4)
            primary_projection[gallery_rows] = authoritative
            stores["primary"][full_name] = primary_projection
            stores["supplemental"][full_name] = supplemental_projection
            models.append({"name": full_name, "checkpoint": str(checkpoint_path.resolve()),
                           "checkpoint_sha256": hash_file(checkpoint_path), "feature_id": checkpoint["feature_id"],
                           "gallery_sha256": hash_file(gallery_path), "reused_gallery_rows": len(gallery_rows),
                           "fresh_cpu_gallery_max_abs_error": error})
            print(f"PROJECTED {full_name}: 735 + 67 rows, gallery error={error:.2g}", flush=True)
    destination = output / "precomputed"
    destination.mkdir(exist_ok=True)
    for key, store in stores.items():
        np.savez_compressed(destination / f"{key}_store.npz", **store)
    complete = {"completed_at": utc_now(), "protocol_sha256": hash_file(output / "protocol.json"),
                "primary_store_sha256": hash_file(destination / "primary_store.npz"),
                "supplemental_store_sha256": hash_file(destination / "supplemental_store.npz"),
                "primary_rows": 735, "supplemental_rows": 67, "all_twelve_heads_authenticated": True,
                "original_438_dino_features_preserved_exactly": True,
                "backbone_preprocessing_and_code_equal": True, "supplemental_parent_plate_isolation_verified": True,
                "no_moa_target_values_read": True, "no_training": True, "models": models,
                "script_sha256": hash_file(__file__)}
    write_json(destination / "complete.json", complete)
    return complete


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "artifacts"
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=root / "prototype_v1")
    parser.add_argument("--expanded", type=Path, default=root / "expanded_samples_v1")
    parser.add_argument("--output", type=Path, default=root / "expanded_moa_probe_v1")
    args = parser.parse_args()
    run(args.old, args.expanded, args.output)
