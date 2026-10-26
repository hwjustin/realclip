"""Frozen image-only inference on the fixed 142-well new-plate MoA cohort.

This creates no text features, labels are not fitted, and no MoA predictions are
scored. All sites and the unchanged original five-channel preprocessing are used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wellclip.common import (CHANNELS, PREPROCESS, hash_file, load_manifest, model_identity,
                            read_json, stable_hash, utc_now, write_json)
from wellclip.encoders import FrozenDINO
from wellclip.features import _atomic_npz, pool_well_features
from wellclip.preprocessing import compute_windows, load_raw, preprocess_array
from wellclip.training import load_run_model, validate_training


@torch.no_grad()
def run(root: Path, supplemental: Path, device: str, batch_size: int) -> dict:
    torch.set_num_threads(4)
    original = root / "CLIP/artifacts/prototype_v1"
    manifest_path = supplemental / "manifest.json"
    manifest = load_manifest(manifest_path)
    cohort_path = supplemental / "cohort.json"
    cohort = read_json(cohort_path)
    if manifest["manifest_id"] != cohort["manifest_id"] or hash_file(manifest_path) != cohort["artifacts"]["manifest.json"]:
        raise ValueError("Supplemental manifest and verified cohort disagree")
    if len(manifest["samples"]) != 142 or sum(len(s["images"]) for s in manifest["samples"]) != 5490:
        raise ValueError("Fixed supplemental cohort changed")
    original_metadata = read_json(original / "features/metadata.json")
    _, receipts = validate_training(original / "training", original_metadata["manifest_id"], original_metadata["feature_id"])
    dino_path = Path(original_metadata["identity"]["model_identities"]["dino"]["path"])
    dino_identity = model_identity(dino_path)
    if dino_identity != original_metadata["identity"]["model_identities"]["dino"]:
        raise ValueError("DINO backbone differs from the sealed original")
    for name, expected in original_metadata["identity"]["implementation_sha256"].items():
        if hash_file(root / "CLIP/src/wellclip" / name) != expected:
            raise ValueError(f"Original feature code changed: {name}")
    code = {name: hash_file(root / "CLIP/src/wellclip" / name)
            for name in ("encoders.py", "features.py", "preprocessing.py", "model.py")}
    payload = {
        "schema_version": 1, "purpose": "Fixed new-plate supplemental MoA evaluation; image-only inference",
        "sample_ids": [s["sample_id"] for s in manifest["samples"]], "sample_count": 142,
        "manifest_id": manifest["manifest_id"], "manifest_sha256": hash_file(manifest_path),
        "cohort_sha256": hash_file(cohort_path), "original_feature_id": original_metadata["feature_id"],
        "original_training_complete_sha256": hash_file(original / "training/complete.json"),
        "fixed_label_sets": {name: value["labels"] for name, value in cohort["fixed_label_sets"].items()},
        "downstream_training": "Original306 train only; original66 validation selection only",
        "downstream_test_access": "Score these142 only after all original306/66-selected MoA probes are sealed; never tune on supplemental outcomes",
        "no_new_captions_or_text_encoding": True, "no_fitting_on_supplemental": True,
        "no_original_parent_or_plate_overlap": True, "preprocessing": PREPROCESS,
        "channels": list(CHANNELS), "image_feature": "unchanged DINO CLS384, all-site per-channel mean then L2",
        "backbone_identity": dino_identity, "code_sha256": code, "script_sha256": hash_file(__file__),
    }
    protocol = {**payload, "protocol_id": stable_hash(payload)}
    protocol_path = supplemental / "supplemental_protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("Supplemental extraction protocol changed; use a separate output directory")
    write_json(protocol_path, protocol)
    output = supplemental / "features"
    output.mkdir(parents=True, exist_ok=True)
    feature_id = stable_hash({"protocol_id": protocol["protocol_id"], "backbone": dino_identity, "feature": payload["image_feature"]})
    image_encoder = None
    features, artifacts, all_windows = [], {}, {}
    for number, sample in enumerate(manifest["samples"]):
        sid = sample["sample_id"]
        path = output / "wells" / f"{sid}.npz"
        receipt_path = path.with_suffix(".json")
        identity = stable_hash({"feature_id": feature_id, "sample": sample})
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["identity"] != identity or hash_file(path) != receipt["sha256"]:
                raise ValueError("Supplemental per-well cache identity changed")
            with np.load(path, allow_pickle=False) as archive:
                if archive["image_ids"].tolist() != [im["image_id"] for im in sample["images"]]:
                    raise ValueError("Supplemental per-site cache ordering changed")
                cls = archive["image_cls"].copy()
                pooled = archive["image_features"].copy()
            np.testing.assert_allclose(pooled, pool_well_features(cls, sample["images"]), atol=1e-6)
        else:
            if image_encoder is None:
                image_encoder = FrozenDINO(dino_path, device=device)
            windows = compute_windows(sample)  # load_raw authenticates every TIFF.
            cls = []
            for start in range(0, len(sample["images"]), batch_size):
                group = sample["images"][start:start + batch_size]
                pixels = torch.stack([preprocess_array(load_raw(im), windows[im["channel"]]) for im in group])
                cls.append(image_encoder.encode(pixels).cpu().numpy())
            cls = np.concatenate(cls).astype(np.float32)
            pooled = pool_well_features(cls, sample["images"])
            _atomic_npz(path, image_cls=cls, image_features=pooled,
                        image_ids=np.asarray([im["image_id"] for im in sample["images"]]),
                        sites=np.asarray([str(im["site"]) for im in sample["images"]]),
                        channels=np.asarray([im["channel"] for im in sample["images"]]))
            receipt = {"identity": identity, "sample_id": sid, "sha256": hash_file(path), "windows": windows,
                       "image_count": len(sample["images"]), "site_count": len({im["site"] for im in sample["images"]}),
                       "completed_at": utc_now()}
            write_json(receipt_path, receipt)
        features.append(pooled)
        all_windows[sid] = receipt["windows"]
        artifacts[str(path.relative_to(output))] = hash_file(path)
        artifacts[str(receipt_path.relative_to(output))] = hash_file(receipt_path)
        if number == 0 or (number + 1) % 10 == 0 or number + 1 == len(manifest["samples"]):
            print(f"Supplemental DINO {number + 1}/142 wells", flush=True)
    values = np.stack(features)
    _atomic_npz(output / "features.npz", sample_ids=np.asarray([s["sample_id"] for s in manifest["samples"]]), image_features=values)
    write_json(output / "windows.json", all_windows)
    artifacts.update({name: hash_file(output / name) for name in ("features.npz", "windows.json")})
    metadata = {"feature_id": feature_id, "manifest_id": manifest["manifest_id"], "protocol_id": protocol["protocol_id"],
                "original_feature_id": original_metadata["feature_id"], "backbone_identity": dino_identity,
                "sample_ids": [s["sample_id"] for s in manifest["samples"]], "image_feature_shape": list(values.shape),
                "artifacts": artifacts, "completed_at": utc_now(), "frozen_backbone": True, "no_text_features": True}
    write_json(output / "metadata.json", metadata)
    if image_encoder is not None:
        del image_encoder
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    projected = {}
    for receipt in receipts:
        name = receipt["name"]
        model, _ = load_run_model(original / "training" / name / "best.pt", device="cpu")
        projected[name] = model.encode_image(torch.from_numpy(values)).numpy()
    _atomic_npz(supplemental / "projected.npz", sample_ids=np.asarray(metadata["sample_ids"]), **projected)
    result = {"protocol_id": protocol["protocol_id"], "manifest_id": manifest["manifest_id"], "feature_id": feature_id,
              "sample_count": 142, "image_count": 5490, "completed_at": utc_now(),
              "original_checkpoint_sha256": {r["name"]: hash_file(original / "training" / r["name"] / "best.pt") for r in receipts},
              "artifacts": {"features/features.npz": hash_file(output / "features.npz"),
                            "features/metadata.json": hash_file(output / "metadata.json"),
                            "projected.npz": hash_file(supplemental / "projected.npz"),
                            "supplemental_protocol.json": hash_file(protocol_path)}}
    write_json(supplemental / "extraction_complete.json", result)
    print("Supplemental frozen image inference complete: 142 wells, six projected galleries", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--supplemental", type=Path, default=root / "CLIP/artifacts/moa_probe_v1/supplemental")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    run(args.root.resolve(), args.supplemental.resolve(), args.device, args.batch_size)
