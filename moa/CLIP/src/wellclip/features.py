"""Resumable, authenticated frozen features for complete treatment wells."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import torch

from .common import (CHANNELS, PREPROCESS, hash_file, load_manifest, model_identity,
                     read_json, stable_hash, utc_now, write_json)
from .encoders import FrozenBiomedCLIP, FrozenDINO
from .preprocessing import compute_windows, load_raw, preprocess_array


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def pool_well_features(image_cls: np.ndarray, images: list[dict]) -> np.ndarray:
    """Mean sites within each fixed channel; site ordering has no meaning."""
    values = np.asarray(image_cls, dtype=np.float32)
    if values.shape != (len(images), 384) or not np.isfinite(values).all():
        raise ValueError("Expected finite per-image CLS features [images,384]")
    keys = [(str(image["site"]), image["channel"]) for image in images]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate site/channel image")
    sites = {site for site, _ in keys}
    if not sites or set(keys) != {(site, channel) for site in sites for channel in CHANNELS}:
        raise ValueError("Each site must contain all five fluorescence channels")
    channels = np.asarray([image["channel"] for image in images])
    result = np.stack([values[channels == channel].mean(axis=0) for channel in CHANNELS])
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero-norm channel features")
    return (result / norms).astype(np.float32)


def _validate_sources(manifest: dict) -> None:
    checked = {}
    for sample in manifest["samples"]:
        description = sample["description"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Missing well description")
        if hashlib.sha256(description.encode("utf-8")).hexdigest() != sample["description_sha256"]:
            raise ValueError(f"Changed description for {sample['sample_id']}")
        for image in sample["images"]:
            for path_key, hash_key in (("path", "sha256"), ("observation_path", "observation_sha256")):
                if path_key not in image:
                    continue
                path, expected = image[path_key], image[hash_key]
                if path not in checked:
                    checked[path] = hash_file(path)
                if checked[path] != expected:
                    raise ValueError(f"Image source bytes changed: {path}")


def _well_cache(path: Path, receipt_path: Path, expected_identity: str, images: list[dict]):
    if not path.exists() and not receipt_path.exists():
        return None
    if not receipt_path.exists():
        # An interrupted atomic NPZ write may finish before its completion receipt.
        return None
    receipt = read_json(receipt_path)
    if receipt.get("identity") != expected_identity or not path.exists() or receipt.get("sha256") != hash_file(path):
        raise ValueError(f"Invalid well feature completion receipt: {receipt_path}")
    with np.load(path, allow_pickle=False) as arrays:
        image_cls = arrays["image_cls"].astype(np.float32)
        pooled = arrays["image_features"].astype(np.float32)
        text = arrays["text_features"].astype(np.float32)
        if arrays["image_ids"].tolist() != [image["image_id"] for image in images]:
            raise ValueError("Cached image ordering changed")
    if pooled.shape != (5, 384) or text.shape != (512,) or not np.isfinite(text).all():
        raise ValueError("Invalid cached well feature shapes")
    if not np.allclose(pooled, pool_well_features(image_cls, images), atol=1e-6):
        raise ValueError("Cached well pooling disagrees with original image CLS features")
    if not np.isclose(np.linalg.norm(text), 1.0, atol=1e-5):
        raise ValueError("Cached text vector is not normalized")
    return pooled, text, receipt


def extract(manifest_path: str | Path, output_dir: str | Path, dino_path: str | Path,
            biomedclip_path: str | Path, device: str = "cuda", batch_size: int = 16) -> dict:
    """Cache every site CLS, its complete-well pooling, and one description vector."""
    if batch_size < 1:
        raise ValueError("A positive extraction batch size is required")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest = load_manifest(manifest_path)
    if not manifest["samples"]:
        raise ValueError("Cannot extract an empty well manifest")
    _validate_sources(manifest)  # Also required when returning a completed cache.
    models = {"dino": model_identity(dino_path), "biomedclip": model_identity(biomedclip_path)}
    identity = {
        "version": "wellclip-frozen-features-v1", "manifest_id": manifest["manifest_id"],
        "model_identities": models, "preprocessing": PREPROCESS,
        "visual_feature": "frozen_DINOv2_small_CLS384_mean_sites_then_channel_L2",
        "text_feature": "official_BiomedCLIP_trained_text_tower_and_projection512_L2_context256",
        "channels": list(CHANNELS), "precision": "float32",
        "implementation_sha256": {name: hash_file(Path(__file__).with_name(name))
                                  for name in ("encoders.py", "features.py", "preprocessing.py")},
        "runtime": {name: version(name) for name in ("torch", "transformers", "open_clip_torch", "numpy", "pillow")},
    }
    feature_id = stable_hash(identity)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata.get("feature_id") != feature_id or metadata.get("identity") != identity:
            raise ValueError("Completed feature cache belongs to another manifest/model/preprocessing")
        for name, digest in metadata["artifacts"].items():
            if hash_file(output_dir / name) != digest:
                raise ValueError(f"Completed feature cache artifact changed: {name}")
        print(f"Verified completed feature cache: {len(manifest['samples'])} wells", flush=True)
        return metadata
    identity_path = output_dir / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Partial feature cache identity changed; use a separate output directory")
    write_json(identity_path, identity)
    image_encoder = text_encoder = None
    sample_ids, image_features, text_features = [], [], []
    windows, receipts, reused = {}, {}, 0
    for index, sample in enumerate(manifest["samples"]):
        sample_id = sample["sample_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
            raise ValueError("Unsafe sample ID for feature cache")
        path = output_dir / "wells" / f"{sample_id}.npz"
        receipt_path = path.with_suffix(".json")
        well_identity = stable_hash({"feature_id": feature_id, "sample": sample})
        cached = _well_cache(path, receipt_path, well_identity, sample["images"])
        if cached is None:
            if image_encoder is None:
                image_encoder = FrozenDINO(dino_path, device)
                text_encoder = FrozenBiomedCLIP(biomedclip_path, device)
            window = compute_windows(sample)
            image_batches = []
            for start in range(0, len(sample["images"]), batch_size):
                batch = sample["images"][start:start + batch_size]
                pixels = torch.stack([preprocess_array(load_raw(image), window[image["channel"]]) for image in batch])
                image_batches.append(image_encoder.encode(pixels).cpu().numpy())
            per_image = np.concatenate(image_batches).astype(np.float32)
            pooled = pool_well_features(per_image, sample["images"])
            token_count = text_encoder.token_counts([sample["description"]])[0]
            text = text_encoder.encode([sample["description"]])[0].cpu().numpy().astype(np.float32)
            _atomic_npz(path, image_cls=per_image, image_features=pooled, text_features=text,
                        image_ids=np.asarray([image["image_id"] for image in sample["images"]]),
                        sites=np.asarray([str(image["site"]) for image in sample["images"]]),
                        channels=np.asarray([image["channel"] for image in sample["images"]]))
            receipt = {"identity": well_identity, "sample_id": sample_id, "sha256": hash_file(path),
                       "windows": window, "token_count": token_count,
                       "site_count": len({str(image["site"]) for image in sample["images"]}),
                       "image_count": len(sample["images"]), "completed_at": utc_now()}
            write_json(receipt_path, receipt)
        else:
            pooled, text, receipt = cached
            reused += 1
        sample_ids.append(sample_id)
        image_features.append(pooled)
        text_features.append(text)
        windows[sample_id] = receipt["windows"]
        receipts[str(path.relative_to(output_dir))] = receipt["sha256"]
        receipts[str(receipt_path.relative_to(output_dir))] = hash_file(receipt_path)
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(manifest["samples"]):
            print(f"Frozen features {index + 1}/{len(manifest['samples'])} wells ({reused} resumed)", flush=True)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample IDs in feature cache")
    _atomic_npz(output_dir / "features.npz", sample_ids=np.asarray(sample_ids),
                image_features=np.stack(image_features), text_features=np.stack(text_features))
    write_json(output_dir / "windows.json", {"preprocessing": PREPROCESS, "wells": windows})
    artifacts = {name: hash_file(output_dir / name) for name in ("features.npz", "windows.json", "identity.json")}
    artifacts.update(receipts)
    metadata = {"feature_id": feature_id, "manifest_id": manifest["manifest_id"], "identity": identity,
                "model_identities": models, "sample_ids": sample_ids, "sample_count": len(sample_ids),
                "artifacts": artifacts, "completed_at": utc_now(), "frozen_backbones": True,
                "image_feature_shape": [len(sample_ids), 5, 384], "text_feature_shape": [len(sample_ids), 512]}
    write_json(metadata_path, metadata)
    return metadata
