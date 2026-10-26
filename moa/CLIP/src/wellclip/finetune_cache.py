"""Authenticated streaming full-token caches for last-block DINO tuning.

Only the frozen image prefix is cached. Final CLS vectors are recomputed through
the trainable tail on every optimization step, preserving genuine gradients.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import re

import numpy as np
import torch

from .common import (PREPROCESS, hash_file, load_manifest, model_identity, read_json,
                     stable_hash, utc_now, write_json)
from .features import pool_well_features
from .finetune_backbone import SplitDINO
from .preprocessing import compute_windows, load_raw, preprocess_array

_VERIFIED_ARRAYS: dict[str, tuple] = {}
_METADATA: dict[str, tuple[tuple, dict]] = {}


def _validate_prefix_sources(manifest: dict) -> None:
    """Authenticate image-only MOA wells and paired training wells alike.

    A held-out MOA well has no caption and never becomes a contrastive example.
    Present captions retain their original byte-level authentication.
    """
    checked = {}

    def check(path, expected):
        if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ValueError(f"Missing or invalid source checksum: {path}")
        path = str(path)
        if path not in checked:
            checked[path] = hash_file(path)
        if checked[path] != expected:
            raise ValueError(f"Image/provenance source bytes changed: {path}")

    samples = manifest["samples"]
    for ordinal, sample in enumerate(samples, start=1):
        if "description" in sample or "description_sha256" in sample:
            description = sample.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError("Missing well description for a paired sample")
            if hashlib.sha256(description.encode("utf-8")).hexdigest() != sample.get("description_sha256"):
                raise ValueError(f"Changed description for {sample['sample_id']}")
        for image in sample["images"]:
            check(image["path"], image.get("sha256"))
            if "observation_path" in image:
                check(image["observation_path"], image.get("observation_sha256"))
            if image.get("role", "treatment") != "treatment":
                raise ValueError("Only treatment images may enter the DINO prefix")
            if "well_id" in image and image["well_id"] != sample.get("well_id"):
                raise ValueError("Image and manifest well identities differ")
        provenance = sample.get("provenance", {})
        for path, expected in provenance.get("files", {}).items():
            check(path, expected)
        if "pairing_path" in provenance:
            check(provenance["pairing_path"], provenance.get("pairing_sha256"))
        if ordinal == 1 or ordinal % 100 == 0 or ordinal == len(samples):
            print(f"Prefix source verification {ordinal}/{len(samples)} wells ({len(checked)} files)", flush=True)


def _safe_sample_id(sample_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
        raise ValueError("Unsafe prefix-cache sample ID")


def _signature(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def _verify_array(path: Path, expected: str) -> None:
    key = str(path.resolve())
    signature = (*_signature(path), expected)
    if _VERIFIED_ARRAYS.get(key) != signature:
        if hash_file(path) != expected:
            raise ValueError(f"Prefix token array checksum mismatch: {path}")
        _VERIFIED_ARRAYS[key] = signature


def _cache_metadata(directory: Path) -> dict:
    path = directory / "metadata.json"
    signature = _signature(path)
    key = str(path.resolve())
    if key not in _METADATA or _METADATA[key][0] != signature:
        value = read_json(path)
        if value.get("cache_id") != stable_hash(value["identity"]):
            raise ValueError("Prefix-cache identity checksum mismatch")
        if value.get("manifest_id") != value["identity"]["manifest_id"]:
            raise ValueError("Prefix-cache manifest binding mismatch")
        _METADATA[key] = (signature, value)
    return _METADATA[key][1]


def load_well_tokens(cache_dir: str | Path, sample_id: str, verify: bool = True) -> np.memmap:
    """Read full tokens in manifest image order without materializing a well.

    SHA256 is checked once per unchanged file in each process. ``verify=False``
    is for a trainer that already called ``validate_prefix_cache``; header and
    completion receipt are still checked.
    """
    _safe_sample_id(sample_id)
    directory = Path(cache_dir).resolve()
    metadata = _cache_metadata(directory)
    record = metadata["well_records"][sample_id]
    relative = f"wells/{sample_id}"
    receipt_path, array_path = directory / (relative + ".json"), directory / (relative + ".npy")
    if hash_file(receipt_path) != metadata["artifacts"][relative + ".json"]:
        raise ValueError("Prefix well receipt checksum mismatch")
    receipt = read_json(receipt_path)
    if (receipt["identity"] != record["identity"] or receipt["cache_id"] != metadata["cache_id"]
            or receipt["image_ids"] != record["image_ids"] or receipt["sample_id"] != sample_id
            or receipt["shape"] != record["shape"] or receipt["dtype"] != record["dtype"]
            or receipt["sha256"] != metadata["artifacts"][relative + ".npy"]):
        raise ValueError("Prefix well identity or image ordering mismatch")
    if verify:
        _verify_array(array_path, receipt["sha256"])
    array = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
        raise ValueError("Prefix token shape or dtype mismatch")
    return array


def validate_prefix_cache(cache_dir: str | Path, manifest_path: str | Path | None = None,
                          verify_arrays: bool = True) -> dict:
    directory = Path(cache_dir).resolve()
    metadata = _cache_metadata(directory)
    if manifest_path is not None:
        manifest = load_manifest(manifest_path)
        if metadata["manifest_id"] != manifest["manifest_id"]:
            raise ValueError("Prefix cache belongs to a different manifest")
        if metadata["sample_ids"] != [s["sample_id"] for s in manifest["samples"]]:
            raise ValueError("Prefix sample order differs from manifest")
    for sample_id in metadata["sample_ids"]:
        load_well_tokens(directory, sample_id, verify=verify_arrays)
    return metadata


def _reference(features_dir: Path | None, samples: list[dict]) -> tuple[dict | None, dict]:
    if features_dir is None:
        return None, {}
    metadata_path, feature_path = features_dir / "metadata.json", features_dir / "features.npz"
    metadata = read_json(metadata_path)
    if metadata.get("artifacts", {}).get("features.npz") != hash_file(feature_path):
        raise ValueError("Reference frozen feature checksum mismatch")
    with np.load(feature_path, allow_pickle=False) as arrays:
        if arrays["sample_ids"].tolist() != [s["sample_id"] for s in samples]:
            raise ValueError("Reference feature rows differ from prefix manifest")
        values = arrays["image_features"].copy()
    if values.shape != (len(samples), 5, 384):
        raise ValueError("Expected standard five-channel frozen DINO reference features")
    reference = {s["sample_id"]: values[i] for i, s in enumerate(samples)}
    identity = {"metadata_path": str(metadata_path), "metadata_sha256": hash_file(metadata_path),
                "archive_sha256": hash_file(feature_path), "feature_id": metadata.get("feature_id")}
    windows_path = features_dir / "windows.json"
    if windows_path.exists():
        window_hash = hash_file(windows_path)
        expected = metadata.get("artifacts", {}).get("windows.json")
        if expected is not None and expected != window_hash:
            raise ValueError("Reference intensity-window checksum mismatch")
        identity["windows_sha256"] = window_hash
    return identity, reference


def cache_prefixes(manifest_path: str | Path, features_dir: str | Path | None,
                   output_dir: str | Path, dino_path: str | Path, device: str = "cuda",
                   batch_size: int = 8, dtype: str = "float32") -> dict:
    """Run and persist only the frozen prefix, with per-well atomic recovery."""
    if batch_size < 1 or dtype not in ("float32", "float16"):
        raise ValueError("Positive microbatch and float32/float16 cache dtype required")
    manifest = load_manifest(manifest_path)
    samples = manifest["samples"]
    if not samples:
        raise ValueError("Cannot cache an empty image collection")
    _validate_prefix_sources(manifest)
    output = Path(output_dir).resolve()
    features_dir = Path(features_dir).resolve() if features_dir is not None else None
    reference_identity, references = _reference(features_dir, samples)
    identity = {
        "version": "dinov2-frozen-first10-fulltokens-v1", "manifest_id": manifest["manifest_id"],
        "model_identity": model_identity(dino_path), "preprocessing": PREPROCESS,
        "split_after_block": 10, "tail_blocks": 2, "prefix_compute_dtype": "float32",
        "array_dtype": dtype, "reference_features": reference_identity,
        "implementations": {name: hash_file(Path(__file__).with_name(name)) for name in
                            ("finetune_backbone.py", "finetune_cache.py", "preprocessing.py")},
        "runtime": {**{name: version(name) for name in ("torch", "transformers", "numpy", "pillow")},
                    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "device_type": torch.device(device).type,
                    "device_name": (torch.cuda.get_device_name(torch.device(device))
                                    if torch.device(device).type == "cuda" else "cpu"),
                    "allow_tf32": torch.backends.cuda.matmul.allow_tf32},
    }
    cache_id = stable_hash(identity)
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Existing prefix cache has a different frozen identity")
    if (output / "metadata.json").exists():
        metadata = validate_prefix_cache(output, manifest_path)
        if metadata["identity"] != identity:
            raise ValueError("Completed prefix cache uses another implementation/model")
        print(f"Verified completed prefix cache: {len(samples)} wells", flush=True)
        return metadata
    if not identity_path.exists():
        write_json(identity_path, identity)
    windows_archive = None
    if features_dir is not None and (features_dir / "windows.json").exists():
        windows_archive = read_json(features_dir / "windows.json")
        if windows_archive.get("preprocessing") != PREPROCESS:
            raise ValueError("Reference intensity windows use another preprocessing")
    model = None
    artifacts, records, reused = {}, {}, 0
    for ordinal, sample in enumerate(samples, start=1):
        sample_id = sample["sample_id"]
        _safe_sample_id(sample_id)
        path = output / "wells" / f"{sample_id}.npy"
        receipt_path = path.with_suffix(".json")
        well_identity = stable_hash({"cache_id": cache_id, "sample": sample})
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if (receipt.get("identity") != well_identity or receipt.get("cache_id") != cache_id
                    or receipt.get("image_ids") != [i["image_id"] for i in sample["images"]]):
                raise ValueError(f"Changed completed prefix well: {sample_id}")
            _verify_array(path, receipt["sha256"])
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != receipt["shape"] or str(array.dtype) != dtype:
                raise ValueError("Completed prefix array header differs")
            del array
            reused += 1
        else:
            if model is None:
                model = SplitDINO(dino_path, device=device)
                if model.split_index != 10 or model.config.hidden_size != 384:
                    raise ValueError("Expected DINOv2-small with frozen first ten blocks")
            window = (windows_archive["wells"][sample_id] if windows_archive is not None
                      else compute_windows(sample))
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
            array, shape, frozen_cls = None, None, []
            try:
                for start in range(0, len(sample["images"]), batch_size):
                    batch = sample["images"][start:start + batch_size]
                    pixels = torch.stack([preprocess_array(load_raw(image), window[image["channel"]])
                                          for image in batch])
                    tokens = model.prefix(pixels)
                    if tokens.ndim != 3 or tokens.shape[-1] != 384 or not torch.isfinite(tokens).all():
                        raise ValueError("Expected finite full prefix tokens [views,tokens,384]")
                    if array is None:
                        shape = [len(sample["images"]), *tokens.shape[1:]]
                        array = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=tuple(shape))
                    array[start:start + len(batch)] = tokens.cpu().numpy().astype(dtype, copy=False)
                    with torch.no_grad():
                        # Check the precise values that will be used by training,
                        # including explicitly requested cache quantization.
                        cached_tokens = torch.from_numpy(np.array(array[start:start + len(batch)], copy=True)).to(device)
                        frozen_cls.append(model.tail(cached_tokens).float().cpu().numpy())
                array.flush()
                del array
                array = None
                per_view = np.concatenate(frozen_cls)
                pooled = pool_well_features(per_view, sample["images"])
                error = None
                if sample_id in references:
                    error = float(np.max(np.abs(pooled - references[sample_id])))
                    tolerance = 2e-5 if dtype == "float32" else 2e-3
                    if not np.allclose(pooled, references[sample_id], atol=tolerance, rtol=tolerance * 10):
                        raise ValueError(f"Split DINO differs from frozen reference for {sample_id}: {error}")
                temporary.replace(path)
                receipt = {"sample_id": sample_id, "cache_id": cache_id, "identity": well_identity,
                           "image_ids": [i["image_id"] for i in sample["images"]],
                           "shape": shape, "dtype": dtype, "sha256": hash_file(path),
                           "windows": window, "frozen_reference_max_abs_error": error,
                           "completed_at": utc_now()}
                write_json(receipt_path, receipt)
            finally:
                if array is not None:
                    del array
                if temporary.exists():
                    temporary.unlink()
        relative = f"wells/{sample_id}"
        artifacts[relative + ".npy"] = receipt["sha256"]
        artifacts[relative + ".json"] = hash_file(receipt_path)
        records[sample_id] = {key: receipt[key] for key in ("identity", "shape", "dtype", "image_ids")}
        if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(samples):
            print(f"Prefix tokens {ordinal}/{len(samples)} wells ({reused} resumed)", flush=True)
    metadata = {"schema_version": 1, "cache_id": cache_id, "manifest_id": manifest["manifest_id"],
                "identity": identity, "sample_ids": [s["sample_id"] for s in samples],
                "sample_count": len(samples), "image_count": sum(len(s["images"]) for s in samples),
                "array_dtype": dtype, "well_records": records, "artifacts": artifacts,
                "completed_at": utc_now()}
    write_json(output / "metadata.json", metadata)
    return metadata
