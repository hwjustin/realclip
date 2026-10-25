"""Portable, complete-well Cell Painting pairs and deterministic preprocessing."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

CHANNELS = ("DNA", "RNA", "ER", "Mito", "AGP")
PREPROCESS = {
    "version": "well-channel-quantile-letterbox-v1",
    "percentiles": [0.5, 99.8],
    "window_scope": "treatment_well_channel_all_sites",
    "fallback": "minmax_then_zero_for_constant",
    "size": 518,
    "resize": "bicubic_antialias_longest_side",
    "padding": "center_zero",
    "rgb": "replicate_grayscale",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
FORMAT = "realclip-h200-well-pairs-v1"


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def image_path(data_root: str | Path, relative_path: str) -> Path:
    """Resolve only portable relative paths, including protection from symlinks."""
    _require(isinstance(relative_path, str) and relative_path and "\\" not in relative_path,
             "Image path must be a portable relative path")
    relative = PurePosixPath(relative_path)
    _require(not relative.is_absolute() and ".." not in relative.parts
             and ":" not in relative_path, "Image path escapes the data root")
    root = Path(data_root).resolve()
    result = (root / relative).resolve()
    _require(result.is_relative_to(root), "Image path escapes the data root through a symlink")
    return result


def _valid_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_manifest(manifest: dict, data_root: str | Path | None = None) -> dict:
    """Validate identities, pairing, windows, paths, and held-out group isolation."""
    payload = {key: value for key, value in manifest.items() if key != "manifest_id"}
    _require(manifest.get("manifest_id") == stable_hash(payload), "Manifest identity mismatch")
    _require(manifest.get("schema_version") == 1 and manifest.get("format") == FORMAT,
             "Unsupported portable manifest format")
    _require(manifest.get("channels") == list(CHANNELS), "Unsupported channel order")
    _require(manifest.get("preprocessing") == PREPROCESS, "Unsupported preprocessing")
    samples = manifest.get("samples")
    _require(isinstance(samples, list) and bool(samples), "Manifest contains no samples")
    sample_ids, well_ids, group_splits, path_hashes = set(), set(), {}, {}
    for sample in samples:
        for key in ("sample_id", "well_id", "source", "plate", "parent_id"):
            _require(isinstance(sample.get(key), str) and bool(sample[key]), f"Missing sample {key}")
        _require(sample["sample_id"] not in sample_ids, "Duplicate sample_id")
        _require(sample["well_id"] not in well_ids, "Duplicate well_id")
        sample_ids.add(sample["sample_id"])
        well_ids.add(sample["well_id"])
        split = sample.get("split")
        _require(split in ("train", "val", "test"), "Unsupported split")
        for group in (("plate", sample["source"], sample["plate"]), ("parent", sample["parent_id"])):
            previous = group_splits.setdefault(group, split)
            _require(previous == split, "Plate or parent occurs in multiple splits")
        text = sample.get("description")
        _require(isinstance(text, str) and bool(text.strip()), "Missing description")
        _require(hashlib.sha256(text.encode("utf-8")).hexdigest() == sample.get("description_sha256"),
                 "Description hash mismatch")
        windows = sample.get("windows", {})
        _require(set(windows) == set(CHANNELS), "Missing channel intensity windows")
        for window in windows.values():
            lower, upper = float(window["lower"]), float(window["upper"])
            _require(math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper <= 65535,
                     "Invalid intensity window")
        images, seen = sample.get("images", []), set()
        _require(isinstance(images, list) and bool(images), "Well contains no images")
        for item in images:
            site, channel = str(item.get("site", "")), item.get("channel")
            _require(site.isdigit() and int(site) >= 0 and channel in CHANNELS, "Invalid site/channel")
            _require((site, channel) not in seen, "Duplicate site/channel image")
            seen.add((site, channel))
            shape = item.get("shape")
            _require(isinstance(shape, list) and len(shape) == 2
                     and all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in shape),
                     "Invalid image shape")
            sha = item.get("sha256")
            _require(_valid_sha(sha), "Invalid image SHA256")
            relative = item.get("path")
            image_path(data_root or ".", relative)
            _require(relative == f"images/{sha}.tif", "Image path must use its content SHA256")
            _require(path_hashes.setdefault(relative, sha) == sha, "Conflicting image SHA256")
        sites = {site for site, _ in seen}
        _require(seen == {(site, channel) for site in sites for channel in CHANNELS},
                 "Each site must contain all five fluorescence channels")
    return manifest


def load_manifest(data_root: str | Path) -> dict:
    root = Path(data_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return validate_manifest(manifest, root)


def load_raw(path: str | Path, shape: list[int], sha256: str | None = None) -> np.ndarray:
    if sha256 is not None and hash_file(path) != sha256:
        raise ValueError(f"Raw image hash mismatch: {path}")
    with Image.open(path) as opened:
        raw = np.asarray(opened).copy()
    _require(raw.ndim == 2 and raw.dtype.kind == "u" and raw.dtype.itemsize == 2,
             f"Expected a two-dimensional uint16 TIFF: {path}")
    _require(list(raw.shape) == shape, f"Raw image shape mismatch: {path}")
    return raw


def preprocess_array(raw: np.ndarray, window: dict) -> torch.Tensor:
    """Match the original full-field, fixed-window 518-pixel transform exactly."""
    _require(raw.ndim == 2 and raw.dtype.kind == "u" and raw.dtype.itemsize == 2,
             "Preprocessing expects a two-dimensional uint16 array")
    lower, upper = float(window["lower"]), float(window["upper"])
    _require(math.isfinite(lower) and math.isfinite(upper) and upper >= lower, "Invalid intensity window")
    unit = np.zeros(raw.shape, dtype=np.float32) if upper == lower else np.clip(
        (raw.astype(np.float32) - lower) / (upper - lower), 0, 1)
    unit = torch.from_numpy(unit).unsqueeze(0).unsqueeze(0)
    height, width = raw.shape
    size = int(PREPROCESS["size"])
    scale = size / max(height, width)
    new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))
    resized = F.interpolate(unit, size=(new_height, new_width), mode="bicubic",
                            align_corners=False, antialias=True).clamp_(0, 1)
    canvas = torch.zeros((1, 1, size, size), dtype=torch.float32)
    top, left = (size - new_height) // 2, (size - new_width) // 2
    canvas[:, :, top:top + new_height, left:left + new_width] = resized
    rgb = canvas[0].expand(3, -1, -1).clone()
    mean = torch.tensor(PREPROCESS["mean"], dtype=torch.float32)[:, None, None]
    std = torch.tensor(PREPROCESS["std"], dtype=torch.float32)[:, None, None]
    return (rgb - mean) / std


class WellDataset(Dataset):
    """One well and its unchanged description; every site contains all channels."""

    def __init__(self, data_root: str | Path, split: str):
        _require(split in ("train", "val", "test", "all"), "Unsupported split")
        self.data_root = Path(data_root).resolve()
        self.manifest = load_manifest(self.data_root)
        self.samples = [s for s in self.manifest["samples"] if split == "all" or s["split"] == split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        images = sorted(sample["images"], key=lambda item: (int(item["site"]), CHANNELS.index(item["channel"])))
        pixels = [preprocess_array(load_raw(image_path(self.data_root, item["path"]), item["shape"]),
                                   sample["windows"][item["channel"]]) for item in images]
        return {"sample_id": sample["sample_id"], "well_id": sample["well_id"],
                "pixels": torch.stack(pixels),
                "channels": torch.tensor([CHANNELS.index(item["channel"]) for item in images], dtype=torch.long),
                "text": sample["description"]}


def collate_wells(batch: list[dict]) -> list[dict]:
    """Keep variable-site wells separate instead of padding or dropping views."""
    return batch


def verify_data(data_root: str | Path, verify_images: bool = False) -> dict:
    manifest = load_manifest(data_root)
    images = {item["path"]: item for s in manifest["samples"] for item in s["images"]}
    for number, item in enumerate(images.values(), 1):
        path = image_path(data_root, item["path"])
        _require(path.is_file(), f"Missing image: {path}")
        if verify_images:
            load_raw(path, item["shape"], item["sha256"])
        if number % 1000 == 0:
            print(f"Verified {number}/{len(images)} image files", flush=True)
    return {"manifest_id": manifest["manifest_id"], "wells": len(manifest["samples"]),
            "unique_images": len(images), "image_sha256_and_dtype_verified": verify_images}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--verify-images", action="store_true", help="Read every TIFF and verify its SHA256, uint16 dtype, and dimensions")
    args = parser.parse_args()
    print(json.dumps(verify_data(args.data_root, args.verify_images), indent=2))


if __name__ == "__main__":
    main()
