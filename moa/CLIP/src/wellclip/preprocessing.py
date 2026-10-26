"""Treatment-only intensity windows and deterministic full-field transforms."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .common import CHANNELS, PREPROCESS, hash_file


def load_raw(image: dict) -> np.ndarray:
    """Read the declared uint16 full field, rejecting changed source bytes."""
    path = Path(image["path"])
    if image.get("sha256") and hash_file(path) != image["sha256"]:
        raise ValueError(f"Raw image hash mismatch: {path}")
    with Image.open(path) as opened:
        array = np.asarray(opened).copy()
    if array.ndim != 2 or array.dtype.kind != "u" or array.dtype.itemsize != 2:
        raise ValueError(f"Expected a two-dimensional uint16 TIFF: {path}")
    if list(array.shape) != list(image["shape"]):
        raise ValueError(f"Raw image shape mismatch: {path}")
    return array


def _histogram_quantile(histogram: np.ndarray, probability: float) -> float:
    """Match NumPy's linear quantile without allocating pooled image pixels."""
    total = int(histogram.sum())
    if total <= 0:
        raise ValueError("Cannot estimate an intensity window from no pixels")
    rank = probability * (total - 1)
    low_rank = int(np.floor(rank))
    high_rank = int(np.ceil(rank))
    cumulative = np.cumsum(histogram, dtype=np.int64)
    low = int(np.searchsorted(cumulative, low_rank, side="right"))
    high = int(np.searchsorted(cumulative, high_rank, side="right"))
    return float(low + (high - low) * (rank - low_rank))


def compute_windows(sample: dict) -> dict:
    """Pool every supplied treatment site separately for each channel."""
    histograms = {channel: np.zeros(65536, dtype=np.int64) for channel in CHANNELS}
    counts = {channel: 0 for channel in CHANNELS}
    seen = set()
    for image in sample["images"]:
        channel = image["channel"]
        if channel not in histograms:
            raise ValueError(f"Unexpected CLIP channel: {channel}")
        key = (str(image["site"]), channel)
        if key in seen:
            raise ValueError(f"Duplicate site/channel: {key}")
        seen.add(key)
        array = load_raw(image)
        histograms[channel] += np.bincount(array.ravel(), minlength=65536)
        counts[channel] += 1
    sites = {site for site, _ in seen}
    if not sites or any((site, channel) not in seen for site in sites for channel in CHANNELS):
        raise ValueError("Each treatment site must contain all five fluorescence channels")
    result = {}
    probabilities = [percentile / 100 for percentile in PREPROCESS["percentiles"]]
    for channel, histogram in histograms.items():
        lower, upper = [_histogram_quantile(histogram, q) for q in probabilities]
        method = "pooled_percentile"
        if upper <= lower:
            occupied = np.flatnonzero(histogram)
            lower, upper = float(occupied[0]), float(occupied[-1])
            method = "minmax" if upper > lower else "constant_zero"
        result[channel] = {"lower": lower, "upper": upper, "method": method}
    return result


def _unit_array(raw: np.ndarray, window: dict) -> np.ndarray:
    if raw.ndim != 2 or raw.dtype.kind != "u" or raw.dtype.itemsize != 2:
        raise ValueError("Preprocessing expects a two-dimensional uint16 array")
    lower, upper = float(window["lower"]), float(window["upper"])
    if not np.isfinite([lower, upper]).all() or upper < lower:
        raise ValueError("Invalid intensity window")
    if upper == lower:
        return np.zeros(raw.shape, dtype=np.float32)
    return np.clip((raw.astype(np.float32) - lower) / (upper - lower), 0, 1)


def preprocess_array(raw: np.ndarray, window: dict):
    """Return ImageNet-normalized RGB [3,518,518], keeping the whole field."""
    import torch
    from torch.nn import functional as functional

    unit = torch.from_numpy(_unit_array(raw, window)).unsqueeze(0).unsqueeze(0)
    height, width = raw.shape
    size = int(PREPROCESS["size"])
    scale = size / max(height, width)
    new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))
    resized = functional.interpolate(unit, size=(new_height, new_width), mode="bicubic",
                                     align_corners=False, antialias=True).clamp_(0, 1)
    canvas = torch.zeros((1, 1, size, size), dtype=torch.float32)
    top, left = (size - new_height) // 2, (size - new_width) // 2
    canvas[:, :, top:top + new_height, left:left + new_width] = resized
    rgb = canvas[0].expand(3, -1, -1).clone()
    mean = torch.tensor(PREPROCESS["mean"], dtype=torch.float32)[:, None, None]
    std = torch.tensor(PREPROCESS["std"], dtype=torch.float32)[:, None, None]
    return (rgb - mean) / std


def render_preview(sample: dict, windows: dict, output_path: str | Path) -> None:
    """Show the first site's upstream display and new treatment-only transform."""
    images = sample["images"]
    site = min({str(image["site"]) for image in images}, key=lambda value: int(value))
    by_channel = {image["channel"]: image for image in images if str(image["site"]) == site}
    cell, title_height = 210, 28
    canvas = Image.new("RGB", (cell * len(CHANNELS), 2 * (cell + title_height) + 36), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{sample['sample_id']} | {sample['source']} | site {site}", fill="black")
    for column, channel in enumerate(CHANNELS):
        image = by_channel[channel]
        with Image.open(image["observation_path"]) as opened:
            original = opened.convert("RGB")
        normalized = Image.fromarray(np.rint(_unit_array(load_raw(image), windows[channel]) * 255).astype(np.uint8)).convert("RGB")
        for row, (label, view) in enumerate([("Observation", original), ("Treatment-only", normalized)]):
            x, y = column * cell, 36 + row * (cell + title_height)
            draw.text((x + 5, y + 5), f"{channel}: {label}", fill="black")
            thumb = ImageOps.pad(view, (cell, cell), method=Image.Resampling.BICUBIC, color="black")
            canvas.paste(thumb, (x, y + title_height))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
