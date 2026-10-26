"""Small shared artifact and identity utilities; no upstream project imports."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_manifest(path: str | Path) -> dict:
    manifest = read_json(path)
    payload = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if manifest.get("manifest_id") != stable_hash(payload):
        raise ValueError("Manifest identity mismatch")
    if manifest.get("channels") != list(CHANNELS) or manifest.get("preprocessing") != PREPROCESS:
        raise ValueError("Unsupported channel/preprocessing configuration")
    return manifest


def model_identity(directory: str | Path) -> dict:
    directory = Path(directory).resolve()
    files = {str(path.relative_to(directory)): hash_file(path)
             for path in sorted(directory.rglob("*")) if path.is_file()
             and not any(part.startswith(".") for part in path.relative_to(directory).parts)}
    if not files:
        raise ValueError(f"No model files in {directory}")
    return {"path": str(directory), "files": files, "sha256": stable_hash(files)}
