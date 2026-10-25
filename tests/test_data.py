from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import pytest
import torch

from realclip_h200.data import (CHANNELS, FORMAT, PREPROCESS, WellDataset, collate_wells,
                               hash_file, load_manifest, preprocess_array, stable_hash, verify_data)


def make_manifest(tmp_path: Path):
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    images = []
    for index, channel in enumerate(CHANNELS):
        raw = np.arange(24, dtype=np.uint16).reshape(4, 6) + index * 100
        source = root / f"source{index}.tif"
        Image.fromarray(raw).save(source)
        sha = hash_file(source)
        path = root / "images" / f"{sha}.tif"
        source.rename(path)
        images.append({"image_id": str(index), "path": str(path.relative_to(root)), "sha256": sha,
                       "shape": [4, 6], "site": "0", "channel": channel})
    text = "Oval nuclei with thin cellular extensions."
    sample = {"sample_id": "one", "well_id": "source|plate|A01", "source": "source", "plate": "plate",
              "well": "A01", "parent_id": "parent", "split": "train", "description": text,
              "description_sha256": hashlib.sha256(text.encode()).hexdigest(), "images": images,
              "windows": {channel: {"lower": index * 100, "upper": index * 100 + 23, "method": "pooled_percentile"}
                          for index, channel in enumerate(CHANNELS)}}
    payload = {"schema_version": 1, "format": FORMAT, "channels": list(CHANNELS),
               "preprocessing": PREPROCESS, "samples": [sample], "lineage": {}}
    manifest = {**payload, "manifest_id": stable_hash(payload)}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, manifest


def save_changed(root, manifest):
    manifest["manifest_id"] = stable_hash({k: v for k, v in manifest.items() if k != "manifest_id"})
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_complete_well_uint16_and_full_field_preprocessing(tmp_path):
    root, manifest = make_manifest(tmp_path)
    dataset = WellDataset(root, "train")
    sample = dataset[0]
    assert sample["pixels"].shape == (5, 3, 518, 518)
    assert sample["pixels"].dtype == torch.float32
    assert sample["channels"].tolist() == [0, 1, 2, 3, 4]
    assert sample["text"] == manifest["samples"][0]["description"]
    assert torch.equal(sample["pixels"][0], sample["pixels"][1])
    # Letterbox bands are zero before ImageNet normalization.
    assert torch.allclose(sample["pixels"][0, :, 0, 0], -torch.tensor(PREPROCESS["mean"]) / torch.tensor(PREPROCESS["std"]))
    batch = [sample]
    assert collate_wells(batch) is batch
    assert verify_data(root, True)["unique_images"] == 5


@pytest.mark.parametrize("mutation,match", [
    (lambda s: s["images"].pop(), "five fluorescence"),
    (lambda s: s.__setitem__("description", "changed"), "Description hash"),
    (lambda s: s["images"][0].__setitem__("path", "../escape.tif"), "escapes"),
    (lambda s: s["images"][0].__setitem__("path", "/tmp/escape.tif"), "escapes"),
])
def test_reject_corrupt_pairing_and_paths(tmp_path, mutation, match):
    root, manifest = make_manifest(tmp_path)
    mutation(manifest["samples"][0])
    save_changed(root, manifest)
    with pytest.raises(ValueError, match=match):
        load_manifest(root)


def test_reject_symlink_escape_and_plate_leakage(tmp_path):
    root, manifest = make_manifest(tmp_path)
    item = manifest["samples"][0]["images"][0]
    path = root / item["path"]
    outside = tmp_path / "outside.tif"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        load_manifest(root)
    path.unlink()
    outside.rename(path)
    second = copy.deepcopy(manifest["samples"][0])
    second.update(sample_id="two", well_id="source|plate|A02", parent_id="other", split="val")
    manifest["samples"].append(second)
    save_changed(root, manifest)
    with pytest.raises(ValueError, match="multiple splits"):
        load_manifest(root)


def test_reject_changed_tiff_bytes(tmp_path):
    root, manifest = make_manifest(tmp_path)
    path = root / manifest["samples"][0]["images"][0]["path"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_data(root, True)


def test_constant_window_and_input_dtype():
    raw = np.full((3, 5), 10, dtype=np.uint16)
    pixels = preprocess_array(raw, {"lower": 10, "upper": 10})
    expected = -torch.tensor(PREPROCESS["mean"]) / torch.tensor(PREPROCESS["std"])
    assert torch.allclose(pixels[:, 200, 200], expected)
    with pytest.raises(ValueError, match="uint16"):
        preprocess_array(raw.astype(np.uint8), {"lower": 0, "upper": 20})


def test_streaming_export_roundtrip_and_resume(tmp_path):
    root, manifest = make_manifest(tmp_path)
    source = copy.deepcopy(manifest)
    source.pop("format")
    source.pop("lineage")
    for sample in source["samples"]:
        sample.pop("windows")
        for item in sample["images"]:
            item["path"] = str(root / item["path"])
            item["observation_path"] = "/never/export/observation.png"
    source["manifest_id"] = stable_hash({k: v for k, v in source.items() if k != "manifest_id"})
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    windows = tmp_path / "windows.json"
    windows.write_text(json.dumps({"preprocessing": PREPROCESS, "wells": {"one": manifest["samples"][0]["windows"]}}))
    spec = importlib.util.spec_from_file_location("export_data", Path(__file__).parents[1] / "scripts/export_data.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    output = tmp_path / "export"
    plan = exporter.export(source_path, windows, output, 20480, plan_only=True)
    assert plan["unique_images"] == 5 and not output.exists()
    index = exporter.export(source_path, windows, output, 20480)
    assert index["complete"]
    restored = tmp_path / "restored"
    restored.mkdir()
    for entry in index["files"]:
        with tarfile.open(output / entry["file_name"]) as archive:
            assert all(not member.name.startswith("/") and ".." not in Path(member.name).parts for member in archive)
            archive.extractall(restored, filter="data")
    assert verify_data(restored, True)["unique_images"] == 5
    assert WellDataset(restored, "train")[0]["text"] == manifest["samples"][0]["description"]
    assert "observation_path" not in (output / "manifest.json").read_text()
    assert str(root) not in (output / "manifest.json").read_text()
    resumed = exporter.export(source_path, windows, output, 20480)
    assert resumed["files"] == index["files"]
    first_image_archive = output / index["files"][1]["file_name"]
    first_image_archive.write_bytes(first_image_archive.read_bytes() + b"broken")
    with pytest.raises(ValueError, match="archive changed"):
        exporter.export(source_path, windows, output, 20480)


def test_ssh_staging_hardlinks_resume_and_preserve_conflicts(tmp_path):
    root, manifest = make_manifest(tmp_path)
    source = copy.deepcopy(manifest)
    source.pop("format")
    source.pop("lineage")
    for sample in source["samples"]:
        sample.pop("windows")
        for item in sample["images"]:
            item["path"] = str(root / item["path"])
    source["manifest_id"] = stable_hash({k: v for k, v in source.items() if k != "manifest_id"})
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    windows = tmp_path / "windows.json"
    windows.write_text(json.dumps({"preprocessing": PREPROCESS, "wells": {"one": manifest["samples"][0]["windows"]}}))
    spec = importlib.util.spec_from_file_location("stage_ssh_data", Path(__file__).parents[1] / "scripts/stage_ssh_data.py")
    staging = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staging)
    output = tmp_path / "staged"
    plan = staging.stage(source_path, windows, output, plan_only=True)
    assert plan["additional_image_bytes"] == 0 and not output.exists()
    result = staging.stage(source_path, windows, output, verify_source=True)
    assert result["new_hardlinks"] == 5 and result["source_sha256_verified"]
    item = manifest["samples"][0]["images"][0]
    original, staged = root / item["path"], output / item["path"]
    assert original.stat().st_ino == staged.stat().st_ino
    again = staging.stage(source_path, windows, output)
    assert again["new_hardlinks"] == 0 and again["reused_files"] == 5
    assert verify_data(output, True)["unique_images"] == 5
    staged.unlink()
    staged.write_bytes(b"existing unrelated file")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        staging.stage(source_path, windows, output)
    assert staged.read_bytes() == b"existing unrelated file"
    assert hash_file(original) == item["sha256"]
