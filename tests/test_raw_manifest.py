"""Source provenance must resolve every frozen image before publishing URLs."""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from realclip_h200.data import CHANNELS, FORMAT, PREPROCESS

SPEC = importlib.util.spec_from_file_location("prepare_raw_manifest", Path(__file__).parents[1] / "scripts/prepare_raw_manifest.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def seal(value):
    value["manifest_id"] = exporter.stable_hash({k: v for k, v in value.items() if k != "manifest_id"})


def fixture(tmp_path):
    source_path, portable_path, pairing_path = [tmp_path / name for name in ("source.json", "portable.json", "pairing.json")]
    images, records = [], []
    for index, channel in enumerate(CHANNELS):
        path = tmp_path / f"image-{index}.tif"
        payload = b"uint16-fixture" + bytes([index])
        path.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        images.append({"image_id": f"im{index}", "site": "0", "channel": channel,
                       "path": str(path), "sha256": sha, "shape": [2, 3]})
        url = f"https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_1/images/batch/images/plate/image-{index}.tif"
        records.append({"raw_path": str(path), "url": url, "source_url": url,
                        "raw_sha256": sha, "raw_bytes": len(payload), "role": "treatment",
                        "well_id": "source|plate|A01", "shape": [2, 3]})
    text = "Oval nuclei and thin extensions."
    sample = {"sample_id": "one", "well_id": "source|plate|A01", "source": "source",
              "plate": "plate", "well": "A01", "parent_id": "parent", "split": "train",
              "description": text, "description_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "images": images, "provenance": {"files": {str(pairing_path): "pending"}}}
    source = {"channels": list(CHANNELS), "preprocessing": PREPROCESS, "samples": [sample]}
    portable_sample = copy.deepcopy(sample)
    portable_sample.pop("provenance")
    portable_sample["windows"] = {channel: {"lower": 0, "upper": 1} for channel in CHANNELS}
    for image in portable_sample["images"]:
        image["path"] = f"images/{image['sha256']}.tif"
    portable = {"schema_version": 1, "format": FORMAT, "channels": list(CHANNELS),
                "preprocessing": PREPROCESS, "samples": [portable_sample], "lineage": {}}
    pairing = {"images": records}
    refresh(source_path, portable_path, pairing_path, source, portable, pairing)
    return source_path, portable_path, pairing_path, source, portable, pairing


def refresh(source_path, portable_path, pairing_path, source, portable, pairing):
    write_json(pairing_path, pairing)
    source["samples"][0]["provenance"]["files"][str(pairing_path)] = exporter.file_hash(pairing_path)
    seal(source)
    write_json(source_path, source)
    portable["lineage"].update(source_manifest_id=source["manifest_id"], source_manifest_sha256=exporter.file_hash(source_path))
    seal(portable)
    write_json(portable_path, portable)


def test_full_source_mapping_deterministic_and_no_local_paths(tmp_path):
    source_path, portable_path, _, _, portable, _ = fixture(tmp_path)
    output = tmp_path / "export"
    first = exporter.prepare(portable_path, source_path, output)
    second = exporter.prepare(portable_path, source_path, tmp_path / "second")
    assert first == second
    dataset = json.loads(gzip.decompress((output / "dataset.json.gz").read_bytes()))
    assert dataset == portable
    rows = [json.loads(line) for line in gzip.decompress((output / "raw_downloads.jsonl.gz").read_bytes()).splitlines()]
    assert len(rows) == 5 and [row["path"] for row in rows] == sorted(row["path"] for row in rows)
    assert first["counts"] == {"wells": 1, "images": 5}
    assert first["raw_bytes"] == 5 * len(b"uint16-fixture\0")
    for name, artifact in first["artifacts"].items():
        raw = (output / name).read_bytes()
        assert len(raw) == artifact["size_bytes"] and hashlib.sha256(raw).hexdigest() == artifact["sha256"]
        assert raw == (tmp_path / "second" / name).read_bytes()
        assert str(tmp_path).encode() not in gzip.decompress(raw)
    assert str(tmp_path) not in (output / "metadata.json").read_text()


@pytest.mark.parametrize("change,match", [
    (lambda p: p["images"].pop(), "Missing public source URL"),
    (lambda p: p["images"].append({**p["images"][0], "source_url": "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/other.tif", "url": "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/other.tif"}), "Conflicting URL"),
    (lambda p: p["images"][0].update(raw_sha256="0" * 64), "image bytes"),
    (lambda p: p["images"][0].update(raw_bytes=12345), "byte count"),
    (lambda p: p["images"][0].update(source_url="https://attacker.invalid/image.tif", url="https://attacker.invalid/image.tif"), "unsigned public"),
    (lambda p: p["images"][0].update(role="control"), "treatment identity"),
])
def test_missing_or_conflicting_sources_are_not_published(tmp_path, change, match):
    source_path, portable_path, pairing_path, source, portable, pairing = fixture(tmp_path)
    change(pairing)
    refresh(source_path, portable_path, pairing_path, source, portable, pairing)
    with pytest.raises(ValueError, match=match):
        exporter.prepare(portable_path, source_path, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_changed_description_and_unbound_provenance_rejected(tmp_path):
    source_path, portable_path, _, _, portable, _ = fixture(tmp_path)
    portable["samples"][0]["description"] = "Changed wording."
    seal(portable)
    write_json(portable_path, portable)
    with pytest.raises(ValueError, match="original description"):
        exporter.build_downloads(portable_path, source_path)
    portable["lineage"]["source_manifest_sha256"] = "0" * 64
    seal(portable)
    write_json(portable_path, portable)
    with pytest.raises(ValueError, match="not bound"):
        exporter.build_downloads(portable_path, source_path)


def test_source_pairing_receipt_is_authenticated(tmp_path):
    source_path, portable_path, pairing_path, _, _, pairing = fixture(tmp_path)
    pairing["images"][0]["raw_bytes"] = 123
    write_json(pairing_path, pairing)
    with pytest.raises(ValueError, match="pairing.json checksum"):
        exporter.build_downloads(portable_path, source_path)
