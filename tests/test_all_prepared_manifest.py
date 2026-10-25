"""Archival prepared wells are independent of the frozen training cohort."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("prepare_all_prepared", Path(__file__).parents[1] / "scripts/prepare_all_prepared.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def write_json(path, value):
    path.write_text(json.dumps(value))


def setup_archive(tmp_path, complete=False, legacy=False):
    input_path = tmp_path / "input.json"
    write_json(input_path, {})
    images = []
    for channel in exporter.CHANNELS:
        path = tmp_path / f"{channel}.tif"
        path.write_bytes(("raw " + channel).encode())
        image = {"role": "treatment", "channel": channel, "field": "0", "well_id": "source|plate|B01",
                 "source_url": f"https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source/images/{channel}.tif",
                 "raw_path": str(path), "raw_sha256": exporter.file_hash(path), "raw_bytes": path.stat().st_size}
        if legacy:
            image.pop("raw_sha256")
            image.pop("raw_bytes")
        images.append(image)
    # Extra channels/control rows never become CLIP raw inputs.
    pairing_path = tmp_path / "pairing.json"
    write_json(pairing_path, {"compound_id": "archive", "dataset_sample_id": "source|plate|B01",
                              "images": images + [{"role": "control"}, {"role": "treatment", "channel": "Brightfield_L"}]})
    training = {"manifest_id": "training-id", "samples": [{"sample_id": "fit", "well_id": "source|plate|A01", "images": []}]}
    prepared_path = tmp_path / "prepared.json"
    write_json(prepared_path, {"samples": [
        {"compound_id": "fit", "dataset_sample_id": "source|plate|A01"},
        {"compound_id": "archive", "dataset_sample_id": "source|plate|B01", "sample_id": "archive",
         "source": "source", "plate": "plate", "treatment_well": "B01", "input_path": str(input_path),
         "input_sha256": exporter.file_hash(input_path), "pairing_path": str(pairing_path)}]})
    result_paths = []
    if complete:
        result_path = tmp_path / "result.json"
        write_json(result_path, {"description": "Preserved archival observation."})
        write_json(tmp_path / "complete.json", {"files": {"result.json": exporter.file_hash(result_path)}})
        result_paths.append(str(result_path))
    inventory_path = tmp_path / "inventory.jsonl"
    inventory_path.write_text("\n".join(json.dumps(row) for row in [
        {"compound_id": "fit", "category": "complete"},
        {"compound_id": "archive", "category": "complete" if complete else "network_or_timeout", "result_paths": result_paths}]))
    return prepared_path, inventory_path, training, input_path


@pytest.mark.parametrize("complete,legacy", [(False, False), (False, True), (True, False)])
def test_archive_export_preserves_status_and_excludes_controls(tmp_path, complete, legacy):
    prepared, inventory, training, _ = setup_archive(tmp_path, complete, legacy)
    first = exporter.archive_samples(prepared, inventory, training)
    assert first == exporter.archive_samples(prepared, inventory, training)
    assert first["total_prepared_wells"] == 2 and len(first["samples"]) == 1
    assert len(first["images"]) == 5
    sample = first["samples"][0]
    assert sample["training_eligible"] is False
    assert sample["description_status"] == ("complete" if complete else "incomplete")
    assert (sample["description"] is not None) == complete
    assert first["provenance"]["legacy_raw_bytes_hashed"] > 0 if legacy else first["provenance"]["legacy_raw_bytes_hashed"] == 0
    assert str(tmp_path) not in json.dumps(first)


def test_changed_prepared_inputs_are_not_exported(tmp_path):
    prepared, inventory, training, input_path = setup_archive(tmp_path)
    input_path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="input checksum"):
        exporter.archive_samples(prepared, inventory, training)
