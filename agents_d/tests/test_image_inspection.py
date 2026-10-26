"""Offline checks for pixel fidelity, source binding and bounded local views."""
from __future__ import annotations

import json
import struct

from PIL import Image
import pytest

from realclip_d.contracts import ImageInput, SampleInput
from realclip_d.storage import file_hash, read_json
from realclip_d.tools.image_inspection import ImageInspection


@pytest.fixture
def sample(tmp_path):
    rows = []
    for role, channel in (("treatment", "DNA"), ("treatment", "AGP"), ("control", "DNA")):
        path = tmp_path / f"private-{role}-{channel}.png"
        image = Image.new("RGB", (16, 12))
        image.putdata([(x * 13, y * 17, (x + y) * 7) for y in range(12) for x in range(16)])
        image.save(path)
        rows.append(ImageInput(image_id=f"{role}_{channel}", path=path, role=role,
            channel=channel, field="site_2", well_id=f"{role}_well", stain=channel + " stain",
            pixel_size_um=0.75, acquisition_settings="Supplied acquisition metadata",
            display_settings="Supplied display mapping", alignment_id=None))
    return SampleInput(sample_id="INSPECTION", compound={"name": "PRIVATE_COMPOUND"}, images=rows)


def crop(tool, image_id="treatment_DNA", **kwargs):
    return tool.crop(image_id, kwargs.get("x", 2), kwargs.get("y", 3),
        kwargs.get("width", 5), kwargs.get("height", 4),
        kwargs.get("question", "One outline or adjacent outlines?"),
        kwargs.get("selection_reason", "Confirm a local structure; not a representative sample."))


def test_crop_preserves_native_pixels_and_source_metadata(sample, tmp_path):
    inspection = ImageInspection(sample, tmp_path / "crops")
    original_hash = file_hash(sample.images[0].path)
    metadata = crop(inspection)
    with Image.open(sample.images[0].path) as original, Image.open(inspection.path_for(metadata["image_id"])) as local:
        assert local.size == (5, 4)
        assert local.mode == original.mode
        assert local.tobytes() == original.crop((2, 3, 7, 7)).tobytes()
    assert file_hash(sample.images[0].path) == original_hash
    assert metadata["parent_sha256"] == original_hash
    assert metadata["crop_box"] == [2, 3, 5, 4]
    assert metadata["parent_image_id"] == "treatment_DNA"
    for key in ("role", "channel", "field", "well_id", "stain", "pixel_size_um",
                "acquisition_settings", "display_settings", "alignment_id"):
        assert metadata[key] == getattr(sample.images[0], key)
    assert metadata["pixel_transform"] == "native_pixel_crop_only"
    assert "path" not in metadata
    assert str(tmp_path) not in json.dumps(metadata)
    manifest = read_json(inspection.directory / "manifest.json")
    assert manifest["originals"][0]["sha256"] == original_hash
    assert manifest["crops"][0]["relative_path"] == metadata["image_id"] + ".png"
    assert str(tmp_path) not in json.dumps(manifest)
    metadata["role"] = "control"
    assert inspection.images[0]["role"] == "treatment"


def test_sixteen_bit_pixels_are_not_scaled(sample, tmp_path):
    image = Image.frombytes("I;16", (16, 12), struct.pack("<192H", *(i * 251 for i in range(192))))
    image.save(sample.images[0].path)
    inspection = ImageInspection(sample, tmp_path / "crops")
    metadata = crop(inspection)
    with Image.open(sample.images[0].path) as original, Image.open(inspection.path_for(metadata["image_id"])) as local:
        assert local.mode == original.mode
        assert local.getextrema()[1] > 255
        assert local.tobytes() == original.crop((2, 3, 7, 7)).tobytes()


@pytest.mark.parametrize("params", [
    {"x": -1}, {"y": -1}, {"width": 0}, {"height": -1},
    {"x": True}, {"y": 1.0}, {"width": "5"},
    {"x": 12}, {"y": 10}, {"width": 17}, {"height": 13},
    {"question": "  "}, {"selection_reason": ""},
])
def test_invalid_crop_does_not_create_an_image(sample, tmp_path, params):
    inspection = ImageInspection(sample, tmp_path / "crops")
    with pytest.raises(ValueError):
        crop(inspection, **params)
    assert inspection.images == []
    assert list(inspection.directory.glob("*.png")) == []


def test_original_tamper_is_rejected_even_for_cached_crop(sample, tmp_path):
    inspection = ImageInspection(sample, tmp_path / "crops")
    crop(inspection)
    Image.new("RGB", (16, 12), "white").save(sample.images[0].path)
    with pytest.raises(ValueError, match="Original image changed"):
        crop(inspection)
    assert len(inspection.images) == 1
    assert read_json(inspection.directory / "manifest.json")["requests"][-1]["status"] == "rejected_original_changed"


def test_crop_tamper_is_rejected_when_resolving_or_reusing(sample, tmp_path):
    inspection = ImageInspection(sample, tmp_path / "crops")
    metadata = crop(inspection)
    path = inspection.path_for(metadata["image_id"])
    Image.new("RGB", (5, 4), "white").save(path)
    with pytest.raises(ValueError, match="crop image changed"):
        inspection.path_for(metadata["image_id"])
    with pytest.raises(ValueError, match="crop image changed"):
        crop(inspection)


def test_unique_crop_limit_allows_reuse_and_records_each_question(sample, tmp_path):
    inspection = ImageInspection(sample, tmp_path / "crops", max_crops=1)
    first = crop(inspection)
    second = crop(inspection, question="Is the boundary clear?")
    assert first["image_id"] == second["image_id"]
    assert second["question"] == "Is the boundary clear?"
    assert len(inspection.images) == 1
    with pytest.raises(ValueError, match="Unique crop limit"):
        crop(inspection, x=3)
    assert len(list(inspection.directory.glob("*.png"))) == 1
    requests = read_json(inspection.directory / "manifest.json")["requests"]
    assert [row["status"] for row in requests] == ["created", "reused", "rejected_crop_limit"]
    assert requests[1]["question"] == "Is the boundary clear?"


def test_pixel_limit_rejects_before_writing(sample, tmp_path):
    inspection = ImageInspection(sample, tmp_path / "crops", max_pixels=19)
    with pytest.raises(ValueError, match="pixel limit"):
        crop(inspection)
    assert inspection.images == []
    assert not list(inspection.directory.glob("*.png"))


def test_crops_cannot_be_parents_and_paths_are_not_ids(sample, tmp_path):
    inspection = ImageInspection(sample, tmp_path / "crops")
    metadata = crop(inspection)
    with pytest.raises(ValueError, match="original supplied image ID"):
        crop(inspection, metadata["image_id"])
    with pytest.raises(ValueError, match="original supplied image ID"):
        crop(inspection, str(sample.images[0].path))
    with pytest.raises(ValueError, match="Unknown generated crop"):
        inspection.path_for("../../original.png")


def test_same_box_across_channels_is_not_assumed_registered(sample, tmp_path):
    inspection = ImageInspection(sample, tmp_path / "crops")
    dna = crop(inspection, "treatment_DNA")
    agp = crop(inspection, "treatment_AGP")
    control = crop(inspection, "control_DNA")
    assert len({row["image_id"] for row in (dna, agp, control)}) == 3
    assert dna["alignment_id"] is None and agp["alignment_id"] is None
    assert dna["channel"] == "DNA" and agp["channel"] == "AGP"
    assert control["role"] == "control"
    # A supplied alignment statement is inherited rather than manufactured.
    sample.images[0].alignment_id = "explicit_grid"
    other = ImageInspection(sample, tmp_path / "aligned_crops")
    assert crop(other)["alignment_id"] == "explicit_grid"


def test_crop_identity_is_stable_and_output_is_not_overwritten(sample, tmp_path):
    first = ImageInspection(sample, tmp_path / "first")
    second = ImageInspection(sample, tmp_path / "second")
    assert crop(first)["image_id"] == crop(second)["image_id"]
    with pytest.raises(ValueError, match="already exists"):
        ImageInspection(sample, tmp_path / "first")
