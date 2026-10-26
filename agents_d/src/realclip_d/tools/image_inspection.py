"""Bounded native-pixel crops from the images supplied for one observation.

Coordinates are in the supplied image's pixel grid. A crop is another view of
existing pixels, not a new optical measurement or an inferred registration.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from ..contracts import SampleInput
from ..inputs import image_manifest
from ..storage import digest, file_hash, write_json


class ImageInspection:
    """Keep generated images and their provenance separate from supplied inputs.

    ``max_pixels`` limits each crop; ``max_crops`` limits unique crop files.
    Repeated requests for the same original and box reuse its generated image,
    while retaining a separate request receipt with the current question.
    """

    def __init__(self, sample: SampleInput, directory: Path, *,
                 max_crops: int = 12, max_pixels: int = 4_000_000):
        if type(max_crops) is not int or max_crops < 0:
            raise ValueError("max_crops must be a nonnegative integer")
        if type(max_pixels) is not int or max_pixels < 1:
            raise ValueError("max_pixels must be a positive integer")
        originals = image_manifest(sample, include_metadata=True)
        self.directory = Path(directory)
        self.max_crops = max_crops
        self.max_pixels = max_pixels
        self._originals = {row["image_id"]: deepcopy(row) for row in originals}
        self._source_paths = {row.image_id: Path(row.path) for row in sample.images}
        self._crops: dict[str, dict[str, Any]] = {}
        self._requests: list[dict[str, Any]] = []
        if (self.directory / "manifest.json").exists():
            raise ValueError("Image inspection output already exists; use a new directory")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._save_manifest()

    @property
    def images(self) -> list[dict[str, Any]]:
        """Generated image metadata, with no filesystem paths exposed."""
        return deepcopy(list(self._crops.values()))

    def _save_manifest(self) -> None:
        write_json(self.directory / "manifest.json", {
            "schema_version": "1",
            "originals": list(self._originals.values()),
            "crops": [{**row, "relative_path": row["image_id"] + ".png"}
                      for row in self._crops.values()],
            "requests": self._requests,
            "limits": {"max_crops": self.max_crops, "max_pixels_per_crop": self.max_pixels},
            "note": "Native decoded pixels only: no resizing, contrast adjustment, new resolution, "
                    "or inferred cross-channel registration. Local crops cannot establish well-wide prevalence.",
        })

    def _receipt(self, *, image_id: str, crop_box: list[int], question: str,
                 selection_reason: str, status: str, crop_id: str | None = None) -> None:
        self._requests.append({"request_index": len(self._requests) + 1,
            "parent_image_id": image_id, "crop_box": crop_box,
            "question": question, "selection_reason": selection_reason,
            "status": status, "crop_image_id": crop_id})
        self._save_manifest()

    def path_for(self, image_id: str) -> Path:
        """Resolve an existing generated crop; never accept a caller's file path."""
        row = self._crops.get(image_id)
        if row is None:
            raise ValueError("Unknown generated crop image ID")
        path = self.directory / (image_id + ".png")
        try:
            if file_hash(path) != row["sha256"]:
                raise ValueError("Generated crop image changed after creation")
        except OSError as error:
            raise ValueError("Generated crop image is missing or unreadable") from error
        return path

    def crop(self, image_id: str, x: int, y: int, width: int, height: int,
             question: str, selection_reason: str) -> dict[str, Any]:
        """Return safe metadata for a lossless PNG cropped from an original.

        The model must supply an observation question and why this region was
        selected. This method records those intentions without interpreting the
        pixels or claiming that the selected region represents the whole well.
        """
        if image_id not in self._originals:
            raise ValueError("Crop parent must be an original supplied image ID; recursive crops are unavailable")
        if any(type(value) is not int for value in (x, y, width, height)):
            raise ValueError("Crop coordinates and dimensions must be integers")
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("Crop requires a nonnegative origin and positive dimensions")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A nonempty observation question is required")
        if not isinstance(selection_reason, str) or not selection_reason.strip():
            raise ValueError("A nonempty region selection reason is required")
        question, selection_reason = question.strip(), selection_reason.strip()
        original = self._originals[image_id]
        box = [x, y, width, height]
        if x + width > original["width"] or y + height > original["height"]:
            self._receipt(image_id=image_id, crop_box=box, question=question,
                          selection_reason=selection_reason, status="rejected_out_of_bounds")
            raise ValueError("Crop box exceeds the original image bounds")
        if width * height > self.max_pixels:
            self._receipt(image_id=image_id, crop_box=box, question=question,
                          selection_reason=selection_reason, status="rejected_pixel_limit")
            raise ValueError("Crop exceeds the per-image pixel limit")
        # Read once, then crop exactly the bytes whose hash was bound at setup.
        # This also prevents a file replacement between hash and image loading.
        try:
            content = self._source_paths[image_id].read_bytes()
        except OSError as error:
            raise ValueError("Original image is missing or unreadable") from error
        if hashlib.sha256(content).hexdigest() != original["sha256"]:
            self._receipt(image_id=image_id, crop_box=box, question=question,
                          selection_reason=selection_reason, status="rejected_original_changed")
            raise ValueError("Original image changed after observation input was bound")
        crop_id = "crop_" + digest({"parent_image_id": image_id,
            "parent_sha256": original["sha256"], "crop_box": box})[:24]
        if crop_id in self._originals:
            raise ValueError("Generated crop ID conflicts with a supplied image ID")
        if crop_id in self._crops:
            self.path_for(crop_id)  # Do not return a silently changed cached image.
            self._receipt(image_id=image_id, crop_box=box, question=question,
                          selection_reason=selection_reason, status="reused", crop_id=crop_id)
            return {**deepcopy(self._crops[crop_id]), "question": question,
                    "selection_reason": selection_reason}
        if len(self._crops) >= self.max_crops:
            self._receipt(image_id=image_id, crop_box=box, question=question,
                          selection_reason=selection_reason, status="rejected_crop_limit")
            raise ValueError("Unique crop limit reached")
        path = self.directory / (crop_id + ".png")
        if path.exists():
            raise ValueError("Generated crop output already exists")
        try:
            with Image.open(BytesIO(content)) as loaded:
                loaded.load()
                local = loaded.crop((x, y, x + width, y + height))
                # PNG is lossless for supported source modes; converting modes
                # here would alter the pixels and is deliberately not attempted.
                local.save(path, format="PNG")
        except (OSError, SyntaxError, ValueError) as error:
            path.unlink(missing_ok=True)
            raise ValueError("Original image cannot be cropped into an unchanged-pixel PNG") from error
        metadata = {**deepcopy(original), "image_id": crop_id,
            "parent_image_id": image_id, "parent_sha256": original["sha256"],
            "crop_box": box, "view": "local", "width": width, "height": height,
            "sha256": file_hash(path), "pixel_transform": "native_pixel_crop_only",
            "question": question, "selection_reason": selection_reason}
        self._crops[crop_id] = metadata
        self._receipt(image_id=image_id, crop_box=box, question=question,
                      selection_reason=selection_reason, status="created", crop_id=crop_id)
        return deepcopy(metadata)
