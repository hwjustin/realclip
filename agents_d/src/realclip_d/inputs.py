"""Explicit input files and a narrow, appearance-only agent handoff."""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

from .contracts import (HypothesisView, HypothesisViewV2, HypothesisViewV3, ObservationInput, ResearchOutput,
                        ResearchResultV2, ResearchResultV3, ResearchResultV4,
                        SampleInput)
from .storage import file_hash


def load_sample(path: Path) -> SampleInput:
    sample = SampleInput.model_validate(json.loads(path.read_text(encoding="utf-8")))
    base = path.resolve().parent
    for image in sample.images:
        image.path = (base / image.path).resolve()
    if sample.cp_file:
        sample.cp_file = (base / sample.cp_file).resolve()
    return sample


def research_input(sample: SampleInput) -> dict:
    return {"sample_id": sample.sample_id, "compound": sample.compound.model_dump(mode="json"),
            "known_moa": sample.known_moa, "context": sample.context.model_dump(mode="json")}


def image_manifest(sample: SampleInput, *, include_metadata: bool = False) -> list[dict]:
    if not sample.images:
        raise ValueError("Observation requires treatment and control images")
    if {image.role for image in sample.images} != {"treatment", "control"}:
        raise ValueError("Observation requires both treatment and control images")
    result = []
    for image in sample.images:
        try:
            with Image.open(image.path) as loaded:
                if loaded.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("Use explicit PNG, JPEG or WEBP input images")
                width, height = loaded.size
                loaded.verify()
        except (OSError, SyntaxError) as error:
            raise ValueError(f"Missing or damaged image: {image.image_id}") from error
        row = {"image_id": image.image_id, "role": image.role,
                       "channel": image.channel, "field": image.field,
                       "view": image.view, "sha256": file_hash(image.path)}
        if include_metadata:
            row.update(image.model_dump(mode="json", exclude={"path"}))
            row.update(width=width, height=height)
        result.append(row)
    if include_metadata:
        by_id = {row["image_id"]: row for row in result}
        for row in result:
            if row["crop_box"]:
                x, y, width, height = row["crop_box"]
                parent = by_id[row["parent_image_id"]]
                if x + width > parent["width"] or y + height > parent["height"]:
                    raise ValueError("crop_box exceeds supplied parent image dimensions")
    return result


def hypothesis_view(sample: SampleInput, research: ResearchOutput,
                    images: list[dict[str, str]]) -> ObservationInput:
    if research.sample_id != sample.sample_id:
        raise ValueError("Research result belongs to another sample")
    if isinstance(research, (ResearchResultV3, ResearchResultV4)):
        guidance = {item.hypothesis_id: item for item in research.observation_guidance}
        hypotheses = [HypothesisViewV3(
            hypothesis_id=h.id, change=h.change, applicability=h.applicability,
            **guidance[h.id].model_dump(exclude={"hypothesis_id"}),
        ) for h in research.morphology_hypotheses]
    else:
        view_type = HypothesisViewV2 if isinstance(research, ResearchResultV2) else HypothesisView
        fields = set(view_type.model_fields)
        hypotheses = [view_type.model_validate(h.model_dump(include=fields)) for h in research.hypotheses]
    # Fail visibly on literal private identifiers in free-text projections.
    # This is a narrow leakage check, not proof against all mechanism synonyms.
    private = [sample.compound.name, sample.compound.inchikey, sample.compound.chembl_id,
               sample.compound.smiles, *sample.known_moa]
    appearance = json.dumps({"sample_id": sample.sample_id, "context": sample.context.model_dump(),
                             "hypotheses": [h.model_dump() for h in hypotheses],
                             "images": images}, ensure_ascii=False)
    for value in private:
        if value and re.search(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", appearance, re.I):
            raise ValueError("Hypothesis projection contains a private compound/MOA identifier; preserve result and revise in a new run")
    return ObservationInput(sample_id=sample.sample_id, context=sample.context,
                            hypotheses=hypotheses, images=images,
                            cp_available=sample.cp_file is not None)
