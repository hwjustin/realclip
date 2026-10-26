"""Read-only source projection for standalone plain-description Summarize.

Only finalized reviewed sentences and their cited observations cross this
boundary. Native byte-identity mappings are supported in this first version;
arbitrary rendered transforms need a later, independently verified adapter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .observation_contracts import ObservationRequest
from .observation_stages import ObservationPipelineResult
from .storage import file_hash, read_json, verify_stage
from .summarize_contracts import (
    ClipVisualManifest, RenderingConfig, SummarizeConstraints,
    TargetTreatmentWell,
)


SHARED_TASK_CONTEXT = (
    "These data are Cell Painting microscopy images. One contrastive "
    "pair consists of the explicitly supplied multi-site, multi-channel image "
    "collection of ONE treatment well and one concise visual caption of that "
    "well. Control images and CellProfiler measurements helped the upstream "
    "Observation report; the final caption must describe treatment appearances "
    "that can stand on their own without control images or compound identity. "
    "Sites are sampled fields, not independent replicate wells. Mixed-stain "
    "channels do not establish a single component's identity. Channel registration "
    "and same-cell correspondence are not presumed. This stage compresses reviewed "
    "text using the supplied context; it does not re-inspect pixels."
)

_IMAGE_FIELDS = (
    "image_id", "role", "well_id", "channel", "field", "view", "sha256",
    "width", "height", "parent_image_id", "crop_box", "alignment_id", "stain",
    "display_settings", "acquisition_settings",
)


def _load_source(directory: Path) -> tuple[ObservationRequest, ObservationPipelineResult, dict[str, dict]]:
    directory = Path(directory)
    receipt = verify_stage(directory)
    view = ObservationRequest.model_validate(read_json(directory / "input.json"))
    result = ObservationPipelineResult.model_validate(read_json(directory / "result.json"))
    if view.sample_id != result.sample_id:
        raise ValueError("Observation input and final result belong to different samples")
    originals: dict[str, dict] = {}
    for row in view.images:
        key = row.get("image_id")
        if not isinstance(key, str) or not key or key in originals:
            raise ValueError("Observation image IDs must be present and unique")
        if row.get("role") not in {"treatment", "control"}:
            raise ValueError("Observation image role must be explicit")
        # The typed manifest below also checks selected hashes and dimensions.
        originals[key] = row
    treatment = [row for row in originals.values() if row["role"] == "treatment"]
    wells = {row.get("well_id") for row in treatment}
    if not treatment or len(wells) != 1 or not all(isinstance(well, str) and well.strip() for well in wells):
        raise ValueError("Observation must identify exactly one treatment well")
    sources = dict(originals)
    crop_path = directory / "crops" / "manifest.json"
    if crop_path.exists():
        if "crops/manifest.json" not in receipt["files"]:
            raise ValueError("Saved crop manifest must be bound by the completion receipt")
        manifest = read_json(crop_path)
        if manifest.get("originals") != view.images:
            raise ValueError("Saved crop originals differ from the bound Observation inputs")
        for entry in manifest.get("crops", []):
            row = {key: value for key, value in entry.items() if key != "relative_path"}
            key, parent = row.get("image_id"), originals.get(row.get("parent_image_id"))
            if not isinstance(key, str) or not key or key in sources or parent is None:
                raise ValueError("Saved crop identity or parent is invalid")
            for name in ("role", "channel", "field", "well_id", "alignment_id", "display_settings", "acquisition_settings", "stain"):
                if row.get(name) != parent.get(name):
                    raise ValueError("Saved crop metadata differs from its original")
            box = row.get("crop_box")
            if not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box):
                raise ValueError("Saved crop bounds must be explicit integers")
            x, y, width, height = box
            if (min(x, y) < 0 or min(width, height) < 1
                    or x + width > parent["width"] or y + height > parent["height"]
                    or (width, height) != (row.get("width"), row.get("height"))
                    or row.get("parent_sha256") != parent.get("sha256")):
                raise ValueError("Saved crop bounds or parent hash are invalid")
            relative = entry.get("relative_path")
            if relative != key + ".png" or Path(relative).name != relative:
                raise ValueError("Saved crop path is invalid")
            relative_bound = "crops/" + relative
            if relative_bound not in receipt["files"] or file_hash(directory / relative_bound) != row.get("sha256"):
                raise ValueError("Saved crop pixels must match the sealed hash")
            from PIL import Image

            with Image.open(directory / relative_bound) as loaded:
                loaded.load()
                if loaded.size != (width, height):
                    raise ValueError("Saved crop dimensions differ from its metadata")
            sources[key] = row
    for observation in result.observations:
        if not set(observation.image_ids) <= set(sources):
            raise ValueError("Final observation cites an unknown source image")
    return view, result, sources


def _target(view: ObservationRequest) -> TargetTreatmentWell:
    treatment = next(row for row in view.images if row["role"] == "treatment")
    return TargetTreatmentWell(sample_id=view.sample_id, well_id=treatment["well_id"],
                               cell_line=view.context.cell_line)


def _validate_manifest(manifest: dict, view: ObservationRequest, sources: dict[str, dict]) -> ClipVisualManifest:
    parsed = ClipVisualManifest.model_validate(manifest)
    if parsed.target_treatment_well != _target(view):
        raise ValueError("CLIP manifest differs from the bound target treatment well")
    for image in parsed.images:
        source = sources.get(image.source_image_ids[0])
        if source is None or source.get("role") != "treatment":
            raise ValueError("CLIP view must map to a known treatment source image")
        for key in ("well_id", "channel", "field", "view", "width", "height", "parent_image_id", "alignment_id", "stain", "display_settings", "acquisition_settings"):
            if getattr(image, key) != source.get(key):
                raise ValueError(f"CLIP view metadata differs from source: {key}")
        if image.crop_box != (tuple(source["crop_box"]) if source.get("crop_box") is not None else None):
            raise ValueError("CLIP crop mapping differs from its source")
        if image.source_sha256 != source.get("sha256") or image.sha256 != image.source_sha256:
            raise ValueError("Only byte-identical views with verified source hashes are supported")
    return parsed


def make_provisional_manifest(observation_dir: Path, channels: list[str]) -> dict:
    """Explicitly select original treatment views for a candidate-only demo.

    No channel default and no implicit ROI inclusion. Final encoder rendering is
    deliberately unspecified, so this is not a ready-to-train image collection.
    """
    if not channels or len(channels) != len(set(channels)) or any(not isinstance(value, str) or not value.strip() for value in channels):
        raise ValueError("Supply a nonempty, unique, explicit list of channels")
    view, _, sources = _load_source(Path(observation_dir))
    originals = [row for row in view.images if row["role"] == "treatment" and row.get("parent_image_id") is None]
    available = {row["channel"] for row in originals}
    if not set(channels) <= available:
        raise ValueError("Requested channels are absent from original treatment views")
    rendering = RenderingConfig(status="provisional", description=(
        "Explicit native original treatment views; byte identity is verified against "
        "Observation metadata. Final encoder resizing, cropping and normalization "
        "are not configured, and fine-detail visibility is unverified."))
    images = []
    for source in originals:
        if source["channel"] not in channels:
            continue
        row = {key: source[key] for key in _IMAGE_FIELDS if key in source}
        row.update(source_image_ids=[source["image_id"]], source_sha256=source["sha256"],
                   rendering=rendering.model_dump(mode="json"), visibility="source_native")
        images.append(row)
    manifest = {"schema_version": "1", "status": "provisional", "sample_id": view.sample_id,
                "target_treatment_well": _target(view).model_dump(mode="json"), "images": images,
                "rendering": rendering.model_dump(mode="json"), "limitations": [
                    "Final encoder resizing, cropping and normalization have not been configured.",
                    "Fine-detail visibility after encoder resizing is unverified; omit details that depend on it.",
                ]}
    return _validate_manifest(manifest, view, sources).model_dump(mode="json")


def _build_legacy_request(observation_dir: Path, manifest: dict, constraints: SummarizeConstraints) -> dict:
    """Reconstruct sealed original requests for readback, never for new runs."""
    constraints = SummarizeConstraints.model_validate(constraints.model_dump(mode="json"))
    view, result, sources = _load_source(Path(observation_dir))
    parsed = _validate_manifest(manifest, view, sources)
    visual_images = parsed.model_dump(mode="json")["images"]
    selected_ids = {key for sentence in result.description_sentences for key in sentence.observation_ids}
    observations = []
    for observation in result.observations:
        if observation.id not in selected_ids:
            continue
        row = observation.model_dump(mode="json", exclude={"cp_feature_ids"})
        row["image_evidence"] = [
            {key: sources[image_id][key] for key in _IMAGE_FIELDS if key in sources[image_id]}
            for image_id in observation.image_ids
        ]
        row["eligible_treatment_image_ids"] = [
            image["image_id"] for image in visual_images
            if set(image["source_image_ids"]) & set(observation.image_ids)
        ]
        observations.append(row)
    model_constraints = constraints.model_dump(mode="json", exclude={"tokenizer_path"})
    model_constraints["output_language"] = "English"
    model_constraints["tokenizer_configured"] = constraints.tokenizer_path is not None
    return {
        "shared_task_context": SHARED_TASK_CONTEXT,
        "target_treatment_well": parsed.target_treatment_well.model_dump(mode="json"),
        "description": result.description,
        "selected_observations": observations,
        "clip_visual_manifest": parsed.model_dump(mode="json"),
        "limitations": list(dict.fromkeys([*result.limitations, *parsed.limitations])),
        "constraints": model_constraints,
    }


def build_request(observation_dir: Path, manifest: dict, constraints: SummarizeConstraints,
                  *, request_version: str = "2") -> dict:
    """Project reviewed text onto the exact target views before summarization.

    Image membership is an input boundary, not a generated-caption check. A
    record spanning excluded treatment views cannot be split safely using its
    image IDs alone, so it is withheld as a whole. Reviewed sentences containing
    both eligible and ineligible observations are explicitly partial context;
    only their independently eligible clauses may be used by the model.
    """
    if request_version == "1":
        return _build_legacy_request(observation_dir, manifest, constraints)
    if request_version != "2":
        raise ValueError("Unsupported Summarize request version")
    constraints = SummarizeConstraints.model_validate(constraints.model_dump(mode="json"))
    view, result, sources = _load_source(Path(observation_dir))
    parsed = _validate_manifest(manifest, view, sources)
    visual_images = parsed.model_dump(mode="json")["images"]
    selected_sources = {image["source_image_ids"][0] for image in visual_images}
    selected_ids = {key for sentence in result.description_sentences for key in sentence.observation_ids}
    available_by_channel: dict[str, list[str]] = {}
    for image in visual_images:
        available_by_channel.setdefault(image["channel"], []).append(image["field"])
    available_by_channel = {key: sorted(set(values)) for key, values in available_by_channel.items()}
    available_fields = sorted({image["field"] for image in visual_images})
    observations = []
    for observation in result.observations:
        if (observation.id not in selected_ids or not observation.visually_confirmed
                or observation.comparison_kind not in {"treatment_only", "treatment_control"}):
            continue
        treatment_ids = [key for key in observation.image_ids if sources[key]["role"] == "treatment"]
        if not treatment_ids or not set(treatment_ids) <= selected_sources:
            continue
        row = observation.model_dump(mode="json", exclude={"cp_feature_ids"})
        row["image_ids"] = treatment_ids
        row["image_evidence"] = [
            {key: sources[image_id][key] for key in _IMAGE_FIELDS if key in sources[image_id]}
            for image_id in treatment_ids
        ]
        row["eligible_treatment_image_ids"] = [
            image["image_id"] for image in visual_images
            if image["source_image_ids"][0] in treatment_ids
        ]
        observed_by_channel: dict[str, list[str]] = {}
        for key in treatment_ids:
            source = sources[key]
            observed_by_channel.setdefault(source["channel"], []).append(source["field"])
        row["target_scope"] = {
            "observed_fields": sorted({sources[key]["field"] for key in treatment_ids}),
            "available_fields": available_fields,
            "observed_fields_by_channel": {key: sorted(set(values)) for key, values in observed_by_channel.items()},
            "available_fields_by_channel": available_by_channel,
            "views": sorted({sources[key]["view"] for key in treatment_ids}),
            "sampling_limit": (
                "These are the cited treatment views, not a prevalence estimate. "
                "Preserve the observation's own scope and qualifiers; availability "
                "of other fields does not establish the appearance there. A local "
                "crop never establishes whole-field coverage or well-wide frequency."
            ),
        }
        row["reviewed_sentence_ids"] = [
            sentence.id for sentence in result.description_sentences
            if observation.id in sentence.observation_ids
        ]
        observations.append(row)
    eligible_ids = {row["id"] for row in observations}
    contexts = []
    for sentence in result.description_sentences:
        allowed = [key for key in sentence.observation_ids if key in eligible_ids]
        if not allowed:
            continue
        contexts.append({
            "sentence_id": sentence.id, "text": sentence.text,
            "eligible_observation_ids": allowed,
            "eligibility": "fully_eligible" if len(allowed) == len(sentence.observation_ids) else "partial",
            "use": "only_clauses_supported_by_eligible_observations",
        })
    model_constraints = constraints.model_dump(mode="json", exclude={"tokenizer_path"})
    model_constraints["output_language"] = "English"
    model_constraints["tokenizer_configured"] = constraints.tokenizer_path is not None
    return {
        "request_version": "2", "shared_task_context": SHARED_TASK_CONTEXT,
        "target_treatment_well": parsed.target_treatment_well.model_dump(mode="json"),
        "reviewed_description_context": contexts, "eligible_observations": observations,
        "clip_visual_manifest": parsed.model_dump(mode="json"),
        "limitations": list(dict.fromkeys([*result.limitations, *parsed.limitations])),
        "constraints": model_constraints,
    }

