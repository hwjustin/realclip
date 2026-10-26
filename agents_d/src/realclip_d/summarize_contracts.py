"""Input contracts for the standalone plain-description Summarize module.

These are new downstream boundaries. They neither modify Observation contracts
nor certify semantic entailment, rendered visibility or training suitability.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .contracts import Contract, NonEmpty


Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


def _unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


class SummarizeConstraints(Contract):
    max_sentences: int = Field(default=3, ge=1, le=10)
    preferred_word_min: int = Field(default=30, ge=0)
    preferred_word_max: int = Field(default=60, ge=1)
    max_words: int = Field(default=70, ge=1)
    tokenizer_path: str | None = None
    max_text_tokens: int | None = Field(default=None, ge=1)
    text_prefix: str = ""

    @model_validator(mode="after")
    def consistent_limits(self):
        if not self.preferred_word_min <= self.preferred_word_max <= self.max_words:
            raise ValueError("Preferred word range must fit within max_words")
        if (self.tokenizer_path is None) != (self.max_text_tokens is None):
            raise ValueError("Tokenizer path and max_text_tokens must be configured together")
        if self.tokenizer_path is not None and not self.tokenizer_path.strip():
            raise ValueError("Tokenizer path cannot be blank")
        return self


class TargetTreatmentWell(Contract):
    sample_id: NonEmpty
    well_id: NonEmpty
    role: Literal["treatment"] = "treatment"
    cell_line: NonEmpty | None = None


class RenderingConfig(Contract):
    status: Literal["provisional", "final"]
    description: NonEmpty
    encoder_input_size: tuple[int, int] | None = None
    resize_method: NonEmpty | None = None
    normalization: NonEmpty | None = None

    @model_validator(mode="after")
    def valid_size(self):
        if self.encoder_input_size is not None and min(self.encoder_input_size) < 1:
            raise ValueError("Encoder input dimensions must be positive")
        return self


class ClipVisualImage(Contract):
    image_id: NonEmpty
    source_image_ids: list[NonEmpty] = Field(min_length=1, max_length=1)
    role: Literal["treatment"]
    well_id: NonEmpty
    channel: NonEmpty
    field: NonEmpty
    view: NonEmpty
    sha256: Sha256
    source_sha256: Sha256
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    parent_image_id: NonEmpty | None = None
    crop_box: tuple[int, int, int, int] | None = None
    alignment_id: NonEmpty | None = None
    stain: NonEmpty | None = None
    display_settings: NonEmpty | None = None
    acquisition_settings: NonEmpty | None = None
    rendering: RenderingConfig
    visibility: Literal["source_native", "unverified"]


class ClipVisualManifest(Contract):
    schema_version: Literal["1"] = "1"
    status: Literal["provisional", "final"]
    sample_id: NonEmpty
    target_treatment_well: TargetTreatmentWell
    images: list[ClipVisualImage] = Field(min_length=1)
    rendering: RenderingConfig
    limitations: list[NonEmpty] = Field(default_factory=list)

    @model_validator(mode="after")
    def same_target(self):
        if self.sample_id != self.target_treatment_well.sample_id:
            raise ValueError("Manifest and target sample differ")
        _unique([row.image_id for row in self.images], "Manifest image IDs")
        _unique([row.source_image_ids[0] for row in self.images], "Manifest source image IDs")
        if any(row.well_id != self.target_treatment_well.well_id for row in self.images):
            raise ValueError("Manifest images must belong to the target treatment well")
        if self.status != self.rendering.status or any(
            row.rendering.status != self.status for row in self.images
        ):
            raise ValueError("Manifest and rendering statuses must agree")
        return self
