"""Explicit, hash-bound handoff revisions and preflight checks; no model calls."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from .contracts import Contract, HypothesisViewV3, NonEmpty, ResearchResultV3, ResearchResultV4, SampleInput
from .inputs import image_manifest
from .observation_contracts import ObservationRequest
from .storage import digest, read_json
from .tools.cp import CPTable


_RESEARCH_TERMS = re.compile(
    r"\bmTOR(?:C[12])?\b|\bMOA\b|\bmechanism(?:s| of action)?\b|"
    r"\bapopto(?:sis|tic)\b|\bmitotic arrest\b|\bautophagy\b|"
    r"\bcytotoxic\b|\bdying\b|\bdead cells?\b|\bdamaged cells?\b", re.I)


def check_observation_text(value: object, sample: SampleInput) -> None:
    """Conservative lexical guard, not proof of semantic anonymity or accuracy."""
    import json

    text = json.dumps(value, ensure_ascii=False)
    private = [sample.compound.name, sample.compound.inchikey, sample.compound.smiles,
               sample.compound.chembl_id, *sample.known_moa]
    # A bare CID is a number and can coincide with legitimate image dimensions
    # or measurement counts. No CID field is projected into this view.
    for word in private:
        if word and re.search(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", text, re.I):
            raise ValueError("Observation view contains a private compound/MOA identifier; supply a reviewed handoff revision")
    if _RESEARCH_TERMS.search(text):
        raise ValueError("Observation view contains research/biological interpretation; supply a reviewed handoff revision")


class HandoffEdit(Contract):
    hypothesis_id: NonEmpty
    field: Literal["change", "applicability", "where_to_look", "what_to_check", "how_to_compare"]
    original: NonEmpty
    replacement: NonEmpty
    reason: NonEmpty


class HandoffRevision(Contract):
    schema_version: Literal["1"] = "1"
    sample_id: NonEmpty
    research_sha256: NonEmpty
    edits: list[HandoffEdit]

    @model_validator(mode="after")
    def unique_edits(self):
        keys = [(edit.hypothesis_id, edit.field) for edit in self.edits]
        if len(keys) != len(set(keys)):
            raise ValueError("Handoff edits must target distinct hypothesis fields")
        return self


def prepare_observation(sample: SampleInput, research: ResearchResultV3 | ResearchResultV4,
                        handoff_path: Path | None = None,
                        revision: HandoffRevision | None = None) -> tuple[ObservationRequest, dict]:
    if not isinstance(research, (ResearchResultV3, ResearchResultV4)):
        raise ValueError("New observations require four-stage research output")
    if sample.sample_id != research.sample_id:
        raise ValueError("Research result belongs to another sample")
    if handoff_path is not None and revision is not None:
        raise ValueError("Provide one handoff revision, not two")
    if handoff_path is not None:
        revision = HandoffRevision.model_validate(read_json(handoff_path))
    revision = revision or HandoffRevision(sample_id=sample.sample_id,
                                           research_sha256=digest(research.model_dump(mode="json")), edits=[])
    if revision.sample_id != sample.sample_id or revision.research_sha256 != digest(research.model_dump(mode="json")):
        raise ValueError("Handoff revision belongs to a different sample or research result")
    guidance = {row.hypothesis_id: row for row in research.observation_guidance}
    original = [HypothesisViewV3(hypothesis_id=row.id, change=row.change, applicability=row.applicability,
                **guidance[row.id].model_dump(exclude={"hypothesis_id"})).model_dump(mode="json")
                for row in research.morphology_hypotheses]
    projected = {row["hypothesis_id"]: dict(row) for row in original}
    for edit in revision.edits:
        row = projected.get(edit.hypothesis_id)
        if row is None or row[edit.field] != edit.original:
            raise ValueError("Handoff revision original text does not match the bound research")
        row[edit.field] = edit.replacement
    cp = CPTable(sample.cp_file, preprocessing=sample.cp_preprocessing, metadata=sample.cp_metadata)
    # CP identifiers and labels will appear in subsequent tool responses, too.
    # Check before client construction rather than leaking a private label on turn two.
    check_observation_text([row.model_dump(exclude={"value"}) for row in cp.rows], sample)
    check_observation_text(cp.metadata.model_dump(mode="json"), sample)
    view = ObservationRequest(sample_id=sample.sample_id, context=sample.context,
                              hypotheses=[HypothesisViewV3.model_validate(row) for row in projected.values()],
                              images=image_manifest(sample, include_metadata=True),
                              cp_available=sample.cp_file is not None, cp_catalogue=cp.catalogue())
    check_observation_text(view.model_dump(mode="json"), sample)
    audit = {"schema_version": "1", "revision": revision.model_dump(mode="json"),
             "original_hypotheses": original, "observation_hypotheses": [h.model_dump(mode="json") for h in view.hypotheses],
             "guard": "Literal identifiers and selected research terms checked; semantic fidelity requires review."}
    return view, audit
