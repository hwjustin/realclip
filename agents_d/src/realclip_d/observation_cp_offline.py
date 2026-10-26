"""Check prepared real well pairs with the new CP contract, without paid calls.

This module has no live mode and never reads credentials. Real image/CP inputs
are exercised using the SDK's local scripted model; no new phenotype is inferred.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .config import Settings
from .contracts import CPMetadata
from .inputs import load_sample
from .observation_offline import network_disabled
from .observation_protocol import prepare_observation
from .storage import file_hash, freeze_json, read_json, run_lock, verify_stage
from .tools.cp import CPTable
from .workflow import _describe, _read_research


async def verify(prepared: Path, output: Path) -> dict:
    prepared, output = prepared.resolve(), output.resolve()
    if prepared == output or prepared in output.parents or output in prepared.parents:
        raise ValueError("Offline output must be separate from prepared input data")
    cases = sorted(prepared.glob("*/pairing.json"))
    if not cases:
        raise ValueError("No prepared treatment/control cases found")
    report = {"network_calls": 0, "paid_model_calls": 0, "cost_usd": 0,
              "biological_observation_performed": False,
              "model": "local scripted SDK model; no image interpretation", "samples": []}
    with network_disabled(), run_lock(output):
        for pair_path in cases:
            source = pair_path.parent
            pair = read_json(pair_path)
            original = load_sample(Path(pair["research_input"]))
            stage = Path(pair["research_stage"])
            research = _read_research(stage, original)
            research_hash = file_hash(stage / "result.json")
            sample_hash = file_hash(source / "sample.json")
            sample = load_sample(source / "sample.json")
            if sample.cp_metadata is None:
                # These inputs establish identity and well matching, but not the
                # exact upstream normalization population or aggregation recipe.
                sample.cp_metadata = CPMetadata(value_kind="preprocessed",
                    comparison_context="Treatment and control are from the same source and plate, with the same published profile file.",
                    quality_notes=["Normalization reference, feature definitions, aggregation method and preservation of order are not verified here; keep them unknown."])
            directory = output / sample.sample_id
            freeze_json(directory / "sample.json", sample.model_dump(mode="json"))
            view, audit = prepare_observation(sample, research, source / "handoff_revision.json")
            freeze_json(directory / "observation_input.json", view.model_dump(mode="json"))
            freeze_json(directory / "handoff_audit.json", audit)
            cp = CPTable(sample.cp_file, preprocessing=sample.cp_preprocessing, metadata=sample.cp_metadata)
            preview = cp.query(feature_names=["Cells_AreaShape_Area"], limit=1)
            if preview.status != "ok" or len(preview.data["rows"]) != 2:
                raise ValueError("Expected one complete actual cell-area comparison")
            assert {r["role"] for r in preview.data["rows"]} == {"treatment", "control"}
            freeze_json(directory / "cp_comparison_preview.json", preview.model_dump(mode="json"))
            result = await _describe(sample, research, Settings(), directory / "observation",
                                     "mock", None, source / "handoff_revision.json")
            verify_stage(directory / "observation")
            if file_hash(stage / "result.json") != research_hash or file_hash(source / "sample.json") != sample_hash:
                raise ValueError("Read-only source artifacts changed")
            sites = {role: sorted({i.field for i in sample.images if i.role == role})
                     for role in ("treatment", "control")}
            if sites != pair["sites"] or len(sample.images) != len(pair["images"]):
                raise ValueError("Offline input does not cover the prepared sites/images")
            report["samples"].append({"sample_id": sample.sample_id, "images": len(sample.images),
                "sites": sites, "cp_rows": len(cp.rows), "cp_comparisons": len(cp.comparisons()),
                "exact_query_returns_both_roles": True,
                "area_comparison_ready": preview.data["comparisons"][0]["comparison_ready"],
                "interpretation_gaps": preview.data["comparisons"][0]["interpretation_gaps"],
                "research_result_unchanged": True, "source_input_unchanged": True,
                "schema_version": result.schema_version, "description": result.description,
                "hypothesis_assessments": [h.assessment for h in result.hypothesis_assessments]})
        freeze_json(output / "summary.json", report)
        lines = ["# CP input and Observation integration: offline check", "",
                 "Real images and well profiles; local scripted model only. No biological image interpretation.",
                 "Paid model calls: 0. Network calls: 0. Cost: $0. Research outputs are unchanged.", "",
                 "| Sample | Images | CP rows | Feature comparisons | Contract |",
                 "|---|---:|---:|---:|---|"]
        lines += [f"| {r['sample_id']} | {r['images']} | {r['cp_rows']} | {r['cp_comparisons']} | {r['schema_version']} |"
                  for r in report["samples"]]
        lines += ["", "Every exact area query returns both roles on the same page. Missing upstream interpretation metadata remains explicit; no physical units, normalization parameters or feature definitions are invented.",
                  "Descriptions remain empty and all hypotheses uncertain because the scripted model performs no biological interpretation. Synthetic integration tests separately exercise linked supporting/conflicting CP evidence and description assembly."]
        (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import json
    print(json.dumps(asyncio.run(verify(args.prepared, args.output)), indent=2))


if __name__ == "__main__":
    main()
